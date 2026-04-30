"""Build a compact JSON payload for the Level 1 briefing.

The goal is to hand the model enough signal to pick the three most
interesting issues without going over the input token budget. We aim for
<= 4000 input tokens which we approximate at ~16000 bytes of JSON text
(~4 chars per token). The summary uses atop's own numbers: sstat
aggregates at the start and end of the window, deltas across the window,
and the top processes by CPU and RSS.
"""

from __future__ import annotations

import json
from typing import Any

from atop_web.llm import prompts, schema
from atop_web.llm.provider import LLMProvider, LLMProviderError
from atop_web.parser.reader import RawLog, Sample

# Token budget is enforced on the JSON string we pass as the user message.
# 4000 tokens x 4 chars/token = 16000 chars. Leave headroom for the prompt
# wrapping text the provider may add.
MAX_INPUT_CHARS = 14_000
DEFAULT_TOP_N = 5


def _pct_ticks(ticks: int, hertz: int, interval: int, ncpu: int) -> float | None:
    if hertz <= 0 or interval <= 0 or ncpu <= 0:
        return None
    return round(ticks / (hertz * interval * ncpu) * 100.0, 2)


def _pages_to_mib(pages: int | None, pagesize: int) -> float | None:
    # ``pages`` is ``None`` when the source rawlog does not carry this
    # counter (for example atop 2.7 has no availablemem). Pass the missing
    # signal through so the model sees "not measured" instead of a zero
    # that looks like a real reading.
    if pages is None:
        return None
    if pagesize <= 0:
        return 0.0
    return round(pages * pagesize / (1024 * 1024), 1)


def _summarize_cpu(first: Sample, last: Sample, hertz: int, ncpu: int) -> dict:
    def bucket(s: Sample) -> dict[str, Any] | None:
        if s.system_cpu is None:
            return None
        a = s.system_cpu.all
        interval = max(1, s.interval)
        fields = ["utime", "stime", "ntime", "Itime", "Stime", "wtime", "itime", "steal", "guest"]
        row: dict[str, Any] = {
            "interval": s.interval,
            "nrcpu": s.system_cpu.nrcpu,
            "lavg1": round(s.system_cpu.lavg1, 2),
            "lavg5": round(s.system_cpu.lavg5, 2),
            "lavg15": round(s.system_cpu.lavg15, 2),
        }
        for f in fields:
            ticks = getattr(a, f)
            row[f + "_pct"] = _pct_ticks(ticks, hertz, interval, ncpu)
        return row

    return {
        "first": bucket(first),
        "last": bucket(last),
    }


def _summarize_mem(first: Sample, last: Sample, pagesize: int) -> dict:
    def bucket(s: Sample) -> dict | None:
        m = s.system_memory
        if m is None:
            return None
        used_pages = max(
            m.physmem - m.freemem - m.cachemem - m.buffermem - m.slabmem, 0
        )
        return {
            "physmem_mib": _pages_to_mib(m.physmem, pagesize),
            "used_mib": _pages_to_mib(used_pages, pagesize),
            "free_mib": _pages_to_mib(m.freemem, pagesize),
            "cache_mib": _pages_to_mib(m.cachemem, pagesize),
            "slab_mib": _pages_to_mib(m.slabmem, pagesize),
            "available_mib": _pages_to_mib(m.availablemem, pagesize),
            "totswap_mib": _pages_to_mib(m.totswap, pagesize),
            "freeswap_mib": _pages_to_mib(m.freeswap, pagesize),
        }

    return {
        "first": bucket(first),
        "last": bucket(last),
    }


def _summarize_disk(first: Sample, last: Sample) -> dict:
    def by_name(s: Sample) -> dict[str, dict]:
        if s.system_disk is None:
            return {}
        return {d.name: d for d in s.system_disk.disks}

    last_by_name = by_name(last)
    first_by_name = by_name(first)
    dt = max(1, last.curtime - first.curtime)
    out: list[dict] = []
    for name, dev in last_by_name.items():
        first_dev = first_by_name.get(name)
        delta_r = max(dev.nrsect - first_dev.nrsect, 0) if first_dev else 0
        delta_w = max(dev.nwsect - first_dev.nwsect, 0) if first_dev else 0
        read_mibs = round(delta_r * 512 / (1024 * 1024) / dt, 2)
        write_mibs = round(delta_w * 512 / (1024 * 1024) / dt, 2)
        out.append(
            {
                "name": name,
                "kind": dev.kind,
                "read_mibps": read_mibs,
                "write_mibps": write_mibs,
                "inflight": dev.inflight,
                "ndisc": dev.ndisc,
            }
        )
    # Trim to the busiest devices so we stay within the byte budget.
    out.sort(key=lambda d: d["read_mibps"] + d["write_mibps"], reverse=True)
    return {"devices": out[:6]}


def _summarize_net(first: Sample, last: Sample) -> dict:
    def by_name(s: Sample) -> dict:
        if s.system_network is None:
            return {}
        return {i.name: i for i in s.system_network.interfaces}

    last_by_name = by_name(last)
    first_by_name = by_name(first)
    dt = max(1, last.curtime - first.curtime)
    out: list[dict] = []
    for name, iface in last_by_name.items():
        first_iface = first_by_name.get(name)
        dr = max(iface.rbyte - first_iface.rbyte, 0) if first_iface else 0
        dt_b = max(iface.sbyte - first_iface.sbyte, 0) if first_iface else 0
        out.append(
            {
                "name": name,
                "type": iface.type,
                "rx_kbps": round(dr / 1000 / dt, 2),
                "tx_kbps": round(dt_b / 1000 / dt, 2),
                "rerrs": iface.rerrs,
                "rdrop": iface.rdrop,
                "serrs": iface.serrs,
                "sdrop": iface.sdrop,
            }
        )
    out.sort(key=lambda d: d["rx_kbps"] + d["tx_kbps"], reverse=True)
    return {"interfaces": out[:6]}


def _summarize_processes(sample: Sample, top_n: int = DEFAULT_TOP_N) -> dict:
    if not sample.processes:
        return {"by_cpu": [], "by_rss": []}
    by_cpu = sorted(
        sample.processes, key=lambda p: p.utime + p.stime, reverse=True
    )[:top_n]
    by_rss = sorted(sample.processes, key=lambda p: p.rmem_kb, reverse=True)[:top_n]

    def row(p) -> dict:
        # Deliberately omit cmdline: it often leaks secrets or is too long.
        return {
            "pid": p.pid,
            "name": p.name,
            "cpu_ticks": p.utime + p.stime,
            "rmem_kb": p.rmem_kb,
            "vmem_kb": p.vmem_kb,
            "dsk_read_sectors": p.rsz,
            "dsk_write_sectors": p.wsz,
        }

    return {"by_cpu": [row(p) for p in by_cpu], "by_rss": [row(p) for p in by_rss]}


def build_briefing_input(rawlog: RawLog) -> dict:
    """Shape ``rawlog`` into a compact summary dict ready for the model."""
    samples = rawlog.samples
    if not samples:
        return {
            "capture": {
                "hostname": rawlog.header.nodename,
                "kernel": rawlog.header.release,
                "aversion": rawlog.header.aversion,
                "sample_count": 0,
            },
            "note": "no samples",
        }
    first = samples[0]
    last = samples[-1]
    hertz = rawlog.header.hertz or 100
    ncpu = first.nrcpu or (first.system_cpu.nrcpu if first.system_cpu else 1) or 1
    pagesize = rawlog.header.pagesize or 4096

    return {
        "capture": {
            "hostname": rawlog.header.nodename,
            "kernel": rawlog.header.release,
            "machine": rawlog.header.machine,
            "aversion": rawlog.header.aversion,
            "hertz": hertz,
            "pagesize": pagesize,
            "ncpu": ncpu,
            "sample_count": len(samples),
            "start": first.curtime,
            "end": last.curtime,
            "duration_seconds": last.curtime - first.curtime,
        },
        "cpu": _summarize_cpu(first, last, hertz, ncpu),
        "memory": _summarize_mem(first, last, pagesize),
        "disk": _summarize_disk(first, last),
        "network": _summarize_net(first, last),
        "processes_first": _summarize_processes(first),
        "processes_last": _summarize_processes(last),
    }


def _truncate_processes(payload: dict) -> bool:
    """Halve the process lists in place. Returns True if anything was dropped."""
    dropped = False
    for key in ("processes_first", "processes_last"):
        block = payload.get(key)
        if not isinstance(block, dict):
            continue
        for sub in ("by_cpu", "by_rss"):
            rows = block.get(sub)
            if isinstance(rows, list) and len(rows) > 1:
                new_len = max(1, len(rows) // 2)
                block[sub] = rows[:new_len]
                dropped = True
    return dropped


def _fit_to_budget(payload: dict, budget: int = MAX_INPUT_CHARS) -> tuple[str, bool]:
    """Shrink ``payload`` until its JSON serialization fits the char budget."""
    truncated = False
    while True:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= budget:
            return text, truncated
        # Prefer dropping process rows first; they are the biggest chunk.
        if _truncate_processes(payload):
            truncated = True
            continue
        # Next, shrink disk/network lists.
        for key in ("disk", "network"):
            block = payload.get(key)
            if isinstance(block, dict):
                for sub in ("devices", "interfaces"):
                    rows = block.get(sub)
                    if isinstance(rows, list) and len(rows) > 1:
                        new_len = max(1, len(rows) // 2)
                        block[sub] = rows[:new_len]
                        truncated = True
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= budget:
            return text, truncated
        # Last resort: hard truncate so we never exceed the budget.
        return text[: budget - 32] + '..."TRUNCATED"}', True


def generate_briefing(provider: LLMProvider, rawlog: RawLog) -> dict:
    """Run the provider against the compact summary and return a briefing dict.

    The caller catches ``LLMProviderError``; any other exception propagates
    and gets logged by the route handler.
    """
    payload = build_briefing_input(rawlog)
    user_text, truncated = _fit_to_budget(payload)
    if truncated:
        user_text = user_text + "\n\nNote: input was truncated to fit budget."
    response = provider.complete_json(
        prompts.SYSTEM_BRIEFING, user_text, schema.BRIEFING_SCHEMA
    )
    return schema.validate_briefing(response)
