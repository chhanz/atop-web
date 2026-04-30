// Range hint badge fallback (see ``scanAssistantTextForRangeHints``)
// --------------------------------------------------------------
// Primary path: the LLM emits ``<range start=... end=... label=.../>`` tags
// which the server forwards as ``range_hint`` SSE events, each rendered by
// ``appendRangeHintBadge``. Because some providers omit the tag even when
// they mention exact ISO timestamps in prose, a fallback scans the finished
// assistant text for ISO8601 UTC timestamps and creates badges for any
// pairs the primary path missed.
//
// Manual cases (no JS unit harness yet):
//   A. LLM emits two <range/> tags -> two badges. Fallback sees the same
//      ISO pairs in text but dedupes via data-start-epoch/data-end-epoch.
//   B. LLM emits no <range/> but body has two ISO timestamps -> fallback
//      creates one badge.
//   C. Body has a single ISO timestamp -> fallback expands by
//      ``state.capture.minRangeSeconds`` (default 300s) on each side.
//   D. ISO timestamps fall outside the capture window -> dropped silently.
//   E. One <range/> tag + a separate ISO pair in prose -> two badges.
(() => {
  const state = {
    session: null,
    samples: null,
    currentIndex: null,
    charts: {},
    lastJob: null,
    pollTimer: null,
    retry: null,
    timeRange: { from: null, to: null, fullStart: null, fullEnd: null },
    tz: "utc",
    busy: false,
    sort: { by: "cpu", order: "desc" },
    procMeta: null,
    lastSummary: null,
    cpuUnit: "pct",
    sysmemView: "memory",
    sysmemUnit: "mib",
    sysmem: null,
    syscpu: null,
    sysdsk: null,
    diskUnit: "iops",
    diskDevice: null,
    sysnet: null,
    netInterface: null,
    llmHealth: null,
    currentJobId: null,
    chatHistory: [],
    chatBusy: false,
    chatOpen: false,
    // Sample interval info from /api/summary. ``minRangeSeconds`` is the
    // enforced minimum width when the user clicks a range hint badge or
    // types a range narrower than a single sample - without it a 10
    // minute tag on a 600s interval capture lands between samples.
    capture: { intervalSeconds: null, minRangeSeconds: 300 },
  };

  const fmt = new Intl.NumberFormat();

  // Time zone aware formatting ----------------------------------------------

  function loadTz() {
    try {
      const saved = window.localStorage.getItem("atop-web.tz");
      if (saved === "utc" || saved === "local") {
        state.tz = saved;
      }
    } catch (e) {
      // localStorage may be unavailable; fall through to default.
    }
  }

  function saveTz() {
    try {
      window.localStorage.setItem("atop-web.tz", state.tz);
    } catch (e) {
      // ignore
    }
  }

  function localOffsetLabel() {
    const off = -new Date().getTimezoneOffset();
    const sign = off >= 0 ? "+" : "-";
    const abs = Math.abs(off);
    const hh = String(Math.floor(abs / 60)).padStart(2, "0");
    const mm = String(abs % 60).padStart(2, "0");
    let abbr = "";
    try {
      const parts = new Intl.DateTimeFormat(undefined, { timeZoneName: "short" }).formatToParts(new Date());
      const tzPart = parts.find((p) => p.type === "timeZoneName");
      if (tzPart) abbr = tzPart.value;
    } catch (e) {
      // ignore
    }
    return `Local ${abbr ? abbr + " " : ""}UTC${sign}${hh}:${mm}`.trim();
  }

  function pad(n, width = 2) {
    return String(n).padStart(width, "0");
  }

  function breakdownEpoch(epoch) {
    const d = new Date(epoch * 1000);
    if (state.tz === "utc") {
      return {
        year: d.getUTCFullYear(),
        month: d.getUTCMonth() + 1,
        day: d.getUTCDate(),
        hour: d.getUTCHours(),
        minute: d.getUTCMinutes(),
        second: d.getUTCSeconds(),
      };
    }
    return {
      year: d.getFullYear(),
      month: d.getMonth() + 1,
      day: d.getDate(),
      hour: d.getHours(),
      minute: d.getMinutes(),
      second: d.getSeconds(),
    };
  }

  function formatDateTime(epoch) {
    if (!epoch) return "";
    const p = breakdownEpoch(epoch);
    return `${p.year}-${pad(p.month)}-${pad(p.day)} ${pad(p.hour)}:${pad(p.minute)}:${pad(p.second)}`;
  }

  function formatChartTick(epoch) {
    if (!epoch) return "";
    const p = breakdownEpoch(epoch);
    return `${pad(p.month)}-${pad(p.day)} ${pad(p.hour)}:${pad(p.minute)}:${pad(p.second)}`;
  }

  function parseTextDateTime(value) {
    if (!value) return null;
    const m = /^\s*(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?\s*$/.exec(value);
    if (!m) return null;
    const [_, y, mo, d, h, mi, s] = m;
    const year = Number(y), mon = Number(mo), day = Number(d);
    const hour = Number(h), minute = Number(mi), second = Number(s || "0");
    if (mon < 1 || mon > 12 || day < 1 || day > 31 || hour > 23 || minute > 59 || second > 59) {
      return null;
    }
    let epochMs;
    if (state.tz === "utc") {
      epochMs = Date.UTC(year, mon - 1, day, hour, minute, second);
    } else {
      epochMs = new Date(year, mon - 1, day, hour, minute, second).getTime();
    }
    if (Number.isNaN(epochMs)) return null;
    return Math.floor(epochMs / 1000);
  }

  function formatBytes(bytes) {
    if (!bytes && bytes !== 0) return "";
    const units = ["B", "KiB", "MiB", "GiB"];
    let n = bytes;
    let i = 0;
    while (n >= 1024 && i < units.length - 1) {
      n /= 1024;
      i++;
    }
    return `${n.toFixed(n >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
  }

  function formatMtime(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    const epoch = Math.floor(d.getTime() / 1000);
    return formatDateTime(epoch);
  }

  function el(id) {
    return document.getElementById(id);
  }

  function escapeHtml(s) {
    return String(s || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  // Session info -----------------------------------------------------------

  function setSysInfo(data) {
    const node = el("sysinfo");
    if (!data) {
      node.textContent = "no session loaded";
      return;
    }
    node.innerHTML = `
      <strong>${escapeHtml(data.hostname || "-")}</strong>
      <span class="muted small">/ ${escapeHtml(data.kernel || "")}</span>
      <span class="muted small">/ ${escapeHtml(data.filename || "")}</span>
      <span class="muted small">/ ${data.sample_count} samples${data.source ? ` (${escapeHtml(data.source)})` : ""}</span>
    `;
  }

  // Progress bar -----------------------------------------------------------

  // ``progressMax`` latches the highest percent we have ever shown for the
  // current job so the bar never moves backwards, even if a late poll
  // response carries a stale payload.
  let progressMax = 0;

  function showProgress() {
    progressMax = 0;
    el("progress-bar-wrap").hidden = false;
    el("progress-retry").hidden = true;
    el("progress-error").hidden = true;
    const detail = el("progress-detail");
    if (detail) {
      detail.hidden = true;
      detail.textContent = "";
    }
  }

  function hideProgress() {
    el("progress-bar-wrap").hidden = true;
    el("progress-error").hidden = true;
    el("progress-retry").hidden = true;
    const detail = el("progress-detail");
    if (detail) {
      detail.hidden = true;
      detail.textContent = "";
    }
  }

  function setProgress(stage, percent, errorMessage, detailText) {
    el("progress-stage").textContent = stage || "";
    // Lock in a monotonically increasing value unless an error reset the
    // bar to 100.
    let value = Math.max(0, Math.min(100, percent || 0));
    if (errorMessage) {
      value = 100;
    } else {
      value = Math.max(progressMax, value);
      progressMax = value;
    }
    el("progress-percent").textContent = `${Math.round(value)}%`;
    el("progress-fill").style.width = `${value}%`;
    const errorNode = el("progress-error");
    const detailNode = el("progress-detail");
    if (errorMessage) {
      errorNode.textContent = errorMessage;
      errorNode.hidden = false;
      el("progress-retry").hidden = !state.retry;
      el("progress-fill").classList.add("error");
      if (detailNode) {
        detailNode.hidden = true;
        detailNode.textContent = "";
      }
    } else {
      errorNode.hidden = true;
      el("progress-retry").hidden = true;
      el("progress-fill").classList.remove("error");
      if (detailNode) {
        if (detailText) {
          detailNode.textContent = detailText;
          detailNode.hidden = false;
        } else {
          detailNode.hidden = true;
          detailNode.textContent = "";
        }
      }
    }
  }

  // Fallback English labels when the server has not filled stage_label yet
  // (this can happen for the initial upload/queued phases that are emitted
  // from the HTTP handler before the parser kicks in). The backend owns the
  // English labels for the parser stages themselves.
  const STAGE_LABELS = {
    pending: "Queued",
    upload_saved: "Uploaded, preparing",
    header: "Validating header",
    scanning: "Scanning records",
    decoding_sstat: "Decoding system stats",
    decoding_tstat: "Decoding process stats",
    parsing: "Parsing rawlog",
    building_samples: "Building samples",
    done: "Done",
    error: "Failed",
  };

  function stageLabel(stage, serverLabel) {
    if (serverLabel) return serverLabel;
    return STAGE_LABELS[stage] || stage || "working";
  }

  // Job polling ------------------------------------------------------------

  function stopPolling() {
    if (state.pollTimer) {
      clearTimeout(state.pollTimer);
      state.pollTimer = null;
    }
  }

  function setBusy(value) {
    state.busy = value;
    const closeBtn = el("picker-close");
    if (closeBtn) closeBtn.disabled = value;
  }

  async function pollJob(jobId) {
    stopPolling();
    state.lastJob = jobId;
    state.currentJobId = jobId;
    setBusy(true);

    const tick = async () => {
      let res;
      try {
        res = await fetch(`api/jobs/${encodeURIComponent(jobId)}`);
      } catch (err) {
        setBusy(false);
        setProgress("Error", 100, `Network error: ${err.message || err}`);
        return;
      }
      if (res.status === 404) {
        setBusy(false);
        setProgress("Error", 100, "Job expired or not found.");
        return;
      }
      if (!res.ok) {
        setBusy(false);
        const msg = await res.text();
        setProgress("Error", 100, `Job poll failed: ${msg}`);
        return;
      }
      const job = await res.json();
      if (job.status === "done" && job.result) {
        setProgress(stageLabel("done", job.stage_label), 100, null, null);
        setTimeout(hideProgress, 800);
        setBusy(false);
        onParseComplete(job.result);
        return;
      }
      if (job.status === "error") {
        setBusy(false);
        setProgress("Error", 100, job.error || "parse failed");
        return;
      }
      setProgress(
        stageLabel(job.stage, job.stage_label),
        job.progress || 0,
        null,
        job.detail || null,
      );
      state.pollTimer = setTimeout(tick, 500);
    };

    tick();
  }

  async function onParseComplete(result) {
    state.session = result.session;
    setSysInfo(result);

    const summaryRes = await fetch(`api/summary?session=${encodeURIComponent(result.session)}`);
    const summary = await summaryRes.json();
    state.lastSummary = summary;
    const tr = summary && summary.time_range ? summary.time_range : {};
    state.capture.intervalSeconds = tr.interval_seconds ?? null;
    state.capture.minRangeSeconds = tr.recommended_min_range_seconds || 300;
    setupTimeRange(summary);
    await loadSamples();
    // Reset per session chat state so history does not leak across captures.
    state.chatHistory = [];
    const chatLog = el("chat-log");
    if (chatLog) {
      chatLog.innerHTML = '<div class="chat-empty">Ask about CPU, memory, disk or network. Try "find problem points in the whole log" or pick a time range and ask "whats wrong in this slice?"</div>';
    }
    refreshChatVisibility();
    updateChatRangeBadge();
    // Fire and forget: the briefing card updates itself when the response
    // arrives. A slow or failing LLM must not block the rest of the UI.
    requestBriefing(state.currentJobId);
  }

  // AI briefing ------------------------------------------------------------

  function briefingShouldRender() {
    const health = state.llmHealth;
    if (!health) return false;
    if (!health.ok) return false;
    if (health.provider === "none") return false;
    return true;
  }

  function severityLabel(sev) {
    if (sev === "critical") return "CRITICAL";
    if (sev === "warning") return "WARN";
    return "INFO";
  }

  function renderBriefingLoading() {
    const section = el("briefing");
    if (!section) return;
    section.hidden = !briefingShouldRender();
    const body = el("briefing-body");
    if (body) body.innerHTML = '<span class="muted">generating briefing...</span>';
    const meta = el("briefing-meta");
    if (meta && state.llmHealth) {
      const parts = [state.llmHealth.provider];
      if (state.llmHealth.model) parts.push(state.llmHealth.model);
      meta.textContent = parts.join(" / ");
    }
  }

  function renderBriefing(payload) {
    const section = el("briefing");
    const body = el("briefing-body");
    if (!section || !body) return;
    section.hidden = !briefingShouldRender();
    const issues = (payload && payload.issues) || [];
    if (!issues.length) {
      body.innerHTML = '<div class="briefing-empty">No notable issues detected.</div>';
      return;
    }
    body.innerHTML = issues
      .map((issue) => {
        const sev = (issue.severity || "info").toLowerCase();
        const hint = issue.metric_hint
          ? `<div class="briefing-hint">${escapeHtml(issue.metric_hint)}</div>`
          : "";
        return `
          <div class="briefing-issue severity-${escapeHtml(sev)}">
            <div class="briefing-title">
              <span class="severity-badge">${escapeHtml(severityLabel(sev))}</span>
              <span>${escapeHtml(issue.title || "")}</span>
            </div>
            <div class="briefing-detail">${escapeHtml(issue.detail || "")}</div>
            ${hint}
          </div>
        `;
      })
      .join("");
  }

  function renderBriefingError(msg) {
    const section = el("briefing");
    const body = el("briefing-body");
    if (!section || !body) return;
    section.hidden = !briefingShouldRender();
    body.innerHTML = `<div class="briefing-error">Briefing failed: ${escapeHtml(msg || "unknown error")}</div>`;
  }

  async function fetchLlmHealth() {
    try {
      const res = await fetch("api/llm/health");
      if (!res.ok) {
        state.llmHealth = { ok: false, provider: "none" };
      } else {
        state.llmHealth = await res.json();
      }
    } catch (err) {
      state.llmHealth = { ok: false, provider: "none", detail: String(err) };
    }
    const section = el("briefing");
    if (section) section.hidden = !briefingShouldRender();
    refreshChatVisibility();
  }

  async function requestBriefing(jobId) {
    if (!jobId) return;
    if (!briefingShouldRender()) return;
    renderBriefingLoading();
    try {
      const res = await fetch(`api/jobs/${encodeURIComponent(jobId)}/briefing`, {
        method: "POST",
      });
      if (!res.ok) {
        let text;
        try {
          const payload = await res.json();
          text = payload.detail || JSON.stringify(payload);
        } catch {
          text = await res.text();
        }
        renderBriefingError(`HTTP ${res.status}: ${text}`);
        return;
      }
      const data = await res.json();
      if (data.status === "error") {
        renderBriefingError(data.error || "provider returned an error");
        return;
      }
      renderBriefing(data);
    } catch (err) {
      renderBriefingError(err.message || String(err));
    }
  }

  function setupBriefing() {
    const btn = el("briefing-regen");
    if (btn) {
      btn.addEventListener("click", () => {
        if (state.currentJobId) requestBriefing(state.currentJobId);
      });
    }
    // Kick off the health probe so the card becomes visible as soon as we
    // know the server has a real provider configured.
    fetchLlmHealth();
  }

  // AI chat ----------------------------------------------------------------

  function chatShouldRender() {
    const health = state.llmHealth;
    if (!health) return false;
    if (!health.ok) return false;
    if (health.provider === "none") return false;
    return Boolean(state.currentJobId);
  }

  function refreshChatVisibility() {
    // The header icon toggle is always exposed when a session is loaded;
    // aria-pressed reflects the panel's open/closed state so the button
    // also acts as the close affordance (in addition to the in-panel
    // close link).
    const toggle = el("chat-toggle");
    const canRender = chatShouldRender();
    if (toggle) {
      toggle.hidden = !canRender;
      toggle.setAttribute("aria-hidden", canRender ? "false" : "true");
      toggle.setAttribute("aria-pressed", String(state.chatOpen));
    }
    const panel = el("chat-panel");
    if (panel && !canRender) {
      panel.hidden = true;
      state.chatOpen = false;
      document.body.removeAttribute("data-chat-open");
    }
  }

  function openChat() {
    if (!chatShouldRender()) return;
    const panel = el("chat-panel");
    if (!panel) return;
    panel.hidden = false;
    state.chatOpen = true;
    document.body.setAttribute("data-chat-open", "1");
    refreshChatVisibility();
    updateChatRangeBadge();
    const input = el("chat-input");
    if (input) setTimeout(() => input.focus(), 50);
  }

  function closeChat() {
    const panel = el("chat-panel");
    if (panel) panel.hidden = true;
    state.chatOpen = false;
    document.body.removeAttribute("data-chat-open");
    refreshChatVisibility();
  }

  function updateChatRangeBadge() {
    const badge = el("chat-range-badge");
    if (!badge) return;
    const from = state.timeRange.from;
    const to = state.timeRange.to;
    if (from && to) {
      badge.textContent = `Range ${formatDateTime(from)} .. ${formatDateTime(to)}`;
      badge.classList.add("active");
    } else {
      badge.textContent = "Full range";
      badge.classList.remove("active");
    }
  }

  function appendChatMessage(role, text) {
    const log = el("chat-log");
    if (!log) return null;
    const empty = log.querySelector(".chat-empty");
    if (empty) empty.remove();
    const wrap = document.createElement("div");
    wrap.className = `chat-message ${role}`;
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    bubble.textContent = text || "";
    wrap.appendChild(bubble);
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
    return bubble;
  }

  function createAssistantPlaceholderBubble() {
    // Renders an "assistant is thinking" bubble in the chat log. The
    // bubble carries two distinct children:
    //   * ``.chat-bubble-thinking`` spinner+label (removed on first
    //     token / range_hint)
    //   * ``.chat-bubble-text`` (added by ``swapPlaceholderToAssistant``
    //     on the swap and kept as a dedicated child so streaming token
    //     updates mutate only the text without touching sibling range
    //     hint badges).
    // Keeping these as separate children prevents the scroll position
    // from jumping when the spinner is replaced, and - critically -
    // isolates text updates from badge DOM so ``appendRangeHintBadge``
    // results survive the rest of the stream.
    const log = el("chat-log");
    if (!log) return null;
    const empty = log.querySelector(".chat-empty");
    if (empty) empty.remove();
    const wrap = document.createElement("div");
    wrap.className = "chat-message assistant";
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    const thinking = document.createElement("span");
    thinking.className = "chat-bubble-thinking";
    thinking.dataset.tokenCount = "0";
    const spinner = document.createElement("span");
    spinner.className = "chat-spinner";
    const label = document.createElement("span");
    label.className = "chat-bubble-thinking-label";
    label.textContent = "thinking...";
    thinking.appendChild(spinner);
    thinking.appendChild(label);
    bubble.appendChild(thinking);
    wrap.appendChild(bubble);
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
    return bubble;
  }

  function setPlaceholderTokenCount(bubble, tokenCount) {
    if (!bubble) return;
    const thinking = bubble.querySelector(".chat-bubble-thinking");
    if (!thinking) return;
    const label = thinking.querySelector(".chat-bubble-thinking-label");
    if (!label) return;
    thinking.dataset.tokenCount = String(tokenCount);
    label.textContent = tokenCount > 0
      ? `responding... (${tokenCount} tokens)`
      : "thinking...";
  }

  function getOrCreateTextSpan(bubble) {
    // Returns the dedicated ``.chat-bubble-text`` child, creating it on
    // first use. Callers use this to mutate only the streaming text
    // portion without clobbering sibling range hint badges. Must NOT
    // clear bubble.textContent, which would delete those siblings.
    let span = bubble.querySelector(".chat-bubble-text");
    if (!span) {
      span = document.createElement("span");
      span.className = "chat-bubble-text";
      bubble.appendChild(span);
    }
    return span;
  }

  function swapPlaceholderToAssistant(bubble, text) {
    // First real content (token or range_hint) arrived; remove the
    // spinner child but leave any other siblings (badges) intact. The
    // streaming text lives in ``.chat-bubble-text``, populated here and
    // then updated by later tokens via ``getOrCreateTextSpan``.
    if (!bubble) return;
    const thinking = bubble.querySelector(".chat-bubble-thinking");
    if (thinking) thinking.remove();
    const span = getOrCreateTextSpan(bubble);
    span.textContent = text;
  }

  function updateAssistantBubbleText(bubble, text) {
    // Token-stream helper: update the text child in place, leaving any
    // badges that ``appendRangeHintBadge`` added as siblings untouched.
    if (!bubble) return;
    const span = getOrCreateTextSpan(bubble);
    span.textContent = text;
  }

  function removePlaceholderBubble(bubble) {
    if (!bubble) return;
    const wrap = bubble.parentElement;
    if (wrap && wrap.classList.contains("chat-message")) wrap.remove();
  }

  function renderMarkdown(text) {
    // Sanitized markdown render used only for assistant replies; user input
    // stays as plain textContent. Returns the original text if either
    // dependency failed to load so the bubble never shows raw script tags
    // or an empty node, which would be worse than un-formatted output.
    if (!text) return "";
    if (!window.marked || !window.DOMPurify) return text;
    try {
      const html = window.marked.parse(text, { async: false });
      return window.DOMPurify.sanitize(html);
    } catch (err) {
      return text;
    }
  }

  function finalizeAssistantBubble(bubble, text) {
    // Replace only the streaming text child with its sanitized markdown
    // render. Sibling range hint badges stay in place so we no longer
    // need the snapshot+reattach dance in ``sendChatMessage``.
    if (!bubble) return;
    const span = getOrCreateTextSpan(bubble);
    const html = renderMarkdown(text);
    if (html === text) {
      span.textContent = text;
    } else {
      span.innerHTML = html;
    }
  }

  function appendChatError(text) {
    const log = el("chat-log");
    if (!log) return;
    const empty = log.querySelector(".chat-empty");
    if (empty) empty.remove();
    const wrap = document.createElement("div");
    wrap.className = "chat-message error";
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";
    bubble.textContent = text;
    wrap.appendChild(bubble);
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
  }

  function appendRangeHintBadge(parentBubble, hint) {
    if (!parentBubble) return;
    const startEpoch = parseIsoToEpoch(hint.start);
    const endEpoch = parseIsoToEpoch(hint.end);
    const labelStart = startEpoch ? formatDateTime(startEpoch) : hint.start || "?";
    const labelEnd = endEpoch ? formatDateTime(endEpoch) : hint.end || "?";
    const reason = (hint.reason || "").trim();
    // Badge label priority:
    //   1. explicit label from <range label='X'/> (hint.label)
    //   2. context keyword resolved by the auto scanner (hint.autoLabel)
    //   3. reason (short description the LLM chose to attach)
    // The ``auto`` flag marks fallback badges so we can render a different
    // prefix without misattributing labels that the LLM explicitly set.
    const explicitLabel = (hint.label || "").trim();
    const autoLabel = (hint.autoLabel || "").trim();
    const headLabel = explicitLabel || autoLabel;
    const shortReason = reason ? truncate(reason, 40) : "";
    const parts = [`${labelStart} .. ${labelEnd}`];
    if (headLabel) parts.push(headLabel);
    else if (shortReason) parts.push(shortReason);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chat-hint-badge";
    btn.textContent = parts.join(" - ");
    // Record the epoch pair as data attributes so the ISO fallback scan
    // (see ``scanAssistantTextForRangeHints``) can dedupe against badges
    // already created by the ``range_hint`` SSE path.
    if (startEpoch) btn.dataset.startEpoch = String(startEpoch);
    if (endEpoch) btn.dataset.endEpoch = String(endEpoch);
    const titleParts = [`${labelStart} .. ${labelEnd}`];
    if (explicitLabel) titleParts.push(`label: ${explicitLabel}`);
    if (autoLabel && !explicitLabel) titleParts.push(`auto: ${autoLabel}`);
    if (reason) titleParts.push(reason);
    titleParts.push("Click to zoom charts to this range.");
    btn.title = titleParts.join("\n");
    btn.addEventListener("click", () => {
      if (!startEpoch || !endEpoch) return;
      const widened = widenRangeIfNarrow(startEpoch, endEpoch);
      const fromInput = el("range-from");
      const toInput = el("range-to");
      if (fromInput) fromInput.value = formatDateTime(widened.from);
      if (toInput) toInput.value = formatDateTime(widened.to);
      applyRange(widened.from, widened.to);
      updateChatRangeBadge();
      if (widened.widened) {
        const sec = state.capture.minRangeSeconds || 0;
        const interval = state.capture.intervalSeconds;
        const detail = interval
          ? `${sec}s (${interval}s sample interval)`
          : `${sec}s`;
        showChatToast(`Range was narrower than the sample interval; expanded to ${detail}.`);
      }
      // Visual feedback: flash the badge border so the user sees which
      // hint was activated, especially useful when several badges stack
      // up in a single reply.
      btn.classList.remove("chat-hint-badge-flash");
      // Force reflow so the animation restarts on repeat clicks.
      void btn.offsetWidth;
      btn.classList.add("chat-hint-badge-flash");
    });
    parentBubble.appendChild(document.createElement("br"));
    parentBubble.appendChild(btn);
    const log = el("chat-log");
    if (log) log.scrollTop = log.scrollHeight;
  }

  function truncate(text, max) {
    if (!text) return "";
    if (text.length <= max) return text;
    return text.slice(0, Math.max(1, max - 1)) + "…";
  }

  function parseIsoToEpoch(iso) {
    if (!iso || typeof iso !== "string") return null;
    const ms = Date.parse(iso);
    if (Number.isNaN(ms)) return null;
    return Math.floor(ms / 1000);
  }

  // Auto label keyword table. Keys are the lower cased source tokens we
  // look for in the surrounding sentence; values are the canonical label
  // we render on the badge. Keeping Korean + English entries side by side
  // so operators get Korean output when they wrote Korean and English
  // otherwise. Priority for ``resolveAutoRangeLabel`` is: explicit
  // keyword match (first in list wins) -> "구간N" fallback.
  const AUTO_LABEL_KEYWORDS = [
    ["최소값", "최소값"],
    ["최솟값", "최솟값"],
    ["min", "min"],
    ["minimum", "min"],
    ["최대값", "최대값"],
    ["최댓값", "최댓값"],
    ["max", "max"],
    ["maximum", "max"],
    ["피크", "피크"],
    ["peak", "peak"],
    ["스파이크", "스파이크"],
    ["spike", "spike"],
    ["평균", "평균"],
    ["avg", "avg"],
    ["average", "avg"],
    ["시작", "시작"],
    ["start", "start"],
    ["종료", "종료"],
    ["end", "end"],
  ];

  function resolveAutoRangeLabel(ctx) {
    // Pure helper so pytest-level node smoke tests can exercise it
    // without booting a DOM. ``ctx`` shape:
    //   { sentence: string, counter: number, context?: string }
    // Returns the canonical label string. Falls back to "구간N".
    const haystack = (ctx.context || ctx.sentence || "").toLowerCase();
    for (const [needle, label] of AUTO_LABEL_KEYWORDS) {
      if (haystack.includes(needle)) return label;
    }
    return `구간${ctx.counter}`;
  }

  function scanAssistantTextForRangeHints(bubble, text) {
    // Fallback ISO scanner; see the file-top comment block for the
    // rationale and manual test cases. Runs after the stream has settled
    // and after primary-path range_hint badges have been re-appended.
    if (!bubble || !text) return;
    const iso = /\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z/g;
    // Walk sentence by sentence so we can enforce the "same or adjacent
    // sentence" pairing rule. Splitting on the sentence terminators from
    // the spec (., \n, ?, !) is intentionally coarse - ISO timestamps
    // never contain them, so no false sentence breaks land mid-stamp.
    const sentences = text.split(/(?<=[\.\n\?\!])/);
    // Flatten every (iso, sentenceIndex, sentenceText) pair in order so
    // the auto-label resolver can inspect the originating sentence for
    // keywords like "peak" or "최소값".
    const stamps = [];
    for (let i = 0; i < sentences.length; i++) {
      const s = sentences[i];
      const matches = s.match(iso);
      if (!matches) continue;
      for (const m of matches) {
        const epoch = parseIsoToEpoch(m);
        if (epoch) stamps.push({ epoch, sentence: i, sentenceText: s, raw: m });
      }
    }
    if (!stamps.length) return;

    // Existing badges' (start,end) pairs for dedupe.
    const existing = new Set();
    bubble.querySelectorAll(".chat-hint-badge").forEach((b) => {
      const s = parseInt(b.dataset.startEpoch || "", 10);
      const e = parseInt(b.dataset.endEpoch || "", 10);
      if (Number.isFinite(s) && Number.isFinite(e)) {
        existing.add(`${s}:${e}`);
      }
    });

    const minRange = Math.max(state.capture.minRangeSeconds || 300, 60);
    const captureStart = state.timeRange.fullStart;
    const captureEnd = state.timeRange.fullEnd;

    const pairs = [];
    let i = 0;
    while (i < stamps.length) {
      const a = stamps[i];
      const b = stamps[i + 1];
      // Pair if second stamp is in the same or an adjacent sentence.
      if (b && b.sentence - a.sentence <= 1) {
        let start = a.epoch;
        let end = b.epoch;
        if (end < start) { const t = start; start = end; end = t; }
        // Build a small context window: the paired sentences plus up to
        // 20 chars of surrounding prose so keywords that sit just
        // before/after the timestamp still register.
        const startIdx = Math.max(0, text.indexOf(a.raw) - 20);
        const endIdx = Math.min(
          text.length,
          text.indexOf(b.raw, startIdx) + b.raw.length + 20
        );
        const context = `${a.sentenceText} ${b.sentenceText} ${text.slice(startIdx, endIdx)}`;
        pairs.push({ start, end, sentence: a.sentenceText, context });
        i += 2;
      } else {
        // Single unpaired stamp -> expand by minRange / 2 on each side.
        const half = Math.ceil(minRange / 2);
        const rawIdx = text.indexOf(a.raw);
        const startIdx = Math.max(0, rawIdx - 20);
        const endIdx = Math.min(text.length, rawIdx + a.raw.length + 20);
        const context = `${a.sentenceText} ${text.slice(startIdx, endIdx)}`;
        pairs.push({
          start: a.epoch - half,
          end: a.epoch + half,
          sentence: a.sentenceText,
          context,
        });
        i += 1;
      }
    }

    // Per-message counter so the next assistant reply starts at 구간1.
    let counter = 0;
    for (const p of pairs) {
      // Capture-window validation: silently drop any range that doesn't
      // intersect the capture bounds. Mirrors the server-side filter so
      // the two paths agree on what counts as in-range.
      if (captureStart && p.end < captureStart) continue;
      if (captureEnd && p.start > captureEnd) continue;
      const key = `${p.start}:${p.end}`;
      if (existing.has(key)) continue;
      existing.add(key);
      counter += 1;
      const autoLabel = resolveAutoRangeLabel({
        sentence: p.sentence,
        context: p.context,
        counter,
      });
      appendRangeHintBadge(bubble, {
        start: new Date(p.start * 1000).toISOString(),
        end: new Date(p.end * 1000).toISOString(),
        reason: "(auto)",
        autoLabel,
      });
    }
  }

  function setChatBusy(busy) {
    state.chatBusy = busy;
    const input = el("chat-input");
    const send = el("chat-send");
    if (input) input.disabled = busy;
    if (send) send.disabled = busy;
  }

  async function sendChatMessage(message) {
    if (!message || state.chatBusy) return;
    const jobId = state.currentJobId;
    if (!jobId) {
      appendChatError("No analysis session loaded.");
      return;
    }
    appendChatMessage("user", message);
    const assistantBubble = createAssistantPlaceholderBubble();
    setChatBusy(true);

    const body = {
      message,
      time_range: {
        start: state.timeRange.from
          ? new Date(state.timeRange.from * 1000).toISOString()
          : null,
        end: state.timeRange.to
          ? new Date(state.timeRange.to * 1000).toISOString()
          : null,
      },
      history: state.chatHistory.slice(-20),
    };

    let assistantText = "";
    let tokenCount = 0;
    let gotError = null;
    try {
      const res = await fetch(
        `api/jobs/${encodeURIComponent(jobId)}/chat/stream`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
          body: JSON.stringify(body),
        }
      );
      if (!res.ok) {
        let detail;
        try {
          detail = (await res.json()).detail;
        } catch {
          detail = await res.text();
        }
        gotError = `HTTP ${res.status}: ${detail}`;
      } else if (!res.body) {
        gotError = "server returned no streaming body";
      } else {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split("\n\n");
          buffer = frames.pop() || "";
          for (const frame of frames) {
            const parsed = parseSseFrame(frame);
            if (!parsed) continue;
            if (parsed.event === "token") {
              assistantText += parsed.data.text || "";
              tokenCount += 1;
              if (tokenCount === 1) {
                // First token arrived; remove the spinner and seed the
                // text child. From here on, only the text child gets
                // mutated so any range hint badges attached during
                // streaming survive as siblings.
                swapPlaceholderToAssistant(assistantBubble, assistantText);
              } else {
                updateAssistantBubbleText(assistantBubble, assistantText);
              }
              const log = el("chat-log");
              if (log) log.scrollTop = log.scrollHeight;
            } else if (parsed.event === "range_hint") {
              // If a range_hint arrives before any token, we still need
              // to drop the spinner so the badge isn't rendered next to
              // "thinking..." markup. The swap adds an empty text span,
              // the badge is then appended as a sibling.
              if (tokenCount === 0) {
                swapPlaceholderToAssistant(assistantBubble, "");
              }
              appendRangeHintBadge(assistantBubble, parsed.data);
            } else if (parsed.event === "error") {
              gotError = parsed.data.message || "provider error";
            } else if (parsed.event === "done") {
              // meta ignored for now; could show token/char count later
            }
          }
        }
      }
    } catch (err) {
      gotError = err.message || String(err);
    }

    setChatBusy(false);
    if (gotError) {
      if (!assistantText) {
        // Drop the placeholder bubble so only the error remains.
        removePlaceholderBubble(assistantBubble);
      }
      appendChatError(`Chat failed: ${gotError}`);
      return;
    }
    if (!assistantText) {
      // Stream closed without any tokens and no error: drop the idle
      // spinner bubble so the log doesn't accumulate empty placeholders.
      removePlaceholderBubble(assistantBubble);
      return;
    }
    // Render the streamed text as sanitized markdown. Only the
    // ``.chat-bubble-text`` child is rewritten; range hint badges that
    // arrived during the stream stay put as bubble siblings, so no
    // snapshot/reattach dance is needed anymore.
    finalizeAssistantBubble(assistantBubble, assistantText);
    // Fallback ISO scan for range hints the LLM mentioned in prose but did
    // not wrap in a <range/> tag. Dedupes against the primary-path badges
    // via their data-*-epoch attributes.
    scanAssistantTextForRangeHints(assistantBubble, assistantText);
    state.chatHistory.push({ role: "user", content: message });
    state.chatHistory.push({ role: "assistant", content: assistantText });
  }

  function parseSseFrame(frame) {
    if (!frame) return null;
    const lines = frame.split("\n");
    let event = null;
    const dataParts = [];
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        event = line.slice(7);
      } else if (line.startsWith("data: ")) {
        dataParts.push(line.slice(6));
      }
    }
    if (!event) return null;
    const rawData = dataParts.join("\n");
    try {
      return { event, data: JSON.parse(rawData) };
    } catch {
      return { event, data: { text: rawData } };
    }
  }

  function setupChat() {
    const toggle = el("chat-toggle");
    const closeBtn = el("chat-close");
    const resetBtn = el("chat-reset");
    const form = el("chat-form");
    const input = el("chat-input");

    if (toggle) toggle.addEventListener("click", () => {
      if (state.chatOpen) closeChat();
      else openChat();
    });
    if (closeBtn) closeBtn.addEventListener("click", closeChat);
    if (resetBtn) {
      resetBtn.addEventListener("click", () => {
        state.chatHistory = [];
        const log = el("chat-log");
        if (log) {
          log.innerHTML = '<div class="chat-empty">Conversation cleared.</div>';
        }
      });
    }
    if (form && input) {
      form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const text = input.value.trim();
        if (!text) return;
        input.value = "";
        sendChatMessage(text);
      });
      input.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" && !ev.shiftKey) {
          ev.preventDefault();
          form.requestSubmit();
        }
      });
    }
    setupChatResize();
    restoreChatPanelWidth();
  }

  // Chat panel resize (splitter) -----------------------------------------

  const CHAT_WIDTH_KEY = "atop-web:chatPanelWidth";
  const CHAT_WIDTH_MIN = 320;

  function chatMaxWidth() {
    return Math.min(Math.floor(window.innerWidth * 0.7), 900);
  }

  function setChatPanelWidth(pxRaw) {
    const max = chatMaxWidth();
    const px = Math.max(CHAT_WIDTH_MIN, Math.min(max, Math.round(pxRaw)));
    document.documentElement.style.setProperty("--chat-panel-width", `${px}px`);
    return px;
  }

  function restoreChatPanelWidth() {
    // On phones (<768px) the panel is a full width bottom drawer, so the
    // saved width is irrelevant; leave the CSS default alone.
    if (window.innerWidth < 768) return;
    try {
      const raw = window.localStorage.getItem(CHAT_WIDTH_KEY);
      if (!raw) return;
      const px = parseInt(raw, 10);
      if (!Number.isFinite(px)) return;
      setChatPanelWidth(px);
    } catch (e) {
      // ignore
    }
  }

  function setupChatResize() {
    const handle = el("chat-resize-handle");
    if (!handle) return;
    let dragging = false;
    let pointerId = null;

    const onMove = (ev) => {
      if (!dragging) return;
      // Panel is docked to the right edge, so width grows as the pointer
      // moves left. ``window.innerWidth - clientX`` gives the distance
      // from the right edge, which is what the panel width should be.
      const width = window.innerWidth - ev.clientX;
      setChatPanelWidth(width);
      ev.preventDefault();
    };

    const onUp = (ev) => {
      if (!dragging) return;
      dragging = false;
      handle.classList.remove("dragging");
      document.body.classList.remove("chat-resizing");
      if (pointerId !== null) {
        try { handle.releasePointerCapture(pointerId); } catch (e) { /* ignore */ }
        pointerId = null;
      }
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      // Persist the final width so the next session restores it.
      try {
        const px = parseInt(
          getComputedStyle(document.documentElement)
            .getPropertyValue("--chat-panel-width"),
          10
        );
        if (Number.isFinite(px)) {
          window.localStorage.setItem(CHAT_WIDTH_KEY, String(px));
        }
      } catch (e) {
        // ignore
      }
    };

    handle.addEventListener("pointerdown", (ev) => {
      if (window.innerWidth < 768) return;
      dragging = true;
      pointerId = ev.pointerId;
      try { handle.setPointerCapture(pointerId); } catch (e) { /* ignore */ }
      handle.classList.add("dragging");
      document.body.classList.add("chat-resizing");
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      window.addEventListener("pointercancel", onUp);
      ev.preventDefault();
    });

    window.addEventListener("resize", () => {
      // Re-clamp to the current viewport's max so a shrinking window does
      // not leave the panel wider than the 70% cap.
      const current = parseInt(
        getComputedStyle(document.documentElement)
          .getPropertyValue("--chat-panel-width"),
        10
      );
      if (Number.isFinite(current)) setChatPanelWidth(current);
    });
  }

  // Upload + server pick flow ---------------------------------------------

  function uploadWithProgress(file) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "api/upload");
      xhr.upload.onprogress = (ev) => {
        if (ev.lengthComputable) {
          const pct = (ev.loaded / ev.total) * 100;
          setProgress(`Uploading ${Math.round(pct)}%`, pct);
        }
      };
      xhr.onerror = () => reject(new Error("network error"));
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            reject(e);
          }
        } else {
          reject(new Error(xhr.responseText || `HTTP ${xhr.status}`));
        }
      };
      const fd = new FormData();
      fd.append("file", file);
      xhr.send(fd);
    });
  }

  async function startUpload(file) {
    stopPolling();
    closeModal({ force: true });
    showProgress();
    setProgress("Uploading 0%", 0);
    state.retry = () => startUpload(file);

    try {
      const resp = await uploadWithProgress(file);
      if (!resp.job_id) {
        onParseComplete(resp);
        hideProgress();
        return;
      }
      setProgress("Uploaded, preparing", 25);
      pollJob(resp.job_id);
    } catch (err) {
      setProgress("Error", 100, err.message || String(err));
    }
  }

  async function startServerOpen(name) {
    if (!name) return;
    stopPolling();
    closeModal({ force: true });
    showProgress();
    setProgress("Queued", 5);
    state.retry = () => startServerOpen(name);

    try {
      const res = await fetch("api/files/parse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) {
        const msg = await res.text();
        setProgress("Error", 100, `Open failed: ${msg}`);
        return;
      }
      const resp = await res.json();
      if (!resp.job_id) {
        onParseComplete(resp);
        hideProgress();
        return;
      }
      setProgress("Parsing rawlog", 50);
      pollJob(resp.job_id);
    } catch (err) {
      setProgress("Error", 100, err.message || String(err));
    }
  }

  // Modal + tabs -----------------------------------------------------------

  let lastFocus = null;

  function modalElements() {
    return {
      modal: el("picker-modal"),
      overlay: el("picker-overlay"),
      close: el("picker-close"),
      tabServer: el("tab-server"),
      tabUpload: el("tab-upload"),
      panelServer: el("panel-server"),
      panelUpload: el("panel-upload"),
    };
  }

  function focusableElements(root) {
    return Array.from(
      root.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'),
    ).filter((n) => !n.disabled && n.offsetParent !== null);
  }

  function openModal() {
    lastFocus = document.activeElement;
    const { modal } = modalElements();
    modal.hidden = false;
    pickDefaultTab().then(() => {
      const focusables = focusableElements(modal);
      if (focusables.length) focusables[0].focus();
    });
  }

  function closeModal({ force } = {}) {
    if (state.busy && !force) return;
    const { modal } = modalElements();
    if (modal.hidden) return;
    modal.hidden = true;
    if (lastFocus && typeof lastFocus.focus === "function") {
      lastFocus.focus();
    }
  }

  function switchTab(which) {
    const { tabServer, tabUpload, panelServer, panelUpload } = modalElements();
    const serverActive = which === "server";
    tabServer.classList.toggle("active", serverActive);
    tabUpload.classList.toggle("active", !serverActive);
    tabServer.setAttribute("aria-selected", String(serverActive));
    tabUpload.setAttribute("aria-selected", String(!serverActive));
    panelServer.hidden = !serverActive;
    panelUpload.hidden = serverActive;
  }

  async function pickDefaultTab() {
    const data = await renderServerBrowser();
    if (data && data.enabled && data.files && data.files.length) {
      switchTab("server");
    } else {
      switchTab("upload");
    }
    return data;
  }

  function trapFocus(event) {
    const { modal } = modalElements();
    if (modal.hidden) return;
    if (event.key !== "Tab") return;
    const focusables = focusableElements(modal);
    if (!focusables.length) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      last.focus();
      event.preventDefault();
    } else if (!event.shiftKey && document.activeElement === last) {
      first.focus();
      event.preventDefault();
    }
  }

  // Server browser ---------------------------------------------------------

  async function renderServerBrowser() {
    const body = el("server-browser-body");
    const pathNode = el("server-browser-path");
    body.innerHTML = '<span class="muted">loading...</span>';
    pathNode.textContent = "";

    let data;
    try {
      const res = await fetch("api/files");
      data = await res.json();
    } catch (err) {
      body.innerHTML = `<span class="muted">failed to list files: ${escapeHtml(err.message || err)}</span>`;
      return null;
    }

    pathNode.textContent = data.log_dir ? `(${data.log_dir})` : "";

    if (!data.enabled) {
      body.innerHTML = `<div class="muted">Log directory ${escapeHtml(data.log_dir || "")} is not available. Use the Upload tab.</div>`;
      return data;
    }
    if (!data.files || !data.files.length) {
      body.innerHTML = `<div class="muted">No atop log files in ${escapeHtml(data.log_dir)}. Use the Upload tab.</div>`;
      return data;
    }

    const rows = data.files
      .map(
        (f) => `
        <tr data-name="${escapeHtml(f.name)}">
          <td>${escapeHtml(f.name)}</td>
          <td>${escapeHtml(f.date_guess || "")}</td>
          <td class="num">${escapeHtml(formatBytes(f.size))}</td>
          <td class="num">${escapeHtml(formatMtime(f.mtime))}</td>
          <td><button type="button" class="linkish" data-open="${escapeHtml(f.name)}">open</button></td>
        </tr>`,
      )
      .join("");

    body.innerHTML = `
      <table class="browser-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Date</th>
            <th>Size</th>
            <th>Modified</th>
            <th></th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;

    body.querySelectorAll("button[data-open]").forEach((btn) => {
      btn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        startServerOpen(btn.getAttribute("data-open"));
      });
    });
    body.querySelectorAll("tr[data-name]").forEach((tr) => {
      tr.addEventListener("click", () => startServerOpen(tr.getAttribute("data-name")));
    });
    return data;
  }

  // Time range controls ----------------------------------------------------

  function rangeInputEpoch(input) {
    const value = input.value;
    if (!value) return { ok: true, epoch: null };
    const epoch = parseTextDateTime(value);
    if (epoch === null) return { ok: false, epoch: null };
    return { ok: true, epoch };
  }

  function validateRangeInputs() {
    const fromInput = el("range-from");
    const toInput = el("range-to");
    const fromResult = rangeInputEpoch(fromInput);
    const toResult = rangeInputEpoch(toInput);
    fromInput.classList.toggle("invalid", !fromResult.ok);
    toInput.classList.toggle("invalid", !toResult.ok);
    const errorNode = el("range-error");
    const applyBtn = el("range-apply");
    if (!fromResult.ok || !toResult.ok) {
      errorNode.textContent = "Expected YYYY-MM-DD HH:MM:SS";
      errorNode.hidden = false;
      applyBtn.disabled = true;
      return null;
    }
    if (fromResult.epoch && toResult.epoch && fromResult.epoch > toResult.epoch) {
      errorNode.textContent = "From must be earlier than To";
      errorNode.hidden = false;
      applyBtn.disabled = true;
      return null;
    }
    errorNode.hidden = true;
    applyBtn.disabled = false;
    return { from: fromResult.epoch, to: toResult.epoch };
  }

  function setupTimeRange(summary) {
    const wrap = el("time-range");
    wrap.hidden = false;
    const start = summary?.time_range?.start;
    const end = summary?.time_range?.end;
    state.timeRange.fullStart = start || null;
    state.timeRange.fullEnd = end || null;
    state.timeRange.from = null;
    state.timeRange.to = null;
    el("range-from").value = start ? formatDateTime(start) : "";
    el("range-to").value = end ? formatDateTime(end) : "";
    validateRangeInputs();
    updateRangeSummary();
  }

  function updateRangeSummary() {
    const from = state.timeRange.from || state.timeRange.fullStart;
    const to = state.timeRange.to || state.timeRange.fullEnd;
    el("range-summary").textContent =
      from && to ? `range: ${formatDateTime(from)} .. ${formatDateTime(to)} (${state.tz.toUpperCase()})` : "";
  }

  function applyRange(fromEpoch, toEpoch) {
    state.timeRange.from = fromEpoch || null;
    state.timeRange.to = toEpoch || null;
    updateRangeSummary();
    updateChatRangeBadge();
    if (state.session) loadSamples();
  }

  function widenRangeIfNarrow(fromEpoch, toEpoch) {
    // Ensure a clicked range hint contains at least ``minRangeSeconds``
    // of wall time so the filtered query returns samples. Narrow ranges
    // come from the LLM when it ignores ``recommended_min_range_seconds``
    // or from captures where a single spike sample sits between
    // infrequent measurements.
    if (!fromEpoch || !toEpoch) return { from: fromEpoch, to: toEpoch, widened: false };
    const minWidth = Math.max(state.capture.minRangeSeconds || 0, 60);
    const width = toEpoch - fromEpoch;
    if (width >= minWidth) {
      return { from: fromEpoch, to: toEpoch, widened: false };
    }
    const center = Math.floor((fromEpoch + toEpoch) / 2);
    const half = Math.ceil(minWidth / 2);
    let newFrom = center - half;
    let newTo = center + half;
    // Clamp to the capture bounds if we know them; otherwise trust the
    // server to return "no samples in range" for out of band values.
    const fs = state.timeRange.fullStart;
    const fe = state.timeRange.fullEnd;
    if (fs && newFrom < fs) { newTo += fs - newFrom; newFrom = fs; }
    if (fe && newTo > fe) { newFrom -= newTo - fe; newTo = fe; if (fs && newFrom < fs) newFrom = fs; }
    return { from: newFrom, to: newTo, widened: true };
  }

  function showChatToast(message) {
    // Transient toast docked inside the chat log so the feedback lives
    // next to the badge the user clicked; no separate DOM plumbing or
    // aria-live setup needed since the log already announces updates.
    const log = el("chat-log");
    if (!log) return;
    const toast = document.createElement("div");
    toast.className = "chat-toast";
    toast.textContent = message;
    log.appendChild(toast);
    log.scrollTop = log.scrollHeight;
    setTimeout(() => {
      toast.classList.add("fade-out");
      setTimeout(() => toast.remove(), 400);
    }, 3200);
  }

  function attachRangeControls() {
    const fromInput = el("range-from");
    const toInput = el("range-to");
    [fromInput, toInput].forEach((node) => {
      node.addEventListener("input", validateRangeInputs);
      node.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") {
          ev.preventDefault();
          applyFromInputs();
        }
      });
    });

    el("range-apply").addEventListener("click", applyFromInputs);

    document.querySelectorAll("button[data-range]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const preset = btn.getAttribute("data-range");
        if (preset === "all") {
          fromInput.value = state.timeRange.fullStart ? formatDateTime(state.timeRange.fullStart) : "";
          toInput.value = state.timeRange.fullEnd ? formatDateTime(state.timeRange.fullEnd) : "";
          validateRangeInputs();
          applyRange(null, null);
          return;
        }
        if (!state.timeRange.fullEnd) return;
        const end = state.timeRange.fullEnd;
        const deltas = { "15m": 900, "1h": 3600, "3h": 10800, "6h": 21600 };
        const delta = deltas[preset] || 0;
        if (!delta) return;
        const start = Math.max(state.timeRange.fullStart || end - delta, end - delta);
        fromInput.value = formatDateTime(start);
        toInput.value = formatDateTime(end);
        validateRangeInputs();
        applyRange(start, end);
      });
    });
  }

  function applyFromInputs() {
    const validated = validateRangeInputs();
    if (!validated) return;
    applyRange(validated.from, validated.to);
  }

  function buildRangeQuery() {
    const qs = [];
    if (state.timeRange.from)
      qs.push(`from=${encodeURIComponent(new Date(state.timeRange.from * 1000).toISOString())}`);
    if (state.timeRange.to)
      qs.push(`to=${encodeURIComponent(new Date(state.timeRange.to * 1000).toISOString())}`);
    return qs.length ? "&" + qs.join("&") : "";
  }

  // Charts -----------------------------------------------------------------

  function destroyCharts() {
    for (const key of Object.keys(state.charts)) {
      state.charts[key].destroy();
    }
    state.charts = {};
  }

  function themeVar(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function makeLineChart(ctxId, label, labels, values, onPoint) {
    const ctx = el(ctxId).getContext("2d");
    const accent = themeVar("--accent", "#4da3ff");
    const tickColor = themeVar("--chart-tick", "#8a93a6");
    const gridColor = themeVar("--chart-grid", "#2b3350");

    const datasets = Array.isArray(label)
      ? label.map((cfg, i) => ({
          label: cfg.label,
          data: values[i],
          borderColor: cfg.color,
          backgroundColor: cfg.color + "33",
          borderWidth: 1.5,
          pointRadius: 2,
          tension: 0.15,
          fill: false,
        }))
      : [
          {
            label,
            data: values,
            borderColor: accent,
            backgroundColor: accent + "33",
            borderWidth: 1.5,
            pointRadius: 2,
            tension: 0.15,
            fill: true,
          },
        ];

    return new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: {
            display: Array.isArray(label),
            labels: { color: tickColor, boxWidth: 12 },
          },
          tooltip: { mode: "index", intersect: false },
        },
        scales: {
          x: {
            ticks: { color: tickColor, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
            grid: { color: gridColor },
          },
          y: {
            ticks: { color: tickColor },
            grid: { color: gridColor },
            beginAtZero: true,
          },
        },
        onClick: (_evt, elements) => {
          if (!onPoint) return;
          if (elements && elements.length) {
            onPoint(elements[0].index);
          }
        },
      },
    });
  }

  // CPU stack configuration. Order matches the stacked draw order (bottom to
  // top in the chart) and the /proc/stat breakdown.
  const CPU_STACK = [
    { key: "utime", label: "user",    cssVar: "--cpu-user",    fallback: "#4da3ff" },
    { key: "ntime", label: "nice",    cssVar: "--cpu-nice",    fallback: "#8be9a8" },
    { key: "stime", label: "system",  cssVar: "--cpu-sys",     fallback: "#ff8a65" },
    { key: "Itime", label: "irq",     cssVar: "--cpu-irq",     fallback: "#ffb454" },
    { key: "Stime", label: "softirq", cssVar: "--cpu-softirq", fallback: "#e8b86b" },
    { key: "steal", label: "steal",   cssVar: "--cpu-steal",   fallback: "#b06bff" },
    { key: "guest", label: "guest",   cssVar: "--cpu-guest",   fallback: "#ff6bc5" },
    { key: "wtime", label: "wait",    cssVar: "--cpu-wait",    fallback: "#6bd6e8" },
    { key: "itime", label: "idle",    cssVar: "--cpu-idle",    fallback: "#3a4466" },
  ];

  // System memory helpers --------------------------------------------------

  function sysmemUnitLabel() {
    return state.sysmemUnit === "gib" ? "GiB" : "MiB";
  }

  function pagesToUnit(pages, pagesize) {
    if (pages === null || pages === undefined) return null;
    if (!pagesize) return 0;
    const bytes = pages * pagesize;
    const div = state.sysmemUnit === "gib" ? 1024 * 1024 * 1024 : 1024 * 1024;
    return Math.round((bytes / div) * 100) / 100;
  }

  function refreshChartHeaders() {
    const cpuTitle = el("chart-cpu-title");
    if (cpuTitle) {
      cpuTitle.textContent =
        state.cpuUnit === "pct"
          ? "CPU (system wide %)"
          : "CPU (system wide ticks)";
    }
    const memTitle = el("chart-mem-title");
    if (memTitle) {
      const unit = sysmemUnitLabel();
      const viewNames = { memory: "Memory", available: "Available", swap: "Swap" };
      memTitle.textContent = `System memory - ${viewNames[state.sysmemView]} (${unit})`;
    }
    document.querySelectorAll("button[data-cpu-unit]").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-cpu-unit") === state.cpuUnit);
    });
    document.querySelectorAll("button[data-sysmem-view]").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-sysmem-view") === state.sysmemView);
    });
    document.querySelectorAll("button[data-sysmem-unit]").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-sysmem-unit") === state.sysmemUnit);
    });
    document.querySelectorAll("button[data-disk-unit]").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-disk-unit") === state.diskUnit);
    });
  }

  function destroyChart(key) {
    if (state.charts[key]) {
      state.charts[key].destroy();
      delete state.charts[key];
    }
  }

  function destroySysmemChart() {
    destroyChart("mem");
  }

  function setChartPlaceholder(canvasId, placeholderId, chartKey, text) {
    const canvas = el(canvasId);
    const ph = el(placeholderId);
    if (!ph || !canvas) return;
    // The canvas now lives inside a .chart-canvas-wrap sized box; hide the
    // wrapper (when present) so the placeholder can take its space cleanly.
    const wrap = canvas.parentElement && canvas.parentElement.classList.contains("chart-canvas-wrap")
      ? canvas.parentElement
      : canvas;
    if (text) {
      destroyChart(chartKey);
      ph.textContent = text;
      ph.hidden = false;
      wrap.style.display = "none";
    } else {
      ph.hidden = true;
      wrap.style.display = "";
    }
  }

  function setMemPlaceholder(text) {
    setChartPlaceholder("chart-mem", "chart-mem-placeholder", "mem", text);
  }

  function setCpuPlaceholder(text) {
    setChartPlaceholder("chart-cpu", "chart-cpu-placeholder", "cpu", text);
  }

  function setDiskPlaceholder(text) {
    setChartPlaceholder("chart-dsk", "chart-dsk-placeholder", "dsk", text);
  }

  function setNetPlaceholder(text) {
    setChartPlaceholder("chart-net", "chart-net-placeholder", "net", text);
  }

  function setNetPpsPlaceholder(text) {
    setChartPlaceholder("chart-net-pps", "chart-net-pps-placeholder", "netpps", text);
  }

  function stackedDatasets(labels, seriesDefs) {
    // seriesDefs: [{ label, color, data }]
    return seriesDefs.map((def) => ({
      label: def.label,
      data: def.data,
      backgroundColor: def.color + "cc",
      borderColor: def.color,
      borderWidth: 1,
      pointRadius: 0,
      tension: 0.15,
      fill: true,
    }));
  }

  function makeStackedAreaChart(ctxId, labels, seriesDefs, unit, onPoint) {
    const ctx = el(ctxId).getContext("2d");
    const tickColor = themeVar("--chart-tick", "#8a93a6");
    const gridColor = themeVar("--chart-grid", "#2b3350");
    return new Chart(ctx, {
      type: "line",
      data: { labels, datasets: stackedDatasets(labels, seriesDefs) },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: true, labels: { color: tickColor, boxWidth: 12 } },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y} ${unit}`,
            },
          },
        },
        scales: {
          x: {
            ticks: { color: tickColor, maxRotation: 0, autoSkip: true, maxTicksLimit: 8 },
            grid: { color: gridColor },
          },
          y: {
            stacked: true,
            ticks: { color: tickColor },
            grid: { color: gridColor },
            beginAtZero: true,
          },
        },
        onClick: (_evt, elements) => {
          if (!onPoint || !elements || !elements.length) return;
          onPoint(elements[0].index);
        },
      },
    });
  }

  function renderSysmemChart(onPoint) {
    destroySysmemChart();
    const sysmem = state.sysmem;
    if (!sysmem || !sysmem.samples || !sysmem.samples.length || !sysmem.pagesize) {
      setMemPlaceholder("System memory data not available for this rawlog.");
      return;
    }
    setMemPlaceholder(null);
    const pagesize = sysmem.pagesize;
    const unit = sysmemUnitLabel();
    const labels = sysmem.samples.map((s) => formatChartTick(s.curtime));

    if (state.sysmemView === "memory") {
      const used = sysmem.samples.map((s) =>
        pagesToUnit(
          Math.max(s.physmem - s.freemem - s.cachemem - s.buffermem - s.slabmem, 0),
          pagesize,
        ),
      );
      const cache = sysmem.samples.map((s) => pagesToUnit(s.cachemem, pagesize));
      const buff = sysmem.samples.map((s) => pagesToUnit(s.buffermem, pagesize));
      const slab = sysmem.samples.map((s) => pagesToUnit(s.slabmem, pagesize));
      const free = sysmem.samples.map((s) => pagesToUnit(s.freemem, pagesize));
      const seriesDefs = [
        { label: "used", color: themeVar("--mem-used", "#ff8a65"), data: used },
        { label: "cache", color: themeVar("--mem-cache", "#4da3ff"), data: cache },
        { label: "buff", color: themeVar("--mem-buff", "#9fbfff"), data: buff },
        { label: "slab", color: themeVar("--mem-slab", "#8a93a6"), data: slab },
        { label: "free", color: themeVar("--mem-free", "#8be9a8"), data: free },
      ];
      state.charts.mem = makeStackedAreaChart("chart-mem", labels, seriesDefs, unit, onPoint);
      return;
    }

    if (state.sysmemView === "available") {
      const avail = sysmem.samples.map((s) => pagesToUnit(s.availablemem, pagesize));
      // atop 2.7 rawlogs predate the kernel MemAvailable counter, so every
      // sample carries availablemem=null. Show an explicit notice instead
      // of an empty chart so operators know the field is missing, not zero.
      const everyMissing = avail.length > 0 && avail.every((v) => v === null || v === undefined);
      if (everyMissing) {
        setMemPlaceholder(
          "available 값은 atop 2.7 로그에서는 지원되지 않습니다 (데이터 없음)."
        );
        return;
      }
      state.charts.mem = makeLineChart("chart-mem", `available ${unit}`, labels, avail, onPoint);
      return;
    }

    // swap view
    if (!sysmem.swap_configured) {
      setMemPlaceholder("Swap is not configured on this host (totswap=0 in all samples).");
      return;
    }
    const swapUsed = sysmem.samples.map((s) =>
      pagesToUnit(Math.max(s.totswap - s.freeswap - s.swapcached, 0), pagesize),
    );
    const swapCached = sysmem.samples.map((s) => pagesToUnit(s.swapcached, pagesize));
    const swapFree = sysmem.samples.map((s) => pagesToUnit(s.freeswap, pagesize));
    const seriesDefs = [
      { label: "swap used", color: themeVar("--mem-swap-used", "#ff8a65"), data: swapUsed },
      { label: "swap cached", color: themeVar("--mem-swap-cached", "#ffb454"), data: swapCached },
      { label: "swap free", color: themeVar("--mem-swap-free", "#8be9a8"), data: swapFree },
    ];
    state.charts.mem = makeStackedAreaChart("chart-mem", labels, seriesDefs, unit, onPoint);
  }

  // Disk helpers ------------------------------------------------------------

  function allDeviceNames(sysdsk) {
    if (!sysdsk || !sysdsk.devices) return [];
    const out = [];
    for (const kind of ["disks", "mdds", "lvms"]) {
      for (const name of sysdsk.devices[kind] || []) {
        if (!out.includes(name)) out.push(name);
      }
    }
    return out;
  }

  function ensureDiskDevice() {
    const select = el("disk-device");
    const names = allDeviceNames(state.sysdsk);
    if (select) {
      select.innerHTML = names.length
        ? names.map((n) => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join("")
        : `<option value="">(no devices)</option>`;
    }
    if (!names.length) {
      state.diskDevice = null;
      return;
    }
    if (!state.diskDevice || !names.includes(state.diskDevice)) {
      state.diskDevice = names[0];
    }
    if (select) select.value = state.diskDevice;
  }

  function findDeviceInSample(entry, name) {
    if (!entry) return null;
    for (const kind of ["disks", "mdds", "lvms"]) {
      for (const d of entry[kind] || []) {
        if (d.name === name) return d;
      }
    }
    return null;
  }

  function diskRateSeries() {
    const sysdsk = state.sysdsk;
    const name = state.diskDevice;
    if (!sysdsk || !sysdsk.samples || !sysdsk.samples.length || !name) {
      return { labels: [], read: [], write: [], unitLabel: "", deviceMissing: true };
    }
    const unitLabel = state.diskUnit === "mibs" ? "MiB/s" : "IOPS";
    const labels = [];
    const read = [];
    const write = [];
    // Rate calculation needs deltas between consecutive samples. The first
    // sample in any window cannot produce a rate, so we show it as null and
    // Chart.js leaves a gap.
    let prev = null;
    for (const entry of sysdsk.samples) {
      labels.push(formatChartTick(entry.curtime));
      const cur = findDeviceInSample(entry, name);
      if (!prev || !cur) {
        read.push(null);
        write.push(null);
        prev = { entry, dev: cur };
        continue;
      }
      const prevDev = findDeviceInSample(prev.entry, name);
      const dt = entry.interval > 0 ? entry.interval : (entry.curtime - prev.entry.curtime) || 0;
      if (!prevDev || dt <= 0) {
        read.push(null);
        write.push(null);
      } else if (state.diskUnit === "mibs") {
        // Sectors are 512 B. Convert to MiB/s = (delta_sectors * 512) / 1048576 / interval.
        const factor = 512 / (1024 * 1024);
        const dr = Math.max(cur.nrsect - prevDev.nrsect, 0) * factor / dt;
        const dw = Math.max(cur.nwsect - prevDev.nwsect, 0) * factor / dt;
        read.push(Math.round(dr * 100) / 100);
        write.push(Math.round(dw * 100) / 100);
      } else {
        const dr = Math.max(cur.nread - prevDev.nread, 0) / dt;
        const dw = Math.max(cur.nwrite - prevDev.nwrite, 0) / dt;
        read.push(Math.round(dr * 100) / 100);
        write.push(Math.round(dw * 100) / 100);
      }
      prev = { entry, dev: cur };
    }
    return { labels, read, write, unitLabel, deviceMissing: false };
  }

  function renderDiskChart(onPoint) {
    destroyChart("dsk");
    if (!state.sysdsk || !state.sysdsk.samples || !state.sysdsk.samples.length) {
      setDiskPlaceholder("Disk data not available for this rawlog.");
      return;
    }
    if (!state.diskDevice) {
      setDiskPlaceholder("No block devices recorded in this rawlog.");
      return;
    }
    setDiskPlaceholder(null);

    const series = diskRateSeries();
    const readColor = themeVar("--dsk-read", "#4da3ff");
    const writeColor = themeVar("--dsk-write", "#ff8a65");

    state.charts.dsk = makeLineChart(
      "chart-dsk",
      [
        { label: `read ${series.unitLabel}`, color: readColor },
        { label: `write ${series.unitLabel}`, color: writeColor },
      ],
      series.labels,
      [series.read, series.write],
      onPoint,
    );

    const diskTitle = el("chart-dsk-title");
    if (diskTitle) {
      diskTitle.textContent = `Disk I/O (${state.diskDevice}) ${series.unitLabel}`;
    }
  }

  // Network helpers ---------------------------------------------------------

  function allInterfaceNames(sysnet) {
    if (!sysnet) return [];
    if (Array.isArray(sysnet.interfaces)) return sysnet.interfaces.slice();
    return [];
  }

  function ensureNetInterface() {
    const select = el("net-interface");
    const names = allInterfaceNames(state.sysnet);
    if (select) {
      select.innerHTML = names.length
        ? names.map((n) => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join("")
        : `<option value="">(no interfaces)</option>`;
    }
    if (!names.length) {
      state.netInterface = null;
      return;
    }
    if (!state.netInterface || !names.includes(state.netInterface)) {
      // Prefer the first non-virtual interface so physical links show up by
      // default; fall back to the first one otherwise.
      const firstEntry = state.sysnet.samples && state.sysnet.samples.find((s) => s.interfaces && s.interfaces.length);
      let chosen = null;
      if (firstEntry) {
        for (const iface of firstEntry.interfaces) {
          if (iface.type && iface.type !== "v") {
            chosen = iface.name;
            break;
          }
        }
      }
      state.netInterface = chosen || names[0];
    }
    if (select) select.value = state.netInterface;
  }

  function findInterfaceInSample(entry, name) {
    if (!entry || !entry.interfaces) return null;
    for (const i of entry.interfaces) {
      if (i.name === name) return i;
    }
    return null;
  }

  function netRateSeries() {
    const sysnet = state.sysnet;
    const name = state.netInterface;
    const empty = { labels: [], rx_kbs: [], tx_kbs: [], rx_pps: [], tx_pps: [] };
    if (!sysnet || !sysnet.samples || !sysnet.samples.length || !name) return empty;
    const labels = [];
    const rx_kbs = [];
    const tx_kbs = [];
    const rx_pps = [];
    const tx_pps = [];
    let prev = null;
    for (const entry of sysnet.samples) {
      labels.push(formatChartTick(entry.curtime));
      const cur = findInterfaceInSample(entry, name);
      if (!prev || !cur) {
        rx_kbs.push(null); tx_kbs.push(null); rx_pps.push(null); tx_pps.push(null);
        prev = { entry, iface: cur };
        continue;
      }
      const prevIf = findInterfaceInSample(prev.entry, name);
      const dt = entry.interval > 0 ? entry.interval : (entry.curtime - prev.entry.curtime) || 0;
      if (!prevIf || dt <= 0) {
        rx_kbs.push(null); tx_kbs.push(null); rx_pps.push(null); tx_pps.push(null);
      } else {
        const dRb = Math.max(cur.rbyte - prevIf.rbyte, 0);
        const dSb = Math.max(cur.sbyte - prevIf.sbyte, 0);
        const dRp = Math.max(cur.rpack - prevIf.rpack, 0);
        const dSp = Math.max(cur.spack - prevIf.spack, 0);
        // NIC datasheets use 1000 base; keep the UI aligned with that.
        rx_kbs.push(Math.round((dRb / dt / 1000) * 100) / 100);
        tx_kbs.push(Math.round((dSb / dt / 1000) * 100) / 100);
        rx_pps.push(Math.round((dRp / dt) * 100) / 100);
        tx_pps.push(Math.round((dSp / dt) * 100) / 100);
      }
      prev = { entry, iface: cur };
    }
    return { labels, rx_kbs, tx_kbs, rx_pps, tx_pps };
  }

  function renderNetCharts(onPoint) {
    destroyChart("net");
    destroyChart("netpps");
    if (!state.sysnet || !state.sysnet.samples || !state.sysnet.samples.length) {
      setNetPlaceholder("Network data not available for this rawlog.");
      setNetPpsPlaceholder("Network data not available for this rawlog.");
      return;
    }
    if (!state.netInterface) {
      setNetPlaceholder("No network interfaces recorded in this rawlog.");
      setNetPpsPlaceholder("No network interfaces recorded in this rawlog.");
      return;
    }
    setNetPlaceholder(null);
    setNetPpsPlaceholder(null);

    const series = netRateSeries();
    const rxColor = themeVar("--dsk-read", "#4da3ff");
    const txColor = themeVar("--dsk-write", "#ff8a65");

    state.charts.net = makeLineChart(
      "chart-net",
      [
        { label: "Rx KB/s", color: rxColor },
        { label: "Tx KB/s", color: txColor },
      ],
      series.labels,
      [series.rx_kbs, series.tx_kbs],
      onPoint,
    );
    state.charts.netpps = makeLineChart(
      "chart-net-pps",
      [
        { label: "Rx pps", color: rxColor },
        { label: "Tx pps", color: txColor },
      ],
      series.labels,
      [series.rx_pps, series.tx_pps],
      onPoint,
    );

    const title = el("chart-net-title");
    if (title) {
      title.textContent = `Network I/O (${state.netInterface}) KB/s`;
    }
    const ppsTitle = el("chart-net-pps-title");
    if (ppsTitle) {
      ppsTitle.textContent = `Network packets/s (${state.netInterface})`;
    }
  }

  function renderSyscpuChart(onPoint) {
    destroyChart("cpu");
    const syscpu = state.syscpu;
    if (!syscpu || !syscpu.samples || !syscpu.samples.length) {
      setCpuPlaceholder("System CPU data not available for this rawlog.");
      return;
    }
    setCpuPlaceholder(null);

    const hertz = syscpu.hertz || 100;
    const unitTicks = state.cpuUnit === "ticks";
    const unitLabel = unitTicks ? "ticks" : "%";
    const labels = syscpu.samples.map((s) => formatChartTick(s.curtime));

    const seriesDefs = CPU_STACK.map((layer) => {
      const data = syscpu.samples.map((s) => {
        const ticks = (s.all && s.all[layer.key]) || 0;
        if (unitTicks) return ticks;
        const interval = s.interval || 0;
        const ncpu = s.nrcpu || syscpu.ncpu || 1;
        const denom = hertz * interval * ncpu;
        if (denom <= 0) return null;
        return Math.round((ticks / denom) * 100 * 100) / 100;
      });
      return {
        label: layer.label,
        color: themeVar(layer.cssVar, layer.fallback),
        data,
      };
    });

    state.charts.cpu = makeStackedAreaChart(
      "chart-cpu",
      labels,
      seriesDefs,
      unitLabel,
      onPoint,
    );
  }

  function renderCharts(samples) {
    destroyCharts();
    const labels = samples.timeline.map(formatChartTick);
    const onPoint = (i) => loadProcesses(i);

    renderSyscpuChart(onPoint);

    renderSysmemChart(onPoint);

    renderDiskChart(onPoint);

    renderNetCharts(onPoint);

    refreshChartHeaders();
  }

  async function loadSamples() {
    if (!state.session) return;
    const query = `session=${encodeURIComponent(state.session)}${buildRangeQuery()}`;
    const [samplesRes, sysmemRes, syscpuRes, sysdskRes, sysnetRes] = await Promise.all([
      fetch(`api/samples?${query}`),
      fetch(`api/samples/system_memory?${query}`),
      fetch(`api/samples/system_cpu?${query}`),
      fetch(`api/samples/system_disk?${query}`),
      fetch(`api/samples/system_network?${query}`),
    ]);
    if (!samplesRes.ok) throw new Error(`samples request failed: ${samplesRes.status}`);
    const data = await samplesRes.json();
    state.samples = data;
    state.sysmem = sysmemRes.ok ? await sysmemRes.json() : null;
    state.syscpu = syscpuRes.ok ? await syscpuRes.json() : null;
    state.sysdsk = sysdskRes.ok ? await sysdskRes.json() : null;
    state.sysnet = sysnetRes.ok ? await sysnetRes.json() : null;
    ensureDiskDevice();
    ensureNetInterface();
    renderCharts(data);
    if (data.count > 0) {
      loadProcesses(data.count - 1);
    } else {
      const tbody = el("proc-table").querySelector("tbody");
      tbody.innerHTML = "";
      el("proc-time").textContent = "no samples in range";
    }
  }

  // Processes table --------------------------------------------------------

  const COLUMN_DEFS = [
    { key: "pid",   label: "PID",                sort: "pid",   numeric: true,  tooltip: "Process identifier. atop tstat.gen.pid." },
    { key: "name",  label: "Name",               sort: "name",  numeric: false, tooltip: "Process name (proc/pid/comm, up to 15 chars)." },
    { key: "state", label: "State",              sort: "state", numeric: false, tooltip: "Linux task state: R(run) S(sleep) D(uninterruptible) Z(zombie) T(stopped) I(idle) E(exit)." },
    { key: "thr",   label: "Thr",                sort: "nthr",  numeric: true,  tooltip: "Thread count (tstat.gen.nthr)." },
    { key: "cpu",   label: "CPU % / ticks",      sort: "cpu",   numeric: true,  tooltip: "Primary: percentage of CPU time consumed in this sample interval, computed as (utime+stime) / (hertz x interval x ncpu) x 100. Secondary (gray): raw clock tick accumulator (utime+stime)." },
    { key: "rmem",  label: "RMEM MiB / KiB",     sort: "rmem",  numeric: true,  tooltip: "Primary: resident memory (RSS) converted from KiB to MiB (1024 base). Secondary (gray): raw KiB value. atop photoproc.h struct mem.rmem." },
    { key: "vmem",  label: "VMEM MiB / KiB",     sort: "vmem",  numeric: true,  tooltip: "Primary: virtual memory size (process address space) converted from KiB to MiB (1024 base). Secondary (gray): raw KiB value. atop struct mem.vmem." },
    { key: "dsk",   label: "DSK MiB / sectors",  sort: "dsk",   numeric: true,  tooltip: "Primary: disk read+write during this sample interval, expressed in MiB (sectors x 512 / 1024 / 1024). Secondary (gray): raw 512 byte sector counters (rsz+wsz)." },
    { key: "net",   label: "NET pkts",           sort: "net",   numeric: true,  tooltip: "TCP + UDP packet counters (sent + received) from the atop network accounting layer. Values stay at 0 when the netatop kernel module is not loaded." },
  ];

  function renderProcThead() {
    const trow = el("proc-thead").querySelector("tr");
    trow.innerHTML = "";
    COLUMN_DEFS.forEach((col) => {
      const th = document.createElement("th");
      th.classList.add("sortable");
      if (col.numeric) th.classList.add("num");
      th.setAttribute("data-sort", col.sort);
      th.setAttribute("data-tip", col.tooltip);
      th.setAttribute("tabindex", "0");
      const arrow = state.sort.by === col.sort ? (state.sort.order === "asc" ? " ▲" : " ▼") : "";
      th.innerHTML = `<span class="th-label">${escapeHtml(col.label)}</span><span class="sort-arrow">${arrow}</span>`;
      th.addEventListener("click", () => onSortClick(col.sort));
      th.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          onSortClick(col.sort);
        }
      });
      trow.appendChild(th);
    });
    attachInfoTooltips();
  }

  function onSortClick(sortKey) {
    if (state.sort.by === sortKey) {
      state.sort.order = state.sort.order === "desc" ? "asc" : "desc";
    } else {
      state.sort.by = sortKey;
      state.sort.order = sortKey === "name" || sortKey === "pid" ? "asc" : "desc";
    }
    if (state.currentIndex !== null) loadProcesses(state.currentIndex);
  }

  function renderProcMeta(meta) {
    const node = el("proc-meta");
    if (!meta) {
      node.textContent = "";
      return;
    }
    const parts = [];
    if (meta.hertz) parts.push(`hertz=${meta.hertz}`);
    if (meta.ncpu) parts.push(`ncpu=${meta.ncpu}`);
    if (meta.interval_sec !== undefined && meta.interval_sec !== null)
      parts.push(`interval=${meta.interval_sec}s`);
    node.textContent = parts.join(" ");
  }

  function fmt1(value) {
    if (value === null || value === undefined) return "-";
    return (Math.round(value * 10) / 10).toFixed(1);
  }

  function renderProcesses(data) {
    renderProcMeta(data.meta);
    el("proc-time").textContent = `t=${formatDateTime(data.curtime)} (ndeviat=${data.ndeviat}, nactproc=${data.nactproc}, sort=${data.sort_by} ${data.order})`;

    const tbody = el("proc-table").querySelector("tbody");
    tbody.innerHTML = "";
    for (const p of data.processes) {
      const tr = document.createElement("tr");
      const dskMib = (p.dsk_read_mb || 0) + (p.dsk_write_mb || 0);
      const netPkts =
        (p.tcp_sent || 0) + (p.tcp_recv || 0) + (p.udp_sent || 0) + (p.udp_recv || 0);
      const netBreakdown = `tcp ${fmt.format((p.tcp_sent||0)+(p.tcp_recv||0))} / udp ${fmt.format((p.udp_sent||0)+(p.udp_recv||0))}`;
      tr.innerHTML = `
        <td class="num">${p.pid}</td>
        <td>${escapeHtml(p.name) || ""}</td>
        <td>${escapeHtml(p.state) || ""}</td>
        <td class="num">${p.nthr}</td>
        <td class="num">${p.cpu_pct === null || p.cpu_pct === undefined ? "-" : fmt1(p.cpu_pct)}<span class="raw-hint">${fmt.format(p.cpu_ticks)} ticks</span></td>
        <td class="num">${fmt1(p.rmem_mb)}<span class="raw-hint">${fmt.format(p.rmem_kb)} KiB</span></td>
        <td class="num">${fmt1(p.vmem_mb)}<span class="raw-hint">${fmt.format(p.vmem_kb)} KiB</span></td>
        <td class="num">${fmt1(dskMib)}<span class="raw-hint">${fmt.format(p.dsk_read_sectors + p.dsk_write_sectors)} sectors</span></td>
        <td class="num">${fmt.format(netPkts)}<span class="raw-hint">${netBreakdown}</span></td>
      `;
      tbody.appendChild(tr);
    }
  }

  async function loadProcesses(index) {
    state.currentIndex = index;
    const res = await fetch(
      `api/processes?session=${encodeURIComponent(state.session)}&index=${index}` +
        `&sort_by=${encodeURIComponent(state.sort.by)}&order=${state.sort.order}&limit=200${buildRangeQuery()}`,
    );
    if (!res.ok) return;
    const data = await res.json();
    renderProcesses(data);
    renderProcThead();
  }

  // Info tooltips ----------------------------------------------------------

  let tooltipNode = null;

  function ensureTooltip() {
    if (tooltipNode) return tooltipNode;
    tooltipNode = document.createElement("div");
    tooltipNode.className = "tooltip";
    tooltipNode.setAttribute("role", "tooltip");
    tooltipNode.hidden = true;
    document.body.appendChild(tooltipNode);
    return tooltipNode;
  }

  function showTooltipFor(target) {
    const tip = ensureTooltip();
    tip.textContent = target.getAttribute("data-tip") || "";
    tip.hidden = false;
    const rect = target.getBoundingClientRect();
    const top = rect.bottom + window.scrollY + 6;
    const left = Math.max(
      8,
      Math.min(window.innerWidth - 340, rect.left + window.scrollX - 8),
    );
    tip.style.top = `${top}px`;
    tip.style.left = `${left}px`;
  }

  function hideTooltip() {
    if (tooltipNode) tooltipNode.hidden = true;
  }

  // activeTooltipNode tracks which element (if any) is currently "pinned" by
  // a touch or click, so the next outside tap can clear it. Hover based
  // tooltips never set this, so they dismiss themselves via mouseleave.
  let activeTooltipNode = null;

  function pinTooltip(node) {
    activeTooltipNode = node;
    showTooltipFor(node);
  }

  function unpinTooltip() {
    activeTooltipNode = null;
    hideTooltip();
  }

  function toggleTooltipFor(node) {
    if (activeTooltipNode === node) {
      unpinTooltip();
    } else {
      pinTooltip(node);
    }
  }

  // Dismiss a pinned tooltip when the user taps anywhere else.
  document.addEventListener("click", (ev) => {
    if (!activeTooltipNode) return;
    if (activeTooltipNode.contains(ev.target)) return;
    unpinTooltip();
  }, true);

  function bindHoverTooltip(node) {
    if (node.dataset.tipBound === "1") return;
    node.dataset.tipBound = "1";
    if (!node.hasAttribute("tabindex")) node.setAttribute("tabindex", "0");
    node.addEventListener("mouseenter", () => {
      if (activeTooltipNode) return; // let the pinned one win
      showTooltipFor(node);
    });
    node.addEventListener("mouseleave", () => {
      if (activeTooltipNode === node) return;
      hideTooltip();
    });
    node.addEventListener("focus", () => showTooltipFor(node));
    node.addEventListener("blur", () => {
      if (activeTooltipNode === node) return;
      hideTooltip();
    });
  }

  // Chart <h2> titles have no competing action, so a single tap toggles the
  // tooltip. Mouse users still get the hover behavior.
  function attachChartTooltips() {
    document.querySelectorAll("h2[data-tip]").forEach((node) => {
      bindHoverTooltip(node);
      if (node.dataset.tapBound === "1") return;
      node.dataset.tapBound = "1";
      node.addEventListener("click", (ev) => {
        ev.stopPropagation();
        toggleTooltipFor(node);
      });
    });
  }

  // Table headers double as sort triggers. To avoid stealing the short tap,
  // only a long press (500 ms) surfaces the tooltip on touch. Short taps
  // fall through to the existing sort click handler.
  function attachHeaderTooltips() {
    document.querySelectorAll("thead th[data-tip]").forEach((node) => {
      bindHoverTooltip(node);
      if (node.dataset.longPressBound === "1") return;
      node.dataset.longPressBound = "1";

      let pressTimer = null;
      let suppressClick = false;

      const start = () => {
        suppressClick = false;
        pressTimer = setTimeout(() => {
          pressTimer = null;
          suppressClick = true;
          pinTooltip(node);
        }, 500);
      };
      const cancel = () => {
        if (pressTimer) {
          clearTimeout(pressTimer);
          pressTimer = null;
        }
      };

      node.addEventListener("touchstart", start, { passive: true });
      node.addEventListener("touchend", cancel);
      node.addEventListener("touchmove", cancel, { passive: true });
      node.addEventListener("touchcancel", cancel);
      node.addEventListener("click", (ev) => {
        if (suppressClick) {
          suppressClick = false;
          ev.stopImmediatePropagation();
          ev.preventDefault();
        }
      }, true);
    });
  }

  function attachInfoTooltips() {
    attachHeaderTooltips();
  }

  // Theme -------------------------------------------------------------------

  function detectInitialTheme() {
    try {
      const saved = window.localStorage.getItem("atop-web.theme");
      if (saved === "dark" || saved === "light") return saved;
    } catch (e) {
      // ignore
    }
    try {
      if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
        return "light";
      }
    } catch (e) {
      // ignore
    }
    return "dark";
  }

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    const btn = el("theme-toggle");
    if (btn) {
      // Label shows the destination of the toggle, not the current theme.
      btn.textContent = theme === "dark" ? "☀ Light" : "☾ Dark";
      btn.setAttribute("aria-pressed", String(theme === "light"));
    }
    // Charts cache the tick/grid colors at construction time; rebuild them
    // when the theme flips so the palette catches up.
    if (state.samples) renderCharts(state.samples);
  }

  function setupTheme() {
    const theme = detectInitialTheme();
    applyTheme(theme);
    const btn = el("theme-toggle");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
      try {
        window.localStorage.setItem("atop-web.theme", next);
      } catch (e) {
        // ignore
      }
      applyTheme(next);
    });
  }

  // Chart unit toggles ------------------------------------------------------

  function loadChartUnits() {
    try {
      const cpu = window.localStorage.getItem("atop-web.cpu_unit");
      if (cpu === "ticks" || cpu === "pct") state.cpuUnit = cpu;
      const view = window.localStorage.getItem("atop-web.sysmem_view");
      if (view === "memory" || view === "available" || view === "swap") {
        state.sysmemView = view;
      }
      const unit = window.localStorage.getItem("atop-web.sysmem_unit");
      if (unit === "mib" || unit === "gib") state.sysmemUnit = unit;
      const disk = window.localStorage.getItem("atop-web.disk_unit");
      if (disk === "iops" || disk === "mibs") state.diskUnit = disk;
    } catch (e) {
      // ignore
    }
  }

  function saveChartUnits() {
    try {
      window.localStorage.setItem("atop-web.cpu_unit", state.cpuUnit);
      window.localStorage.setItem("atop-web.sysmem_view", state.sysmemView);
      window.localStorage.setItem("atop-web.sysmem_unit", state.sysmemUnit);
      window.localStorage.setItem("atop-web.disk_unit", state.diskUnit);
    } catch (e) {
      // ignore
    }
  }

  function rerenderMemoryChart() {
    if (!state.sysmem) {
      refreshChartHeaders();
      return;
    }
    renderSysmemChart((i) => loadProcesses(i));
    refreshChartHeaders();
  }

  function rerenderCpuChart() {
    if (!state.syscpu) {
      refreshChartHeaders();
      return;
    }
    renderSyscpuChart((i) => loadProcesses(i));
    refreshChartHeaders();
  }

  function rerenderDiskChart() {
    renderDiskChart((i) => loadProcesses(i));
    refreshChartHeaders();
  }

  function rerenderNetCharts() {
    renderNetCharts((i) => loadProcesses(i));
    refreshChartHeaders();
  }

  function setupChartUnits() {
    loadChartUnits();
    document.querySelectorAll("button[data-cpu-unit]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const next = btn.getAttribute("data-cpu-unit");
        if (next !== state.cpuUnit) {
          state.cpuUnit = next;
          saveChartUnits();
          rerenderCpuChart();
        }
      });
    });
    document.querySelectorAll("button[data-sysmem-view]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const next = btn.getAttribute("data-sysmem-view");
        if (next !== state.sysmemView) {
          state.sysmemView = next;
          saveChartUnits();
          rerenderMemoryChart();
        }
      });
    });
    document.querySelectorAll("button[data-sysmem-unit]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const next = btn.getAttribute("data-sysmem-unit");
        if (next !== state.sysmemUnit) {
          state.sysmemUnit = next;
          saveChartUnits();
          rerenderMemoryChart();
        }
      });
    });
    document.querySelectorAll("button[data-disk-unit]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const next = btn.getAttribute("data-disk-unit");
        if (next !== state.diskUnit) {
          state.diskUnit = next;
          saveChartUnits();
          rerenderDiskChart();
        }
      });
    });
    const diskSelect = el("disk-device");
    if (diskSelect) {
      diskSelect.addEventListener("change", () => {
        state.diskDevice = diskSelect.value || null;
        rerenderDiskChart();
      });
    }
    const netSelect = el("net-interface");
    if (netSelect) {
      netSelect.addEventListener("change", () => {
        state.netInterface = netSelect.value || null;
        rerenderNetCharts();
      });
    }
    refreshChartHeaders();
    attachChartTooltips();
  }

  // Wiring ------------------------------------------------------------------

  function setupTz() {
    loadTz();
    const select = el("tz-select");
    el("tz-local-option").textContent = localOffsetLabel();
    select.value = state.tz;
    select.addEventListener("change", () => {
      state.tz = select.value;
      saveTz();
      if (state.samples) renderCharts(state.samples);
      if (state.lastSummary) {
        const startText = state.lastSummary.time_range?.start ? formatDateTime(state.lastSummary.time_range.start) : "";
        const endText = state.lastSummary.time_range?.end ? formatDateTime(state.lastSummary.time_range.end) : "";
        if (!state.timeRange.from) el("range-from").value = startText;
        if (!state.timeRange.to) el("range-to").value = endText;
        validateRangeInputs();
      }
      updateRangeSummary();
      if (state.currentIndex !== null) loadProcesses(state.currentIndex);
    });
  }

  function setupModalEvents() {
    const { overlay, close, tabServer, tabUpload } = modalElements();
    el("open-picker").addEventListener("click", openModal);
    overlay.addEventListener("click", () => closeModal());
    close.addEventListener("click", () => closeModal());
    tabServer.addEventListener("click", () => switchTab("server"));
    tabUpload.addEventListener("click", () => switchTab("upload"));
    el("server-refresh").addEventListener("click", renderServerBrowser);

    document.addEventListener("keydown", (ev) => {
      const modal = el("picker-modal");
      if (modal.hidden) return;
      if (ev.key === "Escape") {
        closeModal();
        ev.preventDefault();
      } else {
        trapFocus(ev);
      }
    });
  }

  function setupDropZone() {
    const dz = el("drop-zone");
    const input = el("file-input");
    const label = el("file-input-label");
    const submit = el("upload-submit");

    input.addEventListener("change", () => {
      label.textContent = input.files.length ? input.files[0].name : "Choose rawlog file";
    });

    submit.addEventListener("click", () => {
      if (!input.files.length) {
        alert("Choose a rawlog file first.");
        return;
      }
      startUpload(input.files[0]);
    });

    ["dragenter", "dragover"].forEach((evt) => {
      dz.addEventListener(evt, (ev) => {
        ev.preventDefault();
        dz.classList.add("hover");
      });
    });
    ["dragleave", "drop"].forEach((evt) => {
      dz.addEventListener(evt, (ev) => {
        ev.preventDefault();
        dz.classList.remove("hover");
      });
    });
    dz.addEventListener("drop", (ev) => {
      const file = ev.dataTransfer?.files?.[0];
      if (file) {
        startUpload(file);
      }
    });
  }

  el("progress-retry").addEventListener("click", () => {
    if (state.retry) state.retry();
  });

  setupTheme();
  setupTz();
  setupChartUnits();
  setupBriefing();
  setupChat();
  renderProcThead();
  setupModalEvents();
  setupDropZone();
  attachRangeControls();
})();
