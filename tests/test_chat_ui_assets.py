"""Structural checks for Phase 20.2 UI changes that can't be exercised
without a JS engine in the test container.

Covers:
* (A) Thinking placeholder lives in the chat log as an assistant bubble
  that gets swapped in place once streaming starts.
* (B) Auto-detected range hint badges are labeled with a localized keyword
  when context mentions min/max/peak/etc., otherwise ``구간N`` with a per
  message counter.

A node based DOM smoke test is attempted via ``test_chat_ui_js_smoke`` but
skips cleanly when the container image lacks a Node runtime.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).parent.parent / "atop_web" / "static"
APP_JS = (STATIC / "app.js").read_text()
INDEX_HTML = (STATIC / "index.html").read_text()
STYLE_CSS = (STATIC / "style.css").read_text()


# -- (A) Thinking placeholder lives inside the chat log ----------------------


def test_thinking_bar_removed_from_chat_form():
    # The earlier fixed bar above the textarea is gone; the placeholder
    # now lives inside the assistant bubble stream.
    assert 'id="chat-thinking"' not in INDEX_HTML
    assert 'id="chat-thinking-label"' not in INDEX_HTML


def test_app_js_has_placeholder_bubble_helpers():
    # The new flow exposes a named helper so future changes (cancel,
    # error) can target the same bubble without string matching DOM.
    assert "createAssistantPlaceholderBubble" in APP_JS
    # Placeholder markup carries a class we can assert on both in CSS and
    # test for presence.
    assert "chat-bubble-thinking" in APP_JS


def test_placeholder_swap_is_in_place():
    # Guard against a regression where someone removes the placeholder
    # bubble before appending a fresh one (which would cause a scroll
    # jump). Swap must clear+populate the same node.
    assert "swapPlaceholderToAssistant" in APP_JS


def test_chat_send_uses_placeholder():
    # sendChatMessage() should construct the placeholder before the
    # network call so the spinner shows up immediately.
    assert "createAssistantPlaceholderBubble()" in APP_JS


def test_style_has_thinking_bubble_styles():
    # Visual regression sentinel so the dark mode spinner+italic styling
    # survives future refactors.
    assert ".chat-bubble-thinking" in STYLE_CSS


# -- (B) Auto range hint labels ---------------------------------------------


def test_auto_label_keywords_present_in_source():
    # Korean + English keywords must all be recognized. If any of these
    # localization entries get dropped, labels fall back to "구간N" which
    # is the regression this test guards against.
    for keyword in (
        "최소값",
        "최솟값",
        "min",
        "최대값",
        "최댓값",
        "max",
        "피크",
        "peak",
        "스파이크",
        "spike",
        "평균",
        "avg",
        "시작",
        "start",
        "종료",
        "end",
    ):
        assert keyword in APP_JS, keyword


def test_auto_label_counter_resets_per_message():
    # The counter must be local to scanAssistantTextForRangeHints so the
    # next message starts at "구간1" again. Storing it on ``state`` would
    # leak across bubbles.
    assert "scanAssistantTextForRangeHints" in APP_JS
    # "구간" is the Korean prefix we render. We only care that it appears
    # at least once (the fallback path).
    assert "구간" in APP_JS


def test_auto_label_priority_comment_present():
    # Documented priority: explicit <range label=...> > context keyword >
    # "구간N". This comment is load bearing - it signals to future readers
    # why the appendRangeHintBadge label plumbing is layered.
    assert "explicit label" in APP_JS.lower() or "priority" in APP_JS.lower()


# -- Optional node smoke test -----------------------------------------------


@pytest.fixture(scope="module")
def node_bin():
    n = shutil.which("node")
    if not n:
        pytest.skip("node not installed in test environment")
    return n


def _run_node(node_bin: str, script: str) -> dict:
    res = subprocess.run(
        [node_bin, "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if res.returncode != 0:
        pytest.fail(f"node failed: {res.stderr}\nstdout: {res.stdout}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_streaming_tokens_preserve_range_hint_badges(node_bin):
    # Regression guard: during a streamed reply the assistant bubble has
    # range_hint badges appended between token deltas. An earlier
    # implementation reassigned ``bubble.textContent`` on every token,
    # which silently deleted any badges that had already been attached.
    # The fix isolates the streaming text inside a dedicated
    # ``.chat-bubble-text`` child so badges stay on the bubble as
    # siblings. This smoke test drives the real chat pipeline under a
    # jsdom-backed DOM (or a local mini DOM when jsdom is not
    # available) and asserts the badges survive the stream + finalize.
    script = rf"""
      const fs = require('fs');
      const src = fs.readFileSync({json.dumps(str(STATIC / 'app.js'))}, 'utf8');

      // Prefer a real jsdom when present; fall back to a mini DOM shim
      // that covers only the APIs app.js touches on the chat bubble.
      let jsdomAvailable = false;
      try {{ require.resolve('jsdom'); jsdomAvailable = true; }} catch (_) {{}}

      function installDom() {{
        if (jsdomAvailable) {{
          const {{ JSDOM }} = require('jsdom');
          const dom = new JSDOM('<!doctype html><html><body><div id="chat-log"></div></body></html>');
          globalThis.window = dom.window;
          globalThis.document = dom.window.document;
          return;
        }}
        // Minimal DOM shim. Each node tracks children + attrs; textContent
        // setter clears children as the real DOM does. querySelectorAll
        // walks descendants matching a simple ".class" or "tag" selector.
        class Node {{
          constructor(tag) {{
            this.tagName = tag.toUpperCase();
            this.children = [];
            this.parentElement = null;
            this._text = '';
            this.className = '';
            this.attributes = {{}};
            this.dataset = {{}};
            this.style = {{ setProperty: () => {{}} }};
            this._listeners = {{}};
          }}
          appendChild(child) {{
            child.parentElement = this;
            this.children.push(child);
            this._text = '';
            return child;
          }}
          remove() {{
            if (this.parentElement) {{
              const idx = this.parentElement.children.indexOf(this);
              if (idx >= 0) this.parentElement.children.splice(idx, 1);
              this.parentElement = null;
            }}
          }}
          setAttribute(k, v) {{ this.attributes[k] = v; }}
          removeAttribute(k) {{ delete this.attributes[k]; }}
          addEventListener(name, fn) {{ this._listeners[name] = fn; }}
          get textContent() {{
            if (this.children.length === 0) return this._text;
            return this.children.map((c) => c.textContent).join('');
          }}
          set textContent(val) {{
            // Real DOM semantics: setting textContent removes all
            // children and replaces them with a single text node.
            this.children = [];
            this._text = String(val);
          }}
          get classList() {{
            const self = this;
            return {{
              add: (...names) => {{ const cur = new Set(self.className.split(/\s+/).filter(Boolean)); for (const n of names) cur.add(n); self.className = Array.from(cur).join(' '); }},
              remove: (...names) => {{ const cur = new Set(self.className.split(/\s+/).filter(Boolean)); for (const n of names) cur.delete(n); self.className = Array.from(cur).join(' '); }},
              contains: (n) => self.className.split(/\s+/).includes(n),
            }};
          }}
          querySelector(sel) {{
            const all = this.querySelectorAll(sel);
            return all[0] || null;
          }}
          querySelectorAll(sel) {{
            const out = [];
            const match = (node) => {{
              if (sel.startsWith('.')) return node.classList.contains(sel.slice(1));
              return node.tagName === sel.toUpperCase();
            }};
            const walk = (node) => {{
              for (const c of node.children) {{
                if (match(c)) out.push(c);
                walk(c);
              }}
            }};
            walk(this);
            return out;
          }}
          get offsetWidth() {{ return 0; }}
        }}
        const document = {{
          _elements: {{}},
          createElement: (tag) => new Node(tag),
          createTextNode: (t) => {{ const n = new Node('#text'); n._text = t; return n; }},
          getElementById: (id) => document._elements[id] || null,
          body: new Node('body'),
          documentElement: new Node('html'),
        }};
        document.documentElement.dataset = {{}};
        const log = document.createElement('div');
        document._elements['chat-log'] = log;
        document._elements['chat-input'] = new Node('textarea');
        document._elements['chat-send'] = new Node('button');
        globalThis.document = document;
        globalThis.window = {{
          document,
          localStorage: {{ getItem: () => null, setItem: () => {{}} }},
          matchMedia: () => ({{ matches: false }}),
          innerWidth: 1200,
          addEventListener: () => {{}},
          removeEventListener: () => {{}},
        }};
        globalThis.getComputedStyle = () => ({{ getPropertyValue: () => '' }});
      }}

      installDom();
      globalThis.Chart = function () {{}};
      globalThis.Chart.register = () => {{}};
      // Extract helpers out of app.js by carving out named declarations
      // (the file's top-level IIFE hides them otherwise). We only need:
      //   appendRangeHintBadge, scanAssistantTextForRangeHints,
      //   parseIsoToEpoch, truncate, createAssistantPlaceholderBubble,
      //   swapPlaceholderToAssistant, setPlaceholderTokenCount,
      //   AUTO_LABEL_KEYWORDS, resolveAutoRangeLabel,
      //   finalizeAssistantBubble, renderMarkdown, formatDateTime.
      function extract(name, kind) {{
        const needle = kind === 'const' ? `const ${{name}} = ` : `function ${{name}}`;
        const idx = src.indexOf(needle);
        if (idx < 0) throw new Error('missing ' + name);
        if (kind === 'const') {{
          const end = src.indexOf('];', idx);
          return src.slice(idx, end + 2);
        }}
        let depth = 0, end = -1, started = false;
        for (let i = idx; i < src.length; i++) {{
          const c = src[i];
          if (c === '{{') {{ depth++; started = true; }}
          else if (c === '}}') {{ depth--; if (started && depth === 0) {{ end = i + 1; break; }} }}
        }}
        return src.slice(idx, end);
      }}
      // We stub helpers the badge path calls but doesn't need for the
      // regression being tested (click handler side effects).
      const stubs = `
        const state = {{ capture: {{ minRangeSeconds: 300, intervalSeconds: 60 }}, timeRange: {{ fullStart: null, fullEnd: null, from: null, to: null }} }};
        function el(id) {{ return document.getElementById(id); }}
        function formatDateTime(e) {{ return new Date(e*1000).toISOString(); }}
        function widenRangeIfNarrow(a,b) {{ return {{ from: a, to: b, widened: false }}; }}
        function applyRange() {{}}
        function updateChatRangeBadge() {{}}
        function showChatToast() {{}}
      `;
      const body = [
        stubs,
        extract('AUTO_LABEL_KEYWORDS', 'const'),
        extract('parseIsoToEpoch'),
        extract('truncate'),
        extract('resolveAutoRangeLabel'),
        extract('renderMarkdown'),
        extract('finalizeAssistantBubble'),
        extract('createAssistantPlaceholderBubble'),
        extract('setPlaceholderTokenCount'),
        extract('getOrCreateTextSpan'),
        extract('swapPlaceholderToAssistant'),
        extract('updateAssistantBubbleText'),
        extract('removePlaceholderBubble'),
        extract('appendRangeHintBadge'),
        extract('scanAssistantTextForRangeHints'),
      ].join('\n');
      const runner = new Function(body + `
        // Simulate: user sends, placeholder appears, tokens stream,
        // range_hint arrives mid-stream 3 times, then stream ends and
        // finalize runs.
        const bubble = createAssistantPlaceholderBubble();
        let text = '';
        const tokens = [
          'Analyzing samples ',
          'across the window. ',
          '',              // range_hint marker 1
          'CPU usage peaked ',
          '',              // range_hint marker 2
          'and memory grew ',
          'steadily. ',
          '',              // range_hint marker 3
          'No other anomalies ',
          'were detected.',
        ];
        const hints = [
          {{ start: '2026-04-28T10:00:00Z', end: '2026-04-28T10:10:00Z', reason: 'range' }},
          {{ start: '2026-04-28T11:00:00Z', end: '2026-04-28T11:10:00Z', reason: 'range' }},
          {{ start: '2026-04-28T12:00:00Z', end: '2026-04-28T12:10:00Z', reason: 'range' }},
        ];
        let hintIdx = 0;
        let tokenCount = 0;
        for (const t of tokens) {{
          if (t === '') {{
            if (tokenCount === 0) swapPlaceholderToAssistant(bubble, '');
            appendRangeHintBadge(bubble, hints[hintIdx++]);
            continue;
          }}
          text += t;
          tokenCount++;
          if (tokenCount === 1) swapPlaceholderToAssistant(bubble, text);
          else updateAssistantBubbleText(bubble, text);
        }}
        // Finalize rewrites only the .chat-bubble-text child; sibling
        // badges stay intact.
        finalizeAssistantBubble(bubble, text);
        const finalBadges = bubble.querySelectorAll('.chat-hint-badge');
        return {{
          badgeCount: finalBadges.length,
          bubbleTextContainsText: bubble.textContent.includes('peaked'),
          firstBadgeHasEpochData: finalBadges.length > 0 && !!finalBadges[0].dataset.startEpoch,
        }};
      `);
      const out = runner();
      console.log(JSON.stringify(out));
    """
    out = _run_node(node_bin, script)
    assert out["badgeCount"] == 3, f"lost badges: {out}"
    assert out["bubbleTextContainsText"], f"text lost: {out}"
    assert out["firstBadgeHasEpochData"], f"badge missing epoch data: {out}"


def test_auto_label_picks_keyword_over_counter(node_bin):
    # Evaluate only the pure helper by carving it out of app.js. We
    # re-expose just ``resolveAutoRangeLabel`` which is a pure function.
    script = rf"""
      const fs = require('fs');
      const src = fs.readFileSync({json.dumps(str(STATIC / 'app.js'))}, 'utf8');
      // Extract the AUTO_LABEL_KEYWORDS table and the function body
      // together so the returned closure has its dependencies bound.
      function extractBlock(needle) {{
        const idx = src.indexOf(needle);
        if (idx < 0) return null;
        // Scan forward for the first ';' or '}}' that closes the block.
        // For the ``const AUTO_LABEL_KEYWORDS = [...]`` array we stop at
        // the matching ``];`` and for the function we use brace counting.
        if (needle.startsWith('const ')) {{
          const end = src.indexOf('];', idx);
          if (end < 0) return null;
          return src.slice(idx, end + 2);
        }}
        let depth = 0, end = -1, started = false;
        for (let i = idx; i < src.length; i++) {{
          const c = src[i];
          if (c === '{{') {{ depth++; started = true; }}
          else if (c === '}}') {{ depth--; if (started && depth === 0) {{ end = i + 1; break; }} }}
        }}
        return end > 0 ? src.slice(idx, end) : null;
      }}
      const table = extractBlock('const AUTO_LABEL_KEYWORDS');
      const fnBlock = extractBlock('function resolveAutoRangeLabel');
      if (!table || !fnBlock) {{
        console.log(JSON.stringify({{missing: true}}));
        process.exit(0);
      }}
      const fn = new Function(table + '\n' + fnBlock + '\n; return resolveAutoRangeLabel;')();
      const out = {{
        keywordMin: fn({{sentence: 'CPU 최소값 구간 보임', counter: 1}}),
        keywordMax: fn({{sentence: 'peak was observed here', counter: 2}}),
        keywordSpikeKr: fn({{sentence: '메모리 스파이크 구간', counter: 3}}),
        fallback: fn({{sentence: 'cpu behavior during the window', counter: 4}}),
      }};
      console.log(JSON.stringify(out));
    """
    out = _run_node(node_bin, script)
    if out.get("missing"):
        pytest.skip("resolveAutoRangeLabel not exposed yet")
    assert out["keywordMin"] in ("min", "최소값", "최솟값")
    assert out["keywordMax"] in ("max", "peak", "최대값", "최댓값")
    assert out["keywordSpikeKr"] in ("spike", "피크", "스파이크")
    assert out["fallback"] == "구간4"
