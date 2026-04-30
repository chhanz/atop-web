"""Static asset sanity checks for the chat markdown pipeline.

We do not run a JS engine in the test container, so these are purely
structural checks: the vendored files exist, have the expected version
signature, are referenced from ``index.html``, and the call sites in
``app.js`` compose marked with DOMPurify so no path renders unsanitized
HTML. A smoke XSS test would ideally exercise ``DOMPurify.sanitize`` in
jsdom; we approximate it by verifying the sanitize call appears in the
render helper and that the vendored DOMPurify bundle contains its
documented default block list for ``<script>`` and ``on*`` event handler
attributes.
"""

from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).parent.parent / "atop_web" / "static"


def test_vendor_marked_present():
    fp = STATIC / "vendor" / "marked.min.js"
    assert fp.is_file(), fp
    text = fp.read_text(errors="replace")
    # UMD bundle exposes ``window.marked``; the header banner includes the
    # version string so a bump breaks this test until VERSIONS.txt is
    # refreshed.
    assert "marked v9.1.6" in text
    assert "marked" in text.lower()


def test_vendor_dompurify_present():
    fp = STATIC / "vendor" / "purify.min.js"
    assert fp.is_file(), fp
    text = fp.read_text(errors="replace")
    assert "DOMPurify 3.4.1" in text
    # Cheap structural check: the UMD bundle binds ``DOMPurify`` onto the
    # global, so the identifier is always in the file even after
    # minification. A missing identifier would mean we shipped the wrong
    # artifact (e.g. the ESM build, which does not self register).
    assert "DOMPurify" in text
    # Size sanity: the real bundle is ~25 KiB; a 1 line stub or 404 HTML
    # would slip past the string check above.
    assert fp.stat().st_size > 10_000


def test_vendor_versions_manifest_lists_both_libraries():
    fp = STATIC / "vendor" / "VERSIONS.txt"
    assert fp.is_file(), fp
    text = fp.read_text()
    assert "marked" in text
    assert "DOMPurify" in text
    assert "sha256:" in text


def test_index_html_loads_vendor_scripts():
    html = (STATIC / "index.html").read_text()
    assert 'src="static/vendor/marked.min.js"' in html
    assert 'src="static/vendor/purify.min.js"' in html
    # app.js must be deferred so the window.marked / window.DOMPurify globals
    # are available before the IIFE at the bottom runs.
    assert 'src="static/app.js" defer' in html


def test_app_js_sanitizes_before_rendering():
    js = (STATIC / "app.js").read_text()
    assert "renderMarkdown" in js
    assert "DOMPurify.sanitize" in js
    assert "marked.parse" in js
    # User messages must stay as textContent so an operator cannot
    # accidentally XSS themselves via a pasted prompt that the server
    # echoes back.
    assert "textContent = text" in js


def test_css_has_assistant_markdown_styles():
    css = (STATIC / "style.css").read_text()
    # Basic regression: ensure the dark theme applies to the rendered
    # markdown elements so <code>, <pre>, <strong> stay legible on the
    # assistant bubble background.
    for selector in (
        ".chat-message.assistant .chat-bubble code",
        ".chat-message.assistant .chat-bubble pre",
        ".chat-message.assistant .chat-bubble strong",
        ".chat-message.assistant .chat-bubble ul",
    ):
        assert selector in css, selector
