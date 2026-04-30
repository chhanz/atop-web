"""Structural sentinel for the Phase 21 dashboard layout.

The dashboard is now a vertical two-row layout (charts on top, process
table below). The chart strip switches between three modes based on the
container's available inline size, not the viewport, so that an open
chat panel shrinking the effective width automatically downgrades the
layout. This test keeps those breakpoints honest.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).parent.parent / "atop_web" / "static"
STYLE_CSS = (STATIC / "style.css").read_text()


def test_main_is_vertical_two_row_layout():
    # ``main`` must be a single column so the charts section stacks on
    # top of the process table.
    m = re.search(r"\nmain\s*\{[^}]*\}", STYLE_CSS, flags=re.DOTALL)
    assert m, "main {} block missing"
    body = m.group(0)
    assert "grid-template-columns" in body
    # 1fr single column; the old 3fr 2fr split must be gone.
    assert "3fr 2fr" not in body, body
    assert "grid-template-columns: 1fr" in body or "grid-template-columns:1fr" in body.replace(" ", "")


def test_charts_declares_container_query_context():
    # ``#charts`` must opt into container queries so the breakpoints
    # below track the available inline size rather than the viewport.
    assert re.search(r"#charts\s*\{[^}]*container-type:\s*inline-size", STYLE_CSS, flags=re.DOTALL), \
        "#charts must set container-type: inline-size"


def test_charts_has_three_container_query_modes():
    # Wide (5 columns), Mid (3+2 via 6-column span), Narrow (1 column).
    # We check for each sentinel independently so a future reshuffle
    # has to touch all three.
    wide = re.search(
        r"@container[^{]*\(min-width:\s*1600px\)\s*\{[^}]*#charts[^{]*\{[^}]*repeat\(5",
        STYLE_CSS,
        flags=re.DOTALL,
    )
    assert wide, "wide mode (>=1600px, 5 columns) missing"

    mid = re.search(
        r"@container[^{]*\(min-width:\s*1100px\)\s*\{[^}]*#charts[^{]*\{[^}]*repeat\(6",
        STYLE_CSS,
        flags=re.DOTALL,
    )
    assert mid, "mid mode (>=1100px, 6-col grid with span 2/3) missing"

    # Narrow mode is the default (no query): plain single column on
    # ``#charts``. The ``1fr`` fallback lives outside @container blocks.
    base = re.search(r"#charts\s*\{[^}]*grid-template-columns:\s*1fr\b", STYLE_CSS, flags=re.DOTALL)
    assert base, "narrow fallback (1 column) missing from base #charts rule"


def test_mid_mode_assigns_spans_to_cards():
    # 3 + 2 layout needs explicit ``grid-column: span N`` on the chart
    # cards in the mid mode. We find the @container (min-width: 1100px)
    # block and require that it contains both ``span 2`` and ``span 3``
    # grid-column assignments. A simpler sentinel than matching the full
    # nested structure; breaks the moment someone drops either span.
    m = re.search(
        r"@container[^{]*\(min-width:\s*1100px\)\s*\{",
        STYLE_CSS,
    )
    assert m, "mid container-query block missing"
    # Walk forward from the opening brace using a depth counter so we
    # isolate the whole @container body (which itself nests selectors).
    start = m.end() - 1
    depth = 0
    end = start
    for i in range(start, len(STYLE_CSS)):
        c = STYLE_CSS[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    block = STYLE_CSS[start:end]
    assert "grid-column: span 2" in block, "mid mode must assign span 2 to leading cards"
    assert "grid-column: span 3" in block, "mid mode must assign span 3 to trailing cards"


def test_side_keeps_min_height_floor():
    # Keeps a sensible vertical floor so the process table has room
    # when the charts row is tall on narrow viewports.
    m = re.search(r"#side\s*\{[^}]*min-height:\s*(\d+)px", STYLE_CSS, flags=re.DOTALL)
    # The value may appear in either the base rule or a media query;
    # accept either but require at least 300px somewhere for #side.
    found = re.findall(r"#side\s*\{[^}]*min-height:\s*(\d+)px", STYLE_CSS, flags=re.DOTALL)
    assert any(int(v) >= 300 for v in found), f"#side needs a floor >=300px, got {found}"
