"""JSON schemas for structured LLM output."""

from __future__ import annotations

# Level 1 briefing: at most three issues, ordered by severity. The schema is
# enforced loosely (validated after parse) because Bedrock Converse does not
# support strict JSON schema for every model.
BRIEFING_SCHEMA: dict = {
    "type": "object",
    "required": ["issues"],
    "properties": {
        "issues": {
            "type": "array",
            "minItems": 0,
            "maxItems": 3,
            "items": {
                "type": "object",
                "required": ["title", "severity", "detail"],
                "properties": {
                    "title": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["info", "warning", "critical"],
                    },
                    "detail": {"type": "string"},
                    "metric_hint": {
                        "type": "string",
                        "description": (
                            "Optional short pointer to a metric or chart "
                            "the user should open for more context."
                        ),
                    },
                },
            },
        }
    },
}

_ALLOWED_SEVERITIES = {"info", "warning", "critical"}


def validate_briefing(payload: object) -> dict:
    """Return ``payload`` normalized to the briefing schema.

    Missing or malformed fields are coerced to safe defaults so a single bad
    model response does not take the whole card down. At most three issues
    are returned; anything beyond the third is dropped.
    """
    if not isinstance(payload, dict):
        return {"issues": []}
    raw_issues = payload.get("issues")
    if not isinstance(raw_issues, list):
        raw_issues = []
    out: list[dict] = []
    for entry in raw_issues[:3]:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        detail = str(entry.get("detail") or "").strip()
        severity = str(entry.get("severity") or "info").strip().lower()
        if severity not in _ALLOWED_SEVERITIES:
            severity = "info"
        metric_hint = entry.get("metric_hint")
        if metric_hint is not None and not isinstance(metric_hint, str):
            metric_hint = None
        if metric_hint is not None:
            metric_hint = metric_hint.strip() or None
        if not title and not detail:
            continue
        item: dict = {
            "title": title or "(untitled)",
            "severity": severity,
            "detail": detail or "(no detail)",
        }
        if metric_hint:
            item["metric_hint"] = metric_hint
        out.append(item)
    return {"issues": out}
