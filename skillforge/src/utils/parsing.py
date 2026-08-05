import json
from typing import Any


def strip_markdown_fence(text: str) -> str:
    """Remove one optional Markdown fence while preserving response content."""
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_json_response(text: str) -> Any:
    """Parse JSON returned either directly or inside a Markdown fence."""
    cleaned = strip_markdown_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {exc.msg}") from exc
