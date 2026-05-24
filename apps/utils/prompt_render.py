"""Prompt-template rendering for SportPrompt.

The only template variable we expand is `{{BRAND_LIST}}` (single-brace
`{BRAND_LIST}` is also accepted for ergonomics). It is replaced with a
newline-separated list of allowed brand identifiers, each prefixed with
"- ", suitable for dropping straight into a closed-vocabulary instruction:

    ALLOWED BRANDS:
    - aggreko
    - bmw
    - rolex
"""

from __future__ import annotations

import re
from typing import Iterable, List


_BRAND_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def normalize_brand(raw: str) -> str:
    """Coerce a brand string into the canonical snake_case identifier.

    Empty / all-symbol inputs collapse to "" — callers should drop those.
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip().lower()
    s = _BRAND_TOKEN_RE.sub("_", s)
    return s.strip("_")


def normalize_brands(values: Iterable[str]) -> List[str]:
    """Normalize + dedupe a list of brand strings, preserving input order."""
    seen: set[str] = set()
    out: List[str] = []
    for v in values or []:
        n = normalize_brand(v)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def render_prompt(prompt: str, allowed_brands: Iterable[str] | None) -> str:
    """Substitute the {{BRAND_LIST}} placeholder with a formatted brand list.

    If the prompt has no placeholder, it is returned unchanged. If the
    placeholder is present but the brand list is empty, it is replaced
    with the literal "(none)" so the model sees the intent unambiguously.
    """
    if not prompt or ("{{BRAND_LIST}}" not in prompt and "{BRAND_LIST}" not in prompt):
        return prompt

    brands = list(allowed_brands or [])
    formatted = "\n".join(f"- {b}" for b in brands) if brands else "(none)"
    return prompt.replace("{{BRAND_LIST}}", formatted).replace(
        "{BRAND_LIST}", formatted
    )
