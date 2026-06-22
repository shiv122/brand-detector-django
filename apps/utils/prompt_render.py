"""Prompt-template rendering for SportPrompt.

Two template variables are expanded:

  * `{{BRAND_LIST}}`   — the canonical brand vocabulary
  * `{{TAGLINE_LIST}}` — the canonical tagline / slogan vocabulary

(single-brace `{BRAND_LIST}` / `{TAGLINE_LIST}` are also accepted for
ergonomics). Each is replaced with a newline-separated list, each entry
prefixed with "- ", suitable for dropping into a closed-vocabulary
instruction:

    ALLOWED BRANDS:
    - Aggreko
    - BMW
    - Rolex

Brands and taglines are stored in DISPLAY form (real casing / spacing) so
the model can emit the true canonical spelling. A separate snake_case key
(`normalize_brand`) is only used for de-duplication and internal matching.

When a prompt is rendered with a non-empty vocabulary, a standardized
CORRECTION guide is appended. It tells the stage-2 formatter to map noisy
OCR text onto the canonical entries and to report every change it makes in a
top-level `corrections` array — so the raw and corrected forms are both
recoverable downstream.
"""

from __future__ import annotations

import re
from typing import Iterable, List


_BRAND_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")


def normalize_brand(raw: str) -> str:
    """Coerce a string into a snake_case matching key.

    Used only for de-duplication / internal matching — NOT for storage or
    display. Empty / all-symbol inputs collapse to "" — callers should drop
    those.
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip().lower()
    s = _BRAND_TOKEN_RE.sub("_", s)
    return s.strip("_")


def normalize_brands(values: Iterable[str]) -> List[str]:
    """Normalize + dedupe a list to snake_case keys, preserving input order."""
    seen: set[str] = set()
    out: List[str] = []
    for v in values or []:
        n = normalize_brand(v)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def normalize_phrase(raw: str) -> str:
    """Trim and collapse internal whitespace while preserving casing.

    This is the display form stored for brands and taglines.
    """
    if not isinstance(raw, str):
        return ""
    return _WS_RE.sub(" ", raw.strip())


def dedupe_phrases(values: Iterable[str]) -> List[str]:
    """Clean + dedupe display phrases, preserving first-seen order.

    De-duplication is case-insensitive (keyed on the snake_case form) but the
    original display spelling is what gets kept.
    """
    seen: set[str] = set()
    out: List[str] = []
    for v in values or []:
        phrase = normalize_phrase(v)
        if not phrase:
            continue
        key = normalize_brand(phrase)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def _format_list(values: List[str]) -> str:
    return "\n".join(f"- {v}" for v in values) if values else "(none)"


def _correction_guide(brands: List[str], taglines: List[str]) -> str:
    """Build the standardized brand/tagline correction instruction.

    Embeds the canonical lists so the guide is self-contained (works even
    when the prompt body has no `{{BRAND_LIST}}` / `{{TAGLINE_LIST}}`
    placeholder).
    """
    sections: List[str] = [
        "",
        "",
        "--- BRAND & TAGLINE CORRECTION ---",
        "You are given a closed vocabulary of canonical brands and taglines. "
        "OCR text is noisy: it may contain typos, character confusions "
        "(0/O, 1/l/I, rn/m), wrong casing, extra/missing spaces, or partial "
        "reads.",
    ]
    if brands:
        sections.append("\nCANONICAL BRANDS:\n" + _format_list(brands))
    if taglines:
        sections.append("\nCANONICAL TAGLINES:\n" + _format_list(taglines))
    sections.append(
        "\nRules:\n"
        "- When an on-screen string clearly refers to one of the canonical "
        "entries above, replace it IN PLACE (in `brands`, `texts`, and any "
        "other field) with the EXACT canonical spelling shown.\n"
        "- Only correct when you are confident it is the same brand/tagline. "
        "If a string does not match any canonical entry, leave it exactly as "
        "the OCR produced it — do NOT invent or force matches.\n"
        "- Record every replacement you make in a top-level `corrections` "
        "array. Each item is an object: "
        '{"raw": "<original OCR string>", "corrected": "<canonical spelling>", '
        '"type": "brand" | "tagline"}.\n'
        "- If you made no corrections, return an empty `corrections` array."
    )
    return "\n".join(sections)


def render_prompt(
    prompt: str,
    allowed_brands: Iterable[str] | None = None,
    taglines: Iterable[str] | None = None,
    correction_enabled: bool = True,
) -> str:
    """Append the brand/tagline correction guide when enabled.

    Correction is an opt-in feature: callers pass `correction_enabled` from the
    SportPrompt toggle. When off, the vocabulary is ignored entirely — brands
    and taglines never reach the model. When on, the standardized correction
    guide (with the canonical lists embedded) is appended to the prompt body,
    and the optional `{{BRAND_LIST}}` / `{{TAGLINE_LIST}}` placeholders are also
    expanded for power users who want inline control.
    """
    if not prompt:
        return prompt

    brands = dedupe_phrases(allowed_brands or []) if correction_enabled else []
    tags = dedupe_phrases(taglines or []) if correction_enabled else []

    brand_block = _format_list(brands)
    tag_block = _format_list(tags)

    rendered = (
        prompt.replace("{{BRAND_LIST}}", brand_block)
        .replace("{BRAND_LIST}", brand_block)
        .replace("{{TAGLINE_LIST}}", tag_block)
        .replace("{TAGLINE_LIST}", tag_block)
    )

    if brands or tags:
        rendered = rendered + _correction_guide(brands, tags)

    return rendered
