"""
OCR request validation classes.
"""

import re

from apps.api.v1.requests.base_request import BaseRequest
from apps.utils.prompt_render import dedupe_phrases


_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$")
_MAX_BRANDS = 256
_MAX_BRAND_LEN = 64
_MAX_TAGLINES = 256
_MAX_TAGLINE_LEN = 160


class OcrRunRequest(BaseRequest):
    """Validation for POST /api/v1/ocr/run.

    Prompt resolution priority (highest first):
      1. `prompt`       — inline override, sent verbatim
      2. `prompt_slug`  — pick a saved prompt by its slug
      3. `sport`        — pick the prompt whose `sport` field matches
    """

    def rules(self):
        if "file" not in self.files or not self.files["file"]:
            self._add_error("file", "file is required")

        if "prompt_slug" in self.data and self.data["prompt_slug"]:
            self._string("prompt_slug", min_length=1, max_length=128)
        else:
            self.data["prompt_slug"] = ""

        if "sport" in self.data and self.data["sport"]:
            self._string("sport", min_length=1, max_length=64)
        else:
            self.data["sport"] = ""

        if "prompt" in self.data and self.data["prompt"]:
            self._string("prompt", max_length=16384)
        else:
            self.data["prompt"] = ""


class SportPromptUpsertRequest(BaseRequest):
    """Validation for creating / updating a SportPrompt."""

    require_slug = True  # set False on PUT (slug comes from URL)

    def rules(self):
        self._required("name", "name")
        self._string("name", min_length=1, max_length=255)

        if self.require_slug:
            self._required("slug", "slug")
        if self.data.get("slug"):
            slug = str(self.data["slug"]).strip().lower()
            if not _SLUG_RE.match(slug):
                self._add_error(
                    "slug",
                    "slug must be lowercase letters/digits/hyphens (2-64 chars)",
                )
            else:
                self.data["slug"] = slug

        if "sport" in self.data and self.data["sport"]:
            self._string("sport", max_length=64)
        else:
            self.data["sport"] = ""

        if "description" in self.data and self.data["description"]:
            self._string("description", max_length=4096)
        else:
            self.data["description"] = ""

        self._required("prompt", "prompt")
        if self.data.get("prompt"):
            self._string("prompt", min_length=1, max_length=16384)

        # allowed_brands / taglines: lists of display-form strings, cleaned
        # (trim + collapse whitespace) and case-insensitively deduped while
        # preserving the original spelling. Anything not a list is rejected;
        # non-string entries are dropped.
        self.data["allowed_brands"] = self._clean_phrase_list(
            "allowed_brands", _MAX_BRANDS, _MAX_BRAND_LEN
        )
        self.data["taglines"] = self._clean_phrase_list(
            "taglines", _MAX_TAGLINES, _MAX_TAGLINE_LEN
        )

        # correction_enabled: opt-in toggle for the brand/tagline correction
        # guide. Accept real bools and the usual truthy strings from forms.
        raw_enabled = self.data.get("correction_enabled", False)
        if isinstance(raw_enabled, bool):
            self.data["correction_enabled"] = raw_enabled
        elif isinstance(raw_enabled, str):
            self.data["correction_enabled"] = raw_enabled.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        else:
            self.data["correction_enabled"] = bool(raw_enabled)

        # Optional reference image (data URL or raw base64). Capped at ~6 MB.
        ref = self.data.get("reference_image")
        if ref is not None and not isinstance(ref, str):
            self._add_error("reference_image", "reference_image must be a base64 string")
            self.data["reference_image"] = None
        elif isinstance(ref, str):
            if len(ref) > 6 * 1024 * 1024:
                self._add_error(
                    "reference_image",
                    "reference_image exceeds 6 MB encoded — re-encode at lower quality",
                )
                self.data["reference_image"] = None
            elif not ref.strip():
                self.data["reference_image"] = None

    def _clean_phrase_list(self, field: str, max_items: int, max_len: int) -> list:
        raw = self.data.get(field, [])
        if raw in (None, ""):
            return []
        if not isinstance(raw, list):
            self._add_error(field, f"{field} must be a list of strings")
            return []
        strings = [s for s in raw if isinstance(s, str)]
        if any(len(s) > max_len for s in strings):
            self._add_error(field, f"each entry must be <= {max_len} chars")
        cleaned = dedupe_phrases(strings)
        if len(cleaned) > max_items:
            self._add_error(field, f"{field} accepts at most {max_items} entries")
            cleaned = cleaned[:max_items]
        return cleaned
