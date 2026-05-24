"""
OCR request validation classes.
"""

import re

from apps.api.v1.requests.base_request import BaseRequest
from apps.utils.prompt_render import normalize_brands


_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$")
_MAX_BRANDS = 256
_MAX_BRAND_LEN = 64


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

        # allowed_brands: list of strings, normalized to snake_case and deduped.
        # Anything not a list is rejected; non-string entries are dropped.
        raw_brands = self.data.get("allowed_brands", [])
        if raw_brands in (None, ""):
            self.data["allowed_brands"] = []
        elif not isinstance(raw_brands, list):
            self._add_error("allowed_brands", "allowed_brands must be a list of strings")
            self.data["allowed_brands"] = []
        else:
            string_brands = [b for b in raw_brands if isinstance(b, str)]
            if any(len(b) > _MAX_BRAND_LEN for b in string_brands):
                self._add_error(
                    "allowed_brands",
                    f"each brand must be <= {_MAX_BRAND_LEN} chars",
                )
            cleaned = normalize_brands(string_brands)
            if len(cleaned) > _MAX_BRANDS:
                self._add_error(
                    "allowed_brands",
                    f"allowed_brands accepts at most {_MAX_BRANDS} entries",
                )
                cleaned = cleaned[:_MAX_BRANDS]
            self.data["allowed_brands"] = cleaned

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
