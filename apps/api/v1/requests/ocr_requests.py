"""
OCR request validation classes.
"""

import json
import re

from apps.api.v1.requests.base_request import BaseRequest


_ALLOWED_FIELD_TYPES = [
    "string",
    "integer",
    "float",
    "boolean",
    "list[string]",
    "list[integer]",
]
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$")
_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")


class OcrRunRequest(BaseRequest):
    """Validation for POST /api/v1/ocr/run (standalone OCR endpoint)."""

    def rules(self):
        if "file" not in self.files or not self.files["file"]:
            self._add_error("file", "file is required")

        if "roi" in self.data and self.data["roi"]:
            self._parse_roi("roi")

        if "include_annotated" in self.data:
            self._boolean("include_annotated")
        else:
            self.data["include_annotated"] = False

        if "template_key" in self.data and self.data["template_key"]:
            self._string("template_key", min_length=1, max_length=128)
        else:
            self.data["template_key"] = "raw"

        if "custom_prompt" in self.data and self.data["custom_prompt"]:
            self._string("custom_prompt", max_length=8192)
        else:
            self.data["custom_prompt"] = ""

    def _parse_roi(self, field: str):
        """Parse roi as a JSON array of 4 floats in 0..1."""
        raw = self.data[field]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                self._add_error(field, "roi must be a JSON array [x1,y1,x2,y2]")
                return

        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            self._add_error(field, "roi must be an array of 4 numbers")
            return

        try:
            roi = [float(v) for v in raw]
        except (TypeError, ValueError):
            self._add_error(field, "roi values must be numbers")
            return

        if not all(0.0 <= v <= 1.0 for v in roi):
            self._add_error(field, "roi values must be normalized to 0..1")
            return

        if roi[2] <= roi[0] or roi[3] <= roi[1]:
            self._add_error(field, "roi must have positive width and height")
            return

        self.data[field] = roi


class CustomTemplateUpsertRequest(BaseRequest):
    """Validation for creating / updating a CustomOcrTemplate."""

    # Set to False on PUT when slug comes from the URL.
    require_slug = True

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

        if "system_prompt" in self.data and self.data["system_prompt"]:
            self._string("system_prompt", max_length=16384)
        else:
            self.data["system_prompt"] = ""

        if "multimodal" in self.data:
            self._boolean("multimodal")
        else:
            self.data["multimodal"] = False

        self._validate_regions()

    def _validate_regions(self):
        raw = self.data.get("regions")
        if raw is None:
            self._add_error("regions", "regions is required")
            return

        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                self._add_error("regions", "regions must be valid JSON")
                return

        if not isinstance(raw, list) or not raw:
            self._add_error("regions", "regions must be a non-empty array")
            return

        if len(raw) > 20:
            self._add_error("regions", "max 20 regions")
            return

        seen_labels: set = set()
        seen_field_names: set = set()
        cleaned: list = []
        for i, region in enumerate(raw):
            if not isinstance(region, dict):
                self._add_error("regions", f"region {i} must be an object")
                continue

            label = str(region.get("label", "")).strip()
            if not label:
                self._add_error("regions", f"region {i} missing label")
                continue
            if not _IDENT_RE.match(label):
                self._add_error(
                    "regions",
                    f"region {i} label must be a valid identifier "
                    "(letters/digits/underscore, max 64 chars)",
                )
                continue
            if label in seen_labels:
                self._add_error("regions", f"duplicate region label: {label}")
                continue
            seen_labels.add(label)

            coords = region.get("coords")
            if (
                not isinstance(coords, (list, tuple))
                or len(coords) != 4
                or not all(isinstance(v, (int, float)) for v in coords)
            ):
                self._add_error(
                    "regions",
                    f"region {i} ({label}) coords must be [x1,y1,x2,y2] numbers",
                )
                continue
            x1, y1, x2, y2 = (float(v) for v in coords)
            if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
                self._add_error(
                    "regions",
                    f"region {i} ({label}) coords must be normalized 0..1",
                )
                continue
            if x2 <= x1 or y2 <= y1:
                self._add_error(
                    "regions",
                    f"region {i} ({label}) coords must have positive area",
                )
                continue

            fields_raw = region.get("expected_fields", [])
            if not isinstance(fields_raw, list):
                self._add_error(
                    "regions",
                    f"region {i} ({label}) expected_fields must be a list",
                )
                continue

            cleaned_fields: list = []
            for j, field_def in enumerate(fields_raw):
                if not isinstance(field_def, dict):
                    self._add_error(
                        "regions",
                        f"region {label} field {j} must be an object",
                    )
                    continue
                fname = str(field_def.get("name", "")).strip()
                if not fname or not _IDENT_RE.match(fname):
                    self._add_error(
                        "regions",
                        f"region {label} field {j} name must be a valid identifier",
                    )
                    continue
                if fname in seen_field_names:
                    self._add_error(
                        "regions",
                        f"duplicate field name across regions: {fname}",
                    )
                    continue
                seen_field_names.add(fname)

                ftype = str(field_def.get("type", "string")).strip()
                if ftype not in _ALLOWED_FIELD_TYPES:
                    self._add_error(
                        "regions",
                        f"region {label} field {fname} type must be one of "
                        f"{_ALLOWED_FIELD_TYPES}",
                    )
                    continue
                fdesc = str(field_def.get("description", "")).strip()[:512]
                cleaned_fields.append(
                    {"name": fname, "type": ftype, "description": fdesc}
                )

            cleaned.append(
                {
                    "label": label,
                    "description": str(region.get("description", "")).strip()[:1024],
                    "coords": [x1, y1, x2, y2],
                    "expected_fields": cleaned_fields,
                }
            )

        if cleaned:
            self.data["regions"] = cleaned
