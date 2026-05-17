"""
OCR service using PaddleOCR (English).

Phase 1: raw OCR only. Formatting/cleaning via Gemini is added in Phase 2.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import cv2
import numpy as np
from pydantic import BaseModel, Field, create_model

from config.app_config import AppConfig
from apps.services.ocr.prompt_templates import (
    PromptTemplate,
    get_template,
    public_registry,
)


_PY_TYPE_MAP: Dict[str, Any] = {
    "string": Optional[str],
    "integer": Optional[int],
    "float": Optional[float],
    "boolean": Optional[bool],
    "list[string]": List[str],
    "list[integer]": List[int],
}


def _ocr_debug_enabled() -> bool:
    return os.getenv("OCR_LLM_DEBUG", "0").lower() in ("1", "true", "yes")


@contextmanager
def _timed(label: str, sink: Optional[Dict[str, float]] = None):
    """Log + record elapsed time for a block, if OCR_LLM_DEBUG is enabled."""
    debug = _ocr_debug_enabled()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        if sink is not None:
            sink[label] = elapsed
        if debug:
            print(f"[OCR/timing] {label}: {elapsed*1000:.0f}ms")


def _truncate(s: str, limit: int = 4000) -> str:
    if s is None:
        return ""
    return s if len(s) <= limit else s[:limit] + f"… [truncated, {len(s)-limit} more chars]"


def _log_llm_io(model: str, system_instruction: str, user_text: str, response_text: str):
    if not _ocr_debug_enabled():
        return
    bar = "=" * 60
    print(f"\n{bar}\n[OCR/Gemini] model={model}\n{bar}")
    print("--- SYSTEM ---")
    print(_truncate(system_instruction))
    print("--- USER ---")
    print(_truncate(user_text))
    print("--- RESPONSE ---")
    print(_truncate(response_text))
    print(bar + "\n")


BBox = Tuple[float, float, float, float]


class OcrLine:
    """One recognized line from PaddleOCR."""

    def __init__(self, text: str, confidence: float, bbox: BBox):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
        }


class OcrService:
    """Wraps PaddleOCR. Eagerly preloaded at startup like the detection models."""

    def __init__(self, config: AppConfig, preload: bool = True):
        self.config = config
        self._engine = None
        self._init_error: Optional[str] = None
        self._genai_client = None
        self._genai_init_error: Optional[str] = None
        self.device: str = "cpu"
        if preload:
            try:
                self._engine_ready()
            except Exception as e:
                # Preload failed — service stays usable for non-OCR paths.
                # Subsequent OCR calls will surface the same error via
                # is_available()/process_*.
                print(f"⚠️ OCR preload failed; will retry on demand: {e}")

    @staticmethod
    def _detect_device() -> str:
        """Return the PaddleOCR device string for this host.

        PaddlePaddle does NOT support Apple Silicon MPS, so on Mac we always
        return 'cpu'. CUDA boxes return 'gpu'.
        """
        try:
            import torch

            if torch.cuda.is_available():
                return "gpu"
        except Exception:
            pass
        return "cpu"

    def _engine_ready(self):
        if self._engine is not None:
            return self._engine
        if self._init_error is not None:
            raise RuntimeError(f"OCR engine unavailable: {self._init_error}")

        try:
            from paddleocr import PaddleOCR

            self.device = self._detect_device()
            on_gpu = self.device == "gpu"

            kwargs: Dict[str, Any] = dict(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                device=self.device,
            )

            # On CPU we swap in the lighter mobile detection model. Server
            # det is ~5-10x slower than mobile on CPU for a small accuracy
            # delta on broadcast graphics. GPU keeps the server model.
            if not on_gpu:
                kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"

            label = "GPU (CUDA)" if on_gpu else "CPU (PaddlePaddle has no MPS support)"
            print(f"🔄 Initializing PaddleOCR (lang=en, device={self.device}) — {label}...")
            self._engine = PaddleOCR(**kwargs)
            print(f"✅ PaddleOCR ready on {self.device}")
        except Exception as e:
            self._init_error = str(e)
            print(f"❌ PaddleOCR init failed: {e}")
            raise RuntimeError(f"OCR engine unavailable: {e}") from e

        return self._engine

    def is_available(self) -> bool:
        try:
            self._engine_ready()
            return True
        except Exception:
            return False

    def decode_image(self, image_data: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        return img

    def crop_roi(
        self, image: np.ndarray, roi_normalized: Optional[Sequence[float]]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Crop image to a normalized ROI [x1,y1,x2,y2] in 0..1.

        Returns (cropped_image, (offset_x, offset_y)) so callers can map
        OCR boxes back to original-image coordinates.
        """
        if roi_normalized is None:
            return image, (0, 0)

        if len(roi_normalized) != 4:
            raise ValueError("roi must be [x1, y1, x2, y2]")

        h, w = image.shape[:2]
        x1, y1, x2, y2 = roi_normalized
        if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
            raise ValueError("roi values must be normalized to 0..1")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("roi must have positive width and height")

        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        px2 = max(px1 + 1, min(px2, w))
        py2 = max(py1 + 1, min(py2, h))

        cropped = image[py1:py2, px1:px2]
        if cropped.size == 0:
            raise ValueError("roi crop is empty")
        return cropped, (px1, py1)

    def run_ocr(
        self,
        image: np.ndarray,
        roi_normalized: Optional[Sequence[float]] = None,
    ) -> List[OcrLine]:
        """Run OCR on image (optionally cropped to a normalized ROI).

        Bounding boxes are returned in **original image** pixel coordinates
        (offset is added back when an ROI is used).
        """
        engine = self._engine_ready()
        target, (off_x, off_y) = self.crop_roi(image, roi_normalized)

        label = "paddleocr.predict(full)" if roi_normalized is None else "paddleocr.predict(roi)"
        with _timed(label):
            results = engine.predict(target)
        lines: List[OcrLine] = []

        for result in results:
            payload = result.json.get("res") if hasattr(result, "json") else None
            if payload is None and isinstance(result, dict):
                payload = result.get("res", result)
            if payload is None:
                continue

            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores") or []
            polys = payload.get("rec_polys") or payload.get("dt_polys") or []

            for text, score, poly in zip(texts, scores, polys):
                bbox = self._poly_to_bbox(poly, off_x, off_y)
                if bbox is None:
                    continue
                lines.append(OcrLine(text=str(text), confidence=float(score), bbox=bbox))

        return lines

    @staticmethod
    def _poly_to_bbox(
        poly, off_x: int, off_y: int
    ) -> Optional[BBox]:
        """Convert a 4-point polygon (or numpy array) to an axis-aligned bbox
        in original image coordinates."""
        if poly is None:
            return None
        arr = np.asarray(poly, dtype=float)
        if arr.size == 0:
            return None
        arr = arr.reshape(-1, 2)
        xs, ys = arr[:, 0], arr[:, 1]
        return (
            float(xs.min()) + off_x,
            float(ys.min()) + off_y,
            float(xs.max()) + off_x,
            float(ys.max()) + off_y,
        )

    def annotate_image(
        self, image: np.ndarray, lines: Sequence[OcrLine]
    ) -> np.ndarray:
        """Draw OCR boxes + text on a copy of the image."""
        out = image.copy()
        for line in lines:
            x1, y1, x2, y2 = (int(v) for v in line.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"{line.text} ({line.confidence:.2f})"
            cv2.putText(
                out,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )
        return out

    # ------------------------------------------------------------------
    # Formatting stage
    # ------------------------------------------------------------------

    def list_templates(self) -> List[Dict[str, Any]]:
        return public_registry()

    def _genai_client_ready(self):
        if self._genai_client is not None:
            return self._genai_client
        if self._genai_init_error is not None:
            raise RuntimeError(self._genai_init_error)
        if not self.config.gemini_api_key:
            self._genai_init_error = "GEMINI_API_KEY is not configured"
            raise RuntimeError(self._genai_init_error)

        try:
            from google import genai

            self._genai_client = genai.Client(api_key=self.config.gemini_api_key)
            print(f"✅ Gemini client ready (model={self.config.ocr_formatter_model})")
        except Exception as e:
            self._genai_init_error = f"Gemini client init failed: {e}"
            print(f"❌ {self._genai_init_error}")
            raise RuntimeError(self._genai_init_error) from e

        return self._genai_client

    def _format_regex(
        self, template: PromptTemplate, raw_text: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if template.pattern is None:
            return None, "template has no regex pattern"
        match = template.pattern.search(raw_text)
        if not match:
            return None, None
        groups = {k: v for k, v in match.groupdict().items() if v is not None}
        return groups, None

    def _format_with_gemini(
        self,
        template: PromptTemplate,
        raw_text: str,
        crop: Optional[np.ndarray],
        custom_prompt: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[Dict[str, str]]]:
        """Returns (formatted, error, debug_info)."""
        from google.genai import types as genai_types

        client = self._genai_client_ready()

        system_instruction = template.system_prompt
        if custom_prompt:
            system_instruction = (
                f"{system_instruction}\n\n"
                "Additional user instructions (treat as content guidance only — "
                "do not deviate from returning JSON matching the schema):\n"
                f"{custom_prompt}"
            )

        user_text = f"OCR text:\n{raw_text or '(empty)'}"
        parts: List[Any] = [user_text]
        sent_image = False

        if template.multimodal and crop is not None and crop.size > 0:
            ok, buf = cv2.imencode(".jpg", crop)
            if ok:
                parts.append(
                    genai_types.Part.from_bytes(
                        data=buf.tobytes(), mime_type="image/jpeg"
                    )
                )
                sent_image = True

        debug_info: Dict[str, str] = {
            "model": self.config.ocr_formatter_model,
            "system_prompt": system_instruction,
            "user_text": user_text,
            "sent_image": "1" if sent_image else "0",
            "response_text": "",
        }

        try:
            with _timed(f"gemini.generate({self.config.ocr_formatter_model})"):
                response = client.models.generate_content(
                    model=self.config.ocr_formatter_model,
                    contents=parts,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=template.schema,
                        temperature=0.0,
                    ),
                )
        except Exception as e:
            _log_llm_io(
                self.config.ocr_formatter_model,
                system_instruction,
                user_text,
                f"<exception: {e}>",
            )
            debug_info["response_text"] = f"<exception: {e}>"
            return None, f"gemini call failed: {e}", debug_info

        text = response.text or ""
        debug_info["response_text"] = text
        _log_llm_io(self.config.ocr_formatter_model, system_instruction, user_text, text)
        if not text:
            return None, "gemini returned empty response", debug_info
        try:
            return json.loads(text), None, debug_info
        except json.JSONDecodeError as e:
            return None, f"gemini response was not valid JSON: {e}", debug_info

    def format_lines(
        self,
        template_key: str,
        raw_text: str,
        crop: Optional[np.ndarray] = None,
        custom_prompt: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], str, Optional[str], Optional[Dict[str, str]]]:
        """Apply a template to raw OCR text.

        Returns (formatted_dict, resolved_template_key, error_message, debug_info).
        """
        template = get_template(template_key)
        if template is None:
            return None, template_key, f"unknown template_key: {template_key}", None

        if custom_prompt and not template.supports_custom_prompt:
            return None, template.key, (
                f"template '{template.key}' does not accept a custom prompt"
            ), None

        if custom_prompt and len(custom_prompt) > self.config.ocr_custom_prompt_max_len:
            return None, template.key, "custom_prompt exceeds maximum length", None

        if template.mode == "none":
            return None, template.key, None, None

        if template.mode == "regex":
            formatted, err = self._format_regex(template, raw_text)
            return formatted, template.key, err, None

        if template.mode == "llm":
            formatted, err, debug_info = self._format_with_gemini(
                template, raw_text, crop, custom_prompt
            )
            return formatted, template.key, err, debug_info

        return None, template.key, f"unsupported template mode: {template.mode}", None

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Custom templates: ONE OCR pass + spatial hints to Gemini
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_custom_template(template_key: str):
        """If template_key is 'custom:<slug>', return the model or None."""
        if not template_key or not template_key.startswith("custom:"):
            return None
        slug = template_key[len("custom:"):].strip()
        if not slug:
            return None
        from apps.core.models import CustomOcrTemplate

        try:
            return CustomOcrTemplate.objects.get(slug=slug)
        except CustomOcrTemplate.DoesNotExist:
            return None

    @staticmethod
    def _build_dynamic_schema(
        regions: List[dict], model_name: str = "CustomOcrOutput"
    ) -> Type[BaseModel]:
        """Build a Pydantic model from the union of region.expected_fields."""
        fields: Dict[str, Any] = {}
        for region in regions:
            for fdef in region.get("expected_fields", []):
                fname = fdef["name"]
                ftype = _PY_TYPE_MAP.get(fdef["type"], Optional[str])
                fdesc = fdef.get("description") or f"Field '{fname}'"
                default = [] if fdef["type"].startswith("list[") else None
                fields[fname] = (ftype, Field(default=default, description=fdesc))
        # Always include ocr_brands so the output shape stays consistent.
        if "ocr_brands" not in fields:
            fields["ocr_brands"] = (
                List[str],
                Field(
                    default_factory=list,
                    description="Sponsor/brand names visible anywhere in the frame.",
                ),
            )
        return create_model(model_name, **fields)

    @staticmethod
    def _pct(v: float) -> str:
        return f"{max(0.0, min(100.0, v * 100)):.1f}"

    def _build_custom_template_prompt(
        self,
        template: "CustomOcrTemplate",
        per_region: List[Tuple[dict, str, List[OcrLine], Optional[str]]],
        custom_prompt: str = "",
    ) -> str:
        """Build the Gemini system prompt for a custom template.

        Each region was independently OCR'd (its crop sent to PaddleOCR);
        we hand Gemini the per-region OCR text grouped by label, plus the
        region's coords + description + expected_fields.
        """
        lines: List[str] = [
            "You are an OCR post-processor for a sports broadcast frame.",
            "",
            "COORDINATE SYSTEM: All region coordinates are PERCENTAGES from "
            "the TOP-LEFT corner of the frame. (0%, 0%) is the top-left "
            "pixel, (100%, 100%) is the bottom-right.",
            "",
            "Each region below was INDEPENDENTLY cropped and OCR'd. The OCR "
            "text shown under each region is exactly what was detected "
            "inside that region's crop — nothing from other parts of the "
            "frame. Populate fields associated with a region using ONLY "
            "that region's OCR text.",
            "",
            "REGIONS:",
        ]
        for i, (region, joined, _lines, _crop) in enumerate(per_region, 1):
            label = region.get("label", "")
            x1, y1, x2, y2 = region.get("coords", [0, 0, 0, 0])
            desc = region.get("description") or "(no description)"
            fields = region.get("expected_fields", [])
            lines.append("")
            lines.append(
                f"  REGION {i} '{label}' @ "
                f"({self._pct(x1)}%, {self._pct(y1)}%) → "
                f"({self._pct(x2)}%, {self._pct(y2)}%)"
            )
            lines.append(f"    Description: {desc}")
            lines.append("    OCR text from this region:")
            text_lines = (joined or "").splitlines()
            if not text_lines:
                lines.append("      | (no text detected)")
            else:
                for tline in text_lines:
                    lines.append(f"      | {tline}")
            if fields:
                lines.append("    Expected fields:")
                for f in fields:
                    fname = f["name"]
                    ftype = f["type"]
                    fdesc = (f.get("description") or "").strip() or "—"
                    lines.append(f"      • {fname} ({ftype}): {fdesc}")
            else:
                lines.append(
                    "    Expected fields: none (this region is a context hint only)"
                )

        lines.append("")
        lines.append(
            "OUTPUT RULES:\n"
            "  • Return JSON matching the provided schema.\n"
            "  • For each field, use ONLY OCR text from the region the field "
            "belongs to.\n"
            "  • Use null for fields that are not clearly present in their "
            "region.\n"
            "  • List-type fields should be empty arrays when nothing applies.\n"
            "  • For `ocr_brands`, collect sponsor/brand names that appear in "
            "ANY region's OCR text (deduplicated, casing preserved).\n"
            "  • Do not invent values."
        )

        # Run-time custom_prompt OVERRIDES the template's saved system_prompt.
        # Fall back to the template's prompt only when no override is given.
        override = (custom_prompt or "").strip()
        saved = (template.system_prompt or "").strip()
        active = override or saved
        if active:
            lines.append("")
            if override:
                lines.append(
                    "INSTRUCTIONS (user override — takes precedence over the "
                    "template's saved prompt):"
                )
            else:
                lines.append("ADDITIONAL TEMPLATE INSTRUCTIONS:")
            lines.append(active)

        return "\n".join(lines)

    def _call_gemini_structured(
        self,
        system_instruction: str,
        user_text: str,
        schema: Type[BaseModel],
        image_for_multimodal: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[dict], Optional[str], Optional[Dict[str, str]]]:
        """Single Gemini call with structured JSON output.

        Returns (formatted, error, debug_info).
        """
        from google.genai import types as genai_types

        debug_info: Dict[str, str] = {
            "model": self.config.ocr_formatter_model,
            "system_prompt": system_instruction,
            "user_text": user_text,
            "sent_image": "0",
            "response_text": "",
        }

        try:
            client = self._genai_client_ready()
        except RuntimeError as e:
            debug_info["response_text"] = f"<no client: {e}>"
            return None, str(e), debug_info

        parts: List[Any] = [user_text]
        if image_for_multimodal is not None and image_for_multimodal.size > 0:
            ok, buf = cv2.imencode(".jpg", image_for_multimodal)
            if ok:
                parts.append(
                    genai_types.Part.from_bytes(
                        data=buf.tobytes(), mime_type="image/jpeg"
                    )
                )
                debug_info["sent_image"] = "1"

        try:
            with _timed(f"gemini.generate({self.config.ocr_formatter_model})"):
                response = client.models.generate_content(
                    model=self.config.ocr_formatter_model,
                    contents=parts,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.0,
                    ),
                )
        except Exception as e:
            _log_llm_io(
                self.config.ocr_formatter_model,
                system_instruction,
                user_text,
                f"<exception: {e}>",
            )
            debug_info["response_text"] = f"<exception: {e}>"
            return None, f"gemini call failed: {e}", debug_info

        text = response.text or ""
        debug_info["response_text"] = text
        _log_llm_io(self.config.ocr_formatter_model, system_instruction, user_text, text)
        if not text:
            return None, "gemini returned empty response", debug_info
        try:
            return json.loads(text), None, debug_info
        except json.JSONDecodeError as e:
            return None, f"gemini response was not valid JSON: {e}", debug_info

    def process_with_custom_template(
        self,
        image: np.ndarray,
        template: "CustomOcrTemplate",
        roi_normalized: Optional[Sequence[float]] = None,
        custom_prompt: str = "",
    ) -> dict:
        """Crop and OCR each region of the template independently, then send
        the per-region text to Gemini together with the region metadata.

        `roi_normalized` is accepted for signature compatibility but ignored
        for custom templates — the template's regions define what gets OCR'd.
        """
        debug = _ocr_debug_enabled()
        timings: Dict[str, float] = {}
        t_total = time.perf_counter()

        import base64

        regions = template.regions or []
        per_region: List[Tuple[dict, str, List[OcrLine], Optional[str]]] = []

        with _timed(f"ocr_regions(n={len(regions)})", timings):
            for region in regions:
                coords = region.get("coords")
                crop_b64: Optional[str] = None
                try:
                    crop_img, _ = self.crop_roi(image, coords)
                    # base64-encode the crop so the frontend can prove
                    # what was actually sent to PaddleOCR for this region.
                    ok, crop_buf = cv2.imencode(".jpg", crop_img)
                    if ok:
                        crop_b64 = (
                            "data:image/jpeg;base64,"
                            + base64.b64encode(crop_buf.tobytes()).decode("ascii")
                        )
                    region_lines = self.run_ocr(image, roi_normalized=coords)
                except Exception as e:
                    print(f"[OCR] region '{region.get('label')}' failed: {e}")
                    region_lines = []
                joined = "\n".join(l.text for l in region_lines)
                per_region.append((region, joined, region_lines, crop_b64))

        # Aggregate stats across regions
        all_lines: List[OcrLine] = []
        region_payload: List[dict] = []
        for region, joined, region_lines, crop_b64 in per_region:
            all_lines.extend(region_lines)
            region_payload.append(
                {
                    "label": region.get("label"),
                    "coords": region.get("coords"),
                    "raw_text": joined,
                    "line_count": len(region_lines),
                    "cropped_image": crop_b64,
                }
            )
        avg_conf = (
            sum(l.confidence for l in all_lines) / len(all_lines)
            if all_lines
            else 0.0
        )
        raw_text = "\n\n".join(
            f"[{r['label']}]\n{r['raw_text'] or '(no text)'}" for r in region_payload
        )

        schema_model = self._build_dynamic_schema(
            regions, model_name=f"Custom_{template.slug.replace('-', '_')}"
        )
        system_instruction = self._build_custom_template_prompt(
            template, per_region, custom_prompt=custom_prompt
        )
        user_text = (
            "Apply the schema using only the per-region OCR text above. "
            "Return the JSON object only."
        )

        with _timed("call_gemini_structured(custom)", timings):
            formatted, format_error, debug_info = self._call_gemini_structured(
                system_instruction=system_instruction,
                user_text=user_text,
                schema=schema_model,
                image_for_multimodal=image if template.multimodal else None,
            )

        elapsed_total = time.perf_counter() - t_total
        if debug:
            print(
                f"[OCR/timing] process_with_custom_template({template.slug}) "
                f"total: {elapsed_total*1000:.0f}ms"
            )
            timings["total"] = elapsed_total

        return {
            "raw_lines": [l.to_dict() for l in all_lines],
            "raw_text": raw_text,
            "confidence_avg": avg_conf,
            "template_key": f"custom:{template.slug}",
            "formatted": formatted,
            "format_error": format_error,
            "regions": region_payload,
            "debug_info": debug_info if debug else None,
            "timings": (
                {k: round(v * 1000, 1) for k, v in timings.items()} if debug else None
            ),
        }

    def process_image_array(
        self,
        image: np.ndarray,
        roi_normalized: Optional[Sequence[float]] = None,
        template_key: str = "raw",
        custom_prompt: str = "",
    ) -> dict:
        """OCR + format on an in-memory ndarray (no annotated image).

        Used by the detection pipeline where we already have the frame as
        a numpy array and don't need a base64-encoded annotated image.
        """
        debug = _ocr_debug_enabled()
        timings: Dict[str, float] = {}
        t_total = time.perf_counter()

        custom_template = self._resolve_custom_template(template_key)
        if custom_template is not None:
            payload = self.process_with_custom_template(
                image,
                custom_template,
                roi_normalized=roi_normalized,
                custom_prompt=custom_prompt,
            )
            if debug:
                elapsed = time.perf_counter() - t_total
                print(
                    f"[OCR/timing] process_image_array(custom) total: "
                    f"{elapsed*1000:.0f}ms"
                )
            return payload

        with _timed("run_ocr", timings):
            lines = self.run_ocr(image, roi_normalized=roi_normalized)
        raw_text = "\n".join(line.text for line in lines)
        avg_conf = (
            sum(line.confidence for line in lines) / len(lines) if lines else 0.0
        )

        crop_for_llm: Optional[np.ndarray] = None
        template = get_template(template_key)
        if template and template.multimodal:
            crop_for_llm, _ = self.crop_roi(image, roi_normalized)

        with _timed("format_lines", timings):
            formatted, resolved_key, format_error, debug_info = self.format_lines(
                template_key,
                raw_text,
                crop=crop_for_llm,
                custom_prompt=custom_prompt,
            )

        elapsed_total = time.perf_counter() - t_total
        if debug:
            print(f"[OCR/timing] process_image_array total: {elapsed_total*1000:.0f}ms")
            timings["total"] = elapsed_total

        return {
            "raw_lines": [line.to_dict() for line in lines],
            "raw_text": raw_text,
            "confidence_avg": avg_conf,
            "template_key": resolved_key,
            "formatted": formatted,
            "format_error": format_error,
            "roi": list(roi_normalized) if roi_normalized else None,
            "debug_info": debug_info if debug else None,
            "timings": {k: round(v * 1000, 1) for k, v in timings.items()} if debug else None,
        }

    @staticmethod
    def bbox_to_normalized_roi(
        bbox_xyxy: Sequence[float],
        image_size: Tuple[int, int],
        padding: int = 0,
    ) -> Optional[List[float]]:
        """Convert a pixel-space bbox [x1,y1,x2,y2] to a normalized ROI.

        image_size is (width, height). Padding is applied in pixel units.
        Returns None if the resulting box is degenerate.
        """
        w, h = image_size
        if w <= 0 or h <= 0:
            return None
        x1, y1, x2, y2 = bbox_xyxy
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        if x2 <= x1 or y2 <= y1:
            return None
        return [float(x1) / w, float(y1) / h, float(x2) / w, float(y2) / h]

    def process_bytes(
        self,
        image_data: bytes,
        roi_normalized: Optional[Sequence[float]] = None,
        include_annotated: bool = False,
        template_key: str = "raw",
        custom_prompt: str = "",
    ) -> dict:
        """Orchestrator used by the standalone /ocr/run endpoint."""
        image = self.decode_image(image_data)

        custom_template = self._resolve_custom_template(template_key)
        if custom_template is not None:
            payload = self.process_with_custom_template(
                image,
                custom_template,
                roi_normalized=roi_normalized,
                custom_prompt=custom_prompt,
            )
            payload["image_size"] = [int(image.shape[1]), int(image.shape[0])]
            if include_annotated and payload.get("raw_lines"):
                lines_for_anno = [
                    OcrLine(
                        text=l["text"],
                        confidence=l["confidence"],
                        bbox=tuple(l["bbox"]),
                    )
                    for l in payload["raw_lines"]
                ]
                annotated = self.annotate_image(image, lines_for_anno)
                _, buf = cv2.imencode(".jpg", annotated)
                import base64

                payload["annotated_image"] = (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(buf.tobytes()).decode("ascii")
                )
            return payload

        debug = _ocr_debug_enabled()
        timings: Dict[str, float] = {}
        t_total = time.perf_counter()

        with _timed("run_ocr", timings):
            lines = self.run_ocr(image, roi_normalized=roi_normalized)

        raw_text = "\n".join(line.text for line in lines)
        avg_conf = (
            sum(line.confidence for line in lines) / len(lines) if lines else 0.0
        )

        crop_for_llm: Optional[np.ndarray] = None
        template = get_template(template_key)
        if template and template.multimodal:
            crop_for_llm, _ = self.crop_roi(image, roi_normalized)

        with _timed("format_lines", timings):
            formatted, resolved_key, format_error, debug_info = self.format_lines(
                template_key,
                raw_text,
                crop=crop_for_llm,
                custom_prompt=custom_prompt,
            )

        elapsed_total = time.perf_counter() - t_total
        if debug:
            print(f"[OCR/timing] process_bytes total: {elapsed_total*1000:.0f}ms")
            timings["total"] = elapsed_total

        payload: dict = {
            "raw_lines": [line.to_dict() for line in lines],
            "raw_text": raw_text,
            "confidence_avg": avg_conf,
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "template_key": resolved_key,
            "formatted": formatted,
            "format_error": format_error,
            "debug_info": debug_info if debug else None,
            "timings": {k: round(v * 1000, 1) for k, v in timings.items()} if debug else None,
        }

        if include_annotated and lines:
            annotated = self.annotate_image(image, lines)
            _, buf = cv2.imencode(".jpg", annotated)
            import base64

            payload["annotated_image"] = (
                "data:image/jpeg;base64,"
                + base64.b64encode(buf.tobytes()).decode("ascii")
            )

        return payload
