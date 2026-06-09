"""
LocateAnything OCR engine — a detection-style alternative to GLM OCR.

It asks the external LocateAnything service to LOCATE brands / logos / text (or
any query) and returns bounding boxes in the ORIGINAL frame's pixel space, ready
to overlay. Stored on Frame.ocr_summary like a GLM result (engine: "locate").

Two layers:
  - task   : how boxes are found — "ocr" (find + read all text, one call) or
             "detection" (find brand/logo/text regions for a query).
  - reader : for the detection task, how each box's TEXT is filled in —
               "rapidocr"  : crop the box + read it with RapidOCR (local, accurate)
               "tesseract" : crop the box + read it with Tesseract (local, fast)
               "locate"    : a second model pass ("detect all text") merged by overlap
               "none"/""   : boxes only (no text)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from config.app_config import AppConfig

_RAPIDOCR = None  # cached RapidOCR engine (loads ONNX models once)


def _get_rapidocr():
    global _RAPIDOCR
    if _RAPIDOCR is None:
        from rapidocr_onnxruntime import RapidOCR

        _RAPIDOCR = RapidOCR()
    return _RAPIDOCR


def _attach_text(det_boxes, ocr_boxes) -> None:
    """Label each detection box with the OCR text whose center falls inside it.

    Boxes from both passes share the same original-frame pixel space, so a box
    "contains" a piece of text when that text's center lies within it. Multiple
    words inside one region are joined in reading order (top-to-bottom, then
    left-to-right). Logo regions with no text simply get no `text`.
    """
    for d in det_boxes:
        box = d.get("box") or []
        if len(box) != 4:
            continue
        dx1, dy1, dx2, dy2 = box
        inside = []
        for o in ocr_boxes:
            label = (o.get("label") or "").strip()
            ob = o.get("box") or []
            if not label or len(ob) != 4:
                continue
            cx = (ob[0] + ob[2]) / 2.0
            cy = (ob[1] + ob[3]) / 2.0
            if dx1 <= cx <= dx2 and dy1 <= cy <= dy2:
                inside.append((round(ob[1], 1), round(ob[0], 1), label))
        if inside:
            inside.sort()
            d["text"] = " ".join(t for _, _, t in inside)


def _crop_box(img, box, pad: int):
    """Crop a detection box from the image with `pad` px on every side.

    Normalizes coordinate order (the model can emit x2<x1), clamps to the image,
    and returns None for degenerate boxes that would crash crop().
    """
    if len(box) != 4:
        return None
    w, h = img.size
    x1, y1, x2, y2 = box
    left = max(0, int(min(x1, x2)) - pad)
    upper = max(0, int(min(y1, y2)) - pad)
    right = min(w, int(max(x1, x2)) + pad)
    lower = min(h, int(max(y1, y2)) + pad)
    if right <= left or lower <= upper:
        return None
    return img.crop((left, upper, right, lower))


class LocateService:
    def __init__(self, config: AppConfig):
        self.config = config

    def is_available(self) -> bool:
        return bool(self.config.locate_host)

    def run(self, image_url: str, task: str = "", query: str = "", reader: str = "") -> Dict[str, Any]:
        from apps.services.ocr.locate_anything_client import LocateAnythingClient

        task = (task or self.config.locate_default_task or "ocr").strip().lower()
        query = (query or "").strip()
        if not query and task in ("detection", "grounding"):
            query = self.config.locate_default_query
        reader = (reader or "").strip().lower()

        client = LocateAnythingClient(
            host=self.config.locate_host,
            timeout_seconds=self.config.locate_timeout_seconds,
        )
        data, timing = client.locate(
            image_url,
            task=task,
            query=query,
            mode=self.config.locate_mode or None,
            max_tokens=self.config.locate_max_new_tokens or None,
        )
        timing_ms = {f"locate_{k}": v for k, v in (timing or {}).items()}

        if data is None:
            return {
                "provider": "locate",
                "engine": "locate",
                "task": task,
                "query": query,
                "reader": reader or None,
                "error": client.load_error() or "LocateAnything request failed.",
                "timing_ms": timing_ms,
            }

        boxes = data.get("boxes") or []

        # For box-detection, fill each box's text with the chosen reader. (The OCR
        # task already reads text inline, so it needs no reader.)
        if task == "detection" and boxes and reader and reader != "none":
            if reader == "rapidocr":
                self._read_local(image_url, boxes, timing_ms, "rapidocr")
            elif reader == "tesseract":
                self._read_local(image_url, boxes, timing_ms, "tesseract")
            elif reader == "locate":
                ocr_data, ocr_timing = client.locate(
                    image_url,
                    task="ocr",
                    query="",
                    mode=self.config.locate_mode or None,
                    max_tokens=self.config.locate_max_new_tokens or None,
                )
                timing_ms.update({f"locate_ocr_{k}": v for k, v in (ocr_timing or {}).items()})
                if ocr_data:
                    _attach_text(boxes, ocr_data.get("boxes") or [])

        return {
            "provider": "locate",
            "engine": "locate",
            "task": task,
            "query": query,
            "reader": reader or None,
            "boxes": boxes,
            "points": data.get("points") or [],
            # dimensions of the frame the boxes are scaled to (NOT image bytes —
            # named to avoid the _slim_result image-key strip).
            "image_size": data.get("image") or {},
            "raw_text": data.get("text") or "",
            "prompt": data.get("prompt") or "",
            "model": data.get("model"),
            "timing_ms": timing_ms,
        }

    def _fetch_image(self, image_url: str, timing_ms, key: str) -> Optional[Any]:
        """Download the source image once for a local reader. Returns a PIL image
        (or None, recording the failure in timing_ms)."""
        import io

        import requests

        try:
            from PIL import Image
        except ImportError as e:  # noqa: BLE001
            timing_ms[f"{key}_error"] = f"Pillow not installed: {e}"
            return None
        try:
            resp = requests.get(image_url, timeout=self.config.locate_timeout_seconds)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:  # noqa: BLE001
            timing_ms[f"{key}_error"] = f"image fetch failed: {e}"
            return None

    def _read_local(self, image_url: str, boxes, timing_ms, engine: str) -> None:
        """Crop each detection box and read it with a local OCR engine."""
        import time

        img = self._fetch_image(image_url, timing_ms, engine)
        if img is None:
            return

        pad = self.config.locate_crop_pad
        t0 = time.monotonic()

        if engine == "rapidocr":
            try:
                ocr = _get_rapidocr()
                import numpy as np
            except Exception as e:  # noqa: BLE001
                timing_ms["rapidocr_error"] = f"rapidocr unavailable: {e}"
                return
            for b in boxes:
                crop = _crop_box(img, b.get("box") or [], pad)
                if crop is None:
                    continue
                try:
                    # recognition-only: the crop IS the text region — skip detection
                    res, _ = ocr(np.array(crop), use_det=False, use_cls=False, use_rec=True)
                except Exception:  # noqa: BLE001
                    res = None
                txt = _rapidocr_text(res)
                if txt:
                    b["text"] = txt
            timing_ms["rapidocr_ms"] = int((time.monotonic() - t0) * 1000)
            return

        # tesseract
        try:
            import pytesseract
        except ImportError as e:  # noqa: BLE001
            timing_ms["tesseract_error"] = f"pytesseract not installed: {e}"
            return
        psm = self.config.locate_tesseract_psm
        for b in boxes:
            crop = _crop_box(img, b.get("box") or [], pad)
            if crop is None:
                continue
            # Tesseract is much better on larger, grayscale text — upscale tiny
            # crops to ~48px tall and drop colour before reading.
            cw, ch = crop.size
            if ch and ch < 48:
                from PIL import Image

                crop = crop.resize((max(1, round(cw * 48 / ch)), 48), Image.LANCZOS)
            crop = crop.convert("L")
            try:
                txt = pytesseract.image_to_string(crop, config=f"--psm {psm}").strip()
            except Exception:  # noqa: BLE001
                txt = ""
            if txt:
                b["text"] = " ".join(txt.split())
        timing_ms["tesseract_ms"] = int((time.monotonic() - t0) * 1000)


def _rapidocr_text(res) -> str:
    """Pull the joined text out of a RapidOCR result (rec-only or full)."""
    if not res:
        return ""
    parts = []
    for item in res:
        for el in item:
            if isinstance(el, str):
                el = el.strip()
                if el:
                    parts.append(el)
                break
    return " ".join(" ".join(parts).split())
