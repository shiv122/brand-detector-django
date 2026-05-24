"""
RQ task functions for asynchronous OCR.

These run inside `python manage.py rqworker ocr` and are the only entry
point that talks to the OCR HTTP endpoint outside of the synchronous
`/ocr/run` route. Detection enqueues one of these per image / per frame
and returns immediately; the frontend polls `/api/v1/ocr/jobs` to learn
when results are ready.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


_OCR_SERVICE = None


def _load_service():
    """Build an OcrService directly — do NOT go through shared_services.

    shared_services instantiates ModelService + ClassificationService at
    import time, which eagerly loads every YOLO weight onto MPS. Doing
    that inside an RQ work-horse (a fork() child of the worker) on macOS
    segfaults with signal 11 — PyTorch/MPS state does not survive fork.
    The OCR worker only needs the HTTP client to RunPod, so we build a
    fresh OcrService here and cache it on the work-horse for the lifetime
    of the job.
    """
    global _OCR_SERVICE
    if _OCR_SERVICE is None:
        from apps.services.ocr.ocr_service import OcrService
        from config.app_config import AppConfig

        _OCR_SERVICE = OcrService(AppConfig())
    return _OCR_SERVICE


def run_ocr_from_path(
    image_path: str,
    prompt: str,
    prompt_meta: Optional[Dict[str, Any]] = None,
    frame_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Worker entry point: read the image off disk and run OCR.

    `frame_id` (when present) makes the worker also persist the OCR result
    onto the matching Frame row so that the video page can hydrate from
    DB on reload — the Redis result is the realtime channel, the DB is
    the durable copy.
    """
    p = Path(image_path)
    if not p.exists():
        return {
            "error": f"image not found: {image_path}",
            "prompt": prompt,
        }

    image_bytes = p.read_bytes()
    service = _load_service()
    result = service.run(image_bytes, prompt)
    if prompt_meta:
        result["prompt_meta"] = prompt_meta

    if frame_id is not None:
        try:
            from apps.core.models import Frame

            Frame.objects.filter(id=frame_id).update(ocr_summary=result)
        except Exception as exc:  # noqa: BLE001 - never fail the job over DB hiccups
            result.setdefault("warnings", []).append(
                f"failed to persist ocr_summary: {exc}"
            )

    return result
