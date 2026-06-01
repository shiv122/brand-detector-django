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

# Fields the UI / exports actually read off a stored OCR result. Everything
# else the engine produces is debug-only and just bloats Redis + the DB +
# the poll payload (multiplied by every frame of a long video).
#   formatted          -> the structured JSON shown in OcrBlock
#   raw_text           -> "Raw response" expander + the XLSX export
#   error/format_error -> error rendering
#   prompt_meta        -> the prompt slug badge
#   timing_ms          -> the "N ms" badge
#   provider/text_formatter_provider/*_id/*_model -> tiny scalars, kept
_RESULT_KEEP_KEYS = frozenset({
    "formatted",
    "raw_text",
    "error",
    "format_error",
    "warnings",
    "prompt_meta",
    "timing_ms",
    "provider",
    "text_formatter_provider",
    "deepseek_text_id",
    "deepseek_text_model",
    "gemini_text_id",
    "gemini_text_model",
})

# Defence-in-depth: keys that could ever carry image bytes/base64. The OCR
# pipeline never puts an image in its result (the worker is handed a *path*,
# not bytes, and returns only text/JSON), but we strip these unconditionally
# so a future change can't silently start persisting images into Redis/DB.
_IMAGE_LIKE_KEYS = frozenset({
    "image", "images", "image_b64", "image_base64", "image_data",
    "annotated_image", "frame", "frame_b64", "frame_image", "thumbnail",
})

# Hard ceiling on the stored raw OCR text. A single frame's text is normally a
# few hundred bytes to a few KB; this only trips on a pathological model dump,
# keeping one bad frame from bloating the row.
_MAX_RAW_TEXT_CHARS = 20_000


def _slim_result(result):
    """Return a storage-safe copy of an OCR result.

    Drops debug-only / redundant fields (glm_ocr_text duplicates raw_text;
    formatter_raw_text and the full prompt are never read), strips any
    image-like key, and caps raw_text. Applied to the copy persisted to Redis
    (the job return value) and the DB — NOT to the synchronous /ocr/run
    response, which keeps the full payload for the single-image debug view.
    """
    if not isinstance(result, dict):
        return result
    slim = {k: v for k, v in result.items()
            if k in _RESULT_KEEP_KEYS and k not in _IMAGE_LIKE_KEYS}
    raw = slim.get("raw_text")
    if isinstance(raw, str) and len(raw) > _MAX_RAW_TEXT_CHARS:
        slim["raw_text"] = raw[:_MAX_RAW_TEXT_CHARS] + "\n…[truncated]"
    return slim


def _cleanup_ocr_input(image_path: str) -> None:
    """Delete the worker's input frame once it has been read.

    Detection writes one downscaled JPEG per OCR'd frame into
    `<static>/ocr_inputs/`; without this they accumulate forever (a long
    video leaves thousands of files = GBs of dead disk). We only delete files
    that actually live under ocr_inputs/ — never an arbitrary path — and never
    let a cleanup hiccup fail the job.

    IMPORTANT: this runs only after `service.run()` returns normally. If the
    work-horse raises (e.g. the RQ job timeout fires mid-call), we skip the
    delete so the file survives for the retried attempt to re-read.
    """
    try:
        from config.app_config import AppConfig

        p = Path(image_path).resolve()
        ocr_dir = (Path(AppConfig().static_dir) / "ocr_inputs").resolve()
        if ocr_dir in p.parents:
            p.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - cleanup must never break the job
        pass


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
    # If service.run raises (incl. the RQ job-timeout death penalty), we fall
    # straight out of this function WITHOUT deleting the input — so the retry
    # can re-read it. Cleanup below only happens once we have a result in hand.
    result = service.run(image_bytes, prompt)
    if prompt_meta:
        result["prompt_meta"] = prompt_meta

    # Trim debug-only / redundant fields and guarantee no image bytes land in
    # Redis (the job return value) or the DB. The sync /ocr/run path keeps the
    # full result; only this queued path is slimmed.
    result = _slim_result(result)

    if frame_id is not None:
        try:
            from apps.core.models import Frame

            Frame.objects.filter(id=frame_id).update(ocr_summary=result)
        except Exception as exc:  # noqa: BLE001 - never fail the job over DB hiccups
            result.setdefault("warnings", []).append(
                f"failed to persist ocr_summary: {exc}"
            )

    # Success or a handled error dict (neither re-raises, so neither retries):
    # the input frame is no longer needed — reclaim the disk.
    _cleanup_ocr_input(image_path)

    return result
