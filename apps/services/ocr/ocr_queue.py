"""
Thin wrapper around django-rq for OCR enqueueing.

Centralises (a) the queue name, (b) sane defaults (timeout, result TTL),
and (c) the graceful fallback when Redis is unreachable so a dead Redis
doesn't take detection down with it — the API still returns detections,
the frontend just sees `ocr_job: { status: "unavailable" }` per item.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


QUEUE_NAME = "ocr"


def enqueue_ocr_job(
    image_path: str,
    prompt: str,
    prompt_meta: Optional[Dict[str, Any]] = None,
    frame_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Enqueue an OCR run. Returns a dict the controller hands to the client.

    Shape:
        { "id": <rq job id>, "status": "queued" }
        { "status": "unavailable", "error": "<reason>" }   on failure
    """
    try:
        import django_rq

        queue = django_rq.get_queue(QUEUE_NAME)
        job = queue.enqueue(
            "apps.services.ocr.ocr_tasks.run_ocr_from_path",
            image_path,
            prompt,
            prompt_meta,
            frame_id,
        )
        return {"id": job.id, "status": "queued"}
    except Exception as exc:  # noqa: BLE001 - any connect/import error is "unavailable"
        return {"status": "unavailable", "error": str(exc)}


def fetch_jobs(job_ids: list[str]) -> Dict[str, Dict[str, Any]]:
    """Return a {job_id: {status, result?, error?}} map for the given IDs.

    Statuses align with RQ: queued / started / finished / failed / deferred /
    canceled. Unknown IDs (TTL expired or never existed) come back as
    `{"status": "unknown"}` so the frontend can stop polling them.
    """
    out: Dict[str, Dict[str, Any]] = {}
    try:
        import django_rq
        from rq.job import Job
        from rq.exceptions import NoSuchJobError

        queue = django_rq.get_queue(QUEUE_NAME)
        connection = queue.connection
    except Exception as exc:  # noqa: BLE001
        for jid in job_ids:
            out[jid] = {"status": "unavailable", "error": str(exc)}
        return out

    for jid in job_ids:
        try:
            job = Job.fetch(jid, connection=connection)
        except NoSuchJobError:
            out[jid] = {"status": "unknown"}
            continue
        except Exception as exc:  # noqa: BLE001
            out[jid] = {"status": "error", "error": str(exc)}
            continue

        raw_status = job.get_status(refresh=True) or "unknown"
        # RQ versions vary: some return a JobStatus enum, some a plain str.
        # Normalise to a string so DRF can JSON-serialise it without coercion.
        status_str = getattr(raw_status, "value", raw_status)
        if not isinstance(status_str, str):
            status_str = str(status_str)
        entry: Dict[str, Any] = {"status": status_str}
        if status_str == "finished":
            entry["result"] = job.result
        elif status_str == "failed":
            entry["error"] = (job.exc_info or "").splitlines()[-1] if job.exc_info else "job failed"
        out[jid] = entry
    return out
