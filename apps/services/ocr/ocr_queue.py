"""
Thin wrapper around django-rq for OCR enqueueing.

Centralises (a) the queue name, (b) sane defaults (timeout, result TTL),
and (c) the graceful fallback when Redis is unreachable so a dead Redis
doesn't take detection down with it — the API still returns detections,
the frontend just sees `ocr_job: { status: "unavailable" }` per item.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


QUEUE_NAME = "ocr"

# Hard cap on how many job IDs a single poll will resolve. A pathological
# client (or a multi-hour video) could send tens of thousands of IDs; even a
# single pipelined round-trip has a cost, so we bound it. The frontend already
# stops polling terminal jobs, so the pending set it sends shrinks over time —
# this is just a backstop against an unbounded request.
_MAX_IDS_PER_POLL = 5000


def _retry_spec():
    """Build an rq.Retry from env, or None to disable.

    A long video enqueues thousands of jobs; transient breakage (Ollama cold
    start, a momentary slot saturation, an RQ work-horse SIGKILL'd by the job
    timeout) would otherwise leave a frame's OCR permanently failed. Retrying
    a few times with spacing lets the job land once the backlog drains.

    Only an *unhandled exception* (incl. the RQ job-timeout) triggers a retry;
    the OCR service returns an error *dict* for expected failures (no raise),
    so those don't burn retries. `interval` requeues the job via the scheduled
    registry — the frontend keeps polling it (non-terminal) until it resolves.
    """
    try:
        max_retries = int(os.environ.get("RQ_OCR_JOB_RETRIES", "3"))
    except ValueError:
        max_retries = 3
    if max_retries <= 0:
        return None
    try:
        from rq import Retry
    except Exception:  # noqa: BLE001 - very old rq without Retry
        return None
    return Retry(max=max_retries, interval=[10, 30, 60])


def enqueue_ocr_job(
    image_url: str,
    prompt: str,
    prompt_meta: Optional[Dict[str, Any]] = None,
    frame_id: Optional[int] = None,
    session_id: Optional[str] = None,
    frame_number: Optional[int] = None,
) -> Dict[str, Any]:
    """Enqueue an OCR run for an image URL. Returns a dict for the client.

    Shape:
        { "id": <rq job id>, "status": "queued" }
        { "status": "unavailable", "error": "<reason>" }   on failure

    The worker persists the result onto a Frame row. Pass EITHER `frame_id`
    (when the row already exists) OR `(session_id, frame_number)` — the latter
    lets the detection path enqueue OCR *before* the Frame row is committed, so
    OCR runs concurrently with the detection call instead of waiting for it.
    """
    try:
        import django_rq

        queue = django_rq.get_queue(QUEUE_NAME)
        enqueue_kwargs: Dict[str, Any] = {}
        retry = _retry_spec()
        if retry is not None:
            enqueue_kwargs["retry"] = retry
        job = queue.enqueue(
            "apps.services.ocr.ocr_tasks.run_ocr_from_url",
            image_url,
            prompt,
            prompt_meta,
            frame_id,
            session_id,
            frame_number,
            **enqueue_kwargs,
        )
        return {"id": job.id, "status": "queued"}
    except Exception as exc:  # noqa: BLE001 - any connect/import error is "unavailable"
        return {"status": "unavailable", "error": str(exc)}


def _status_str(job) -> str:
    """Read a job's status as a plain string WITHOUT a fresh Redis round-trip.

    `Job.fetch_many` has already loaded the status off the job hash via
    `restore()`, so `get_status(refresh=False)` returns the just-fetched value.
    RQ versions vary between a JobStatus enum and a plain str — normalise.
    """
    raw = job.get_status(refresh=False) or "unknown"
    s = getattr(raw, "value", raw)
    return s if isinstance(s, str) else str(s)


def _finished_result(job) -> Any:
    """Best-effort return value for a finished job.

    Prefer the value `restore()` already loaded inline (free); only hit the
    Result registry if it wasn't inlined. Never raise — a single unreadable
    result must not fail the whole poll.
    """
    inline = getattr(job, "_result", None)
    if inline is not None:
        return inline
    for getter in ("return_value", "result"):
        try:
            attr = getattr(job, getter)
            return attr(refresh=False) if callable(attr) else attr
        except Exception:  # noqa: BLE001
            continue
    return None


def _failure_error(job) -> str:
    try:
        res = job.latest_result()
        if res is not None and getattr(res, "exc_string", None):
            return res.exc_string.strip().splitlines()[-1]
    except Exception:  # noqa: BLE001
        pass
    try:
        info = job.exc_info
        if info:
            return info.strip().splitlines()[-1]
    except Exception:  # noqa: BLE001
        pass
    return "job failed"


def fetch_jobs(job_ids: list[str]) -> Dict[str, Dict[str, Any]]:
    """Return a {job_id: {status, result?, error?}} map for the given IDs.

    Statuses align with RQ: queued / started / finished / failed / deferred /
    scheduled / canceled. Unknown IDs (TTL expired or never existed) come back
    as `{"status": "unknown"}` so the frontend can stop polling them.

    All pending IDs are resolved in a SINGLE pipelined Redis round-trip via
    `Job.fetch_many` — previously this looped `Job.fetch` + a refreshing
    `get_status` per ID (2N round-trips), which on a long video's thousands of
    pending jobs took tens of seconds and pinned a web worker per poll, the
    main cause of "everything stops". Only the few jobs that are actually
    finished/failed pay an extra read for their result/error payload.
    """
    out: Dict[str, Dict[str, Any]] = {}
    ids: List[str] = list(job_ids)[:_MAX_IDS_PER_POLL]
    if not ids:
        return out
    try:
        import django_rq
        from rq.job import Job

        queue = django_rq.get_queue(QUEUE_NAME)
        connection = queue.connection
        jobs = Job.fetch_many(ids, connection=connection)
    except Exception as exc:  # noqa: BLE001 - Redis down / import error
        for jid in ids:
            out[jid] = {"status": "unavailable", "error": str(exc)}
        return out

    for jid, job in zip(ids, jobs):
        if job is None:
            # TTL expired or never existed — terminal from the client's view.
            out[jid] = {"status": "unknown"}
            continue
        try:
            status_str = _status_str(job)
            entry: Dict[str, Any] = {"status": status_str}
            if status_str == "finished":
                entry["result"] = _finished_result(job)
            elif status_str == "failed":
                entry["error"] = _failure_error(job)
            out[jid] = entry
        except Exception as exc:  # noqa: BLE001 - never let one job break the poll
            out[jid] = {"status": "error", "error": str(exc)}
    return out
