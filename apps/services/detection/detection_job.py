"""
RQ job entrypoints for background video detection.

This is the seam that decouples a detection run from the HTTP/SSE connection
that started it. ``run_detection`` is what the `detection` RQ worker executes;
it just delegates to the shared ``DetectionService.process_session`` (which
owns all the logic, so the API request path and the worker path can't drift).

``enqueue_detection_job`` is the SINGLE place a job is put on the queue — used
by the API enqueue endpoint, the in-worker auto-resume on a recoverable error,
the manual Resume endpoint, and the stalled-session reaper. There is no
RQ-level retry on purpose: a blind retry would restart from ``start_frame=0``
and lose progress. Resume is always explicit, from ``processed_frames``.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("apps.detection")

QUEUE_NAME = "detection"


def run_detection(session_id: str, start_frame: int = 0) -> dict:
    """RQ work-horse: run (or resume) detection for one session.

    Imported lazily so importing this module (e.g. to enqueue) never pulls in
    the heavy model stack. In the worker process the first call loads the
    shared services once; subsequent jobs reuse them.
    """
    from apps.api.v1.shared_services import _detection_service

    return _detection_service.process_session(session_id, start_frame=start_frame)


KNOWN_QUEUES = ("detection", "ocr")


def _queue_counts(queue) -> dict:
    return {
        "pending": queue.count,
        "started": len(queue.started_job_registry),
        "failed": len(queue.failed_job_registry),
        "deferred": len(queue.deferred_job_registry),
        "scheduled": len(queue.scheduled_job_registry),
    }


def queue_overview() -> dict:
    """Per-queue job depth + LIVE worker count.

    The worker count is the important bit: a queue with jobs but 0 workers is
    why "everything stays queued". Never raises — a Redis outage comes back as
    an `error` field so the UI can still render.
    """
    out: dict = {}
    try:
        import django_rq
        from rq import Worker
    except Exception as exc:  # noqa: BLE001
        return {name: {"error": str(exc), "total": 0, "workers": 0} for name in KNOWN_QUEUES}

    for name in KNOWN_QUEUES:
        try:
            q = django_rq.get_queue(name)
            counts = _queue_counts(q)
            out[name] = {
                **counts,
                "total": sum(counts.values()),
                "workers": len(Worker.all(queue=q)),
            }
        except Exception as exc:  # noqa: BLE001 - Redis down / lookup error
            out[name] = {"error": str(exc), "total": 0, "workers": 0}
    return out


def purge_queue(queue_name: str) -> dict:
    """Drop pending jobs + clear the failed/deferred/scheduled registries for a
    queue. Returns the per-bucket count removed.

    Deliberately does NOT touch the `started` registry: a started job is one a
    worker is actively running right now (e.g. a long detection), and deleting
    its record mid-run just orphans bookkeeping without stopping the work. To
    stop a running detection, cancel its session instead.
    """
    import django_rq
    from rq.job import Job
    from rq.exceptions import NoSuchJobError

    q = django_rq.get_queue(queue_name)
    removed = {"pending": q.count}
    q.empty()
    registries = (
        ("failed", q.failed_job_registry),
        ("deferred", q.deferred_job_registry),
        ("scheduled", q.scheduled_job_registry),
    )
    for label, registry in registries:
        n = 0
        try:
            for jid in registry.get_job_ids():
                try:
                    Job.fetch(jid, connection=registry.connection).delete()
                    n += 1
                except NoSuchJobError:
                    try:
                        registry.remove(jid)
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001 - best-effort per job
                    pass
        except Exception:  # noqa: BLE001
            pass
        removed[label] = n
    removed["total"] = sum(removed.values())
    return removed


def enqueue_detection_job(session_id: str, start_frame: int = 0) -> Optional[str]:
    """Enqueue a detection job. Returns the RQ job id, or None if the queue /
    Redis is unavailable (the caller decides how to surface that)."""
    try:
        import django_rq

        queue = django_rq.get_queue(QUEUE_NAME)
        job = queue.enqueue(
            "apps.services.detection.detection_job.run_detection",
            session_id,
            start_frame,
        )
        logger.info(
            "Enqueued detection job %s session=%s start_frame=%s",
            job.id, session_id, start_frame,
        )
        return job.id
    except Exception as exc:  # noqa: BLE001 - Redis down / import error
        logger.exception("Failed to enqueue detection job for %s: %s", session_id, exc)
        return None
