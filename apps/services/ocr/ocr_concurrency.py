"""
Cross-worker concurrency limiter for outgoing Ollama (GLM-OCR) calls.

Ollama serves one request at a time per loaded model (OLLAMA_NUM_PARALLEL=1).
We run multiple RQ workers (12 by default) — without a limiter they all
race against the single Ollama slot, latency per call collapses from
~5s to >100s, and the queue effectively stalls (the "stuck at 6%"
failure mode).

This module gives the workers a shared counter in Redis. Before a worker
calls Ollama it has to acquire a slot; if the in-flight count is already
at the cap, it sleeps and retries. Slots have a safety TTL so a crashed
worker can't permanently leak one.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

_INFLIGHT_KEY = "ocr:ollama:inflight"
_SLOT_SAFETY_TTL = 600  # seconds — outlasts any reasonable single Ollama call


def _get_redis():
    """Get the same Redis connection RQ uses, so the counter is visible
    to every worker that processes the `ocr` queue."""
    from django_rq.queues import get_connection

    return get_connection("ocr")


def _max_concurrent() -> int:
    """How many concurrent Ollama calls the cluster as a whole is allowed.
    Default 2 — slightly above OLLAMA_NUM_PARALLEL=1 so one request is
    actively executing while another is hitting the server's accept queue."""
    try:
        return max(1, int(os.environ.get("LOCAL_OCR_OLLAMA_MAX_CONCURRENT", "2")))
    except ValueError:
        return 2


def _acquire_timeout() -> float:
    """Maximum time a worker is willing to wait for a slot before giving up.

    MUST be strictly less than the RQ job timeout (RQ_OCR_TIMEOUT, default
    600s). Previously this defaulted to 300s while the job timeout was 180s —
    so a job waiting for a slot was killed by RQ (death penalty) before it
    could raise its own clean TimeoutError, leaving the slot/limiter state
    ragged and the failure un-retryable. With the default at 150s the limiter
    times out first: the job ends cleanly and (via rq.Retry) is requeued to
    try again once the backlog drains."""
    try:
        return float(os.environ.get("LOCAL_OCR_OLLAMA_SLOT_TIMEOUT_SECONDS", "150"))
    except ValueError:
        return 150.0


@contextmanager
def ollama_slot() -> Iterator[None]:
    """Block until a slot is available, then run the wrapped call.

    Raises:
        TimeoutError: if no slot becomes available within the configured
            timeout. The caller should treat this as a transient failure
            (e.g. retry the RQ job later, not crash the worker).
    """
    max_concurrent = _max_concurrent()
    deadline = time.monotonic() + _acquire_timeout()
    r = _get_redis()
    poll = 0.25  # short — Ollama calls are seconds, not minutes
    acquired = False

    while True:
        # INCR is atomic; the first incrementer also sets the safety TTL so
        # a SIGKILL'd worker can't permanently hold a slot.
        n = r.incr(_INFLIGHT_KEY)
        if n == 1:
            r.expire(_INFLIGHT_KEY, _SLOT_SAFETY_TTL)
        if n <= max_concurrent:
            acquired = True
            break
        # No room — give the slot back and wait.
        r.decr(_INFLIGHT_KEY)
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"OCR slot timeout: {_acquire_timeout():.0f}s waiting for a "
                f"slot (max_concurrent={max_concurrent}). Ollama is likely "
                "saturated or unreachable."
            )
        # Light exponential creep so we don't hammer Redis under contention.
        time.sleep(poll)
        poll = min(poll * 1.5, 2.0)

    try:
        yield
    finally:
        if acquired:
            try:
                r.decr(_INFLIGHT_KEY)
            except Exception:
                # If Redis hiccupped at the end, the safety TTL will reclaim
                # the slot. Never let the cleanup raise over the actual call.
                logger.warning("ollama_slot: failed to DECR inflight counter")


def inflight_count() -> int:
    """Diagnostic: current in-flight count, or 0 if Redis is unreachable."""
    try:
        val = _get_redis().get(_INFLIGHT_KEY)
    except Exception:
        return 0
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def reset_inflight() -> None:
    """Clear the counter — call once at worker boot in case a previous
    crash left a stale value larger than the actual number of workers."""
    try:
        _get_redis().delete(_INFLIGHT_KEY)
    except Exception:
        logger.warning("reset_inflight: failed to clear counter (Redis down?)")
