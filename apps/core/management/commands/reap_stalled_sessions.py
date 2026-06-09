"""
Reaper for detection sessions whose worker died WITHOUT running any cleanup —
an OOM SIGKILL, a container redeploy mid-run, a hard crash. In those cases no
in-process handler fires, so the session would otherwise sit in PROCESSING
forever. This command finds them by stale heartbeat and re-enqueues them from
processed_frames (bounded by DETECTION_MAX_ATTEMPTS).

It deliberately does NOT import the model stack (only the ORM + the enqueue
helper), so running it on a tight loop is cheap. Run it on a loop in
production (supervisord) with SKIP_MODEL_PRELOAD=1:

    python manage.py reap_stalled_sessions --loop --interval 30

Or one-shot from cron:

    python manage.py reap_stalled_sessions
"""

import logging
import time
from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db.models import Q
from django.utils import timezone

from apps.core.models import ProcessingSession
from apps.core.enums import ProcessingStatus
from apps.services.detection.detection_job import enqueue_detection_job

logger = logging.getLogger("apps.detection")


class Command(BaseCommand):
    help = "Re-enqueue detection sessions whose worker died (stale heartbeat)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--loop", action="store_true",
            help="Run continuously instead of a single pass.",
        )
        parser.add_argument(
            "--interval", type=int, default=30,
            help="Seconds between scans in --loop mode (default 30).",
        )

    def handle(self, *args, **options):
        if options["loop"]:
            interval = max(5, int(options["interval"]))
            logger.info("Reaper loop started (interval=%ss)", interval)
            while True:
                try:
                    self._scan_once()
                except Exception:  # noqa: BLE001 - never let the loop die
                    logger.exception("Reaper scan failed")
                time.sleep(interval)
        else:
            n = self._scan_once()
            self.stdout.write(self.style.SUCCESS(f"Reaper revived {n} session(s)."))

    def _scan_once(self) -> int:
        stale_seconds = int(getattr(settings, "DETECTION_HEARTBEAT_STALE_SECONDS", 120))
        now = timezone.now()
        heartbeat_cutoff = now - timedelta(seconds=stale_seconds)
        # A QUEUED session that never started is only an orphan after a longer
        # grace period (the worker may simply be busy with another job).
        orphan_cutoff = now - timedelta(seconds=stale_seconds * 3)

        # 1) PROCESSING with a stale (or missing) heartbeat -> worker is dead.
        dead = ProcessingSession.objects.filter(
            status=ProcessingStatus.PROCESSING.value
        ).filter(
            Q(heartbeat_at__lt=heartbeat_cutoff)
            | Q(heartbeat_at__isnull=True, updated_at__lt=heartbeat_cutoff)
        )

        # 2) QUEUED but never picked up for a long time -> job likely lost.
        orphans = ProcessingSession.objects.filter(
            status=ProcessingStatus.QUEUED.value,
            heartbeat_at__isnull=True,
            updated_at__lt=orphan_cutoff,
        )

        revived = 0
        for session in list(dead) + list(orphans):
            if self._revive(session):
                revived += 1
        return revived

    def _revive(self, session: ProcessingSession) -> bool:
        max_attempts = int(getattr(settings, "DETECTION_MAX_ATTEMPTS", 5))
        if (session.attempts or 0) >= max_attempts:
            session.mark_failed(
                f"Worker died and max attempts ({max_attempts}) reached.",
                stop_reason="stalled",
            )
            logger.error(
                "Reaper giving up on session=%s after %s attempts",
                session.session_id, session.attempts,
            )
            return False

        if not session.source_url:
            session.mark_failed(
                "Worker died and no source_url to resume from.",
                stop_reason="stalled",
            )
            logger.error("Reaper cannot resume session=%s (no source)", session.session_id)
            return False

        start_frame = session.processed_frames or 0
        session.status = ProcessingStatus.QUEUED.value
        session.stop_reason = "stalled"
        session.error_message = "Worker stopped responding; resumed by reaper."
        # Clear the heartbeat so a second reaper pass before pickup doesn't
        # immediately re-trigger on this same row.
        session.heartbeat_at = None
        session.save(
            update_fields=["status", "stop_reason", "error_message", "heartbeat_at", "updated_at"]
        )
        job_id = enqueue_detection_job(session.session_id, start_frame=start_frame)
        logger.warning(
            "Reaper revived session=%s from frame=%s (attempt %s, job=%s)",
            session.session_id, start_frame, session.attempts, job_id,
        )
        return job_id is not None
