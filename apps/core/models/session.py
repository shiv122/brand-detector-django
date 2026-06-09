from django.db import models
from django.utils import timezone
from apps.core.enums import ProcessingStatus


class ProcessingSession(models.Model):
    """Model for tracking video/image processing sessions.

    The detection job runs in a background RQ worker, NOT inside the HTTP/SSE
    request, so this row is the durable source of truth for a run: the worker
    writes progress + a heartbeat here, the SSE progress endpoint reads from
    here, and the reaper resumes from here. `source_url` + `processed_frames`
    are what make resume-from-where-it-stopped possible.
    """

    session_id = models.CharField(max_length=255, unique=True, db_index=True)
    video_filename = models.CharField(max_length=255, null=True, blank=True)
    video_path = models.CharField(max_length=500, null=True, blank=True)
    processed_video_path = models.CharField(max_length=500, null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=[(status.value, status.value) for status in ProcessingStatus],
        default=ProcessingStatus.PENDING.value,
        db_index=True,
    )
    frames_per_second = models.IntegerField(default=2)
    confidence_threshold = models.FloatField(default=0.5)
    # NOTE: total_frames is the number of frames we EXPECT to process at the
    # target fps (duration_seconds * frames_per_second), NOT the video's raw
    # frame count. That makes processed_frames / total_frames a true progress
    # ratio everywhere (sessions list, dashboard, SSE denominator).
    total_frames = models.IntegerField(default=0)
    processed_frames = models.IntegerField(default=0)
    settings = models.JSONField(default=dict, blank=True)  # Store processing settings in JSON format

    # --- Durable-job fields (background worker + resume + observability) ---
    # The original source the worker (re)fetches frames from. For the normal
    # flow this is the presigned Spaces GET URL; for a direct upload it is a
    # local path under static_dir that we keep until the run completes.
    source_url = models.TextField(blank=True, default="")
    source_is_local = models.BooleanField(default=False)
    # Detected source fps (informational / sanity — the worker samples by time
    # via ffmpeg's fps= filter, so a misread here no longer breaks sampling).
    video_fps = models.FloatField(default=0.0)
    # Full per-run params the worker needs (conf, flags, weights, resolved OCR
    # prompt + meta) so it never depends on the originating HTTP request.
    run_params = models.JSONField(default=dict, blank=True)
    # Why the run stopped: eof | error | exception | cancelled | interrupted.
    stop_reason = models.CharField(max_length=32, blank=True, default="")
    # Human-readable failure detail (last line of the traceback), surfaced to
    # the UI so a stop is never silent again.
    error_message = models.TextField(blank=True, default="")
    # Liveness: the worker bumps this every few seconds. A PROCESSING session
    # whose heartbeat is stale is treated as dead and re-enqueued by the reaper.
    heartbeat_at = models.DateTimeField(null=True, blank=True, db_index=True)
    # How many times this session has been (re)enqueued — bounds resume retries.
    attempts = models.IntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "processing_sessions"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["session_id"]),
            models.Index(fields=["status"]),
            models.Index(fields=["created_at"]),
            models.Index(fields=["status", "heartbeat_at"]),
        ]

    def __str__(self):
        return f"Session {self.session_id} - {self.status}"

    def mark_queued(self):
        """Mark session as queued for a background worker."""
        self.status = ProcessingStatus.QUEUED.value
        self.save(update_fields=["status", "updated_at"])

    def mark_processing(self):
        """Mark session as processing and stamp an initial heartbeat."""
        self.status = ProcessingStatus.PROCESSING.value
        self.heartbeat_at = timezone.now()
        self.error_message = ""
        self.stop_reason = ""
        self.save(
            update_fields=["status", "heartbeat_at", "error_message", "stop_reason", "updated_at"]
        )

    def touch_heartbeat(self, processed_frames: int | None = None):
        """Bump liveness (and optionally progress) cheaply.

        Called by the worker on a throttled cadence; keeps the write set tiny
        so it is safe to call often without hammering the DB.
        """
        self.heartbeat_at = timezone.now()
        fields = ["heartbeat_at", "updated_at"]
        if processed_frames is not None:
            self.processed_frames = processed_frames
            fields.append("processed_frames")
        self.save(update_fields=fields)

    def mark_completed(self, stop_reason: str = "eof"):
        """Mark session as completed."""
        self.status = ProcessingStatus.COMPLETED.value
        self.stop_reason = stop_reason
        self.completed_at = timezone.now()
        self.save(
            update_fields=["status", "stop_reason", "completed_at", "updated_at"]
        )

    def mark_failed(self, error_message: str = "", stop_reason: str = "exception"):
        """Mark session as failed (unrecoverable) with a reason."""
        self.status = ProcessingStatus.FAILED.value
        self.stop_reason = stop_reason
        if error_message:
            self.error_message = error_message[:2000]
        self.save(
            update_fields=["status", "stop_reason", "error_message", "updated_at"]
        )

    def mark_interrupted(self, error_message: str = "", stop_reason: str = "interrupted"):
        """Mark session as interrupted (stopped early but resumable)."""
        self.status = ProcessingStatus.INTERRUPTED.value
        self.stop_reason = stop_reason
        if error_message:
            self.error_message = error_message[:2000]
        self.save(
            update_fields=["status", "stop_reason", "error_message", "updated_at"]
        )
