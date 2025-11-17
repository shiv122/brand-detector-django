from django.db import models
from django.utils import timezone
from apps.core.enums import ProcessingStatus


class ProcessingSession(models.Model):
    """Model for tracking video/image processing sessions"""

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
    total_frames = models.IntegerField(default=0)
    processed_frames = models.IntegerField(default=0)
    settings = models.JSONField(default=dict, blank=True)  # Store processing settings in JSON format
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
        ]

    def __str__(self):
        return f"Session {self.session_id} - {self.status}"

    def mark_completed(self):
        """Mark session as completed"""
        self.status = ProcessingStatus.COMPLETED.value
        self.completed_at = timezone.now()
        self.save(update_fields=["status", "completed_at", "updated_at"])

    def mark_failed(self):
        """Mark session as failed"""
        self.status = ProcessingStatus.FAILED.value
        self.save(update_fields=["status", "updated_at"])

    def mark_processing(self):
        """Mark session as processing"""
        self.status = ProcessingStatus.PROCESSING.value
        self.save(update_fields=["status", "updated_at"])
