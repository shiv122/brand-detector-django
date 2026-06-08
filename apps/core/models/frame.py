from django.db import models
from apps.core.models.session import ProcessingSession


class Frame(models.Model):
    """Model for storing processed video frames"""

    session = models.ForeignKey(
        ProcessingSession,
        on_delete=models.CASCADE,
        related_name="frames",
        db_index=True,
    )
    frame_number = models.IntegerField(db_index=True)
    # Local disk path (only when Spaces is unconfigured; "" once uploaded).
    frame_path = models.CharField(max_length=500, blank=True, default="")
    # Public URL the dashboard/OCR use — a DigitalOcean Spaces URL when
    # configured, else a local /static/ path.
    frame_url = models.CharField(max_length=500)
    # Object key in the Spaces bucket (blank when stored locally).
    s3_key = models.CharField(max_length=500, blank=True, default="")
    timestamp = models.FloatField(default=0.0)
    total_detections = models.IntegerField(default=0)
    ocr_summary = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "frames"
        ordering = ["session", "frame_number"]
        unique_together = [["session", "frame_number"]]
        indexes = [
            models.Index(fields=["session", "frame_number"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self):
        return f"Frame {self.frame_number} - Session {self.session.session_id}"
