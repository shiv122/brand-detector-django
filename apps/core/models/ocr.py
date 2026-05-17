from django.db import models

from apps.core.models.detection import Detection
from apps.core.models.frame import Frame
from apps.core.models.session import ProcessingSession


class Ocr(models.Model):
    """OCR result attached to a Frame and optionally a specific Detection."""

    session = models.ForeignKey(
        ProcessingSession,
        on_delete=models.CASCADE,
        related_name="ocr_results",
        db_index=True,
    )
    frame = models.ForeignKey(
        Frame,
        on_delete=models.CASCADE,
        related_name="ocr_results",
        db_index=True,
    )
    detection = models.ForeignKey(
        Detection,
        on_delete=models.CASCADE,
        related_name="ocr",
        null=True,
        blank=True,
        db_index=True,
    )
    template_key = models.CharField(max_length=64, default="raw")
    custom_prompt = models.TextField(blank=True)
    raw_text = models.TextField(blank=True)
    raw_lines = models.JSONField(default=list)
    formatted = models.JSONField(null=True, blank=True)
    roi = models.JSONField(null=True, blank=True)
    confidence_avg = models.FloatField(default=0.0)
    format_error = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "ocr_results"
        ordering = ["frame", "created_at"]
        indexes = [
            models.Index(fields=["session", "frame"]),
            models.Index(fields=["template_key"]),
        ]

    def __str__(self):
        target = f"detection {self.detection_id}" if self.detection_id else "frame"
        return f"OCR({self.template_key}) on {target} — frame {self.frame_id}"
