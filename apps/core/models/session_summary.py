from django.db import models
from apps.core.models.session import ProcessingSession


class SessionSummary(models.Model):
    """Denormalized summary data for quick access"""

    session = models.OneToOneField(
        ProcessingSession,
        on_delete=models.CASCADE,
        related_name="summary",
        db_index=True,
    )
    total_detections = models.IntegerField(default=0)
    unique_logos = models.JSONField(default=list)  # List of unique logo names
    logo_counts = models.JSONField(default=dict)  # {logo_name: count}
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "session_summaries"

    def __str__(self):
        return f"Summary for Session {self.session.session_id}"
