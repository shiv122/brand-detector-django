from django.db import models
from apps.core.models.detection import Detection


class Classification(models.Model):
    """Model for storing classification results for detections"""

    detection = models.ForeignKey(
        Detection,
        on_delete=models.CASCADE,
        related_name="classifications",
        db_index=True,
    )
    class_id = models.IntegerField()
    class_name = models.CharField(max_length=100)
    confidence = models.FloatField()
    rank = models.IntegerField(default=1)  # 1 = top-1, 2 = top-2, etc.
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "classifications"
        ordering = ["detection", "rank"]
        indexes = [
            models.Index(fields=["detection", "rank"]),
        ]

    def __str__(self):
        return f"{self.class_name} ({self.confidence:.2f}) - Rank {self.rank}"
