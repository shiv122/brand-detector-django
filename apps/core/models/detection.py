from django.db import models
from apps.core.models.session import ProcessingSession
from apps.core.models.frame import Frame


class Detection(models.Model):
    """Model for storing detection results"""
    
    frame = models.ForeignKey(
        Frame,
        on_delete=models.CASCADE,
        related_name="detections",
        db_index=True
    )
    session = models.ForeignKey(
        ProcessingSession,
        on_delete=models.CASCADE,
        related_name="detections",
        db_index=True
    )
    class_id = models.IntegerField()
    class_name = models.CharField(max_length=100, db_index=True)
    confidence = models.FloatField(db_index=True)
    bbox_x1 = models.FloatField()
    bbox_y1 = models.FloatField()
    bbox_x2 = models.FloatField()
    bbox_y2 = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    class Meta:
        db_table = "detections"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["session", "class_name"]),
            models.Index(fields=["frame"]),
            models.Index(fields=["class_name", "confidence"]),
        ]
    
    def __str__(self):
        return f"{self.class_name} ({self.confidence:.2f}) - Frame {self.frame.frame_number}"
    
    @property
    def bbox(self):
        """Get bounding box as list"""
        return [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2]
    
    def set_bbox(self, bbox_list):
        """Set bounding box from list"""
        self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2 = bbox_list

