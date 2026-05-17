from django.db import models


class CustomOcrTemplate(models.Model):
    """User-defined OCR template with labelled regions used as spatial hints.

    At runtime we OCR the image ONCE (whole frame, or whatever ROI the caller
    passes) and feed Gemini both the OCR lines AND the saved region template
    (label + description + %-coords + expected_fields). Gemini then matches
    each OCR line to a region by position and emits a JSON object matching
    the dynamic schema built from `expected_fields`.

    `regions` shape (list of):
      {
        "label": "current_hole_hud",
        "description": "shows hole number, par, player, score",
        "coords": [x1, y1, x2, y2],     # normalized 0..1, top-left origin
        "expected_fields": [
          {"name": "hole", "type": "integer", "description": "..."}
        ]
      }
    """

    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=255, unique=True, db_index=True)
    sport = models.CharField(max_length=64, blank=True, db_index=True)
    description = models.TextField(blank=True)
    regions = models.JSONField(default=list)
    system_prompt = models.TextField(blank=True)
    multimodal = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "custom_ocr_templates"
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.name} ({self.slug})"
