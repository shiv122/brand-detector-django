from django.db import models


class SportPrompt(models.Model):
    """User-editable OCR prompt attached to a sport.

    The prompt is sent verbatim to the serverless OCR endpoint (RunPod)
    alongside the image. The endpoint returns a JSON object whose shape is
    governed entirely by the prompt — no schema is enforced on our side.
    """

    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=255, unique=True, db_index=True)
    sport = models.CharField(max_length=64, blank=True, db_index=True)
    description = models.TextField(blank=True)
    prompt = models.TextField()
    # Canonical brand names in display form (e.g. "Coca-Cola"). Substituted
    # into the prompt wherever the `{{BRAND_LIST}}` placeholder appears, and
    # used as the closed vocabulary the formatter corrects noisy OCR towards.
    allowed_brands = models.JSONField(default=list, blank=True)
    # Canonical taglines / slogans in display form (e.g. "Just Do It"). Same
    # role as allowed_brands but for multi-word phrases.
    taglines = models.JSONField(default=list, blank=True)
    # When on, the brand/tagline correction guide (+ vocabulary) is appended to
    # the rendered prompt so the formatter snaps noisy OCR onto the canonical
    # entries. When off, brands/taglines are stored but never sent to the model.
    correction_enabled = models.BooleanField(default=False)
    reference_image_path = models.CharField(max_length=500, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "sport_prompts"
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.name} ({self.slug})"
