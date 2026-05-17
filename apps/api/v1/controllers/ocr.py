"""
OCR Controller — standalone /ocr/* endpoints.
"""

from django.urls import re_path
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from apps.api.v1.requests.ocr_requests import (
    CustomTemplateUpsertRequest,
    OcrRunRequest,
)
from apps.api.v1.shared_services import _ocr_service
from apps.core.models import CustomOcrTemplate


def optional_slash_path(route, view, name=None):
    return [re_path(rf"^{route}/?$", view, name=name)]


@extend_schema(
    summary="Run OCR on an image",
    description=(
        "Run PaddleOCR (English) on a single uploaded image. Optionally restrict "
        "to a normalized ROI [x1, y1, x2, y2] in 0..1. Pass a `template_key` "
        "(see /ocr/templates) to post-process the OCR text via regex or Gemini. "
        "Pass `custom_prompt` to override or augment the template's prompt."
    ),
    request={
        "multipart/form-data": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "format": "binary",
                    "description": "Image file to OCR",
                },
                "roi": {
                    "type": "string",
                    "description": (
                        "Optional JSON array of 4 normalized floats "
                        "[x1, y1, x2, y2] in 0..1"
                    ),
                },
                "include_annotated": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include base64-encoded annotated image in response",
                },
                "template_key": {
                    "type": "string",
                    "default": "raw",
                    "description": "Key from /ocr/templates, e.g. 'golf_hole'",
                },
                "custom_prompt": {
                    "type": "string",
                    "description": (
                        "Optional override or augmentation of the template's prompt. "
                        "Only allowed on templates with supports_custom_prompt=true."
                    ),
                },
            },
            "required": ["file"],
        }
    },
    responses={
        200: {
            "example": {
                "raw_text": "Hole 7",
                "raw_lines": [
                    {"text": "Hole 7", "confidence": 0.98, "bbox": [10, 20, 80, 60]}
                ],
                "confidence_avg": 0.98,
                "image_size": [1280, 720],
            }
        }
    },
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def run_ocr(request):
    """Standalone OCR endpoint."""
    validation = OcrRunRequest(request.data, request.FILES)
    if validation.fails():
        return validation.errors_response()

    data = validation.validated()
    file = request.FILES["file"]
    roi = data.get("roi")
    include_annotated = bool(data.get("include_annotated", False))
    template_key = data.get("template_key", "raw") or "raw"
    custom_prompt = data.get("custom_prompt", "") or ""

    if not _ocr_service.is_available():
        return Response(
            {"error": "OCR engine is not available"},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        payload = _ocr_service.process_bytes(
            file.read(),
            roi_normalized=roi,
            include_annotated=include_annotated,
            template_key=template_key,
            custom_prompt=custom_prompt,
        )
    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
    except Exception as e:
        return Response(
            {"error": f"OCR failed: {e}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return Response(payload)


@extend_schema(
    summary="List OCR formatting templates",
    description=(
        "Return the registry of pre-defined OCR formatting templates. Each "
        "template declares its mode (none/regex/llm), whether it supports a "
        "custom prompt, whether it's multimodal, and the JSON schema of its "
        "output."
    ),
    responses={
        200: {
            "example": {
                "templates": [
                    {
                        "key": "golf_hole",
                        "label": "Golf hole number",
                        "description": "Read the current hole number...",
                        "mode": "llm",
                        "multimodal": True,
                        "supports_custom_prompt": True,
                        "schema": {"type": "object", "properties": {}},
                    }
                ]
            }
        }
    },
)
@api_view(["GET"])
def list_templates(request):
    """Return the registry of OCR formatting templates."""
    return Response({"templates": _ocr_service.list_templates()})


def _serialize_custom_template(t: CustomOcrTemplate) -> dict:
    return {
        "id": t.id,
        "name": t.name,
        "slug": t.slug,
        "sport": t.sport,
        "description": t.description,
        "regions": t.regions or [],
        "system_prompt": t.system_prompt or "",
        "multimodal": bool(t.multimodal),
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "updated_at": t.updated_at.isoformat() if t.updated_at else None,
    }


@extend_schema(
    summary="List / create custom OCR templates",
    description=(
        "Custom OCR templates carry labelled regions used as spatial hints. At "
        "runtime we OCR the image once and pass Gemini both the OCR lines and "
        "the template's regions; Gemini maps lines to labels and emits a JSON "
        "object matching a schema built from `expected_fields`."
    ),
    responses={200: {"example": {"custom_templates": []}}},
)
@api_view(["GET", "POST"])
@parser_classes([JSONParser])
def custom_templates(request):
    if request.method == "GET":
        items = [_serialize_custom_template(t) for t in CustomOcrTemplate.objects.all()]
        return Response({"custom_templates": items})

    validation = CustomTemplateUpsertRequest(request.data)
    if validation.fails():
        return validation.errors_response()
    data = validation.validated()

    if CustomOcrTemplate.objects.filter(slug=data["slug"]).exists():
        return Response(
            {"errors": {"slug": [f"slug '{data['slug']}' already exists"]}},
            status=status.HTTP_409_CONFLICT,
        )

    template = CustomOcrTemplate.objects.create(
        name=data["name"],
        slug=data["slug"],
        sport=data.get("sport", ""),
        description=data.get("description", ""),
        regions=data["regions"],
        system_prompt=data.get("system_prompt", ""),
        multimodal=bool(data.get("multimodal", False)),
    )
    return Response(
        _serialize_custom_template(template), status=status.HTTP_201_CREATED
    )


@extend_schema(
    summary="Retrieve / update / delete a custom OCR template",
    description="GET / PUT / DELETE by slug.",
)
@api_view(["GET", "PUT", "DELETE"])
@parser_classes([JSONParser])
def custom_template_detail(request, slug: str):
    try:
        template = CustomOcrTemplate.objects.get(slug=slug)
    except CustomOcrTemplate.DoesNotExist:
        return Response({"error": "not found"}, status=status.HTTP_404_NOT_FOUND)

    if request.method == "GET":
        return Response(_serialize_custom_template(template))

    if request.method == "DELETE":
        template.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    payload = dict(request.data or {})
    payload.setdefault("slug", slug)
    validation = CustomTemplateUpsertRequest(payload)
    validation.require_slug = False
    if validation.fails():
        return validation.errors_response()
    data = validation.validated()

    new_slug = data["slug"]
    if new_slug != template.slug:
        if CustomOcrTemplate.objects.filter(slug=new_slug).exists():
            return Response(
                {"errors": {"slug": [f"slug '{new_slug}' already exists"]}},
                status=status.HTTP_409_CONFLICT,
            )
        template.slug = new_slug

    template.name = data["name"]
    template.sport = data.get("sport", "")
    template.description = data.get("description", "")
    template.regions = data["regions"]
    template.system_prompt = data.get("system_prompt", "")
    template.multimodal = bool(data.get("multimodal", False))
    template.save()
    return Response(_serialize_custom_template(template))


urlpatterns = [
    *optional_slash_path("run", run_ocr, name="ocr-run"),
    *optional_slash_path("templates", list_templates, name="ocr-templates"),
    *optional_slash_path(
        "custom-templates", custom_templates, name="ocr-custom-templates"
    ),
    *optional_slash_path(
        r"custom-templates/(?P<slug>[a-z0-9][a-z0-9-]*)",
        custom_template_detail,
        name="ocr-custom-template-detail",
    ),
]
