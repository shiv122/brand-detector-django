"""
OCR Controller — /ocr/run + /ocr/sport-prompts (CRUD).
"""

import base64
import binascii
import re
from pathlib import Path
from typing import Optional, Tuple

from django.conf import settings as django_settings
from django.urls import re_path
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from apps.api.v1.requests.ocr_requests import (
    OcrRunRequest,
    SportPromptUpsertRequest,
)
from apps.api.v1.shared_services import _config, _ocr_service
from apps.core.models import SportPrompt
from apps.services.ocr.ocr_queue import fetch_jobs
from apps.utils.prompt_render import render_prompt


def optional_slash_path(route, view, name=None):
    return [re_path(rf"^{route}/?$", view, name=name)]


# ---------------------------------------------------------------------------
# Reference-image helpers (stored on disk under <static>/sport_prompts/)
# ---------------------------------------------------------------------------

_DATA_URL_RE = re.compile(r"^data:image/(?P<mime>jpeg|jpg|png|webp);base64,(?P<data>.+)$")


def _decode_reference_image(raw: Optional[str]) -> Optional[Tuple[bytes, str]]:
    if not raw:
        return None
    m = _DATA_URL_RE.match(raw.strip())
    if m:
        ext = m.group("mime").lower().replace("jpeg", "jpg")
        try:
            return base64.b64decode(m.group("data"), validate=True), ext
        except binascii.Error:
            return None
    try:
        return base64.b64decode(raw, validate=True), "jpg"
    except binascii.Error:
        return None


def _reference_image_dir() -> Path:
    p = Path(_config.static_dir) / "sport_prompts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_reference_image(slug: str, raw: Optional[str]) -> Optional[str]:
    decoded = _decode_reference_image(raw)
    if decoded is None:
        return None
    image_bytes, ext = decoded
    out_dir = _reference_image_dir()
    for stale in out_dir.glob(f"{slug}.*"):
        try:
            stale.unlink()
        except OSError:
            pass
    out_path = out_dir / f"{slug}.{ext}"
    out_path.write_bytes(image_bytes)
    return f"sport_prompts/{out_path.name}"


def _reference_image_url(rel_path: str) -> str:
    if not rel_path:
        return ""
    return f"{django_settings.STATIC_URL.rstrip('/')}/{rel_path.lstrip('/')}"


def _delete_reference_image(rel_path: str) -> None:
    if not rel_path:
        return
    p = Path(_config.static_dir) / rel_path.lstrip("/")
    try:
        if p.exists():
            p.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# /ocr/run
# ---------------------------------------------------------------------------


@extend_schema(
    summary="Run OCR on an image via the serverless endpoint",
    description=(
        "Sends the uploaded image + a prompt to the configured RunPod OCR "
        "endpoint. Prompt resolution priority: `prompt` (inline) > "
        "`prompt_slug` > `sport`."
    ),
    request={
        "multipart/form-data": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "format": "binary"},
                "sport": {
                    "type": "string",
                    "description": "Sport name; resolves to the saved prompt whose "
                    "`sport` matches. Prefers the prompt with slug == sport.",
                },
                "prompt_slug": {
                    "type": "string",
                    "description": "Slug of a saved SportPrompt.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Inline prompt; overrides prompt_slug / sport.",
                },
            },
            "required": ["file"],
        }
    },
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def run_ocr(request):
    validation = OcrRunRequest(request.data, request.FILES)
    if validation.fails():
        return validation.errors_response()
    data = validation.validated()

    file = request.FILES["file"]
    inline_prompt = (data.get("prompt") or "").strip()
    prompt_slug = (data.get("prompt_slug") or "").strip()
    sport = (data.get("sport") or "").strip()

    prompt: str = ""
    prompt_meta = {}
    allowed_brands: list[str] = []
    if inline_prompt:
        prompt = inline_prompt
        prompt_meta = {"source": "inline"}
    elif prompt_slug:
        sp = SportPrompt.objects.filter(slug=prompt_slug).first()
        if sp is None:
            return Response(
                {"error": f"no SportPrompt with slug '{prompt_slug}'"},
                status=status.HTTP_404_NOT_FOUND,
            )
        prompt = sp.prompt
        allowed_brands = list(sp.allowed_brands or [])
        prompt_meta = {
            "source": "stored",
            "slug": sp.slug,
            "name": sp.name,
            "sport": sp.sport,
        }
    elif sport:
        # Prefer a prompt whose slug equals the sport name (seeded canonical
        # row); otherwise fall back to any prompt tagged with this sport.
        sport_lower = sport.lower()
        sp = (
            SportPrompt.objects.filter(slug=sport_lower).first()
            or SportPrompt.objects.filter(sport__iexact=sport_lower)
            .order_by("-updated_at")
            .first()
        )
        if sp is None:
            return Response(
                {"error": f"no SportPrompt for sport '{sport}'"},
                status=status.HTTP_404_NOT_FOUND,
            )
        prompt = sp.prompt
        allowed_brands = list(sp.allowed_brands or [])
        prompt_meta = {
            "source": "sport",
            "slug": sp.slug,
            "name": sp.name,
            "sport": sp.sport,
        }
    else:
        return Response(
            {"error": "one of prompt / prompt_slug / sport is required"},
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    prompt = render_prompt(prompt, allowed_brands)
    if allowed_brands:
        prompt_meta["allowed_brands"] = allowed_brands

    if not _ocr_service.is_available():
        missing = []
        if not _config.local_ocr_ollama_host:
            missing.append("LOCAL_OCR_OLLAMA_HOST")
        if not _config.local_ocr_ollama_model:
            missing.append("LOCAL_OCR_OLLAMA_MODEL")
        if not _config.deepseek_text_api_key:
            missing.append("DEEPSEEK_TEXT_API_KEY")
        err = (
            f"OCR endpoint is not configured — set "
            f"{', '.join(missing) or '<unknown>'} in .env, then restart."
        )
        return Response(
            {"error": err},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    result = _ocr_service.run(file.read(), prompt)
    result["prompt_meta"] = prompt_meta
    return Response(result)


# ---------------------------------------------------------------------------
# /ocr/sport-prompts (CRUD)
# ---------------------------------------------------------------------------


def _serialize_prompt(p: SportPrompt) -> dict:
    return {
        "id": p.id,
        "name": p.name,
        "slug": p.slug,
        "sport": p.sport,
        "description": p.description or "",
        "prompt": p.prompt,
        "allowed_brands": list(p.allowed_brands or []),
        "reference_image_path": p.reference_image_path or "",
        "reference_image_url": _reference_image_url(p.reference_image_path or ""),
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }


@extend_schema(summary="List / create sport prompts")
@api_view(["GET", "POST"])
@parser_classes([JSONParser])
def sport_prompts(request):
    if request.method == "GET":
        items = [_serialize_prompt(p) for p in SportPrompt.objects.all()]
        return Response({"sport_prompts": items})

    validation = SportPromptUpsertRequest(request.data)
    if validation.fails():
        return validation.errors_response()
    data = validation.validated()

    if SportPrompt.objects.filter(slug=data["slug"]).exists():
        return Response(
            {"errors": {"slug": [f"slug '{data['slug']}' already exists"]}},
            status=status.HTTP_409_CONFLICT,
        )

    ref_rel = _save_reference_image(data["slug"], data.get("reference_image"))
    p = SportPrompt.objects.create(
        name=data["name"],
        slug=data["slug"],
        sport=data.get("sport", ""),
        description=data.get("description", ""),
        prompt=data["prompt"],
        allowed_brands=data.get("allowed_brands", []),
        reference_image_path=ref_rel or "",
    )
    return Response(_serialize_prompt(p), status=status.HTTP_201_CREATED)


@extend_schema(summary="Retrieve / update / delete a sport prompt by slug")
@api_view(["GET", "PUT", "DELETE"])
@parser_classes([JSONParser])
def sport_prompt_detail(request, slug: str):
    try:
        p = SportPrompt.objects.get(slug=slug)
    except SportPrompt.DoesNotExist:
        return Response({"error": "not found"}, status=status.HTTP_404_NOT_FOUND)

    if request.method == "GET":
        return Response(_serialize_prompt(p))

    if request.method == "DELETE":
        _delete_reference_image(p.reference_image_path or "")
        p.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    payload = dict(request.data or {})
    payload.setdefault("slug", slug)
    validation = SportPromptUpsertRequest(payload)
    validation.require_slug = False
    if validation.fails():
        return validation.errors_response()
    data = validation.validated()

    new_slug = data["slug"]
    if new_slug != p.slug:
        if SportPrompt.objects.filter(slug=new_slug).exists():
            return Response(
                {"errors": {"slug": [f"slug '{new_slug}' already exists"]}},
                status=status.HTTP_409_CONFLICT,
            )
        p.slug = new_slug

    p.name = data["name"]
    p.sport = data.get("sport", "")
    p.description = data.get("description", "")
    p.prompt = data["prompt"]
    p.allowed_brands = data.get("allowed_brands", [])

    if "reference_image" in (request.data or {}):
        raw = data.get("reference_image")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            _delete_reference_image(p.reference_image_path or "")
            p.reference_image_path = ""
        else:
            ref_rel = _save_reference_image(p.slug, raw)
            if ref_rel:
                p.reference_image_path = ref_rel

    p.save()
    return Response(_serialize_prompt(p))


# ---------------------------------------------------------------------------
# /ocr/jobs — status + results for queued OCR runs
# ---------------------------------------------------------------------------


@extend_schema(
    summary="Poll one or more OCR job statuses",
    description=(
        "Pass `ids` as a repeated query param OR a comma-separated list. "
        "Returns `{ jobs: { <id>: { status, result?, error? } } }`. "
        "Statuses: queued, started, finished, failed, deferred, canceled, "
        "unknown (TTL expired), unavailable (Redis down)."
    ),
)
@api_view(["GET"])
def ocr_jobs_status(request):
    raw = request.GET.getlist("ids") or []
    # Support both repeated ?ids=a&ids=b and ?ids=a,b shapes — the frontend
    # default for arrays in URLSearchParams varies, this side accepts either.
    flat: list[str] = []
    for item in raw:
        flat.extend(part.strip() for part in item.split(",") if part.strip())

    if not flat:
        return Response({"jobs": {}})

    return Response({"jobs": fetch_jobs(flat)})


# ---------------------------------------------------------------------------

urlpatterns = [
    *optional_slash_path("run", run_ocr, name="ocr-run"),
    *optional_slash_path("jobs", ocr_jobs_status, name="ocr-jobs"),
    *optional_slash_path("sport-prompts", sport_prompts, name="ocr-sport-prompts"),
    *optional_slash_path(
        r"sport-prompts/(?P<slug>[a-z0-9][a-z0-9-]*)",
        sport_prompt_detail,
        name="ocr-sport-prompt-detail",
    ),
]
