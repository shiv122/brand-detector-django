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
from apps.api.v1.shared_services import (
    _config,
    _locate_service,
    _ocr_service,
    _spaces_service,
)
from apps.core.models import Frame, SportPrompt
from apps.services.ocr.ocr_queue import enqueue_ocr_job, fetch_jobs
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
# Prompt resolution — shared by /ocr/run and /ocr/frame
# ---------------------------------------------------------------------------


def _resolve_prompt(inline_prompt: str, prompt_slug: str, sport: str):
    """Resolve (prompt, prompt_meta) from inline > slug > sport.

    Returns (prompt, prompt_meta, None) on success, or (None, None, Response)
    when the input is missing / no matching SportPrompt exists.
    """
    inline_prompt = (inline_prompt or "").strip()
    prompt_slug = (prompt_slug or "").strip()
    sport = (sport or "").strip()

    if inline_prompt:
        return render_prompt(inline_prompt, []), {"source": "inline"}, None

    if prompt_slug:
        sp = SportPrompt.objects.filter(slug=prompt_slug).first()
        if sp is None:
            return None, None, Response(
                {"error": f"no SportPrompt with slug '{prompt_slug}'"},
                status=status.HTTP_404_NOT_FOUND,
            )
        source = "stored"
    elif sport:
        sport_lower = sport.lower()
        sp = (
            SportPrompt.objects.filter(slug=sport_lower).first()
            or SportPrompt.objects.filter(sport__iexact=sport_lower)
            .order_by("-updated_at")
            .first()
        )
        if sp is None:
            return None, None, Response(
                {"error": f"no SportPrompt for sport '{sport}'"},
                status=status.HTTP_404_NOT_FOUND,
            )
        source = "sport"
    else:
        return None, None, Response(
            {"error": "one of prompt / prompt_slug / sport is required"},
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    allowed_brands = list(sp.allowed_brands or [])
    meta = {"source": source, "slug": sp.slug, "name": sp.name, "sport": sp.sport}
    if allowed_brands:
        meta["allowed_brands"] = allowed_brands
    return render_prompt(sp.prompt, allowed_brands), meta, None


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
    engine = (request.data.get("engine") or "glm").strip().lower()

    # Resolve engine + its inputs before doing any upload.
    if engine == "locate":
        if not _locate_service.is_available():
            return Response(
                {"error": "LocateAnything endpoint is not configured — set LOCATE_HOST."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        prompt, prompt_meta = None, {"engine": "locate"}
    else:
        prompt, prompt_meta, err = _resolve_prompt(
            data.get("prompt"), data.get("prompt_slug"), data.get("sport")
        )
        if err is not None:
            return err
        if not _ocr_service.is_available():
            return Response(
                {"error": "GLM OCR endpoint is not configured — set GLM_OCR_HOST."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

    if not _spaces_service.is_configured():
        return Response(
            {
                "error": "Spaces is not configured — needed to host the upload "
                "for the OCR service to fetch. Set the DO_* env vars."
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # The OCR engines fetch by URL, so push the upload to Spaces first, then hand
    # over its public URL.
    import uuid

    ext = (Path(file.name).suffix or ".jpg").lower()
    key = f"ocr_uploads/{uuid.uuid4().hex}{ext}"
    try:
        image_url = _spaces_service.upload_bytes(
            key, file.read(), content_type=file.content_type or "image/jpeg"
        )
    except Exception as exc:  # noqa: BLE001
        return Response(
            {"error": f"failed to upload image to Spaces: {exc}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )

    if engine == "locate":
        result = _locate_service.run(
            image_url,
            task=(request.data.get("task") or ""),
            query=(request.data.get("query") or ""),
            reader=(request.data.get("reader") or ""),
        )
    else:
        result = _ocr_service.run(image_url, prompt)
    result["prompt_meta"] = prompt_meta
    result["image_url"] = image_url
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
        "GET with `ids` as a repeated query param OR comma-separated list, "
        "OR POST with `{ ids: [...] }` body (use POST when the ID list is "
        "large enough to overflow request-line limits ~4KB). "
        "Returns `{ jobs: { <id>: { status, result?, error? } } }`. "
        "Statuses: queued, started, finished, failed, deferred, canceled, "
        "unknown (TTL expired), unavailable (Redis down)."
    ),
)
@api_view(["GET", "POST"])
def ocr_jobs_status(request):
    flat: list[str] = []
    if request.method == "POST":
        body_ids = request.data.get("ids") if hasattr(request, "data") else None
        if isinstance(body_ids, list):
            flat = [str(i).strip() for i in body_ids if str(i).strip()]
        elif isinstance(body_ids, str):
            flat = [p.strip() for p in body_ids.split(",") if p.strip()]
    else:
        # Support both repeated ?ids=a&ids=b and ?ids=a,b shapes — the frontend
        # default for arrays in URLSearchParams varies, this side accepts either.
        for item in request.GET.getlist("ids") or []:
            flat.extend(part.strip() for part in item.split(",") if part.strip())

    if not flat:
        return Response({"jobs": {}})

    return Response({"jobs": fetch_jobs(flat)})


# ---------------------------------------------------------------------------
# /ocr/frame — enqueue OCR for an already-detected frame (on-demand, async)
# ---------------------------------------------------------------------------


@extend_schema(
    summary="Enqueue OCR for a stored detection frame",
    description=(
        "Runs OCR on a frame that detection already produced, as a background "
        "job that calls the external GLM OCR API. The job carries the frame's "
        "image PATH (never image bytes); the result is written back to "
        "Frame.ocr_summary and is also pollable via /ocr/jobs. "
        "Prompt resolution: `prompt` > `prompt_slug` > `sport`."
    ),
)
@api_view(["POST"])
@parser_classes([JSONParser])
def enqueue_frame_ocr(request):
    data = request.data or {}
    raw_id = data.get("frame_id")
    try:
        frame_id = int(raw_id)
    except (TypeError, ValueError):
        return Response(
            {"error": "frame_id (integer) is required"},
            status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    frame = Frame.objects.filter(id=frame_id).first()
    if frame is None:
        return Response({"error": "frame not found"}, status=status.HTTP_404_NOT_FOUND)
    # GLM OCR fetches by URL — the frame must have a fetchable (Spaces) URL.
    image_url = frame.frame_url or ""
    if not image_url.startswith(("http://", "https://")):
        return Response(
            {
                "error": "frame has no public URL (was it stored locally? "
                "configure Spaces so frames are uploaded)."
            },
            status=status.HTTP_409_CONFLICT,
        )

    engine = (data.get("engine") or "glm").strip().lower()
    if engine == "locate":
        if not _locate_service.is_available():
            return Response(
                {"error": "LocateAnything endpoint is not configured — set LOCATE_HOST."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        task = (data.get("task") or "").strip()
        query = (data.get("query") or "").strip()
        reader = (data.get("reader") or "").strip()
        meta = {"engine": "locate", "source": "locate"}
        if task:
            meta["task"] = task
        if query:
            meta["query"] = query
        if reader:
            meta["reader"] = reader
        job = enqueue_ocr_job(
            image_url=image_url,
            prompt="",
            prompt_meta=meta,
            frame_id=frame.id,
            engine="locate",
            task=task,
            query=query,
            reader=reader,
        )
        return Response({"ocr_job": job, "frame_id": frame.id, "prompt_meta": meta})

    prompt, prompt_meta, err = _resolve_prompt(
        data.get("prompt"), data.get("prompt_slug"), data.get("sport")
    )
    if err is not None:
        return err

    if not _ocr_service.is_available():
        return Response(
            {"error": "GLM OCR endpoint is not configured — set GLM_OCR_HOST."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    job = enqueue_ocr_job(
        image_url=image_url,
        prompt=prompt,
        prompt_meta=prompt_meta,
        frame_id=frame.id,
    )
    return Response({"ocr_job": job, "frame_id": frame.id, "prompt_meta": prompt_meta})


# ---------------------------------------------------------------------------

urlpatterns = [
    *optional_slash_path("run", run_ocr, name="ocr-run"),
    *optional_slash_path("frame", enqueue_frame_ocr, name="ocr-frame"),
    *optional_slash_path("jobs", ocr_jobs_status, name="ocr-jobs"),
    *optional_slash_path("sport-prompts", sport_prompts, name="ocr-sport-prompts"),
    *optional_slash_path(
        r"sport-prompts/(?P<slug>[a-z0-9][a-z0-9-]*)",
        sport_prompt_detail,
        name="ocr-sport-prompt-detail",
    ),
]
