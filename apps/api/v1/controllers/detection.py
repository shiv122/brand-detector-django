"""
Detection Controller - Laravel-style (thin controllers)
"""

from django.urls import path, re_path
from django.http import StreamingHttpResponse
from django.views.decorators.http import require_GET
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes
from apps.api.v1.requests.detection_requests import (
    UpdateConfigRequest,
    SwitchWeightRequest,
    DetectImagesRequest,
    DetectVideoRequest,
)

# Import shared services (initialized once to prevent duplicate logs)
from apps.api.v1.shared_services import (
    _config,
    _model_service,
    _image_service,
    _classification_service,
    _counting_service,
    _detection_service,
    _spaces_service,
)


def optional_slash_path(route, view, name=None):
    """Helper to create URL patterns that work with or without trailing slashes"""
    return [
        re_path(rf"^{route}/?$", view, name=name),
    ]


urlpatterns = []


@extend_schema(
    summary="Root detection endpoint",
    description="Returns API status",
    responses={200: {"example": {"message": "Detection API", "status": "running"}}},
)
@api_view(["GET"])
def index(request):
    """Root detection endpoint"""
    return Response({"message": "Detection API", "status": "running"})


@extend_schema(
    summary="Health check",
    description="Check if detection service is healthy and model is loaded",
    responses={200: {"example": {"status": "healthy", "model_loaded": True}}},
)
@api_view(["GET"])
def health(request):
    """Health check for detection service"""
    return Response(
        {
            "status": "healthy",
            "model_loaded": _detection_service.is_model_loaded(),
        }
    )


@extend_schema(
    summary="Get device information",
    description="Get information about the current device (GPU/CPU)",
    responses={200: {"example": {"device": "mps", "device_name": "Apple M1"}}},
)
@api_view(["GET"])
def device(request):
    """Get device information"""
    return Response(_model_service.get_device_info())


@extend_schema(
    summary="Get or update configuration",
    description="GET: Get current detection configuration. POST: Update configuration (frames_per_second, confidence_threshold)",
    request=UpdateConfigRequest,
    responses={
        200: {
            "description": "GET returns config, POST returns success message",
            "examples": {
                "get": {"frames_per_second": 2, "confidence_threshold": 0.5},
                "post": {"message": "Configuration updated successfully"},
            },
        }
    },
)
@api_view(["GET", "POST"])
@parser_classes([JSONParser])
def config(request):
    """Get or update configuration - same path for GET and POST to match old backend"""
    if request.method == "GET":
        return Response(_config.to_dict())
    else:  # POST
        validation = UpdateConfigRequest(request.data)
        if validation.fails():
            return validation.errors_response()
        return _detection_service.update_config(validation.validated())


@extend_schema(
    summary="Get available weights",
    description="Get list of available model weights and default weight. Send weight_name with each detection request to specify which model to use.",
    responses={
        200: {
            "example": {
                "available_weights": [{"name": "best.pt", "size": 40800000}],
                "default_weight": "best.pt",
            }
        }
    },
)
@api_view(["GET"])
def weights(request):
    """Get list of available weights (from the external detector box when
    DETECTOR_HOST is set, else local)."""
    available, default = _detection_service.available_detection_weights()
    return Response(
        {
            "available_weights": available,
            "default_weight": default,
        }
    )


@extend_schema(
    summary="[DEPRECATED] Switch model weight",
    description="Deprecated: Send weight_name with each detection request instead.",
    request=SwitchWeightRequest,
    responses={200: {"example": {"message": "Switched to weight: best.pt", "deprecated": True}}},
)
@api_view(["POST"])
@parser_classes([JSONParser])
def switch_weight(request):
    """[DEPRECATED] Switch to a different weight. Send weight_name with each request instead."""
    validation = SwitchWeightRequest(request.data)

    if validation.fails():
        return validation.errors_response()

    return _detection_service.switch_weight_handler(validation.validated())


@extend_schema(
    summary="Detect logos in images",
    description="Detect logos in one or more uploaded images",
    request={
        "multipart/form-data": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string", "format": "binary"},
                    "description": "Image files to process",
                },
                "confidence_threshold": {
                    "type": "number",
                    "format": "float",
                    "default": 0.5,
                    "description": "Confidence threshold (0.0-1.0)",
                },
                "enable_classification": {
                    "type": "boolean",
                    "default": False,
                    "description": "Run asset classification on each detection",
                },
                "classification_weight_name": {
                    "type": "string",
                    "description": "Classification model weight to use when enable_classification is true",
                },
            },
            "required": ["files"],
        }
    },
    responses={
        200: {"example": {"results": [{"detections": [], "total_detections": 0}]}}
    },
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def detect_images(request):
    """Detect logos in images"""
    # Validate request data (BaseRequest now handles QueryDict immutability)
    validation = DetectImagesRequest(request.data, request.FILES)

    if validation.fails():
        return validation.errors_response()

    # Check for files
    files = request.FILES.getlist("files")
    if not files:
        return Response(
            {"error": "files", "message": "At least one file is required"}, status=422
        )

    # Pass validated data directly to service - no need to modify request.data
    validated_data = validation.validated()
    return _detection_service.detect_images_handler(request, validated_data)


@extend_schema(
    summary="Detect logos in video",
    description="Detect logos in uploaded video or video from URL. Returns Server-Sent Events (SSE) stream.",
    request={
        "multipart/form-data": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "format": "binary",
                    "description": "Video file to process",
                },
                "file_url": {
                    "type": "string",
                    "format": "uri",
                    "description": "URL of video to download and process",
                },
                "frames_per_second": {
                    "type": "integer",
                    "default": 2,
                    "description": "Frames per second to process (1-30)",
                },
                "confidence_threshold": {
                    "type": "number",
                    "format": "float",
                    "default": 0.5,
                    "description": "Confidence threshold (0.0-1.0)",
                },
                "create_video": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to create processed video with annotations",
                },
                "enable_classification": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to enable logo classification",
                },
                "enable_ocr": {
                    "type": "boolean",
                    "default": False,
                    "description": "OCR each processed frame via the external GLM endpoint. Requires Spaces configured.",
                },
                "sport": {
                    "type": "string",
                    "description": "Sport name; resolves to the saved OCR prompt for that sport (used when enable_ocr is true).",
                },
                "prompt_slug": {
                    "type": "string",
                    "description": "Slug of a saved SportPrompt to use for OCR (overrides sport).",
                },
                "prompt": {
                    "type": "string",
                    "description": "Inline OCR prompt (overrides prompt_slug / sport).",
                },
            },
        }
    },
    responses={
        200: {
            "description": "Server-Sent Events stream",
            "content": {
                "text/event-stream": {
                    "example": 'data: {"type": "frame", "frame_number": 0, "detections": []}\n\n',
                }
            },
        }
    },
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def detect_video(request):
    """Queue a video for background detection.

    Returns 202 with a session_id immediately — processing runs in the
    `detection` RQ worker, NOT on this request. Subscribe to
    GET /video/detect/{session_id}/events for the live SSE progress feed.
    """
    validation = DetectVideoRequest(request.data, request.FILES)

    if validation.fails():
        return validation.errors_response()

    validated_data = validation.validated()
    return _detection_service.enqueue_video_handler(request, validated_data)


# Plain Django view (NOT @api_view): DRF runs content negotiation against the
# request's Accept header, and since only JSONRenderer is configured it answers
# `text/event-stream` with 406 "Could not satisfy the request Accept header".
# An SSE endpoint returns raw bytes and must bypass DRF's renderer pipeline.
@require_GET
def video_events(request, session_id):
    """Reconnectable SSE progress feed for a detection session.

    GET /api/v1/video/detect/{session_id}/events?since={last_frame_number}
    """
    # Cursor is the last frame_number already seen; -1 (the default when absent)
    # means "from the very start" so frame 0 is included.
    try:
        since = int(request.GET.get("since", -1))
    except (TypeError, ValueError):
        since = -1

    response = StreamingHttpResponse(
        _detection_service.iter_progress_events(session_id, since=since),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"  # disable nginx buffering for SSE
    return response


@extend_schema(
    summary="Resume a stopped/interrupted detection session",
    description=(
        "Re-enqueue an interrupted or failed session from where it left off "
        "(processed_frames). No-op if the session is already active or "
        "complete. The source video is re-read from the stored source_url."
    ),
    parameters=[
        OpenApiParameter(
            name="session_id", type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH, description="Session ID",
        ),
    ],
    responses={200: {"example": {"ok": True, "status": "queued", "start_frame": 1492}}},
)
@api_view(["POST"])
def resume_video(request, session_id):
    """Resume a stopped session from its last processed frame."""
    result = _detection_service.resume_session(session_id)
    if not result.get("ok"):
        return Response(result, status=404 if "not found" in str(result.get("error", "")).lower() else 400)
    return Response(result)


@extend_schema(
    summary="Cancel a running/queued detection session",
    description=(
        "Stops a session that is queued or processing. The worker notices "
        "within ~2s and stops; the session is left INTERRUPTED and can be "
        "resumed later from where it stopped."
    ),
    parameters=[
        OpenApiParameter(
            name="session_id", type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH, description="Session ID",
        ),
    ],
    responses={200: {"example": {"ok": True, "status": "interrupted"}}},
)
@api_view(["POST"])
def cancel_video(request, session_id):
    """Cancel a queued/running session (stays resumable)."""
    result = _detection_service.cancel_session(session_id)
    if not result.get("ok"):
        return Response(result, status=404)
    return Response(result)


@extend_schema(
    summary="Detection/OCR queue status",
    description="Per-queue pending/started/failed counts and LIVE worker count. "
    "A queue with jobs but 0 workers is why sessions stay 'queued'.",
    responses={200: {"example": {"detection": {"pending": 3, "workers": 1, "total": 3}}}},
)
@api_view(["GET"])
def queue_status(request):
    """Queue depths + worker counts."""
    return Response(_detection_service.get_queue_status())


@extend_schema(
    summary="Clear a job queue",
    description="Drop all pending jobs from a queue (default: detection). For "
    "the detection queue, also marks not-yet-started sessions as cancelled so "
    "they don't get re-queued. A running job is left alone.",
    request={"application/json": {"type": "object", "properties": {
        "queue": {"type": "string", "enum": ["detection", "ocr"], "default": "detection"},
    }}},
    responses={200: {"example": {"ok": True, "queue": "detection", "removed": {"total": 12}, "cancelled_sessions": 3}}},
)
@api_view(["POST"])
@parser_classes([JSONParser])
def queue_clear(request):
    """Clear a job queue (detection or ocr)."""
    queue = (request.data or {}).get("queue", "detection")
    if queue not in ("detection", "ocr"):
        return Response({"ok": False, "error": f"unknown queue '{queue}'"}, status=400)
    result = _detection_service.clear_queue(queue)
    return Response(result, status=200 if result.get("ok") else 502)


@extend_schema(
    summary="Get a presigned URL to upload a video straight to Spaces",
    description=(
        "Returns a presigned PUT URL the browser uploads the video to directly "
        "(bypassing this backend, so large files don't hit the proxy), plus a "
        "presigned GET URL to pass back as `file_url` on /video/detect."
    ),
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Original filename (used for the extension)."},
                "content_type": {"type": "string", "description": "MIME type, e.g. video/mp4."},
            },
        }
    },
    responses={
        200: {
            "example": {
                "upload_url": "https://bucket.region.digitaloceanspaces.com/video_uploads/ab12.mp4?X-Amz-...",
                "download_url": "https://bucket.region.digitaloceanspaces.com/video_uploads/ab12.mp4?X-Amz-...",
                "key": "video_uploads/ab12.mp4",
            }
        }
    },
)
@api_view(["POST"])
@parser_classes([JSONParser])
def video_upload_url(request):
    """Mint presigned upload (PUT) + download (GET) URLs for a video on Spaces."""
    import uuid
    from pathlib import Path as _Path
    from rest_framework import status as _status

    if not _spaces_service.is_configured():
        return Response(
            {"error": "Spaces is not configured — set the DO_* env vars."},
            status=_status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # Only allow video extensions so the presigned URL can't be used to stash
    # arbitrary file types (.html/.svg/.js) in the bucket. Objects are private
    # (no public-read ACL), but this keeps the upload surface narrow.
    allowed_exts = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v", ".mpeg", ".mpg"}
    data = request.data or {}
    filename = str(data.get("filename") or "video.mp4")
    ext = (_Path(filename).suffix or ".mp4").lower()
    if ext not in allowed_exts:
        return Response(
            {"error": f"unsupported video extension '{ext}'"},
            status=_status.HTTP_422_UNPROCESSABLE_ENTITY,
        )
    key = f"video_uploads/{uuid.uuid4().hex}{ext}"

    # GET must outlive the upload + the whole detection download; keep it long.
    try:
        upload_url = _spaces_service.presigned_put_url(key, expires=3600)
        download_url = _spaces_service.presigned_get_url(key, expires=21600)
    except Exception as exc:  # noqa: BLE001
        return Response(
            {"error": f"failed to presign Spaces URL: {exc}"},
            status=_status.HTTP_502_BAD_GATEWAY,
        )

    return Response({"upload_url": upload_url, "download_url": download_url, "key": key})


@extend_schema(
    summary="Get session summary",
    description="Get summary of detection session including total detections and logo counts",
    parameters=[
        OpenApiParameter(
            name="session_id",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            description="Session ID from video processing",
        ),
    ],
    responses={
        200: {
            "example": {
                "session_id": "video_123",
                "total_frames_processed": 100,
                "logo_totals": {},
            }
        }
    },
)
@api_view(["GET"])
def session_summary(request, session_id):
    """Get session summary"""
    return Response(_detection_service.get_session_summary(session_id))


@extend_schema(
    summary="Get real-time CSV files",
    description="Get real-time CSV files generated during video processing",
    parameters=[
        OpenApiParameter(
            name="session_id",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            description="Session ID from video processing",
        ),
    ],
    responses={
        200: {
            "example": {
                "csv_files": {"main": "/static/csv_reports/file.csv"},
                "session_id": "video_123",
            }
        }
    },
)
@api_view(["GET"])
def realtime_csv(request, session_id):
    """Get real-time CSV files for a session"""
    csv_files = _detection_service.get_realtime_csv_files(session_id)
    return Response({"csv_files": csv_files, "session_id": session_id})


@extend_schema(
    summary="Export session to CSV",
    description="Export session detection data to CSV file from database",
    request={
        "application/json": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to export"},
                "filename_prefix": {
                    "type": "string",
                    "description": "Optional filename prefix",
                },
            },
            "required": ["session_id"],
        }
    },
    responses={
        200: {
            "example": {
                "message": "CSV files exported successfully",
                "csv_files": {},
                "session_id": "video_123",
            }
        }
    },
)
@api_view(["POST"])
@parser_classes([JSONParser])
def export_csv(request):
    """Export session data to CSV files"""
    session_id = request.data.get("session_id")
    filename_prefix = request.data.get("filename_prefix")

    if not session_id:
        return Response({"error": "session_id is required"}, status=400)

    csv_files = _detection_service.export_session_to_csv(session_id, filename_prefix)
    return Response(
        {
            "message": "CSV files exported successfully",
            "csv_files": csv_files,
            "session_id": session_id,
        }
    )


@extend_schema(
    summary="List CSV files",
    description="Get list of all available CSV files",
    responses={
        200: {
            "example": {
                "csv_files": [
                    {"filename": "file.csv", "path": "/static/csv_reports/file.csv"}
                ]
            }
        }
    },
)
@api_view(["GET"])
def csv_files(request):
    """Get list of available CSV files"""
    csv_files = _detection_service.get_available_csv_files()
    return Response({"csv_files": csv_files})


@extend_schema(
    summary="Download CSV file",
    description="Download a specific CSV file",
    parameters=[
        OpenApiParameter(
            name="filename",
            type=OpenApiTypes.STR,
            location=OpenApiParameter.PATH,
            description="CSV filename to download",
        ),
    ],
    responses={200: {"description": "CSV file download"}},
)
@api_view(["GET"])
def download_csv(request, filename):
    """Download a specific CSV file"""
    from django.http import FileResponse
    from pathlib import Path
    from django.conf import settings

    csv_dir = Path(settings.STATIC_ROOT or settings.STATICFILES_DIRS[0]) / "csv_reports"
    file_path = csv_dir / filename

    if not file_path.exists():
        return Response({"error": "File not found"}, status=404)

    return FileResponse(
        open(file_path, "rb"),
        content_type="text/csv",
        filename=filename,
    )


@extend_schema(
    summary="Cleanup CSV files",
    description="Clean up old CSV files, keeping only the most recent ones",
    parameters=[
        OpenApiParameter(
            name="max_files",
            type=OpenApiTypes.INT,
            location=OpenApiParameter.QUERY,
            description="Maximum number of files to keep (1-200)",
            default=50,
        ),
    ],
    responses={
        200: {
            "example": {"message": "Cleaned up old CSV files, keeping 50 most recent"}
        }
    },
)
@api_view(["DELETE"])
def cleanup_csv(request):
    """Clean up old CSV files"""
    max_files = int(request.GET.get("max_files", 50))

    if max_files < 1 or max_files > 200:
        return Response({"error": "max_files must be between 1 and 200"}, status=400)

    _detection_service.cleanup_old_csv_files(max_files)
    return Response(
        {"message": f"Cleaned up old CSV files, keeping {max_files} most recent"}
    )


# URL patterns - Using regex to support both with and without trailing slashes
urlpatterns = [
    path("", index, name="detection-index"),  # GET /api/v1/
    *optional_slash_path(
        "health", health, name="detection-health"
    ),  # GET /api/v1/health or /api/v1/health/
    *optional_slash_path(
        "device", device, name="detection-device"
    ),  # GET /api/v1/device or /api/v1/device/
    *optional_slash_path(
        "config", config, name="detection-config"
    ),  # GET/POST /api/v1/config or /api/v1/config/
    *optional_slash_path(
        "weights", weights, name="detection-weights"
    ),  # GET /api/v1/weights or /api/v1/weights/
    *optional_slash_path(
        "weights/switch", switch_weight, name="detection-weights-switch"
    ),  # POST /api/v1/weights/switch or /api/v1/weights/switch/
    *optional_slash_path(
        "images/detect", detect_images, name="detection-images"
    ),  # POST /api/v1/images/detect or /api/v1/images/detect/
    *optional_slash_path(
        "video/detect", detect_video, name="detection-video"
    ),  # POST /api/v1/video/detect — queues a background job, returns session_id
    *optional_slash_path(
        r"video/detect/(?P<session_id>[^/]+)/events",
        video_events,
        name="detection-video-events",
    ),  # GET /api/v1/video/detect/{id}/events — reconnectable SSE progress feed
    *optional_slash_path(
        r"video/detect/(?P<session_id>[^/]+)/resume",
        resume_video,
        name="detection-video-resume",
    ),  # POST /api/v1/video/detect/{id}/resume — resume from processed_frames
    *optional_slash_path(
        r"video/detect/(?P<session_id>[^/]+)/cancel",
        cancel_video,
        name="detection-video-cancel",
    ),  # POST /api/v1/video/detect/{id}/cancel — stop (stays resumable)
    *optional_slash_path(
        "video/queue", queue_status, name="detection-queue-status"
    ),  # GET /api/v1/video/queue — depths + live worker counts
    *optional_slash_path(
        "video/queue/clear", queue_clear, name="detection-queue-clear"
    ),  # POST /api/v1/video/queue/clear — drop pending jobs (+cancel sessions)
    *optional_slash_path(
        "video/upload-url", video_upload_url, name="detection-video-upload-url"
    ),  # POST /api/v1/video/upload-url — presigned Spaces upload + download URLs
    *optional_slash_path(
        r"session/(?P<session_id>[^/]+)/summary",
        session_summary,
        name="detection-session-summary",
    ),  # GET /api/v1/session/{id}/summary or /api/v1/session/{id}/summary/
    *optional_slash_path(
        r"session/(?P<session_id>[^/]+)/realtime-csv",
        realtime_csv,
        name="detection-realtime-csv",
    ),  # GET /api/v1/session/{id}/realtime-csv or /api/v1/session/{id}/realtime-csv/
    *optional_slash_path(
        "session/export-csv", export_csv, name="detection-export-csv"
    ),  # POST /api/v1/session/export-csv or /api/v1/session/export-csv/
    *optional_slash_path(
        "csv-files", csv_files, name="detection-csv-files"
    ),  # GET /api/v1/csv-files or /api/v1/csv-files/
    *optional_slash_path(
        r"csv-files/download/(?P<filename>[^/]+)",
        download_csv,
        name="detection-download-csv",
    ),  # GET /api/v1/csv-files/download/{filename} or /api/v1/csv-files/download/{filename}/
    *optional_slash_path(
        "csv-files/cleanup", cleanup_csv, name="detection-cleanup-csv"
    ),  # DELETE /api/v1/csv-files/cleanup or /api/v1/csv-files/cleanup/
]
