"""
Detection Controller - Laravel-style (thin controllers)
"""

from django.urls import path, re_path
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
    description="Get list of available model weights and current weight",
    responses={
        200: {
            "example": {
                "available_weights": [{"name": "best.pt", "size": 40800000}],
                "current_weight": "best.pt",
            }
        }
    },
)
@api_view(["GET"])
def weights(request):
    """Get list of available weights"""
    return Response(
        {
            "available_weights": _detection_service.get_available_weights(),
            "current_weight": _detection_service.get_current_weight(),
        }
    )


@extend_schema(
    summary="Switch model weight",
    description="Switch to a different YOLO model weight",
    request=SwitchWeightRequest,
    responses={200: {"example": {"message": "Switched to weight: best.pt"}}},
)
@api_view(["POST"])
@parser_classes([JSONParser])
def switch_weight(request):
    """Switch to a different weight"""
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
    """Detect logos in video"""
    # Validate request data (BaseRequest now handles QueryDict immutability)
    validation = DetectVideoRequest(request.data, request.FILES)

    if validation.fails():
        return validation.errors_response()

    # Pass validated data directly to service - no need to modify request.data
    validated_data = validation.validated()
    return _detection_service.detect_video_handler(request, validated_data)


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
    ),  # POST /api/v1/video/detect or /api/v1/video/detect/
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
