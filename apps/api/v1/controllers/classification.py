"""
Classification Controller - Laravel-style (thin controllers)
"""

from django.urls import path, re_path
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.response import Response
from apps.api.v1.requests.classification_requests import (
    SwitchClassificationWeightRequest,
    ClassifyImagesRequest,
)

# Import shared services (initialized once to prevent duplicate logs)
from apps.api.v1.shared_services import _classification_service


def optional_slash_path(route, view, name=None):
    """Helper to create URL patterns that work with or without trailing slashes"""
    return [
        re_path(rf"^{route}/?$", view, name=name),
    ]


urlpatterns = []


@api_view(["GET"])
def index(request):
    """Root classification endpoint"""
    return Response({"message": "Classification API", "status": "running"})


@api_view(["GET"])
def health(request):
    """Health check for classification service"""
    return Response(
        {
            "status": "healthy",
            "model_loaded": _classification_service.is_model_loaded(),
        }
    )


@api_view(["GET"])
def weights(request):
    """Get list of available classification weights"""
    return Response(
        {
            "available_weights": _classification_service.get_available_weights(),
            "current_weight": _classification_service.get_current_weight(),
        }
    )


@api_view(["POST"])
@parser_classes([JSONParser])
def switch_weight(request):
    """Switch to a different classification weight"""
    validation = SwitchClassificationWeightRequest(request.data)

    if validation.fails():
        return validation.errors_response()

    return _classification_service.switch_weight_handler(validation.validated())


@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def classify_images(request):
    """Classify images and return top-k predictions"""
    # Validate request data
    validation = ClassifyImagesRequest(request.data, request.FILES)

    if validation.fails():
        return validation.errors_response()

    # Check for files
    files = request.FILES.getlist("files")
    if not files:
        return Response(
            {"error": "files", "message": "At least one file is required"}, status=422
        )

    # Pass validated data to service
    request.data.update(validation.validated())
    return _classification_service.classify_images_handler(request)


# URL patterns - Using regex to support both with and without trailing slashes
urlpatterns = [
    path("", index, name="classification-index"),  # GET /api/v1/classification/
    *optional_slash_path(
        "health", health, name="classification-health"
    ),  # GET /api/v1/classification/health or /api/v1/classification/health/
    *optional_slash_path(
        "weights", weights, name="classification-weights"
    ),  # GET /api/v1/classification/weights or /api/v1/classification/weights/
    *optional_slash_path(
        "weights/switch", switch_weight, name="classification-weights-switch"
    ),  # POST /api/v1/classification/weights/switch or /api/v1/classification/weights/switch/
    *optional_slash_path(
        "images/classify", classify_images, name="classification-images"
    ),  # POST /api/v1/classification/images/classify or /api/v1/classification/images/classify/
]
