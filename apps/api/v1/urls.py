"""
API v1 URL configuration - Laravel-style
Matches old backend structure: /api/* for detection, /api/classification/* for classification
"""

from django.urls import path, include

urlpatterns = [
    # Detection endpoints (no /detection/ prefix to match old backend)
    # Includes: /api/v1/health, /api/v1/device, /api/v1/config, etc.
    path("", include("apps.api.v1.controllers.detection")),
    # Classification endpoints
    # Includes: /api/v1/classification/health, /api/v1/classification/weights, etc.
    path("classification/", include("apps.api.v1.controllers.classification")),
]
