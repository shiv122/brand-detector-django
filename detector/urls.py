"""
URL configuration for detector project.
"""

import os
from pathlib import Path

from django.contrib import admin
from django.http import JsonResponse
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularRedocView,
    SpectacularSwaggerView,
)

_BUILD_TIME_FILE = Path("/app/BUILD_TIME")


def root_index(_request):
    build_time = "dev"
    if _BUILD_TIME_FILE.exists():
        try:
            build_time = _BUILD_TIME_FILE.read_text().strip() or "unknown"
        except OSError:
            build_time = "unknown"
    return JsonResponse(
        {
            "service": "detector-backend",
            "status": "ok",
            "build_time": build_time,
            "git_sha": os.environ.get("BUILD_GIT_SHA", "unknown"),
        }
    )


urlpatterns = [
    path("", root_index, name="root-index"),
    path("admin/", admin.site.urls),
    path("api/", include("apps.api.urls")),
    # API Documentation
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("redoc/", SpectacularRedocView.as_view(url_name="schema"), name="redoc"),
]

# WhiteNoise middleware handles static file serving automatically
# It serves files from STATIC_ROOT at /static/ URL with compression and caching
# For /api/v1/static/ compatibility, we add a URL pattern that redirects to /static/
if not settings.DEBUG:
    # In production, add URL pattern for /api/v1/static/ compatibility
    # WhiteNoise serves /static/ automatically, this redirects /api/v1/static/ to /static/
    urlpatterns += [
        re_path(
            r"^api/v1/static/(?P<path>.*)$",
            serve,
            {"document_root": settings.STATIC_ROOT},
        ),
    ]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
