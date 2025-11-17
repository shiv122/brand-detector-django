"""
URL configuration for detector project.
"""

from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularRedocView,
    SpectacularSwaggerView,
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("apps.api.urls")),
    # API Documentation
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("redoc/", SpectacularRedocView.as_view(url_name="schema"), name="redoc"),
]

# Serve static and media files
# In production, use a web server (nginx) or WhiteNoise for static files
# This is a fallback for development and when DEBUG=True
if settings.DEBUG:
    # Serve static files at /static/ (standard Django)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    # Also serve static files at /api/v1/static/ for frontend compatibility
    urlpatterns += [
        re_path(
            r"^api/v1/static/(?P<path>.*)$",
            serve,
            {"document_root": settings.STATIC_ROOT},
        ),
    ]
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
else:
    # In production, serve static files at /api/v1/static/ for frontend compatibility
    # Note: For production, it's recommended to use nginx or WhiteNoise middleware
    # This is a fallback that works but is not optimal for high traffic
    urlpatterns += [
        re_path(
            r"^api/v1/static/(?P<path>.*)$",
            serve,
            {"document_root": settings.STATIC_ROOT},
        ),
        re_path(
            r"^static/(?P<path>.*)$",
            serve,
            {"document_root": settings.STATIC_ROOT},
        ),
    ]
