"""
API URL configuration - Laravel-style
"""

from django.urls import path, include

urlpatterns = [
    # API v1
    path("v1/", include("apps.api.v1.urls")),
]
