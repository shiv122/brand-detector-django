"""
Dashboard Controller - Statistics and analytics endpoints
"""

from django.urls import re_path
from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema
from django.db.models import Count, Q, Sum
from django.utils import timezone
from datetime import timedelta
from apps.core.models import ProcessingSession, Detection, Frame, Classification
from apps.core.enums import ProcessingStatus


def optional_slash_path(route, view, name=None):
    """Helper to create URL patterns that work with or without trailing slashes"""
    return [
        re_path(rf"^{route}/?$", view, name=name),
    ]


urlpatterns = []


@extend_schema(
    summary="Get dashboard statistics",
    description="Get comprehensive statistics for the dashboard including totals, top brands, recent activity, and processing queue",
    responses={
        200: {
            "example": {
                "overview": {
                    "total_detections": 12847,
                    "images_processed": 3421,
                    "videos_processed": 287,
                    "total_sessions": 3708,
                },
                "top_brands": [
                    {"name": "Nike", "detections": 234, "percentage": 18.7},
                    {"name": "Adidas", "detections": 189, "percentage": 15.1},
                ],
                "recent_activity": [
                    {
                        "id": 1,
                        "session_id": "video_123",
                        "type": "video",
                        "name": "sports_highlights.mp4",
                        "detections": 45,
                        "status": "completed",
                        "created_at": "2024-01-01T12:00:00Z",
                    }
                ],
                "processing_queue": [
                    {
                        "session_id": "video_456",
                        "name": "video_001.mp4",
                        "progress": 75,
                        "status": "processing",
                    }
                ],
                "brand_distribution": [
                    {"date": "2024-01-01", "Nike": 10, "Adidas": 8},
                ],
            }
        }
    },
)
@api_view(["GET"])
def dashboard_stats(request):
    """Get dashboard statistics"""
    # Overview statistics
    total_detections = Detection.objects.count()
    
    # Count image sessions (sessions without video_path)
    images_processed = ProcessingSession.objects.filter(
        video_path__isnull=True
    ).count()
    
    # Count video sessions (sessions with video_path)
    videos_processed = ProcessingSession.objects.filter(
        video_path__isnull=False
    ).exclude(video_path="").count()
    
    total_sessions = ProcessingSession.objects.count()
    
    # Top brands (most detected logos)
    top_brands_data = (
        Detection.objects.values("class_name")
        .annotate(detections=Count("id"))
        .order_by("-detections")[:10]
    )
    
    total_brand_detections = sum(item["detections"] for item in top_brands_data)
    top_brands = [
        {
            "name": item["class_name"],
            "detections": item["detections"],
            "percentage": round((item["detections"] / total_brand_detections * 100), 1)
            if total_brand_detections > 0
            else 0,
        }
        for item in top_brands_data
    ]
    
    # Recent activity (last 10 completed sessions)
    recent_sessions = (
        ProcessingSession.objects.filter(status=ProcessingStatus.COMPLETED.value)
        .order_by("-created_at")[:10]
        .select_related()
    )
    
    recent_activity = []
    for session in recent_sessions:
        # Get detection count for this session
        detection_count = Detection.objects.filter(session=session).count()
        
        # Determine type based on video_path
        session_type = "video" if session.video_path else "image"
        
        # Get filename
        filename = session.video_filename or f"session_{session.session_id[:8]}"
        
        recent_activity.append(
            {
                "id": session.id,
                "session_id": session.session_id,
                "type": session_type,
                "name": filename,
                "detections": detection_count,
                "status": session.status,
                "created_at": session.created_at.isoformat(),
            }
        )
    
    # Processing queue (active sessions)
    active_sessions = ProcessingSession.objects.filter(
        status__in=[
            ProcessingStatus.PROCESSING.value,
            ProcessingStatus.PENDING.value,
        ]
    ).order_by("-created_at")[:5]
    
    processing_queue = []
    for session in active_sessions:
        # Calculate progress
        if session.total_frames > 0:
            progress = int((session.processed_frames / session.total_frames) * 100)
        else:
            progress = 0
        
        filename = session.video_filename or f"session_{session.session_id[:8]}"
        
        processing_queue.append(
            {
                "session_id": session.session_id,
                "name": filename,
                "progress": progress,
                "status": session.status,
            }
        )
    
    # Brand distribution over time (last 30 days)
    thirty_days_ago = timezone.now() - timedelta(days=30)
    recent_detections = Detection.objects.filter(
        created_at__gte=thirty_days_ago
    ).select_related("session")
    
    # Group by date and brand
    brand_distribution = {}
    for detection in recent_detections:
        date_str = detection.created_at.date().isoformat()
        if date_str not in brand_distribution:
            brand_distribution[date_str] = {}
        brand_name = detection.class_name
        brand_distribution[date_str][brand_name] = (
            brand_distribution[date_str].get(brand_name, 0) + 1
        )
    
    # Convert to list format
    brand_distribution_list = [
        {"date": date, **brands} for date, brands in sorted(brand_distribution.items())
    ]
    
    # Asset statistics (classifications)
    total_assets = Classification.objects.count()
    
    # Top assets (most classified)
    top_assets_data = (
        Classification.objects.filter(rank=1)  # Only top-1 classifications
        .values("class_name")
        .annotate(count=Count("id"))
        .order_by("-count")[:10]
    )
    
    top_assets = [
        {
            "name": item["class_name"],
            "count": item["count"],
        }
        for item in top_assets_data
    ]
    
    # Assets per brand (brands with their top assets)
    assets_per_brand = {}
    classifications_with_detections = (
        Classification.objects.filter(rank=1)
        .select_related("detection")
        .values("detection__class_name", "class_name")
        .annotate(count=Count("id"))
        .order_by("-count")
    )
    
    for item in classifications_with_detections:
        brand_name = item["detection__class_name"]
        asset_name = item["class_name"]
        count = item["count"]
        
        if brand_name not in assets_per_brand:
            assets_per_brand[brand_name] = []
        
        assets_per_brand[brand_name].append({
            "asset_name": asset_name,
            "count": count,
        })
    
    # Limit to top 5 brands and top 3 assets per brand
    assets_per_brand_limited = {}
    for brand_name, assets in list(assets_per_brand.items())[:5]:
        assets_per_brand_limited[brand_name] = sorted(
            assets, key=lambda x: x["count"], reverse=True
        )[:3]
    
    # Detection types breakdown (video vs image)
    video_detections = Detection.objects.filter(
        session__video_path__isnull=False
    ).exclude(session__video_path="").count()
    
    image_detections = Detection.objects.filter(
        session__video_path__isnull=True
    ).count()
    
    return Response(
        {
            "overview": {
                "total_detections": total_detections,
                "images_processed": images_processed,
                "videos_processed": videos_processed,
                "total_sessions": total_sessions,
                "total_assets": total_assets,
            },
            "top_brands": top_brands,
            "top_assets": top_assets,
            "assets_per_brand": assets_per_brand_limited,
            "detection_types": {
                "video": video_detections,
                "image": image_detections,
            },
            "recent_activity": recent_activity,
            "processing_queue": processing_queue,
            "brand_distribution": brand_distribution_list,
        }
    )


# URL patterns
urlpatterns = [
    *optional_slash_path("dashboard/stats", dashboard_stats, name="dashboard-stats"),
]

