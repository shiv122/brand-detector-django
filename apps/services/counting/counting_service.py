"""
Counting service for tracking detections per session (MySQL-based) - Laravel-style
"""

from typing import Dict
from apps.core.models import ProcessingSession, SessionSummary, Detection


class CountingService:
    """Service to handle logo counting and session summaries using MySQL"""

    def process_frame_detections(
        self,
        session: ProcessingSession,
        frame_number: int,
        detections: list,
        timestamp: float = None,
    ) -> Dict[str, int]:
        """Process detections for a frame and update counts"""
        # Count logos in this frame
        frame_logo_counts = {}
        for detection in detections:
            logo_name = detection.class_name
            frame_logo_counts[logo_name] = frame_logo_counts.get(logo_name, 0) + 1

        # Update session summary
        self._update_session_summary(session)

        return frame_logo_counts

    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary of detection session"""
        try:
            session = ProcessingSession.objects.get(session_id=session_id)
        except ProcessingSession.DoesNotExist:
            return {
                "session_id": session_id,
                "total_frames_processed": 0,
                "logo_totals": {},
                "total_detections": 0,
                "unique_logos": [],
            }

        # Get or create summary
        summary, _ = SessionSummary.objects.get_or_create(session=session)

        # Get frame count
        total_frames = session.frames.count()

        return {
            "session_id": session_id,
            "total_frames_processed": total_frames,
            "logo_totals": summary.logo_counts or {},
            "total_detections": summary.total_detections,
            "unique_logos": summary.unique_logos or [],
        }

    def _update_session_summary(self, session: ProcessingSession):
        """Update session summary with current counts"""
        from django.db.models import Count

        # Get logo counts from database
        logo_counts = (
            Detection.objects.filter(session=session)
            .values("class_name")
            .annotate(count=Count("id"))
            .order_by("-count")
        )

        logo_counts_dict = {item["class_name"]: item["count"] for item in logo_counts}
        # Get unique logos - use set to ensure uniqueness, then sort
        unique_logos = sorted(
            set(
                Detection.objects.filter(session=session)
                .values_list("class_name", flat=True)
            )
        )
        total_detections = sum(logo_counts_dict.values())

        # Get or create summary
        summary, _ = SessionSummary.objects.get_or_create(session=session)

        # Update summary
        summary.logo_counts = logo_counts_dict
        summary.unique_logos = unique_logos
        summary.total_detections = total_detections
        summary.save()

    def finalize_session(self, session_id: str):
        """Finalize session and update summary"""
        try:
            session = ProcessingSession.objects.get(session_id=session_id)
            self._update_session_summary(session)
        except ProcessingSession.DoesNotExist:
            pass
