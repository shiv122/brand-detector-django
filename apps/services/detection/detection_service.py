"""
Detection service for logo detection in images and videos - Laravel-style
"""

import cv2
import numpy as np
import secrets
import time
import json
import os
import subprocess
import shutil
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Generator
from datetime import datetime
from django.http import StreamingHttpResponse, FileResponse
from rest_framework.response import Response
from rest_framework import status
from config.app_config import AppConfig
from apps.services.model.model_service import ModelService, DetectionResult
from apps.services.image.image_service import ImageService
from apps.services.classification.classification_service import ClassificationService
from apps.services.counting.counting_service import CountingService
from apps.core.models import ProcessingSession, Frame, Detection, Classification
from apps.core.enums import ProcessingStatus
from apps.utils.file_helpers import ensure_directory_exists
import requests


class DetectionService:
    """Service for logo detection"""

    def __init__(
        self,
        config: AppConfig,
        model_service: ModelService,
        image_service: ImageService,
        classification_service: Optional[ClassificationService] = None,
        counting_service: Optional[CountingService] = None,
    ):
        self.config = config
        self.model_service = model_service
        self.image_service = image_service
        self.classification_service = classification_service
        self.counting_service = counting_service
        self.csv_dir = Path(self.config.static_dir) / "csv_reports"

        # Ensure directories exist
        self._setup_directories()

    def _setup_directories(self):
        """Setup required directories"""
        ensure_directory_exists(self.config.static_dir)
        ensure_directory_exists(self.config.frames_dir)
        ensure_directory_exists(Path(self.config.static_dir) / "temp_frames")
        ensure_directory_exists(str(self.csv_dir))

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_service.is_loaded()

    def get_available_weights(self) -> List[dict]:
        """Get list of available weights"""
        return self.model_service.get_available_weights()

    def get_current_weight(self) -> str:
        """Get the currently selected weight"""
        return self.model_service.get_current_model_name()

    def switch_weight(self, weight_name: str) -> bool:
        """Switch to a different weight"""
        return self.model_service.switch_model(weight_name)

    def detect_in_image(
        self, image_data: bytes, confidence_threshold: float = 0.5
    ) -> Tuple[List[DetectionResult], Optional[np.ndarray]]:
        """Detect logos in a single image"""
        return self.model_service.detect_in_image(image_data, confidence_threshold)

    def _crop_detection_box(
        self, frame: np.ndarray, bbox: List[float], padding: int = 40
    ) -> np.ndarray:
        """Crop detection box from frame with padding"""
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]

        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(width, int(x2) + padding)
        y2 = min(height, int(y2) + padding)

        cropped = frame[y1:y2, x1:x2]
        return cropped

    def _classify_detection(
        self, frame: np.ndarray, detection: DetectionResult
    ) -> Optional[List]:
        """Classify a detection by cropping and running classification model"""
        if (
            not self.classification_service
            or not self.classification_service.is_model_loaded()
        ):
            return None

        try:
            cropped = self._crop_detection_box(frame, detection.bbox, padding=40)

            if cropped.size == 0:
                return None

            _, buffer = cv2.imencode(".jpg", cropped)
            image_bytes = buffer.tobytes()

            classification_results = self.classification_service.classify_image(
                image_bytes, top_k=3
            )
            return [r.to_dict() for r in classification_results]
        except Exception as e:
            print(f"[CLASSIFICATION] Error classifying detection: {str(e)}")
            return None

    def update_config(self, data: dict) -> Response:
        """Update configuration (data is already validated)"""
        self.config.frames_per_second = data["frames_per_second"]
        self.config.confidence_threshold = data["confidence_threshold"]
        return Response({"message": "Configuration updated successfully"})

    def switch_weight_handler(self, data: dict) -> Response:
        """Handle weight switching (data is already validated)"""
        weight_name = data["weight_name"]
        success = self.switch_weight(weight_name)

        if success:
            return Response({"message": f"Switched to weight: {weight_name}"})

        return Response(
            {"error": f"Failed to switch to weight: {weight_name}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    def detect_images_handler(self, request, validated_data: dict = None) -> Response:
        """Handle image detection request (data is already validated)"""
        if not self.is_model_loaded():
            return Response(
                {"error": "Model not loaded"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        files = request.FILES.getlist("files")
        # Use validated_data if provided, otherwise fall back to request.data
        if validated_data is not None:
            confidence_threshold = float(
                validated_data.get("confidence_threshold", 0.5)
            )
        else:
            confidence_threshold = float(request.data.get("confidence_threshold", 0.5))

        results = []
        for file in files:
            if not self.image_service.validate_image_file(file.content_type, file.name):
                results.append(
                    {
                        "detections": [],
                        "total_detections": 0,
                        "error": f"File {file.name} is not a valid image",
                    }
                )
                continue

            try:
                contents = file.read()
                detections, annotated_image = self.detect_in_image(
                    contents, confidence_threshold
                )

                annotated_image_b64 = None
                if annotated_image is not None:
                    annotated_image_b64 = self.image_service.image_to_base64(
                        annotated_image
                    )

                results.append(
                    {
                        "detections": [detection.to_dict() for detection in detections],
                        "total_detections": len(detections),
                        "annotated_image": annotated_image_b64,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "detections": [],
                        "total_detections": 0,
                        "error": str(e),
                    }
                )

        return Response({"results": results})

    def detect_video_handler(
        self, request, validated_data: dict = None
    ) -> StreamingHttpResponse:
        """Handle video detection request with SSE streaming (data is already validated)"""
        if not self.is_model_loaded():
            return Response(
                {"error": "Model not loaded"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        file = request.FILES.get("file")
        # Use validated_data if provided, otherwise fall back to request.data
        if validated_data is not None:
            file_url = validated_data.get("file_url")
            frames_per_second = int(validated_data.get("frames_per_second", 2))
            confidence_threshold = float(
                validated_data.get("confidence_threshold", 0.5)
            )
            create_video = validated_data.get("create_video", False)
            enable_classification = validated_data.get("enable_classification", False)
        else:
            file_url = request.data.get("file_url")
            frames_per_second = int(request.data.get("frames_per_second", 2))
            confidence_threshold = float(request.data.get("confidence_threshold", 0.5))
            create_video = request.data.get("create_video", False)
            enable_classification = request.data.get("enable_classification", False)

        if file_url:
            # Stream video from URL with download progress
            response = StreamingHttpResponse(
                self._detect_video_from_url_stream(
                    file_url,
                    frames_per_second,
                    confidence_threshold,
                    create_video,
                    enable_classification,
                ),
                content_type="text/event-stream",
            )
        else:
            # Process uploaded file with streaming
            response = StreamingHttpResponse(
                self._detect_video_stream(
                    file,
                    frames_per_second,
                    confidence_threshold,
                    create_video,
                    enable_classification,
                ),
                content_type="text/event-stream",
            )

        # Set headers for SSE (Connection: keep-alive is not allowed in WSGI, removed)
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"  # Disable buffering for nginx if used
        return response

    def _detect_video_from_url_stream(
        self,
        file_url: str,
        frames_per_second: int,
        confidence_threshold: float,
        create_video: bool,
        enable_classification: bool,
    ) -> Generator[str, None, None]:
        """Stream video processing from URL with SSE"""
        video_path = None
        try:
            filename = file_url.split("/")[-1].split("?")[0] or "video.mp4"

            if not file_url.startswith(("http://", "https://")):
                yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid URL format'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'download_status', 'status': 'Connecting to server...', 'percentage': 0})}\n\n"

            # Download video synchronously with proper headers
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(
                    file_url, stream=True, timeout=300, headers=headers
                )
            except requests.exceptions.RequestException as e:
                error_msg = f"Failed to connect to URL: {str(e)}"
                print(f"[DOWNLOAD ERROR] {error_msg}")
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                return

            if response.status_code != 200:
                error_msg = f"Failed to download video: HTTP {response.status_code}"
                print(f"[DOWNLOAD ERROR] {error_msg}")
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                return

            video_filename = f"uploaded_{int(time.time())}_{filename}"
            video_path = Path(self.config.static_dir) / video_filename

            # Ensure parent directory exists
            video_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[DOWNLOAD] Saving to: {video_path}")

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            progress_update_interval = (
                max(1024 * 1024, total_size // 100) if total_size > 0 else 1024 * 1024
            )
            last_progress_update = 0

            print(
                f"[DOWNLOAD] Starting download: {video_filename} (Size: {total_size / (1024*1024):.2f} MB)"
                if total_size > 0
                else f"[DOWNLOAD] Starting download: {video_filename} (Size: Unknown)"
            )

            try:
                with open(video_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            if (
                                downloaded - last_progress_update
                                >= progress_update_interval
                            ):
                                last_progress_update = downloaded
                                if total_size > 0:
                                    percentage = int((downloaded / total_size) * 100)
                                    mb_downloaded = downloaded / (1024 * 1024)
                                    mb_total = total_size / (1024 * 1024)
                                    status_msg = f"Downloading... {mb_downloaded:.2f}MB / {mb_total:.2f}MB ({percentage}%)"
                                    yield f"data: {json.dumps({'type': 'download_status', 'status': status_msg, 'percentage': percentage})}\n\n"
                                else:
                                    mb_downloaded = downloaded / (1024 * 1024)
                                    estimated_percentage = min(
                                        95, int((mb_downloaded / 500) * 100)
                                    )
                                    status_msg = f"Downloaded {mb_downloaded:.2f}MB..."
                                    yield f"data: {json.dumps({'type': 'download_status', 'status': status_msg, 'percentage': estimated_percentage})}\n\n"

                print(
                    f"[DOWNLOAD] Download complete: {downloaded / (1024*1024):.2f} MB"
                )
                yield f"data: {json.dumps({'type': 'download_status', 'status': 'Download complete, starting processing...', 'percentage': 100})}\n\n"
            except IOError as e:
                error_msg = f"Failed to write video file: {str(e)}"
                print(f"[DOWNLOAD ERROR] {error_msg}")
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
                return

            # Process video
            session_id = f"video_{int(time.time())}_{filename.replace('.', '_')}"
            for event in self._process_video_stream(
                str(video_path),
                session_id,
                frames_per_second,
                confidence_threshold,
                create_video,
                enable_classification,
            ):
                yield event

        except Exception as e:
            import traceback

            error_msg = f"Error processing video from URL: {str(e)}"
            print(f"[DOWNLOAD ERROR] {error_msg}")
            print(f"[DOWNLOAD ERROR] Traceback: {traceback.format_exc()}")
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
        finally:
            if video_path and video_path.exists() and not create_video:
                try:
                    os.unlink(video_path)
                    print(f"[DOWNLOAD] Cleaned up temporary file: {video_path}")
                except Exception as e:
                    print(f"[DOWNLOAD] Failed to cleanup file {video_path}: {str(e)}")

    def _detect_video_stream(
        self,
        file,
        frames_per_second: int,
        confidence_threshold: float,
        create_video: bool,
        enable_classification: bool,
    ) -> Generator[str, None, None]:
        """Stream video processing with SSE"""
        video_path = None
        try:
            # Save uploaded file
            from django.conf import settings

            video_filename = f"uploaded_{int(time.time())}_{file.name}"
            video_path = (
                Path(settings.STATIC_ROOT or settings.STATICFILES_DIRS[0])
                / video_filename
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)

            with open(video_path, "wb") as f:
                for chunk in file.chunks():
                    f.write(chunk)

            # Create session
            session_id = f"video_{int(time.time())}_{file.name.replace('.', '_')}"

            # Process video with streaming
            for event in self._process_video_stream(
                str(video_path),
                session_id,
                frames_per_second,
                confidence_threshold,
                create_video,
                enable_classification,
            ):
                yield event

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Error processing video: {str(e)}'})}\n\n"
        finally:
            if video_path and video_path.exists() and not create_video:
                try:
                    os.unlink(video_path)
                except:
                    pass

    def _process_video_stream(
        self,
        video_path: str,
        session_id: str,
        frames_per_second: int,
        confidence_threshold: float,
        create_video: bool,
        enable_classification: bool,
    ) -> Generator[str, None, None]:
        """Process video and stream results via SSE"""
        # Prepare settings in JSON format
        settings = {
            "enable_classification": enable_classification,
            "create_video": create_video,
            "model_weight": self.get_current_weight(),
        }

        # Create or get session
        session, created = ProcessingSession.objects.get_or_create(
            session_id=session_id,
            defaults={
                "video_filename": Path(video_path).name,
                "frames_per_second": frames_per_second,
                "confidence_threshold": confidence_threshold,
                "status": ProcessingStatus.PENDING.value,
                "settings": settings,
            },
        )

        if not created:
            session.frames_per_second = frames_per_second
            session.confidence_threshold = confidence_threshold
            session.settings = settings
            session.save()

        session.mark_processing()
        session.video_path = video_path
        session.save()

        # Initialize CSV file for real-time export
        csv_file_path, csv_writer = self._initialize_realtime_csv(session_id)

        cap = None
        try:
            # Get video information
            video_fps, total_frames, width, height = self.image_service.get_video_info(
                video_path
            )
            skip_frames = self.image_service.calculate_skip_frames(
                video_fps, frames_per_second
            )

            session.total_frames = total_frames
            session.save()

            estimated_processed_frames = (
                total_frames // skip_frames if skip_frames > 0 else total_frames
            )

            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Starting video processing...', 'estimated_total_frames': estimated_processed_frames})}\n\n"

            # Process frames
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            processed_count = 0
            frame_prefix = secrets.token_hex(8)
            temp_frames_dir = Path(self.config.static_dir) / "temp_frames"
            temp_frames_dir.mkdir(exist_ok=True)
            detection_results = {}

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % skip_frames == 0:
                    # Run detection
                    detections, annotated_frame = self.model_service.detect_in_frame(
                        frame, confidence_threshold
                    )

                    if annotated_frame is not None:
                        # Run classification if enabled
                        classification_data = []
                        if enable_classification:
                            for detection in detections:
                                classification = self._classify_detection(
                                    frame, detection
                                )
                                if classification:
                                    classification_data.append(classification)

                        # Calculate timestamp
                        timestamp = frame_count / video_fps

                        # Save frame
                        frame_filename = (
                            f"frame_{frame_prefix}_{processed_count:06d}.jpg"
                        )
                        frame_path = Path(self.config.frames_dir) / frame_filename
                        cv2.imwrite(str(frame_path), annotated_frame)

                        frame_url = f"/api/v1/static/frames/{frame_filename}"

                        # Create frame in database
                        db_frame = Frame.objects.create(
                            session=session,
                            frame_number=processed_count,
                            frame_path=str(frame_path),
                            frame_url=frame_url,
                            timestamp=timestamp,
                            total_detections=len(detections),
                        )

                        # Save detections to database and prepare detection dicts with classification
                        detection_dicts = []
                        for idx, detection in enumerate(detections):
                            db_detection = Detection.objects.create(
                                frame=db_frame,
                                session=session,
                                class_id=detection.class_id,
                                class_name=detection.class_name,
                                confidence=detection.confidence,
                                bbox_x1=detection.bbox[0],
                                bbox_y1=detection.bbox[1],
                                bbox_x2=detection.bbox[2],
                                bbox_y2=detection.bbox[3],
                            )

                            # Get classification for this detection
                            detection_classification = None
                            if enable_classification and classification_data:
                                if idx < len(classification_data):
                                    detection_classification = classification_data[idx]

                            # Save classifications if available
                            if detection_classification:
                                for rank, cls in enumerate(detection_classification, 1):
                                    Classification.objects.create(
                                        detection=db_detection,
                                        class_id=cls["class_id"],
                                        class_name=cls["class_name"],
                                        confidence=cls["confidence"],
                                        rank=rank,
                                    )

                            # Create detection dict with classification
                            detection_dict = detection.to_dict()
                            if detection_classification:
                                detection_dict["classification"] = (
                                    detection_classification[:3]
                                )  # Top 3 only
                            detection_dicts.append(detection_dict)

                            # Write to real-time CSV
                            self._write_to_realtime_csv(
                                csv_writer,
                                csv_file_path,
                                db_detection,
                                processed_count,
                                timestamp,
                                detection_classification,
                            )

                        # Update counting
                        frame_logo_counts = {}
                        if self.counting_service:
                            frame_logo_counts = (
                                self.counting_service.process_frame_detections(
                                    session, processed_count, detections, timestamp
                                )
                            )

                        # Store for video creation
                        detection_results[frame_count] = (detections, annotated_frame)

                        # Stream frame data via SSE (minimal data - no large session_summary)
                        frame_data = {
                            "type": "frame",
                            "frame_number": processed_count,
                            "frame_url": frame_url,
                            "detections": detection_dicts,
                            "total_detections": len(detections),
                            "timestamp": timestamp,
                            "logo_counts": frame_logo_counts,
                        }
                        yield f"data: {json.dumps(frame_data)}\n\n"

                        # Send session summary only every 10 frames to reduce data size
                        if processed_count % 10 == 0 and self.counting_service:
                            summary = self.get_session_summary(session_id)
                            # Remove unique_logos array (too large) - only send counts
                            summary_data = {
                                "type": "summary",
                                "session_id": summary.get("session_id"),
                                "total_frames_processed": summary.get(
                                    "total_frames_processed", 0
                                ),
                                "logo_totals": summary.get("logo_totals", {}),
                                "total_detections": summary.get("total_detections", 0),
                                "realtime_csv_files": summary.get("realtime_csv_files", {}),
                            }
                            yield f"data: {json.dumps(summary_data)}\n\n"

                        processed_count += 1
                        session.processed_frames = processed_count
                        session.save()

                frame_count += 1

            # Finalize CSV
            if csv_file_path:
                csv_file_path.close()

            # Finalize session
            if self.counting_service:
                self.counting_service.finalize_session(session_id)
                # Send final summary on completion
                summary = self.get_session_summary(session_id)
                summary_data = {
                    "type": "summary",
                    "session_id": summary.get("session_id"),
                    "total_frames_processed": summary.get("total_frames_processed", 0),
                    "logo_totals": summary.get("logo_totals", {}),
                    "total_detections": summary.get("total_detections", 0),
                    "realtime_csv_files": summary.get("realtime_csv_files", {}),
                }
                yield f"data: {json.dumps(summary_data)}\n\n"

            # Send completion
            processed_video_url = (
                f"/api/v1/static/{Path(video_path).name}" if create_video else None
            )
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Video processing completed', 'total_frames': processed_count, 'processed_video_url': processed_video_url})}\n\n"

            session.mark_completed()

            # Create video in background if requested
            if create_video:
                self._create_video_background(
                    video_path,
                    session,
                    detection_results,
                    video_fps,
                    frame_prefix,
                    temp_frames_dir,
                )

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            session.mark_failed()
            yield f"data: {json.dumps({'type': 'error', 'message': f'Error processing video: {str(e)}'})}\n\n"
        finally:
            if cap:
                cap.release()

    def _initialize_realtime_csv(self, session_id: str) -> tuple:
        """Initialize real-time CSV file for session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{timestamp}_{session_id[:8]}"
        filename = f"detection_report_{unique_id}.csv"
        csv_path = self.csv_dir / filename

        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        # Write creation timestamp
        creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow(["Created", creation_time])
        csv_writer.writerow([])  # Empty row for spacing
        csv_writer.writerow(
            [
                "Brand",
                "Frame Number",
                "Timestamp",
                "Confidence",
                "Bounding Box",
                "Classification",
            ]
        )

        return csv_file, csv_writer

    def _write_to_realtime_csv(
        self,
        csv_writer,
        csv_file,
        detection: Detection,
        frame_number: int,
        timestamp: float,
        classification: Optional[List] = None,
    ):
        """Write detection to real-time CSV"""
        box_str = f"[{detection.bbox_x1:.1f},{detection.bbox_y1:.1f},{detection.bbox_x2:.1f},{detection.bbox_y2:.1f}]"
        confidence_str = f"{detection.confidence:.3f}"

        if classification and len(classification) > 0:
            top_class = classification[0]
            classification_str = (
                f"{top_class['class_name']} ({top_class['confidence']:.2%})"
            )
            if len(classification) > 1:
                classification_str += f" | {classification[1]['class_name']} ({classification[1]['confidence']:.2%})"
        else:
            classification_str = "N/A"

        csv_writer.writerow(
            [
                detection.class_name,
                frame_number,
                f"{timestamp:.2f}",
                confidence_str,
                box_str,
                classification_str,
            ]
        )
        # Flush immediately to disk for real-time access
        csv_file.flush()

    def _create_video_background(
        self,
        video_path: str,
        session: ProcessingSession,
        detection_results: dict,
        fps: int,
        frame_prefix: str,
        temp_frames_dir: Path,
    ):
        """Create video in background using FFmpeg"""
        import threading

        def create_video():
            try:
                cap = cv2.VideoCapture(video_path)
                frame_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Find nearest processed frame
                    nearest_frame = self._find_nearest_processed_frame(
                        frame_count, detection_results.keys()
                    )

                    if nearest_frame in detection_results:
                        detections, annotated_frame = detection_results[nearest_frame]
                        if detections:
                            annotated_frame = self._apply_detections_to_frame(
                                frame, detections
                            )
                    else:
                        annotated_frame = frame

                    # Save to temp directory
                    frame_filename = f"frame_{frame_prefix}_{frame_count:06d}.jpg"
                    temp_frame_path = temp_frames_dir / frame_filename
                    cv2.imwrite(str(temp_frame_path), annotated_frame)
                    frame_count += 1

                cap.release()

                # Create video with FFmpeg
                processed_video_path = (
                    Path(self.config.static_dir)
                    / f"processed_{int(time.time())}_{Path(video_path).name}"
                )
                self._create_video_from_frames(
                    temp_frames_dir,
                    processed_video_path,
                    fps,
                    frame_count,
                    frame_prefix,
                )

                session.processed_video_path = str(processed_video_path)
                session.save()

                # Cleanup
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
                if video_path and os.path.exists(video_path):
                    os.unlink(video_path)

            except Exception as e:
                print(f"[VIDEO CREATION ERROR] {str(e)}")

        thread = threading.Thread(target=create_video, daemon=True)
        thread.start()

    def _find_nearest_processed_frame(
        self, current_frame: int, processed_frames: list
    ) -> int:
        """Find the nearest processed frame"""
        if not processed_frames:
            return current_frame

        processed_frames = sorted(processed_frames)
        nearest = processed_frames[0]
        min_distance = abs(current_frame - nearest)

        for frame in processed_frames:
            distance = abs(current_frame - frame)
            if distance < min_distance:
                min_distance = distance
                nearest = frame

        return nearest

    def _apply_detections_to_frame(
        self, frame: np.ndarray, detections: List[DetectionResult]
    ) -> np.ndarray:
        """Apply detections to a frame by drawing bounding boxes"""
        annotated_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{detection.class_name} {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

        return annotated_frame

    def _create_video_from_frames(
        self,
        frames_dir: Path,
        output_path: Path,
        fps: int,
        total_frames: int,
        frame_prefix: str,
    ):
        """Create MP4 video from frames using FFmpeg"""
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frames_dir / f"frame_{frame_prefix}_%06d.jpg"),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]

            process = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
            )

            print(f"Successfully created processed video: {output_path}")

        except FileNotFoundError:
            raise Exception(
                "FFmpeg not found. Please install FFmpeg to process videos."
            )
        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg failed to create video: {e.stderr.decode()}")

    def get_session_summary(self, session_id: str) -> dict:
        """Get summary of detection session"""
        # Always get the session first
        try:
            session = ProcessingSession.objects.get(session_id=session_id)
        except ProcessingSession.DoesNotExist:
            return {
                "session_id": session_id,
                "total_frames_processed": 0,
                "logo_totals": {},
                "total_detections": 0,
                "unique_logos": [],
                "realtime_csv_files": {},
            }

        # If counting service exists, use it but still add realtime CSV files
        if self.counting_service:
            summary = self.counting_service.get_session_summary(session_id)
            # Ensure realtime CSV files are included
            if "realtime_csv_files" not in summary:
                realtime_csv_files = self.get_realtime_csv_files(session_id)
                summary["realtime_csv_files"] = realtime_csv_files
            return summary

        # Otherwise, calculate summary manually
        from django.db.models import Count

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
                Detection.objects.filter(session=session).values_list(
                    "class_name", flat=True
                )
            )
        )

        summary = {
            "session_id": session_id,
            "total_frames_processed": session.processed_frames,
            "logo_totals": logo_counts_dict,
            "total_detections": sum(logo_counts_dict.values()),
            "unique_logos": unique_logos,
        }

        # Add realtime CSV files to summary
        realtime_csv_files = self.get_realtime_csv_files(session_id)
        summary["realtime_csv_files"] = realtime_csv_files

        return summary

    def get_realtime_csv_files(self, session_id: str) -> dict:
        """Get real-time CSV files for a session"""
        csv_files = []
        for csv_file in self.csv_dir.glob(f"*{session_id[:8]}*.csv"):
            csv_files.append({"main": f"/api/v1/static/csv_reports/{csv_file.name}"})
        return csv_files[0] if csv_files else {}

    def export_session_to_csv(
        self, session_id: str, filename_prefix: str = None
    ) -> dict:
        """Export session data to CSV from database"""
        try:
            session = ProcessingSession.objects.get(session_id=session_id)
        except ProcessingSession.DoesNotExist:
            return {}

        if not filename_prefix:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = f"{timestamp}_{session_id[:8]}"
            filename_prefix = f"detection_export_{unique_id}"

        csv_path = self.csv_dir / f"{filename_prefix}.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Brand",
                    "Frame Number",
                    "Timestamp",
                    "Count in Frame",
                    "Confidences",
                    "Bounding Boxes",
                ]
            )

            # Get all frames for session
            frames = Frame.objects.filter(session=session).order_by("frame_number")

            for frame in frames:
                # Group detections by brand
                from collections import defaultdict

                brand_to_boxes = defaultdict(list)
                brand_to_confidences = defaultdict(list)

                detections = Detection.objects.filter(frame=frame)
                for detection in detections:
                    brand_name = detection.class_name
                    bbox = f"[{detection.bbox_x1:.1f},{detection.bbox_y1:.1f},{detection.bbox_x2:.1f},{detection.bbox_y2:.1f}]"
                    brand_to_boxes[brand_name].append(bbox)
                    brand_to_confidences[brand_name].append(
                        f"{detection.confidence:.3f}"
                    )

                for brand, boxes in brand_to_boxes.items():
                    count_in_frame = len(boxes)
                    boxes_str = ", ".join(boxes)
                    confidences_str = ", ".join(brand_to_confidences[brand])
                    writer.writerow(
                        [
                            brand,
                            frame.frame_number,
                            f"{frame.timestamp:.2f}",
                            count_in_frame,
                            confidences_str,
                            boxes_str,
                        ]
                    )

        return {"main": f"/api/v1/static/csv_reports/{csv_path.name}"}

    def get_available_csv_files(self) -> list:
        """Get list of available CSV files"""
        csv_files = []
        for csv_file in self.csv_dir.glob("*.csv"):
            csv_files.append(
                {
                    "filename": csv_file.name,
                    "path": f"/api/v1/static/csv_reports/{csv_file.name}",
                    "size": csv_file.stat().st_size,
                    "created": datetime.fromtimestamp(
                        csv_file.stat().st_ctime
                    ).isoformat(),
                }
            )
        return sorted(csv_files, key=lambda x: x["created"], reverse=True)

    def cleanup_old_csv_files(self, max_files: int = 50):
        """Clean up old CSV files, keeping only the most recent ones"""
        csv_files = sorted(
            self.csv_dir.glob("*.csv"), key=lambda x: x.stat().st_ctime, reverse=True
        )

        if len(csv_files) > max_files:
            for old_file in csv_files[max_files:]:
                try:
                    old_file.unlink()
                except Exception as e:
                    print(f"Error deleting old CSV file {old_file}: {e}")
