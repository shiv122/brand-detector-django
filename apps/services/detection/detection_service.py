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
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Generator
from datetime import datetime
from django.conf import settings as django_settings
from django.db import close_old_connections
from django.utils import timezone
from django.http import StreamingHttpResponse, FileResponse
from rest_framework.response import Response
from rest_framework import status
from config.app_config import AppConfig
from apps.services.model.model_service import ModelService, DetectionResult
from apps.services.image.image_service import ImageService
from apps.services.classification.classification_service import ClassificationService
from apps.services.counting.counting_service import CountingService
from apps.core.models import (
    ProcessingSession,
    Frame,
    Detection,
    Classification,
    SportPrompt,
)
from apps.core.enums import ProcessingStatus, RESUMABLE_STATUSES, ACTIVE_STATUSES
from apps.utils.file_helpers import ensure_directory_exists
from apps.utils.video_helpers import (
    probe_video,
    FfmpegFrameReader,
    VideoProbeError,
    FrameReadError,
)
from apps.services.ocr.ocr_queue import enqueue_ocr_job
from apps.utils.prompt_render import render_prompt

import ipaddress
import socket
from urllib.parse import urlparse, urljoin

import requests

logger = logging.getLogger("apps.detection")


def _sse(payload: dict) -> str:
    """Serialize a dict as one SSE `data:` event."""
    return f"data: {json.dumps(payload)}\n\n"


# Cap on a downloaded video (file_url flow). Guards against a disk-fill DoS from
# an unbounded URL. Default 20 GB so full-length broadcasts aren't rejected;
# override with MAX_VIDEO_DOWNLOAD_BYTES. Ensure the static volume has the disk.
MAX_VIDEO_DOWNLOAD_BYTES = int(
    os.getenv("MAX_VIDEO_DOWNLOAD_BYTES", str(20 * 1024 * 1024 * 1024))
)
_MAX_DOWNLOAD_REDIRECTS = 5


def _host_resolves_to_blocked_ip(host: str) -> bool:
    """True if any address `host` resolves to is private/loopback/link-local/
    reserved/multicast — i.e. an SSRF target inside our own network."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True  # can't resolve -> treat as unsafe
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return True
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        ):
            return True
    return False


def assert_safe_public_url(url: str) -> None:
    """Raise ValueError unless `url` is an http(s) URL whose host resolves only
    to public addresses. Guards the user-supplied file_url against SSRF
    (cloud metadata, loopback, internal services)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http(s) URLs are allowed")
    if not parsed.hostname:
        raise ValueError("URL has no host")
    if _host_resolves_to_blocked_ip(parsed.hostname):
        raise ValueError("URL resolves to a disallowed (internal) address")


class DetectionService:
    """Service for logo detection"""

    def __init__(
        self,
        config: AppConfig,
        model_service: ModelService,
        image_service: ImageService,
        classification_service: Optional[ClassificationService] = None,
        counting_service: Optional[CountingService] = None,
        spaces_service=None,
        ocr_service=None,
    ):
        self.config = config
        self.model_service = model_service
        self.image_service = image_service
        self.classification_service = classification_service
        self.counting_service = counting_service
        self.spaces_service = spaces_service
        self.ocr_service = ocr_service
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
        self, image_data: bytes, confidence_threshold: float = 0.5,
        weight_name: Optional[str] = None
    ) -> Tuple[List[DetectionResult], Optional[np.ndarray]]:
        """Detect logos in a single image"""
        return self.model_service.detect_in_image(image_data, confidence_threshold, weight_name=weight_name)

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
        self, frame: np.ndarray, detection: DetectionResult,
        classification_weight_name: Optional[str] = None
    ) -> Optional[List]:
        """Classify a detection by cropping and running classification model"""
        if not self.classification_service:
            return None
        if not classification_weight_name and not self.classification_service.is_model_loaded():
            return None

        try:
            cropped = self._crop_detection_box(frame, detection.bbox, padding=40)

            if cropped.size == 0:
                return None

            _, buffer = cv2.imencode(".jpg", cropped)
            image_bytes = buffer.tobytes()

            classification_results = self.classification_service.classify_image(
                image_bytes, top_k=3, weight_name=classification_weight_name
            )
            return [r.to_dict() for r in classification_results]
        except Exception as e:
            print(f"[CLASSIFICATION] Error classifying detection: {str(e)}")
            return None

    def _store_frame_image(
        self, frame_filename: str, jpg_bytes: bytes, annotated_frame: np.ndarray
    ) -> Tuple[str, str, str]:
        """Persist an annotated frame and return (frame_url, s3_key, local_path).

        Uploads to DigitalOcean Spaces when configured (frame_url = public S3
        URL, local_path = ""); on any failure — or when Spaces is unconfigured —
        falls back to a local /static file so detection never breaks.
        """
        if self.spaces_service and self.spaces_service.is_configured() and jpg_bytes:
            key = f"{self.config.spaces_frames_prefix}/{frame_filename}"
            try:
                url = self.spaces_service.upload_bytes(
                    key, jpg_bytes, content_type="image/jpeg"
                )
                return url, key, ""
            except Exception as e:  # noqa: BLE001 - fall back to local on any error
                print(f"[SPACES] upload failed for {key}: {e}; using local /static")

        frame_path = Path(self.config.frames_dir) / frame_filename
        if jpg_bytes:
            frame_path.write_bytes(jpg_bytes)
        else:
            cv2.imwrite(str(frame_path), annotated_frame)
        return f"/static/frames/{frame_filename}", "", str(frame_path)

    def _resolve_ocr_prompt(
        self, source: Optional[dict]
    ) -> Optional[Tuple[str, dict]]:
        """Resolve an OCR prompt + metadata from the validated request data.

        Returns (prompt_text, prompt_meta) or None when no OCR is requested
        / no matching prompt can be found. Priority: inline `prompt` >
        `prompt_slug` > `sport`.
        """
        if not source:
            return None
        inline = (source.get("prompt") or "").strip()
        slug = (source.get("prompt_slug") or "").strip()
        sport = (source.get("sport") or "").strip()

        if inline:
            return render_prompt(inline, []), {"source": "inline"}

        if slug:
            sp = SportPrompt.objects.filter(slug=slug).first()
            if sp is not None:
                brands = list(sp.allowed_brands or [])
                meta = {
                    "source": "stored",
                    "slug": sp.slug,
                    "name": sp.name,
                    "sport": sp.sport,
                }
                if brands:
                    meta["allowed_brands"] = brands
                return render_prompt(sp.prompt, brands), meta

        if sport:
            sport_lower = sport.lower()
            sp = (
                SportPrompt.objects.filter(slug=sport_lower).first()
                or SportPrompt.objects.filter(sport__iexact=sport_lower)
                .order_by("-updated_at")
                .first()
            )
            if sp is not None:
                brands = list(sp.allowed_brands or [])
                meta = {
                    "source": "sport",
                    "slug": sp.slug,
                    "name": sp.name,
                    "sport": sp.sport,
                }
                if brands:
                    meta["allowed_brands"] = brands
                return render_prompt(sp.prompt, brands), meta
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
        weight_name = None
        enable_classification = False
        classification_weight_name = None
        if validated_data is not None:
            confidence_threshold = float(
                validated_data.get("confidence_threshold", 0.5)
            )
            weight_name = validated_data.get("weight_name")
            enable_classification = bool(
                validated_data.get("enable_classification", False)
            )
            classification_weight_name = validated_data.get(
                "classification_weight_name"
            )
        else:
            confidence_threshold = float(request.data.get("confidence_threshold", 0.5))
            weight_name = request.data.get("weight_name")
            enable_classification = bool(
                request.data.get("enable_classification", False)
            )
            classification_weight_name = request.data.get("classification_weight_name")

        if not weight_name and not self.is_model_loaded():
            return Response(
                {"error": "Model not loaded"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        files = request.FILES.getlist("files")

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
                    contents, confidence_threshold, weight_name=weight_name
                )

                annotated_image_b64 = None
                if annotated_image is not None:
                    annotated_image_b64 = self.image_service.image_to_base64(
                        annotated_image
                    )

                detection_dicts = [d.to_dict() for d in detections]

                if enable_classification and detections:
                    frame = cv2.imdecode(
                        np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is not None:
                        for det, det_dict in zip(detections, detection_dicts):
                            classification_data = self._classify_detection(
                                frame, det, classification_weight_name
                            )
                            if classification_data:
                                det_dict["classification"] = classification_data

                result_dict: dict = {
                    "detections": detection_dicts,
                    "total_detections": len(detections),
                    "annotated_image": annotated_image_b64,
                }

                results.append(result_dict)
            except Exception as e:
                results.append(
                    {
                        "detections": [],
                        "total_detections": 0,
                        "error": str(e),
                    }
                )

        return Response({"results": results})

    def enqueue_video_handler(self, request, validated_data: dict = None) -> Response:
        """Create a session, persist everything the worker needs, enqueue the
        background detection job, and return immediately (202).

        NO frame processing happens on this request. The browser then
        subscribes to ``GET /video/detect/{session_id}/events`` for progress.
        Because the RQ worker owns the job, closing the browser, a proxy idle
        timeout, or a redeploy can no longer stop detection — the job keeps
        running and the reaper resumes it if its worker ever dies.
        """
        from apps.services.detection.detection_job import enqueue_detection_job

        source = validated_data if validated_data is not None else request.data
        file_url = source.get("file_url")
        frames_per_second = int(source.get("frames_per_second", 2))
        confidence_threshold = float(source.get("confidence_threshold", 0.5))
        create_video = bool(source.get("create_video", False))
        enable_classification = bool(source.get("enable_classification", False))
        weight_name = source.get("weight_name")
        classification_weight_name = source.get("classification_weight_name")
        enable_ocr = bool(source.get("enable_ocr", False))

        # Resolve the source into something the worker can re-read at will
        # (a presigned Spaces URL, or a persisted local upload path).
        file = request.FILES.get("file")
        try:
            if file is not None:
                source_url = self._persist_uploaded_file(file)
                source_is_local = True
                filename = file.name
            elif file_url:
                source_url, source_is_local = file_url, False
                filename = file_url.split("/")[-1].split("?")[0] or "video.mp4"
            else:
                return Response(
                    {"error": "Either file or file_url must be provided"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to accept detection source")
            return Response(
                {"error": f"Failed to accept video: {exc}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Resolve the OCR prompt once and STORE it, so the worker never needs
        # the originating request. None means OCR stays off.
        ocr_resolution = self._resolve_ocr_prompt(source) if enable_ocr else None

        run_params = {
            "frames_per_second": frames_per_second,
            "confidence_threshold": confidence_threshold,
            "create_video": create_video,
            "enable_classification": enable_classification,
            "weight_name": weight_name,
            "classification_weight_name": classification_weight_name,
            "enable_ocr": bool(ocr_resolution),
        }
        if ocr_resolution:
            run_params["ocr_prompt"], run_params["ocr_meta"] = ocr_resolution

        session_id = f"video_{int(time.time())}_{secrets.token_hex(5)}"
        session = ProcessingSession.objects.create(
            session_id=session_id,
            video_filename=filename,
            frames_per_second=frames_per_second,
            confidence_threshold=confidence_threshold,
            status=ProcessingStatus.QUEUED.value,
            settings={
                "enable_classification": enable_classification,
                "enable_ocr": bool(ocr_resolution),
                "create_video": create_video,
                "model_weight": weight_name or self.get_current_weight(),
            },
            source_url=source_url,
            source_is_local=source_is_local,
            run_params=run_params,
        )

        job_id = enqueue_detection_job(session_id, start_frame=0)
        if job_id is None:
            session.mark_failed(
                "Could not enqueue detection job (queue unavailable)",
                stop_reason="enqueue_failed",
            )
            logger.error("Enqueue failed for session %s", session_id)
            return Response(
                {"error": "Detection queue is unavailable; please retry."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        logger.info(
            "Queued detection session=%s fps=%s conf=%s source=%s",
            session_id, frames_per_second, confidence_threshold,
            "local-upload" if source_is_local else "url",
        )
        return Response(
            {
                "session_id": session_id,
                "status": session.status,
                "events_url": f"/api/v1/video/detect/{session_id}/events",
            },
            status=status.HTTP_202_ACCEPTED,
        )

    # ------------------------------------------------------------------ #
    # Source handling — turn the request's file/url into something the     #
    # background worker can (re-)read at will, including on resume.         #
    # ------------------------------------------------------------------ #
    def _persist_uploaded_file(self, file) -> str:
        """Save a direct upload under static_dir/uploads and return its path.

        Unlike the old flow (which streamed the upload into a temp file tied to
        the request and deleted it immediately), this persists the file so the
        background worker — and any resume — can read it after the request that
        created it is long gone. Cleaned up when the session completes.
        """
        uploads_dir = Path(self.config.static_dir) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        safe_ext = (Path(file.name).suffix or ".mp4").lower()
        dest = uploads_dir / f"{int(time.time())}_{secrets.token_hex(6)}{safe_ext}"
        with open(dest, "wb") as f:
            for chunk in file.chunks():
                f.write(chunk)
        return str(dest)

    def _download_to_temp(self, file_url: str) -> str:
        """Download a (public, SSRF-validated) URL to a temp file; return path.

        Reused by the worker on every (re)attempt. Raises on any failure so the
        caller can mark the session interrupted/failed with a real reason —
        never silently. SSRF-safe: validates the host on every redirect hop.
        """
        if not file_url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL format")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        current_url = file_url
        response = None
        for _ in range(_MAX_DOWNLOAD_REDIRECTS + 1):
            assert_safe_public_url(current_url)
            response = requests.get(
                current_url, stream=True, timeout=300, headers=headers,
                allow_redirects=False,
            )
            if response.is_redirect or response.is_permanent_redirect:
                loc = response.headers.get("Location")
                response.close()
                if not loc:
                    raise ValueError("Redirect without Location header")
                current_url = urljoin(current_url, loc)
                continue
            break
        else:
            raise ValueError("Too many redirects")

        if response.status_code != 200:
            response.close()
            raise ValueError(f"download failed: HTTP {response.status_code}")

        declared = int(response.headers.get("content-length", 0) or 0)
        if declared and declared > MAX_VIDEO_DOWNLOAD_BYTES:
            response.close()
            raise ValueError("Video exceeds the maximum allowed size")

        filename = file_url.split("/")[-1].split("?")[0] or "video.mp4"
        dest = Path(self.config.static_dir) / f"dl_{int(time.time())}_{secrets.token_hex(6)}_{filename}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        try:
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > MAX_VIDEO_DOWNLOAD_BYTES:
                        raise ValueError("Video exceeds the maximum allowed size")
                    f.write(chunk)
        except Exception:
            try:
                os.unlink(dest)
            except OSError:
                pass
            raise
        finally:
            response.close()
        logger.info("Downloaded source %.1f MB -> %s", downloaded / (1024 * 1024), dest)
        return str(dest)

    def _resolve_source(self, session: ProcessingSession) -> Tuple[str, bool]:
        """Return (local_video_path, is_temp) for the worker to read.

        Local uploads are read in place (is_temp=False — kept until completion).
        URL sources are downloaded fresh to a temp file (is_temp=True — deleted
        after the run). Raises if the source can't be obtained.
        """
        if session.source_is_local:
            if not session.source_url or not os.path.exists(session.source_url):
                raise FileNotFoundError(
                    f"uploaded source missing: {session.source_url!r}"
                )
            return session.source_url, False
        if not session.source_url:
            raise ValueError("session has no source_url to process")
        return self._download_to_temp(session.source_url), True

    # ------------------------------------------------------------------ #
    # The worker core — runs INSIDE the RQ detection worker, NOT the       #
    # request. Independent of any socket, fully resumable, logs loudly.    #
    # ------------------------------------------------------------------ #
    def process_session(self, session_id: str, start_frame: int = 0) -> dict:
        """Run (or resume) detection for a session to completion.

        Robustness contract:
          * A single bad packet no longer ends the run — ffmpeg skips it. If
            ffmpeg aborts mid-stream we mark INTERRUPTED (resumable), never a
            false COMPLETED.
          * Memory is flat: one frame in flight at a time, nothing accumulated.
          * Idempotent: already-persisted frame numbers are skipped, so a
            resume can't duplicate rows.
          * Progress + a heartbeat are written to the DB continuously, so the
            SSE endpoint and the reaper always see the truth.
        """
        try:
            session = ProcessingSession.objects.get(session_id=session_id)
        except ProcessingSession.DoesNotExist:
            logger.error("process_session: unknown session %s", session_id)
            return {"status": "unknown", "session_id": session_id}

        # Respect a cancel that landed while this job was still queued (the
        # worker hadn't picked it up yet). Don't flip it back to PROCESSING.
        if (
            session.status == ProcessingStatus.INTERRUPTED.value
            and session.stop_reason == "cancelled"
        ):
            logger.info("process_session: session %s cancelled before start", session_id)
            return {"status": "cancelled", "session_id": session_id}

        params = session.run_params or {}
        fps = int(session.frames_per_second or params.get("frames_per_second") or 1)
        conf = float(session.confidence_threshold)
        weight_name = params.get("weight_name") or self.config.selected_weight
        classification_weight_name = params.get("classification_weight_name")
        enable_classification = bool(params.get("enable_classification"))
        create_video = bool(params.get("create_video"))
        ocr_resolution = None
        if params.get("enable_ocr") and params.get("ocr_prompt"):
            ocr_resolution = (params["ocr_prompt"], params.get("ocr_meta") or {})
        ocr_active = bool(
            ocr_resolution and self.ocr_service and self.ocr_service.is_available()
        )

        session.attempts = (session.attempts or 0) + 1
        session.save(update_fields=["attempts", "updated_at"])
        session.mark_processing()
        logger.info(
            "Detection START session=%s start_frame=%s attempt=%s fps=%s weight=%s",
            session_id, start_frame, session.attempts, fps, weight_name,
        )

        video_path = None
        is_temp = False
        csv_file = None
        reader = None
        processed = start_frame
        loop_error = ""
        temp_frames_dir = None
        cancelled = False
        try:
            video_path, is_temp = self._resolve_source(session)
            session.video_path = video_path
            session.save(update_fields=["video_path", "updated_at"])

            info = probe_video(video_path)
            width, height, duration = info["width"], info["height"], info["duration"]
            session.video_fps = info["fps"]
            estimated = int(round(duration * fps)) if duration > 0 else 0
            # Never shrink the denominator on resume.
            session.total_frames = max(estimated, session.total_frames or 0, start_frame)
            session.save(update_fields=["video_fps", "total_frames", "updated_at"])

            # Frame-number naming is stable across resumes (matters for the
            # optional create_video encode).
            frame_prefix = params.get("frame_prefix") or secrets.token_hex(8)
            if params.get("frame_prefix") != frame_prefix:
                params["frame_prefix"] = frame_prefix
                session.run_params = params
                session.save(update_fields=["run_params", "updated_at"])

            if create_video:
                temp_frames_dir = Path(self.config.static_dir) / "temp_frames" / session_id
                temp_frames_dir.mkdir(parents=True, exist_ok=True)

            csv_file, csv_writer = self._open_realtime_csv(
                session_id, append=start_frame > 0
            )

            # Idempotency window: frame numbers already persisted at/after the
            # resume point (a time-seek can overlap them). Loaded once.
            existing = set(
                Frame.objects.filter(
                    session=session, frame_number__gte=start_frame
                ).values_list("frame_number", flat=True)
            )

            reader = FfmpegFrameReader(
                video_path, fps, width, height, start_frame=start_frame
            )
            last_hb = time.monotonic()
            summary_every = 25
            for index, timestamp, frame in reader:
                if index in existing:
                    processed = max(processed, index + 1)
                    continue
                self._process_one_frame(
                    session=session,
                    frame=frame,
                    index=index,
                    timestamp=timestamp,
                    conf=conf,
                    weight_name=weight_name,
                    enable_classification=enable_classification,
                    classification_weight_name=classification_weight_name,
                    csv_writer=csv_writer,
                    csv_file=csv_file,
                    ocr_active=ocr_active,
                    ocr_resolution=ocr_resolution,
                    frame_prefix=frame_prefix,
                    create_video=create_video,
                    temp_frames_dir=temp_frames_dir,
                )
                processed = index + 1
                now = time.monotonic()
                if now - last_hb >= 2.0:
                    session.touch_heartbeat(processed_frames=processed)
                    last_hb = now
                    # Honor an external cancel. cancel_session flips status away
                    # from PROCESSING; touch_heartbeat above only writes
                    # heartbeat/processed (never status), so this read is truthful.
                    live_status = (
                        ProcessingSession.objects.filter(session_id=session_id)
                        .values_list("status", flat=True)
                        .first()
                    )
                    if live_status != ProcessingStatus.PROCESSING.value:
                        logger.info(
                            "Detection CANCELLED session=%s at frame=%s (status=%s)",
                            session_id, processed, live_status,
                        )
                        cancelled = True
                        break
                if processed % summary_every == 0 and self.counting_service:
                    self.counting_service.finalize_session(session_id)
        except (VideoProbeError, FrameReadError) as exc:
            loop_error = f"{type(exc).__name__}: {exc}"
            logger.error("Detection source/decode error session=%s: %s", session_id, loop_error)
        except Exception as exc:  # noqa: BLE001 - any per-frame failure is recoverable
            loop_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Detection loop crashed session=%s at frame=%s", session_id, processed)
        finally:
            if reader is not None:
                reader.close()
            if csv_file is not None:
                try:
                    csv_file.close()
                except Exception:  # noqa: BLE001
                    pass

        # Persist final progress + summary regardless of outcome.
        session.processed_frames = processed
        session.heartbeat_at = timezone.now()
        session.save(update_fields=["processed_frames", "heartbeat_at", "updated_at"])
        if self.counting_service:
            try:
                self.counting_service.finalize_session(session_id)
            except Exception:  # noqa: BLE001
                logger.exception("finalize_session failed for %s", session_id)

        if cancelled:
            # Session is already INTERRUPTED(cancelled) via cancel_session; just
            # leave it resumable and stop. Drop a temp download.
            if is_temp and video_path:
                self._safe_unlink(video_path)
            logger.info(
                "Detection STOPPED (cancelled) session=%s processed=%s",
                session_id, processed,
            )
            return {"status": "cancelled", "session_id": session_id,
                    "processed_frames": processed}

        clean_eof = bool(reader is not None and reader.clean_eof and not loop_error)

        # Decide outcome HONESTLY.
        if not clean_eof:
            reason = "read_error" if (reader is not None and not loop_error) else "exception"
            detail = loop_error or (
                f"ffmpeg aborted before EOF: {reader.stderr_tail[-300:]}"
                if reader is not None else "stream did not reach EOF"
            )
            return self._finish_recoverable(session, processed, reason, detail, is_temp, video_path)

        # Clean EOF -> truly done.
        if is_temp and video_path:
            self._safe_unlink(video_path)
        if session.source_is_local and session.source_url:
            self._safe_unlink(session.source_url)  # uploaded source no longer needed

        processed_video_url = None
        if create_video and start_frame == 0 and temp_frames_dir is not None:
            processed_video_url = self._encode_session_video(
                session, temp_frames_dir, frame_prefix, fps
            )
        elif create_video and start_frame > 0:
            logger.warning(
                "Session %s used create_video but was resumed; skipping video "
                "encode (frames + CSV are complete).", session_id,
            )
        if temp_frames_dir is not None:
            shutil.rmtree(temp_frames_dir, ignore_errors=True)

        session.mark_completed(stop_reason="eof")
        logger.info(
            "Detection COMPLETE session=%s processed=%s/%s",
            session_id, processed, session.total_frames,
        )
        return {
            "status": "completed",
            "session_id": session_id,
            "processed_frames": processed,
            "processed_video_url": processed_video_url,
        }

    def _finish_recoverable(
        self, session, processed, reason, detail, is_temp, video_path
    ) -> dict:
        """Mark a non-clean stop and auto-resume (bounded by attempts)."""
        from apps.services.detection.detection_job import enqueue_detection_job
        from django.conf import settings as _s

        # Drop a temp download; the resume re-fetches it. Keep local uploads.
        if is_temp and video_path:
            self._safe_unlink(video_path)

        max_attempts = int(getattr(_s, "DETECTION_MAX_ATTEMPTS", 5))
        if session.attempts < max_attempts:
            session.error_message = detail[:2000]
            session.stop_reason = reason
            session.status = ProcessingStatus.QUEUED.value
            session.save(
                update_fields=["error_message", "stop_reason", "status", "updated_at"]
            )
            enqueue_detection_job(session.session_id, start_frame=processed)
            logger.warning(
                "Detection INTERRUPTED session=%s (%s) — re-enqueued from frame %s "
                "(attempt %s/%s): %s",
                session.session_id, reason, processed, session.attempts,
                max_attempts, detail[:200],
            )
            return {"status": "requeued", "session_id": session.session_id,
                    "processed_frames": processed, "start_frame": processed}

        session.mark_failed(
            f"Gave up after {session.attempts} attempts. Last error: {detail}",
            stop_reason=reason,
        )
        logger.error(
            "Detection FAILED session=%s after %s attempts: %s",
            session.session_id, session.attempts, detail[:300],
        )
        return {"status": "failed", "session_id": session.session_id,
                "processed_frames": processed}

    def _process_one_frame(
        self, *, session, frame, index, timestamp, conf, weight_name,
        enable_classification, classification_weight_name, csv_writer, csv_file,
        ocr_active, ocr_resolution, frame_prefix, create_video, temp_frames_dir,
    ):
        """Detect + persist a single frame. Mirrors the old per-frame block but
        writes nothing to in-memory accumulators (flat memory).

        Ordering matters: the SSE progress feed runs in a DIFFERENT process and
        emits a frame the instant its row is visible, then never re-emits it. If
        we created the Frame and only attached its OCR job id afterwards, the
        feed would emit the frame before the handle existed and the UI would
        never see OCR. So we do the slow work (model + S3 uploads + OCR enqueue)
        FIRST, then commit the Frame + its ocr_job_id + detections in ONE atomic
        transaction — the feed only ever sees a fully-formed frame.
        """
        from django.db import IntegrityError, transaction

        detections, annotated_frame = self.model_service.detect_in_frame(
            frame, conf, weight_name=weight_name
        )
        if annotated_frame is None:
            annotated_frame = frame

        classification_data = []
        if enable_classification:
            for detection in detections:
                classification = self._classify_detection(
                    frame, detection, classification_weight_name
                )
                if classification:
                    classification_data.append(classification)

        # --- Slow, non-DB work first (kept OUT of the transaction) ---
        frame_filename = f"frame_{frame_prefix}_{index:06d}.jpg"
        ok, buf = cv2.imencode(".jpg", annotated_frame)
        jpg_bytes = buf.tobytes() if ok else b""
        frame_url, s3_key, local_path = self._store_frame_image(
            frame_filename, jpg_bytes, annotated_frame
        )

        # OCR uses the ORIGINAL (un-annotated) frame so the burned-in boxes/labels
        # aren't mis-read as text. Upload it now; enqueue happens in the txn below
        # (needs the frame id) so the job id commits together with the frame.
        ocr_url = None
        if ocr_active and ocr_resolution:
            ok_raw, raw_buf = cv2.imencode(".jpg", frame)
            raw_bytes = raw_buf.tobytes() if ok_raw else b""
            ocr_url, _, _ = self._store_frame_image(
                f"ocr_{frame_filename}", raw_bytes, frame
            )

        if create_video and temp_frames_dir is not None:
            cv2.imwrite(str(temp_frames_dir / frame_filename), annotated_frame)

        # --- Fast DB writes, atomic so the frame becomes visible all-at-once ---
        try:
            with transaction.atomic():
                db_frame = Frame.objects.create(
                    session=session,
                    frame_number=index,
                    frame_path=local_path,
                    frame_url=frame_url,
                    s3_key=s3_key,
                    timestamp=timestamp,
                    total_detections=len(detections),
                )

                # Enqueue OCR inside the txn so ocr_job_id is committed WITH the
                # frame — the SSE feed never sees a frame missing its handle.
                if ocr_active and ocr_resolution and ocr_url:
                    if ocr_url.startswith(("http://", "https://")):
                        prompt, meta = ocr_resolution
                        job_info = enqueue_ocr_job(
                            image_url=ocr_url, prompt=prompt, prompt_meta=meta,
                            frame_id=db_frame.id,
                        )
                        jid = (job_info or {}).get("id")
                        if jid:
                            db_frame.ocr_job_id = jid
                            db_frame.save(update_fields=["ocr_job_id"])
                    else:
                        logger.warning(
                            "OCR enabled but frame URL is not public (Spaces not "
                            "configured?) — skipping OCR for frame %s", index,
                        )

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
                    detection_classification = None
                    if enable_classification and idx < len(classification_data):
                        detection_classification = classification_data[idx]
                    if detection_classification:
                        for rank, cls in enumerate(detection_classification, 1):
                            Classification.objects.create(
                                detection=db_detection,
                                class_id=cls["class_id"],
                                class_name=cls["class_name"],
                                confidence=cls["confidence"],
                                rank=rank,
                            )
                    self._write_to_realtime_csv(
                        csv_writer, csv_file, db_detection, index, timestamp,
                        detection_classification,
                    )
        except IntegrityError:
            # Raced/overlapped on resume — this frame is already done.
            return

    def _open_realtime_csv(self, session_id: str, append: bool = False) -> tuple:
        """Open the realtime CSV for write (new) or append (resume)."""
        if append:
            existing = sorted(
                self.csv_dir.glob(f"*{session_id[:8]}*.csv"),
                key=lambda p: p.stat().st_mtime,
            )
            if existing:
                f = open(existing[-1], "a", newline="", encoding="utf-8")
                return f, csv.writer(f)
        return self._initialize_realtime_csv(session_id)

    def _encode_session_video(
        self, session, temp_frames_dir: Path, frame_prefix: str, fps: int
    ) -> Optional[str]:
        """Encode buffered annotated frames into an MP4 (create_video path)."""
        try:
            count = len(list(temp_frames_dir.glob(f"frame_{frame_prefix}_*.jpg")))
            if count == 0:
                return None
            out_name = f"processed_{int(time.time())}_{session.session_id}.mp4"
            out_path = Path(self.config.static_dir) / out_name
            self._create_video_from_frames(
                temp_frames_dir, out_path, fps, count, frame_prefix
            )
            session.processed_video_path = str(out_path)
            session.save(update_fields=["processed_video_path", "updated_at"])
            return f"/static/{out_name}"
        except Exception:  # noqa: BLE001
            logger.exception("create_video encode failed for %s", session.session_id)
            return None

    @staticmethod
    def _safe_unlink(path):
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError:
            logger.warning("Could not delete %s", path)

    # ------------------------------------------------------------------ #
    # Read-only progress feed. Disposable: a disconnect ends THIS         #
    # generator only; the worker keeps running. Reconnect with ?since=N.  #
    # ------------------------------------------------------------------ #
    def iter_progress_events(
        self, session_id: str, since: int = 0, poll_interval: float = 0.5
    ) -> Generator[str, None, None]:
        """SSE generator that tails session progress from the DB.

        Replays frames with frame_number > `since`, then live-tails until the
        session reaches a terminal state. Safe to disconnect and reconnect at
        any time (pass the last frame_number you saw as `since`).
        """
        try:
            session = ProcessingSession.objects.get(session_id=session_id)
        except ProcessingSession.DoesNotExist:
            yield _sse({"type": "error", "message": "Session not found"})
            return

        yield _sse({
            "type": "status",
            "message": "Subscribed",
            "estimated_total_frames": session.total_frames or 0,
            "processed_frames": session.processed_frames or 0,
            "session_id": session_id,
            "session_status": session.status,
        })

        last = since
        run_params = session.run_params or {}
        ocr_enabled = bool(run_params.get("enable_ocr"))
        # Frames we've streamed that don't have their (async) OCR result yet.
        pending_ocr: set = set()
        # OCR finishes AFTER detection, so once detection is done keep the feed
        # open briefly to deliver late results — bounded so a stuck/failed OCR
        # job can't hang the stream forever (~30s).
        drain_polls = int(30 / poll_interval) if poll_interval else 60
        try:
            while True:
                close_old_connections()
                session.refresh_from_db()

                new_frames = list(
                    Frame.objects.filter(session=session, frame_number__gt=last)
                    .order_by("frame_number")[:200]
                )
                for f in new_frames:
                    last = f.frame_number
                    yield _sse(self._frame_event(f))
                    if ocr_enabled and f.ocr_summary is None:
                        pending_ocr.add(f.frame_number)

                # OCR runs async on the `ocr` queue and writes its result back to
                # Frame.ocr_summary later. Re-emit each result the moment it lands
                # so the grid's OCR data + the "done/total" counter fill in live.
                if pending_ocr:
                    ready = Frame.objects.filter(
                        session=session, frame_number__in=list(pending_ocr)
                    ).exclude(ocr_summary__isnull=True)
                    for f in ready:
                        pending_ocr.discard(f.frame_number)
                        yield _sse({
                            "type": "ocr_result",
                            "frame_id": f.id,
                            "frame_number": f.frame_number,
                            "ocr_summary": f.ocr_summary,
                        })

                if new_frames:
                    summary = self.get_session_summary(session_id)
                    yield _sse({
                        "type": "summary",
                        "session_id": summary.get("session_id"),
                        "total_frames_processed": summary.get("total_frames_processed", 0),
                        "logo_totals": summary.get("logo_totals", {}),
                        "total_detections": summary.get("total_detections", 0),
                        "realtime_csv_files": summary.get("realtime_csv_files", {}),
                    })

                terminal = session.status in (
                    ProcessingStatus.COMPLETED.value,
                    ProcessingStatus.FAILED.value,
                    ProcessingStatus.INTERRUPTED.value,
                )
                more = Frame.objects.filter(
                    session=session, frame_number__gt=last
                ).exists()
                if terminal and not more:
                    # Detection is done, but keep draining late OCR results.
                    if ocr_enabled and pending_ocr and drain_polls > 0:
                        drain_polls -= 1
                        yield ": ocr-drain\n\n"
                        time.sleep(poll_interval)
                        continue
                    if session.status == ProcessingStatus.COMPLETED.value:
                        yield _sse({
                            "type": "complete",
                            "message": "Video processing completed",
                            "total_frames": session.processed_frames,
                            "processed_video_url": (
                                f"/static/{Path(session.processed_video_path).name}"
                                if session.processed_video_path else None
                            ),
                            "creating_video": False,
                        })
                    else:
                        # Interrupted/failed — tell the client it can resume.
                        yield _sse({
                            "type": "error",
                            "message": session.error_message
                            or f"Detection {session.status}",
                            "session_status": session.status,
                            "stop_reason": session.stop_reason,
                            "processed_frames": session.processed_frames,
                            "can_resume": session.status in RESUMABLE_STATUSES,
                        })
                    return

                if not new_frames:
                    # Comment line doubles as a proxy keep-alive.
                    yield ": keep-alive\n\n"
                    time.sleep(poll_interval)
        except GeneratorExit:
            # Client went away. The worker is unaffected — just stop tailing.
            return

    def _frame_event(self, f: Frame) -> dict:
        """Reconstruct a `frame` SSE payload from persisted rows (for replay)."""
        detections = []
        logo_counts: dict = {}
        for d in f.detections.all():
            entry = {
                "bbox": [d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2],
                "confidence": d.confidence,
                "class_id": d.class_id,
                "class_name": d.class_name,
            }
            cls = list(d.classifications.order_by("rank").values(
                "class_id", "class_name", "confidence"
            )) if hasattr(d, "classifications") else []
            if cls:
                entry["classification"] = cls
            detections.append(entry)
            logo_counts[d.class_name] = logo_counts.get(d.class_name, 0) + 1
        return {
            "type": "frame",
            "frame_id": f.id,
            "frame_number": f.frame_number,
            "frame_url": f.frame_url,
            "detections": detections,
            "total_detections": f.total_detections,
            "timestamp": f.timestamp,
            "logo_counts": logo_counts,
            "ocr_summary": f.ocr_summary,
            # Hand back the OCR job handle so the client can poll /ocr/jobs (the
            # results grid renders OCR from this). The async OCR task also
            # persists to ocr_summary above as the durable copy.
            "ocr_job": {"id": f.ocr_job_id, "status": "queued"} if f.ocr_job_id else None,
        }

    def resume_session(self, session_id: str) -> dict:
        """Re-enqueue a stopped session from where it left off.

        Used by the Resume button and as the building block of the reaper.
        Idempotent: a session already active is left alone.
        """
        from apps.services.detection.detection_job import enqueue_detection_job

        try:
            session = ProcessingSession.objects.get(session_id=session_id)
        except ProcessingSession.DoesNotExist:
            return {"ok": False, "error": "Session not found"}

        if session.status == ProcessingStatus.COMPLETED.value:
            return {"ok": True, "status": "completed", "message": "Already complete"}
        if session.status in (
            ProcessingStatus.QUEUED.value,
            ProcessingStatus.PROCESSING.value,
            ProcessingStatus.PENDING.value,
        ):
            return {"ok": True, "status": session.status, "message": "Already active"}
        if not session.source_url:
            return {"ok": False, "error": "No source to resume from"}

        start_frame = session.processed_frames or 0
        # A manual resume resets the attempt budget and clears the prior
        # stop reason (incl. a previous cancel) so the worker doesn't refuse it.
        session.attempts = 0
        session.status = ProcessingStatus.QUEUED.value
        session.stop_reason = ""
        session.error_message = ""
        session.save(
            update_fields=["attempts", "status", "stop_reason", "error_message", "updated_at"]
        )
        enqueue_detection_job(session_id, start_frame=start_frame)
        logger.info("Resume requested session=%s from frame=%s", session_id, start_frame)
        return {
            "ok": True,
            "status": "queued",
            "session_id": session_id,
            "start_frame": start_frame,
            "events_url": f"/api/v1/video/detect/{session_id}/events",
        }

    def cancel_session(self, session_id: str) -> dict:
        """Stop a running/queued session. The running worker notices the status
        flip within ~2s and stops; a still-queued job refuses to start. Marked
        INTERRUPTED(cancelled) so it stays resumable from processed_frames.
        """
        try:
            session = ProcessingSession.objects.get(session_id=session_id)
        except ProcessingSession.DoesNotExist:
            return {"ok": False, "error": "Session not found"}

        if session.status not in ACTIVE_STATUSES:
            return {"ok": True, "status": session.status, "message": "Not active"}

        session.mark_interrupted("Cancelled by user", stop_reason="cancelled")
        logger.info("Cancel requested session=%s at frame=%s", session_id, session.processed_frames)
        return {
            "ok": True,
            "status": session.status,
            "session_id": session_id,
            "processed_frames": session.processed_frames,
        }

    def get_queue_status(self) -> dict:
        """Queue depths + live worker counts (detection + ocr)."""
        from apps.services.detection.detection_job import queue_overview

        return queue_overview()

    def clear_queue(self, queue_name: str = "detection", cancel_sessions: bool = True) -> dict:
        """Drop everything pending on a queue. For the detection queue, also
        mark not-yet-started sessions (QUEUED/PENDING) as cancelled so they
        don't linger or get revived by the reaper. A running PROCESSING job is
        left alone — use Stop/cancel for that.
        """
        from apps.services.detection.detection_job import purge_queue

        try:
            removed = purge_queue(queue_name)
        except Exception as exc:  # noqa: BLE001 - Redis down / lookup error
            logger.exception("Failed to clear queue %s", queue_name)
            return {"ok": False, "error": str(exc), "queue": queue_name}

        cancelled = 0
        if queue_name == "detection" and cancel_sessions:
            cancelled = ProcessingSession.objects.filter(
                status__in=[
                    ProcessingStatus.QUEUED.value,
                    ProcessingStatus.PENDING.value,
                ]
            ).update(
                status=ProcessingStatus.INTERRUPTED.value,
                stop_reason="cleared",
                error_message="Queue cleared by user",
            )

        logger.info(
            "Cleared queue %s: removed=%s cancelled_sessions=%s",
            queue_name, removed.get("total"), cancelled,
        )
        return {
            "ok": True,
            "queue": queue_name,
            "removed": removed,
            "cancelled_sessions": cancelled,
        }

    def _initialize_realtime_csv(self, session_id: str) -> tuple:
        """Initialize real-time CSV file for session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{timestamp}_{session_id[:8]}"
        filename = f"detection_report_{unique_id}.csv"
        csv_path = self.csv_dir / filename

        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        creation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow(["Created", creation_time])
        csv_writer.writerow([])
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
        csv_file.flush()

    def _create_processed_video(
        self,
        video_path: str,
        session: ProcessingSession,
        detection_results: dict,
        fps: int,
        frame_prefix: str,
        temp_frames_dir: Path,
    ) -> Optional[str]:
        """Render the annotated output video with FFmpeg, synchronously.

        Returns the public `/static/...` URL on success, or None on failure.
        Runs inline in the SSE generator (NOT a background thread) so the
        caller can emit a `video_ready` event once the file actually exists —
        otherwise the client's "Creating video" indicator never clears.
        """
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
            processed_video_name = f"processed_{int(time.time())}_{Path(video_path).name}"
            processed_video_path = Path(self.config.static_dir) / processed_video_name
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

            return f"/static/{processed_video_name}"

        except Exception as e:
            print(f"[VIDEO CREATION ERROR] {str(e)}")
            return None

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
            csv_files.append({"main": f"/static/csv_reports/{csv_file.name}"})
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

        return {"main": f"/static/csv_reports/{csv_path.name}"}

    def get_available_csv_files(self) -> list:
        """Get list of available CSV files"""
        csv_files = []
        for csv_file in self.csv_dir.glob("*.csv"):
            csv_files.append(
                {
                    "filename": csv_file.name,
                    "path": f"/static/csv_reports/{csv_file.name}",
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
