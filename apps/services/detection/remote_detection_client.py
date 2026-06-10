"""
HTTP client for the external detector service (detection + classification).

POSTs an image URL to {DETECTOR_HOST}/detect; the service fetches the image from
that URL (DigitalOcean Spaces), runs YOLO detection + per-box classification on
its GPU, and returns both combined. No image bytes cross this wire — only the
URL (same contract as the GLM OCR client).

Enabled by setting DETECTOR_HOST. When unset, the detection path falls back to
the in-process YOLO (ModelService) so nothing breaks without the service.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Optional

import requests

logger = logging.getLogger("apps.detection")

_RETRY_STATUSES: frozenset[int] = frozenset({429, 502, 503, 504})


class RemoteDetectionError(RuntimeError):
    """Detector service unreachable or failing after retries."""


class RemoteDetectionClient:
    def __init__(
        self,
        host: str,
        api_key: str = "",
        timeout_seconds: float = 60.0,
        retries: int = 3,
        retry_base_delay: float = 1.0,
    ):
        self.host = host.rstrip("/")
        self.api_key = api_key or ""
        self.timeout_seconds = timeout_seconds
        self.retries = max(1, retries)
        self.retry_base_delay = max(0.1, retry_base_delay)

    def _headers(self) -> dict:
        h = {"content-type": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def detect(
        self,
        image_url: str,
        weight_name: Optional[str] = None,
        classification_weight_name: Optional[str] = None,
        confidence: Optional[float] = None,
        classify: bool = True,
        top_k: Optional[int] = None,
        crop_padding: Optional[int] = None,
    ) -> list[dict]:
        """Detect + classify the image at `image_url`.

        Returns the service's `detections` list verbatim — each item already in
        the shape we persist: {bbox:[x1,y1,x2,y2], confidence, class_id,
        class_name, classification:[{class_id,class_name,confidence,rank}]|None}.
        Raises RemoteDetectionError on hard failure (after retries).
        """
        body: dict[str, Any] = {"image_url": image_url, "classify": classify}
        if weight_name:
            body["weight_name"] = weight_name
        if classification_weight_name:
            body["classification_weight_name"] = classification_weight_name
        if confidence is not None:
            body["confidence"] = confidence
        if top_k is not None:
            body["top_k"] = top_k
        if crop_padding is not None:
            body["crop_padding"] = crop_padding
        data = self._post("/detect", body)
        return data.get("detections", []) or []

    def list_detection_weights(self) -> dict:
        return self._get("/weights")

    def list_classification_weights(self) -> dict:
        return self._get("/classification/weights")

    def health(self) -> dict:
        return self._get("/health")

    # --- transport ----------------------------------------------------------
    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.host}{path}"
        last = "unknown error"
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(
                    url, json=body, headers=self._headers(), timeout=self.timeout_seconds
                )
            except requests.RequestException as e:
                last = f"request failed: {e}"
                if attempt < self.retries:
                    self._backoff(attempt)
                    continue
                raise RemoteDetectionError(f"detector {url}: {last}") from e

            if resp.status_code in _RETRY_STATUSES and attempt < self.retries:
                last = f"HTTP {resp.status_code}"
                self._backoff(attempt)
                continue
            if resp.status_code != 200:
                # 4xx (bad weight / bad image) won't fix on retry — surface now.
                raise RemoteDetectionError(
                    f"detector {url} HTTP {resp.status_code}: {resp.text[:200]}"
                )
            try:
                return resp.json()
            except ValueError as e:
                raise RemoteDetectionError(f"detector {url}: non-JSON response") from e
        raise RemoteDetectionError(f"detector {url}: {last}")

    def _get(self, path: str) -> dict:
        url = f"{self.host}{path}"
        resp = requests.get(url, headers=self._headers(), timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def _backoff(self, attempt: int) -> None:
        delay = self.retry_base_delay * (2 ** (attempt - 1))
        delay *= 1.0 + random.uniform(0, 0.3)
        time.sleep(min(delay, 30.0))
