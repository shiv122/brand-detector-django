"""
HTTP client for the external LocateAnything service.

POSTs an image URL (+ task/query) to {LOCATE_HOST}/locate; the service fetches
the image from that URL (DigitalOcean Spaces) and returns detected bounding
boxes / points with coordinates in the ORIGINAL frame's pixel space. No image
bytes cross this wire — only the URL. Mirrors GlmOcrClient.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

_RETRY_STATUSES: frozenset[int] = frozenset({429, 502, 503, 504})


def _retry_attempts() -> int:
    try:
        return max(1, int(os.environ.get("LOCATE_RETRIES", "3")))
    except ValueError:
        return 3


def _retry_base_delay() -> float:
    try:
        return max(0.1, float(os.environ.get("LOCATE_RETRY_BASE_DELAY", "1.0")))
    except ValueError:
        return 1.0


class LocateAnythingClient:
    def __init__(self, host: str, timeout_seconds: float = 180.0):
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._last_error: Optional[str] = None

    def load_error(self) -> Optional[str]:
        return self._last_error

    def locate(
        self,
        image_url: str,
        task: str = "detection",
        query: Optional[str] = None,
        prompt: Optional[str] = None,
        mode: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[Optional[dict], dict]:
        """Return (result, timing).

        `result` is the service JSON (boxes / points / image / text) or None on
        failure (see load_error()).
        """
        self._last_error = None
        url = f"{self.host}/locate"
        body: dict[str, Any] = {"image_url": image_url, "task": task}
        if query:
            body["query"] = query
        if prompt:
            body["prompt"] = prompt
        if mode:
            body["mode"] = mode
        if max_tokens:
            body["max_tokens"] = max_tokens

        attempts = _retry_attempts()
        base_delay = _retry_base_delay()
        for attempt in range(1, attempts + 1):
            t0 = time.monotonic()
            try:
                resp = requests.post(url, json=body, timeout=self.timeout_seconds)
            except requests.RequestException as e:
                self._last_error = f"LocateAnything request failed: {e}"
                if attempt < attempts:
                    self._backoff(base_delay, attempt)
                    continue
                return None, {}

            net_ms = int((time.monotonic() - t0) * 1000)

            if resp.status_code in _RETRY_STATUSES and attempt < attempts:
                self._backoff(base_delay, attempt)
                continue

            if resp.status_code != 200:
                self._last_error = (
                    f"LocateAnything HTTP {resp.status_code}: {resp.text[:200]}"
                )
                return None, {"network_ms": net_ms}

            try:
                data = resp.json()
            except ValueError:
                self._last_error = "LocateAnything returned non-JSON response"
                return None, {"network_ms": net_ms}

            if isinstance(data, dict) and data.get("error"):
                self._last_error = str(data["error"])
                return None, {"network_ms": net_ms}

            timing = dict(data.get("timing_ms") or {})
            timing["network_ms"] = net_ms
            return data, timing

        return None, {}

    @staticmethod
    def _backoff(base_delay: float, attempt: int) -> None:
        delay = base_delay * (2 ** (attempt - 1))
        delay *= 1.0 + random.uniform(0, 0.3)
        time.sleep(min(delay, 30.0))
