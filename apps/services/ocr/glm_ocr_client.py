"""
HTTP client for the external GLM OCR service.

POSTs an image URL + prompt to {GLM_OCR_HOST}/ocr; the GLM service fetches the
image from that URL (DigitalOcean Spaces) and returns the extracted text. No
image bytes cross this wire — only the URL.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_RETRY_STATUSES: frozenset[int] = frozenset({429, 502, 503, 504})


def _retry_attempts() -> int:
    try:
        return max(1, int(os.environ.get("GLM_OCR_RETRIES", "3")))
    except ValueError:
        return 3


def _retry_base_delay() -> float:
    try:
        return max(0.1, float(os.environ.get("GLM_OCR_RETRY_BASE_DELAY", "1.0")))
    except ValueError:
        return 1.0


class GlmOcrClient:
    def __init__(
        self,
        host: str,
        model: str,
        max_new_tokens: int = 2048,
        timeout_seconds: float = 120.0,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self._last_error: Optional[str] = None

    def load_error(self) -> Optional[str]:
        return self._last_error

    def extract_text(self, image_url: str, prompt: str) -> tuple[Optional[str], dict]:
        """Return (text, timing). text is None on failure (see load_error())."""
        self._last_error = None
        url = f"{self.host}/ocr"
        body = {
            "image_url": image_url,
            "prompt": prompt,
            "model": self.model,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
        }

        attempts = _retry_attempts()
        base_delay = _retry_base_delay()
        for attempt in range(1, attempts + 1):
            t0 = time.monotonic()
            try:
                resp = requests.post(url, json=body, timeout=self.timeout_seconds)
            except requests.RequestException as e:
                self._last_error = f"GLM OCR request failed: {e}"
                if attempt < attempts:
                    self._backoff(base_delay, attempt)
                    continue
                return None, {}

            net_ms = int((time.monotonic() - t0) * 1000)

            if resp.status_code in _RETRY_STATUSES and attempt < attempts:
                self._backoff(base_delay, attempt)
                continue

            if resp.status_code != 200:
                self._last_error = f"GLM OCR HTTP {resp.status_code}: {resp.text[:200]}"
                return None, {"network_ms": net_ms}

            try:
                data = resp.json()
            except ValueError:
                self._last_error = "GLM OCR returned non-JSON response"
                return None, {"network_ms": net_ms}

            if isinstance(data, dict) and data.get("error"):
                self._last_error = str(data["error"])
                return None, {"network_ms": net_ms}

            text = (data.get("text") or "").strip()
            timing = dict(data.get("timing_ms") or {})
            timing["network_ms"] = net_ms
            return text, timing

        return None, {}

    def extract_blocks(
        self, image_url: str
    ) -> tuple[Optional[str], list, dict]:
        """POST /parse — returns (text, blocks, timing).

        `blocks` is the glmocr pipeline's per-region output: a list of
        {index, label, content, bbox_2d:[x1,y1,x2,y2]} (pixel coordinates).
        text is None and blocks is [] on failure (see load_error()). The
        worker drives its own per-region prompts, so no prompt is sent.
        """
        self._last_error = None
        url = f"{self.host}/parse"
        body = {"image_url": image_url, "model": self.model}

        attempts = _retry_attempts()
        base_delay = _retry_base_delay()
        for attempt in range(1, attempts + 1):
            t0 = time.monotonic()
            try:
                resp = requests.post(url, json=body, timeout=self.timeout_seconds)
            except requests.RequestException as e:
                self._last_error = f"GLM OCR /parse request failed: {e}"
                if attempt < attempts:
                    self._backoff(base_delay, attempt)
                    continue
                return None, [], {}

            net_ms = int((time.monotonic() - t0) * 1000)

            if resp.status_code in _RETRY_STATUSES and attempt < attempts:
                self._backoff(base_delay, attempt)
                continue

            if resp.status_code != 200:
                self._last_error = f"GLM OCR /parse HTTP {resp.status_code}: {resp.text[:200]}"
                return None, [], {"network_ms": net_ms}

            try:
                data = resp.json()
            except ValueError:
                self._last_error = "GLM OCR /parse returned non-JSON response"
                return None, [], {"network_ms": net_ms}

            if isinstance(data, dict) and data.get("error"):
                self._last_error = str(data["error"])
                return None, [], {"network_ms": net_ms}

            text = (data.get("text") or "").strip()
            blocks = data.get("blocks") or []
            if not isinstance(blocks, list):
                blocks = []
            timing = dict(data.get("timing_ms") or {})
            timing["network_ms"] = net_ms
            return text, blocks, timing

        return None, [], {}

    @staticmethod
    def _backoff(base_delay: float, attempt: int) -> None:
        delay = base_delay * (2 ** (attempt - 1))
        delay *= 1.0 + random.uniform(0, 0.3)
        time.sleep(min(delay, 30.0))
