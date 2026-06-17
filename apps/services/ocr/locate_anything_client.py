"""
HTTP client for the external LocateAnything service.

POSTs an image URL + a task/query to {LOCATE_HOST}/locate; the service fetches
the image (DigitalOcean Spaces) itself, runs NVIDIA LocateAnything-3B, and
returns the raw model text plus parsed bounding boxes whose labels are the
text it read. No image bytes cross this wire — only the URL.

This mirrors GlmOcrClient so OcrService can treat it as a drop-in stage-1
text extractor: `extract_text()` returns (text, boxes, timing), where `text`
is the readable content (box labels joined, falling back to the cleaned raw
output) ready to hand to the stage-2 formatter.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from typing import List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_RETRY_STATUSES: frozenset[int] = frozenset({429, 502, 503, 504})

# Strip the model's box markup so the fallback (no parsed labels) still yields
# clean text: drop whole <box>...</box> blocks (they hold the <int> coords),
# then any remaining tag (<ref>/<box>/<123>), keeping the wrapped ref text.
_BOX_BLOCK_RE = re.compile(r"<box>.*?</box>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


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


def _clean_markup(raw: str) -> str:
    if not raw:
        return ""
    s = _BOX_BLOCK_RE.sub(" ", raw)
    s = _TAG_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _text_from(boxes: list, raw: str) -> str:
    """Best readable text: box labels (the read text), else cleaned raw output."""
    labels = [str(b.get("label")).strip() for b in boxes if b.get("label")]
    if labels:
        return "\n".join(labels)
    return _clean_markup(raw)


class LocateAnythingClient:
    def __init__(self, host: str, timeout_seconds: float = 180.0):
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._last_error: Optional[str] = None

    def load_error(self) -> Optional[str]:
        return self._last_error

    def extract_text(
        self,
        image_url: str,
        task: str = "ocr",
        query: str = "",
        mode: str = "hybrid",
        max_new_tokens: int = 2048,
        max_side: Optional[int] = None,
    ) -> Tuple[Optional[str], List[dict], dict, dict]:
        """Return (text, boxes, image, timing).

        text is None on failure (see load_error()). `boxes` carry the read text
        as `label` plus pixel `box` coords; `image` is {width, height} of the
        original image so callers can map boxes onto what the user sees.
        """
        self._last_error = None
        url = f"{self.host}/locate"
        body = {
            "image_url": image_url,
            "task": task,
            "query": query or None,
            "mode": mode,
            "max_tokens": max_new_tokens,
        }
        if max_side is not None:
            body["max_side"] = max_side

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
                return None, [], {}, {}

            net_ms = int((time.monotonic() - t0) * 1000)

            if resp.status_code in _RETRY_STATUSES and attempt < attempts:
                self._backoff(base_delay, attempt)
                continue

            if resp.status_code != 200:
                self._last_error = (
                    f"LocateAnything HTTP {resp.status_code}: {resp.text[:200]}"
                )
                return None, [], {}, {"network_ms": net_ms}

            try:
                data = resp.json()
            except ValueError:
                self._last_error = "LocateAnything returned non-JSON response"
                return None, [], {}, {"network_ms": net_ms}

            if isinstance(data, dict) and data.get("error"):
                self._last_error = str(data["error"])
                return None, [], {}, {"network_ms": net_ms}

            boxes = data.get("boxes") or []
            text = _text_from(boxes, data.get("text") or "")
            img = data.get("image") or {}
            image = {"width": img.get("width"), "height": img.get("height")}
            timing = dict(data.get("timing_ms") or {})
            timing["network_ms"] = net_ms
            timing["boxes"] = len(boxes)
            return text, boxes, image, timing

        return None, [], {}, {}

    @staticmethod
    def _backoff(base_delay: float, attempt: int) -> None:
        delay = base_delay * (2 ** (attempt - 1))
        delay *= 1.0 + random.uniform(0, 0.3)
        time.sleep(min(delay, 30.0))
