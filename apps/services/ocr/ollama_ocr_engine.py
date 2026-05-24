"""
Ollama OCR engine — calls a local Ollama server (default
http://localhost:11434) running a vision model (e.g. GLM-OCR) to extract
raw text from an image.

Mirrors the shape of GlmOcrEngine so OcrService can swap between the two
based on `LOCAL_OCR_BACKEND`. No model loading happens here — Ollama is a
separate process that holds the weights and serves them over HTTP.
"""

from __future__ import annotations

import base64
import time
from typing import Optional

import requests


class OllamaOcrEngine:
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

    def is_loaded(self) -> bool:
        # Ollama loads on its own schedule; we report True because there's
        # no client-side load to wait for.
        return True

    def extract_text(
        self, image_bytes: bytes, prompt: str
    ) -> tuple[Optional[str], dict]:
        """Return (text, timing) — text is None if the call failed.

        timing keys (in ms): network, ollama_total_duration, ollama_eval.
        `load_ms` is reported as 0 to match GlmOcrEngine's contract.
        `inference_ms` aliases `ollama_total_duration` for the same reason.
        """
        self._last_error = None
        url = f"{self.host}/api/chat"
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    # Ollama expects bare base64 (no `data:` prefix).
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": self.max_new_tokens,
            },
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(url, json=body, timeout=self.timeout_seconds)
        except requests.RequestException as e:
            self._last_error = (
                f"Could not reach Ollama at {self.host} — is `ollama serve` "
                f"running? {e}"
            )
            return None, {"load_ms": 0, "inference_ms": 0, "network_ms": 0}
        network_ms = int((time.perf_counter() - t0) * 1000)

        if not resp.ok:
            self._last_error = (
                f"Ollama {resp.status_code}: {resp.text[:300]}"
            )
            return None, {
                "load_ms": 0,
                "inference_ms": 0,
                "network_ms": network_ms,
            }

        try:
            data = resp.json()
        except ValueError as e:
            self._last_error = f"Ollama response was not JSON: {e}"
            return None, {
                "load_ms": 0,
                "inference_ms": 0,
                "network_ms": network_ms,
            }

        message = data.get("message") or {}
        text = message.get("content")
        if not isinstance(text, str):
            self._last_error = f"Ollama response missing 'message.content': {data}"
            return None, {
                "load_ms": 0,
                "inference_ms": 0,
                "network_ms": network_ms,
            }

        # Ollama timings are in nanoseconds. We expose total_duration (the
        # whole request including model load) and eval_duration (token gen
        # only) so cold starts are visible in the response.
        total_ns = int(data.get("total_duration") or 0)
        eval_ns = int(data.get("eval_duration") or 0)
        total_ms = total_ns // 1_000_000
        eval_ms = eval_ns // 1_000_000

        return text.strip(), {
            "load_ms": 0,
            "inference_ms": total_ms or network_ms,
            "network_ms": network_ms,
            "ollama_total_ms": total_ms,
            "ollama_eval_ms": eval_ms,
        }
