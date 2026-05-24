"""
OCR service — two-stage pipeline behind a single provider name "local":

  1. GLM-OCR via a remote Ollama HTTP server (the GLM_OCR container) extracts
     raw text from the image.
  2. DeepSeek text API (api.deepseek.com) formats the extracted text into
     JSON using the sport prompt supplied by the caller.

Result shape (formatted / raw_text / prompt / timing_ms / format_error) is
preserved so downstream callers don't change.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests

from config.app_config import AppConfig


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def _ocr_debug_enabled() -> bool:
    return os.getenv("OCR_LLM_DEBUG", "0").lower() in ("1", "true", "yes")


_BARE_NAN_RE = re.compile(
    r"(?<=[:,\[\s])(nan|NaN|Infinity|-Infinity)(?=[,\]\}\s])"
)


def _normalize_json_constants(text: str) -> str:
    """Replace bare `nan`/`NaN`/`Infinity` tokens with `null`.

    VLMs occasionally emit Python-style float literals ("score": nan) that
    strict `json.loads` rejects. The regex only matches outside string
    literals (by requiring a JSON structural delimiter on either side).
    """
    if not text:
        return text
    return _BARE_NAN_RE.sub("null", text)


def _strip_fence(text: str) -> str:
    """Return the inner content of a ```json ... ``` fence, or the input as-is.

    Also handles UNCLOSED fences: when the model's output is truncated mid-
    stream the trailing ``` is missing.
    """
    if not text:
        return ""
    stripped = text.strip()
    m = _FENCE_RE.match(stripped)
    if m:
        return m.group(1).strip()
    if stripped.startswith("```"):
        nl = stripped.find("\n")
        inner = stripped[nl + 1:] if nl != -1 else stripped[3:]
        if inner.rstrip().endswith("```"):
            inner = inner.rstrip()[:-3]
        return inner.strip()
    return stripped


def _repair_truncated_json(text: str) -> Optional[Any]:
    """Best-effort repair of JSON that got truncated mid-stream."""
    if not text:
        return None

    open_stack: List[str] = []
    in_string = False
    escape = False
    last_safe_end: Optional[int] = None

    for i, c in enumerate(text):
        if escape:
            escape = False
            continue
        if in_string:
            if c == "\\":
                escape = True
            elif c == '"':
                in_string = False
                last_safe_end = i + 1
            continue
        if c == '"':
            in_string = True
            continue
        if c in "{[":
            open_stack.append(c)
            continue
        if c == "}":
            if not open_stack or open_stack[-1] != "{":
                return None
            open_stack.pop()
            last_safe_end = i + 1
            continue
        if c == "]":
            if not open_stack or open_stack[-1] != "[":
                return None
            open_stack.pop()
            last_safe_end = i + 1
            continue
        if c == ",":
            last_safe_end = i
            continue

    if last_safe_end is None:
        return None

    prefix = text[:last_safe_end]
    rem_stack: List[str] = []
    in_s = False
    esc = False
    for c in prefix:
        if esc:
            esc = False
            continue
        if in_s:
            if c == "\\":
                esc = True
            elif c == '"':
                in_s = False
            continue
        if c == '"':
            in_s = True
            continue
        if c in "{[":
            rem_stack.append(c)
        elif c == "}" and rem_stack and rem_stack[-1] == "{":
            rem_stack.pop()
        elif c == "]" and rem_stack and rem_stack[-1] == "[":
            rem_stack.pop()

    closer = "".join("]" if b == "[" else "}" for b in reversed(rem_stack))
    repaired = prefix + closer
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _dedupe_string_lists(obj: Any) -> Any:
    """Recursively dedupe lists-of-strings inside the result while preserving order."""
    if isinstance(obj, dict):
        return {k: _dedupe_string_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and all(isinstance(x, str) for x in obj):
            seen: set = set()
            out: List[str] = []
            for x in obj:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            return out
        return [_dedupe_string_lists(x) for x in obj]
    return obj


def _parse_assistant_text(text: str, prompt: str, base: Dict[str, Any]) -> Dict[str, Any]:
    """Strip fence + parse JSON (with repair fallback) and merge into `base`."""
    inner = _strip_fence(text)
    normalized = _normalize_json_constants(inner)
    formatted: Optional[Any] = None
    format_error: Optional[str] = None
    if normalized:
        try:
            formatted = json.loads(normalized)
        except json.JSONDecodeError as e:
            repaired = _repair_truncated_json(normalized)
            if repaired is not None:
                formatted = repaired
                format_error = (
                    "endpoint output was truncated; salvaged via "
                    "best-effort JSON repair"
                )
            else:
                format_error = (
                    f"endpoint returned non-JSON inside the fence: {e}"
                )

    if formatted is not None:
        formatted = _dedupe_string_lists(formatted)

    result = {
        **base,
        "formatted": formatted,
        "raw_text": text,
        "prompt": prompt,
    }
    if format_error:
        result["format_error"] = format_error
    return result


class OcrService:
    """Two-stage OCR: GLM-OCR (remote Ollama) extracts text, DeepSeek formats."""

    def __init__(self, config: AppConfig):
        self.config = config

    @property
    def provider(self) -> str:
        return self.config.ocr_provider

    def is_available(self) -> bool:
        return bool(
            self.config.local_ocr_ollama_host
            and self.config.local_ocr_ollama_model
            and self.config.deepseek_text_api_key
            and self.config.deepseek_text_base_url
        )

    def run(self, image_data: bytes, prompt: str) -> Dict[str, Any]:
        if self.provider != "local":
            return {
                "error": (
                    f"Unknown OCR_PROVIDER={self.provider!r}. Only 'local' is "
                    f"supported."
                ),
                "prompt": prompt,
            }
        return self._run_local(image_data, prompt)

    # ----------------------------------------------------------------- local

    def _run_local(self, image_data: bytes, prompt: str) -> Dict[str, Any]:
        if not self.config.deepseek_text_api_key:
            return {
                "error": (
                    "DEEPSEEK_TEXT_API_KEY is not set — needed to format "
                    "extracted text into JSON."
                ),
                "prompt": prompt,
            }

        debug = _ocr_debug_enabled()

        # Stage 1: extract text via remote Ollama (GLM_OCR container).
        from apps.services.ocr.ollama_ocr_engine import OllamaOcrEngine

        engine = OllamaOcrEngine(
            host=self.config.local_ocr_ollama_host,
            model=self.config.local_ocr_ollama_model,
            max_new_tokens=self.config.local_ocr_max_new_tokens,
            timeout_seconds=self.config.local_ocr_ollama_timeout_seconds,
        )

        if debug:
            print(
                f"\n[OCR/Local] extract via ollama "
                f"model={self.config.local_ocr_ollama_model} "
                f"host={self.config.local_ocr_ollama_host} "
                f"image={len(image_data)} bytes"
            )

        extracted_text, ocr_timing = engine.extract_text(
            image_data, self.config.local_ocr_extract_prompt
        )
        glm_timing: Dict[str, Any] = {
            "glm_ocr_load": ocr_timing.get("load_ms", 0),
            "glm_ocr_inference": ocr_timing.get("inference_ms", 0),
            "glm_ocr_backend": "ollama",
        }
        if "network_ms" in ocr_timing:
            glm_timing["glm_ocr_network"] = ocr_timing["network_ms"]
        if "ollama_total_ms" in ocr_timing:
            glm_timing["ollama_total_ms"] = ocr_timing["ollama_total_ms"]
            glm_timing["ollama_eval_ms"] = ocr_timing["ollama_eval_ms"]

        if extracted_text is None:
            return {
                "error": engine.load_error() or "Local OCR extraction failed.",
                "prompt": prompt,
                "timing_ms": glm_timing,
            }

        if debug:
            print(
                f"[OCR/Local] extract ok load={ocr_timing.get('load_ms', 0)}ms "
                f"inf={ocr_timing.get('inference_ms', 0)}ms "
                f"text_len={len(extracted_text)}"
            )

        # Stage 2: DeepSeek text API formats into JSON.
        format_result = self._call_deepseek_text(prompt, extracted_text)
        if "error" in format_result:
            return {
                **format_result,
                "raw_text": extracted_text,
                "prompt": prompt,
                "provider": "local",
                "timing_ms": {
                    **glm_timing,
                    **format_result.get("timing_ms", {}),
                },
            }

        formatter_text = format_result["text"]
        usage = format_result.get("usage", {})
        base = {
            "provider": "local",
            "timing_ms": {
                **glm_timing,
                "deepseek_text_network": format_result.get("network_ms"),
                "deepseek_text_completion_tokens": usage.get("completion_tokens"),
                "deepseek_text_total_tokens": usage.get("total_tokens"),
            },
            "deepseek_text_id": format_result.get("response_id"),
            "deepseek_text_model": format_result.get("model"),
            "glm_ocr_text": extracted_text,
        }
        result = _parse_assistant_text(formatter_text, prompt, base)
        result["raw_text"] = extracted_text
        result["formatter_raw_text"] = formatter_text
        return result

    # ---------------------------------------------- DeepSeek text helper

    def _call_deepseek_text(self, prompt: str, ocr_text: str) -> Dict[str, Any]:
        """POST api.deepseek.com /chat/completions to format OCR text into JSON.

        Returns either:
          {"text", "network_ms", "usage", "response_id", "model"}
        or:
          {"error", "timing_ms": {"network": ...}}
        """
        url = self.config.deepseek_text_base_url.rstrip("/") + "/chat/completions"
        body: Dict[str, Any] = {
            "model": self.config.deepseek_text_model,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "Below is OCR text extracted from an image. Apply the "
                        "instructions in the system message to this text and "
                        "return the result as the system message specifies.\n\n"
                        "--- OCR TEXT ---\n"
                        f"{ocr_text}\n"
                        "--- END ---"
                    ),
                },
            ],
            "max_tokens": self.config.deepseek_text_max_tokens,
            "temperature": self.config.deepseek_text_temperature,
            "stream": False,
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                url,
                json=body,
                headers={
                    "Authorization": f"Bearer {self.config.deepseek_text_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.deepseek_text_timeout_seconds,
            )
        except requests.RequestException as e:
            return {"error": f"DeepSeek text API request failed: {e}"}
        network_ms = int((time.perf_counter() - t0) * 1000)

        if not resp.ok:
            return {
                "error": f"DeepSeek text API {resp.status_code}: {resp.text[:300]}",
                "timing_ms": {"network": network_ms},
            }

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            return {
                "error": f"DeepSeek text response was not JSON: {e}",
                "timing_ms": {"network": network_ms},
            }

        choices = data.get("choices") or []
        if not choices:
            return {
                "error": f"DeepSeek text API returned no choices: {data}",
                "timing_ms": {"network": network_ms},
            }
        message = choices[0].get("message") or {}
        text = message.get("content")
        if not isinstance(text, str):
            text = json.dumps(message)

        return {
            "text": text,
            "network_ms": network_ms,
            "usage": data.get("usage") or {},
            "response_id": data.get("id"),
            "model": data.get("model"),
        }
