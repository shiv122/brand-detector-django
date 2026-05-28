"""
OCR service — two-stage pipeline behind a single provider name "local":

  1. GLM-OCR via a remote Ollama HTTP server (the GLM_OCR container) extracts
     raw text from the image.
  2. A text formatter (DeepSeek or Gemini, chosen by TEXT_FORMATTER_PROVIDER)
     turns the extracted text into JSON using the sport prompt.

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
        if not (self.config.local_ocr_ollama_host and self.config.local_ocr_ollama_model):
            return False
        provider = self.config.text_formatter_provider
        if provider == "gemini":
            return bool(self.config.gemini_text_api_key and self.config.gemini_text_base_url)
        # default: deepseek
        return bool(self.config.deepseek_text_api_key and self.config.deepseek_text_base_url)

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
        formatter_provider = self.config.text_formatter_provider
        if formatter_provider == "gemini":
            if not self.config.gemini_text_api_key:
                return {
                    "error": (
                        "GEMINI_TEXT_API_KEY is not set — needed to format "
                        "extracted text into JSON."
                    ),
                    "prompt": prompt,
                }
        else:
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

        # Bound the number of concurrent calls into Ollama across ALL RQ
        # workers via a Redis counter — Ollama serves one request at a
        # time per model (OLLAMA_NUM_PARALLEL=1), so without this limit
        # 12 workers race against 1 slot, latency collapses, and the
        # queue stalls. The slot is held only across the HTTP call.
        from apps.services.ocr.ocr_concurrency import ollama_slot

        try:
            with ollama_slot():
                extracted_text, ocr_timing = engine.extract_text(
                    image_data, self.config.local_ocr_extract_prompt
                )
        except TimeoutError as e:
            # No Ollama capacity within the wait budget — surface this as a
            # clean failure rather than letting the worker hang. The frontend
            # will see it as a failed job and stop polling that ID.
            if debug:
                print(f"[OCR/Local] slot timeout: {e}")
            return {
                "error": str(e),
                "prompt": prompt,
                "glm_timing": {"glm_ocr_backend": "ollama", "slot_timeout": True},
            }
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

        # Stage 2: text formatter (DeepSeek or Gemini) shapes raw text into JSON.
        if formatter_provider == "gemini":
            format_result = self._call_gemini_text(prompt, extracted_text)
            timing_prefix = "gemini_text"
        else:
            format_result = self._call_deepseek_text(prompt, extracted_text)
            timing_prefix = "deepseek_text"

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

        if debug:
            print(
                f"[OCR/Local] format via {formatter_provider} "
                f"model={format_result.get('model')} "
                f"net={format_result.get('network_ms', 0)}ms "
                f"tokens=in:{usage.get('prompt_tokens', usage.get('input_tokens', 0))}/"
                f"out:{usage.get('completion_tokens', usage.get('output_tokens', 0))}"
            )

        base = {
            "provider": "local",
            "timing_ms": {
                **glm_timing,
                f"{timing_prefix}_network": format_result.get("network_ms"),
                f"{timing_prefix}_completion_tokens": usage.get("completion_tokens"),
                f"{timing_prefix}_total_tokens": usage.get("total_tokens"),
            },
            f"{timing_prefix}_id": format_result.get("response_id"),
            f"{timing_prefix}_model": format_result.get("model"),
            "text_formatter_provider": formatter_provider,
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

    # ---------------------------------------------- Gemini text helper

    def _call_gemini_text(self, prompt: str, ocr_text: str) -> Dict[str, Any]:
        """POST generativelanguage.googleapis.com to format OCR text into JSON.

        Uses Gemini's native REST API (not the OpenAI-compat shim) so we get
        proper usage metadata and avoid auth-header quirks. The system prompt
        goes in `systemInstruction`, the user text in `contents`.
        """
        model = self.config.gemini_text_model
        url = (
            f"{self.config.gemini_text_base_url.rstrip('/')}"
            f"/models/{model}:generateContent"
        )
        body: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Below is OCR text extracted from an image. Apply the "
                                "instructions in the system message to this text and "
                                "return the result as the system message specifies.\n\n"
                                "--- OCR TEXT ---\n"
                                f"{ocr_text}\n"
                                "--- END ---"
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": self.config.gemini_text_temperature,
                "maxOutputTokens": self.config.gemini_text_max_tokens,
            },
        }

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                url,
                json=body,
                headers={
                    "x-goog-api-key": self.config.gemini_text_api_key,
                    "Content-Type": "application/json",
                },
                timeout=self.config.gemini_text_timeout_seconds,
            )
        except requests.RequestException as e:
            return {"error": f"Gemini text API request failed: {e}"}
        network_ms = int((time.perf_counter() - t0) * 1000)

        if not resp.ok:
            return {
                "error": f"Gemini text API {resp.status_code}: {resp.text[:300]}",
                "timing_ms": {"network": network_ms},
            }

        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            return {
                "error": f"Gemini text response was not JSON: {e}",
                "timing_ms": {"network": network_ms},
            }

        candidates = data.get("candidates") or []
        if not candidates:
            return {
                "error": f"Gemini text API returned no candidates: {data}",
                "timing_ms": {"network": network_ms},
            }
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        if not text:
            text = json.dumps(candidates[0])

        usage_md = data.get("usageMetadata") or {}
        # Normalize keys to match the deepseek shape so downstream code
        # (which reads usage.prompt_tokens / completion_tokens) keeps working.
        usage = {
            "prompt_tokens": usage_md.get("promptTokenCount"),
            "completion_tokens": usage_md.get("candidatesTokenCount"),
            "total_tokens": usage_md.get("totalTokenCount"),
        }

        return {
            "text": text,
            "network_ms": network_ms,
            "usage": usage,
            "response_id": data.get("responseId"),
            "model": data.get("modelVersion") or model,
        }
