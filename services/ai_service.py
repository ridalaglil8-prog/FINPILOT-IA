"""External LLM integration (OpenAI-compatible chat-completions API).

Configuration comes exclusively from the environment / ``.env``:

    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS

The API key is never logged, never printed and never rendered in the UI.
Every failure mode (missing key, timeout, HTTP error, rate limit) is translated
into a short, user-safe message; the application stays fully functional in
Demo mode without any key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import LLM_DEFAULTS

try:  # optional: the app runs without python-dotenv too
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - optional dependency
    pass

try:
    import requests
except ImportError:  # pragma: no cover - dependency guard
    requests = None  # type: ignore


class AIServiceError(RuntimeError):
    """User-safe message describing why the assistant could not answer."""


@dataclass(frozen=True)
class LLMSettings:
    api_key: Optional[str]
    base_url: str
    model: str
    timeout: int

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def get_settings() -> LLMSettings:
    """Read configuration from the environment (never from hard-coded values)."""
    key = os.getenv("LLM_API_KEY", "").strip() or None
    base_url = (os.getenv("LLM_BASE_URL", "").strip() or LLM_DEFAULTS["base_url"]).rstrip("/")
    model = os.getenv("LLM_MODEL", "").strip() or LLM_DEFAULTS["model"]
    try:
        timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", str(LLM_DEFAULTS["timeout"])))
    except ValueError:
        timeout = int(LLM_DEFAULTS["timeout"])
    return LLMSettings(api_key=key, base_url=base_url, model=model, timeout=max(1, timeout))


def is_configured() -> bool:
    return get_settings().configured


def status_text() -> str:
    """Short status line for the Settings / Assistant pages.

    Never renders any part of the key: the UI only reports whether one is
    configured, and which model would be used.
    """
    settings = get_settings()
    if not settings.configured:
        return "not configured (LLM_API_KEY missing)"
    return f"configured — model {settings.model}, API key detected"


def _error_from_response(response: Any) -> AIServiceError:
    status = getattr(response, "status_code", None)
    if status == 401:
        return AIServiceError("the LLM API key was rejected (HTTP 401). Check LLM_API_KEY in your .env file.")
    if status == 429:
        return AIServiceError("the LLM provider is rate-limiting requests (HTTP 429). Try again shortly.")
    if status and status >= 500:
        return AIServiceError(f"the LLM provider is unavailable (HTTP {status}). Try again later.")
    detail = ""
    try:
        detail = str(response.json().get("error", {}))[:200]
    except Exception:
        detail = (response.text or "")[:200]
    return AIServiceError(f"the LLM request failed (HTTP {status}). {detail}")


def chat(user_message: str, *, system_prompt: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """Send one chat-completions request and return the assistant's text.

    Raises :class:`AIServiceError` for every failure mode.
    """
    settings = get_settings()
    if not settings.configured:
        raise AIServiceError(
            "No AI key configured. Set LLM_API_KEY in a .env file (see .env.example) "
            "to enable the assistant. Everything else in FinPilot AI works without it."
        )
    if requests is None:
        raise AIServiceError("the 'requests' package is not installed.")

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for item in (history or [])[-6:]:
        if item.get("role") in ("user", "assistant") and item.get("content"):
            messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": user_message})

    url = f"{settings.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }
    body: Dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "temperature": 0.2,
    }
    try:
        response = requests.post(url, headers=headers, json=body, timeout=settings.timeout)
    except requests.exceptions.Timeout as exc:
        raise AIServiceError(
            f"the LLM provider did not answer within {settings.timeout} s."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise AIServiceError(
            "could not reach the LLM provider (network or DNS error). "
            "Check LLM_BASE_URL and your Internet connection."
        ) from exc

    if response.status_code != 200:
        raise _error_from_response(response)

    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise AIServiceError("the LLM provider returned an unexpected response shape.") from exc

    if not isinstance(content, str) or not content.strip():
        raise AIServiceError("the LLM provider returned an empty answer.")
    return content.strip()
