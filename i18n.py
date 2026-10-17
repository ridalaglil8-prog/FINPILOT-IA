"""Single shared translation helper for FinPilot AI.

Locales live in ``locales/fr.json`` and ``locales/en.json`` and must contain
the same set of keys (enforced by ``tests/test_i18n.py``).

``t()`` never raises: a missing key falls back to English, then to the key
itself, so a translation gap degrades gracefully instead of crashing a page.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

LOCALES_DIR = Path(__file__).resolve().parent / "locales"
DEFAULT_LANGUAGE = "en"
FALLBACK_LANGUAGE = "en"


@lru_cache(maxsize=None)
def load_locale(language: str) -> Dict[str, Any]:
    """Load and cache a locale dictionary."""
    path = LOCALES_DIR / f"{language}.json"
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("locale file must contain a JSON object")
        return data
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, ValueError):
        return {}


def available_languages() -> Dict[str, str]:
    return {"fr": "Français", "en": "English"}


def current_language() -> str:
    try:
        import streamlit as st

        return st.session_state.get("language", DEFAULT_LANGUAGE)
    except Exception:  # pragma: no cover - outside a Streamlit runtime
        return DEFAULT_LANGUAGE


def set_language(language: str) -> None:
    if language not in available_languages():
        language = DEFAULT_LANGUAGE
    try:
        import streamlit as st

        st.session_state["language"] = language
    except Exception:  # pragma: no cover
        pass


def translate(language: str, key: str, **kwargs: Any) -> str:
    """Translate ``key`` for ``language`` with optional ``str.format`` values."""
    value = load_locale(language).get(key)
    if value is None and language != FALLBACK_LANGUAGE:
        value = load_locale(FALLBACK_LANGUAGE).get(key)
    if value is None:
        value = key
    if not isinstance(value, str):
        value = str(value)
    if kwargs:
        try:
            return value.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return value
    return value


def t(key: str, **kwargs: Any) -> str:
    """Translate ``key`` using the language selected in the session."""
    return translate(current_language(), key, **kwargs)


def tw(code: str, **kwargs: Any) -> str:
    """Translate a machine-readable *code*, optionally carrying a detail part.

    Codes are emitted by the business layer (``portfolio_service``,
    ``market_data``...) so that they can stay language-neutral. Two shapes are
    accepted:

    * ``"invalid_weights"``              -> ``code.invalid_weights``
    * ``"dropped_dates: 12"``            -> ``code.dropped_dates`` + the raw
      detail appended in parentheses, e.g. *"... (12)"*.

    Unknown codes degrade to the raw text instead of raising, so a new
    backend message never crashes a page.
    """
    text = str(code).strip()
    prefix, sep, detail = text.partition(":")
    key = f"code.{prefix.strip()}"
    message = translate(current_language(), key)
    if message == key:  # not a known code -> show the raw text
        message = text
    detail = detail.strip()
    if kwargs:
        try:
            message = message.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            pass
    if sep and detail:
        return f"{message} ({detail})"
    return message
