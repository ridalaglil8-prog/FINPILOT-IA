"""Guard: every translation key referenced in the codebase must exist.

Catches the classic regression where a page calls ``t("new.key")`` but the
locale files were not updated (or only one of them was).
"""
from __future__ import annotations

import re
from pathlib import Path

import i18n

ROOT = Path(i18n.__file__).resolve().parent  # project root (i18n.py lives there)
SCAN_DIRS = ("pages", "components", "services", "app.py", "ai", "analytics",
             "optimization", "data")
CODE_SOURCES = tuple(
    p for d in SCAN_DIRS
    for p in ([ROOT / d] if (ROOT / d).is_file() else (ROOT / d).rglob("*.py"))
)

T_CALL = re.compile(r"""\bt\(\s*["']([a-zA-Z0-9_.]+)["']""")
TW_CALL = re.compile(r"""\btw\(\s*["']([a-zA-Z0-9_.]+)["']""")


def _referenced_keys() -> set:
    keys: set = set()
    for path in CODE_SOURCES:
        text = path.read_text(encoding="utf-8")
        keys.update(T_CALL.findall(text))
        keys.update(TW_CALL.findall(text))
    return keys


def test_source_files_exist():
    assert CODE_SOURCES, "no source files were scanned"


def test_every_referenced_key_exists_in_both_locales():
    missing_en, missing_fr = [], []
    for key in sorted(_referenced_keys()):
        if key not in i18n.load_locale("en"):
            missing_en.append(key)
        if key not in i18n.load_locale("fr"):
            missing_fr.append(key)
    assert not missing_en, f"missing in en.json: {missing_en}"
    assert not missing_fr, f"missing in fr.json: {missing_fr}"


def test_backend_codes_are_translatable():
    """Codes emitted by services/analytics must resolve via ``tw``."""
    expected = (
        "no_assets", "duplicate_assets", "invalid_weights", "empty_weights",
        "invalid_capital", "invalid_date_range", "missing_prices",
        "insufficient_history", "portfolio_calculation_failed",
        "dropped_dates", "fx_missing", "benchmark_simulated",
        "benchmark_overlap", "correlation_unavailable", "short_history",
    )
    for language in ("en", "fr"):
        locale = i18n.load_locale(language)
        for code in expected:
            assert f"code.{code}" in locale, f"code.{code} missing from {language}.json"


def test_format_placeholders_match_across_locales():
    """A key using {placeholders} in English must use the same ones in French."""
    english, french = i18n.load_locale("en"), i18n.load_locale("fr")
    pattern = re.compile(r"\{([a-zA-Z0-9_]+)\}")
    mismatches = []
    for key in english:
        en_slots = set(pattern.findall(english[key]))
        fr_slots = set(pattern.findall(french.get(key, "")))
        if en_slots != fr_slots:
            mismatches.append((key, sorted(en_slots), sorted(fr_slots)))
    assert not mismatches, f"placeholder mismatches: {mismatches}"
