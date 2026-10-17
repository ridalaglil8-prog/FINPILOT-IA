"""Localization tests: key parity between locales and safe fallbacks."""
from __future__ import annotations

import json

import i18n


def _keys(language: str) -> set:
    with (i18n.LOCALES_DIR / f"{language}.json").open(encoding="utf-8") as fh:
        return set(json.load(fh).keys())


def test_locale_files_exist_and_are_objects():
    for language in ("en", "fr"):
        data = i18n.load_locale(language)
        assert isinstance(data, dict) and data


def test_locales_have_identical_key_sets():
    assert _keys("en") == _keys("fr")


def test_translations_are_non_empty_strings():
    for language in ("en", "fr"):
        for key, value in i18n.load_locale(language).items():
            assert isinstance(value, str) and value.strip(), f"{language}:{key}"


def test_translate_with_placeholders():
    english = i18n.translate("en", "sidebar.version", version="1.0.0")
    french = i18n.translate("fr", "sidebar.version", version="1.0.0")
    assert "1.0.0" in english and "Version" in english
    assert "1.0.0" in french and "{version}" not in french


def test_missing_placeholder_degrades_gracefully():
    # no kwargs supplied for a template key -> the raw template is returned
    assert "{" in i18n.translate("en", "sidebar.version")


def test_unknown_key_falls_back_to_itself():
    assert i18n.translate("en", "definitely.not.a.key") == "definitely.not.a.key"


def test_unknown_language_falls_back_to_english():
    assert i18n.translate("xx", "nav.dashboard") == i18n.translate("en", "nav.dashboard")


def test_missing_locale_file_returns_empty_dict():
    assert i18n.load_locale("zz") == {}


def test_tw_translates_codes_with_detail():
    english = i18n.translate("en", "code.missing_prices")
    message = i18n.tw("missing_prices: SPY, QQQ")
    assert message.startswith(english)
    assert "SPY" in message


def test_tw_unknown_code_returns_raw_text():
    assert i18n.tw("some brand new backend message") == "some brand new backend message"


def test_available_languages():
    languages = i18n.available_languages()
    assert set(languages) == {"en", "fr"}
    assert languages["fr"] == "Français"
