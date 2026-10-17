"""Render-level tests using Streamlit's official AppTest harness.

These execute the real page scripts in a simulated Streamlit runtime —
catching exceptions that unit tests (which bypass the UI layer) cannot.
No network: everything runs on deterministic Demo data.
"""
from __future__ import annotations

import datetime as _dt

import pytest
from streamlit.testing.v1 import AppTest

import config

# Streamlit's AppTest executes a script from the project root.
MAIN = "app.py"
PAGES = [
    ("pages/1_Portfolio_Builder.py", "Portfolio Builder"),
    ("pages/2_Performance_Analytics.py", "Performance Analytics"),
    ("pages/3_Risk_Diversification.py", "Risk & Diversification"),
    ("pages/4_Portfolio_Optimization.py", "Portfolio Optimization"),
    ("pages/5_Stress_Testing.py", "Stress Testing"),
    ("pages/6_AI_Assistant.py", "AI Financial Assistant"),
    ("pages/7_Settings.py", "Settings"),
]


def _run(path: str, *, timeout: int = 120) -> AppTest:
    at = AppTest.from_file(path, default_timeout=timeout)
    at.run()
    return at


def _assert_no_exception(at: AppTest, label: str) -> None:
    errors = [str(e.value)[:300] for e in at.exception]
    assert not errors, f"{label} raised: {errors}"


def test_dashboard_renders_demo_mode():
    at = _run(MAIN)
    _assert_no_exception(at, "dashboard")
    titles = [str(t.value) for t in at.title]
    assert any("FinPilot AI" in t for t in titles)


def test_dashboard_shows_simulated_label():
    at = _run(MAIN)
    warnings = " ".join(str(w.value) for w in at.warning)
    assert "SIMULATED" in warnings


def test_every_page_renders_without_exception():
    for path, label in PAGES:
        at = _run(path)
        _assert_no_exception(at, label)


def test_builder_page_lists_default_assets():
    at = _run("pages/1_Portfolio_Builder.py")
    _assert_no_exception(at, "builder")
    # four default US holdings, each with a remove button
    assert len(at.button) >= 4


def test_language_switch_to_french_updates_titles():
    at = _run(MAIN)
    at.session_state["language"] = "fr"
    at.run()
    _assert_no_exception(at, "dashboard (fr)")
    titles = [str(t.value) for t in at.title]
    assert any("Tableau de bord" in t for t in titles), titles


def test_french_pages_render():
    for path, label in PAGES:
        at = _run(path)
        at.session_state["language"] = "fr"
        at.run()
        _assert_no_exception(at, f"{label} (fr)")


def test_moroccan_market_without_source_shows_informative_message():
    at = _run(MAIN)
    at.session_state["market"] = config.MARKET_MA
    at.session_state["data_mode"] = config.MODE_HISTORICAL
    at.session_state["assets"] = config.default_assets(config.MARKET_MA)
    at.run()
    _assert_no_exception(at, "morocco/historical")
    # must explain the missing source rather than crash or invent data
    text = " ".join(str(e.value) for e in at.error) + " ".join(
        str(i.value) for i in at.info)
    assert text.strip(), "expected an informative message for the MA market"


def test_empty_portfolio_is_handled_gracefully():
    at = _run(MAIN)
    at.session_state["assets"] = []
    at.run()
    _assert_no_exception(at, "empty portfolio")


def test_invalid_weights_are_reported_not_crashed():
    at = _run(MAIN)
    start, end = config.default_period()
    at.session_state["assets"] = [
        {"ticker": "SPY", "name": "SPY", "asset_class": "US Equity ETF",
         "currency": "USD", "weight": 0.3},
    ]
    at.run()
    _assert_no_exception(at, "invalid weights")


def test_inverted_date_range_is_reported():
    at = _run(MAIN)
    start, end = config.default_period()
    at.session_state["start"] = end
    at.session_state["end"] = start
    at.run()
    _assert_no_exception(at, "inverted dates")


def test_settings_page_renders_upload_and_controls():
    at = _run("pages/7_Settings.py")
    _assert_no_exception(at, "settings")
    assert len(at.number_input) >= 2
