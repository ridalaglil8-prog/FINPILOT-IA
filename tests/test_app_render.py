"""Render-level tests using Streamlit's official AppTest harness.

These execute the real page scripts in a simulated Streamlit runtime —
catching exceptions that unit tests (which bypass the UI layer) cannot.
No network: everything runs on deterministic Demo data.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import config

# Streamlit's AppTest resolves script paths differently across versions:
# older releases resolve them against the process CWD, newer ones (>=1.65)
# against the file that calls AppTest.from_file().  Anchoring every path to
# the repository root keeps the tests correct under both behaviours.
_ROOT = Path(__file__).resolve().parent.parent

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
    script = Path(path)
    if not script.is_absolute():
        script = _ROOT / script
    at = AppTest.from_file(str(script), default_timeout=timeout)
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


def _metrics(at) -> dict:
    return {str(m.label): str(m.value) for m in at.metric}


def test_stress_page_shocks_default_to_zero_not_to_the_widget_minimum():
    """Regression: ``st.number_input`` defaults to ``value="min"``, so shock
    inputs seeded with ``min_value=-100`` opened on a −100 % scenario that
    wiped out the whole portfolio, and the class-level presets never worked."""
    from streamlit.elements.lib import policies

    # the duplication warning is latched once per process — reset the latch so
    # this test really proves the page never emits it
    policies._shown_default_value_warning = False

    at = _run("pages/5_Stress_Testing.py")
    _assert_no_exception(at, "stress")

    shocks = {str(ni.key): float(ni.value) for ni in at.number_input}
    assert shocks, "expected shock inputs"
    assert all(value > -100.0 for value in shocks.values()), shocks
    assert shocks["stress_group_US Equity ETF"] == -10.0   # default preset
    assert shocks["stress_asset_SPY"] == 0.0               # not a fake −100 %

    metrics = _metrics(at)
    # 45 % + 25 % equity at −10 %  =>  −7 % of the portfolio
    assert metrics["Estimated impact"] == "-7.00 %"
    assert "-100.00" not in " ".join(metrics.values())

    # seeding the widget keys must not trigger Streamlit's duplication warning
    duplication = [str(w.value) for w in at.warning if "Session State API" in str(w.value)]
    assert not duplication, duplication


def test_stress_scenario_is_applied_to_the_current_portfolio_value():
    """"Value before" must match the value the dashboard reports today,
    not the initial capital invested."""
    import config
    from components.kpi_cards import fmt_money
    from services.portfolio_service import build_context

    at = _run("pages/5_Stress_Testing.py")
    _assert_no_exception(at, "stress")

    current_value = build_context(config.default_state())["metrics"]["current_value"]
    metrics = _metrics(at)
    assert metrics["Portfolio value before"] == fmt_money(current_value, "USD")
    if current_value != 100_000.0:
        assert metrics["Portfolio value before"] != fmt_money(100_000.0, "USD")


def test_stress_preset_switch_applies_class_shocks():
    """A preset must reach every asset of the class (zero-valued per-asset
    widgets used to cancel the class shock entirely)."""
    at = _run("pages/5_Stress_Testing.py")
    at.radio(key="stress_preset").set_value("bond")
    at.run()
    _assert_no_exception(at, "stress (bond preset)")

    shocks = {str(ni.key): float(ni.value) for ni in at.number_input}
    assert shocks["stress_group_Bond ETF"] == -8.0
    assert shocks["stress_group_US Equity ETF"] == 0.0
    # 20 % bond weight × −8 %  =>  −1.6 %
    assert _metrics(at)["Estimated impact"] == "-1.60 %"


def test_stress_per_asset_shock_overrides_its_class_shock():
    at = _run("pages/5_Stress_Testing.py")
    at.radio(key="stress_preset").set_value("bond")
    at.run()
    at.number_input(key="stress_asset_TLT").set_value(5.0)   # exempt +5 %
    at.run()
    _assert_no_exception(at, "stress (asset override)")

    # the explicit per-asset +5 % replaces the −8 % class shock on TLT
    assert _metrics(at)["Estimated impact"] == "+1.00 %"


def test_stress_scenario_stored_for_the_assistant_matches_the_screen():
    at = _run("pages/5_Stress_Testing.py")
    _assert_no_exception(at, "stress")

    scenario = at.session_state["stress_test"]
    assert scenario.get("signature")
    assert [g["currency"] for g in scenario["groups"]] == ["USD"]
    assert scenario["groups"][0]["impact_pct"] == pytest.approx(-0.07)
    # the AI is told exactly what the page displays
    assert _metrics(at)["Estimated impact"] == "-7.00 %"


def test_optimization_results_are_dropped_when_the_portfolio_changes():
    """Results computed on another portfolio must never stay on screen (or
    reach the assistant) — the user is asked to run the optimisation again."""
    at = _run("pages/4_Portfolio_Optimization.py")
    _assert_no_exception(at, "optimization (initial)")
    assert any("Run optimization" in str(i.value) for i in at.info)

    at.button(key="opt_run").click()
    at.run()
    _assert_no_exception(at, "optimization (run)")
    assert "optimization" in at.session_state
    assert not any("Run optimization" in str(i.value) for i in at.info)

    # a valid rebalancing (weights still total 100 %) invalidates the result
    for asset, weight in zip(at.session_state["assets"], [0.60, 0.20, 0.10, 0.10]):
        asset["weight"] = weight
    at.run()
    _assert_no_exception(at, "optimization (changed)")
    assert any("Run optimization" in str(i.value) for i in at.info)
    assert not at.metric, "stale optimization metrics must not be rendered"
