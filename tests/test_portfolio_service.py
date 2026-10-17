"""End-to-end context test: session-state-like input -> computed context."""
from __future__ import annotations

import pytest

import config
from services import portfolio_service as PS


def _state(**overrides):
    state = config.default_state()
    state["data_mode"] = config.MODE_DEMO
    state["market"] = config.MARKET_US
    start, end = config.default_period()
    state["start"], state["end"] = start, end
    state.update(overrides)
    return state


def test_demo_context_is_usable_and_labelled_simulated():
    context = PS.build_context(_state())
    assert PS.is_usable(context)
    assert context["simulated"] is True
    assert context["currencies"] == ["USD"]
    assert context["multi_currency"] is False
    assert context["last_observation"]


def test_metrics_are_internally_consistent():
    context = PS.build_context(_state())
    metrics = context["metrics"]
    capital = context["groups"][0]["capital"]
    assert metrics["current_value"] == context["groups"][0]["wealth"].iloc[-1]
    assert metrics["cumulative_return"] == (
        metrics["current_value"] / capital - 1.0)
    assert metrics["annualized_volatility"] >= 0
    assert metrics["max_drawdown"] <= 0


def test_demo_context_is_reproducible_across_calls():
    first = PS.build_context(_state())
    second = PS.build_context(_state())
    assert first["metrics"]["current_value"] == second["metrics"]["current_value"]
    assert first["metrics"]["sharpe_ratio"] == second["metrics"]["sharpe_ratio"]


def test_empty_portfolio_is_reported_not_raised():
    context = PS.build_context(_state(assets=[]))
    assert not PS.is_usable(context)
    assert "no_assets" in context["errors"]


def test_invalid_weights_stop_the_pipeline():
    context = PS.build_context(_state(assets=[
        {"ticker": "SPY", "name": "SPY", "asset_class": "ETF",
         "currency": "USD", "weight": 0.3},
    ]))
    assert not PS.is_usable(context)
    assert "invalid_weights" in context["errors"]


def test_invalid_date_range_is_reported():
    start, end = config.default_period()
    context = PS.build_context(_state(start=end, end=start))
    assert not PS.is_usable(context)
    assert "invalid_date_range" in context["errors"]


def test_invalid_capital_is_reported():
    context = PS.build_context(_state(capital=-1000.0))
    assert not PS.is_usable(context)
    assert "invalid_capital" in context["errors"]


def test_multi_currency_portfolio_is_split_and_never_summed():
    assets = [
        {"ticker": "SPY", "name": "SPY", "asset_class": "US Equity ETF",
         "currency": "USD", "weight": 0.5},
        {"ticker": "SIM_MA_BANK", "name": "SIM bank", "asset_class": "MA Equity (simulated)",
         "currency": "MAD", "weight": 0.5},
    ]
    context = PS.build_context(_state(assets=assets, benchmark=None))
    assert PS.is_usable(context)
    assert context["multi_currency"] is True
    assert set(context["currencies"]) == {"USD", "MAD"}
    assert len(context["groups"]) == 2
    # each sub-portfolio is analysed on its own capital share
    for group in context["groups"]:
        assert group["capital"] == 50_000.0
        assert abs(sum(group["weights"].values()) - 1.0) < 1e-9


def test_multi_currency_with_fx_rate_reports_combined_total():
    assets = [
        {"ticker": "SPY", "name": "SPY", "asset_class": "US Equity ETF",
         "currency": "USD", "weight": 0.5},
        {"ticker": "SIM_MA_BANK", "name": "SIM bank", "asset_class": "MA Equity (simulated)",
         "currency": "MAD", "weight": 0.5},
    ]
    context = PS.build_context(_state(assets=assets, benchmark=None, fx_usd_mad=10.0))
    assert context["fx"] is not None
    # totals are merged into the first-seen currency (USD): 50 000 USD +
    # 50 000 MAD at MAD->USD = 0.1  =>  55 000 USD
    assert context["fx"]["currency"] == "USD"
    assert context["fx"]["total"] == pytest.approx(55_000.0)
    assert context["fx"]["rates_used"]["MAD_USD"] == pytest.approx(0.1)
    assert "fx_missing" not in context["warnings"]


def test_allocation_rows_carry_currency_and_value():
    context = PS.build_context(_state())
    for row in context["allocation"]:
        assert row["currency"] == "USD"
        assert row["value"] > 0
        assert 0 < row["weight_pct"] <= 100
