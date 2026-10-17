"""Stress-scenario tests: first-order impact of hypothetical shocks."""
from __future__ import annotations

import pytest

from analytics import stress as Stress


WEIGHTS = {"AAA": 0.6, "BBB": 0.4}


def test_normalize_shock_accepts_percent_and_decimal():
    assert Stress.normalize_shock(-10) == pytest.approx(-0.10)
    assert Stress.normalize_shock(-0.10) == pytest.approx(-0.10)
    assert Stress.normalize_shock(15) == pytest.approx(0.15)
    assert Stress.normalize_shock(0.0) == 0.0


def test_normalize_shock_rejects_impossible_values():
    for bad in (-200, 10_000, float("nan"), "abc"):
        with pytest.raises(Stress.ShockError):
            Stress.normalize_shock(bad)


def test_apply_shocks_impact_and_values():
    result = Stress.apply_shocks(WEIGHTS, {"AAA": -0.10, "BBB": -0.05}, 100_000.0)
    assert result["impact_pct"] == pytest.approx(0.6 * -0.10 + 0.4 * -0.05)
    assert result["impact_value"] == pytest.approx(-8_000.0)
    assert result["value_after"] == pytest.approx(92_000.0)
    row = {r["ticker"]: r for r in result["rows"]}
    assert row["AAA"]["value_before"] == pytest.approx(60_000.0)
    assert row["AAA"]["value_after"] == pytest.approx(54_000.0)
    assert row["BBB"]["impact"] == pytest.approx(-2_000.0)


def test_apply_shocks_leaves_unshocked_assets_unchanged():
    result = Stress.apply_shocks(WEIGHTS, {"AAA": -0.20}, 10_000.0)
    assert result["impact_pct"] == pytest.approx(-0.12)
    row = {r["ticker"]: r for r in result["rows"]}
    assert row["BBB"]["shock"] == 0.0
    assert row["BBB"]["impact"] == 0.0


def test_apply_shocks_positive_scenario():
    result = Stress.apply_shocks(WEIGHTS, {"AAA": 0.10, "BBB": 0.10}, 100_000.0)
    assert result["impact_pct"] == pytest.approx(0.10)
    assert result["value_after"] == pytest.approx(110_000.0)


def test_apply_shocks_requires_valid_weights():
    with pytest.raises(Stress.ShockError):
        Stress.apply_shocks({"A": 0.5}, {"A": -0.1}, 1_000.0)   # sums to 50 %
    with pytest.raises(Stress.ShockError):
        Stress.apply_shocks({}, {}, 1_000.0)


def test_apply_shocks_requires_valid_capital():
    for capital in (0.0, -1.0, float("nan")):
        with pytest.raises(Stress.ShockError):
            Stress.apply_shocks(WEIGHTS, {"AAA": -0.1}, capital)


def test_group_shocks_precedence():
    assets = [
        {"ticker": "AAA", "asset_class": "US Equity ETF"},
        {"ticker": "BBB", "asset_class": "US Stock"},
        {"ticker": "CCC", "asset_class": "Bond ETF"},
    ]
    shocks = Stress.group_shocks(
        assets,
        {"US Equity ETF": -10, "US Stock": -20},
        {"BBB": -5},
    )
    assert shocks["AAA"] == pytest.approx(-0.10)
    assert shocks["BBB"] == pytest.approx(-0.05)   # explicit asset wins
    assert "CCC" not in shocks                     # no class shock defined


def test_merge_shocks():
    merged = Stress.merge_shocks({"A": -0.1}, {"B": 5, "C": 0})
    assert merged["A"] == pytest.approx(-0.1)
    assert merged["B"] == pytest.approx(0.05)
    assert "C" not in merged                       # zero shocks are dropped


def test_assumptions_only_list_shocked_assets():
    result = Stress.apply_shocks(WEIGHTS, {"BBB": 0.05}, 1_000.0)
    assert [a["ticker"] for a in result["assumptions"]] == ["BBB"]
