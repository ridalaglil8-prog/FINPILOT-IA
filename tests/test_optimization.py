"""Optimization tests: deterministic prices, known long-only solutions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimization import optimizer


def _prices(seed: int = 7, days: int = 400) -> pd.DataFrame:
    """Two synthetic assets with very different risk/return profiles.

    * LOW  — daily vol 0.3 %, positive drift
    * HIGH — daily vol 2.0 %, negative drift
    Fixed seed => deterministic output on every machine.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=days)
    low = 0.0006 + 0.003 * rng.standard_normal(days)
    high = -0.0004 + 0.02 * rng.standard_normal(days)
    return pd.DataFrame({
        "LOW": 100.0 * np.cumprod(1.0 + low),
        "HIGH": 100.0 * np.cumprod(1.0 + high),
    }, index=dates)


def test_prepare_inputs_shapes_and_finiteness():
    prices = _prices()
    mu, cov = optimizer.prepare_inputs(prices)
    assert list(mu.index) == ["LOW", "HIGH"]
    assert cov.shape == (2, 2)
    assert np.isfinite(mu.to_numpy()).all()
    assert np.isfinite(cov.to_numpy()).all()
    # HIGH is much more volatile than LOW by construction
    assert cov.loc["HIGH", "HIGH"] > 10 * cov.loc["LOW", "LOW"]


def test_prepare_inputs_rejects_short_history():
    prices = _prices(days=30)
    with pytest.raises(optimizer.OptimizationError, match="observation"):
        optimizer.prepare_inputs(prices)


def test_prepare_inputs_rejects_too_few_assets():
    with pytest.raises(optimizer.OptimizationError, match="two assets"):
        optimizer.prepare_inputs(_prices()[["LOW"]])


def test_prepare_inputs_drops_incomplete_rows():
    # a handful of gaps is fine: those rows are dropped listwise
    prices = _prices()
    prices.iloc[10, 0] = np.nan
    mu, cov = optimizer.prepare_inputs(prices)
    assert np.isfinite(mu.to_numpy()).all()


def test_prepare_inputs_rejects_too_many_missing_prices():
    prices = _prices(days=70)
    prices.iloc[:20, 0] = np.nan   # leaves 49 complete rows < 60 required
    with pytest.raises(optimizer.OptimizationError, match="observation"):
        optimizer.prepare_inputs(prices)


def test_prepare_inputs_rejects_constant_asset():
    prices = _prices()
    prices["FLAT"] = 100.0
    with pytest.raises(optimizer.OptimizationError, match="constant"):
        optimizer.prepare_inputs(prices)


def test_prepare_inputs_rejects_empty_frame():
    with pytest.raises(optimizer.OptimizationError):
        optimizer.prepare_inputs(pd.DataFrame())


def test_min_volatility_concentrates_in_low_risk_asset():
    result = optimizer.min_volatility(_prices())
    weights = result["weights"]
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert weights["LOW"] > 0.95
    assert weights["HIGH"] >= 0.0
    assert result["performance"]["volatility"] >= 0.0


def test_max_sharpe_prefers_the_positive_drift_asset():
    result = optimizer.max_sharpe(_prices())
    weights = result["weights"]
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert weights["LOW"] > 0.95


def test_weight_bounds_are_respected():
    result = optimizer.min_volatility(_prices(), weight_bounds=(0.0, 0.5))
    weights = result["weights"]
    for value in weights.values():
        assert 0.0 <= value <= 0.5 + 1e-6
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4)


def test_invalid_bounds_rejected():
    with pytest.raises(optimizer.OptimizationError):
        optimizer.optimize(_prices(), "min_volatility", weight_bounds=(-0.1, 1.0))
    with pytest.raises(optimizer.OptimizationError):
        optimizer.optimize(_prices(), "min_volatility", weight_bounds=(0.9, 0.1))


def test_unknown_objective_rejected():
    with pytest.raises(optimizer.OptimizationError):
        optimizer.optimize(_prices(), "maximum_fun")


def test_evaluate_weights_matches_manual_math():
    mu, cov = optimizer.prepare_inputs(_prices())
    result = optimizer.evaluate_weights({"LOW": 1.0, "HIGH": 0.0}, mu, cov, 0.02)
    assert result["expected_return"] == pytest.approx(float(mu["LOW"]), rel=1e-9)
    assert result["volatility"] == pytest.approx(
        float(np.sqrt(cov.loc["LOW", "LOW"] * 252)), rel=1e-9)
    assert result["sharpe"] == pytest.approx((mu["LOW"] - 0.02) / result["volatility"],
                                             rel=1e-9)


def test_evaluate_weights_unknown_asset():
    mu, cov = optimizer.prepare_inputs(_prices())
    with pytest.raises(optimizer.OptimizationError):
        optimizer.evaluate_weights({"ZZZ": 1.0}, mu, cov, 0.02)


def test_compare_allocations_columns():
    frame = optimizer.compare_allocations({"A": 0.7, "B": 0.3}, {"A": 0.4, "C": 0.6})
    assert set(frame.columns) == {"ticker", "current", "optimized", "change"}
    by_ticker = frame.set_index("ticker")
    assert by_ticker.loc["A", "current"] == pytest.approx(70.0)
    assert by_ticker.loc["A", "optimized"] == pytest.approx(40.0)
    assert by_ticker.loc["A", "change"] == pytest.approx(-30.0)
    assert by_ticker.loc["C", "current"] == pytest.approx(0.0)


def test_result_is_json_serialisable_for_the_ai_payload():
    import json

    result = optimizer.min_volatility(_prices())
    payload = {k: v for k, v in result.items()
               if k in ("objective", "engine", "weights", "performance")}
    json.dumps(payload, default=str)  # must not raise
