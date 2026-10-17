"""Risk-metric tests with hand-computed expectations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics import returns as Ret
from analytics import risk as Risk
from analytics.returns import InsufficientDataError


def test_annualized_volatility_known_value():
    # sample std (ddof=1) of [+1 %, -1 %] = sqrt(2)*1e-2 ; x sqrt(252)
    returns = pd.Series([0.01, -0.01])
    expected = np.sqrt(2) * 0.01 * np.sqrt(252)
    assert Risk.annualized_volatility(returns) == pytest.approx(expected, rel=1e-12)


def test_annualized_volatility_needs_two_observations():
    with pytest.raises(InsufficientDataError):
        Risk.annualized_volatility(pd.Series([0.01]))
    with pytest.raises(InsufficientDataError):
        Risk.annualized_volatility(pd.Series(dtype=float))


def test_annualized_volatility_rejects_bad_periods():
    with pytest.raises(ValueError):
        Risk.annualized_volatility(pd.Series([0.01, 0.02]), periods=0)


def test_sharpe_ratio_formula():
    assert Risk.sharpe_ratio(0.10, 0.20, 0.02) == pytest.approx(0.4)


def test_sharpe_ratio_zero_volatility_convention():
    assert Risk.sharpe_ratio(0.10, 0.0, 0.02) == 0.0


def test_sharpe_ratio_rejects_non_finite():
    for bad in (None, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Risk.sharpe_ratio(bad, 0.2, 0.02)


def test_sharpe_from_returns_wrapper():
    returns = pd.Series([0.01, -0.01])
    annual_return = Ret.annualized_return(returns)
    annual_vol = Risk.annualized_volatility(returns)
    expected = Risk.sharpe_ratio(annual_return, annual_vol, 0.02)
    assert Risk.sharpe_from_returns(returns, 0.02) == pytest.approx(expected, rel=1e-9)


def test_sharpe_from_returns_zero_vol_is_finite():
    assert Risk.sharpe_from_returns(pd.Series([0.01] * 10), 0.02) == 0.0


def test_max_drawdown_from_returns():
    # wealth path: 1.10 -> 0.55 -> 0.825 ; peak 1.10 -> deepest dip -50 %
    returns = pd.Series([0.10, -0.50, 0.50])
    assert Risk.max_drawdown(returns=returns) == pytest.approx(-0.50)


def test_max_drawdown_from_wealth_path():
    wealth = pd.Series([100.0, 120.0, 90.0, 110.0])
    assert Risk.max_drawdown(wealth=wealth) == pytest.approx(-0.25)


def test_max_drawdown_monotonic_series_is_zero():
    returns = pd.Series([0.01, 0.01, 0.01])
    assert Risk.max_drawdown(returns=returns) == pytest.approx(0.0)


def test_max_drawdown_total_loss():
    returns = pd.Series([-1.0])
    assert Risk.max_drawdown(returns=returns) == -1.0


def test_max_drawdown_requires_input():
    with pytest.raises(ValueError):
        Risk.max_drawdown()


def test_drawdown_series_shape():
    returns = pd.Series([0.10, -0.50, 0.50])
    series = Risk.drawdown_series(returns)
    assert float(series.iloc[1]) == pytest.approx(-0.50)
    assert float(series.iloc[-1]) == pytest.approx(-0.25)
    assert float(series.max()) == pytest.approx(0.0)


def test_historical_var_and_cvar():
    returns = pd.Series([0.01, 0.02, -0.03, -0.04, 0.0])
    var = Risk.historical_var(returns, confidence=0.8)
    cvar = Risk.historical_cvar(returns, confidence=0.8)
    # 20th percentile of the sorted sample, linearly interpolated
    assert var == pytest.approx(0.032, abs=1e-12)
    # the only observation at or below that threshold is -4 %
    assert cvar == pytest.approx(0.04, abs=1e-12)
    assert cvar >= var  # expected shortfall is at least the VaR


def test_var_confidence_validation():
    with pytest.raises(ValueError):
        Risk.historical_var(pd.Series([0.01, 0.02]), confidence=1.0)
    with pytest.raises(ValueError):
        Risk.historical_var(pd.Series([0.01, 0.02]), confidence=0.2)


def test_portfolio_variance_and_volatility():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=["A", "B"], columns=["A", "B"])
    weights = {"A": 0.5, "B": 0.5}
    # w'Σw = 0.25*0.04 + 0.25*0.04 = 0.02 (uncorrelated assets)
    assert Risk.portfolio_variance(cov, weights) == pytest.approx(0.02)
    assert Risk.portfolio_volatility(cov, weights) == pytest.approx(np.sqrt(0.02 * 252))


def test_portfolio_variance_full_correlation_adds_up():
    cov = pd.DataFrame([[0.04, 0.04], [0.04, 0.04]], index=["A", "B"], columns=["A", "B"])
    # perfectly correlated: variance equals the single-asset variance
    assert Risk.portfolio_variance(cov, {"A": 0.5, "B": 0.5}) == pytest.approx(0.04)


def test_portfolio_variance_missing_column():
    cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    with pytest.raises(KeyError):
        Risk.portfolio_variance(cov, {"ZZZ": 1.0})


def test_portfolio_variance_empty_matrix():
    with pytest.raises(InsufficientDataError):
        Risk.portfolio_variance(pd.DataFrame(), {"A": 1.0})


def test_portfolio_variance_negative_is_clamped():
    cov = pd.DataFrame([[-0.04]], index=["A"], columns=["A"])
    assert Risk.portfolio_variance(cov, {"A": 1.0}) == 0.0


def test_concentration_metrics():
    metrics = Risk.concentration_metrics({"A": 0.5, "B": 0.5})
    assert metrics["herfindahl"] == pytest.approx(0.5)
    assert metrics["effective_n"] == pytest.approx(2.0)
    assert metrics["max_weight"] == pytest.approx(0.5)
    assert metrics["top3_weight"] == pytest.approx(1.0)
    assert metrics["n_assets"] == 2


def test_concentration_metrics_empty():
    metrics = Risk.concentration_metrics({})
    assert metrics["n_assets"] == 0
    assert metrics["effective_n"] == 0.0


def test_concentration_metrics_ignores_non_finite():
    metrics = Risk.concentration_metrics({"A": 0.6, "B": 0.4, "C": float("nan")})
    assert metrics["n_assets"] == 2
    assert metrics["herfindahl"] == pytest.approx(0.52)
