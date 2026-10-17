"""Returns engine tests — deterministic inputs, hand-computed expectations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics import returns as Ret
from analytics.returns import InsufficientDataError


def _prices(values, dates=None):
    index = dates or pd.bdate_range("2023-01-02", periods=len(values))
    return pd.Series(values, index=index, dtype=float)


def test_daily_returns_basic():
    prices = _prices([100.0, 110.0, 99.0])
    result = Ret.daily_returns(prices)
    assert len(result) == 2
    assert result.iloc[0] == pytest.approx(0.10)
    assert result.iloc[1] == pytest.approx(-0.10)


def test_daily_returns_drops_first_observation_and_nan():
    # A NaN price breaks the return on both sides: no value is ever filled in.
    prices = _prices([100.0, np.nan, 120.0, 132.0])
    result = Ret.daily_returns(prices)
    # only the clean 120 -> 132 step survives (+10 %)
    assert len(result) == 1
    assert result.iloc[0] == pytest.approx(0.10)


def test_daily_returns_requires_two_observations():
    with pytest.raises(InsufficientDataError):
        Ret.daily_returns(_prices([100.0]))
    with pytest.raises(InsufficientDataError):
        Ret.daily_returns(pd.Series(dtype=float))


def test_cumulative_returns_compounding():
    returns = pd.Series([0.10, -0.10])
    cumulative = Ret.cumulative_returns(returns)
    assert float(cumulative.iloc[-1]) == pytest.approx(1.1 * 0.9 - 1.0)


def test_wealth_index_from_capital():
    returns = pd.Series([0.10, -0.10])
    wealth = Ret.wealth_index(1000.0, returns)
    assert float(wealth.iloc[0]) == pytest.approx(1100.0)
    assert float(wealth.iloc[1]) == pytest.approx(990.0)


def test_wealth_index_rejects_invalid_capital():
    returns = pd.Series([0.01, 0.02])
    for capital in (0.0, -5.0, float("nan")):
        with pytest.raises(ValueError):
            Ret.wealth_index(capital, returns)


def test_annualized_return_geometric_convention():
    # 252 daily returns of exactly +1 % -> CAGR = 1.01^252 - 1
    returns = pd.Series([0.01] * 252)
    expected = (1.01 ** 252) - 1.0
    assert Ret.annualized_return(returns) == pytest.approx(expected, rel=1e-9)


def test_annualized_return_one_trading_day():
    # a single +5 % day annualises to (1.05)^252 - 1
    returns = pd.Series([0.05])
    assert Ret.annualized_return(returns) == pytest.approx((1.05 ** 252) - 1.0, rel=1e-9)


def test_annualized_return_total_loss_is_minus_one():
    returns = pd.Series([-1.0, 0.1])
    assert Ret.annualized_return(returns) == -1.0


def test_annualized_return_rejects_bad_periods():
    with pytest.raises(ValueError):
        Ret.annualized_return(pd.Series([0.01, 0.01]), periods=0)


def test_portfolio_returns_weighted_average():
    frame = pd.DataFrame({"A": [0.10, -0.05], "B": [0.00, 0.20]})
    result = Ret.portfolio_returns(frame, {"A": 0.6, "B": 0.4})
    assert float(result.iloc[0]) == pytest.approx(0.06)
    assert float(result.iloc[1]) == pytest.approx(0.05)


def test_portfolio_returns_rejects_invalid_weights():
    frame = pd.DataFrame({"A": [0.1, 0.1], "B": [0.1, 0.1]})
    with pytest.raises(ValueError):
        Ret.portfolio_returns(frame, {"A": 0.7, "B": 0.7})      # sums to 140 %
    with pytest.raises(ValueError):
        Ret.portfolio_returns(frame, {"A": -0.5, "B": 1.5})     # short position
    with pytest.raises(ValueError):
        Ret.portfolio_returns(frame, {})                        # empty


def test_portfolio_returns_unknown_asset():
    frame = pd.DataFrame({"A": [0.1, 0.1]})
    with pytest.raises(KeyError):
        Ret.portfolio_returns(frame, {"ZZZ": 1.0})


def test_portfolio_returns_enforces_min_observations():
    frame = pd.DataFrame({"A": [0.1], "B": [0.2]})
    with pytest.raises(InsufficientDataError):
        Ret.portfolio_returns(frame, {"A": 0.5, "B": 0.5}, min_observations=5)


def test_portfolio_returns_drops_incomplete_rows():
    frame = pd.DataFrame({"A": [0.1, np.nan, 0.3], "B": [0.2, 0.2, np.inf]})
    result = Ret.portfolio_returns(frame, {"A": 0.5, "B": 0.5})
    # only the first row is complete and finite
    assert len(result) == 1
    assert float(result.iloc[0]) == pytest.approx(0.15)


def test_validate_weight_map_messages():
    problems = Ret.validate_weight_map({"A": 1.2})
    assert any("exceeds" in p for p in problems)
    problems = Ret.validate_weight_map({"A": 0.4, "B": 0.5})
    assert any("100" in p for p in problems)
    assert Ret.validate_weight_map({"A": 1.0}) == []


def test_returns_summary_bundle():
    returns = pd.Series([0.01, -0.02, 0.03])
    summary = Ret.returns_summary(returns)
    assert summary["observations"] == 3
    assert summary["cumulative_return"] == pytest.approx((1.01 * 0.98 * 1.03) - 1.0)
    assert summary["annualized_volatility"] > 0
