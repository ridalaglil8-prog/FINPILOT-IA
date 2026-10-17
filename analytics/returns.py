"""Return calculations.

Conventions
-----------
* Daily (simple) returns:  ``r_t = P_t / P_{t-1} - 1``
* Cumulative return:       ``R_T = prod(1 + r_t) - 1``
* Wealth index:            ``W_t = W_0 * prod_{s<=t}(1 + r_s)``
* Annualised return:       ``r_ann = (1 + R_T) ** (252 / n) - 1``
  (geometric / CAGR convention, ``n`` = number of daily observations)
* Portfolio return:        ``r_p = sum_i w_i * r_i`` with weights summing to 1.

Data handling
-------------
Missing prices are **never** silently filled. Rows containing non-finite
values for at least one selected asset are dropped (listwise deletion) and the
number of dropped rows is reported to the caller so the UI can warn about it.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Union

import numpy as np
import pandas as pd

from config import TRADING_DAYS, WEIGHT_SUM_TOLERANCE

SeriesOrFrame = Union[pd.Series, pd.DataFrame]


class InsufficientDataError(ValueError):
    """Raised when there is not enough data to compute a statistic."""


def _as_frame(data: SeriesOrFrame) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        return data.to_frame()
    if isinstance(data, pd.DataFrame):
        return data
    raise TypeError("expected a pandas Series or DataFrame")


def _clean(values: SeriesOrFrame) -> SeriesOrFrame:
    """Coerce to float and mask non-finite values as NaN (no filling)."""
    if isinstance(values, pd.DataFrame):
        out = values.apply(pd.to_numeric, errors="coerce")
        return out.replace([np.inf, -np.inf], np.nan)
    out = pd.to_numeric(values, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def daily_returns(prices: SeriesOrFrame, *, dropna: bool = True) -> SeriesOrFrame:
    """Simple period-over-period returns of a price series/frame.

    The first observation has no previous price and is therefore dropped.
    """
    if prices is None or len(prices) == 0:
        raise InsufficientDataError("no price observations supplied")
    clean = _clean(prices)
    rets = clean.pct_change(fill_method=None)
    rets = rets.iloc[1:]
    if dropna:
        rets = rets.dropna(how="all") if isinstance(rets, pd.DataFrame) else rets.dropna()
    if len(rets) == 0:
        raise InsufficientDataError("fewer than two price observations supplied")
    return rets


def cumulative_returns(returns: SeriesOrFrame) -> SeriesOrFrame:
    """Cumulative (compounded) return from a return series."""
    if returns is None or len(returns) == 0:
        raise InsufficientDataError("no return observations supplied")
    clean = _clean(returns)
    if isinstance(clean, pd.DataFrame):
        return (1.0 + clean.fillna(0.0)).cumprod() - 1.0
    return (1.0 + clean.fillna(0.0)).cumprod() - 1.0


def wealth_index(capital: float, returns: pd.Series) -> pd.Series:
    """Portfolio wealth path starting at ``capital``."""
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("capital must be a finite positive number")
    if returns is None or len(returns) == 0:
        raise InsufficientDataError("no return observations supplied")
    clean = _clean(returns).fillna(0.0)
    return float(capital) * (1.0 + clean).cumprod()


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Geometric annualised return: ``(1+R)**(periods/n) - 1``."""
    if returns is None or len(returns) == 0:
        raise InsufficientDataError("no return observations supplied")
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            raise TypeError("annualized_return expects a single return series")
        returns = returns.iloc[:, 0]
    vals = _clean(returns).dropna()
    n = int(len(vals))
    if n == 0:
        raise InsufficientDataError("no finite return observations supplied")
    if periods <= 0:
        raise ValueError("periods must be positive")
    total = float((1.0 + vals).prod())
    if not np.isfinite(total):
        raise InsufficientDataError("non-finite cumulative return")
    if total <= 0:
        # Complete loss of capital: the geometric return is bounded at -100 %.
        return -1.0
    return float(total ** (periods / n) - 1.0)


def aligned_returns(
    asset_returns: pd.DataFrame,
    tickers: Iterable[str],
) -> pd.DataFrame:
    """Select ``tickers`` and drop rows with any missing/non-finite value.

    Returns an empty frame if nothing usable remains.
    """
    tickers = [str(t) for t in tickers]
    missing = [t for t in tickers if t not in asset_returns.columns]
    if missing:
        raise KeyError(f"unknown assets: {', '.join(missing)}")
    sub = _clean(asset_returns[tickers])
    return sub.dropna(how="any")


def portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    min_observations: int = 1,
) -> pd.Series:
    """Weighted portfolio return series ``r_p = sum_i w_i r_i``.

    * weights must reference existing columns, be finite, non-negative and
      sum to 1 (within ``WEIGHT_SUM_TOLERANCE``);
    * rows with a missing value for any selected asset are dropped;
    * ``InsufficientDataError`` is raised when fewer than
      ``min_observations`` complete rows remain.
    """
    if not weights:
        raise ValueError("empty weight vector")
    problems = validate_weight_map(weights)
    if problems:
        raise ValueError("; ".join(problems))

    tickers = list(weights.keys())
    w = np.array([float(weights[t]) for t in tickers], dtype=float)
    aligned = aligned_returns(asset_returns, tickers)
    if len(aligned) < min_observations:
        raise InsufficientDataError(
            f"only {len(aligned)} complete return observations "
            f"(minimum {min_observations} required)"
        )
    values = aligned.to_numpy(dtype=float) @ w
    out = pd.Series(values, index=aligned.index, name="portfolio")
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def validate_weight_map(weights: Mapping[str, float]) -> List[str]:
    """Validate a ``{ticker: weight}`` map (weights in [0, 1], sum == 1)."""
    problems: List[str] = []
    if not weights:
        return ["portfolio is empty"]
    for key, value in weights.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            problems.append(f"weight for {key} is not a number")
            continue
        if not np.isfinite(v):
            problems.append(f"weight for {key} is not finite")
        elif v < 0:
            problems.append(f"weight for {key} is negative (long-only portfolios)")
        elif v > 1 + WEIGHT_SUM_TOLERANCE:
            problems.append(f"weight for {key} exceeds 100 %")
    if not problems:
        total = float(sum(float(v) for v in weights.values()))
        if abs(total - 1.0) > 1e-6:
            problems.append(f"weights total {total * 100:.4f} % instead of 100 %")
    return problems


def returns_summary(returns: pd.Series, periods: int = TRADING_DAYS) -> Dict[str, float]:
    """Small convenience bundle used by tests and the AI payload builder."""
    from analytics.risk import annualized_volatility, max_drawdown  # local import

    vals = _clean(returns).dropna()
    if len(vals) == 0:
        raise InsufficientDataError("no return observations supplied")
    cum = float((1.0 + vals).prod() - 1.0)
    return {
        "observations": int(len(vals)),
        "cumulative_return": cum,
        "annualized_return": annualized_return(vals, periods=periods),
        "annualized_volatility": annualized_volatility(vals, periods=periods),
        "max_drawdown": max_drawdown(vals),
    }
