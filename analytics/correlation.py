"""Correlation and dependence analysis.

``correlation_matrix`` uses **pairwise** deletion of missing observations
(each pair uses every date where both assets have a return), while
``complete_case_matrix`` uses listwise deletion. Which one is used is
reported to the UI so that the analyst knows how many observations fed
each estimate.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from analytics.returns import InsufficientDataError, _clean
from config import TRADING_DAYS


def correlation_matrix(returns: pd.DataFrame, *, min_observations: int = 2) -> pd.DataFrame:
    """Pairwise Pearson correlation matrix of asset returns."""
    if returns is None or returns.empty:
        raise InsufficientDataError("no return observations supplied")
    if returns.shape[1] < 1:
        raise InsufficientDataError("at least one asset is required")
    clean = _clean(returns)
    if clean.shape[1] == 1:
        # A single asset has a perfect (trivial) self-correlation.
        name = clean.columns[0]
        return pd.DataFrame([[1.0]], index=[name], columns=[name])
    corr = clean.corr(method="pearson", min_periods=min_observations)
    if corr.isna().all().all():
        raise InsufficientDataError("not enough overlapping observations for correlation")
    return corr


def complete_case_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Rows where *every* asset has a finite return (listwise deletion)."""
    return _clean(returns).dropna(how="any")


def covariance_matrix(returns: pd.DataFrame, *, annualize: bool = False) -> pd.DataFrame:
    """Sample covariance of daily returns (optionally annualised)."""
    clean = complete_case_matrix(returns)
    if len(clean) < 2:
        raise InsufficientDataError("at least two complete observations are required")
    cov = clean.cov(ddof=1)
    if annualize:
        cov = cov * TRADING_DAYS
    if not np.isfinite(cov.to_numpy(dtype=float)).all():
        raise InsufficientDataError("covariance matrix contains non-finite values")
    return cov


def average_pairwise_correlation(corr: pd.DataFrame) -> float:
    """Mean correlation across the off-diagonal of a correlation matrix."""
    if corr is None or corr.empty:
        raise InsufficientDataError("empty correlation matrix")
    values = corr.to_numpy(dtype=float)
    n = values.shape[0]
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    pair_values = values[mask]
    pair_values = pair_values[np.isfinite(pair_values)]
    if pair_values.size == 0:
        return 0.0
    return float(np.mean(pair_values))


def risk_contributions(
    covariance: pd.DataFrame,
    weights: Dict[str, float],
    *,
    annualize: bool = True,
) -> pd.DataFrame:
    """Absolute and percentage risk contributions of each asset.

    ``RC_i = w_i * (Σ w)_i``; percentages sum to 100 % by construction.
    """
    tickers = list(weights.keys())
    missing = [t for t in tickers if t not in covariance.columns]
    if missing:
        raise KeyError(f"missing covariance columns: {', '.join(missing)}")
    cov = covariance.loc[tickers, tickers].to_numpy(dtype=float)
    if not np.isfinite(cov).all():
        raise InsufficientDataError("covariance matrix contains non-finite values")
    w = np.array([float(weights[t]) for t in tickers], dtype=float)
    marginal = cov @ w
    contribution = w * marginal
    total = float(contribution.sum())
    if total <= 0 or not np.isfinite(total):
        pct = np.zeros_like(contribution)
    else:
        pct = contribution / total
    factor = TRADING_DAYS if annualize else 1.0
    return pd.DataFrame(
        {
            "ticker": tickers,
            "weight": w,
            "risk_contribution": contribution * factor,
            "risk_contribution_pct": pct * 100.0,
        }
    ).set_index("ticker")


def rolling_volatility(
    returns: pd.Series,
    window: int = 60,
    periods: int = TRADING_DAYS,
) -> pd.Series:
    """Rolling annualised volatility (for charting)."""
    clean = _clean(returns).dropna()
    if len(clean) < max(2, min(window, 5)):
        raise InsufficientDataError("not enough observations for a rolling window")
    return clean.rolling(window).std(ddof=1) * np.sqrt(periods)
