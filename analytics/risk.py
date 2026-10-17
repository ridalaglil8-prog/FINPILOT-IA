"""Risk metrics.

Formulas (all annualised with 252 trading days unless stated otherwise)
------------------------------------------------------------------------
* Annualised volatility: ``sigma_ann = std(r, ddof=1) * sqrt(252)``
* Sharpe ratio:          ``S = (r_ann - r_f) / sigma_ann``
  Both terms are annualised; the risk-free rate is an annual effective rate
  configured by the user (default 2 %).
  *Zero-volatility convention:* when ``sigma_ann <= ZERO_VOLATILITY_EPSILON``
  (1e-12 — covers constant series whose sample std is floating-point noise)
  the ratio is mathematically undefined; FinPilot AI reports ``0.0`` instead
  of ``inf`` or ``nan`` so that tables and charts stay finite.
* Maximum drawdown:      ``MDD = min_t (W_t / max_{s<=t} W_s - 1)``
  computed on the cumulative wealth path (reported as a negative number).
* Portfolio variance:    ``sigma_p^2 = w' Σ w`` with ``Σ`` the sample
  covariance matrix of daily returns (annualised by ``TRADING_DAYS``).
* Historical VaR / CVaR: empirical quantile / tail mean of the return
  distribution at the requested confidence level (losses positive).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from analytics.returns import InsufficientDataError, _clean, wealth_index
from config import TRADING_DAYS

# A sample std below this threshold is treated as exactly zero. Constant
# price series produce a tiny non-zero std from floating-point rounding, which
# would otherwise explode the Sharpe ratio (division by ~1e-20).
ZERO_VOLATILITY_EPSILON = 1e-12


def annualized_volatility(
    returns: pd.Series,
    periods: int = TRADING_DAYS,
    *,
    ddof: int = 1,
) -> float:
    """Annualised standard deviation of a return series."""
    if returns is None or len(returns) == 0:
        raise InsufficientDataError("no return observations supplied")
    if periods <= 0:
        raise ValueError("periods must be positive")
    vals = _clean(returns).dropna()
    if len(vals) < 2:
        raise InsufficientDataError("at least two return observations are required")
    std = float(vals.std(ddof=ddof))
    if not np.isfinite(std):
        raise InsufficientDataError("non-finite volatility estimate")
    if std <= ZERO_VOLATILITY_EPSILON:
        # Constant series: floating-point noise in the sample std must not
        # leak through as a huge Sharpe ratio — see the zero-vol convention.
        return 0.0
    return float(std * np.sqrt(periods))


def sharpe_ratio(
    annual_return: float,
    annual_volatility: float,
    risk_free_rate: float,
) -> float:
    """``(r_ann - r_f) / sigma_ann`` with a documented zero-vol convention."""
    for name, value in (
        ("annual_return", annual_return),
        ("annual_volatility", annual_volatility),
        ("risk_free_rate", risk_free_rate),
    ):
        if value is None or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if annual_volatility <= 0:
        return 0.0
    return float((annual_return - risk_free_rate) / annual_volatility)


def sharpe_from_returns(
    returns: pd.Series,
    risk_free_rate: float,
    periods: int = TRADING_DAYS,
) -> float:
    """Convenience wrapper: annualise return and volatility, then ratio."""
    from analytics.returns import annualized_return

    vals = _clean(returns).dropna()
    if len(vals) == 0:
        raise InsufficientDataError("no return observations supplied")
    return sharpe_ratio(
        annualized_return(vals, periods=periods),
        annualized_volatility(vals, periods=periods),
        risk_free_rate,
    )


def max_drawdown(returns: Optional[pd.Series] = None, wealth: Optional[pd.Series] = None) -> float:
    """Maximum drawdown as a negative fraction (e.g. ``-0.23`` = -23 %)."""
    if wealth is None:
        if returns is None:
            raise ValueError("either returns or wealth must be supplied")
        vals = _clean(returns).dropna()
        if len(vals) == 0:
            raise InsufficientDataError("no return observations supplied")
        wealth = (1.0 + vals).cumprod()
    else:
        wealth = pd.to_numeric(wealth, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(wealth) == 0:
            raise InsufficientDataError("no wealth observations supplied")

    values = wealth.to_numpy(dtype=float)
    if (values <= 0).any():
        # Capital wiped out at some point -> drawdown is 100 %.
        return -1.0
    running_max = np.maximum.accumulate(values)
    drawdowns = values / running_max - 1.0
    return float(np.min(drawdowns))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown path (for charting)."""
    vals = _clean(returns).dropna()
    if len(vals) == 0:
        raise InsufficientDataError("no return observations supplied")
    wealth = (1.0 + vals).cumprod()
    running_max = wealth.cummax()
    return wealth / running_max - 1.0


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR: the loss (positive number) that is not exceeded with
    the given probability."""
    if not 0.5 <= confidence < 1.0:
        raise ValueError("confidence must be in [0.5, 1.0)")
    vals = _clean(returns).dropna()
    if len(vals) < 2:
        raise InsufficientDataError("at least two return observations are required")
    quantile = float(np.quantile(vals.to_numpy(dtype=float), 1.0 - confidence))
    return float(max(-quantile, 0.0))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected shortfall: average loss beyond the historical VaR."""
    vals = _clean(returns).dropna()
    if len(vals) < 2:
        raise InsufficientDataError("at least two return observations are required")
    threshold = float(np.quantile(vals.to_numpy(dtype=float), 1.0 - confidence))
    tail = vals[vals <= threshold]
    if tail.empty:
        return 0.0
    return float(max(-float(tail.mean()), 0.0))


def portfolio_variance(covariance: pd.DataFrame, weights: Mapping[str, float]) -> float:
    """``w' Σ w`` for a covariance matrix given as a DataFrame."""
    if covariance is None or covariance.empty:
        raise InsufficientDataError("empty covariance matrix")
    tickers = list(weights.keys())
    missing = [t for t in tickers if t not in covariance.columns]
    if missing:
        raise KeyError(f"missing covariance columns: {', '.join(missing)}")
    sub = covariance.loc[tickers, tickers].to_numpy(dtype=float)
    w = np.array([float(weights[t]) for t in tickers], dtype=float)
    if not np.isfinite(sub).all():
        raise InsufficientDataError("covariance matrix contains non-finite values")
    var = float(w @ sub @ w)
    return max(var, 0.0)


def portfolio_volatility(
    covariance: pd.DataFrame,
    weights: Mapping[str, float],
    periods: int = TRADING_DAYS,
) -> float:
    """Annualised portfolio volatility ``sqrt(w'Σw * 252)`` (daily covariance)."""
    var = portfolio_variance(covariance, weights)
    return float(np.sqrt(var * periods))


def concentration_metrics(weights: Mapping[str, float]) -> Dict[str, float]:
    """Concentration of an allocation.

    * ``herfindahl`` = ``sum(w_i^2)`` in [1/n, 1]
    * ``effective_n`` = ``1 / herfindahl`` (effective number of holdings)
    * ``max_weight`` / ``top3_weight`` = largest / three largest weights
    """
    clean = {k: float(v) for k, v in weights.items() if np.isfinite(v)}
    if not clean:
        return {"herfindahl": 0.0, "effective_n": 0.0, "max_weight": 0.0,
                "top3_weight": 0.0, "n_assets": 0}
    vals = np.array(list(clean.values()), dtype=float)
    hhi = float(np.sum(vals ** 2))
    return {
        "herfindahl": hhi,
        "effective_n": float(1.0 / hhi) if hhi > 0 else 0.0,
        "max_weight": float(vals.max()),
        "top3_weight": float(np.sort(vals)[::-1][:3].sum()),
        "n_assets": int(len(vals)),
    }
