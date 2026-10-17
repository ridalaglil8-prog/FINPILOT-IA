"""Portfolio optimisation.

Primary engine: **PyPortfolioOpt** (Efficient Frontier, long-only bounds).
If PyPortfolioOpt/cvxpy is unavailable or its solver fails, a documented
SciPy SLSQP fallback with identical constraints is used, and the caller is
told via ``engine`` in the result so the UI can disclose it.

Inputs are validated before optimisation:

* at least ``MIN_OPTIMIZATION_OBSERVATIONS`` complete (aligned) price rows
* no missing/non-finite prices in the optimisation window
* every asset must have a strictly positive, non-zero variance
  (a constant asset makes the covariance matrix singular)

Results are normalised (weights sum to exactly 1.0, clipped to bounds).

DISCLAIMER: optimisation uses *historical* estimates (mean returns and a
sample covariance matrix) under strong assumptions. It does NOT guarantee
better future performance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from analytics.correlation import covariance_matrix
from config import MIN_OPTIMIZATION_OBSERVATIONS, TRADING_DAYS

ENGINE_PYPFOPT = "PyPortfolioOpt"
ENGINE_SCIPY = "SciPy SLSQP (fallback)"

DEFAULT_RISK_FREE_RATE = 0.02
MIN_WEIGHT = 0.0
MAX_WEIGHT = 1.0


class OptimizationError(ValueError):
    """Optimisation cannot be performed; message is safe for end users."""


def _try_import_pyportfolioopt() -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    try:
        from pypfopt import EfficientFrontier, expected_returns, risk_models  # type: ignore
        return EfficientFrontier, (expected_returns, risk_models), None
    except Exception as exc:  # ImportError or a broken cvxpy install
        return None, None, f"{type(exc).__name__}: {exc}"


def prepare_inputs(
    prices: pd.DataFrame,
    *,
    min_observations: int = MIN_OPTIMIZATION_OBSERVATIONS,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Validate prices and return ``(expected_returns, covariance_matrix)``.

    ``expected_returns`` are annualised historical (arithmetic) means;
    ``covariance_matrix`` is the daily sample covariance.
    """
    if prices is None or not isinstance(prices, pd.DataFrame) or prices.empty:
        raise OptimizationError("no price history available for optimisation")
    if prices.shape[1] < 2:
        raise OptimizationError("optimisation requires at least two assets")

    numeric = prices.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(how="any")
    if numeric.empty:
        raise OptimizationError(
            "no date has a complete set of prices for every selected asset"
        )
    if len(numeric) < min_observations:
        raise OptimizationError(
            f"only {len(numeric)} complete observation(s); optimisation requires "
            f"at least {min_observations} trading days of aligned history"
        )

    constant = [c for c in numeric.columns if float(numeric[c].std(ddof=1)) <= 0]
    if constant:
        raise OptimizationError(
            "these assets have a constant (zero-variance) price series and cannot "
            f"be optimised: {', '.join(map(str, constant))}"
        )

    daily_returns = numeric.pct_change(fill_method=None).iloc[1:]
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(daily_returns) < max(2, min_observations - 1):
        raise OptimizationError("not enough return observations after alignment")

    mu = daily_returns.mean() * TRADING_DAYS
    cov = daily_returns.cov(ddof=1)
    if not np.isfinite(mu.to_numpy()).all():
        raise OptimizationError("non-finite expected return estimates")
    if not np.isfinite(cov.to_numpy()).all():
        raise OptimizationError("non-finite covariance estimates")
    return mu, cov


def _clean_weights(weights: Mapping[str, float],
                   bound: Tuple[float, float] = (MIN_WEIGHT, MAX_WEIGHT)) -> Dict[str, float]:
    cleaned = {k: float(np.clip(v, *bound)) for k, v in weights.items()}
    total = float(sum(cleaned.values()))
    if total <= 0:
        raise OptimizationError("the optimiser returned an empty portfolio")
    cleaned = {k: v / total for k, v in cleaned.items()}
    # Kill floating point dust so displayed weights sum to 100 % exactly.
    return {k: round(v, 6) for k, v in cleaned.items()}


def evaluate_weights(
    weights: Mapping[str, float],
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Dict[str, float]:
    """Historical expected return, volatility and Sharpe of ``weights``."""
    tickers = [t for t in weights if t in mu.index]
    if len(tickers) != len(list(weights)):
        raise OptimizationError("weights reference assets missing from the estimates")
    w = np.array([float(weights[t]) for t in tickers], dtype=float)
    if w.sum() <= 0:
        raise OptimizationError("weights must sum to a positive number")
    w = w / w.sum()
    returns_vector = mu.loc[tickers].to_numpy(dtype=float)
    cov_matrix = cov.loc[tickers, tickers].to_numpy(dtype=float)
    if not np.isfinite(returns_vector).all() or not np.isfinite(cov_matrix).all():
        raise OptimizationError("estimates contain non-finite values")

    expected = float(w @ returns_vector)
    variance = float(w @ cov_matrix @ w)
    volatility = float(np.sqrt(max(variance, 0.0) * TRADING_DAYS))
    if volatility <= 0:
        sharpe = 0.0
    else:
        sharpe = float((expected - risk_free_rate) / volatility)
    return {
        "expected_return": expected,
        "volatility": volatility,
        "sharpe": sharpe,
        "weights": {t: float(v) for t, v in zip(tickers, w)},
    }


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
def _optimize_pyportfolioopt(
    objective: str,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
    weight_bounds: Tuple[float, float],
) -> Optional[Dict[str, float]]:
    EfficientFrontier, modules, error = _try_import_pyportfolioopt()
    if EfficientFrontier is None:
        return None
    expected_returns, risk_models = modules
    try:
        # Use our own validated estimates (identical methodology).
        ef = EfficientFrontier(mu.to_numpy(dtype=float),
                               cov.to_numpy(dtype=float),
                               weight_bounds=weight_bounds)
        if objective == "max_sharpe":
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        else:
            ef.min_volatility()
        raw = ef.clean_weights()
        labels = [str(c) for c in mu.index]
        weights = {label: float(raw[i]) for i, label in enumerate(labels)}
        return weights
    except Exception:
        # Solver failure (infeasible max_sharpe with negative expected returns,
        # missing solver, ...) -> caller falls back to SciPy.
        return None


def _optimize_scipy(
    objective: str,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
    weight_bounds: Tuple[float, float],
) -> Dict[str, float]:
    from scipy.optimize import minimize

    labels = [str(c) for c in mu.index]
    m = len(labels)
    returns_vector = mu.to_numpy(dtype=float)
    cov_matrix = cov.to_numpy(dtype=float)
    bounds = [weight_bounds] * m
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    x0 = np.full(m, 1.0 / m)

    if objective == "max_sharpe":
        def objective_fn(w: np.ndarray) -> float:
            ret = float(w @ returns_vector)
            vol = float(np.sqrt(max(w @ cov_matrix @ w, 0.0)))
            if vol <= 1e-12:
                return 1e6
            return -float((ret - risk_free_rate) / vol)
    else:
        def objective_fn(w: np.ndarray) -> float:
            return float(w @ cov_matrix @ w)

    result = minimize(
        objective_fn, x0, method="SLSQP", bounds=bounds,
        constraints=constraints, options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success and not np.isfinite(result.fun):
        raise OptimizationError(
            f"the optimiser failed to converge ({result.message})"
        )
    weights = {label: float(value) for label, value in zip(labels, result.x)}
    return weights


def optimize(
    prices: pd.DataFrame,
    objective: str = "min_volatility",
    *,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    weight_bounds: Tuple[float, float] = (MIN_WEIGHT, MAX_WEIGHT),
    min_observations: int = MIN_OPTIMIZATION_OBSERVATIONS,
) -> Dict[str, Any]:
    """Run a long-only optimisation over ``prices``.

    Returns a dict with normalised ``weights``, the ``objective``, the
    ``engine`` actually used and the historical ``performance`` of the result.
    """
    if objective not in ("min_volatility", "max_sharpe"):
        raise OptimizationError(f"unknown objective: {objective}")
    if weight_bounds[0] < 0 or weight_bounds[1] > 1 or weight_bounds[0] >= weight_bounds[1]:
        raise OptimizationError("invalid weight bounds (long-only expects 0 <= w <= 1)")

    mu, cov = prepare_inputs(prices, min_observations=min_observations)

    engine = ENGINE_PYPFOPT
    weights = _optimize_pyportfolioopt(objective, mu, cov, risk_free_rate, weight_bounds)
    if weights is None:
        engine = ENGINE_SCIPY
        weights = _optimize_scipy(objective, mu, cov, risk_free_rate, weight_bounds)

    weights = _clean_weights(weights, bound=weight_bounds)
    performance = evaluate_weights(weights, mu, cov, risk_free_rate)
    return {
        "objective": objective,
        "engine": engine,
        "weights": weights,
        "performance": performance,
        "expected_returns": mu.to_dict(),
        "risk_free_rate": risk_free_rate,
        "observations": int(len(prices.dropna(how="any"))),
    }


def min_volatility(prices: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    return optimize(prices, "min_volatility", **kwargs)


def max_sharpe(prices: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
    return optimize(prices, "max_sharpe", **kwargs)


def compare_allocations(
    current: Mapping[str, float],
    proposed: Mapping[str, float],
) -> pd.DataFrame:
    """Side-by-side allocation comparison (current vs optimised)."""
    tickers = sorted(set(current) | set(proposed))
    rows = []
    for ticker in tickers:
        cur = float(current.get(ticker, 0.0))
        new = float(proposed.get(ticker, 0.0))
        rows.append({
            "ticker": ticker,
            "current": cur * 100.0,
            "optimized": new * 100.0,
            "change": (new - cur) * 100.0,
        })
    return pd.DataFrame(rows).sort_values("optimized", ascending=False).reset_index(drop=True)
