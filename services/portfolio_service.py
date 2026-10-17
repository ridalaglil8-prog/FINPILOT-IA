"""Portfolio service: turns session state into a fully computed context.

This is the single place where market data, weighting, currency grouping and
the analytics engine are combined. Every page (and the AI assistant) consumes
the dictionary returned by :func:`build_context`, so what the user sees and
what the model is told are always the same numbers.

Currency rule
-------------
Assets are grouped by currency. Each group is analysed as an independent
sub-portfolio (weights renormalised inside the group) and reported in its own
currency. Totals are merged **only** when an explicit FX rate is configured.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from analytics import correlation as Corr
from analytics import portfolio as Port
from analytics import risk as Risk
from analytics import returns as Ret
from config import (
    MIN_HISTORY_OBSERVATIONS,
    MIN_RELIABLE_OBSERVATIONS,
    TRADING_DAYS,
)
from services import market_data
from services import currency as FX

DEFAULT_RISK_FREE_RATE = 0.02


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _state_weights(assets: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for asset in assets:
        ticker = str(asset.get("ticker", "")).strip()
        if not ticker:
            continue
        weights[ticker] = weights.get(ticker, 0.0) + float(asset.get("weight", 0.0))
    return weights


def _finite(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan)


def _metric_block(
    wealth: pd.Series,
    returns: pd.Series,
    capital: float,
    risk_free_rate: float,
    weights: Mapping[str, float],
) -> Dict[str, float]:
    """Compute the full metric block for one currency sub-portfolio."""
    current_value = float(wealth.iloc[-1]) if len(wealth) else float(capital)
    block: Dict[str, float] = {
        "initial_capital": float(capital),
        "current_value": current_value,
        "cumulative_return": float(current_value / capital - 1.0) if capital else 0.0,
        "observations": int(len(returns)),
    }
    block["annualized_return"] = Ret.annualized_return(returns, periods=TRADING_DAYS)
    block["annualized_volatility"] = Risk.annualized_volatility(returns, periods=TRADING_DAYS)
    block["sharpe_ratio"] = Risk.sharpe_ratio(
        block["annualized_return"], block["annualized_volatility"], risk_free_rate
    )
    block["max_drawdown"] = Risk.max_drawdown(returns=returns)
    block["var_95"] = Risk.historical_var(returns, confidence=0.95)
    block["cvar_95"] = Risk.historical_cvar(returns, confidence=0.95)
    concentration = Risk.concentration_metrics(weights)
    block.update(concentration)
    return block


def _empty_context(state: Mapping[str, Any], errors: List[str]) -> Dict[str, Any]:
    return {
        "ok": False,
        "errors": errors,
        "warnings": [],
        "market": state.get("market"),
        "data_mode": state.get("data_mode"),
        "start": str(state.get("start", "")),
        "end": str(state.get("end", "")),
        "assets": [],
        "weights_pct": {},
        "allocation": [],
        "groups": [],
        "metrics": {},
        "currencies": [],
        "currency": None,
        "simulated": state.get("data_mode") == "demo",
        "data_source": "—",
        "last_observation": None,
        "prices": None,
        "asset_returns": None,
        "wealth": None,
        "benchmark_ticker": state.get("benchmark"),
        "benchmark_returns": None,
        "benchmark_metrics": {},
        "correlation": None,
        "risk_contributors": None,
        "multi_currency": False,
    }


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def build_context(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute everything the UI needs. Never raises for expected failures:
    problems are returned in ``context['errors']``."""
    assets = list(state.get("assets") or [])
    if not assets:
        return _empty_context(state, ["no_assets"])

    errors: List[str] = []
    warnings: List[str] = []

    # 1. weights -----------------------------------------------------------
    weights = _state_weights(assets)
    if len(weights) != len([a for a in assets if str(a.get("ticker", "")).strip()]):
        errors.append("duplicate_assets")
    weight_problems = Port.validate_weights(weights)
    if weight_problems:
        errors.append("invalid_weights")
        warnings.extend(weight_problems)

    total_weight = float(sum(weights.values()))
    if weight_problems and abs(total_weight) <= 0:
        return _empty_context(state, errors + ["empty_weights"])
    try:
        normalized = Port.normalize_weights(weights) if total_weight > 0 else {}
    except ValueError as exc:
        errors.append(str(exc))
        normalized = {}

    capital = float(state.get("capital", 0.0))
    if not np.isfinite(capital) or capital <= 0:
        errors.append("invalid_capital")
        capital = 0.0

    start = state.get("start")
    end = state.get("end")
    try:
        start_date = start if isinstance(start, _dt.date) else _dt.date.fromisoformat(str(start))
        end_date = end if isinstance(end, _dt.date) else _dt.date.fromisoformat(str(end))
    except ValueError:
        return _empty_context(state, errors + ["invalid_date_range"])
    if end_date < start_date:
        return _empty_context(state, errors + ["invalid_date_range"])

    market = str(state.get("market", "US"))
    data_mode = str(state.get("data_mode", "demo"))
    risk_free = float(state.get("risk_free_rate", DEFAULT_RISK_FREE_RATE))
    if not np.isfinite(risk_free):
        risk_free = DEFAULT_RISK_FREE_RATE

    if errors:
        # Stop early: analytics on an invalid portfolio would be misleading.
        return _empty_context(state, errors)

    # 2. market data -------------------------------------------------------
    tickers = list(weights.keys())
    try:
        prices, meta = market_data.fetch_prices(
            tickers, start_date, end_date,
            market=market, data_mode=data_mode,
            imported=state.get("imported_prices"),
        )
    except market_data.MarketDataError as exc:
        return _empty_context(state, errors + [str(exc)])

    prices = _finite(prices)
    missing_cols = [t for t in tickers if t not in prices.columns]
    if missing_cols:
        return _empty_context(state, [f"missing_prices: {', '.join(missing_cols)}"])

    aligned = prices[tickers].dropna(how="any")
    dropped = int(len(prices) - len(aligned))
    if dropped:
        warnings.append(f"dropped_dates:{dropped}")
    if len(aligned) < MIN_HISTORY_OBSERVATIONS + 1:
        return _empty_context(state, ["insufficient_history"])

    asset_returns = Ret.daily_returns(aligned)
    if len(asset_returns) < MIN_HISTORY_OBSERVATIONS:
        return _empty_context(state, ["insufficient_history"])

    # 3. currency grouping -------------------------------------------------
    by_currency: Dict[str, List[Dict[str, Any]]] = {}
    for asset in assets:
        currency = FX.normalize_currency(asset.get("currency"))
        by_currency.setdefault(currency, []).append(asset)

    fx_rates = FX.user_fx_rates(state.get("fx_usd_mad"))
    currencies = list(by_currency.keys())
    multi_currency = FX.needs_conversion(currencies)
    combined = FX.combine_totals(
        {c: sum(float(a.get("weight", 0.0)) for a in group) * capital
         for c, group in by_currency.items()},
        fx_rates,
    ) if multi_currency else None
    if multi_currency and combined is None:
        warnings.append("fx_missing")

    groups: List[Dict[str, Any]] = []
    for currency in currencies:
        group_assets = by_currency[currency]
        group_weights = Port.sub_weights(group_assets)          # sums to 1 inside group
        group_tickers = list(group_weights.keys())
        group_share = float(sum(float(a.get("weight", 0.0)) for a in group_assets))
        group_capital = capital * group_share

        try:
            group_returns = Ret.portfolio_returns(
                asset_returns, group_weights, min_observations=MIN_HISTORY_OBSERVATIONS
            )
        except (ValueError, Ret.InsufficientDataError) as exc:
            errors.append(f"{currency}: {exc}")
            continue

        wealth = Ret.wealth_index(group_capital, group_returns)
        metrics = _metric_block(wealth, group_returns, group_capital, risk_free, group_weights)
        groups.append({
            "currency": currency,
            "weight_pct": group_share * 100.0,
            "capital": group_capital,
            "assets": group_tickers,
            "weights": group_weights,
            "returns": group_returns,
            "wealth": wealth,
            "metrics": metrics,
        })

    if not groups:
        return _empty_context(state, errors + ["portfolio_calculation_failed"])

    groups.sort(key=lambda g: g["weight_pct"], reverse=True)
    primary = groups[0]

    # 4. benchmark ---------------------------------------------------------
    benchmark_ticker = state.get("benchmark")
    benchmark_returns = None
    benchmark_metrics: Dict[str, Any] = {}
    if benchmark_ticker:
        bench_prices, bench_meta, bench_warning = market_data.fetch_benchmark(
            benchmark_ticker, start_date, end_date,
            market=market, data_mode=data_mode,
            imported=state.get("imported_prices"),
        )
        if bench_warning:
            warnings.append(bench_warning)
        elif bench_prices is not None:
            bench_frame = _finite(bench_prices)
            # yfinance returns a DataFrame: reduce the single benchmark column
            # to a Series so downstream float() calls stay well-defined.
            if isinstance(bench_frame, pd.DataFrame):
                bench_frame = bench_frame.iloc[:, 0]
            bench_returns_all = Ret.daily_returns(bench_frame)
            benchmark_returns = bench_returns_all.reindex(primary["returns"].index).dropna()
            if len(benchmark_returns) >= 2:
                benchmark_metrics = {
                    "ticker": benchmark_ticker,
                    "cumulative_return": Ret.cumulative_returns(benchmark_returns).iloc[-1],
                    "annualized_return": Ret.annualized_return(benchmark_returns),
                    "annualized_volatility": Risk.annualized_volatility(benchmark_returns),
                    "max_drawdown": Risk.max_drawdown(returns=benchmark_returns),
                }
                if bench_meta and bench_meta.get("simulated"):
                    warnings.append("benchmark_simulated")
            else:
                benchmark_returns = None
                warnings.append(f"benchmark_overlap:{benchmark_ticker}")

    # 5. correlation & risk contributions (primary group) ------------------
    correlation_block = None
    risk_contributors = None
    primary_tickers = primary["assets"]
    corr_frame = None
    try:
        corr_frame = Corr.correlation_matrix(asset_returns[primary_tickers])
        correlation_block = {
            "labels": [str(c) for c in corr_frame.columns],
            "matrix": [[None if not np.isfinite(v) else round(float(v), 4)
                        for v in row] for row in corr_frame.to_numpy()],
            "average": round(Corr.average_pairwise_correlation(corr_frame), 4),
        }
        cov = Corr.covariance_matrix(asset_returns[primary_tickers])
        rc = Corr.risk_contributions(cov, primary["weights"], annualize=True)
        rc = rc.sort_values("risk_contribution_pct", ascending=False)
        risk_contributors = [
            {"ticker": str(idx), "weight_pct": round(float(row["weight"]) * 100, 2),
             "risk_pct": round(float(row["risk_contribution_pct"]), 2)}
            for idx, row in rc.iterrows()
        ]
    except Exception:  # correlation needs >= 2 assets / enough overlap
        warnings.append("correlation_unavailable")

    # 6. allocation --------------------------------------------------------
    last_prices = {t: float(aligned[t].iloc[-1]) for t in tickers}
    allocation = Port.allocation_table(assets, last_prices, capital)

    # 7. warnings ----------------------------------------------------------
    if data_mode == "demo":
        warnings.append("simulated_data")
    if len(asset_returns) < MIN_RELIABLE_OBSERVATIONS:
        warnings.append(f"short_history:{len(asset_returns)}")

    metrics = dict(primary["metrics"])
    if benchmark_metrics:
        metrics["benchmark_cumulative_return"] = benchmark_metrics["cumulative_return"]
        metrics["benchmark_annualized_return"] = benchmark_metrics["annualized_return"]
        metrics["benchmark_annualized_volatility"] = benchmark_metrics["annualized_volatility"]
    if correlation_block and correlation_block.get("average") is not None:
        metrics["average_correlation"] = correlation_block["average"]

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "market": market,
        "data_mode": data_mode,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "assets": tickers,
        "weights_pct": {k: v * 100.0 for k, v in weights.items()},
        "allocation": allocation,
        "groups": groups,
        "metrics": metrics,
        "currencies": currencies,
        "currency": primary["currency"],
        "multi_currency": multi_currency,
        "fx": combined,
        "fx_rates": fx_rates,
        "simulated": bool(meta.get("simulated", False)),
        "data_source": market_data.describe_source(meta),
        "data_meta": meta,
        "last_observation": meta.get("date_max"),
        "prices": aligned,
        "asset_returns": asset_returns,
        "wealth": primary["wealth"],
        "benchmark_ticker": benchmark_ticker,
        "benchmark_returns": benchmark_returns,
        "benchmark_metrics": benchmark_metrics,
        "correlation": correlation_block,
        "risk_contributors": risk_contributors,
        "risk_free_rate": risk_free,
        "warnings": warnings,
    }


def is_usable(context: Mapping[str, Any]) -> bool:
    return bool(context.get("ok")) and not context.get("errors")
