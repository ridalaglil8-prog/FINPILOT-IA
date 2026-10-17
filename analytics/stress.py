"""Hypothetical stress scenarios.

A shock is a *percentage* change applied to an asset's price, expressed as a
decimal (``-0.10`` = a 10 % fall). The estimated impact on a long-only
portfolio is:

    impact_pct = sum_i ( w_i * shock_i )
    value_after = capital * (1 + impact_pct)

This is an instantaneous, first-order approximation: it assumes weights are
applied to the pre-shock portfolio and ignores correlations between the
shocked assets, rebalancing, and second-order effects. Results are labelled
as **hypothetical scenarios, never forecasts**.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


class ShockError(ValueError):
    """Invalid shock definition."""


def normalize_shock(value: float) -> float:
    """Accept ``-10`` (percent) or ``-0.10`` (decimal) and return decimal."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ShockError("shock must be a number") from exc
    if not np.isfinite(v):
        raise ShockError("shock must be a finite number")
    if abs(v) > 1.5:  # clearly given in percent
        v = v / 100.0
    if v < -1.0:
        raise ShockError("a shock cannot lose more than 100 % of an asset")
    if v > 5.0:
        raise ShockError("shock is unreasonably large")
    return float(v)


def group_shocks(
    assets: Sequence[Mapping[str, Any]],
    asset_class_shocks: Mapping[str, float],
    asset_shocks: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Expand per-asset-class shocks to per-asset shocks.

    Explicit per-asset shocks take precedence over the group shock.
    """
    shocks: Dict[str, float] = {}
    asset_shocks = asset_shocks or {}
    for asset in assets:
        ticker = str(asset["ticker"])
        if ticker in asset_shocks:
            shocks[ticker] = normalize_shock(asset_shocks[ticker])
            continue
        asset_class = str(asset.get("asset_class", ""))
        if asset_class in asset_class_shocks and asset_class_shocks[asset_class] not in (None, 0):
            shocks[ticker] = normalize_shock(asset_class_shocks[asset_class])
    return shocks


def apply_shocks(
    weights: Mapping[str, float],
    shocks: Mapping[str, float],
    capital: float,
    *,
    currencies: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Apply shocks to a *single-currency* portfolio.

    Returns per-asset rows, the total percentage impact, the monetary
    impact and the value after the scenario. Assets without a shock are
    assumed unchanged (shock = 0).
    """
    if not weights:
        raise ShockError("the portfolio is empty")
    if not np.isfinite(capital) or capital <= 0:
        raise ShockError("capital must be a finite positive number")
    total_weight = float(sum(float(w) for w in weights.values()))
    if abs(total_weight - 1.0) > 1e-6:
        raise ShockError(
            f"weights total {total_weight * 100:.2f} % — they must total 100 % "
            "before running a scenario"
        )

    rows: List[Dict[str, Any]] = []
    impact_pct = 0.0
    for ticker, weight in weights.items():
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0:
            raise ShockError(f"invalid weight for {ticker}")
        raw = shocks.get(ticker, 0.0)
        shock = normalize_shock(raw) if raw not in (None, "") else 0.0
        before = float(capital) * weight
        after = before * (1.0 + shock)
        delta = after - before
        impact_pct += weight * shock
        rows.append(
            {
                "ticker": ticker,
                "currency": (currencies or {}).get(ticker, ""),
                "weight": weight,
                "shock": shock,
                "value_before": before,
                "value_after": after,
                "impact": delta,
            }
        )

    value_after = float(capital) * (1.0 + impact_pct)
    return {
        "capital": float(capital),
        "impact_pct": float(impact_pct),
        "impact_value": value_after - float(capital),
        "value_after": value_after,
        "rows": rows,
        "assumptions": [
            {"ticker": r["ticker"], "shock": r["shock"]} for r in rows if r["shock"] != 0.0
        ],
    }


def merge_shocks(
    asset_shocks: Mapping[str, float],
    extra: Mapping[str, float],
) -> Dict[str, float]:
    """Merge two shock maps (``extra`` wins) validating every entry."""
    merged = dict(asset_shocks or {})
    for key, value in (extra or {}).items():
        merged[key] = normalize_shock(value)
    return {k: v for k, v in merged.items() if v != 0.0}
