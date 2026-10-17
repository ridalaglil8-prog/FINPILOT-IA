"""Portfolio-level primitives: weights, allocation and values."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from analytics.returns import InsufficientDataError, validate_weight_map, wealth_index
from config import CURRENCY_SYMBOLS, TRADING_DAYS, WEIGHT_SUM_TOLERANCE


def validate_weights(weights: Mapping[str, float]) -> List[str]:
    """Return a list of human-readable problems (empty list == valid).

    Rules for the MVP: long-only (0 % <= w <= 100 %) and total exactly 100 %.
    """
    return validate_weight_map(weights)


def normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Scale weights so they sum to 1.0 (raises on empty/all-zero input)."""
    if not weights:
        raise ValueError("empty weight vector")
    values = np.array([float(v) for v in weights.values()], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("weights contain non-finite values")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return {k: float(v) / total for k, v in zip(weights.keys(), values)}


def percent_to_weights(weights_pct: Mapping[str, float]) -> Dict[str, float]:
    """Convert a percentage mapping (0-100) to fractional weights."""
    return {k: float(v) / 100.0 for k, v in weights_pct.items()}


def weights_to_percent(weights: Mapping[str, float]) -> Dict[str, float]:
    return {k: float(v) * 100.0 for k, v in weights.items()}


def split_by_currency(
    assets: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Mapping[str, Any]]]:
    """Group asset records by their currency (unknown currency -> "USD"? no:
    grouped under ``"UNKNOWN"`` so it can never silently mix currencies)."""
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for asset in assets:
        currency = str(asset.get("currency") or "UNKNOWN").upper()
        groups.setdefault(currency, []).append(asset)
    return groups


def group_weight_total(assets: Sequence[Mapping[str, Any]]) -> float:
    return float(sum(float(a.get("weight", 0.0)) for a in assets))


def sub_weights(assets: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Normalised weights *within* a currency group.

    Used so that each currency sub-portfolio is analysed on its own terms
    (no cross-currency arithmetic without an explicit FX rate).
    """
    raw = {str(a["ticker"]): float(a.get("weight", 0.0)) for a in assets}
    total = float(sum(raw.values()))
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return {k: v / total for k, v in raw.items()}


def allocation_table(
    assets: Sequence[Mapping[str, Any]],
    last_prices: Mapping[str, float],
    capital: float,
) -> List[Dict[str, Any]]:
    """Per-asset allocation rows: weight, monetary value, currency, price.

    Values are computed as ``capital * weight`` and are **never** summed
    across currencies by this function.
    """
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("capital must be a finite positive number")
    rows: List[Dict[str, Any]] = []
    for asset in assets:
        weight = float(asset.get("weight", 0.0))
        value = float(capital) * weight
        price = last_prices.get(str(asset["ticker"]))
        rows.append(
            {
                "ticker": str(asset["ticker"]),
                "name": asset.get("name", asset["ticker"]),
                "asset_class": asset.get("asset_class", ""),
                "currency": str(asset.get("currency", "UNKNOWN")).upper(),
                "weight": weight,
                "weight_pct": weight * 100.0,
                "value": value,
                "last_price": float(price) if price is not None and np.isfinite(price) else None,
                "value_display": format_money(value, str(asset.get("currency", ""))),
            }
        )
    return rows


def totals_by_currency(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Sum allocation values *within* each currency only."""
    totals: Dict[str, float] = {}
    for row in rows:
        currency = str(row.get("currency", "UNKNOWN"))
        totals[currency] = totals.get(currency, 0.0) + float(row.get("value", 0.0))
    return totals


def format_money(amount: float, currency: str) -> str:
    symbol = CURRENCY_SYMBOLS.get(str(currency).upper(), f"{currency} ")
    return f"{symbol}{amount:,.2f}"


def portfolio_value(capital: float, returns: pd.Series) -> float:
    """Final wealth given an initial capital and a return stream."""
    wealth = wealth_index(capital, returns)
    if len(wealth) == 0:
        raise InsufficientDataError("no return observations supplied")
    value = float(wealth.iloc[-1])
    if not np.isfinite(value):
        raise InsufficientDataError("non-finite portfolio value")
    return value


def shares_from_weights(capital: float, weights: Mapping[str, float],
                        prices: Mapping[str, float]) -> Dict[str, float]:
    """Indicative number of shares bought at the last price (informational)."""
    if not np.isfinite(capital) or capital <= 0:
        raise ValueError("capital must be a finite positive number")
    out: Dict[str, float] = {}
    for ticker, weight in weights.items():
        price = prices.get(ticker)
        if price is None or not np.isfinite(price) or price <= 0:
            raise ValueError(f"missing or invalid price for {ticker}")
        out[ticker] = float(capital) * float(weight) / float(price)
    return out


def is_valid_weight_sum(weights: Mapping[str, float], tol: float = WEIGHT_SUM_TOLERANCE) -> bool:
    if not weights:
        return False
    total = float(sum(float(v) for v in weights.values()))
    return abs(total - 1.0) <= max(tol, 1e-9)
