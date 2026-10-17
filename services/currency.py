"""Currency handling.

FinPilot AI refuses to mix USD and MAD unless an explicit, documented FX rate
is available (provided by the user in Settings or through the ``FX_USD_MAD``
environment variable). Without it, every calculation stays inside a single
currency and totals are reported per currency.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

KNOWN_CURRENCIES = ("USD", "MAD")


class FXUnavailableError(ValueError):
    """Raised when a conversion was requested without an explicit rate."""


def normalize_currency(currency: Optional[str]) -> str:
    return str(currency or "UNKNOWN").upper()


def currencies_of(assets: Iterable[Mapping[str, Any]]) -> List[str]:
    seen: List[str] = []
    for asset in assets:
        currency = normalize_currency(asset.get("currency"))
        if currency not in seen:
            seen.append(currency)
    return seen


def needs_conversion(currencies: Iterable[str]) -> bool:
    """True when more than one currency is present (i.e. totals cannot be merged)."""
    distinct = {normalize_currency(c) for c in currencies}
    return len(distinct) > 1


def fx_rate(source: str, target: str, rates: Optional[Mapping[str, float]]) -> Optional[float]:
    """Return the rate ``source -> target`` or ``None`` if unavailable.

    ``rates`` is keyed by currency pair: ``{"USD_MAD": 10.1, "MAD_USD": 0.099}``.
    """
    source, target = normalize_currency(source), normalize_currency(target)
    if source == target:
        return 1.0
    if not rates:
        return None
    direct = rates.get(f"{source}_{target}")
    for candidate in (direct, rates.get(f"{source}{target}")):
        if candidate is None:
            continue
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            return value
    inverse = rates.get(f"{target}_{source}") or rates.get(f"{target}{source}")
    try:
        inverse_value = float(inverse)
    except (TypeError, ValueError):
        return None
    if np.isfinite(inverse_value) and inverse_value > 0:
        return 1.0 / inverse_value
    return None


def convert(amount: float, source: str, target: str,
            rates: Optional[Mapping[str, float]]) -> float:
    """Convert ``amount`` or raise :class:`FXUnavailableError`."""
    rate = fx_rate(source, target, rates)
    if rate is None:
        raise FXUnavailableError(
            f"no FX rate available for {normalize_currency(source)} -> "
            f"{normalize_currency(target)}. Provide one in Settings to combine totals."
        )
    return float(amount) * rate


def combine_totals(
    totals: Mapping[str, float],
    rates: Optional[Mapping[str, float]],
    *,
    target: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Combine per-currency totals into one figure when every required rate
    exists. Returns ``None`` when a rate is missing (caller must then display
    the totals separately — never guess a conversion)."""
    if not totals:
        return None
    currencies = list(totals.keys())
    if len(currencies) == 1:
        currency = currencies[0]
        return {"currency": currency, "total": float(totals[currency]),
                "converted": dict(totals), "rates_used": {}}
    target = normalize_currency(target or currencies[0])
    converted: Dict[str, float] = {}
    rates_used: Dict[str, float] = {}
    for currency in currencies:
        rate = fx_rate(currency, target, rates)
        if rate is None:
            return None
        converted[currency] = float(totals[currency]) * rate
        rates_used[f"{currency}_{target}"] = rate
    return {
        "currency": target,
        "total": float(sum(converted.values())),
        "converted": converted,
        "rates_used": rates_used,
    }


def user_fx_rates(fx_usd_mad: Optional[float]) -> Dict[str, float]:
    """Build a rate table from the single user-provided USD/MAD quote."""
    rates: Dict[str, float] = {}
    if fx_usd_mad is None:
        return rates
    try:
        value = float(fx_usd_mad)
    except (TypeError, ValueError):
        return rates
    if np.isfinite(value) and value > 0:
        rates["USD_MAD"] = value
        rates["MAD_USD"] = 1.0 / value
    return rates


def format_amount(amount: float, currency: str) -> str:
    from config import CURRENCY_SYMBOLS

    symbol = CURRENCY_SYMBOLS.get(normalize_currency(currency), f"{normalize_currency(currency)} ")
    return f"{symbol}{float(amount):,.2f}"
