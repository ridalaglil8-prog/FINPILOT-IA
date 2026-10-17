"""Market-data facade used by the UI.

Responsibilities
----------------
* pick the right provider for ``(market, data_mode)``
* cache network/synthetic fetches with ``st.cache_data``
* attach provenance metadata (source, date range, currency, warnings)
* propagate :class:`MarketDataError` — a failed *real* request is never
  silently replaced by simulated data

Demo mode is deterministic and offline; it never touches the network.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from config import MARKET_MA, MARKET_US, MODE_DEMO, MODE_HISTORICAL
from data import demo_data
from data.providers import morocco_market, us_market

BENCHMARK_SOURCE = "benchmark"


class MarketDataError(RuntimeError):
    """Single public exception of this facade: an actionable, user-safe
    message about unavailable market data. Provider-specific errors are
    caught and converted into this type."""


# ---------------------------------------------------------------------------
# Cached fetches
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1800, max_entries=64)
def _fetch_demo(tickers: Tuple[str, ...], start_iso: str, end_iso: str,
                seed: int = 0) -> pd.DataFrame:
    return demo_data.demo_prices(
        list(tickers),
        _dt.date.fromisoformat(start_iso),
        _dt.date.fromisoformat(end_iso),
        seed=demo_data.DEFAULT_SEED + seed,
    )


@st.cache_data(show_spinner=False, ttl=1800, max_entries=64)
def _fetch_us(tickers: Tuple[str, ...], start_iso: str, end_iso: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    return us_market.fetch_prices(
        list(tickers),
        _dt.date.fromisoformat(start_iso),
        _dt.date.fromisoformat(end_iso),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_prices(
    tickers: Sequence[str],
    start: _dt.date,
    end: _dt.date,
    *,
    market: str,
    data_mode: str,
    imported: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return ``(prices, meta)`` or raise :class:`MarketDataError`."""
    tickers = [str(t).strip() for t in dict.fromkeys(tickers) if str(t).strip()]
    if not tickers:
        raise MarketDataError("no assets selected")
    if end < start:
        raise MarketDataError("the end date must not be before the start date")

    if data_mode == MODE_DEMO:
        try:
            frame = _fetch_demo(tuple(tickers), start.isoformat(), end.isoformat())
        except ValueError as exc:
            raise MarketDataError(str(exc)) from exc
        meta = {
            "source": "Demo data (simulated)",
            "simulated": True,
            "currency": "USD" if market == MARKET_US else "MAD",
            "rows": int(len(frame)),
            "columns": [str(c) for c in frame.columns],
            "date_min": pd.Timestamp(frame.index.min()).date().isoformat(),
            "date_max": pd.Timestamp(frame.index.max()).date().isoformat(),
            "warnings": ["Prices are simulated for demonstration purposes."],
        }
        return frame, meta

    # --- Historical (real) data -------------------------------------------
    if market == MARKET_US:
        try:
            frame, meta = _fetch_us(tuple(tickers), start.isoformat(), end.isoformat())
        except us_market.MarketDataError as exc:
            raise MarketDataError(str(exc)) from exc
        meta["simulated"] = False
        return frame, meta

    if market == MARKET_MA:
        try:
            frame, meta = morocco_market.fetch_prices(tickers, start, end, imported)
        except morocco_market.MarketDataError as exc:
            raise MarketDataError(str(exc)) from exc
        return frame, meta

    raise MarketDataError(f"unsupported market: {market}")


def fetch_benchmark(
    ticker: Optional[str],
    start: _dt.date,
    end: _dt.date,
    *,
    market: str,
    data_mode: str,
    imported: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[str]]:
    """Fetch a benchmark series.

    Returns ``(prices, meta, warning)``; a benchmark failure is **not fatal** —
    it yields ``(None, None, warning)`` so the rest of the page still renders.
    """
    if not ticker:
        return None, None, None
    try:
        frame, meta = fetch_prices(
            [ticker], start, end, market=market, data_mode=data_mode, imported=imported
        )
        return frame, meta, None
    except MarketDataError as exc:
        return None, None, f"Benchmark {ticker}: {exc}"


def provider_universe(market: str, data_mode: str,
                      imported: Optional[pd.DataFrame] = None) -> List[Dict[str, str]]:
    """Selectable assets for the current market/mode."""
    from config import asset_metadata

    if data_mode == MODE_DEMO:
        return asset_metadata(market, demo=True) or asset_metadata(market, demo=False)
    if market == MARKET_MA:
        rows = morocco_market.universe(imported)
        if rows:
            return rows
        return []  # explicit: no invented Moroccan instruments
    return us_market.universe()


def provider_benchmarks(market: str, data_mode: str,
                        imported: Optional[pd.DataFrame] = None) -> List[Dict[str, str]]:
    from config import benchmark_universe

    if data_mode == MODE_DEMO:
        return benchmark_universe(market, demo=True)
    if market == MARKET_MA:
        return []  # no verified Moroccan index feed bundled
    return us_market.benchmarks()


def describe_source(meta: Optional[Dict[str, Any]]) -> str:
    if not meta:
        return "—"
    label = "SIMULATED — Demo data" if meta.get("simulated") else meta.get("source", "—")
    return str(label)
