"""Moroccan market data provider.

FinPilot AI makes **no assumption** that Casablanca Stock Exchange securities
are available through yfinance. This provider therefore supports exactly two
sources, in this order of preference:

1. a CSV of historical prices imported by the user (Settings page) — real,
   user-supplied observations;
2. Demo mode synthetic series (clearly labelled *SIMULATED*, produced by
   ``data.demo_data``).

If neither is available the provider raises :class:`MarketDataError` with an
actionable message. No Moroccan ticker symbol, price, index or historical
observation is bundled with or invented by this application.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from config import MOROCCO_ASSETS

SOURCE_NAME = "CSV import"
CURRENCY = "MAD"


class MarketDataError(RuntimeError):
    """No Moroccan data source available. Message is safe for end users."""


def universe(imported: Optional[pd.DataFrame] = None) -> List[Dict[str, str]]:
    """Assets selectable in the Moroccan market.

    * metadata declared in ``config.MOROCCO_ASSETS`` (verified by the project
      owner) comes first;
    * then any column found in the user-imported CSV.
    """
    assets: List[Dict[str, str]] = [dict(a) for a in MOROCCO_ASSETS]
    known = {a["ticker"] for a in assets}
    if imported is not None and not imported.empty:
        for column in imported.columns:
            ticker = str(column)
            if ticker not in known:
                assets.append({
                    "ticker": ticker,
                    "name": f"{ticker} (imported CSV)",
                    "asset_class": "Imported (CSV)",
                    "currency": CURRENCY,
                })
                known.add(ticker)
    return assets


def metadata_for(ticker: str, imported: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    for asset in universe(imported):
        if asset["ticker"] == ticker:
            return dict(asset)
    return {"ticker": ticker, "name": ticker, "asset_class": "Imported (CSV)",
            "currency": CURRENCY}


def has_source(imported: Optional[pd.DataFrame]) -> bool:
    return imported is not None and isinstance(imported, pd.DataFrame) and not imported.empty


def fetch_prices(
    tickers: Sequence[str],
    start: _dt.date,
    end: _dt.date,
    imported: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return Moroccan prices from the user-imported CSV within ``[start, end]``.

    Raises :class:`MarketDataError` when no CSV was imported, when the
    requested tickers are absent, or when the range yields < 2 observations.
    """
    tickers = [str(t).strip() for t in dict.fromkeys(tickers) if str(t).strip()]
    if not tickers:
        raise MarketDataError("no tickers selected")
    if end < start:
        raise MarketDataError("the end date must not be before the start date")
    if not has_source(imported):
        raise MarketDataError(
            "No Moroccan market data source is available. Import a CSV file "
            "with historical prices on the Settings page, or switch to Demo "
            "mode (simulated data) in the sidebar."
        )

    frame = imported.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame.index = pd.to_datetime(frame.index)
        except Exception as exc:
            raise MarketDataError("the imported data has no valid date index") from exc

    missing = [t for t in tickers if t not in frame.columns]
    if missing:
        raise MarketDataError(
            "the imported CSV does not contain: " + ", ".join(missing) +
            ". Available: " + ", ".join(str(c) for c in frame.columns[:15])
        )

    subset = frame[tickers]
    subset = subset.loc[(subset.index >= pd.Timestamp(start)) & (subset.index <= pd.Timestamp(end))]
    subset = subset.apply(pd.to_numeric, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
    subset = subset.dropna(how="all").sort_index()

    if len(subset) < 2:
        raise MarketDataError(
            "fewer than two observations in the imported data for the selected "
            "period. Widen the analysis range or import a longer history."
        )

    warnings: List[str] = []
    gap_rows = int(subset.isna().any(axis=1).sum())
    if gap_rows:
        warnings.append(
            f"{gap_rows} date(s) have incomplete prices in the imported CSV "
            "(values are never forward-filled)"
        )

    meta: Dict[str, Any] = {
        "source": SOURCE_NAME,
        "currency": CURRENCY,
        "rows": int(len(subset)),
        "columns": [str(c) for c in subset.columns],
        "date_min": pd.Timestamp(subset.index.min()).date().isoformat(),
        "date_max": pd.Timestamp(subset.index.max()).date().isoformat(),
        "warnings": warnings,
        "simulated": False,
    }
    return subset, meta
