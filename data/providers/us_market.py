"""US market data provider (yfinance).

The provider is deliberately strict: if the upstream request fails or returns
nothing, it raises :class:`MarketDataError` — it **never** falls back to
simulated data. Demo data is only ever produced by ``data.demo_data`` and is
labelled as such by the UI.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from config import US_ASSETS, US_BENCHMARKS

SOURCE_NAME = "yfinance"
CURRENCY = "USD"
REQUEST_TIMEOUT = 20  # seconds


class MarketDataError(RuntimeError):
    """Upstream market data is unavailable. Message is safe for end users."""


def universe() -> List[Dict[str, str]]:
    return [dict(a) for a in US_ASSETS]


def benchmarks() -> List[Dict[str, str]]:
    return [dict(b) for b in US_BENCHMARKS]


def metadata_for(ticker: str) -> Dict[str, str]:
    for asset in US_ASSETS:
        if asset["ticker"] == ticker:
            return dict(asset)
    for bench in US_BENCHMARKS:
        if bench["ticker"] == ticker:
            return {"ticker": ticker, "name": bench["name"],
                    "asset_class": "Benchmark", "currency": CURRENCY}
    return {"ticker": ticker, "name": ticker, "asset_class": "Stock", "currency": CURRENCY}


def _flatten_download(raw: Any, tickers: Sequence[str]) -> pd.DataFrame:
    """Normalise yfinance output (single or multi-index columns) to a frame."""
    if raw is None or raw.empty:
        raise MarketDataError("the data provider returned an empty response")

    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        # yfinance returns (field, ticker) or (ticker, field) depending on version.
        if "Close" in frame.columns.get_level_values(0):
            frame = frame["Close"]
        elif "Close" in frame.columns.get_level_values(-1):
            frame = frame.xs("Close", axis=1, level=-1)
        else:
            frame.columns = frame.columns.droplevel(-1)
    elif frame.columns.name in ("Price", "Ticker") or "Close" in frame.columns:
        frame = frame[["Close"]] if "Close" in frame.columns else frame

    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([pd.NA], float("nan"))
    keep = [c for c in tickers if c in frame.columns]
    frame = frame[keep]
    frame = frame.dropna(how="all")
    if frame.empty:
        raise MarketDataError(
            "no historical prices were returned for the selected assets"
        )
    return frame


def fetch_prices(
    tickers: Sequence[str],
    start: _dt.date,
    end: _dt.date,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Download daily close prices for ``tickers`` (US market).

    Raises :class:`MarketDataError` on any failure (network, ticker unknown,
    empty payload). ``end`` is inclusive for the caller and exclusive for
    yfinance, so one day is added.
    """
    tickers = [str(t).strip() for t in dict.fromkeys(tickers) if str(t).strip()]
    if not tickers:
        raise MarketDataError("no tickers selected")
    if end < start:
        raise MarketDataError("the end date must not be before the start date")

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise MarketDataError(
            "the yfinance package is not installed (pip install -r requirements.txt)"
        ) from exc

    yf_end = end + _dt.timedelta(days=1)
    try:
        raw = yf.download(
            tickers=list(tickers),
            start=pd.Timestamp(start),
            end=pd.Timestamp(yf_end),
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )
    except Exception as exc:  # network errors, timeouts, parsing
        raise MarketDataError(
            f"US market data request failed ({type(exc).__name__}). "
            "Check your Internet connection or use Demo mode."
        ) from exc

    try:
        frame = _flatten_download(raw, tickers)
    except MarketDataError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise MarketDataError(
            f"could not parse the provider response ({type(exc).__name__})"
        ) from exc

    missing = [t for t in tickers if t not in frame.columns]
    warnings: List[str] = []
    if missing:
        warnings.append(
            "no data returned for: " + ", ".join(missing) +
            " (check the ticker symbol or the selected period)"
        )
    frame = frame.sort_index()
    if len(frame) < 2:
        raise MarketDataError(
            "fewer than two trading days were returned for the selected period"
        )

    meta: Dict[str, Any] = {
        "source": SOURCE_NAME,
        "currency": CURRENCY,
        "rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "date_min": pd.Timestamp(frame.index.min()).date().isoformat(),
        "date_max": pd.Timestamp(frame.index.max()).date().isoformat(),
        "warnings": warnings,
        "simulated": False,
    }
    return frame, meta
