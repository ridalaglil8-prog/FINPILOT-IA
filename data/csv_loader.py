"""CSV historical-price import with strict validation.

Accepted layouts
----------------
1. **Wide** — ``date, SPY, QQQ, ...`` (first column dates, other columns prices)
2. **Long** — ``date, ticker, price`` (a column whose name matches
   ``ticker/symbol/asset/actif`` and a price column)

Column matching is case-insensitive and accent-tolerant; common aliases are
recognised (``date/Date/Date de cotation``, ``close/Close/clôture``,
``adj close/Adj_Close``, ...).

Validation guarantees
---------------------
* every accepted row has a parseable date and a finite, strictly positive price
* dates are unique (duplicates keep the **last** occurrence, with a warning)
* the frame is sorted by date
* at least two observations are required
* nothing is forward/backward-filled: gaps stay gaps

Failures raise :class:`CSVValidationError` with a message that is safe to
show to end users (no stack traces).
"""
from __future__ import annotations

import io
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

MAX_ROWS = 200_000
MAX_COLUMNS = 200

DATE_ALIASES = ("date", "datetime", "timestamp", "time", "jour", "date de cotation",
                "date de negociation", "seance")
PRICE_ALIASES = ("price", "close", "adj close", "adjusted close", "adj_close",
                 "cloture", "cloture ajustee", "last", "value", "val", "prix",
                 "cours", "course", "px_last")
TICKER_ALIASES = ("ticker", "symbol", "asset", "actif", "code", "instrument", "security")


class CSVValidationError(ValueError):
    """A user-safe message explaining why an uploaded CSV was rejected."""


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _norm(name: Any) -> str:
    return " ".join(_strip_accents(name).lower().replace("_", " ").replace("-", " ").split())


def _match(columns: List[Any], aliases: Tuple[str, ...]) -> Optional[Any]:
    normalized = {_norm(c): c for c in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for norm_name, original in normalized.items():
        for alias in aliases:
            if norm_name and (norm_name == alias or alias in norm_name.split()):
                return original
    return None


def _read(source: Union[str, Path, io.BytesIO, io.StringIO, Any]) -> pd.DataFrame:
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise CSVValidationError(f"file not found: {path}")
            df = pd.read_csv(path)
        elif hasattr(source, "getvalue"):  # Streamlit UploadedFile / StringIO
            payload = source.getvalue()
            if isinstance(payload, bytes):
                df = pd.read_csv(io.BytesIO(payload))
            else:
                df = pd.read_csv(io.StringIO(payload))
        else:
            df = pd.read_csv(source)
    except CSVValidationError:
        raise
    except UnicodeDecodeError as exc:
        raise CSVValidationError("the file must be UTF-8 encoded CSV") from exc
    except pd.errors.EmptyDataError as exc:
        raise CSVValidationError("the file is empty") from exc
    except pd.errors.ParserError as exc:
        raise CSVValidationError(f"malformed CSV: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise CSVValidationError(f"could not read the file: {exc}") from exc

    if df.empty:
        raise CSVValidationError("the file contains no data rows")
    if len(df) > MAX_ROWS:
        raise CSVValidationError(f"the file has more than {MAX_ROWS} rows")
    if df.shape[1] > MAX_COLUMNS:
        raise CSVValidationError(f"the file has more than {MAX_COLUMNS} columns")
    if df.shape[1] < 2:
        raise CSVValidationError("the file must contain at least a date column and a price column")
    return df


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse dates, preferring the unambiguous ISO-8601 format.

    ``dayfirst`` alone would silently swap month and day on ISO input
    (``2023-01-02`` -> 1 February), so ISO is tried first and ``dayfirst``
    only fills what ISO could not parse (e.g. ``03/01/2023`` exports).
    Non-parsable values become ``NaT`` — never guessed.
    """
    iso = pd.to_datetime(values, errors="coerce", format="ISO8601")
    remaining = iso.isna()
    if remaining.any():
        iso[remaining] = pd.to_datetime(values[remaining], errors="coerce", dayfirst=True)
    return iso


def _is_numeric_like(column: pd.Series, threshold: float = 0.8) -> bool:
    """True when at least ``threshold`` of the values are parseable as numbers."""
    return pd.to_numeric(column, errors="coerce").notna().mean() >= threshold


def load_price_csv(
    source: Union[str, Path, Any],
    *,
    default_ticker: str = "ASSET",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Parse and validate a historical-price CSV.

    Returns ``(wide_prices, meta)`` where ``wide_prices`` has a DatetimeIndex
    and one float column per asset, and ``meta`` carries import diagnostics
    (``rows``, ``date_min``, ``date_max``, ``columns``, ``warnings``,
    ``format``).
    """
    raw = _read(source)
    warnings: List[str] = []

    date_col = _match(list(raw.columns), DATE_ALIASES)
    ticker_col = _match(list(raw.columns), TICKER_ALIASES)

    if date_col is None:
        # Fall back to the first column that parses as dates for most rows.
        # Numeric columns are skipped: pandas would otherwise interpret the
        # integers 1, 2, 3... as epoch timestamps and accept them as dates.
        for column in raw.columns:
            if pd.api.types.is_numeric_dtype(raw[column]):
                continue
            parsed = _parse_dates(raw[column])
            if parsed.notna().mean() >= 0.8:
                date_col = column
                break
    if date_col is None:
        raise CSVValidationError(
            "no date column found (expected a column named 'date', 'Date', 'datetime', ...)"
        )

    price_col = _match(list(raw.columns), PRICE_ALIASES)
    if price_col in (date_col, ticker_col):
        price_col = None

    fmt: str
    if ticker_col is not None:
        # ---- long layout: date, ticker, price ----------------------------
        fmt = "long"
        if price_col is None:
            numeric_candidates = [
                c for c in raw.columns
                if c != date_col and c != ticker_col and _is_numeric_like(raw[c])
            ]
            if not numeric_candidates:
                raise CSVValidationError(
                    "no numeric price column found (expected 'price', 'close', 'cloture', ...)"
                )
            price_col = numeric_candidates[0]
        frame = raw[[date_col, price_col, ticker_col]].copy()
        frame.columns = ["date", "price", "ticker"]
    else:
        # ---- wide layout: date, then one column per asset ----------------
        fmt = "wide"
        # 0.5 threshold: a few gaps must not disqualify an asset column.
        candidates = [c for c in raw.columns
                      if c != date_col and _is_numeric_like(raw[c], threshold=0.5)]
        if not candidates:
            raise CSVValidationError(
                "no numeric price column found (expected 'price', 'close', 'cloture', ...)"
            )
        if len(candidates) == 1 and price_col is not None:
            # a single anonymous price column -> one asset named default_ticker
            frame = raw[[date_col, candidates[0]]].copy()
            frame.columns = ["date", "price"]
            frame["ticker"] = str(default_ticker).strip() or "ASSET"
            fmt = "long"
        else:
            frame = raw[[date_col] + candidates].copy()
            frame.columns = ["date"] + [str(c) for c in candidates]

    frame["date"] = _parse_dates(frame["date"])
    bad_dates = int(frame["date"].isna().sum())
    if bad_dates:
        warnings.append(f"{bad_dates} row(s) with an unparseable date were ignored")
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        raise CSVValidationError("no row contains a valid date")

    if "price" in frame.columns:
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        bad_prices = int(frame["price"].isna().sum())
        if bad_prices:
            warnings.append(f"{bad_prices} row(s) with a non-numeric price were ignored")
        frame = frame.dropna(subset=["price"])

        non_positive = int((frame["price"] <= 0).sum())
        if non_positive:
            warnings.append(f"{non_positive} row(s) with a non-positive price were ignored")
            frame = frame[frame["price"] > 0]

        if frame.empty:
            raise CSVValidationError("no row contains a valid positive price")
        if len(frame) < 2:
            raise CSVValidationError("at least two valid price observations are required")
    else:
        asset_columns = [c for c in frame.columns if c != "date"]
        numeric = frame[asset_columns].apply(pd.to_numeric, errors="coerce")
        bad_prices = int(numeric.isna().sum().sum())
        if bad_prices:
            warnings.append(f"{bad_prices} cell(s) with a non-numeric price were ignored")
        non_positive = int((numeric <= 0).sum().sum())
        if non_positive:
            warnings.append(f"{non_positive} cell(s) with a non-positive price were ignored")
        frame[asset_columns] = numeric
        if numeric.isna().all(axis=None):
            raise CSVValidationError("no row contains a valid positive price")

    if fmt == "long":
        frame["ticker"] = frame["ticker"].astype(str).str.strip().replace({"": np.nan})
        missing_ticker = int(frame["ticker"].isna().sum())
        if missing_ticker:
            warnings.append(f"{missing_ticker} row(s) without a ticker were ignored")
        frame = frame.dropna(subset=["ticker"])
        if frame.empty:
            raise CSVValidationError("no row contains a ticker symbol")
        duplicated = int(frame.duplicated(subset=["date", "ticker"]).sum())
        if duplicated:
            warnings.append(f"{duplicated} duplicated (date, ticker) row(s): last value kept")
        wide = frame.pivot_table(index="date", columns="ticker", values="price", aggfunc="last")
        tickers = sorted(frame["ticker"].unique())
        wide = wide[tickers]
    else:
        duplicated = int(frame["date"].duplicated().sum())
        if duplicated:
            warnings.append(f"{duplicated} duplicated date(s): last value kept")
        wide = frame.drop_duplicates(subset=["date"], keep="last").set_index("date")

    if len(wide) < 2:
        raise CSVValidationError("at least two valid price observations are required")

    wide = wide.sort_index()
    wide = wide.replace([np.inf, -np.inf], np.nan)
    if wide.isna().all(axis=None):
        raise CSVValidationError("the file contains no usable price observations")
    partial = wide.isna().any(axis=1)
    if partial.any():
        warnings.append(
            f"{int(partial.sum())} date(s) with incomplete prices were kept as gaps "
            "(no values are forward-filled)"
        )
    if not wide.index.is_unique:
        wide = wide[~wide.index.duplicated(keep="last")]

    meta: Dict[str, Any] = {
        "format": fmt,
        "rows": int(len(wide)),
        "columns": [str(c) for c in wide.columns],
        "date_min": wide.index.min().date().isoformat(),
        "date_max": wide.index.max().date().isoformat(),
        "warnings": warnings,
        "currency": "MAD",
    }
    return wide, meta


def validate_price_frame(frame: pd.DataFrame, *, min_rows: int = 2) -> List[str]:
    """Validate an already-loaded wide price frame; returns problems."""
    problems: List[str] = []
    if frame is None or not isinstance(frame, pd.DataFrame):
        return ["price data is not a DataFrame"]
    if frame.empty:
        return ["price data is empty"]
    if not isinstance(frame.index, pd.DatetimeIndex):
        problems.append("index is not a DatetimeIndex")
    if len(frame) < min_rows:
        problems.append(f"only {len(frame)} observation(s) (minimum {min_rows})")
    if frame.shape[1] < 1:
        problems.append("no asset columns")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        problems.append("non-finite price values present")
    if (numeric <= 0).any(axis=None):
        problems.append("non-positive prices present")
    return problems
