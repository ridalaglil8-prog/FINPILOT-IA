"""Data-layer tests: deterministic demo series, providers, Moroccan CSV path."""
from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd
import pytest

from data import demo_data
from data.providers import morocco_market

START = _dt.date(2023, 1, 2)
END = _dt.date(2023, 6, 30)


def test_demo_prices_are_deterministic():
    first = demo_data.demo_prices(["SPY", "QQQ"], START, END)
    second = demo_data.demo_prices(["SPY", "QQQ"], START, END)
    pd.testing.assert_frame_equal(first, second)


def test_demo_prices_differ_per_ticker():
    frame = demo_data.demo_prices(["SPY", "QQQ"], START, END)
    assert not frame["SPY"].equals(frame["QQQ"])


def test_demo_prices_shape_and_positivity():
    frame = demo_data.demo_prices(["SPY"], START, END)
    assert len(frame) > 100
    assert (frame > 0).all(axis=None)
    assert np.isfinite(frame.to_numpy()).all()
    assert isinstance(frame.index, pd.DatetimeIndex)
    # Monday-Friday calendar only
    assert (frame.index.dayofweek < 5).all()


def test_demo_asset_parameters_are_stable():
    assert demo_data.asset_parameters("SPY") == demo_data.asset_parameters("SPY")
    mu, sigma = demo_data.asset_parameters("SPY")
    assert 0.03 <= mu <= 0.13
    assert 0.09 <= sigma <= 0.38


def test_demo_prices_rejects_invalid_range():
    with pytest.raises(ValueError):
        demo_data.demo_prices(["SPY"], END, START)
    with pytest.raises(ValueError):
        demo_data.demo_prices([], START, END)
    with pytest.raises(ValueError):
        demo_data.demo_prices(["SPY"], START, START)   # fewer than 2 sessions


def test_demo_prices_are_always_labelled_simulated():
    assert demo_data.is_simulated() is True


# ---------------------------------------------------------------------------
# Moroccan provider (CSV import path)
# ---------------------------------------------------------------------------
def _imported() -> pd.DataFrame:
    index = pd.bdate_range("2023-01-02", periods=80)
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "ATW": 400 * np.cumprod(1 + 0.001 + 0.01 * rng.standard_normal(len(index))),
        "LDH": 280 * np.cumprod(1 + 0.0005 + 0.012 * rng.standard_normal(len(index))),
    }, index=index)


def test_morocco_universe_from_imported_csv():
    imported = _imported()
    universe = morocco_market.universe(imported)
    tickers = {a["ticker"] for a in universe}
    assert {"ATW", "LDH"} <= tickers
    assert all(a["currency"] == "MAD" for a in universe if a["ticker"] in {"ATW", "LDH"})


def test_morocco_universe_without_source_is_empty():
    # No Moroccan instrument is bundled: nothing is invented.
    assert morocco_market.universe(None) == []
    assert morocco_market.universe(pd.DataFrame()) == []


def test_morocco_fetch_requires_a_source():
    with pytest.raises(morocco_market.MarketDataError, match="No Moroccan"):
        morocco_market.fetch_prices(["ATW"], START, END, None)


def test_morocco_fetch_returns_meta_and_subset():
    imported = _imported()
    frame, meta = morocco_market.fetch_prices(
        ["ATW"], imported.index.min().date(), imported.index.max().date(), imported)
    assert list(frame.columns) == ["ATW"]
    assert meta["source"] == "CSV import"
    assert meta["simulated"] is False
    assert meta["currency"] == "MAD"
    assert meta["rows"] == len(frame)


def test_morocco_fetch_unknown_ticker_lists_available():
    imported = _imported()
    with pytest.raises(morocco_market.MarketDataError, match="ATW"):
        morocco_market.fetch_prices(["ZZZ"], START, END, imported)


def test_morocco_fetch_rejects_too_short_range():
    imported = _imported()
    with pytest.raises(morocco_market.MarketDataError, match="two observations"):
        morocco_market.fetch_prices(
            ["ATW"], imported.index.min().date(), imported.index.min().date(), imported)


def test_morocco_fetch_flags_gap_rows():
    imported = _imported()
    imported.iloc[5, 0] = np.nan
    _, meta = morocco_market.fetch_prices(
        ["ATW", "LDH"], imported.index.min().date(), imported.index.max().date(), imported)
    assert any("incomplete" in w for w in meta["warnings"])
