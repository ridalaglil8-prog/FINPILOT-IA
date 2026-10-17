"""CSV import validation tests (wide and long layouts, rejection paths)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from data import csv_loader


def _csv(text: str) -> io.StringIO:
    return io.StringIO(text)


WIDE = """date,ATW,LDH
2023-01-02,412.50,285.10
2023-01-03,415.00,286.40
2023-01-04,410.00,288.00
"""

LONG = """Date,Ticker,Prix
2023-01-02,ATW,412.50
2023-01-03,ATW,415.00
2023-01-02,LDH,285.10
2023-01-03,LDH,286.40
"""


def test_wide_csv_layout():
    frame, meta = csv_loader.load_price_csv(_csv(WIDE))
    assert list(frame.columns) == ["ATW", "LDH"]
    assert len(frame) == 3
    assert meta["format"] == "wide"
    assert meta["date_min"] == "2023-01-02"
    assert meta["date_max"] == "2023-01-04"
    assert frame.loc["2023-01-03", "ATW"] == pytest.approx(415.0)


def test_long_csv_layout():
    frame, meta = csv_loader.load_price_csv(_csv(LONG))
    assert list(frame.columns) == ["ATW", "LDH"]
    assert meta["format"] == "long"
    assert frame.loc["2023-01-02", "LDH"] == pytest.approx(285.10)


def test_dates_are_sorted_and_unique():
    messy = """date,price
2023-01-04,10
2023-01-02,12
2023-01-03,11
"""
    frame, meta = csv_loader.load_price_csv(_csv(messy), default_ticker="TEST")
    assert list(frame.index) == sorted(frame.index)
    assert not frame.index.duplicated().any()


def test_duplicate_dates_keep_last_with_warning():
    dup = """date,price
2023-01-02,10
2023-01-02,12
2023-01-03,11
"""
    frame, meta = csv_loader.load_price_csv(_csv(dup), default_ticker="TEST")
    assert frame.loc["2023-01-02", "TEST"] == pytest.approx(12.0)
    assert any("duplicated" in w for w in meta["warnings"])


def test_non_positive_prices_are_ignored_with_warning():
    bad = """date,price
2023-01-02,10
2023-01-03,-5
2023-01-04,11
"""
    frame, meta = csv_loader.load_price_csv(_csv(bad), default_ticker="TEST")
    assert len(frame) == 2
    assert (frame > 0).all(axis=None)
    assert any("non-positive" in w for w in meta["warnings"])


def test_gaps_are_kept_not_filled():
    gaps = """date,A,B
2023-01-02,10,20
2023-01-03,,21
2023-01-04,12,22
"""
    frame, meta = csv_loader.load_price_csv(_csv(gaps))
    assert np.isnan(frame.loc["2023-01-03", "A"])
    assert frame.loc["2023-01-03", "B"] == pytest.approx(21.0)
    assert any("gaps" in w for w in meta["warnings"])


def test_unparseable_dates_are_dropped_with_warning():
    broken = """date,price
not-a-date,10
2023-01-03,11
2023-01-04,12
"""
    frame, meta = csv_loader.load_price_csv(_csv(broken), default_ticker="TEST")
    assert len(frame) == 2
    assert any("unparseable" in w for w in meta["warnings"])


def test_rejects_empty_file():
    with pytest.raises(csv_loader.CSVValidationError):
        csv_loader.load_price_csv(_csv(""))


def test_rejects_file_without_date():
    with pytest.raises(csv_loader.CSVValidationError, match="date"):
        csv_loader.load_price_csv(_csv("a,b\n1,2\n3,4\n"))


def test_rejects_file_without_numeric_price():
    with pytest.raises(csv_loader.CSVValidationError):
        csv_loader.load_price_csv(_csv("date,note\n2023-01-02,hello\n2023-01-03,world\n"))


def test_rejects_single_observation():
    with pytest.raises(csv_loader.CSVValidationError, match="two"):
        csv_loader.load_price_csv(_csv("date,price\n2023-01-02,10\n"))


def test_rejects_all_invalid_prices():
    with pytest.raises(csv_loader.CSVValidationError):
        csv_loader.load_price_csv(_csv("date,price\n2023-01-02,-1\n2023-01-03,0\n"))


def test_missing_file_path():
    with pytest.raises(csv_loader.CSVValidationError, match="not found"):
        csv_loader.load_price_csv("does_not_exist.csv")


def test_validate_price_frame_problems():
    good = pd.DataFrame({"A": [1.0, 2.0]},
                        index=pd.to_datetime(["2023-01-02", "2023-01-03"]))
    assert csv_loader.validate_price_frame(good) == []
    assert csv_loader.validate_price_frame(None)
    assert csv_loader.validate_price_frame(pd.DataFrame())
    assert csv_loader.validate_price_frame(good.head(1), min_rows=2)
    bad = good.copy()
    bad.iloc[0, 0] = np.inf
    assert any("non-finite" in p for p in csv_loader.validate_price_frame(bad))
    negative = pd.DataFrame({"A": [-1.0, 2.0]}, index=good.index)
    assert any("non-positive" in p for p in csv_loader.validate_price_frame(negative))
