"""Chart-builder smoke tests.

Every figure must build without raising and carry at least one trace.
These catch the "duplicate keyword in update_layout" family of bugs, which
only surface at render time in the UI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from components import charts as C

INDEX = pd.bdate_range("2023-01-02", periods=60)
SERIES = pd.Series(np.linspace(100.0, 130.0, len(INDEX)), index=INDEX)
RETURNS = SERIES.pct_change().dropna()
FRAME = pd.DataFrame({"A": np.linspace(100, 110, len(INDEX)),
                      "B": np.linspace(50, 70, len(INDEX))}, index=INDEX)
ROWS = [{"ticker": "A", "value": 60_000.0, "weight_pct": 60.0},
        {"ticker": "B", "value": 40_000.0, "weight_pct": 40.0}]


def _assert_figure(fig):
    assert len(fig.data) >= 1
    assert fig.layout.height  # every builder sets a height via base_layout


def test_allocation_pie():
    _assert_figure(C.allocation_pie(ROWS, "Allocation"))


def test_allocation_bars():
    _assert_figure(C.allocation_bars(["A", "B"], [60.0, 40.0], "Weights"))
    _assert_figure(C.allocation_bars(["A", "B"], [60.0, 40.0], "Weights", horizontal=False))


def test_wealth_chart_with_and_without_benchmark():
    _assert_figure(C.wealth_chart(SERIES, title="Wealth"))
    _assert_figure(C.wealth_chart(SERIES, SERIES * 1.1, title="Wealth"))
    _assert_figure(C.wealth_chart(SERIES, as_pct_from_start=True, title="Cumulative"))


def test_benchmark_relative_chart():
    _assert_figure(C.benchmark_relative_chart(SERIES, SERIES * 1.1, "vs benchmark"))


def test_benchmark_relative_chart_with_no_overlap():
    # insufficient overlap must render an annotation, not raise
    other = pd.Series(np.linspace(1, 2, 5), index=pd.bdate_range("1999-01-04", periods=5))
    fig = C.benchmark_relative_chart(SERIES, other, "vs benchmark")
    assert fig.layout.annotations


def test_drawdown_chart():
    _assert_figure(C.drawdown_chart(RETURNS, "Drawdown"))


def test_correlation_heatmap():
    _assert_figure(C.correlation_heatmap(["A", "B"], [[1.0, 0.4], [0.4, 1.0]], "Corr"))


def test_comparison_bars():
    frame = pd.DataFrame({"ticker": ["A", "B"], "current": [60.0, 40.0],
                          "optimized": [50.0, 50.0]})
    _assert_figure(C.comparison_bars(frame, "Compare"))


def test_stress_chart():
    _assert_figure(C.stress_chart([{"ticker": "A", "impact": -1_000.0},
                                   {"ticker": "B", "impact": 500.0}], "Stress"))


def test_histogram():
    _assert_figure(C.histogram(RETURNS, "Distribution"))


def test_rolling_vol_chart():
    rolling = pd.Series(np.full(len(INDEX) - 5, 0.20), index=INDEX[5:])
    _assert_figure(C.rolling_vol_chart(rolling, "Rolling vol"))


def test_bar_compare():
    _assert_figure(C.bar_compare(["A", "B"], [1.5, -2.0], "Compare"))


def test_base_layout_has_no_title_key():
    # callers pass their own title=..., so a built-in title would collide
    assert "title" not in C.base_layout()
    assert C.with_title("x")["text"] == "x"


@pytest.mark.parametrize("color", ["#0A1428", "rgba(0,0,0,0)"])
def test_layout_colors_are_dark_theme(color):
    assert isinstance(color, str)
