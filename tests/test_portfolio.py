"""Portfolio primitives: weights, allocation values and currency grouping."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics import portfolio as Port
from analytics.returns import InsufficientDataError


ASSETS = [
    {"ticker": "AAA", "name": "Alpha", "asset_class": "US Stock",
     "currency": "USD", "weight": 0.6},
    {"ticker": "BBB", "name": "Beta", "asset_class": "US Stock",
     "currency": "USD", "weight": 0.4},
]


def test_validate_weights_accepts_long_only_100():
    assert Port.validate_weights({"A": 0.6, "B": 0.4}) == []


def test_validate_weights_rejects_short_and_excess():
    assert Port.validate_weights({"A": -0.2, "B": 1.2})
    assert Port.validate_weights({"A": 1.2})
    assert Port.validate_weights({"A": 0.4, "B": 0.4})   # sums to 80 %
    assert Port.validate_weights({})                      # empty


def test_normalize_weights_scales_to_one():
    result = Port.normalize_weights({"A": 30.0, "B": 10.0})
    assert result == {"A": pytest.approx(0.75), "B": pytest.approx(0.25)}


def test_normalize_weights_rejects_bad_input():
    with pytest.raises(ValueError):
        Port.normalize_weights({})
    with pytest.raises(ValueError):
        Port.normalize_weights({"A": 0.0, "B": 0.0})
    with pytest.raises(ValueError):
        Port.normalize_weights({"A": float("nan")})


def test_percent_conversions_round_trip():
    weights = {"A": 0.25, "B": 0.75}
    assert Port.percent_to_weights(Port.weights_to_percent(weights)) == weights


def test_split_by_currency_never_mixes():
    assets = ASSETS + [{"ticker": "CCC", "currency": "MAD", "weight": 0.0}]
    groups = Port.split_by_currency(assets)
    assert set(groups) == {"USD", "MAD"}
    assert [a["ticker"] for a in groups["USD"]] == ["AAA", "BBB"]


def test_sub_weights_renormalise_within_group():
    group = ASSETS + [{"ticker": "CCC", "currency": "MAD", "weight": 0.5}]
    sub = Port.sub_weights([a for a in group if a["currency"] == "USD"])
    assert sum(sub.values()) == pytest.approx(1.0)
    assert sub["AAA"] == pytest.approx(0.6)


def test_sub_weights_rejects_all_zero():
    with pytest.raises(ValueError):
        Port.sub_weights([{"ticker": "A", "weight": 0.0}])


def test_allocation_table_values_and_currency_isolation():
    rows = Port.allocation_table(ASSETS, {"AAA": 120.0, "BBB": 40.0}, 100_000.0)
    assert rows[0]["value"] == pytest.approx(60_000.0)
    assert rows[1]["value"] == pytest.approx(40_000.0)
    assert rows[0]["last_price"] == pytest.approx(120.0)
    assert Port.totals_by_currency(rows) == {"USD": pytest.approx(100_000.0)}


def test_allocation_table_rejects_bad_capital():
    for capital in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            Port.allocation_table(ASSETS, {"AAA": 1.0, "BBB": 1.0}, capital)


def test_allocation_table_handles_missing_price():
    rows = Port.allocation_table(ASSETS, {"AAA": 120.0}, 1_000.0)
    assert rows[1]["last_price"] is None


def test_totals_by_currency_splits_mixed_portfolios():
    rows = [
        {"currency": "USD", "value": 60_000.0},
        {"currency": "MAD", "value": 40_000.0},
        {"currency": "USD", "value": 10_000.0},
    ]
    totals = Port.totals_by_currency(rows)
    assert totals["USD"] == pytest.approx(70_000.0)
    assert totals["MAD"] == pytest.approx(40_000.0)


def test_portfolio_value():
    returns = pd.Series([0.10, -0.10])
    assert Port.portfolio_value(1_000.0, returns) == pytest.approx(990.0)


def test_portfolio_value_requires_observations():
    with pytest.raises(InsufficientDataError):
        Port.portfolio_value(1_000.0, pd.Series(dtype=float))


def test_shares_from_weights():
    shares = Port.shares_from_weights(10_000.0, {"AAA": 0.6, "BBB": 0.4},
                                      {"AAA": 120.0, "BBB": 40.0})
    assert shares["AAA"] == pytest.approx(50.0)
    assert shares["BBB"] == pytest.approx(100.0)


def test_shares_from_weights_requires_prices():
    with pytest.raises(ValueError):
        Port.shares_from_weights(10_000.0, {"AAA": 1.0}, {})


def test_is_valid_weight_sum():
    assert Port.is_valid_weight_sum({"A": 0.5, "B": 0.5})
    assert not Port.is_valid_weight_sum({"A": 0.5})
    assert not Port.is_valid_weight_sum({})


def test_format_money():
    assert Port.format_money(1234.5, "USD") == "$1,234.50"
    assert "MAD" in Port.format_money(1234.5, "MAD")
