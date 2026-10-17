"""Currency rules: totals are combined only with an explicit FX rate."""
from __future__ import annotations

import pytest

from services import currency as FX


def test_normalize_currency():
    assert FX.normalize_currency("usd") == "USD"
    assert FX.normalize_currency(None) == "UNKNOWN"


def test_currencies_of_preserves_order():
    assets = [{"currency": "MAD"}, {"currency": "USD"}, {"currency": "MAD"}]
    assert FX.currencies_of(assets) == ["MAD", "USD"]


def test_needs_conversion():
    assert not FX.needs_conversion(["USD"])
    assert not FX.needs_conversion(["USD", "USD"])
    assert FX.needs_conversion(["USD", "MAD"])


def test_fx_rate_direct_and_inverse():
    rates = {"USD_MAD": 10.0}
    assert FX.fx_rate("USD", "MAD", rates) == pytest.approx(10.0)
    assert FX.fx_rate("MAD", "USD", rates) == pytest.approx(0.1)
    assert FX.fx_rate("USD", "USD", rates) == 1.0


def test_fx_rate_missing_returns_none():
    assert FX.fx_rate("USD", "MAD", None) is None
    assert FX.fx_rate("USD", "MAD", {}) is None
    assert FX.fx_rate("USD", "MAD", {"EUR_USD": 1.1}) is None


def test_fx_rate_rejects_invalid_values():
    assert FX.fx_rate("USD", "MAD", {"USD_MAD": 0.0}) is None
    assert FX.fx_rate("USD", "MAD", {"USD_MAD": -3.0}) is None
    assert FX.fx_rate("USD", "MAD", {"USD_MAD": float("nan")}) is None


def test_convert_raises_without_rate():
    with pytest.raises(FX.FXUnavailableError):
        FX.convert(100.0, "USD", "MAD", None)


def test_convert_with_rate():
    assert FX.convert(100.0, "USD", "MAD", {"USD_MAD": 10.0}) == pytest.approx(1000.0)


def test_combine_totals_returns_none_without_rate():
    # Never guess a conversion: without a rate the caller must show totals apart.
    assert FX.combine_totals({"USD": 100.0, "MAD": 200.0}, None) is None


def test_combine_totals_with_explicit_rate():
    result = FX.combine_totals({"USD": 100.0, "MAD": 200.0}, {"USD_MAD": 10.0},
                               target="MAD")
    assert result is not None
    assert result["currency"] == "MAD"
    assert result["total"] == pytest.approx(100 * 10.0 + 200.0)
    assert result["rates_used"]["USD_MAD"] == pytest.approx(10.0)


def test_combine_totals_single_currency():
    result = FX.combine_totals({"USD": 123.45}, None)
    assert result["total"] == pytest.approx(123.45)
    assert result["rates_used"] == {}


def test_combine_totals_empty():
    assert FX.combine_totals({}, None) is None


def test_user_fx_rates_validation():
    assert FX.user_fx_rates(10.0)["USD_MAD"] == pytest.approx(10.0)
    assert FX.user_fx_rates(10.0)["MAD_USD"] == pytest.approx(0.1)
    assert FX.user_fx_rates(None) == {}
    assert FX.user_fx_rates(0.0) == {}
    assert FX.user_fx_rates(-1.0) == {}
    assert FX.user_fx_rates("abc") == {}


def test_format_amount():
    assert FX.format_amount(1500.0, "USD") == "$1,500.00"
    assert FX.format_amount(1500.0, "MAD") == "MAD 1,500.00"
