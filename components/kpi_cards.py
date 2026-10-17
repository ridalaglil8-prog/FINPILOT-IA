"""KPI cards and small formatting helpers shared by every page."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import streamlit as st

from config import CURRENCY_SYMBOLS
from i18n import t
from services.currency import format_amount


def fmt_pct(value: Optional[float], *, signed: bool = True, digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    number = float(value) * 100.0
    sign = "+" if signed and number > 0 else ""
    return f"{sign}{number:.{digits}f} %"


def fmt_num(value: Optional[float], digits: int = 2) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):,.{digits}f}"


def fmt_money(value: Optional[float], currency: Optional[str]) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return format_amount(float(value), currency or "")


def _delta_text(value: Optional[float], label: str) -> Optional[str]:
    if value is None:
        return None
    return f"{fmt_pct(value)} · {label}"


def render_kpi_cards(metrics: Mapping[str, Any], currency: str,
                     benchmark: Optional[Mapping[str, Any]] = None) -> None:
    """Two rows of four professional KPI cards."""
    if not metrics:
        st.info(t("kpi.unavailable"))
        return

    row1 = st.columns(4)
    row1[0].metric(t("kpi.capital"), fmt_money(metrics.get("initial_capital"), currency))
    row1[1].metric(
        t("kpi.value"),
        fmt_money(metrics.get("current_value"), currency),
        delta=fmt_pct(metrics.get("cumulative_return")) if metrics.get("cumulative_return") is not None else None,
        delta_color="normal",
    )
    row1[2].metric(t("kpi.annualized"), fmt_pct(metrics.get("annualized_return")))
    row1[3].metric(t("kpi.volatility"), fmt_pct(metrics.get("annualized_volatility"), signed=False))

    row2 = st.columns(4)
    row2[0].metric(t("kpi.sharpe"), fmt_num(metrics.get("sharpe_ratio")))
    drawdown = metrics.get("max_drawdown")
    row2[1].metric(t("kpi.drawdown"), fmt_pct(drawdown, signed=False))
    row2[2].metric(t("kpi.var95"), fmt_pct(metrics.get("var_95"), signed=False))
    row2[3].metric(t("kpi.observations"), fmt_num(metrics.get("observations"), digits=0))


def render_benchmark_card(benchmark: Mapping[str, Any]) -> None:
    if not benchmark:
        return
    columns = st.columns(4)
    columns[0].metric(t("bench.label", ticker=benchmark.get("ticker", "")),
                      fmt_pct(benchmark.get("cumulative_return")))
    columns[1].metric(t("bench.annualized"), fmt_pct(benchmark.get("annualized_return")))
    columns[2].metric(t("bench.volatility"), fmt_pct(benchmark.get("annualized_volatility"), signed=False))
    columns[3].metric(t("bench.drawdown"), fmt_pct(benchmark.get("max_drawdown"), signed=False))


def render_data_provenance(context: Mapping[str, Any]) -> None:
    """Source / period / last observation / currency / warnings strip."""
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    col1.caption(f"🗄 {t('prov.source')}: **{context.get('data_source', '—')}**")
    col2.caption(
        f"🗓 {t('prov.range')}: {context.get('start', '—')} → {context.get('end', '—')}"
    )
    col3.caption(f"📅 {t('prov.last')}: {context.get('last_observation') or '—'}")
    col4.caption(f"💱 {t('prov.currency')}: {', '.join(context.get('currencies', []) or ['—'])}")

    if context.get("simulated"):
        st.warning(t("warn.simulated_data"))
    for warning in context.get("warnings", []):
        if warning == "simulated_data":
            continue
        _render_warning(warning)


def _render_warning(warning: str) -> None:
    from i18n import tw

    message = tw(warning)
    if warning == "fx_missing":
        st.info(message)
    else:
        st.warning(message)


def render_errors(errors: List[str]) -> None:
    from i18n import tw

    for error in errors:
        st.error(tw(error))


def currency_label(currency: str) -> str:
    return f"{CURRENCY_SYMBOLS.get(currency, '')}{currency}"
