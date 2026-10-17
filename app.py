"""FinPilot AI — Dashboard (main entry point).

Run with::

    streamlit run app.py

The page stays concise: navigation and session bootstrap live in the sidebar
component, metrics come from ``services.portfolio_service`` and all figures
are built by ``components.charts``.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="FinPilot AI — Dashboard", page_icon="🧭", layout="wide")

from analytics.risk import drawdown_series
from components import charts as C
from components import kpi_cards as KPI
from components.sidebar import render_sidebar
from config import APP_NAME, MARKETS
from i18n import t
from services.portfolio_service import build_context, is_usable


def _render_dashboard(context: dict) -> None:
    KPI.render_data_provenance(context)

    currency = context.get("currency") or "USD"
    if context.get("multi_currency"):
        st.info(t("warn.multi_currency"))
        for group in context["groups"]:
            with st.container(border=True):
                weight_label = t("dashboard.subportfolio",
                                 weight=f"{group['weight_pct']:.1f}")
                st.markdown(f"**{group['currency']}** · {weight_label}")
                KPI.render_kpi_cards(group["metrics"], group["currency"])
    else:
        KPI.render_kpi_cards(context["metrics"], currency,
                             context.get("benchmark_metrics"))

    st.subheader(t("dashboard.allocation_title"))
    left, right = st.columns([1, 1], gap="large")
    with left:
        value_key = "weight_pct" if context.get("multi_currency") else "value"
        fig = C.allocation_pie(
            context["allocation"],
            t("dashboard.allocation_pie", currency="" if value_key == "weight_pct" else currency),
            value_key=value_key,
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        weights_pct = [row["weight_pct"] for row in context["allocation"]]
        fig = C.allocation_bars(
            [row["ticker"] for row in context["allocation"]],
            weights_pct,
            t("dashboard.allocation_bars"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader(t("dashboard.performance_title"))
    wealth = context.get("wealth")
    benchmark_returns = context.get("benchmark_returns")
    benchmark_wealth = None
    if benchmark_returns is not None and len(benchmark_returns):
        benchmark_wealth = float(wealth.iloc[0]) * (1.0 + benchmark_returns).cumprod()

    fig = C.wealth_chart(
        wealth, benchmark_wealth,
        title=t("dashboard.wealth_title"),
        y_title=currency,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(t("dashboard.benchmark_title"))
    if context.get("benchmark_metrics"):
        KPI.render_benchmark_card(context["benchmark_metrics"])
        if benchmark_wealth is not None:
            fig = C.benchmark_relative_chart(
                wealth, benchmark_wealth,
                title=t("dashboard.rebased_title", ticker=context.get("benchmark_ticker", "")),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(t("bench.unavailable"))

    fig = C.drawdown_chart(drawdown_series(context["groups"][0]["returns"]),
                           title=t("dashboard.drawdown_title"))
    st.plotly_chart(fig, use_container_width=True)

    market = context.get("market", "US")
    info = MARKETS.get(market, {})
    with st.container(border=True):
        st.markdown(f"**{t('dashboard.market_info')}**")
        st.markdown(
            f"- {t('info.market')}: {info.get('label', market)} — {info.get('exchange', '—')}\n"
            f"- {t('info.currency')}: {', '.join(context.get('currencies', []))}\n"
            f"- {t('info.source')}: {context.get('data_source', '—')}\n"
            f"- {t('info.last')}: {context.get('last_observation') or '—'}\n"
            f"- {t('info.simulated')}: {t('yes' if context.get('simulated') else 'no')}"
        )


def main() -> None:
    state = render_sidebar()
    st.title(f"{APP_NAME} · {t('page.dashboard')}")
    st.caption(t("dashboard.subtitle"))

    context = build_context(state)
    if not is_usable(context):
        st.info(t("dashboard.empty_hint"))
        KPI.render_errors(context.get("errors", []))
        return

    _render_dashboard(context)


if __name__ == "__main__":
    main()
