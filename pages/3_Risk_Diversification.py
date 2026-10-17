"""Risk & Diversification — correlation, risk contributions, concentration."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="FinPilot AI — Risk & Diversification", page_icon="🛡️", layout="wide")

from analytics import correlation as Corr
from analytics import risk as Risk
from components import charts as C
from components import kpi_cards as KPI
from components.sidebar import render_sidebar
from components.streamlit_compat import plotly_chart
from i18n import t
from services.portfolio_service import build_context, is_usable


def _render_concentration(metrics) -> None:
    st.markdown(f"#### {t('risk.concentration_title')}")
    cols = st.columns(5)
    cols[0].metric(t("risk.herfindahl"), f"{metrics.get('herfindahl', 0.0):.4f}")
    cols[1].metric(t("risk.effective_n"), f"{metrics.get('effective_n', 0.0):.2f}")
    cols[2].metric(t("risk.max_weight"), KPI.fmt_pct(metrics.get("max_weight"), signed=False))
    cols[3].metric(t("risk.top3"), KPI.fmt_pct(metrics.get("top3_weight"), signed=False))
    cols[4].metric(t("risk.n_assets"), f"{metrics.get('n_assets', 0):.0f}")
    if metrics.get("n_assets", 0) <= 1:
        st.info(t("risk.single_asset_note"))


def main() -> None:
    state = render_sidebar()
    st.title(f"🛡️ {t('page.risk')}")
    st.caption(t("risk.subtitle"))

    context = build_context(state)
    if not is_usable(context):
        st.info(t("dashboard.empty_hint"))
        KPI.render_errors(context.get("errors", []))
        return

    KPI.render_data_provenance(context)
    group = context["groups"][0]
    returns = group["returns"]
    currency = group["currency"]

    KPI.render_kpi_cards(group["metrics"], currency, context.get("benchmark_metrics"))

    st.metric(t("risk.cvar95"), KPI.fmt_pct(group["metrics"].get("cvar_95"), signed=False))

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(f"#### {t('risk.correlation_title')}")
        correlation = context.get("correlation")
        if correlation and correlation.get("labels"):
            plotly_chart(
                C.correlation_heatmap(correlation["labels"], correlation["matrix"],
                                      title=t("risk.correlation_title")),
            )
            st.caption(t("risk.avg_correlation",
                         value=f"{correlation.get('average', 0.0):.3f}"))
        else:
            st.info(t("risk.no_correlation"))

    with right:
        st.markdown(f"#### {t('risk.risk_contrib_title')}")
        contributors = context.get("risk_contributors")
        if contributors:
            plotly_chart(
                C.allocation_bars([c["ticker"] for c in contributors],
                                  [c["risk_pct"] for c in contributors],
                                  title=t("risk.risk_contrib_title")),
            )
            st.caption(t("risk.risk_contrib_note"))
        else:
            st.info(t("risk.no_correlation"))

    st.markdown("---")
    _render_concentration(group["metrics"])

    st.markdown("---")
    tab_hist, tab_vol = st.tabs([t("risk.histogram_title"),
                                 t("risk.rolling_vol_title", window=60)])
    with tab_hist:
        plotly_chart(C.histogram(returns, title=t("risk.histogram_title")),)
    with tab_vol:
        try:
            rolling = Corr.rolling_volatility(returns, window=60)
            plotly_chart(C.rolling_vol_chart(rolling, title=t("risk.rolling_vol_title",
                                                                 window=60)),)
        except Exception:
            st.info(t("code.short_history"))


if __name__ == "__main__":
    main()
