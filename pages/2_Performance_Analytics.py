"""Performance Analytics — wealth path, cumulative return, monthly heatmap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FinPilot AI — Performance", page_icon="📈", layout="wide")

from analytics import returns as Ret
from analytics.risk import drawdown_series
from components import charts as C
from components import kpi_cards as KPI
from components.sidebar import render_sidebar
from components.streamlit_compat import plotly_chart
from i18n import t
from services.portfolio_service import build_context, is_usable

MONTHS_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_FR = ["Janv", "Févr", "Mars", "Avril", "Mai", "Juin",
             "Juil", "Août", "Sept", "Oct", "Nov", "Déc"]


def _monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Month-by-month compounded returns as a year x month DataFrame (%)."""
    monthly = (1.0 + returns).groupby([returns.index.year, returns.index.month]).prod() - 1.0
    frame = (monthly * 100.0).unstack(level=1)
    frame.columns = [int(m) for m in frame.columns]
    return frame


def _monthly_heatmap(frame: pd.DataFrame, language: str) -> "object":
    import plotly.graph_objects as go

    labels = (MONTHS_FR if language == "fr" else MONTHS_EN)
    months = list(frame.columns)
    fig = go.Figure(go.Heatmap(
        z=frame.to_numpy(dtype=float),
        x=[labels[m - 1] for m in months],
        y=[str(y) for y in frame.index],
        colorscale="RdYlGn", zmid=0,
        colorbar=dict(title="%", thickness=12),
        hovertemplate="%{y} %{x}<br>%{z:.2f} %<extra></extra>",
        text=[[f"{v:.1f}" if np.isfinite(v) else "" for v in row] for row in frame.to_numpy()],
        texttemplate="%{text}",
    ))
    fig.update_layout(
        height=380, margin=dict(l=40, r=20, t=45, b=30),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E6EDF7"),
        title=dict(text=t("perf.monthly_title"), font=dict(size=15, color="#E6EDF7")),
        xaxis=dict(title=""), yaxis=dict(title="", autorange="reversed"),
    )
    return fig


def main() -> None:
    state = render_sidebar()
    st.title(f"📈 {t('page.performance')}")
    st.caption(t("perf.subtitle"))

    context = build_context(state)
    if not is_usable(context):
        st.info(t("dashboard.empty_hint"))
        KPI.render_errors(context.get("errors", []))
        return

    KPI.render_data_provenance(context)
    group = context["groups"][0]
    returns = group["returns"]
    wealth = group["wealth"]
    currency = group["currency"]
    benchmark_wealth = None
    if context.get("benchmark_returns") is not None and len(context["benchmark_returns"]):
        benchmark_wealth = float(wealth.iloc[0]) * (1.0 + context["benchmark_returns"]).cumprod()

    KPI.render_kpi_cards(group["metrics"], currency, context.get("benchmark_metrics"))

    tab_wealth, tab_cumulative, tab_monthly, tab_drawdown = st.tabs([
        t("dashboard.wealth_title"),
        t("perf.cumulative_title"),
        t("perf.monthly_title"),
        t("dashboard.drawdown_title"),
    ])

    with tab_wealth:
        plotly_chart(
            C.wealth_chart(wealth, benchmark_wealth,
                           title=t("dashboard.wealth_title"), y_title=currency),
        )

    with tab_cumulative:
        cumulative = Ret.cumulative_returns(returns) * 100.0
        benchmark_cumulative = None
        if benchmark_wealth is not None:
            benchmark_cumulative = (benchmark_wealth / float(benchmark_wealth.iloc[0]) - 1.0) * 100.0
        plotly_chart(
            C.wealth_chart(wealth, benchmark_wealth, title=t("perf.cumulative_title"),
                           as_pct_from_start=True),
        )
        st.caption(f"{t('perf.total_return')}: {float(cumulative.iloc[-1]):+.2f} %"
                   + (f" · {context.get('benchmark_ticker', '')}: "
                      f"{float(benchmark_cumulative.iloc[-1]):+.2f} %"
                      if benchmark_cumulative is not None else ""))

    with tab_monthly:
        frame = _monthly_returns(returns)
        plotly_chart(_monthly_heatmap(frame, state.get("language", "en")),)
        st.caption(t("perf.monthly_note"))

        summary = st.columns(4)
        monthly_values = frame.to_numpy(dtype=float)
        finite = monthly_values[np.isfinite(monthly_values)]
        if finite.size:
            best = float(np.nanmax(finite))
            worst = float(np.nanmin(finite))
            positive = float((finite > 0).mean()) * 100.0
            summary[0].metric(t("perf.best_month"), f"{best:+.2f} %")
            summary[1].metric(t("perf.worst_month"), f"{worst:+.2f} %")
            summary[2].metric(t("perf.positive_months"), f"{positive:.0f} %")
            summary[3].metric(t("perf.total_return"),
                              f"{float(Ret.cumulative_returns(returns).iloc[-1]) * 100:+.2f} %")

    with tab_drawdown:
        plotly_chart(
            C.drawdown_chart(drawdown_series(returns), title=t("dashboard.drawdown_title")),
        )

    if context.get("benchmark_metrics"):
        st.subheader(t("perf.benchmark_title"))
        KPI.render_benchmark_card(context["benchmark_metrics"])
        if benchmark_wealth is not None:
            plotly_chart(
                C.benchmark_relative_chart(
                    wealth, benchmark_wealth,
                    title=t("dashboard.rebased_title",
                            ticker=context.get("benchmark_ticker", ""))),
            )
    else:
        st.info(t("perf.no_benchmark"))


if __name__ == "__main__":
    main()
