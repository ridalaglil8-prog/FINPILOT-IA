"""Portfolio Optimization — long-only min-volatility / max-Sharpe solutions."""
from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="FinPilot AI — Optimization", page_icon="⚖️", layout="wide")

from components import charts as C
from components import kpi_cards as KPI
from components.sidebar import render_sidebar
from i18n import t
from optimization import optimizer
from services.portfolio_service import build_context, is_usable


def _run_optimization(context, objective: str, max_weight: float, risk_free_rate: float):
    """Optimize the primary (largest-currency) sub-portfolio."""
    group = context["groups"][0]
    prices = context["prices"][group["assets"]]
    result = optimizer.optimize(
        prices, objective,
        risk_free_rate=risk_free_rate,
        weight_bounds=(0.0, float(max_weight)),
    )
    current = optimizer.evaluate_weights(
        group["weights"],
        *optimizer.prepare_inputs(prices),
        risk_free_rate,
    )
    comparison = optimizer.compare_allocations(group["weights"], result["weights"])
    return {
        "objective": objective,
        "engine": result["engine"],
        "weights": result["weights"],
        "performance": result["performance"],
        "current_performance": current,
        "comparison": comparison,
        "currency": group["currency"],
        "assets": group["assets"],
        "risk_free_rate": risk_free_rate,
        "observations": result["observations"],
        "disclaimer": True,
    }


def main() -> None:
    state = render_sidebar()
    st.title(f"⚖️ {t('page.optimization')}")
    st.caption(t("opt.subtitle"))

    context = build_context(state)
    if not is_usable(context):
        st.info(t("dashboard.empty_hint"))
        KPI.render_errors(context.get("errors", []))
        return

    KPI.render_data_provenance(context)
    risk_free_rate = float(context.get("risk_free_rate", 0.02))

    with st.sidebar:
        st.markdown(f"**{t('opt.objective_label')}**")
        objective = st.radio(
            t("opt.objective_label"),
            options=["min_volatility", "max_sharpe"],
            format_func=lambda o: t("opt.min_vol" if o == "min_volatility" else "opt.max_sharpe"),
            key="opt_objective", label_visibility="collapsed",
        )
        max_weight = st.slider(
            t("opt.max_weight_label"), min_value=5, max_value=100, value=100, step=5,
            help=t("opt.max_weight_help"),
        ) / 100.0

    if len(context["groups"][0]["assets"]) < 2:
        st.info(t("opt.need_valid"))
        return

    run = st.button(t("opt.run"), key="opt_run", type="primary")

    if run:
        with st.spinner(t("ai.thinking")):
            try:
                st.session_state["optimization"] = _run_optimization(
                    context, objective, max_weight, risk_free_rate
                )
            except optimizer.OptimizationError as exc:
                st.session_state.pop("optimization", None)
                st.error(str(exc))

    result = st.session_state.get("optimization")
    if not result:
        st.info(t("opt.hint"))
        return

    st.warning(t("opt.disclaimer"))
    st.caption(t("opt.engine", engine=result["engine"]))

    current_perf = result["current_performance"]
    new_perf = result["performance"]

    st.subheader(t("opt.results_title"))
    metrics_frame = pd.DataFrame([
        {"metric": t("opt.expected_return"),
         t("opt.current_col"): round(current_perf["expected_return"] * 100.0, 2),
         t("opt.optimized_col"): round(new_perf["expected_return"] * 100.0, 2)},
        {"metric": t("opt.volatility"),
         t("opt.current_col"): round(current_perf["volatility"] * 100.0, 2),
         t("opt.optimized_col"): round(new_perf["volatility"] * 100.0, 2)},
        {"metric": t("opt.sharpe"),
         t("opt.current_col"): round(current_perf["sharpe"], 3),
         t("opt.optimized_col"): round(new_perf["sharpe"], 3)},
    ])
    st.dataframe(metrics_frame, use_container_width=True, hide_index=True)

    row1 = st.columns(3)
    row1[0].metric(t("opt.expected_return"), KPI.fmt_pct(new_perf["expected_return"]),
                   delta=f"{(new_perf['expected_return'] - current_perf['expected_return']) * 100:+.2f} pp")
    row1[1].metric(t("opt.volatility"), KPI.fmt_pct(new_perf["volatility"], signed=False),
                   delta=f"{(new_perf['volatility'] - current_perf['volatility']) * 100:+.2f} pp",
                   delta_color="inverse")
    row1[2].metric(t("opt.sharpe"), KPI.fmt_num(new_perf["sharpe"]),
                   delta=f"{new_perf['sharpe'] - current_perf['sharpe']:+.2f}")

    st.subheader(t("opt.allocations_title"))
    st.plotly_chart(C.comparison_bars(result["comparison"], t("opt.table_title")),
                    use_container_width=True)

    frame = result["comparison"].copy()
    frame.columns = [t("common.asset"), t("opt.current_col"), t("opt.optimized_col"),
                     t("opt.change_col")]
    st.dataframe(frame, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
