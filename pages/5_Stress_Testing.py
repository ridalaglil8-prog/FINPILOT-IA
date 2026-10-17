"""Stress Testing — hypothetical shocks on assets or asset classes."""
from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="FinPilot AI — Stress Testing", page_icon="🧪", layout="wide")

from analytics import stress as Stress
from components import charts as C
from components import kpi_cards as KPI
from components.sidebar import render_sidebar
from i18n import t
from services.portfolio_service import build_context, is_usable

PRESETS = ("us", "ma", "mixed", "bond", "custom")


def _shock_key(asset_class: str) -> str:
    return f"stress_group_{asset_class}"


def _asset_shock_key(ticker: str) -> str:
    return f"stress_asset_{ticker}"


def _run_scenario(context) -> dict:
    """Apply the configured shocks to every currency sub-portfolio."""
    groups_out = []
    for group in context["groups"]:
        class_shocks = {
            str(a.get("asset_class", "")): float(st.session_state.get(
                _shock_key(str(a.get("asset_class", ""))), 0.0))
            for a in context["allocation"]
            if a.get("currency") == group["currency"]
        }
        # keep only classes actually present in this group
        class_shocks = {k: v for k, v in class_shocks.items() if k}
        asset_shocks = {
            ticker: float(st.session_state.get(_asset_shock_key(ticker), 0.0))
            for ticker in group["assets"]
        }
        shocks = Stress.group_shocks(
            [{"ticker": t, "asset_class": next(
                (str(a.get("asset_class", "")) for a in context["allocation"]
                 if a["ticker"] == t), "")}
             for t in group["assets"]],
            class_shocks, asset_shocks,
        )
        result = Stress.apply_shocks(
            group["weights"], shocks, group["capital"],
            currencies={a["ticker"]: group["currency"] for a in context["allocation"]},
        )
        result["currency"] = group["currency"]
        result["wealth_before"] = float(group["wealth"].iloc[-1])
        groups_out.append(result)
    return {"groups": groups_out, "simulated": context.get("simulated", False)}


def main() -> None:
    state = render_sidebar()
    st.title(f"🧪 {t('page.stress')}")
    st.caption(t("stress.subtitle"))

    context = build_context(state)
    if not is_usable(context):
        st.info(t("dashboard.empty_hint"))
        KPI.render_errors(context.get("errors", []))
        return

    KPI.render_data_provenance(context)
    st.warning(t("stress.hypothetical"))

    # --- scenario definition ---------------------------------------------
    st.subheader(t("stress.presets_title"))
    preset_labels = {
        "us": t("stress.preset_us"), "ma": t("stress.preset_ma"),
        "mixed": t("stress.preset_mixed"), "bond": t("stress.preset_bond"),
        "custom": t("stress.custom"),
    }
    preset = st.radio(t("stress.presets_title"), list(preset_labels.keys()),
                      format_func=lambda p: preset_labels[p], horizontal=True,
                      key="stress_preset", label_visibility="collapsed")

    asset_classes = sorted({str(a.get("asset_class", "")) for a in context["allocation"]
                            if str(a.get("asset_class", ""))})

    if preset != "custom":
        if preset == "us":
            shocks = {"US Equity ETF": -10.0, "US Stock": -10.0}
        elif preset == "ma":
            shocks = {c: -15.0 for c in asset_classes if c.startswith("MA")}
        elif preset == "bond":
            shocks = {"Bond ETF": -8.0}
        else:  # mixed
            shocks = {c: (-20.0 if "Equity" in c or "Stock" in c else -5.0)
                      for c in asset_classes}
        for asset_class in asset_classes:
            st.session_state[_shock_key(asset_class)] = shocks.get(asset_class, 0.0)

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.markdown(f"#### {t('stress.group_title')}")
        for asset_class in asset_classes:
            st.number_input(
                f"{asset_class}", min_value=-100.0, max_value=100.0, step=1.0,
                key=_shock_key(asset_class), help=t("stress.shock_help"),
            )
    with col_right:
        st.markdown(f"#### {t('stress.asset_title')}")
        for row in context["allocation"]:
            st.number_input(
                f"{row['ticker']} ({row['currency']})", min_value=-100.0, max_value=100.0,
                step=1.0, key=_asset_shock_key(row["ticker"]), help=t("stress.shock_help"),
            )

    any_shock = any(
        float(st.session_state.get(_shock_key(c), 0.0)) != 0.0 for c in asset_classes
    ) or any(float(st.session_state.get(_asset_shock_key(r["ticker"]), 0.0)) != 0.0
             for r in context["allocation"])

    st.button(t("stress.run"), key="stress_run", type="primary", disabled=not any_shock)

    if not any_shock:
        st.info(t("stress.no_shocks"))
        return

    try:
        scenario = _run_scenario(context)
        st.session_state["stress_test"] = scenario
    except Stress.ShockError as exc:
        st.session_state.pop("stress_test", None)
        st.error(str(exc))
        return

    # --- results ----------------------------------------------------------
    st.subheader(t("stress.results_title"))
    for result in scenario["groups"]:
        currency = result["currency"]
        with st.container(border=True):
            st.markdown(f"**{currency}**")
            row = st.columns(4)
            row[0].metric(t("stress.value_before"), KPI.fmt_money(result["capital"], currency))
            row[1].metric(t("stress.impact_pct"), KPI.fmt_pct(result["impact_pct"], signed=False),
                          delta=f"{result['impact_pct'] * 100:+.2f} pp")
            row[2].metric(t("stress.impact_value"),
                          KPI.fmt_money(result["impact_value"], currency))
            row[3].metric(t("stress.value_after"),
                          KPI.fmt_money(result["value_after"], currency))

            if result["assumptions"]:
                st.markdown(f"**{t('stress.assumptions')}** · " + ", ".join(
                    f"{a['ticker']} {a['shock'] * 100:+.1f} %" for a in result["assumptions"]
                ))

            affected = [r for r in result["rows"] if r["shock"] != 0.0]
            if affected:
                st.plotly_chart(C.stress_chart(affected, t("stress.table_impact")),
                                use_container_width=True)

            frame = pd.DataFrame([
                {
                    t("stress.table_ticker"): r["ticker"],
                    t("stress.table_shock"): f"{r['shock'] * 100:+.1f} %",
                    t("stress.table_before"): round(r["value_before"], 2),
                    t("stress.table_after"): round(r["value_after"], 2),
                    t("stress.table_impact"): round(r["impact"], 2),
                }
                for r in result["rows"]
            ])
            st.dataframe(frame, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
