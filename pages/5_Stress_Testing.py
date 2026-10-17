"""Stress Testing — hypothetical shocks on assets or asset classes.

Scenario inputs are live: results are recomputed on every rerun, so what is
shown (and what the AI assistant is later told) always matches the shocks
currently on screen. The scenario is applied to the portfolio's *current*
value, per currency sub-portfolio.
"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

st.set_page_config(page_title="FinPilot AI — Stress Testing", page_icon="🧪", layout="wide")

from analytics import stress as Stress
from components import charts as C
from components import kpi_cards as KPI
from components.sidebar import render_sidebar
from i18n import t
from services.portfolio_service import build_context, is_usable, stress_fingerprint

PRESETS = ("us", "ma", "mixed", "bond", "custom")


def _shock_key(asset_class: str) -> str:
    return f"stress_group_{asset_class}"


def _asset_shock_key(ticker: str) -> str:
    return f"stress_asset_{ticker}"


def _as_float(value, default: float = 0.0) -> float:
    """Session values may be missing or malformed — never let that reach math."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _preset_shocks(preset: str, asset_classes) -> dict:
    """Percentage shocks defined by a preset (empty for ``custom``)."""
    if preset == "ma":
        return {c: -15.0 for c in asset_classes if str(c).startswith("MA")}
    if preset == "bond":
        return {"Bond ETF": -8.0}
    if preset == "mixed":
        return {c: (-20.0 if ("Equity" in c or "Stock" in c) else -5.0)
                for c in asset_classes}
    if preset == "us":
        return {"US Equity ETF": -10.0, "US Stock": -10.0}
    return {}


def _apply_preset(preset, asset_classes, tickers) -> None:
    """Replace the whole scenario with a preset (``custom`` edits nothing).

    Called as an ``on_change`` callback — i.e. only when the preset actually
    changes — and once while seeding the default, so it can never overwrite a
    manual edit during an unrelated rerun.
    """
    if preset == "custom":
        return
    shocks = _preset_shocks(preset, asset_classes)
    for asset_class in asset_classes:
        st.session_state[_shock_key(asset_class)] = float(shocks.get(asset_class, 0.0))
    for ticker in tickers:
        st.session_state[_asset_shock_key(ticker)] = 0.0


def _on_preset_change(asset_classes, tickers) -> None:
    _apply_preset(st.session_state.get("stress_preset", PRESETS[0]),
                  asset_classes, tickers)


def _run_scenario(context) -> dict:
    """Apply the configured shocks to every currency sub-portfolio."""
    groups_out = []
    for group in context["groups"]:
        currency = group["currency"]
        class_shocks = {
            asset_class: _as_float(st.session_state.get(_shock_key(asset_class)))
            for asset_class in {
                str(a.get("asset_class", "")) for a in context["allocation"]
                if a.get("currency") == currency and str(a.get("asset_class", ""))
            }
        }
        # Per-asset shocks are passed only when non-zero: an untouched widget
        # reads 0 and must not cancel a class-level shock (an explicit
        # non-zero value still wins over its class, as documented).
        asset_shocks = {}
        for ticker in group["assets"]:
            value = _as_float(st.session_state.get(_asset_shock_key(ticker)))
            if value:
                asset_shocks[ticker] = value

        assets = [
            {"ticker": str(ticker),
             "asset_class": next((str(a.get("asset_class", ""))
                                  for a in context["allocation"]
                                  if str(a["ticker"]) == str(ticker)), "")}
            for ticker in group["assets"]
        ]
        shocks = Stress.group_shocks(assets, class_shocks, asset_shocks)

        # The scenario hits the portfolio as it stands *today*, so "value
        # before" matches the current value shown on the dashboard.
        current_value = float(group["wealth"].iloc[-1])
        result = Stress.apply_shocks(
            group["weights"], shocks, current_value,
            currencies={str(a["ticker"]): str(a.get("currency", ""))
                        for a in context["allocation"]},
        )
        result["currency"] = currency
        result["wealth_before"] = current_value
        groups_out.append(result)
    return {"groups": groups_out, "simulated": context.get("simulated", False)}


def _render_scenario(scenario: dict) -> None:
    st.subheader(t("stress.results_title"))
    for result in scenario["groups"]:
        currency = result["currency"]
        with st.container(border=True):
            st.markdown(f"**{currency}**")
            row = st.columns(4)
            row[0].metric(t("stress.value_before"), KPI.fmt_money(result["capital"], currency))
            row[1].metric(t("stress.impact_pct"), KPI.fmt_pct(result["impact_pct"]),
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

    asset_classes = sorted({str(a.get("asset_class", "")) for a in context["allocation"]
                            if str(a.get("asset_class", ""))})
    tickers = [str(a["ticker"]) for a in context["allocation"]]

    # Seed the default preset once, before any widget key exists, so the page
    # opens with a coherent scenario instead of empty (or defaulted) inputs.
    if "stress_preset" not in st.session_state:
        _apply_preset(PRESETS[0], asset_classes, tickers)

    # Every shock key must exist before its widget is created, otherwise
    # ``number_input`` would default to its minimum (−100 %). The keys are
    # seeded rather than passed as ``value=`` on purpose: an explicit default
    # plus a Session State write makes Streamlit warn about the duplication.
    for asset_class in asset_classes:
        st.session_state.setdefault(_shock_key(asset_class), 0.0)
    for ticker in tickers:
        st.session_state.setdefault(_asset_shock_key(ticker), 0.0)

    preset_labels = {
        "us": t("stress.preset_us"), "ma": t("stress.preset_ma"),
        "mixed": t("stress.preset_mixed"), "bond": t("stress.preset_bond"),
        "custom": t("stress.custom"),
    }
    st.radio(t("stress.presets_title"), list(preset_labels.keys()),
             format_func=lambda p: preset_labels[p], horizontal=True,
             key="stress_preset", on_change=_on_preset_change,
             args=(asset_classes, tickers), label_visibility="collapsed")

    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        st.markdown(f"#### {t('stress.group_title')}")
        for asset_class in asset_classes:
            st.number_input(
                f"{asset_class}", min_value=-100.0, max_value=100.0,
                step=1.0,
                key=_shock_key(asset_class), help=t("stress.shock_help"),
            )
    with col_right:
        st.markdown(f"#### {t('stress.asset_title')}")
        for row in context["allocation"]:
            st.number_input(
                f"{row['ticker']} ({row['currency']})", min_value=-100.0, max_value=100.0,
                step=1.0,
                key=_asset_shock_key(row["ticker"]), help=t("stress.shock_help"),
            )

    any_shock = any(_as_float(st.session_state.get(_shock_key(c))) for c in asset_classes) or any(
        _as_float(st.session_state.get(_asset_shock_key(r["ticker"])))
        for r in context["allocation"])

    if not any_shock:
        # Nothing is shocked: drop any previously stored scenario so no other
        # page (or the assistant) can quote an obsolete one.
        st.session_state.pop("stress_test", None)
        st.info(t("stress.no_shocks"))
        return

    try:
        scenario = _run_scenario(context)
    except (ValueError, TypeError) as exc:      # ShockError derives from ValueError
        st.session_state.pop("stress_test", None)
        st.error(str(exc))
        return

    scenario["signature"] = stress_fingerprint(context)
    st.session_state["stress_test"] = scenario

    _render_scenario(scenario)


if __name__ == "__main__":
    main()
