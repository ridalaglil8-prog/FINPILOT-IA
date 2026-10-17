"""Portfolio Builder — compose the portfolio every other page analyses.

The builder never computes analytics itself: it edits ``st.session_state``
(portfolio, capital, period, benchmark) and validates the weights. The shared
``services.portfolio_service`` is the single source of truth used by all pages.
"""
from __future__ import annotations

import datetime as _dt

import streamlit as st

st.set_page_config(page_title="FinPilot AI — Portfolio Builder", page_icon="🧱", layout="wide")

import config
from components.sidebar import render_sidebar
from i18n import t
from services import market_data

WEIGHT_TOLERANCE_PCT = 0.01  # 0.01 pp of tolerance on the 100 % total


def _weight_key(ticker: str) -> str:
    return f"builder_weight_{ticker}"


def _sync_weight_keys(assets: list) -> None:
    """Make sure every holding has a live number-input value in session state."""
    for asset in assets:
        st.session_state.setdefault(_weight_key(str(asset["ticker"])),
                                    float(asset.get("weight", 0.0)) * 100.0)


def _remove_asset(ticker: str) -> None:
    state = st.session_state
    state["assets"] = [a for a in state["assets"] if str(a["ticker"]) != ticker]
    state.pop(_weight_key(ticker), None)
    st.rerun()


def _add_asset(meta: dict) -> None:
    state = st.session_state
    state["assets"] = list(state["assets"]) + [{
        "ticker": meta["ticker"],
        "name": meta.get("name", meta["ticker"]),
        "asset_class": meta.get("asset_class", ""),
        "currency": meta.get("currency", config.market_currency(state["market"])),
        "weight": 0.0,
    }]
    st.session_state.setdefault(_weight_key(meta["ticker"]), 0.0)
    st.rerun()


def _persist_weights() -> None:
    """Read the live widget values back into the session portfolio."""
    for asset in st.session_state["assets"]:
        ticker = str(asset["ticker"])
        asset["weight"] = float(st.session_state.get(_weight_key(ticker), 0.0)) / 100.0


def _render_holdings() -> tuple[float, bool]:
    """Render the holdings editor; returns ``(total_weight_pct, valid)``."""
    assets = st.session_state["assets"]
    if not assets:
        st.info(t("builder.empty"))
        return 0.0, False

    st.markdown(f"#### {t('builder.weights_title')}")
    total = 0.0
    for asset in assets:
        ticker = str(asset["ticker"])
        w_col, n_col, v_col, r_col = st.columns([4, 3, 2, 1])
        with w_col:
            st.markdown(f"**{ticker}** — {asset.get('name', ticker)}")
            st.caption(f"{asset.get('asset_class', '')} · {asset.get('currency', '')}")
        with n_col:
            st.number_input(
                t("common.weight") + " (%)",
                min_value=0.0, max_value=100.0, step=5.0,
                key=_weight_key(ticker),
                label_visibility="collapsed",
            )
        with v_col:
            weight = float(st.session_state.get(_weight_key(ticker), 0.0))
            total += weight
            value = st.session_state["capital"] * weight / 100.0
            st.markdown(f"**{value:,.0f}**")
        with r_col:
            st.button("🗑", key=f"builder_rm_{ticker}", on_click=_remove_asset,
                      args=(ticker,), help=t("builder.remove"))

    valid = abs(total - 100.0) <= WEIGHT_TOLERANCE_PCT
    st.progress(min(total, 100.0) / 100.0,
                text=f"{t('builder.weight_total')}: {total:.2f} %")
    if valid:
        st.success(t("builder.weights_valid"))
    else:
        st.warning(t("builder.weights_invalid", total=f"{total:.2f}"))
    return total, valid


def _render_add_asset(universe: list) -> None:
    available = [a for a in universe
                 if a["ticker"] not in {str(x["ticker"]) for x in st.session_state["assets"]}]
    if not available:
        return
    st.markdown(f"#### {t('builder.add_label')}")
    labels = {f"{a['ticker']} — {a.get('name', '')}": a for a in available}
    choice = st.selectbox(t("builder.add_label"), list(labels.keys()),
                          label_visibility="collapsed", help=t("builder.add_help"))
    if st.button(t("builder.add_button"), key="builder_add_btn"):
        _add_asset(labels[choice])


def main() -> None:
    state = render_sidebar()
    st.title(f"🧱 {t('page.builder')}")
    st.caption(t("builder.subtitle"))

    market = state["market"]
    data_mode = state["data_mode"]
    universe = market_data.provider_universe(market, data_mode, state.get("imported_prices"))

    if market == config.MARKET_MA and data_mode == config.MODE_HISTORICAL:
        st.info(t("builder.ma_hint"))
    if not universe:
        st.warning(t("builder.empty_universe"))
        _render_settings(state, benchmarks=[])
        return

    _sync_weight_keys(state["assets"])

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(f"#### {t('builder.assets_label')}")
        _render_add_asset(universe)
        _render_holdings()

    with right:
        st.markdown(f"#### {t('settings.section_general')}")
        st.number_input(
            t("builder.capital_label"), min_value=1_000.0, max_value=1e9, step=10_000.0,
            key="capital", help=t("builder.capital_help"),
        )
        st.date_input(t("builder.start_label"), key="start")
        st.date_input(t("builder.end_label"), key="end")

        benchmarks = market_data.provider_benchmarks(market, data_mode,
                                                     state.get("imported_prices"))
        if benchmarks:
            labels = [f"{b['ticker']} — {b.get('name', '')}" for b in benchmarks]
            tickers = [b["ticker"] for b in benchmarks]
            current = state.get("benchmark")
            index = tickers.index(current) if current in tickers else 0
            selection = st.selectbox(t("builder.benchmark_label"), labels, index=index,
                                     key="benchmark_select", help=t("builder.no_benchmark"))
            state["benchmark"] = tickers[labels.index(selection)]
        else:
            st.info(t("builder.no_benchmark"))
            state["benchmark"] = None

    _render_settings(state)


def _render_settings(state) -> None:
    """Apply / reset actions (bottom of the page)."""
    total, valid = (None, None)
    assets = state.get("assets", [])
    if assets:
        total = sum(float(st.session_state.get(_weight_key(str(a["ticker"])), 0.0))
                    for a in assets)
        valid = abs(total - 100.0) <= WEIGHT_TOLERANCE_PCT

    apply_col, reset_col = st.columns([1, 1])
    with apply_col:
        if st.button(t("common.apply"), key="builder_apply", type="primary",
                     disabled=(not assets or not valid)):
            _persist_weights()
            start = st.session_state.get("start")
            end = st.session_state.get("end")
            if isinstance(start, tuple):
                start = start[0] if start else None
            if isinstance(end, tuple):
                end = end[0] if end else None
            if isinstance(start, _dt.datetime):
                start = start.date()
            if isinstance(end, _dt.datetime):
                end = end.date()
            if start:
                state["start"] = start
            if end:
                state["end"] = end
            capital = st.session_state.get("capital")
            if capital:
                state["capital"] = float(capital)
            st.success(t("builder.saved"))
    with reset_col:
        if st.button(t("common.reset"), key="builder_reset"):
            state["assets"] = config.default_assets(state["market"])
            state["benchmark"] = config.default_benchmark(state["market"])
            for asset in state["assets"]:
                st.session_state.pop(_weight_key(str(asset["ticker"])), None)
            st.rerun()


if __name__ == "__main__":
    main()
