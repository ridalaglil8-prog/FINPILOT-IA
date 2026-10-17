"""Sidebar navigation, language, data-mode and market selectors.

``render_sidebar()`` also initialises the session state, so every page simply
calls it first and can rely on the defaults being present.
"""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

import config
from i18n import available_languages, t


# ---------------------------------------------------------------------------
# session defaults
# ---------------------------------------------------------------------------
def ensure_state() -> Dict[str, Any]:
    """Populate ``st.session_state`` with defaults (idempotent)."""
    defaults = config.default_state()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value
    if not isinstance(st.session_state.get("assets"), list):
        st.session_state["assets"] = config.default_assets(st.session_state["market"])
    return st.session_state


def _on_language_change() -> None:
    # ``key="language"`` already wrote the value; nothing else to do.
    pass


def _on_market_change() -> None:
    """Switching market resets assets/benchmark to that market's defaults."""
    market = st.session_state.get("market", config.MARKET_US)
    st.session_state["assets"] = config.default_assets(market)
    st.session_state["benchmark"] = config.default_benchmark(market)


def _on_mode_change() -> None:
    """Data-mode switch: nothing to reset, prices are re-fetched on the next run."""
    return


# ---------------------------------------------------------------------------
# navigation
# ---------------------------------------------------------------------------
PAGES = [
    ("app.py", "nav.dashboard", "🏠"),
    ("pages/1_Portfolio_Builder.py", "nav.builder", "🧱"),
    ("pages/2_Performance_Analytics.py", "nav.performance", "📈"),
    ("pages/3_Risk_Diversification.py", "nav.risk", "🛡️"),
    ("pages/4_Portfolio_Optimization.py", "nav.optimization", "⚖️"),
    ("pages/5_Stress_Testing.py", "nav.stress", "🧪"),
    ("pages/6_AI_Assistant.py", "nav.ai", "🤖"),
    ("pages/7_Settings.py", "nav.settings", "⚙️"),
]


def render_navigation() -> None:
    st.markdown(f"### {config.APP_NAME}")
    st.caption(t("sidebar.tagline"))
    st.markdown("")
    for path, key, icon in PAGES:
        try:
            st.page_link(path, label=t(key), icon=icon)
        except Exception:  # pragma: no cover - only if a page file is missing
            # Degrade to a plain link instead of crashing every page.
            st.markdown(f"{icon} [{t(key)}]({path})")


def render_sidebar() -> Dict[str, Any]:
    """Render the whole sidebar and return the (initialised) session state."""
    state = ensure_state()

    with st.sidebar:
        render_navigation()
        st.divider()

        st.markdown(f"**{t('sidebar.preferences')}**")

        language_options = list(available_languages().keys())
        language_labels = available_languages()
        st.selectbox(
            t("sidebar.language"),
            options=language_options,
            format_func=lambda code: language_labels[code],
            key="language",
            on_change=_on_language_change,
            help=t("sidebar.language_help"),
        )

        mode_labels = {
            config.MODE_DEMO: t("mode.demo"),
            config.MODE_HISTORICAL: t("mode.historical"),
        }
        st.radio(
            t("sidebar.data_mode"),
            options=list(config.DATA_MODES),
            format_func=lambda mode: mode_labels[mode],
            key="data_mode",
            on_change=_on_mode_change,
            help=t("sidebar.data_mode_help"),
        )

        market_labels = {
            config.MARKET_US: t("market.us"),
            config.MARKET_MA: t("market.ma"),
        }
        st.selectbox(
            t("sidebar.market"),
            options=[config.MARKET_US, config.MARKET_MA],
            format_func=lambda code: market_labels[code],
            key="market",
            on_change=_on_market_change,
            help=t("sidebar.market_help"),
        )

        st.divider()
        with st.expander(t("sidebar.help")):
            st.markdown(t("sidebar.help_body"))
        with st.expander(t("sidebar.disclaimer")):
            st.markdown(t("sidebar.disclaimer_body"))

        st.caption(t("sidebar.version", version="1.0.0"))

    return state
