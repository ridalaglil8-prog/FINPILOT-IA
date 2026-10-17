"""Settings — CSV import, risk-free rate, FX rate, AI status, session reset."""
from __future__ import annotations

import numpy as np
import streamlit as st

st.set_page_config(page_title="FinPilot AI — Settings", page_icon="⚙️", layout="wide")

from components.sidebar import render_sidebar
from config import RISK_FREE_RATE_BOUNDS
from data import csv_loader
from i18n import t
from services import ai_service


def _render_csv_import(state) -> None:
    st.markdown(f"#### {t('settings.csv_section')}")
    st.caption(t("settings.csv_help"))

    with st.expander(t("settings.csv_example_title")):
        st.code(
            "date,ATW,LDH\n"
            "2023-01-02,412.50,285.10\n"
            "2023-01-03,415.00,286.40\n"
            "\n# ou format long :\n"
            "date,ticker,price\n"
            "2023-01-02,ATW,412.50\n"
            "2023-01-03,ATW,415.00",
            language="text",
        )

    uploaded = st.file_uploader(t("settings.csv_upload_label"), type=["csv"],
                                key="csv_uploader")
    if uploaded is not None:
        try:
            frame, meta = csv_loader.load_price_csv(uploaded)
        except csv_loader.CSVValidationError as exc:
            st.error(f"{t('settings.csv_error')}: {exc}")
        else:
            state["imported_prices"] = frame
            state["import_revision"] = int(state.get("import_revision", 0)) + 1
            st.session_state["csv_meta"] = meta
            st.success(t("settings.csv_loaded",
                         rows=meta["rows"], columns=len(meta["columns"])))

    meta = st.session_state.get("csv_meta")
    imported = state.get("imported_prices")
    if imported is not None and meta:
        st.info(f"{t('settings.csv_range')}: {meta['date_min']} → {meta['date_max']}")
        st.caption(f"{t('settings.csv_columns')}: {', '.join(meta['columns'])}")
        for warning in meta.get("warnings", []):
            st.warning(warning)
        if st.button(t("settings.csv_clear"), key="csv_clear"):
            state["imported_prices"] = None
            state["import_revision"] = int(state.get("import_revision", 0)) + 1
            st.session_state.pop("csv_meta", None)
            st.rerun()


def main() -> None:
    state = render_sidebar()
    st.title(f"⚙️ {t('page.settings')}")
    st.caption(t("settings.subtitle"))

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(f"#### {t('settings.section_general')}")
        rate_pct = st.number_input(
            t("settings.risk_free_label"),
            min_value=RISK_FREE_RATE_BOUNDS[0] * 100,
            max_value=RISK_FREE_RATE_BOUNDS[1] * 100,
            value=float(state.get("risk_free_rate", 0.02)) * 100.0,
            step=0.25, help=t("settings.risk_free_help"), key="risk_free_input",
        )
        if st.button(t("common.save"), key="save_risk_free"):
            state["risk_free_rate"] = float(rate_pct) / 100.0
            st.success(t("settings.risk_free_saved"))

        st.markdown(f"#### {t('settings.fx_section')}")
        fx_value = state.get("fx_usd_mad")
        fx_input = st.number_input(
            t("settings.fx_label"), min_value=0.0, max_value=1000.0, step=0.01,
            value=float(fx_value) if fx_value else 0.0,
            help=t("settings.fx_help"), key="fx_input",
        )
        fx_col, clear_col = st.columns(2)
        with fx_col:
            if st.button(t("common.save"), key="save_fx"):
                if fx_input and np.isfinite(fx_input) and fx_input > 0:
                    state["fx_usd_mad"] = float(fx_input)
                    st.success(t("settings.fx_saved"))
                else:
                    state["fx_usd_mad"] = None
                    st.info(t("settings.fx_cleared"))
        with clear_col:
            if st.button(t("common.clear"), key="clear_fx"):
                state["fx_usd_mad"] = None
                st.info(t("settings.fx_cleared"))

        st.markdown(f"#### {t('settings.ai_section')}")
        st.caption(f"{t('settings.ai_status_label')}: {ai_service.status_text()}")
        if not ai_service.is_configured():
            st.caption(t("ai.configure_hint"))

    with right:
        _render_csv_import(state)

        st.markdown(f"#### {t('settings.session_section')}")
        if st.button(t("settings.reset_label"), key="reset_session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success(t("settings.reset_done"))

    st.caption(t("sidebar.disclaimer_body"))


if __name__ == "__main__":
    main()
