"""AI Financial Assistant — explains *calculated* results, never invents them.

Python computes every number (``services.portfolio_service``, the optimizer and
the stress engine). The LLM only receives a structured JSON payload built from
those results and is asked to explain them in the selected language.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="FinPilot AI — Assistant", page_icon="🤖", layout="wide")

from ai import prompts
from components import kpi_cards as KPI
from components.sidebar import render_sidebar
from i18n import current_language, t
from services import ai_service
from services.portfolio_service import build_context


def _enriched_payload(context, language: str) -> str:
    """Attach the latest optimization / stress results, if any, to the payload."""
    enriched = dict(context)
    optimization = st.session_state.get("optimization")
    if optimization:
        enriched["optimization"] = {
            "objective": optimization.get("objective"),
            "engine": optimization.get("engine"),
            "weights_pct": {k: round(v * 100.0, 2)
                            for k, v in (optimization.get("weights") or {}).items()},
            "performance": optimization.get("performance"),
            "current_performance": optimization.get("current_performance"),
        }
    scenario = st.session_state.get("stress_test")
    if scenario:
        enriched["stress_test"] = [
            {
                "currency": g.get("currency"),
                "impact_pct": round(float(g.get("impact_pct", 0.0)), 6),
                "impact_value": round(float(g.get("impact_value", 0.0)), 2),
                "value_after": round(float(g.get("value_after", 0.0)), 2),
                "assumptions": g.get("assumptions", []),
            }
            for g in scenario.get("groups", [])
        ]
    return prompts.build_payload(enriched, language)


def _render_history() -> None:
    for message in st.session_state.get("ai_messages", []):
        role = message.get("role")
        with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
            st.markdown(message.get("content", ""))


def _ask(payload: str, question: str, language: str) -> None:
    history = st.session_state.get("ai_messages", [])
    st.session_state["ai_messages"] = history + [{"role": "user", "content": question}]
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)
    try:
        answer = ai_service.chat(
            prompts.build_user_message(payload, question),
            system_prompt=prompts.build_system_prompt(language),
            history=history,
        )
    except ai_service.AIServiceError as exc:
        st.session_state["ai_messages"] = history
        with st.chat_message("assistant", avatar="🤖"):
            st.error(str(exc))
        return
    st.session_state["ai_messages"] = st.session_state["ai_messages"] + [
        {"role": "assistant", "content": answer}
    ]
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(answer)


def main() -> None:
    state = render_sidebar()
    st.title(f"🤖 {t('page.ai')}")
    st.caption(t("ai.subtitle"))

    language = current_language()
    configured = ai_service.is_configured()

    with st.expander(t("ai.status_label"), expanded=not configured):
        if configured:
            st.success(ai_service.status_text())
            st.caption(t("ai.privacy_note"))
        else:
            st.warning(t("ai.not_configured"))
            st.caption(t("ai.configure_hint"))

    context = build_context(state)
    usable = bool(context.get("ok")) and not context.get("errors")

    if not usable:
        st.info(t("ai.no_portfolio"))
        KPI.render_errors(context.get("errors", []))
    else:
        KPI.render_data_provenance(context)

    if st.button(t("ai.clear"), key="ai_clear"):
        st.session_state["ai_messages"] = []
        st.rerun()

    _render_history()

    if not usable:
        return

    payload = _enriched_payload(context, language)

    pending = None
    with st.expander(t("ai.suggested_title")):
        for index, question in enumerate(prompts.example_questions(language)):
            if st.button(question, key=f"ai_q_{index}"):
                pending = question

    question = st.chat_input(t("ai.input_hint"))
    if pending:
        question = pending
    if question and question.strip():
        _ask(payload, question.strip(), language)

    st.caption(t("ai.answered_by"))


if __name__ == "__main__":
    main()
