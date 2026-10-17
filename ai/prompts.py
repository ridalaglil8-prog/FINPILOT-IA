"""LLM prompt construction for the FinPilot AI assistant.

Design rule
-----------
Python computes every number. The model only *explains* structured results
that it is given. The system prompt explicitly forbids inventing prices,
statistics, market observations or optimisation outputs.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional

ROUND = 4

LANG_NAMES = {"fr": "French (français)", "en": "English"}

SYSTEM_TEMPLATE = """You are FinPilot AI, the educational assistant embedded in the
FinPilot AI portfolio-analytics application. Your ONLY job is to explain results
that were computed by the application's Python analytics engine and provided to
you in the user message.

Language: answer ALWAYS in {language}.

Hard rules:
1. Never invent, estimate or recall prices, returns, volatility, Sharpe ratios,
   drawdowns, correlations, index levels or optimisation weights. If a figure is
   not in the provided payload, say it is not available.
2. Never do your own authoritative calculations. Explain the numbers you were given.
3. This is an educational tool: no trading instructions, no promises of returns,
   no predictions presented as facts. Stress scenarios are hypothetical, and
   optimisation is historical, not a forecast.
4. Be concise and clear (short paragraphs or 4-6 bullet points), use plain
   language, and define jargon in one clause.
5. If the payload shows a warning or an unavailable metric, mention it honestly.
6. End with one short sentence reminding the user that this is educational
   information, not investment advice.
"""

def build_system_prompt(language: str = "en") -> str:
    return SYSTEM_TEMPLATE.format(language=LANG_NAMES.get(language, LANG_NAMES["en"]))


def _round(value: Any) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, ROUND)


def _metrics_block(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "initial_capital", "current_value", "cumulative_return", "annualized_return",
        "annualized_volatility", "sharpe_ratio", "max_drawdown", "observations",
        "var_95", "cvar_95", "herfindahl", "effective_n", "max_weight",
        "average_correlation", "benchmark_cumulative_return",
    )
    return {k: _round(metrics[k]) for k in keys if k in metrics}


def build_payload(context: Mapping[str, Any], language: str = "en") -> str:
    """Serialise the analytics context into a compact, structured JSON payload.

    ``context`` is produced by ``services.portfolio_service`` — the same dict
    the UI renders, so the model can never see numbers the user did not see.
    """
    payload: Dict[str, Any] = {
        "language": language,
        "market": context.get("market"),
        "data_mode": context.get("data_mode"),
        "currency": context.get("currency"),
        "currencies": context.get("currencies"),
        "period": {"start": context.get("start"), "end": context.get("end")},
        "data_source": context.get("data_source"),
        "last_observation": context.get("last_observation"),
        "simulation_warning": bool(context.get("simulated")),
        "portfolio": {
            "assets": context.get("assets"),
            "weights_pct": context.get("weights_pct"),
            "allocation": [
                {"ticker": a.get("ticker"), "weight_pct": _round(a.get("weight_pct")),
                 "currency": a.get("currency"), "value": _round(a.get("value"))}
                for a in context.get("allocation", [])
            ],
        },
        "metrics": _metrics_block(context.get("metrics", {})),
        "sub_portfolios": [
            {"currency": g.get("currency"),
             "assets": g.get("assets"),
             "weight_pct": _round(g.get("weight_pct")),
             "metrics": _metrics_block(g.get("metrics", {}))}
            for g in context.get("groups", [])
        ],
        "correlation": context.get("correlation"),
        "top_risk_contributors": context.get("risk_contributors"),
        "optimization": context.get("optimization"),
        "stress_test": context.get("stress_test"),
        "warnings": context.get("warnings", []),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return text


def build_user_message(payload: str, question: str) -> str:
    """Attach the user's question to the structured payload."""
    return f"{payload}\n\nUser question: {question}"


def example_questions(language: str = "en") -> List[str]:
    """Suggested prompts shown under the chat input."""
    if language == "fr":
        return [
            "Explique mon ratio de Sharpe et ce qu'il signifie.",
            "Ma porte-feuille est-elle trop concentrée ?",
            "Compare mon allocation actuelle à l'allocation optimisée.",
            "Résume le scénario de stress et son impact.",
            "Pourquoi la volatilité annuelle est-elle calculée sur 252 jours ?",
        ]
    return [
        "Explain my Sharpe ratio and what it means.",
        "Is my portfolio too concentrated?",
        "Compare my current allocation with the optimised one.",
        "Summarise the stress scenario and its impact.",
        "Why is annual volatility computed over 252 trading days?",
    ]
