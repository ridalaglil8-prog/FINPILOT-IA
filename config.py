"""FinPilot AI — static application configuration.

This module is intentionally free of any Streamlit or network side effects so
that it can be imported by the analytics engine and by the test-suite.

Formulas, conventions and assumptions are documented in ``README.md``.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Identity / branding
# ---------------------------------------------------------------------------
APP_NAME = "FinPilot AI"
PAGE_ICON = "🧭"
LAYOUT = "wide"

# ---------------------------------------------------------------------------
# Financial conventions
# ---------------------------------------------------------------------------
TRADING_DAYS = 252                 # US annualisation convention (252 trading days)
DEFAULT_RISK_FREE_RATE = 0.02      # 2.0 % annualised, configurable in Settings
RISK_FREE_RATE_BOUNDS = (0.0, 0.15)

MIN_HISTORY_OBSERVATIONS = 2       # absolute minimum to compute anything
MIN_RELIABLE_OBSERVATIONS = 60     # below this we warn (optimisation needs more)
MIN_OPTIMIZATION_OBSERVATIONS = 60 # hard floor for portfolio optimisation
WEIGHT_SUM_TOLERANCE = 1e-6
ANALYSIS_PERIOD_YEARS = 3

# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------
MARKET_US = "US"
MARKET_MA = "MA"
MARKETS: Dict[str, Dict[str, Any]] = {
    MARKET_US: {
        "label": "United States",
        "exchange": "NYSE / NASDAQ",
        "currency": "USD",
        "benchmark": "^GSPC",
        "benchmark_label": "S&P 500",
    },
    MARKET_MA: {
        "label": "Maroc / Morocco",
        "exchange": "Casablanca Stock Exchange",
        "currency": "MAD",
        "benchmark": None,          # no verified index feed available -> user CSV / demo
        "benchmark_label": "",
    },
}

CURRENCY_SYMBOLS = {"USD": "$", "MAD": "MAD "}

# ---------------------------------------------------------------------------
# US asset universe (public metadata only — no embedded prices).
# Prices are fetched from yfinance in "Historical Data" mode and generated
# deterministically in "Demo" mode.
# ---------------------------------------------------------------------------
US_ASSETS: List[Dict[str, str]] = [
    {"ticker": "SPY",   "name": "SPDR S&P 500 ETF",            "asset_class": "US Equity ETF",  "currency": "USD"},
    {"ticker": "QQQ",   "name": "Invesco QQQ Trust",           "asset_class": "US Equity ETF",  "currency": "USD"},
    {"ticker": "IWM",   "name": "iShares Russell 2000 ETF",    "asset_class": "US Equity ETF",  "currency": "USD"},
    {"ticker": "VTI",   "name": "Vanguard Total Stock Market", "asset_class": "US Equity ETF",  "currency": "USD"},
    {"ticker": "AAPL",  "name": "Apple Inc.",                  "asset_class": "US Stock",       "currency": "USD"},
    {"ticker": "MSFT",  "name": "Microsoft Corp.",             "asset_class": "US Stock",       "currency": "USD"},
    {"ticker": "GOOGL", "name": "Alphabet Inc. Class A",       "asset_class": "US Stock",       "currency": "USD"},
    {"ticker": "AMZN",  "name": "Amazon.com Inc.",             "asset_class": "US Stock",       "currency": "USD"},
    {"ticker": "NVDA",  "name": "NVIDIA Corp.",                "asset_class": "US Stock",       "currency": "USD"},
    {"ticker": "JPM",   "name": "JPMorgan Chase & Co.",        "asset_class": "US Stock",       "currency": "USD"},
    {"ticker": "JNJ",   "name": "Johnson & Johnson",           "asset_class": "US Stock",       "currency": "USD"},
    {"ticker": "XOM",   "name": "Exxon Mobil Corp.",           "asset_class": "US Stock",       "currency": "USD"},
    {"ticker": "WMT",   "name": "Walmart Inc.",                "asset_class": "US Stock",       "currency": "USD"},
    {"ticker": "V",     "name": "Visa Inc.",                   "asset_class": "US Stock",       "currency": "USD"},
    {"ticker": "PG",    "name": "Procter & Gamble Co.",        "asset_class": "US Stock",       "currency": "USD"},
    {"ticker": "TLT",   "name": "iShares 20+ Year Treasury",   "asset_class": "Bond ETF",       "currency": "USD"},
    {"ticker": "GLD",   "name": "SPDR Gold Shares",            "asset_class": "Commodity ETF",  "currency": "USD"},
    {"ticker": "BND",   "name": "Vanguard Total Bond Market",  "asset_class": "Bond ETF",       "currency": "USD"},
]

US_BENCHMARKS: List[Dict[str, str]] = [
    {"ticker": "^GSPC", "name": "S&P 500 Index", "currency": "USD"},
    {"ticker": "^IXIC", "name": "NASDAQ Composite", "currency": "USD"},
    {"ticker": "SPY",   "name": "SPDR S&P 500 ETF", "currency": "USD"},
]

# ---------------------------------------------------------------------------
# Moroccan market universe.
#
# IMPORTANT: FinPilot AI ships with **no** Moroccan securities by default.
# Casablanca Stock Exchange instruments are deliberately NOT listed here,
# because no verified market-data feed is bundled with the application and
# inventing tickers/prices would be misleading. Users can:
#   * use Demo mode (clearly labelled SIMULATED instruments), or
#   * import their own historical prices as CSV (Settings page), or
#   * extend this list with metadata they have verified themselves.
# ---------------------------------------------------------------------------
MOROCCO_ASSETS: List[Dict[str, str]] = []

# Clearly-labelled synthetic instruments used only in Demo mode for the
# Moroccan market view. These are NOT real Casablanca listings.
MOROCCO_DEMO_ASSETS: List[Dict[str, str]] = [
    {"ticker": "SIM_MA_BANK",  "name": "SIMULATED Moroccan banks basket",     "asset_class": "MA Equity (simulated)", "currency": "MAD"},
    {"ticker": "SIM_MA_TEL",   "name": "SIMULATED telecom basket",            "asset_class": "MA Equity (simulated)", "currency": "MAD"},
    {"ticker": "SIM_MA_IND",   "name": "SIMULATED industrials basket",        "asset_class": "MA Equity (simulated)", "currency": "MAD"},
    {"ticker": "SIM_MA_AGRI",  "name": "SIMULATED agri-food basket",          "asset_class": "MA Equity (simulated)", "currency": "MAD"},
    {"ticker": "SIM_MA_IX",    "name": "SIMULATED broad MA market index",     "asset_class": "MA Index (simulated)",  "currency": "MAD"},
]

# ---------------------------------------------------------------------------
# Data modes
# ---------------------------------------------------------------------------
MODE_DEMO = "demo"
MODE_HISTORICAL = "historical"
DATA_MODES = (MODE_DEMO, MODE_HISTORICAL)

# ---------------------------------------------------------------------------
# LLM defaults (secrets always come from the environment / .env)
# ---------------------------------------------------------------------------
LLM_DEFAULTS = {
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "timeout": 30,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def asset_metadata(market: str, demo: bool = False) -> List[Dict[str, str]]:
    """Return the configurable asset universe for ``market``."""
    if market == MARKET_MA:
        if demo:
            return list(MOROCCO_DEMO_ASSETS)
        return list(MOROCCO_ASSETS)
    return list(US_ASSETS)


def benchmark_universe(market: str, demo: bool = False) -> List[Dict[str, str]]:
    """Return selectable benchmarks for ``market`` (may be empty)."""
    if market == MARKET_MA:
        return list(MOROCCO_DEMO_ASSETS[-1:]) if demo else []
    return list(US_BENCHMARKS)


def default_benchmark(market: str) -> Optional[str]:
    if market == MARKET_MA:
        return None
    return MARKETS[market]["benchmark"]


def market_currency(market: str) -> str:
    return MARKETS[market]["currency"]


def default_assets(market: str) -> List[Dict[str, Any]]:
    """A sensible starting portfolio for the selected market (weights = 100 %)."""
    if market == MARKET_MA:
        assets = MOROCCO_DEMO_ASSETS[:4]
        weights = [0.40, 0.25, 0.20, 0.15]
    else:
        assets = [a for a in US_ASSETS if a["ticker"] in ("SPY", "QQQ", "TLT", "GLD")]
        weights = [0.45, 0.25, 0.20, 0.10]
    return [
        {"ticker": a["ticker"], "name": a["name"], "asset_class": a["asset_class"],
         "currency": a["currency"], "weight": w}
        for a, w in zip(assets, weights)
    ]


def default_period() -> tuple[_dt.date, _dt.date]:
    end = _dt.date.today()
    start = end - _dt.timedelta(days=365 * ANALYSIS_PERIOD_YEARS)
    return start, end


def default_state() -> Dict[str, Any]:
    """Fresh session-state template (no Streamlit dependency)."""
    start, end = default_period()
    return {
        "language": "en",
        "market": MARKET_US,
        "data_mode": MODE_DEMO,
        "capital": 100000.0,
        "risk_free_rate": DEFAULT_RISK_FREE_RATE,
        "assets": default_assets(MARKET_US),
        "benchmark": default_benchmark(MARKET_US),
        "start": start,
        "end": end,
        "fx_usd_mad": None,             # explicit user-provided rate (or None)
        "imported_prices": None,        # wide DataFrame imported from CSV
        "import_revision": 0,           # cache-buster after a new CSV upload
        "asset_shocks": {},             # ticker -> decimal shock
        "ai_messages": [],
        "show_debug": False,
    }
