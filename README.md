# FinPilot AI 🧭

**AI-powered investment portfolio management & analytics platform** — an
educational FinTech project built with **Streamlit**.

FinPilot AI lets you build portfolios on the **US** and **Moroccan** markets,
analyse their historical performance and risk, optimise allocations, run
hypothetical stress scenarios, and ask an AI assistant to explain every metric
— in **French or English**.

> ⚠️ **Educational project.** FinPilot AI never executes trades, never stores
> money and never promises returns. Every figure is either **historical** or
> **hypothetical**. This is not investment advice.

---

## ✨ Features

| Page | What it does |
|------|--------------|
| 🏠 **Dashboard** | KPI cards (capital, value, return, volatility, Sharpe, drawdown, VaR), allocation charts, wealth path, benchmark comparison, data provenance |
| 🧱 **Portfolio Builder** | Add/remove assets, set weights with live validation (weights must total 100 %), capital, benchmark and analysis period |
| 📈 **Performance Analytics** | Wealth path, cumulative return, monthly-returns heatmap, drawdown, benchmark relative performance |
| 🛡️ **Risk & Diversification** | Correlation heatmap, risk contributions, concentration (Herfindahl, effective *n*), return distribution, rolling volatility |
| ⚖️ **Portfolio Optimization** | Long-only **min-volatility** and **max-Sharpe** (PyPortfolioOpt, SciPy fallback) with a current-vs-optimized comparison |
| 🧪 **Stress Testing** | Hypothetical shocks per asset class or per asset, presets, estimated monetary and percentage impact |
| 🤖 **AI Financial Assistant** | Explains *your calculated* results via an OpenAI-compatible LLM — never invents numbers |
| ⚙️ **Settings** | Moroccan CSV import, risk-free rate, explicit FX rate, AI status, session reset |

### Design principles

- **Never invent data.** Demo mode is clearly labelled *SIMULATED*; a failed
  real-data request surfaces an error and is **never** silently replaced by
  synthetic prices.
- **Never mix currencies silently.** USD and MAD totals are reported
  separately unless *you* provide an explicit FX rate.
- **Python computes, the LLM explains.** The assistant only ever receives a
  structured JSON payload produced by the analytics engine.
- **Secrets stay in the environment.** No API key is ever hard-coded, logged
  or rendered.

---

## 🚀 Quick start

### 1. Requirements

- Python **3.10 or 3.11** (verified on CPython 3.10.11, Windows)
- pip

### 2. Install

```powershell
# from the project root
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

Or on Windows, use the bundled bootstrap script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

### 3. Run

```powershell
streamlit run app.py
```

The app opens on **http://localhost:8501**. The **Dashboard works immediately
in Demo mode with no API key and no Internet connection.**

---

## 🗂️ Project structure

```
finpilot-ai/
├── app.py                     # Dashboard entry point (kept concise)
├── config.py                  # Static configuration, asset universes, defaults
├── i18n.py                    # Shared translation helper (t / tw)
├── conftest.py                # Pytest bootstrap (offline, deterministic)
├── requirements.txt
├── .env.example               # LLM + FX configuration template
├── .streamlit/config.toml     # Dark navy/cyan theme, server settings
├── pages/                     # Streamlit multipage navigation
│   ├── 1_Portfolio_Builder.py
│   ├── 2_Performance_Analytics.py
│   ├── 3_Risk_Diversification.py
│   ├── 4_Portfolio_Optimization.py
│   ├── 5_Stress_Testing.py
│   ├── 6_AI_Assistant.py
│   └── 7_Settings.py
├── components/                # Reusable presentation components
│   ├── sidebar.py             # Navigation + language/mode/market selectors
│   ├── kpi_cards.py           # KPI cards, provenance strip, formatters
│   └── charts.py              # Plotly builders (consistent dark styling)
├── services/                  # Orchestration layer
│   ├── portfolio_service.py   # session state -> fully computed context
│   ├── market_data.py         # Provider facade + caching + provenance
│   ├── currency.py            # FX rules (never mix without an explicit rate)
│   └── ai_service.py          # OpenAI-compatible LLM client
├── data/
│   ├── demo_data.py           # Deterministic synthetic prices (offline)
│   ├── csv_loader.py          # Validated CSV import (wide & long layouts)
│   └── providers/
│       ├── us_market.py       # yfinance (strict, no silent fallback)
│       └── morocco_market.py  # CSV-based, no invented tickers
├── analytics/                 # Pure, independently testable finance engine
│   ├── returns.py             # daily/cumulative/annualised/portfolio returns
│   ├── risk.py                # vol, Sharpe, drawdown, VaR/CVaR, concentration
│   ├── correlation.py         # correlation, covariance, risk contributions
│   └── stress.py              # hypothetical shock engine
├── optimization/
│   └── optimizer.py           # PyPortfolioOpt + documented SciPy fallback
├── ai/
│   └── prompts.py             # System prompt + structured JSON payload
├── locales/
│   ├── en.json                # Complete English dictionary
│   └── fr.json                # Complete French dictionary (same keys)
└── tests/                     # 191 deterministic pytest tests
```

**Separation of concerns:** `analytics/`, `optimization/`, `data/` and
`services/currency.py` are Streamlit-free and unit-testable. Only
`components/` and `pages/` touch Streamlit.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and fill in what you need. **The app runs fully
in Demo mode without any of these values.**

```dotenv
# --- LLM assistant (OpenAI-compatible chat-completions endpoint) ---
LLM_API_KEY=
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT_SECONDS=30

# --- Optional, explicit FX conversion ---
FX_USD_MAD=

# --- Development ---
FINPILOT_DEBUG=0
```

`.env` is listed in `.gitignore` and must never be committed. Keys are read
with `python-dotenv` at startup and are never rendered in the UI (the Settings
page only reports whether a key is configured, never any part of it).

Any OpenAI-compatible endpoint works (OpenAI, Azure-compatible proxies,
Ollama, vLLM, Mistral…) — just set `LLM_BASE_URL` and `LLM_MODEL`.

---

## 🌐 Markets & data

### US market
Real data via **yfinance** (`Historical data` mode). The configurable universe
lives in `config.US_ASSETS` (ETFs and large caps) with the **S&P 500**
(`^GSPC`) as the default benchmark. If a download fails, you get a clear error
— never silently substituted synthetic prices.

### Moroccan market
FinPilot AI **ships with no Casablanca securities** on purpose: no verified
data feed is bundled, and inventing tickers or prices would be misleading.
You can:

1. **Import a CSV** of historical prices on the *Settings* page, or
2. Use **Demo mode**, which clearly labelled *SIMULATED* baskets.

Accepted CSV layouts (case- and accent-tolerant column names):

```csv
# Wide: one column per asset
date,ATW,LDH
2023-01-02,412.50,285.10
2023-01-03,415.00,286.40
```

```csv
# Long: date, ticker, price
date,ticker,price
2023-01-02,ATW,412.50
2023-01-03,ATW,415.00
```

Validation guarantees: parseable unique dates, finite strictly-positive
prices, sorted index, ≥ 2 observations. **Gaps stay gaps** — nothing is
forward- or backward-filled, and every ignored row is reported as a warning.

### Multi-currency
Assets are grouped by currency and each group is analysed as an independent
sub-portfolio. Totals are merged **only** when you provide an explicit
USD → MAD rate (Settings or `FX_USD_MAD`). Without it, totals stay separate —
the app will never guess a conversion.

---

## 📐 Financial conventions

| Metric | Formula / convention |
|--------|----------------------|
| Daily return | `r_t = P_t / P_{t-1} - 1` |
| Cumulative return | `R_T = Π(1 + r_t) - 1` |
| Wealth index | `W_t = W_0 · Π(1 + r_s)` |
| Annualised return | `(1 + R_T)^(252/n) - 1` (geometric / CAGR) |
| Annualised volatility | `std(r, ddof=1) · √252` |
| Sharpe ratio | `(r_ann − r_f) / σ_ann` (both annualised) |
| Maximum drawdown | `min_t (W_t / running_max(W)_t − 1)` |
| Portfolio variance | `wᵀ Σ w` (`Σ` = sample covariance of daily returns) |
| Risk contribution | `RC_i = w_i · (Σw)_i`, percentages sum to 100 % |
| Herfindahl index | `Σ w_i²`, effective *n* = `1 / HHI` |
| Historical VaR / CVaR | empirical quantile / tail mean (losses positive) |
| Stress impact | `impact = Σ w_i · shock_i`, `value_after = current_value · (1 + impact)` (per currency sub-portfolio) |

**Assumptions & edge cases** (all covered by tests):

- **252 trading days** is the default annualisation convention.
- The **risk-free rate** is an annual effective rate, configurable in
  Settings (default 2 %).
- **Zero volatility**: a volatility ≤ 1e-12 is treated as exactly 0 so the
  Sharpe ratio reports `0.0` instead of exploding.
- **Total loss**: the geometric return is bounded at −100 %; a drawdown of a
  wiped-out portfolio is −100 %.
- **Missing prices are never filled.** Rows with any non-finite value are
  dropped listwise and the count is surfaced as a warning.
- Insufficient history, empty portfolios, invalid weights, inverted date
  ranges and non-finite inputs produce **clear user-facing messages**, not
  stack traces.
- **Optimisation is historical**, based on estimated means and a sample
  covariance matrix under strong assumptions. It does **not** guarantee better
  future performance.
- **Stored results expire.** Optimisation and stress results carry a
  fingerprint of the portfolio, period, dataset and constraints they were
  computed from. Change any of them and the result is dropped — the page asks
  you to run it again instead of showing (or letting the assistant quote)
  numbers that no longer describe your portfolio.

---

## 🧪 Testing

```powershell
pytest tests -q
```

**191 tests** run deterministically and **fully offline** (no network, no API
keys). They cover daily/cumulative/annualised returns, volatility, Sharpe,
drawdown, VaR/CVaR, weight validation, portfolio values, correlation, risk
contributions, optimisation constraints, missing data, invalid inputs, CSV
validation (wide/long/rejection paths), currency consistency, stress shocks,
locale key parity, and the end-to-end `build_context` pipeline.

`conftest.py` injects the project root into `sys.path` and sets
`FINPILOT_TESTING=1` so no test ever touches the network.

---

## 🌍 Bilingual interface

Every string comes from `locales/en.json` / `locales/fr.json` through the
single `i18n.t()` helper. The two files are enforced to have **identical key
sets** by `tests/test_i18n.py`. Switch language with the sidebar selector —
no page is duplicated per language.

Backend codes (e.g. `invalid_weights`) are translated by `i18n.tw()`, which
falls back to the raw text for unknown codes so a new message can never crash
a page.

---

## 🔒 Security & reliability

- API credentials live **only** in environment variables / `.env`
  (git-ignored). Never in code, logs or the UI.
- Uploaded CSVs are size-limited and strictly validated; failures return
  user-safe messages.
- External API errors, timeouts, DNS failures and rate limits are converted
  into short actionable messages.
- Invalid numeric values are rejected before reaching any financial
  calculation.
- No database, no brokerage integration, no trading — by design for this MVP.

---

## 🗺️ Roadmap ideas

- Verified Casablanca Stock Exchange data adapter (provider interface already
  in place in `data/providers/morocco_market.py`)
- Live FX feed with caching and staleness warnings
- Efficient-frontier plot and allocation drift over time
- Exportable reports (PDF/CSV)
- Additional risk models (CVaR optimisation, Black–Litterman)

---

## 📄 License & disclaimer

For educational use. The authors accept no liability for decisions made using
this software. Past performance — real or simulated — is not indicative of
future results.
