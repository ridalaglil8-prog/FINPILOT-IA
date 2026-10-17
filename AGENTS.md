# AGENTS.md — FinPilot AI

Guidance for AI coding agents (and new contributors) working in this repo.

## What this project is

FinPilot AI is an **educational** Streamlit portfolio-analytics app. It
computes historical metrics, optimises allocations and runs hypothetical
stress scenarios on US and Moroccan instruments. **It never executes trades
and never promises returns.**

## Non-negotiable rules

1. **Streamlit is the web framework.** Do not introduce React/Next.js/Flask/
   FastAPI, or a separate frontend build step.
2. **Never invent financial data.**
   - Demo data is synthetic, deterministic, and must stay labelled *SIMULATED*
     everywhere it reaches the UI.
   - A failed **real** data request must surface an error. **Never** fall back
     silently to synthetic prices.
   - No Moroccan ticker, price, index or observation may be hard-coded. The
     Casablanca universe is user-supplied (CSV) or demo-only.
3. **Never mix USD and MAD** into a single total without an explicit,
   user-provided FX rate. Report per-currency totals instead.
4. **Python computes, the LLM explains.** The assistant only ever receives a
   structured JSON payload built by `ai/prompts.py` from real computed
   results. Do not ask the model to produce authoritative numbers.
5. **Secrets live only in the environment** (`.env`, git-ignored). Never
   hard-code, log or render an API key.
6. **No fabricated metrics.** Do not hard-code expected numbers into the
   real-data path to make a page look complete.
7. **Educational disclaimer stays visible.** Never phrase output as advice,
   a prediction, or a guarantee.

## Architecture

```
pages/          Streamlit UI (one file per nav item)  — thin, no math
components/     Reusable presentation (sidebar, KPI cards, charts)
services/       Orchestration: portfolio_service, market_data, currency, ai_service
analytics/      Pure finance engine — no Streamlit import, fully unit-tested
optimization/   PyPortfolioOpt + documented SciPy fallback
data/           demo_data, csv_loader, providers/{us_market,morocco_market}
ai/             prompts (system prompt + JSON payload)
locales/        en.json / fr.json — identical key sets
```

**The key contract:** `services/portfolio_service.build_context(state)` turns
session state into one fully-computed context dict. Every page, the dashboard
and the AI payload consume **that same dict**, so the UI and the model can
never disagree.

### Session state contract (`config.default_state()`)

| Key | Meaning |
|-----|---------|
| `language` | `"en"` \| `"fr"` |
| `market` | `MARKET_US` \| `MARKET_MA` |
| `data_mode` | `MODE_DEMO` \| `MODE_HISTORICAL` |
| `capital` | initial investment (float) |
| `risk_free_rate` | annual effective rate for Sharpe |
| `assets` | list of `{ticker, name, asset_class, currency, weight}` |
| `benchmark` | ticker or `None` |
| `start`, `end` | `datetime.date` analysis window |
| `fx_usd_mad` | explicit rate or `None` |
| `imported_prices` | wide `DataFrame` from CSV or `None` |
| `optimization` | last optimizer result (set by the page) |
| `stress_test` | last scenario result (set by the page) |

## Conventions

- **Annualisation:** 252 trading days (`config.TRADING_DAYS`).
- **Weights:** long-only, fractions summing to 1.0; the builder validates
  100 % in percent points.
- **Missing data:** dropped listwise and reported as a warning. Never filled.
- **Errors:** business modules raise typed exceptions
  (`MarketDataError`, `OptimizationError`, `ShockError`, `CSVValidationError`,
  `InsufficientDataError`, `AIServiceError`) with user-safe messages. Pages
  catch them and render a message — never a stack trace.

## Localization

- Add every user-facing string to **both** `locales/en.json` and
  `locales/fr.json`. Key parity is enforced by `tests/test_i18n.py`.
- Use `i18n.t("key")` in UI code; `i18n.tw("code")` for backend status codes
  (supports `"code: detail"`).
- Never duplicate a page per language.

## Charts

`components/charts.py` builders all share `base_layout()`.

> ⚠️ `base_layout()` intentionally omits `title`. Pass your own
> `title=with_title("...")`. Passing `title=` while `base_layout` also
> returned one caused `update_layout() got multiple values for keyword
> argument 'title'` — a crash that only appeared at render time. The same
> applies to `xaxis`/`yaxis`: use `fig.update_xaxes(...)` instead of passing
> them alongside `base_layout`.

## Testing

```powershell
pytest tests -q          # 147+ tests, offline and deterministic
```

- Tests use **deterministic synthetic data** with hand-computed expectations.
  No network, no API keys (`conftest.py` sets `FINPILOT_TESTING=1`).
- New analytics code needs tests. Run the suite and **fix failures before
  claiming completion**.
- `tests/test_charts.py` smoke-tests every chart builder — it catches the
  duplicate-kwarg class of bug cheaply.

## Dev loop

```powershell
.venv\Scripts\python.exe -m pytest tests -q -p no:warnings
.venv\Scripts\streamlit run app.py        # http://localhost:8501
```

Verify in a real browser, not just by importing modules: a page can import
cleanly and still fail while rendering.

## Where to change things

| Task | Files |
|------|-------|
| Add an asset to the US universe | `config.US_ASSETS` (metadata only, no prices) |
| Add a metric | `analytics/` → expose in `portfolio_service._metric_block` → render in `components/kpi_cards.py` |
| Add a page | `pages/N_Name.py` + `components/sidebar.py` `PAGES` + locale keys |
| Add a translation | `locales/en.json` **and** `locales/fr.json` |
| Change a chart | `components/charts.py` (+ `tests/test_charts.py`) |
| Add an LLM knob | `.env.example`, `config.LLM_DEFAULTS`, `services/ai_service.py` |
| Add a stress preset | `pages/5_Stress_Testing.py` `PRESETS` + locale keys |
