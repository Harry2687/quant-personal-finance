# Quant Personal Finance

A quantitative, after-tax asset allocation, simulation, and live portfolio tracking package tailored for an Australian tax-resident investor (39% marginal tax rate example: 37% bracket + 2% Medicare levy) seeking Ex-Australia liquid market diversification (given existing AUD exposure via salary, emergency cash, and domestic property).

## Quantitatively Derived Portfolio Strategy

Starting from a clean 6-asset Ex-Australia candidate universe (`BGBL`, `HGBL`, `BEMG`, `VAF`, `PMGOLD`, `BTC`) using forward-looking after-tax Capital Market Assumptions (CMAs), Ledoit-Wolf shrunk weekly covariance (removing ASX-vs-UTC timezone lag), and tail-aware CRRA utility optimization, the recommended liquid growth allocation is **85% complete Ex-Australia global equities + 15% non-equity diversifiers**, with cash kept strictly outside the portfolio:

| Asset | Ticker | Target Weight | Ex-BTC Variant | Role & Quantitative Rationale | Platform | MER |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Global Developed Equities (Unhedged, ex-AU)** | `BGBL.AX` | **70%** | **75%** | Core global growth engine; low distribution tax drag (~0.66%); natural AUD crash-cushioning effect. | Betashares Direct | 0.08% |
| **Emerging Markets Equities (MSCI EM)** | `BEMG.AX` | **15%** | **15%** | Completes global equity coverage (including South Korea, excluded by `BGBL`); low weekly correlation to `BGBL` (0.60); non-distributing accumulating UCITS swap structure avoids ongoing 39% AU income tax drag. | Betashares Direct | 0.35% |
| **Physical Gold** | `PMGOLD.AX` | **12%** | **10%** | Zero ongoing income tax drag; CPI cost-base indexation shelters its inflation return; genuine crisis convexity (+1.43% mean return and 0.0% joint crash rate during `BGBL`'s worst 5% weeks). | Betashares Direct | 0.15% |
| **Bitcoin** | `BTC-AUD` | **3%** | **0%** | Optional high-volatility asymmetric sleeve; sized modestly (0-5%) due to positive post-2020 equity correlation (0.25-0.28 weekly) and downside crash contagion (18.8%). | Exchange / Spot ETF | 0.60% (trade) / ~0.5% ETF |

* **Australian CGT System Modeled (2026-27 Tax Reform):**
  * Models the *Treasury Laws Amendment (Tax Reform No. 1) Act 2026* (effective **1 July 2027**), which replaces the 50% CGT discount for individuals/trusts with **CPI cost-base indexation** (assets held >= 12 months, gains only; capital losses use unindexed reduced cost base) and a **30% minimum tax floor** (`max(30%, MTR)` = 39% on real gains at 39% MTR).
  * Gains accrued prior to 1 July 2027 remain grandfathered under the legacy 50% discount rules via a 1 July 2027 market-value reset; cashflow-directed rebalancing avoids realizing 39% tax on real gains.

* **Why 0% Australian Equities (`A200`), 0% `HGBL` (AUD-Hedged), and 0% `VAF` (Australian Bonds)?**
  * **Ex-Australia constraint:** Salary, AUD emergency cash, and domestic property already create substantial Australian/AUD exposure on the total household balance sheet.
  * **`HGBL`:** Higher volatility (15.0%-16.9%), worse crash weeks (-4.32% vs `BGBL`'s -3.86%), and higher tax drag (~1.06% from TOFA FX distributions); receives 0% in 2020+ covariance and all tail-aware CRRA models.
  * **`VAF`:** Yields ~3.24% net nominal (~0.63% net geometric real) after 39% tax—barely above a risk-free external high-interest savings account (3.05% net at 5.0% gross)—with duration risk and positive post-2020 equity correlation (+0.26).
* **Illustrative Simulation Defaults** *(override locally in gitignored `data/profile.json`)*:
  * **Emergency Cash**: $30,000 kept external to the portfolio in a HISA (5.0% gross / 3.05% net after 39% tax; or mortgage offset if applicable).
  * **Starting Investable Capital**: $100,000 deployed lump sum.
  * **Ongoing Contribution**: $1,000 per fortnight ($26,000/year), directed dynamically toward underweight assets (cashflow-directed rebalancing to minimize CGT).
* **Note on Superannuation**: For an Australian investor in higher tax brackets, concessional superannuation contributions (15% tax vs marginal rate) or FHSSS should be evaluated alongside this outside-super portfolio.

---

## Repository Structure

```text
├── src/portfolio/
│   ├── assumptions.py                # Centralized AU tax rules, asset specs, forward CMAs, and data paths
│   ├── analysis/
│   │   ├── fetch_data.py             # Return-space splicing pipeline fetching clean 6-asset daily series
│   │   ├── optimize.py               # From-scratch MVO, CRRA utility, and CMA sensitivity search
│   │   ├── distributions.py          # Clean moments, Student-t/NIG fits, Gaussian KDE, and weekly tail dependence
│   │   ├── validate.py               # Historical backtest with band rebalancing, CGT, fees, and regime stress tests
│   │   └── project.py                # 5,000-path Politis-Romano stationary bootstrap with forward CMAs & path-dependent DCA
│   └── live/
│       ├── tracker.py                # Live portfolio engine (Betashares CSV/Gmail via gws + Swyftx read-only API)
│       ├── cli.py                    # Standalone terminal CLI for live portfolio view & next deposit split
│       └── mcp_server.py             # Local Stdio MCP server exposing live portfolio & DCA tools to Antigravity CLI
├── data/                             # Local gitignored data, personal profile & generated charts
│   ├── profile.json                  # Optional personal overrides (capital, DCA, tax rate, ledger start date)
│   ├── market_data.parquet           # Clean historical price dataset
│   ├── portfolio_ledger.db           # Local SQLite trade ledger
│   └── charts/                       # Generated interactive Altair HTML charts
│       ├── optimization_frontier.html
│       ├── return_distributions.html
│       ├── portfolio_validation.html
│       └── portfolio_projections.html
├── pyproject.toml / uv.lock          # Package metadata, entrypoints, and dependencies managed via uv
└── README.md                         # System documentation
```

---

## Live Portfolio CLI & MCP Server (Betashares Direct + Swyftx)

* **Swyftx (`BTC`):** Pulls live balance and buy history directly via read-only API key stored in macOS Keychain (`security add-generic-password -U -a "$USER" -s "swyftx-api-key" -w`).
* **Betashares Direct (`BGBL`, `BEMG`, `PMGOLD`):** Backfilled via CSV export and auto-synced from `orders@notification.betashares.com.au` order confirmation emails via `gws gmail` into `data/portfolio_ledger.db` (deduplicated by date, ticker, side, and amount).
* **MCP Server:** Registered in `~/.gemini/config/mcp_config.json` as `portfolio` (`uv run --directory <repo> portfolio-mcp`), exposing `get_live_portfolio`, `get_next_dca_plan`, `sync_betashares_emails`, `import_betashares_csv`, and `get_trade_history`.
* **Standalone CLI (No AI Required):**
```bash
uv run portfolio                          # Live portfolio table + suggested fortnightly deposit split
uv run portfolio --amount 2000            # Custom deposit amount
uv run portfolio --no-sync                # Skip Gmail check for faster refresh
uv run portfolio trades                   # View recorded Betashares trades
uv run portfolio import-csv <path>        # Import a Betashares CSV export
```
*(Optional: install globally with `uv tool install -e .` to run `portfolio` from any directory.)*

---

## Quantitative Analysis & Simulation Suite

### 1. Refresh Clean Market Data
Fetches daily adjusted closes from November 2014 to present and splices newer ETFs (`BGBL`, `HGBL`, `BEMG`, `PMGOLD`) back to established proxies (`VGS`, `VGAD`, `IEM`, `GOLD`) in return space without forward-filling sparse quotes:
```bash
uv run portfolio-fetch
```

### 2. Search for Optimal Allocation From Scratch
Runs Ledoit-Wolf shrunk weekly covariance estimation, after-tax Mean-Variance optimization, empirical tail-aware CRRA utility optimization, and a BTC/Gold CMA sensitivity sweep:
```bash
uv run portfolio-optimize
```
* Generates `data/charts/optimization_frontier.html`.

### 3. Model Return Distributions & Weekly Tail Dependence
Fits Normal, Student-t, and NIG distributions to clean daily log returns, computes true Gaussian KDE curves, and measures timezone-aligned weekly crash co-movement:
```bash
uv run portfolio-distributions
```
* Generates `data/charts/return_distributions.html`.

### 4. Validate Against Historical Regimes & Frictions
Backtests the recommended allocation (`70/15/12/3`) against the Ex-BTC variant (`75/15/10`), the original `85/10/5` portfolio, 100% `BGBL`, `85/15 BGBL/BEMG`, and `60/40 BGBL/VAF`, incorporating ongoing dividend/FITO tax drag, 5% tolerance-band rebalancing CGT, trading costs, and regime stress tests:
```bash
uv run portfolio-validate
```
* Generates `data/charts/portfolio_validation.html`.

### 5. Project Forward Real & Nominal Wealth (20-Year Stationary Bootstrap)
Runs a 5,000-path Politis-Romano stationary block bootstrap shifted to forward after-tax CMAs, simulating path-dependent fortnightly cashflow-directed DCA rebalancing, a 10% emergency sell band with CPI-indexed CGT, trading costs, inflation (2.5% CPI), and terminal exit CGT:
```bash
uv run portfolio-project
```
* Generates `data/charts/portfolio_projections.html`.

---

## Development Checks (Formatting, Linting & Type Checking)

```bash
uv run ruff format --check
uv run ruff check
uv run pyrefly check
```
