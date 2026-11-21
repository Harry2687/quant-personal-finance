import os

import altair as alt
import numpy as np
import polars as pl

from portfolio.analysis.fetch_data import DATA_FILE, fetch_market_data
from portfolio.assumptions import (
    ANNUAL_DCA,
    ASSET_ORDER,
    ASSETS,
    CGT_REAL_GAIN_RATE,
    CHARTS_DIR,
    FORTNIGHTLY_DCA,
    HISA_NET_REAL,
    INFLATION_RATE,
    NO_CRYPTO_OPTIMAL,
    OPTIMAL_ALLOCATION,
    ORIGINAL_ALLOCATION,
    REBALANCE_SELL_BAND,
    STARTING_CAPITAL,
)

CHART_FILE = CHARTS_DIR / "portfolio_projections.html"
TRADING_DAYS = 252
MEAN_BLOCK_LEN = (
    21  # Politis-Romano stationary bootstrap expected block length (~1 month)
)
N_SIMS = 5000
MAX_YEARS = 20


def load_drift_adjusted_daily_returns() -> np.ndarray:
    """Shift clean historical daily log returns to match forward after-tax CMAs while preserving tails."""
    if not os.path.exists(DATA_FILE):
        fetch_market_data(DATA_FILE)
    df = pl.read_parquet(DATA_FILE).sort("date")
    prices = df.select(ASSET_ORDER).to_numpy()
    raw_rets = prices[1:] / prices[:-1] - 1.0
    log_rets = np.log1p(raw_rets)

    shifted = np.zeros_like(raw_rets)
    for j, code in enumerate(ASSET_ORDER):
        target_ann_net = ASSETS[code].cma_net_nominal
        target_daily_arith = (1.0 + target_ann_net) ** (1.0 / TRADING_DAYS) - 1.0
        mean_exp = float(np.mean(np.exp(log_rets[:, j])))
        delta = np.log((1.0 + target_daily_arith) / mean_exp)
        shifted[:, j] = np.expm1(log_rets[:, j] + delta)
    return shifted


def generate_stationary_bootstrap_indices(
    n_obs: int, n_steps: int, n_sims: int, mean_block: float, seed: int = 42
) -> np.ndarray:
    """Politis & Romano (1994) stationary block bootstrap with geometric random block lengths."""
    rng = np.random.default_rng(seed)
    p_new = 1.0 / mean_block
    idx = np.empty((n_sims, n_steps), dtype=np.int32)
    idx[:, 0] = rng.integers(0, n_obs, size=n_sims)

    for t in range(1, n_steps):
        restart = rng.random(n_sims) < p_new
        nxt = (idx[:, t - 1] + 1) % n_obs
        n_restarts = int(np.sum(restart))
        if n_restarts > 0:
            nxt[restart] = rng.integers(0, n_obs, size=n_restarts)
        idx[:, t] = nxt
    return idx


def simulate_dca_paths(
    daily_rets: np.ndarray,
    idx: np.ndarray,
    weights: dict[str, float],
    rebalance_band: float = REBALANCE_SELL_BAND,
) -> dict[str, np.ndarray]:
    """Vectorized path-dependent DCA simulation with cashflow rebalancing, emergency sell band, CPI-indexed CGT, and fees."""
    n_sims, n_days = idx.shape
    n_assets = len(ASSET_ORDER)
    w_target = np.array([weights.get(c, 0.0) for c in ASSET_ORDER], dtype=float)
    trade_costs = np.array([ASSETS[c].trade_cost for c in ASSET_ORDER], dtype=float)
    daily_cpi_factor = (1.0 + INFLATION_RATE) ** (1.0 / TRADING_DAYS)

    entry_cost = float(np.sum(w_target * trade_costs))
    vals = np.broadcast_to(
        STARTING_CAPITAL * (1.0 - entry_cost) * w_target, (n_sims, n_assets)
    ).copy()
    unindexed_basis = vals.copy()
    indexed_basis = vals.copy()

    # Time-weighted return (TWR) index of the actual DCA-rebalanced portfolio (excludes cashflow sizing)
    twr_idx = np.full(n_sims, 1.0 - entry_cost)
    twr_peak = twr_idx.copy()
    max_dd = np.zeros(n_sims)

    # Track exact year-end nominal wealth, per-asset exit CGT, and DCA portfolio TWR (years 1..MAX_YEARS)
    yr_wealth_nom = np.zeros((n_sims, MAX_YEARS))
    yr_exit_cgt = np.zeros((n_sims, MAX_YEARS))
    yr_twr = np.zeros((n_sims, MAX_YEARS))

    # 26 fortnightly contributions per 252 trading days (~every 9.69 trading days)
    dca_days = set(np.round(np.linspace(10, TRADING_DAYS, 26)).astype(int).tolist())

    for d in range(n_days):
        tot_before = np.sum(vals, axis=1)
        r_d = daily_rets[idx[:, d]]  # (n_sims, n_assets)
        vals *= 1.0 + r_d
        indexed_basis *= daily_cpi_factor
        r_mkt = np.sum(vals, axis=1) / tot_before - 1.0

        day_in_year = (d % TRADING_DAYS) + 1

        # Cashflow-directed DCA: direct each fortnightly deposit toward underweight assets
        if day_in_year in dca_days:
            tot = np.sum(vals, axis=1, keepdims=True)
            target_post = (tot + FORTNIGHTLY_DCA) * w_target
            deficit = np.maximum(target_post - vals, 0.0)
            def_sum = np.sum(deficit, axis=1, keepdims=True)
            alloc_frac = np.where(
                def_sum > 0, deficit / np.where(def_sum > 0, def_sum, 1.0), w_target
            )
            contrib = FORTNIGHTLY_DCA * alloc_frac
            net_contrib = contrib * (1.0 - trade_costs)
            vals += net_contrib
            unindexed_basis += contrib
            indexed_basis += contrib

        # Monthly emergency sell-band check when DCA cashflows alone cannot prevent extreme drift
        rebal_drag = np.zeros(n_sims)
        if (d + 1) % 21 == 0:
            tot = np.sum(vals, axis=1)
            cur_w = vals / tot[:, None]
            breached = np.max(np.abs(cur_w - w_target), axis=1) >= rebalance_band
            if np.any(breached):
                b_vals = vals[breached]
                b_unidx = unindexed_basis[breached]
                b_basis = indexed_basis[breached]
                b_tot = tot[breached]
                b_target = b_tot[:, None] * w_target
                trades = b_target - b_vals
                sells = np.maximum(-trades, 0.0)
                buys = np.maximum(trades, 0.0)

                sell_frac = np.where(
                    b_vals > 0, sells / np.where(b_vals > 0, b_vals, 1.0), 0.0
                )
                unidx_sold = b_unidx * sell_frac
                basis_sold = b_basis * sell_frac
                real_gains = np.maximum(sells - basis_sold, 0.0)
                cap_losses = np.maximum(unidx_sold - sells, 0.0)
                net_gain = np.maximum(
                    np.sum(real_gains, axis=1) - np.sum(cap_losses, axis=1), 0.0
                )
                cgt = net_gain * CGT_REAL_GAIN_RATE
                fees = np.sum((sells + buys) * trade_costs, axis=1)

                post_tot = b_tot - cgt - fees
                rebal_drag[breached] = (cgt + fees) / b_tot
                new_vals = post_tot[:, None] * w_target
                buy_add = np.maximum(new_vals - (b_vals - sells), 0.0)
                vals[breached] = new_vals
                unindexed_basis[breached] = b_unidx - unidx_sold + buy_add
                indexed_basis[breached] = b_basis - basis_sold + buy_add

        twr_idx *= (1.0 + r_mkt) * (1.0 - rebal_drag)
        twr_peak = np.maximum(twr_peak, twr_idx)
        max_dd = np.minimum(max_dd, twr_idx / twr_peak - 1.0)

        if day_in_year == TRADING_DAYS:
            yr = d // TRADING_DAYS
            yr_wealth_nom[:, yr] = np.sum(vals, axis=1)
            # Per-asset liquidation CGT: indexation applies to gains only; losses use unindexed cost base
            exit_real_gains = np.maximum(vals - indexed_basis, 0.0)
            exit_cap_losses = np.maximum(unindexed_basis - vals, 0.0)
            net_exit_gain = np.maximum(
                np.sum(exit_real_gains, axis=1) - np.sum(exit_cap_losses, axis=1), 0.0
            )
            yr_exit_cgt[:, yr] = net_exit_gain * CGT_REAL_GAIN_RATE
            yr_twr[:, yr] = twr_idx

    return {
        "yr_wealth_nom": yr_wealth_nom,
        "yr_exit_cgt": yr_exit_cgt,
        "yr_twr": yr_twr,
        "max_dd": max_dd,
    }


def run_projections():
    daily_rets = load_drift_adjusted_daily_returns()
    n_obs = daily_rets.shape[0]
    n_days = MAX_YEARS * TRADING_DAYS

    idx = generate_stationary_bootstrap_indices(
        n_obs=n_obs, n_steps=n_days, n_sims=N_SIMS, mean_block=MEAN_BLOCK_LEN, seed=42
    )

    strategies = {
        "New Optimal (70/15/12/3)": OPTIMAL_ALLOCATION,
        "New Optimal Ex-BTC (75/15/10)": NO_CRYPTO_OPTIMAL,
        "Original Target (85/10/5)": ORIGINAL_ALLOCATION,
        "100% Global Developed (BGBL)": {"BGBL": 1.0},
    }

    sim_results = {
        name: simulate_dca_paths(daily_rets, idx, w) for name, w in strategies.items()
    }
    primary = sim_results["New Optimal (70/15/12/3)"]

    pl.Config.set_tbl_cols(10)
    pl.Config.set_tbl_width_chars(150)

    horizons = [1, 3, 5, 10, 20]
    cagr_rows = []
    wealth_real_rows = []
    wealth_nom_rows = []

    for y in horizons:
        infl = (1.0 + INFLATION_RATE) ** y
        twr_y = primary["yr_twr"][:, y - 1]
        cagr_nom = twr_y ** (1.0 / y) - 1.0
        cagr_real = (1.0 + cagr_nom) / (1.0 + INFLATION_RATE) - 1.0

        p05, p25, p50, p75, p95 = np.percentile(cagr_real, [5, 25, 50, 75, 95])
        p_real_loss = float(np.mean(cagr_real < 0.0) * 100.0)
        p_beat_hisa = float(np.mean(cagr_real > HISA_NET_REAL) * 100.0)

        cagr_rows.append(
            {
                "Horizon": f"{y}-Year",
                "5th (Bear)": f"{p05 * 100:.1f}%",
                "25th": f"{p25 * 100:.1f}%",
                "Median": f"{p50 * 100:.1f}%",
                "75th": f"{p75 * 100:.1f}%",
                "95th (Bull)": f"{p95 * 100:.1f}%",
                "P(Real Loss)": f"{p_real_loss:.1f}%",
                "P(Beat HISA)": f"{p_beat_hisa:.1f}%",
            }
        )

        # Path-dependent DCA wealth (pre-exit-CGT and net after per-asset CPI-indexed exit CGT)
        w_nom = primary["yr_wealth_nom"][:, y - 1]
        exit_cgt = primary["yr_exit_cgt"][:, y - 1]
        w_net_liq_nom = w_nom - exit_cgt
        w_real = w_nom / infl
        w_net_liq_real = w_net_liq_nom / infl
        total_in = STARTING_CAPITAL + ANNUAL_DCA * y

        rp = np.percentile(w_real, [5, 25, 50, 75, 95])
        rp_liq = float(np.median(w_net_liq_real))
        wealth_real_rows.append(
            {
                "Horizon": f"{y}-Year",
                "Nom Invested": f"${total_in:,.0f}",
                "5th (Real)": f"${rp[0]:,.0f}",
                "25th (Real)": f"${rp[1]:,.0f}",
                "Median (Real)": f"${rp[2]:,.0f}",
                "Median Net-Liq": f"${rp_liq:,.0f}",
                "95th (Real)": f"${rp[4]:,.0f}",
            }
        )

        np_p = np.percentile(w_nom, [5, 25, 50, 75, 95])
        wealth_nom_rows.append(
            {
                "Horizon": f"{y}-Year",
                "Total Invested": f"${total_in:,.0f}",
                "5th (Nom)": f"${np_p[0]:,.0f}",
                "25th (Nom)": f"${np_p[1]:,.0f}",
                "Median (Nom)": f"${np_p[2]:,.0f}",
                "75th (Nom)": f"${np_p[3]:,.0f}",
                "95th (Nom)": f"${np_p[4]:,.0f}",
            }
        )

    print(
        f"\n=== 1. New Optimal (70 BGBL / 15 BEMG / 12 Gold / 3 BTC): Real After-Tax TWR Distribution ({N_SIMS:,} Stationary Bootstrap Paths, DCA + {int(REBALANCE_SELL_BAND * 100)}% Sell Band) ==="
    )
    print(pl.DataFrame(cagr_rows))

    print(
        f"\n=== 2. Terminal DCA Wealth in REAL 2026 AUD (${STARTING_CAPITAL:,.0f} + ${FORTNIGHTLY_DCA:,.0f}/fn, Cashflow Rebalancing + {int(REBALANCE_SELL_BAND * 100)}% Emergency Sell Band) ==="
    )
    print(pl.DataFrame(wealth_real_rows))

    print("\n=== 3. Terminal DCA Wealth in NOMINAL AUD (2.5% CPI Assumed) ===")
    print(pl.DataFrame(wealth_nom_rows))

    # Head-to-head 20-year comparison across strategies
    comp_rows = []
    infl_20 = (1.0 + INFLATION_RATE) ** MAX_YEARS
    for name, res in sim_results.items():
        w20_real = res["yr_wealth_nom"][:, -1] / infl_20
        w20_liq_real = (
            res["yr_wealth_nom"][:, -1] - res["yr_exit_cgt"][:, -1]
        ) / infl_20
        twr_nom_cagr = res["yr_twr"][:, -1] ** (1.0 / MAX_YEARS) - 1.0
        twr_real_cagr = (1.0 + twr_nom_cagr) / (1.0 + INFLATION_RATE) - 1.0
        comp_rows.append(
            {
                "Strategy (20Y Horizon)": name,
                "Med Real TWR": f"{np.median(twr_real_cagr) * 100:.2f}%",
                "Med Nom TWR": f"{np.median(twr_nom_cagr) * 100:.2f}%",
                "5th %ile Real DCA": f"${np.percentile(w20_real, 5):,.0f}",
                "Median Real DCA": f"${np.median(w20_real):,.0f}",
                "Med Net-Liq Real": f"${np.median(w20_liq_real):,.0f}",
                "Med 20Y MaxDD": f"{np.median(res['max_dd']) * 100:.1f}%",
                "95th Worst MaxDD": f"{np.percentile(res['max_dd'], 5) * 100:.1f}%",
            }
        )
    print(
        "\n=== 4. 20-Year Forward Strategy Comparison (Real 2026 AUD, Forward CMAs + Empirical Tails) ==="
    )
    print(pl.DataFrame(comp_rows))

    # Build true data-driven fan chart directly from simulated year-by-year Real 2026 AUD wealth
    chart_records = []
    for y in range(1, MAX_YEARS + 1):
        infl = (1.0 + INFLATION_RATE) ** y
        w_real_y = primary["yr_wealth_nom"][:, y - 1] / infl
        p05, p25, p50, p75, p95 = np.percentile(w_real_y, [5, 25, 50, 75, 95])
        # Real purchasing power of cumulative dollars contributed
        contrib_real = STARTING_CAPITAL / infl + sum(
            ANNUAL_DCA / ((1.0 + INFLATION_RATE) ** (y - k + 0.5))
            for k in range(1, y + 1)
        )
        chart_records.append(
            {
                "Year": y,
                "Real_Contributions": round(float(contrib_real), 2),
                "P05_Bear": round(float(p05), 2),
                "P25_Lower": round(float(p25), 2),
                "P50_Median": round(float(p50), 2),
                "P75_Upper": round(float(p75), 2),
                "P95_Bull": round(float(p95), 2),
            }
        )

    chart_df = pl.DataFrame(chart_records)
    base = alt.Chart(chart_df).encode(x=alt.X("Year:O", title="Years Elapsed"))

    band_90 = base.mark_area(opacity=0.15, color="#1f77b4").encode(
        y=alt.Y("P05_Bear:Q", title="Portfolio Value (Real 2026 AUD)"), y2="P95_Bull:Q"
    )
    band_50 = base.mark_area(opacity=0.30, color="#1f77b4").encode(
        y="P25_Lower:Q", y2="P75_Upper:Q"
    )
    median_line = base.mark_line(color="#1f77b4", strokeWidth=3).encode(
        y="P50_Median:Q",
        tooltip=[
            "Year:O",
            "P05_Bear:Q",
            "P25_Lower:Q",
            "P50_Median:Q",
            "P75_Upper:Q",
            "P95_Bull:Q",
        ],
    )
    cost_line = base.mark_line(
        color="#7f7f7f", strokeDash=[4, 4], strokeWidth=2
    ).encode(y="Real_Contributions:Q")

    chart = (band_90 + band_50 + median_line + cost_line).properties(
        title=f"20-Year Real Wealth Fan Chart: 70/15/12/3 Optimal Portfolio (${STARTING_CAPITAL:,.0f} + ${FORTNIGHTLY_DCA:,.0f}/fn DCA, Forward CMAs)",
        width=860,
        height=460,
    )
    CHART_FILE.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(CHART_FILE))
    print(f"\nSaved data-driven projection fan chart to {CHART_FILE}")


if __name__ == "__main__":
    run_projections()
