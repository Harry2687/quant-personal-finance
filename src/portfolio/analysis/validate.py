import os
from datetime import date

import altair as alt
import numpy as np
import polars as pl

from portfolio.analysis.fetch_data import DATA_FILE, fetch_market_data
from portfolio.assumptions import (
    ASSET_ORDER,
    ASSETS,
    CHARTS_DIR,
    HISA_NET_NOMINAL,
    LONG_TERM_CGT_RATE,
    NO_CRYPTO_OPTIMAL,
    OPTIMAL_ALLOCATION,
    ORIGINAL_ALLOCATION,
    STARTING_CAPITAL,
)

CHART_FILE = CHARTS_DIR / "portfolio_validation.html"
TRADING_DAYS = 252


def load_net_returns() -> tuple[pl.DataFrame, np.ndarray]:
    if not os.path.exists(DATA_FILE):
        fetch_market_data(DATA_FILE)
    df = pl.read_parquet(DATA_FILE).sort("date")
    prices = df.select(ASSET_ORDER).to_numpy()
    raw_rets = prices[1:] / prices[:-1] - 1.0

    # Apply per-asset ongoing after-tax distribution drag (incl. A200 franking & FITO)
    drags_daily = np.array(
        [ASSETS[c].annual_tax_drag / TRADING_DAYS for c in ASSET_ORDER]
    )
    net_rets = raw_rets - drags_daily
    return df.slice(1), net_rets


def calculate_metrics(daily_rets: np.ndarray, rf_annual: float) -> dict:
    n_days = len(daily_rets)
    cum = np.cumprod(1.0 + daily_rets)
    cagr = float(cum[-1] ** (TRADING_DAYS / n_days) - 1.0)
    vol = float(np.std(daily_rets, ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (cagr - rf_annual) / vol if vol > 0 else 0.0

    # Standard Sortino: downside semi-deviation below daily risk-free rate across all days
    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0
    shortfall = np.minimum(daily_rets - rf_daily, 0.0)
    downside_dev = float(np.sqrt(np.mean(shortfall**2)) * np.sqrt(TRADING_DAYS))
    sortino = (cagr - rf_annual) / downside_dev if downside_dev > 0 else 0.0

    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd))
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    return {
        "CAGR": f"{cagr * 100:.2f}%",
        "Volatility": f"{vol * 100:.2f}%",
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "Max Drawdown": f"{max_dd * 100:.2f}%",
        "Calmar": round(calmar, 2),
        "_cum": cum,
    }


def simulate_portfolio(
    net_rets: np.ndarray,
    weights: dict[str, float],
    rebalance_band: float = 0.05,
    apply_frictions: bool = True,
) -> tuple[np.ndarray, float]:
    """Simulate portfolio with tolerance-band rebalancing, trading costs, and long-term CGT."""
    n_days, n_assets = net_rets.shape
    w_target = np.array([weights.get(c, 0.0) for c in ASSET_ORDER], dtype=float)
    trade_costs = np.array([ASSETS[c].trade_cost for c in ASSET_ORDER], dtype=float)

    # Initial deployment (deduct entry trade cost if frictions enabled)
    entry_cost = float(np.sum(w_target * trade_costs)) if apply_frictions else 0.0
    values = STARTING_CAPITAL * (1.0 - entry_cost) * w_target.copy()
    cost_basis = values.copy()

    daily_port_rets = np.zeros(n_days)
    total_turnover = 0.0

    for d in range(n_days):
        prev_total = float(np.sum(values))
        values *= 1.0 + net_rets[d]
        cur_total = float(np.sum(values))

        # Check monthly (every 21 trading days) for tolerance-band breach
        if d > 0 and d % 21 == 0 and cur_total > 0:
            cur_w = values / cur_total
            max_drift = float(np.max(np.abs(cur_w - w_target)))
            if max_drift >= rebalance_band:
                target_vals = cur_total * w_target
                trades = target_vals - values
                sells = np.maximum(-trades, 0.0)
                buys = np.maximum(trades, 0.0)
                total_turnover += float(np.sum(sells)) / cur_total

                if apply_frictions:
                    # Realized capital gains tax on sold appreciated parcels (50% CGT discount)
                    cgt_paid = 0.0
                    for i in range(n_assets):
                        if sells[i] > 0 and values[i] > 0:
                            sell_frac = sells[i] / values[i]
                            basis_sold = cost_basis[i] * sell_frac
                            gain = max(sells[i] - basis_sold, 0.0)
                            cgt_paid += gain * LONG_TERM_CGT_RATE
                            cost_basis[i] -= basis_sold
                    fee_paid = float(np.sum((sells + buys) * trade_costs))
                    post_total = cur_total - cgt_paid - fee_paid
                    values = post_total * w_target
                    # Update cost basis for newly purchased units
                    for i in range(n_assets):
                        if buys[i] > 0:
                            cost_basis[i] += max(
                                values[i] - (target_vals[i] - buys[i]), 0.0
                            )
                else:
                    values = target_vals
                cur_total = float(np.sum(values))

        daily_port_rets[d] = cur_total / prev_total - 1.0

    ann_turnover = total_turnover / (n_days / TRADING_DAYS)
    return daily_port_rets, ann_turnover


def run_validation():
    df_dates, net_rets = load_net_returns()
    dates = df_dates["date"].to_list()

    portfolios = {
        "1. New Optimal (70 BGBL / 15 BEMG / 12 Gold / 3 BTC)": OPTIMAL_ALLOCATION,
        "2. New Optimal Ex-BTC (75 BGBL / 15 BEMG / 10 Gold)": NO_CRYPTO_OPTIMAL,
        "3. Original Target (85 BGBL / 10 Gold / 5 BTC)": ORIGINAL_ALLOCATION,
        "4. 100% Global Developed (BGBL)": {"BGBL": 1.0},
        "5. All-Equity Global (85 BGBL / 15 BEMG)": {"BGBL": 0.85, "BEMG": 0.15},
        "6. Traditional 60/40 (BGBL / VAF Bonds)": {"BGBL": 0.60, "VAF": 0.40},
    }

    pl.Config.set_tbl_cols(10)
    pl.Config.set_tbl_width_chars(150)

    print(
        f"\n=== Clean Historical Backtest ({dates[0]} to {dates[-1]}, After-Tax + Band Rebal CGT & Fees) ==="
    )
    print(
        "Note: Historical CAGRs reflect 2014-2026 boom conditions (BTC +56%/yr, Gold +12%/yr, BGBL +13%/yr); see optimize_allocation.py & project_portfolio.py for forward CMAs."
    )

    summary_rows = []
    trajectories = {}
    for name, w in portfolios.items():
        rets, turnover = simulate_portfolio(
            net_rets, w, rebalance_band=0.05, apply_frictions=True
        )
        metrics = calculate_metrics(rets, HISA_NET_NOMINAL)
        trajectories[name] = metrics["_cum"]
        summary_rows.append(
            {
                "Portfolio": name,
                "Hist CAGR (Net)": metrics["CAGR"],
                "Volatility": metrics["Volatility"],
                "Sharpe": metrics["Sharpe"],
                "Sortino": metrics["Sortino"],
                "Max Drawdown": metrics["Max Drawdown"],
                "Calmar": metrics["Calmar"],
                "Ann Turnover": f"{turnover * 100:.1f}%",
            }
        )
    print(pl.DataFrame(summary_rows))

    # Clean Historical Regime Stress Tests (incl. Post-2021 BTC Peak)
    regimes = [
        ("2020 Covid Crash & Rebound", date(2020, 1, 1), date(2020, 12, 31)),
        ("2022 Inflation & Rate Shock", date(2022, 1, 1), date(2022, 12, 31)),
        ("Post-2021 Peak (Nov 2021-Present)", date(2021, 11, 10), dates[-1]),
        ("2023-Present Expansion", date(2023, 1, 1), dates[-1]),
    ]

    date_arr = np.array(dates)
    regime_rows = []
    for reg_name, s_d, e_d in regimes:
        mask = (date_arr >= s_d) & (date_arr <= e_d)
        sub_rets = net_rets[mask]

        r_new, _ = simulate_portfolio(
            sub_rets, OPTIMAL_ALLOCATION, apply_frictions=True
        )
        r_nobtc, _ = simulate_portfolio(
            sub_rets, NO_CRYPTO_OPTIMAL, apply_frictions=True
        )
        r_orig, _ = simulate_portfolio(
            sub_rets, ORIGINAL_ALLOCATION, apply_frictions=True
        )
        r_eq, _ = simulate_portfolio(sub_rets, {"BGBL": 1.0}, apply_frictions=True)

        m_new = calculate_metrics(r_new, HISA_NET_NOMINAL)
        m_nobtc = calculate_metrics(r_nobtc, HISA_NET_NOMINAL)
        m_orig = calculate_metrics(r_orig, HISA_NET_NOMINAL)
        m_eq = calculate_metrics(r_eq, HISA_NET_NOMINAL)
        regime_rows.append(
            {
                "Regime": reg_name,
                "New Opt CAGR / DD": f"{m_new['CAGR']} / {m_new['Max Drawdown']}",
                "Ex-BTC CAGR / DD": f"{m_nobtc['CAGR']} / {m_nobtc['Max Drawdown']}",
                "Orig 85/10/5 CAGR / DD": f"{m_orig['CAGR']} / {m_orig['Max Drawdown']}",
                "100% BGBL CAGR / DD": f"{m_eq['CAGR']} / {m_eq['Max Drawdown']}",
            }
        )

    print(
        "\n=== Clean Historical Regime Stress Tests (With Fixed Gold Data & Frictions) ==="
    )
    print(pl.DataFrame(regime_rows))

    # Plot historical comparative trajectories
    plot_names = [
        "1. New Optimal (70 BGBL / 15 BEMG / 12 Gold / 3 BTC)",
        "2. New Optimal Ex-BTC (75 BGBL / 15 BEMG / 10 Gold)",
        "3. Original Target (85 BGBL / 10 Gold / 5 BTC)",
        "4. 100% Global Developed (BGBL)",
        "6. Traditional 60/40 (BGBL / VAF Bonds)",
    ]
    chart_rows = []
    for i, d in enumerate(dates):
        for pname in plot_names:
            chart_rows.append(
                {
                    "date": d,
                    "Portfolio": pname,
                    "Value": round(float(STARTING_CAPITAL * trajectories[pname][i]), 2),
                }
            )

    chart = (
        alt.Chart(pl.DataFrame(chart_rows))
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(
                "Value:Q",
                title="Portfolio Value (AUD, Log Scale, After-Tax & Frictions)",
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color("Portfolio:N", title="Strategy"),
            tooltip=["date:T", "Portfolio:N", "Value:Q"],
        )
        .properties(
            title=f"Clean Historical Portfolio Trajectories (${STARTING_CAPITAL:,.0f} Starting Capital, 2014-2026)",
            width=860,
            height=460,
        )
    )
    CHART_FILE.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(CHART_FILE))
    print(f"\nSaved validation chart to {CHART_FILE}")


if __name__ == "__main__":
    run_validation()
