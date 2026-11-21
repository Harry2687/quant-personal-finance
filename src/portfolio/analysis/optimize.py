import os
from datetime import date

import altair as alt
import numpy as np
import polars as pl
from scipy.optimize import minimize

from portfolio.analysis.fetch_data import DATA_FILE, fetch_market_data
from portfolio.assumptions import (
    ASSET_ORDER,
    ASSETS,
    CHARTS_DIR,
    HISA_NET_NOMINAL,
    INFLATION_RATE,
    NO_CRYPTO_OPTIMAL,
    OPTIMAL_ALLOCATION,
    ORIGINAL_ALLOCATION,
)

CHART_FILE = CHARTS_DIR / "optimization_frontier.html"
WEEKS_PER_YEAR = 52.0


def load_weekly_returns(
    df: pl.DataFrame, start_date: date | None = None
) -> tuple[np.ndarray, list[date]]:
    """Resample ASX-aligned daily prices to weekly (Friday) simple returns to remove timezone lag."""
    sub = df if start_date is None else df.filter(pl.col("date") >= start_date)
    weekly = (
        sub.sort("date")
        .group_by_dynamic("date", every="1w")
        .agg([pl.col(c).last() for c in ASSET_ORDER])
        .sort("date")
    )
    prices = weekly.select(ASSET_ORDER).to_numpy()
    rets = prices[1:] / prices[:-1] - 1.0
    return rets, weekly["date"].to_list()[1:]


def ledoit_wolf_constant_corr(rets: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) covariance shrinkage toward the constant-correlation target."""
    t, n = rets.shape
    x = rets - np.mean(rets, axis=0, keepdims=True)
    sample = (x.T @ x) / t

    var = np.diag(sample)
    std = np.sqrt(var)
    corr = sample / np.outer(std, std)
    r_bar = (np.sum(corr) - n) / (n * (n - 1))

    target = r_bar * np.outer(std, std)
    np.fill_diagonal(target, var)

    x2 = x**2
    phi_mat = (x2.T @ x2) / t - sample**2
    pi_hat = float(np.sum(phi_mat))

    theta_ii = np.sum((x2 - var) ** 2, axis=0) / t
    rho_diag = float(np.sum(theta_ii))
    rho_off = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x_ij = x[:, i] * x[:, j]
            cov_i = np.mean((x2[:, i] - var[i]) * (x_ij - sample[i, j]))
            cov_j = np.mean((x2[:, j] - var[j]) * (x_ij - sample[i, j]))
            rho_off += (r_bar / 2.0) * (
                (std[j] / std[i]) * cov_i + (std[i] / std[j]) * cov_j
            )
    rho_hat = rho_diag + rho_off

    gamma_hat = float(np.sum((target - sample) ** 2))
    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
    shrink = float(np.clip(kappa / t, 0.0, 1.0))

    cov_shrunk = shrink * target + (1.0 - shrink) * sample
    return cov_shrunk, shrink


def recenter_empirical_returns(
    weekly_rets: np.ndarray, target_annual_net: np.ndarray
) -> np.ndarray:
    """Shift empirical log returns so annualized arithmetic expectation matches forward after-tax CMAs."""
    log_r = np.log1p(weekly_rets)
    target_weekly_arith = (1.0 + target_annual_net) ** (1.0 / WEEKS_PER_YEAR) - 1.0
    shifted = np.zeros_like(weekly_rets)
    for j in range(weekly_rets.shape[1]):
        mean_exp = float(np.mean(np.exp(log_r[:, j])))
        delta = np.log((1.0 + target_weekly_arith[j]) / mean_exp)
        shifted[:, j] = np.expm1(log_r[:, j] + delta)
    return shifted


def _solve_slsqp(obj, bounds, x0_list, extra_constraints=()):
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}, *extra_constraints]
    best = None
    for x0 in x0_list:
        res = minimize(
            obj,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-10, "maxiter": 500},
        )
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("Optimization failed to converge")
    w = np.maximum(best.x, 0.0)
    return w / np.sum(w)


def _initial_guesses(n: int, bounds: list[tuple[float, float]]) -> list[np.ndarray]:
    guesses = [np.full(n, 1.0 / n)]
    rng = np.random.default_rng(42)
    for _ in range(8):
        g = rng.dirichlet(np.ones(n))
        g = np.clip(g, [b[0] for b in bounds], [b[1] for b in bounds])
        if np.sum(g) > 0:
            guesses.append(g / np.sum(g))
    return guesses


def optimize_max_sharpe(
    mu: np.ndarray, cov: np.ndarray, rf: float, bounds: list[tuple[float, float]]
) -> np.ndarray:
    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        return -(ret - rf) / vol if vol > 1e-9 else 1e6

    return _solve_slsqp(neg_sharpe, bounds, _initial_guesses(len(mu), bounds))


def optimize_min_vol(
    cov: np.ndarray,
    bounds: list[tuple[float, float]],
    target_ret: float | None = None,
    mu: np.ndarray | None = None,
) -> np.ndarray:
    extra = ()
    if target_ret is not None and mu is not None:
        extra = ({"type": "eq", "fun": lambda w: float(w @ mu) - target_ret},)
    return _solve_slsqp(
        lambda w: float(np.sqrt(w @ cov @ w)),
        bounds,
        _initial_guesses(cov.shape[0], bounds),
        extra_constraints=extra,
    )


def optimize_crra_utility(
    recentered_weekly: np.ndarray, gamma: float, bounds: list[tuple[float, float]]
) -> np.ndarray:
    """Maximize expected CRRA power utility on empirical tail-preserving weekly returns."""

    def neg_utility(w):
        port_gross = np.maximum(1.0 + recentered_weekly @ w, 1e-4)
        if abs(gamma - 1.0) < 1e-6:
            return -float(np.mean(np.log(port_gross)))
        return -float(np.mean((port_gross ** (1.0 - gamma) - 1.0) / (1.0 - gamma)))

    return _solve_slsqp(
        neg_utility, bounds, _initial_guesses(recentered_weekly.shape[1], bounds)
    )


def portfolio_stats(
    w: np.ndarray,
    mu_net: np.ndarray,
    cov_ann: np.ndarray,
    recentered_weekly: np.ndarray,
) -> dict:
    exp_nom = float(w @ mu_net)
    vol = float(np.sqrt(w @ cov_ann @ w))
    sharpe = (exp_nom - HISA_NET_NOMINAL) / vol if vol > 0 else 0.0

    port_w = recentered_weekly @ w
    q05 = np.percentile(port_w, 5)
    cvar95_wk = float(np.mean(port_w[port_w <= q05]))
    geo_nom = float(np.exp(np.mean(np.log1p(port_w)) * WEEKS_PER_YEAR) - 1.0)
    geo_real = (1.0 + geo_nom) / (1.0 + INFLATION_RATE) - 1.0

    return {
        "Arith_Nom": exp_nom,
        "Geo_Nom": geo_nom,
        "Geo_Real": geo_real,
        "Vol": vol,
        "Sharpe": sharpe,
        "CVaR95_Wk": cvar95_wk,
    }


def format_weights(w: np.ndarray) -> str:
    parts = [
        f"{c}:{v * 100:.0f}%" for c, v in zip(ASSET_ORDER, w, strict=True) if v >= 0.005
    ]
    return " / ".join(parts)


def run_optimization():
    if not os.path.exists(DATA_FILE):
        fetch_market_data(DATA_FILE)
    df = pl.read_parquet(DATA_FILE).sort("date")

    wk_full, _ = load_weekly_returns(df)
    wk_recent, _ = load_weekly_returns(df, start_date=date(2020, 1, 1))

    cov_wk_full, shrink_full = ledoit_wolf_constant_corr(wk_full)
    cov_wk_recent, shrink_recent = ledoit_wolf_constant_corr(wk_recent)
    cov_ann_full = cov_wk_full * WEEKS_PER_YEAR
    cov_ann_recent = cov_wk_recent * WEEKS_PER_YEAR

    mu_net = np.array([ASSETS[c].cma_net_nominal for c in ASSET_ORDER])
    mu_gross = np.array([ASSETS[c].cma_gross_nominal for c in ASSET_ORDER])
    tax_drags = np.array([ASSETS[c].annual_tax_drag for c in ASSET_ORDER])
    vols_full = np.sqrt(np.diag(cov_ann_full))
    vols_recent = np.sqrt(np.diag(cov_ann_recent))

    pl.Config.set_tbl_cols(12)
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(180)
    pl.Config.set_fmt_str_lengths(120)

    print("\n=== 1. Forward CMAs (Ex-Australia Universe, 39% AU Marginal Tax Rate) ===")
    cma_rows = []
    for i, c in enumerate(ASSET_ORDER):
        s = ASSETS[c]
        cma_rows.append(
            {
                "Asset": c,
                "MER": f"{s.mer * 100:.2f}%",
                "Gross Nom": f"{mu_gross[i] * 100:.2f}%",
                "Tax Drag": f"{tax_drags[i] * 100:.2f}%",
                "Net Nom": f"{mu_net[i] * 100:.2f}%",
                "Net Real": f"{s.cma_net_real * 100:.2f}%",
                "Wk Vol (Full)": f"{vols_full[i] * 100:.1f}%",
                "Wk Vol (2020+)": f"{vols_recent[i] * 100:.1f}%",
            }
        )
    print(pl.DataFrame(cma_rows))
    print(
        f"Ledoit-Wolf Shrinkage Intensity -> Full (2014-2026): {shrink_full:.3f} | Recent (2020-2026): {shrink_recent:.3f}"
    )

    print(
        "\n=== 2. Weekly Correlation Matrix (2020-2026 Post-Institutional Regime) ==="
    )
    corr_recent = np.corrcoef(wk_recent.T)
    corr_df = pl.DataFrame(
        {
            "Asset": ASSET_ORDER,
            **{c: np.round(corr_recent[:, i], 3) for i, c in enumerate(ASSET_ORDER)},
        }
    )
    print(corr_df)

    rec_full = recenter_empirical_returns(wk_full, mu_net)
    rec_recent = recenter_empirical_returns(wk_recent, mu_net)

    # Order: [BGBL, HGBL, BEMG, VAF, PMGOLD, BTC]
    bounds_unconstrained = [(0.0, 1.0)] * len(ASSET_ORDER)
    bounds_guard5 = [
        (0.0, 1.0),  # BGBL
        (0.0, 0.50),  # HGBL
        (0.0, 0.15),  # BEMG (capped near global market-cap + EM risk guardrail)
        (0.0, 1.0),  # VAF
        (0.0, 0.15),  # PMGOLD
        (0.0, 0.05),  # BTC <= 5%
    ]
    bounds_guard20em = list(bounds_guard5)
    bounds_guard20em[2] = (0.0, 0.20)  # BEMG <= 20%
    bounds_guard10btc = list(bounds_guard5)
    bounds_guard10btc[5] = (0.0, 0.10)  # BTC <= 10%

    def _to_vec(w_dict: dict[str, float]) -> np.ndarray:
        return np.array([w_dict.get(c, 0.0) for c in ASSET_ORDER])

    candidates = [
        (
            "MVO Max-Sharpe (Unconstrained)",
            optimize_max_sharpe(
                mu_net, cov_ann_full, HISA_NET_NOMINAL, bounds_unconstrained
            ),
        ),
        (
            "MVO Max-Sharpe (BEMG<=15%, Gold<=15%, BTC<=5%)",
            optimize_max_sharpe(mu_net, cov_ann_full, HISA_NET_NOMINAL, bounds_guard5),
        ),
        (
            "MVO Max-Sharpe (2020+ Cov, Guarded)",
            optimize_max_sharpe(
                mu_net, cov_ann_recent, HISA_NET_NOMINAL, bounds_guard5
            ),
        ),
        (
            "MVO Max-Sharpe (BEMG<=20%, Guarded)",
            optimize_max_sharpe(
                mu_net, cov_ann_full, HISA_NET_NOMINAL, bounds_guard20em
            ),
        ),
        (
            "CRRA Aggressive (gamma=2.0, BTC<=10%)",
            optimize_crra_utility(rec_full, 2.0, bounds_guard10btc),
        ),
        (
            "CRRA Growth (gamma=3.0, Guarded)",
            optimize_crra_utility(rec_full, 3.0, bounds_guard5),
        ),
        (
            "CRRA Growth (gamma=3.0, 2020+ Tails)",
            optimize_crra_utility(rec_recent, 3.0, bounds_guard5),
        ),
        (
            "CRRA Balanced (gamma=4.5, Guarded)",
            optimize_crra_utility(rec_full, 4.5, bounds_guard5),
        ),
        ("Recommended Rounded (70/15/12/3)", _to_vec(OPTIMAL_ALLOCATION)),
        ("Recommended Ex-BTC (75/15/10)", _to_vec(NO_CRYPTO_OPTIMAL)),
        ("Benchmark: Original 85/10/5", _to_vec(ORIGINAL_ALLOCATION)),
        ("Benchmark: 100% BGBL", _to_vec({"BGBL": 1.0})),
        ("Benchmark: 60% BGBL / 40% VAF", _to_vec({"BGBL": 0.60, "VAF": 0.40})),
    ]

    print(
        "\n=== 3. From-Scratch Quantitative Optimization Results (Ex-Australia, After-Tax CMAs) ==="
    )
    opt_rows = []
    for label, w in candidates:
        st = portfolio_stats(w, mu_net, cov_ann_full, rec_full)
        opt_rows.append(
            {
                "Portfolio": label,
                "Net Geo Nom": f"{st['Geo_Nom'] * 100:.2f}%",
                "Net Geo Real": f"{st['Geo_Real'] * 100:.2f}%",
                "Ann Vol": f"{st['Vol'] * 100:.2f}%",
                "Sharpe": round(st["Sharpe"], 3),
                "Wk CVaR95": f"{st['CVaR95_Wk'] * 100:.2f}%",
                "Weights": format_weights(w),
            }
        )
    print(pl.DataFrame(opt_rows))

    print(
        "\n=== 4. CMA Sensitivity Sweep (CRRA Growth gamma=3.0, BEMG<=15%, Gold<=15%, BTC<=10%) ==="
    )
    sens_rows = []
    for btc_cma in [0.04, 0.10, 0.15, 0.20]:
        for gold_cma in [0.025, 0.04, 0.06]:
            mu_s = mu_net.copy()
            mu_s[ASSET_ORDER.index("PMGOLD")] = gold_cma
            mu_s[ASSET_ORDER.index("BTC")] = btc_cma
            rec_s = recenter_empirical_returns(wk_full, mu_s)
            w_s = optimize_crra_utility(rec_s, 3.0, bounds_guard10btc)
            st_s = portfolio_stats(w_s, mu_s, cov_ann_full, rec_s)
            sens_rows.append(
                {
                    "BTC CMA": f"{btc_cma * 100:.0f}%",
                    "Gold CMA": f"{gold_cma * 100:.1f}%",
                    "Net Geo Real": f"{st_s['Geo_Real'] * 100:.2f}%",
                    "Vol": f"{st_s['Vol'] * 100:.1f}%",
                    "Sharpe": round(st_s["Sharpe"], 3),
                    "Optimal Weights": format_weights(w_s),
                }
            )
    print(pl.DataFrame(sens_rows))

    # Efficient Frontier chart (Guarded)
    w_minv = optimize_min_vol(cov_ann_full, bounds_guard5)
    min_r = float(w_minv @ mu_net)
    max_r = float(np.dot(np.array([0.80, 0.0, 0.15, 0.0, 0.0, 0.05]), mu_net))
    target_rets = np.linspace(min_r, max_r, 35)
    frontier_rows = []
    for tr in target_rets:
        try:
            wf = optimize_min_vol(cov_ann_full, bounds_guard5, target_ret=tr, mu=mu_net)
            st = portfolio_stats(wf, mu_net, cov_ann_full, rec_full)
            frontier_rows.append(
                {
                    "Volatility": round(st["Vol"] * 100, 2),
                    "Expected_Return": round(st["Arith_Nom"] * 100, 2),
                    "Sharpe": round(st["Sharpe"], 3),
                    "Weights": format_weights(wf),
                }
            )
        except RuntimeError:
            continue

    frontier_df = pl.DataFrame(frontier_rows)
    highlight_rows = []
    for label, w in [
        ("Max-Sharpe (Guarded)", candidates[1][1]),
        ("Recommended (70/15/12/3)", _to_vec(OPTIMAL_ALLOCATION)),
        ("Recommended Ex-BTC (75/15/10)", _to_vec(NO_CRYPTO_OPTIMAL)),
        ("Original 85/10/5", _to_vec(ORIGINAL_ALLOCATION)),
        ("100% BGBL", _to_vec({"BGBL": 1.0})),
        ("60/40 BGBL/VAF", _to_vec({"BGBL": 0.60, "VAF": 0.40})),
    ]:
        st = portfolio_stats(w, mu_net, cov_ann_full, rec_full)
        highlight_rows.append(
            {
                "Label": label,
                "Volatility": round(st["Vol"] * 100, 2),
                "Expected_Return": round(st["Arith_Nom"] * 100, 2),
                "Sharpe": round(st["Sharpe"], 3),
                "Weights": format_weights(w),
            }
        )
    highlight_df = pl.DataFrame(highlight_rows)

    line = (
        alt.Chart(frontier_df)
        .mark_line(color="#1f77b4", strokeWidth=2.5)
        .encode(
            x=alt.X(
                "Volatility:Q", title="Annualized Volatility (%, Weekly Shrunk Cov)"
            ),
            y=alt.Y("Expected_Return:Q", title="After-Tax Expected Nominal Return (%)"),
            tooltip=["Volatility:Q", "Expected_Return:Q", "Sharpe:Q", "Weights:N"],
        )
    )
    pts = (
        alt.Chart(highlight_df)
        .mark_point(filled=True, size=110)
        .encode(
            x="Volatility:Q",
            y="Expected_Return:Q",
            color=alt.Color("Label:N", title="Portfolio"),
            tooltip=[
                "Label:N",
                "Volatility:Q",
                "Expected_Return:Q",
                "Sharpe:Q",
                "Weights:N",
            ],
        )
    )
    chart = (line + pts).properties(
        title="After-Tax Efficient Frontier (Ex-AU Universe with BEMG, Forward CMAs, BTC<=5%, Gold<=15%)",
        width=820,
        height=460,
    )
    CHART_FILE.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(CHART_FILE))
    print(f"\nSaved efficient frontier chart to {CHART_FILE}")


if __name__ == "__main__":
    run_optimization()
