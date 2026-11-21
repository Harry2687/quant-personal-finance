import os
from datetime import date

import altair as alt
import numpy as np
import polars as pl
from scipy import stats

from portfolio.analysis.fetch_data import DATA_FILE, fetch_market_data
from portfolio.assumptions import ASSET_ORDER, CHARTS_DIR

CHART_FILE = CHARTS_DIR / "return_distributions.html"


def _weekly_log_returns(
    df: pl.DataFrame, start_date: date | None = None
) -> dict[str, np.ndarray]:
    sub = df if start_date is None else df.filter(pl.col("date") >= start_date)
    weekly = (
        sub.sort("date")
        .group_by_dynamic("date", every="1w")
        .agg([pl.col(c).last() for c in ASSET_ORDER])
        .sort("date")
    )
    return {c: np.diff(np.log(weekly[c].to_numpy())) for c in ASSET_ORDER}


def analyze_distributions():
    if not os.path.exists(DATA_FILE):
        fetch_market_data(DATA_FILE)
    df = pl.read_parquet(DATA_FILE).sort("date")
    trading_days = 252

    daily_log_rets = {c: np.diff(np.log(df[c].to_numpy())) for c in ASSET_ORDER}
    weekly_log_rets = _weekly_log_returns(df)
    weekly_2020 = _weekly_log_returns(df, start_date=date(2020, 1, 1))

    moments = []
    fit_records = []
    fitted_params: dict[str, tuple[tuple, tuple]] = {}

    for a in ASSET_ORDER:
        r = daily_log_rets[a]
        mean_ann = float(np.mean(r) * trading_days)
        vol_ann = float(np.std(r) * np.sqrt(trading_days))
        skew = float(stats.skew(r))
        kurt = float(stats.kurtosis(r))
        _jb_stat, jb_p = stats.jarque_bera(r)

        norm_params = stats.norm.fit(r)
        t_params = stats.t.fit(r)
        nig_params = stats.norminvgauss.fit(r)
        fitted_params[a] = (norm_params, t_params)

        ll_norm = float(np.sum(stats.norm.logpdf(r, *norm_params)))
        ll_t = float(np.sum(stats.t.logpdf(r, *t_params)))
        ll_nig = float(np.sum(stats.norminvgauss.logpdf(r, *nig_params)))

        aic_norm = 2 * 2 - 2 * ll_norm
        aic_t = 2 * 3 - 2 * ll_t
        aic_nig = 2 * 4 - 2 * ll_nig

        moments.append(
            {
                "Asset": a,
                "Hist Mean (Log)": f"{mean_ann * 100:.2f}%",
                "Daily Vol (Ann)": f"{vol_ann * 100:.2f}%",
                "Skewness": round(skew, 3),
                "Excess Kurtosis": round(kurt, 3),
                "JB p-value": f"{jb_p:.2e}",
            }
        )

        best = min(
            [("Normal", aic_norm), ("Student-t", aic_t), ("NIG", aic_nig)],
            key=lambda x: x[1],
        )[0]
        fit_records.append(
            {
                "Asset": a,
                "Student-t df (nu)": round(float(t_params[0]), 2),
                "Student-t Scale": round(float(t_params[2]), 4),
                "AIC (Normal)": round(aic_norm, 1),
                "AIC (Student-t)": round(aic_t, 1),
                "AIC (NIG)": round(aic_nig, 1),
                "Best Fit Model": best,
            }
        )

    pl.Config.set_tbl_cols(10)
    pl.Config.set_tbl_width_chars(140)

    print("\n=== 1. Empirical Moments of Clean Daily Log Returns (2014-2026) ===")
    print(pl.DataFrame(moments))

    print("\n=== 2. Parametric Distribution Fits & Model Selection ===")
    print(pl.DataFrame(fit_records))

    # Weekly tail dependence (eliminates ASX-vs-UTC timezone lag between BGBL and BTC)
    bgbl_w = weekly_log_rets["BGBL"]
    q05_bgbl = float(np.percentile(bgbl_w, 5))
    idx_crash = np.where(bgbl_w <= q05_bgbl)[0]

    tail_rows = []
    for a in ASSET_ORDER:
        rw = weekly_log_rets[a]
        q05_a = float(np.percentile(rw, 5))
        cond_prob = float(np.mean(rw[idx_crash] <= q05_a) * 100.0)
        crash_mean = float(np.mean(rw[idx_crash]) * 100.0)
        corr_full = float(np.corrcoef(bgbl_w, rw)[0, 1])
        corr_2020 = float(np.corrcoef(weekly_2020["BGBL"], weekly_2020[a])[0, 1])
        tail_rows.append(
            {
                "Asset": a,
                "Wk 5th %ile": f"{q05_a * 100:.2f}%",
                "Mean on BGBL Crash Wk": f"{crash_mean:.2f}%",
                "P(a<=q05 | BGBL<=q05)": f"{cond_prob:.1f}%",
                "Wk Corr BGBL (Full)": round(corr_full, 3),
                "Wk Corr BGBL (2020+)": round(corr_2020, 3),
            }
        )

    print(
        f"\n=== 3. Weekly Tail Dependence & Crisis Co-Movement (BGBL Worst 5% Weeks, N={len(idx_crash)}) ==="
    )
    print(pl.DataFrame(tail_rows))

    # True Gaussian KDE vs Fitted Student-t vs Fitted Normal for all 7 assets
    chart_rows = []
    for a in ASSET_ORDER:
        r = daily_log_rets[a]
        lo, hi = np.percentile(r, [0.5, 99.5])
        span = max(abs(lo), abs(hi))
        xs = np.linspace(-span, span, 180)
        kde = stats.gaussian_kde(r)
        norm_params, t_params = fitted_params[a]

        for x, y_kde, y_t, y_n in zip(
            xs,
            kde(xs),
            stats.t.pdf(xs, *t_params),
            stats.norm.pdf(xs, *norm_params),
            strict=True,
        ):
            chart_rows.append(
                {
                    "Asset": a,
                    "Return": round(float(x), 5),
                    "Density": round(float(y_kde), 3),
                    "Type": "Empirical KDE",
                }
            )
            chart_rows.append(
                {
                    "Asset": a,
                    "Return": round(float(x), 5),
                    "Density": round(float(y_t), 3),
                    "Type": "Fitted Student-t",
                }
            )
            chart_rows.append(
                {
                    "Asset": a,
                    "Return": round(float(x), 5),
                    "Density": round(float(y_n), 3),
                    "Type": "Fitted Normal",
                }
            )

    chart_df = pl.DataFrame(chart_rows)
    charts = []
    for a in ASSET_ORDER:
        sub = chart_df.filter(pl.col("Asset") == a)
        c = (
            alt.Chart(sub)
            .mark_line()
            .encode(
                x=alt.X("Return:Q", title="Daily Log Return"),
                y=alt.Y("Density:Q", title="Probability Density"),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=["Empirical KDE", "Fitted Student-t", "Fitted Normal"],
                        range=["#2ca02c", "#1f77b4", "#d62728"],
                    ),
                ),
                strokeDash=alt.condition(
                    alt.datum.Type == "Fitted Normal", alt.value([4, 4]), alt.value([0])
                ),
            )
            .properties(title=f"{a} Daily Log Returns", width=260, height=210)
        )
        charts.append(c)

    panel = alt.vconcat(alt.hconcat(*charts[:3]), alt.hconcat(*charts[3:]))
    CHART_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel.save(str(CHART_FILE))
    print(f"\nSaved distribution visualization to {CHART_FILE}")


if __name__ == "__main__":
    analyze_distributions()
