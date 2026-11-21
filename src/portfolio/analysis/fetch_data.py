from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import yfinance as yf

from portfolio.assumptions import ASSET_ORDER, ASSETS, MARKET_DATA_PATH

DATA_FILE = MARKET_DATA_PATH


def _download_series(ticker: str, start_date: str, end_date: str) -> pl.DataFrame:
    hist = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
    if hist.empty:
        raise RuntimeError(f"No data returned by yfinance for {ticker}")
    dates = [d.date() for d in hist.index]
    prices = hist["Close"].astype(float).tolist()
    return (
        pl.DataFrame({"date": dates, "price": prices})
        .drop_nulls()
        .unique(subset=["date"], keep="last")
        .sort("date")
    )


def _splice_return_space(
    calendar: pl.DataFrame,
    actual_df: pl.DataFrame,
    proxy_df: pl.DataFrame | None,
    min_actual_date: str | None,
) -> np.ndarray:
    """Splice actual and proxy in return space without forward-filling sparse actual quotes."""
    joined = calendar.join(actual_df.rename({"price": "actual"}), on="date", how="left")
    if proxy_df is not None:
        joined = joined.join(proxy_df.rename({"price": "proxy"}), on="date", how="left")
    else:
        joined = joined.with_columns(pl.lit(None, dtype=pl.Float64).alias("proxy"))

    dates = joined["date"].to_list()
    actual = joined["actual"].to_numpy().astype(float)
    proxy = joined["proxy"].to_numpy().astype(float)

    # Fill rare isolated feed gaps in proxy (or actual when no proxy exists)
    def _ffill_gaps(arr: np.ndarray, max_gap: int = 5) -> np.ndarray:
        out = arr.copy()
        gap = 0
        for i in range(1, len(out)):
            if np.isnan(out[i]) and not np.isnan(out[i - 1]) and gap < max_gap:
                out[i] = out[i - 1]
                gap += 1
            else:
                gap = 0
        return out

    if proxy_df is not None:
        proxy = _ffill_gaps(proxy, max_gap=5)
    else:
        actual = _ffill_gaps(actual, max_gap=5)

    cutoff = (
        date.fromisoformat(min_actual_date) if min_actual_date else date(1900, 1, 1)
    )

    n = len(dates)
    rets = np.zeros(n, dtype=float)
    for i in range(1, n):
        use_actual = (
            dates[i] >= cutoff
            and not np.isnan(actual[i])
            and not np.isnan(actual[i - 1])
            and actual[i - 1] > 0
        )
        if use_actual:
            rets[i] = actual[i] / actual[i - 1] - 1.0
        elif not np.isnan(proxy[i]) and not np.isnan(proxy[i - 1]) and proxy[i - 1] > 0:
            rets[i] = proxy[i] / proxy[i - 1] - 1.0
        else:
            rets[i] = 0.0

    # Anchor reconstructed price series to the latest valid actual price
    valid_actual_idx = np.where(~np.isnan(actual))[0]
    anchor_price = (
        float(actual[valid_actual_idx[-1]]) if len(valid_actual_idx) > 0 else 100.0
    )

    # Trim any leading NaN returns if proxy starts slightly later
    prices = np.full(n, np.nan, dtype=float)
    last_valid = n - 1
    while last_valid > 0 and np.isnan(rets[last_valid]):
        last_valid -= 1

    prices[last_valid] = anchor_price
    for i in range(last_valid, 0, -1):
        if np.isnan(rets[i]):
            break
        prices[i - 1] = prices[i] / (1.0 + rets[i])

    return prices


def fetch_market_data(output_path: str | Path = DATA_FILE) -> pl.DataFrame:
    start_date = "2014-11-18"
    end_date = datetime.now(UTC).strftime("%Y-%m-%d")

    raw_cache: dict[str, pl.DataFrame] = {}
    all_tickers = set()
    for code in ASSET_ORDER:
        spec = ASSETS[code]
        all_tickers.add(spec.ticker)
        if spec.proxy_ticker:
            all_tickers.add(spec.proxy_ticker)

    for ticker in sorted(all_tickers):
        raw_cache[ticker] = _download_series(ticker, start_date, end_date)

    # Master ASX trading calendar from VGS.AX + BGBL.AX union
    cal = (
        pl.concat(
            [
                raw_cache["VGS.AX"].select("date"),
                raw_cache["BGBL.AX"].select("date"),
            ]
        )
        .unique()
        .sort("date")
    )

    # Trim trailing calendar dates where any required actual ticker has not yet settled
    latest_dates: list[date] = [
        raw_cache[ASSETS[c].ticker]["date"].to_list()[-1] for c in ASSET_ORDER
    ]
    latest_common = min(latest_dates)
    cal = cal.filter(pl.col("date") <= latest_common)

    columns: dict[str, object] = {"date": cal["date"]}
    for code in ASSET_ORDER:
        spec = ASSETS[code]
        actual_df = raw_cache[spec.ticker]
        proxy_df = raw_cache[spec.proxy_ticker] if spec.proxy_ticker else None
        columns[code] = _splice_return_space(
            cal, actual_df, proxy_df, spec.min_actual_date
        )

    df = pl.DataFrame(columns).fill_nan(None).drop_nulls().sort("date")

    # Data quality verification
    print(
        f"Fetched {len(df)} ASX trading days ({df['date'][0]} to {df['date'][-1]}) across {len(ASSET_ORDER)} assets"
    )
    qc_rows = []
    for code in ASSET_ORDER:
        p = df[code].to_numpy()
        r = p[1:] / p[:-1] - 1.0
        zero_pct = float(np.mean(r == 0.0) * 100.0)
        qc_rows.append(
            {
                "Asset": code,
                "Zero-Ret Days": f"{int(np.sum(r == 0.0))} ({zero_pct:.1f}%)",
                "Min Day": f"{np.min(r) * 100:.2f}%",
                "Max Day": f"{np.max(r) * 100:.2f}%",
                "Hist CAGR": f"{((p[-1] / p[0]) ** (252.0 / len(r)) - 1.0) * 100:.2f}%",
            }
        )
    print(pl.DataFrame(qc_rows))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"Saved clean market data to {out_path}")
    return df


if __name__ == "__main__":
    fetch_market_data()
