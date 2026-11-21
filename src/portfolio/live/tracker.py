import csv
import json
import os
import re
import shutil
import sqlite3
import subprocess
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yfinance as yf

from portfolio.assumptions import (
    FORTNIGHTLY_DCA,
    LEDGER_DB_PATH,
    LEDGER_START_DATE,
    OPTIMAL_ALLOCATION,
)

DB_PATH = LEDGER_DB_PATH
SWYFTX_BASE = "https://api.swyftx.com.au"
SWYFTX_KEYCHAIN_SERVICE = "swyftx-api-key"
DEFAULT_LEDGER_START = LEDGER_START_DATE


def _gws_bin() -> str:
    found = shutil.which("gws")
    if found:
        return found
    for candidate in ("/opt/homebrew/bin/gws", "/usr/local/bin/gws"):
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("gws binary not found in PATH or /opt/homebrew/bin/gws")


def _connect_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            external_id TEXT,
            trade_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            price_aud REAL NOT NULL,
            total_aud REAL NOT NULL,
            brokerage_aud REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_emails (
            message_id TEXT PRIMARY KEY,
            subject TEXT,
            processed_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _parse_currency(val: str) -> float:
    cleaned = val.replace("$", "").replace(",", "").strip()
    return abs(float(cleaned)) if cleaned else 0.0


def _trade_exists(
    conn: sqlite3.Connection,
    trade_date: str,
    ticker: str,
    side: str,
    total_aud: float,
    external_id: str | None = None,
) -> bool:
    if external_id:
        row = conn.execute(
            "SELECT 1 FROM trades WHERE external_id = ?", (external_id,)
        ).fetchone()
        if row:
            return True
    # Deduplicate CSV vs Gmail confirmations on exact date, ticker, side, and rounded AUD amount
    row = conn.execute(
        """
        SELECT 1 FROM trades
        WHERE trade_date = ? AND ticker = ? AND side = ? AND ABS(total_aud - ?) < 0.05
        """,
        (trade_date, ticker, side, total_aud),
    ).fetchone()
    return row is not None


def import_betashares_csv(csv_path: str, db_path: Path = DB_PATH) -> dict:
    """Import a Betashares Direct transaction history CSV into the local SQLite ledger."""
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    conn = _connect_db(db_path)
    inserted = 0
    skipped = 0
    now_iso = datetime.now(UTC).isoformat()

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            act = (row.get("Activity Type") or "").strip().lower()
            if act not in {"buy", "sell", "reinvestment", "drp"}:
                continue
            raw_sym = (row.get("Symbol") or "").strip()
            if not raw_sym:
                continue
            ticker = raw_sym.split(":")[0].upper()
            side = "SELL" if act == "sell" else "BUY"
            trade_dt = datetime.strptime(
                row["Effective Date"].strip(), "%d/%m/%Y"
            ).strftime("%Y-%m-%d")
            qty = abs(float(row["Quantity"].replace(",", "").strip()))
            price = _parse_currency(row.get("Price", ""))
            total = _parse_currency(row.get("Gross", ""))
            brok = _parse_currency(row.get("Brokerage", ""))

            if _trade_exists(conn, trade_dt, ticker, side, total):
                skipped += 1
                continue

            conn.execute(
                """
                INSERT INTO trades (
                    source, external_id, trade_date, ticker, side,
                    quantity, price_aud, total_aud, brokerage_aud, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "betashares_csv",
                    None,
                    trade_dt,
                    ticker,
                    side,
                    qty,
                    price,
                    total,
                    brok,
                    now_iso,
                ),
            )
            inserted += 1

    conn.commit()
    conn.close()
    return {"csv_path": str(path), "inserted": inserted, "skipped_duplicates": skipped}


def _gws_json(args: list[str]) -> dict:
    out = subprocess.check_output(
        [_gws_bin(), *args], text=True, stderr=subprocess.DEVNULL
    )
    idx = out.find("{")
    if idx == -1:
        raise RuntimeError(f"Unexpected gws output: {out[:200]}")
    return json.loads(out[idx:])


def _parse_betashares_order_body(body: str) -> dict | None:
    """Parse structured trade confirmation email from orders@notification.betashares.com.au."""
    m_hdr = re.search(
        r"Your\s+(buy|sell|recurring)\s+order\s+for\s+([A-Z0-9]+)\s+has\s+been\s+filled",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    m_qty = re.search(r"Quantity\s+([0-9,]+\.?\d*)", body, re.DOTALL)
    if not (m_hdr and m_qty):
        return None

    m_oid = re.search(r"Order ID:\s*([A-Z0-9-]+)", body)
    m_dt = re.search(r"filled at\s+(.+?)\s*\(AE[SD]T\)", body, re.DOTALL)
    m_px = re.search(r"Average price\s+\$([0-9,]+\.\d+)", body, re.DOTALL)
    m_tot = re.search(r"Total\s+\$([0-9,]+\.\d+)", body, re.DOTALL)

    side = "SELL" if m_hdr.group(1).lower() == "sell" else "BUY"
    ticker = m_hdr.group(2).upper()
    qty = float(m_qty.group(1).replace(",", ""))
    price = float(m_px.group(1).replace(",", "")) if m_px else 0.0
    total = float(m_tot.group(1).replace(",", "")) if m_tot else round(qty * price, 2)

    trade_date = ""
    if m_dt:
        dt_str = re.sub(r"\s+", " ", m_dt.group(1)).strip()
        for fmt in ("%A, %d %B %Y", "%d %B %Y"):
            try:
                trade_date = datetime.strptime(dt_str, fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

    return {
        "order_id": m_oid.group(1) if m_oid else None,
        "trade_date": trade_date,
        "ticker": ticker,
        "side": side,
        "quantity": qty,
        "price_aud": price,
        "total_aud": total,
    }


def sync_betashares_gmail(
    since_date: str = DEFAULT_LEDGER_START, db_path: Path = DB_PATH
) -> dict:
    """Fetch new Betashares Direct order confirmations from Gmail via gws CLI."""
    conn = _connect_db(db_path)
    seen_ids = {
        r["message_id"]
        for r in conn.execute("SELECT message_id FROM processed_emails").fetchall()
    }

    q_date = since_date.replace("-", "/")
    query = f"from:orders@notification.betashares.com.au after:{q_date}"
    res = _gws_json(
        [
            "gmail",
            "users",
            "messages",
            "list",
            "--params",
            json.dumps({"userId": "me", "q": query, "maxResults": 100}),
        ]
    )
    msgs = res.get("messages", [])

    inserted = 0
    deduped = 0
    already_processed = 0
    now_iso = datetime.now(UTC).isoformat()

    for m in msgs:
        mid = m["id"]
        if mid in seen_ids:
            already_processed += 1
            continue

        body = subprocess.check_output(
            [_gws_bin(), "gmail", "+read", "--id", mid],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        parsed = _parse_betashares_order_body(body)
        if parsed and parsed["trade_date"]:
            if _trade_exists(
                conn,
                parsed["trade_date"],
                parsed["ticker"],
                parsed["side"],
                parsed["total_aud"],
                external_id=parsed["order_id"],
            ):
                deduped += 1
            else:
                conn.execute(
                    """
                    INSERT INTO trades (
                        source, external_id, trade_date, ticker, side,
                        quantity, price_aud, total_aud, brokerage_aud, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "betashares_gmail",
                        parsed["order_id"] or mid,
                        parsed["trade_date"],
                        parsed["ticker"],
                        parsed["side"],
                        parsed["quantity"],
                        parsed["price_aud"],
                        parsed["total_aud"],
                        0.0,
                        now_iso,
                    ),
                )
                inserted += 1

        conn.execute(
            "INSERT OR IGNORE INTO processed_emails (message_id, subject, processed_at) VALUES (?, ?, ?)",
            (mid, parsed["ticker"] if parsed else "other", now_iso),
        )

    conn.commit()
    conn.close()
    return {
        "query": query,
        "emails_found": len(msgs),
        "new_trades_inserted": inserted,
        "matched_existing_csv": deduped,
        "already_processed": already_processed,
    }


def _get_swyftx_api_key() -> str:
    env_key = os.environ.get("SWYFTX_API_KEY", "").strip()
    if env_key:
        return env_key
    return subprocess.check_output(
        ["security", "find-generic-password", "-s", SWYFTX_KEYCHAIN_SERVICE, "-w"],
        text=True,
    ).strip()


def fetch_swyftx_live(since_date: str = DEFAULT_LEDGER_START) -> dict:
    """Pull live Swyftx balances and recent BTC buy history using read-only API key."""
    key = _get_swyftx_api_key()
    headers = {"Content-Type": "application/json", "User-Agent": "PortfolioMCP/1.0"}
    req = urllib.request.Request(
        f"{SWYFTX_BASE}/auth/refresh/",
        data=json.dumps({"apiKey": key}).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=12) as r:
        token = json.loads(r.read())["accessToken"]

    auth_h = {"Authorization": f"Bearer {token}", "User-Agent": "PortfolioMCP/1.0"}
    with urllib.request.urlopen(
        urllib.request.Request(f"{SWYFTX_BASE}/user/balance/", headers=auth_h),
        timeout=12,
    ) as r:
        balances_raw = json.loads(r.read())

    with urllib.request.urlopen(
        urllib.request.Request(f"{SWYFTX_BASE}/markets/assets/", headers=auth_h),
        timeout=12,
    ) as r:
        asset_map = {str(a["id"]): a.get("code", "") for a in json.loads(r.read())}

    holdings: dict[str, float] = {}
    for b in balances_raw:
        amt = float(b.get("availableBalance", 0) or 0) + float(
            b.get("stakingBalance", 0) or 0
        )
        code = asset_map.get(str(b.get("assetId", "")), "")
        if code and amt > 1e-8:
            holdings[code] = amt

    # Fetch recent BTC (assetId=3) buy history since portfolio start for exact cost basis
    with urllib.request.urlopen(
        urllib.request.Request(
            f"{SWYFTX_BASE}/history/all/type/3/?limit=50", headers=auth_h
        ),
        timeout=12,
    ) as r:
        hist = json.loads(r.read())

    cutoff_dt = date.fromisoformat(since_date)
    btc_buys: list[dict[str, Any]] = []
    for h in hist:
        act = (h.get("actionType") or "").lower()
        status = (h.get("status") or "").lower()
        ts = h.get("updated") or h.get("created") or 0
        if "buy" in act and status == "complete" and str(h.get("asset")) == "3" and ts:
            dt = datetime.fromtimestamp(ts / 1000.0, tz=UTC).date()
            if dt >= cutoff_dt and h.get("quantity"):
                # quantityAsset=1 means quantity is AUD spend
                btc_buys.append(
                    {
                        "trade_date": dt.isoformat(),
                        "ticker": "BTC",
                        "side": "BUY",
                        "total_aud": float(h["quantity"]),
                    }
                )

    btc_cost_basis = sum(float(b["total_aud"]) for b in btc_buys)
    return {
        "holdings": holdings,
        "btc_units": holdings.get("BTC", 0.0),
        "aud_cash": holdings.get("AUD", 0.0),
        "btc_cost_basis_aud": round(btc_cost_basis, 2),
        "btc_buys": sorted(btc_buys, key=lambda x: x["trade_date"]),
    }


def fetch_live_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch latest AUD market prices via yfinance."""
    yf_map = {t: ("BTC-AUD" if t == "BTC" else f"{t}.AX") for t in tickers}
    prices: dict[str, float] = {}
    for code, yf_sym in yf_map.items():
        hist = yf.Ticker(yf_sym).history(period="5d")
        if not hist.empty:
            prices[code] = float(hist["Close"].dropna().iloc[-1])
    return prices


def _xirr(cashflows: list[tuple[date, float]]) -> float | None:
    """Compute annualized Money-Weighted Return (XIRR) via Newton-Raphson."""
    if len(cashflows) < 2:
        return None
    t0 = min(d for d, _ in cashflows)
    times = np.array([(d - t0).days / 365.25 for d, _ in cashflows], dtype=float)
    amts = np.array([a for _, a in cashflows], dtype=float)
    if np.max(times) < 0.05:  # Avoid noisy annualized extrapolation on <18 days
        return None

    r = 0.08
    for _ in range(80):
        denom = (1.0 + r) ** times
        f = float(np.sum(amts / denom))
        df = float(np.sum(-times * amts / ((1.0 + r) ** (times + 1.0))))
        if abs(df) < 1e-12:
            break
        nxt = r - f / df
        if nxt <= -0.99:
            nxt = -0.5
        if abs(nxt - r) < 1e-7:
            return float(nxt)
        r = nxt
    return None


def get_betashares_positions(
    db_path: Path = DB_PATH,
) -> tuple[dict[str, dict], list[dict]]:
    conn = _connect_db(db_path)
    rows = conn.execute(
        "SELECT * FROM trades ORDER BY trade_date ASC, id ASC"
    ).fetchall()
    conn.close()

    lots: dict[str, list[dict]] = {}
    all_trades = []
    for r in rows:
        t = dict(r)
        all_trades.append(t)
        sym = t["ticker"]
        lots.setdefault(sym, [])
        if t["side"] == "BUY":
            lots[sym].append(
                {
                    "trade_date": t["trade_date"],
                    "quantity": float(t["quantity"]),
                    "cost_aud": float(t["total_aud"]) + float(t["brokerage_aud"]),
                }
            )
        else:
            # FIFO reduction on sells
            rem = float(t["quantity"])
            while rem > 1e-9 and lots[sym]:
                lot = lots[sym][0]
                if lot["quantity"] <= rem + 1e-9:
                    rem -= lot["quantity"]
                    lots[sym].pop(0)
                else:
                    frac = rem / lot["quantity"]
                    lot["quantity"] -= rem
                    lot["cost_aud"] *= 1.0 - frac
                    rem = 0.0

    today = date.today()
    positions: dict[str, dict[str, Any]] = {}
    for sym, sym_lots in lots.items():
        qty = sum(lot["quantity"] for lot in sym_lots)
        if qty <= 1e-6:
            continue
        cost = sum(lot["cost_aud"] for lot in sym_lots)
        lt_qty = sum(
            lot["quantity"]
            for lot in sym_lots
            if (today - date.fromisoformat(lot["trade_date"])).days >= 365
        )
        positions[sym] = {
            "units": qty,
            "cost_basis_aud": cost,
            "lt_cgt_eligible_units": lt_qty,
            "lt_cgt_eligible_pct": (lt_qty / qty * 100.0) if qty > 0 else 0.0,
            "earliest_lot_date": sym_lots[0]["trade_date"],
        }
    return positions, all_trades


def get_live_portfolio(sync_gmail: bool = True, db_path: Path = DB_PATH) -> dict:
    """Assemble live portfolio across Betashares Direct (CSV + Gmail) and Swyftx API."""
    sync_info = None
    if sync_gmail:
        try:
            sync_info = sync_betashares_gmail(db_path=db_path)
        except (
            subprocess.SubprocessError,
            OSError,
            RuntimeError,
            json.JSONDecodeError,
        ) as e:
            sync_info = {"error": str(e)}

    bs_positions, bs_trades = get_betashares_positions(db_path)
    swyftx = fetch_swyftx_live()

    all_tickers = sorted(
        set(OPTIMAL_ALLOCATION.keys()) | set(bs_positions.keys()) | {"BTC"}
    )
    prices = fetch_live_prices(all_tickers)

    today = date.today()
    holdings: list[dict[str, Any]] = []
    total_value = 0.0
    total_cost = 0.0

    for ticker in all_tickers:
        px = prices.get(ticker, 0.0)
        if ticker == "BTC":
            units = swyftx["btc_units"]
            cost = swyftx["btc_cost_basis_aud"]
            earliest = (
                swyftx["btc_buys"][0]["trade_date"] if swyftx["btc_buys"] else None
            )
            lt_pct = (
                100.0
                if earliest and (today - date.fromisoformat(earliest)).days >= 365
                else 0.0
            )
            platform = "Swyftx"
        else:
            pos = bs_positions.get(ticker, {})
            units = pos.get("units", 0.0)
            cost = pos.get("cost_basis_aud", 0.0)
            earliest = pos.get("earliest_lot_date")
            lt_pct = pos.get("lt_cgt_eligible_pct", 0.0)
            platform = "Betashares Direct"

        val = units * px
        total_value += val
        total_cost += cost
        gain = val - cost
        gain_pct = (gain / cost * 100.0) if cost > 0 else 0.0

        holdings.append(
            {
                "ticker": ticker,
                "platform": platform,
                "units": round(units, 8 if ticker == "BTC" else 4),
                "price_aud": round(px, 2),
                "value_aud": round(val, 2),
                "cost_basis_aud": round(cost, 2),
                "unrealized_pnl_aud": round(gain, 2),
                "unrealized_pnl_pct": round(gain_pct, 2),
                "target_weight_pct": round(
                    OPTIMAL_ALLOCATION.get(ticker, 0.0) * 100.0, 2
                ),
                "lt_cgt_eligible_pct": round(lt_pct, 1),
                "earliest_lot_date": earliest,
            }
        )

    for h in holdings:
        cur_w = (h["value_aud"] / total_value * 100.0) if total_value > 0 else 0.0
        h["current_weight_pct"] = round(cur_w, 2)
        h["drift_pct"] = round(cur_w - h["target_weight_pct"], 2)

    # Cashflows for XIRR and 100% BGBL benchmark comparison
    cashflows: list[tuple[date, float]] = []
    for t in bs_trades:
        dt = date.fromisoformat(t["trade_date"])
        amt = -float(t["total_aud"]) if t["side"] == "BUY" else float(t["total_aud"])
        cashflows.append((dt, amt))
    for b in swyftx["btc_buys"]:
        cashflows.append((date.fromisoformat(b["trade_date"]), -float(b["total_aud"])))
    cashflows.append((today, total_value))

    xirr_val = _xirr(cashflows)
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100.0) if total_cost > 0 else 0.0

    # Benchmark: what if all exact same AUD cashflows bought 100% BGBL on those dates?
    bgbl_bench_val = None
    if cashflows[:-1] and "BGBL" in prices:
        min_dt = min(d for d, _ in cashflows[:-1])
        bgbl_hist = yf.Ticker("BGBL.AX").history(start=min_dt.isoformat())
        if not bgbl_hist.empty:
            hist_dates = [d.date() for d in bgbl_hist.index]
            hist_closes = bgbl_hist["Close"].astype(float).tolist()
            bench_units = 0.0
            for d, amt in cashflows[:-1]:
                # Find nearest trading day on or before/after d
                idx = min(
                    range(len(hist_dates)), key=lambda i: abs((hist_dates[i] - d).days)
                )
                bench_units += (-amt) / hist_closes[idx]
            bgbl_bench_val = round(bench_units * prices["BGBL"], 2)

    return {
        "as_of": datetime.now(UTC).isoformat(),
        "target_allocation": OPTIMAL_ALLOCATION,
        "summary": {
            "total_value_aud": round(total_value, 2),
            "total_cost_basis_aud": round(total_cost, 2),
            "unrealized_pnl_aud": round(total_pnl, 2),
            "unrealized_pnl_pct": round(total_pnl_pct, 2),
            "xirr_annualized_pct": round(xirr_val * 100.0, 2)
            if xirr_val is not None
            else None,
            "benchmark_100pct_bgbl_value_aud": bgbl_bench_val,
            "vs_bgbl_benchmark_aud": round(total_value - bgbl_bench_val, 2)
            if bgbl_bench_val is not None
            else None,
            "swyftx_aud_cash": round(swyftx["aud_cash"], 2),
        },
        "holdings": holdings,
        "gmail_sync": sync_info,
    }


def get_next_dca_plan(
    amount: float = FORTNIGHTLY_DCA,
    sync_gmail: bool = True,
    db_path: Path = DB_PATH,
    portfolio: dict | None = None,
) -> dict:
    """Compute exact zero-sell cashflow-directed allocation for the next DCA deposit."""
    port = (
        portfolio
        if portfolio is not None
        else get_live_portfolio(sync_gmail=sync_gmail, db_path=db_path)
    )
    tot = port["summary"]["total_value_aud"]
    post_tot = tot + amount

    by_ticker = {h["ticker"]: h for h in port["holdings"]}
    deficits = {}
    for ticker, tw in OPTIMAL_ALLOCATION.items():
        cur_val = by_ticker.get(ticker, {}).get("value_aud", 0.0)
        deficits[ticker] = max(post_tot * tw - cur_val, 0.0)

    def_sum = sum(deficits.values())
    buys = {}
    post_weights = {}
    for ticker, tw in OPTIMAL_ALLOCATION.items():
        alloc = (amount * deficits[ticker] / def_sum) if def_sum > 0 else amount * tw
        buys[ticker] = round(alloc, 2)
        cur_val = by_ticker.get(ticker, {}).get("value_aud", 0.0)
        post_weights[ticker] = round((cur_val + alloc) / post_tot * 100.0, 2)

    # Estimate how many fortnights until the most overweight asset reaches target without selling
    max_over_tot = max(
        (
            by_ticker.get(t, {}).get("value_aud", 0.0) / tw
            for t, tw in OPTIMAL_ALLOCATION.items()
            if tw > 0
        ),
        default=tot,
    )
    cash_to_balance = max(max_over_tot - tot, 0.0)
    est_fortnights = round(cash_to_balance / amount, 1) if amount > 0 else 0.0

    sell_alerts = [
        {
            "ticker": h["ticker"],
            "drift_pct": h["drift_pct"],
            "note": "Exceeds +5% tolerance band (consider selling >12m lots only if cashflow cannot absorb).",
        }
        for h in port["holdings"]
        if h["drift_pct"] >= 5.0 and h["ticker"] != "BGBL"
    ]

    return {
        "deposit_amount_aud": amount,
        "current_total_aud": tot,
        "post_deposit_total_aud": round(post_tot, 2),
        "recommended_buys_aud": buys,
        "post_buy_weights_pct": post_weights,
        "estimated_cash_to_full_target_aud": round(cash_to_balance, 2),
        "estimated_fortnights_to_target": est_fortnights,
        "sell_band_alerts": sell_alerts,
    }


def get_trade_history(ticker: str | None = None, db_path: Path = DB_PATH) -> list[dict]:
    conn = _connect_db(db_path)
    if ticker:
        rows = conn.execute(
            "SELECT * FROM trades WHERE ticker = ? ORDER BY trade_date DESC, id DESC",
            (ticker.upper(),),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY trade_date DESC, id DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    print(json.dumps(get_live_portfolio(sync_gmail=True), indent=2))
