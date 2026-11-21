import argparse
import json

import polars as pl

from portfolio.assumptions import FORTNIGHTLY_DCA
from portfolio.live.tracker import (
    get_live_portfolio,
    get_next_dca_plan,
    get_trade_history,
    import_betashares_csv,
    sync_betashares_gmail,
)


def _fmt_signed_aud(val: float) -> str:
    sign = "+" if val >= 0 else "-"
    return f"{sign}${abs(val):,.2f}"


def _fmt_signed_pct(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}%"


def _print_portfolio(port: dict) -> None:
    s = port["summary"]
    sync = port.get("gmail_sync") or {}

    pl.Config.set_tbl_cols(12)
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(160)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)

    print(f"\n=== Live Portfolio ({port['as_of'][:19]} UTC) ===")
    rows = []
    for h in sorted(port["holdings"], key=lambda x: -x["value_aud"]):
        units_str = f"{h['units']:.8f}" if h["ticker"] == "BTC" else f"{h['units']:.4f}"
        rows.append(
            {
                "Asset": h["ticker"],
                "Platform": h["platform"],
                "Units": units_str,
                "Price": f"${h['price_aud']:,.2f}",
                "Value": f"${h['value_aud']:,.2f}",
                "Cost": f"${h['cost_basis_aud']:,.2f}",
                "P&L": f"{_fmt_signed_aud(h['unrealized_pnl_aud'])} ({_fmt_signed_pct(h['unrealized_pnl_pct'])})",
                "Weight": f"{h['current_weight_pct']:.2f}%",
                "Target": f"{h['target_weight_pct']:.1f}%",
                "Drift": _fmt_signed_pct(h["drift_pct"]),
            }
        )
    print(pl.DataFrame(rows))

    print(
        f"Total Value : ${s['total_value_aud']:,.2f} AUD  |  "
        f"Cost Basis: ${s['total_cost_basis_aud']:,.2f} AUD  |  "
        f"Unrealized P&L: {_fmt_signed_aud(s['unrealized_pnl_aud'])} ({_fmt_signed_pct(s['unrealized_pnl_pct'])})"
    )
    if s["benchmark_100pct_bgbl_value_aud"] is not None:
        print(
            f"100% BGBL   : ${s['benchmark_100pct_bgbl_value_aud']:,.2f} AUD  |  "
            f"Relative vs BGBL: {_fmt_signed_aud(s['vs_bgbl_benchmark_aud'])}"
        )
    if sync:
        if "error" in sync:
            print(f"Gmail Sync  : Warning ({sync['error']})")
        else:
            print(
                f"Gmail Sync  : {sync.get('emails_found', 0)} emails checked, "
                f"{sync.get('new_trades_inserted', 0)} new trades added"
            )


def _print_dca_plan(plan: dict, holdings: list[dict] | None = None) -> None:
    prices = {h["ticker"]: h["price_aud"] for h in (holdings or [])}
    print(
        f"\n=== Suggested Next Deposit Split (${plan['deposit_amount_aud']:,.2f} AUD, Zero-Sell Rebalancing) ==="
    )
    rows = []
    for ticker, buy_aud in plan["recommended_buys_aud"].items():
        px = prices.get(ticker, 0.0)
        est_units = (buy_aud / px) if (px > 0 and buy_aud > 0) else 0.0
        unit_str = (
            f"~{est_units:.6f}"
            if ticker == "BTC" and est_units > 0
            else (f"~{est_units:.2f}" if est_units > 0 else "-")
        )
        rows.append(
            {
                "Asset": ticker,
                "Buy (AUD)": f"${buy_aud:,.2f}",
                "Est Units": unit_str,
                "Post-Buy Weight": f"{plan['post_buy_weights_pct'][ticker]:.2f}%",
            }
        )
    print(pl.DataFrame(rows))
    print(
        f"Post-Deposit Total: ${plan['post_deposit_total_aud']:,.2f} AUD  |  "
        f"Est. To Full Target: ~{plan['estimated_fortnights_to_target']:.1f} fortnights "
        f"(${plan['estimated_cash_to_full_target_aud']:,.2f})"
    )
    for alert in plan.get("sell_band_alerts", []):
        print(
            f"ALERT [{alert['ticker']}]: drift {_fmt_signed_pct(alert['drift_pct'])} - {alert['note']}"
        )


def _print_trades(trades: list[dict]) -> None:
    if not trades:
        print("No trades recorded in local ledger.")
        return
    pl.Config.set_tbl_cols(10)
    pl.Config.set_tbl_rows(50)
    pl.Config.set_tbl_width_chars(150)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    pl.Config.set_tbl_hide_column_data_types(True)
    rows = [
        {
            "Date": t["trade_date"],
            "Side": t["side"],
            "Asset": t["ticker"],
            "Quantity": f"{t['quantity']:.4f}",
            "Price": f"${t['price_aud']:,.2f}",
            "Total (AUD)": f"${t['total_aud']:,.2f}",
            "Source": t["source"],
        }
        for t in trades
    ]
    print("\n=== Betashares Direct Trade History ===")
    print(pl.DataFrame(rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="portfolio",
        description="Live portfolio viewer and zero-sell DCA planner (Betashares Direct + Swyftx).",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=FORTNIGHTLY_DCA,
        help=f"Deposit amount in AUD for DCA plan (default: {FORTNIGHTLY_DCA:.0f}).",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip Gmail sync before fetching live balances.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output raw JSON instead of formatted tables.",
    )

    sub = parser.add_subparsers(dest="command")
    sub.add_parser(
        "status",
        help="Show live portfolio summary and suggested next deposit split (default).",
    )

    p_dca = sub.add_parser("dca", help="Show only the suggested next deposit split.")
    p_dca.add_argument(
        "--amount", type=float, default=None, help="Deposit amount in AUD."
    )

    p_trades = sub.add_parser("trades", help="Show recorded Betashares Direct trades.")
    p_trades.add_argument(
        "--ticker", type=str, default=None, help="Filter by ticker (e.g. BGBL)."
    )

    p_imp = sub.add_parser(
        "import-csv", help="Import a Betashares Direct activity CSV export."
    )
    p_imp.add_argument("csv_path", type=str, help="Path to Betashares CSV file.")

    sub.add_parser(
        "sync", help="Sync new Betashares Direct order emails from Gmail via gws."
    )

    args = parser.parse_args()
    cmd = args.command or "status"
    sync_gmail = not args.no_sync

    if cmd == "sync":
        res = sync_betashares_gmail()
        print(json.dumps(res, indent=2))
        return

    if cmd == "import-csv":
        res = import_betashares_csv(args.csv_path)
        print(json.dumps(res, indent=2))
        return

    if cmd == "trades":
        trades = get_trade_history(ticker=args.ticker)
        if args.as_json:
            print(json.dumps(trades, indent=2))
        else:
            _print_trades(trades)
        return

    if cmd == "dca":
        amt = args.amount if args.amount is not None else FORTNIGHTLY_DCA
        port = get_live_portfolio(sync_gmail=sync_gmail)
        plan = get_next_dca_plan(amount=amt, sync_gmail=False, portfolio=port)
        if args.as_json:
            print(json.dumps(plan, indent=2))
        else:
            _print_dca_plan(plan, port["holdings"])
        return

    # Default: status (live portfolio + suggested next deposit split)
    port = get_live_portfolio(sync_gmail=sync_gmail)
    plan = get_next_dca_plan(amount=args.amount, sync_gmail=False, portfolio=port)
    if args.as_json:
        print(json.dumps({"portfolio": port, "dca_plan": plan}, indent=2))
    else:
        _print_portfolio(port)
        _print_dca_plan(plan, port["holdings"])


if __name__ == "__main__":
    main()
