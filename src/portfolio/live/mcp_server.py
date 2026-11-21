from mcp.server.mcpserver import MCPServer

from portfolio.assumptions import FORTNIGHTLY_DCA
from portfolio.live.tracker import (
    get_live_portfolio as _get_live_portfolio,
)
from portfolio.live.tracker import (
    get_next_dca_plan as _get_next_dca_plan,
)
from portfolio.live.tracker import (
    get_trade_history as _get_trade_history,
)
from portfolio.live.tracker import (
    import_betashares_csv as _import_betashares_csv,
)
from portfolio.live.tracker import (
    sync_betashares_gmail as _sync_betashares_gmail,
)

mcp = MCPServer(
    "portfolio",
    instructions=(
        "Read-only personal investment portfolio tracker combining Betashares Direct "
        "(local SQLite ledger auto-synced via gws gmail) and Swyftx (live read-only API "
        "using macOS Keychain) with live ASX/BTC prices via yfinance."
    ),
)


@mcp.tool()
def get_live_portfolio(sync_gmail: bool = True) -> dict:
    """Return live portfolio balances, weights, drift vs 70/15/12/3 target, unrealized P&L, XIRR, and 100% BGBL benchmark comparison."""
    return _get_live_portfolio(sync_gmail=sync_gmail)


@mcp.tool()
def get_next_dca_plan(amount: float = FORTNIGHTLY_DCA, sync_gmail: bool = True) -> dict:
    """Compute exact zero-sell cashflow-directed buy splits for the next fortnightly DCA deposit toward the 70/15/12/3 target."""
    return _get_next_dca_plan(amount=amount, sync_gmail=sync_gmail)


@mcp.tool()
def sync_betashares_emails() -> dict:
    """Sync new Betashares Direct order fill confirmation emails from Gmail via gws CLI into the local ledger."""
    return _sync_betashares_gmail()


@mcp.tool()
def import_betashares_csv(csv_path: str) -> dict:
    """Import a Betashares Direct transaction history CSV export into the local SQLite ledger (deduplicates existing trades)."""
    return _import_betashares_csv(csv_path=csv_path)


@mcp.tool()
def get_trade_history(ticker: str | None = None) -> list[dict]:
    """List recorded Betashares Direct trades in the local ledger, optionally filtered by ASX ticker (e.g. BGBL, BEMG, PMGOLD)."""
    return _get_trade_history(ticker=ticker)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
