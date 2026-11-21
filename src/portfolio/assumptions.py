import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_DIR / "data"
CHARTS_DIR = DATA_DIR / "charts"
MARKET_DATA_PATH = DATA_DIR / "market_data.parquet"
LEDGER_DB_PATH = DATA_DIR / "portfolio_ledger.db"
PROFILE_PATH = DATA_DIR / "profile.json"


def _load_profile() -> dict[str, Any]:
    if PROFILE_PATH.exists():
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    return {}


_PROFILE = _load_profile()

# Default illustrative values; override locally in gitignored data/profile.json
STARTING_CAPITAL = float(_PROFILE.get("starting_capital", 100_000.0))
FORTNIGHTLY_DCA = float(_PROFILE.get("fortnightly_dca", 1_000.0))
ANNUAL_DCA = FORTNIGHTLY_DCA * 26
EMERGENCY_CASH = float(_PROFILE.get("emergency_cash", 30_000.0))

MARGINAL_TAX_RATE = float(_PROFILE.get("marginal_tax_rate", 0.39))  # 37% + 2% Medicare
CGT_DISCOUNT = 0.50
LONG_TERM_CGT_RATE = MARGINAL_TAX_RATE * (1.0 - CGT_DISCOUNT)  # 19.5%
INFLATION_RATE = 0.025  # RBA target midpoint

HISA_GROSS_RATE = float(_PROFILE.get("hisa_gross_rate", 0.05))
HISA_NET_NOMINAL = HISA_GROSS_RATE * (1.0 - MARGINAL_TAX_RATE)  # 3.05%
HISA_NET_REAL = (1.0 + HISA_NET_NOMINAL) / (1.0 + INFLATION_RATE) - 1.0
LEDGER_START_DATE = str(_PROFILE.get("ledger_start_date", "2020-01-01"))

# Ex-Australia candidate universe (AU exposure covered by salary, HISA cash, and future AU property)
ASSET_ORDER = ["BGBL", "HGBL", "BEMG", "VAF", "PMGOLD", "BTC"]

# Placeholder defaults updated after running optimize_allocation.py
OPTIMAL_ALLOCATION = {
    "BGBL": 0.70,
    "BEMG": 0.15,
    "PMGOLD": 0.12,
    "BTC": 0.03,
}
NO_CRYPTO_OPTIMAL = {
    "BGBL": 0.75,
    "BEMG": 0.15,
    "PMGOLD": 0.10,
}
ORIGINAL_ALLOCATION = {
    "BGBL": 0.85,
    "PMGOLD": 0.10,
    "BTC": 0.05,
}


@dataclass(frozen=True)
class AssetSpec:
    code: str
    name: str
    ticker: str
    proxy_ticker: str | None
    min_actual_date: str | None
    mer: float
    # Forward-looking gross nominal arithmetic return (net of MER & unrecoverable internal WHT)
    cma_gross_nominal: float
    # Expected annual taxable AU cash distribution yield
    dist_yield: float
    # Fraction of cash distribution that is 50%-discounted capital gain
    cg_discount_frac: float = 0.0
    # Foreign income tax offset as fraction of cash distribution (grossed up)
    fito_frac_of_cash: float = 0.0
    # One-way trading fee + estimated bid-ask spread
    trade_cost: float = 0.0005

    @property
    def annual_tax_drag(self) -> float:
        """Net annual Australian income tax drag relative to untaxed total return."""
        if self.dist_yield <= 0.0:
            return 0.0
        cg_part = self.dist_yield * self.cg_discount_frac
        inc_part = self.dist_yield - cg_part
        fito = inc_part * self.fito_frac_of_cash
        gross_inc = inc_part + fito
        net_inc_tax = gross_inc * MARGINAL_TAX_RATE - fito
        return net_inc_tax + cg_part * LONG_TERM_CGT_RATE

    @property
    def cma_net_nominal(self) -> float:
        """Expected ongoing after-tax nominal return (before eventual exit CGT)."""
        return self.cma_gross_nominal - self.annual_tax_drag

    @property
    def cma_net_real(self) -> float:
        return (1.0 + self.cma_net_nominal) / (1.0 + INFLATION_RATE) - 1.0


ASSETS: dict[str, AssetSpec] = {
    "BGBL": AssetSpec(
        code="BGBL",
        name="Global Developed Equities (Unhedged, ex-AU)",
        ticker="BGBL.AX",
        proxy_ticker="VGS.AX",
        min_actual_date="2023-06-01",
        mer=0.0008,
        cma_gross_nominal=0.080,
        dist_yield=0.022,
        cg_discount_frac=0.20,
        fito_frac_of_cash=0.14,
        trade_cost=0.0005,
    ),
    "HGBL": AssetSpec(
        code="HGBL",
        name="Global Developed Equities (AUD Hedged, ex-AU)",
        ticker="HGBL.AX",
        proxy_ticker="VGAD.AX",
        min_actual_date="2023-06-01",
        mer=0.0011,
        cma_gross_nominal=0.079,
        dist_yield=0.032,  # Higher distributions from TOFA FX forward gains
        cg_discount_frac=0.10,
        fito_frac_of_cash=0.09,
        trade_cost=0.0008,
    ),
    "BEMG": AssetSpec(
        code="BEMG",
        name="Emerging Markets Equities (MSCI EM, Accumulating Swap)",
        ticker="BEMG.AX",
        proxy_ticker="IEM.AX",  # Exact MSCI EM index match (incl. South Korea)
        min_actual_date="2026-01-01",
        mer=0.0035,
        # ~8.4% gross EM return minus ~0.4% unrecoverable internal WHT inside MSCI NTR swap
        cma_gross_nominal=0.080,
        dist_yield=0.0,  # Non-distributing accumulating UCITS swap structure (0% ongoing AU tax drag)
        trade_cost=0.0015,  # Slightly wider spread for newer ASX complex ETF
    ),
    "VAF": AssetSpec(
        code="VAF",
        name="Australian Investment Grade Bonds",
        ticker="VAF.AX",
        proxy_ticker=None,
        min_actual_date=None,
        mer=0.0010,
        cma_gross_nominal=0.045,
        dist_yield=0.040,
        cg_discount_frac=0.0,
        trade_cost=0.0005,
    ),
    "PMGOLD": AssetSpec(
        code="PMGOLD",
        name="Physical Gold (AUD)",
        ticker="PMGOLD.AX",
        proxy_ticker="GOLD.AX",
        min_actual_date="2022-01-01",
        mer=0.0015,
        cma_gross_nominal=0.040,  # ~1.5% real above 2.5% CPI
        dist_yield=0.0,
        trade_cost=0.0008,
    ),
    "BTC": AssetSpec(
        code="BTC",
        name="Bitcoin (AUD)",
        ticker="BTC-AUD",
        proxy_ticker=None,
        min_actual_date=None,
        mer=0.0,
        cma_gross_nominal=0.100,  # Base-case moderated forward arithmetic return
        dist_yield=0.0,
        trade_cost=0.0080,  # Swyftx 0.60% fee + ~0.20% spread
    ),
}
