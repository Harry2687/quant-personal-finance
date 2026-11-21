import json
from dataclasses import dataclass
from datetime import date
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
INFLATION_RATE = 0.025  # RBA target midpoint

# Treasury Laws Amendment (Tax Reform No. 1) Act 2026 (effective 1 July 2027):
# Replaces the 50% CGT discount with CPI cost-base indexation and a 30% minimum tax floor.
CGT_REFORM_DATE = date(2027, 7, 1)
CGT_MIN_TAX_RATE = 0.30
CGT_REAL_GAIN_RATE = max(CGT_MIN_TAX_RATE, MARGINAL_TAX_RATE)  # 39% at 39% MTR
# Legacy 50% discount applies only to gains accrued prior to 1 July 2027 (held >12 months)
LEGACY_CGT_DISCOUNT = 0.50
LEGACY_LONG_TERM_CGT_RATE = MARGINAL_TAX_RATE * (1.0 - LEGACY_CGT_DISCOUNT)
# Approx real-gain share of AMIT ETF capital gain distributions after CPI indexation
AMIT_CG_REAL_SHARE = 0.75

HISA_GROSS_RATE = float(_PROFILE.get("hisa_gross_rate", 0.05))
HISA_NET_NOMINAL = HISA_GROSS_RATE * (1.0 - MARGINAL_TAX_RATE)  # 3.05%
HISA_NET_REAL = (1.0 + HISA_NET_NOMINAL) / (1.0 + INFLATION_RATE) - 1.0
LEDGER_START_DATE = str(_PROFILE.get("ledger_start_date", "2020-01-01"))


def calculate_indexed_cgt(
    proceeds: float,
    indexed_cost_base: float,
    tax_rate: float = CGT_REAL_GAIN_RATE,
) -> float:
    """Compute post-1 July 2027 Australian CGT on real indexed capital gains (gains only)."""
    real_gain = max(proceeds - indexed_cost_base, 0.0)
    return real_gain * tax_rate


# Ex-Australia candidate universe (AU exposure covered by salary, HISA cash, and future AU property)
ASSET_ORDER = ["BGBL", "HGBL", "BEMG", "VAF", "PMGOLD", "BTC"]
# Emergency taxable sell threshold when DCA inflows alone cannot absorb drift under 39% real CGT
REBALANCE_SELL_BAND = 0.10

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
    # Fraction of cash distribution that is realized AMIT capital gain (indexed post-1 July 2027)
    cg_dist_frac: float = 0.0
    # Foreign income tax offset as fraction of cash distribution (grossed up)
    fito_frac_of_cash: float = 0.0
    # One-way trading fee + estimated bid-ask spread
    trade_cost: float = 0.0005

    @property
    def annual_tax_drag(self) -> float:
        """Net annual Australian income tax drag relative to untaxed total return."""
        if self.dist_yield <= 0.0:
            return 0.0
        cg_part = self.dist_yield * self.cg_dist_frac
        inc_part = self.dist_yield - cg_part
        fito = inc_part * self.fito_frac_of_cash
        gross_inc = inc_part + fito
        net_inc_tax = gross_inc * MARGINAL_TAX_RATE - fito
        # Post-1 July 2027: AMIT real indexed capital gains taxed at max(30%, MTR)
        return net_inc_tax + cg_part * AMIT_CG_REAL_SHARE * CGT_REAL_GAIN_RATE

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
        cma_gross_nominal=0.090,  # ~8.2% gross / ~7.5% net geometric nominal (~4.9% net real)
        dist_yield=0.022,
        cg_dist_frac=0.20,
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
        cma_gross_nominal=0.089,  # Same equity ERP as BGBL minus higher MER & FX hedge friction
        dist_yield=0.032,  # Higher distributions from TOFA FX forward gains
        cg_dist_frac=0.10,
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
        # ~9.6% gross EM arith minus ~0.4% unrecoverable internal WHT inside MSCI NTR swap (~8.0% net geo)
        cma_gross_nominal=0.092,
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
        cma_gross_nominal=0.048,
        dist_yield=0.040,
        cg_dist_frac=0.0,
        trade_cost=0.0005,
    ),
    "PMGOLD": AssetSpec(
        code="PMGOLD",
        name="Physical Gold (AUD)",
        ticker="PMGOLD.AX",
        proxy_ticker="GOLD.AX",
        min_actual_date="2022-01-01",
        mer=0.0015,
        cma_gross_nominal=0.050,  # ~3.9% geometric nominal (~1.4% real compound above 2.5% CPI)
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
        cma_gross_nominal=0.120,  # Moderated forward arithmetic return (equity ERP + 3% risk premium)
        dist_yield=0.0,
        trade_cost=0.0080,  # Swyftx 0.60% fee + ~0.20% spread
    ),
}
