"""Risk protection for rebate/maker strategies.

Rebate-harvesting strategies have one structural failure mode: adverse
selection. When news breaks and price moves against you, takers eat your
resting bids first — you earn a few cents of rebate but take a much larger
loss on the inventory.

This module enforces:

1. **Per-market inventory cap** — never hold more than `POLY_MAX_POSITION_PER_MARKET_USD`
   worth of shares on any single outcome token. Default $50.
2. **Per-sleeve aggregate cap** — never let one sleeve's total open exposure
   exceed a fraction of bankroll. Default 10%.
3. **Global open-position cap** — never let ALL sleeves combined exceed 50%
   of bankroll in open inventory.
4. **Net rebate P&L** — computes (rebates earned) − (MTM loss on inventory)
   so the dashboard shows TRUE economics, not gross rebate.

All caps are HARD — `check_pre_trade()` returns False if any would be violated
by a proposed intent, and the intent is dropped before paper fill simulation.

Reads from FillRow × current Market.params_json.last_mid to compute live MTM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select

from .db.models import FillRow, Market, OrderIntentRow
from .db.session import SessionLocal

log = structlog.get_logger()

BANKROLL_USD = Decimal(os.environ.get("POLY_BANKROLL_USD", "1000"))
MAX_POSITION_PER_MARKET_USD = Decimal(
    os.environ.get("POLY_MAX_POSITION_PER_MARKET_USD", "50")
)
MAX_SLEEVE_EXPOSURE_FRACTION = Decimal(
    os.environ.get("POLY_MAX_SLEEVE_EXPOSURE_FRACTION", "0.10")
)
MAX_GLOBAL_EXPOSURE_FRACTION = Decimal(
    os.environ.get("POLY_MAX_GLOBAL_EXPOSURE_FRACTION", "0.50")
)
RISK_ENABLED = os.environ.get("POLY_RISK_CHECKS_ENABLED", "1") not in ("0", "false", "no")


@dataclass(frozen=True)
class RiskDecision:
    allow: bool
    reason: str | None
    current_market_exposure_usd: Decimal
    current_sleeve_exposure_usd: Decimal
    current_global_exposure_usd: Decimal


async def _open_exposure_usd(
    market_cond: Optional[str] = None,
    sleeve_id: Optional[str] = None,
) -> Decimal:
    """Sum of notional on non-rejected fills for markets still open.

    A fill is "open inventory" if:
      - it was NOT rejected
      - the market's params_json.closed is falsy / missing
    """
    async with SessionLocal() as db:
        # Get all non-rejected fills joined with their intent and market.
        q = (
            select(FillRow, OrderIntentRow, Market)
            .join(OrderIntentRow, FillRow.client_order_id == OrderIntentRow.client_order_id)
            .outerjoin(Market, Market.condition_id == OrderIntentRow.market_condition_id)
            .where(FillRow.rejected == False)  # noqa: E712
        )
        if market_cond is not None:
            q = q.where(OrderIntentRow.market_condition_id == market_cond)
        if sleeve_id is not None:
            q = q.where(OrderIntentRow.sleeve_id == sleeve_id)
        rows = (await db.execute(q)).all()
    total = Decimal("0")
    for fill, _intent, market in rows:
        params = (market.params_json if market else None) or {}
        if params.get("closed"):
            continue
        try:
            total += Decimal(fill.notional_usd or "0")
        except Exception:
            continue
    return total


async def check_pre_trade(
    *,
    market_condition_id: str,
    sleeve_id: str,
    proposed_notional_usd: Decimal,
) -> RiskDecision:
    """Hard pre-trade risk check. Returns (allow, reason)."""
    if not RISK_ENABLED:
        return RiskDecision(True, None, Decimal("0"), Decimal("0"), Decimal("0"))

    mkt_exp = await _open_exposure_usd(market_cond=market_condition_id)
    sleeve_exp = await _open_exposure_usd(sleeve_id=sleeve_id)
    global_exp = await _open_exposure_usd()

    # Check each cap with the *proposed* trade added.
    new_mkt = mkt_exp + proposed_notional_usd
    new_sleeve = sleeve_exp + proposed_notional_usd
    new_global = global_exp + proposed_notional_usd

    sleeve_cap = BANKROLL_USD * MAX_SLEEVE_EXPOSURE_FRACTION
    global_cap = BANKROLL_USD * MAX_GLOBAL_EXPOSURE_FRACTION

    if new_mkt > MAX_POSITION_PER_MARKET_USD:
        return RiskDecision(
            False,
            f"per-market cap: {new_mkt} > {MAX_POSITION_PER_MARKET_USD}",
            mkt_exp, sleeve_exp, global_exp,
        )
    if new_sleeve > sleeve_cap:
        return RiskDecision(
            False,
            f"per-sleeve cap: {new_sleeve} > {sleeve_cap}",
            mkt_exp, sleeve_exp, global_exp,
        )
    if new_global > global_cap:
        return RiskDecision(
            False,
            f"global exposure cap: {new_global} > {global_cap}",
            mkt_exp, sleeve_exp, global_exp,
        )
    return RiskDecision(True, None, mkt_exp, sleeve_exp, global_exp)


# ---------------------------------------------------------------------------
# Net rebate P&L  (rebates earned MINUS MTM loss on open inventory)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NetRebateReport:
    gross_rebates_usd: float        # positive = rebates earned (negative fees)
    open_inventory_notional_usd: float
    open_inventory_mtm_pnl_usd: float  # negative = we're underwater on holdings
    net_rebate_pnl_usd: float       # gross_rebates + mtm (both signed)
    open_position_count: int


async def compute_net_rebate_pnl() -> NetRebateReport:
    """Rebates are only real profit if the inventory they created doesn't lose more.

    gross_rebates   = SUM(-fees_usd) across non-rejected fills  (fees are negative for makers)
    mtm_pnl         = SUM((last_mid - fill_price) * shares) across open positions
    net_rebate_pnl  = gross_rebates + mtm_pnl
    """
    gross_rebates = 0.0
    mtm_pnl = 0.0
    open_notional = 0.0
    open_count = 0
    async with SessionLocal() as db:
        rows = (await db.execute(
            select(FillRow, OrderIntentRow, Market)
            .join(OrderIntentRow, FillRow.client_order_id == OrderIntentRow.client_order_id)
            .outerjoin(Market, Market.condition_id == OrderIntentRow.market_condition_id)
            .where(FillRow.rejected == False)  # noqa: E712
        )).all()
    for fill, intent, market in rows:
        try:
            fees = float(fill.fees_usd or "0")
            gross_rebates += -fees  # rebates are negative fees
        except (TypeError, ValueError):
            pass
        params = (market.params_json if market else None) or {}
        if params.get("closed"):
            continue
        try:
            price = float(fill.avg_price or "0")
            shares = float(fill.filled_size_shares or "0")
        except (TypeError, ValueError):
            continue
        if price == 0 or shares == 0:
            continue
        open_notional += price * shares
        open_count += 1
        last_mid = params.get("last_mid")
        if last_mid is None:
            continue
        try:
            mid = float(last_mid)
        except (TypeError, ValueError):
            continue
        # For BUY, PnL = (current_mid - entry_price) * shares.
        mtm_pnl += (mid - price) * shares
    return NetRebateReport(
        gross_rebates_usd=round(gross_rebates, 4),
        open_inventory_notional_usd=round(open_notional, 2),
        open_inventory_mtm_pnl_usd=round(mtm_pnl, 4),
        net_rebate_pnl_usd=round(gross_rebates + mtm_pnl, 4),
        open_position_count=open_count,
    )


# ---------------------------------------------------------------------------
# Inventory snapshot for dashboard
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InventoryPosition:
    market_condition_id: str
    sleeve_id: str
    side: str
    shares: float
    entry_price: float
    cost_usd: float
    last_mid: Optional[float]
    mtm_pnl_usd: Optional[float]


async def current_inventory() -> list[InventoryPosition]:
    """Net open inventory aggregated per (market, sleeve, side)."""
    positions: dict[tuple[str, str, str], dict] = {}
    async with SessionLocal() as db:
        rows = (await db.execute(
            select(FillRow, OrderIntentRow, Market)
            .join(OrderIntentRow, FillRow.client_order_id == OrderIntentRow.client_order_id)
            .outerjoin(Market, Market.condition_id == OrderIntentRow.market_condition_id)
            .where(FillRow.rejected == False)  # noqa: E712
        )).all()
    for fill, intent, market in rows:
        params = (market.params_json if market else None) or {}
        if params.get("closed"):
            continue
        try:
            price = float(fill.avg_price or "0")
            shares = float(fill.filled_size_shares or "0")
        except (TypeError, ValueError):
            continue
        if price == 0 or shares == 0:
            continue
        key = (intent.market_condition_id or "", intent.sleeve_id, intent.side)
        p = positions.setdefault(key, {
            "shares": 0.0, "cost": 0.0,
            "last_mid": params.get("last_mid"),
        })
        p["shares"] += shares
        p["cost"] += price * shares
    out: list[InventoryPosition] = []
    for (cond, sleeve, side), p in positions.items():
        avg = p["cost"] / p["shares"] if p["shares"] > 0 else 0.0
        last_mid_v = None
        try:
            last_mid_v = float(p["last_mid"]) if p["last_mid"] is not None else None
        except (TypeError, ValueError):
            last_mid_v = None
        mtm = (last_mid_v - avg) * p["shares"] if last_mid_v is not None else None
        out.append(InventoryPosition(
            market_condition_id=cond,
            sleeve_id=sleeve,
            side=side,
            shares=round(p["shares"], 2),
            entry_price=round(avg, 4),
            cost_usd=round(p["cost"], 2),
            last_mid=round(last_mid_v, 4) if last_mid_v is not None else None,
            mtm_pnl_usd=round(mtm, 4) if mtm is not None else None,
        ))
    return sorted(out, key=lambda x: -abs(x.mtm_pnl_usd or 0))
