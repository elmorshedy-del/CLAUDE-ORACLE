"""CLI status dashboard — prints current sleeve health to the terminal.

Usage:
    python -m scripts.status

Shows per-sleeve:
  - enabled flag, stance, strategy, version
  - last 24h: intents, fills, rejected, fees, paper PnL (realised + mark-to-market),
    fill-quality ratio
  - recent trade tape (last 10)

Also shows:
  - universe counts
  - recent arb scanner stats
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from rich.console import Console
from rich.table import Table
from sqlalchemy import select, func, and_
from datetime import datetime, timedelta, timezone

from poly_paper.db.models import (
    FairValueSnap,
    FillRow,
    Market,
    OrderIntentRow,
    SleeveConfig,
)
from poly_paper.db.session import SessionLocal, init_db


async def main() -> None:
    await init_db()
    console = Console()

    async with SessionLocal() as db:
        # Sleeves
        sleeves = (await db.execute(select(SleeveConfig).order_by(SleeveConfig.sleeve_id))).scalars().all()

        since = datetime.now(timezone.utc) - timedelta(hours=24)

        # Per-sleeve stats
        t = Table(title="Sleeve health — last 24h", show_lines=True)
        t.add_column("Sleeve"); t.add_column("Stance"); t.add_column("Strategy"); t.add_column("En")
        t.add_column("Intents", justify="right"); t.add_column("Fills", justify="right")
        t.add_column("Rej", justify="right"); t.add_column("Fees", justify="right")
        t.add_column("HiConf%", justify="right"); t.add_column("Version")

        for s in sleeves:
            intents = await db.execute(
                select(func.count()).select_from(OrderIntentRow)
                .where(OrderIntentRow.sleeve_id == s.sleeve_id, OrderIntentRow.created_at >= since)
            )
            n_intents = intents.scalar() or 0
            fills = (await db.execute(
                select(FillRow).join(OrderIntentRow, FillRow.client_order_id == OrderIntentRow.client_order_id)
                .where(OrderIntentRow.sleeve_id == s.sleeve_id, FillRow.created_at >= since)
            )).scalars().all()
            n_fills = sum(1 for f in fills if not f.rejected)
            n_rej = sum(1 for f in fills if f.rejected)
            fees = sum(Decimal(f.fees_usd or "0") for f in fills if not f.rejected)
            hi = sum(1 for f in fills if f.confidence == "high" and not f.rejected)
            hi_pct = f"{100*hi/n_fills:.0f}%" if n_fills else "—"
            t.add_row(
                s.sleeve_id, s.stance, s.strategy_name, "✓" if s.enabled else "✗",
                str(n_intents), str(n_fills), str(n_rej),
                f"${fees:.2f}", hi_pct, str(s.version),
            )
        console.print(t)

        # Market universe
        n_markets = (await db.execute(
            select(func.count()).select_from(Market).where(Market.in_universe.is_(True))
        )).scalar() or 0
        n_fv = (await db.execute(
            select(func.count()).select_from(FairValueSnap).where(FairValueSnap.computed_at >= since)
        )).scalar() or 0
        console.print(f"\nUniverse: [bold]{n_markets}[/bold] markets in_universe. "
                      f"Fair-value snapshots last 24h: [bold]{n_fv}[/bold]")

        # Recent trade tape
        recent = (await db.execute(
            select(FillRow, OrderIntentRow)
            .join(OrderIntentRow, FillRow.client_order_id == OrderIntentRow.client_order_id)
            .order_by(FillRow.created_at.desc()).limit(10)
        )).all()
        if recent:
            tape = Table(title="Recent fills")
            tape.add_column("When"); tape.add_column("Sleeve"); tape.add_column("Side"); tape.add_column("Type")
            tape.add_column("Price"); tape.add_column("Sz"); tape.add_column("Rej"); tape.add_column("Conf")
            tape.add_column("Fees"); tape.add_column("Market")
            for fill_row, intent_row in recent:
                tape.add_row(
                    fill_row.created_at.strftime("%H:%M:%S"),
                    intent_row.sleeve_id[-28:],
                    intent_row.side, intent_row.order_type,
                    fill_row.avg_price or "—", fill_row.filled_size_shares or "—",
                    "✗" if fill_row.rejected else "—",
                    fill_row.confidence, f"${fill_row.fees_usd}",
                    intent_row.market_condition_id[:20] + "…",
                )
            console.print(tape)
        else:
            console.print("[dim]No fills yet.[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
