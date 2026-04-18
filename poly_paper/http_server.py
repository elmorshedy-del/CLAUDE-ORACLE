"""Tiny HTTP server exposing health + metrics endpoints.

Railway sends healthchecks and wants a responsive HTTP port. This runs
alongside the trading engine loops in the same process (via asyncio.gather).

Endpoints:
  GET /              — redirect hint
  GET /healthz       — liveness (always 200 if process is up)
  GET /readyz        — readiness (200 only if last tick was recent)
  GET /metrics       — sleeve health snapshot as JSON
  GET /tape          — last 50 intents+fills as JSON

No external deps beyond stdlib + the existing DB session.
"""

from __future__ import annotations

import json as _json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from aiohttp import web  # aiohttp is transitively available; if not, we add it.
from sqlalchemy import func, select

from .db.models import FillRow, Market, OrderIntentRow, SleeveConfig
from .db.session import SessionLocal


# Time of the last successful main-loop tick; written by runner.py each cycle.
# Readiness = "we ticked within the last TICK_TOLERANCE_SECONDS".
LAST_TICK_AT: datetime | None = None
TICK_TOLERANCE_SECONDS = float(os.environ.get("POLY_TICK_TOLERANCE", "120"))


def record_tick() -> None:
    """Called by the runner at the end of every successful tick."""
    global LAST_TICK_AT
    LAST_TICK_AT = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def index(_req: web.Request) -> web.Response:
    return web.json_response({
        "service": "poly-paper",
        "endpoints": ["/healthz", "/readyz", "/metrics", "/tape"],
    })


async def healthz(_req: web.Request) -> web.Response:
    # Always 200 if the process is responding.
    return web.json_response({"status": "ok"})


async def readyz(_req: web.Request) -> web.Response:
    if LAST_TICK_AT is None:
        return web.json_response({"status": "warming_up"}, status=503)
    age = (datetime.now(timezone.utc) - LAST_TICK_AT).total_seconds()
    if age > TICK_TOLERANCE_SECONDS:
        return web.json_response(
            {"status": "stale", "last_tick_seconds_ago": age},
            status=503,
        )
    return web.json_response({
        "status": "ready",
        "last_tick_at": LAST_TICK_AT.isoformat(),
        "last_tick_seconds_ago": age,
    })


async def metrics(_req: web.Request) -> web.Response:
    since = datetime.now(timezone.utc) - timedelta(hours=24)
    async with SessionLocal() as db:
        sleeves = (await db.execute(select(SleeveConfig).order_by(SleeveConfig.sleeve_id))).scalars().all()
        out = []
        for s in sleeves:
            n_intents = (await db.execute(
                select(func.count()).select_from(OrderIntentRow).where(
                    OrderIntentRow.sleeve_id == s.sleeve_id,
                    OrderIntentRow.created_at >= since,
                )
            )).scalar() or 0
            fills = (await db.execute(
                select(FillRow).join(OrderIntentRow, FillRow.client_order_id == OrderIntentRow.client_order_id).where(
                    OrderIntentRow.sleeve_id == s.sleeve_id,
                    FillRow.created_at >= since,
                )
            )).scalars().all()
            n_fills = sum(1 for f in fills if not f.rejected)
            n_rej = sum(1 for f in fills if f.rejected)
            fees = sum(Decimal(f.fees_usd or "0") for f in fills if not f.rejected)
            hi = sum(1 for f in fills if f.confidence == "high" and not f.rejected)
            hi_pct = (100 * hi / n_fills) if n_fills else 0.0
            out.append({
                "sleeve_id": s.sleeve_id,
                "stance": s.stance,
                "strategy": s.strategy_name,
                "enabled": s.enabled,
                "version": s.version,
                "intents_24h": n_intents,
                "fills_24h": n_fills,
                "rejected_24h": n_rej,
                "fees_24h_usd": float(fees),
                "high_conf_fill_pct": round(hi_pct, 1),
                "bankroll_usd": float(Decimal(s.bankroll_usd)),
                "max_position_usd": float(Decimal(s.max_position_usd)),
                "min_edge_bps": s.min_edge_bps,
            })
        n_markets = (await db.execute(
            select(func.count()).select_from(Market).where(Market.in_universe.is_(True))
        )).scalar() or 0
    return web.json_response({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "markets_in_universe": n_markets,
        "last_tick_at": LAST_TICK_AT.isoformat() if LAST_TICK_AT else None,
        "sleeves": out,
    })


async def tape(_req: web.Request) -> web.Response:
    async with SessionLocal() as db:
        rows = (await db.execute(
            select(FillRow, OrderIntentRow)
            .join(OrderIntentRow, FillRow.client_order_id == OrderIntentRow.client_order_id)
            .order_by(FillRow.created_at.desc()).limit(50)
        )).all()
        out = [
            {
                "fill_id": f.fill_id,
                "created_at": f.created_at.isoformat(),
                "sleeve_id": i.sleeve_id,
                "market_condition_id": i.market_condition_id,
                "side": i.side,
                "order_type": i.order_type,
                "rejected": f.rejected,
                "confidence": f.confidence,
                "avg_price": f.avg_price,
                "filled_size_shares": f.filled_size_shares,
                "fees_usd": f.fees_usd,
                "slippage_bps": f.slippage_bps,
                "notes": f.notes,
                "reasoning": i.reasoning,
            }
            for f, i in rows
        ]
    return web.json_response(out)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/readyz", readyz)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/tape", tape)
    return app


async def serve_forever() -> None:
    port = int(os.environ.get("PORT", "8080"))  # Railway sets PORT.
    app = build_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=port)
    await site.start()
    # Keep the coroutine alive — the server runs in background tasks.
    import asyncio
    while True:
        await asyncio.sleep(3600)
