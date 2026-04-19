"""Tiny HTTP server exposing health + metrics endpoints.

Railway sends healthchecks and wants a responsive HTTP port. This runs
alongside the trading engine loops in the same process (via asyncio.gather).

Endpoints:
  GET /              — redirect hint
  GET /healthz       — liveness (always 200 if process is up)
  GET /readyz        — readiness (200 only if last tick was recent)
  GET /metrics       — sleeve health snapshot as JSON
  GET /tape          — last 50 intents+fills as JSON (supports start/end filters)
  GET /export/manifest — lightweight inventory of the full export bundle
  GET /export/download — compressed multi-file export of all persisted audit data

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
from .export_bundle import (
    ExportFilter,
    build_manifest,
    fill_intent_payload,
    parse_export_datetime,
    prepare_export_path,
    write_export_bundle,
)


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
    # Serve the HTML dashboard at /.
    import os
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return web.Response(text=f.read(), content_type="text/html")
    except FileNotFoundError:
        return web.json_response({
            "service": "poly-paper",
            "endpoints": ["/healthz", "/readyz", "/metrics", "/tape"],
            "note": "dashboard HTML missing; falling back to JSON",
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
    payload = await _metrics_payload()
    return web.json_response(payload)


def _bad_request(message: str) -> web.HTTPBadRequest:
    return web.HTTPBadRequest(
        text=_json.dumps({"error": message}),
        content_type="application/json",
    )


def _export_filter_from_request(req: web.Request) -> ExportFilter:
    start_raw = req.query.get("start")
    end_raw = req.query.get("end")
    try:
        start = parse_export_datetime(start_raw) if start_raw else None
        end = parse_export_datetime(end_raw, is_end=True) if end_raw else None
    except ValueError as exc:
        raise _bad_request(f"Invalid start/end value: {exc}") from exc
    if start is not None and end is not None and start >= end:
        raise _bad_request("start must be earlier than end")
    return ExportFilter(start=start, end=end)


async def _metrics_payload() -> dict:
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
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "markets_in_universe": n_markets,
        "last_tick_at": LAST_TICK_AT.isoformat() if LAST_TICK_AT else None,
        "sleeves": out,
    }


async def tape(_req: web.Request) -> web.Response:
    export_filter = _export_filter_from_request(_req)
    raw_limit = _req.query.get("limit", "50")
    try:
        limit = max(1, min(int(raw_limit), 1000))
    except ValueError:
        limit = 50
    payload = await _tape_payload(limit=limit, export_filter=export_filter)
    return web.json_response(payload)


async def _tape_payload(
    limit: int = 50,
    export_filter: ExportFilter | None = None,
) -> list[dict]:
    export_filter = export_filter or ExportFilter()
    async with SessionLocal() as db:
        stmt = (
            select(FillRow, OrderIntentRow)
            .join(OrderIntentRow, FillRow.client_order_id == OrderIntentRow.client_order_id)
            .order_by(FillRow.created_at.desc()).limit(limit)
        )
        if export_filter.start is not None:
            stmt = stmt.where(FillRow.created_at >= export_filter.start)
        if export_filter.end is not None:
            stmt = stmt.where(FillRow.created_at < export_filter.end)
        rows = (await db.execute(stmt)).all()
        return [fill_intent_payload(fill, intent) for fill, intent in rows]


async def export_manifest(_req: web.Request) -> web.Response:
    export_filter = _export_filter_from_request(_req)
    payload = await build_manifest(export_filter=export_filter)
    return web.json_response(payload)


async def export_download(_req: web.Request) -> web.StreamResponse:
    export_filter = _export_filter_from_request(_req)
    metrics_payload = await _metrics_payload()
    tape_payload = await _tape_payload(limit=100, export_filter=export_filter)
    export_path = prepare_export_path(export_filter)
    await write_export_bundle(
        export_path,
        metrics_payload=metrics_payload,
        tape_payload=tape_payload,
        export_filter=export_filter,
    )
    response = web.FileResponse(path=export_path)
    response.headers["Content-Disposition"] = f'attachment; filename="{export_path.name}"'
    response.content_type = "application/zip"
    return response


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
    app.router.add_get("/export/manifest", export_manifest)
    app.router.add_get("/export/download", export_download)
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
