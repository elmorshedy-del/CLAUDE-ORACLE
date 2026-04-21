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
import time
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


async def weather_quality(_req: web.Request) -> web.Response:
    """Forecast quality metrics: is our data improving predictions?

    Returns per-city-per-kind:
      - calibration: Brier score, BSS vs climatology, reliability bins, sharpness
      - ngr_status: is a fit available? improvement over raw?
      - data_coverage: resolved records available for training
      - trend: Brier score over rolling 7d windows (sparkline)
    """
    from .weather_calibration import (
        WeatherForecastRecord, compute_calibration,
    )
    from .db.models import NGRFitRow
    from .feeds.open_meteo import CITIES

    kinds = ["temperature_max_day", "precipitation_sum_period"]
    slices = []

    async with SessionLocal() as db:
        for city in CITIES.keys():
            for kind in kinds:
                # Resolved & unresolved counts
                total = (await db.execute(
                    select(func.count()).select_from(WeatherForecastRecord).where(
                        WeatherForecastRecord.city == city,
                        WeatherForecastRecord.kind == kind,
                    )
                )).scalar() or 0
                resolved = (await db.execute(
                    select(func.count()).select_from(WeatherForecastRecord).where(
                        WeatherForecastRecord.city == city,
                        WeatherForecastRecord.kind == kind,
                        WeatherForecastRecord.resolved_at.is_not(None),
                    )
                )).scalar() or 0
                # Most recent NGR fit for this slice
                ngr_row = (await db.execute(
                    select(NGRFitRow).where(
                        NGRFitRow.city == city, NGRFitRow.kind == kind,
                    ).order_by(NGRFitRow.fitted_at.desc()).limit(1)
                )).scalar_one_or_none()
                if total == 0:
                    continue  # no data yet, skip to keep response compact
                slices.append({
                    "city": city, "kind": kind,
                    "total": total, "resolved": resolved,
                    "ngr": None if ngr_row is None else {
                        "fitted_at": ngr_row.fitted_at.isoformat(),
                        "n_samples": ngr_row.n_training_samples,
                        "crps_raw": round(ngr_row.mean_crps_raw, 4),
                        "crps_ngr": round(ngr_row.mean_crps_train, 4),
                        "improvement_pct": round(ngr_row.improvement_pct, 1),
                    },
                })
    # Add calibration reports (only for slices with enough resolved data).
    for s in slices:
        try:
            report = await compute_calibration(
                city=s["city"], kind=s["kind"], min_records=20,
            )
        except Exception:
            report = None
        if report is None:
            s["calibration"] = None
            continue
        s["calibration"] = {
            "n_resolved": report.n_resolved,
            "brier": round(report.brier_score, 4),
            "brier_climatology": round(report.brier_score_climatology, 4),
            "brier_skill_score": round(report.brier_skill_score, 4),
            "base_rate": round(report.empirical_base_rate, 3),
            "sharpness": round(report.sharpness, 3),
            "auc": None if report.auc is None else round(report.auc, 3),
            "reliability": [
                {
                    "bin_low": round(b.bin_low, 2),
                    "bin_high": round(b.bin_high, 2),
                    "n": b.n_forecasts,
                    "predicted": round(b.mean_predicted, 3),
                    "observed": round(b.observed_frequency, 3),
                }
                for b in report.reliability_bins
            ],
            "raw_brier": None if report.raw_brier_score is None else round(report.raw_brier_score, 4),
        }
    return web.json_response({"slices": slices})


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

async def pnl(_req: web.Request) -> web.Response:
    """Per-trade P&L and win-rate summary.

    For each non-rejected fill, compute realized P&L:
      - BUY @ price p, size s shares → cost = p*s, resolved_value in [0, 1]
      - realized = (resolved_value - p) * s - fees
    For unresolved fills, mark-to-market against the latest book mid.

    Returns aggregated totals per sleeve AND the per-trade breakdown (last 200).
    """
    from .db.models import FillRow, Market, OrderIntentRow, SleeveConfig
    import httpx

    per_sleeve: dict[str, dict] = {}
    trades: list[dict] = []

    async with SessionLocal() as db:
        rows = (await db.execute(
            select(FillRow, OrderIntentRow)
            .join(OrderIntentRow, FillRow.client_order_id == OrderIntentRow.client_order_id)
            .where(FillRow.rejected == False)  # noqa: E712
            .order_by(FillRow.created_at.desc())
            .limit(500)
        )).all()

        # Collect condition_ids we need resolution info for.
        cond_ids = list({i.market_condition_id for _, i in rows if i.market_condition_id})
        markets_by_cond: dict[str, Market] = {}
        if cond_ids:
            markets = (await db.execute(
                select(Market).where(Market.condition_id.in_(cond_ids))
            )).scalars().all()
            for m in markets:
                markets_by_cond[m.condition_id] = m

    for f, i in rows:
        try:
            price = float(f.avg_price or "0")
            shares = float(f.filled_size_shares or "0")
            fees = float(f.fees_usd or "0")
        except (TypeError, ValueError):
            continue
        if shares == 0:
            continue
        cost = price * shares
        m = markets_by_cond.get(i.market_condition_id)
        # Resolution status from params_json end_unix + closed flag if available.
        resolved_value = None
        status = "open"
        if m is not None:
            params = m.params_json or {}
            now = time.time()
            end_unix = params.get("end_unix") or 0
            if end_unix and end_unix < now:
                # Market has passed resolution time; try to read closed price from params.
                closed_yes_price = params.get("closed_yes_price")
                if closed_yes_price is not None:
                    # Caller must know which token this intent was on. For a YES
                    # BUY (btc "Up" or weather bucket YES) a win = closed_yes_price
                    # close to 1. For "Down" BUY we bought the NO token which
                    # resolves to (1 - closed_yes_price) in a binary.
                    # Conservative: treat `closed_yes_price >= 0.5` as YES won.
                    if closed_yes_price >= 0.5:
                        resolved_value = 1.0
                    else:
                        resolved_value = 0.0
                    status = "resolved"

        if resolved_value is None:
            # Mark-to-market against last known mid if available.
            params = (m.params_json if m else None) or {}
            mid = params.get("last_mid")
            if mid is not None:
                resolved_value = float(mid)
                status = "mtm"
            else:
                # No resolution AND no mid — use entry price as MTM placeholder (zero P&L).
                resolved_value = price
                status = "unresolved"

        pnl_usd = (resolved_value - price) * shares - fees  # fees negative for rebate
        won = resolved_value >= 0.5 if status == "resolved" else None
        trades.append({
            "fill_id": f.fill_id,
            "created_at": f.created_at.isoformat(),
            "sleeve_id": i.sleeve_id,
            "side": i.side,
            "price": round(price, 4),
            "shares": round(shares, 2),
            "cost_usd": round(cost, 2),
            "fees_usd": round(fees, 4),
            "resolved_value": round(resolved_value, 4) if resolved_value is not None else None,
            "pnl_usd": round(pnl_usd, 4),
            "status": status,
            "won": won,
        })
        agg = per_sleeve.setdefault(i.sleeve_id, {
            "sleeve_id": i.sleeve_id, "n_trades": 0, "n_resolved": 0, "n_wins": 0,
            "realized_pnl_usd": 0.0, "mtm_pnl_usd": 0.0, "unresolved_n": 0,
            "total_fees_usd": 0.0, "total_notional_usd": 0.0,
        })
        agg["n_trades"] += 1
        agg["total_notional_usd"] += cost
        agg["total_fees_usd"] += fees
        if status == "resolved":
            agg["n_resolved"] += 1
            if won:
                agg["n_wins"] += 1
            agg["realized_pnl_usd"] += pnl_usd
        elif status == "mtm":
            agg["mtm_pnl_usd"] += pnl_usd
        else:
            agg["unresolved_n"] += 1

    # Compute win rates and round.
    for a in per_sleeve.values():
        a["win_rate_pct"] = round(100 * a["n_wins"] / a["n_resolved"], 1) if a["n_resolved"] > 0 else None
        for k in ("realized_pnl_usd", "mtm_pnl_usd", "total_fees_usd", "total_notional_usd"):
            a[k] = round(a[k], 2)

    return web.json_response({
        "sleeves": sorted(per_sleeve.values(), key=lambda x: -x["n_trades"]),
        "trades": trades[:200],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })


async def export_stream(req: web.Request) -> web.StreamResponse:
    """Streaming export of intents + fills + forecasts over a date range.

    Query params:
        start   - ISO date (YYYY-MM-DD). Default: 7 days ago.
        end     - ISO date (YYYY-MM-DD). Default: now.
        table   - intents | fills | forecasts | all. Default: all.
        format  - csv | json. Default: csv.

    Uses aiohttp StreamResponse + server-side batched queries so memory stays
    flat even on years of data. Flushes every 500 rows. Non-blocking for other
    endpoints (each request in its own event-loop task).
    """
    from datetime import datetime as _dt
    from .db.models import FillRow, OrderIntentRow

    start = req.query.get("start")
    end = req.query.get("end")
    table = req.query.get("table", "all")
    fmt = req.query.get("format", "csv").lower()

    try:
        start_dt = _dt.fromisoformat(start) if start else datetime.now(timezone.utc) - timedelta(days=7)
        end_dt = _dt.fromisoformat(end) if end else datetime.now(timezone.utc)
    except ValueError:
        return web.json_response({"error": "invalid start/end — use YYYY-MM-DD"}, status=400)

    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    ct = "application/json" if fmt == "json" else "text/csv"
    fname = f"poly-paper_{table}_{start_dt.date()}_{end_dt.date()}.{fmt}"
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": f"{ct}; charset=utf-8",
            "Content-Disposition": f'attachment; filename="{fname}"',
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",  # disable any nginx buffering
        },
    )
    await resp.prepare(req)

    import csv as _csv
    import io as _io
    import json as _jsn

    def _dump_csv_row(writer: "_csv.writer", row: dict, fp: "_io.StringIO") -> bytes:
        writer.writerow([row.get(k, "") for k in row.keys()])
        data = fp.getvalue().encode()
        fp.seek(0)
        fp.truncate(0)
        return data

    async def stream_one(label: str, batch_iter):
        """Yield rows in CSV or NDJSON form."""
        header_written = False
        fp = _io.StringIO()
        writer = _csv.writer(fp)
        async for row in batch_iter:
            if fmt == "json":
                line = _jsn.dumps({"table": label, **row}, default=str) + "\n"
                await resp.write(line.encode())
            else:
                if not header_written:
                    writer.writerow([f"table={label}"])
                    writer.writerow(list(row.keys()))
                    await resp.write(fp.getvalue().encode()); fp.seek(0); fp.truncate(0)
                    header_written = True
                writer.writerow([row.get(k, "") for k in row.keys()])
                await resp.write(fp.getvalue().encode()); fp.seek(0); fp.truncate(0)

    async def paginate_query(model, time_col):
        """Page through a table, yielding dicts."""
        page_size = 500
        last_id = 0
        while True:
            async with SessionLocal() as db:
                rows = (await db.execute(
                    select(model)
                    .where(time_col >= start_dt, time_col <= end_dt, model.id > last_id)
                    .order_by(model.id.asc())
                    .limit(page_size)
                )).scalars().all()
            if not rows:
                return
            for r in rows:
                d = {c.name: getattr(r, c.name) for c in r.__table__.columns}
                yield d
            last_id = rows[-1].id

    tables_to_export = []
    if table in ("all", "intents"):
        tables_to_export.append(("intents", OrderIntentRow, OrderIntentRow.created_at))
    if table in ("all", "fills"):
        tables_to_export.append(("fills", FillRow, FillRow.created_at))
    if table in ("all", "forecasts"):
        try:
            from .weather_calibration import WeatherForecastRecord
            tables_to_export.append(("forecasts", WeatherForecastRecord, WeatherForecastRecord.recorded_at))
        except Exception:
            pass

    for label, model, tcol in tables_to_export:
        try:
            await stream_one(label, paginate_query(model, tcol))
        except Exception as e:
            # Write an error marker but keep streaming other tables.
            err_line = f"# export_error {label}: {e}\n".encode()
            await resp.write(err_line)

    await resp.write_eof()
    return resp


async def risk_snapshot(_req: web.Request) -> web.Response:
    """Live inventory + net rebate P&L (gross rebates MINUS MTM loss)."""
    from .risk import compute_net_rebate_pnl, current_inventory
    report = await compute_net_rebate_pnl()
    positions = await current_inventory()
    return web.json_response({
        "net_rebate": {
            "gross_rebates_usd": report.gross_rebates_usd,
            "open_inventory_notional_usd": report.open_inventory_notional_usd,
            "open_inventory_mtm_pnl_usd": report.open_inventory_mtm_pnl_usd,
            "net_rebate_pnl_usd": report.net_rebate_pnl_usd,
            "open_position_count": report.open_position_count,
        },
        "positions": [
            {
                "market_condition_id": p.market_condition_id,
                "sleeve_id": p.sleeve_id,
                "side": p.side,
                "shares": p.shares,
                "entry_price": p.entry_price,
                "cost_usd": p.cost_usd,
                "last_mid": p.last_mid,
                "mtm_pnl_usd": p.mtm_pnl_usd,
            }
            for p in positions
        ],
        "caps": {
            "max_position_per_market_usd": float(os.environ.get("POLY_MAX_POSITION_PER_MARKET_USD", "50")),
            "max_sleeve_exposure_fraction": float(os.environ.get("POLY_MAX_SLEEVE_EXPOSURE_FRACTION", "0.10")),
            "max_global_exposure_fraction": float(os.environ.get("POLY_MAX_GLOBAL_EXPOSURE_FRACTION", "0.50")),
            "bankroll_usd": float(os.environ.get("POLY_BANKROLL_USD", "1000")),
        },
    })


async def arb_stats(_req: web.Request) -> web.Response:
    """Latest arb scan summary + historical arb fills."""
    from .arb_scanner import get_latest_arb_stats
    from .db.models import FillRow, OrderIntentRow
    latest = get_latest_arb_stats()

    async with SessionLocal() as db:
        rows = (await db.execute(
            select(FillRow, OrderIntentRow)
            .join(OrderIntentRow, FillRow.client_order_id == OrderIntentRow.client_order_id)
            .where(OrderIntentRow.sleeve_id.like("%arb%"))
            .order_by(FillRow.created_at.desc())
            .limit(50)
        )).all()
    recent = [
        {
            "created_at": f.created_at.isoformat(),
            "sleeve_id": i.sleeve_id,
            "market_condition_id": i.market_condition_id,
            "rejected": f.rejected,
            "price": f.avg_price,
            "shares": f.filled_size_shares,
            "fees": f.fees_usd,
            "reasoning": (i.reasoning or "")[:180],
        }
        for f, i in rows
    ]

    # Simple aggregates over arb fills: total notional, avg gap, fire rate.
    from sqlalchemy import func as _f
    async with SessionLocal() as db:
        fire_count = (await db.execute(
            select(_f.count()).select_from(FillRow)
            .join(OrderIntentRow, FillRow.client_order_id == OrderIntentRow.client_order_id)
            .where(OrderIntentRow.sleeve_id.like("%arb%"))
        )).scalar() or 0
        reject_count = (await db.execute(
            select(_f.count()).select_from(FillRow)
            .join(OrderIntentRow, FillRow.client_order_id == OrderIntentRow.client_order_id)
            .where(OrderIntentRow.sleeve_id.like("%arb%"))
            .where(FillRow.rejected == True)  # noqa: E712
        )).scalar() or 0

    return web.json_response({
        "latest_scan": latest,
        "lifetime": {
            "arb_fills_total": fire_count,
            "arb_fills_rejected": reject_count,
            "arb_fills_filled": fire_count - reject_count,
        },
        "recent": recent,
    })


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/readyz", readyz)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/tape", tape)
    app.router.add_get("/pnl", pnl)
    app.router.add_get("/weather-quality", weather_quality)
    app.router.add_get("/export", export_stream)
    app.router.add_get("/risk", risk_snapshot)
    app.router.add_get("/arb-stats", arb_stats)
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
