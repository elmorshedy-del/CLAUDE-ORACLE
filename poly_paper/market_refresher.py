"""Market resolution refresher.

For every market we've ever filled on, query Gamma periodically and update
`params_json` with:
  - closed:           bool
  - closed_yes_price: final resolution price (YES token) when closed
  - last_mid:         current mid for mark-to-market PnL

This fills the gap where the BTC runner only refreshes BTC markets, leaving
weather/arb/global trades marked "unresolved" forever even after Polymarket
has officially closed them. /pnl reads params_json to decide P&L status,
so this loop is what makes P&L actually show resolved trades.

Runs every `POLY_MKT_REFRESH_INTERVAL_SEC` (default: 5 minutes).
"""

from __future__ import annotations

import asyncio
import json as _json
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog
from sqlalchemy import select

from .db.models import FillRow, Market, OrderIntentRow
from .db.session import SessionLocal

log = structlog.get_logger()

REFRESH_INTERVAL_SECONDS = float(os.environ.get("POLY_MKT_REFRESH_INTERVAL_SEC", "300"))
ENABLED = os.environ.get("POLY_MKT_REFRESH_ENABLED", "1") not in ("0", "false", "no")
GAMMA = "https://gamma-api.polymarket.com"


async def _fetch_market(
    client: httpx.AsyncClient, condition_id: str,
) -> Optional[dict]:
    """Hit Gamma /markets?condition_ids=... and return the one match or None."""
    try:
        r = await client.get(
            f"{GAMMA}/markets",
            params={"condition_ids": condition_id},
            headers={"User-Agent": "poly-paper/0.4"},
        )
        if r.status_code != 200:
            return None
        payload = r.json()
    except Exception:
        return None
    if not payload:
        return None
    if isinstance(payload, list):
        return payload[0] if payload else None
    return payload


def _extract_resolution_fields(mkt: dict) -> dict:
    """Return params updates with closed/closed_yes_price/last_mid."""
    out: dict = {}
    out["closed"] = bool(mkt.get("closed"))
    if out["closed"]:
        op = mkt.get("outcomePrices")
        if isinstance(op, str):
            try:
                op = _json.loads(op)
            except Exception:
                op = None
        if op and len(op) >= 1:
            try:
                out["closed_yes_price"] = float(op[0])
            except (TypeError, ValueError):
                pass
    ltp = mkt.get("lastTradePrice")
    bid = mkt.get("bestBid")
    ask = mkt.get("bestAsk")
    try:
        if ltp is not None:
            out["last_mid"] = float(ltp)
        elif bid is not None and ask is not None:
            out["last_mid"] = (float(bid) + float(ask)) / 2.0
    except (TypeError, ValueError):
        pass
    return out


async def refresh_once() -> dict:
    """One pass — refresh every Market row that we've ever traded on OR have in universe."""
    async with SessionLocal() as db:
        # Collect condition_ids we care about: any market in our DB.
        rows = (await db.execute(select(Market))).scalars().all()
        condition_ids = [r.condition_id for r in rows if r.condition_id]

    refreshed = 0
    closed_count = 0
    errors = 0
    async with httpx.AsyncClient(timeout=10, limits=httpx.Limits(max_connections=10)) as client:
        for cid in condition_ids:
            try:
                mkt = await _fetch_market(client, cid)
            except Exception:
                errors += 1
                continue
            if mkt is None:
                continue
            updates = _extract_resolution_fields(mkt)
            if not updates:
                continue
            async with SessionLocal() as db:
                row = (await db.execute(
                    select(Market).where(Market.condition_id == cid)
                )).scalar_one_or_none()
                if row is None:
                    continue
                merged = dict(row.params_json or {})
                merged.update(updates)
                row.params_json = merged
                row.last_seen_at = datetime.now(timezone.utc)
                await db.commit()
            refreshed += 1
            if updates.get("closed"):
                closed_count += 1
    return {
        "markets_checked": len(condition_ids),
        "refreshed": refreshed,
        "newly_or_still_closed": closed_count,
        "errors": errors,
    }


async def run_market_refresher_forever() -> None:
    if not ENABLED:
        log.info("market_refresher_disabled")
        return
    while True:
        try:
            stats = await refresh_once()
            log.info("market_refresher_tick", **stats)
        except Exception as e:
            log.error("market_refresher_failed", err=str(e))
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
