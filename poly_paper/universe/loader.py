"""Market universe management — rules-based auto-add for BTC up/down markets.

This module is responsible for scanning Polymarket, finding markets that
match our strategy families, classifying them, and recording them in the
`markets` table with `in_universe=True` so the runner can act on them.

Rules are EXPLICIT (declared below) so the set of markets the system touches
is always predictable and auditable.

Phase 2 strategy families:
  btc_up_down_5m   — Polymarket 5-minute BTC up/down markets
  btc_up_down_15m  — Polymarket 15-minute BTC up/down markets

Later phases add:
  btc_range_daily, btc_above_daily, eth_up_down_*, etc.
"""

from __future__ import annotations

import json as _json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import Market


@dataclass(frozen=True)
class UniverseRule:
    """Declarative rule for auto-adding markets to the universe.

    A market is admitted IFF:
      - matching_strategy_family is set (classifier matched it)
      - min_liquidity_usd <= market.liquidity
      - max_universe_size not yet exceeded for this family
    """

    strategy_family: str
    min_liquidity_usd: float
    max_universe_size: int  # cap so we can't run away


DEFAULT_RULES: list[UniverseRule] = [
    UniverseRule(strategy_family="btc_up_down_5m", min_liquidity_usd=500, max_universe_size=50),
    UniverseRule(strategy_family="btc_up_down_15m", min_liquidity_usd=500, max_universe_size=50),
]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# Slugs like "btc-updown-5m-1776555900" where the number is the unix resolution ts.
_BTC_UPDOWN_SLUG = re.compile(r"^btc-updown-(5m|15m)-(\d+)$")


def classify_btc_updown(market_or_event_dict: dict) -> tuple[str, dict] | None:
    """If the event/market looks like a BTC up/down market, return (family, params).

    Params include:
      resolution_unix      : int — when the market resolves
      window_seconds       : 300 or 900
      start_unix, end_unix : start and end of the reference window (same as resolution for up/down)
      outcomes             : [{"outcome": "Up", "token_id": "..."}, ...]
    Returns None if not a match.
    """
    slug = market_or_event_dict.get("slug", "")
    m = _BTC_UPDOWN_SLUG.match(slug)
    if not m:
        return None
    window, res_unix = m.group(1), int(m.group(2))
    window_seconds = 300 if window == "5m" else 900
    family = f"btc_up_down_{window}"
    # Polymarket "up/down" events have a single market inside; we pull its token ids.
    markets = market_or_event_dict.get("markets", [])
    if not markets:
        # Already a market, not event.
        markets = [market_or_event_dict]
    inner = markets[0]
    outcomes = inner.get("outcomes")
    if isinstance(outcomes, str):
        outcomes = _json.loads(outcomes)
    clob = inner.get("clobTokenIds") or inner.get("clob_token_ids")
    if isinstance(clob, str):
        clob = _json.loads(clob)
    if not outcomes or not clob or len(outcomes) != len(clob):
        return None
    tokens = [
        {"outcome": o, "token_id": str(t)}
        for o, t in zip(outcomes, clob)
    ]
    return family, {
        "resolution_unix": res_unix,
        "window_seconds": window_seconds,
        # For up/down markets, "start" is at resolution_unix - window_seconds.
        "start_unix": res_unix - window_seconds,
        "end_unix": res_unix,
        "outcomes": tokens,
        "resolution_source": "chainlink_btc_usd",
    }


# ---------------------------------------------------------------------------
# Discovery + persistence
# ---------------------------------------------------------------------------

async def discover_and_upsert(
    db: AsyncSession,
    events_or_markets: list[dict],
    *,
    rules: list[UniverseRule] = DEFAULT_RULES,
) -> dict:
    """Scan fetched events/markets, classify, enforce rules, upsert into DB.

    Returns counters: {added, updated, skipped_low_liq, skipped_cap}.
    """
    rule_by_family = {r.strategy_family: r for r in rules}

    current_counts: dict[str, int] = {}
    # Count existing in-universe markets per family for cap enforcement.
    for fam in rule_by_family:
        result = await db.execute(
            select(Market).where(
                Market.strategy_family == fam, Market.in_universe.is_(True)
            )
        )
        current_counts[fam] = len(result.scalars().all())

    counters = {"added": 0, "updated": 0, "skipped_low_liq": 0, "skipped_cap": 0, "not_classified": 0}

    for ev in events_or_markets:
        classified = classify_btc_updown(ev)
        if classified is None:
            counters["not_classified"] += 1
            continue
        family, params = classified
        rule = rule_by_family.get(family)
        if rule is None:
            continue

        # Drill into inner market for id / liquidity / volume.
        inner = (ev.get("markets") or [ev])[0]
        cond_id = inner.get("conditionId") or inner.get("condition_id")
        if not cond_id:
            continue
        liq = float(inner.get("liquidity") or inner.get("liquidityClob") or 0)
        vol = float(inner.get("volume24hr") or inner.get("volume24hrClob") or 0)
        question = inner.get("question") or ev.get("title") or ""
        slug = inner.get("slug") or ev.get("slug") or ""

        # Look up existing.
        existing = (
            await db.execute(select(Market).where(Market.condition_id == cond_id))
        ).scalar_one_or_none()

        if existing is None:
            # Gate on liquidity + cap.
            if liq < rule.min_liquidity_usd:
                counters["skipped_low_liq"] += 1
                continue
            if current_counts[family] >= rule.max_universe_size:
                counters["skipped_cap"] += 1
                continue
            db.add(
                Market(
                    condition_id=cond_id,
                    question=question,
                    slug=slug,
                    category="crypto",
                    strategy_family=family,
                    end_date_iso=inner.get("endDate") or inner.get("end_date_iso"),
                    tokens_json=params["outcomes"],
                    params_json={
                        k: v for k, v in params.items() if k != "outcomes"
                    },
                    in_universe=True,
                    last_volume_24h_usd=vol,
                    last_liquidity_usd=liq,
                )
            )
            current_counts[family] += 1
            counters["added"] += 1
        else:
            # Refresh liquidity/volume; don't remove from universe just because liq dipped.
            existing.last_volume_24h_usd = vol
            existing.last_liquidity_usd = liq
            existing.last_seen_at = datetime.now(timezone.utc)
            # Update resolution / MTM params from Gamma payload so /pnl has fresh data.
            # Gamma returns outcomePrices (closed markets) and lastTradePrice / bestBid /
            # bestAsk for open markets. We persist:
            #   closed:           bool
            #   closed_yes_price: float 0-1 (resolution price for YES token)
            #   last_mid:         float — current mid for mark-to-market
            params_updates: dict = dict(existing.params_json or {})
            mkt_inner = (market_or_event_dict.get("markets") or [market_or_event_dict])[0]
            is_closed = bool(mkt_inner.get("closed"))
            params_updates["closed"] = is_closed
            if is_closed:
                op = mkt_inner.get("outcomePrices")
                if isinstance(op, str):
                    try:
                        op = _json.loads(op)
                    except Exception:
                        op = None
                if op and len(op) >= 1:
                    try:
                        params_updates["closed_yes_price"] = float(op[0])
                    except (TypeError, ValueError):
                        pass
            # Mark-to-market: prefer lastTradePrice, fall back to (bid+ask)/2.
            ltp = mkt_inner.get("lastTradePrice")
            bid = mkt_inner.get("bestBid")
            ask = mkt_inner.get("bestAsk")
            try:
                if ltp is not None:
                    params_updates["last_mid"] = float(ltp)
                elif bid is not None and ask is not None:
                    params_updates["last_mid"] = (float(bid) + float(ask)) / 2.0
            except (TypeError, ValueError):
                pass
            existing.params_json = params_updates
            counters["updated"] += 1

    await db.commit()
    return counters


def expired_markets(markets: list[Market]) -> list[Market]:
    """Markets whose resolution time has passed — should leave the universe."""
    now = time.time()
    out = []
    for m in markets:
        res = (m.params_json or {}).get("resolution_unix")
        if res is not None and res < now - 60:  # 60s buffer for resolution delays
            out.append(m)
    return out
