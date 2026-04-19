"""Continuous arb scanner — catches transient arbs that static probes miss.

Runs as a separate loop in the runner. Every N seconds:
  1. Refresh candidate universe (binary markets AND neg_risk events).
  2. For each candidate, check ask-sum vs $1.
  3. If a true arb appears and survives fees, synthesise a bundle of OrderIntents
     through the bundle_arb strategy.
  4. Execute via the router (paper mode).

Honesty baked in:
  - Before classifying an event as "mutually exclusive," we require either:
      (a) negRisk == True on all markets in the event, OR
      (b) a binary market with exactly 2 outcomes (YES + NO).
    Nested date-ladder events (e.g. "BTC hits $150k by June OR by December") are
    NEVER classified as mutually exclusive and therefore never generate arb intents.
  - We skip events where ANY market has no book (missing books mean the arb
    safety condition isn't verified).
  - We tag every intent's reasoning with WHY we think exclusivity holds so
    later inspection is easy.

Expected hit rate: low on an efficient exchange. We budget for this to find
very few (or zero) arbs per hour — the value is in catching the edge cases
that DO appear during book imbalances.
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import httpx
import structlog
from sqlalchemy import select

from .db.models import FillRow, OrderIntentRow, SleeveConfig
from .db.session import SessionLocal
from .exec.models import (
    BookLevel,
    ExecutionMode,
    MarketCategory,
    OrderBook,
    SleeveConfig as ExecSleeveConfig,
    SleeveStance,
)
from .exec.router import execute_order
from .polymarket.rest import classify_market
from .strategies.bundle_arb import (
    ArbContext,
    OutcomeQuote,
    default_bundle_arb_sleeves,
    evaluate_bundle,
)

log = structlog.get_logger()

SCAN_INTERVAL_SECONDS = float(os.environ.get("POLY_ARB_SCAN_SECONDS", "30"))
MIN_EVENT_LIQUIDITY_USD = float(os.environ.get("POLY_ARB_MIN_LIQ", "1000"))


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


async def _fetch_candidate_events(client: httpx.AsyncClient) -> list[dict]:
    """Pull neg_risk events across ALL major Polymarket tags.

    These are guaranteed mutually exclusive and candidates for bundle arb.
    Bundle/ladder arb is strategy-agnostic — it doesn't care if the market is
    about BTC or presidential elections, only whether the math works. So we
    scan the entire universe, not just sleeves-I've-seeded-for.
    """
    events: list[dict] = []
    seen: set[str] = set()
    # Full coverage: sports, politics, geo, weather, crypto, economics, culture.
    tags = [
        "sports", "politics", "elections", "culture",
        "tennis", "nba", "nhl", "mlb", "nfl", "soccer", "mma", "boxing",
        "weather", "natural-disasters", "climate",
        "crypto", "bitcoin", "ethereum",
        "economy", "finance", "business",
        "technology", "ai", "science",
        "entertainment", "awards",
        "geopolitics", "world",
    ]
    for tag in tags:
        try:
            r = await client.get(
                f"{GAMMA}/events",
                params={"active": "true", "closed": "false", "tag_slug": tag, "limit": 100},
                headers={"User-Agent": "poly-paper/0.3"},
            )
            if r.status_code != 200:
                continue
            for ev in r.json():
                eid = ev.get("id")
                if eid and eid not in seen:
                    seen.add(eid)
                    events.append(ev)
        except Exception as e:
            log.warning("arb_scan_event_fetch_failed", tag=tag, err=str(e))
    return events


async def _fetch_binary_markets(client: httpx.AsyncClient) -> list[dict]:
    """Pull highest-liquidity binary (YES/NO) markets."""
    out: list[dict] = []
    for offset in (0, 500):
        try:
            r = await client.get(
                f"{GAMMA}/markets",
                params={"active": "true", "closed": "false", "limit": 500, "offset": offset},
                headers={"User-Agent": "poly-paper/0.2"},
            )
            r.raise_for_status()
            out.extend(r.json())
        except Exception as e:
            log.warning("arb_scan_markets_fetch_failed", offset=offset, err=str(e))
    # Only tradeable binaries with enough liquidity.
    filtered = []
    for m in out:
        if not (m.get("acceptingOrders") and m.get("enableOrderBook")):
            continue
        if float(m.get("liquidity") or 0) < MIN_EVENT_LIQUIDITY_USD:
            continue
        outcomes = m.get("outcomes")
        if isinstance(outcomes, str):
            outcomes = _json.loads(outcomes)
        clob = m.get("clobTokenIds")
        if isinstance(clob, str):
            clob = _json.loads(clob)
        if outcomes and len(outcomes) == 2 and clob and len(clob) == 2:
            filtered.append(m)
    return filtered


# ---------------------------------------------------------------------------
# Book fetching
# ---------------------------------------------------------------------------

async def _fetch_book(client: httpx.AsyncClient, token_id: str) -> OrderBook | None:
    try:
        r = await client.get(f"{CLOB}/book", params={"token_id": token_id})
        r.raise_for_status()
        raw = r.json()
    except Exception:
        return None
    if "error" in raw:
        return None
    bids = sorted(
        [BookLevel(price=Decimal(b["price"]), size=Decimal(b["size"])) for b in raw.get("bids", [])],
        key=lambda lv: lv.price,
        reverse=True,
    )
    asks = sorted(
        [BookLevel(price=Decimal(a["price"]), size=Decimal(a["size"])) for a in raw.get("asks", [])],
        key=lambda lv: lv.price,
    )
    return OrderBook(
        token_id=token_id,
        market_condition_id=raw.get("market", ""),
        timestamp_ms=int(raw.get("timestamp", "0")),
        bids=bids,
        asks=asks,
    )


# ---------------------------------------------------------------------------
# Classification: safe vs unsafe for bundle arb
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArbCandidate:
    """A group of markets whose outcomes are KNOWN to be mutually exclusive."""

    kind: str  # "binary" or "neg_risk_event"
    event_or_market_id: str
    label: str  # human-readable
    category: MarketCategory
    outcomes: list[dict]  # [{"outcome": "...", "token_id": "...", "market_condition_id": "..."}]


def _is_safe_binary(market: dict) -> bool:
    outcomes = market.get("outcomes")
    if isinstance(outcomes, str):
        outcomes = _json.loads(outcomes)
    # Must be exactly YES/NO (order-independent).
    if not outcomes or len(outcomes) != 2:
        return False
    lo = {o.lower() for o in outcomes}
    return lo == {"yes", "no"}


def _neg_risk_event_safe(event: dict) -> bool:
    """True iff ALL markets in the event have negRisk=True.

    negRisk is Polymarket's mechanism for guaranteeing exactly-one-outcome-wins
    across a group of related markets. When all markets share this flag, buying
    the YES side of each is guaranteed to pay $1 total at resolution.
    """
    mkts = event.get("markets") or []
    if len(mkts) < 3:
        return False
    tradeable = [
        m for m in mkts
        if m.get("acceptingOrders") and m.get("enableOrderBook") and m.get("negRisk")
    ]
    if len(tradeable) < 3:
        return False
    # Also require every tradeable market to have a clobTokenIds list.
    for m in tradeable:
        c = m.get("clobTokenIds")
        if isinstance(c, str):
            c = _json.loads(c)
        if not c or len(c) < 2:
            return False
    return True


def build_candidates(events: list[dict], markets: list[dict]) -> list[ArbCandidate]:
    out: list[ArbCandidate] = []

    # 1) neg_risk events.
    for ev in events:
        if not _neg_risk_event_safe(ev):
            continue
        cat = classify_market(ev)
        outs: list[dict] = []
        for m in ev.get("markets") or []:
            if not (m.get("acceptingOrders") and m.get("enableOrderBook") and m.get("negRisk")):
                continue
            outcomes = m.get("outcomes")
            if isinstance(outcomes, str):
                outcomes = _json.loads(outcomes)
            clob = m.get("clobTokenIds")
            if isinstance(clob, str):
                clob = _json.loads(clob)
            if not outcomes or not clob or len(outcomes) != len(clob):
                continue
            # "YES token" — the token whose outcome label is "Yes".
            yi = 0
            for i, o in enumerate(outcomes):
                if o == "Yes":
                    yi = i
                    break
            outs.append({
                "outcome": m.get("question", "")[:60],
                "token_id": clob[yi],
                "market_condition_id": m.get("conditionId") or m.get("condition_id", ""),
            })
        if len(outs) >= 3:
            out.append(ArbCandidate(
                kind="neg_risk_event",
                event_or_market_id=ev.get("id", ""),
                label=ev.get("title", "")[:80],
                category=cat,
                outcomes=outs,
            ))

    # 2) Binary markets (YES/NO).
    for m in markets:
        if not _is_safe_binary(m):
            continue
        cat = classify_market(m)
        outcomes = m.get("outcomes")
        if isinstance(outcomes, str):
            outcomes = _json.loads(outcomes)
        clob = m.get("clobTokenIds")
        if isinstance(clob, str):
            clob = _json.loads(clob)
        out.append(ArbCandidate(
            kind="binary",
            event_or_market_id=m.get("conditionId", ""),
            label=m.get("question", "")[:80],
            category=cat,
            outcomes=[
                {"outcome": outcomes[0], "token_id": clob[0],
                 "market_condition_id": m.get("conditionId", "")},
                {"outcome": outcomes[1], "token_id": clob[1],
                 "market_condition_id": m.get("conditionId", "")},
            ],
        ))
    return out


# ---------------------------------------------------------------------------
# Scan one candidate
# ---------------------------------------------------------------------------

async def _scan_candidate(
    candidate: ArbCandidate,
    sleeves: list[ExecSleeveConfig],
    client: httpx.AsyncClient,
) -> int:
    """Fetch books, build ArbContext, evaluate across sleeves, execute if any triggers.

    Returns number of bundle arbs fired.
    """
    # Fetch all books in parallel.
    books = await asyncio.gather(*[_fetch_book(client, o["token_id"]) for o in candidate.outcomes])
    # Need every book present.
    if any(b is None for b in books):
        return 0

    quotes = [
        OutcomeQuote(
            outcome=o["outcome"], token_id=o["token_id"], book=book,
        )
        for o, book in zip(candidate.outcomes, books)
    ]
    # We use the candidate's representative market_condition_id (first outcome's).
    ctx = ArbContext(
        market_condition_id=candidate.outcomes[0]["market_condition_id"],
        category=candidate.category,
        quotes=quotes,
    )

    fired = 0
    for sleeve in sleeves:
        decision = evaluate_bundle(sleeve, ctx)
        if not decision.intents:
            continue
        # For each leg of the bundle, we need the corresponding book to execute against.
        book_by_token = {q.token_id: q.book for q in quotes}

        async with SessionLocal() as db:
            for intent in decision.intents:
                # Persist intent row.
                db.add(OrderIntentRow(
                    client_order_id=intent.client_order_id,
                    sleeve_id=intent.sleeve_id,
                    market_condition_id=intent.market_condition_id,
                    token_id=intent.token_id,
                    side=intent.side.value,
                    order_type=intent.order_type.value,
                    limit_price=str(intent.limit_price) if intent.limit_price else None,
                    size_usd=str(intent.size_usd) if intent.size_usd else None,
                    size_shares=str(intent.size_shares) if intent.size_shares else None,
                    edge_bps=intent.edge_bps,
                    category=intent.category.value,
                    reasoning=intent.reasoning,
                ))
                # Execute via router.
                fill = await execute_order(
                    intent,
                    mode=ExecutionMode.PAPER,
                    book=book_by_token[intent.token_id],
                    category=candidate.category,
                )
                db.add(FillRow(
                    fill_id=fill.fill_id,
                    client_order_id=intent.client_order_id,
                    mode=fill.mode.value,
                    rejected=fill.rejected,
                    filled_size_shares=str(fill.filled_size_shares),
                    avg_price=str(fill.avg_price) if fill.avg_price is not None else None,
                    notional_usd=str(fill.notional_usd),
                    fees_usd=str(fill.fees_usd),
                    gas_usd=str(fill.gas_usd),
                    confidence=fill.confidence.value,
                    slippage_bps=fill.slippage_bps,
                    latency_ms=fill.latency_ms,
                    legs_json=[
                        {"price": str(l.price), "size_shares": str(l.size_shares), "role": l.role}
                        for l in fill.legs
                    ],
                    notes=fill.notes,
                ))
            await db.commit()

        log.info(
            "arb_fired",
            sleeve=sleeve.sleeve_id,
            candidate_kind=candidate.kind,
            candidate=candidate.label,
            n_legs=len(decision.intents),
            gap_bps=decision.gap_bps,
            net_edge_bps=decision.net_edge_bps,
        )
        fired += 1
        # Only fire ONE sleeve's bundle per candidate per scan — avoid double
        # exposure on the same arb across stances.
        break

    return fired


# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------

async def scan_once(sleeves: list[ExecSleeveConfig]) -> dict:
    """One pass through candidates. Returns counters."""
    async with httpx.AsyncClient(
        timeout=8, limits=httpx.Limits(max_connections=20),
    ) as client:
        t0 = time.time()
        events = await _fetch_candidate_events(client)
        markets = await _fetch_binary_markets(client)
        candidates = build_candidates(events, markets)
        log.info(
            "arb_scan_started",
            n_events=len(events), n_markets=len(markets),
            n_candidates=len(candidates),
        )

        # Evaluate up to 100 candidates per pass (sorted by implied liquidity below).
        candidates = candidates[:100]
        fired_total = 0
        # Do 10 in parallel.
        sem = asyncio.Semaphore(10)
        async def _one(cand: ArbCandidate) -> int:
            async with sem:
                try:
                    return await _scan_candidate(cand, sleeves, client)
                except Exception as e:
                    log.warning("arb_candidate_failed", label=cand.label, err=str(e))
                    return 0
        results = await asyncio.gather(*[_one(c) for c in candidates])
        fired_total = sum(results)
        return {
            "elapsed_sec": round(time.time() - t0, 2),
            "candidates_scanned": len(candidates),
            "arbs_fired": fired_total,
        }


async def run_arb_scanner_forever() -> None:
    from .runner import _load_exec_sleeve

    while True:
        try:
            async with SessionLocal() as db:
                rows = (await db.execute(
                    select(SleeveConfig).where(
                        SleeveConfig.enabled.is_(True),
                        SleeveConfig.strategy_name == "bundle_arb",
                    )
                )).scalars().all()
            sleeves = [_load_exec_sleeve(r) for r in rows]
            if not sleeves:
                log.warning("arb_scanner_no_sleeves")
            else:
                stats = await scan_once(sleeves)
                log.info("arb_scan_done", **stats)
        except Exception as e:
            log.error("arb_scanner_iteration_failed", err=str(e))
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)
