"""The runner — main loop tying everything together.

Per tick (default 10 seconds):
  1. Refresh BTC spot from Coinbase.
  2. Recompute annualised realized vol (cached, refreshed every 30 min).
  3. Fetch active BTC up/down events from Polymarket Gamma.
  4. Classify + upsert into markets table (rules-based).
  5. For each market in universe:
       a. Fetch L2 books for UP and DOWN tokens.
       b. Build MarketContext.
       c. For each sleeve, call strategies.btc_updown.evaluate().
       d. Log the fair-value snapshot regardless of trade decision.
       e. If a decision produced an intent, route it through execute_order()
          (paper mode). Log the intent and fill.
       f. Persist a book snapshot for replay.
  6. Sleep until next tick.

Graceful: a failure on one market does not abort the tick. Failures are
structlog-logged with full context.
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select

from .db.models import (
    BookSnap,
    FairValueSnap,
    FillRow,
    Market,
    OrderIntentRow,
    SleeveConfig,
)
from .db.session import SessionLocal, init_db
from .exec.models import (
    ExecutionMode,
    MarketCategory,
    OrderBook,
    OrderIntent,
    SleeveConfig as ExecSleeveConfig,
    SleeveStance,
)
from .exec.router import execute_order
from .feeds.btc_spot import BTCSpotFeed
from .logging_setup import configure_structlog
from .polymarket.rest import PolymarketRest
from .strategies.btc_updown import MarketContext, default_btc_up_down_sleeves, evaluate
from .strategies.bundle_arb import (
    ArbContext, OutcomeQuote, default_bundle_arb_sleeves, evaluate_bundle,
)
from .strategies.fair_value import realized_vol_annualised
from .universe.loader import discover_and_upsert

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Config (env-driven)
# ---------------------------------------------------------------------------

TICK_SECONDS = float(os.environ.get("POLY_TICK_SECONDS", "10"))
VOL_REFRESH_SECONDS = float(os.environ.get("POLY_VOL_REFRESH_SECONDS", "1800"))  # 30 min
BANKROLL_USD = Decimal(os.environ.get("POLY_BANKROLL_USD", "1000"))
# Which strategy families to run. Phase 2: BTC up/down only.
ENABLED_FAMILIES = os.environ.get("POLY_FAMILIES", "btc_up_down_5m,btc_up_down_15m").split(",")
MODE = ExecutionMode(os.environ.get("POLY_MODE", "paper"))


# ---------------------------------------------------------------------------
# Volatility cache
# ---------------------------------------------------------------------------

class VolCache:
    def __init__(self) -> None:
        self._value: float | None = None
        self._refreshed_at: float = 0

    async def get(self, spot_feed: BTCSpotFeed) -> float:
        now = time.time()
        if self._value is None or (now - self._refreshed_at) > VOL_REFRESH_SECONDS:
            closes = await spot_feed.get_history_closes(granularity_seconds=3600, n_bars=24 * 14)
            self._value = realized_vol_annualised(closes, bar_seconds=3600, robust=True)
            self._refreshed_at = now
            log.info("vol_refreshed", sigma_annual=self._value, n_bars=len(closes))
        return self._value


# ---------------------------------------------------------------------------
# Sleeve bootstrap
# ---------------------------------------------------------------------------

async def ensure_sleeves_seeded() -> None:
    """Create sleeve config rows if they don't exist yet."""
    async with SessionLocal() as db:
        for fam in ENABLED_FAMILIES:
            fam = fam.strip()
            if not fam:
                continue
            # Two strategy families share the same market universe: directional
            # btc_updown and market-agnostic bundle_arb. They complement each other.
            all_configs = (
                default_btc_up_down_sleeves(strategy_family=fam, total_bankroll_usd=BANKROLL_USD)
                + default_bundle_arb_sleeves(strategy_family=fam, total_bankroll_usd=BANKROLL_USD)
            )
            for exec_cfg in all_configs:
                existing = (
                    await db.execute(
                        select(SleeveConfig).where(SleeveConfig.sleeve_id == exec_cfg.sleeve_id)
                    )
                ).scalar_one_or_none()
                if existing:
                    continue
                db.add(
                    SleeveConfig(
                        sleeve_id=exec_cfg.sleeve_id,
                        stance=exec_cfg.stance.value,
                        strategy_name=exec_cfg.strategy_name,
                        market_selector=exec_cfg.market_selector,
                        bankroll_usd=str(exec_cfg.bankroll_usd),
                        max_position_usd=str(exec_cfg.max_position_usd),
                        min_edge_bps=exec_cfg.min_edge_bps,
                        max_cross_spread_bps=exec_cfg.max_cross_spread_bps,
                        enabled=exec_cfg.enabled,
                        version=exec_cfg.version,
                        notes=exec_cfg.notes,
                        extra_json={"min_gross_edge_bps": exec_cfg.min_gross_edge_bps},
                    )
                )
        await db.commit()

    # Weather sleeves — separate universe (not indexed by strategy_family).
    from .weather_runner import ensure_weather_sleeves_seeded
    await ensure_weather_sleeves_seeded(BANKROLL_USD)


def _load_exec_sleeve(row: SleeveConfig) -> ExecSleeveConfig:
    return ExecSleeveConfig(
        sleeve_id=row.sleeve_id,
        stance=SleeveStance(row.stance),
        strategy_name=row.strategy_name,
        market_selector=row.market_selector,
        bankroll_usd=Decimal(row.bankroll_usd),
        max_position_usd=Decimal(row.max_position_usd),
        min_edge_bps=row.min_edge_bps,
        min_gross_edge_bps=(row.extra_json or {}).get("min_gross_edge_bps", row.min_edge_bps),
        max_cross_spread_bps=row.max_cross_spread_bps,
        enabled=row.enabled,
        version=row.version,
        notes=row.notes,
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

async def discover_markets(api: PolymarketRest, db) -> dict:
    """Pull active BTC up/down events from Gamma and upsert into DB."""
    # crypto-prices tag captures the up/down events.
    import subprocess
    # We use the httpx client behind PolymarketRest for the main GET, but that
    # helper is oriented to /markets. For events we go direct.
    import httpx as _httpx
    async with _httpx.AsyncClient(timeout=15) as client:
        r = await client.get(
            "https://gamma-api.polymarket.com/events",
            params={
                "active": "true",
                "closed": "false",
                "tag_slug": "crypto-prices",
                "limit": 200,
                "order": "volume24hr",
            },
            headers={"User-Agent": "poly-paper/0.2"},
        )
        r.raise_for_status()
        events = r.json()
    counters = await discover_and_upsert(db, events)
    return counters


# ---------------------------------------------------------------------------
# Main tick
# ---------------------------------------------------------------------------

async def tick(
    api: PolymarketRest,
    spot_feed: BTCSpotFeed,
    vol: VolCache,
) -> None:
    t0 = time.time()
    async with SessionLocal() as db:
        # 1. Refresh universe.
        try:
            counters = await discover_markets(api, db)
            log.info("universe", **counters)
        except Exception as e:
            log.warning("universe_refresh_failed", error=str(e))

        # 2. Spot + vol.
        spot_quote = await spot_feed.get_spot()
        sigma = await vol.get(spot_feed)
        log.info("spot", price=spot_quote.price, source=spot_quote.source, sigma_annual=round(sigma, 4))

        # 3. Load active universe.
        markets = (
            await db.execute(
                select(Market).where(
                    Market.in_universe.is_(True),
                    Market.strategy_family.in_([f.strip() for f in ENABLED_FAMILIES]),
                )
            )
        ).scalars().all()

        # Filter to markets that haven't resolved yet.
        now = time.time()
        active = [m for m in markets if (m.params_json or {}).get("end_unix", 0) > now]
        if not active:
            log.info("tick_no_active_markets", total_in_db=len(markets))
            return

        # 4. Load sleeves (by strategy family).
        sleeve_rows = (await db.execute(select(SleeveConfig).where(SleeveConfig.enabled.is_(True)))).scalars().all()
        sleeves_by_family = {}
        for row in sleeve_rows:
            # market_selector looks like "strategy_family=btc_up_down_5m"
            if "strategy_family=" in row.market_selector:
                fam = row.market_selector.split("=", 1)[1]
                sleeves_by_family.setdefault(fam, []).append(_load_exec_sleeve(row))

        n_intents = 0
        n_skips = 0
        for m in active:
            fam = m.strategy_family or ""
            sleeves = sleeves_by_family.get(fam, [])
            if not sleeves:
                continue
            try:
                await _evaluate_market(m, sleeves, spot_quote.price, sigma, api, db, MODE)
            except Exception as e:
                log.warning("market_eval_failed", condition=m.condition_id, err=str(e))
                continue

        await db.commit()
        log.info("tick_done", elapsed_sec=round(time.time() - t0, 2), active_markets=len(active))
        # Mark the readiness probe healthy.
        from .http_server import record_tick
        record_tick()


async def _evaluate_market(
    m: Market,
    sleeves: list[ExecSleeveConfig],
    spot: float,
    sigma: float,
    api: PolymarketRest,
    db,
    mode: ExecutionMode,
) -> None:
    # Fetch books for UP and DOWN.
    tokens = m.tokens_json or []
    books: dict[str, OrderBook] = {}
    token_ids: dict[str, str] = {}
    for tok in tokens:
        outcome = tok.get("outcome")
        tid = tok.get("token_id")
        token_ids[outcome] = tid
        try:
            books[outcome] = await api.get_book(tid)
        except LookupError:
            log.debug("no_book", market=m.condition_id, token=tid)
            return  # book not ready yet

    # Persist book snaps (one per side).
    for outcome, book in books.items():
        db.add(_build_book_snap(token_ids[outcome], book))

    params = m.params_json or {}
    end_unix = params.get("end_unix", 0)
    seconds_to_res = max(end_unix - time.time(), 1.0)

    ctx = MarketContext(
        market_condition_id=m.condition_id,
        strategy_family=m.strategy_family or "",
        seconds_to_resolution=seconds_to_res,
        spot=spot,
        sigma_annual=sigma,
        books=books,
        token_ids=token_ids,
    )

    # Log fair value ONCE per market per tick (same across all sleeves).
    any_sleeve = sleeves[0]
    probe = evaluate(any_sleeve, ctx)
    db.add(FairValueSnap(
        market_condition_id=m.condition_id,
        token_id=token_ids.get("Up", ""),
        probability=probe.fair_value.probability,
        ci_low=probe.fair_value.ci_low,
        ci_high=probe.fair_value.ci_high,
        spot=spot,
        sigma_annual=sigma,
        horizon_seconds=seconds_to_res,
        model="up_down",
    ))

    # Evaluate all sleeves.
    for sleeve in sleeves:
        decision = evaluate(sleeve, ctx)
        if decision.intent is None:
            log.debug(
                "sleeve_skip",
                sleeve=sleeve.sleeve_id, market=m.condition_id,
                reason=decision.reason_skipped, gross_bps=decision.gross_edge_bps,
            )
            continue

        # Persist the intent row.
        db.add(_intent_to_row(decision.intent))

        # Execute via router (paper mode in Phase 2).
        side = decision.chosen_outcome
        fill = await execute_order(
            decision.intent,
            mode=mode,
            book=books[side] if side else None,
            category=MarketCategory.CRYPTO,
        )

        # Persist fill row.
        db.add(_fill_to_row(fill, decision.intent.client_order_id))

        log.info(
            "intent_executed",
            sleeve=sleeve.sleeve_id, market=m.condition_id, outcome=side,
            rejected=fill.rejected, conf=fill.confidence.value,
            avg_price=str(fill.avg_price) if fill.avg_price else None,
            fees=str(fill.fees_usd), notes=fill.notes[:120],
        )


# ---------------------------------------------------------------------------
# Row mappers
# ---------------------------------------------------------------------------

def _intent_to_row(intent: OrderIntent) -> OrderIntentRow:
    return OrderIntentRow(
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
    )


def _fill_to_row(fill, client_order_id: str) -> FillRow:
    return FillRow(
        fill_id=fill.fill_id,
        client_order_id=client_order_id,
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
            {"price": str(leg.price), "size_shares": str(leg.size_shares), "role": leg.role}
            for leg in fill.legs
        ],
        notes=fill.notes,
    )


def _build_book_snap(token_id: str, book: OrderBook) -> BookSnap:
    return BookSnap(
        token_id=token_id,
        best_bid=float(book.best_bid) if book.best_bid is not None else None,
        best_bid_size=float(book.bids[0].size) if book.bids else None,
        best_ask=float(book.best_ask) if book.best_ask is not None else None,
        best_ask_size=float(book.asks[0].size) if book.asks else None,
        depth_1pct_buy=float(book.depth_within(_side("BUY"), Decimal("0.01"))),
        depth_1pct_sell=float(book.depth_within(_side("SELL"), Decimal("0.01"))),
        timestamp_ms=book.timestamp_ms,
    )


def _side(s):
    from .exec.models import Side
    return Side(s)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_forever() -> None:
    configure_structlog()
    log.info("runner_starting", mode=MODE.value, tick_seconds=TICK_SECONDS, families=ENABLED_FAMILIES)
    await init_db()
    await ensure_sleeves_seeded()

    vol = VolCache()
    async with PolymarketRest() as api, BTCSpotFeed() as spot_feed:
        # Warm vol cache before first tick.
        await vol.get(spot_feed)
        while True:
            try:
                await tick(api, spot_feed, vol)
            except Exception as e:
                log.error("tick_failed", error=str(e))
            await asyncio.sleep(TICK_SECONDS)


if __name__ == "__main__":
    asyncio.run(run_forever())
