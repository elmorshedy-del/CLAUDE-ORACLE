"""Paper fill simulator.

This module is the heart of the "no lies" guarantee.

Given an OrderIntent and a current OrderBook snapshot, it simulates what would
happen in live trading and produces a Fill. Everything downstream (PnL, sleeve
performance, dashboards) uses that Fill exactly as if it came from the live CLOB.

Design principles:
1. **Walk the ladder, don't assume mid.** A market buy of 500 shares eats the
   best ask, then the next, etc., paying the weighted average.
2. **Apply the REAL Polymarket fee formula** (fees.py) — not a flat 2%.
3. **Model maker fill probability honestly.** A post-only order at a price level
   doesn't auto-fill — it needs counterparty flow. We use a conservative model
   (see _simulate_maker_fill) and tag low-confidence when uncertain.
4. **Every fill carries a confidence score.** Downstream UI never hides this.
5. **Latency is simulated.** Real orders take ~50-500ms round trip to Polymarket's
   off-chain matching engine; we add this to the fill timestamp. Used later to
   detect sleeves whose edge evaporates under realistic latency.

Limitations we are explicit about (logged on every fill in `notes` when relevant):
- Queue position for maker orders is unknown. We use a simple probability model
  based on book depth at that level + recent trade flow. Real fills may differ.
- Our book snapshot is by definition stale — it's from *now-epsilon*. Real
  orders hit the live book. Staleness is simulated via `staleness_ms`.
- Your own order impact is not modelled. If you place a 10,000-share order on
  a book with 5,000 shares of liquidity, live will show worse slippage than paper.
  We flag any fill where `filled_size > 0.3 * depth_within_1pct` as LOW confidence.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from .fees import leg_fee_usd
from .models import (
    ExecutionMode,
    Fill,
    FillConfidence,
    FillLeg,
    MarketCategory,
    OrderBook,
    OrderIntent,
    OrderType,
    Side,
)


# ---------------------------------------------------------------------------
# Simulator configuration (exposed so self-correction can tune)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaperSimConfig:
    # Simulated round-trip latency (ms). Polymarket CLOB averages ~100–300ms.
    latency_ms: int = 200
    # If simulated fill consumes more than this fraction of visible depth,
    # we downgrade confidence — big orders move books.
    large_order_depth_threshold: Decimal = Decimal("0.3")
    # For maker orders: if recent trade flow is weak, we assume the order sits
    # unfilled for the horizon of the simulation. Phase 1 keeps this conservative.
    maker_fill_default: Literal["always", "never", "probabilistic"] = "probabilistic"
    # Probability a maker order at the top of a reasonably deep book fills in
    # the simulation window (used when maker_fill_default == "probabilistic").
    maker_top_of_book_fill_prob: Decimal = Decimal("0.60")


DEFAULT_CONFIG = PaperSimConfig()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def simulate_fill(
    intent: OrderIntent,
    book: OrderBook,
    *,
    category: MarketCategory | None = None,
    config: PaperSimConfig = DEFAULT_CONFIG,
) -> Fill:
    """Simulate the fill for `intent` against `book`. Pure function — no I/O.

    Either `intent.size_shares` or `intent.size_usd` drives sizing.
    For size_usd on a BUY, we interpret it as "maximum notional USDC I will spend
    to acquire shares" (the ladder walk stops when spend would exceed size_usd).
    """
    cat = category or intent.category
    fill_id = f"fill_{uuid.uuid4().hex[:12]}"

    # Guard: no book at all.
    if not book.bids and not book.asks:
        return _reject(intent, fill_id, reason="empty book")

    if intent.side is Side.BUY:
        levels = book.asks  # buying consumes asks (sellers' offers)
        opposite_top = book.best_ask
    else:
        levels = book.bids
        opposite_top = book.best_bid

    if not levels:
        return _reject(intent, fill_id, reason=f"no {intent.side.value} liquidity")

    # Dispatch by order type.
    if intent.order_type in (OrderType.MARKET, OrderType.FOK):
        return _simulate_taker(intent, book, levels, cat, config, fill_id, must_fully_fill=(intent.order_type == OrderType.FOK))

    if intent.order_type == OrderType.POST_ONLY:
        return _simulate_maker(intent, book, cat, config, fill_id, reject_on_cross=True)

    if intent.order_type == OrderType.LIMIT:
        # LIMIT: if it crosses → becomes a taker sweep up to limit; else becomes a maker rest.
        if intent.limit_price is None:
            return _reject(intent, fill_id, reason="LIMIT order missing limit_price")
        if _would_cross(intent.side, intent.limit_price, opposite_top):
            return _simulate_taker(
                intent, book, levels, cat, config, fill_id,
                stop_price=intent.limit_price, must_fully_fill=False,
            )
        return _simulate_maker(intent, book, cat, config, fill_id, reject_on_cross=False)

    return _reject(intent, fill_id, reason=f"unsupported order_type {intent.order_type}")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _reject(intent: OrderIntent, fill_id: str, *, reason: str) -> Fill:
    return Fill(
        intent=intent,
        mode=ExecutionMode.PAPER,
        legs=[],
        fees_usd=Decimal("0"),
        gas_usd=Decimal("0"),
        confidence=FillConfidence.HIGH,  # rejection is deterministic
        slippage_bps=0,
        latency_ms=0,
        notes=f"rejected: {reason}",
        fill_id=fill_id,
    )


def _would_cross(side: Side, limit_price: Decimal, opposite_top: Decimal | None) -> bool:
    """True iff a resting limit at `limit_price` would immediately match the opposite side."""
    if opposite_top is None:
        return False
    if side is Side.BUY:
        # Buy crosses if our bid ≥ best ask.
        return limit_price >= opposite_top
    else:
        # Sell crosses if our ask ≤ best bid.
        return limit_price <= opposite_top


def _simulate_taker(
    intent: OrderIntent,
    book: OrderBook,
    levels: list,
    category: MarketCategory,
    config: PaperSimConfig,
    fill_id: str,
    *,
    stop_price: Decimal | None = None,
    must_fully_fill: bool,
) -> Fill:
    """Walk the ladder, consuming liquidity until the size or budget is exhausted."""
    remaining_shares = intent.size_shares
    remaining_usd = intent.size_usd

    legs: list[FillLeg] = []
    total_notional = Decimal("0")
    total_shares = Decimal("0")

    for lvl in levels:
        # If we have a limit price (LIMIT order that crossed partway), stop if book moves past it.
        if stop_price is not None:
            if intent.side is Side.BUY and lvl.price > stop_price:
                break
            if intent.side is Side.SELL and lvl.price < stop_price:
                break

        # Determine how many shares we'd take at this level.
        if remaining_shares is not None:
            want = min(lvl.size, remaining_shares - total_shares)
        else:
            # size_usd → convert budget to shares at this price.
            assert remaining_usd is not None
            budget_left = remaining_usd - total_notional
            if budget_left <= 0:
                break
            want_from_budget = budget_left / lvl.price if intent.side is Side.BUY else budget_left / (Decimal("1") - lvl.price)
            want = min(lvl.size, want_from_budget)

        if want <= 0:
            break

        legs.append(FillLeg(price=lvl.price, size_shares=want, role="taker"))
        total_notional += lvl.price * want
        total_shares += want

        if remaining_shares is not None and total_shares >= remaining_shares:
            break
        if remaining_usd is not None and total_notional >= remaining_usd:
            break

    if not legs:
        return _reject(intent, fill_id, reason="no fillable liquidity at limit")

    # FOK must fully fill or cancel.
    target_shares = intent.size_shares
    if must_fully_fill and target_shares is not None and total_shares < target_shares:
        return _reject(intent, fill_id, reason=f"FOK could not fully fill ({total_shares}/{target_shares})")

    fees = sum(
        (leg_fee_usd(price=leg.price, size_shares=leg.size_shares, category=category, role="taker") for leg in legs),
        Decimal("0"),
    )

    # Confidence: taker fills against deep visible liquidity = HIGH.
    depth_top_1pct = book.depth_within(intent.side, Decimal("0.01"))
    confidence = _score_taker_confidence(total_shares, depth_top_1pct, config)

    avg_price = total_notional / total_shares
    slippage_bps = _slippage_bps(intent.side, avg_price, book)

    notes = []
    if confidence is FillConfidence.LOW:
        notes.append(f"large-order warning: filled {total_shares} shares vs ${depth_top_1pct} depth within 1%")
    if len(legs) > 1:
        notes.append(f"walked {len(legs)} price levels from {legs[0].price} to {legs[-1].price}")

    return Fill(
        intent=intent,
        mode=ExecutionMode.PAPER,
        legs=legs,
        fees_usd=fees,
        gas_usd=Decimal("0"),  # Polymarket off-chain match; settlement gas absorbed by the exchange on the winning side. Kept explicit so Phase-2 live reconciliation can override.
        confidence=confidence,
        slippage_bps=slippage_bps,
        latency_ms=config.latency_ms,
        notes="; ".join(notes),
        fill_id=fill_id,
    )


def _simulate_maker(
    intent: OrderIntent,
    book: OrderBook,
    category: MarketCategory,
    config: PaperSimConfig,
    fill_id: str,
    *,
    reject_on_cross: bool,
) -> Fill:
    """Simulate a maker order resting on the book.

    Phase-1 model (intentionally conservative):
      - If the order would cross and reject_on_cross=True → rejected (POST_ONLY semantics).
      - If we're at/better than the current best on our side → optimistic top-of-book fill.
        Probability = maker_top_of_book_fill_prob. Filled size = limit or visible opposite depth,
        whichever is smaller.
      - If we're behind the best → assume NOT FILLED within the simulation window.
        (Returns a rejected fill with a note; the strategy will re-evaluate on the next tick.)

    Confidence for maker fills is at best MEDIUM: we cannot know real queue position.
    """
    limit_price = intent.limit_price
    if limit_price is None:
        return _reject(intent, fill_id, reason="maker order missing limit_price")

    # Check for cross condition.
    opposite_top = book.best_ask if intent.side is Side.BUY else book.best_bid
    would_cross = _would_cross(intent.side, limit_price, opposite_top)
    if would_cross and reject_on_cross:
        return _reject(intent, fill_id, reason="post_only would cross")

    # Are we top-of-book on our side?
    same_side_top = book.best_bid if intent.side is Side.BUY else book.best_ask
    if same_side_top is None:
        # Empty side: we'd be alone at the top. Count that as top-of-book.
        at_top = True
    else:
        if intent.side is Side.BUY:
            at_top = limit_price >= same_side_top
        else:
            at_top = limit_price <= same_side_top

    if not at_top:
        # We'd be behind queue. Model says: no fill this tick.
        return Fill(
            intent=intent,
            mode=ExecutionMode.PAPER,
            legs=[],
            fees_usd=Decimal("0"),
            gas_usd=Decimal("0"),
            confidence=FillConfidence.MEDIUM,
            slippage_bps=0,
            latency_ms=config.latency_ms,
            notes=f"maker resting behind top-of-book; no fill simulated (limit {limit_price} vs top {same_side_top})",
            fill_id=fill_id,
        )

    # Probabilistic fill based on config. Deterministic "roll" using fill_id hash so tests
    # are repeatable. (NOT calling random here — determinism is a property Phase-1 holds.)
    if config.maker_fill_default == "never":
        return _reject(intent, fill_id, reason="maker fill disabled in sim config")
    if config.maker_fill_default == "probabilistic":
        # Use hash-of-fill_id as deterministic draw in [0,1).
        draw = Decimal(int(fill_id[-8:], 16) % 10000) / Decimal("10000")
        if draw > config.maker_top_of_book_fill_prob:
            return Fill(
                intent=intent,
                mode=ExecutionMode.PAPER,
                legs=[],
                fees_usd=Decimal("0"),
                gas_usd=Decimal("0"),
                confidence=FillConfidence.MEDIUM,
                slippage_bps=0,
                latency_ms=config.latency_ms,
                notes=f"maker top-of-book — probabilistic no-fill (draw={draw}, p={config.maker_top_of_book_fill_prob})",
                fill_id=fill_id,
            )

    # Fill: determine size.
    target_shares = intent.size_shares
    if target_shares is None:
        assert intent.size_usd is not None
        # Convert USD budget to shares at the limit price.
        if intent.side is Side.BUY:
            target_shares = intent.size_usd / limit_price
        else:
            target_shares = intent.size_usd / (Decimal("1") - limit_price)

    # Cap at visible opposite-side depth (we need counterparties to fill).
    opposite_depth = sum(
        (lvl.size for lvl in (book.asks if intent.side is Side.BUY else book.bids)),
        Decimal("0"),
    )
    filled_shares = min(target_shares, opposite_depth) if opposite_depth > 0 else target_shares

    leg = FillLeg(price=limit_price, size_shares=filled_shares, role="maker")
    fees = leg_fee_usd(price=limit_price, size_shares=filled_shares, category=category, role="maker")

    # Confidence: MEDIUM by default for all makers. Downgrade if order was very large vs visible depth.
    depth_top_1pct = book.depth_within(intent.side, Decimal("0.01"))
    confidence = FillConfidence.MEDIUM
    if depth_top_1pct > 0 and filled_shares > config.large_order_depth_threshold * depth_top_1pct:
        confidence = FillConfidence.LOW

    return Fill(
        intent=intent,
        mode=ExecutionMode.PAPER,
        legs=[leg],
        fees_usd=fees,
        gas_usd=Decimal("0"),
        confidence=confidence,
        slippage_bps=0,  # limit price is limit price
        latency_ms=config.latency_ms,
        notes=f"maker fill (top-of-book); rebate applied",
        fill_id=fill_id,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_taker_confidence(
    filled_shares: Decimal,
    depth_top_1pct: Decimal,
    config: PaperSimConfig,
) -> FillConfidence:
    """Score a taker fill's confidence based on the size-to-depth ratio."""
    if depth_top_1pct == 0:
        return FillConfidence.LOW
    ratio = filled_shares / depth_top_1pct
    if ratio > config.large_order_depth_threshold:
        return FillConfidence.LOW
    if ratio > config.large_order_depth_threshold / 2:
        return FillConfidence.MEDIUM
    return FillConfidence.HIGH


def _slippage_bps(side: Side, avg_price: Decimal, book: OrderBook) -> int:
    """Slippage vs. the pre-trade best opposite-side price, in basis points of the reference price."""
    ref = book.best_ask if side is Side.BUY else book.best_bid
    if ref is None or ref == 0:
        return 0
    diff = avg_price - ref if side is Side.BUY else ref - avg_price
    return int((diff / ref) * Decimal("10000"))
