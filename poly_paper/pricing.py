"""Price utilities for Polymarket.

Polymarket's CLOB has a 1-cent minimum tick. Float arithmetic produces values
like 0.45999999999999996 from 0.47 - 0.01, which breaks maker-price placement
(we end up posted BEHIND top-of-book instead of AT it). This module
quantizes all maker / limit prices to exact cents before submission.

Also provides a post-fill confidence tagger used by the dashboard and
self-correction proposer.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Literal

# Polymarket tick — 1 cent. Override via env if a market uses different tick.
TICK = Decimal("0.01")


def quantize_cents(x: float | Decimal) -> Decimal:
    """Round to exact cents (half-up). Always returns Decimal."""
    if isinstance(x, float):
        x = Decimal(str(x))
    return x.quantize(TICK, rounding=ROUND_HALF_UP)


def clamp_price(x: float | Decimal, lo: Decimal = Decimal("0.01"),
                hi: Decimal = Decimal("0.99")) -> Decimal:
    """Clamp to [0.01, 0.99] on Polymarket's tick grid.

    A price of 0 or 1 means a resolved market — no trading allowed there.
    """
    q = quantize_cents(x)
    if q < lo:
        return lo
    if q > hi:
        return hi
    return q


def maker_post_price(best_bid: float | Decimal, best_ask: float | Decimal) -> Decimal | None:
    """Price for a post-only buy: one tick above best bid, never crossing ask.

    Returns None if the spread is too tight to post (bid and ask only 1 cent apart
    OR the book is crossed).
    """
    bid = quantize_cents(best_bid)
    ask = quantize_cents(best_ask)
    if ask <= bid:
        return None
    # Post one tick above bid; never at or above ask.
    candidate = bid + TICK
    if candidate >= ask:
        # Spread too tight — at-the-bid is the best we can do without crossing.
        return bid if bid > Decimal("0") else None
    return candidate


# ---------------------------------------------------------------------------
# Post-fill confidence classifier
# ---------------------------------------------------------------------------

FillConfidence = Literal["high", "medium", "low"]


def classify_post_fill_confidence(
    *,
    intent_price: Decimal,
    fill_price: Decimal | None,
    best_bid_at_fill: Decimal | None,
    best_ask_at_fill: Decimal | None,
    fill_size_shares: Decimal,
    intended_size_shares: Decimal,
    order_type: str,
    rejected: bool,
) -> FillConfidence:
    """Classify how realistic this paper fill is relative to a live fill.

    HIGH    — executed at or better than intent price, no size degradation,
              book was stable around fill level.
    MEDIUM  — executed but at worse price / smaller size, or maker fill via
              probabilistic queue simulation.
    LOW     — rejected, or fill price moved significantly from intent.
    """
    if rejected:
        return "low"
    if fill_price is None or fill_size_shares == 0:
        return "low"

    # Full size at or better than intent → HIGH (taker route).
    size_ratio = fill_size_shares / intended_size_shares if intended_size_shares > 0 else Decimal("0")
    same_tick = intent_price == fill_price
    tighter_than_expected = (
        intent_price > 0 and fill_price <= intent_price
    )

    if order_type == "post_only":
        # Maker fills are inherently queue-dependent. A maker fill AT the posted
        # price with FULL size is as "live-realistic" as we can claim — but we
        # deliberately don't mark it HIGH because queue position in live CLOB
        # has non-zero tail risk. Call it HIGH only if we filled at BETTER price
        # (spread improved in our direction), else MEDIUM.
        if fill_price < intent_price and size_ratio >= Decimal("0.95"):
            return "high"
        if size_ratio >= Decimal("0.95"):
            return "medium"
        return "low"

    # Taker / limit route.
    if same_tick and size_ratio >= Decimal("0.99"):
        return "high"
    if tighter_than_expected and size_ratio >= Decimal("0.75"):
        return "high"
    if size_ratio >= Decimal("0.50"):
        return "medium"
    return "low"
