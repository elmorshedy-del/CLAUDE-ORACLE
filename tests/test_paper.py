"""Tests for the paper fill simulator.

These tests encode the invariants a good paper simulator must hold. If any
test fails, the fill simulator is lying about something.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from poly_paper.exec.models import (
    BookLevel,
    ExecutionMode,
    MarketCategory,
    OrderBook,
    OrderIntent,
    OrderType,
    Side,
)
from poly_paper.exec.paper import DEFAULT_CONFIG, PaperSimConfig, simulate_fill


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def book(bids: list[tuple[str, str]], asks: list[tuple[str, str]]) -> OrderBook:
    """Build an OrderBook from (price, size) tuples. Bids should be descending; asks ascending."""
    return OrderBook(
        token_id="tok",
        market_condition_id="cond",
        timestamp_ms=0,
        bids=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in bids],
        asks=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in asks],
    )


def intent(
    *,
    side: Side = Side.BUY,
    order_type: OrderType = OrderType.MARKET,
    size_shares: str | None = None,
    size_usd: str | None = None,
    limit_price: str | None = None,
    category: MarketCategory = MarketCategory.SPORTS,
) -> OrderIntent:
    return OrderIntent(
        sleeve_id="test",
        market_condition_id="cond",
        token_id="tok",
        side=side,
        order_type=order_type,
        limit_price=Decimal(limit_price) if limit_price else None,
        size_shares=Decimal(size_shares) if size_shares else None,
        size_usd=Decimal(size_usd) if size_usd else None,
        category=category,
        client_order_id="t1",
    )


# ---------------------------------------------------------------------------
# MARKET / taker fills
# ---------------------------------------------------------------------------

class TestMarketBuy:
    def test_fills_at_best_ask_when_enough_depth(self) -> None:
        b = book(bids=[("0.49", "1000")], asks=[("0.51", "1000")])
        f = simulate_fill(intent(size_shares="100"), b)
        assert not f.rejected
        assert len(f.legs) == 1
        assert f.legs[0].price == Decimal("0.51")
        assert f.legs[0].size_shares == Decimal("100")
        assert f.legs[0].role == "taker"
        assert f.mode == ExecutionMode.PAPER

    def test_walks_the_ladder(self) -> None:
        b = book(
            bids=[("0.49", "1000")],
            asks=[("0.51", "30"), ("0.52", "40"), ("0.53", "1000")],
        )
        f = simulate_fill(intent(size_shares="100"), b)
        assert len(f.legs) == 3
        assert f.legs[0].price == Decimal("0.51")
        assert f.legs[0].size_shares == Decimal("30")
        assert f.legs[1].price == Decimal("0.52")
        assert f.legs[1].size_shares == Decimal("40")
        assert f.legs[2].price == Decimal("0.53")
        assert f.legs[2].size_shares == Decimal("30")
        # avg_price = (30*0.51 + 40*0.52 + 30*0.53)/100 = 52.0/100 = 0.52
        assert f.avg_price == Decimal("0.5200")

    def test_slippage_reported_in_bps(self) -> None:
        b = book(
            bids=[("0.49", "1000")],
            asks=[("0.50", "10"), ("0.52", "1000")],  # best ask 0.50, next 0.52
        )
        f = simulate_fill(intent(size_shares="100"), b)
        # avg = (10*0.50 + 90*0.52)/100 = 0.518; ref = 0.50; slippage = 0.018/0.5 = 3.6%
        # 3.6% = 360 bps
        assert f.slippage_bps == 360

    def test_rejects_when_no_asks(self) -> None:
        b = book(bids=[("0.49", "1000")], asks=[])
        f = simulate_fill(intent(size_shares="100"), b)
        assert f.rejected
        assert "no BUY liquidity" in f.notes

    def test_size_in_usd_stops_at_budget(self) -> None:
        b = book(
            bids=[("0.49", "1000")],
            asks=[("0.50", "10"), ("0.60", "1000")],
        )
        # Budget $20. First leg: 10 shares * $0.50 = $5 spent. Remaining $15 / $0.60 = 25 shares.
        f = simulate_fill(intent(size_usd="20"), b)
        assert len(f.legs) == 2
        assert f.legs[0].size_shares == Decimal("10")
        assert f.legs[1].price == Decimal("0.60")
        assert f.legs[1].size_shares == Decimal("25")

    def test_fok_rejects_if_cannot_fully_fill(self) -> None:
        b = book(bids=[], asks=[("0.50", "10")])
        f = simulate_fill(intent(order_type=OrderType.FOK, size_shares="100"), b)
        assert f.rejected
        assert "FOK" in f.notes


class TestTakerConfidence:
    def test_small_fill_is_high_confidence(self) -> None:
        b = book(bids=[], asks=[("0.50", "10000")])
        f = simulate_fill(intent(size_shares="50"), b)
        assert f.confidence.value == "high"

    def test_large_fill_is_low_confidence(self) -> None:
        # Order eats >30% of visible top-1% depth.
        b = book(bids=[], asks=[("0.50", "100")])
        f = simulate_fill(intent(size_shares="90"), b)
        assert f.confidence.value == "low"


# ---------------------------------------------------------------------------
# Fees
# ---------------------------------------------------------------------------

class TestFees:
    def test_taker_fee_applied_on_market_buy(self) -> None:
        b = book(bids=[], asks=[("0.50", "1000")])
        f = simulate_fill(intent(size_shares="100", category=MarketCategory.SPORTS), b)
        # notional 50, sports peak 0.75% at p=0.5 → fee = 50 * 0.0075 = 0.375
        assert f.fees_usd == Decimal("0.375000")

    def test_no_fees_on_geopolitics(self) -> None:
        b = book(bids=[], asks=[("0.50", "1000")])
        f = simulate_fill(intent(size_shares="100", category=MarketCategory.GEOPOLITICS), b)
        assert f.fees_usd == Decimal("0")


# ---------------------------------------------------------------------------
# POST_ONLY / maker fills
# ---------------------------------------------------------------------------

class TestPostOnly:
    def test_rejects_if_would_cross(self) -> None:
        b = book(bids=[("0.49", "1000")], asks=[("0.51", "1000")])
        # Buy at 0.52 would cross (>= best ask 0.51).
        f = simulate_fill(
            intent(order_type=OrderType.POST_ONLY, size_shares="10", limit_price="0.52"),
            b,
        )
        assert f.rejected
        assert "cross" in f.notes

    def test_rests_at_top_of_book_medium_confidence(self) -> None:
        # Disable the probabilistic no-fill for this test by forcing always-fill.
        cfg = PaperSimConfig(maker_fill_default="always")
        b = book(bids=[("0.49", "1000")], asks=[("0.51", "1000")])
        # Buy at 0.50 — above current best bid 0.49 so becomes new top-of-book, doesn't cross 0.51.
        f = simulate_fill(
            intent(order_type=OrderType.POST_ONLY, size_shares="10", limit_price="0.50"),
            b,
            config=cfg,
        )
        assert not f.rejected
        assert f.legs[0].role == "maker"
        assert f.legs[0].price == Decimal("0.50")
        # 10 shares on a 1000-deep book both sides = thick book, small size → HIGH.
        # If the book were thin or our size dominated, this would downgrade.
        assert f.confidence.value in ("high", "medium")

    def test_maker_rebate_applied(self) -> None:
        cfg = PaperSimConfig(maker_fill_default="always")
        b = book(bids=[("0.49", "1000")], asks=[("0.51", "1000")])
        f = simulate_fill(
            intent(
                order_type=OrderType.POST_ONLY,
                size_shares="100",
                limit_price="0.50",
                category=MarketCategory.SPORTS,
            ),
            b,
            config=cfg,
        )
        # Rebate should be negative (cash received).
        assert f.fees_usd < 0


# ---------------------------------------------------------------------------
# LIMIT
# ---------------------------------------------------------------------------

class TestLimit:
    def test_limit_that_crosses_becomes_taker_up_to_limit(self) -> None:
        b = book(
            bids=[("0.49", "1000")],
            asks=[("0.50", "20"), ("0.52", "1000"), ("0.54", "1000")],
        )
        # Buy up to 0.52 for 500 shares. Should fill 20 @ 0.50 + 480 @ 0.52 = 500.
        f = simulate_fill(
            intent(order_type=OrderType.LIMIT, size_shares="500", limit_price="0.52"),
            b,
        )
        assert not f.rejected
        assert sum(leg.size_shares for leg in f.legs) == Decimal("500")
        assert all(leg.price <= Decimal("0.52") for leg in f.legs)
        assert all(leg.role == "taker" for leg in f.legs)

    def test_limit_stops_at_limit_price_even_partial(self) -> None:
        b = book(
            bids=[("0.49", "1000")],
            asks=[("0.50", "20"), ("0.55", "1000")],
        )
        # Buy up to 0.52 for 500 shares. Only 20 available at/below limit.
        f = simulate_fill(
            intent(order_type=OrderType.LIMIT, size_shares="500", limit_price="0.52"),
            b,
        )
        assert sum(leg.size_shares for leg in f.legs) == Decimal("20")

    def test_limit_not_crossing_becomes_maker(self) -> None:
        cfg = PaperSimConfig(maker_fill_default="always")
        b = book(bids=[("0.49", "1000")], asks=[("0.51", "1000")])
        # Buy at 0.50 doesn't cross (best ask is 0.51) → rests as maker.
        f = simulate_fill(
            intent(order_type=OrderType.LIMIT, size_shares="10", limit_price="0.50"),
            b,
            config=cfg,
        )
        assert not f.rejected
        assert f.legs[0].role == "maker"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_same_intent_same_book() -> None:
    """Same inputs produce the same Fill, except for the random fill_id. This is crucial
    for replay-based backtesting and for comparing config versions."""
    b = book(bids=[("0.49", "1000")], asks=[("0.51", "1000")])
    i = intent(size_shares="100")
    f1 = simulate_fill(i, b)
    f2 = simulate_fill(i, b)
    # fill_id differs, but all economic quantities are identical.
    assert f1.legs == f2.legs
    assert f1.fees_usd == f2.fees_usd
    assert f1.confidence == f2.confidence
    assert f1.slippage_bps == f2.slippage_bps
