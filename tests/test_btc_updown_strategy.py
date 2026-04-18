"""Tests for the BTC up/down strategy evaluation logic.

We build synthetic MarketContexts and verify the strategy's decisions match
what the math says they should be.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from poly_paper.exec.models import (
    BookLevel,
    MarketCategory,
    OrderBook,
    OrderType,
    Side,
    SleeveConfig,
    SleeveStance,
)
from poly_paper.strategies.btc_updown import (
    MarketContext,
    default_btc_up_down_sleeves,
    evaluate,
)


def _book(bids: list[tuple[str, str]], asks: list[tuple[str, str]], token_id: str = "tok") -> OrderBook:
    return OrderBook(
        token_id=token_id,
        market_condition_id="cond",
        timestamp_ms=0,
        bids=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in bids],
        asks=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in asks],
    )


def _ctx(
    up_book: OrderBook,
    down_book: OrderBook,
    *,
    seconds_to_resolution: float = 300,
    sigma_annual: float = 0.7,
    spot: float = 100_000.0,
) -> MarketContext:
    return MarketContext(
        market_condition_id="cond",
        strategy_family="btc_up_down_5m",
        seconds_to_resolution=seconds_to_resolution,
        spot=spot,
        sigma_annual=sigma_annual,
        books={"Up": up_book, "Down": down_book},
        token_ids={"Up": "tok_up", "Down": "tok_dn"},
    )


def _sleeve(**kwargs) -> SleeveConfig:
    base = {
        "sleeve_id": "test",
        "stance": SleeveStance.AGGRESSIVE,
        "strategy_name": "btc_up_down",
        "market_selector": "strategy_family=btc_up_down_5m",
        "bankroll_usd": Decimal("1000"),
        "max_position_usd": Decimal("30"),
        "min_edge_bps": 80,
        "min_gross_edge_bps": 50,
        "max_cross_spread_bps": 200,
        "enabled": True,
        "version": 1,
        "notes": "test",
    }
    base.update(kwargs)
    return SleeveConfig(**base)


class TestNoEdgeCase:
    def test_fair_market_no_trade(self) -> None:
        # Both sides priced at 0.50 — theoretical fair value. No edge.
        up = _book([("0.495", "1000")], [("0.50", "1000")])
        dn = _book([("0.495", "1000")], [("0.50", "1000")])
        d = evaluate(_sleeve(), _ctx(up, dn))
        assert d.intent is None
        assert d.reason_skipped is not None
        # Either the gross-edge gate or the net-edge gate should trip.
        assert "edge" in d.reason_skipped.lower()


class TestEdgeCase:
    def test_up_underpriced_triggers_buy_up(self) -> None:
        # Up priced at 0.40, truly ~0.50 — huge edge. Down at 0.60.
        up = _book([("0.39", "1000")], [("0.40", "1000")])
        dn = _book([("0.59", "1000")], [("0.60", "1000")])
        d = evaluate(_sleeve(), _ctx(up, dn))
        assert d.intent is not None
        assert d.chosen_outcome == "Up"
        assert d.intent.side is Side.BUY
        assert d.intent.token_id == "tok_up"

    def test_down_underpriced_triggers_buy_down(self) -> None:
        up = _book([("0.59", "1000")], [("0.60", "1000")])
        dn = _book([("0.39", "1000")], [("0.40", "1000")])
        d = evaluate(_sleeve(), _ctx(up, dn))
        assert d.intent is not None
        assert d.chosen_outcome == "Down"
        assert d.intent.token_id == "tok_dn"


class TestBookSanity:
    def test_disjointed_book_rejected(self) -> None:
        # Both sides quote 0.30 ask — sum = 0.60 — book is broken.
        up = _book([("0.29", "1000")], [("0.30", "1000")])
        dn = _book([("0.29", "1000")], [("0.30", "1000")])
        d = evaluate(_sleeve(), _ctx(up, dn))
        assert d.intent is None
        assert "sanity fail" in (d.reason_skipped or "")


class TestStanceGating:
    def test_conservative_requires_bigger_edge(self) -> None:
        # Edge present but only ~1% — conservative should skip, aggressive should trade.
        up = _book([("0.48", "1000")], [("0.49", "1000")])
        dn = _book([("0.50", "1000")], [("0.51", "1000")])
        sleeves = default_btc_up_down_sleeves(
            strategy_family="btc_up_down_5m", total_bankroll_usd=Decimal("1000")
        )
        cons = next(s for s in sleeves if s.stance is SleeveStance.CONSERVATIVE)
        agg = next(s for s in sleeves if s.stance is SleeveStance.AGGRESSIVE)

        d_cons = evaluate(cons, _ctx(up, dn))
        d_agg = evaluate(agg, _ctx(up, dn))
        # Conservative's min_edge_bps=300 should reject; aggressive's 80 may accept.
        assert d_cons.intent is None
        # (aggressive result depends on fees; at p~0.5 for CRYPTO fees are peak ~1.8%,
        # so a 1% edge may or may not survive net. We only assert conservative skips.)

    def test_conservative_uses_post_only(self) -> None:
        # Build a case with enough edge for conservative to trade.
        # Up priced very cheap, Down at ~0.80. Vol far higher than typical → pushes fair far from 0.5.
        # Simplest: short horizon + normal vol → fair ~0.50. Up at 0.20 gives 30% edge.
        up = _book([("0.19", "1000")], [("0.20", "1000")])
        dn = _book([("0.79", "1000")], [("0.80", "1000")])
        sleeves = default_btc_up_down_sleeves(
            strategy_family="btc_up_down_5m", total_bankroll_usd=Decimal("1000")
        )
        cons = next(s for s in sleeves if s.stance is SleeveStance.CONSERVATIVE)
        d = evaluate(cons, _ctx(up, dn))
        assert d.intent is not None
        # Conservative must use post-only.
        assert d.intent.order_type is OrderType.POST_ONLY


class TestReasoningCaptured:
    def test_intent_reasoning_includes_key_numbers(self) -> None:
        up = _book([("0.19", "1000")], [("0.20", "1000")])
        dn = _book([("0.79", "1000")], [("0.80", "1000")])
        d = evaluate(_sleeve(), _ctx(up, dn))
        assert d.intent is not None
        reasoning = d.intent.reasoning
        for must_have in ("fv=", "spot=", "sigma=", "gross=", "net="):
            assert must_have in reasoning
