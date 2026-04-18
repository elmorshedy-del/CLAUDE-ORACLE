"""Tests for bundle arbitrage strategy."""

from __future__ import annotations

from decimal import Decimal

import pytest

from poly_paper.exec.models import (
    BookLevel,
    MarketCategory,
    OrderBook,
    OrderType,
    SleeveConfig,
    SleeveStance,
)
from poly_paper.strategies.bundle_arb import (
    ArbContext,
    OutcomeQuote,
    default_bundle_arb_sleeves,
    evaluate_bundle,
)


def _book(asks: list[tuple[str, str]], bids: list[tuple[str, str]] | None = None) -> OrderBook:
    return OrderBook(
        token_id="tok",
        market_condition_id="cond",
        timestamp_ms=0,
        bids=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in (bids or [])],
        asks=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in asks],
    )


def _ctx(asks_per_outcome: dict[str, list[tuple[str, str]]]) -> ArbContext:
    quotes = [
        OutcomeQuote(outcome=o, token_id=f"tok_{o}", book=_book(asks_list))
        for o, asks_list in asks_per_outcome.items()
    ]
    return ArbContext(
        market_condition_id="cond",
        category=MarketCategory.CRYPTO,
        quotes=quotes,
    )


def _sleeve(min_edge_bps: int = 50, max_pos: str = "100") -> SleeveConfig:
    return SleeveConfig(
        sleeve_id="arb_test",
        stance=SleeveStance.AGGRESSIVE,
        strategy_name="bundle_arb",
        market_selector="test",
        bankroll_usd=Decimal("1000"),
        max_position_usd=Decimal(max_pos),
        min_edge_bps=min_edge_bps,
        min_gross_edge_bps=min_edge_bps,
        max_cross_spread_bps=1000,
        enabled=True,
        version=1,
        notes="",
    )


class TestNoArb:
    def test_sum_exactly_one_no_arb(self) -> None:
        ctx = _ctx({"Up": [("0.50", "100")], "Down": [("0.50", "100")]})
        d = evaluate_bundle(_sleeve(), ctx)
        assert d.intents == []
        assert "no arb" in (d.reason_skipped or "")

    def test_sum_over_one_no_arb(self) -> None:
        ctx = _ctx({"Up": [("0.52", "100")], "Down": [("0.50", "100")]})
        d = evaluate_bundle(_sleeve(), ctx)
        assert d.intents == []

    def test_too_few_outcomes(self) -> None:
        ctx = _ctx({"Only": [("0.40", "100")]})
        d = evaluate_bundle(_sleeve(), ctx)
        assert d.intents == []


class TestArbDetection:
    def test_clean_arb_triggers_bundle(self) -> None:
        # sum_asks = 0.48 + 0.48 = 0.96 → gap = 400bps
        # Crypto fee at p=0.48: ~0.018 * 4 * 0.48 * 0.52 = ~1.797% → fee contribution
        #   per leg = 1.797% * 0.48 = ~0.863% → two legs = ~1.73% → fees = ~173bps
        # Net ≈ 400 - 173 = 227bps — far above our 50bps threshold.
        ctx = _ctx({
            "Up":   [("0.48", "100")],
            "Down": [("0.48", "100")],
        })
        d = evaluate_bundle(_sleeve(min_edge_bps=50), ctx)
        assert len(d.intents) == 2
        assert d.gap_bps > 0
        assert d.net_edge_bps >= 50

    def test_thin_arb_rejected_by_conservative(self) -> None:
        # Small gap that gets eaten by fees.
        ctx = _ctx({
            "Up":   [("0.499", "100")],
            "Down": [("0.499", "100")],
        })
        sleeves = default_bundle_arb_sleeves(
            strategy_family="btc_up_down_5m", total_bankroll_usd=Decimal("1000")
        )
        cons = next(s for s in sleeves if s.stance is SleeveStance.CONSERVATIVE)
        d = evaluate_bundle(cons, ctx)
        # Very small gap — fees will dominate.
        assert d.intents == []

    def test_bundle_intents_are_all_limit_taker(self) -> None:
        ctx = _ctx({
            "Up":   [("0.40", "500")],
            "Down": [("0.40", "500")],
        })
        d = evaluate_bundle(_sleeve(min_edge_bps=50), ctx)
        assert len(d.intents) == 2
        for i in d.intents:
            assert i.order_type is OrderType.LIMIT
            assert i.limit_price is not None
            assert i.size_shares is not None
            # All intents in the bundle must have the same share count (guarantee condition).
        assert d.intents[0].size_shares == d.intents[1].size_shares

    def test_bundle_size_capped_by_shallow_leg(self) -> None:
        # One leg has 10 shares of best-ask depth; other has 1000. Bundle must
        # cap at 10 (otherwise the small leg can't fill fully and arb breaks).
        ctx = _ctx({
            "Up":   [("0.40", "10")],
            "Down": [("0.40", "1000")],
        })
        d = evaluate_bundle(_sleeve(min_edge_bps=50, max_pos="1000"), ctx)
        assert len(d.intents) == 2
        for i in d.intents:
            assert i.size_shares == Decimal("10")

    def test_bundle_size_capped_by_bankroll(self) -> None:
        # Deep books, tiny max_position. Should cap at max_position/sum_asks.
        ctx = _ctx({
            "Up":   [("0.40", "10000")],
            "Down": [("0.40", "10000")],
        })
        # max_pos = 10 USD, sum_asks = 0.80, so max 12.5 shares.
        d = evaluate_bundle(_sleeve(min_edge_bps=50, max_pos="10"), ctx)
        assert len(d.intents) == 2
        for i in d.intents:
            assert i.size_shares <= Decimal("13")


class TestMultiOutcome:
    def test_three_outcome_bundle(self) -> None:
        # Three mutually-exclusive outcomes. Sum asks = 0.30 + 0.30 + 0.30 = 0.90.
        # Arb gap = 10% = 1000bps gross.
        ctx = _ctx({
            "A": [("0.30", "100")],
            "B": [("0.30", "100")],
            "C": [("0.30", "100")],
        })
        d = evaluate_bundle(_sleeve(min_edge_bps=50), ctx)
        assert len(d.intents) == 3
        assert d.gap_bps == 1000
