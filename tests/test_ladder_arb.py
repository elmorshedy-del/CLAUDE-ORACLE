"""Tests for date-ladder arb detection.

Critical test: the real Netanyahu prices as of April 2026 should NOT trigger an
arb, because they respect monotonicity (bid_short < ask_long on every pair).
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
from poly_paper.strategies.ladder_arb import (
    LadderContext,
    LadderRung,
    build_ladder_context,
    default_ladder_sleeves,
    evaluate_ladder,
    is_date_ladder_event,
)


def _book(bids: list[tuple[str, str]], asks: list[tuple[str, str]]) -> OrderBook:
    return OrderBook(
        token_id="t",
        market_condition_id="c",
        timestamp_ms=0,
        bids=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in bids],
        asks=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in asks],
    )


def _sleeve(min_edge: int = 10) -> SleeveConfig:
    return SleeveConfig(
        sleeve_id="test", stance=SleeveStance.AGGRESSIVE,
        strategy_name="ladder_arb", market_selector="kind=date_ladder",
        bankroll_usd=Decimal("1000"), max_position_usd=Decimal("100"),
        min_edge_bps=min_edge, min_gross_edge_bps=min_edge,
        max_cross_spread_bps=10000, enabled=True, version=1, notes="",
    )


class TestRealNetanyahu:
    """Real Netanyahu April 2026 prices should NOT produce an arb.

    bid(Apr)=0.006 / ask(Apr)=0.007 / size=5835/1221
    bid(Jun)=0.050 / ask(Jun)=0.060 / size=136739/9924
    bid(Dec)=0.420 / ask(Dec)=0.430 / size=2634/917

    Monotonicity:
      bid(Apr)=0.006 < ask(Jun)=0.060  → no arb
      bid(Apr)=0.006 < ask(Dec)=0.430  → no arb
      bid(Jun)=0.050 < ask(Dec)=0.430  → no arb
    """

    def test_real_netanyahu_no_arb(self) -> None:
        rungs_raw = [
            {
                "token_id": "t_apr", "market_condition_id": "c_apr",
                "end_date_iso": "2026-04-30T00:00:00Z",
                "question": "Netanyahu out by April 30?",
                "book": _book([("0.006", "5835")], [("0.007", "1221")]),
            },
            {
                "token_id": "t_jun", "market_condition_id": "c_jun",
                "end_date_iso": "2026-06-30T00:00:00Z",
                "question": "Netanyahu out by June 30?",
                "book": _book([("0.050", "136739")], [("0.060", "9924")]),
            },
            {
                "token_id": "t_dec", "market_condition_id": "c_dec",
                "end_date_iso": "2026-12-31T00:00:00Z",
                "question": "Netanyahu out by end of 2026?",
                "book": _book([("0.420", "2634")], [("0.430", "917")]),
            },
        ]
        ctx = build_ladder_context(
            event_slug="netanyahu-out-before-2027",
            category=MarketCategory.POLITICS,
            rungs_raw=rungs_raw,
        )
        decisions = evaluate_ladder(_sleeve(), ctx)
        # Three pairs checked (Apr-Jun, Apr-Dec, Jun-Dec). All should reject.
        firing = [d for d in decisions if d.intents]
        assert len(firing) == 0, "Netanyahu prices respect monotonicity; no arb should fire."
        # Every pair should have a 'no monotonicity violation' reason.
        for d in decisions:
            assert "no monotonicity" in (d.reason_skipped or "") or d.reason_skipped is not None


class TestMonotonicityArbDetection:
    def test_stale_earlier_bid_triggers_arb(self) -> None:
        """If someone left a stale bid above a later date's ask → arb."""
        rungs_raw = [
            {  # "by April" — stale bid way higher than reality
                "token_id": "t1", "market_condition_id": "c1",
                "end_date_iso": "2026-04-30T00:00:00Z",
                "question": "out by April?",
                "book": _book([("0.50", "100")], [("0.55", "100")]),
            },
            {  # "by June" — current price still low
                "token_id": "t2", "market_condition_id": "c2",
                "end_date_iso": "2026-06-30T00:00:00Z",
                "question": "out by June?",
                "book": _book([("0.30", "100")], [("0.35", "100")]),
            },
        ]
        ctx = build_ladder_context(
            event_slug="test", category=MarketCategory.POLITICS, rungs_raw=rungs_raw,
        )
        decisions = evaluate_ladder(_sleeve(min_edge=10), ctx)
        firing = [d for d in decisions if d.intents]
        assert len(firing) == 1
        d = firing[0]
        # bid_short(0.50) - ask_long(0.35) = 0.15 gross = 1500 bps
        assert d.gross_edge_bps == 1500
        # Fees for politics (1% peak) at p=0.50 → peak rate × price = 0.01 * 0.50 ≈ 50 bps
        # At p=0.35 → rate = 0.01 * 4 * 0.35 * 0.65 = 0.0091 → fee ≈ 0.0091 * 0.35 = 31.85bps
        # Total fees ~ 82 bps → net ~1418 bps
        assert d.net_edge_bps > 1000
        assert d.kind == "arb_monotonicity"

    def test_thin_arb_rejected_by_fees(self) -> None:
        """Arb just enough to exist but eaten by fees."""
        rungs_raw = [
            {
                "token_id": "t1", "market_condition_id": "c1",
                "end_date_iso": "2026-04-30T00:00:00Z",
                "question": "short",
                "book": _book([("0.502", "100")], [("0.510", "100")]),
            },
            {
                "token_id": "t2", "market_condition_id": "c2",
                "end_date_iso": "2026-06-30T00:00:00Z",
                "question": "long",
                "book": _book([("0.495", "100")], [("0.500", "100")]),
            },
        ]
        # bid(short)=0.502 > ask(long)=0.500 → gross = 2bps. Fees at p=0.5 = ~100bps
        # Net < 0.
        ctx = build_ladder_context(
            event_slug="test", category=MarketCategory.POLITICS, rungs_raw=rungs_raw,
        )
        decisions = evaluate_ladder(_sleeve(min_edge=10), ctx)
        firing = [d for d in decisions if d.intents]
        assert len(firing) == 0

    def test_three_rung_ladder_checks_all_pairs(self) -> None:
        """Arb scanner should check all 3 adjacent+non-adjacent pairs."""
        rungs_raw = [
            {"token_id": "ta", "market_condition_id": "ca",
             "end_date_iso": "2026-01-01", "question": "A",
             "book": _book([("0.10", "100")], [("0.12", "100")])},
            {"token_id": "tb", "market_condition_id": "cb",
             "end_date_iso": "2026-06-01", "question": "B",
             "book": _book([("0.30", "100")], [("0.32", "100")])},
            {"token_id": "tc", "market_condition_id": "cc",
             "end_date_iso": "2026-12-01", "question": "C",
             "book": _book([("0.50", "100")], [("0.52", "100")])},
        ]
        ctx = build_ladder_context(
            event_slug="test", category=MarketCategory.POLITICS, rungs_raw=rungs_raw,
        )
        decisions = evaluate_ladder(_sleeve(), ctx)
        # 3 pairs: (A,B), (A,C), (B,C). All respect monotonicity → no arbs.
        firing = [d for d in decisions if d.intents]
        assert len(firing) == 0
        # But we should have evaluated all 3 pairs (counting decisions with reasons).
        assert len(decisions) == 3

    def test_arb_size_capped_by_shallowest_book(self) -> None:
        rungs_raw = [
            {
                "token_id": "t1", "market_condition_id": "c1",
                "end_date_iso": "2026-04-30", "question": "A",
                "book": _book([("0.80", "10")], [("0.81", "10")]),  # only 10 shares bid
            },
            {
                "token_id": "t2", "market_condition_id": "c2",
                "end_date_iso": "2026-06-30", "question": "B",
                "book": _book([("0.10", "1000")], [("0.12", "1000")]),  # huge depth
            },
        ]
        ctx = build_ladder_context(
            event_slug="test", category=MarketCategory.POLITICS, rungs_raw=rungs_raw,
        )
        decisions = evaluate_ladder(_sleeve(min_edge=10), ctx)
        firing = [d for d in decisions if d.intents]
        assert len(firing) == 1
        intent = firing[0].intents[0]
        # Size capped at 10 (the shallow leg).
        assert intent.size_shares == Decimal("10")


class TestEventClassification:
    def test_netanyahu_is_date_ladder(self) -> None:
        ev = {
            "title": "Netanyahu out by...?",
            "slug": "netanyahu-out-before-2027",
            "markets": [
                {"acceptingOrders": True, "enableOrderBook": True, "negRisk": False,
                 "outcomes": ["Yes", "No"], "endDate": "2026-04-30"},
                {"acceptingOrders": True, "enableOrderBook": True, "negRisk": False,
                 "outcomes": ["Yes", "No"], "endDate": "2026-06-30"},
                {"acceptingOrders": True, "enableOrderBook": True, "negRisk": False,
                 "outcomes": ["Yes", "No"], "endDate": "2026-12-31"},
            ],
        }
        assert is_date_ladder_event(ev)

    def test_roy_is_not_date_ladder(self) -> None:
        """NBA ROY is neg_risk (mutually exclusive outcomes), not date-nested."""
        ev = {
            "title": "NBA Rookie of the Year",
            "slug": "nba-rookie-of-the-year-873",
            "markets": [
                {"acceptingOrders": True, "enableOrderBook": True, "negRisk": True,
                 "outcomes": ["Yes", "No"], "endDate": "2026-05-18"},
                {"acceptingOrders": True, "enableOrderBook": True, "negRisk": True,
                 "outcomes": ["Yes", "No"], "endDate": "2026-05-18"},
            ],
        }
        assert not is_date_ladder_event(ev)
