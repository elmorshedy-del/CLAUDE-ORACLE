"""Tests for self-correction proposer.

These tests verify the CRITICAL safety property: self-correction NEVER proposes
changes outside hard bounds. Auto-applied changes stay inside the default
envelope; anything outside is flagged for manual review.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from poly_paper.selfcorrect import (
    HARD_CEILING_POSITION_PCT,
    HARD_FLOOR_MIN_EDGE_BPS,
    HARD_FLOOR_MIN_GROSS_BPS,
    ProposedChange,
    SleeveMetrics,
    _classify_status,
    _classify_status_pos,
    propose_changes,
)


def _sm(**kwargs) -> SleeveMetrics:
    base = dict(
        sleeve_id="btc_up_down_5m__balanced",
        strategy_name="btc_up_down",
        stance="balanced",
        version=1,
        current_min_edge_bps=150,
        current_min_gross_bps=150,
        current_max_position_usd=Decimal("15"),
        current_max_cross_spread_bps=50,
        current_bankroll_usd=Decimal("1000"),
    )
    base.update(kwargs)
    return SleeveMetrics(**base)


class TestDecisionRules:
    def test_no_fills_no_proposals(self) -> None:
        """Quiet system with 0 data → no changes."""
        m = _sm(n_intents=0, n_fills=0, n_rejected=0)
        assert len(propose_changes([m])) == 0

    def test_bad_fill_quality_tightens_cross_spread(self) -> None:
        """Sleeve with many LOW-confidence fills should get cross spread tightened."""
        m = _sm(n_fills=150, n_high_conf_fills=30)  # 20% high-conf
        proposals = propose_changes([m])
        assert len(proposals) >= 1
        cross_change = next((p for p in proposals if p.field_name == "max_cross_spread_bps"), None)
        assert cross_change is not None
        assert cross_change.new_value < cross_change.old_value

    def test_good_fill_quality_no_change(self) -> None:
        m = _sm(n_fills=150, n_high_conf_fills=140)  # 93% high-conf
        proposals = propose_changes([m])
        cross_changes = [p for p in proposals if p.field_name == "max_cross_spread_bps"]
        assert len(cross_changes) == 0

    def test_high_rejection_reduces_position(self) -> None:
        """High rejection rate should shrink max_position_usd."""
        m = _sm(n_fills=30, n_rejected=20)  # 40% rejection
        proposals = propose_changes([m])
        pos_change = next((p for p in proposals if p.field_name == "max_position_usd"), None)
        assert pos_change is not None
        assert Decimal(str(pos_change.new_value)) < m.current_max_position_usd

    def test_large_realized_slippage_flags_for_review(self) -> None:
        """Slippage eating >50% of intent edge = manual review flag."""
        m = _sm(n_fills=60, mean_intent_edge_bps=200.0, mean_realised_slippage_bps=150.0)
        proposals = propose_changes([m])
        flagged = [p for p in proposals if p.status == "flagged"]
        assert len(flagged) >= 1
        assert any(p.field_name == "_review_required" for p in flagged)

    def test_unknown_sleeve_skipped(self) -> None:
        """No default config known → refuse to propose anything."""
        m = _sm(sleeve_id="totally__unknown", n_fills=150, n_high_conf_fills=20)
        proposals = propose_changes([m])
        assert len(proposals) == 0


class TestHardBounds:
    def test_classify_respects_hard_floor(self) -> None:
        """An int change below hard floor is flagged, not applied."""
        c = ProposedChange(
            sleeve_id="x", field_name="min_edge_bps",
            old_value=150, new_value=10,  # way below floor=25
            rationale="test",
        )
        status = _classify_status(c, default_value=150, bound=2.0, hard_floor=HARD_FLOOR_MIN_EDGE_BPS, hard_ceiling=None)
        assert status == "flagged"

    def test_classify_respects_bound_envelope(self) -> None:
        """Change outside the default envelope is flagged."""
        # 3x the default — above BOUND=2.0
        c = ProposedChange(
            sleeve_id="x", field_name="max_cross_spread_bps",
            old_value=50, new_value=300,
            rationale="test",
        )
        status = _classify_status(c, default_value=100, bound=2.0, hard_floor=0, hard_ceiling=None)
        assert status == "flagged"

    def test_classify_inside_bounds_is_applied(self) -> None:
        c = ProposedChange(
            sleeve_id="x", field_name="max_cross_spread_bps",
            old_value=100, new_value=50,
            rationale="test",
        )
        status = _classify_status(c, default_value=100, bound=2.0, hard_floor=0, hard_ceiling=None)
        assert status == "applied"

    def test_pos_above_hard_ceiling_flagged(self) -> None:
        """Position above 5% of bankroll is flagged."""
        bankroll = Decimal("1000")
        # 6% position = $60 > $50 hard ceiling
        c = ProposedChange(
            sleeve_id="x", field_name="max_position_usd",
            old_value=Decimal("15"), new_value=Decimal("60"),
            rationale="test",
        )
        status = _classify_status_pos(c, default_value=Decimal("15"), hard_ceiling=bankroll * HARD_CEILING_POSITION_PCT)
        assert status == "flagged"


class TestMetricsProperties:
    def test_high_conf_fraction(self) -> None:
        m = _sm(n_fills=100, n_high_conf_fills=80)
        assert m.high_conf_fraction == 0.8

    def test_rejection_rate(self) -> None:
        m = _sm(n_fills=30, n_rejected=10)
        assert m.rejection_rate == 0.25

    def test_zero_denominator_safety(self) -> None:
        m = _sm(n_fills=0, n_rejected=0)
        assert m.high_conf_fraction == 0.0
        assert m.rejection_rate == 0.0
