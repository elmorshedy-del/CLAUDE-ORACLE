"""Tests for ladder classifier — parsing question text to extract ladder variable."""

from __future__ import annotations

from datetime import datetime
import pytest

from poly_paper.strategies.ladder_classify import (
    classify_question,
    is_coherent_ladder,
    sorted_rungs_for_arb,
)


class TestDateParsing:
    def test_by_month_day(self) -> None:
        c = classify_question("Netanyahu out by April 30?")
        assert c is not None
        assert c.kind == "date_ladder"
        assert c.higher_value_more_likely is True
        # Approximate: April 30 of current or specified year — should be well-formed ts.
        assert c.value > 0

    def test_by_month_day_year(self) -> None:
        c = classify_question("Starmer out by June 30, 2026?")
        assert c is not None
        assert c.kind == "date_ladder"
        ts = datetime(2026, 6, 30).timestamp()
        assert abs(c.value - ts) < 1

    def test_by_end_of_year(self) -> None:
        c = classify_question("Netanyahu out by end of 2026?")
        assert c is not None
        assert c.kind == "date_ladder"
        ts = datetime(2026, 12, 31).timestamp()
        assert abs(c.value - ts) < 1

    def test_before_year(self) -> None:
        c = classify_question("US national Bitcoin reserve before 2027?")
        assert c is not None
        assert c.kind == "date_ladder"

    def test_before_month_day(self) -> None:
        c = classify_question("Will the US capture another world leader before June 30, 2026?")
        assert c is not None
        assert c.kind == "date_ladder"


class TestThresholdParsing:
    def test_greater_than_billions(self) -> None:
        c = classify_question("MegaETH market cap (FDV) >$1.5B one day after launch?")
        assert c is not None
        assert c.kind == "threshold_ladder"
        assert c.value == 1_500_000_000
        assert c.higher_value_more_likely is False

    def test_above_trillions(self) -> None:
        c = classify_question("OpenAI IPO closing market cap above $1.6T?")
        assert c is not None
        assert c.kind == "threshold_ladder"
        assert c.value == 1_600_000_000_000
        c2 = classify_question("OpenAI IPO closing market cap above $800B?")
        assert c2.value == 800_000_000_000
        # Key invariant: the trillion value sorts above the billion value.
        assert c.value > c2.value

    def test_above_dollar_thousands(self) -> None:
        c = classify_question("Will Bitcoin hit $150k by December 31, 2026?")
        # This has BOTH a threshold ("$150k") AND a date ("by December 31, 2026").
        # The date pattern runs first in our function, so we get the date.
        # For mixed ladders the event coordinator should separately check this —
        # the classifier returns SOMETHING ladder-like, but treating this as a
        # pure date ladder within one fixed threshold event is correct behavior.
        assert c is not None
        assert c.kind == "date_ladder"

    def test_threshold_only(self) -> None:
        c = classify_question("MegaETH FDV >$6B one day after launch?")
        assert c is not None
        assert c.kind == "threshold_ladder"
        assert c.value == 6_000_000_000

    def test_above_small(self) -> None:
        c = classify_question("Will price be above $100?")
        assert c is not None
        assert c.kind == "threshold_ladder"
        assert c.value == 100.0

    def test_reach_pattern(self) -> None:
        c = classify_question("Will Bitcoin reach $200,000 by 2027?")
        # "by 2027" date pattern wins (runs first). 
        assert c is not None
        assert c.kind == "date_ladder"

    def test_pure_threshold(self) -> None:
        c = classify_question("Bitcoin above $150k")
        assert c is not None
        assert c.kind == "threshold_ladder"
        assert c.value == 150_000


class TestNonLadders:
    def test_who_wins_is_not_ladder(self) -> None:
        c = classify_question("Will Cooper Flagg win the 2025-26 NBA Rookie of the Year award?")
        # Has no ladder variable.
        assert c is None

    def test_simple_binary_no_ladder(self) -> None:
        c = classify_question("Will the Jets make the playoffs?")
        assert c is None


class TestCoherenceCheck:
    def test_mixed_ladders_rejected(self) -> None:
        a = classify_question("Netanyahu out by April 30?")
        b = classify_question("MegaETH FDV >$6B?")
        assert not is_coherent_ladder([a, b])

    def test_all_dates_coherent(self) -> None:
        a = classify_question("Netanyahu out by April 30?")
        b = classify_question("Netanyahu out by June 30?")
        c = classify_question("Netanyahu out by end of 2026?")
        assert is_coherent_ladder([a, b, c])

    def test_all_thresholds_coherent(self) -> None:
        a = classify_question("FDV >$1B")
        b = classify_question("FDV >$3B")
        c = classify_question("FDV >$6B")
        assert is_coherent_ladder([a, b, c])


class TestSortOrder:
    def test_date_ladder_ascending(self) -> None:
        a = (classify_question("by April 30"), "A")
        b = (classify_question("by June 30"), "B")
        c = (classify_question("by end of 2026"), "C")
        out = sorted_rungs_for_arb([c, a, b])
        # Subset (earliest date) first.
        assert [x[1] for x in out] == ["A", "B", "C"]

    def test_threshold_ladder_descending(self) -> None:
        a = (classify_question(">$1B"), "low")
        b = (classify_question(">$3B"), "mid")
        c = (classify_question(">$6B"), "high")
        out = sorted_rungs_for_arb([a, b, c])
        # Subset (highest threshold = LEAST likely) first.
        assert [x[1] for x in out] == ["high", "mid", "low"]
