"""Tests for weather strategy."""

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
from poly_paper.strategies.weather import (
    WeatherBucket,
    WeatherFairValue,
    WeatherMarketEval,
    default_weather_sleeves,
    evaluate_bucket,
    extract_city,
    parse_precipitation_bucket,
    parse_temperature_bucket,
    parse_weather_bucket,
    precipitation_fair_value,
    temperature_fair_value,
)


# ---------------------------------------------------------------------------
# Parser tests — against REAL Polymarket question formats
# ---------------------------------------------------------------------------

class TestTemperatureParsing:
    def test_between_fahrenheit(self) -> None:
        b = parse_temperature_bucket("Will the highest temperature in Seattle be between 54-55°F on April 17?")
        assert b is not None
        assert b.kind == "temperature_max_day"
        # 54F = 12.22C; 56F (upper exclusive) = 13.33C
        assert abs(b.lower - 12.222) < 0.01
        assert abs(b.upper - 13.333) < 0.01

    def test_single_celsius(self) -> None:
        b = parse_temperature_bucket("Will the highest temperature in Hong Kong be 22°C on April 17?")
        assert b is not None
        assert b.lower == 22.0
        assert b.upper == 23.0

    def test_or_below_celsius(self) -> None:
        b = parse_temperature_bucket("Will the highest temperature in Hong Kong be 21°C or below on April 17?")
        assert b is not None
        assert b.lower is None
        assert b.upper == 22.0  # "21 or below" = anything <22

    def test_or_above_fahrenheit(self) -> None:
        b = parse_temperature_bucket("Will the highest temperature in Seattle be 70°F or above on April 17?")
        assert b is not None
        # 70F = 21.111C
        assert abs(b.lower - 21.111) < 0.01
        assert b.upper is None

    def test_non_temperature_returns_none(self) -> None:
        assert parse_temperature_bucket("Will Cooper Flagg win Rookie of the Year?") is None


class TestPrecipitationParsing:
    def test_between_inches(self) -> None:
        b = parse_precipitation_bucket("Will NYC have between 2 and 3 inches of precipitation in April?")
        assert b is not None
        assert b.kind == "precipitation_sum_period"
        # 2 inches = 50.8mm; 3 inches = 76.2mm
        assert abs(b.lower - 50.8) < 0.01
        assert abs(b.upper - 76.2) < 0.01

    def test_between_mm(self) -> None:
        b = parse_precipitation_bucket("Will London have between 20-30mm of precipitation in April?")
        assert b is not None
        assert b.lower == 20.0
        assert b.upper == 30.0

    def test_less_than_mm(self) -> None:
        b = parse_precipitation_bucket("Will London have less than 20mm of precipitation in April?")
        assert b is not None
        assert b.lower is None
        assert b.upper == 20.0

    def test_more_than_inches(self) -> None:
        b = parse_precipitation_bucket("Will NYC have more than 6 inches of precipitation in April?")
        assert b is not None
        # 6 inches = 152.4mm
        assert abs(b.lower - 152.4) < 0.01
        assert b.upper is None


class TestCityExtraction:
    def test_paris(self) -> None:
        assert extract_city("Highest temperature in Paris on April 16?", "Will the highest temperature in Paris be 14°C on April 16?") == "paris"

    def test_hong_kong_matches_before_seoul(self) -> None:
        # "hong kong" must be extracted even though "k" appears in both — longest first.
        assert extract_city("", "Will the highest temperature in Hong Kong be 22°C on April 17?") == "hong_kong"

    def test_seattle(self) -> None:
        assert extract_city("Highest temperature in Seattle on April 17?", "") == "seattle"

    def test_not_found(self) -> None:
        assert extract_city("Polymarket revenue by 2027", "Will Polymarket revenue hit $5M?") is None


# ---------------------------------------------------------------------------
# Fair-value computation
# ---------------------------------------------------------------------------

class TestTemperatureFairValue:
    def test_fraction_in_bucket(self) -> None:
        # 10 ensemble members, 4 in [15, 17)
        members = [14.0, 14.9, 15.0, 15.5, 16.0, 16.9, 17.0, 17.5, 18.0, 20.0]
        b = WeatherBucket(kind="temperature_max_day", lower=15.0, upper=17.0, raw_question="")
        fv = temperature_fair_value(members, b)
        assert fv.probability == 0.4
        assert fv.ensemble_size == 10
        assert fv.members_in_bucket == 4

    def test_open_upper_bound(self) -> None:
        members = [15.0, 20.0, 25.0]
        b = WeatherBucket(kind="temperature_max_day", lower=20.0, upper=None, raw_question="")
        fv = temperature_fair_value(members, b)
        assert fv.probability == pytest.approx(2 / 3)

    def test_empty_ensemble(self) -> None:
        b = WeatherBucket(kind="temperature_max_day", lower=10.0, upper=20.0, raw_question="")
        fv = temperature_fair_value([], b)
        assert fv.probability == 0.0
        assert fv.ensemble_size == 0


class TestPrecipitationFairValue:
    def test_fraction_in_bucket(self) -> None:
        # 10 member totals, 3 in [50, 100)
        members = [20, 40, 55, 70, 95, 100, 120, 150, 200, 300]
        b = WeatherBucket(kind="precipitation_sum_period", lower=50.0, upper=100.0, raw_question="")
        fv = precipitation_fair_value(members, b)
        assert fv.probability == 0.3
        assert fv.members_in_bucket == 3


# ---------------------------------------------------------------------------
# Decision tests
# ---------------------------------------------------------------------------

def _book(bids: list[tuple[str, str]], asks: list[tuple[str, str]]) -> OrderBook:
    return OrderBook(
        token_id="t", market_condition_id="c", timestamp_ms=0,
        bids=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in bids],
        asks=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in asks],
    )


def _sleeve(min_edge: int = 300, max_cross: int = 100) -> SleeveConfig:
    return SleeveConfig(
        sleeve_id="weather__test",
        stance=SleeveStance.BALANCED,
        strategy_name="weather",
        market_selector="kind=weather_bucket",
        bankroll_usd=Decimal("1000"),
        max_position_usd=Decimal("15"),
        min_edge_bps=min_edge,
        min_gross_edge_bps=min_edge,
        max_cross_spread_bps=max_cross,
        enabled=True, version=1, notes="",
    )


class TestDecisionGating:
    def test_positive_edge_triggers_maker(self) -> None:
        # fv=0.25, ask=0.18 → gross 7% = 700bps
        members = [15.0] * 10 + [20.0] * 30  # 10/40 = 25% in bucket [15, 17)
        b = WeatherBucket(kind="temperature_max_day", lower=15.0, upper=17.0, raw_question="q")
        fv = temperature_fair_value(members, b)
        book = _book([("0.17", "100")], [("0.18", "100")])
        eval_ = WeatherMarketEval(
            token_id="tok", market_condition_id="cond", bucket=b,
            fair_value=fv, book=book, event_title="test",
        )
        d = evaluate_bucket(_sleeve(min_edge=300), eval_)
        assert d.intent is not None
        assert d.intent.order_type == OrderType.POST_ONLY
        # Maker price = bid+tick = 0.18 (then min with ask-tick=0.17) = 0.17
        assert abs(float(d.intent.limit_price) - 0.17) < 1e-6

    def test_zero_edge_skipped(self) -> None:
        members = [15.0] * 5 + [20.0] * 5
        b = WeatherBucket(kind="temperature_max_day", lower=15.0, upper=17.0, raw_question="q")
        fv = temperature_fair_value(members, b)
        book = _book([("0.49", "100")], [("0.52", "100")])  # ask > fv (0.50)
        eval_ = WeatherMarketEval(
            token_id="tok", market_condition_id="cond", bucket=b,
            fair_value=fv, book=book, event_title="test",
        )
        d = evaluate_bucket(_sleeve(), eval_)
        assert d.intent is None
        assert "gross edge too small" in (d.reason_skipped or "")

    def test_small_ensemble_refused(self) -> None:
        # 5 members is below our safety threshold of 10.
        members = [15.0, 16.0, 15.5, 16.5, 15.8]
        b = WeatherBucket(kind="temperature_max_day", lower=15.0, upper=17.0, raw_question="q")
        fv = temperature_fair_value(members, b)  # all 5 in bucket → p=1.0
        book = _book([("0.10", "100")], [("0.15", "100")])
        eval_ = WeatherMarketEval(
            token_id="tok", market_condition_id="cond", bucket=b,
            fair_value=fv, book=book, event_title="test",
        )
        d = evaluate_bucket(_sleeve(), eval_)
        assert d.intent is None
        assert "ensemble too small" in (d.reason_skipped or "")

    def test_no_ask_skipped(self) -> None:
        members = [15.0] * 40
        b = WeatherBucket(kind="temperature_max_day", lower=15.0, upper=17.0, raw_question="q")
        fv = temperature_fair_value(members, b)
        book = _book([("0.50", "100")], [])  # empty ask side
        eval_ = WeatherMarketEval(
            token_id="tok", market_condition_id="cond", bucket=b,
            fair_value=fv, book=book, event_title="test",
        )
        d = evaluate_bucket(_sleeve(), eval_)
        assert d.intent is None


class TestDefaultSleeves:
    def test_three_stances_created(self) -> None:
        sleeves = default_weather_sleeves(total_bankroll_usd=Decimal("1000"))
        assert len(sleeves) == 3
        stances = {s.stance for s in sleeves}
        assert stances == {SleeveStance.CONSERVATIVE, SleeveStance.BALANCED, SleeveStance.AGGRESSIVE}
        # Conservative < balanced < aggressive in risk.
        c = next(s for s in sleeves if s.stance == SleeveStance.CONSERVATIVE)
        b = next(s for s in sleeves if s.stance == SleeveStance.BALANCED)
        a = next(s for s in sleeves if s.stance == SleeveStance.AGGRESSIVE)
        assert c.min_edge_bps > b.min_edge_bps > a.min_edge_bps
        assert c.max_position_usd < b.max_position_usd < a.max_position_usd
