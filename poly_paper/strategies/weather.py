"""Weather strategy sleeves — temperature and precipitation buckets.

Two market shapes:
  1. TEMPERATURE-ON-DATE:  "Highest temperature in Paris on April 16?"
       N mutually-exclusive buckets ("14°C", "15-16°C", "17°C or above", etc).
       negRisk=True. Short horizon (same-day resolution).
       Open-Meteo's hourly forecast gives us max temperature over the day
       from 40 ensemble members → non-parametric probability per bucket.

  2. PRECIPITATION-MONTH:  "Precipitation in NYC in April?"
       N mutually-exclusive total-rainfall buckets (mm or inches).
       negRisk=True. Longer horizon (end-of-month resolution).
       Open-Meteo ensemble daily precip sum → probability per bucket.

The strategy's fair value for each bucket is the **fraction of ensemble members
whose outcome falls in that bucket.** Compare to `ask` price; if market prices
the bucket substantially lower than our ensemble probability, we have positive
expected value. Standard min_edge_bps gate + fee adjustment.

Why this is likely to have real edge (vs BTC/sports):
  - Polymarket's weather book liquidity is thin ($1k-$40k per bucket).
  - Most Polymarket participants aren't running ensemble forecasts.
  - Open-Meteo's ECMWF-based models are calibrated for short-horizon temperature
    forecasts to within ~1°C — comparable to the bucket width.
  - We compute probability NON-PARAMETRICALLY (counting ensemble members), so
    we don't need to assume Gaussian or parametric errors.

Design:
  - Weather markets use our existing `bundle_arb` layer for bundle-arb detection
    (already works — they're negRisk events).
  - THIS module adds DIRECTIONAL trading: buying individual buckets we think are
    under-priced relative to the ensemble.
  - Sleeves: conservative / balanced / aggressive, same pattern as btc_updown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal, Optional
from uuid import uuid4

from ..exec.fees import CATEGORY_TAKER_PEAK_RATE, MAKER_REBATE_SHARE, taker_fee_rate
from ..exec.models import (
    MarketCategory,
    OrderBook,
    OrderIntent,
    OrderType,
    Side,
    SleeveConfig,
    SleeveStance,
)

# ---------------------------------------------------------------------------
# Question parsing
# ---------------------------------------------------------------------------

# Temperature patterns
_TEMP_BETWEEN_F = re.compile(r"between\s+(\d+)\s*[-–]\s*(\d+)\s*°?\s*F", re.I)
_TEMP_BETWEEN_C = re.compile(r"between\s+(\d+)\s*[-–]\s*(\d+)\s*°?\s*C", re.I)
_TEMP_BE_F = re.compile(r"\b(?:temperature.*?be)\s+(\d+)\s*°?\s*F\b", re.I)
_TEMP_BE_C = re.compile(r"\b(?:temperature.*?be)\s+(\d+)\s*°?\s*C\b", re.I)
_TEMP_OR_BELOW = re.compile(r"(\d+)\s*°?\s*([FC])\s+or\s+below", re.I)
_TEMP_OR_ABOVE = re.compile(r"(\d+)\s*°?\s*([FC])\s+or\s+above", re.I)

# Precipitation patterns
_PRECIP_BETWEEN_MM = re.compile(r"between\s+(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*mm", re.I)
_PRECIP_BETWEEN_IN = re.compile(r"between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s+inches", re.I)
_PRECIP_LESS_MM = re.compile(r"less\s+than\s+(\d+(?:\.\d+)?)\s*mm", re.I)
_PRECIP_LESS_IN = re.compile(r"less\s+than\s+(\d+(?:\.\d+)?)\s+inches", re.I)
_PRECIP_MORE_MM = re.compile(r"(?:more|greater)\s+than\s+(\d+(?:\.\d+)?)\s*mm", re.I)
_PRECIP_MORE_IN = re.compile(r"(?:more|greater)\s+than\s+(\d+(?:\.\d+)?)\s+inches", re.I)


@dataclass(frozen=True)
class WeatherBucket:
    """A single bucket within a weather event.

    Semantics: the outcome variable (max-temp / total-precip) falls within
    [lower_inclusive, upper_exclusive). `None` = open-ended on that side.
    Units normalized: temperature in Celsius, precipitation in millimetres.
    """

    kind: Literal["temperature_max_day", "precipitation_sum_period"]
    lower: float | None     # inclusive
    upper: float | None     # exclusive
    raw_question: str


def _f_to_c(f: float) -> float:
    return (f - 32) * 5 / 9


def _in_to_mm(x: float) -> float:
    return x * 25.4


def parse_temperature_bucket(question: str) -> Optional[WeatherBucket]:
    q = question.strip()

    # "between X-Y °F" or "between X and Y °F"
    m = _TEMP_BETWEEN_F.search(q)
    if m:
        lo_f, hi_f = float(m.group(1)), float(m.group(2))
        # Polymarket's "between 54-55°F" means values in [54, 56) roughly — exclusive
        # upper and inclusive lower is the standard convention for integer temps.
        return WeatherBucket(
            kind="temperature_max_day",
            lower=_f_to_c(lo_f),
            upper=_f_to_c(hi_f + 1),
            raw_question=q,
        )
    m = _TEMP_BETWEEN_C.search(q)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return WeatherBucket(
            kind="temperature_max_day",
            lower=lo, upper=hi + 1,
            raw_question=q,
        )
    # "or below"
    m = _TEMP_OR_BELOW.search(q)
    if m:
        val, unit = float(m.group(1)), m.group(2).upper()
        c = _f_to_c(val + 1) if unit == "F" else (val + 1)
        return WeatherBucket(
            kind="temperature_max_day",
            lower=None, upper=c,
            raw_question=q,
        )
    # "or above"
    m = _TEMP_OR_ABOVE.search(q)
    if m:
        val, unit = float(m.group(1)), m.group(2).upper()
        c = _f_to_c(val) if unit == "F" else val
        return WeatherBucket(
            kind="temperature_max_day",
            lower=c, upper=None,
            raw_question=q,
        )
    # "be X°F"  / "be X°C" — single degree
    m = _TEMP_BE_F.search(q)
    if m:
        val = float(m.group(1))
        lo_c, hi_c = _f_to_c(val), _f_to_c(val + 1)
        return WeatherBucket(
            kind="temperature_max_day",
            lower=lo_c, upper=hi_c,
            raw_question=q,
        )
    m = _TEMP_BE_C.search(q)
    if m:
        val = float(m.group(1))
        return WeatherBucket(
            kind="temperature_max_day",
            lower=val, upper=val + 1,
            raw_question=q,
        )
    return None


def parse_precipitation_bucket(question: str) -> Optional[WeatherBucket]:
    q = question.strip()
    m = _PRECIP_BETWEEN_MM.search(q)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return WeatherBucket(
            kind="precipitation_sum_period",
            lower=lo, upper=hi,
            raw_question=q,
        )
    m = _PRECIP_BETWEEN_IN.search(q)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return WeatherBucket(
            kind="precipitation_sum_period",
            lower=_in_to_mm(lo), upper=_in_to_mm(hi),
            raw_question=q,
        )
    m = _PRECIP_LESS_MM.search(q)
    if m:
        return WeatherBucket(
            kind="precipitation_sum_period",
            lower=None, upper=float(m.group(1)),
            raw_question=q,
        )
    m = _PRECIP_LESS_IN.search(q)
    if m:
        return WeatherBucket(
            kind="precipitation_sum_period",
            lower=None, upper=_in_to_mm(float(m.group(1))),
            raw_question=q,
        )
    m = _PRECIP_MORE_MM.search(q)
    if m:
        return WeatherBucket(
            kind="precipitation_sum_period",
            lower=float(m.group(1)), upper=None,
            raw_question=q,
        )
    m = _PRECIP_MORE_IN.search(q)
    if m:
        return WeatherBucket(
            kind="precipitation_sum_period",
            lower=_in_to_mm(float(m.group(1))), upper=None,
            raw_question=q,
        )
    return None


def parse_weather_bucket(question: str) -> Optional[WeatherBucket]:
    """Try temperature then precipitation patterns."""
    t = parse_temperature_bucket(question)
    if t is not None:
        return t
    return parse_precipitation_bucket(question)


# ---------------------------------------------------------------------------
# City extraction
# ---------------------------------------------------------------------------

_CITY_MAP: dict[str, str] = {
    "nyc": "nyc",
    "new york": "nyc",
    "seattle": "seattle",
    "london": "london",
    "seoul": "seoul",
    "hong kong": "hong_kong",
    "paris": "paris",
}

# Cities we have lat/lon for. Extended list lives in feeds/open_meteo.CITIES.
# Any future additions should update both.


def extract_city(event_title: str, question: str) -> Optional[str]:
    """Pull the city key out of event title or question."""
    blob = f"{event_title} {question}".lower()
    # Longest-first to match 'hong kong' before 'seoul' etc.
    for phrase in sorted(_CITY_MAP.keys(), key=len, reverse=True):
        if phrase in blob:
            return _CITY_MAP[phrase]
    return None


# ---------------------------------------------------------------------------
# Fair-value computation (non-parametric, counting ensemble members)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeatherFairValue:
    probability: float
    # Diagnostics for logs / dashboard.
    ensemble_size: int
    members_in_bucket: int
    bucket: WeatherBucket
    horizon_days: int


def temperature_fair_value(
    ensemble_max_temps_c: list[float],
    bucket: WeatherBucket,
) -> WeatherFairValue:
    """P(max day temperature falls in bucket), from ensemble member daily maxes."""
    if not ensemble_max_temps_c:
        return WeatherFairValue(
            probability=0.0, ensemble_size=0, members_in_bucket=0,
            bucket=bucket, horizon_days=1,
        )
    hits = sum(
        1 for t in ensemble_max_temps_c
        if (bucket.lower is None or t >= bucket.lower)
        and (bucket.upper is None or t < bucket.upper)
    )
    return WeatherFairValue(
        probability=hits / len(ensemble_max_temps_c),
        ensemble_size=len(ensemble_max_temps_c),
        members_in_bucket=hits,
        bucket=bucket,
        horizon_days=1,
    )


def precipitation_fair_value(
    ensemble_totals_mm: list[float],
    bucket: WeatherBucket,
) -> WeatherFairValue:
    """P(cumulative precipitation total falls in bucket), from per-member sums."""
    if not ensemble_totals_mm:
        return WeatherFairValue(
            probability=0.0, ensemble_size=0, members_in_bucket=0,
            bucket=bucket, horizon_days=0,
        )
    hits = sum(
        1 for t in ensemble_totals_mm
        if (bucket.lower is None or t >= bucket.lower)
        and (bucket.upper is None or t < bucket.upper)
    )
    return WeatherFairValue(
        probability=hits / len(ensemble_totals_mm),
        ensemble_size=len(ensemble_totals_mm),
        members_in_bucket=hits,
        bucket=bucket,
        horizon_days=len(ensemble_totals_mm),  # placeholder
    )


# ---------------------------------------------------------------------------
# Sleeve definitions
# ---------------------------------------------------------------------------

def default_weather_sleeves(
    *,
    total_bankroll_usd: Decimal,
) -> list[SleeveConfig]:
    """Three canonical sleeves for the weather strategy."""
    bank = total_bankroll_usd
    return [
        SleeveConfig(
            sleeve_id="weather__conservative",
            stance=SleeveStance.CONSERVATIVE,
            strategy_name="weather",
            market_selector="kind=weather_bucket",
            bankroll_usd=bank,
            max_position_usd=bank * Decimal("0.005"),
            min_edge_bps=500,       # 5% net edge after fees
            min_gross_edge_bps=500,
            max_cross_spread_bps=0,
            enabled=True,
            version=1,
            notes="Weather conservative: 5% net edge threshold, post-only",
        ),
        SleeveConfig(
            sleeve_id="weather__balanced",
            stance=SleeveStance.BALANCED,
            strategy_name="weather",
            market_selector="kind=weather_bucket",
            bankroll_usd=bank,
            max_position_usd=bank * Decimal("0.015"),
            min_edge_bps=300,
            min_gross_edge_bps=300,
            max_cross_spread_bps=100,
            enabled=True,
            version=1,
            notes="Weather balanced: 3% edge threshold",
        ),
        SleeveConfig(
            sleeve_id="weather__aggressive",
            stance=SleeveStance.AGGRESSIVE,
            strategy_name="weather",
            market_selector="kind=weather_bucket",
            bankroll_usd=bank,
            max_position_usd=bank * Decimal("0.03"),
            min_edge_bps=150,
            min_gross_edge_bps=150,
            max_cross_spread_bps=300,
            enabled=True,
            version=1,
            notes="Weather aggressive: 1.5% edge threshold",
        ),
    ]


# ---------------------------------------------------------------------------
# Decision generation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeatherMarketEval:
    """One bucket within a weather event, evaluated for trading."""

    token_id: str
    market_condition_id: str
    bucket: WeatherBucket
    fair_value: WeatherFairValue
    book: OrderBook
    event_title: str


@dataclass(frozen=True)
class WeatherDecision:
    intent: OrderIntent | None
    fair_value: WeatherFairValue
    market_ask: float | None
    gross_edge_bps: int
    net_edge_bps: int
    reason_skipped: str | None


def evaluate_bucket(
    sleeve: SleeveConfig,
    bucket_eval: WeatherMarketEval,
    category: MarketCategory = MarketCategory.WEATHER,
) -> WeatherDecision:
    """Evaluate whether buying this bucket's YES token is positive EV after fees."""
    fv = bucket_eval.fair_value
    book = bucket_eval.book
    if book.best_ask is None:
        return WeatherDecision(
            intent=None, fair_value=fv, market_ask=None,
            gross_edge_bps=0, net_edge_bps=0,
            reason_skipped="no ask",
        )
    ask = float(book.best_ask)
    p = fv.probability
    gross = p - ask
    gross_bps = int(gross * 10000)

    # Taker fee at ask price.
    tfee = float(taker_fee_rate(Decimal(str(ask)), category)) * ask
    # Maker route: one tick above best bid. Bid may be None (empty side).
    best_bid = float(book.best_bid) if book.best_bid is not None else 0.0
    best_ask_side = ask  # already defined
    tick = 0.01
    post_price = min(best_bid + tick, best_ask_side - tick)
    can_post = 0 < post_price < 1

    net_edge_taker = p - ask - tfee
    net_edge_taker_bps = int(net_edge_taker * 10000)

    net_edge_maker = -1.0
    if can_post:
        rebate = float(CATEGORY_TAKER_PEAK_RATE[category]) * 4 * post_price * (1 - post_price) * float(MAKER_REBATE_SHARE) * post_price
        net_edge_maker = (p - post_price) + rebate
    net_edge_maker_bps = int(net_edge_maker * 10000)

    if gross_bps < sleeve.min_gross_edge_bps:
        return WeatherDecision(
            intent=None, fair_value=fv, market_ask=ask,
            gross_edge_bps=gross_bps,
            net_edge_bps=max(net_edge_maker_bps, net_edge_taker_bps),
            reason_skipped=(
                f"gross edge too small: gross={gross_bps}bps < {sleeve.min_gross_edge_bps}bps "
                f"(fv={p:.3f} ask={ask:.3f}, ensemble {fv.members_in_bucket}/{fv.ensemble_size})"
            ),
        )

    # Refuse trades with too-small ensemble (un-calibrated probability).
    if fv.ensemble_size < 10:
        return WeatherDecision(
            intent=None, fair_value=fv, market_ask=ask,
            gross_edge_bps=gross_bps,
            net_edge_bps=max(net_edge_maker_bps, net_edge_taker_bps),
            reason_skipped=f"ensemble too small: {fv.ensemble_size}",
        )

    # Route choice.
    route: str | None = None
    if can_post and net_edge_maker_bps >= sleeve.min_edge_bps:
        route = "maker"
    elif net_edge_taker_bps >= sleeve.min_edge_bps and sleeve.max_cross_spread_bps > 0:
        mid = (best_bid + ask) / 2 if best_bid > 0 else ask
        cross_bps = int((ask - mid) / max(mid, 1e-9) * 10000)
        if cross_bps <= sleeve.max_cross_spread_bps:
            route = "taker"

    if route is None:
        return WeatherDecision(
            intent=None, fair_value=fv, market_ask=ask,
            gross_edge_bps=gross_bps,
            net_edge_bps=max(net_edge_maker_bps, net_edge_taker_bps),
            reason_skipped=(
                f"net edge insufficient: maker={net_edge_maker_bps}bps "
                f"taker={net_edge_taker_bps}bps thresh={sleeve.min_edge_bps}bps"
            ),
        )

    # Size by sleeve.
    size_usd = sleeve.max_position_usd
    if route == "maker":
        order_type = OrderType.POST_ONLY
        limit_price = Decimal(str(post_price))
        used_net = net_edge_maker_bps
        used_price = post_price
    else:
        order_type = OrderType.LIMIT
        limit_price = Decimal(str(ask))
        used_net = net_edge_taker_bps
        used_price = ask

    intent = OrderIntent(
        sleeve_id=sleeve.sleeve_id,
        market_condition_id=bucket_eval.market_condition_id,
        token_id=bucket_eval.token_id,
        side=Side.BUY,
        order_type=order_type,
        limit_price=limit_price,
        size_usd=size_usd,
        category=category,
        edge_bps=used_net,
        reasoning=(
            f"BUY weather bucket via {route}: fv={p:.4f} "
            f"(ensemble {fv.members_in_bucket}/{fv.ensemble_size}), "
            f"ask={ask:.4f}, gross={gross_bps}bps net={used_net}bps. "
            f"Bucket=[{bucket_eval.bucket.lower},{bucket_eval.bucket.upper}) {bucket_eval.bucket.kind}. "
            f"Event: {bucket_eval.event_title[:80]}"
        ),
        client_order_id=f"weather_{uuid4().hex[:12]}",
    )
    return WeatherDecision(
        intent=intent, fair_value=fv, market_ask=ask,
        gross_edge_bps=gross_bps, net_edge_bps=used_net,
        reason_skipped=None,
    )
