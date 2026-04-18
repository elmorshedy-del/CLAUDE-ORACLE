"""Weather strategy runner — concurrent loop evaluating weather-bucket markets.

Runs alongside the BTC tick and arb scanner. Every N seconds:
  1. Pull Polymarket 'weather' tag events that are tradeable.
  2. For each event, parse every market's bucket (temp or precip).
  3. Fetch Open-Meteo ensemble for the city/date.
  4. Compute fair values per bucket.
  5. For each weather sleeve, evaluate every bucket. Execute intents via router.

Design notes:
  - Temperature markets (same-day resolution) are the hot path — short horizon
    = tight ensemble spread = most accurate fair values.
  - Precipitation markets (month totals) have wider uncertainty but also fatter
    buckets, so signal may still be there.
  - We fetch Open-Meteo forecasts PER-CITY, not per-market (cache within a tick).
  - We fetch each event's book once and evaluate all buckets against it.
  - Skip events older than our ensemble horizon (16 days from now).
"""

from __future__ import annotations

import asyncio
import json as _json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import structlog
from sqlalchemy import select

from .db.models import FillRow, OrderIntentRow, SleeveConfig
from .db.session import SessionLocal
from .exec.models import (
    BookLevel,
    ExecutionMode,
    MarketCategory,
    OrderBook,
    SleeveConfig as ExecSleeveConfig,
    SleeveStance,
)
from .exec.router import execute_order
from .feeds.open_meteo import CITIES, OpenMeteo, EnsembleSample
from .strategies.weather import (
    WeatherMarketEval,
    default_weather_sleeves,
    evaluate_bucket,
    extract_city,
    parse_weather_bucket,
    precipitation_fair_value,
    temperature_fair_value,
)

log = structlog.get_logger()

SCAN_INTERVAL_SECONDS = float(os.environ.get("POLY_WEATHER_SCAN_SECONDS", "300"))
WEATHER_ENABLED = os.environ.get("POLY_WEATHER_ENABLED", "1") not in ("0", "false", "no")

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


# ---------------------------------------------------------------------------
# Polymarket fetching
# ---------------------------------------------------------------------------

async def _fetch_weather_events(client: httpx.AsyncClient) -> list[dict]:
    out: list[dict] = []
    try:
        r = await client.get(
            f"{GAMMA}/events",
            params={"active": "true", "closed": "false", "tag_slug": "weather", "limit": 200},
            headers={"User-Agent": "poly-paper/0.3"},
        )
        r.raise_for_status()
        out = r.json()
    except Exception as e:
        log.warning("weather_events_fetch_failed", err=str(e))
    return out or []


async def _fetch_book(client: httpx.AsyncClient, token_id: str) -> OrderBook | None:
    try:
        r = await client.get(f"{CLOB}/book", params={"token_id": token_id})
        r.raise_for_status()
        raw = r.json()
    except Exception:
        return None
    if "error" in raw:
        return None
    bids = sorted(
        [BookLevel(price=Decimal(b["price"]), size=Decimal(b["size"])) for b in raw.get("bids", [])],
        key=lambda lv: lv.price, reverse=True,
    )
    asks = sorted(
        [BookLevel(price=Decimal(a["price"]), size=Decimal(a["size"])) for a in raw.get("asks", [])],
        key=lambda lv: lv.price,
    )
    return OrderBook(
        token_id=token_id,
        market_condition_id=raw.get("market", ""),
        timestamp_ms=int(raw.get("timestamp", "0")),
        bids=bids, asks=asks,
    )


# ---------------------------------------------------------------------------
# Event classification and ensemble extraction
# ---------------------------------------------------------------------------

@dataclass
class WeatherEvent:
    title: str
    slug: str
    end_date: str
    city: str
    markets: list[dict]   # Polymarket market dicts
    target_date: str      # ISO YYYY-MM-DD — what day we need forecast for
    is_month_aggregate: bool  # True for precipitation-month events


def _classify_event(ev: dict) -> WeatherEvent | None:
    mkts = ev.get("markets") or []
    tradeable = [m for m in mkts if m.get("acceptingOrders") and m.get("enableOrderBook")]
    if len(tradeable) < 2:
        return None
    # Only take events where every market parses to a bucket.
    buckets_found = 0
    for m in tradeable:
        if parse_weather_bucket(m.get("question", "")) is not None:
            buckets_found += 1
    if buckets_found < len(tradeable) / 2:
        return None
    # City.
    city = extract_city(ev.get("title", ""), tradeable[0].get("question", "") if tradeable else "")
    if city not in CITIES:
        return None
    # Target date. For "highest temp in X on April 17?" use endDate or parse.
    end_date = ev.get("endDate", "") or (tradeable[0].get("endDate", "") if tradeable else "")
    end_date = end_date[:10] if end_date else ""
    # Month aggregate = "Precipitation in X in April" (title mentions month, not date).
    title_lower = ev.get("title", "").lower()
    is_month_aggregate = "in april" in title_lower or "in may" in title_lower or "in june" in title_lower or "in march" in title_lower
    return WeatherEvent(
        title=ev.get("title", ""),
        slug=ev.get("slug", ""),
        end_date=end_date,
        city=city,
        markets=tradeable,
        target_date=end_date,
        is_month_aggregate=is_month_aggregate,
    )


def _extract_max_temps_for_date(ensemble: list[EnsembleSample], target_date: str) -> list[float]:
    """Return daily tmax_c (Celsius) for `target_date` across ensemble members."""
    out = []
    for s in ensemble:
        for d in s.daily:
            if d.date == target_date and d.tmax_c is not None:
                out.append(d.tmax_c)
                break
    return out


def _extract_precip_totals_for_range(
    ensemble: list[EnsembleSample],
    start_date: str,
    end_date: str,
) -> list[float]:
    """Return per-member cumulative precip (mm) over [start_date, end_date]."""
    out = []
    for s in ensemble:
        total = 0.0
        any_found = False
        for d in s.daily:
            if start_date <= d.date <= end_date and d.precip_mm is not None:
                total += d.precip_mm
                any_found = True
        if any_found:
            out.append(total)
    return out


# ---------------------------------------------------------------------------
# One event evaluation
# ---------------------------------------------------------------------------

async def _evaluate_event(
    event: WeatherEvent,
    sleeves: list[ExecSleeveConfig],
    om: OpenMeteo,
    client: httpx.AsyncClient,
) -> int:
    """Returns number of intents fired."""
    # 1. Pull ensemble for the city.
    city_info = {k: v for k, v in CITIES[event.city].items() if k != "station"}
    try:
        ensemble = await om.ensemble_daily(**city_info, days=16)
    except Exception as e:
        log.warning("weather_ensemble_fetch_failed", city=event.city, err=str(e))
        return 0
    if not ensemble or len(ensemble) < 10:
        log.debug("weather_ensemble_too_small", city=event.city, n=len(ensemble))
        return 0

    # 2. Determine the ensemble slice for each market's bucket.
    if event.is_month_aggregate:
        # For April events: start = first day of month, end = target_date.
        today = datetime.now(timezone.utc).date()
        # Use the minimum of today and the event end_date as range start to avoid
        # double-counting observed-so-far precip (which we'd need historical data for).
        # For now: sum from today forward to end_date.
        start_d = today.isoformat()
        end_d = event.target_date or start_d
        precip_totals = _extract_precip_totals_for_range(ensemble, start_d, end_d)
        max_temps = None
    else:
        # Same-day temperature event.
        if not event.target_date:
            return 0
        max_temps = _extract_max_temps_for_date(ensemble, event.target_date)
        precip_totals = None

    if max_temps is not None and len(max_temps) < 10:
        log.debug("weather_temp_ensemble_too_small", city=event.city, date=event.target_date, n=len(max_temps))
        return 0
    if precip_totals is not None and len(precip_totals) < 10:
        log.debug("weather_precip_ensemble_too_small", city=event.city, n=len(precip_totals))
        return 0

    # 3. For each market in the event: parse bucket, fetch book, compute fv, evaluate across sleeves.
    fired = 0
    for m in event.markets:
        bucket = parse_weather_bucket(m.get("question", ""))
        if bucket is None:
            continue
        clob = m.get("clobTokenIds")
        if isinstance(clob, str):
            clob = _json.loads(clob)
        outcomes = m.get("outcomes")
        if isinstance(outcomes, str):
            outcomes = _json.loads(outcomes)
        if not clob or not outcomes or len(clob) < 2:
            continue
        # Get YES token id.
        yes_idx = 0
        for i, o in enumerate(outcomes):
            if o == "Yes":
                yes_idx = i
                break
        yes_token = clob[yes_idx]
        book = await _fetch_book(client, yes_token)
        if book is None:
            continue

        # Compute fair value.
        if bucket.kind == "temperature_max_day":
            fv = temperature_fair_value(max_temps or [], bucket)
        else:
            fv = precipitation_fair_value(precip_totals or [], bucket)

        eval_ = WeatherMarketEval(
            token_id=yes_token,
            market_condition_id=m.get("conditionId", ""),
            bucket=bucket,
            fair_value=fv,
            book=book,
            event_title=event.title,
        )

        # Try each sleeve.
        for sleeve in sleeves:
            decision = evaluate_bucket(sleeve, eval_, category=MarketCategory.WEATHER)
            if decision.intent is None:
                log.debug(
                    "weather_skip",
                    sleeve=sleeve.sleeve_id, ev_title=event.title[:40],
                    fv=round(fv.probability, 3),
                    ask=decision.market_ask,
                    gross_bps=decision.gross_edge_bps,
                    reason=decision.reason_skipped,
                )
                continue

            async with SessionLocal() as db:
                db.add(OrderIntentRow(
                    client_order_id=decision.intent.client_order_id,
                    sleeve_id=decision.intent.sleeve_id,
                    market_condition_id=decision.intent.market_condition_id,
                    token_id=decision.intent.token_id,
                    side=decision.intent.side.value,
                    order_type=decision.intent.order_type.value,
                    limit_price=str(decision.intent.limit_price) if decision.intent.limit_price else None,
                    size_usd=str(decision.intent.size_usd) if decision.intent.size_usd else None,
                    size_shares=str(decision.intent.size_shares) if decision.intent.size_shares else None,
                    edge_bps=decision.intent.edge_bps,
                    category=decision.intent.category.value,
                    reasoning=decision.intent.reasoning,
                ))
                fill = await execute_order(
                    decision.intent, mode=ExecutionMode.PAPER,
                    book=book, category=MarketCategory.WEATHER,
                )
                db.add(FillRow(
                    fill_id=fill.fill_id,
                    client_order_id=decision.intent.client_order_id,
                    mode=fill.mode.value,
                    rejected=fill.rejected,
                    filled_size_shares=str(fill.filled_size_shares),
                    avg_price=str(fill.avg_price) if fill.avg_price is not None else None,
                    notional_usd=str(fill.notional_usd),
                    fees_usd=str(fill.fees_usd),
                    gas_usd=str(fill.gas_usd),
                    confidence=fill.confidence.value,
                    slippage_bps=fill.slippage_bps,
                    latency_ms=fill.latency_ms,
                    legs_json=[
                        {"price": str(l.price), "size_shares": str(l.size_shares), "role": l.role}
                        for l in fill.legs
                    ],
                    notes=fill.notes,
                ))
                await db.commit()

            log.info(
                "weather_fired",
                sleeve=sleeve.sleeve_id,
                event=event.title[:60],
                bucket=bucket.raw_question[:55],
                fv=round(fv.probability, 3),
                ask=decision.market_ask,
                gross_bps=decision.gross_edge_bps,
                net_bps=decision.net_edge_bps,
            )
            fired += 1
            # One sleeve per bucket — don't stack multiple sleeves on the same position.
            break
    return fired


# ---------------------------------------------------------------------------
# Sleeve seeding helper — called from runner.ensure_sleeves_seeded()
# ---------------------------------------------------------------------------

async def ensure_weather_sleeves_seeded(total_bankroll_usd: Decimal) -> None:
    if not WEATHER_ENABLED:
        return
    async with SessionLocal() as db:
        for exec_cfg in default_weather_sleeves(total_bankroll_usd=total_bankroll_usd):
            existing = (await db.execute(
                select(SleeveConfig).where(SleeveConfig.sleeve_id == exec_cfg.sleeve_id)
            )).scalar_one_or_none()
            if existing:
                continue
            db.add(SleeveConfig(
                sleeve_id=exec_cfg.sleeve_id,
                stance=exec_cfg.stance.value,
                strategy_name=exec_cfg.strategy_name,
                market_selector=exec_cfg.market_selector,
                bankroll_usd=str(exec_cfg.bankroll_usd),
                max_position_usd=str(exec_cfg.max_position_usd),
                min_edge_bps=exec_cfg.min_edge_bps,
                max_cross_spread_bps=exec_cfg.max_cross_spread_bps,
                enabled=exec_cfg.enabled,
                version=exec_cfg.version,
                notes=exec_cfg.notes,
                extra_json={"min_gross_edge_bps": exec_cfg.min_gross_edge_bps},
            ))
        await db.commit()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def scan_once() -> dict:
    from .runner import _load_exec_sleeve

    async with SessionLocal() as db:
        rows = (await db.execute(
            select(SleeveConfig).where(
                SleeveConfig.enabled.is_(True),
                SleeveConfig.strategy_name == "weather",
            )
        )).scalars().all()
    sleeves = [_load_exec_sleeve(r) for r in rows]
    if not sleeves:
        return {"sleeves": 0, "events": 0, "fired": 0, "elapsed_sec": 0}

    t0 = time.time()
    async with httpx.AsyncClient(timeout=10, limits=httpx.Limits(max_connections=10)) as client:
        raw_events = await _fetch_weather_events(client)
        classified = [c for c in (_classify_event(ev) for ev in raw_events) if c is not None]
        log.info(
            "weather_scan_started",
            raw_events=len(raw_events),
            classified=len(classified),
        )
        async with OpenMeteo() as om:
            fired_total = 0
            # Sequentially across events — don't hammer Open-Meteo.
            for ev in classified:
                try:
                    fired = await _evaluate_event(ev, sleeves, om, client)
                    fired_total += fired
                except Exception as e:
                    log.warning("weather_event_failed", slug=ev.slug, err=str(e))
    return {
        "elapsed_sec": round(time.time() - t0, 2),
        "events_evaluated": len(classified),
        "intents_fired": fired_total,
    }


async def run_weather_forever() -> None:
    if not WEATHER_ENABLED:
        log.info("weather_disabled")
        return
    while True:
        try:
            stats = await scan_once()
            log.info("weather_scan_done", **stats)
        except Exception as e:
            log.error("weather_scan_failed", err=str(e))
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)
