"""Weather outcome resolver.

Runs hourly (cheap). For each unresolved WeatherForecastRecord:
  1. Check if Polymarket's market has resolved (via Gamma API `closed=true`).
  2. Fetch the YES token's resolved price — if $1.00, bucket was hit.
  3. Record observed_value by fetching Open-Meteo's HISTORICAL endpoint for
     that city+date (gives us the actual observed tmax or precip total that
     NGR needs for training).
  4. Call `attach_outcome()` to populate observed_outcome, observed_value,
     resolved_at across every forecast row for that token.

Why we need BOTH the boolean outcome AND the observed value:
  - Boolean outcome (bucket hit?) → used by calibration Brier/reliability.
  - Observed value (actual tmax_c or precip_mm) → used by NGR training.

This is the data flywheel: each resolved market contributes a training sample
to NGR (per-city-per-kind) AND a calibration point to Brier/BSS tracking.
Run it on a schedule and the measurement + adjustment system is self-feeding.
"""

from __future__ import annotations

import asyncio
import json as _json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import structlog
from sqlalchemy import func, select

from .db.session import SessionLocal
from .feeds.open_meteo import CITIES
from .weather_calibration import WeatherForecastRecord, attach_outcome

log = structlog.get_logger()

RESOLVE_INTERVAL_SECONDS = float(os.environ.get("POLY_RESOLVE_INTERVAL_SEC", "3600"))  # hourly
ENABLED = os.environ.get("POLY_RESOLVE_ENABLED", "1") not in ("0", "false", "no")

GAMMA = "https://gamma-api.polymarket.com"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------------------------
# Polymarket resolution check
# ---------------------------------------------------------------------------

async def _market_resolved(
    client: httpx.AsyncClient, condition_id: str,
) -> Optional[tuple[bool, str | None]]:
    """Return (yes_won, resolution_date_iso) or None if not yet resolved / unknown.

    We query Gamma for the market and check `closed=True` + `outcomePrices`.
    If closed and outcomePrices = ["1", "0"] (or ["0", "1"]), we know the result.
    """
    try:
        r = await client.get(
            f"{GAMMA}/markets", params={"condition_ids": condition_id},
            headers={"User-Agent": "poly-paper/0.3"},
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.debug("resolve_fetch_failed", cond=condition_id[:20], err=str(e))
        return None
    if not data:
        return None
    m = data[0] if isinstance(data, list) else data
    if not m.get("closed"):
        return None
    # Outcome prices: [yes_resolved_price, no_resolved_price] usually
    prices = m.get("outcomePrices")
    if isinstance(prices, str):
        prices = _json.loads(prices)
    if not prices or len(prices) < 2:
        return None
    outcomes = m.get("outcomes")
    if isinstance(outcomes, str):
        outcomes = _json.loads(outcomes)
    yes_idx = 0
    for i, o in enumerate(outcomes or []):
        if o == "Yes":
            yes_idx = i
            break
    try:
        yes_price = float(prices[yes_idx])
    except (ValueError, TypeError, IndexError):
        return None
    yes_won = yes_price > 0.5
    resolution_date = m.get("endDate", "")[:10] if m.get("endDate") else None
    return yes_won, resolution_date


# ---------------------------------------------------------------------------
# Observed value fetch (Open-Meteo archive)
# ---------------------------------------------------------------------------

async def _fetch_observed_tmax(
    client: httpx.AsyncClient, city: str, date_iso: str,
) -> Optional[float]:
    city_info = CITIES.get(city)
    if city_info is None:
        return None
    try:
        r = await client.get(
            OPEN_METEO_ARCHIVE,
            params={
                "latitude": city_info["latitude"],
                "longitude": city_info["longitude"],
                "start_date": date_iso,
                "end_date": date_iso,
                "daily": "temperature_2m_max",
                "timezone": "auto",
            },
            timeout=10,
        )
        r.raise_for_status()
        d = r.json()
        tmaxes = d.get("daily", {}).get("temperature_2m_max", [])
        return float(tmaxes[0]) if tmaxes else None
    except Exception as e:
        log.debug("observed_tmax_fetch_failed", city=city, date=date_iso, err=str(e))
        return None


async def _fetch_observed_precip_sum(
    client: httpx.AsyncClient, city: str, start_date: str, end_date: str,
) -> Optional[float]:
    city_info = CITIES.get(city)
    if city_info is None:
        return None
    try:
        r = await client.get(
            OPEN_METEO_ARCHIVE,
            params={
                "latitude": city_info["latitude"],
                "longitude": city_info["longitude"],
                "start_date": start_date,
                "end_date": end_date,
                "daily": "precipitation_sum",
                "timezone": "auto",
            },
            timeout=15,
        )
        r.raise_for_status()
        d = r.json()
        daily = d.get("daily", {}).get("precipitation_sum", [])
        if not daily:
            return None
        # Sum across the range, treating nulls as 0.
        return float(sum(v for v in daily if v is not None))
    except Exception as e:
        log.debug("observed_precip_fetch_failed", city=city, err=str(e))
        return None


# ---------------------------------------------------------------------------
# Resolver loop
# ---------------------------------------------------------------------------

async def resolve_once() -> dict:
    """Single pass: find unresolved records, check market status, attach truth."""
    async with SessionLocal() as db:
        # Get distinct tokens with unresolved forecasts.
        unresolved_tokens = (await db.execute(
            select(
                WeatherForecastRecord.token_id,
                WeatherForecastRecord.market_condition_id,
                WeatherForecastRecord.city,
                WeatherForecastRecord.kind,
                WeatherForecastRecord.bucket_lower,
                WeatherForecastRecord.bucket_upper,
                func.min(WeatherForecastRecord.recorded_at).label("first_seen"),
            )
            .where(WeatherForecastRecord.resolved_at.is_(None))
            .group_by(
                WeatherForecastRecord.token_id,
                WeatherForecastRecord.market_condition_id,
                WeatherForecastRecord.city,
                WeatherForecastRecord.kind,
                WeatherForecastRecord.bucket_lower,
                WeatherForecastRecord.bucket_upper,
            )
        )).all()

    resolved_count = 0
    not_yet = 0
    errors = 0
    async with httpx.AsyncClient(timeout=15, limits=httpx.Limits(max_connections=10)) as client:
        for row in unresolved_tokens:
            token_id, condition_id, city, kind, lo, hi, first_seen = row
            if not condition_id:
                continue
            try:
                status = await _market_resolved(client, condition_id)
            except Exception as e:
                log.warning("resolve_check_failed", cond=condition_id[:20], err=str(e))
                errors += 1
                continue
            if status is None:
                not_yet += 1
                continue
            yes_won, resolution_date = status

            # Fetch observed value for the relevant day/range.
            observed_value: Optional[float] = None
            if kind == "temperature_max_day" and resolution_date:
                observed_value = await _fetch_observed_tmax(client, city, resolution_date)
            elif kind == "precipitation_sum_period" and resolution_date:
                # For month-aggregate events: fetch from 1st of month to resolution_date.
                start_date = resolution_date[:7] + "-01"
                observed_value = await _fetch_observed_precip_sum(
                    client, city, start_date, resolution_date,
                )

            try:
                n_updated = await attach_outcome(
                    token_id=token_id,
                    observed_outcome=yes_won,
                    observed_value=observed_value,
                )
                resolved_count += n_updated
                log.info(
                    "weather_resolved",
                    token=token_id[:20], city=city, kind=kind,
                    yes_won=yes_won, observed_value=observed_value,
                    n_records=n_updated,
                )
            except Exception as e:
                log.warning("resolve_attach_failed", token=token_id[:20], err=str(e))
                errors += 1
    return {
        "unresolved_tokens": len(unresolved_tokens),
        "records_updated": resolved_count,
        "still_pending": not_yet,
        "errors": errors,
    }


async def run_resolver_forever() -> None:
    if not ENABLED:
        log.info("weather_resolver_disabled")
        return
    while True:
        try:
            stats = await resolve_once()
            log.info("weather_resolver_tick", **stats)
        except Exception as e:
            log.error("weather_resolver_failed", err=str(e))
        await asyncio.sleep(RESOLVE_INTERVAL_SECONDS)
