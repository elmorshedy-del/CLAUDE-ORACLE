"""Daily NGR trainer — fits calibration coefficients from resolved forecasts.

Runs as a separate loop on `POLY_NGR_TRAIN_INTERVAL_SEC` (default: daily).

For every (city, kind) slice with enough resolved observations, we:
  1. Pull historical (ensemble_mean, ensemble_var, observed_value) triples.
  2. Fit NGR by minimum-CRPS.
  3. Compute improvement: (raw_CRPS - NGR_CRPS) / raw_CRPS * 100.
  4. Persist to `ngr_fits` table.

Weather runner at prediction time calls `latest_fit(city, kind)` and uses
NGR probabilities when a fit exists, else falls back to raw ensemble counts.

IMPORTANT: We DO NOT hot-swap raw ensemble data for observations here. The
ensemble_mean/ensemble_var stored on `WeatherForecastRecord` is what we had
at forecast time — training uses that exact pair (forecast, outcome) so the
fit reflects real prediction-time uncertainty, not hindsight.

BOOTSTRAPPING: Until you have ~30 resolved pairs per slice, this produces
no fits, and weather runner keeps using raw. The dashboard reports
coverage so you can see when NGR will kick in per city.
"""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog
from sqlalchemy import desc, select

from .db.models import NGRFitRow
from .db.session import SessionLocal
from .ngr import NGRFit, fit_ngr
from .weather_calibration import WeatherForecastRecord

log = structlog.get_logger()

TRAIN_INTERVAL_SECONDS = float(os.environ.get("POLY_NGR_TRAIN_INTERVAL_SEC", "86400"))  # daily
MIN_TRAINING_SAMPLES = int(os.environ.get("POLY_NGR_MIN_SAMPLES", "30"))
MAX_TRAINING_AGE_DAYS = float(os.environ.get("POLY_NGR_MAX_AGE_DAYS", "180"))
ENABLED = os.environ.get("POLY_NGR_ENABLED", "1") not in ("0", "false", "no")


# ---------------------------------------------------------------------------
# Training data extraction
# ---------------------------------------------------------------------------

@dataclass
class TrainingSlice:
    """One (city, kind) slice with its historical training data."""
    city: str
    kind: str
    ensemble_means: list[float]
    ensemble_vars: list[float]
    observations: list[float]

    @property
    def n(self) -> int:
        return len(self.observations)


async def gather_training_slices(max_age_days: float = MAX_TRAINING_AGE_DAYS) -> list[TrainingSlice]:
    """Group resolved WeatherForecastRecord rows into (city, kind) slices.

    Training data per slice = (ensemble_mean_value, ensemble_var_value, observed_value)
    for every resolved record. Observed_value is the actual continuous variable
    (tmax_c for temperature events, total precip_mm for precipitation events).

    Same-slice records may represent different BUCKETS on different DAYS — that's
    fine: NGR fits the predictive distribution of the continuous variable
    (y ~ N(a + b·mean, √(c + d·var))), which is the same model across buckets.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    async with SessionLocal() as db:
        rows = (await db.execute(
            select(WeatherForecastRecord).where(
                WeatherForecastRecord.resolved_at.is_not(None),
                WeatherForecastRecord.observed_value.is_not(None),
                WeatherForecastRecord.ensemble_mean_value.is_not(None),
                WeatherForecastRecord.ensemble_var_value.is_not(None),
                WeatherForecastRecord.recorded_at >= cutoff,
            )
        )).scalars().all()

    # Group by (city, kind).
    # Deduplicate: same (token_id, recorded_day) can have MANY rows if multiple
    # sleeves evaluated — but the ensemble moments are the same for all. Take
    # one representative per (token_id, date) to avoid over-weighting.
    groups: dict[tuple[str, str], dict[tuple[str, str], WeatherForecastRecord]] = {}
    for r in rows:
        key = (r.city, r.kind)
        groups.setdefault(key, {})
        # Dedup key: one sample per (token_id, YYYY-MM-DD of recorded_at).
        dedup_key = (r.token_id, r.recorded_at.date().isoformat())
        # Keep earliest record (most "fresh" for given horizon).
        existing = groups[key].get(dedup_key)
        if existing is None or r.recorded_at < existing.recorded_at:
            groups[key][dedup_key] = r

    slices: list[TrainingSlice] = []
    for (city, kind), recs in groups.items():
        means = [r.ensemble_mean_value for r in recs.values()]
        vars_ = [r.ensemble_var_value for r in recs.values()]
        obs = [r.observed_value for r in recs.values()]
        slices.append(TrainingSlice(
            city=city, kind=kind,
            ensemble_means=means, ensemble_vars=vars_, observations=obs,
        ))
    return slices


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

async def persist_fit(city: str, kind: str, fit: NGRFit) -> None:
    improvement_pct = 0.0
    if fit.mean_crps_raw > 0:
        improvement_pct = 100.0 * (fit.mean_crps_raw - fit.mean_crps_train) / fit.mean_crps_raw
    async with SessionLocal() as db:
        row = NGRFitRow(
            city=city,
            kind=kind,
            a=fit.a,
            b=fit.b,
            c=fit.c,
            d=fit.d,
            n_training_samples=fit.n_training_samples,
            mean_crps_train=fit.mean_crps_train,
            mean_crps_raw=fit.mean_crps_raw,
            improvement_pct=improvement_pct,
        )
        db.add(row)
        await db.commit()


async def latest_fit(city: str, kind: str) -> Optional[NGRFit]:
    """Load the most recent fit for this slice, or None if none exists."""
    async with SessionLocal() as db:
        row = (await db.execute(
            select(NGRFitRow)
            .where(NGRFitRow.city == city, NGRFitRow.kind == kind)
            .order_by(desc(NGRFitRow.fitted_at))
            .limit(1)
        )).scalar_one_or_none()
    if row is None:
        return None
    return NGRFit(
        a=row.a, b=row.b, c=row.c, d=row.d,
        n_training_samples=row.n_training_samples,
        mean_crps_train=row.mean_crps_train,
        mean_crps_raw=row.mean_crps_raw,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def train_once() -> dict:
    """One pass of training across all slices."""
    slices = await gather_training_slices()
    n_fit = 0
    n_skip_too_small = 0
    for sl in slices:
        if sl.n < MIN_TRAINING_SAMPLES:
            n_skip_too_small += 1
            log.info(
                "ngr_skip_too_small", city=sl.city, kind=sl.kind,
                n=sl.n, min_required=MIN_TRAINING_SAMPLES,
            )
            continue
        try:
            fit = fit_ngr(sl.ensemble_means, sl.ensemble_vars, sl.observations)
            await persist_fit(sl.city, sl.kind, fit)
            n_fit += 1
            log.info(
                "ngr_fit_persisted", city=sl.city, kind=sl.kind,
                n_samples=fit.n_training_samples,
                crps_raw=round(fit.mean_crps_raw, 4),
                crps_ngr=round(fit.mean_crps_train, 4),
                improvement_pct=round(100 * (fit.mean_crps_raw - fit.mean_crps_train) / max(fit.mean_crps_raw, 1e-9), 1),
            )
        except Exception as e:
            log.warning("ngr_fit_failed", city=sl.city, kind=sl.kind, err=str(e))
    return {
        "slices_evaluated": len(slices),
        "fits_persisted": n_fit,
        "skipped_too_small": n_skip_too_small,
    }


async def run_ngr_trainer_forever() -> None:
    if not ENABLED:
        log.info("ngr_trainer_disabled")
        return
    while True:
        try:
            stats = await train_once()
            log.info("ngr_trainer_tick", **stats)
        except Exception as e:
            log.error("ngr_trainer_failed", err=str(e))
        await asyncio.sleep(TRAIN_INTERVAL_SECONDS)
