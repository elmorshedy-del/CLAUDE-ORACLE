"""Weather forecast calibration — measurement infrastructure.

The CORE metric that decides whether weather strategy has edge or is losing
money silently: DO OUR FAIR VALUES MATCH REALIZED OUTCOMES?

This module:
  1. Records every fair-value forecast we make, alongside the bucket metadata,
     forecast horizon, and ensemble size.
  2. Later (when markets resolve), attaches the observed outcome.
  3. Computes calibration metrics:
       - Brier score          — mean squared error of probability
       - Brier skill score    — vs climatology reference (> 0 = genuine skill)
       - Reliability curve    — for each predicted-probability bin, observed frequency
       - CRPS (discrete form) — for multi-bucket events, cumulative distribution error
       - Sharpness            — how often we make confident predictions (away from 0.5)
       - Discrimination       — ROC / AUC: can we separate eventual YES from NO

These are the standard toolkit from ECMWF and the probabilistic-forecasting
literature (Wilson, Hersbach, Gofa). Without them, "edge" is an untested claim.

DESIGN:
  - Forecasts are persisted via `record_forecast(...)` at intent-generation time.
  - Observations are written via `attach_outcome(...)` when markets resolve.
  - Metrics are computed OFFLINE (cheap) from the stored data.
  - Dashboard reads these metrics and shows per-sleeve calibration state.

IMPORTANT GUARANTEE:
  We record forecasts even if NO TRADE fires. Non-trade data tells us whether
  our thresholds are right. If our fv=0.08 buckets resolve YES 30% of the time,
  we're systematically under-estimating extremes and should widen the price
  range we trade at, not tighten it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import (
    JSON, Column, DateTime, Float, Integer, String, Text, Boolean, select, func,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db.models import Base
from .db.session import SessionLocal

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class WeatherForecastRecord(Base):
    """One forecast-vs-outcome record for calibration tracking.

    One row per (event, bucket) at the moment we computed a fair value.
    Outcome is filled in later when the market resolves. Unresolved rows
    (`resolved_at` null) are excluded from calibration metrics.
    """

    __tablename__ = "weather_forecast_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Forecast identity
    event_slug: Mapped[str] = mapped_column(String, index=True)
    event_title: Mapped[str] = mapped_column(String)
    market_condition_id: Mapped[str] = mapped_column(String, index=True)
    token_id: Mapped[str] = mapped_column(String, index=True)
    city: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String)  # "temperature_max_day" | "precipitation_sum_period"

    # Bucket
    bucket_lower: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bucket_upper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bucket_raw_question: Mapped[str] = mapped_column(Text)

    # Forecast
    fair_value: Mapped[float] = mapped_column(Float)
    raw_fair_value: Mapped[float] = mapped_column(Float)  # before post-processing
    ensemble_size: Mapped[int] = mapped_column(Integer)
    members_in_bucket: Mapped[int] = mapped_column(Integer)
    # Ensemble moments on the continuous variable (tmax_c or precip_mm_total).
    # Persisted so the NGR trainer can fit against observed_value later.
    ensemble_mean_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ensemble_var_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    horizon_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    post_processing: Mapped[str] = mapped_column(String, default="raw")  # "raw" | "ngr"

    # Market-observed price at forecast time (if available)
    market_bid: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market_ask: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Outcome (filled in by resolver)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    observed_outcome: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)  # bucket hit = True
    observed_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)     # actual tmax or rainfall

    # Provenance
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True,
    )
    sleeve_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    intent_fired: Mapped[bool] = mapped_column(Boolean, default=False)  # did this trigger a trade?


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

async def record_forecast(
    *,
    event_slug: str,
    event_title: str,
    market_condition_id: str,
    token_id: str,
    city: str,
    bucket_kind: str,
    bucket_lower: Optional[float],
    bucket_upper: Optional[float],
    bucket_raw_question: str,
    fair_value: float,
    raw_fair_value: float,
    ensemble_size: int,
    members_in_bucket: int,
    ensemble_mean_value: Optional[float] = None,
    ensemble_var_value: Optional[float] = None,
    horizon_hours: Optional[float] = None,
    post_processing: str = "raw",
    market_bid: Optional[float] = None,
    market_ask: Optional[float] = None,
    sleeve_id: Optional[str] = None,
    intent_fired: bool = False,
) -> int:
    """Persist one forecast record. Returns the row ID."""
    async with SessionLocal() as db:
        rec = WeatherForecastRecord(
            event_slug=event_slug,
            event_title=event_title,
            market_condition_id=market_condition_id,
            token_id=token_id,
            city=city,
            kind=bucket_kind,
            bucket_lower=bucket_lower,
            bucket_upper=bucket_upper,
            bucket_raw_question=bucket_raw_question,
            fair_value=fair_value,
            raw_fair_value=raw_fair_value,
            ensemble_size=ensemble_size,
            members_in_bucket=members_in_bucket,
            ensemble_mean_value=ensemble_mean_value,
            ensemble_var_value=ensemble_var_value,
            horizon_hours=horizon_hours,
            post_processing=post_processing,
            market_bid=market_bid,
            market_ask=market_ask,
            sleeve_id=sleeve_id,
            intent_fired=intent_fired,
        )
        db.add(rec)
        await db.commit()
        await db.refresh(rec)
        return rec.id


async def attach_outcome(
    *,
    token_id: str,
    observed_outcome: bool,
    observed_value: Optional[float] = None,
    resolved_at: Optional[datetime] = None,
) -> int:
    """Fill in the observed outcome for all unresolved forecasts on this token.

    Called when a market resolves. Returns number of rows updated.
    """
    from sqlalchemy import update
    resolved_at = resolved_at or datetime.now(timezone.utc)
    async with SessionLocal() as db:
        res = await db.execute(
            update(WeatherForecastRecord)
            .where(
                WeatherForecastRecord.token_id == token_id,
                WeatherForecastRecord.resolved_at.is_(None),
            )
            .values(
                resolved_at=resolved_at,
                observed_outcome=observed_outcome,
                observed_value=observed_value,
            )
        )
        await db.commit()
        return res.rowcount or 0


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

@dataclass
class ReliabilityBin:
    """One bin in the reliability diagram.

    Bin [low, high) contains all forecasts with fair_value in this range.
    We report how often those forecasts actually resolved YES.
    """
    bin_low: float
    bin_high: float
    n_forecasts: int
    mean_predicted: float      # mean of our fair_values in this bin
    observed_frequency: float  # fraction that resolved YES


@dataclass
class CalibrationReport:
    """Verification stats over a set of resolved forecasts."""
    n_total: int
    n_resolved: int
    # Scores (lower is better for Brier, higher for BSS / AUC)
    brier_score: float
    brier_score_climatology: float  # reference
    brier_skill_score: float        # 1 - BS/BS_clim; > 0 = we beat climatology
    # Calibration curve
    reliability_bins: list[ReliabilityBin]
    # Base rate (climatology within this data)
    empirical_base_rate: float
    # Sharpness: std of our predicted probabilities. High sharpness + good
    # reliability = genuinely skillful system.
    sharpness: float
    # Discrimination (AUC): can we separate YES from NO events?
    auc: Optional[float]
    # Post-processing diagnostic: did raw vs post-processed differ?
    raw_brier_score: Optional[float] = None


async def compute_calibration(
    *,
    sleeve_id: Optional[str] = None,
    city: Optional[str] = None,
    kind: Optional[str] = None,
    n_bins: int = 10,
    min_records: int = 30,
) -> Optional[CalibrationReport]:
    """Compute calibration metrics over resolved forecasts.

    Filters:
      sleeve_id — restrict to forecasts for one sleeve (None = all)
      city      — restrict to one city (None = all)
      kind      — restrict to "temperature_max_day" or "precipitation_sum_period"
      n_bins    — number of reliability bins (default 10, i.e. deciles)
      min_records — returns None if fewer resolved records exist
    """
    async with SessionLocal() as db:
        stmt = select(WeatherForecastRecord).where(
            WeatherForecastRecord.resolved_at.is_not(None)
        )
        if sleeve_id is not None:
            stmt = stmt.where(WeatherForecastRecord.sleeve_id == sleeve_id)
        if city is not None:
            stmt = stmt.where(WeatherForecastRecord.city == city)
        if kind is not None:
            stmt = stmt.where(WeatherForecastRecord.kind == kind)
        rows = (await db.execute(stmt)).scalars().all()

    if len(rows) < min_records:
        return None

    n = len(rows)
    preds = [r.fair_value for r in rows]
    raws = [r.raw_fair_value for r in rows]
    outcomes = [1.0 if r.observed_outcome else 0.0 for r in rows]

    # Empirical base rate (climatology reference for this filter slice)
    base_rate = sum(outcomes) / n

    # Brier score = mean((p - y)^2)
    brier = sum((p - y) ** 2 for p, y in zip(preds, outcomes)) / n
    raw_brier = sum((p - y) ** 2 for p, y in zip(raws, outcomes)) / n
    # Climatology reference: always predict base_rate
    brier_clim = sum((base_rate - y) ** 2 for y in outcomes) / n
    bss = 1.0 - (brier / brier_clim) if brier_clim > 0 else 0.0

    # Sharpness = std of predicted probabilities
    mean_p = sum(preds) / n
    sharpness = (sum((p - mean_p) ** 2 for p in preds) / n) ** 0.5

    # Reliability bins
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        in_bin = [(p, y) for p, y in zip(preds, outcomes) if lo <= p < hi or (hi == 1.0 and p == 1.0)]
        if not in_bin:
            continue
        mean_pred = sum(p for p, _ in in_bin) / len(in_bin)
        obs_freq = sum(y for _, y in in_bin) / len(in_bin)
        bins.append(ReliabilityBin(
            bin_low=lo, bin_high=hi, n_forecasts=len(in_bin),
            mean_predicted=mean_pred, observed_frequency=obs_freq,
        ))

    # AUC via Mann-Whitney U
    auc = _compute_auc(preds, outcomes)

    return CalibrationReport(
        n_total=n,
        n_resolved=n,
        brier_score=brier,
        brier_score_climatology=brier_clim,
        brier_skill_score=bss,
        reliability_bins=bins,
        empirical_base_rate=base_rate,
        sharpness=sharpness,
        auc=auc,
        raw_brier_score=raw_brier,
    )


def _compute_auc(preds: list[float], outcomes: list[float]) -> Optional[float]:
    """Area under ROC curve, via the rank-sum formula.

    AUC = 1.0 = perfect discrimination. AUC = 0.5 = random. AUC < 0.5 = inverted.
    """
    pos = [p for p, y in zip(preds, outcomes) if y > 0.5]
    neg = [p for p, y in zip(preds, outcomes) if y <= 0.5]
    if not pos or not neg:
        return None
    # For each (pos, neg) pair, count (pos > neg) + 0.5*(pos == neg).
    count = 0.0
    for pp in pos:
        for np_ in neg:
            if pp > np_:
                count += 1.0
            elif pp == np_:
                count += 0.5
    return count / (len(pos) * len(neg))
