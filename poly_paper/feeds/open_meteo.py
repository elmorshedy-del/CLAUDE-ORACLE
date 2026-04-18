"""Open-Meteo feed — free forecasts for weather-market fair values.

https://open-meteo.com — free tier (no key), 10k calls/day. Returns:
  - Hourly/daily forecasts up to 16 days ahead
  - Historical archive for calibration
  - ENSEMBLE models — multiple weather models (GFS, ECMWF, etc.) with
    probability distributions, not just point estimates

Why this works for Polymarket weather markets:
  Polymarket has "precipitation in NYC in April" type markets. These resolve
  based on observed totals at specific weather stations (e.g. Central Park).
  Open-Meteo's ensemble gives us a PROBABILITY DISTRIBUTION over future
  rainfall, which we can use to compute fair prices on "rainfall < X inches"
  type buckets.

Strategy plumbing comes in Phase 3 — this module provides the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


# Common Polymarket weather-market cities — lat/lon.
CITIES: dict[str, dict] = {
    "nyc":       {"latitude": 40.7789, "longitude": -73.9692, "station": "Central Park, NYC"},
    "seattle":   {"latitude": 47.6062, "longitude": -122.3321, "station": "Seattle-Tacoma Intl"},
    "london":    {"latitude": 51.4700, "longitude": -0.4543,   "station": "Heathrow"},
    "seoul":     {"latitude": 37.5665, "longitude": 126.9780},
    "hong_kong": {"latitude": 22.3193, "longitude": 114.1694},
}


@dataclass
class DailyForecast:
    date: str  # YYYY-MM-DD
    tmax_c: float | None
    tmin_c: float | None
    precip_mm: float | None


@dataclass
class EnsembleSample:
    """One member of the ensemble model. Each member is one possible future."""

    member_id: int
    daily: list[DailyForecast]


class OpenMeteo:
    def __init__(self, timeout: float = 15.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenMeteo":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def daily_forecast(
        self,
        *,
        latitude: float,
        longitude: float,
        days: int = 16,
        timezone: str = "auto",
    ) -> list[DailyForecast]:
        """Point forecast (deterministic — one model's median).

        Use for quick sanity checks. For probability estimation, use `ensemble_daily`.
        """
        r = await self._client.get(
            OPEN_METEO,
            params={
                "latitude": latitude, "longitude": longitude,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "forecast_days": days, "timezone": timezone,
            },
        )
        r.raise_for_status()
        d = r.json()
        days_data = d.get("daily", {})
        out = []
        for i, date in enumerate(days_data.get("time", [])):
            out.append(DailyForecast(
                date=date,
                tmax_c=days_data.get("temperature_2m_max", [None])[i],
                tmin_c=days_data.get("temperature_2m_min", [None])[i],
                precip_mm=days_data.get("precipitation_sum", [None])[i],
            ))
        return out

    async def ensemble_daily(
        self,
        *,
        latitude: float,
        longitude: float,
        days: int = 16,
        model: str = "icon_seamless",  # GFS-alternative; check docs for options
    ) -> list[EnsembleSample]:
        """Ensemble forecast — returns N members of the weather-model ensemble.

        Open-Meteo's ICON-seamless gives ~40 ensemble members. Each member is
        a self-consistent possible future. Aggregating across members gives a
        probability distribution over any outcome.

        Returns: one EnsembleSample per member, each with `days` DailyForecasts.
        """
        r = await self._client.get(
            OPEN_METEO_ENSEMBLE,
            params={
                "latitude": latitude, "longitude": longitude,
                "models": model,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "forecast_days": days,
                "timezone": "auto",
            },
        )
        r.raise_for_status()
        d = r.json()
        days_data = d.get("daily", {})
        dates = days_data.get("time", [])
        samples: dict[int, list[DailyForecast]] = {}
        # Open-Meteo returns lists like "temperature_2m_max_member01" for each member
        # (1-indexed: member01, member02, ..., memberNN). The flat "temperature_2m_max"
        # field is the deterministic mean; we skip it in favor of per-member samples.
        member_ids: set[int] = set()
        for k in days_data.keys():
            if not k.startswith("temperature_2m_max_member"):
                continue
            try:
                mid = int(k.split("member")[1])
                member_ids.add(mid)
            except ValueError:
                continue
        # Fallback: if the API returns no per-member fields (some models flatten),
        # treat the flat forecast as a single sample.
        if not member_ids:
            return [EnsembleSample(member_id=0, daily=await self._flatten(days_data))]

        for mid in sorted(member_ids):
            tmax = days_data.get(f"temperature_2m_max_member{mid:02d}", [])
            tmin = days_data.get(f"temperature_2m_min_member{mid:02d}", [])
            prcp = days_data.get(f"precipitation_sum_member{mid:02d}", [])
            daily = []
            for i, date in enumerate(dates):
                daily.append(DailyForecast(
                    date=date,
                    tmax_c=tmax[i] if i < len(tmax) else None,
                    tmin_c=tmin[i] if i < len(tmin) else None,
                    precip_mm=prcp[i] if i < len(prcp) else None,
                ))
            samples[mid] = daily
        return [EnsembleSample(member_id=k, daily=v) for k, v in samples.items()]

    @staticmethod
    async def _flatten(days_data: dict) -> list[DailyForecast]:
        out = []
        for i, date in enumerate(days_data.get("time", [])):
            out.append(DailyForecast(
                date=date,
                tmax_c=days_data.get("temperature_2m_max", [None])[i],
                tmin_c=days_data.get("temperature_2m_min", [None])[i],
                precip_mm=days_data.get("precipitation_sum", [None])[i],
            ))
        return out


# ---------------------------------------------------------------------------
# Probability derivation helpers
# ---------------------------------------------------------------------------

def prob_cumulative_precip_exceeds(
    ensemble: list[EnsembleSample],
    *,
    start_date: str,
    end_date: str,
    threshold_mm: float,
) -> float:
    """P(sum of daily precipitation across [start_date, end_date] > threshold_mm).

    Computed as the fraction of ensemble members whose summed precipitation
    exceeds the threshold. This is a NON-PARAMETRIC estimate — we're literally
    counting possible futures.

    Example use: "Total precipitation in NYC in April > 4 inches?" → threshold_mm=101.6,
    start_date="2026-04-01", end_date="2026-04-30".
    """
    hits = 0
    totals = []
    for sample in ensemble:
        total = 0.0
        any_found = False
        for day in sample.daily:
            if start_date <= day.date <= end_date and day.precip_mm is not None:
                total += day.precip_mm
                any_found = True
        if any_found:
            totals.append(total)
            if total > threshold_mm:
                hits += 1
    if not totals:
        return 0.5  # degenerate; signal "we don't know"
    return hits / len(totals)


def prob_max_tmax_exceeds(
    ensemble: list[EnsembleSample],
    *,
    start_date: str,
    end_date: str,
    threshold_c: float,
) -> float:
    """P(any day's max temperature in [start, end] > threshold_c)."""
    hits = 0
    n = 0
    for sample in ensemble:
        any_found = False
        peak = -1e9
        for day in sample.daily:
            if start_date <= day.date <= end_date and day.tmax_c is not None:
                any_found = True
                peak = max(peak, day.tmax_c)
        if any_found:
            n += 1
            if peak > threshold_c:
                hits += 1
    if n == 0:
        return 0.5
    return hits / n
