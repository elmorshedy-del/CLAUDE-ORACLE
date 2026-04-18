"""Fair-value math for BTC price markets.

All probabilities computed under the assumption that BTC spot follows
geometric Brownian motion under the physical measure:

    log(S_T / S_0) ~ Normal(mu*T - sigma^2*T/2, sigma^2*T)

For short horizons (minutes to hours), mu is effectively zero — the drift
component is dominated by volatility, and using mu=0 is slightly conservative
(biases toward 50/50, making us *less* confident in any directional edge,
which is the correct direction of error for Phase 2).

We expose:
- prob_up_over_horizon: P(S_T > S_0), i.e. "up or down" markets.
- prob_in_range: P(low <= S_T <= high), i.e. "between X and Y on date D".
- prob_above: P(S_T > level), i.e. "above X on date D" (close-basis).

NOT included yet (needs barrier-option math — Phase 3):
- prob_ever_touches: P(max_{t<=T} S_t >= level), i.e. "will BTC reach X by date Y?"
  Uses reflection principle. Different math. Adding later.

All horizons are in YEARS (standard for annualised vol). Use `years_from_seconds()`
to convert. All vol inputs are ANNUALISED (e.g. 0.70 for 70% annualised vol).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

# Seconds in a year — we use 365*24*60*60 so realized-vol computed from spot
# history (which sees weekends) annualises consistently. Crypto trades 24/7.
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60


def years_from_seconds(seconds: float) -> float:
    return seconds / SECONDS_PER_YEAR


@dataclass(frozen=True)
class FairValue:
    """A fair-value estimate with diagnostic context.

    Strategies consume this and decide whether to act. The diagnostic fields
    are logged so that later we can measure calibration: how often did our
    fair-value actually match the resolution rate?
    """

    probability: float              # in [0, 1]
    # Inputs used (logged for calibration):
    spot: float
    sigma_annual: float             # annualised vol assumption
    horizon_years: float
    model: Literal["up_down", "range", "above", "below"]
    # Upper/lower bounds of the 95% confidence interval on our probability
    # accounting for vol estimation error. A wide CI means we should be
    # conservative in sizing.
    ci_low: float
    ci_high: float

    def edge_vs_market(self, buy_yes_price: float) -> float:
        """Return expected value per $1 notional if buying YES at `buy_yes_price`.

        Positive edge = profitable before fees. Subtract fees to get net.
        """
        return self.probability - buy_yes_price


# ---------------------------------------------------------------------------
# Core probability models
# ---------------------------------------------------------------------------

def prob_up_over_horizon(
    sigma_annual: float,
    horizon_years: float,
    drift_annual: float = 0.0,
) -> float:
    """P(S_T > S_0) under GBM. Spot-independent — only depends on drift and vol."""
    if horizon_years <= 0:
        return 0.5
    sigma_t = sigma_annual * math.sqrt(horizon_years)
    # log(S_T/S_0) has mean (drift - sigma^2/2)*T and std sigma*sqrt(T).
    # P(log > 0) = 1 - Phi(-mean/std) = Phi(mean/std).
    mean = (drift_annual - sigma_annual**2 / 2) * horizon_years
    z = mean / sigma_t
    return float(norm.cdf(z))


def prob_in_range(
    spot: float,
    low: float,
    high: float,
    sigma_annual: float,
    horizon_years: float,
    drift_annual: float = 0.0,
) -> float:
    """P(low <= S_T <= high) under GBM. Closing-price basis."""
    if horizon_years <= 0:
        return 1.0 if low <= spot <= high else 0.0
    if low >= high:
        return 0.0
    sigma_t = sigma_annual * math.sqrt(horizon_years)
    mean = (drift_annual - sigma_annual**2 / 2) * horizon_years
    # log(S_T) = log(spot) + drift_term + vol_term, so log(S_T/spot) ~ N(mean, sigma_t^2).
    z_low = (math.log(low / spot) - mean) / sigma_t
    z_high = (math.log(high / spot) - mean) / sigma_t
    return float(norm.cdf(z_high) - norm.cdf(z_low))


def prob_above(
    spot: float,
    level: float,
    sigma_annual: float,
    horizon_years: float,
    drift_annual: float = 0.0,
) -> float:
    """P(S_T > level) under GBM. Closing-price basis."""
    if horizon_years <= 0:
        return 1.0 if spot > level else 0.0
    sigma_t = sigma_annual * math.sqrt(horizon_years)
    mean = (drift_annual - sigma_annual**2 / 2) * horizon_years
    z = (math.log(level / spot) - mean) / sigma_t
    return float(1.0 - norm.cdf(z))


# ---------------------------------------------------------------------------
# Wrappers that produce full FairValue objects with confidence intervals
# ---------------------------------------------------------------------------

def fv_up_down(
    sigma_annual: float,
    horizon_seconds: float,
    *,
    spot: float = 0.0,
    sigma_se: float = 0.1,  # relative std error on vol estimate (10% by default)
    drift_annual: float = 0.0,
) -> FairValue:
    """Fair value for 'will S close higher than it started over the next T seconds?'"""
    T = years_from_seconds(horizon_seconds)
    p = prob_up_over_horizon(sigma_annual, T, drift_annual=drift_annual)
    # Bootstrap CI using vol uncertainty.
    lo_vol = sigma_annual * (1 - 1.96 * sigma_se)
    hi_vol = sigma_annual * (1 + 1.96 * sigma_se)
    p_lo = prob_up_over_horizon(max(lo_vol, 1e-6), T, drift_annual=drift_annual)
    p_hi = prob_up_over_horizon(hi_vol, T, drift_annual=drift_annual)
    ci_low, ci_high = min(p_lo, p_hi), max(p_lo, p_hi)
    return FairValue(
        probability=p,
        spot=spot,
        sigma_annual=sigma_annual,
        horizon_years=T,
        model="up_down",
        ci_low=ci_low,
        ci_high=ci_high,
    )


def fv_range(
    spot: float,
    low: float,
    high: float,
    sigma_annual: float,
    horizon_seconds: float,
    *,
    sigma_se: float = 0.1,
    drift_annual: float = 0.0,
) -> FairValue:
    T = years_from_seconds(horizon_seconds)
    p = prob_in_range(spot, low, high, sigma_annual, T, drift_annual=drift_annual)
    lo_vol = max(sigma_annual * (1 - 1.96 * sigma_se), 1e-6)
    hi_vol = sigma_annual * (1 + 1.96 * sigma_se)
    p_lo = prob_in_range(spot, low, high, lo_vol, T, drift_annual=drift_annual)
    p_hi = prob_in_range(spot, low, high, hi_vol, T, drift_annual=drift_annual)
    return FairValue(
        probability=p,
        spot=spot,
        sigma_annual=sigma_annual,
        horizon_years=T,
        model="range",
        ci_low=min(p_lo, p_hi),
        ci_high=max(p_lo, p_hi),
    )


def fv_above(
    spot: float,
    level: float,
    sigma_annual: float,
    horizon_seconds: float,
    *,
    sigma_se: float = 0.1,
    drift_annual: float = 0.0,
) -> FairValue:
    T = years_from_seconds(horizon_seconds)
    p = prob_above(spot, level, sigma_annual, T, drift_annual=drift_annual)
    lo_vol = max(sigma_annual * (1 - 1.96 * sigma_se), 1e-6)
    hi_vol = sigma_annual * (1 + 1.96 * sigma_se)
    p_lo = prob_above(spot, level, lo_vol, T, drift_annual=drift_annual)
    p_hi = prob_above(spot, level, hi_vol, T, drift_annual=drift_annual)
    return FairValue(
        probability=p,
        spot=spot,
        sigma_annual=sigma_annual,
        horizon_years=T,
        model="above",
        ci_low=min(p_lo, p_hi),
        ci_high=max(p_lo, p_hi),
    )


# ---------------------------------------------------------------------------
# Realized volatility from price history
# ---------------------------------------------------------------------------

def realized_vol_annualised(
    prices: list[float] | np.ndarray,
    *,
    bar_seconds: float,
    robust: bool = True,
) -> float:
    """Compute annualised volatility from equally-spaced historical prices.

    Uses log-returns. If robust=True, uses the median absolute deviation
    scaled to std, which is less sensitive to outlier bars (crypto has plenty).
    """
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 2:
        raise ValueError("need at least 2 prices")
    log_returns = np.diff(np.log(arr))
    if robust:
        # MAD = median(|x - median(x)|); consistent stdev estimator is MAD * 1.4826.
        mad = np.median(np.abs(log_returns - np.median(log_returns)))
        sigma_bar = float(mad * 1.4826)
    else:
        sigma_bar = float(np.std(log_returns, ddof=1))
    # Scale bar-vol to annual.
    bars_per_year = SECONDS_PER_YEAR / bar_seconds
    return sigma_bar * math.sqrt(bars_per_year)
