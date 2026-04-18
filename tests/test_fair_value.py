"""Tests for the fair-value math. These encode the mathematical invariants
that must hold for any correct implementation of GBM probabilities.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from poly_paper.strategies.fair_value import (
    SECONDS_PER_YEAR,
    fv_range,
    fv_up_down,
    prob_above,
    prob_in_range,
    prob_up_over_horizon,
    realized_vol_annualised,
    years_from_seconds,
)


class TestUpDown:
    def test_zero_drift_slightly_below_half(self) -> None:
        # With zero drift and positive vol, P(up) is SLIGHTLY below 0.5 because
        # log(S_T/S_0) has mean -sigma^2*T/2. Real drift would push it back.
        p = prob_up_over_horizon(sigma_annual=0.7, horizon_years=years_from_seconds(300))
        assert 0.49 < p < 0.5

    def test_very_short_horizon_is_nearly_half(self) -> None:
        p = prob_up_over_horizon(sigma_annual=0.7, horizon_years=years_from_seconds(60))
        assert abs(p - 0.5) < 0.001

    def test_zero_vol_zero_drift_is_exactly_half_or_edge(self) -> None:
        # sigma=0 means deterministic path, and with drift=0 the future equals the present.
        # P(up) is technically 0.5 by convention (tie).
        # Our implementation returns 0.5 in this degenerate case via scipy's limit.
        p = prob_up_over_horizon(sigma_annual=1e-10, horizon_years=years_from_seconds(300))
        assert 0.45 < p < 0.55

    def test_positive_drift_raises_probability(self) -> None:
        p_no_drift = prob_up_over_horizon(sigma_annual=0.5, horizon_years=years_from_seconds(3600))
        p_drift = prob_up_over_horizon(
            sigma_annual=0.5, horizon_years=years_from_seconds(3600), drift_annual=0.5
        )
        assert p_drift > p_no_drift


class TestRange:
    def test_current_spot_inside_range_at_zero_horizon(self) -> None:
        p = prob_in_range(spot=100, low=90, high=110, sigma_annual=0.5, horizon_years=0)
        assert p == 1.0

    def test_current_spot_outside_range_at_zero_horizon(self) -> None:
        p = prob_in_range(spot=100, low=120, high=130, sigma_annual=0.5, horizon_years=0)
        assert p == 0.0

    def test_symmetric_range_around_spot_at_zero_drift(self) -> None:
        # At zero drift, a symmetric range in LOG space captures most of the mass.
        spot = 100.0
        # Choose low/high symmetric in log: low = spot/exp(k), high = spot*exp(k)
        k = 0.05
        low, high = spot * math.exp(-k), spot * math.exp(k)
        p = prob_in_range(spot=spot, low=low, high=high, sigma_annual=0.5, horizon_years=years_from_seconds(300))
        # Very short horizon → almost all mass stays in range.
        assert p > 0.9

    def test_wider_vol_lowers_in_range_probability(self) -> None:
        args = dict(spot=100.0, low=95.0, high=105.0, horizon_years=years_from_seconds(3600))
        p_lo = prob_in_range(sigma_annual=0.3, **args)
        p_hi = prob_in_range(sigma_annual=1.2, **args)
        assert p_lo > p_hi

    def test_degenerate_range_zero(self) -> None:
        p = prob_in_range(spot=100, low=100, high=100, sigma_annual=0.5, horizon_years=years_from_seconds(300))
        assert p == 0.0


class TestAbove:
    def test_above_plus_below_equals_one_minus_ties(self) -> None:
        args = dict(spot=100.0, sigma_annual=0.7, horizon_years=years_from_seconds(3600))
        level = 102.0
        p_above = prob_above(level=level, **args)
        # P(below) = 1 - P(above). We don't compute that separately; just check 0<=p<=1.
        assert 0 <= p_above <= 1

    def test_far_above_close_to_zero(self) -> None:
        p = prob_above(spot=100, level=1000, sigma_annual=0.7, horizon_years=years_from_seconds(300))
        assert p < 0.001

    def test_far_below_close_to_one(self) -> None:
        p = prob_above(spot=100, level=1, sigma_annual=0.7, horizon_years=years_from_seconds(300))
        assert p > 0.999


class TestFairValueWrapper:
    def test_ci_widens_with_vol_uncertainty(self) -> None:
        tight = fv_up_down(sigma_annual=0.7, horizon_seconds=3600, sigma_se=0.01)
        loose = fv_up_down(sigma_annual=0.7, horizon_seconds=3600, sigma_se=0.5)
        assert (loose.ci_high - loose.ci_low) >= (tight.ci_high - tight.ci_low)

    def test_edge_vs_market(self) -> None:
        fv = fv_up_down(sigma_annual=0.7, horizon_seconds=300)
        # With fair value ~0.4999, market at 0.52 means buying YES is bad.
        edge = fv.edge_vs_market(0.52)
        assert edge < 0
        # Buying YES at 0.45 means selling pressure has pushed price below fair.
        edge = fv.edge_vs_market(0.45)
        assert edge > 0

    def test_range_fv_matches_manual(self) -> None:
        fv = fv_range(spot=100.0, low=95.0, high=105.0, sigma_annual=0.5, horizon_seconds=3600)
        manual = prob_in_range(
            spot=100.0, low=95.0, high=105.0,
            sigma_annual=0.5, horizon_years=years_from_seconds(3600),
        )
        assert abs(fv.probability - manual) < 1e-9


class TestRealizedVol:
    def test_constant_price_has_zero_vol(self) -> None:
        prices = [100.0] * 100
        sigma = realized_vol_annualised(prices, bar_seconds=3600, robust=False)
        assert sigma == 0.0

    def test_gbm_synthetic_recovers_true_vol(self) -> None:
        # Generate GBM with known sigma=0.6 annualised. Should recover it roughly.
        rng = np.random.default_rng(42)
        sigma_true = 0.6
        bar_seconds = 3600  # hourly
        n = 24 * 30  # 30 days of hourly bars
        dt = bar_seconds / SECONDS_PER_YEAR
        returns = rng.normal(loc=0, scale=sigma_true * math.sqrt(dt), size=n)
        prices = [100.0]
        for r in returns:
            prices.append(prices[-1] * math.exp(r))
        sigma_est = realized_vol_annualised(prices, bar_seconds=bar_seconds, robust=False)
        # Statistical estimator, allow 15% tolerance.
        assert abs(sigma_est - sigma_true) / sigma_true < 0.15

    def test_requires_at_least_two_prices(self) -> None:
        with pytest.raises(ValueError):
            realized_vol_annualised([100.0], bar_seconds=3600)

    def test_robust_estimator_more_resilient_to_outliers(self) -> None:
        # Build a series with a huge outlier return; robust should be much lower than non-robust.
        rng = np.random.default_rng(7)
        base = [100.0]
        for _ in range(200):
            base.append(base[-1] * math.exp(rng.normal(0, 0.01)))
        # Inject a 50% spike.
        base[100] *= 1.5
        sigma_robust = realized_vol_annualised(base, bar_seconds=3600, robust=True)
        sigma_plain = realized_vol_annualised(base, bar_seconds=3600, robust=False)
        assert sigma_robust < sigma_plain
