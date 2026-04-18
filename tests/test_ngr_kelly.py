"""Tests for NGR post-processing and Kelly sizing."""

from __future__ import annotations

from decimal import Decimal

import math
import numpy as np
import pytest

from poly_paper.ngr import (
    NGRFit, bucket_probability, crps_normal, ensemble_moments, fit_ngr,
)
from poly_paper.kelly import (
    KellyResult, correlation_adjusted_stake, kelly_for_yes_buy,
)


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------

class TestCRPS:
    def test_perfect_point_forecast(self) -> None:
        # If sigma is tiny and mu hits obs, CRPS should be near 0.
        assert crps_normal(obs=20.0, mu=20.0, sigma=0.001) < 0.001

    def test_biased_forecast_has_higher_crps_than_unbiased(self) -> None:
        unbiased = crps_normal(obs=20.0, mu=20.0, sigma=1.0)
        biased = crps_normal(obs=20.0, mu=23.0, sigma=1.0)
        assert biased > unbiased

    def test_crps_degenerate_sigma_zero(self) -> None:
        # Degrades to absolute error.
        assert crps_normal(obs=20.0, mu=22.0, sigma=0.0) == 2.0


# ---------------------------------------------------------------------------
# Ensemble moments
# ---------------------------------------------------------------------------

class TestEnsembleMoments:
    def test_mean_and_variance(self) -> None:
        mean, var = ensemble_moments([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mean == 3.0
        # Sample variance (n-1) of [1..5] = 2.5
        assert abs(var - 2.5) < 1e-9

    def test_single_member_zero_variance(self) -> None:
        mean, var = ensemble_moments([7.0])
        assert mean == 7.0
        assert var == 0.0

    def test_empty_safe(self) -> None:
        mean, var = ensemble_moments([])
        assert mean == 0.0
        assert var == 0.0


# ---------------------------------------------------------------------------
# NGR fitting
# ---------------------------------------------------------------------------

class TestNGRFit:
    def test_fit_unbiased_data_recovers_identity(self) -> None:
        """If ensemble mean is unbiased and spread is calibrated, NGR should
        recover roughly a=0, b=1."""
        np.random.seed(0)
        n = 100
        truth = np.random.normal(15.0, 2.0, n)
        ens_means = truth + np.random.normal(0, 0.3, n)
        ens_vars = np.full(n, 0.3**2)
        fit = fit_ngr(list(ens_means), list(ens_vars), list(truth))
        assert abs(fit.a) < 1.0
        assert 0.8 < fit.b < 1.2

    def test_fit_biased_data_corrects_bias(self) -> None:
        """If ensemble is biased by a constant, NGR should absorb it into `a`."""
        np.random.seed(1)
        n = 80
        truth = np.random.normal(20.0, 3.0, n)
        ens_means = truth + 2.0 + np.random.normal(0, 0.3, n)  # bias of +2°C
        ens_vars = np.full(n, 0.3**2)
        fit = fit_ngr(list(ens_means), list(ens_vars), list(truth))
        # a + b*mean(ens) should recover mean(truth).
        # If b ≈ 1, then a ≈ -2.
        predicted_mean = fit.a + fit.b * float(ens_means.mean())
        assert abs(predicted_mean - float(truth.mean())) < 0.5

    def test_fit_improves_crps_on_biased_underdispersive(self) -> None:
        """The whole point: NGR should reduce CRPS on bad ensembles."""
        np.random.seed(42)
        n = 60
        truth = np.random.normal(20.0, 3.0, n)
        ens_means = truth + np.random.normal(0.8, 0.5, n)
        ens_vars = np.full(n, 0.5)  # deliberately too narrow
        fit = fit_ngr(list(ens_means), list(ens_vars), list(truth))
        assert fit.mean_crps_train < fit.mean_crps_raw


# ---------------------------------------------------------------------------
# NGR apply + bucket probability
# ---------------------------------------------------------------------------

class TestNGRApply:
    def test_apply_returns_mu_sigma(self) -> None:
        fit = NGRFit(a=0.0, b=1.0, c=0.0, d=1.0, n_training_samples=50,
                     mean_crps_train=0.1, mean_crps_raw=0.2)
        mu, sigma = fit.apply(ensemble_mean=20.0, ensemble_var=4.0)
        assert mu == 20.0
        assert abs(sigma - 2.0) < 1e-9

    def test_probability_in_bucket_with_known_normal(self) -> None:
        # N(20, 1) → P(19 <= x < 21) ≈ 0.6827
        fit = NGRFit(a=0.0, b=1.0, c=1.0, d=0.0, n_training_samples=50,
                     mean_crps_train=0.1, mean_crps_raw=0.2)
        p = fit.probability_in_bucket(ensemble_mean=20.0, ensemble_var=0.0,
                                      lower=19.0, upper=21.0)
        assert abs(p - 0.6827) < 0.005

    def test_probability_open_upper(self) -> None:
        # N(20, 1), P(x >= 22) ≈ 0.0228
        fit = NGRFit(a=0.0, b=1.0, c=1.0, d=0.0, n_training_samples=50,
                     mean_crps_train=0.1, mean_crps_raw=0.2)
        p = fit.probability_in_bucket(ensemble_mean=20.0, ensemble_var=0.0,
                                      lower=22.0, upper=None)
        assert abs(p - 0.0228) < 0.005


class TestBucketProbabilityRouter:
    def test_raw_when_no_fit(self) -> None:
        p, method = bucket_probability(members=[20, 21, 22], lower=20.0, upper=22.0)
        # members 20 and 21 are in [20, 22), 22 is not.
        assert method == "raw"
        assert abs(p - 2 / 3) < 1e-9

    def test_ngr_when_fit_provided(self) -> None:
        fit = NGRFit(a=0.0, b=1.0, c=1.0, d=0.0, n_training_samples=50,
                     mean_crps_train=0.1, mean_crps_raw=0.2)
        p, method = bucket_probability(members=[20.0, 20.0, 20.0],
                                       lower=19.0, upper=21.0, fit=fit)
        assert method == "ngr"
        assert abs(p - 0.6827) < 0.005


# ---------------------------------------------------------------------------
# Kelly
# ---------------------------------------------------------------------------

class TestKelly:
    def test_no_edge_zero_stake(self) -> None:
        k = kelly_for_yes_buy(probability=0.3, ask_price=0.4,
                              bankroll_usd=Decimal("1000"))
        assert k.stake_usd == Decimal("0")
        assert k.full_kelly <= 0

    def test_positive_edge_positive_stake(self) -> None:
        k = kelly_for_yes_buy(probability=0.5, ask_price=0.3,
                              bankroll_usd=Decimal("1000"), kelly_fraction=0.25)
        # Full Kelly: f* = 0.5 - 0.5*0.3/0.7 = 0.5 - 0.2143 = 0.2857
        # Quarter Kelly: 0.0714 → stake $71.43
        assert k.full_kelly > 0
        assert k.stake_usd > 0
        assert abs(k.full_kelly - (0.5 - 0.5 * 0.3 / 0.7)) < 1e-6

    def test_max_fraction_caps_stake(self) -> None:
        # Huge edge but max_fraction caps us at 5% of bankroll.
        k = kelly_for_yes_buy(probability=0.95, ask_price=0.10,
                              bankroll_usd=Decimal("1000"),
                              kelly_fraction=1.0, max_fraction=0.05)
        # full Kelly would be massive here, should be clipped.
        assert k.fractional == 0.05
        assert k.stake_usd == Decimal("50.00")

    def test_full_kelly_vs_quarter(self) -> None:
        """Quarter Kelly should stake 1/4 of full Kelly (when both below cap)."""
        args = dict(probability=0.5, ask_price=0.4, bankroll_usd=Decimal("10000"),
                    max_fraction=0.5)
        full = kelly_for_yes_buy(**args, kelly_fraction=1.0)
        quarter = kelly_for_yes_buy(**args, kelly_fraction=0.25)
        # stakes differ by factor of 4 (approximately — quantized to cents).
        ratio = float(full.stake_usd) / float(quarter.stake_usd)
        assert abs(ratio - 4.0) < 0.1

    def test_uncertainty_penalty_shrinks(self) -> None:
        base = kelly_for_yes_buy(probability=0.5, ask_price=0.3,
                                 bankroll_usd=Decimal("1000"),
                                 kelly_fraction=0.25, uncertainty_penalty=1.0)
        shrunk = kelly_for_yes_buy(probability=0.5, ask_price=0.3,
                                   bankroll_usd=Decimal("1000"),
                                   kelly_fraction=0.25, uncertainty_penalty=0.5)
        assert shrunk.stake_usd < base.stake_usd

    def test_invalid_ask_returns_zero(self) -> None:
        for bad in (0.0, 1.0, -0.1, 1.5):
            k = kelly_for_yes_buy(probability=0.5, ask_price=bad,
                                  bankroll_usd=Decimal("1000"))
            assert k.stake_usd == Decimal("0")


class TestCorrelationAdjustment:
    def test_unscaled_when_under_cap(self) -> None:
        stakes = [Decimal("10"), Decimal("15"), Decimal("20")]
        adj = correlation_adjusted_stake(
            individual_stakes_usd=stakes,
            max_group_fraction_of_bankroll=0.1,
            bankroll_usd=Decimal("1000"),
        )
        # Total 45 < cap 100 → unchanged.
        assert adj == stakes

    def test_scaled_when_over_cap(self) -> None:
        stakes = [Decimal("60"), Decimal("60")]
        adj = correlation_adjusted_stake(
            individual_stakes_usd=stakes,
            max_group_fraction_of_bankroll=0.05,
            bankroll_usd=Decimal("1000"),
        )
        # Cap = 50. Total = 120. Scale = 50/120. Each = 25.
        assert sum(adj) == Decimal("50.00")
        # Proportional
        assert adj[0] == adj[1]
