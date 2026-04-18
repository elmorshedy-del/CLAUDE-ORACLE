"""Non-homogeneous Gaussian Regression (NGR) — statistical post-processing for ensemble forecasts.

Also known as Ensemble Model Output Statistics (EMOS) in the weather forecasting literature.
Reference: Gneiting, Raftery, Westveld, Goldman (2005), "Calibrated Probabilistic Forecasting
Using Ensemble Model Output Statistics and Minimum CRPS Estimation", Mon. Wea. Rev.

THE PROBLEM WITH RAW ENSEMBLES:
  Raw ensemble forecasts are systematically:
    1) BIASED: ensemble mean drifts from observed value (station bias, model bias).
    2) UNDER-DISPERSIVE: ensemble spread too narrow → we underestimate uncertainty.
       Reliability diagrams on raw ensembles look "S-shaped" (over-confident).

NGR FIX:
  Given observed y and ensemble (m_1, ..., m_K), model y as Gaussian:

      y ~ Normal(a + b*mean(m), sqrt(c + d * var(m)))

  The mean is a linear correction of the ensemble mean.
  The VARIANCE is a linear function of the ensemble variance — this is
  the "nonhomogeneous" part: variance varies with forecast uncertainty.

  Parameters (a, b, c, d) are fit by MINIMUM CRPS (not MLE — CRPS is a proper
  scoring rule and directly optimizes forecast sharpness AND reliability).

  After fitting, we replace the raw bucket probability (count members / K) with
  the integral of the fitted Normal over the bucket range.

WHAT THIS BUYS US:
  - Bias correction: station-specific + seasonal + model drift all absorbed.
  - Reliable spread: no more over-confident predictions on calm days.
  - Probabilities become calibrated → better Kelly sizing, better edge detection.

TRAINING DATA REQUIREMENT:
  Need a history of (ensemble_forecast, observed_value) pairs. We build this
  from Open-Meteo's historical archive endpoint for each city. Minimum ~30 days
  of data for a stable fit; literature recommends 40+ for production.

WHAT WE DEGRADE TO WHEN NO TRAINING DATA:
  With < 20 training samples, we return the raw ensemble probability but flag
  the record as `post_processing="raw"`. The calibration tracker will later
  tell us whether raw is good enough for this city/horizon — frequently it isn't.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


# ---------------------------------------------------------------------------
# CRPS for a Normal distribution (closed form)
# ---------------------------------------------------------------------------

def crps_normal(obs: float, mu: float, sigma: float) -> float:
    """Continuous Ranked Probability Score for a Gaussian forecast.

    Hersbach (2000). Lower is better. Closed form:
        CRPS = sigma * [ z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (obs - mu) / sigma.
    """
    if sigma <= 0:
        # Degenerate case: treat as point forecast.
        return abs(obs - mu)
    z = (obs - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))


# ---------------------------------------------------------------------------
# NGR fit
# ---------------------------------------------------------------------------

@dataclass
class NGRFit:
    """Fitted NGR parameters for one (city, horizon, variable) combination.

        y ~ N(a + b*ensemble_mean, sqrt(c + d*ensemble_var))
    """
    a: float
    b: float
    c: float  # must be >= 0
    d: float  # must be >= 0
    n_training_samples: int
    mean_crps_train: float
    # Diagnostic: what was the raw CRPS on the same training data?
    mean_crps_raw: float

    def apply(self, ensemble_mean: float, ensemble_var: float) -> tuple[float, float]:
        """Return (calibrated_mu, calibrated_sigma) for the predictive Normal."""
        mu = self.a + self.b * ensemble_mean
        var = max(self.c + self.d * max(ensemble_var, 0.0), 1e-6)
        return mu, math.sqrt(var)

    def probability_in_bucket(
        self, ensemble_mean: float, ensemble_var: float,
        lower: Optional[float], upper: Optional[float],
    ) -> float:
        """P(y in [lower, upper)) under the calibrated Normal."""
        mu, sigma = self.apply(ensemble_mean, ensemble_var)
        p_lo = norm.cdf(lower, loc=mu, scale=sigma) if lower is not None else 0.0
        p_hi = norm.cdf(upper, loc=mu, scale=sigma) if upper is not None else 1.0
        return max(0.0, min(1.0, p_hi - p_lo))


def fit_ngr(
    ensemble_means: list[float],
    ensemble_vars: list[float],
    observations: list[float],
    *,
    initial_params: Optional[tuple[float, float, float, float]] = None,
) -> NGRFit:
    """Fit NGR parameters by minimum-CRPS.

    Uses log-transform on variance coefficients to enforce c>0, d>0 during
    optimisation, then exponentiates back.
    """
    ens_mean_arr = np.array(ensemble_means, dtype=float)
    ens_var_arr = np.array(ensemble_vars, dtype=float)
    obs_arr = np.array(observations, dtype=float)

    # Starting point: intercept=0, slope=1, var components small-positive.
    if initial_params is None:
        a0 = float(obs_arr.mean() - ens_mean_arr.mean())
        b0 = 1.0
        c0 = float(np.var(obs_arr - ens_mean_arr))  # residual variance ≈ c
        d0 = 0.5
        initial_params = (a0, b0, c0, d0)

    def objective(params: np.ndarray) -> float:
        a, b, log_c, log_d = params
        c = math.exp(log_c)
        d = math.exp(log_d)
        mu = a + b * ens_mean_arr
        var = c + d * ens_var_arr
        sigma = np.sqrt(np.maximum(var, 1e-6))
        total = 0.0
        for obs, m, s in zip(obs_arr, mu, sigma):
            total += crps_normal(float(obs), float(m), float(s))
        return total / len(obs_arr)

    a0, b0, c0, d0 = initial_params
    x0 = np.array([a0, b0, math.log(max(c0, 1e-4)), math.log(max(d0, 1e-4))])
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})

    a, b, log_c, log_d = result.x
    c = math.exp(log_c)
    d = math.exp(log_d)

    # Compute raw CRPS for comparison (treat ensemble as N(mean, sqrt(var))).
    raw_total = 0.0
    for obs, m, v in zip(obs_arr, ens_mean_arr, ens_var_arr):
        raw_total += crps_normal(float(obs), float(m), math.sqrt(max(float(v), 1e-6)))
    raw_crps = raw_total / len(obs_arr)

    return NGRFit(
        a=float(a), b=float(b), c=float(c), d=float(d),
        n_training_samples=len(obs_arr),
        mean_crps_train=float(result.fun),
        mean_crps_raw=float(raw_crps),
    )


# ---------------------------------------------------------------------------
# Helper: compute ensemble mean and variance from a list of member values
# ---------------------------------------------------------------------------

def ensemble_moments(members: list[float]) -> tuple[float, float]:
    """Return (mean, variance) of ensemble member values. Sample variance (n-1)."""
    if not members:
        return 0.0, 0.0
    arr = np.array(members, dtype=float)
    mean = float(arr.mean())
    var = float(arr.var(ddof=1)) if len(arr) > 1 else 0.0
    return mean, var


# ---------------------------------------------------------------------------
# Bucket probability with optional NGR
# ---------------------------------------------------------------------------

def bucket_probability(
    *,
    members: list[float],
    lower: Optional[float],
    upper: Optional[float],
    fit: Optional[NGRFit] = None,
) -> tuple[float, str]:
    """Return (probability, method) for bucket [lower, upper).

    If `fit` is provided, use NGR-calibrated Normal. Otherwise count members.
    """
    if fit is not None and members:
        mean, var = ensemble_moments(members)
        p = fit.probability_in_bucket(mean, var, lower, upper)
        return p, "ngr"
    # Raw non-parametric count
    if not members:
        return 0.0, "raw"
    hits = sum(
        1 for v in members
        if (lower is None or v >= lower) and (upper is None or v < upper)
    )
    return hits / len(members), "raw"
