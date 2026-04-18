"""Fractional Kelly bet sizing.

Kelly Criterion (Kelly 1956) computes the bankroll fraction that maximizes
expected geometric growth given a known edge:

    f* = (b*p - q) / b
         = (p * b - (1-p)) / b
         = p - (1-p)/b

where:
    p = probability of winning
    q = 1 - p
    b = net decimal odds on win (for Polymarket: b = (1/ask) - 1)

For YES token trading at ask price `a` with our fair-value `p`:
    win payoff   = (1 - a) / a per unit staked
    loss payoff  = -1 per unit staked
    b = (1 - a) / a
    f* = p - (1-p) * a / (1-a)

IMPORTANT CAVEATS FROM LITERATURE:
  - Full Kelly assumes EXACT knowledge of p. Over-estimating p leads to ruin.
  - Fractional Kelly (half, quarter) trades off growth rate for variance +
    protection against model error. Half-Kelly gets ~75% of full Kelly growth
    with ~50% of the variance (Thorp, Downey).
  - Correlated bets: Kelly per-bet overestimates safe size when bets are
    correlated. In our weather case, multiple buckets within the same event
    are perfectly correlated (exactly one can win) — we cap total stake
    across correlated buckets.
  - When uncertainty on p is large, shrink further. We use Kelly/4 as default.

OUTPUT: a dollar stake, computed as `kelly_fraction * current_bankroll`, then
clipped by max_position_usd per-sleeve and by book depth per-venue.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


@dataclass
class KellyResult:
    full_kelly: float        # f* (can be negative → don't bet)
    fractional: float        # kelly fraction we'll actually use (clipped)
    stake_usd: Decimal       # recommended dollar stake
    expected_growth_rate: float  # at `fractional` size, log growth per trade
    rationale: str


def kelly_for_yes_buy(
    *,
    probability: float,
    ask_price: float,
    bankroll_usd: Decimal,
    kelly_fraction: float = 0.25,
    max_fraction: float = 0.05,
    uncertainty_penalty: float = 1.0,
) -> KellyResult:
    """Kelly stake for buying YES at `ask_price` with our estimated `probability`.

    Args:
      probability: our fair-value P(YES).
      ask_price: market price per YES share, in [0, 1].
      bankroll_usd: current capital.
      kelly_fraction: 0.25 = quarter-Kelly (default). Literature standard.
      max_fraction: absolute cap on fraction of bankroll per trade (default 5%).
      uncertainty_penalty: multiplicative shrinkage when we're unsure of p.
        1.0 = no shrinkage; 0.5 = double-down on fractional Kelly.

    Returns KellyResult. `stake_usd` is 0 if edge is non-positive.
    """
    if ask_price <= 0 or ask_price >= 1:
        return KellyResult(
            full_kelly=0.0, fractional=0.0, stake_usd=Decimal("0"),
            expected_growth_rate=0.0,
            rationale=f"invalid ask={ask_price}",
        )
    if probability <= ask_price:
        # No edge. Never bet.
        return KellyResult(
            full_kelly=0.0, fractional=0.0, stake_usd=Decimal("0"),
            expected_growth_rate=0.0,
            rationale=f"no edge: p={probability:.4f} <= ask={ask_price:.4f}",
        )
    # Kelly formula
    b = (1.0 - ask_price) / ask_price
    p = probability
    q = 1.0 - p
    f_star = (b * p - q) / b   # == p - q/b == p - (1-p) * ask / (1-ask)

    if f_star <= 0:
        return KellyResult(
            full_kelly=f_star, fractional=0.0, stake_usd=Decimal("0"),
            expected_growth_rate=0.0,
            rationale=f"Kelly<=0 (f*={f_star:.4f})",
        )

    # Apply fractional Kelly + uncertainty shrinkage + hard cap.
    frac = min(f_star * kelly_fraction * uncertainty_penalty, max_fraction)
    # Expected log-growth at this fraction:
    # E[log(1 + X*f)] where X is win payoff per unit stake.
    # For a single YES bet: win = +(1-ask)/ask, loss = -1
    #   E[log] = p*log(1 + f*b) + (1-p)*log(1 - f)
    try:
        import math
        eg = p * math.log(1 + frac * b) + q * math.log(max(1 - frac, 1e-9))
    except Exception:
        eg = 0.0

    stake = (Decimal(str(frac)) * bankroll_usd).quantize(Decimal("0.01"))
    return KellyResult(
        full_kelly=f_star,
        fractional=frac,
        stake_usd=stake,
        expected_growth_rate=eg,
        rationale=(
            f"Kelly: p={p:.4f} ask={ask_price:.4f} b={b:.3f} f*={f_star:.4f} "
            f"→ frac={frac:.4f} (κ={kelly_fraction}, cap={max_fraction}, "
            f"penalty={uncertainty_penalty})"
        ),
    )


def correlation_adjusted_stake(
    *,
    individual_stakes_usd: list[Decimal],
    max_group_fraction_of_bankroll: float,
    bankroll_usd: Decimal,
) -> list[Decimal]:
    """Scale down correlated bets so the sum of stakes doesn't exceed a group cap.

    Weather-buckets within a single event are mutually exclusive YES tokens.
    Buying all of them is equivalent to buying "any of these buckets wins",
    which is a less risky (but also less rewarding) composite. Kelly-sum > 1
    is dangerous because bets are perfectly negatively correlated.

    Cap: total stake on the group ≤ max_group_fraction_of_bankroll * bankroll.
    """
    if not individual_stakes_usd:
        return []
    total = sum(individual_stakes_usd, Decimal("0"))
    cap = Decimal(str(max_group_fraction_of_bankroll)) * bankroll_usd
    if total <= cap:
        return list(individual_stakes_usd)
    scale = cap / total if total > 0 else Decimal("1")
    return [(s * scale).quantize(Decimal("0.01")) for s in individual_stakes_usd]
