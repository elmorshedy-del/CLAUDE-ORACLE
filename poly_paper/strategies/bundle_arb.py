"""Bundle arbitrage — risk-free trades from outcome-price inconsistencies.

The core observation: in a market where outcomes are mutually exclusive and
collectively exhaustive, the sum of ask prices across ALL outcomes should be
AT LEAST $1.00 (otherwise someone buying every side wins a guaranteed $1.00
while paying less than $1.00).

Examples on Polymarket:
- Binary "Up or Down": 2 outcomes. Sum of asks should be >= 1.00.
- Multi-outcome "What price will BTC hit in 2026": N buckets. Sum of asks
  across all mutually-exclusive buckets should be >= 1.00.

When `sum_asks < 1.00`, the gap (after accounting for taker fees on each leg)
is RISK-FREE profit per $1.00 of guaranteed payout.

This module detects arbs and generates a BUNDLE of matched OrderIntents
(one per leg). The runner executes them together; if any leg fails to fill,
the others must be cancelled / inventoried. Phase 2 treats this as
"all-or-nothing in paper" (we assume all fills succeed or none do); Phase 3
adds leg-by-leg failure handling for live mode.

Edge math:
    sum_asks = sum of best_ask across all outcomes
    gap = 1.00 - sum_asks
    total_fee_rate ≈ sum over legs of taker_fee_rate(price_i) × price_i
    net_edge = gap - total_fee_rate
    require: net_edge >= min_edge_bps (sleeve threshold)

Size: capped at the minimum best_ask_size across legs, and by max_position_usd.

Why this works on Polymarket specifically:
    Polymarket's CLOB matching is off-chain but order submission is signed.
    Stale quotes on less-trafficked markets CAN leave arbs on the board for
    seconds-to-minutes. Plus, new 5m/15m BTC markets are created continuously,
    and their initial books are often mis-stitched for a brief window.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence
from uuid import uuid4

from ..exec.fees import taker_fee_rate
from ..exec.models import (
    MarketCategory,
    OrderBook,
    OrderIntent,
    OrderType,
    Side,
    SleeveConfig,
    SleeveStance,
)


@dataclass(frozen=True)
class OutcomeQuote:
    outcome: str
    token_id: str
    book: OrderBook


@dataclass(frozen=True)
class ArbContext:
    """Everything a bundle arb sleeve needs to evaluate one market group."""

    market_condition_id: str
    category: MarketCategory
    quotes: Sequence[OutcomeQuote]  # one per mutually-exclusive outcome
    # For binary markets this is 2; for multi-outcome events it's N.


@dataclass(frozen=True)
class ArbDecision:
    """One decision, containing zero or more intents (all of a bundle, or none)."""

    intents: list[OrderIntent]
    gap_bps: int           # 1.00 - sum_asks, in bps
    net_edge_bps: int      # after taker fees
    reason_skipped: str | None
    chosen_size_shares: Decimal | None = None


# ---------------------------------------------------------------------------
# Sleeve definitions
# ---------------------------------------------------------------------------

def default_bundle_arb_sleeves(
    *,
    strategy_family: str,
    total_bankroll_usd: Decimal,
) -> list[SleeveConfig]:
    """Three stances for bundle arb.

    Conservative: only takes arbs with fat margin post-fees (>= 100bps net).
                  Small notionals, won't touch illiquid books.
    Balanced:     ~40bps net threshold, medium notionals.
    Aggressive:   ~15bps net threshold, larger notionals, accepts thinner depth.
    """
    bank = total_bankroll_usd
    return [
        SleeveConfig(
            sleeve_id=f"{strategy_family}__conservative",
            stance=SleeveStance.CONSERVATIVE,
            strategy_name="bundle_arb",
            market_selector=f"strategy_family={strategy_family}",
            bankroll_usd=bank,
            max_position_usd=bank * Decimal("0.02"),   # 2% per arb
            min_edge_bps=100,       # 1% net after fees
            min_gross_edge_bps=100,
            max_cross_spread_bps=1000,  # arbs require taking liquidity
            enabled=True,
            version=1,
            notes="Conservative bundle arb: 1.00% net edge threshold, taker-only",
        ),
        SleeveConfig(
            sleeve_id=f"{strategy_family}__balanced",
            stance=SleeveStance.BALANCED,
            strategy_name="bundle_arb",
            market_selector=f"strategy_family={strategy_family}",
            bankroll_usd=bank,
            max_position_usd=bank * Decimal("0.05"),
            min_edge_bps=40,
            min_gross_edge_bps=40,
            max_cross_spread_bps=1000,
            enabled=True,
            version=1,
            notes="Balanced bundle arb: 0.40% net edge threshold",
        ),
        SleeveConfig(
            sleeve_id=f"{strategy_family}__aggressive",
            stance=SleeveStance.AGGRESSIVE,
            strategy_name="bundle_arb",
            market_selector=f"strategy_family={strategy_family}",
            bankroll_usd=bank,
            max_position_usd=bank * Decimal("0.10"),
            min_edge_bps=15,
            min_gross_edge_bps=15,
            max_cross_spread_bps=1000,
            enabled=True,
            version=1,
            notes="Aggressive bundle arb: 0.15% net edge threshold",
        ),
    ]


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_bundle(
    sleeve: SleeveConfig,
    ctx: ArbContext,
) -> ArbDecision:
    """Check whether a risk-free arb exists and, if so, build a matched bundle."""
    # 1) Need at least 2 outcomes, each with a visible best_ask and some size.
    if len(ctx.quotes) < 2:
        return ArbDecision(intents=[], gap_bps=0, net_edge_bps=0,
                           reason_skipped="too few outcomes")

    asks: list[tuple[OutcomeQuote, Decimal, Decimal]] = []  # (q, price, size)
    for q in ctx.quotes:
        if q.book.best_ask is None or not q.book.asks:
            return ArbDecision(intents=[], gap_bps=0, net_edge_bps=0,
                               reason_skipped=f"missing ask for outcome {q.outcome}")
        asks.append((q, q.book.best_ask, q.book.asks[0].size))

    # 2) Compute sum of asks and raw gap.
    sum_asks = sum((p for _, p, _ in asks), Decimal("0"))
    if sum_asks >= Decimal("1"):
        return ArbDecision(intents=[], gap_bps=0, net_edge_bps=0,
                           reason_skipped=f"no arb: sum_asks={sum_asks:.4f} >= 1.0")
    gap = Decimal("1") - sum_asks
    gap_bps = int(gap * Decimal("10000"))

    # 3) Compute total taker fees over the bundle.
    # If we buy size S shares on each leg, the GUARANTEED payout is $S (one
    # outcome resolves YES paying $1/share). Total cost before fees = sum_asks × S.
    # Fee per leg = taker_fee_rate(p_i) × (p_i × S). So fee-as-fraction-of-payout is
    # sum over legs of taker_fee_rate(p_i) × p_i.
    fee_fraction = sum(
        (taker_fee_rate(p, ctx.category) * p for _, p, _ in asks),
        Decimal("0"),
    )
    net_edge = gap - fee_fraction
    net_edge_bps = int(net_edge * Decimal("10000"))

    if net_edge_bps < sleeve.min_edge_bps:
        return ArbDecision(
            intents=[], gap_bps=gap_bps, net_edge_bps=net_edge_bps,
            reason_skipped=(
                f"arb too thin after fees: gap={gap_bps}bps, fees={int(fee_fraction*10000)}bps, "
                f"net={net_edge_bps}bps < threshold {sleeve.min_edge_bps}bps"
            ),
        )

    # 4) Determine shares per leg.
    # Bundle must buy equal shares on every leg (or the guarantee breaks).
    # Size cap from:
    #   (a) shallowest leg's best_ask_size (we only consume top-level quote to
    #       keep execution clean; walking the ladder per leg in paper is fine,
    #       but introduces realizable-slippage risk we defer to Phase 3).
    #   (b) max_position_usd / sum_asks (total dollars into the bundle).
    min_ask_size = min(size for _, _, size in asks)
    max_shares_by_cash = sleeve.max_position_usd / max(sum_asks, Decimal("0.0001"))
    target_shares = min(min_ask_size, max_shares_by_cash)

    if target_shares <= 0:
        return ArbDecision(intents=[], gap_bps=gap_bps, net_edge_bps=net_edge_bps,
                           reason_skipped="zero viable size")

    # 5) Build the bundle of intents (one per leg). A common group_id in
    # reasoning ties them together in logs for later analysis.
    bundle_id = f"arb_{uuid4().hex[:10]}"
    intents: list[OrderIntent] = []
    for q, price, _size in asks:
        intents.append(
            OrderIntent(
                sleeve_id=sleeve.sleeve_id,
                market_condition_id=ctx.market_condition_id,
                token_id=q.token_id,
                side=Side.BUY,
                # LIMIT at the best ask executes as taker (we cross the spread exactly).
                order_type=OrderType.LIMIT,
                limit_price=price,
                size_shares=target_shares,
                category=ctx.category,
                edge_bps=net_edge_bps,
                reasoning=(
                    f"bundle {bundle_id} leg: BUY {q.outcome} at {price:.4f}. "
                    f"sum_asks={sum_asks:.4f} gap={gap_bps}bps "
                    f"fee={int(fee_fraction*10000)}bps net={net_edge_bps}bps "
                    f"bundle_size={target_shares}"
                ),
                client_order_id=f"intent_{bundle_id}_{q.outcome[:6]}_{uuid4().hex[:8]}",
            )
        )

    return ArbDecision(
        intents=intents,
        gap_bps=gap_bps,
        net_edge_bps=net_edge_bps,
        reason_skipped=None,
        chosen_size_shares=target_shares,
    )
