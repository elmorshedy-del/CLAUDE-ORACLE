"""BTC up/down strategy — first real end-to-end strategy.

For each active BTC up/down market (5m or 15m) we:
  1. Compute the time remaining to resolution (horizon in seconds).
  2. Compute fair value P(UP) using GBM with current realized vol.
  3. Fetch the token book for UP and DOWN sides.
  4. Compute edge for BUY UP and BUY DOWN (ignore selling — no inventory yet).
  5. For each sleeve (conservative/balanced/aggressive), if edge meets threshold
     AND fee-adjusted EV is positive, generate an OrderIntent.

Key design details:
- Fees are netted into the edge check before deciding to trade. A sleeve NEVER
  generates an intent whose fee-adjusted EV is negative.
- We check both BUY UP (betting it goes up) and BUY DOWN (betting it goes down).
  The underpriced side is always the one with the better edge.
- We take the MID of the NO side's book as a cross-check on the UP side's book
  (in a two-outcome market, YES + NO should sum to $1.00 ignoring spread).
  If the cross-check disagrees sharply, we flag and skip — book is disjointed.

Sleeves are pure configuration; the strategy logic is identical. That way the
self-correction module only has to tune numbers, not rewrite logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable
from uuid import uuid4

from ..exec.fees import CATEGORY_TAKER_PEAK_RATE, MAKER_REBATE_SHARE, taker_fee_rate
from ..exec.models import (
    MarketCategory,
    OrderBook,
    OrderIntent,
    OrderType,
    Side,
    SleeveConfig,
    SleeveStance,
)
from .fair_value import FairValue, fv_up_down


# ---------------------------------------------------------------------------
# Strategy-level inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketContext:
    """Everything a strategy needs to evaluate one market."""

    market_condition_id: str
    strategy_family: str  # "btc_up_down_5m" or "btc_up_down_15m"
    # Unix seconds until the resolution event.
    seconds_to_resolution: float
    # Current BTC spot (for logging / fair-value recompute).
    spot: float
    sigma_annual: float  # estimated from realized vol
    # Token books keyed by outcome label ("Up" / "Down").
    books: dict[str, OrderBook]
    # Token IDs keyed by outcome label.
    token_ids: dict[str, str]


@dataclass(frozen=True)
class StrategyDecision:
    """Output of a single sleeve's evaluation of one market."""

    intent: OrderIntent | None  # None means "no trade"
    fair_value: FairValue
    # Diagnostic fields always populated even when intent is None.
    chosen_outcome: str | None  # "Up" / "Down" / None
    gross_edge_bps: int          # edge before fees (may be negative)
    net_edge_bps: int            # edge after taker fee (if taker) or rebate (if maker)
    reason_skipped: str | None   # if intent is None, why?


# ---------------------------------------------------------------------------
# Sleeve archetypes
# ---------------------------------------------------------------------------

def default_btc_up_down_sleeves(
    *,
    strategy_family: str,
    total_bankroll_usd: Decimal,
) -> list[SleeveConfig]:
    """Build the three canonical sleeves for a given BTC up/down family.

    Thresholds and sizes chosen to be conservative at launch. Self-correction
    can tune these over time inside hard bounds (see Phase 5).

    Stances differ in:
      - min_edge_bps:     conservative demands bigger edges
      - max_cross_spread: conservative = 0 (post-only), aggressive crosses freely
      - max_position_usd: scaled 0.5% / 1.5% / 3% of bankroll
    """
    bank = total_bankroll_usd
    return [
        SleeveConfig(
            sleeve_id=f"{strategy_family}__conservative",
            stance=SleeveStance.CONSERVATIVE,
            strategy_name="btc_up_down",
            market_selector=f"strategy_family={strategy_family}",
            bankroll_usd=bank,
            max_position_usd=bank * Decimal("0.005"),  # 0.5%
            min_edge_bps=300,  # 3% net edge after fees
            min_gross_edge_bps=300,  # and at least 3% DIRECTIONAL edge — no spread harvesting
            max_cross_spread_bps=0,  # post-only only
            enabled=True,
            version=1,
            notes="Conservative BTC up/down: post-only, 3% gross+net edge",
        ),
        SleeveConfig(
            sleeve_id=f"{strategy_family}__balanced",
            stance=SleeveStance.BALANCED,
            strategy_name="btc_up_down",
            market_selector=f"strategy_family={strategy_family}",
            bankroll_usd=bank,
            max_position_usd=bank * Decimal("0.015"),  # 1.5%
            min_edge_bps=150,
            min_gross_edge_bps=150,
            max_cross_spread_bps=50,
            enabled=True,
            version=1,
            notes="Balanced BTC up/down: limit + post-only, 1.5% gross+net edge",
        ),
        SleeveConfig(
            sleeve_id=f"{strategy_family}__aggressive",
            stance=SleeveStance.AGGRESSIVE,
            strategy_name="btc_up_down",
            market_selector=f"strategy_family={strategy_family}",
            bankroll_usd=bank,
            max_position_usd=bank * Decimal("0.030"),  # 3%
            min_edge_bps=80,
            min_gross_edge_bps=50,  # allow slightly more spread-capture for aggressive
            max_cross_spread_bps=200,
            enabled=True,
            version=1,
            notes="Aggressive BTC up/down: taker allowed, 0.8% net / 0.5% gross edge",
        ),
    ]


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    sleeve: SleeveConfig,
    ctx: MarketContext,
) -> StrategyDecision:
    """Compute a StrategyDecision for one market under one sleeve config."""
    # 1) Fair value for UP.
    fv = fv_up_down(
        sigma_annual=ctx.sigma_annual,
        horizon_seconds=max(ctx.seconds_to_resolution, 1.0),
        spot=ctx.spot,
    )
    p_up = fv.probability
    p_down = 1 - p_up

    book_up = ctx.books.get("Up")
    book_dn = ctx.books.get("Down")
    if book_up is None or book_dn is None:
        return StrategyDecision(
            intent=None, fair_value=fv, chosen_outcome=None,
            gross_edge_bps=0, net_edge_bps=0,
            reason_skipped="missing book for Up or Down side",
        )

    # 2) Cross-check: YES ask + NO ask should be ≈ $1 (slightly more due to spread).
    #    If they sum to < 0.97 or > 1.05 the book is broken or stale.
    ask_up = float(book_up.best_ask) if book_up.best_ask is not None else None
    ask_dn = float(book_dn.best_ask) if book_dn.best_ask is not None else None
    if ask_up is None or ask_dn is None:
        return StrategyDecision(
            intent=None, fair_value=fv, chosen_outcome=None,
            gross_edge_bps=0, net_edge_bps=0,
            reason_skipped="missing best ask on one side",
        )
    cross_sum = ask_up + ask_dn
    if cross_sum < 0.95 or cross_sum > 1.06:
        return StrategyDecision(
            intent=None, fair_value=fv, chosen_outcome=None,
            gross_edge_bps=0, net_edge_bps=0,
            reason_skipped=f"book sanity fail: ask_up+ask_dn={cross_sum:.4f}",
        )

    # 3) Compute edge for BUY UP and BUY DOWN (only buys — Phase 2 never sells).
    edge_up = p_up - ask_up           # gross EV of buying UP at ask
    edge_dn = p_down - ask_dn         # gross EV of buying DOWN at ask

    # Pick the better side.
    if edge_up >= edge_dn:
        chosen = "Up"
        gross_edge = edge_up
        exec_price = ask_up
        p_win = p_up
        book = book_up
    else:
        chosen = "Down"
        gross_edge = edge_dn
        exec_price = ask_dn
        p_win = p_down
        book = book_dn

    gross_edge_bps = int(gross_edge * 10000)

    # 4) Decide taker vs maker based on sleeve stance.
    #    Conservative (max_cross_spread=0) → post-only only.
    #    Others → taker allowed if post-only wouldn't fill.
    category = MarketCategory.CRYPTO
    taker_fee = float(taker_fee_rate(Decimal(str(exec_price)), category))

    # Option A: taker at current ask.
    net_edge_taker = gross_edge - taker_fee * exec_price  # fee is on notional
    net_edge_taker_bps = int(net_edge_taker * 10000)

    # Option B: post-only one tick inside the spread on our side (improves top of book).
    # For a BUY order, "improve" = higher than current best bid.
    best_bid = float(book.best_bid) if book.best_bid is not None else 0.0
    best_ask_side = float(book.best_ask) if book.best_ask is not None else 1.0
    tick = 0.01  # Polymarket's min tick on most markets; we confirm per-market in Phase 3.
    post_price = min(best_bid + tick, best_ask_side - tick)
    if post_price <= 0 or post_price >= 1:
        # Can't post inside; fall back to taker consideration only.
        post_price = 0.0
    # Maker rebate is positive for us (cash received). Net gain on fill:
    #   (p_win - post_price) + rebate
    # Rebate rate: 25% of taker fee at post_price.
    maker_rebate = float(CATEGORY_TAKER_PEAK_RATE[category]) * 4 * post_price * (1 - post_price) * float(MAKER_REBATE_SHARE)
    net_edge_maker = (p_win - post_price) + maker_rebate * post_price if post_price > 0 else -1.0
    net_edge_maker_bps = int(net_edge_maker * 10000)

    # 5) Pick best viable route subject to sleeve rules.
    min_edge_bps = sleeve.min_edge_bps
    min_gross = sleeve.min_gross_edge_bps
    max_cross_bps = sleeve.max_cross_spread_bps

    # HARD GATE: require meaningful DIRECTIONAL edge before any trade.
    # This prevents the strategy from generating intents purely to harvest
    # the bid-ask spread on fairly-priced markets (a market-making strategy
    # needs different risk controls — inventory, adverse selection — than
    # this system has).
    if gross_edge_bps < min_gross:
        return StrategyDecision(
            intent=None, fair_value=fv, chosen_outcome=chosen,
            gross_edge_bps=gross_edge_bps,
            net_edge_bps=max(net_edge_maker_bps, net_edge_taker_bps),
            reason_skipped=(
                f"directional edge too small: gross={gross_edge_bps}bps "
                f"< min_gross={min_gross}bps"
            ),
        )

    best_net_bps = max(net_edge_maker_bps, net_edge_taker_bps)

    route: str | None = None
    # Prefer maker (post_only) if it's viable — smaller edge threshold needed because fees are negative.
    if post_price > 0 and net_edge_maker_bps >= min_edge_bps:
        route = "maker"
    elif net_edge_taker_bps >= min_edge_bps and max_cross_bps > 0:
        # How far inside the spread we'd have to cross: (exec_price - best_bid) in bps of exec_price.
        # For a buy, "crossing" means paying the ask — the spread we cross is (ask - mid).
        mid = (best_bid + best_ask_side) / 2 if best_bid > 0 and best_ask_side < 1 else exec_price
        cross_bps = int((exec_price - mid) / max(mid, 1e-9) * 10000)
        if cross_bps <= max_cross_bps:
            route = "taker"

    if route is None:
        return StrategyDecision(
            intent=None, fair_value=fv, chosen_outcome=chosen,
            gross_edge_bps=gross_edge_bps,
            net_edge_bps=best_net_bps,
            reason_skipped=(
                f"edge too thin: gross={gross_edge_bps}bps, "
                f"maker={net_edge_maker_bps}bps, taker={net_edge_taker_bps}bps, "
                f"threshold={min_edge_bps}bps"
            ),
        )

    # 6) Build the intent.
    size_usd = sleeve.max_position_usd
    if route == "maker":
        order_type = OrderType.POST_ONLY
        limit_price = Decimal(str(post_price))
        used_price = post_price
        used_net_edge_bps = net_edge_maker_bps
    else:
        order_type = OrderType.LIMIT  # LIMIT at the current ask → executes as taker
        limit_price = Decimal(str(exec_price))
        used_price = exec_price
        used_net_edge_bps = net_edge_taker_bps

    intent = OrderIntent(
        sleeve_id=sleeve.sleeve_id,
        market_condition_id=ctx.market_condition_id,
        token_id=ctx.token_ids[chosen],
        side=Side.BUY,
        order_type=order_type,
        limit_price=limit_price,
        size_usd=size_usd,
        category=category,
        edge_bps=used_net_edge_bps,
        reasoning=(
            f"BUY {chosen} at {used_price:.4f} via {route}. "
            f"fv={p_win:.4f} (ci [{fv.ci_low:.4f},{fv.ci_high:.4f}]) "
            f"spot={ctx.spot:.2f} sigma={ctx.sigma_annual:.3f} "
            f"T_sec={ctx.seconds_to_resolution:.0f} "
            f"gross={gross_edge_bps}bps net={used_net_edge_bps}bps"
        ),
        client_order_id=f"intent_{uuid4().hex[:16]}",
    )

    return StrategyDecision(
        intent=intent,
        fair_value=fv,
        chosen_outcome=chosen,
        gross_edge_bps=gross_edge_bps,
        net_edge_bps=used_net_edge_bps,
        reason_skipped=None,
    )


def evaluate_all(
    sleeves: Iterable[SleeveConfig],
    ctx: MarketContext,
) -> list[StrategyDecision]:
    """Convenience: evaluate one market against multiple sleeves."""
    return [evaluate(s, ctx) for s in sleeves]
