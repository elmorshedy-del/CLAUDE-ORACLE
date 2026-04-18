"""Calendar-ladder arb detector — the ONE real risk-free trade inside nested date ladders.

Context: Events like "Netanyahu out by ___" have N markets at increasing dates.
Outcomes are NESTED, not mutually exclusive:
    P(out by Apr) <= P(out by Jun) <= P(out by Dec)

There is NO simple "buy all YES" arb here. But there IS a structural
arbitrage when prices violate monotonicity:

    If ask(short_date) > bid(long_date), you can:
        - BUY long_date YES  @ ask_long  (pays off IF out by long_date)
        - SELL short_date YES @ bid_short (pays out IF NOT out by short_date,
          and you keep the difference; equivalent to buying NO on short_date)
    Net exposure: paid (ask_long - bid_short) upfront.
    At resolution:
        - out by short_date: long_date ALSO resolves YES → net payoff = +1 - 1 = 0
          (you lose the initial stake minus fees)
        - out between short_date and long_date: long resolves YES, short resolved NO
          → net payoff = +1 - 0 = +1 (big win)
        - not out by long_date: both resolve NO → net payoff = 0 - 0 = 0
          (you lose initial stake)

    ACTUALLY this isn't a pure arb either — it's a LIMITED-LOSS calendar spread.

The ONE actual risk-free trade:
    If  ask(short_date) + bid_NO(long_date) < 1.00  ... hmm that's also not right.

Let me think carefully:
    market_short: resolves YES if out by date T1.
    market_long:  resolves YES if out by date T2 > T1.
    Define events:
        A = out by T1           -> short=YES, long=YES
        B = out in (T1, T2]     -> short=NO,  long=YES
        C = not out by T2       -> short=NO,  long=NO

    Legs: sell short_YES (receive bid_short), buy long_YES (pay ask_long).
        case A: payoff = -1 + 1 = 0. Net = 0 + bid_short - ask_long.
        case B: payoff =  0 + 1 = 1. Net = 1 + bid_short - ask_long.
        case C: payoff =  0 + 0 = 0. Net = 0 + bid_short - ask_long.

    So min_payoff = bid_short - ask_long (in A and C).
    max_payoff   = bid_short - ask_long + 1 (in B).

    For a RISK-FREE ARB: need min_payoff >= 0 → bid_short >= ask_long.
    That's the actual arb condition: **earlier-date bid must exceed later-date ask**.

On Netanyahu right now:
    - bid(Apr 30) = 0.006
    - ask(Jun 30) = 0.060
    - bid(Jun 30) = 0.050
    - ask(Dec 31) = 0.430

    bid(Apr) = 0.006 < ask(Jun) = 0.060 → NO arb
    bid(Jun) = 0.050 < ask(Dec) = 0.430 → NO arb
    Prices respect monotonicity. Efficient.

But this CAN be violated by stale quotes. Worth scanning.

What this module does:
  1. Detect date-ladder events by parsing market titles / endDate fields.
  2. Order the markets chronologically.
  3. For each adjacent pair (T1, T2), check if bid(T1) >= ask(T2) after fees.
  4. If yes, emit a 2-leg bundle: SELL T1_YES + BUY T2_YES of equal size.

We also optionally expose CALENDAR-SPREAD opportunities (not risk-free, but
a structured directional bet on when the event occurs) — these get tagged
"calendar_spread" rather than "arb" so they never trade under a true-arb gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterable

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
from uuid import uuid4


@dataclass(frozen=True)
class LadderRung:
    """One date-indexed YES market in a nested date-ladder event."""

    token_id: str
    market_condition_id: str
    end_date_iso: str           # e.g. "2026-04-30"
    end_timestamp: int          # unix
    question: str
    book: OrderBook
    best_bid: Decimal | None
    best_ask: Decimal | None
    best_bid_size: Decimal
    best_ask_size: Decimal


@dataclass(frozen=True)
class LadderContext:
    event_slug: str
    category: MarketCategory
    rungs: list[LadderRung]      # sorted by end_timestamp ASCENDING


@dataclass(frozen=True)
class LadderDecision:
    """One arb opportunity detected on an adjacent rung pair."""

    short_rung: LadderRung
    long_rung: LadderRung
    intents: list[OrderIntent]
    # 'arb_monotonicity' = risk-free arb (rare).
    # 'calendar_spread' = structured directional bet (limited loss; not pure arb).
    kind: str
    gross_edge_bps: int
    net_edge_bps: int
    reason_skipped: str | None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def is_date_ladder_event(event: dict) -> bool:
    """Heuristic: event has >= 2 tradeable markets, all binary YES/NO, NOT negRisk,
    AND the event title matches a date-ladder pattern ('by ...', 'before ...',
    'all time high by ...', etc.), AND markets have distinct endDates.
    """
    mkts = event.get("markets") or []
    tradeable = [
        m for m in mkts
        if m.get("acceptingOrders") and m.get("enableOrderBook")
        and not m.get("negRisk")
    ]
    if len(tradeable) < 2:
        return False
    # All binary YES/NO?
    for m in tradeable:
        outcomes = m.get("outcomes")
        if isinstance(outcomes, str):
            import json as _json
            outcomes = _json.loads(outcomes)
        if not outcomes or len(outcomes) != 2:
            return False
        if {o.lower() for o in outcomes} != {"yes", "no"}:
            return False
    # Distinct end-dates?
    dates = {m.get("endDate", "")[:10] for m in tradeable}
    if len(dates) < 2:
        return False
    # Event title hints?
    title = (event.get("title", "") or event.get("slug", "") or "").lower()
    hint_patterns = [
        "by ___", "by...", "out by", "all time high by", "ipo by",
        "hit $", "before 2027", "before 2026", "before gta",
    ]
    return any(h in title for h in hint_patterns)


# ---------------------------------------------------------------------------
# Sleeves
# ---------------------------------------------------------------------------

def default_ladder_sleeves(
    *,
    total_bankroll_usd: Decimal,
) -> list[SleeveConfig]:
    """Sleeves for date-ladder monotonicity arbs.

    Only emits TRUE arbs (monotonicity violations). Calendar spreads are
    disabled at the sleeve level for now — they need a separate sleeve with
    different risk controls (they have bounded loss, not zero loss).
    """
    bank = total_bankroll_usd
    return [
        SleeveConfig(
            sleeve_id="ladder_arb__conservative",
            stance=SleeveStance.CONSERVATIVE,
            strategy_name="ladder_arb",
            market_selector="kind=date_ladder",
            bankroll_usd=bank,
            max_position_usd=bank * Decimal("0.01"),
            min_edge_bps=50,
            min_gross_edge_bps=50,
            max_cross_spread_bps=10000,  # arbs need taking liquidity
            enabled=True,
            version=1,
            notes="Date-ladder monotonicity arb, conservative",
        ),
        SleeveConfig(
            sleeve_id="ladder_arb__aggressive",
            stance=SleeveStance.AGGRESSIVE,
            strategy_name="ladder_arb",
            market_selector="kind=date_ladder",
            bankroll_usd=bank,
            max_position_usd=bank * Decimal("0.05"),
            min_edge_bps=10,
            min_gross_edge_bps=10,
            max_cross_spread_bps=10000,
            enabled=True,
            version=1,
            notes="Date-ladder monotonicity arb, aggressive",
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_ladder(
    sleeve: SleeveConfig,
    ctx: LadderContext,
) -> list[LadderDecision]:
    """Walk adjacent rung pairs; emit a decision per viable arb."""
    decisions: list[LadderDecision] = []
    rungs = ctx.rungs
    # Also consider non-adjacent pairs: any short-date / long-date pair where
    # short's end < long's end. This catches arbs across skipped rungs.
    for i, short in enumerate(rungs):
        for long in rungs[i + 1 :]:
            decisions.append(_evaluate_pair(sleeve, ctx, short, long))
    return [d for d in decisions if d.intents or d.reason_skipped]


def _evaluate_pair(
    sleeve: SleeveConfig,
    ctx: LadderContext,
    short: LadderRung,
    long: LadderRung,
) -> LadderDecision:
    """Check the arb condition bid(short) >= ask(long)."""
    bid_short = short.best_bid
    ask_long = long.best_ask
    if bid_short is None or ask_long is None:
        return LadderDecision(
            short_rung=short, long_rung=long, intents=[],
            kind="arb_monotonicity", gross_edge_bps=0, net_edge_bps=0,
            reason_skipped="missing quotes",
        )

    # Gross arb margin (no fees yet).
    gross = bid_short - ask_long
    gross_bps = int(gross * Decimal("10000"))

    if gross <= 0:
        return LadderDecision(
            short_rung=short, long_rung=long, intents=[],
            kind="arb_monotonicity", gross_edge_bps=gross_bps, net_edge_bps=gross_bps,
            reason_skipped=f"no monotonicity violation (bid_short={bid_short} ask_long={ask_long})",
        )

    # Fee on the SELL at bid_short (taker): fee ≈ taker_rate(bid_short) × bid_short.
    # Fee on the BUY at ask_long (taker): fee ≈ taker_rate(ask_long) × ask_long.
    # Both are fractions of notional = size × price.
    fee_short = taker_fee_rate(bid_short, ctx.category) * bid_short
    fee_long = taker_fee_rate(ask_long, ctx.category) * ask_long
    fee_total_frac = fee_short + fee_long

    net = gross - fee_total_frac
    net_bps = int(net * Decimal("10000"))

    if net_bps < sleeve.min_edge_bps:
        return LadderDecision(
            short_rung=short, long_rung=long, intents=[],
            kind="arb_monotonicity", gross_edge_bps=gross_bps, net_edge_bps=net_bps,
            reason_skipped=(
                f"arb too thin after fees: gross={gross_bps}bps, "
                f"fees={int(fee_total_frac*10000)}bps, net={net_bps}bps"
            ),
        )

    # Size: cap at min(bid_short_size, ask_long_size, max_position_usd).
    min_size = min(short.best_bid_size, long.best_ask_size)
    max_shares_by_cash = sleeve.max_position_usd / max(ask_long, Decimal("0.0001"))
    shares = min(min_size, max_shares_by_cash)
    if shares <= 0:
        return LadderDecision(
            short_rung=short, long_rung=long, intents=[],
            kind="arb_monotonicity", gross_edge_bps=gross_bps, net_edge_bps=net_bps,
            reason_skipped="zero size",
        )

    bundle_id = f"ladder_{uuid4().hex[:10]}"
    # NOTE: Phase 2 paper mode does BUY-only (no short sells). A "SELL YES short_rung"
    # is equivalent to "BUY NO short_rung" on the other token of the same condition.
    # We don't have the NO token_id in our LadderRung (we only stored the YES side).
    # For Phase 2 we EMIT THE BUY LEG and log a note explaining the short leg is
    # pending; the full bundle will be executable in Phase 3 when we track both
    # YES and NO tokens per market.

    intents: list[OrderIntent] = [
        OrderIntent(
            sleeve_id=sleeve.sleeve_id,
            market_condition_id=long.market_condition_id,
            token_id=long.token_id,
            side=Side.BUY,
            order_type=OrderType.LIMIT,
            limit_price=ask_long,
            size_shares=shares,
            category=ctx.category,
            edge_bps=net_bps,
            reasoning=(
                f"ladder arb leg 1 of 2 (bundle {bundle_id}): "
                f"BUY long YES ({long.question[:50]}) at {ask_long:.4f}. "
                f"Short leg at bid_short={bid_short:.4f} on {short.question[:50]} "
                f"pending NO-token plumbing (Phase 3). "
                f"gross={gross_bps}bps net={net_bps}bps"
            ),
            client_order_id=f"intent_{bundle_id}_long_{uuid4().hex[:8]}",
        ),
    ]

    return LadderDecision(
        short_rung=short, long_rung=long, intents=intents,
        kind="arb_monotonicity", gross_edge_bps=gross_bps, net_edge_bps=net_bps,
        reason_skipped=None,
    )


# ---------------------------------------------------------------------------
# Context-building helper
# ---------------------------------------------------------------------------

def _parse_end_ts(iso: str) -> int:
    """Parse ISO datetime → unix seconds."""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return 0


def build_ladder_context(
    *,
    event_slug: str,
    category: MarketCategory,
    rungs_raw: list[dict],  # {token_id, market_condition_id, end_date_iso, question, book}
) -> LadderContext:
    rungs = []
    for r in rungs_raw:
        book: OrderBook = r["book"]
        rungs.append(LadderRung(
            token_id=r["token_id"],
            market_condition_id=r["market_condition_id"],
            end_date_iso=r["end_date_iso"],
            end_timestamp=_parse_end_ts(r["end_date_iso"]),
            question=r["question"],
            book=book,
            best_bid=book.best_bid,
            best_ask=book.best_ask,
            best_bid_size=(book.bids[0].size if book.bids else Decimal("0")),
            best_ask_size=(book.asks[0].size if book.asks else Decimal("0")),
        ))
    rungs.sort(key=lambda r: r.end_timestamp)
    return LadderContext(event_slug=event_slug, category=category, rungs=rungs)
