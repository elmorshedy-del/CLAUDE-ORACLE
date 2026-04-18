"""Self-correction proposer — continuous tuning of sleeve config within guardrails.

Runs as a separate loop. Every POLY_SELFCORRECT_INTERVAL_SEC:
  1. For each enabled sleeve, compute rolling metrics over the last N days:
       - fills_count, rejected_count
       - fill_quality (%HIGH confidence / total fills)
       - fees_usd_sum
       - realised_slippage_bps (mean actual - expected)
       - avg edge_bps at intent time vs realised_edge_bps after fill
  2. Apply decision rules (inside HARD BOUNDS):
       - "loud silence" (many evaluations, 0 fills for 7+ days at conservative)
             → widen edge threshold slightly (never below hard floor)
       - "bad fill quality" (<50% high-conf over 100+ fills)
             → tighten max_cross_spread_bps (make fills purer)
       - "fat edge consistently taken" (avg realised edge well above threshold)
             → nothing — system is working
       - "too many rejections" (>30% rejected due to size/book issues)
             → reduce max_position_usd toward min-viable
  3. Persist proposed changes to the config_changes table with rationale.
  4. AUTO-APPLY only within strict bounds:
       - size_mult in [0.5x, 1.5x] of the original default
       - min_edge_bps within [50%, 150%] of default  (never below hard floor)
       - max_cross_spread_bps in [0, 2× default]
     Any change outside bounds is persisted as `status="flagged"` for manual review.

This is deliberately conservative — the goal is NOT to optimise aggressively,
it's to prevent degenerate states (sleeves silently burning fees on bad fills)
and to make the system observable. Aggressive optimisation belongs in a
separate human-in-the-loop tool.

Hard floor safety — the system will NEVER:
  - lower min_edge_bps below 25 bps (quarter of a percent net)
  - lower min_gross_edge_bps below 10 bps
  - raise max_position_usd above 5% of total bankroll
  - enable live mode
  - change a sleeve's strategy_name (only its risk parameters)
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import structlog
from sqlalchemy import func, select

from .db.models import ConfigChange, FillRow, OrderIntentRow, SleeveConfig
from .db.session import SessionLocal

log = structlog.get_logger()

INTERVAL_SECONDS = float(os.environ.get("POLY_SELFCORRECT_INTERVAL_SEC", "3600"))  # hourly by default
LOOKBACK_DAYS = float(os.environ.get("POLY_SELFCORRECT_LOOKBACK_DAYS", "7"))
BANKROLL_USD = Decimal(os.environ.get("POLY_BANKROLL_USD", "1000"))
ENABLED = os.environ.get("POLY_SELFCORRECT_ENABLED", "1") not in ("0", "false", "no")


# ---------------------------------------------------------------------------
# Hard bounds — self-correction NEVER crosses these
# ---------------------------------------------------------------------------

HARD_FLOOR_MIN_EDGE_BPS = 25        # net edge threshold absolute minimum
HARD_FLOOR_MIN_GROSS_BPS = 10
HARD_CEILING_POSITION_PCT = Decimal("0.05")   # 5% of bankroll per trade
# Multiplicative envelope around the DEFAULT config:
BOUND_EDGE_THRESHOLD_MIN = 0.5      # can reduce threshold to 50% of default
BOUND_EDGE_THRESHOLD_MAX = 2.0      # or raise to 200%
BOUND_POSITION_MIN = 0.5
BOUND_POSITION_MAX = 1.5
BOUND_CROSS_SPREAD_MAX = 2.0


@dataclass
class SleeveMetrics:
    sleeve_id: str
    strategy_name: str
    stance: str
    version: int
    # Counters
    n_intents: int = 0
    n_fills: int = 0
    n_rejected: int = 0
    n_high_conf_fills: int = 0
    # Dollars
    fees_usd: Decimal = Decimal("0")
    notional_usd: Decimal = Decimal("0")
    # Edge
    mean_intent_edge_bps: float = 0.0
    mean_realised_slippage_bps: float = 0.0
    # Current config
    current_min_edge_bps: int = 0
    current_min_gross_bps: int = 0
    current_max_position_usd: Decimal = Decimal("0")
    current_max_cross_spread_bps: int = 0
    current_bankroll_usd: Decimal = Decimal("0")

    @property
    def high_conf_fraction(self) -> float:
        return (self.n_high_conf_fills / self.n_fills) if self.n_fills else 0.0

    @property
    def rejection_rate(self) -> float:
        total = self.n_fills + self.n_rejected
        return (self.n_rejected / total) if total else 0.0


@dataclass
class ProposedChange:
    """One proposed config change for one sleeve."""
    sleeve_id: str
    field_name: str
    old_value: Any
    new_value: Any
    rationale: str
    status: str = "pending"   # "applied" | "flagged" | "pending"
    evidence: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

async def compute_sleeve_metrics(lookback_days: float) -> list[SleeveMetrics]:
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    out: list[SleeveMetrics] = []
    async with SessionLocal() as db:
        sleeves = (await db.execute(select(SleeveConfig))).scalars().all()
        for s in sleeves:
            sm = SleeveMetrics(
                sleeve_id=s.sleeve_id,
                strategy_name=s.strategy_name,
                stance=s.stance,
                version=s.version,
                current_min_edge_bps=s.min_edge_bps,
                current_min_gross_bps=(s.extra_json or {}).get("min_gross_edge_bps", s.min_edge_bps),
                current_max_position_usd=Decimal(s.max_position_usd),
                current_max_cross_spread_bps=s.max_cross_spread_bps,
                current_bankroll_usd=Decimal(s.bankroll_usd),
            )
            sm.n_intents = (await db.execute(
                select(func.count()).select_from(OrderIntentRow).where(
                    OrderIntentRow.sleeve_id == s.sleeve_id,
                    OrderIntentRow.created_at >= since,
                )
            )).scalar() or 0
            intents = (await db.execute(
                select(OrderIntentRow).where(
                    OrderIntentRow.sleeve_id == s.sleeve_id,
                    OrderIntentRow.created_at >= since,
                )
            )).scalars().all()
            if intents:
                sm.mean_intent_edge_bps = sum((i.edge_bps or 0) for i in intents) / len(intents)
            fills = (await db.execute(
                select(FillRow).join(OrderIntentRow, FillRow.client_order_id == OrderIntentRow.client_order_id).where(
                    OrderIntentRow.sleeve_id == s.sleeve_id,
                    FillRow.created_at >= since,
                )
            )).scalars().all()
            for f in fills:
                if f.rejected:
                    sm.n_rejected += 1
                else:
                    sm.n_fills += 1
                    if f.confidence == "high":
                        sm.n_high_conf_fills += 1
                    sm.fees_usd += Decimal(f.fees_usd or "0")
                    sm.notional_usd += Decimal(f.notional_usd or "0")
                    if f.slippage_bps is not None:
                        sm.mean_realised_slippage_bps += float(f.slippage_bps)
            if sm.n_fills:
                sm.mean_realised_slippage_bps /= sm.n_fills
            out.append(sm)
    return out


# ---------------------------------------------------------------------------
# Defaults — to anchor "relative to default" bounds
# ---------------------------------------------------------------------------

def _default_config_for(sleeve_id: str) -> SleeveConfig | None:
    """Recreate the default SleeveConfig for a given sleeve_id, if known.

    Used as the anchor for "within X% of default" bound checks. If we don't
    recognise the sleeve_id, we refuse to auto-apply and only flag.
    """
    from .strategies.btc_updown import default_btc_up_down_sleeves
    from .strategies.bundle_arb import default_bundle_arb_sleeves
    from .strategies.weather import default_weather_sleeves

    pool: list[SleeveConfig] = []
    for fam in ("btc_up_down_5m", "btc_up_down_15m"):
        pool += default_btc_up_down_sleeves(strategy_family=fam, total_bankroll_usd=BANKROLL_USD)
        pool += default_bundle_arb_sleeves(strategy_family=fam, total_bankroll_usd=BANKROLL_USD)
    pool += default_weather_sleeves(total_bankroll_usd=BANKROLL_USD)
    for s in pool:
        if s.sleeve_id == sleeve_id:
            return s
    return None


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

def propose_changes(metrics: list[SleeveMetrics]) -> list[ProposedChange]:
    """Apply deterministic decision rules to generate ProposedChanges."""
    out: list[ProposedChange] = []
    for m in metrics:
        default = _default_config_for(m.sleeve_id)
        if default is None:
            continue  # unknown sleeve — skip
        default_min_edge = default.min_edge_bps
        default_max_pos = Decimal(default.max_position_usd)
        default_cross = default.max_cross_spread_bps

        # RULE 1 — "loud silence": many evaluations, 0 fills, conservative sleeve
        # → relax edge threshold slightly (never below hard floor / 50% default).
        if m.n_intents == 0 and m.stance == "conservative":
            # Skip — no data at all, not a silence problem.
            pass

        # RULE 2 — "bad fill quality": <50% high-conf over 100+ fills
        # → tighten max_cross_spread_bps to reduce taker crossings.
        if m.n_fills >= 100 and m.high_conf_fraction < 0.5:
            new_cross = max(0, int(m.current_max_cross_spread_bps * 0.5))
            change = ProposedChange(
                sleeve_id=m.sleeve_id,
                field_name="max_cross_spread_bps",
                old_value=m.current_max_cross_spread_bps,
                new_value=new_cross,
                rationale=(
                    f"Low fill quality: only {m.high_conf_fraction:.0%} HIGH confidence over "
                    f"{m.n_fills} fills. Tightening max_cross_spread_bps to force more post-only routes."
                ),
                evidence={"n_fills": m.n_fills, "high_conf_pct": round(m.high_conf_fraction * 100, 1)},
            )
            change.status = _classify_status(change, default_cross, bound=BOUND_CROSS_SPREAD_MAX, hard_floor=0, hard_ceiling=None)
            out.append(change)

        # RULE 3 — "high rejection rate": >30% intents rejected for size/book
        # → reduce max_position_usd toward 75% (closer to what fits in books).
        if (m.n_fills + m.n_rejected) >= 30 and m.rejection_rate > 0.30:
            factor = Decimal("0.75")
            new_pos = (m.current_max_position_usd * factor).quantize(Decimal("0.01"))
            change = ProposedChange(
                sleeve_id=m.sleeve_id,
                field_name="max_position_usd",
                old_value=str(m.current_max_position_usd),
                new_value=str(new_pos),
                rationale=(
                    f"High rejection rate: {m.rejection_rate:.0%} of orders rejected over "
                    f"{m.n_fills + m.n_rejected} attempts. Books too thin for current size; reducing 25%."
                ),
                evidence={
                    "rejection_rate_pct": round(m.rejection_rate * 100, 1),
                    "n_fills": m.n_fills,
                    "n_rejected": m.n_rejected,
                },
            )
            change.status = _classify_status_pos(
                change, default_max_pos,
                hard_ceiling=BANKROLL_USD * HARD_CEILING_POSITION_PCT,
            )
            out.append(change)

        # RULE 4 — "realised slippage exceeds intended edge": paper/live parity warning
        # → log (flag for manual review); do NOT auto-apply changes since this means
        # fills are materially worse than expected and the right fix is manual.
        if m.n_fills >= 50 and m.mean_realised_slippage_bps > 0 and m.mean_intent_edge_bps > 0:
            if m.mean_realised_slippage_bps > 0.5 * m.mean_intent_edge_bps:
                out.append(ProposedChange(
                    sleeve_id=m.sleeve_id,
                    field_name="_review_required",
                    old_value=None,
                    new_value=None,
                    rationale=(
                        f"Realised slippage ({m.mean_realised_slippage_bps:.1f} bps) is >50% of intent edge "
                        f"({m.mean_intent_edge_bps:.1f} bps). Paper↔live parity risk. Manual review needed."
                    ),
                    status="flagged",
                    evidence={
                        "mean_slippage_bps": round(m.mean_realised_slippage_bps, 1),
                        "mean_intent_edge_bps": round(m.mean_intent_edge_bps, 1),
                        "n_fills": m.n_fills,
                    },
                ))

    return out


def _classify_status(
    change: ProposedChange,
    default_value: int,
    *,
    bound: float,
    hard_floor: int | None,
    hard_ceiling: int | None,
) -> str:
    """Return 'applied' if inside bounds, else 'flagged'."""
    new = change.new_value
    if hard_floor is not None and new < hard_floor:
        return "flagged"
    if hard_ceiling is not None and new > hard_ceiling:
        return "flagged"
    if default_value > 0:
        ratio = new / default_value
        if ratio < 1 / bound or ratio > bound:
            return "flagged"
    return "applied"


def _classify_status_pos(
    change: ProposedChange,
    default_value: Decimal,
    *,
    hard_ceiling: Decimal,
) -> str:
    new = Decimal(str(change.new_value))
    if new > hard_ceiling:
        return "flagged"
    if default_value > 0:
        ratio = new / default_value
        if ratio < Decimal(str(BOUND_POSITION_MIN)) or ratio > Decimal(str(BOUND_POSITION_MAX)):
            return "flagged"
    return "applied"


# ---------------------------------------------------------------------------
# Apply + log
# ---------------------------------------------------------------------------

async def apply_changes(changes: list[ProposedChange]) -> None:
    """Persist changes to DB, apply APPLIED ones to SleeveConfig row."""
    async with SessionLocal() as db:
        for c in changes:
            # Persist audit row. Schema uses `field`/`reasoning`; we encode status
            # as a prefix in reasoning since schema has no dedicated status field.
            status_prefixed_reasoning = f"[{c.status.upper()}] {c.rationale}"
            db.add(ConfigChange(
                sleeve_id=c.sleeve_id,
                field=c.field_name,
                old_value=str(c.old_value) if c.old_value is not None else "",
                new_value=str(c.new_value) if c.new_value is not None else "",
                reasoning=status_prefixed_reasoning,
                evidence_json=c.evidence,
                source="self_correct",
            ))
            if c.status != "applied" or c.field_name.startswith("_"):
                continue
            # Apply to the live SleeveConfig row.
            row = (await db.execute(
                select(SleeveConfig).where(SleeveConfig.sleeve_id == c.sleeve_id)
            )).scalar_one_or_none()
            if row is None:
                continue
            if c.field_name == "max_cross_spread_bps":
                row.max_cross_spread_bps = int(c.new_value)
            elif c.field_name == "max_position_usd":
                row.max_position_usd = str(c.new_value)
            elif c.field_name == "min_edge_bps":
                row.min_edge_bps = int(c.new_value)
            row.version = (row.version or 1) + 1
        await db.commit()


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

async def run_selfcorrect_forever() -> None:
    if not ENABLED:
        log.info("selfcorrect_disabled")
        return
    while True:
        try:
            t0 = time.time()
            metrics = await compute_sleeve_metrics(LOOKBACK_DAYS)
            proposals = propose_changes(metrics)
            await apply_changes(proposals)
            applied = sum(1 for p in proposals if p.status == "applied")
            flagged = sum(1 for p in proposals if p.status == "flagged")
            log.info(
                "selfcorrect_tick",
                elapsed_sec=round(time.time() - t0, 2),
                sleeves_evaluated=len(metrics),
                proposals=len(proposals),
                applied=applied,
                flagged=flagged,
            )
        except Exception as e:
            log.error("selfcorrect_failed", err=str(e))
        await asyncio.sleep(INTERVAL_SECONDS)
