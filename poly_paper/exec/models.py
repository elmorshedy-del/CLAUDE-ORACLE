"""Shared data models for the execution layer.

These models are used identically in paper and live mode. Every order intent,
fill, book snapshot, and config passes through these types — so paper and live
cannot silently diverge on what an order *is*.

All monetary values are in USDC. All prices are in [0, 1] (Polymarket binary).
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExecutionMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    # Crosses the spread, takes liquidity, pays taker fees.
    MARKET = "market"
    # Rests on book; if it would cross, it still crosses (may take).
    LIMIT = "limit"
    # Rests on book; rejected if it would cross. Guaranteed maker.
    POST_ONLY = "post_only"
    # Fill-or-kill: execute in full immediately as taker, or reject entirely.
    FOK = "fok"


class MarketCategory(str, Enum):
    """Polymarket fee categories (April 2026). Peak taker rate differs per category.

    Reference: https://docs.polymarket.com/ (see fees.py for rates)
    """

    CRYPTO = "crypto"
    SPORTS = "sports"
    POLITICS = "politics"
    FINANCE = "finance"
    ECONOMICS = "economics"
    TECH = "tech"
    CULTURE = "culture"
    WEATHER = "weather"
    MENTIONS = "mentions"
    GEOPOLITICS = "geopolitics"  # fee-free
    OTHER = "other"


class FillConfidence(str, Enum):
    """How trustworthy is this paper fill relative to what live would do?

    HIGH    -> taker fill deep into visible liquidity; live will match within slippage bounds.
    MEDIUM  -> small maker fill in a liquid book; likely to fill live but queue position uncertain.
    LOW     -> large maker fill, illiquid book, or walked ladder past a stale snapshot; live will deviate.
    N_A     -> live mode (real fill, no simulation needed).
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    N_A = "n/a"


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------

class BookLevel(BaseModel):
    price: Decimal
    size: Decimal

    model_config = ConfigDict(frozen=True)

    @field_validator("price")
    @classmethod
    def _p(cls, v: Decimal) -> Decimal:
        if not (Decimal("0") <= v <= Decimal("1")):
            raise ValueError(f"price {v} out of [0,1]")
        return v


class OrderBook(BaseModel):
    """Level-2 order book snapshot for one outcome token.

    Polymarket returns bids sorted ascending by price. We normalise so that
    `bids` is descending (best bid first) and `asks` is ascending (best ask first).
    """

    token_id: str
    market_condition_id: str
    timestamp_ms: int
    bids: list[BookLevel] = Field(default_factory=list)
    asks: list[BookLevel] = Field(default_factory=list)
    hash: str | None = None

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Decimal | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    def depth_within(self, side: Side, pct_from_top: Decimal) -> Decimal:
        """Sum of size within `pct_from_top` of the best price on the given side."""
        if side is Side.BUY:
            # When buying we consume asks.
            if not self.asks:
                return Decimal("0")
            best = self.asks[0].price
            limit = best * (Decimal("1") + pct_from_top)
            return sum((lvl.size for lvl in self.asks if lvl.price <= limit), Decimal("0"))
        else:
            if not self.bids:
                return Decimal("0")
            best = self.bids[0].price
            limit = best * (Decimal("1") - pct_from_top)
            return sum((lvl.size for lvl in self.bids if lvl.price >= limit), Decimal("0"))


# ---------------------------------------------------------------------------
# Order / Fill
# ---------------------------------------------------------------------------

class OrderIntent(BaseModel):
    """What a strategy wants to do. Created by sleeves, passed to the router.

    `size_usd` OR `size_shares` must be set, not both. Router converts to shares
    against the current book when dispatching.
    """

    sleeve_id: str
    market_condition_id: str
    token_id: str
    side: Side
    order_type: OrderType
    limit_price: Decimal | None = None  # required for LIMIT and POST_ONLY
    size_usd: Decimal | None = None
    size_shares: Decimal | None = None
    category: MarketCategory = MarketCategory.OTHER
    # Strategy's self-reported edge in cents at the moment of intent.
    # Logged for later attribution / self-correction.
    edge_bps: int | None = None
    reasoning: str = ""
    client_order_id: str  # idempotency + audit trail

    @field_validator("size_usd", "size_shares")
    @classmethod
    def _pos(cls, v: Decimal | None) -> Decimal | None:
        if v is not None and v <= 0:
            raise ValueError("size must be positive")
        return v

    def model_post_init(self, _: object) -> None:  # noqa: D401
        if self.size_usd is None and self.size_shares is None:
            raise ValueError("must specify size_usd or size_shares")
        if self.size_usd is not None and self.size_shares is not None:
            raise ValueError("specify only one of size_usd / size_shares")
        if self.order_type in (OrderType.LIMIT, OrderType.POST_ONLY) and self.limit_price is None:
            raise ValueError(f"{self.order_type} requires limit_price")


class FillLeg(BaseModel):
    """One price level's worth of a fill (walks-the-ladder = multiple legs)."""

    price: Decimal
    size_shares: Decimal
    # Whether this leg was a maker (posted to book) or taker (crossed spread).
    role: Literal["maker", "taker"]


class Fill(BaseModel):
    """The authoritative record of what happened for an intent.

    Paper and live both produce Fill objects with identical schemas.
    """

    intent: OrderIntent
    mode: ExecutionMode
    # Empty list = rejected / unfilled.
    legs: list[FillLeg] = Field(default_factory=list)
    fees_usd: Decimal = Decimal("0")  # positive = paid, negative = rebate received
    gas_usd: Decimal = Decimal("0")  # tiny on Polygon; often zero if gasless
    confidence: FillConfidence = FillConfidence.N_A
    slippage_bps: int = 0  # vs. intent price or intent-time mid
    latency_ms: int = 0
    # Free-form diagnostics.
    notes: str = ""
    # Populated by router for log correlation.
    fill_id: str

    @property
    def filled_size_shares(self) -> Decimal:
        return sum((leg.size_shares for leg in self.legs), Decimal("0"))

    @property
    def notional_usd(self) -> Decimal:
        """Sum of leg price × leg size. Not including fees."""
        return sum((leg.price * leg.size_shares for leg in self.legs), Decimal("0"))

    @property
    def avg_price(self) -> Decimal | None:
        if not self.legs or self.filled_size_shares == 0:
            return None
        return self.notional_usd / self.filled_size_shares

    @property
    def fully_filled(self) -> bool:
        target = self.intent.size_shares
        if target is None:
            return False  # sized in USD; router compares notional separately
        return abs(self.filled_size_shares - target) < Decimal("0.0001")

    @property
    def rejected(self) -> bool:
        return len(self.legs) == 0


# ---------------------------------------------------------------------------
# Sleeve config (the unit of self-correction)
# ---------------------------------------------------------------------------

class SleeveStance(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class SleeveConfig(BaseModel):
    """Per-sleeve tunable parameters. Versioned and logged on every change.

    This is what the self-correction proposer reads and modifies.
    """

    sleeve_id: str
    stance: SleeveStance
    strategy_name: str  # e.g. "sports_pinnacle_fade", "btc_range_vol", "bundle_arb"
    market_selector: str  # JSON-ish descriptor of market universe
    bankroll_usd: Decimal
    max_position_usd: Decimal
    min_edge_bps: int  # net (after-fee) edge below this → don't trade
    # Minimum DIRECTIONAL (gross, before-fee) edge required. Prevents pure
    # spread-capture trades that look profitable on paper but have no thesis.
    # Self-correction can tune this within bounds.
    min_gross_edge_bps: int = 0
    max_cross_spread_bps: int = 0  # 0 = post-only; larger = taker tolerance
    enabled: bool = True
    version: int = 1
    notes: str = ""
