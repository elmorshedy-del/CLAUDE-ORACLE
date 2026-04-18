"""Database schema. SQLite-compatible for local dev, Postgres-compatible for Railway.

All tables use `String` primary keys so that `client_order_id` and `fill_id` from
the exec layer can be used directly as keys without re-hashing. All monetary
values stored as strings (to preserve Decimal precision) — queried back as
Decimals in Python.

Schema overview:
  - markets           : tradeable universe members with metadata and category
  - sleeve_configs    : versioned per-sleeve tunables (self-correction changes these)
  - order_intents     : every intent a sleeve generated (whether filled or not)
  - fills             : every fill (paper or live) linked to an intent
  - fair_value_snaps  : fair-value computations logged for calibration analysis
  - book_snaps        : occasional L2 snapshots — useful for replay
  - config_changes    : audit log of every sleeve config change with reasoning
  - sleeve_pnl_hourly : rolled-up sleeve PnL for dashboards
"""

from __future__ import annotations

import datetime as _dt

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class Market(Base):
    __tablename__ = "markets"

    condition_id: Mapped[str] = mapped_column(String, primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    slug: Mapped[str] = mapped_column(String, index=True)
    category: Mapped[str] = mapped_column(String, index=True)  # e.g. "crypto", "sports"
    strategy_family: Mapped[str | None] = mapped_column(String, index=True, nullable=True)
    # e.g. "btc_up_down_5m", "btc_up_down_15m", "btc_range_daily"
    end_date_iso: Mapped[str | None] = mapped_column(String, nullable=True)
    # The two outcome token IDs, stored as JSON for flexibility.
    tokens_json: Mapped[list] = mapped_column(JSON)  # [{"outcome": "Up", "token_id": "..."}]
    # Parsed strategy-specific params (e.g. {"barrier": 78000, "direction": "above"}).
    params_json: Mapped[dict] = mapped_column(JSON, default=dict)
    in_universe: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    first_seen_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_seen_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # Most-recent snapshot metrics — updated by the runner periodically.
    last_volume_24h_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_liquidity_usd: Mapped[float | None] = mapped_column(Float, nullable=True)


class SleeveConfig(Base):
    __tablename__ = "sleeve_configs"

    sleeve_id: Mapped[str] = mapped_column(String, primary_key=True)
    stance: Mapped[str] = mapped_column(String)  # conservative / balanced / aggressive
    strategy_name: Mapped[str] = mapped_column(String, index=True)
    market_selector: Mapped[str] = mapped_column(Text)  # e.g. "strategy_family=btc_up_down_5m"
    bankroll_usd: Mapped[str] = mapped_column(String)  # Decimal as string
    max_position_usd: Mapped[str] = mapped_column(String)
    min_edge_bps: Mapped[int] = mapped_column(Integer)
    max_cross_spread_bps: Mapped[int] = mapped_column(Integer)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    notes: Mapped[str] = mapped_column(Text, default="")
    extra_json: Mapped[dict] = mapped_column(JSON, default=dict)  # strategy-specific
    updated_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


class OrderIntentRow(Base):
    __tablename__ = "order_intents"

    client_order_id: Mapped[str] = mapped_column(String, primary_key=True)
    sleeve_id: Mapped[str] = mapped_column(String, ForeignKey("sleeve_configs.sleeve_id"), index=True)
    market_condition_id: Mapped[str] = mapped_column(String, ForeignKey("markets.condition_id"), index=True)
    token_id: Mapped[str] = mapped_column(String, index=True)
    side: Mapped[str] = mapped_column(String)
    order_type: Mapped[str] = mapped_column(String)
    limit_price: Mapped[str | None] = mapped_column(String, nullable=True)
    size_usd: Mapped[str | None] = mapped_column(String, nullable=True)
    size_shares: Mapped[str | None] = mapped_column(String, nullable=True)
    edge_bps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    category: Mapped[str] = mapped_column(String)
    reasoning: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    fills = relationship("FillRow", back_populates="intent")


class FillRow(Base):
    __tablename__ = "fills"

    fill_id: Mapped[str] = mapped_column(String, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String, ForeignKey("order_intents.client_order_id"), index=True)
    mode: Mapped[str] = mapped_column(String, index=True)  # paper / live
    rejected: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    filled_size_shares: Mapped[str] = mapped_column(String, default="0")
    avg_price: Mapped[str | None] = mapped_column(String, nullable=True)
    notional_usd: Mapped[str] = mapped_column(String, default="0")
    fees_usd: Mapped[str] = mapped_column(String, default="0")
    gas_usd: Mapped[str] = mapped_column(String, default="0")
    confidence: Mapped[str] = mapped_column(String, default="n/a")
    slippage_bps: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    legs_json: Mapped[list] = mapped_column(JSON, default=list)
    notes: Mapped[str] = mapped_column(Text, default="")
    # Resolved payout — filled in later by a resolution watcher.
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    resolved_winner: Mapped[str | None] = mapped_column(String, nullable=True)
    resolved_pnl_usd: Mapped[str | None] = mapped_column(String, nullable=True)
    resolved_at: Mapped[_dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    intent = relationship("OrderIntentRow", back_populates="fills")


class FairValueSnap(Base):
    __tablename__ = "fair_value_snaps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_condition_id: Mapped[str] = mapped_column(String, ForeignKey("markets.condition_id"), index=True)
    token_id: Mapped[str] = mapped_column(String, index=True)
    probability: Mapped[float] = mapped_column(Float)
    ci_low: Mapped[float] = mapped_column(Float)
    ci_high: Mapped[float] = mapped_column(Float)
    spot: Mapped[float] = mapped_column(Float)
    sigma_annual: Mapped[float] = mapped_column(Float)
    horizon_seconds: Mapped[float] = mapped_column(Float)
    model: Mapped[str] = mapped_column(String)  # up_down / range / above
    computed_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class BookSnap(Base):
    __tablename__ = "book_snaps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    token_id: Mapped[str] = mapped_column(String, index=True)
    best_bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_bid_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_ask_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    depth_1pct_buy: Mapped[float] = mapped_column(Float, default=0)
    depth_1pct_sell: Mapped[float] = mapped_column(Float, default=0)
    timestamp_ms: Mapped[int] = mapped_column(Integer)
    captured_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class ConfigChange(Base):
    __tablename__ = "config_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sleeve_id: Mapped[str] = mapped_column(String, ForeignKey("sleeve_configs.sleeve_id"), index=True)
    field: Mapped[str] = mapped_column(String)
    old_value: Mapped[str] = mapped_column(Text)
    new_value: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String)  # "manual" / "self_correct" / "migration"
    reasoning: Mapped[str] = mapped_column(Text, default="")
    evidence_json: Mapped[dict] = mapped_column(JSON, default=dict)  # stats that triggered change
    approved_by: Mapped[str | None] = mapped_column(String, nullable=True)
    applied_at: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class SleevePnlHourly(Base):
    __tablename__ = "sleeve_pnl_hourly"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sleeve_id: Mapped[str] = mapped_column(String, ForeignKey("sleeve_configs.sleeve_id"), index=True)
    hour_bucket: Mapped[_dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    n_intents: Mapped[int] = mapped_column(Integer, default=0)
    n_fills: Mapped[int] = mapped_column(Integer, default=0)
    n_rejected: Mapped[int] = mapped_column(Integer, default=0)
    gross_pnl_usd: Mapped[str] = mapped_column(String, default="0")
    fees_usd: Mapped[str] = mapped_column(String, default="0")
    high_conf_fill_pct: Mapped[float] = mapped_column(Float, default=0)
    avg_realised_edge_bps: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        Index("ix_sleeve_pnl_sleeve_hour", "sleeve_id", "hour_bucket", unique=True),
    )
