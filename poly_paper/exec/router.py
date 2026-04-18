"""Execution router — the single entry point for all order execution.

THIS is the file that enforces "paper and live share the same code path".

Every sleeve, every strategy, every signal calls `execute_order(intent, mode, ...)`.
This router dispatches to the paper simulator or the live CLOB client depending
on mode, but the OrderIntent going in and the Fill coming out are the same types
with the same semantics.

If you want to know whether a sleeve that looks good in paper will work in live:
the *only* difference is the last step inside this file — one branch calls
`simulate_fill()`, the other calls `execute_live()`. Everything up to and
downstream from the router is identical.
"""

from __future__ import annotations

from .live import execute_live
from .models import (
    ExecutionMode,
    Fill,
    MarketCategory,
    OrderBook,
    OrderIntent,
)
from .paper import DEFAULT_CONFIG, PaperSimConfig, simulate_fill


async def execute_order(
    intent: OrderIntent,
    mode: ExecutionMode,
    *,
    book: OrderBook | None = None,
    category: MarketCategory | None = None,
    sim_config: PaperSimConfig = DEFAULT_CONFIG,
) -> Fill:
    """Execute (or simulate) an order intent.

    PAPER mode requires `book` — the caller must supply a current L2 snapshot.
    LIVE mode ignores `book` (real book lives in Polymarket's matcher).
    """
    if mode is ExecutionMode.PAPER:
        if book is None:
            raise ValueError("paper mode requires a book snapshot")
        return simulate_fill(intent, book, category=category, config=sim_config)

    if mode is ExecutionMode.LIVE:
        return await execute_live(intent)

    raise ValueError(f"unknown execution mode: {mode}")
