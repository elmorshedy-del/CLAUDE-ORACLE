"""Live execution — stub for Phase 1.

In Phase 2 this will use py-clob-client to sign and submit orders to the
Polymarket CLOB. For now it raises loudly so that an accidental mode=LIVE
call never silently runs through paper code.

When Phase 2 lands, this module will:
  1. Sign orders with the user's Polygon private key (EIP-712).
  2. Submit via py-clob-client's `post_order()` (with post_only flag honored).
  3. Poll order status / subscribe to the user WebSocket channel for fills.
  4. Build an identical Fill object from the real execution data.
  5. Emit parity metrics (paper-vs-live fee/slippage deltas) for the dashboard.

The contract: `execute_live(intent)` returns a Fill with mode=LIVE.
Everything downstream (logging, PnL, self-correction) is mode-agnostic.
"""

from __future__ import annotations

from .models import Fill, OrderIntent


async def execute_live(intent: OrderIntent) -> Fill:  # pragma: no cover
    raise NotImplementedError(
        "Live execution arrives in Phase 2. "
        "Until then, mode=LIVE is refused at the router level."
    )
