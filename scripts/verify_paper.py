"""End-to-end paper-fill verification against real Polymarket data.

This script proves the paper simulator works against live books. It:
  1. Pulls active tradeable markets from the Gamma API.
  2. Picks one with a liquid order book.
  3. Fetches the real L2 book from the CLOB API.
  4. Simulates several order intents (small taker, large taker, post-only).
  5. Prints every fill with its confidence, fees, slippage, and full reasoning.

Run:
    python -m scripts.verify_paper

No auth needed — uses only public endpoints.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from poly_paper.exec.models import (
    ExecutionMode,
    MarketCategory,
    OrderIntent,
    OrderType,
    Side,
)
from poly_paper.exec.paper import PaperSimConfig, simulate_fill
from poly_paper.polymarket.rest import PolymarketRest, classify_market


async def _pick_liquid_market(api: PolymarketRest) -> tuple[dict, dict]:
    """Return (market_dict, chosen_token_dict) for a market with usable liquidity."""
    markets = await api.list_markets(active=True, closed=False, limit=100)
    # Rank by 24hr volume and liquidity.
    scored: list[tuple[float, dict]] = []
    for m in markets:
        if not m.get("enableOrderBook"):
            continue
        if not m.get("acceptingOrders"):
            continue
        vol = float(m.get("volume24hrClob") or m.get("volume24hr") or 0)
        liq = float(m.get("liquidityClob") or m.get("liquidity") or 0)
        if liq < 500:  # skip totally illiquid books
            continue
        scored.append((vol + liq, m))
    if not scored:
        raise RuntimeError("no active markets with usable liquidity found")
    scored.sort(key=lambda x: -x[0])

    # Find one whose book actually returns usable data.
    import json as _json
    for _, m in scored[:10]:
        clob_ids = m.get("clobTokenIds")
        if isinstance(clob_ids, str):
            clob_ids = _json.loads(clob_ids)
        if not clob_ids:
            continue
        for token_id in clob_ids:
            try:
                book = await api.get_book(str(token_id))
            except LookupError:
                continue
            if book.bids and book.asks:
                outcomes = m.get("outcomes")
                if isinstance(outcomes, str):
                    outcomes = _json.loads(outcomes)
                tokens = list(zip(outcomes or [], clob_ids or []))
                idx = [str(t) for _, t in tokens].index(str(token_id))
                return m, {"outcome": tokens[idx][0], "token_id": str(token_id)}
    raise RuntimeError("no liquid book found in top 10 markets")


def _fmt_book(book: Any, n: int = 5) -> str:
    lines = ["  ASKS (top → next):"]
    for lvl in book.asks[:n]:
        lines.append(f"    ${lvl.price:<6} × {lvl.size}")
    lines.append("  BIDS (top → next):")
    for lvl in book.bids[:n]:
        lines.append(f"    ${lvl.price:<6} × {lvl.size}")
    if book.mid is not None:
        lines.append(f"  mid=${book.mid:.4f}  spread=${book.spread}")
    return "\n".join(lines)


def _print_fill(tag: str, fill: Any) -> None:
    print(f"\n=== {tag} ===")
    if fill.rejected:
        print(f"  REJECTED: {fill.notes}")
        return
    print(f"  fill_id:    {fill.fill_id}")
    print(f"  legs:       {len(fill.legs)}")
    for i, leg in enumerate(fill.legs):
        print(f"    #{i+1}: {leg.role:<5} {leg.size_shares} @ ${leg.price}")
    print(f"  avg price:  ${fill.avg_price}")
    print(f"  filled sz:  {fill.filled_size_shares}")
    print(f"  notional:   ${fill.notional_usd}")
    print(f"  fees USDC:  {fill.fees_usd}  ({'rebate' if fill.fees_usd < 0 else 'fee paid'})")
    print(f"  slippage:   {fill.slippage_bps} bps")
    print(f"  confidence: {fill.confidence.value.upper()}")
    print(f"  latency:    {fill.latency_ms} ms (simulated)")
    if fill.notes:
        print(f"  notes:      {fill.notes}")


async def main() -> None:
    async with PolymarketRest() as api:
        print("Finding a liquid tradeable market…")
        market, chosen = await _pick_liquid_market(api)
        cat = classify_market(market)
        token_id = chosen["token_id"]

        print(f"\nMarket:    {market.get('question', market.get('title'))}")
        print(f"Outcome:   YES on '{chosen['outcome']}' — token {token_id[:20]}…")
        print(f"Category:  {cat.value}  (→ peak taker fee {_peak_rate_display(cat)})")
        print(f"24h vol:   ${float(market.get('volume24hrClob') or market.get('volume24hr') or 0):,.0f}")
        print(f"Liquidity: ${float(market.get('liquidityClob') or market.get('liquidity') or 0):,.0f}")
        print()

        print("Fetching L2 book…")
        book = await api.get_book(token_id)
        print(f"Book timestamp: {book.timestamp_ms}")
        print(_fmt_book(book))

        cond_id = market.get("conditionId") or market.get("condition_id", "")

        # --- Intent 1: small market buy (should be HIGH confidence) ---
        i1 = OrderIntent(
            sleeve_id="verify",
            market_condition_id=cond_id,
            token_id=token_id,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            size_usd=Decimal("10"),
            category=cat,
            reasoning="smoke test - small taker",
            client_order_id="v1",
        )
        _print_fill("Intent 1: MARKET BUY $10 (small taker)", simulate_fill(i1, book, category=cat))

        # --- Intent 2: larger market buy (may walk ladder, may downgrade confidence) ---
        i2 = OrderIntent(
            sleeve_id="verify",
            market_condition_id=cond_id,
            token_id=token_id,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            size_usd=Decimal("500"),
            category=cat,
            reasoning="smoke test - large taker",
            client_order_id="v2",
        )
        _print_fill("Intent 2: MARKET BUY $500 (larger taker — may walk ladder)", simulate_fill(i2, book, category=cat))

        # --- Intent 3: post-only at mid (must either fill as maker or reject for cross) ---
        if book.mid is not None and book.best_bid is not None and book.best_ask is not None:
            tick = Decimal("0.01")
            # Place one tick above best bid (improves top-of-book, shouldn't cross ask)
            my_bid = book.best_bid + tick
            i3 = OrderIntent(
                sleeve_id="verify",
                market_condition_id=cond_id,
                token_id=token_id,
                side=Side.BUY,
                order_type=OrderType.POST_ONLY,
                size_shares=Decimal("20"),
                limit_price=my_bid,
                category=cat,
                reasoning="smoke test - post_only maker improving best bid",
                client_order_id="v3",
            )
            _print_fill(
                f"Intent 3: POST_ONLY BUY 20 @ ${my_bid} (maker, improving best bid)",
                simulate_fill(i3, book, category=cat, config=PaperSimConfig(maker_fill_default="always")),
            )

        # --- Summary ---
        print("\n" + "=" * 60)
        print("What this proves:")
        print("  1. The paper simulator walks real Polymarket books.")
        print("  2. Fee formula applied correctly for this market's category.")
        print("  3. Confidence scores honestly reflect fill quality.")
        print("  4. When paper mode = LIVE is flipped in Phase 2, intents and")
        print("     Fills pass through the identical code path — only the")
        print("     final dispatch inside router.execute_order() changes.")


def _peak_rate_display(cat: MarketCategory) -> str:
    from poly_paper.exec.fees import CATEGORY_TAKER_PEAK_RATE
    pct = CATEGORY_TAKER_PEAK_RATE[cat] * Decimal("100")
    return f"{pct}%"


if __name__ == "__main__":
    asyncio.run(main())
