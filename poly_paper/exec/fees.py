"""Polymarket fee formula — verified against docs as of April 2026.

Key references:
- https://docs.polymarket.com/developers/market-makers/maker-rebates-program
- https://www.predictionhunt.com/blog/polymarket-fees-complete-guide

Model:
- Taker fee peaks at price p=0.50 and decreases symmetrically toward 0 and 1.
- Formula used here:  fee_rate(p) = peak_rate * 4 * p * (1 - p)
  This is a clean parabola with value=peak at p=0.5 and value=0 at p∈{0,1}.
- Maker rebate = taker_fee × rebate_share (typically 20–25%). The rebate is
  paid out daily, not per-trade, but we apply it at fill time for PnL accuracy
  (reversible if Polymarket changes its mechanism).
- Fees are charged on notional (price * shares) for USDC-denominated reporting.

NOTE: Polymarket's exact internal formula uses a form C * p * feeRate * (p*(1-p))^exp
      and fees on BUY are collected in shares while SELL fees are collected in USDC.
      For paper-trading PnL purposes we unify everything to USDC-equivalent. This
      keeps our numbers comparable across trades. When we integrate the live CLOB
      client in Phase 2, the client itself applies the real on-chain fee, and we
      *compare* our estimate against the actual fee — any drift gets flagged in
      the parity panel.

If Polymarket tweaks these rates, update CATEGORY_TAKER_PEAK_RATE and re-run tests.
"""

from __future__ import annotations

from decimal import Decimal

from .models import MarketCategory, OrderType, Side

# Peak taker rates (at p=0.50) per category. Source: Polymarket docs / Prediction Hunt, April 2026.
# Values are fractional (0.018 = 1.8%).
CATEGORY_TAKER_PEAK_RATE: dict[MarketCategory, Decimal] = {
    MarketCategory.CRYPTO:      Decimal("0.0180"),
    MarketCategory.ECONOMICS:   Decimal("0.0150"),
    MarketCategory.MENTIONS:    Decimal("0.0156"),
    MarketCategory.CULTURE:     Decimal("0.0125"),
    MarketCategory.WEATHER:     Decimal("0.0125"),
    MarketCategory.FINANCE:     Decimal("0.0100"),
    MarketCategory.POLITICS:    Decimal("0.0100"),
    MarketCategory.TECH:        Decimal("0.0100"),
    MarketCategory.SPORTS:      Decimal("0.0075"),
    MarketCategory.GEOPOLITICS: Decimal("0"),      # fee-free
    MarketCategory.OTHER:       Decimal("0.0100"), # conservative default
}

# Fraction of taker fee that is rebated to the maker who provided the liquidity.
# Polymarket docs say 20–25% depending on market / rebate pool dynamics. We use 25%
# as a best-case estimate; if the real rebate averages lower, maker strategies
# will look slightly better in paper than live — we log this caveat.
MAKER_REBATE_SHARE = Decimal("0.25")


def taker_fee_rate(price: Decimal, category: MarketCategory) -> Decimal:
    """Effective taker fee rate as a fraction of notional, at the given price.

    Returns 0 for fee-free categories. Symmetric around p=0.5, zero at p∈{0,1}.
    """
    if price <= 0 or price >= 1:
        return Decimal("0")
    peak = CATEGORY_TAKER_PEAK_RATE.get(category, Decimal("0"))
    # peak * 4p(1-p) — parabola with maximum = peak at p=0.5, zeros at 0 and 1.
    return peak * Decimal("4") * price * (Decimal("1") - price)


def maker_rebate_rate(price: Decimal, category: MarketCategory) -> Decimal:
    """Effective maker rebate rate (fraction of notional). Returned as a NEGATIVE number
    to signal cash received.
    """
    taker = taker_fee_rate(price, category)
    return -taker * MAKER_REBATE_SHARE


def leg_fee_usd(
    *,
    price: Decimal,
    size_shares: Decimal,
    category: MarketCategory,
    role: str,  # "maker" or "taker"
) -> Decimal:
    """Fee on one fill leg, in USDC.

    Positive = fee paid. Negative = rebate received.
    """
    notional = price * size_shares
    if role == "taker":
        return notional * taker_fee_rate(price, category)
    elif role == "maker":
        return notional * maker_rebate_rate(price, category)
    raise ValueError(f"unknown role: {role}")


def fee_for_order_type(order_type: OrderType) -> str:
    """Default role under the given order_type, assuming the order does NOT cross the spread.

    LIMIT orders that cross become takers in live — we have to inspect the book to know.
    This helper only supplies the *default* assumption for order_type. The paper simulator
    inspects the book and may reclassify legs.
    """
    if order_type in (OrderType.MARKET, OrderType.FOK):
        return "taker"
    if order_type == OrderType.POST_ONLY:
        return "maker"
    if order_type == OrderType.LIMIT:
        return "maker"  # default; simulator reclassifies if the order crosses
    raise ValueError(order_type)
