"""Tests for the Polymarket fee formula.

Any change to fees.py must keep these passing — they encode Polymarket's published
fee behaviour as of April 2026. If Polymarket updates its docs, update the rates
in fees.py and re-run these tests.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from poly_paper.exec.fees import (
    CATEGORY_TAKER_PEAK_RATE,
    MAKER_REBATE_SHARE,
    leg_fee_usd,
    maker_rebate_rate,
    taker_fee_rate,
)
from poly_paper.exec.models import MarketCategory


class TestTakerFeeCurve:
    """Polymarket's taker fee peaks at price=0.50, is 0 at 0 and 1, and is symmetric."""

    @pytest.mark.parametrize("category", list(MarketCategory))
    def test_peak_at_midpoint(self, category: MarketCategory) -> None:
        peak_expected = CATEGORY_TAKER_PEAK_RATE[category]
        got = taker_fee_rate(Decimal("0.50"), category)
        assert got == peak_expected, f"{category}: expected {peak_expected} at p=0.5, got {got}"

    @pytest.mark.parametrize("p", [Decimal("0"), Decimal("1")])
    def test_zero_at_extremes(self, p: Decimal) -> None:
        for cat in MarketCategory:
            assert taker_fee_rate(p, cat) == Decimal("0")

    def test_symmetric_around_half(self) -> None:
        for p in (Decimal("0.1"), Decimal("0.25"), Decimal("0.33"), Decimal("0.4")):
            left = taker_fee_rate(p, MarketCategory.CRYPTO)
            right = taker_fee_rate(Decimal("1") - p, MarketCategory.CRYPTO)
            assert left == right, f"asymmetric at p={p}: {left} vs {right}"

    def test_decreases_toward_extremes(self) -> None:
        rates = [taker_fee_rate(Decimal(p), MarketCategory.CRYPTO) for p in ("0.1", "0.25", "0.5", "0.75", "0.9")]
        assert rates[0] < rates[1] < rates[2]
        assert rates[2] > rates[3] > rates[4]

    def test_geopolitics_always_free(self) -> None:
        for p in (Decimal("0.05"), Decimal("0.5"), Decimal("0.95")):
            assert taker_fee_rate(p, MarketCategory.GEOPOLITICS) == Decimal("0")


class TestMakerRebate:
    def test_rebate_is_negative_cash_flow(self) -> None:
        # Rebate returned as a negative rate so callers can add it to fees and
        # compute net PnL without special-casing.
        rate = maker_rebate_rate(Decimal("0.5"), MarketCategory.CRYPTO)
        assert rate < 0

    def test_rebate_is_share_of_taker_fee(self) -> None:
        for p in (Decimal("0.2"), Decimal("0.5"), Decimal("0.8")):
            taker = taker_fee_rate(p, MarketCategory.CRYPTO)
            rebate = maker_rebate_rate(p, MarketCategory.CRYPTO)
            assert rebate == -taker * MAKER_REBATE_SHARE

    def test_geopolitics_rebate_zero(self) -> None:
        assert maker_rebate_rate(Decimal("0.5"), MarketCategory.GEOPOLITICS) == Decimal("0")


class TestLegFee:
    def test_taker_leg_fee_matches_rate(self) -> None:
        price, size = Decimal("0.50"), Decimal("100")
        got = leg_fee_usd(price=price, size_shares=size, category=MarketCategory.CRYPTO, role="taker")
        # notional = 50, peak crypto = 1.80% → fee = 0.90
        assert got == Decimal("0.9000")

    def test_maker_leg_fee_is_rebate(self) -> None:
        price, size = Decimal("0.50"), Decimal("100")
        got = leg_fee_usd(price=price, size_shares=size, category=MarketCategory.CRYPTO, role="maker")
        # taker fee = 0.90, rebate = 25% = 0.225 → returned as -0.225
        assert got == Decimal("-0.225000")

    def test_unknown_role_raises(self) -> None:
        with pytest.raises(ValueError):
            leg_fee_usd(
                price=Decimal("0.5"),
                size_shares=Decimal("1"),
                category=MarketCategory.CRYPTO,
                role="bogus",
            )


class TestRatesMatchPublishedDocs:
    """Sanity-check the published peak rates. If these drift, Polymarket changed their fees."""

    expected = {
        MarketCategory.CRYPTO: Decimal("0.0180"),
        MarketCategory.SPORTS: Decimal("0.0075"),
        MarketCategory.POLITICS: Decimal("0.0100"),
        MarketCategory.FINANCE: Decimal("0.0100"),
        MarketCategory.GEOPOLITICS: Decimal("0"),
    }

    @pytest.mark.parametrize("cat,rate", list(expected.items()))
    def test_peak_rate(self, cat: MarketCategory, rate: Decimal) -> None:
        assert CATEGORY_TAKER_PEAK_RATE[cat] == rate
