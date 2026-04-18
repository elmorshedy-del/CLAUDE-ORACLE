"""BTC spot price and history feed.

Uses Coinbase's public Exchange API (no auth, no rate-limit concerns for our
polling frequency). Coinbase is used because Polymarket's "up/down" markets
resolve via Chainlink BTC/USD which aggregates from major exchanges, and
Coinbase is the largest single USD-pair source in the Chainlink feed.

Fallback path: if Coinbase fails, try Kraken public API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import httpx

COINBASE_BASE = "https://api.exchange.coinbase.com"
KRAKEN_BASE = "https://api.kraken.com/0/public"


@dataclass
class SpotQuote:
    price: float
    timestamp_unix: float
    source: Literal["coinbase", "kraken"]


class BTCSpotFeed:
    """Simple async client with spot ticker + historical candles."""

    def __init__(self, timeout: float = 8.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout)
        # Tiny in-memory cache for spot — polls at most once per 2 seconds.
        self._cached: SpotQuote | None = None
        self._cache_ttl_sec = 2.0

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "BTCSpotFeed":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Spot
    # ------------------------------------------------------------------

    async def get_spot(self) -> SpotQuote:
        now = time.time()
        if self._cached is not None and (now - self._cached.timestamp_unix) < self._cache_ttl_sec:
            return self._cached
        try:
            q = await self._coinbase_ticker()
            self._cached = q
            return q
        except Exception:
            q = await self._kraken_ticker()
            self._cached = q
            return q

    async def _coinbase_ticker(self) -> SpotQuote:
        r = await self._client.get(f"{COINBASE_BASE}/products/BTC-USD/ticker")
        r.raise_for_status()
        d = r.json()
        return SpotQuote(price=float(d["price"]), timestamp_unix=time.time(), source="coinbase")

    async def _kraken_ticker(self) -> SpotQuote:
        r = await self._client.get(f"{KRAKEN_BASE}/Ticker?pair=XBTUSD")
        r.raise_for_status()
        d = r.json()
        pair_data = next(iter(d["result"].values()))
        # 'c' field = [last_price, last_size]
        return SpotQuote(
            price=float(pair_data["c"][0]),
            timestamp_unix=time.time(),
            source="kraken",
        )

    # ------------------------------------------------------------------
    # History (for realized vol)
    # ------------------------------------------------------------------

    async def get_history_closes(
        self,
        *,
        granularity_seconds: int = 3600,
        n_bars: int = 24 * 30,
    ) -> list[float]:
        """Return N most recent CLOSE prices, oldest-first, for a single bar size.

        granularity must be one of Coinbase's supported values:
        60, 300, 900, 3600, 21600, 86400 (1m, 5m, 15m, 1h, 6h, 1d).
        """
        if granularity_seconds not in {60, 300, 900, 3600, 21600, 86400}:
            raise ValueError(f"unsupported granularity {granularity_seconds}")
        # Coinbase's candles endpoint returns at most 300 bars per call.
        # We respect that limit; caller requests feasible n_bars.
        if n_bars > 300:
            n_bars = 300
        end = int(time.time())
        start = end - granularity_seconds * n_bars
        params = {
            "granularity": granularity_seconds,
            "start": start,
            "end": end,
        }
        r = await self._client.get(f"{COINBASE_BASE}/products/BTC-USD/candles", params=params)
        r.raise_for_status()
        raw = r.json()
        # Coinbase returns [[time, low, high, open, close, volume], ...] descending by time.
        # Sort ascending and take closes.
        raw_sorted = sorted(raw, key=lambda c: c[0])
        closes = [float(c[4]) for c in raw_sorted]
        return closes
