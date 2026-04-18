"""Polymarket REST client (public endpoints only — no auth needed for Phase 1).

We use httpx async client. Two base URLs:
- Gamma API:  https://gamma-api.polymarket.com  — market discovery, categories, tags
- CLOB API:   https://clob.polymarket.com       — L2 order books, public market data

Phase 2 will add authenticated CLOB endpoints (order placement, user channel WS).
Phase 1 only needs public data: enumerate markets, fetch books, classify.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import httpx

from ..exec.models import BookLevel, MarketCategory, OrderBook

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


class PolymarketRest:
    """Thin async wrapper. Reuses a single httpx client across calls."""

    def __init__(self, timeout: float = 10.0) -> None:
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "PolymarketRest":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Gamma: market discovery
    # ------------------------------------------------------------------

    async def list_markets(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
        tag: str | None = None,
    ) -> list[dict]:
        """List markets from the Gamma API, filtered for tradability by default."""
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
        }
        if tag is not None:
            params["tag"] = tag
        r = await self._client.get(f"{GAMMA_BASE}/markets", params=params)
        r.raise_for_status()
        data = r.json()
        # Gamma returns a bare list.
        return data if isinstance(data, list) else data.get("data", [])

    async def get_market(self, condition_id: str) -> dict:
        r = await self._client.get(f"{GAMMA_BASE}/markets", params={"condition_ids": condition_id})
        r.raise_for_status()
        body = r.json()
        items = body if isinstance(body, list) else body.get("data", [])
        if not items:
            raise LookupError(f"no market with condition_id {condition_id}")
        return items[0]

    # ------------------------------------------------------------------
    # CLOB: order book
    # ------------------------------------------------------------------

    async def get_book(self, token_id: str) -> OrderBook:
        """Fetch L2 book for an outcome token. Returns normalised OrderBook.

        Polymarket returns bids ASCENDING by price (worst → best). We flip to
        DESCENDING (best bid first) so best_bid == bids[0]. Asks remain ASC
        (best ask == asks[0]) — note that Polymarket usually already serves them
        asc but we re-sort defensively.
        """
        r = await self._client.get(f"{CLOB_BASE}/book", params={"token_id": token_id})
        r.raise_for_status()
        raw = r.json()
        if "error" in raw:
            raise LookupError(raw["error"])

        bids_raw = raw.get("bids", [])
        asks_raw = raw.get("asks", [])

        bids = sorted(
            (BookLevel(price=Decimal(b["price"]), size=Decimal(b["size"])) for b in bids_raw),
            key=lambda lv: lv.price,
            reverse=True,
        )
        asks = sorted(
            (BookLevel(price=Decimal(a["price"]), size=Decimal(a["size"])) for a in asks_raw),
            key=lambda lv: lv.price,
        )

        return OrderBook(
            token_id=token_id,
            market_condition_id=raw.get("market", ""),
            timestamp_ms=int(raw.get("timestamp", "0")),
            bids=bids,
            asks=asks,
            hash=raw.get("hash"),
        )


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

# Tag-to-category mapping. Polymarket uses free-form tags; we map them to our
# fee-enum so the paper simulator charges the right rate. When a market has
# multiple tags we pick the highest-fee one (conservative).
#
# Update this as Polymarket adds categories. Unknown → OTHER (1% default).
_TAG_TO_CATEGORY: dict[str, MarketCategory] = {
    "crypto": MarketCategory.CRYPTO,
    "bitcoin": MarketCategory.CRYPTO,
    "ethereum": MarketCategory.CRYPTO,
    "sports": MarketCategory.SPORTS,
    "nba": MarketCategory.SPORTS,
    "nfl": MarketCategory.SPORTS,
    "mlb": MarketCategory.SPORTS,
    "nhl": MarketCategory.SPORTS,
    "tennis": MarketCategory.SPORTS,
    "soccer": MarketCategory.SPORTS,
    "politics": MarketCategory.POLITICS,
    "elections": MarketCategory.POLITICS,
    "us-politics": MarketCategory.POLITICS,
    "geopolitics": MarketCategory.GEOPOLITICS,
    "finance": MarketCategory.FINANCE,
    "economics": MarketCategory.ECONOMICS,
    "tech": MarketCategory.TECH,
    "culture": MarketCategory.CULTURE,
    "weather": MarketCategory.WEATHER,
    "mentions": MarketCategory.MENTIONS,
}

# Fee-ordering for "pick highest fee" tiebreaker (higher index → higher fee).
_CATEGORY_FEE_ORDER: list[MarketCategory] = [
    MarketCategory.GEOPOLITICS,  # 0%
    MarketCategory.SPORTS,       # 0.75%
    MarketCategory.POLITICS,
    MarketCategory.TECH,
    MarketCategory.FINANCE,      # 1.00%
    MarketCategory.CULTURE,
    MarketCategory.WEATHER,      # 1.25%
    MarketCategory.ECONOMICS,    # 1.50%
    MarketCategory.MENTIONS,     # 1.56%
    MarketCategory.CRYPTO,       # 1.80%
    MarketCategory.OTHER,
]


def classify_market(market_dict: dict) -> MarketCategory:
    """Return the category to use for fee calculation.

    Reads tags / events.tags / title heuristics. When ambiguous, returns the
    highest-fee matching category (conservative for PnL estimation).
    """
    tags: list[str] = []

    # Flat tags on the market itself.
    if isinstance(market_dict.get("tags"), list):
        tags.extend(str(t).lower() for t in market_dict["tags"])

    # Event-level tags (Gamma embeds event info).
    for ev in market_dict.get("events", []) or []:
        if isinstance(ev, dict):
            for t in ev.get("tags", []) or []:
                if isinstance(t, dict) and "slug" in t:
                    tags.append(str(t["slug"]).lower())
                elif isinstance(t, str):
                    tags.append(t.lower())

    matched = {_TAG_TO_CATEGORY[t] for t in tags if t in _TAG_TO_CATEGORY}
    if not matched:
        # Weak title-based fallback.
        title = (market_dict.get("question") or market_dict.get("title") or "").lower()
        if any(w in title for w in ("bitcoin", "btc", "ethereum", "eth", "crypto")):
            matched.add(MarketCategory.CRYPTO)
        elif any(w in title for w in ("election", "president", "congress", "senate")):
            matched.add(MarketCategory.POLITICS)
        elif any(w in title for w in ("game", "match", "vs.", "vs ")):
            matched.add(MarketCategory.SPORTS)

    if not matched:
        return MarketCategory.OTHER

    # Pick highest-fee category from matches.
    for cat in reversed(_CATEGORY_FEE_ORDER):
        if cat in matched:
            return cat
    return MarketCategory.OTHER
