"""Main entry point — runs the paper trading engine forever.

Usage:
    python -m scripts.run_paper

Env vars:
    DATABASE_URL               e.g. sqlite+aiosqlite:///./poly_paper.db (default)
                               or postgresql://user:pass@host:5432/db (Railway)
    POLY_BANKROLL_USD          starting paper bankroll (default: 1000)
    POLY_MODE                  "paper" or "live" (default: paper; live not yet
                               implemented — will refuse)
    POLY_TICK_SECONDS          main loop tick (default: 10)
    POLY_FAMILIES              comma-separated strategy families to enable
                               (default: btc_up_down_5m,btc_up_down_15m)
    POLY_ARB_SCAN_SECONDS      arb scanner interval (default: 30)
    POLY_ARB_MIN_LIQ           min per-market liquidity to scan (default: 1000)
"""

from __future__ import annotations

import asyncio
import os

import structlog

from poly_paper.arb_scanner import run_arb_scanner_forever
from poly_paper.http_server import serve_forever as serve_http_forever
from poly_paper.runner import run_forever as run_main_loop_forever


async def main() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer() if os.environ.get("JSON_LOGS") else structlog.dev.ConsoleRenderer(),
        ],
    )
    # Run the HTTP server, the directional-strategy loop, and the arb scanner concurrently.
    # If any raises, log it but keep the others running.
    await asyncio.gather(
        serve_http_forever(),
        run_main_loop_forever(),
        run_arb_scanner_forever(),
        return_exceptions=False,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
