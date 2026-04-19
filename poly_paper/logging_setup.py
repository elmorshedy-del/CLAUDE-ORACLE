"""Shared structlog setup with optional JSONL persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

import structlog


_WRITE_LOCK = Lock()


def _event_log_path() -> Path:
    configured = os.environ.get("POLY_EVENT_LOG_PATH")
    if configured:
        path = Path(configured)
    elif Path("/data").exists():
        path = Path("/data/poly_paper_events.jsonl")
    else:
        path = Path("./logs/poly_paper_events.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _persist_event(_logger: Any, _method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Mirror every structured log event to a JSONL file for export."""
    try:
        payload = json.dumps(event_dict, default=str, separators=(",", ":"))
        with _WRITE_LOCK:
            with _event_log_path().open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
    except Exception:
        # Logging should never break the trading loops.
        pass
    return event_dict


def configure_structlog() -> None:
    """Configure structlog once for both stdout and exportable JSONL logs."""
    renderer = (
        structlog.processors.JSONRenderer()
        if os.environ.get("JSON_LOGS")
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _persist_event,
            renderer,
        ],
    )
