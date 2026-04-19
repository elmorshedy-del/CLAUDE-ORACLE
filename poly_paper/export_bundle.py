"""Export helpers for rich, lightweight audit bundles.

The trading engine already persists the important paper-trading artifacts in the
database. This module packages them into a compressed multi-file export so a
download stays fast and organised:

- summary/manifest.json
- summary/README.txt
- summary/metrics_snapshot.json
- summary/recent_tape.json
- tables/*.ndjson (one file per logical table)
- logs/*.jsonl (optional raw runtime logs if present on disk)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, inspect as sa_inspect, select

from .db.models import (
    BookSnap,
    ConfigChange,
    FairValueSnap,
    FillRow,
    Market,
    OrderIntentRow,
    SleeveConfig,
    SleevePnlHourly,
)
from .db.session import SessionLocal


@dataclass(frozen=True)
class TableExportSpec:
    name: str
    model: type[Any]
    archive_path: str


EXPORT_TABLES: tuple[TableExportSpec, ...] = (
    TableExportSpec("markets", Market, "tables/markets.ndjson"),
    TableExportSpec("sleeve_configs", SleeveConfig, "tables/sleeve_configs.ndjson"),
    TableExportSpec("order_intents", OrderIntentRow, "tables/order_intents.ndjson"),
    TableExportSpec("fills", FillRow, "tables/fills.ndjson"),
    TableExportSpec("fair_value_snaps", FairValueSnap, "tables/fair_value_snaps.ndjson"),
    TableExportSpec("book_snaps", BookSnap, "tables/book_snaps.ndjson"),
    TableExportSpec("config_changes", ConfigChange, "tables/config_changes.ndjson"),
    TableExportSpec("sleeve_pnl_hourly", SleevePnlHourly, "tables/sleeve_pnl_hourly.ndjson"),
)

EXPORT_TTL_SECONDS = int(os.environ.get("POLY_EXPORT_TTL_SECONDS", "3600"))
EXPORT_DIR = Path(os.environ.get("POLY_EXPORT_DIR", tempfile.gettempdir())) / "claude_oracle_exports"


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _build_readme() -> str:
    return "\n".join([
        "CLAUDE-ORACLE export layout",
        "",
        "summary/manifest.json",
        "  Lightweight inventory of the bundle with table counts and file sizes.",
        "",
        "summary/metrics_snapshot.json",
        "  Current /metrics payload for quick inspection.",
        "",
        "summary/recent_tape.json",
        "  Recent fills preview from the tape endpoint.",
        "",
        "tables/*.ndjson",
        "  Durable source-of-truth rows exported one table per file.",
        "  NDJSON keeps the archive compact and easy to stream into analysis tools.",
        "",
        "logs/*.jsonl",
        "  Raw runtime event logs if a file sink exists on disk.",
    ])


def _primary_key_columns(model: type[Any]) -> list[Any]:
    mapper = sa_inspect(model)
    return list(mapper.primary_key)


def _row_to_dict(row: Any) -> dict[str, Any]:
    mapper = sa_inspect(row.__class__)
    data: dict[str, Any] = {}
    for column in mapper.columns:
        value = getattr(row, column.key)
        if isinstance(value, datetime):
            data[column.key] = value.isoformat()
        else:
            data[column.key] = value
    return data


def _candidate_log_paths() -> list[Path]:
    configured = os.environ.get("POLY_EVENT_LOG_PATH")
    paths: list[Path] = []
    if configured:
        paths.append(Path(configured))
    paths.extend([
        Path("/data/poly_paper_events.jsonl"),
        Path("./poly_paper_events.jsonl"),
        Path("./logs/poly_paper_events.jsonl"),
    ])
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def existing_log_exports() -> list[tuple[str, Path]]:
    exports: list[tuple[str, Path]] = []
    for path in _candidate_log_paths():
        if path.exists() and path.is_file():
            exports.append((f"logs/{path.name}", path))
    return exports


def prepare_export_path() -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - EXPORT_TTL_SECONDS
    for old in EXPORT_DIR.glob("claude-oracle-export-*.zip"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except FileNotFoundError:
            continue
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return EXPORT_DIR / f"claude-oracle-export-{stamp}.zip"


async def build_manifest() -> dict[str, Any]:
    async with SessionLocal() as db:
        files: list[dict[str, Any]] = []
        for spec in EXPORT_TABLES:
            count = (await db.execute(select(func.count()).select_from(spec.model))).scalar() or 0
            files.append({
                "path": spec.archive_path,
                "format": "ndjson",
                "source": spec.name,
                "row_count": count,
            })

    for archive_path, path in existing_log_exports():
        files.append({
            "path": archive_path,
            "format": path.suffix.lstrip(".") or "jsonl",
            "source": "runtime_logs",
            "size_bytes": path.stat().st_size,
        })

    return {
        "format": "claude-oracle-export",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }


async def _write_table_to_zip(zf: zipfile.ZipFile, spec: TableExportSpec) -> None:
    async with SessionLocal() as db:
        stmt = select(spec.model).order_by(*_primary_key_columns(spec.model))
        stream = await db.stream_scalars(stmt)
        with zf.open(spec.archive_path, "w") as handle:
            async for row in stream:
                payload = json.dumps(
                    _row_to_dict(row),
                    default=_json_default,
                    separators=(",", ":"),
                ).encode("utf-8")
                handle.write(payload + b"\n")


async def write_export_bundle(
    destination: Path,
    *,
    metrics_payload: dict[str, Any],
    tape_payload: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = await build_manifest()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("summary/manifest.json", json.dumps(manifest, indent=2, default=_json_default))
        zf.writestr("summary/README.txt", _build_readme())
        zf.writestr("summary/metrics_snapshot.json", json.dumps(metrics_payload, indent=2, default=_json_default))
        zf.writestr("summary/recent_tape.json", json.dumps(tape_payload, indent=2, default=_json_default))

        for spec in EXPORT_TABLES:
            await _write_table_to_zip(zf, spec)

        for archive_path, path in existing_log_exports():
            zf.write(path, archive_path)

    return manifest
