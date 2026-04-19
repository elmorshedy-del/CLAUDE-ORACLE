"""Export helpers for rich, lightweight audit bundles.

The trading engine already persists the important paper-trading artifacts in the
database. This module packages them into a compressed multi-file export so a
download stays fast and organised:

- summary/manifest.json
- summary/README.txt
- summary/export_scope.json
- summary/metrics_snapshot.json
- summary/recent_tape.json
- tables/*.ndjson (one file per logical table)
- derived/*.ndjson (joined / analysis-friendly exports)
- logs/*.jsonl (optional raw runtime logs if present on disk)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
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
from .weather_calibration import WeatherForecastRecord


@dataclass(frozen=True)
class TableExportSpec:
    name: str
    model: type[Any]
    archive_path: str
    time_field: str | None = None


@dataclass(frozen=True)
class ExportFilter:
    start: datetime | None = None
    end: datetime | None = None

    def has_bounds(self) -> bool:
        return self.start is not None or self.end is not None

    def contains(self, value: datetime | None) -> bool:
        if value is None:
            return not self.has_bounds()
        instant = _normalize_datetime(value)
        if self.start is not None and instant < self.start:
            return False
        if self.end is not None and instant >= self.end:
            return False
        return True

    def to_manifest(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "semantics": "start inclusive, end exclusive",
        }

    def filename_suffix(self) -> str:
        parts: list[str] = []
        if self.start is not None:
            parts.append(f"from-{self.start.strftime('%Y%m%d')}")
        if self.end is not None:
            end_day = (self.end - timedelta(seconds=1)).strftime("%Y%m%d")
            parts.append(f"to-{end_day}")
        return "-".join(parts)


EXPORT_TABLES: tuple[TableExportSpec, ...] = (
    TableExportSpec("markets", Market, "tables/markets.ndjson"),
    TableExportSpec("sleeve_configs", SleeveConfig, "tables/sleeve_configs.ndjson"),
    TableExportSpec("order_intents", OrderIntentRow, "tables/order_intents.ndjson", time_field="created_at"),
    TableExportSpec("fills", FillRow, "tables/fills.ndjson", time_field="created_at"),
    TableExportSpec("fair_value_snaps", FairValueSnap, "tables/fair_value_snaps.ndjson", time_field="computed_at"),
    TableExportSpec("book_snaps", BookSnap, "tables/book_snaps.ndjson", time_field="captured_at"),
    TableExportSpec("config_changes", ConfigChange, "tables/config_changes.ndjson", time_field="applied_at"),
    TableExportSpec("sleeve_pnl_hourly", SleevePnlHourly, "tables/sleeve_pnl_hourly.ndjson", time_field="hour_bucket"),
    TableExportSpec(
        "weather_forecast_records",
        WeatherForecastRecord,
        "tables/weather_forecast_records.ndjson",
        time_field="recorded_at",
    ),
)

EXPORT_TTL_SECONDS = int(os.environ.get("POLY_EXPORT_TTL_SECONDS", "3600"))
EXPORT_DIR = Path(os.environ.get("POLY_EXPORT_DIR", tempfile.gettempdir())) / "claude_oracle_exports"
FULL_TAPE_ARCHIVE_PATH = "derived/full_tape.ndjson"


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return _datetime_iso(value)
    return str(value)


def _build_readme(export_filter: ExportFilter) -> str:
    scope = "All persisted history."
    if export_filter.has_bounds():
        scope = (
            "Filtered export window.\n"
            f"  start: {export_filter.start.isoformat() if export_filter.start else 'open'}\n"
            f"  end:   {export_filter.end.isoformat() if export_filter.end else 'open'}\n"
            "  semantics: start inclusive, end exclusive"
        )
    return "\n".join([
        "CLAUDE-ORACLE export layout",
        "",
        "summary/export_scope.json",
        f"  {scope}",
        "",
        "summary/manifest.json",
        "  Lightweight inventory of the bundle with table counts and file sizes.",
        "",
        "summary/metrics_snapshot.json",
        "  Current /metrics payload for quick inspection.",
        "",
        "summary/recent_tape.json",
        "  Recent fills preview from the filtered tape query.",
        "",
        "tables/*.ndjson",
        "  Durable source-of-truth rows exported one table per file.",
        "  NDJSON keeps the archive compact and easy to stream into analysis tools.",
        "",
        "derived/full_tape.ndjson",
        "  Full joined fill+intent export for analysis without manual joins.",
        "",
        "logs/*.jsonl",
        "  Raw runtime event logs if a file sink exists on disk.",
        "  If a time filter is supplied, only events inside that window are included.",
    ])


def _primary_key_columns(model: type[Any]) -> list[Any]:
    mapper = sa_inspect(model)
    return list(mapper.primary_key)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _datetime_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _normalize_datetime(value).isoformat()


def parse_export_datetime(raw: str, *, is_end: bool = False) -> datetime:
    text = raw.strip()
    if len(text) == 10:
        parsed_date = date.fromisoformat(text)
        base = datetime.combine(parsed_date, dt_time.min, tzinfo=timezone.utc)
        return base + timedelta(days=1) if is_end else base
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return _normalize_datetime(parsed)


def _row_to_dict(row: Any) -> dict[str, Any]:
    mapper = sa_inspect(row.__class__)
    data: dict[str, Any] = {}
    for column in mapper.columns:
        value = getattr(row, column.key)
        if isinstance(value, datetime):
            data[column.key] = _datetime_iso(value)
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


def prepare_export_path(export_filter: ExportFilter | None = None) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - EXPORT_TTL_SECONDS
    for old in EXPORT_DIR.glob("claude-oracle-export-*.zip"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except FileNotFoundError:
            continue
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = ""
    if export_filter and export_filter.has_bounds():
        suffix = f"-{export_filter.filename_suffix()}"
    return EXPORT_DIR / f"claude-oracle-export-{stamp}{suffix}.zip"


def _apply_time_filter(stmt: Any, spec: TableExportSpec, export_filter: ExportFilter) -> Any:
    if not export_filter.has_bounds() or spec.time_field is None:
        return stmt
    column = getattr(spec.model, spec.time_field)
    if export_filter.start is not None:
        stmt = stmt.where(column >= export_filter.start)
    if export_filter.end is not None:
        stmt = stmt.where(column < export_filter.end)
    return stmt


def _log_entry_manifest(
    archive_path: str,
    path: Path,
    *,
    export_filter: ExportFilter,
) -> dict[str, Any]:
    entry = {
        "path": archive_path,
        "format": path.suffix.lstrip(".") or "jsonl",
        "source": "runtime_logs",
        "size_bytes": path.stat().st_size,
    }
    if export_filter.has_bounds():
        entry["window_field"] = "timestamp"
        entry["filtered"] = True
    return entry


async def _full_tape_count(export_filter: ExportFilter) -> int:
    async with SessionLocal() as db:
        stmt = select(func.count()).select_from(FillRow)
        if export_filter.start is not None:
            stmt = stmt.where(FillRow.created_at >= export_filter.start)
        if export_filter.end is not None:
            stmt = stmt.where(FillRow.created_at < export_filter.end)
        return (await db.execute(stmt)).scalar() or 0


async def build_manifest(export_filter: ExportFilter | None = None) -> dict[str, Any]:
    export_filter = export_filter or ExportFilter()
    async with SessionLocal() as db:
        files: list[dict[str, Any]] = []
        for spec in EXPORT_TABLES:
            stmt = _apply_time_filter(select(func.count()).select_from(spec.model), spec, export_filter)
            count = (await db.execute(stmt)).scalar() or 0
            entry = {
                "path": spec.archive_path,
                "format": "ndjson",
                "source": spec.name,
                "row_count": count,
            }
            if spec.time_field is not None:
                entry["window_field"] = spec.time_field
            files.append(entry)

    files.append({
        "path": FULL_TAPE_ARCHIVE_PATH,
        "format": "ndjson",
        "source": "full_tape",
        "row_count": await _full_tape_count(export_filter),
        "window_field": "created_at",
    })

    for archive_path, path in existing_log_exports():
        files.append(_log_entry_manifest(archive_path, path, export_filter=export_filter))

    return {
        "format": "claude-oracle-export",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "filters": export_filter.to_manifest(),
        "files": files,
    }


async def _write_table_to_zip(
    zf: zipfile.ZipFile,
    spec: TableExportSpec,
    export_filter: ExportFilter,
) -> None:
    async with SessionLocal() as db:
        stmt = select(spec.model).order_by(*_primary_key_columns(spec.model))
        stmt = _apply_time_filter(stmt, spec, export_filter)
        stream = await db.stream_scalars(stmt)
        with zf.open(spec.archive_path, "w") as handle:
            async for row in stream:
                payload = json.dumps(
                    _row_to_dict(row),
                    default=_json_default,
                    separators=(",", ":"),
                ).encode("utf-8")
                handle.write(payload + b"\n")


def fill_intent_payload(fill: FillRow, intent: OrderIntentRow) -> dict[str, Any]:
    return {
        "fill_id": fill.fill_id,
        "client_order_id": intent.client_order_id,
        "created_at": _datetime_iso(fill.created_at),
        "fill_created_at": _datetime_iso(fill.created_at),
        "intent_created_at": _datetime_iso(intent.created_at),
        "sleeve_id": intent.sleeve_id,
        "market_condition_id": intent.market_condition_id,
        "token_id": intent.token_id,
        "side": intent.side,
        "order_type": intent.order_type,
        "limit_price": intent.limit_price,
        "size_usd": intent.size_usd,
        "size_shares": intent.size_shares,
        "edge_bps": intent.edge_bps,
        "category": intent.category,
        "mode": fill.mode,
        "rejected": fill.rejected,
        "confidence": fill.confidence,
        "avg_price": fill.avg_price,
        "filled_size_shares": fill.filled_size_shares,
        "notional_usd": fill.notional_usd,
        "fees_usd": fill.fees_usd,
        "gas_usd": fill.gas_usd,
        "slippage_bps": fill.slippage_bps,
        "latency_ms": fill.latency_ms,
        "legs_json": fill.legs_json,
        "notes": fill.notes,
        "reasoning": intent.reasoning,
        "resolved": fill.resolved,
        "resolved_winner": fill.resolved_winner,
        "resolved_pnl_usd": fill.resolved_pnl_usd,
        "resolved_at": _datetime_iso(fill.resolved_at),
    }


async def _write_full_tape_to_zip(zf: zipfile.ZipFile, export_filter: ExportFilter) -> None:
    async with SessionLocal() as db:
        stmt = (
            select(FillRow, OrderIntentRow)
            .join(OrderIntentRow, FillRow.client_order_id == OrderIntentRow.client_order_id)
            .order_by(FillRow.created_at.desc(), FillRow.fill_id.desc())
        )
        if export_filter.start is not None:
            stmt = stmt.where(FillRow.created_at >= export_filter.start)
        if export_filter.end is not None:
            stmt = stmt.where(FillRow.created_at < export_filter.end)
        stream = await db.stream(stmt)
        with zf.open(FULL_TAPE_ARCHIVE_PATH, "w") as handle:
            async for row in stream:
                fill, intent = row
                payload = json.dumps(
                    fill_intent_payload(fill, intent),
                    default=_json_default,
                    separators=(",", ":"),
                ).encode("utf-8")
                handle.write(payload + b"\n")


def _parse_log_line_timestamp(line: str) -> datetime | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, str):
        return None
    try:
        return parse_export_datetime(timestamp)
    except ValueError:
        return None


def _write_filtered_log_to_zip(
    zf: zipfile.ZipFile,
    archive_path: str,
    path: Path,
    export_filter: ExportFilter,
) -> None:
    if not export_filter.has_bounds():
        zf.write(path, archive_path)
        return
    with path.open("r", encoding="utf-8") as source, zf.open(archive_path, "w") as handle:
        for line in source:
            if not line.strip():
                continue
            timestamp = _parse_log_line_timestamp(line)
            if timestamp is None or not export_filter.contains(timestamp):
                continue
            handle.write(line.encode("utf-8"))


async def write_export_bundle(
    destination: Path,
    *,
    metrics_payload: dict[str, Any],
    tape_payload: list[dict[str, Any]],
    export_filter: ExportFilter | None = None,
) -> dict[str, Any]:
    export_filter = export_filter or ExportFilter()
    manifest = await build_manifest(export_filter=export_filter)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("summary/manifest.json", json.dumps(manifest, indent=2, default=_json_default))
        zf.writestr("summary/README.txt", _build_readme(export_filter))
        zf.writestr("summary/export_scope.json", json.dumps(export_filter.to_manifest(), indent=2))
        zf.writestr("summary/metrics_snapshot.json", json.dumps(metrics_payload, indent=2, default=_json_default))
        zf.writestr("summary/recent_tape.json", json.dumps(tape_payload, indent=2, default=_json_default))

        for spec in EXPORT_TABLES:
            await _write_table_to_zip(zf, spec, export_filter)

        await _write_full_tape_to_zip(zf, export_filter)

        for archive_path, path in existing_log_exports():
            _write_filtered_log_to_zip(zf, archive_path, path, export_filter)

    return manifest
