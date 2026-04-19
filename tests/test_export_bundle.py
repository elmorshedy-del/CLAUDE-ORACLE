from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import poly_paper.export_bundle as export_bundle
import poly_paper.http_server as http_server
from poly_paper.db.models import Base, FillRow, Market, OrderIntentRow, SleeveConfig
from poly_paper.weather_calibration import WeatherForecastRecord


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


@pytest.fixture
async def isolated_session(monkeypatch: pytest.MonkeyPatch, tmp_path):
    db_path = tmp_path / "export-test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    session_local = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    monkeypatch.setattr(export_bundle, "SessionLocal", session_local)
    monkeypatch.setattr(http_server, "SessionLocal", session_local)
    monkeypatch.setattr(export_bundle, "existing_log_exports", lambda: [])

    try:
        yield session_local
    finally:
        await engine.dispose()


async def _seed_rows(session_local: async_sessionmaker[AsyncSession]) -> None:
    async with session_local() as db:
        db.add(Market(
            condition_id="market-1",
            question="Will rainfall exceed 5mm?",
            slug="rainfall-5mm",
            category="weather",
            strategy_family="weather",
            tokens_json=[{"outcome": "Yes", "token_id": "tok-1"}],
            params_json={"bucket": "5mm+"},
            in_universe=True,
            first_seen_at=_dt("2026-04-16T00:00:00Z"),
            last_seen_at=_dt("2026-04-18T23:00:00Z"),
            last_volume_24h_usd=1250.0,
            last_liquidity_usd=500.0,
        ))
        db.add(SleeveConfig(
            sleeve_id="weather__balanced",
            stance="balanced",
            strategy_name="weather",
            market_selector="strategy_family=weather",
            bankroll_usd="1000",
            max_position_usd="25",
            min_edge_bps=120,
            max_cross_spread_bps=40,
            enabled=True,
            version=2,
            notes="seed",
            extra_json={"city": "NYC"},
            updated_at=_dt("2026-04-18T00:00:00Z"),
        ))
        db.add_all([
            OrderIntentRow(
                client_order_id="co-in",
                sleeve_id="weather__balanced",
                market_condition_id="market-1",
                token_id="tok-1",
                side="buy",
                order_type="limit",
                limit_price="0.53",
                size_usd="10",
                size_shares="18.87",
                edge_bps=220,
                category="weather",
                reasoning="in-window intent",
                created_at=_dt("2026-04-18T12:00:00Z"),
            ),
            OrderIntentRow(
                client_order_id="co-out",
                sleeve_id="weather__balanced",
                market_condition_id="market-1",
                token_id="tok-1",
                side="buy",
                order_type="limit",
                limit_price="0.48",
                size_usd="7",
                size_shares="14.58",
                edge_bps=140,
                category="weather",
                reasoning="out-of-window intent",
                created_at=_dt("2026-04-16T12:00:00Z"),
            ),
            FillRow(
                fill_id="fill-in",
                client_order_id="co-in",
                mode="paper",
                rejected=False,
                filled_size_shares="18.87",
                avg_price="0.53",
                notional_usd="10",
                fees_usd="-0.02",
                gas_usd="0",
                confidence="high",
                slippage_bps=3,
                latency_ms=25,
                legs_json=[],
                notes="in-window fill",
                resolved=True,
                resolved_winner="Yes",
                resolved_pnl_usd="8.87",
                resolved_at=_dt("2026-04-18T15:00:00Z"),
                created_at=_dt("2026-04-18T12:00:02Z"),
            ),
            FillRow(
                fill_id="fill-out",
                client_order_id="co-out",
                mode="paper",
                rejected=False,
                filled_size_shares="14.58",
                avg_price="0.48",
                notional_usd="7",
                fees_usd="-0.01",
                gas_usd="0",
                confidence="medium",
                slippage_bps=2,
                latency_ms=30,
                legs_json=[],
                notes="out-of-window fill",
                resolved=False,
                created_at=_dt("2026-04-16T12:00:03Z"),
            ),
            WeatherForecastRecord(
                event_slug="rain-1",
                event_title="Rain in NYC",
                market_condition_id="market-1",
                token_id="tok-1",
                city="NYC",
                kind="precipitation_sum_period",
                bucket_lower=5.0,
                bucket_upper=None,
                bucket_raw_question="5mm+",
                fair_value=0.62,
                raw_fair_value=0.58,
                ensemble_size=40,
                members_in_bucket=25,
                horizon_hours=18.0,
                post_processing="ngr",
                market_bid=0.5,
                market_ask=0.56,
                resolved_at=_dt("2026-04-18T20:00:00Z"),
                observed_outcome=True,
                observed_value=6.1,
                recorded_at=_dt("2026-04-18T11:45:00Z"),
                sleeve_id="weather__balanced",
                intent_fired=True,
            ),
            WeatherForecastRecord(
                event_slug="rain-0",
                event_title="Rain in NYC",
                market_condition_id="market-1",
                token_id="tok-1",
                city="NYC",
                kind="precipitation_sum_period",
                bucket_lower=5.0,
                bucket_upper=None,
                bucket_raw_question="5mm+",
                fair_value=0.41,
                raw_fair_value=0.39,
                ensemble_size=40,
                members_in_bucket=16,
                horizon_hours=48.0,
                post_processing="raw",
                market_bid=0.38,
                market_ask=0.42,
                recorded_at=_dt("2026-04-16T09:00:00Z"),
                sleeve_id="weather__balanced",
                intent_fired=False,
            ),
        ])
        await db.commit()


def test_parse_export_datetime_date_end_is_next_day() -> None:
    start = export_bundle.parse_export_datetime("2026-04-18")
    end = export_bundle.parse_export_datetime("2026-04-18", is_end=True)

    assert start == _dt("2026-04-18T00:00:00Z")
    assert end == _dt("2026-04-19T00:00:00Z")


@pytest.mark.asyncio
async def test_filtered_export_bundle_includes_weather_full_tape_and_filtered_logs(
    isolated_session,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    await _seed_rows(isolated_session)

    runtime_log = tmp_path / "runtime.jsonl"
    runtime_log.write_text(
        "\n".join([
            json.dumps({"timestamp": "2026-04-18T12:30:00Z", "event": "tick_done"}),
            json.dumps({"timestamp": "2026-04-16T12:30:00Z", "event": "tick_done"}),
        ]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        export_bundle,
        "existing_log_exports",
        lambda: [("logs/runtime.jsonl", runtime_log)],
    )

    export_filter = export_bundle.ExportFilter(
        start=_dt("2026-04-18T00:00:00Z"),
        end=_dt("2026-04-19T00:00:00Z"),
    )

    manifest = await export_bundle.build_manifest(export_filter=export_filter)
    files = {entry["path"]: entry for entry in manifest["files"]}

    assert files["tables/order_intents.ndjson"]["row_count"] == 1
    assert files["tables/fills.ndjson"]["row_count"] == 1
    assert files["tables/weather_forecast_records.ndjson"]["row_count"] == 1
    assert files["derived/full_tape.ndjson"]["row_count"] == 1
    assert files["tables/markets.ndjson"]["row_count"] == 1

    destination = tmp_path / "bundle.zip"
    await export_bundle.write_export_bundle(
        destination,
        metrics_payload={"markets_in_universe": 1},
        tape_payload=[{"fill_id": "fill-in"}],
        export_filter=export_filter,
    )

    with zipfile.ZipFile(destination) as zf:
        names = set(zf.namelist())
        assert "summary/export_scope.json" in names
        assert "tables/weather_forecast_records.ndjson" in names
        assert "derived/full_tape.ndjson" in names
        assert "logs/runtime.jsonl" in names

        weather_rows = [
            json.loads(line)
            for line in zf.read("tables/weather_forecast_records.ndjson").decode("utf-8").splitlines()
        ]
        assert len(weather_rows) == 1
        assert weather_rows[0]["recorded_at"] == "2026-04-18T11:45:00+00:00"

        tape_rows = [
            json.loads(line)
            for line in zf.read("derived/full_tape.ndjson").decode("utf-8").splitlines()
        ]
        assert len(tape_rows) == 1
        assert tape_rows[0]["client_order_id"] == "co-in"
        assert tape_rows[0]["intent_created_at"] == "2026-04-18T12:00:00+00:00"

        fill_rows = zf.read("tables/fills.ndjson").decode("utf-8").splitlines()
        assert len(fill_rows) == 1

        log_rows = zf.read("logs/runtime.jsonl").decode("utf-8").splitlines()
        assert len(log_rows) == 1
        assert json.loads(log_rows[0])["timestamp"] == "2026-04-18T12:30:00Z"
