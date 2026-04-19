# poly-paper

Polymarket paper trading platform with paper↔live execution parity.

**Core promise:** every sleeve that shows profit in paper produces the same
trades in live. Paper and live share the exact same code path through the
exec router; only the final dispatch differs. Every paper fill carries a
`HIGH / MEDIUM / LOW` confidence tag so the dashboard never silently hides
the fact that e.g. a large maker order in an illiquid book would execute
differently live.

105 tests passing. Live-tested against Polymarket's real API.

---

## Deploy to Railway (first 10 minutes)

### 1. Push to GitHub
```bash
cd poly-paper
git init
git add .
git commit -m "initial commit"
git remote add origin git@github.com:YOURUSER/poly-paper.git
git push -u origin main
```

### 2. Create the Railway service
1. Go to railway.app → New Project → Deploy from GitHub repo → select `poly-paper`
2. Railway reads `nixpacks.toml` and `railway.toml` automatically
3. It will build and fail on the first deploy because there's no database yet

### 3. Add Postgres
1. In the Railway project: **+ New → Database → Add PostgreSQL**
2. Railway auto-creates a `DATABASE_URL` env var and wires it to your service
3. The code translates `postgres://` → `postgresql+asyncpg://` automatically

### 4. Set environment variables
In the Railway service → Variables, set:
```
POLY_MODE=paper
POLY_BANKROLL_USD=1000
JSON_LOGS=1
```
Everything else has sensible defaults (see `.env.example`).

### 5. Deploy
Push to `main` or click **Deploy** in Railway. Watch the Deploy Logs pane.
When you see `tick_done` messages arriving, it's working.

### 6. Verify
Railway will expose a public URL. Hit:
- `https://YOUR-URL/` → **web dashboard** (auto-refreshing sleeve health, recent trades, system status)
- `https://YOUR-URL/healthz` → `{"status":"ok"}`
- `https://YOUR-URL/readyz` → `{"status":"ready", ...}` after first tick
- `https://YOUR-URL/metrics` → per-sleeve stats JSON
- `https://YOUR-URL/tape` → last 50 trades
- `https://YOUR-URL/tape?start=2026-04-18&end=2026-04-18&limit=200` → filtered tape slice for one day
- `https://YOUR-URL/export/manifest` → lightweight inventory of the full export
- `https://YOUR-URL/export/manifest?start=2026-04-18&end=2026-04-18` → inventory for a filtered one-day export
- `https://YOUR-URL/export/download` → zipped NDJSON export of all persisted audit tables
- `https://YOUR-URL/export/download?start=2026-04-18&end=2026-04-18` → lighter one-day zip

---

## Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests (must be 105 passing)
pytest

# Verify paper fill simulator against live Polymarket
python -m scripts.verify_paper

# Run the engine locally (SQLite by default)
python -m scripts.run_paper

# In another terminal — CLI status dashboard
python -m scripts.status
```

---

## Current state — what exists

### Working (162 tests)
- **Exec layer** — `OrderIntent`/`Fill`/`OrderBook` Pydantic models, paper fill
  simulator that walks real Polymarket L2 books, verified per-category fee
  formula, shared router enforcing paper↔live parity.
- **Feeds** — Coinbase BTC spot + Kraken fallback; Open-Meteo free 40-member
  ensemble weather forecasts (live-verified: 39 members per city);
  Polymarket Gamma/CLOB REST client.
- **Strategies**
  - `btc_updown` — directional BTC up/down 5m/15m with three sleeves
  - `bundle_arb` — risk-free bundle detection for binary YES/NO and `negRisk` events
  - `ladder_arb` + `ladder_classify` — monotonicity detection for date and
    threshold ladders; parses ladder variable from question text
  - `weather` — temperature + precipitation buckets with non-parametric
    or NGR-calibrated fair values
- **Research-rigor modules** (new)
  - `ngr` — Non-homogeneous Gaussian Regression post-processing for
    ensembles. Minimum-CRPS fitting. Standard ECMWF-style EMOS that
    corrects raw ensemble bias and under-dispersion. On biased synthetic
    data, cuts CRPS by ~55%.
  - `weather_calibration` — persists every forecast with bucket metadata,
    attaches observed outcomes at resolution, computes Brier score, Brier
    Skill Score vs climatology, reliability bins, sharpness, AUC.
    Without this, "edge" is an untested claim.
  - `kelly` — fractional Kelly sizing replaces flat `max_position_usd`.
    Quarter-Kelly default (per Thorp, Downey). Correlation-adjusted group
    caps for mutually-exclusive buckets within one event.
- **Universe** — rules-based auto-discovery with hard size caps.
- **Runner + arb scanner + weather runner** — three concurrent strategy loops.
- **Self-correction proposer** — continuous tuning with HARD BOUNDS.
- **Web dashboard at `/`** — auto-refreshing sleeve health, recent trades,
  system status. No build step.
- **HTTP endpoints** — `/`, `/healthz`, `/readyz`, `/metrics`, `/tape`,
  `/export/manifest`, `/export/download`.
- **Export bundle** — supports optional `start` / `end` filters, includes
  `derived/full_tape.ndjson` for joined fill+intent history, and exports
  `tables/weather_forecast_records.ndjson` for weather calibration audits.

### Empirical findings from real Polymarket data
**The honest results from live scans:**

| Probe | Scope | Real arbs found |
|---|---|---|
| Binary complementary (ask_YES + ask_NO < $1) | 267 high-liquidity markets | 0 |
| neg_risk event bundle (Σ asks < $1) | 305 events | 0 (5 false positives from missing books) |
| Date/threshold ladder monotonicity | 1,222 events, 167 coherent ladders | 0 actionable |
| BTC 5m/15m directional edge | 51 markets, 153 sleeve evaluations | 0 (gross edges 5-34 bps vs 50+ bps threshold) |

**Polymarket is efficient at the static snapshot level on most dimensions.**
The system correctly refuses to manufacture trades when no edge exists —
that's the telling-the-truth behaviour, not a bug. Real alpha likely requires
catching transient imbalances (the continuous arb scanner) or strategies with
genuine models (weather ensemble, directional theses, LLM judgment).

### Not yet built
- Live execution (`exec/live.py` is a stub that raises NotImplementedError)
- News LLM judge (deferred until calibration system exists)
- Full PnL attribution (realised + mark-to-market)

---

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./poly_paper.db` | Railway Postgres URL auto-translated |
| `PORT` | `8080` | HTTP server port; Railway sets this automatically |
| `POLY_MODE` | `paper` | `live` not implemented; router refuses |
| `POLY_BANKROLL_USD` | `1000` | total paper bankroll |
| `POLY_TICK_SECONDS` | `10` | main loop interval |
| `POLY_TICK_TOLERANCE` | `120` | `/readyz` fails if no tick within this many seconds |
| `POLY_VOL_REFRESH_SECONDS` | `1800` | BTC realized vol refresh cadence |
| `POLY_FAMILIES` | `btc_up_down_5m,btc_up_down_15m` | enabled strategy families |
| `POLY_ARB_SCAN_SECONDS` | `30` | arb scanner loop interval |
| `POLY_ARB_MIN_LIQ` | `1000` | min market liquidity for arb scan |
| `JSON_LOGS` | unset | set to `1` for JSON logs (Railway) |
| `POLY_EVENT_LOG_PATH` | auto | optional JSONL runtime log path; included in `/export/download` if present |

### Export semantics

- `start` and `end` are optional on `/tape`, `/export/manifest`, and `/export/download`
- Date-only values like `2026-04-18` are treated as UTC day boundaries
- `start` is inclusive and `end` is exclusive internally, so `start=2026-04-18&end=2026-04-18` gives one full UTC day
- Reference tables such as `markets` and `sleeve_configs` stay global for context
- Heavy time-series tables are filtered by window to keep the zip lighter
- `summary/recent_tape.json` is a preview slice; `derived/full_tape.ndjson` is the full joined export for the chosen window

---

## Project layout

```
poly_paper/
├── exec/              # shared paper↔live execution layer
│   ├── models.py      # OrderIntent, Fill, OrderBook, FillConfidence, SleeveConfig
│   ├── fees.py        # verified Polymarket per-category fee formula
│   ├── paper.py       # paper fill simulator walking real books
│   ├── live.py        # live stub (Phase 4 will implement)
│   └── router.py      # single execute_order() entry point
├── polymarket/rest.py # public Gamma/CLOB client + category classifier
├── feeds/
│   ├── btc_spot.py    # Coinbase + Kraken fallback
│   └── open_meteo.py  # 40-member ensemble weather forecasts
├── strategies/
│   ├── fair_value.py  # GBM math for BTC markets
│   ├── btc_updown.py  # directional 5m/15m sleeves
│   ├── bundle_arb.py  # binary+negRisk bundle arbs
│   ├── ladder_arb.py  # date/threshold ladder monotonicity
│   └── ladder_classify.py  # parse ladder variable from question text
├── universe/loader.py # rules-based market auto-discovery
├── db/                # async SQLAlchemy (SQLite local, Postgres prod)
├── runner.py          # main tick loop
├── arb_scanner.py     # continuous arb scanner (separate loop)
└── http_server.py     # /healthz /readyz /metrics /tape /export/*

scripts/
├── run_paper.py       # production entry point
├── verify_paper.py    # paper sim smoke test vs live Polymarket
└── status.py          # CLI dashboard

tests/                 # 105 tests across fees, paper, strategies, arbs, classifier
```

---

## Honesty principles

1. **No trade without measurable edge** — every sleeve declares `min_edge_bps`
   and `min_gross_edge_bps`. Gross gate prevents spread-harvesting trades
   that look profitable on paper but have no directional thesis.

2. **Fill confidence logged on every fill** — MEDIUM/LOW confidence fills
   surface in `/metrics`; a "winning" sleeve whose fills are mostly LOW
   confidence will underperform live and the dashboard flags it.

3. **Book sanity check before every binary trade** — if `ask_YES + ask_NO`
   is outside `[$0.95, $1.06]` we skip (book stale/disjointed).

4. **Arb safety always verifiable** — bundle arbs only fire on binary YES/NO
   or Polymarket-tagged `negRisk` events. Ladder arbs only fire on coherent
   date-only or threshold-only ladders with strict monotonicity violations.

5. **Rejected intents are logged too** — "no trade" is data. Universe of
   evaluations ÷ trades-fired shows per-sleeve selectivity over time.

---

## License

Private. Built for a specific user.
