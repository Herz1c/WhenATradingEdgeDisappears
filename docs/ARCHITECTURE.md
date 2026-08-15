# Architecture

The stack has three layers: a recording tier that never makes decisions, a dataset tier
that turns raw records into causal tensors, and a modelling and evaluation tier. I kept
them separate on purpose. My rule was record first, decide later, because a recording
decision you regret is unrecoverable and a modelling decision you regret is not.

---

## 1. Recording tier

### Services

15 services across 9 sources, each an independent `asyncio` task:

| Source | Services |
|---|---|
| Binance | spot `BTCUSDT`, USD-M perpetual `BTCUSDT` |
| Polymarket | BTC 5m market WS, REST poller, strike poller, resolution watcher, RTDS |
| Bybit | linear `BTCUSDT` |
| Coinbase Advanced | `BTC-USD` |
| Hyperliquid | REST `l2Book` |
| Deribit | options |
| Chainlink | Data Streams (direct, credentialed), public delayed page |
| Alternative.me | Fear & Greed index |

Profiles select subsets (`run-all`, `run-core-sources`, `run-external-venues`,
`run-oracle-strike`, `run-research-stack`) so a partial stack can run on a
constrained machine without editing code.

### Supervisor

`RecorderSupervisor` in `src/market_recorders/supervisor.py` owns a list of
`ManagedServiceState`, each running in its own task under `_run_service_loop`.
Crashed services restart with exponential backoff plus jitter (1 s → 30 s).
`FatalServiceError` stops the whole supervisor rather than looping on an
unrecoverable condition. Shutdown is graceful via `request_stop()` from a
SIGINT/SIGTERM handler, so open hours finalise instead of being truncated.

### Storage

```
data/<source>/<market>/<symbol>/YYYY/MM/DD/HH/<kind>.jsonl.zst
data/<source>/<market>/<symbol>/YYYY/MM/DD/HH/<kind>.manifest.json
```

Append-only zstd-compressed JSONL, hourly rotation on UTC, one manifest per file. Kinds
are `ws`, `snapshot`, `l2`, `lifecycle`, `discovery`, and source-specific variants. The
raw payload is always preserved. Decoded metadata is added alongside it, never in place
of it.

### Timestamps

Every record carries:

- `recv_ts_ns` — local wall clock at receipt, `time.time_ns()`
- `recv_ts_iso` — ISO 8601, nanosecond precision
- `recv_ts_source` — `"python_time_time_ns_epoch_wall_clock"`
- source timestamps, renamed by unit (`_ms`, `_us`, `_ns`) with `_iso` variants

HTTP recorders add `capture_timing` (`request_start_ts_ns`, `response_end_ts_ns`,
`response_midpoint_ts_ns`, `response_latency_ns`), which makes a REST observation's
uncertainty interval explicit rather than assumed.

Utilities: `src/market_recorders/time_utils.py`.

### Quality classes

`src/market_recorders/file_quality.py` — `HourlyQualityTracker` marks each hour,
worst-wins:

`continuity_compromised` > `stale_context` > `recovered` > `active_tail_tolerant`
> `finalized_clean`

Stored in the manifest as `quality_class` and `quality_flags`. Training queries filter
on it. This is how a connection drop stops being an invisible bias.

### Integrity monitors

- **Binance** (`order_book_qc.py`): local book reconstruction with sequence-gap
  detection, bootstrap/resync snapshots, zero-quantity level deletion, periodic
  parity telemetry, forced resnapshot on parity failure. USD-M handles the
  first post-snapshot bridge event, then uses futures `pu` continuity.
- **Bybit**: `continuity_state` on every row, so snapshot resets and
  continuity-uncertain segments are machine-readable.
- **Coinbase**: explicit gap / out-of-order / heartbeat-reset telemetry with a
  controlled resubscribe rather than silent continuation.
- **Polymarket**: reader/worker split (below).

### The Polymarket slow-consumer fix

This was the most consequential recording bug in the project, and it is worth describing
because the failure was silent.

The original design ran one loop per message: receive, track, serialise, compress, write.
At the market channel's 300 to 600 msg/s it could not keep up. The `websockets` library
queue absorbed minutes of frames, so `recv_ts` looked fresh while the payload's source
time was 60 to 180 seconds stale. TCP backpressure eventually caused server-side
disconnects. Worst of all, the decision-window tail of every market was lost, which is
the part that matters. Roughly 92 % of the v2 dataset rows were affected before I found
it.

The fix in `src/polymarket_recorder/ws_client.py` splits the loop:

- a **reader task** does nothing but drain the socket and stamp `recv_ts_ns` at true wire
  arrival, so recorded lag reflects the network rather than my own backlog
- a **worker loop** does all per-record work from an in-process deque, in batches of 20
  with an explicit yield, and emits `ws_ingest_stats` so queue depth is observable
- if queue depth exceeds a hard limit the backlog is dropped and the connection
  hard-reconnects, because for book state fresh beats complete
- a market's writers linger past its close, 30 seconds by default, so events in flight at
  the boundary still land in the closing market's files
- a pre-window hard resync fires about 90 seconds before close, to guarantee a fresh book
  entering the decision window

## 2. Dataset tier

Raw JSONL → canonical Parquet → episode tensors.

- **Canonical** (`src/canonical/`): per-source normalisation into a common
  schema with provenance carried through.
- **Historical episodes** (`tools/build_btc_5m_episode_dataset.py`): each 5-minute market
  becomes a fixed-grid tensor with a source-freshness mask. The builder orders events by
  source time, not receive time, so the mask does not make the tensor causal. This path is
  retained only as a counterfactual diagnostic and requires an explicit opt-in.
- **Splits** (`tools/build_splits_v2.py`): see [METHODOLOGY.md](METHODOLOGY.md).
- **Leakage-safe loading** (`src/model_factory/dataset_loader.py`): enforces the
  temporal split, a leakage-column blacklist, an eligibility filter, and a
  dataset hash. Training-mode loads of placeholder configs are refused outright.

## 3. Modelling and evaluation tier

- **Model factory** (`src/model_factory/`): 14 registered models, YAML-declared
  feature sets, provenance written with every artifact.
- **Sequence models**: TCN and GRU over 200 ms episode tensors, per-member Platt
  calibration, seed ensembles.
- **Execution simulator** (`src/execution_simulator/`, `src/backtest/`):
  deterministic replay with configurable fill modes (`cap_drop`, `fill_worse`),
  latency injection, fee schedule, and depth-walk slippage.
- **Policy and locks** (`src/policy/`, `artifacts/strategy_locks/`): historical paths read
  parameters from the same locally dated JSON lock. The public checkout can check stored
  output consistency, but absent data/model artifacts and external history prevent an
  end-to-end reproduction or independently authenticated freeze claim.
- **Shadow bot** (`src/live_bot/`): runs a locked strategy live against real
  feeds, logs decisions, sends no orders. The live path is gated behind
  `ENABLE_REAL_ORDERS`, a risk gate, and a per-market notional cap.

## 4. QC

`py -m market_recorders qc-day --date YYYY-MM-DD --out ./data` reports
source-tier counts, manifest quality classes, cross-source transport lag
(p50/p95/max of `recv_minus_source`), negative-lag incidence, silent gaps,
freshness spikes, and backlog bursts.

Small negative `recv_minus_source` values are expected from benign clock skew. They are
telemetry, not permission to join on source timestamps.

`tools/check_recorder_liveness.py` runs daily on a schedule. I added it after a recording
outage from 2026-07-05 to 07-10 went unnoticed for five days.
