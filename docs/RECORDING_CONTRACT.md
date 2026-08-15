# Recording Contract

## Purpose

This document defines the recording and ingestion contract for the BTC/USD microstructure stack and the Polymarket BTC 5-minute Up/Down market family. The goal is to preserve causal, training-safe raw data that can later support labeling, feature extraction, and auditability without rerecording.

## Recording Windows

### Polymarket BTC 5-minute Up/Down

- Markets open exactly at UTC minute marks divisible by 5.
- A market is active when `open_ts - 150s <= now < close_ts`.
- The old market must remain active through its exact close boundary.
- The next market may become active during its pre-open window while the previous market is still active.
- Overlap must be real: the subscribed asset set is the union of all active markets.
- Adjacent markets must have positive core overlap in `.ws` and `.l2`.
- A market must never be unsubscribed early just because a newer market is selectable.
- Post-close capture may exist for debug or verification, but it must never substitute for missing live-window capture.

## Canonical Time Policy

### Default cross-source join key

- `recv_ts_ns` is the default canonical cross-source join key.
- `recv_ts_iso` is the human-readable companion timestamp.

### Safe event-time families

- Exchange venue event timestamps that are emitted densely enough to represent the corresponding source event may be used for source-local lag/QC.
- Examples include:
  - Coinbase Advanced channel timestamps such as `updates_event_time_ns`, `trades_time_ns`, `message_timestamp_ns`
  - Binance trade/depth event timestamps where explicitly normalized as source-event fields
  - Polymarket RTDS / Chainlink RTDS message or price timestamps for RTDS-specific freshness telemetry

### Unsafe generic join clocks

The following must not be used as generic cross-source join timestamps:

- Polymarket metadata timestamps when `timestamp_semantics` is not `source_event`
- public delayed Chainlink display timestamps
- Chainlink onchain `updated_at_s`
- FNG daily timestamps as intraday event times
- Hyperliquid historical endpoint timestamps such as `fundingHistory`
- Deribit sparse metadata timestamps

### Causality

- No future leakage is allowed.
- If a source timestamp is delayed-display, metadata-only, sparse metadata, or verification-only, it must remain classified that way in QC and downstream readers.

## File-State Policy

- Training readers are finalized-only by default.
- Active-file reads are opt-in and tail-tolerant only.
- Active files remain excluded unless `tail_tolerant=True` is explicitly requested.
- Manifest `record_count` must reflect readable raw rows on finalized files.
- Low-frequency sources must still finalize correctly when no later write occurs inside the old hour.

## Data Retention Policy

The stack must retain:

- raw `.ws`
- normalized `.l2`
- `.rest`
- `.strike`
- `.discovery`
- `.lifecycle`
- manifests
- enough metadata to reproduce processing and QC decisions

## Required Per-Session / Per-File Metadata

Manifests and lifecycle context should preserve:

- recorder service identity
- runtime session id
- runtime start timestamp
- workspace version / git SHA when available
- redacted config snapshot and config fingerprint
- source family / venue / market family
- Polymarket market slug / id
- Polymarket open / close / active-window boundaries
- active-market-set lifecycle telemetry
- restart / reconnect lifecycle markers

## Source Role Classification

- `polymarket/btc_updown_5m`: primary training
- `binance/spot`, `binance/usdm`, `coinbase`, `bybit`, `hyperliquid`: primary or secondary live context depending on file kind
- `polymarket/rtds/crypto_prices_chainlink/btc_usd`: secondary live proxy context
- `chainlink_public_delayed/public_stream_page`: verification-only delayed comparison layer
- `chainlink/onchain`: verification-only
- `deribit/options/BTC`: secondary context
- `fng`: slow/daily secondary context

## Reader Safety

- Do not silently drop problematic rows just to make audits look clean.
- Surface unsafe or off-market rows through explicit flags, quarantine logic, lifecycle events, or QC states.
- Preserve selected-market cleanliness in Polymarket `.l2`.
- Preserve `label_safe=false` isolation for invalid strike rows.
- Preserve order-book-health gating for local book features.
