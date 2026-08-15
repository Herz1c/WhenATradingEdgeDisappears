"""Live trading bot wrapping the poly_l2_only_v2 strategy.

Strategy spec frozen in:
  artifacts_cleaned/poly_l2_only_v2/STRATEGY_FINAL_2026-06-03.md

Design priorities:
  - LOW LATENCY hot path. WS frame -> decision in << 1 ms (excluding the
    LightGBM predict call, which is ~1-3 ms and unavoidable).
  - Order submission is fire-and-forget (asyncio task) so the hot path
    never blocks on HTTP.
  - Logging via background daemon thread + lock-free queue. The hot path
    just does queue.put_nowait() which is nanosecond-level.
  - No JSON parsing or file I/O in the hot path.
  - Pre-allocated numpy buffer for predictions.
  - Asset-id -> market-state lookup is O(1) dict.
"""
