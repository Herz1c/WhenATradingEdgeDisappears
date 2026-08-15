"""Gate: is today's canonical bias usable enough to let the live bot trade?

Reproduces the bot's own bias extraction (src/live_bot/btc_state.py::
reload_canonical_bias): bias = synthetic_corrected - median(exchange mids) on
the last price_valid row of today's canonical parquet.

Policy (per operator decision): the delayed-residual bias is a SLOWLY-changing
quantity, so its exact age barely matters. We allow the bias to be used as long
as the underlying value is no older than BIAS_MAX_AGE_S (default 24h) -- whether
it is `active_window` or `carried_forward`. We do NOT require it to be active.
This still rejects the genuinely-stale failure mode (e.g. the multi-day
carried-forward value during the chainlink outage), which is what actually
caused the bad trades.

Exit 0 -> READY ; Exit 1 -> NOT READY.

Env overrides:
    BIAS_MAX_AGE_S   max age of the applied bias value, seconds (default 86400 = 24h)
    BIAS_SANITY_USD  reject if |bias| exceeds this (default 250)
"""
from __future__ import annotations

import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

CANONICAL_DIR = Path("data/canonical/live_reference_events_v1")
MAX_AGE_S = float(os.environ.get("BIAS_MAX_AGE_S", "86400"))   # 24 hours
SANITY_USD = float(os.environ.get("BIAS_SANITY_USD", "250"))


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def main() -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = CANONICAL_DIR / f"{today}.parquet"
    if not path.exists():
        print(f"NOT READY: canonical parquet missing: {path}")
        return 1

    try:
        import polars as pl

        df = pl.read_parquet(
            path,
            columns=[
                "ts_seconds", "synthetic_corrected", "price_valid",
                "binance_spot_mid", "binance_usdm_mid", "hyperliquid_mid",
                "bias_active", "bias_mode",
                "applied_bias_age_seconds", "bias_state_age_seconds",
                "rolling_bias",
            ],
        ).filter(pl.col("price_valid"))
    except Exception as exc:  # noqa: BLE001
        print(f"NOT READY: failed to read canonical parquet: {exc!r}")
        return 1

    if df.height == 0:
        print("NOT READY: no price_valid rows in canonical parquet")
        return 1

    last = df.sort("ts_seconds").tail(1).row(0, named=True)

    synthetic = last.get("synthetic_corrected")
    if not _finite(synthetic):
        print("NOT READY: last price_valid row has no synthetic_corrected")
        return 1

    mids = [float(last[k]) for k in ("binance_spot_mid", "binance_usdm_mid", "hyperliquid_mid")
            if _finite(last.get(k)) and float(last.get(k)) > 0]
    if not mids:
        print("NOT READY: last row has no exchange mids")
        return 1

    bot_bias = float(synthetic) - statistics.median(mids)
    ts = int(last["ts_seconds"])
    row_age_s = datetime.now(timezone.utc).timestamp() - ts
    mode = last.get("bias_mode")
    active = bool(last.get("bias_active"))

    # Age of the actual bias VALUE the bot will apply (works for both active and
    # carried_forward). Fall back across the available age columns.
    value_age = None
    for k in ("applied_bias_age_seconds", "bias_state_age_seconds"):
        if _finite(last.get(k)):
            value_age = float(last[k]); break

    print(f"latest row: ts={ts} ({datetime.fromtimestamp(ts, timezone.utc):%H:%M:%SZ}) "
          f"mode={mode} active={active}")
    va = f"{value_age/3600:.1f}h" if value_age is not None else "unknown"
    print(f"  bias = {bot_bias:+.3f} USD | bias-value age {va} | row age {row_age_s/60:.1f} min")

    problems = []
    if value_age is None:
        problems.append("no bias value established yet (never had an active observation)")
    elif value_age > MAX_AGE_S:
        problems.append(f"bias value too old ({value_age/3600:.1f}h > {MAX_AGE_S/3600:.0f}h)")
    if abs(bot_bias) > SANITY_USD:
        problems.append(f"|bias| {abs(bot_bias):.1f} exceeds sanity limit {SANITY_USD:.0f}")

    if problems:
        print("NOT READY: " + "; ".join(problems))
        return 1

    print(f"READY: bias {bot_bias:+.3f} USD ({mode}, value age {va})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
