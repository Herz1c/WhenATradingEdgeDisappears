"""Compute the live_reference_events bias at a given moment WITHOUT rebuilding
the whole day.

WHY THIS IS IDENTICAL TO THE BACKTEST/FULL BUILD (not an approximation):
The delayed-residual bias at time T depends ONLY on chainlink_public_delayed
observations whose causal window covers T -- i.e. observations with
chainlink_ts in [T-LOOKBACK, T-DELAY] = [T-3600s, T-1800s] -- and the
synthetic price at those same seconds. Everything earlier in the day is
irrelevant to the bias at T. So we reuse the EXACT pipeline functions
(_spot_only_premium_price, _build_delayed_bias_residuals,
_apply_delayed_residual_bias) on a short window ending at T and get the
bit-identical value the full-day build would produce for T's finalized second.

It parses only the last ~N hours of raw files instead of 1-2 full days, so it
runs in seconds, not the ~14-min full-day build (no more CPU jet-engine).

NOTE ON SCOPE: exact for the ACTIVE-bias case (active_window) -- which is the
only case the bot is allowed to trade on. For carried_forward (no active
residual at T) a windowed view can't see a last-valid-bias older than the
window; that's fine because the readiness gate refuses to trade then anyway.

Usage:
    py -3 tools/compute_live_bias.py                 # bias right now
    py -3 tools/compute_live_bias.py --at 2026-05-29T03:59:59Z   # at a past second (for validation)
    py -3 tools/compute_live_bias.py --window-hours 3 --max-workers 4
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, "src")

from chainlink_recorder.live_reference_events import (  # noqa: E402
    DEFAULT_POLICY_PATH,
    SECOND_NS,
    STALE_SECONDS,
    _apply_delayed_residual_bias,
    _build_delayed_bias_residuals,
    _hour_key,
    _is_finite,
    _list_source_tasks,
    _run_parallel,
    _spot_only_premium_price,
    load_dataset_policy,
)

LOOKBACK_S = 3600
# Window must comfortably cover the residual lookback plus staleness slack.
SAFETY_MARGIN_S = 180


def _parse_at(value: str | None) -> int:
    if not value:
        return int(_time.time())
    v = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _windowed_tasks(root: Path, policy, policy_path: Path, source_name: str,
                    window_start_ts: int, target_ts: int) -> list[dict]:
    """All raw-file tasks for `source_name` whose hour overlaps the window."""
    d_from = datetime.fromtimestamp(window_start_ts, timezone.utc).date()
    d_to = datetime.fromtimestamp(target_ts, timezone.utc).date()
    tasks = _list_source_tasks(
        root=root, policy=policy, policy_path=policy_path,
        source_name=source_name, date_from=d_from, date_to=d_to,
    )
    kept = []
    for t in tasks:
        try:
            hour = int(str(t["hour_str"]))
            d = datetime.fromisoformat(t["date_str"]).date()
        except (ValueError, KeyError):
            continue
        file_start = int(datetime(d.year, d.month, d.day, hour, tzinfo=timezone.utc).timestamp())
        # Keep the hour-file if it overlaps [window_start, target].
        if file_start + 3600 > window_start_ts and file_start <= target_ts:
            kept.append(t)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    ap.add_argument("--at", default=None, help="UTC ISO time to evaluate (default: now)")
    ap.add_argument("--window-hours", type=float, default=3.0)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--json", action="store_true", help="emit only the JSON result")
    args = ap.parse_args()

    t0 = _time.time()
    root = Path(args.root)
    policy_path = Path(args.policy)
    policy = load_dataset_policy(policy_path)
    primary = list(policy.live_reference_primary_inputs)

    target_ts = _parse_at(args.at)
    window_start_ts = target_ts - int(args.window_hours * 3600)

    # 1) Load only the recent hour-files for each needed source (reusing the
    #    exact parser/quarantine path via _process_source_file inside _run_parallel).
    tasks: list[dict] = []
    for src in (*primary, "chainlink_public_delayed"):
        tasks.extend(_windowed_tasks(root, policy, policy_path, src, window_start_ts, target_ts))
    decisions = _run_parallel(tasks, max_workers=args.max_workers)

    by_source: dict[str, list] = {}
    for dcn in decisions:
        by_source.setdefault(dcn.source_name, []).append(dcn)

    eligible_updates: dict[str, list] = {}
    eligible_hours: dict[str, set] = {}
    for src in primary:
        decs = by_source.get(src, [])
        rows = [r for d in decs if d.eligible for r in d.rows]
        rows.sort(key=lambda r: (r.effective_ts_seconds, r.recv_ts_ns))
        eligible_updates[src] = rows
        eligible_hours[src] = {(d.date_str, d.hour_str) for d in decs if d.eligible}

    calibration_rows = [r for d in by_source.get("chainlink_public_delayed", [])
                        if d.eligible for r in d.rows]
    calibration_rows.sort(key=lambda r: (r.available_ts_seconds, r.recv_ts_ns))

    if not any(eligible_updates.get(s) for s in primary):
        out = {"ok": False, "reason": "no eligible primary (binance_spot) data in window",
               "target_ts": target_ts, "window_hours": args.window_hours}
        print(json.dumps(out, indent=2)); return 1

    # End at the latest finalized eligible second <= target (matches the full
    # build, which excludes the still-active current hour -> last finalized second).
    end_ts = min(target_ts, max(r.effective_ts_seconds for s in primary for r in eligible_updates[s]))
    start_ts = max(window_start_ts, end_ts - (LOOKBACK_S + SAFETY_MARGIN_S))

    # 2) Per-second synthetic loop -- replicated verbatim from
    #    build_live_reference_events (spot-only synthetic, STALE_SECONDS gate).
    idx = {s: 0 for s in primary}
    cur = {s: None for s in primary}
    synthetic_raw_history: dict[int, float] = {}
    rows: list[dict] = []
    for ts in range(start_ts, end_ts + 1):
        hour = _hour_key(ts)
        mids: list[float] = []
        row: dict = {"ts_seconds": ts}
        for s in primary:
            ups = eligible_updates[s]
            while idx[s] < len(ups) and ups[idx[s]].effective_ts_seconds <= ts:
                cur[s] = ups[idx[s]]; idx[s] += 1
            u = cur[s]
            if hour not in eligible_hours[s] or u is None or u.recv_ts_ns > ts * SECOND_NS:
                row[f"{s}_mid"] = math.nan; continue
            staleness = max(0.0, (ts * SECOND_NS - u.recv_ts_ns) / SECOND_NS)
            if staleness <= STALE_SECONDS:
                row[f"{s}_mid"] = float(u.mid); mids.append(float(u.mid))
            else:
                row[f"{s}_mid"] = math.nan
        spot_mid = row.get("binance_spot_mid")
        synth = _spot_only_premium_price(spot_mid=float(spot_mid) if _is_finite(spot_mid) else None)
        price_valid = bool(_is_finite(synth))
        row["synthetic_raw"] = float(synth) if price_valid else math.nan
        row["price_valid"] = price_valid
        row["source_count"] = len(mids)
        synthetic_raw_history[ts] = row["synthetic_raw"]
        rows.append(row)

    # 3) Exact residual + bias application (the parity-critical pipeline funcs).
    residuals, anomalies = _build_delayed_bias_residuals(
        calibration_rows=calibration_rows, synthetic_raw_history=synthetic_raw_history)
    _apply_delayed_residual_bias(rows=rows, residuals=residuals)

    # 4) Report the bias at the latest finalized second -- the value the bot reads.
    valid = [r for r in rows if r.get("price_valid")]
    if not valid:
        out = {"ok": False, "reason": "no price_valid second in window", "target_ts": end_ts}
        print(json.dumps(out, indent=2)); return 1
    last = valid[-1]
    mids = [float(last[f"{s}_mid"]) for s in primary
            if _is_finite(last.get(f"{s}_mid")) and float(last.get(f"{s}_mid")) > 0]
    bot_bias = (float(last["synthetic_corrected"]) - statistics.median(mids)) if mids else None
    age_s = int(_time.time()) - int(last["ts_seconds"])
    result = {
        "ok": True,
        "ts": int(last["ts_seconds"]),
        "ts_iso": datetime.fromtimestamp(last["ts_seconds"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bias_mode": last["bias_mode"],
        "bias_active": bool(last["bias_active"]),
        "rolling_bias": round(float(last["rolling_bias"]), 4),
        "synthetic_raw": round(float(last["synthetic_raw"]), 4),
        "synthetic_corrected": round(float(last["synthetic_corrected"]), 4),
        "bot_extracted_bias": round(bot_bias, 4) if bot_bias is not None else None,
        "row_age_seconds": age_s,
        "n_residuals": len(residuals),
        "n_calibration_obs": len(calibration_rows),
        "anomalies": anomalies,
        "elapsed_seconds": round(_time.time() - t0, 1),
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
