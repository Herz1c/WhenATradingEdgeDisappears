"""In-process live bias computation for the live bot.

This is the windowed bias computation (a few seconds, not 14 minutes) wrapped
as an importable function so the bot can call it on its own loop -- no
external canonical_daemon, no waiting on someone to rerun
build-live-reference-events. Bit-identical to the full build for the latest
finalized second (since the residual bias at time T only depends on data in
[T-LOOKBACK, T] and we use the EXACT pipeline functions on a windowed slice).

See tools/compute_live_bias.py for the standalone CLI version and validation
against the full build.
"""
from __future__ import annotations

import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from chainlink_recorder.live_reference_events import (
    DEFAULT_POLICY_PATH,
    SECOND_NS,
    SOURCE_SPECS,
    STALE_SECONDS,
    SourceFileDecision,
    _apply_delayed_residual_bias,
    _build_delayed_bias_residuals,
    _hour_key,
    _is_finite,
    _list_source_tasks,
    _parse_rows_for_spec,
    _run_parallel,
    _spot_only_premium_price,
    load_dataset_policy,
)


def _process_active_hour_file(task: dict) -> SourceFileDecision:
    """Parse a task that the strict pipeline would reject as
    'manifest_quality_class:active_tail_tolerant'. This is the in-progress hour
    -- valid raw data, just not yet finalized. The dataset pipeline excludes it
    for parity; the LIVE bot needs it to get a bias for the current second."""
    spec = SOURCE_SPECS[str(task["source_name"])]
    raw_path = Path(task["raw_path"])
    try:
        rows = _parse_rows_for_spec(raw_path, spec, int(task["availability_offset_seconds"]))
    except Exception:
        return SourceFileDecision(
            source_name=spec.source_name, date_str=str(task["date_str"]),
            hour_str=str(task["hour_str"]), raw_path=str(raw_path),
            manifest_path="", eligible=False, reason_category="parse_failure",
            quality_class="active_tail_tolerant", file_state="active",
            source_tier=None, valid_row_count=0, rows=(),
        )
    return SourceFileDecision(
        source_name=spec.source_name, date_str=str(task["date_str"]),
        hour_str=str(task["hour_str"]), raw_path=str(raw_path),
        manifest_path="", eligible=True, reason_category=None,
        quality_class="active_tail_tolerant", file_state="active",
        source_tier=None, valid_row_count=len(rows), rows=tuple(rows),
    )

LOOKBACK_S = 3600
SAFETY_MARGIN_S = 180


def _windowed_tasks(root: Path, policy, policy_path: Path, source_name: str,
                    window_start_ts: int, target_ts: int) -> list[dict]:
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
        if file_start + 3600 > window_start_ts and file_start <= target_ts:
            kept.append(t)
    return kept


def compute_live_bias_value(
    *,
    root: str | Path = "data",
    policy_path: str | Path | None = None,
    target_ts: int | None = None,
    window_hours: float = 6.0,
    max_workers: int = 4,
    include_active_hour: bool = True,
) -> dict | None:
    """Compute the live bias at `target_ts` (default: now). Returns a dict with
    the latest finalized second's bias info, or None if no eligible data.

    Returns:
        {
          "ts": int,                        # epoch seconds, latest finalized
          "rolling_bias": float,            # the canonical rolling_bias
          "synthetic_corrected": float,
          "synthetic_raw": float,
          "mids": dict[str, float],         # source -> mid at ts
          "bot_extracted_bias": float,      # synthetic_corrected - median(mids)
          "bias_active": bool,
          "bias_mode": str,
          "value_age_seconds": float,       # age of the bias value being applied
          "elapsed_seconds": float,
        }
    """
    t0 = time.time()
    root = Path(root)
    policy_path = Path(policy_path) if policy_path else DEFAULT_POLICY_PATH
    policy = load_dataset_policy(policy_path)
    primary = list(policy.live_reference_primary_inputs)

    target_ts = int(target_ts) if target_ts else int(time.time())
    window_start_ts = target_ts - int(window_hours * 3600)

    tasks: list[dict] = []
    for src in (*primary, "chainlink_public_delayed"):
        tasks.extend(_windowed_tasks(root, policy, policy_path, src, window_start_ts, target_ts))
    decisions = list(_run_parallel(tasks, max_workers=max_workers))

    # When include_active_hour is True, take any task the strict pipeline
    # rejected as active_tail_tolerant (the in-progress hour) and re-parse it
    # leniently. Live bot needs this so it can compute a bias for "right now"
    # instead of waiting up to an hour for the current hour to finalize.
    if include_active_hour:
        rejected_active = [t for t, d in zip(tasks, decisions)
                           if (not d.eligible) and d.reason_category
                           and "active_tail_tolerant" in str(d.reason_category)]
        if rejected_active:
            extras = [_process_active_hour_file(t) for t in rejected_active]
            # Replace the original ineligible decisions with the lenient ones
            rejected_paths = {str(t["raw_path"]) for t in rejected_active}
            decisions = [d for d in decisions if d.raw_path not in rejected_paths] + extras

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
        return None

    end_ts = min(target_ts, max(r.effective_ts_seconds for s in primary for r in eligible_updates[s]))
    start_ts = max(window_start_ts, end_ts - (LOOKBACK_S + SAFETY_MARGIN_S))

    idx = {s: 0 for s in primary}
    cur = {s: None for s in primary}
    synthetic_raw_history: dict[int, float] = {}
    rows_out: list[dict] = []
    for ts in range(start_ts, end_ts + 1):
        hour = _hour_key(ts)
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
                row[f"{s}_mid"] = float(u.mid)
            else:
                row[f"{s}_mid"] = math.nan
        spot_mid = row.get("binance_spot_mid")
        synth = _spot_only_premium_price(spot_mid=float(spot_mid) if _is_finite(spot_mid) else None)
        price_valid = bool(_is_finite(synth))
        row["synthetic_raw"] = float(synth) if price_valid else math.nan
        row["price_valid"] = price_valid
        synthetic_raw_history[ts] = row["synthetic_raw"]
        rows_out.append(row)

    residuals, _ = _build_delayed_bias_residuals(
        calibration_rows=calibration_rows, synthetic_raw_history=synthetic_raw_history)
    _apply_delayed_residual_bias(rows=rows_out, residuals=residuals)

    valid = [r for r in rows_out if r.get("price_valid")]
    if not valid:
        return None
    last = valid[-1]

    mids = {s: float(last[f"{s}_mid"]) for s in primary
            if _is_finite(last.get(f"{s}_mid")) and float(last.get(f"{s}_mid")) > 0}
    bot_bias = None
    if mids:
        bot_bias = float(last["synthetic_corrected"]) - statistics.median(list(mids.values()))

    return {
        "ts": int(last["ts_seconds"]),
        "rolling_bias": float(last["rolling_bias"]),
        "synthetic_corrected": float(last["synthetic_corrected"]),
        "synthetic_raw": float(last["synthetic_raw"]),
        "mids": mids,
        "bot_extracted_bias": bot_bias,
        "bias_active": bool(last.get("bias_active", False)),
        "bias_mode": last.get("bias_mode"),
        "value_age_seconds": float(last.get("applied_bias_age_seconds")
                                   or last.get("bias_state_age_seconds")
                                   or 0.0),
        "elapsed_seconds": time.time() - t0,
    }
