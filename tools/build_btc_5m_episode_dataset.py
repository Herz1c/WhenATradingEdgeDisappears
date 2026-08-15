"""Build historical BTC binary-market tensors from source-time replay.

Unlike build_sequence_dataset_from_fv.py, this does not start from the sparse
fair-value decision parquet.  It replays raw PM L2, RTDS, and optional Coinbase
source-time streams and emits one full 300-second sequence per market.

CAUTION: this builder preserves the historical counterfactual method so existing
artifacts can be inspected.  It is not a causal/live replay.  Running it requires
the explicit ``--allow-source-time-counterfactual`` acknowledgement.

Historical clock policy:
  * PM event clock: Polymarket payload/source timestamp.
  * RTDS availability clock: chainlink_ts + configured latency.
  * CEX event clock: Coinbase payload/source timestamp.
  * recv timestamps are diagnostics only.

Training code must use valid_mask for losses/metrics.  The mask enforces source
freshness, not causal receive-time availability, and therefore cannot make these
tensors live-replicable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fair_value.extractor import FairValueState
from tools.build_fair_value_v2_source_time import (
    NS,
    WARMUP_S,
    _auto_dates,
    _best_dir,
    _empty_cex_day,
    _raw_roots,
    _read_pm_frames,
    _rtds_price_at,
    _slug_from_path,
    load_coinbase_day,
    load_resolutions,
    load_rtds_day,
)
from tools.build_sequence_dataset_from_fv import (
    QUOTE_COLS,
    _array_stats,
    _expand_date_filter,
    _normalize,
    _save_npz,
    _split_dates,
)
from tools.train_fair_value_v2_source_time import feature_cols

DEFAULT_FEATURE_SOURCE = ROOT / "data" / "datasets" / "fair_value_v2_source_time"
DEFAULT_OUT = ROOT / "data" / "datasets" / "btc_5m_episodes_v1"

AUDIT_COLS = [
    "pm_source_age_s",
    "pm_delivery_lag_s",
    "pm_recv_lag_s",
    "cex_age_s",
    "coinbase_source_lag_s",
    "rtds_source_age_s",
    "rtds_avail_age_s",
    "rtds_latency_s",
]


def _uses_cex(feature_set: str) -> bool:
    return feature_set not in {"pm_rtds_safe"}


def _episode_grid(episode_seconds: int, cadence_s: float) -> tuple[np.ndarray, np.ndarray]:
    step_ns = round(float(cadence_s) * NS)
    if step_ns <= 0:
        raise ValueError("cadence must be positive")
    t_ns = np.arange(0, int(episode_seconds) * NS, step_ns, dtype=np.int64)
    t_s = (t_ns.astype(np.float64) / NS).astype(np.float32)
    ttc_s = (float(episode_seconds) - t_s).astype(np.float32)
    return t_s, ttc_s.astype(np.float32)


def _load_feature_names(feature_source: Path, feature_set: str) -> list[str]:
    files = sorted(feature_source.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no feature-source parquet files found in {feature_source}")
    sample = pd.read_parquet(files[0])
    features = feature_cols(sample, feature_set=feature_set)
    if "implied_p_up" not in features:
        raise SystemExit("feature_set must include implied_p_up for residual sequence models")
    return features


def _finite_or_nan(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _source_valid(
    audit: dict[str, float],
    *,
    uses_cex: bool,
    max_pm_source_age_s: float | None,
    max_cex_age_s: float | None,
    max_rtds_source_age_s: float | None,
) -> bool:
    if max_pm_source_age_s is not None:
        v = audit.get("pm_source_age_s", float("nan"))
        if not math.isfinite(v) or v > max_pm_source_age_s:
            return False
    if uses_cex and max_cex_age_s is not None:
        v = audit.get("cex_age_s", float("nan"))
        if not math.isfinite(v) or v > max_cex_age_s:
            return False
    if max_rtds_source_age_s is not None:
        v = audit.get("rtds_source_age_s", float("nan"))
        if not math.isfinite(v) or v > max_rtds_source_age_s:
            return False
    return True


_G: dict[str, Any] = {}


def _init_worker(
    res,
    rtds,
    rtds_cl,
    cexbt,
    cexag,
    features,
    t_grid,
    ttc_grid,
    require_cex,
    rtds_latency_s,
    max_pm_source_age_s,
    max_cex_age_s,
    max_rtds_source_age_s,
    min_valid_steps,
    min_valid_frac,
):
    _G.clear()
    _G.update({
        "res": res,
        "rtds": rtds,
        "rtds_cl": rtds_cl,
        "cexbt": cexbt,
        "cexag": cexag,
        "features": features,
        "t_grid": t_grid,
        "ttc_grid": ttc_grid,
        "require_cex": require_cex,
        "rtds_latency_s": rtds_latency_s,
        "max_pm_source_age_s": max_pm_source_age_s,
        "max_cex_age_s": max_cex_age_s,
        "max_rtds_source_age_s": max_rtds_source_age_s,
        "min_valid_steps": min_valid_steps,
        "min_valid_frac": min_valid_frac,
    })


def _market_worker(path_s: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    return _build_one_market(Path(path_s), **_G)


def _build_one_market(
    pm_path: Path,
    *,
    res: dict[str, dict[str, Any]],
    rtds,
    rtds_cl,
    cexbt,
    cexag,
    features: list[str],
    t_grid: np.ndarray,
    ttc_grid: np.ndarray,
    require_cex: bool,
    rtds_latency_s: float,
    max_pm_source_age_s: float | None,
    max_cex_age_s: float | None,
    max_rtds_source_age_s: float | None,
    min_valid_steps: int,
    min_valid_frac: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    slug = _slug_from_path(pm_path)
    audit: dict[str, Any] = {"path": str(pm_path), "slug": slug, "skipped_reason": None}
    if slug is None or slug not in res:
        audit["skipped_reason"] = "missing_resolution"
        return None, audit
    meta = res[slug]
    open_s = int(meta["open_s"])
    close_s = int(meta["close_s"])
    if not (open_s and close_s and close_s > open_s):
        audit["skipped_reason"] = "bad_market_window"
        return None, audit
    episode_seconds = round(close_s - open_s)
    if episode_seconds <= 0:
        audit["skipped_reason"] = "bad_episode_seconds"
        return None, audit

    strike = meta.get("open_price")
    if not strike or float(strike) <= 0:
        strike = _rtds_price_at(rtds_cl[0], rtds_cl[1], open_s * NS)
    if strike is None or float(strike) <= 0:
        audit["skipped_reason"] = "missing_strike"
        return None, audit

    lo_ns = (open_s - WARMUP_S) * NS
    hi_ns = close_s * NS
    pm_frames, pm_audit = _read_pm_frames(pm_path)
    pm_frames = [x for x in pm_frames if lo_ns <= x[0] <= hi_ns]
    audit["pm"] = pm_audit
    audit["pm_frames_in_window"] = len(pm_frames)
    if not pm_frames:
        audit["skipped_reason"] = "no_pm_frames"
        return None, audit

    cts, cbid, cask, cbq, caq, clag = cexbt
    ci0 = int(np.searchsorted(cts, lo_ns, "left"))
    ci1 = int(np.searchsorted(cts, hi_ns, "right"))
    ats, aq, asd = cexag
    ai0 = int(np.searchsorted(ats, lo_ns, "left"))
    ai1 = int(np.searchsorted(ats, hi_ns, "right"))
    ravail, rcl, rpx, rlag = rtds
    ri0 = int(np.searchsorted(ravail, lo_ns, "left"))
    ri1 = int(np.searchsorted(ravail, hi_ns, "right"))
    if require_cex and ci1 - ci0 == 0:
        audit["skipped_reason"] = "no_coinbase_window"
        return None, audit
    if ri1 - ri0 == 0:
        audit["skipped_reason"] = "no_rtds_window"
        return None, audit

    events: list[tuple[int, int, Any]] = []
    for source_ns, recv_ns, seq, rec, lag_s in pm_frames:
        events.append((int(source_ns), 0, (rec, float(lag_s), int(recv_ns))))
    if require_cex:
        for i in range(ci0, ci1):
            events.append((int(cts[i]), 1, (float(cbid[i]), float(cask[i]), float(cbq[i]), float(caq[i]), float(clag[i]))))
        # Only full feature set needs CEX trade-flow fields; updating trades is
        # harmless but still optional at the model feature-selection layer.
        for i in range(ai0, ai1):
            events.append((int(ats[i]), 2, (float(aq[i]), float(asd[i]))))
    for i in range(ri0, ri1):
        events.append((int(ravail[i]), 3, (int(rcl[i]), float(rpx[i]), float(rlag[i]))))
    events.sort(key=lambda x: (x[0], x[1]))

    seq_len = int(t_grid.size)
    x = np.full((seq_len, len(features)), np.nan, dtype=np.float32)
    quotes = np.full((seq_len, len(QUOTE_COLS)), np.nan, dtype=np.float32)
    audit_arr = np.full((seq_len, len(AUDIT_COLS)), np.nan, dtype=np.float32)
    now_ns_arr = np.zeros(seq_len, dtype=np.int64)
    row_present = np.zeros(seq_len, dtype=bool)
    source_valid = np.zeros(seq_len, dtype=bool)

    st = FairValueState(require_cex=require_cex)
    st.strike = float(strike)
    st.open_s = open_s
    st.close_s = close_s
    st.pm.market_open_s = open_s
    st.pm.market_close_s = close_s

    latest_pm_source_ns = 0
    latest_pm_recv_ns = 0
    latest_pm_delivery_lag_s = float("nan")
    latest_cex_source_ns = 0
    latest_cex_delivery_lag_s = float("nan")
    latest_rtds_avail_ns = 0
    latest_rtds_cl_ns = 0
    ei = 0
    n_events = len(events)

    for slot, t_s in enumerate(t_grid):
        now_ns = int(open_s * NS + round(float(t_s) * NS))
        now_ns_arr[slot] = now_ns
        while ei < n_events and events[ei][0] <= now_ns:
            ts, kind, data = events[ei]
            if kind == 0:
                rec, lag_s, recv_ns = data
                st.update_pm(rec)
                latest_pm_source_ns = int(ts)
                latest_pm_recv_ns = int(recv_ns)
                latest_pm_delivery_lag_s = float(lag_s)
            elif kind == 1:
                bid, ask, bq, aq_, lag_s = data
                st.update_binance_bookticker(ts, bid, ask, bq, aq_)
                latest_cex_source_ns = int(ts)
                latest_cex_delivery_lag_s = float(lag_s)
            elif kind == 2:
                st.update_binance_trade(ts, data[0], data[1])
            else:
                cl_ns, price, _source_lag_s = data
                st.update_rtds(ts, cl_ns, price)
                latest_rtds_avail_ns = int(ts)
                latest_rtds_cl_ns = int(cl_ns)
            ei += 1

        feat = st.features(now_ns)
        if feat is None:
            continue
        row_present[slot] = True
        x[slot, :] = _finite_or_nan([float(feat.get(c, np.nan)) for c in features])
        quotes[slot, :] = _finite_or_nan([float(feat.get(c, np.nan)) for c in QUOTE_COLS])
        audit_row = {
            "pm_source_age_s": (now_ns - latest_pm_source_ns) / NS if latest_pm_source_ns else float("nan"),
            "pm_delivery_lag_s": latest_pm_delivery_lag_s,
            "pm_recv_lag_s": (latest_pm_recv_ns - latest_pm_source_ns) / NS
            if latest_pm_recv_ns and latest_pm_source_ns else float("nan"),
            "cex_age_s": (now_ns - latest_cex_source_ns) / NS if latest_cex_source_ns else float("nan"),
            "coinbase_source_lag_s": latest_cex_delivery_lag_s,
            "rtds_source_age_s": (now_ns - latest_rtds_cl_ns) / NS if latest_rtds_cl_ns else float("nan"),
            "rtds_avail_age_s": (now_ns - latest_rtds_avail_ns) / NS if latest_rtds_avail_ns else float("nan"),
            "rtds_latency_s": float(rtds_latency_s),
        }
        audit_arr[slot, :] = _finite_or_nan([float(audit_row.get(c, np.nan)) for c in AUDIT_COLS])
        finite_features = bool(np.isfinite(x[slot, :]).all())
        source_valid[slot] = finite_features and _source_valid(
            audit_row,
            uses_cex=require_cex,
            max_pm_source_age_s=max_pm_source_age_s,
            max_cex_age_s=max_cex_age_s,
            max_rtds_source_age_s=max_rtds_source_age_s,
        )

    valid_mask = row_present & source_valid
    valid_count = int(valid_mask.sum())
    if valid_count < min_valid_steps or valid_count / max(1, seq_len) < min_valid_frac:
        audit["skipped_reason"] = "insufficient_valid_steps"
        audit["valid_count"] = valid_count
        audit["row_present_count"] = int(row_present.sum())
        return None, audit

    out = {
        "date": pd.Timestamp(open_s, unit="s", tz="UTC").date().isoformat(),
        "market_slug": slug,
        "open_s": int(open_s),
        "close_s": int(close_s),
        "strike": float(strike),
        "resolved_up": int(meta["resolved_up"]),
        "x": x,
        "quotes": quotes,
        "audit": audit_arr,
        "now_ns": now_ns_arr,
        "row_present": row_present,
        "source_valid": source_valid,
        "valid_mask": valid_mask,
        "valid_count": valid_count,
        "row_present_count": int(row_present.sum()),
    }
    audit["skipped_reason"] = None
    audit["valid_count"] = valid_count
    audit["row_present_count"] = int(row_present.sum())
    return out, audit


def _build_one_date(
    d: str,
    *,
    roots: list[Path],
    out_daily: Path,
    workers: int,
    max_markets: int,
    features: list[str],
    feature_set: str,
    t_grid: np.ndarray,
    ttc_grid: np.ndarray,
    rtds_latency_s: float,
    max_pm_source_age_s: float | None,
    max_cex_age_s: float | None,
    max_rtds_source_age_s: float | None,
    min_valid_steps: int,
    min_valid_frac: float,
    rebuild: bool,
) -> dict[str, Any]:
    t0 = time.time()
    out_daily.mkdir(parents=True, exist_ok=True)
    out_path = out_daily / f"{d}.npz"
    if out_path.exists() and not rebuild:
        return {"date": d, "status": "skipped_exists", "output": str(out_path), "elapsed_s": 0.0}

    require_cex = _uses_cex(feature_set)
    pm_dir = _best_dir(roots, "pm", d)
    res, res_stats = load_resolutions(d, roots)
    rtds, rtds_cl, rtds_stats = load_rtds_day(d, roots, rtds_latency_s)
    if require_cex:
        cexbt, cexag, cex_stats = load_coinbase_day(d, roots)
    else:
        cexbt, cexag, cex_stats = _empty_cex_day()
    pm_files = sorted(pm_dir.glob("*.l2.jsonl.zst")) if pm_dir else []
    if max_markets:
        pm_files = pm_files[:max_markets]

    missing = []
    if not pm_files:
        missing.append("pm")
    if not res:
        missing.append("resolution")
    if not rtds[0].size:
        missing.append("rtds")
    if require_cex and not cexbt[0].size:
        missing.append("coinbase")
    stats: dict[str, Any] = {
        "date": d,
        "status": "built",
        "roots": {
            "pm": str(pm_dir) if pm_dir else None,
            "resolution": res_stats.get("root"),
            "rtds": rtds_stats.get("root"),
            "coinbase": cex_stats.get("root"),
        },
        "pm_markets": len(pm_files),
        "resolutions": len(res),
        "rtds_rows": int(rtds[0].size),
        "coinbase_tickers": int(cexbt[0].size),
        "coinbase_trades": int(cexag[0].size),
        "require_cex": require_cex,
        "source_stats": {"rtds": rtds_stats, "coinbase": cex_stats},
        "skips": {},
    }
    if missing:
        stats["status"] = "missing_sources"
        stats["note"] = f"missing {missing}"
        stats["elapsed_s"] = round(time.time() - t0, 2)
        return stats

    initargs = (
        res, rtds, rtds_cl, cexbt, cexag, features, t_grid, ttc_grid,
        require_cex, rtds_latency_s, max_pm_source_age_s, max_cex_age_s,
        max_rtds_source_age_s, min_valid_steps, min_valid_frac,
    )
    episodes = []
    audits = []
    if workers > 1 and len(pm_files) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(pm_files)),
                                 initializer=_init_worker,
                                 initargs=initargs) as ex:
            for ep, audit in ex.map(_market_worker, [str(p) for p in pm_files], chunksize=4):
                audits.append(audit)
                if ep is not None:
                    episodes.append(ep)
                else:
                    reason = audit.get("skipped_reason") or "unknown"
                    stats["skips"][reason] = int(stats["skips"].get(reason, 0)) + 1
    else:
        _init_worker(*initargs)
        for pm_file in pm_files:
            ep, audit = _market_worker(str(pm_file))
            audits.append(audit)
            if ep is not None:
                episodes.append(ep)
            else:
                reason = audit.get("skipped_reason") or "unknown"
                stats["skips"][reason] = int(stats["skips"].get(reason, 0)) + 1

    stats["episodes"] = len(episodes)
    stats["valid_timesteps"] = int(sum(int(ep["valid_count"]) for ep in episodes))
    stats["present_timesteps"] = int(sum(int(ep["row_present_count"]) for ep in episodes))
    stats["valid_frac"] = round(stats["valid_timesteps"] / max(1, len(episodes) * len(t_grid)), 4) if episodes else 0.0
    stats["present_frac"] = round(stats["present_timesteps"] / max(1, len(episodes) * len(t_grid)), 4) if episodes else 0.0
    stats["market_audit_sample"] = audits[:20]

    if episodes:
        np.savez(
            out_path,
            X_raw=np.stack([ep["x"] for ep in episodes], axis=0).astype(np.float32),
            valid_mask=np.stack([ep["valid_mask"] for ep in episodes], axis=0),
            row_present_mask=np.stack([ep["row_present"] for ep in episodes], axis=0),
            source_valid_mask=np.stack([ep["source_valid"] for ep in episodes], axis=0),
            y=np.asarray([ep["resolved_up"] for ep in episodes], dtype=np.int8),
            p_market=np.stack([
                ep["quotes"][:, QUOTE_COLS.index("implied_p_up")] for ep in episodes
            ], axis=0).astype(np.float32),
            quotes=np.stack([ep["quotes"] for ep in episodes], axis=0).astype(np.float32),
            audit=np.stack([ep["audit"] for ep in episodes], axis=0).astype(np.float32),
            now_ns=np.stack([ep["now_ns"] for ep in episodes], axis=0).astype(np.int64),
            market_slug=np.asarray([ep["market_slug"] for ep in episodes]),
            date=np.asarray([ep["date"] for ep in episodes]),
            open_s=np.asarray([ep["open_s"] for ep in episodes], dtype=np.int64),
            close_s=np.asarray([ep["close_s"] for ep in episodes], dtype=np.int64),
            strike=np.asarray([ep["strike"] for ep in episodes], dtype=np.float32),
            valid_count=np.asarray([ep["valid_count"] for ep in episodes], dtype=np.int16),
            t_s=t_grid.astype(np.float32),
            ttc_s=ttc_grid.astype(np.float32),
        )
        stats["output"] = str(out_path)
    else:
        stats["status"] = "no_episodes"
    stats["elapsed_s"] = round(time.time() - t0, 2)
    return stats


def _stack_days(day_paths: list[Path]) -> dict[str, Any]:
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in day_paths:
        d = np.load(path, allow_pickle=True)
        for key in d.files:
            if key in {"t_s", "ttc_s"}:
                continue
            arrays[key].append(d[key])
    return {k: np.concatenate(v, axis=0) for k, v in arrays.items()}


def _write_splits(out_dir: Path, daily_dir: Path, dates: list[str], train_end: str,
                  test_start: str, val_days: int, features: list[str],
                  compress: bool) -> dict[str, Any]:
    splits = _split_dates(dates, train_end, test_start, val_days)
    paths_by_date = {p.name[:10]: p for p in daily_dir.glob("*.npz")}
    split_arrays: dict[str, dict[str, Any]] = {}
    for split, split_dates in splits.items():
        day_paths = [paths_by_date[d] for d in split_dates if d in paths_by_date]
        if not day_paths:
            raise SystemExit(f"split {split} has no daily episode shards")
        split_arrays[split] = _stack_days(day_paths)

    mean, std = _array_stats(split_arrays["train"]["X_raw"], split_arrays["train"]["valid_mask"])
    for split, arr in split_arrays.items():
        x = _normalize(arr.pop("X_raw"), arr["valid_mask"], mean, std)
        _save_npz(out_dir / f"{split}.npz", compress=compress, X=x, **arr)

    split_manifest = {
        "train": [d for d in splits["train"] if d in paths_by_date],
        "val": [d for d in splits["val"] if d in paths_by_date],
        "test": [d for d in splits["test"] if d in paths_by_date],
        "train_markets": int(split_arrays["train"]["y"].shape[0]),
        "val_markets": int(split_arrays["val"]["y"].shape[0]),
        "test_markets": int(split_arrays["test"]["y"].shape[0]),
    }
    norm = {
        "feature_names": features,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "fit_split": "train",
        "fit_valid_timesteps": int(split_arrays["train"]["valid_mask"].sum()),
    }
    (out_dir / "normalization.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
    (out_dir / "splits.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    return {"splits": split_manifest, "normalization": norm}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dates", default="auto", help="auto, all, comma list, or YYYY-MM-DD:YYYY-MM-DD ranges")
    ap.add_argument("--raw-roots", default="data/raw;D:/RawDataStorage")
    ap.add_argument("--feature-source", type=Path, default=DEFAULT_FEATURE_SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--feature-set", default="cex_ticker_min",
                    choices=("full", "pm_rtds_safe", "cex_oracle_gap", "cex_oracle_core", "cex_ticker_min"))
    ap.add_argument("--episode-seconds", type=int, default=300)
    ap.add_argument("--cadence-s", type=float, default=1.0)
    ap.add_argument("--cadence-ms", type=float, default=None,
                    help="optional alias for cadence; overrides --cadence-s")
    ap.add_argument("--rtds-latency-s", type=float, default=2.5)
    ap.add_argument("--max-pm-source-age-s", type=float, default=2.0)
    ap.add_argument("--max-cex-age-s", type=float, default=2.0)
    ap.add_argument("--max-rtds-source-age-s", type=float, default=60.0)
    ap.add_argument("--min-valid-steps", type=int, default=20)
    ap.add_argument("--min-valid-frac", type=float, default=0.05)
    ap.add_argument("--train-end", default="2026-05-13")
    ap.add_argument("--test-start", default="2026-05-14")
    ap.add_argument("--val-days", type=int, default=2)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 2))
    ap.add_argument("--max-markets", type=int, default=0)
    ap.add_argument("--compress", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--skip-splits", action="store_true",
                    help="write daily shards and manifest only; useful for partial/debug date ranges")
    ap.add_argument(
        "--allow-source-time-counterfactual",
        action="store_true",
        help=("required acknowledgement that this historical builder replays events at "
              "source time and cannot support causal/live performance claims"),
    )
    args = ap.parse_args()
    if not args.allow_source_time_counterfactual:
        ap.error(
            "refusing to build a non-causal dataset without "
            "--allow-source-time-counterfactual; a causal rebuild must order all "
            "sources by recorded receive/availability time"
        )
    if args.cadence_ms is not None:
        args.cadence_s = float(args.cadence_ms) / 1000.0

    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    daily_dir = out_dir / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_dir.mkdir(parents=True, exist_ok=True)

    roots = _raw_roots(args.raw_roots)
    require_cex = _uses_cex(args.feature_set)
    if args.dates.lower() in {"auto", "all"}:
        dates = _auto_dates(roots, require_coinbase=require_cex)
    else:
        filter_dates = _expand_date_filter(args.dates)
        dates = sorted(filter_dates) if filter_dates else []
    if not dates:
        raise SystemExit("no dates selected")

    feature_source = args.feature_source if args.feature_source.is_absolute() else ROOT / args.feature_source
    features = _load_feature_names(feature_source, args.feature_set)
    t_grid, ttc_grid = _episode_grid(args.episode_seconds, args.cadence_s)
    print(f"Building SOURCE-TIME COUNTERFACTUAL btc_5m_episodes_v1 dates={len(dates)} feature_set={args.feature_set} "
          f"features={len(features)} seq_len={len(t_grid)} workers={args.workers} roots={[str(r) for r in roots]}",
          flush=True)

    results = []
    t0 = time.time()
    for d in dates:
        stats = _build_one_date(
            d, roots=roots, out_daily=daily_dir, workers=args.workers,
            max_markets=args.max_markets, features=features, feature_set=args.feature_set,
            t_grid=t_grid, ttc_grid=ttc_grid, rtds_latency_s=args.rtds_latency_s,
            max_pm_source_age_s=args.max_pm_source_age_s,
            max_cex_age_s=args.max_cex_age_s if require_cex else None,
            max_rtds_source_age_s=args.max_rtds_source_age_s,
            min_valid_steps=args.min_valid_steps,
            min_valid_frac=args.min_valid_frac,
            rebuild=args.rebuild,
        )
        results.append(stats)
        print(f"  {d} {stats['status']}: episodes={stats.get('episodes', 0)} "
              f"present={stats.get('present_frac', 0):.2%} valid={stats.get('valid_frac', 0):.2%} "
              f"skip={stats.get('skips', {})} {stats.get('elapsed_s', 0)}s",
              flush=True)

    built_dates = [r["date"] for r in results if r.get("episodes", 0) > 0 and (daily_dir / f"{r['date']}.npz").exists()]
    if args.skip_splits:
        split_info = {"splits": {"daily_only": built_dates}, "normalization": None}
    else:
        split_info = _write_splits(out_dir, daily_dir, built_dates, args.train_end, args.test_start,
                                   args.val_days, features, args.compress)

    manifest = {
        "dataset_version": "btc_5m_episodes_v1",
        "output_dir": str(out_dir),
        "daily_dir": str(daily_dir),
        "raw_roots": [str(r) for r in roots],
        "feature_source": str(feature_source),
        "feature_set": args.feature_set,
        "feature_names": features,
        "quote_names": QUOTE_COLS,
        "audit_names": AUDIT_COLS,
        "require_cex": require_cex,
        "clock_policy": {
            "causal_availability": False,
            "classification": "source_time_counterfactual_not_live_achievable",
            "pm": "source_ts_ns",
            "rtds": "chainlink_ts_ns + rtds_latency_s",
            "cex": "source_ts_ns",
            "recv_ts": "audit_only",
            "warning": (
                "Do not use this dataset for causal, live-achievable, forward-test, "
                "or trading-performance claims. Rebuild all streams on recorded "
                "receive/availability time and retrain."
            ),
        },
        "freshness_policy": {
            "max_pm_source_age_s": args.max_pm_source_age_s,
            "max_cex_age_s": args.max_cex_age_s if require_cex else None,
            "max_rtds_source_age_s": args.max_rtds_source_age_s,
            "min_valid_steps": args.min_valid_steps,
            "min_valid_frac": args.min_valid_frac,
        },
        "episode": {
            "episode_seconds": args.episode_seconds,
            "cadence_s": args.cadence_s,
            "seq_len": len(t_grid),
            "t_s": t_grid.tolist(),
            "ttc_s": ttc_grid.tolist(),
        },
        "splits": split_info["splits"],
        "dates": results,
        "elapsed_s": round(time.time() - t0, 2),
    }
    (out_dir / "feature_names.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (out_dir / "quote_names.json").write_text(json.dumps(QUOTE_COLS, indent=2), encoding="utf-8")
    (out_dir / "audit_names.json").write_text(json.dumps(AUDIT_COLS, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(out_dir),
        "features": len(features),
        "seq_len": len(t_grid),
        "splits": split_info["splits"],
        "elapsed_s": manifest["elapsed_s"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
