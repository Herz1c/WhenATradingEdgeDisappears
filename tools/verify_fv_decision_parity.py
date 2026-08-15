"""Verify fair-value live ENTER decisions against an exact raw-data replay.

This intentionally does NOT need Polymarket resolution labels. For every ENTER
in logs/fair_value_bot/fv_decisions_<date>.jsonl it replays the raw recorder
streams (Polymarket L2 + RTDS + Coinbase) and emits a feature row at the exact
live snapshot_ts_ns. PASS means the backtest replay would take the same side at
that exact timestamp.

Usage:
  py -3 tools/verify_fv_decision_parity.py 2026-06-29 10:19
  py -3 tools/verify_fv_decision_parity.py 2026-06-29 10:19 --raw-root data/raw
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.build_fair_value_v2_source_time import (  # noqa: E402
    _best_dir,
    _empty_cex_day,
    _raw_roots,
    build_market,
    load_coinbase_day,
    load_rtds_day,
)

ART = Path(os.getenv("FV_ARTIFACTS_DIR", ROOT / "artifacts" / "fair_value_v2_source_time"))
NS = 1_000_000_000
BOOK_COHERENCE_TOL = 0.03


def _logit(p: float) -> float:
    p = min(1 - 1e-4, max(1e-4, float(p)))
    return math.log(p / (1 - p))


def _iso(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / NS, tz=dt.timezone.utc).isoformat(timespec="milliseconds")


def _parse_hhmm(date_s: str, hhmm: str | None) -> int:
    if not hhmm:
        return 0
    h, m = map(int, hhmm.split(":"))
    y, mo, d = map(int, date_s.split("-"))
    return int(dt.datetime(y, mo, d, h, m, tzinfo=dt.timezone.utc).timestamp() * NS)


def _predict(row: dict[str, Any], features: list[str], booster: lgb.Booster, calibrator) -> float:
    x = np.array([[float(row.get(c, np.nan)) for c in features]], dtype=float)
    raw = float(booster.predict(x, raw_score=True)[0])
    p = 1.0 / (1.0 + math.exp(-(_logit(float(row["implied_p_up"])) + raw)))
    return float(calibrator.transform([p])[0])


def _action(p: float, up_ask: float, dn_ask: float,
            ev_thr: float, price_lo: float, price_hi: float) -> tuple[str, float, float]:
    ev_up = p - up_ask
    ev_dn = (1.0 - p) - dn_ask
    if ev_up >= ev_thr and ev_up >= ev_dn and price_lo < up_ask < price_hi:
        return "UP", ev_up, ev_dn
    if ev_dn >= ev_thr and price_lo < dn_ask < price_hi:
        return "DOWN", ev_up, ev_dn
    return "SKIP", ev_up, ev_dn


def _book_quality_reason(row: dict[str, Any]) -> str | None:
    up_mid = float(row.get("up_mid") or 0.0)
    dn_mid = float(row.get("down_mid") or 0.0)
    if up_mid <= 0.0 or dn_mid <= 0.0:
        return "one_sided_book"
    mid_sum = up_mid + dn_mid
    if abs(1.0 - mid_sum) > BOOK_COHERENCE_TOL:
        return f"incoherent_mid_sum:{mid_sum:.3f}"
    if float(row.get("up_book_evts_5s") or 0.0) <= 0.0 or float(row.get("down_book_evts_5s") or 0.0) <= 0.0:
        return "stale_book_side_inactive_5s"
    return None


def _expected_decision(row: dict[str, Any], action: str, *,
                       max_exec_pm_lag_s: float,
                       use_book_quality: bool) -> tuple[str, str | None]:
    if action == "SKIP":
        return "SKIP_EDGE", None
    if use_book_quality:
        reason = _book_quality_reason(row)
        if reason is not None:
            return "SKIP_BOOK", reason
    lag = float(row.get("pm_source_lag_s") if row.get("pm_source_lag_s") is not None else float("nan"))
    if not math.isfinite(lag) or lag > max_exec_pm_lag_s:
        return "SKIP_EXEC_STALE_BOOK", f"pm_lag_s>{max_exec_pm_lag_s:.3f}"
    return "ENTER", None


def _source_gap_report(ts_ns: int, rtds, rtds_cl, cexbt, cexag) -> dict[str, dict[str, float | int | None]]:
    out = {}
    for name, arr in (
        ("rtds_recv", rtds[0]),
        ("rtds_chainlink", rtds_cl[0]),
        ("cex_bookticker", cexbt[0]),
        ("cex_trades", cexag[0]),
    ):
        a = np.asarray(arr)
        lo = ts_ns - 10 * NS
        hi = ts_ns + 10 * NS
        before = a[a <= ts_ns]
        after = a[a >= ts_ns]
        out[name] = {
            "n_pm10s": int(((a >= lo) & (a <= hi)).sum()),
            "age_before_s": (round(float((ts_ns - before[-1]) / NS), 3) if before.size else None),
            "next_after_s": (round(float((after[0] - ts_ns) / NS), 3) if after.size else None),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", help="UTC date, e.g. 2026-06-29")
    ap.add_argument("since_hhmm", nargs="?", default=None, help="optional UTC HH:MM filter")
    ap.add_argument("--raw-root", default="data/raw", help="raw recorder root; semicolon-separated roots allowed")
    ap.add_argument("--artifacts-dir", default=str(ART), help="fair-value model artifact directory")
    ap.add_argument("--decisions-dir", default=str(ROOT / "logs" / "fair_value_bot"))
    ap.add_argument("--cex", default="coinbase", choices=["coinbase"])
    ap.add_argument("--require-cex", action=argparse.BooleanOptionalAction, default=None,
                    help="default is inferred from artifacts/feature_policy.json")
    ap.add_argument("--workers", type=int, default=1,
                    help="CEX loader workers; keep 1 on Windows when debugging")
    ap.add_argument("--ev-thr", type=float, default=0.10)
    ap.add_argument("--price-lo", type=float, default=0.30)
    ap.add_argument("--price-hi", type=float, default=0.70)
    ap.add_argument("--rtds-latency-s", type=float, default=2.5)
    ap.add_argument("--decisions", default="ENTER",
                    help="comma-separated live decisions to verify, e.g. ENTER,SKIP_BOOK,SKIP_EXEC_STALE_BOOK")
    ap.add_argument("--include-blocked", action="store_true",
                    help="shortcut for --decisions ENTER,SKIP_BOOK,SKIP_EXEC_STALE_BOOK")
    ap.add_argument("--max-exec-pm-lag-s", type=float, default=1.0)
    ap.add_argument("--no-book-quality", action="store_true")
    ap.add_argument("--max-mismatches", type=int, default=20)
    ap.add_argument("--raw-stale-pm-lag-s", type=float, default=5.0,
                    help="if the replay's PM book is older than this while the live bot's was "
                         "fresh, classify the disagreement as RAW_RECORDER_STALE (recorder data "
                         "invalid at decision time), not as a model/action mismatch")
    args = ap.parse_args()
    art = Path(args.artifacts_dir)
    wanted_decisions = {x.strip() for x in args.decisions.split(",") if x.strip()}
    if args.include_blocked:
        wanted_decisions.update({"ENTER", "SKIP_BOOK", "SKIP_EXEC_STALE_BOOK"})

    t0 = _parse_hhmm(args.date, args.since_hhmm)
    decisions_path = Path(args.decisions_dir) / f"fv_decisions_{args.date}.jsonl"
    if not decisions_path.exists():
        print(f"missing {decisions_path}")
        return 2

    rows = []
    with decisions_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("decision") not in wanted_decisions:
                continue
            ts = int(rec.get("snapshot_ts_ns") or 0)
            if ts >= t0:
                rows.append(rec)
    if not rows:
        print("no ENTER decisions in requested window")
        return 0

    features = json.loads((art / "features.json").read_text())
    policy_path = art / "feature_policy.json"
    feature_policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    require_cex = bool(feature_policy.get("require_cex", True)) if args.require_cex is None else bool(args.require_cex)
    booster = lgb.Booster(model_file=str(art / "model.txt"))
    calibrator = joblib.load(art / "calibrator.pkl")

    roots = _raw_roots(args.raw_root)
    print(f"loading raw sources date={args.date} raw_root={args.raw_root} artifacts={art} "
          f"feature_set={feature_policy.get('feature_set', 'legacy_full')} require_cex={require_cex} ...")
    rtds, rtds_cl, rtds_stats = load_rtds_day(args.date, roots, args.rtds_latency_s)
    if require_cex:
        cexbt, cexag, cex_stats = load_coinbase_day(args.date, roots)
    else:
        cexbt, cexag, cex_stats = _empty_cex_day()
    print(f"source counts: rtds={len(rtds[0])} cex_bookticker={len(cexbt[0])} cex_trades={len(cexag[0])}")
    print(f"source roots: rtds={rtds_stats.get('root')} coinbase={cex_stats.get('root')}")

    by_slug: dict[str, list[dict]] = defaultdict(list)
    for rec in rows:
        by_slug[str(rec["market_slug"])].append(rec)

    pm_dir = _best_dir(roots, "pm", args.date)
    if pm_dir is None:
        print(f"missing raw PM L2 dir date={args.date} roots={[str(x) for x in roots]}")
        return 2
    replay_rows: dict[str, dict[int, dict]] = {}
    missing_raw = []
    for slug, recs in by_slug.items():
        paths = sorted(pm_dir.glob(f"*{slug}*.l2.jsonl.zst"))
        if not paths:
            missing_raw.append(slug)
            replay_rows[slug] = {}
            continue
        open_s = int(slug.split("-")[-1])
        res = {
            slug: {
                "open_s": open_s,
                "close_s": open_s + 300,
                "resolved_up": 0,
                "open_price": None,
                "close_price": None,
            }
        }
        emit_ns = [int(r["snapshot_ts_ns"]) for r in recs]
        built, _audit = build_market(paths[0], res, rtds, rtds_cl, cexbt, cexag,
                                     args.rtds_latency_s, emit_ns=emit_ns,
                                     require_cex=require_cex)
        replay_rows[slug] = {int(r["now_ns"]): r for r in built}

    ok = []
    mismatches = []
    raw_stale = []
    unverified = []
    for rec in rows:
        slug = str(rec["market_slug"])
        ts = int(rec["snapshot_ts_ns"])
        live_decision = str(rec.get("decision"))
        live_side = str(rec.get("side")) if rec.get("side") is not None else None
        row = replay_rows.get(slug, {}).get(ts)
        if row is None:
            unverified.append((rec, _source_gap_report(ts, rtds, rtds_cl, cexbt, cexag)))
            continue
        p_bt = _predict(row, features, booster, calibrator)
        bt_action, evu, evd = _action(
            p_bt,
            float(row["up_best_ask"]),
            float(row["down_best_ask"]),
            args.ev_thr,
            args.price_lo,
            args.price_hi,
        )
        expected_decision, expected_reason = _expected_decision(
            row, bt_action,
            max_exec_pm_lag_s=args.max_exec_pm_lag_s,
            use_book_quality=not args.no_book_quality,
        )
        item = {
            "iso": rec.get("iso") or _iso(ts),
            "slug": slug,
            "live_decision": live_decision,
            "expected_decision": expected_decision,
            "live_side": live_side,
            "bt_side": bt_action if bt_action != "SKIP" else None,
            "live_reason": rec.get("book_reason") or rec.get("exec_reason"),
            "expected_reason": expected_reason,
            "live_p": float(rec.get("p_model") or np.nan),
            "bt_p": p_bt,
            "p_diff": abs(float(rec.get("p_model") or np.nan) - p_bt),
            "live_fill": float(rec.get("fill_quote") or np.nan),
            "bt_up_ask": float(row["up_best_ask"]),
            "bt_down_ask": float(row["down_best_ask"]),
            "bt_ev_up": evu,
            "bt_ev_down": evd,
            "bt_pm_source_lag_s": float(row.get("pm_source_lag_s") or np.nan),
            "bt_pm_delivery_lag_s": float(row.get("pm_delivery_lag_s") or np.nan),
            "live_pm_source_lag_s": (float(rec["pm_source_lag_s"])
                                     if rec.get("pm_source_lag_s") is not None else None),
            "bt_up_book_evts_5s": float(row.get("up_book_evts_5s") or 0.0),
            "bt_down_book_evts_5s": float(row.get("down_book_evts_5s") or 0.0),
            "parity_strict": rec.get("parity_strict"),
            "feature_source": rec.get("feature_source"),
            "rtds_source": rec.get("rtds_source"),
            "cex_source": rec.get("cex_source"),
        }
        reason_ok = True
        if expected_decision == "SKIP_BOOK":
            reason_ok = str(rec.get("book_reason")) == str(expected_reason)
        elif expected_decision == "SKIP_EXEC_STALE_BOOK":
            reason_ok = str(rec.get("exec_reason", "")).startswith(str(expected_reason))
        side_ok = expected_decision.startswith("SKIP") or bt_action == live_side
        if live_decision == expected_decision and side_ok and reason_ok:
            ok.append(item)
            continue
        # Disagreement. If the raw recorder file simply has no fresh PM data at
        # the decision instant while the live bot demonstrably had a fresh
        # direct book, the comparison is invalid: the recorder is at fault,
        # not the model/action logic. Count it separately so recorder health
        # is visible and real parity breaks are not diluted.
        live_pm_lag = item["live_pm_source_lag_s"]
        bt_src_lag = item["bt_pm_source_lag_s"]
        bt_dlv_lag = item["bt_pm_delivery_lag_s"]
        recorder_stale = (
            (math.isfinite(bt_src_lag) and bt_src_lag > args.raw_stale_pm_lag_s)
            or (math.isfinite(bt_dlv_lag) and bt_dlv_lag > args.raw_stale_pm_lag_s)
            or not math.isfinite(bt_src_lag)
        )
        live_fresh = live_pm_lag is not None and live_pm_lag <= args.max_exec_pm_lag_s
        if (live_decision == "ENTER"
                and expected_decision in {"SKIP_BOOK", "SKIP_EXEC_STALE_BOOK"}
                and recorder_stale and live_fresh):
            item["classification"] = "RAW_RECORDER_STALE"
            raw_stale.append(item)
        else:
            mismatches.append(item)

    suffix = f" since {args.since_hhmm} UTC" if args.since_hhmm else ""
    verified = ok + mismatches + raw_stale
    comparable = ok + mismatches
    print(f"\nDecisions{suffix}: {len(rows)}  wanted={sorted(wanted_decisions)}")
    print(f"verified exact: {len(verified)}/{len(rows)}")
    print(f"decision agreement (raw-stale excluded): {len(ok)}/{len(comparable)}"
          if comparable else "decision agreement: n/a (all verified rows were raw-stale)")
    if raw_stale:
        print(f"raw recorder stale: {len(raw_stale)}/{len(verified)} "
              f"(recorder had no fresh PM data at decision time; comparison invalid, "
              f"NOT a model/action mismatch)")
    if ok:
        diffs = np.array([x["p_diff"] for x in comparable], dtype=float)
        print(f"p_diff verified: mean={np.nanmean(diffs):.4f} p95={np.nanquantile(diffs, 0.95):.4f}")
        from collections import Counter
        print(f"live decision counts: {dict(Counter(x['live_decision'] for x in verified))}")
        print(f"expected decision counts: {dict(Counter(x['expected_decision'] for x in verified))}")

    if missing_raw:
        print(f"\nmissing raw L2 files: {missing_raw}")

    if mismatches:
        print("\nMISMATCHES")
        for x in mismatches[:max(0, args.max_mismatches)]:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()},
                             separators=(",", ":")))
        if len(mismatches) > args.max_mismatches:
            print(f"... {len(mismatches) - args.max_mismatches} more mismatch(es) omitted")

    if raw_stale:
        print("\nRAW RECORDER STALE (comparison invalid — recorder, not model)")
        for x in raw_stale[:max(0, args.max_mismatches)]:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()},
                             separators=(",", ":")))
        if len(raw_stale) > args.max_mismatches:
            print(f"... {len(raw_stale) - args.max_mismatches} more raw-stale row(s) omitted")

    if unverified:
        print("\nUNVERIFIED")
        for rec, gaps in unverified:
            brief = {
                "iso": rec.get("iso") or _iso(int(rec["snapshot_ts_ns"])),
                "slug": rec.get("market_slug"),
                "side": rec.get("side"),
                "live_p": rec.get("p_model"),
                "fill": rec.get("fill_quote"),
                "reason": "exact raw replay emitted no feature row",
                "source_gaps": gaps,
                "parity_strict": rec.get("parity_strict"),
                "feature_source": rec.get("feature_source"),
                "rtds_source": rec.get("rtds_source"),
                "cex_source": rec.get("cex_source"),
            }
            print(json.dumps(brief, separators=(",", ":")))

    if mismatches or unverified:
        print("\nFAIL: at least one decision is not exactly reproducible from recorder raw data.")
        return 1
    if raw_stale:
        print("\nRAW-STALE: no model/action mismatches, but the recorder had no fresh PM data at "
              f"{len(raw_stale)} decision instant(s). Fix/verify the recorder before trusting parity.")
        return 3
    print("\nPASS: every requested decision is exactly reproducible from recorder raw data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
