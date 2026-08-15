"""Verify TCN shadow decisions against a source-time daily episode replay.

This compares live shadow logs from src/live_bot/tcn_shadow_bot.py with a daily
episode shard built by tools/build_btc_5m_episode_dataset.py.  It is the first
pass parity check: model probability/action/side/strategy_id at the logged
market+step must reproduce from raw source-time replay.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from train_episode_tcn import ResidualTCN, _logit_np  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms"
DEFAULT_TCN_ART = ROOT / "artifacts" / "btc_5m_episode_tcn_c64_b7_cal_ttc15_90"
DEFAULT_LOCK = ROOT / "artifacts" / "strategy_locks" / "tcn_double_strategy_v1_lock.json"
DEFAULT_DECISIONS = ROOT / "logs" / "tcn_shadow_bot"
DEFAULT_OUT = ROOT / "artifacts" / "tcn_shadow_parity"

UP_BID = 0
UP_ASK = 1
DN_BID = 2
DN_ASK = 3


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    if isinstance(x, np.ndarray):
        y = np.clip(x.astype(np.float64, copy=False), -50.0, 50.0)
        return (1.0 / (1.0 + np.exp(-y))).astype(np.float32, copy=False)
    y = max(-50.0, min(50.0, float(x)))
    return 1.0 / (1.0 + math.exp(-y))


def load_model(tcn_dir: Path, n_features: int, device: torch.device) -> ResidualTCN:
    report = json.loads((tcn_dir / "tcn_report.json").read_text(encoding="utf-8"))
    cfg = report["model"]
    model = ResidualTCN(
        n_features=n_features,
        channels=int(cfg["channels"]),
        blocks=int(cfg["blocks"]),
        kernel_size=int(cfg["kernel_size"]),
        dropout=0.0,
        residual_scale=float(cfg.get("residual_scale", 0.5)),
    ).to(device)
    model.load_state_dict(torch.load(tcn_dir / "model.pt", map_location=device))
    model.eval()
    return model


@torch.no_grad()
def predict_delta(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    outs: list[np.ndarray] = []
    for i in range(0, x.shape[0], batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).to(device)
        outs.append(model(xb).detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outs, axis=0)


def normalize_daily(daily: np.lib.npyio.NpzFile, norm: dict[str, Any]) -> np.ndarray:
    if "X_raw" not in daily.files:
        if "X" in daily.files:
            return daily["X"].astype(np.float32, copy=False)
        raise ValueError("daily shard must contain X_raw or X")
    x = daily["X_raw"].astype(np.float32, copy=False)
    mask = daily["valid_mask"].astype(bool, copy=False)
    mean = np.asarray(norm["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(norm["std"], dtype=np.float32).reshape(1, 1, -1)
    std[std <= 1e-12] = 1.0
    y = (x - mean) / std
    y[~np.isfinite(y)] = 0.0
    y[~mask] = 0.0
    return y.astype(np.float32, copy=False)


def load_strategies(lock_path: Path) -> dict[str, dict[str, Any]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    out = {}
    for row in lock["strategies"]:
        if not row.get("enabled", True):
            continue
        hours = row.get("utc_hours_allowed")
        out[str(row["strategy_id"])] = {
            "beta": float(row["beta"]),
            "ev_min": float(row["ev_min"]),
            "price_lo": float(row["price_min_exclusive"]),
            "price_hi": float(row["price_max_exclusive"]),
            "ttc_min": float(row["ttc_min_s_exclusive"]),
            "ttc_max": float(row["ttc_max_s_inclusive"]),
            "utc_hours": None if hours == "all" else {int(x) for x in hours},
        }
    return out


def book_reason(raw: np.ndarray, idx: dict[str, int]) -> str | None:
    up_mid = float(raw[idx["up_mid"]])
    dn_mid = float(raw[idx["down_mid"]])
    if up_mid <= 0.0 or dn_mid <= 0.0:
        return "one_sided_book"
    mid_sum = up_mid + dn_mid
    if abs(1.0 - mid_sum) > 0.03:
        return f"incoherent_mid_sum:{mid_sum:.3f}"
    if float(raw[idx["up_book_evts_5s"]]) <= 0.0 or float(raw[idx["down_book_evts_5s"]]) <= 0.0:
        return "stale_book_side_inactive_5s"
    return None


def expected_action(
    *,
    raw: np.ndarray,
    quotes: np.ndarray,
    valid: bool,
    p_market: float,
    delta: float,
    strategy: dict[str, Any],
    idx: dict[str, int],
    ev_margin: float = 0.0,
) -> dict[str, Any]:
    if not valid:
        return {"decision": "SKIP_SOURCE", "side": None, "p_strategy": None, "ev": None}
    p = float(sigmoid(float(_logit_np(np.asarray([p_market], dtype=np.float32))[0]) + strategy["beta"] * delta))
    # Quotes are stored in float32 in the episode shard.  Without rounding,
    # an exact 20c book level can come back as 0.20000000298 and incorrectly
    # pass a strict >0.20 strategy gate that live rejects.
    up_ask = round(float(quotes[UP_ASK]), 4)
    dn_ask = round(float(quotes[DN_ASK]), 4)
    ev_up = p - up_ask
    ev_dn = (1.0 - p) - dn_ask
    ev_gate = float(strategy["ev_min"]) + float(ev_margin)
    side = None
    fill = None
    ev = None
    if ev_up >= ev_gate and ev_up >= ev_dn and strategy["price_lo"] < up_ask < strategy["price_hi"]:
        side, fill, ev = "UP", up_ask, ev_up
    elif ev_dn >= ev_gate and ev_dn > ev_up and strategy["price_lo"] < dn_ask < strategy["price_hi"]:
        side, fill, ev = "DOWN", dn_ask, ev_dn
    if side is None:
        return {"decision": "SKIP_NO_EDGE", "side": None, "p_strategy": p, "ev": None}
    bad = book_reason(raw, idx)
    if bad is not None:
        return {"decision": "SKIP_BOOK", "side": side, "p_strategy": p, "ev": ev, "book_reason": bad}
    return {"decision": "ENTER", "side": side, "p_strategy": p, "fill_quote": fill, "ev": ev}


def strategy_active(strategy: dict[str, Any], now_ns: int, ttc_s: float) -> bool:
    if not (strategy["ttc_min"] < float(ttc_s) <= strategy["ttc_max"]):
        return False
    hours = strategy.get("utc_hours")
    if hours is None:
        return True
    from datetime import UTC, datetime
    return datetime.fromtimestamp(int(now_ns) / 1e9, tz=UTC).hour in hours


def load_decision_log(path: Path, date: str) -> list[dict[str, Any]]:
    f = path / f"tcn_decisions_{date}.jsonl"
    if not f.exists():
        raise SystemExit(f"missing decisions file: {f}")
    out = []
    with f.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            out.append(rec)
    return out


def comparable_decisions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        rec for rec in records
        if rec.get("strategy_id") and rec.get("market_slug") and rec.get("step_idx") is not None
    ]


def live_source_skip_steps(records: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
    out: set[tuple[str, str, int]] = set()
    for rec in records:
        if rec.get("decision") != "SKIP_SOURCE":
            continue
        slug = str(rec.get("market_slug") or "")
        strategy_id = str(rec.get("strategy_id") or "")
        raw_step = rec.get("step_idx")
        if raw_step is None:
            raw_step = (rec.get("audit") or {}).get("step_idx")
        if not slug or not strategy_id or raw_step is None:
            continue
        out.add((slug, strategy_id, int(raw_step)))
    return out


def expected_entries_in_window(
    *,
    daily: np.lib.npyio.NpzFile,
    delta: np.ndarray,
    strategies: dict[str, dict[str, Any]],
    idx: dict[str, int],
    window_min_ns: int,
    window_max_ns: int,
    blocked_steps: set[tuple[str, str, int]] | None = None,
    ev_margin: float = 0.0,
) -> list[dict[str, Any]]:
    slugs = daily["market_slug"].astype(str)
    now_ns = daily["now_ns"].astype(np.int64, copy=False)
    close_s = daily["close_s"].astype(np.int64, copy=False)
    valid = daily["valid_mask"].astype(bool, copy=False)
    p_market = daily["p_market"].astype(np.float32, copy=False)
    quotes = daily["quotes"].astype(np.float32, copy=False)
    raw_x = daily["X_raw"].astype(np.float32, copy=False) if "X_raw" in daily.files else None
    blocked_steps = blocked_steps or set()
    out: list[dict[str, Any]] = []
    for ep, slug in enumerate(slugs.tolist()):
        if close_s[ep] <= 0:
            continue
        for strategy_id, strategy in strategies.items():
            for step in range(now_ns.shape[1]):
                ts = int(now_ns[ep, step])
                if ts <= 0 or ts < window_min_ns or ts > window_max_ns:
                    continue
                ttc_s = (int(close_s[ep]) * 1_000_000_000 - ts) / 1_000_000_000.0
                if not strategy_active(strategy, ts, ttc_s):
                    continue
                if (slug, strategy_id, int(step)) in blocked_steps:
                    continue
                raw = raw_x[ep, step] if raw_x is not None else np.zeros(max(idx.values()) + 1, dtype=np.float32)
                exp = expected_action(
                    raw=raw,
                    quotes=quotes[ep, step],
                    valid=bool(valid[ep, step]),
                    p_market=float(p_market[ep, step]),
                    delta=float(delta[ep, step]),
                    strategy=strategy,
                    idx=idx,
                    ev_margin=ev_margin,
                )
                if exp["decision"] != "ENTER":
                    continue
                out.append({
                    "market_slug": slug,
                    "strategy_id": strategy_id,
                    "step_idx": int(step),
                    "snapshot_ts_ns": ts,
                    "ttc_s": round(ttc_s, 3),
                    "side": exp.get("side"),
                    "p_strategy": exp.get("p_strategy"),
                    "fill_quote": exp.get("fill_quote"),
                    "ev": exp.get("ev"),
                })
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("date")
    ap.add_argument("--decisions-dir", type=Path, default=DEFAULT_DECISIONS)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--daily-npz", type=Path, default=None)
    ap.add_argument("--tcn-artifacts", type=Path, default=DEFAULT_TCN_ART)
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ev-margin", type=float, default=None,
                    help="Extra EV buffer above the strategy lock threshold; defaults to ev_margin in log or 0.")
    args = ap.parse_args()

    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    daily_path = args.daily_npz or (dataset / "daily" / f"{args.date}.npz")
    if not daily_path.exists():
        raise SystemExit(
            f"missing daily shard {daily_path}\n"
            "Build it first, e.g.\n"
            f"  py -3 tools/build_btc_5m_episode_dataset.py --dates {args.date} "
            "--raw-roots \"data/raw;D:/RawDataStorage\" --out data/datasets/btc_5m_episodes_v1_200ms "
            "--feature-set cex_ticker_min --cadence-ms 200 --skip-splits --rebuild"
        )
    decision_records = load_decision_log(args.decisions_dir if args.decisions_dir.is_absolute()
                                         else ROOT / args.decisions_dir, args.date)
    decisions = comparable_decisions(decision_records)
    if not decisions:
        raise SystemExit("no TCN decisions to verify")
    source_blocked_steps = live_source_skip_steps(decision_records)
    if args.ev_margin is None:
        ev_margin = 0.0
        for rec in decisions:
            if rec.get("ev_margin") is not None:
                ev_margin = float(rec["ev_margin"])
                break
    else:
        ev_margin = float(args.ev_margin)
    decision_ts = [
        int(r["snapshot_ts_ns"]) for r in decisions
        if r.get("snapshot_ts_ns") is not None
    ]
    window_min_ns = min(decision_ts) if decision_ts else 0
    window_max_ns = max(decision_ts) if decision_ts else 2**63 - 1

    norm = json.loads((dataset / "normalization.json").read_text(encoding="utf-8"))
    features = [str(x) for x in norm["feature_names"]]
    idx = {name: features.index(name) for name in (
        "up_mid", "down_mid", "up_book_evts_5s", "down_book_evts_5s",
    )}
    daily = np.load(daily_path, allow_pickle=False)
    x_norm = normalize_daily(daily, norm)
    valid = daily["valid_mask"].astype(bool, copy=False)
    x_model = np.concatenate([x_norm, valid[:, :, None].astype(np.float32)], axis=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.tcn_artifacts if args.tcn_artifacts.is_absolute() else ROOT / args.tcn_artifacts,
                       x_model.shape[-1], device)
    delta = predict_delta(model, x_model, args.batch_size, device)
    strategies = load_strategies(args.lock if args.lock.is_absolute() else ROOT / args.lock)
    slugs = daily["market_slug"].astype(str)
    slug_to_ep = {s: i for i, s in enumerate(slugs.tolist())}

    rows = []
    for rec in decisions:
        slug = str(rec["market_slug"])
        strategy_id = str(rec["strategy_id"])
        step = int(rec["step_idx"])
        ep = slug_to_ep.get(slug)
        if ep is None or strategy_id not in strategies or step < 0 or step >= valid.shape[1]:
            rows.append({"live": rec, "status": "missing_replay"})
            continue
        p_market = float(daily["p_market"][ep, step])
        exp = expected_action(
            raw=daily["X_raw"][ep, step] if "X_raw" in daily.files else np.zeros(len(features), dtype=np.float32),
            quotes=daily["quotes"][ep, step],
            valid=bool(valid[ep, step]),
            p_market=p_market,
            delta=float(delta[ep, step]),
            strategy=strategies[strategy_id],
            idx=idx,
            ev_margin=ev_margin,
        )
        live_decision = str(rec.get("decision"))
        live_side = rec.get("side")
        p_live = rec.get("p_strategy")
        p_exp = exp.get("p_strategy")
        rows.append({
            "market_slug": slug,
            "strategy_id": strategy_id,
            "step_idx": step,
            "live_decision": live_decision,
            "expected_decision": exp["decision"],
            "live_side": live_side,
            "expected_side": exp.get("side"),
            "decision_ok": live_decision == exp["decision"],
            "side_ok": (live_side == exp.get("side")) or exp["decision"].startswith("SKIP"),
            "p_live": p_live,
            "p_expected": p_exp,
            "p_abs_diff": None if p_live is None or p_exp is None else abs(float(p_live) - float(p_exp)),
            "live": rec,
            "expected": exp,
        })

    comparable = [r for r in rows if r.get("status") != "missing_replay"]
    decision_ok = [r for r in comparable if r["decision_ok"]]
    side_ok = [r for r in comparable if r["side_ok"]]
    p_pairs = [(float(r["p_live"]), float(r["p_expected"])) for r in comparable
               if r.get("p_live") is not None and r.get("p_expected") is not None]
    if len(p_pairs) >= 2:
        p_arr = np.asarray(p_pairs, dtype=np.float64)
        p_corr = float(np.corrcoef(p_arr[:, 0], p_arr[:, 1])[0, 1])
        p_mad = float(np.mean(np.abs(p_arr[:, 0] - p_arr[:, 1])))
        p_p95 = float(np.percentile(np.abs(p_arr[:, 0] - p_arr[:, 1]), 95))
    else:
        p_corr = p_mad = p_p95 = float("nan")
    enter_live = [r for r in comparable if r["live_decision"] == "ENTER"]
    enter_expected = [r for r in comparable if r["expected_decision"] == "ENTER"]
    enter_both = [r for r in comparable if r["live_decision"] == "ENTER" and r["expected_decision"] == "ENTER"]
    full_expected_entries = expected_entries_in_window(
        daily=daily,
        delta=delta,
        strategies=strategies,
        idx=idx,
        window_min_ns=window_min_ns,
        window_max_ns=window_max_ns,
        blocked_steps=source_blocked_steps,
        ev_margin=ev_margin,
    )
    live_enter_keys = {
        (str(r["live"]["market_slug"]), str(r["live"]["strategy_id"]))
        for r in enter_live
    }
    expected_enter_keys = {
        (str(r["market_slug"]), str(r["strategy_id"]))
        for r in full_expected_entries
    }
    live_enter_exact_keys = {
        (
            str(r["live"]["market_slug"]),
            str(r["live"]["strategy_id"]),
            int(r["live"]["step_idx"]),
            str(r["live"].get("side")),
        )
        for r in enter_live
    }
    expected_enter_exact_keys = {
        (
            str(r["market_slug"]),
            str(r["strategy_id"]),
            int(r["step_idx"]),
            str(r.get("side")),
        )
        for r in full_expected_entries
    }
    missed_expected_entries = [
        r for r in full_expected_entries
        if (str(r["market_slug"]), str(r["strategy_id"])) not in live_enter_keys
    ]
    extra_live_enter_keys = sorted(live_enter_keys - expected_enter_keys)
    missed_expected_exact_entries = [
        r for r in full_expected_entries
        if (
            str(r["market_slug"]),
            str(r["strategy_id"]),
            int(r["step_idx"]),
            str(r.get("side")),
        ) not in live_enter_exact_keys
    ]
    extra_live_enter_exact_keys = sorted(live_enter_exact_keys - expected_enter_exact_keys)
    summary = {
        "date": args.date,
        "window_min_ns": window_min_ns,
        "window_max_ns": window_max_ns,
        "decisions": len(rows),
        "comparable": len(comparable),
        "missing_replay": len(rows) - len(comparable),
        "live_source_blocked_steps": len(source_blocked_steps),
        "ev_margin": ev_margin,
        "decision_agreement": f"{len(decision_ok)}/{len(comparable)}",
        "decision_agreement_rate": len(decision_ok) / len(comparable) if comparable else 0.0,
        "side_agreement": f"{len(side_ok)}/{len(comparable)}",
        "side_agreement_rate": len(side_ok) / len(comparable) if comparable else 0.0,
        "live_counts": dict(Counter(r["live_decision"] for r in comparable)),
        "expected_counts": dict(Counter(r["expected_decision"] for r in comparable)),
        "live_enter": len(enter_live),
        "expected_enter": len(enter_expected),
        "enter_overlap": len(enter_both),
        "full_replay_expected_enter": len(full_expected_entries),
        "missed_expected_enter": len(missed_expected_entries),
        "extra_live_enter_strategy_keys": extra_live_enter_keys[:50],
        "extra_live_enter_strategy_keys_count": len(extra_live_enter_keys),
        "exact_enter_overlap": len(live_enter_exact_keys & expected_enter_exact_keys),
        "missed_expected_exact_enter": len(missed_expected_exact_entries),
        "extra_live_enter_exact_keys": extra_live_enter_exact_keys[:50],
        "extra_live_enter_exact_keys_count": len(extra_live_enter_exact_keys),
        "p_corr": p_corr,
        "mean_abs_p_diff": p_mad,
        "p95_abs_p_diff": p_p95,
    }
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"tcn_shadow_parity_{args.date}.json").write_text(
        json.dumps({
            "summary": summary,
            "rows": rows,
            "full_replay_expected_entries": full_expected_entries,
            "missed_expected_entries": missed_expected_entries,
            "missed_expected_exact_entries": missed_expected_exact_entries,
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if len(decision_ok) != len(comparable):
        print(f"FAIL: mismatches written to {out_dir / f'tcn_shadow_parity_{args.date}.json'}")
        return 2
    if missed_expected_exact_entries or extra_live_enter_exact_keys:
        print(f"FAIL: enter-set mismatch written to {out_dir / f'tcn_shadow_parity_{args.date}.json'}")
        return 2
    print("PASS: all comparable TCN shadow decisions reproduce from daily source-time replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
