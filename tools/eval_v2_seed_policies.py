"""Seed-variance evaluation of the v2 retrained TCNs on policy PnL (Phase 4.2/5).

Global Brier/AUC don't decide anything — the decision metric is the PnL of the
locked two-slot strategy driven by each seed's deltas, on model-OOS data:

  - v2 test split (2026-06-28..07-02): report-only during training
  - holdout daily shards (2026-07-03+): never touched by training/selection

Also evaluates the mean-delta ensemble of the c64/b7 seeds. Caveat printed in
the report: the locked EV/beta thresholds were tuned for the v1 model's delta
scale, so this measures signal TRANSFER of v2 models into the locked policy,
not a re-optimized v2 policy.

Output: artifacts/audit_v1/v2_seed_policy_eval.json (+ console table)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from backtest.episode_strategy_backtester import (  # noqa: E402
    HOLD, EpisodeData, load_lock, load_tcn_model, logit_np, predict_delta,
    run_locked_strategies,
)

V2 = ROOT / "data" / "datasets" / "btc_5m_episodes_v2_200ms"
DAILY = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms" / "daily"
LOCK = ROOT / "artifacts" / "strategy_locks" / "tcn_double_strategy_v1_lock.json"
OUT = ROOT / "artifacts" / "audit_v1" / "v2_seed_policy_eval.json"

SEED_DIRS = [ROOT / "artifacts" / f"tcn_v2_c64_b7_ttc15_150_seed{s}"
             for s in (11, 7, 23, 42, 101)]
PROBE_DIR = ROOT / "artifacts" / "tcn_v2_c96_b8_ttc15_150_seed11"
HOLDOUT_START = "2026-07-03"


def _norm() -> tuple[np.ndarray, np.ndarray]:
    n = json.loads((V2 / "normalization.json").read_text(encoding="utf-8"))
    mean = np.asarray(n["mean"], dtype=np.float32)
    std = np.asarray(n["std"], dtype=np.float32)
    std[std <= 1e-12] = 1.0
    return mean, std


def load_v2_test_arrays() -> dict[str, np.ndarray]:
    z = np.load(V2 / "test.npz", allow_pickle=False)
    return {k: z[k] for k in ("X", "valid_mask", "y", "p_market", "quotes",
                              "now_ns", "market_slug", "date", "open_s")}


def load_holdout_arrays() -> dict[str, np.ndarray]:
    mean, std = _norm()
    days = sorted(p.stem for p in DAILY.glob("*.npz") if p.stem >= HOLDOUT_START)
    parts: dict[str, list[np.ndarray]] = {}
    for d in days:
        z = np.load(DAILY / f"{d}.npz", allow_pickle=False)
        X = z["X_raw"].astype(np.float32)
        vm = z["valid_mask"].astype(bool)
        X = (X - mean) / std
        X[~vm] = 0.0
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        blob = {"X": X, "valid_mask": vm}
        for k in ("y", "p_market", "quotes", "now_ns", "market_slug", "date", "open_s"):
            blob[k] = z[k]
        for k, v in blob.items():
            parts.setdefault(k, []).append(v)
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def to_episode_data(arrs: dict[str, np.ndarray], split: str, delta: np.ndarray) -> EpisodeData:
    p_market = arrs["p_market"].astype(np.float32)
    p_market[~np.isfinite(p_market)] = 0.5
    return EpisodeData(
        split=split,
        quotes=arrs["quotes"].astype(np.float32),
        valid=arrs["valid_mask"].astype(bool),
        y=arrs["y"].astype(np.int8),
        date=arrs["date"],
        market_slug=arrs["market_slug"],
        now_ns=arrs["now_ns"].astype(np.int64),
        open_s=arrs["open_s"].astype(np.int64),
        base_logit=logit_np(p_market),
        delta=delta,
        cadence_s=0.2,
    )


def main() -> int:
    import torch
    torch.set_num_threads(8)
    specs, _lock = load_lock(LOCK)
    eval_sets = {
        "v2_test_0628_0702": load_v2_test_arrays(),
        "holdout_0703plus": load_holdout_arrays(),
    }
    for name, arrs in eval_sets.items():
        days = sorted(set(str(d) for d in arrs["date"]))
        print(f"{name}: {arrs['y'].shape[0]} markets, days {days[0]}..{days[-1]} ({len(days)}d)")

    model_dirs = [d for d in SEED_DIRS + [PROBE_DIR] if d.exists()]
    report: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "note": "locked v1 thresholds applied to v2 deltas — measures signal "
                            "transfer, not a re-optimized v2 policy",
                    "models": {}, "ensemble": {}}
    deltas_cache: dict[str, dict[str, np.ndarray]] = {}
    t0 = time.time()
    for mdir in model_dirs:
        mname = mdir.name
        deltas_cache[mname] = {}
        report["models"][mname] = {}
        for sname, arrs in eval_sets.items():
            X = np.concatenate([arrs["X"], arrs["valid_mask"][:, :, None].astype(np.float32)], axis=2)
            model, _rep, device = load_tcn_model(mdir, n_features=X.shape[2])
            delta = predict_delta(model, X, batch_size=64, device=device)
            deltas_cache[mname][sname] = delta
            data = to_episode_data(arrs, sname, delta)
            res = run_locked_strategies(data, specs, HOLD)
            per_slot = {sid: {"trades": s["summary"]["trades"],
                              "pnl": round(s["summary"]["total_pnl"], 2)}
                        for sid, s in res["per_strategy"].items()}
            comb = res["combined"]["summary"]
            report["models"][mname][sname] = {
                "combined_trades": comb["trades"],
                "combined_pnl": round(comb["total_pnl"], 2),
                "positive_days": f"{comb['positive_days']}/{comb['active_days']}",
                "per_slot": per_slot,
            }
            print(f"{mname} @ {sname}: {comb['trades']} trades, "
                  f"PnL {comb['total_pnl']:+.2f} ({time.time() - t0:.0f}s)", flush=True)

    # mean-delta ensemble of the c64/b7 seeds
    seed_names = [d.name for d in SEED_DIRS if d.exists()]
    for sname, arrs in eval_sets.items():
        ens_delta = np.mean([deltas_cache[m][sname] for m in seed_names], axis=0)
        data = to_episode_data(arrs, sname, ens_delta)
        res = run_locked_strategies(data, specs, HOLD)
        comb = res["combined"]["summary"]
        report["ensemble"][sname] = {
            "members": seed_names,
            "combined_trades": comb["trades"],
            "combined_pnl": round(comb["total_pnl"], 2),
            "positive_days": f"{comb['positive_days']}/{comb['active_days']}",
            "per_slot": {sid: {"trades": s["summary"]["trades"],
                               "pnl": round(s["summary"]["total_pnl"], 2)}
                         for sid, s in res["per_strategy"].items()},
        }
        print(f"ENSEMBLE @ {sname}: {comb['trades']} trades, PnL {comb['total_pnl']:+.2f}")

    # seed-variance summary on combined PnL
    for sname in eval_sets:
        pnls = [report["models"][m][sname]["combined_pnl"] for m in seed_names]
        report.setdefault("seed_variance", {})[sname] = {
            "pnls": pnls, "mean": round(float(np.mean(pnls)), 2),
            "std": round(float(np.std(pnls)), 2),
            "min": min(pnls), "max": max(pnls),
        }
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["seed_variance"], indent=1))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
