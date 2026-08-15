"""Policy grid search for the v2 retrained TCNs — snooping-safe protocol (Phase 5).

The v1 lock died in the audit because its config was picked directly on the
evaluation data. This search pre-registers the protocol instead:

  SELECTION window: v2 val+test days (2026-06-23..07-02, 10 days) — the model
      saw val only for early stopping; policies are selected HERE.
  HOLDOUT: daily shards 2026-07-03+ — evaluated ONCE for the configs chosen by
      the pre-registered rules below. The headline result is how the
      selection-rule winners perform OOS, never the best OOS row.

Pre-registered selection rules (all reported, chosen before looking at holdout):
  R1 top-5 by score = total_pnl / max(1, |max_daily_drawdown|), trades >= 15
  R2 top-5 by total_pnl, trades >= 15
  R3 Pareto front of (total_pnl, -max_drawdown), trades >= 15 (risk/return
     efficient set — answers my "optimal risk profile AND pnl" question)

Grid: ev_min x beta x ttc window x price band, hold-to-resolution (active
exits already shown dominated), buy delay 2 s, slippage cap 0.03, 5.1 shares.
Betas extend above 1.25 because v2 deltas are better calibrated => smaller;
EV floors extend below 0.075 for the same reason.

Models: c96/b8 seed11 (best transfer in v2_seed_policy_eval) and the 5-seed
c64/b7 mean-delta ensemble.

Output: artifacts/tcn_v2_policy_search/
  grid_<model>_selection.csv, grid_<model>_holdout.csv, report.json, report.md
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from backtest.episode_strategy_backtester import (  # noqa: E402
    HOLD, EpisodeData, StrategySpec, find_entries, load_tcn_model, logit_np,
    predict_delta, simulate_trade,
)

V2 = ROOT / "data" / "datasets" / "btc_5m_episodes_v2_200ms"
DAILY = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms" / "daily"
OUT_DIR = ROOT / "artifacts" / "tcn_v2_policy_search"
HOLDOUT_START = "2026-07-03"
MIN_TRADES_SEL = 15
SHARES = 5.1

ENSEMBLE_DIRS = [ROOT / "artifacts" / f"tcn_v2_c64_b7_ttc15_150_seed{s}"
                 for s in (11, 7, 23, 42, 101)]
PROBE_DIR = ROOT / "artifacts" / "tcn_v2_c96_b8_ttc15_150_seed11"

EVS = [0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125]
BETAS = [0.5, 1.0, 1.5, 2.0]
TTCS = [(15.0, 50.0), (30.0, 75.0), (50.0, 75.0), (50.0, 90.0), (15.0, 90.0),
        (75.0, 120.0), (75.0, 150.0), (90.0, 150.0), (15.0, 150.0), (120.0, 180.0)]
PRICES = [(0.10, 0.90), (0.20, 0.80)]


def _norm() -> tuple[np.ndarray, np.ndarray]:
    n = json.loads((V2 / "normalization.json").read_text(encoding="utf-8"))
    mean = np.asarray(n["mean"], dtype=np.float32)
    std = np.asarray(n["std"], dtype=np.float32)
    std[std <= 1e-12] = 1.0
    return mean, std


KEYS = ("valid_mask", "y", "p_market", "quotes", "now_ns", "market_slug", "date", "open_s")


def load_selection_arrays() -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {}
    for split in ("val", "test"):
        z = np.load(V2 / f"{split}.npz", allow_pickle=False)
        for k in ("X",) + KEYS:
            parts.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


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
        parts.setdefault("X", []).append(X)
        for k in KEYS:
            parts.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def to_episode_data(arrs: dict[str, np.ndarray], split: str, delta: np.ndarray) -> EpisodeData:
    p_market = arrs["p_market"].astype(np.float32)
    p_market[~np.isfinite(p_market)] = 0.5
    return EpisodeData(
        split=split, quotes=arrs["quotes"].astype(np.float32),
        valid=arrs["valid_mask"].astype(bool), y=arrs["y"].astype(np.int8),
        date=arrs["date"], market_slug=arrs["market_slug"],
        now_ns=arrs["now_ns"].astype(np.int64), open_s=arrs["open_s"].astype(np.int64),
        base_logit=logit_np(p_market), delta=delta, cadence_s=0.2,
    )


def model_deltas(arrs: dict[str, np.ndarray], model_dirs: list[Path]) -> np.ndarray:
    X = np.concatenate([arrs["X"], arrs["valid_mask"][:, :, None].astype(np.float32)], axis=2)
    deltas = []
    for mdir in model_dirs:
        model, _rep, device = load_tcn_model(mdir, n_features=X.shape[2])
        deltas.append(predict_delta(model, X, batch_size=64, device=device))
    return np.mean(deltas, axis=0)


def make_specs() -> list[StrategySpec]:
    specs = []
    for ev in EVS:
        for beta in BETAS:
            for lo, hi in TTCS:
                for plo, phi in PRICES:
                    specs.append(StrategySpec(
                        strategy_id=f"b{beta:g}_ttc{lo:g}-{hi:g}_ev{ev:g}_px{plo:g}-{phi:g}",
                        beta=beta, ttc_min=lo, ttc_max=hi, ev_min=ev,
                        price_lo=plo, price_hi=phi, shares=SHARES))
    return specs


def run_config(data: EpisodeData, spec: StrategySpec) -> dict:
    entries = find_entries(data, spec)
    trades = [simulate_trade(data, e, spec, HOLD) for e in entries]
    by_day: dict[str, float] = {}
    for t in trades:
        by_day[t["date"]] = by_day.get(t["date"], 0.0) + t["pnl"]
    day_vals = np.asarray([by_day[k] for k in sorted(by_day)], dtype=np.float64)
    equity = np.cumsum(day_vals) if day_vals.size else day_vals
    dd = equity - np.maximum.accumulate(equity) if equity.size else equity
    pnl = float(sum(t["pnl"] for t in trades))
    max_dd = float(dd.min()) if dd.size else 0.0
    wins = sum(1 for t in trades if t["resolved_win"])
    return {
        "config": spec.strategy_id, "beta": spec.beta, "ttc_min": spec.ttc_min,
        "ttc_max": spec.ttc_max, "ev_min": spec.ev_min,
        "price_lo": spec.price_lo, "price_hi": spec.price_hi,
        "trades": len(trades),
        "pnl": round(pnl, 2),
        "avg_pnl": round(pnl / len(trades), 4) if trades else 0.0,
        "win_rate": round(wins / len(trades), 3) if trades else None,
        "active_days": int(len(by_day)),
        "positive_days": int(sum(1 for v in by_day.values() if v > 0)),
        "worst_day": round(float(day_vals.min()), 2) if day_vals.size else 0.0,
        "max_dd": round(max_dd, 2),
        "score": round(pnl / max(1.0, abs(max_dd)), 3),
    }


def pareto_front(rows: list[dict]) -> list[dict]:
    """Non-dominated set on (pnl max, |max_dd| min)."""
    front = []
    for r in rows:
        dominated = any(
            (o["pnl"] >= r["pnl"] and abs(o["max_dd"]) <= abs(r["max_dd"])
             and (o["pnl"] > r["pnl"] or abs(o["max_dd"]) < abs(r["max_dd"])))
            for o in rows)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: -r["pnl"])


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    import torch
    torch.set_num_threads(8)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sel_arrs = load_selection_arrays()
    hold_arrs = load_holdout_arrays()
    sel_days = sorted(set(map(str, sel_arrs["date"])))
    hold_days = sorted(set(map(str, hold_arrs["date"])))
    print(f"selection: {sel_arrs['y'].shape[0]} mkts {sel_days[0]}..{sel_days[-1]} ({len(sel_days)}d)")
    print(f"holdout:   {hold_arrs['y'].shape[0]} mkts {hold_days[0]}..{hold_days[-1]} ({len(hold_days)}d)")

    models = {
        "c96b8": [PROBE_DIR],
        "ens_c64b7": [d for d in ENSEMBLE_DIRS if d.exists()],
    }
    specs = make_specs()
    print(f"grid: {len(specs)} configs x {len(models)} models")

    report: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "protocol": "select on 06-23..07-02, single-shot holdout eval of "
                                "pre-registered rules R1/R2/R3; headline = OOS of "
                                "selection winners",
                    "selection_days": sel_days, "holdout_days": hold_days,
                    "grid_size": len(specs), "min_trades_sel": MIN_TRADES_SEL,
                    "models": {}}

    for mname, mdirs in models.items():
        print(f"== {mname}: forwarding deltas...", flush=True)
        sel_data = to_episode_data(sel_arrs, "selection", model_deltas(sel_arrs, mdirs))
        hold_data = to_episode_data(hold_arrs, "holdout", model_deltas(hold_arrs, mdirs))
        sel_rows, hold_rows = [], []
        for i, spec in enumerate(specs):
            sel_rows.append(run_config(sel_data, spec))
            hold_rows.append(run_config(hold_data, spec))
            if (i + 1) % 100 == 0:
                print(f"  {mname} {i + 1}/{len(specs)} ({time.time() - t0:.0f}s)", flush=True)
        write_csv(OUT_DIR / f"grid_{mname}_selection.csv", sel_rows)
        write_csv(OUT_DIR / f"grid_{mname}_holdout.csv", hold_rows)

        hold_by_cfg = {r["config"]: r for r in hold_rows}
        eligible = [r for r in sel_rows if r["trades"] >= MIN_TRADES_SEL]
        r1 = sorted(eligible, key=lambda r: (-r["score"], -r["pnl"]))[:5]
        r2 = sorted(eligible, key=lambda r: -r["pnl"])[:5]
        r3 = pareto_front(eligible)[:10]

        def attach_oos(rows):
            out = []
            for r in rows:
                h = hold_by_cfg[r["config"]]
                out.append({"config": r["config"],
                            "sel_trades": r["trades"], "sel_pnl": r["pnl"],
                            "sel_max_dd": r["max_dd"], "sel_score": r["score"],
                            "oos_trades": h["trades"], "oos_pnl": h["pnl"],
                            "oos_max_dd": h["max_dd"], "oos_worst_day": h["worst_day"],
                            "oos_positive_days": f"{h['positive_days']}/{h['active_days']}"})
            return out

        report["models"][mname] = {
            "eligible_configs": len(eligible),
            "R1_top_score": attach_oos(r1),
            "R2_top_pnl": attach_oos(r2),
            "R3_pareto": attach_oos(r3),
        }
        for rule in ("R1_top_score", "R2_top_pnl", "R3_pareto"):
            rows = report["models"][mname][rule]
            tot = sum(x["oos_pnl"] for x in rows)
            print(f"  {mname} {rule}: mean OOS pnl of picks = "
                  f"{tot / max(1, len(rows)):+.2f} over {len(rows)} configs")

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    lines = ["# v2 policy grid search — selection vs holdout", "",
             f"protocol: {report['protocol']}", ""]
    for mname, m in report["models"].items():
        lines += [f"## {mname} (eligible {m['eligible_configs']})", ""]
        for rule in ("R1_top_score", "R2_top_pnl", "R3_pareto"):
            lines += [f"### {rule}", "",
                      "| config | sel trades | sel pnl | sel DD | OOS trades | OOS pnl | OOS DD | OOS posD |",
                      "|---|---:|---:|---:|---:|---:|---:|---|"]
            for x in m[rule]:
                lines.append(f"| {x['config']} | {x['sel_trades']} | {x['sel_pnl']:+.2f} | "
                             f"{x['sel_max_dd']:.2f} | {x['oos_trades']} | {x['oos_pnl']:+.2f} | "
                             f"{x['oos_max_dd']:.2f} | {x['oos_positive_days']} |")
            lines.append("")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {OUT_DIR} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
