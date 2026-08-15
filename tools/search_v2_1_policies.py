"""Search v2.1: calibrated EV + EV-proportional sizing + portfolio (Plan v2.1, Phase 2).

Converts the v2 ensemble's calibration advantage into a PnL/risk-optimal
policy, with the discipline learned from the v1 audit:

  - probabilities: per-member Platt (val_platt_l2) -> mean prob, wrapped into
    an effective delta so the shared engine stays unchanged
  - EV floors: data-driven quantile grid from v2_transfer_diagnosis.json
  - sizing: shares = clip(k * (EV - floor), 0, s_max); trades are simulated
    once at 1 share and sizing variants are evaluated by scaling unit PnL
  - daily loss cap -10 USD simulated at evaluation (entries stop once realized
    PnL of already-closed markets breaches the cap that UTC day)
  - selection era = 2026-06-23..07-11 (everything <= 07-11 is spent); inside
    it a 2-fold mini walk-forward prunes configs that don't validate forward;
    final rank = R1 (pnl / max(1, |maxDD|)); the only honest OOS is the
    forward shadow from 07-12 on
  - portfolio: among R1 survivors pick 2-3 slots with low daily-PnL
    correlation maximizing combined PnL/DD; then size_scale so that
    P5(daily PnL) ~= -10 USD on live-coverage days (07-03+)

Output: artifacts/tcn_v2_1_policy_search/{grid.csv, report.json, report.md}
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
    predict_delta, sigmoid, simulate_trade,
)

DAILY = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms" / "daily"
V2_NORM = ROOT / "data" / "datasets" / "btc_5m_episodes_v2_200ms" / "normalization.json"
V2_DIRS = [ROOT / "artifacts" / f"tcn_v2_c64_b7_ttc15_150_seed{s}" for s in (11, 7, 23, 42, 101)]
DIAG = ROOT / "artifacts" / "audit_v1" / "v2_transfer_diagnosis.json"
OUT_DIR = ROOT / "artifacts" / "tcn_v2_1_policy_search"

SEL_START, SEL_END = "2026-06-23", "2026-07-11"
LIVE_COVERAGE_START = "2026-07-03"
TTCS = [(15.0, 50.0), (30.0, 75.0), (50.0, 75.0), (50.0, 90.0),
        (75.0, 120.0), (75.0, 150.0), (15.0, 90.0), (15.0, 150.0)]
PRICES = [(0.10, 0.90), (0.20, 0.80)]
BETAS = [0.75, 1.0, 1.25]
SIZE_K = [25.0, 50.0, 100.0, 200.0]
SIZE_MAX = [5.0, 10.0, 20.0]
DAILY_CAP = -10.0
MIN_TRADES = 15
WF_FOLDS = [(8, 11), (11, 15)]     # (select first N days, validate through day M)
PORTFOLIO_MAX_SLOTS = 3
PORTFOLIO_CORR_MAX = 0.5
BUDGET_P5 = -10.0


def load_norm(path: Path):
    n = json.loads(path.read_text(encoding="utf-8"))
    mean = np.asarray(n["mean"], dtype=np.float32)
    std = np.asarray(n["std"], dtype=np.float32)
    std[std <= 1e-12] = 1.0
    return mean, std


def load_shards() -> dict:
    days = sorted(p.stem for p in DAILY.glob("*.npz") if SEL_START <= p.stem <= SEL_END)
    parts: dict[str, list[np.ndarray]] = {}
    for d in days:
        z = np.load(DAILY / f"{d}.npz", allow_pickle=False)
        for k in ("X_raw", "valid_mask", "y", "p_market", "quotes", "now_ns",
                  "market_slug", "date", "open_s", "close_s"):
            parts.setdefault(k, []).append(z[k])
    out = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    out["days"] = days
    return out


def calibrated_effective_delta(arrs: dict) -> np.ndarray:
    """Per-member Platt -> mean prob -> effective delta vs market logit."""
    mean, std = load_norm(V2_NORM)
    X = arrs["X_raw"].astype(np.float32)
    vm = arrs["valid_mask"].astype(bool)
    Xn = (X - mean) / std
    Xn[~vm] = 0.0
    np.nan_to_num(Xn, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    Xn = np.concatenate([Xn, vm[:, :, None].astype(np.float32)], axis=2)
    p_market = arrs["p_market"].astype(np.float32)
    p_market[~np.isfinite(p_market)] = 0.5
    base_logit = logit_np(p_market)
    probs = []
    for mdir in V2_DIRS:
        model, rep, device = load_tcn_model(mdir, n_features=Xn.shape[2])
        delta = predict_delta(model, Xn, batch_size=64, device=device)
        cal = json.loads((mdir / "tcn_report.json").read_text(encoding="utf-8"))[
            "calibrators"]["val_platt_l2"]
        probs.append(sigmoid(float(cal["coef"]) * (base_logit + delta) + float(cal["intercept"])))
    p_ens = np.clip(np.mean(probs, axis=0), 1e-5, 1 - 1e-5)
    return (logit_np(p_ens) - base_logit).astype(np.float32)


def to_episode_data(arrs: dict, delta: np.ndarray) -> EpisodeData:
    p_market = arrs["p_market"].astype(np.float32)
    p_market[~np.isfinite(p_market)] = 0.5
    return EpisodeData(
        split="selection", quotes=arrs["quotes"].astype(np.float32),
        valid=arrs["valid_mask"].astype(bool), y=arrs["y"].astype(np.int8),
        date=arrs["date"], market_slug=arrs["market_slug"],
        now_ns=arrs["now_ns"].astype(np.int64), open_s=arrs["open_s"].astype(np.int64),
        base_logit=logit_np(p_market), delta=delta, cadence_s=0.2,
    )


def eval_sized(trades: list[dict], k: float, s_max: float, floor: float,
               days: list[str], daily_cap: float | None = DAILY_CAP) -> dict:
    """Scale unit-share trades by EV-proportional sizing + daily loss cap.
    Trades must be sorted by decision time. The cap uses realized PnL of
    markets already closed before each new entry (honest information set)."""
    by_day_real: dict[str, list[tuple[int, float]]] = {}   # close_ts -> pnl (sized)
    day_pnl = {d: 0.0 for d in days}
    n_taken = 0
    wins = 0
    for t in trades:
        d = t["date"]
        shares = min(s_max, k * max(0.0, t["entry_ev"] - floor))
        if shares <= 0.0:
            continue
        if daily_cap is not None:
            realized = sum(p for ts, p in by_day_real.get(d, [])
                           if ts <= t["decision_ns"])
            if realized <= daily_cap:
                continue
        pnl = t["unit_pnl"] * shares
        day_pnl[d] = day_pnl.get(d, 0.0) + pnl
        by_day_real.setdefault(d, []).append((t["close_ns"], pnl))
        n_taken += 1
        wins += int(t["resolved_win"])
    vals = np.asarray([day_pnl[d] for d in sorted(day_pnl)], dtype=np.float64)
    equity = np.cumsum(vals)
    dd = equity - np.maximum.accumulate(equity)
    total = float(vals.sum())
    return {
        "trades": n_taken, "pnl": round(total, 2),
        "win_rate": round(wins / n_taken, 3) if n_taken else None,
        "worst_day": round(float(vals.min()), 2) if vals.size else 0.0,
        "max_dd": round(float(dd.min()), 2) if vals.size else 0.0,
        "positive_days": int((vals > 0).sum()),
        "score": round(total / max(1.0, abs(float(dd.min()))), 3) if vals.size else 0.0,
        "daily": {d: round(day_pnl[d], 4) for d in sorted(day_pnl)},
    }


def main() -> int:
    import torch
    torch.set_num_threads(8)
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    diag = json.loads(DIAG.read_text(encoding="utf-8"))
    floors = diag["proposed_floor_grid_v2_platt"]
    print(f"EV floor grid (from diagnosis): {floors}")

    arrs = load_shards()
    days = arrs["days"]
    print(f"selection era: {arrs['y'].shape[0]} mkts, {len(days)} days")
    delta = calibrated_effective_delta(arrs)
    data = to_episode_data(arrs, delta)
    print(f"deltas ready ({time.time() - t0:.0f}s)", flush=True)

    # --- stage 1: entry configs, unit-share trades cached -------------------
    entry_rows = []
    for beta in BETAS:
        for lo, hi in TTCS:
            for plo, phi in PRICES:
                for floor in floors:
                    spec = StrategySpec(
                        strategy_id=f"b{beta:g}_ttc{lo:g}-{hi:g}_ev{floor:g}_px{plo:g}-{phi:g}",
                        beta=beta, ttc_min=lo, ttc_max=hi, ev_min=floor,
                        price_lo=plo, price_hi=phi, shares=1.0)
                    entries = find_entries(data, spec)
                    trades = []
                    for e in entries:
                        tr = simulate_trade(data, e, spec, HOLD, shares=1.0)
                        ep = int(e["ep"])
                        trades.append({
                            "date": tr["date"], "unit_pnl": tr["pnl"],
                            "entry_ev": float(e["ev"]),
                            "resolved_win": tr["resolved_win"],
                            "decision_ns": int(data.now_ns[ep, int(e["step"])]),
                            "close_ns": int(arrs["close_s"][ep]) * 1_000_000_000,
                        })
                    trades.sort(key=lambda t: t["decision_ns"])
                    entry_rows.append({"spec": spec, "floor": floor, "trades": trades})
        print(f"entry stage beta={beta:g} done ({time.time() - t0:.0f}s)", flush=True)

    # --- stage 2: sizing variants, WF prune, R1 rank -------------------------
    grid_rows = []
    for er in entry_rows:
        for k in SIZE_K:
            for s_max in SIZE_MAX:
                full = eval_sized(er["trades"], k, s_max, er["floor"], days)
                if full["trades"] < MIN_TRADES:
                    continue
                wf_ok = True
                for sel_n, val_n in WF_FOLDS:
                    sel_days, val_days = days[:sel_n], days[sel_n:val_n]
                    sel = eval_sized([t for t in er["trades"] if t["date"] in set(sel_days)],
                                     k, s_max, er["floor"], sel_days)
                    val = eval_sized([t for t in er["trades"] if t["date"] in set(val_days)],
                                     k, s_max, er["floor"], val_days)
                    if sel["score"] <= 0 or val["pnl"] <= 0:
                        wf_ok = False
                        break
                daily = full.pop("daily")
                grid_rows.append({
                    "config": f"{er['spec'].strategy_id}_k{k:g}_smax{s_max:g}",
                    "beta": er["spec"].beta, "ttc_min": er["spec"].ttc_min,
                    "ttc_max": er["spec"].ttc_max, "floor": er["floor"],
                    "price_lo": er["spec"].price_lo, "size_k": k, "size_max": s_max,
                    **full, "wf_survives": wf_ok, "_daily": daily,
                })
    print(f"grid evaluated: {len(grid_rows)} sized configs ({time.time() - t0:.0f}s)")

    survivors = [r for r in grid_rows if r["wf_survives"]]
    survivors.sort(key=lambda r: (-r["score"], -r["pnl"]))
    print(f"WF survivors: {len(survivors)}")

    # --- stage 3: portfolio + budget scaling ---------------------------------
    portfolio, chosen_daily = [], []
    for r in survivors[:40]:
        d = np.asarray([r["_daily"][x] for x in sorted(r["_daily"])])
        if any(np.corrcoef(d, c)[0, 1] > PORTFOLIO_CORR_MAX
               for c in chosen_daily if d.std() > 0 and c.std() > 0):
            continue
        portfolio.append(r)
        chosen_daily.append(d)
        if len(portfolio) == PORTFOLIO_MAX_SLOTS:
            break
    combo = np.sum(chosen_daily, axis=0) if chosen_daily else np.zeros(len(days))
    live_days_mask = np.asarray([d >= LIVE_COVERAGE_START for d in sorted(days)])
    live_daily = combo[live_days_mask]
    p5 = float(np.percentile(live_daily, 5)) if live_daily.size else 0.0
    size_scale = round(abs(BUDGET_P5) / abs(p5), 3) if p5 < 0 else 1.0
    equity = np.cumsum(combo)
    dd = equity - np.maximum.accumulate(equity)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": "selection era 06-23..07-11 (spent); per-member platt -> mean prob; "
                    "R1 rank after 2-fold WF prune; forward shadow from 07-12 is the only OOS",
        "days": days, "floors": floors,
        "grid_rows": len(grid_rows), "wf_survivors": len(survivors),
        "top20_r1": [{k2: v for k2, v in r.items() if k2 != "_daily"} for r in survivors[:20]],
        "portfolio": {
            "slots": [{k2: v for k2, v in r.items() if k2 != "_daily"} for r in portfolio],
            "combined_pnl": round(float(combo.sum()), 2),
            "combined_max_dd": round(float(dd.min()), 2) if combo.size else 0.0,
            "combined_worst_day": round(float(combo.min()), 2) if combo.size else 0.0,
            "combined_score": round(float(combo.sum()) / max(1.0, abs(float(dd.min()))), 3),
            "live_coverage_p5_daily": round(p5, 2),
            "size_scale_for_budget": size_scale,
            "budget_p5_daily": BUDGET_P5,
        },
    }
    with (OUT_DIR / "grid.csv").open("w", newline="", encoding="utf-8") as f:
        rows = [{k2: v for k2, v in r.items() if k2 != "_daily"} for r in grid_rows]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("== TOP 5 R1 (WF survivors):")
    for r in survivors[:5]:
        print(f"  {r['config']:52s} pnl {r['pnl']:+8.2f} DD {r['max_dd']:7.2f} "
              f"worst {r['worst_day']:7.2f} trades {r['trades']:4d} score {r['score']:.2f}")
    print("== PORTFOLIO:", [r["config"] for r in portfolio])
    print(json.dumps(report["portfolio"], indent=1, default=str)[:600])
    print(f"-> {OUT_DIR} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
