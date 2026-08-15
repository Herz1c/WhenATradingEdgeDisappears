"""Diagnose why the v2 ensemble under-trades v1 thresholds (Plan v2.1, Phase 1).

On the selection era (2026-06-23..07-11, daily shards), compare:
  1. |delta| distributions: v1 model vs v2 ensemble (valid steps, TTC 15-150)
  2. achievable EV distributions and how often each crosses the v1-era floors
  3. calibration effect: per-member Platt (val_platt_l2 from each tcn_report)
     -> mean prob, vs raw mean-logit; Brier per TTC bucket vs market baseline
  4. quantile map of positive EV under the calibrated ensemble -> the
     data-driven EV floor grid for the Phase 2 search

Both models see their own normalization (v1 norm for the v1 model, v2 norm for
the v2 members), built from the shards' X_raw.

Output: artifacts/audit_v1/v2_transfer_diagnosis.json + console summary.
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
    load_tcn_model, logit_np, predict_delta, sigmoid,
)

DAILY = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms" / "daily"
V1_NORM = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms" / "normalization.json"
V2_NORM = ROOT / "data" / "datasets" / "btc_5m_episodes_v2_200ms" / "normalization.json"
V1_MODEL = ROOT / "artifacts" / "btc_5m_episode_tcn_c64_b7_cal_ttc15_90"
V2_DIRS = [ROOT / "artifacts" / f"tcn_v2_c64_b7_ttc15_150_seed{s}" for s in (11, 7, 23, 42, 101)]
OUT = ROOT / "artifacts" / "audit_v1" / "v2_transfer_diagnosis.json"

SEL_START, SEL_END = "2026-06-23", "2026-07-11"
TTC_LO, TTC_HI = 15.0, 150.0
V1_FLOORS = [0.02, 0.04, 0.075, 0.125]
UP_ASK, DN_ASK = 1, 3


def load_norm(path: Path) -> tuple[np.ndarray, np.ndarray]:
    n = json.loads(path.read_text(encoding="utf-8"))
    mean = np.asarray(n["mean"], dtype=np.float32)
    std = np.asarray(n["std"], dtype=np.float32)
    std[std <= 1e-12] = 1.0
    return mean, std


def load_shards() -> dict[str, np.ndarray]:
    days = sorted(p.stem for p in DAILY.glob("*.npz") if SEL_START <= p.stem <= SEL_END)
    parts: dict[str, list[np.ndarray]] = {}
    for d in days:
        z = np.load(DAILY / f"{d}.npz", allow_pickle=False)
        for k in ("X_raw", "valid_mask", "y", "p_market", "quotes", "date"):
            parts.setdefault(k, []).append(z[k])
    out = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    out["days"] = days
    return out


def forward(arrs: dict, norm_path: Path, model_dirs: list[Path]) -> list[np.ndarray]:
    mean, std = load_norm(norm_path)
    X = arrs["X_raw"].astype(np.float32)
    vm = arrs["valid_mask"].astype(bool)
    Xn = (X - mean) / std
    Xn[~vm] = 0.0
    np.nan_to_num(Xn, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    Xn = np.concatenate([Xn, vm[:, :, None].astype(np.float32)], axis=2)
    deltas = []
    for mdir in model_dirs:
        model, _rep, device = load_tcn_model(mdir, n_features=Xn.shape[2])
        deltas.append(predict_delta(model, Xn, batch_size=64, device=device))
    return deltas


def platt_params(mdir: Path) -> tuple[float, float]:
    rep = json.loads((mdir / "tcn_report.json").read_text(encoding="utf-8"))
    cal = rep["calibrators"]["val_platt_l2"]
    return float(cal["coef"]), float(cal["intercept"])


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def q(a: np.ndarray, qs=(50, 75, 90, 95, 99)) -> dict:
    if a.size == 0:
        return {}
    return {f"q{k}": round(float(np.percentile(a, k)), 5) for k in qs} | {
        "max": round(float(a.max()), 5), "n": int(a.size)}


def main() -> int:
    import torch
    torch.set_num_threads(8)
    t0 = time.time()
    arrs = load_shards()
    n_ep, T = arrs["valid_mask"].shape
    print(f"selection era: {n_ep} episodes over {len(arrs['days'])} days "
          f"({arrs['days'][0]}..{arrs['days'][-1]})")

    ttc = 300.0 - np.arange(T, dtype=np.float32) * 0.2
    band = (ttc > TTC_LO) & (ttc <= TTC_HI)
    mask = arrs["valid_mask"].astype(bool) & band[None, :]

    p_market = arrs["p_market"].astype(np.float32)
    p_market[~np.isfinite(p_market)] = 0.5
    base_logit = logit_np(p_market)
    y_step = np.repeat(arrs["y"].astype(np.float32)[:, None], T, axis=1)
    up_ask = arrs["quotes"][:, :, UP_ASK].astype(np.float32)
    dn_ask = arrs["quotes"][:, :, DN_ASK].astype(np.float32)

    print("forwarding v1 model...", flush=True)
    d_v1 = forward(arrs, V1_NORM, [V1_MODEL])[0]
    print(f"forwarding v2 members... ({time.time() - t0:.0f}s)", flush=True)
    d_v2 = forward(arrs, V2_NORM, V2_DIRS)
    d_v2_mean = np.mean(d_v2, axis=0)

    # probability variants
    p_v1 = sigmoid(base_logit + d_v1)                       # v1 raw (beta=1)
    p_v2_raw = sigmoid(base_logit + d_v2_mean)              # v2 raw mean-logit
    member_probs = []
    for delta, mdir in zip(d_v2, V2_DIRS):
        coef, icpt = platt_params(mdir)
        member_probs.append(sigmoid(coef * (base_logit + delta) + icpt))
    p_v2_platt = np.mean(member_probs, axis=0)              # per-member platt -> mean prob

    report: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "era": f"{arrs['days'][0]}..{arrs['days'][-1]}",
                    "episodes": int(n_ep), "ttc_band": [TTC_LO, TTC_HI]}

    # 1) delta distributions
    report["delta_abs"] = {
        "v1": q(np.abs(d_v1[mask])),
        "v2_mean": q(np.abs(d_v2_mean[mask])),
        "v2_members_q90": [round(float(np.percentile(np.abs(d[mask]), 90)), 5) for d in d_v2],
    }

    # 2) EV distributions + floor crossing rates
    def ev_of(p):
        return np.maximum(p - up_ask, (1.0 - p) - dn_ask)
    evs = {"v1_raw": ev_of(p_v1), "v2_raw": ev_of(p_v2_raw), "v2_platt": ev_of(p_v2_platt)}
    report["ev"] = {}
    for name, ev in evs.items():
        e = ev[mask]
        pos = e[e > 0]
        crossing = {f"ge_{f:g}": int((e >= f).sum()) for f in V1_FLOORS}
        report["ev"][name] = {"positive": q(pos), "steps_crossing_v1_floors": crossing}

    # 3) Brier per TTC bucket
    buckets = [(15.0, 50.0), (50.0, 90.0), (90.0, 150.0)]
    report["brier_by_ttc"] = {}
    for lo, hi in buckets:
        m = arrs["valid_mask"].astype(bool) & ((ttc > lo) & (ttc <= hi))[None, :]
        row = {
            "market": round(brier(p_market[m], y_step[m]), 6),
            "v2_raw": round(brier(p_v2_raw[m], y_step[m]), 6),
            "v2_platt": round(brier(p_v2_platt[m], y_step[m]), 6),
            "v1_raw": round(brier(p_v1[m], y_step[m]), 6),
            "n": int(m.sum()),
        }
        report["brier_by_ttc"][f"ttc_{lo:g}_{hi:g}"] = row

    # 4) floor grid proposal from calibrated ensemble positive-EV quantiles
    pos = evs["v2_platt"][mask]
    pos = pos[pos > 0]
    floor_grid = sorted({round(float(np.percentile(pos, k)), 4)
                         for k in (50, 70, 80, 90, 95, 98)})
    report["proposed_floor_grid_v2_platt"] = floor_grid

    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("delta_abs", "brier_by_ttc", "proposed_floor_grid_v2_platt")}, indent=1))
    for name in ("v1_raw", "v2_raw", "v2_platt"):
        print(name, report["ev"][name]["steps_crossing_v1_floors"])
    print(f"-> {OUT} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
