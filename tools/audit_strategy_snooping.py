"""Anti-snooping audit of the locked TCN double strategy (Phase 2.1 + 2.2).

Per strategy slot (selection pressure differs: the early slot + its UTC hour
filter were picked on TEST, the late slot was picked on a 2-day val):

1. Stationary bootstrap (Politis-Romano) over daily PnL: CIs for total PnL and
   daily Sharpe; variants with top-1/top-3 days removed (concentration).
2. Deflated Sharpe Ratio (Bailey & Lopez de Prado) with a sensitivity sweep
   over the number of trials, since the true search universe spans ~10^2-10^5.
3. White's Reality Check: the entry-spec universe from
   search_tcn_fullmarket_exit_policy (840 specs, hold exit) plus the early-slot
   hour-window variants is REGENERATED through the shared engine to get
   per-candidate daily PnL, then a stationary-bootstrap max-statistic test asks
   how often a universe this size produces the observed winner by luck.
4. Day concentration (leave-one-day-out, top-k removal) and regime buckets
   (BTC realized vol from the strike series, per-day valid_frac = recorder era).

Outputs: artifacts/audit_v1/snooping_report.json + snooping_report.md with an
explicit D2 verdict per slot.

Usage:
    py tools/audit_strategy_snooping.py                  # full (regenerates universe, ~20-40 min)
    py tools/audit_strategy_snooping.py --skip-wrc       # bootstrap+DSR+concentration only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from backtest.episode_strategy_backtester import (  # noqa: E402
    HOLD, EpisodeData, StrategySpec, find_entries, load_episode_split, load_lock,
    simulate_trade, summarize,
)

DEFAULT_DATASET = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms"
DEFAULT_TCN = ROOT / "artifacts" / "btc_5m_episode_tcn_c64_b7_cal_ttc15_90"
DEFAULT_LOCK = ROOT / "artifacts" / "strategy_locks" / "tcn_double_strategy_v1_lock.json"
OUT_DIR = ROOT / "artifacts" / "audit_v1"
DELTA_CACHE = OUT_DIR / "delta_test_cache.npy"

RNG_SEED = 20260710
B_BOOT = 10_000
MEAN_BLOCK_DAYS = 3.0


# ---------------------------------------------------------------------------
# Bootstrap / statistics helpers
# ---------------------------------------------------------------------------

def stationary_bootstrap_idx(n: int, b: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """(b, n) index matrix from the Politis-Romano stationary bootstrap."""
    p = 1.0 / max(1.0, mean_block)
    starts = rng.integers(0, n, size=(b, n))
    restart = rng.random((b, n)) < p
    restart[:, 0] = True
    idx = np.zeros((b, n), dtype=np.int64)
    for t in range(n):
        if t == 0:
            idx[:, 0] = starts[:, 0]
        else:
            cont = (idx[:, t - 1] + 1) % n
            idx[:, t] = np.where(restart[:, t], starts[:, t], cont)
    return idx


def sharpe(daily: np.ndarray) -> float:
    if daily.size < 2 or daily.std(ddof=1) == 0:
        return 0.0
    return float(daily.mean() / daily.std(ddof=1))


def boot_stats(daily: np.ndarray, rng: np.random.Generator, b: int = B_BOOT) -> dict:
    n = daily.size
    idx = stationary_bootstrap_idx(n, b, MEAN_BLOCK_DAYS, rng)
    samples = daily[idx]                      # (b, n)
    totals = samples.sum(axis=1)
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    stds[stds == 0] = np.inf
    sharpes = means / stds
    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {
        "days": int(n),
        "total_pnl": float(daily.sum()),
        "total_pnl_ci95": ci(totals),
        "p_total_leq_0": float(np.mean(totals <= 0.0)),
        "daily_sharpe": sharpe(daily),
        "daily_sharpe_ci95": ci(sharpes),
        "p_sharpe_leq_0": float(np.mean(sharpes <= 0.0)),
    }


def deflated_sharpe(daily: np.ndarray, n_trials: int) -> dict:
    """DSR per Bailey & Lopez de Prado (2014). Returns PSR vs the expected-max
    Sharpe benchmark under n_trials independent trials."""
    n = daily.size
    sr = sharpe(daily)
    if n < 4:
        return {"n_trials": n_trials, "dsr": None, "note": "too few days"}
    x = (daily - daily.mean()) / (daily.std(ddof=1) or 1.0)
    g3 = float(np.mean(x ** 3))
    g4 = float(np.mean(x ** 4))
    # expected max SR of n_trials iid N(0, var_sr) trials
    var_sr = 1.0 / n  # variance of SR estimator under H0 (SR=0), per LdP
    emc = 0.5772156649015329
    if n_trials <= 1:
        sr0 = 0.0
    else:
        z1 = _norm_ppf(1.0 - 1.0 / n_trials)
        z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
        sr0 = math.sqrt(var_sr) * ((1.0 - emc) * z1 + emc * z2)
    denom = math.sqrt(max(1e-12, (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr) / (n - 1)))
    z = (sr - sr0) / denom
    return {
        "n_trials": n_trials,
        "sr_daily": sr,
        "sr_benchmark_expected_max": sr0,
        "dsr_prob_sr_gt_benchmark": _norm_cdf(z),
        "passes_dsr_gt_0.95": bool(_norm_cdf(z) > 0.95),
    }


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(q: float) -> float:
    # Acklam's rational approximation; adequate for audit purposes
    if not 0.0 < q < 1.0:
        raise ValueError(q)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        u = math.sqrt(-2 * math.log(q))
        return (((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]) / \
               ((((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1)
    if q > phigh:
        u = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]) / \
               ((((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1)
    u = q - 0.5
    r = u * u
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * u / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ---------------------------------------------------------------------------
# Daily series
# ---------------------------------------------------------------------------

def daily_series(trades: list[dict], all_days: list[str]) -> np.ndarray:
    by_day = {d: 0.0 for d in all_days}
    for t in trades:
        by_day[t["date"]] = by_day.get(t["date"], 0.0) + float(t["pnl"])
    return np.asarray([by_day[d] for d in sorted(by_day)], dtype=np.float64)


def drop_top_k(daily: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or daily.size <= k:
        return daily
    order = np.argsort(daily)          # ascending; top days at the end
    return daily[order[: daily.size - k]]


def concentration(trades: list[dict], all_days: list[str]) -> dict:
    daily = daily_series(trades, all_days)
    days_sorted = sorted(all_days)
    order = np.argsort(daily)[::-1]
    top_days = [
        {"date": days_sorted[i], "pnl": round(float(daily[i]), 3)} for i in order[:5]
    ]
    total = float(daily.sum())
    lodo_min = float(total - daily.max())            # remove best day
    return {
        "total_pnl": round(total, 3),
        "top5_days": top_days,
        "pnl_wo_top1": round(float(drop_top_k(daily, 1).sum()), 3),
        "pnl_wo_top3": round(float(drop_top_k(daily, 3).sum()), 3),
        "top1_share_of_pnl": round(float(daily.max()) / total, 3) if total > 0 else None,
        "lodo_worst_total": round(lodo_min, 3),
    }


# ---------------------------------------------------------------------------
# White's Reality Check universe
# ---------------------------------------------------------------------------

def make_universe_specs() -> list[StrategySpec]:
    """Reconstruct the searched entry-spec universe (hold exit): the fullmarket
    grid (4 betas x 10 ttc windows x 7 EVs x 3 price bands) plus the early-slot
    hour-window variants that were compared on test."""
    betas = [0.5, 0.75, 1.0, 1.25]
    windows = [(50.0, 75.0), (75.0, 120.0), (75.0, 150.0), (90.0, 150.0),
               (90.0, 180.0), (90.0, 240.0), (90.0, 300.0), (120.0, 240.0),
               (120.0, 300.0), (150.0, 300.0)]
    evs = [0.0, 0.02, 0.05, 0.075, 0.10, 0.125, 0.15]
    prices = [(0.10, 0.90), (0.20, 0.80), (0.30, 0.70)]
    specs: list[StrategySpec] = []
    for beta in betas:
        for lo, hi in windows:
            for ev in evs:
                for plo, phi in prices:
                    specs.append(StrategySpec(
                        strategy_id=f"u_b{beta:g}_ttc{lo:g}-{hi:g}_ev{ev:g}_px{plo:g}-{phi:g}",
                        beta=beta, ttc_min=lo, ttc_max=hi, ev_min=ev,
                        price_lo=plo, price_hi=phi))
    # hour-window variants of the early candidate (test-compared in
    # two_strategy_early_hour_filtered_report.json)
    hour_windows = [None, range(2, 14), range(3, 11), range(4, 16), range(5, 17), range(6, 18)]
    for hw in hour_windows:
        specs.append(StrategySpec(
            strategy_id=f"u_early_hours_{'all' if hw is None else f'{hw.start}-{hw.stop - 1}'}",
            beta=1.0, ttc_min=75.0, ttc_max=120.0, ev_min=0.075,
            price_lo=0.20, price_hi=0.80,
            utc_hours=None if hw is None else frozenset(hw)))
    return specs


def universe_daily_pnl(data: EpisodeData, specs: list[StrategySpec],
                       all_days: list[str]) -> tuple[np.ndarray, list[str]]:
    """(n_specs, n_days) daily PnL for every candidate, hold-to-resolution."""
    rows = np.zeros((len(specs), len(all_days)), dtype=np.float64)
    day_index = {d: i for i, d in enumerate(sorted(all_days))}
    t0 = time.time()
    for i, spec in enumerate(specs):
        entries = find_entries(data, spec)
        for e in entries:
            tr = simulate_trade(data, e, spec, HOLD)
            rows[i, day_index[tr["date"]]] += tr["pnl"]
        if (i + 1) % 100 == 0:
            print(f"  universe {i + 1}/{len(specs)} ({time.time() - t0:.0f}s)", flush=True)
    return rows, [s.strategy_id for s in specs]


def whites_reality_check(universe: np.ndarray, rng: np.random.Generator,
                         b: int = 2000) -> dict:
    """H0: no candidate has positive expected daily PnL. Statistic = max mean.
    Stationary bootstrap over days, candidates recentered by their sample mean."""
    n_specs, n_days = universe.shape
    means = universe.mean(axis=1)
    observed_max = float(means.max())
    best_idx = int(means.argmax())
    idx = stationary_bootstrap_idx(n_days, b, MEAN_BLOCK_DAYS, rng)
    centered = universe - means[:, None]
    boot_max = np.empty(b)
    for j in range(b):
        boot_max[j] = centered[:, idx[j]].mean(axis=1).max()
    p_value = float(np.mean(boot_max >= observed_max))
    return {
        "n_candidates": int(n_specs),
        "n_days": int(n_days),
        "observed_max_mean_daily_pnl": observed_max,
        "best_candidate_idx": best_idx,
        "wrc_p_value": p_value,
        "passes_wrc_10pct": bool(p_value < 0.10),
        "bootstrap_reps": b,
    }


# ---------------------------------------------------------------------------
# Regime buckets
# ---------------------------------------------------------------------------

def regime_table(data: EpisodeData, trades: list[dict], all_days: list[str]) -> list[dict]:
    """Per-day: PnL, trades, BTC realized vol (from the strike series), mean
    valid_frac (recorder-era proxy)."""
    import numpy as np
    z = {}
    days_sorted = sorted(all_days)
    strikes_by_day: dict[str, list[tuple[int, float]]] = {d: [] for d in days_sorted}
    valid_by_day: dict[str, list[float]] = {d: [] for d in days_sorted}
    valid_frac = data.valid.mean(axis=1)
    for i in range(data.n_ep):
        d = str(data.date[i])
        strikes_by_day[d].append((int(data.open_s[i]), float('nan')))
        valid_by_day[d].append(float(valid_frac[i]))
    # strike array is not on EpisodeData; reload lazily
    npz = np.load(DEFAULT_DATASET / f"{data.split}.npz", allow_pickle=False)
    strike = npz["strike"].astype(np.float64)
    open_s = npz["open_s"].astype(np.int64)
    date = npz["date"]
    strikes_by_day = {d: [] for d in days_sorted}
    for i in range(len(strike)):
        strikes_by_day[str(date[i])].append((int(open_s[i]), float(strike[i])))
    pnl_by_day: dict[str, float] = {d: 0.0 for d in days_sorted}
    n_by_day: dict[str, int] = {d: 0 for d in days_sorted}
    for t in trades:
        pnl_by_day[t["date"]] += float(t["pnl"])
        n_by_day[t["date"]] += 1
    rows = []
    for d in days_sorted:
        pts = sorted(strikes_by_day[d])
        px = np.asarray([p for _, p in pts if np.isfinite(p) and p > 0])
        if px.size >= 10:
            rets = np.diff(np.log(px))
            vol_bp = float(np.std(rets) * 1e4)   # per-5min log-ret std in bp
        else:
            vol_bp = float("nan")
        rows.append({
            "date": d,
            "pnl": round(pnl_by_day[d], 3),
            "trades": n_by_day[d],
            "btc_5m_vol_bp": round(vol_bp, 2) if np.isfinite(vol_bp) else None,
            "mean_valid_frac": round(float(np.mean(valid_by_day[d])), 3) if valid_by_day[d] else None,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--tcn-artifacts", type=Path, default=DEFAULT_TCN)
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--split", default="test")
    ap.add_argument("--skip-wrc", action="store_true")
    ap.add_argument("--wrc-boot", type=int, default=2000)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    specs, lock = load_lock(args.lock if args.lock.is_absolute() else ROOT / args.lock)
    print("loading split + TCN deltas...", flush=True)
    if DELTA_CACHE.exists():
        delta = np.load(DELTA_CACHE)
        data = load_episode_split(args.dataset, args.split, delta=delta)
    else:
        data = load_episode_split(args.dataset, args.split, tcn_dir=args.tcn_artifacts)
        np.save(DELTA_CACHE, data.delta)
    all_days = sorted(set(str(d) for d in data.date))
    print(f"{data.n_ep} episodes over {len(all_days)} days", flush=True)

    report: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "split": args.split, "days": all_days,
                    "bootstrap": {"B": B_BOOT, "mean_block_days": MEAN_BLOCK_DAYS,
                                  "seed": RNG_SEED},
                    "slots": {}}

    trades_by_slot: dict[str, list[dict]] = {}
    for spec in specs:
        entries = find_entries(data, spec)
        trades_by_slot[spec.strategy_id] = [simulate_trade(data, e, spec, HOLD) for e in entries]
    trades_by_slot["combined"] = [t for ts in trades_by_slot.values() for t in ts]

    # --- 2.1 bootstrap + DSR per slot -------------------------------------
    # honest trials sensitivity: early was argmax over a test-ranked universe;
    # late went through a val->test channel with ~25 test evaluations
    trials_grid = {"early_tcn_75_120_utc02_13": [846, 5000, 58800],
                   "late_tcn_50_75_all_day": [1, 25, 846],
                   "combined": [846, 5000, 58800]}
    for sid, trades in trades_by_slot.items():
        daily = daily_series(trades, all_days)
        slot = {
            "summary": summarize(trades, data.n_ep),
            "bootstrap_full": boot_stats(daily, rng),
            "bootstrap_wo_top1": boot_stats(drop_top_k(daily, 1), rng),
            "bootstrap_wo_top3": boot_stats(drop_top_k(daily, 3), rng),
            "concentration": concentration(trades, all_days),
            "dsr_sensitivity": [deflated_sharpe(daily, n) for n in trials_grid.get(sid, [846])],
        }
        report["slots"][sid] = slot
        print(f"{sid}: total={slot['summary']['total_pnl']:.2f} "
              f"CI95={slot['bootstrap_full']['total_pnl_ci95']} "
              f"P(<=0)={slot['bootstrap_full']['p_total_leq_0']:.3f} "
              f"woTop1 P(<=0)={slot['bootstrap_wo_top1']['p_total_leq_0']:.3f}", flush=True)

    # --- 2.3-lite: WRC over the regenerated universe -----------------------
    if not args.skip_wrc:
        print("regenerating search universe for White's Reality Check...", flush=True)
        uni_specs = make_universe_specs()
        universe, labels = universe_daily_pnl(data, uni_specs, all_days)
        np.savez_compressed(OUT_DIR / "wrc_universe_daily_pnl.npz",
                            daily_pnl=universe, labels=np.asarray(labels))
        # Independent RNG stream: sharing one generator with the per-slot
        # bootstraps above makes this p-value depend on how many slots ran
        # first, so an independent recomputation could not match it exactly.
        wrc_rng = np.random.default_rng(RNG_SEED + 1)
        wrc = whites_reality_check(universe, wrc_rng, b=args.wrc_boot)
        wrc["best_candidate"] = labels[wrc.pop("best_candidate_idx")]
        # where do the locked configs rank inside the universe?
        totals = universe.sum(axis=1)
        rank = {}
        for probe in ("u_early_hours_2-13", "u_b1.25_ttc50-75_ev0.125_px0.2-0.8",
                      "u_b1_ttc75-120_ev0.075_px0.2-0.8"):
            if probe in labels:
                i = labels.index(probe)
                rank[probe] = {"total_pnl": round(float(totals[i]), 2),
                               "rank_by_total_pnl": int((totals > totals[i]).sum()) + 1}
        wrc["locked_config_ranks"] = rank
        report["whites_reality_check"] = wrc
        print(f"WRC: p={wrc['wrc_p_value']:.4f} best={wrc['best_candidate']}", flush=True)

    # --- 2.2 regime table ---------------------------------------------------
    report["regime_by_day_combined"] = regime_table(data, trades_by_slot["combined"], all_days)

    # --- D2 verdict ----------------------------------------------------------
    verdicts = {}
    for sid in ("early_tcn_75_120_utc02_13", "late_tcn_50_75_all_day"):
        s = report["slots"][sid]
        ci_wo_top1 = s["bootstrap_wo_top1"]["total_pnl_ci95"]
        dsr_honest = s["dsr_sensitivity"][-1]["dsr_prob_sr_gt_benchmark"]
        fails = []
        if ci_wo_top1[0] <= 0.0:
            fails.append(f"bootstrap 95% CI includes 0 after dropping top day ({ci_wo_top1})")
        if dsr_honest is not None and dsr_honest <= 0.5:
            fails.append(f"DSR at honest trials <= 0.5 ({dsr_honest:.3f})")
        verdicts[sid] = {"kill_criteria_hit": fails, "survives_d2": not fails}
    if "whites_reality_check" in report:
        if not report["whites_reality_check"]["passes_wrc_10pct"]:
            for sid in verdicts:
                verdicts[sid]["kill_criteria_hit"].append(
                    f"WRC p={report['whites_reality_check']['wrc_p_value']:.3f} >= 0.10")
                verdicts[sid]["survives_d2"] = False
    report["d2_verdicts"] = verdicts

    (OUT_DIR / "snooping_report.json").write_text(json.dumps(report, indent=1),
                                                  encoding="utf-8")
    print(f"-> {OUT_DIR / 'snooping_report.json'}")
    for sid, v in verdicts.items():
        print(f"D2 {sid}: {'SURVIVES' if v['survives_d2'] else 'KILL'} {v['kill_criteria_hit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
