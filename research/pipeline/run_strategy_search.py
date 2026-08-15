#!/usr/bin/env python3
"""Quiet-mode strategy search driver.

Trimmed first pass: hybrid mode only, Optuna TPE, 30 trials × 20 MC weeks ×
5k anchors/day. Capped at ~20% CPU + ~4GB RAM (best-effort).

Output:
  experiments/strategy_search/<timestamp>/
    leaderboard.csv          - one row per completed trial
    trial_<id>/
      params.json
      per_week.json
      grade.json

Usage:
  py -3 run_strategy_search.py                          # full trimmed search
  py -3 run_strategy_search.py --smoke                  # 3 trials × 5 weeks (fast)
  py -3 run_strategy_search.py --trials 30 --weeks 20
  py -3 run_strategy_search.py --cpu-pct 0.20 --mem-cap-mb 4096
"""
from __future__ import annotations

# ── Apply resource caps BEFORE importing heavy libs ──────────────────────────
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backtest.resource_caps import apply_quiet_mode

# Defaults — will be overridden by argv parsing below
_DEFAULT_CPU_PCT = 0.20
_DEFAULT_MEM_MB = 4096

# Quick argv peek so caps go on early
def _peek_arg(name: str, cast, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            try: return cast(sys.argv[i + 1])
            except Exception: pass
    return default

_cpu_pct = _peek_arg("--cpu-pct", float, _DEFAULT_CPU_PCT)
_mem_mb  = _peek_arg("--mem-cap-mb", int, _DEFAULT_MEM_MB)
apply_quiet_mode(cpu_pct=_cpu_pct, mem_cap_mb=_mem_mb, verbose=True)

# ── Now the heavy imports ────────────────────────────────────────────────────
import argparse, csv, gc, io, json, time
from datetime import datetime

import optuna

from backtest.grading import grade_strategy
from backtest.mc_weeks import generate_mc_weeks
from backtest.multi_market_engine import MultiMarketEngine, cleanup
from backtest.search_space import suggest_hybrid_config, fixed_baseline_config

sys.path.insert(0, "src")
from trading.signals import SignalEngine

# Encode stderr/stdout as utf-8 (Windows console quirks)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ── Per-trial backtest ───────────────────────────────────────────────────────

def evaluate_config(cfg, signals: SignalEngine, weeks: list[list[str]],
                    starting_usd: float, max_anchors_per_day: int,
                    fee_scale: float = 1.0, seed: int = 0,
                    progress_cb=None) -> tuple[list[dict], float]:
    per_week: list[dict] = []
    t0 = time.time()
    for i, week_dates in enumerate(weeks):
        engine = MultiMarketEngine(cfg, signals, starting_balance_usd=starting_usd,
                                   verbose=False, seed=seed + i)
        s = engine.run(week_dates, max_anchors_per_day=max_anchors_per_day,
                       fee_scale=fee_scale)
        per_week.append(s)
        if progress_cb:
            progress_cb(i + 1, len(weeks), s)
        del engine
        cleanup()
    return per_week, time.time() - t0


# ── Optuna objective ─────────────────────────────────────────────────────────

def make_objective(signals, weeks, starting_usd, max_anchors_per_day,
                   out_root: Path, seed: int):
    leaderboard_path = out_root / "leaderboard.csv"
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    if not leaderboard_path.exists():
        with leaderboard_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["trial", "score", "rejected", "mean_calmar_clean",
                        "median_net_pnl_usd", "mean_net_pnl_usd",
                        "p10_net_pnl_usd", "n_weeks_with_positive_pnl",
                        "disqualification_rate", "elapsed_sec", "params"])

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        cfg, params = suggest_hybrid_config(trial)
        per_week, _ = evaluate_config(cfg, signals, weeks, starting_usd,
                                      max_anchors_per_day, seed=seed)
        grade = grade_strategy(per_week)

        # Persist trial
        tdir = out_root / f"trial_{trial.number:03d}"
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "params.json").write_text(json.dumps(params, indent=2))
        (tdir / "per_week.json").write_text(json.dumps(per_week, indent=2, default=str))
        (tdir / "grade.json").write_text(json.dumps(grade, indent=2))

        elapsed = time.time() - t0
        with leaderboard_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([trial.number, f"{grade['score']:.4f}", grade["rejected"],
                        f"{grade['mean_calmar_clean']:.4f}",
                        f"{grade['median_net_pnl_usd']:.4f}",
                        f"{grade['mean_net_pnl_usd']:.4f}",
                        f"{grade['p10_net_pnl_usd']:.4f}",
                        grade["n_weeks_with_positive_pnl"],
                        f"{grade['disqualification_rate']:.3f}",
                        f"{elapsed:.1f}", json.dumps(params)])

        print(f"[trial {trial.number:03d}] score={grade['score']:+.3f} "
              f"medPnL=${grade['median_net_pnl_usd']:+.2f} "
              f"calmar={grade['mean_calmar_clean']:+.2f} "
              f"posWk={grade['n_weeks_with_positive_pnl']}/{grade['n_weeks']} "
              f"disq={grade['disqualification_rate']:.2f} ({elapsed:.0f}s)",
              file=sys.stderr)
        return grade["score"]

    return objective


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--weeks", type=int, default=20)
    ap.add_argument("--anchors-per-day", type=int, default=5000)
    ap.add_argument("--starting-usd", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu-pct", type=float, default=_DEFAULT_CPU_PCT,
                    help="(parsed early; recorded only here)")
    ap.add_argument("--mem-cap-mb", type=int, default=_DEFAULT_MEM_MB,
                    help="(parsed early; recorded only here)")
    ap.add_argument("--smoke", action="store_true",
                    help="3 trials × 5 weeks × 1000 anchors/day")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--baseline-only", action="store_true",
                    help="Evaluate the fixed market_maker preset only — no search")
    args = ap.parse_args()

    if args.smoke:
        args.trials = 3
        args.weeks = 5
        args.anchors_per_day = 1000

    # Output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if args.out else Path("experiments") / "strategy_search" / ts
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\n[search] output = {out_root}", file=sys.stderr)

    # Generate / load MC weeks
    weeks = generate_mc_weeks(n_weeks=args.weeks, seed=args.seed)
    (out_root / "mc_weeks.json").write_text(json.dumps(weeks, indent=2))
    print(f"[search] {len(weeks)} MC weeks loaded "
          f"(first: {weeks[0]})", file=sys.stderr)

    # Load signals once (heavy — keep for the whole search)
    print(f"[search] loading SignalEngine ...", file=sys.stderr)
    signals = SignalEngine(artifact_root="artifacts_cleaned")
    gc.collect()

    # Baseline first
    print(f"\n[search] evaluating fixed baseline (market_maker preset) ...", file=sys.stderr)
    base_cfg = fixed_baseline_config()
    base_per_week, base_elapsed = evaluate_config(
        base_cfg, signals, weeks, args.starting_usd, args.anchors_per_day,
        seed=args.seed,
    )
    base_grade = grade_strategy(base_per_week)
    (out_root / "baseline_grade.json").write_text(json.dumps(base_grade, indent=2))
    (out_root / "baseline_per_week.json").write_text(json.dumps(base_per_week, indent=2, default=str))
    print(f"[baseline] score={base_grade['score']:+.3f} "
          f"medPnL=${base_grade['median_net_pnl_usd']:+.2f} "
          f"posWk={base_grade['n_weeks_with_positive_pnl']}/{base_grade['n_weeks']} "
          f"({base_elapsed:.0f}s)", file=sys.stderr)

    if args.baseline_only:
        return

    # Optuna TPE search
    print(f"\n[search] starting Optuna TPE: {args.trials} trials × "
          f"{args.weeks} weeks × ~{args.anchors_per_day} anchors/day", file=sys.stderr)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"hybrid_{ts}",
                                storage=f"sqlite:///{out_root / 'study.db'}",
                                load_if_exists=False)
    objective = make_objective(signals, weeks, args.starting_usd,
                               args.anchors_per_day, out_root, args.seed)
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True,
                   show_progress_bar=False)

    # Final report
    print("\n" + "=" * 70, file=sys.stderr)
    print("SEARCH COMPLETE", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    best = study.best_trial
    print(f"  Best trial #{best.number}  score={best.value:+.3f}", file=sys.stderr)
    print(f"  Best params:", file=sys.stderr)
    for k, v in best.params.items():
        print(f"    {k} = {v}", file=sys.stderr)
    print(f"  Baseline score: {base_grade['score']:+.3f}", file=sys.stderr)
    print(f"  Leaderboard:    {out_root / 'leaderboard.csv'}", file=sys.stderr)

    # Persist best
    (out_root / "best_trial.json").write_text(json.dumps({
        "trial_number": best.number,
        "score": best.value,
        "params": best.params,
    }, indent=2))


if __name__ == "__main__":
    main()
