#!/usr/bin/env python3
"""Optuna search over IntentFilter parameters using the execution_intent engine.

Quiet mode. Grades each candidate across N MC weeks via the Calmar-style scorer.
"""
from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backtest.resource_caps import apply_quiet_mode

def _peek(name, cast, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            try: return cast(sys.argv[i + 1])
            except Exception: pass
    return default

apply_quiet_mode(cpu_pct=_peek("--cpu-pct", float, 0.20),
                 mem_cap_mb=_peek("--mem-cap-mb", int, 4096),
                 verbose=True)

import argparse, csv, gc, io, json, time
from datetime import datetime

import optuna

from backtest.execution_intent_engine import ExecutionIntentEngine, IntentFilter
from backtest.grading import grade_strategy
from backtest.mc_weeks import generate_mc_weeks
from trading.signals import SignalEngine

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def evaluate(filt, signals, weeks, max_intents_per_day, max_margin_usd):
    per_week = []
    for w in weeks:
        eng = ExecutionIntentEngine(signals, filt, starting_balance_usd=100.0)
        s = eng.run(w, max_intents_per_day=max_intents_per_day,
                    filled_only=True, max_margin_usd=max_margin_usd)
        per_week.append(s)
        del eng
        gc.collect()
    return per_week


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--weeks", type=int, default=10)
    ap.add_argument("--intents-per-day", type=int, default=50_000)
    ap.add_argument("--margin-cap", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--final-weeks", type=int, default=20,
                    help="Grade winner on this many MC weeks at the end")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if args.out else Path("experiments") / "intent_search" / ts
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[search] output = {out_root}", file=sys.stderr)

    weeks = generate_mc_weeks(n_weeks=args.weeks, seed=args.seed)
    (out_root / "mc_weeks.json").write_text(json.dumps(weeks, indent=2))
    print(f"[search] {len(weeks)} MC weeks loaded", file=sys.stderr)

    print(f"[search] loading SignalEngine ...", file=sys.stderr)
    signals = SignalEngine(artifact_root="artifacts_cleaned")

    # Leaderboard CSV
    lb = out_root / "leaderboard.csv"
    with lb.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["trial", "score", "rejected", "mean_calmar", "median_pnl",
             "mean_pnl", "p10_pnl", "n_pos_weeks", "disq_rate",
             "elapsed_sec", "params"])

    def objective(trial):
        t0 = time.time()
        sp_lo = trial.suggest_float("signed_pred_min", 0.00, 0.20)
        sp_hi = trial.suggest_float("signed_pred_max", sp_lo + 0.01, 0.40)
        max_sev = trial.suggest_float("max_p_severe", 0.05, 1.0)
        max_ext = trial.suggest_float("max_p_extreme", 0.10, 1.0)
        max_mod = trial.suggest_float("max_p_moderate", 0.10, 1.0)
        max_ttc = trial.suggest_int("max_t_to_close_s", 15, 300)
        min_ttc = trial.suggest_int("min_t_to_close_s", 0, 30)
        min_fp = trial.suggest_float("min_fill_prob", 0.0, 0.95)
        quote_size = trial.suggest_int("quote_size", 1, 5)
        filt = IntentFilter(
            signed_pred_min=sp_lo, signed_pred_max=sp_hi,
            max_p_severe=max_sev, max_p_extreme=max_ext, max_p_moderate=max_mod,
            max_t_to_close_s=max_ttc, min_t_to_close_s=min_ttc,
            min_fill_prob=min_fp, quote_size=quote_size,
            max_concurrent_positions=10**6,
        )
        per_week = evaluate(filt, signals, weeks, args.intents_per_day, args.margin_cap)
        grade = grade_strategy(per_week)

        tdir = out_root / f"trial_{trial.number:03d}"
        tdir.mkdir(parents=True, exist_ok=True)
        params = {"signed_pred_min": sp_lo, "signed_pred_max": sp_hi,
                  "max_p_severe": max_sev, "max_p_extreme": max_ext, "max_p_moderate": max_mod,
                  "max_t_to_close_s": max_ttc, "min_t_to_close_s": min_ttc,
                  "min_fill_prob": min_fp, "quote_size": quote_size}
        (tdir / "params.json").write_text(json.dumps(params, indent=2))
        (tdir / "grade.json").write_text(json.dumps(grade, indent=2))
        (tdir / "per_week.json").write_text(json.dumps(per_week, indent=2, default=str))

        elapsed = time.time() - t0
        with lb.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [trial.number, f"{grade['score']:.4f}", grade["rejected"],
                 f"{grade['mean_calmar_clean']:.4f}",
                 f"{grade['median_net_pnl_usd']:.4f}",
                 f"{grade['mean_net_pnl_usd']:.4f}",
                 f"{grade['p10_net_pnl_usd']:.4f}",
                 grade["n_weeks_with_positive_pnl"],
                 f"{grade['disqualification_rate']:.3f}",
                 f"{elapsed:.1f}", json.dumps(params)])

        print(f"[trial {trial.number:03d}] score={grade['score']:+.3f}  "
              f"medPnL=${grade['median_net_pnl_usd']:+.2f}  "
              f"meanPnL=${grade['mean_net_pnl_usd']:+.2f}  "
              f"calmar={grade['mean_calmar_clean']:+.2f}  "
              f"posWk={grade['n_weeks_with_positive_pnl']}/{grade['n_weeks']}  "
              f"disq={grade['disqualification_rate']:.2f}  "
              f"({elapsed:.0f}s)", file=sys.stderr)
        return grade["score"]

    print(f"[search] Optuna TPE: {args.trials} trials × {args.weeks} weeks", file=sys.stderr)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"intent_{ts}",
                                storage=f"sqlite:///{out_root / 'study.db'}",
                                load_if_exists=False)
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True,
                   show_progress_bar=False)

    print("\n" + "=" * 70, file=sys.stderr)
    print("SEARCH COMPLETE", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    best = study.best_trial
    print(f"  Best trial #{best.number}  score={best.value:+.3f}", file=sys.stderr)
    for k, v in best.params.items():
        print(f"    {k} = {v}", file=sys.stderr)

    # Final grade on a larger set of MC weeks
    if args.final_weeks > args.weeks:
        print(f"\n[final] grading winner on {args.final_weeks} MC weeks ...", file=sys.stderr)
        bigger_weeks = generate_mc_weeks(n_weeks=args.final_weeks, seed=args.seed)
        bp = best.params
        winner_filt = IntentFilter(
            signed_pred_min=bp["signed_pred_min"], signed_pred_max=bp["signed_pred_max"],
            max_p_severe=bp["max_p_severe"], max_p_extreme=bp["max_p_extreme"],
            max_p_moderate=bp["max_p_moderate"],
            max_t_to_close_s=bp["max_t_to_close_s"], min_t_to_close_s=bp["min_t_to_close_s"],
            min_fill_prob=bp["min_fill_prob"], quote_size=bp["quote_size"],
            max_concurrent_positions=10**6,
        )
        per_week = evaluate(winner_filt, signals, bigger_weeks,
                            args.intents_per_day, args.margin_cap)
        final_grade = grade_strategy(per_week)
        (out_root / "final_grade.json").write_text(json.dumps(final_grade, indent=2))
        (out_root / "final_per_week.json").write_text(json.dumps(per_week, indent=2, default=str))
        print(f"[final] score={final_grade['score']:+.3f}  "
              f"medPnL=${final_grade['median_net_pnl_usd']:+.2f}  "
              f"meanPnL=${final_grade['mean_net_pnl_usd']:+.2f}  "
              f"posWk={final_grade['n_weeks_with_positive_pnl']}/{final_grade['n_weeks']}  "
              f"disq={final_grade['disqualification_rate']:.2f}", file=sys.stderr)

    (out_root / "best_params.json").write_text(json.dumps({
        "trial": best.number, "score": best.value, "params": best.params}, indent=2))
    print(f"\n  Output dir: {out_root}", file=sys.stderr)


if __name__ == "__main__":
    main()
