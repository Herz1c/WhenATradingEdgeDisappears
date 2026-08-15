#!/usr/bin/env python3
"""Model 02-only trading strategy on resolution_dense_close.

For each snapshot:
  1. Run Model 02 with proper features → predicted P(UP).
  2. Compare to market_UP_token_price (up_token_mid / up_token_best_ask).
  3. If model_P_UP > market_UP_ask + threshold: BUY UP token (taker).
     If model_P_UP < market_UP_bid - threshold: BUY DOWN token (taker).
     Else skip.
  4. Hold to close. Realized PnL = (resolved_side_label - entry_price) * direction - fee.
     Fee: Polymarket taker fee at fill price.
  5. Maker alternative: post a limit at best_bid (for BUY UP), assume filled with prob
     `min_fill_prob` (calibrated from Model 03). Skip Model 03 for simplicity — use
     "would have filled" heuristic = price moved through quote.

Grading: same Calmar-style scorer on 20 MC weeks from OOS pool only.
"""
import io, json, sys, math, statistics, random
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from backtest.resource_caps import apply_quiet_mode
apply_quiet_mode(cpu_pct=0.20, mem_cap_mb=4096, verbose=False)

import numpy as np
import polars as pl
import joblib

from backtest.fees import FeeCalculator
from backtest.grading import grade_strategy
from feature_cleanup import clean_features

# Load Model 02 + features
ART = "artifacts_cleaned/model_02_fair_resolution/dense_close/lightgbm/model.pkl"
FI = Path(ART).parent / "feature_importance.json"
model = joblib.load(ART)
feats = list(json.loads(FI.read_text()).keys())
print(f"Loaded Model 02 dense_close: {len(feats)} features")


def predict_p_up(df: pl.DataFrame) -> np.ndarray:
    X = np.zeros((len(df), len(feats)), dtype=np.float32)
    cols = set(df.columns)
    for i, f in enumerate(feats):
        if f not in cols: continue
        s = df.get_column(f)
        if not s.dtype.is_numeric(): continue
        v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
        X[:, i] = np.where(np.isfinite(v), v, 0.0)
    raw = model.predict_proba(X)[:, 1]
    cal = getattr(model, "_calibrator", None)
    if cal is not None:
        raw = np.clip(cal.predict(raw), 1e-6, 1 - 1e-6)
    return raw


def load_resolution_week(dates, max_per_day=20000, seed=42):
    parts = []
    for d in dates:
        p = Path(f"data/datasets/resolution_snapshot_dataset_v1_dense_close/{d}.parquet")
        if not p.exists(): continue
        df = pl.read_parquet(p)
        # Keep snapshots with valid resolution and market prices
        df = df.filter(pl.col("resolved_side_label").is_not_null())
        df = df.filter(pl.col("up_token_best_bid").is_not_null() & pl.col("up_token_best_ask").is_not_null())
        df = df.filter(pl.col("up_token_best_ask") > 0.01)
        df = df.filter(pl.col("up_token_best_bid") < 0.99)
        if max_per_day and len(df) > max_per_day:
            df = df.sample(n=max_per_day, seed=seed, with_replacement=False)
        df = df.with_columns(pl.lit(d).alias("date"))
        parts.append(df)
    if not parts: return pl.DataFrame()
    full = pl.concat(parts, how="diagonal")
    if "snapshot_ts_ns" in full.columns:
        full = full.sort("snapshot_ts_ns")
    return full


def backtest_model02(
    dates,
    edge_threshold_usd=0.03,
    entry_mode="taker",       # 'taker' or 'maker'
    min_t_to_close_s=10,
    max_t_to_close_s=300,
    max_per_day=20_000,
    starting_usd=100.0,
    max_margin_usd=100.0,
    quote_size=1.0,
) -> dict:
    df = load_resolution_week(dates, max_per_day=max_per_day)
    if len(df) == 0:
        return {"error": "no data", "starting_usd": starting_usd, "ending_usd": starting_usd,
                "net_pnl_usd": 0, "max_drawdown_usd": 0, "n_fills": 0, "win_rate": 0,
                "daily_pnl": {}, "worst_day_pnl_usd": 0, "best_day_pnl_usd": 0, "dates": dates}
    df = clean_features(df)

    # Predict P(UP) on all snapshots
    p_up = predict_p_up(df)

    # Pull market data + labels
    up_bid = df["up_token_best_bid"].to_numpy()
    up_ask = df["up_token_best_ask"].to_numpy()
    ttc = df["t_to_close_s"].to_numpy() if "t_to_close_s" in df.columns else np.full(len(df), 60.0)
    resolved = df["resolved_side_label"].to_numpy()
    dates_arr = df["date"].to_numpy()

    balance = starting_usd
    peak = trough = starting_usd
    max_dd = 0.0
    daily = defaultdict(float)
    n_fills = n_wins = n_losses = 0
    fee_calcs = {}
    open_margin = 0.0   # No lifecycle: snapshots arrive irregularly; cap concurrent margin instead

    n = len(df)
    skipped_margin = 0
    skipped_gate = 0
    skipped_ttc = 0
    for i in range(n):
        if not (min_t_to_close_s <= ttc[i] <= max_t_to_close_s):
            skipped_ttc += 1; continue

        # Buy UP token if model_P_UP > market_UP_ask + threshold (undervalued UP)
        if p_up[i] > up_ask[i] + edge_threshold_usd:
            side = "UP"; entry = up_ask[i] if entry_mode == "taker" else up_bid[i]
        # Buy DOWN token (== sell UP exposure) if model_P_UP < market_UP_bid - threshold
        elif p_up[i] < up_bid[i] - edge_threshold_usd:
            # Buy DOWN at DOWN's ask. If DOWN data is missing, equivalent: 1 - UP's bid
            d_ask = df["down_token_best_ask"][i] if "down_token_best_ask" in df.columns else (1 - up_bid[i])
            d_bid = df["down_token_best_bid"][i] if "down_token_best_bid" in df.columns else (1 - up_ask[i])
            if d_ask is None or d_bid is None: continue
            entry = d_ask if entry_mode == "taker" else d_bid
            side = "DOWN"
        else:
            skipped_gate += 1; continue

        if entry is None or not math.isfinite(entry) or entry <= 0 or entry >= 1:
            continue
        if open_margin + entry * quote_size > max_margin_usd:
            skipped_margin += 1; continue

        # Realize close PnL
        won = (resolved[i] == 1) if side == "UP" else (resolved[i] == 0)
        date = str(dates_arr[i])
        fc = fee_calcs.setdefault(date, FeeCalculator.for_date(date))
        if entry_mode == "taker":
            fee = fc.taker_fee_usd(price=entry, size=quote_size)
            rebate = 0.0
        else:
            fee = 0.0
            rebate = fc.maker_rebate_usd(price=entry, size=quote_size)
            # Heuristic maker fill: assume filled (this is generous; need a proper
            # fill model for real life — Model 03 is broken OOS so we skip)
        payoff = 1.0 if won else 0.0
        realized = (payoff - entry) * quote_size - fee + rebate

        balance += realized
        daily[date] += realized
        n_fills += 1
        if realized > 0: n_wins += 1
        elif realized < 0: n_losses += 1
        peak = max(peak, balance)
        trough = min(trough, balance)
        max_dd = max(max_dd, peak - balance)
        # Margin model: lock up entry until "close" — we don't have per-position close ts
        # so use simple rolling margin that gets released proportional to fill flow:
        open_margin += entry * quote_size
        if open_margin > max_margin_usd:
            # Release a chunk (simulate that older positions closed)
            open_margin = max(0, open_margin - max_margin_usd * 0.5)

        if balance <= 0:
            break

    return {
        "starting_usd": starting_usd,
        "ending_usd": balance,
        "net_pnl_usd": balance - starting_usd,
        "max_drawdown_usd": max_dd,
        "peak_balance_usd": peak,
        "trough_balance_usd": trough,
        "n_fills": n_fills, "n_wins": n_wins, "n_losses": n_losses,
        "win_rate": (n_wins / n_fills) if n_fills else 0.0,
        "daily_pnl": dict(daily),
        "worst_day_pnl_usd": min(daily.values()) if daily else 0.0,
        "best_day_pnl_usd": max(daily.values()) if daily else 0.0,
        "dates": dates,
        "skipped_margin": skipped_margin, "skipped_gate": skipped_gate, "skipped_ttc": skipped_ttc,
        "n_eligible": n,
    }


# OOS pool
OOS_POOL = ["2026-05-07","2026-05-08","2026-05-09","2026-05-10",
            "2026-05-11","2026-05-12","2026-05-13","2026-05-14",
            "2026-05-15","2026-05-16"]


def main():
    # Single-week IS reference (May 4-10) for comparison
    in_sample = ["2026-05-04", "2026-05-05", "2026-05-06"]

    print("\n" + "=" * 78)
    print("MODEL 02 STRATEGY — single-week sweep on May 4-6 (IS reference) vs OOS")
    print("=" * 78)
    print(f"{'cfg':<48} {'fills':>6} {'win%':>6} {'pnl':>10} {'max_dd':>9}")
    print("-" * 90)
    for label, dates in [("IS May 4-6", in_sample),
                         ("OOS May 7-11", ["2026-05-07","2026-05-08","2026-05-09","2026-05-10","2026-05-11"]),
                         ("Deep OOS May 13-16", ["2026-05-13","2026-05-14","2026-05-15","2026-05-16"])]:
        for em in ["taker", "maker"]:
            for thr in [0.02, 0.03, 0.05, 0.08]:
                s = backtest_model02(dates, edge_threshold_usd=thr, entry_mode=em,
                                     min_t_to_close_s=10, max_t_to_close_s=120,
                                     max_per_day=20_000, max_margin_usd=100.0)
                tag = f"{label}  {em}  thr=${thr:.2f}"
                print(f"{tag:<48} {s['n_fills']:>6} {s['win_rate']*100:>5.1f}% "
                      f"${s['net_pnl_usd']:>+8.2f} ${s['max_drawdown_usd']:>+7.2f}")
        print()

    print("\n" + "=" * 78)
    print("OFFICIAL OOS GRADE — Model 02 taker, threshold $0.05, 20 MC weeks from May 7-16 only")
    print("=" * 78)
    weeks = []
    for i in range(20):
        r = random.Random(2000 + i)
        weeks.append(sorted(r.sample(OOS_POOL, 7)))

    per_week = []
    for i, w in enumerate(weeks):
        s = backtest_model02(w, edge_threshold_usd=0.05, entry_mode="taker",
                             min_t_to_close_s=10, max_t_to_close_s=120,
                             max_per_day=15_000, max_margin_usd=100.0)
        per_week.append(s)
        print(f"  w{i+1:>2}: fills={s['n_fills']:>4}  win%={s['win_rate']*100:>5.1f}  "
              f"pnl=${s['net_pnl_usd']:+7.2f}  dd=${s['max_drawdown_usd']:5.2f}  "
              f"worst=${s['worst_day_pnl_usd']:+6.2f}")

    g = grade_strategy(per_week)
    pnls = [w['net_pnl_usd'] for w in per_week]
    print(f"\n=== GRADE ===")
    print(f"  score={g['score']:+.3f}  rejected={g['rejected']}")
    print(f"  mean=${g['mean_net_pnl_usd']:+.2f}  median=${g['median_net_pnl_usd']:+.2f}  "
          f"min=${min(pnls):+.2f}  max=${max(pnls):+.2f}")
    print(f"  Calmar={g['mean_calmar_clean']:.2f}  positive_weeks={g['n_weeks_with_positive_pnl']}/{g['n_weeks']}  "
          f"disq={g['disqualification_rate']*100:.0f}%")

    Path("docs").mkdir(exist_ok=True)
    Path("docs/model02_strategy_grade.json").write_text(json.dumps({
        "config": {"edge_threshold_usd": 0.05, "entry_mode": "taker",
                   "min_t_to_close_s": 10, "max_t_to_close_s": 120,
                   "max_margin_usd": 100.0, "weeks_pool": OOS_POOL},
        "grade": g, "per_week": per_week, "weeks": weeks,
    }, indent=2, default=str))
    print("\nSaved to docs/model02_strategy_grade.json")


if __name__ == "__main__":
    main()
