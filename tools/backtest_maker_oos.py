"""OOS maker-fill backtest for model 02.

Simulates posting a BUY LIMIT (POST_ONLY maker) for the DOWN token whenever
the model fires a DOWN signal, instead of crossing the book with a FAK.

Fill rule (per signal at snapshot t, market M, limit price P):
  We look at all subsequent dense_close snapshots of M in [t, market_close_ts).
  If at any future snapshot down_token_best_ask <= P, we count it as filled
  at price P. Otherwise the order is cancelled at close with no PnL.

  This is an OPTIMISTIC maker model -- it assumes:
    (a) zero submit latency (order is live at snapshot time),
    (b) we jump the head of queue if ask crosses our price.
  Real fills will be worse than this; the gap is the queue/latency premium
  a real maker pays. The point of the sim is to see whether the strategy
  has positive expectancy with maker economics AT ALL.

Edge gate: same edge_dn = up_bid - p_up used by the taker engine, so a
side-by-side comparison is apples-to-apples on the SIGNAL side; the only
thing that differs is the EXECUTION model.

Maker fee on Polymarket = 0 (sometimes a rebate; we model 0 to stay
pessimistic on fees but optimistic on fills -- net is a fair upper bound).

Usage:
  py -3 tools/backtest_maker_oos.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from feature_cleanup import clean_features  # noqa: E402

DATASET = REPO / "data" / "datasets" / "resolution_snapshot_dataset_v1_dense_close"
ART = REPO / "artifacts_cleaned" / "model_02_fair_resolution" / "dense_close" / "lightgbm" / "model.pkl"

# Pure OOS — these dates are NOT in train (2026-04-19..2026-05-13) and NOT
# in test (2026-05-15..2026-05-20). The model has never seen them.
OOS_DATES = ["2026-05-21", "2026-05-22", "2026-05-26", "2026-05-29"]

TTC_MIN_S = 10.0
TTC_MAX_S = 60.0
SIZE = 1.0          # 1 contract per signal — PnL ≈ dollars per signal
MAX_POS_PER_MARKET = 1


def load_data(dates: list[str]) -> pl.DataFrame:
    parts: list[pl.DataFrame] = []
    for d in dates:
        p = DATASET / f"{d}.parquet"
        if not p.exists():
            print(f"  missing dense_close for {d} -- skipping")
            continue
        df = pl.read_parquet(p)
        df = df.filter(pl.col("resolved_side_label").is_not_null())
        # Need both up + down book sides to be live so we can simulate limit posting
        df = df.filter(pl.col("up_token_best_bid").is_not_null())
        df = df.filter(pl.col("up_token_best_ask").is_not_null())
        df = df.filter(pl.col("down_token_best_bid").is_not_null())
        df = df.filter(pl.col("down_token_best_ask").is_not_null())
        df = df.filter(pl.col("up_token_best_ask") > 0.01)
        df = df.filter(pl.col("up_token_best_bid") < 0.99)
        df = df.with_columns(pl.lit(d).alias("date"))
        parts.append(df)
        print(f"  {d}: {len(df):>6} eligible snapshots")
    if not parts:
        raise SystemExit("no OOS data found")
    full = pl.concat(parts, how="diagonal").sort("snapshot_ts_ns")
    print(f"  TOTAL: {len(full):>6} snapshots, {full['market_slug'].n_unique()} markets")
    return full


def predict(model, feats: list[str], df: pl.DataFrame) -> np.ndarray:
    X = np.zeros((len(df), len(feats)), dtype=np.float32)
    cols = set(df.columns)
    for i, f in enumerate(feats):
        if f not in cols:
            continue
        s = df.get_column(f)
        if not s.dtype.is_numeric():
            continue
        v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
        X[:, i] = np.where(np.isfinite(v), v, 0.0)
    raw = model.predict_proba(X)[:, 1]
    cal = getattr(model, "_calibrator", None)
    if cal is not None:
        raw = np.clip(cal.predict(raw), 1e-6, 1 - 1e-6)
    return raw


def build_market_future_asks(df: pl.DataFrame) -> dict[str, list[tuple[int, float]]]:
    """Per-market sorted list of (snapshot_ts_ns, down_token_best_ask)
    so we can answer 'did the ask ever drop to <= P before close?' cheaply."""
    slugs = df["market_slug"].to_numpy()
    ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
    da = df["down_token_best_ask"].to_numpy().astype(float)
    idx: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for i in range(len(df)):
        idx[str(slugs[i])].append((int(ts[i]), float(da[i])))
    # df was already sorted by snapshot_ts_ns, so each list is sorted.
    return idx


def simulate(df: pl.DataFrame, p_up: np.ndarray,
             *,
             edge_threshold: float,
             limit_strategy: str,
             tick_size: float = 0.01,
             ) -> dict:
    """limit_strategy:
        'best_bid'              -- sit on top-of-bid (or below if someone's at our level)
        'best_ask_minus_tick'   -- jump inside the spread, become the best bid
        'midpoint'              -- post at (bid+ask)/2 rounded down to tick
        'best_bid_minus_tick'   -- 1 tick below best bid (queue position behind nothing)
    """
    up_bid = df["up_token_best_bid"].to_numpy().astype(float)
    dn_bid = df["down_token_best_bid"].to_numpy().astype(float)
    dn_ask = df["down_token_best_ask"].to_numpy().astype(float)
    ttc = df["t_to_close_s"].to_numpy().astype(float)
    resolved = df["resolved_side_label"].to_numpy().astype(int)
    snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
    close_ts = df["market_close_ts_ns"].to_numpy().astype(np.int64)
    slugs = df["market_slug"].to_numpy()
    dates = df["date"].to_numpy()

    edge_dn = up_bid - p_up

    future_ask = build_market_future_asks(df)
    pos_per_mkt: dict[str, int] = defaultdict(int)

    n_signal = n_filled = n_unfilled = n_win = n_loss = 0
    total_pnl = 0.0
    pnl_by_date: dict[str, float] = defaultdict(float)
    fills_by_date: dict[str, int] = defaultdict(int)

    for i in range(len(df)):
        if not (TTC_MIN_S <= ttc[i] <= TTC_MAX_S):
            continue
        if edge_dn[i] < edge_threshold:
            continue
        if dn_ask[i] <= tick_size or dn_ask[i] >= 1 - tick_size:
            continue
        slug = str(slugs[i])
        if pos_per_mkt[slug] >= MAX_POS_PER_MARKET:
            continue

        if limit_strategy == "best_bid":
            limit = dn_bid[i]
        elif limit_strategy == "best_ask_minus_tick":
            limit = round(dn_ask[i] - tick_size, 2)
        elif limit_strategy == "midpoint":
            limit = round((dn_bid[i] + dn_ask[i]) / 2.0, 2)
        elif limit_strategy == "best_bid_minus_tick":
            limit = round(dn_bid[i] - tick_size, 2)
        else:
            raise ValueError(limit_strategy)
        if limit <= 0 or limit >= 1:
            continue
        n_signal += 1
        pos_per_mkt[slug] += 1

        # Did the ask ever drop to <= limit before market close?
        t0 = int(snap_ts[i])
        tend = int(close_ts[i])
        filled = False
        for t, a in future_ask[slug]:
            if t <= t0:
                continue
            if t >= tend:
                break
            if a <= limit:
                filled = True
                break

        if not filled:
            n_unfilled += 1
            continue
        n_filled += 1
        won = (resolved[i] == 0)   # DOWN wins iff resolved_side_label == 0 (UP=1)
        payoff = 1.0 if won else 0.0
        pnl = (payoff - limit) * SIZE      # maker fee = 0 on Polymarket
        total_pnl += pnl
        date = str(dates[i])
        pnl_by_date[date] += pnl
        fills_by_date[date] += 1
        if won: n_win += 1
        else:   n_loss += 1

    return {
        "edge_threshold": edge_threshold,
        "limit_strategy": limit_strategy,
        "n_signals": n_signal,
        "n_filled": n_filled,
        "n_unfilled": n_unfilled,
        "fill_rate": n_filled / max(1, n_signal),
        "n_win": n_win, "n_loss": n_loss,
        "win_rate": n_win / max(1, n_filled),
        "total_pnl_usd": total_pnl,
        "avg_pnl_per_fill": total_pnl / max(1, n_filled),
        "pnl_by_date": dict(pnl_by_date),
        "fills_by_date": dict(fills_by_date),
    }


def main() -> None:
    print(f"loading OOS data: {OOS_DATES}")
    df = load_data(OOS_DATES)
    df = clean_features(df)

    print("\nloading model...")
    model = joblib.load(ART)
    feats = list(json.loads((ART.parent / "feature_importance.json").read_text()).keys())
    print(f"  {len(feats)} features, calibrator={'yes' if getattr(model,'_calibrator',None) else 'no'}")

    print("scoring snapshots...")
    p_up = predict(model, feats, df)
    edge_dn_all = df["up_token_best_bid"].to_numpy().astype(float) - p_up
    print(f"  edge_dn distribution: min={edge_dn_all.min():.3f} p25={np.percentile(edge_dn_all,25):.3f}"
          f" p50={np.percentile(edge_dn_all,50):.3f} p75={np.percentile(edge_dn_all,75):.3f}"
          f" p95={np.percentile(edge_dn_all,95):.3f} max={edge_dn_all.max():.3f}")

    print("\n" + "=" * 110)
    print(f"{'strategy':>22} | {'thr':>5} | {'signals':>7} | {'fills':>5} | {'fill%':>5}"
          f" | {'win%':>5} | {'PnL_$':>9} | {'PnL/fill':>8}")
    print("-" * 110)

    strategies = ["best_bid_minus_tick", "best_bid", "midpoint", "best_ask_minus_tick"]
    thresholds = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]

    all_results = []
    for strat in strategies:
        for thr in thresholds:
            r = simulate(df, p_up, edge_threshold=thr, limit_strategy=strat)
            all_results.append(r)
            print(f"{strat:>22} | {thr:>5.2f} | {r['n_signals']:>7d} | {r['n_filled']:>5d}"
                  f" | {r['fill_rate']:>5.1%} | {r['win_rate']:>5.1%} | {r['total_pnl_usd']:>+9.2f}"
                  f" | {r['avg_pnl_per_fill']:>+8.4f}")
        print("-" * 110)

    # Pick best by PnL with at least 10 fills (avoid silly winners on 1 fill)
    creditable = [r for r in all_results if r["n_filled"] >= 10]
    if creditable:
        best = max(creditable, key=lambda r: r["total_pnl_usd"])
        print(f"\nBEST (>= 10 fills): {best['limit_strategy']} @ thr={best['edge_threshold']:.2f}")
        print(f"  signals={best['n_signals']}  fills={best['n_filled']}  fill_rate={best['fill_rate']:.1%}")
        print(f"  win_rate={best['win_rate']:.1%}  total_pnl=${best['total_pnl_usd']:+.2f}"
              f"  per-fill=${best['avg_pnl_per_fill']:+.4f}")
        print(f"  pnl by date: {best['pnl_by_date']}")
        print(f"  fills by date: {best['fills_by_date']}")

    out = REPO / "logs" / "backtest_maker_oos.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull sweep written to {out}")


if __name__ == "__main__":
    main()
