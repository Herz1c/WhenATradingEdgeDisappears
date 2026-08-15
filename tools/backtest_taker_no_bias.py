"""Re-evaluate the no-bias calibrated model as a TAKER execution model
instead of maker. Compare apples-to-apples: same OOS days, same model,
same refined entry rule.

Taker assumptions:
  - Buying side's best-ask is paid in full (cross the spread).
  - Fill is GUARANTEED whenever a signal fires (no fill-rate uncertainty).
  - Polymarket taker fee: 0.072 * price * (1 - price) per share.
  - 1-share entries, same per-market caps (2 per side, 10s gap).

For each fill:
  pnl = payoff(1 or 0) - entry_price - taker_fee
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

ART_DIR = REPO / "artifacts_cleaned" / "model_no_bias_v1"
DC = REPO / "data" / "datasets" / "resolution_snapshot_dataset_v1_dense_close_no_bias"
TEST_DAYS = ["2026-05-21", "2026-05-22", "2026-05-26", "2026-05-29"]

TTC_MIN_S, TTC_MAX_S = 10.0, 60.0
MIN_PRICE = 0.30
MAX_POS = 2
MIN_GAP_S = 10.0
SIZE = 1.0


def taker_fee_per_share(price: float) -> float:
    """Polymarket taker fee model: 0.072 * price * (1 - price)."""
    return 0.072 * price * (1.0 - price)


def load_split(dates):
    parts = []
    for d in dates:
        p = DC / f"{d}.parquet"
        if not p.exists(): continue
        df = pl.read_parquet(p)
        df = df.filter(pl.col("resolved_side_label").is_not_null())
        df = df.filter(pl.col("up_token_best_bid").is_not_null())
        df = df.filter(pl.col("up_token_best_ask").is_not_null())
        df = df.filter(pl.col("down_token_best_bid").is_not_null())
        df = df.filter(pl.col("down_token_best_ask").is_not_null())
        df = df.filter(pl.col("up_token_best_ask") > 0.01)
        df = df.filter(pl.col("up_token_best_bid") < 0.99)
        df = df.filter((pl.col("t_to_close_s") > TTC_MIN_S) &
                       (pl.col("t_to_close_s") < TTC_MAX_S))
        df = df.with_columns(pl.lit(d).alias("date"))
        parts.append(df)
    return pl.concat(parts, how="diagonal").sort("snapshot_ts_ns")


def build_X(df, feats):
    X = np.zeros((len(df), len(feats)), dtype=np.float32)
    cols = set(df.columns)
    for i, f in enumerate(feats):
        if f not in cols: continue
        s = df.get_column(f)
        if not s.dtype.is_numeric(): continue
        v = s.cast(pl.Float32, strict=False).fill_null(0.0).fill_nan(0.0).to_numpy()
        X[:, i] = np.where(np.isfinite(v), v, 0.0)
    return X


def simulate_taker(df, p_up, *, threshold=0.05, min_price=MIN_PRICE,
                   max_pos=MAX_POS, min_gap_s=MIN_GAP_S):
    """Both-sides taker. Buy at the ask of the chosen side. Fill guaranteed.
    Returns rich per-fill records so we can post-filter."""
    up_ask = df["up_token_best_ask"].to_numpy().astype(float)
    up_bid = df["up_token_best_bid"].to_numpy().astype(float)
    dn_ask = df["down_token_best_ask"].to_numpy().astype(float)
    dn_bid = df["down_token_best_bid"].to_numpy().astype(float)
    ttc = df["t_to_close_s"].to_numpy().astype(float)
    resolved = df["resolved_side_label"].to_numpy().astype(int)
    snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
    slugs = df["market_slug"].to_numpy()
    dates = df["date"].to_numpy()
    delta_raw = df["delta_to_strike_raw"].to_numpy().astype(float)

    edge_up = p_up - up_ask
    edge_dn = up_bid - p_up

    pos_per_mkt_side = defaultdict(int)
    last_entry_ns = {}
    gap_ns = int(min_gap_s * 1e9)
    fills = []

    for i in range(len(df)):
        eu = edge_up[i]; ed = edge_dn[i]
        if eu < threshold and ed < threshold: continue
        side = "UP" if eu >= ed else "DOWN"
        if side == "UP" and eu < threshold: continue
        if side == "DOWN" and ed < threshold: continue
        entry_price = up_ask[i] if side == "UP" else dn_ask[i]
        if not (min_price <= entry_price < 1.0): continue
        slug = str(slugs[i])
        key = (slug, side)
        if pos_per_mkt_side[key] >= max_pos: continue
        last = last_entry_ns.get(key)
        if last is not None and (int(snap_ts[i]) - last) < gap_ns: continue
        pos_per_mkt_side[key] += 1
        last_entry_ns[key] = int(snap_ts[i])

        # Taker: fill guaranteed at the ask price.
        won = (resolved[i] == 1) if side == "UP" else (resolved[i] == 0)
        payoff = 1.0 if won else 0.0
        fee = taker_fee_per_share(entry_price)
        pnl = (payoff - entry_price - fee) * SIZE

        fills.append({
            "date": str(dates[i]),
            "slug": slug,
            "side": side,
            "ttc": float(ttc[i]),
            "entry_price": float(entry_price),
            "edge": float(eu if side == "UP" else ed),
            "p_up": float(p_up[i]),
            "delta_raw": float(delta_raw[i]) if np.isfinite(delta_raw[i]) else 0.0,
            "fee": float(fee),
            "won": int(won),
            "pnl": float(pnl),
        })
    return fills


def apply_refined_rule(fills):
    """Skip UP at entry_price in [0.36, 0.46) AND skip raw_delta in [51, 86]."""
    out = []
    for f in fills:
        if f["side"] == "UP" and 0.36 <= f["entry_price"] < 0.46:
            continue
        if 51.0 <= f["delta_raw"] <= 86.0:
            continue
        out.append(f)
    return out


def summarize(fills, label):
    if not fills:
        print(f"  {label}: 0 fills"); return
    wins = sum(f["won"] for f in fills)
    pnls = np.array([f["pnl"] for f in fills])
    print(f"  {label}: n={len(fills):>4} win={wins/len(fills):>5.1%} "
          f"avg=${pnls.mean():>+.4f} total=${pnls.sum():>+.2f}")


def print_per_day(fills, label):
    print(f"\n=== {label} per-day ===")
    print(f"  {'date':<12} {'fills':>6} {'win%':>6} {'avg_fee':>8} "
          f"{'avg_entry':>9} {'avg_pnl':>9} {'total':>9}")
    for d in TEST_DAYS:
        sub = [f for f in fills if f["date"] == d]
        if not sub:
            print(f"  {d:<12} 0"); continue
        wins = sum(f["won"] for f in sub)
        pnls = np.array([f["pnl"] for f in sub])
        fees = np.array([f["fee"] for f in sub])
        entries = np.array([f["entry_price"] for f in sub])
        print(f"  {d:<12} {len(sub):>6} {wins/len(sub)*100:>5.1f}% "
              f"${fees.mean():>+.4f} ${entries.mean():>+.3f} "
              f"${pnls.mean():>+.4f} ${pnls.sum():>+.2f}")


def main():
    model = joblib.load(ART_DIR / "model.pkl")
    feats = json.loads((ART_DIR / "experiment_manifest.json").read_text())["features"]
    cal = getattr(model, "_calibrator", None)
    df = load_split(TEST_DAYS)
    df_clean = clean_features(df)
    X = build_X(df_clean, feats)
    p_up = cal.predict(model.predict_proba(X)[:, 1]) if cal else model.predict_proba(X)[:, 1]

    # ── Threshold sweep, taker, both sides, no refined rule yet ──────────
    print(f"=== TAKER threshold sweep (both sides, no extra rule) ===")
    print(f"{'thr':>5} {'sig':>5} {'win%':>5} {'avg_entry':>9} {'avg_fee':>8}"
          f" {'avg_pnl':>8} {'total':>9}"
          f" | UP_n={'':>3} UP_w%={'':>4} UP_pnl={'':>7}"
          f" | DN_n={'':>3} DN_w%={'':>4} DN_pnl={'':>7}")
    print("-" * 130)
    for thr in [0.02, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]:
        fills = simulate_taker(df, p_up, threshold=thr)
        if not fills:
            print(f"{thr:>5.2f} 0"); continue
        wins = sum(f["won"] for f in fills)
        pnls = np.array([f["pnl"] for f in fills])
        fees = np.array([f["fee"] for f in fills])
        entries = np.array([f["entry_price"] for f in fills])
        ups = [f for f in fills if f["side"] == "UP"]
        dns = [f for f in fills if f["side"] == "DOWN"]
        u_pnl = sum(f["pnl"] for f in ups); u_w = sum(f["won"] for f in ups)
        d_pnl = sum(f["pnl"] for f in dns); d_w = sum(f["won"] for f in dns)
        print(f"{thr:>5.2f} {len(fills):>5d} {wins/len(fills):>5.1%} "
              f"${entries.mean():>+.3f} ${fees.mean():>+.4f} "
              f"${pnls.mean():>+.4f} ${pnls.sum():>+8.2f}"
              f" | UP_n={len(ups):>3} UP_w%={u_w/max(1,len(ups))*100:>3.0f}% UP_pnl=${u_pnl:>+6.2f}"
              f" | DN_n={len(dns):>3} DN_w%={d_w/max(1,len(dns))*100:>3.0f}% DN_pnl=${d_pnl:>+6.2f}")

    # ── Apply refined rule at threshold 0.05 ─────────────────────────────
    print(f"\n=== TAKER + REFINED RULE (skip UP@[0.36,0.46), skip raw_delta in [51,86]) ===")
    for thr in [0.02, 0.05, 0.07, 0.10, 0.15]:
        fills_all = simulate_taker(df, p_up, threshold=thr)
        fills_ref = apply_refined_rule(fills_all)
        print(f"\n--- threshold = {thr} ---")
        summarize(fills_all, "baseline (no rule)")
        summarize(fills_ref, "refined")
        if fills_ref:
            print_per_day(fills_ref, f"REFINED @ thr={thr}")

    # ── Compare maker vs taker on the same recommended config ───────────
    print(f"\n=== HEAD-TO-HEAD vs MAKER at the recommended config ===")
    print(f"  metric                       maker         taker(refined,thr=0.05)")
    fills_t = simulate_taker(df, p_up, threshold=0.05)
    fills_t = apply_refined_rule(fills_t)
    n_t = len(fills_t)
    if n_t > 0:
        w_t = sum(f["won"] for f in fills_t)
        pnl_t = sum(f["pnl"] for f in fills_t)
        ent_t = np.mean([f["entry_price"] for f in fills_t])
        fee_t = np.mean([f["fee"] for f in fills_t])
        # Maker numbers from previous run (hard-coded from output):
        # n=507, 87.6% win, +$93.97, avg_entry=$0.59, fee=$0
        print(f"  fills                        507           {n_t}")
        print(f"  win rate                     87.6%         {w_t/n_t:.1%}")
        print(f"  avg entry                    $0.590        ${ent_t:.3f}")
        print(f"  avg fee/share                $0.000        ${fee_t:.4f}")
        print(f"  total PnL ($1 size)          $+93.97       ${pnl_t:+.2f}")
        print(f"  $/fill                       $+0.185       ${pnl_t/n_t:+.4f}")
        print(f"  daily PnL avg                $+23.49       ${pnl_t/4:+.2f}")


if __name__ == "__main__":
    main()
