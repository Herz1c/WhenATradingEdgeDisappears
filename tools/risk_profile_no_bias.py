"""Deep risk-profile analysis of the no-bias model on OOS test data.

For the recommended threshold, report:
  - markets per day vs fills per day (saturation rate)
  - distribution of fills per market
  - entry-price histogram + summary
  - edge histogram at entry
  - per-trade PnL distribution (best, worst, median)
  - max drawdown (sequential trade ordering)
  - longest losing/winning streak
  - PnL concentration: top N% of trades' contribution
  - per-day breakdown
  - up-side vs down-side comparison
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, Counter
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
THRESHOLD = 0.05      # the recommended one from training
MIN_PRICE = 0.30
MAX_POS = 2
MIN_GAP_S = 10.0
TTC_MIN_S, TTC_MAX_S = 10.0, 60.0
SIZE = 1.0


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


def simulate_and_record_fills(df, p_up, *, threshold=THRESHOLD):
    """Same as the maker_sim_both_sides but records every single fill so
    we can analyse the distribution afterward."""
    up_bid = df["up_token_best_bid"].to_numpy().astype(float)
    up_ask = df["up_token_best_ask"].to_numpy().astype(float)
    dn_bid = df["down_token_best_bid"].to_numpy().astype(float)
    dn_ask = df["down_token_best_ask"].to_numpy().astype(float)
    ttc = df["t_to_close_s"].to_numpy().astype(float)
    resolved = df["resolved_side_label"].to_numpy().astype(int)
    snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
    close_ts = df["market_close_ts_ns"].to_numpy().astype(np.int64)
    slugs = df["market_slug"].to_numpy()
    dates = df["date"].to_numpy()

    edge_up = p_up - up_ask
    edge_dn = up_bid - p_up

    fut_up_ask = defaultdict(list)
    fut_dn_ask = defaultdict(list)
    for i in range(len(df)):
        fut_up_ask[str(slugs[i])].append((int(snap_ts[i]), float(up_ask[i])))
        fut_dn_ask[str(slugs[i])].append((int(snap_ts[i]), float(dn_ask[i])))

    pos_per_mkt_side = defaultdict(int)
    last_entry_ns = {}
    gap_ns = int(MIN_GAP_S * 1e9)

    fills = []   # list of dicts: each fill has all the info we want to analyse

    for i in range(len(df)):
        eu = edge_up[i]; ed = edge_dn[i]
        if eu < threshold and ed < threshold: continue
        side = "UP" if eu >= ed else "DOWN"
        if side == "UP" and eu < threshold: continue
        if side == "DOWN" and ed < threshold: continue
        bid = up_bid[i] if side == "UP" else dn_bid[i]
        ask = up_ask[i] if side == "UP" else dn_ask[i]
        if not (MIN_PRICE <= ask < 1.0): continue
        if not (0 < bid < ask): continue
        slug = str(slugs[i])
        key = (slug, side)
        if pos_per_mkt_side[key] >= MAX_POS: continue
        last = last_entry_ns.get(key)
        if last is not None and (int(snap_ts[i]) - last) < gap_ns: continue
        limit = round((bid + ask) / 2.0, 2)
        if limit < MIN_PRICE: continue
        if not (0.02 <= limit <= 0.98): continue

        pos_per_mkt_side[key] += 1
        last_entry_ns[key] = int(snap_ts[i])

        fut = fut_up_ask[slug] if side == "UP" else fut_dn_ask[slug]
        filled = False
        for t, a in fut:
            if t <= int(snap_ts[i]): continue
            if t >= int(close_ts[i]): break
            if a <= limit:
                filled = True; break
        if not filled: continue
        won = (resolved[i] == 1) if side == "UP" else (resolved[i] == 0)
        edge_at_entry = eu if side == "UP" else ed
        pnl = (1.0 if won else 0.0) - limit
        fills.append({
            "date": str(dates[i]),
            "slug": slug,
            "side": side,
            "ttc": float(ttc[i]),
            "snap_ts_ns": int(snap_ts[i]),
            "limit": limit,
            "edge": float(edge_at_entry),
            "p_up_calib": float(p_up[i]),
            "won": won,
            "pnl": float(pnl) * SIZE,
        })
    return fills


def pct(arr, q): return float(np.percentile(arr, q)) if len(arr) else float("nan")


def main():
    print(f"loading model from {ART_DIR}")
    model = joblib.load(ART_DIR / "model.pkl")
    # IMPORTANT: must use the feature ORDER from training (manifest), not
    # the importance-sorted order in feature_importance.json -- the
    # LightGBM model is sensitive to column position.
    feats = json.loads((ART_DIR / "experiment_manifest.json").read_text())["features"]
    cal = getattr(model, "_calibrator", None)
    if cal is None:
        print("WARNING: model has no calibrator attached")

    print(f"loading TEST days: {TEST_DAYS}")
    df = load_split(TEST_DAYS)
    print(f"  {len(df)} snapshots, {df['market_slug'].n_unique()} unique markets")
    df_clean = clean_features(df)
    X = build_X(df_clean, feats)
    raw = model.predict_proba(X)[:, 1]
    p_up = cal.predict(raw) if cal is not None else raw

    fills = simulate_and_record_fills(df, p_up, threshold=THRESHOLD)
    if not fills:
        print("no fills."); return

    # ── Macro-level stats ────────────────────────────────────────────────
    n = len(fills)
    n_win = sum(1 for f in fills if f["won"])
    pnl_total = sum(f["pnl"] for f in fills)
    print(f"\n=== MACRO  (threshold={THRESHOLD}) ===")
    print(f"  total fills           : {n}")
    print(f"  win rate              : {n_win/n:.1%}")
    print(f"  total PnL             : ${pnl_total:+.2f}")
    print(f"  per-fill PnL          : ${pnl_total/n:+.4f}")

    # ── Saturation: markets vs fills ─────────────────────────────────────
    markets_total = df["market_slug"].n_unique()
    markets_traded = len({f["slug"] for f in fills})
    fills_per_market = Counter(f["slug"] for f in fills)
    print(f"\n=== MARKET SATURATION ===")
    print(f"  unique markets in test       : {markets_total}")
    print(f"  unique markets traded        : {markets_traded} ({markets_traded/markets_total:.1%})")
    print(f"  fills per traded market      : "
          f"min={min(fills_per_market.values())} "
          f"median={int(np.median(list(fills_per_market.values())))} "
          f"max={max(fills_per_market.values())} "
          f"mean={n/markets_traded:.2f}")
    print(f"  (max possible per market = 2 per side x 2 sides = 4)")
    # Histogram of fills per market
    fpm_hist = Counter(fills_per_market.values())
    print(f"  fills-per-market histogram:")
    for k in sorted(fpm_hist):
        print(f"    {k} fills: {fpm_hist[k]} markets")

    # ── Entry price + edge ────────────────────────────────────────────────
    limits = np.array([f["limit"] for f in fills])
    edges  = np.array([f["edge"]  for f in fills])
    pnls   = np.array([f["pnl"]   for f in fills])
    print(f"\n=== ENTRY PRICE ===")
    print(f"  mean=${limits.mean():.3f}  median=${np.median(limits):.3f}"
          f"  p10=${pct(limits,10):.3f}  p25=${pct(limits,25):.3f}"
          f"  p75=${pct(limits,75):.3f}  p90=${pct(limits,90):.3f}")
    print(f"\n=== EDGE AT ENTRY (= calibrated p_up - market_ask for chosen side) ===")
    print(f"  mean={edges.mean():.4f}  median={np.median(edges):.4f}"
          f"  p25={pct(edges,25):.4f}  p75={pct(edges,75):.4f}"
          f"  p90={pct(edges,90):.4f}  max={edges.max():.4f}")
    print(f"\n=== PER-TRADE PNL ===")
    print(f"  mean=${pnls.mean():+.4f}  median=${np.median(pnls):+.4f}")
    print(f"  worst=${pnls.min():+.4f}  best=${pnls.max():+.4f}")
    print(f"  std=${pnls.std():.4f}  sharpe=PnL_mean/PnL_std = {pnls.mean()/max(pnls.std(),1e-9):.3f}")

    # ── Drawdown analysis on sequentially-ordered fills ──────────────────
    fills_sorted = sorted(fills, key=lambda f: f["snap_ts_ns"])
    cum = np.cumsum([f["pnl"] for f in fills_sorted])
    peak = np.maximum.accumulate(cum)
    dd = cum - peak  # negative drawdown
    print(f"\n=== DRAWDOWN (sequential fills, $1 size) ===")
    print(f"  final equity (after all {len(fills_sorted)} trades): ${cum[-1]:+.2f}")
    print(f"  peak equity                                          : ${peak.max():+.2f}")
    print(f"  max drawdown                                         : ${dd.min():.2f}")
    print(f"  peak-to-trough drawdown                              : ${peak.max() - cum[np.argmin(dd)]:.2f}")

    # ── Win/loss streaks ─────────────────────────────────────────────────
    cur_w = cur_l = max_w = max_l = 0
    for f in fills_sorted:
        if f["won"]:
            cur_w += 1; cur_l = 0
            max_w = max(max_w, cur_w)
        else:
            cur_l += 1; cur_w = 0
            max_l = max(max_l, cur_l)
    print(f"\n=== STREAKS ===")
    print(f"  longest winning streak : {max_w}")
    print(f"  longest losing streak  : {max_l}")

    # ── PnL concentration ────────────────────────────────────────────────
    abs_pnl = np.abs(pnls)
    rank = np.argsort(-abs_pnl)
    top10_pct = max(1, int(n * 0.10))
    top10_contribution = pnls[rank[:top10_pct]].sum()
    top1_pct = max(1, int(n * 0.01))
    top1_contribution = pnls[rank[:top1_pct]].sum()
    print(f"\n=== CONCENTRATION ===")
    print(f"  top  1%% of trades by |PnL| ({top1_pct} fills) contribute ${top1_contribution:+.2f}"
          f" ({top1_contribution/pnl_total*100:+.1f}% of total)")
    print(f"  top 10%% of trades by |PnL| ({top10_pct} fills) contribute ${top10_contribution:+.2f}"
          f" ({top10_contribution/pnl_total*100:+.1f}% of total)")

    # ── Side breakdown ───────────────────────────────────────────────────
    print(f"\n=== UP vs DOWN ===")
    for side in ("UP", "DOWN"):
        sf = [f for f in fills if f["side"] == side]
        if not sf: continue
        s_pnls = np.array([f["pnl"] for f in sf])
        s_limits = np.array([f["limit"] for f in sf])
        s_wins = sum(1 for f in sf if f["won"])
        print(f"  {side:<5}: n={len(sf)}  win={s_wins/len(sf):.1%}  "
              f"avg_entry=${s_limits.mean():.3f}  median_entry=${np.median(s_limits):.3f}  "
              f"avg_pnl=${s_pnls.mean():+.4f}  total=${s_pnls.sum():+.2f}")

    # ── Per-day summary ──────────────────────────────────────────────────
    print(f"\n=== PER DAY ===")
    print(f"  {'date':<12} {'markets':>8} {'fills':>6} {'win%':>5} {'avg_entry':>9} "
          f"{'avg_pnl':>8} {'total':>8}")
    for d in TEST_DAYS:
        df_d = [f for f in fills if f["date"] == d]
        if not df_d:
            print(f"  {d:<12} {'-':>8} {0:>6} {'-':>5} {'-':>9} {'-':>8} {'$0':>8}")
            continue
        unique_mkt = len({f["slug"] for f in df_d})
        wins = sum(1 for f in df_d if f["won"])
        d_pnls = np.array([f["pnl"] for f in df_d])
        d_lim = np.array([f["limit"] for f in df_d])
        print(f"  {d:<12} {unique_mkt:>8} {len(df_d):>6} {wins/len(df_d):>4.1%} "
              f"${d_lim.mean():>7.3f} ${d_pnls.mean():>+6.4f} ${d_pnls.sum():>+7.2f}")


if __name__ == "__main__":
    main()
