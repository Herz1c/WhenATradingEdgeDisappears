"""Day-by-day, price-bucket-by-price-bucket drilldown of the refined rule:
   skip UP at limit in [0.36, 0.46) AND skip raw_delta in [51, 86]

For each (date, price_bucket, side):
   - fills
   - win rate
   - average PnL
   - total PnL

Plus stratification by edge bucket to spot concentration risk.
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


def simulate_with_context(df, p_up, *, threshold=0.02, min_price=0.30,
                          max_pos=2, min_gap_s=10.0):
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
    delta_raw = df["delta_to_strike_raw"].to_numpy().astype(float)

    edge_up = p_up - up_ask
    edge_dn = up_bid - p_up
    fut_up_ask = defaultdict(list)
    fut_dn_ask = defaultdict(list)
    for i in range(len(df)):
        fut_up_ask[str(slugs[i])].append((int(snap_ts[i]), float(up_ask[i])))
        fut_dn_ask[str(slugs[i])].append((int(snap_ts[i]), float(dn_ask[i])))
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
        bid = up_bid[i] if side == "UP" else dn_bid[i]
        ask = up_ask[i] if side == "UP" else dn_ask[i]
        if not (min_price <= ask < 1.0): continue
        if not (0 < bid < ask): continue
        slug = str(slugs[i])
        key = (slug, side)
        if pos_per_mkt_side[key] >= max_pos: continue
        last = last_entry_ns.get(key)
        if last is not None and (int(snap_ts[i]) - last) < gap_ns: continue
        limit = round((bid + ask) / 2.0, 2)
        if limit < min_price: continue
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
            "limit": limit,
            "edge": float(edge_at_entry),
            "delta_raw": float(delta_raw[i]) if np.isfinite(delta_raw[i]) else 0.0,
            "won": int(won),
            "pnl": float(pnl) * SIZE,
        })
    return fills


# Price buckets
PRICE_BINS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


def bucket_idx(limit):
    for i in range(len(PRICE_BINS) - 1):
        if PRICE_BINS[i] <= limit < PRICE_BINS[i+1]:
            return i
    return len(PRICE_BINS) - 2


def apply_refined_rule(fills):
    """Skip UP at limit in [0.36, 0.46) AND skip raw_delta in [51, 86]."""
    out = []
    for f in fills:
        if f["side"] == "UP" and 0.36 <= f["limit"] < 0.46:
            continue
        if 51.0 <= f["delta_raw"] <= 86.0:
            continue
        out.append(f)
    return out


def print_grid(title, fills_filtered):
    """Print: per (date, price_bucket, side) -> n fills, win%, total PnL."""
    print(f"\n=== {title} ===")
    print(f"  {len(fills_filtered)} total fills")
    # Group
    by_cell = defaultdict(list)
    for f in fills_filtered:
        b = bucket_idx(f["limit"])
        by_cell[(f["date"], b, f["side"])].append(f)
    # Print table
    header = f"  {'date':<12} {'side':<5} | "
    for i in range(len(PRICE_BINS) - 1):
        header += f"{f'[{PRICE_BINS[i]:.2f}-{PRICE_BINS[i+1]:.2f})':<14}"
    header += f"{'TOTAL':<14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    grand_n = 0; grand_w = 0; grand_p = 0.0
    for date in TEST_DAYS:
        for side in ("UP", "DOWN"):
            row = f"  {date:<12} {side:<5} | "
            row_n = 0; row_w = 0; row_p = 0.0
            for b in range(len(PRICE_BINS) - 1):
                sub = by_cell.get((date, b, side), [])
                if sub:
                    n = len(sub)
                    w = sum(f["won"] for f in sub)
                    p = sum(f["pnl"] for f in sub)
                    cell = f"{n}|{w/n*100:.0f}%|${p:+.2f}"
                    row_n += n; row_w += w; row_p += p
                else:
                    cell = "-"
                row += f"{cell:<14}"
            if row_n > 0:
                row += f"{row_n}|{row_w/row_n*100:.0f}%|${row_p:+.2f}"
                grand_n += row_n; grand_w += row_w; grand_p += row_p
            else:
                row += "-"
            print(row)
        print("  " + "-" * (len(header) - 2))
    print(f"\n  TOTAL: {grand_n} fills, {grand_w}/{grand_n} = {grand_w/max(1,grand_n)*100:.1f}% win, "
          f"PnL ${grand_p:+.2f}")


def main():
    model = joblib.load(ART_DIR / "model.pkl")
    feats = json.loads((ART_DIR / "experiment_manifest.json").read_text())["features"]
    cal = getattr(model, "_calibrator", None)
    df = load_split(TEST_DAYS)
    df_clean = clean_features(df)
    X = build_X(df_clean, feats)
    p_up = cal.predict(model.predict_proba(X)[:, 1]) if cal else model.predict_proba(X)[:, 1]
    fills = simulate_with_context(df, p_up, threshold=0.02)

    print_grid("BASELINE (thr 0.02, no rule)", fills)
    refined = apply_refined_rule(fills)
    print_grid("REFINED RULE (skip UP@[0.36,0.46), skip raw_delta in [51,86])", refined)

    # Concentration on the refined rule: PnL by top-N markets
    print(f"\n=== concentration: top markets by |PnL| (refined rule) ===")
    by_market = defaultdict(list)
    for f in refined:
        by_market[f["slug"]].append(f)
    market_pnls = [(slug, sum(g["pnl"] for g in grp), len(grp))
                   for slug, grp in by_market.items()]
    market_pnls.sort(key=lambda kv: -abs(kv[1]))
    total = sum(p for _, p, _ in market_pnls)
    cum = 0
    print(f"  top 10 markets by |PnL|:")
    for slug, p, n in market_pnls[:10]:
        cum += p
        print(f"    {slug:<32}  fills={n}  PnL=${p:+.2f}  cum_share={cum/total*100:+.1f}%")

    # Day-summary table
    print(f"\n=== refined rule per-day summary ===")
    print(f"  {'date':<12} {'fills':>6} {'win%':>6} {'avg':>9} {'total':>9}"
          f"  {'UP_n':>5} {'UP_win%':>8} {'UP_pnl':>9}"
          f"  {'DN_n':>5} {'DN_win%':>8} {'DN_pnl':>9}")
    for date in TEST_DAYS:
        df_day = [f for f in refined if f["date"] == date]
        ups = [f for f in df_day if f["side"] == "UP"]
        dns = [f for f in df_day if f["side"] == "DOWN"]
        if not df_day:
            print(f"  {date:<12} 0"); continue
        d_pnl = sum(f["pnl"] for f in df_day)
        d_w = sum(f["won"] for f in df_day)
        u_pnl = sum(f["pnl"] for f in ups); u_w = sum(f["won"] for f in ups)
        d2_pnl = sum(f["pnl"] for f in dns); d2_w = sum(f["won"] for f in dns)
        print(f"  {date:<12} {len(df_day):>6} {d_w/len(df_day)*100:>5.1f}% "
              f"${d_pnl/len(df_day):>+7.4f} ${d_pnl:>+7.2f}"
              f"  {len(ups):>5} {u_w/max(1,len(ups))*100:>7.1f}% ${u_pnl:>+7.2f}"
              f"  {len(dns):>5} {d2_w/max(1,len(dns))*100:>7.1f}% ${d2_pnl:>+7.2f}")


if __name__ == "__main__":
    main()
