"""Quick backtest of the CURRENT live config on OOS days:
  - threshold 0.20
  - both sides allowed
  - taker fills at ask + Polymarket fee
  - min price 0.40, max 0.90 (both sides)
  - skip UP in [0.36, 0.46) [historical refined rule, mostly subsumed by 0.40 floor now]
  - skip raw_delta in [+/-51, +/-86]
  - max 1 entry per (market, side); max 2 entries per market total
  - 10s gap between consecutive entries on same market
"""
from __future__ import annotations
import json, sys
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

# Current live config:
THRESHOLD            = 0.20
MIN_PRICE            = 0.40
MAX_PRICE            = 0.90
SKIP_UP_LO, SKIP_UP_HI = 0.36, 0.46
SKIP_DELTA_LO, SKIP_DELTA_HI = 51.0, 86.0
MAX_POS_PER_MARKET   = 2
MAX_POS_PER_SIDE     = 1
MIN_GAP_S            = 10.0
TTC_MIN_S, TTC_MAX_S = 10.0, 60.0
SIZE                 = 1.0


def taker_fee_per_share(price): return 0.072 * price * (1 - price)


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


def simulate(df, p_up):
    up_bid = df["up_token_best_bid"].to_numpy().astype(float)
    up_ask = df["up_token_best_ask"].to_numpy().astype(float)
    dn_bid = df["down_token_best_bid"].to_numpy().astype(float)
    dn_ask = df["down_token_best_ask"].to_numpy().astype(float)
    ttc = df["t_to_close_s"].to_numpy().astype(float)
    resolved = df["resolved_side_label"].to_numpy().astype(int)
    snap_ts = df["snapshot_ts_ns"].to_numpy().astype(np.int64)
    slugs = df["market_slug"].to_numpy()
    dates = df["date"].to_numpy()
    delta_raw = df["delta_to_strike_raw"].to_numpy().astype(float)

    edge_up = p_up - up_ask
    edge_dn = up_bid - p_up

    pos_per_mkt = defaultdict(int)
    pos_per_mkt_side: dict[tuple[str, str], int] = defaultdict(int)
    last_entry_ns: dict[str, int] = {}
    gap_ns = int(MIN_GAP_S * 1e9)
    fills = []

    for i in range(len(df)):
        eu, ed = edge_up[i], edge_dn[i]
        if eu < THRESHOLD and ed < THRESHOLD: continue
        side = "UP" if eu >= ed else "DOWN"
        if side == "UP" and eu < THRESHOLD: continue
        if side == "DOWN" and ed < THRESHOLD: continue

        # raw-delta danger zone
        if SKIP_DELTA_LO <= abs(delta_raw[i]) <= SKIP_DELTA_HI:
            continue

        slug = str(slugs[i])
        # per-market and per-side caps
        if pos_per_mkt[slug] >= MAX_POS_PER_MARKET: continue
        if pos_per_mkt_side[(slug, side)] >= MAX_POS_PER_SIDE: continue
        # 10s gap
        last = last_entry_ns.get(slug)
        if last is not None and (int(snap_ts[i]) - last) < gap_ns: continue

        # side-specific price gates + UP skip band
        if side == "UP":
            entry = up_ask[i]
            if SKIP_UP_LO <= entry < SKIP_UP_HI: continue
            if not (MIN_PRICE <= entry <= MAX_PRICE): continue
        else:
            entry = dn_ask[i]
            if not (MIN_PRICE <= entry <= MAX_PRICE): continue

        pos_per_mkt[slug] += 1
        pos_per_mkt_side[(slug, side)] += 1
        last_entry_ns[slug] = int(snap_ts[i])

        won = (resolved[i] == 1) if side == "UP" else (resolved[i] == 0)
        fee = taker_fee_per_share(entry)
        pnl = ((1.0 if won else 0.0) - entry - fee) * SIZE
        fills.append({"date": str(dates[i]), "side": side, "entry": entry,
                      "won": int(won), "pnl": pnl, "fee": fee})
    return fills


def main():
    model = joblib.load(ART_DIR / "model.pkl")
    feats = json.loads((ART_DIR / "experiment_manifest.json").read_text())["features"]
    cal = getattr(model, "_calibrator", None)
    print(f"loading OOS days: {TEST_DAYS}")
    df = load_split(TEST_DAYS)
    print(f"  {len(df)} eligible snapshots, {df['market_slug'].n_unique()} markets")
    df_clean = clean_features(df)
    X = build_X(df_clean, feats)
    p_up = cal.predict(model.predict_proba(X)[:, 1]) if cal else model.predict_proba(X)[:, 1]
    fills = simulate(df, p_up)
    if not fills: print("no fills"); return

    n = len(fills); wins = sum(f["won"] for f in fills)
    total = sum(f["pnl"] for f in fills)
    print(f"\n=== TOTAL ({len(TEST_DAYS)} days) ===")
    print(f"  fills:           {n}")
    print(f"  fills/day:       {n/len(TEST_DAYS):.1f}")
    print(f"  win rate:        {wins/n:.1%}")
    print(f"  total PnL ($1):  ${total:+.2f}")
    print(f"  per-fill PnL:    ${total/n:+.4f}")
    print(f"  daily avg ($1):  ${total/len(TEST_DAYS):+.2f}")
    print(f"  daily avg @ 5sh: ${5*total/len(TEST_DAYS):+.2f}")

    print(f"\n=== PER DAY ===")
    print(f"  {'date':<12} {'fills':>6} {'win%':>6} {'avg_entry':>10} {'$/fill':>8} {'total':>9}")
    for d in TEST_DAYS:
        sub = [f for f in fills if f["date"] == d]
        if not sub: print(f"  {d:<12} 0"); continue
        w = sum(f["won"] for f in sub)
        pnls = np.array([f["pnl"] for f in sub])
        entries = np.array([f["entry"] for f in sub])
        print(f"  {d:<12} {len(sub):>6} {w/len(sub)*100:>5.1f}% ${entries.mean():>+.3f}"
              f"  ${pnls.mean():>+.4f} ${pnls.sum():>+.2f}")

    print(f"\n=== BY SIDE ===")
    for side in ("UP","DOWN"):
        sub = [f for f in fills if f["side"] == side]
        if not sub: continue
        w = sum(f["won"] for f in sub)
        s_pnl = sum(f["pnl"] for f in sub)
        s_ent = np.mean([f["entry"] for f in sub])
        print(f"  {side:<5}: n={len(sub):>4}  win={w/len(sub):>5.1%}  "
              f"avg_entry=${s_ent:.3f}  total=${s_pnl:+.2f}  $/fill=${s_pnl/len(sub):+.4f}")

    # Per-day per-side breakdown
    print(f"\n=== PER DAY × SIDE ===")
    print(f"  {'date':<12} {'UP_n':>5} {'UP_win%':>8} {'UP_pnl':>8} {'DN_n':>5} {'DN_win%':>8} {'DN_pnl':>8}")
    for d in TEST_DAYS:
        ups = [f for f in fills if f["date"] == d and f["side"] == "UP"]
        dns = [f for f in fills if f["date"] == d and f["side"] == "DOWN"]
        u_pnl = sum(f["pnl"] for f in ups); u_w = sum(f["won"] for f in ups)
        d_pnl = sum(f["pnl"] for f in dns); d_w = sum(f["won"] for f in dns)
        print(f"  {d:<12} {len(ups):>5} {u_w/max(1,len(ups))*100:>7.1f}% ${u_pnl:>+6.2f} "
              f"{len(dns):>5} {d_w/max(1,len(dns))*100:>7.1f}% ${d_pnl:>+6.2f}")

if __name__ == "__main__":
    main()
