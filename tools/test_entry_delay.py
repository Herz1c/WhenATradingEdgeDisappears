"""Execution-latency stress test of the FINAL strategy
(one entry/market, buckets 10-50s, thr 0.15, fill[0.25,0.60]).

The bot DECIDES at tick T but the order FILLS `delay` seconds later, at the
best-ask prevailing at T+delay (committed market order = worst case: we are
filled even if the price moved against us). Resolution/outcome is unchanged.

Reports PnL / DD / PF for delay = 0,1,2,3s, plus how much the fill price moved.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import polars as pl

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from calib2d_ev_backtest import load_model, load_days, score_model_p, fit_iso  # noqa
from search_robust_strategy import base_frame, NOTIONAL  # noqa
from search_reentry_strategy import build_candidates, apply_policy  # noqa

REPO = TOOLS.parent
ART = REPO / "artifacts_cleaned" / "poly_l2_only_v2"
CALIB = ["2026-04-22", "2026-04-30"]
SELECT = {"2026-05-07", "2026-05-15", "2026-05-21"}
EVAL = ["2026-05-07", "2026-05-15", "2026-05-21",
        "2026-05-28", "2026-06-01", "2026-06-10", "2026-06-11"]
NS = 1_000_000_000


def metrics(pnl, won, dates):
    pnl = np.asarray(pnl)
    eq = np.cumsum(pnl); dd = np.maximum.accumulate(eq) - eq
    net = float(pnl.sum()); mdd = float(dd.max())
    gw = float(pnl[pnl > 0].sum()); gl = float(-pnl[pnl < 0].sum())
    pf = gw / gl if gl > 0 else float("inf")
    pd = defaultdict(float)
    for d, p in zip(dates, pnl):
        pd[d] += p
    return {"n": len(pnl), "net": net, "mdd": mdd, "ddpnl": mdd/net if net > 0 else float("inf"),
            "pf": pf, "win": float(np.mean(won)),
            "pos_days": sum(1 for v in pd.values() if v > 0), "per_day": dict(pd)}


def main():
    model, feats, _ = load_model(ART)
    df_cal = load_days(CALIB)
    iso = fit_iso(score_model_p(model, feats, df_cal),
                  df_cal["label_up"].to_numpy().astype(np.float64))
    print("scoring 7 OOS days ...", flush=True)
    base = base_frame(model, feats, EVAL, iso)

    by_market = build_candidates(base)
    entries = apply_policy(by_market, 1, 0.0, 0.0)   # final strategy = 1/market
    entries.sort(key=lambda r: r["recv_ts_ns"])
    print(f"entries: {len(entries)}", flush=True)

    # Per-market full tick stream (raw asks) for delayed-price lookup.
    slugs = {e["market_slug"] for e in entries}
    raw = load_days(EVAL).select(["market_slug", "recv_ts_ns", "up_best_ask", "down_best_ask"])
    sub = raw.filter(pl.col("market_slug").is_in(list(slugs))).sort("recv_ts_ns")
    # Keep only TRADEABLE ticks per side (0 < ask < 1); ask==0 means no offer on
    # that side (empty book) and is NOT a fillable price.
    streams = {}   # slug -> {1:(ts_up,ask_up), 0:(ts_dn,ask_dn)}
    for part in sub.partition_by("market_slug"):
        s = part["market_slug"][0]
        ts = part["recv_ts_ns"].to_numpy()
        ua = part["up_best_ask"].to_numpy().astype(np.float64)
        da = part["down_best_ask"].to_numpy().astype(np.float64)
        mu = (ua > 0.0) & (ua < 1.0)
        md = (da > 0.0) & (da < 1.0)
        streams[s] = {1: (ts[mu], ua[mu]), 0: (ts[md], da[md])}

    def fill_at(slug, side, t_ns, delay_s):
        ts, ask = streams[slug][side]
        if len(ts) == 0:
            return None, True
        target = t_ns + int(delay_s * NS)
        idx = np.searchsorted(ts, target, side="left")
        fallback = idx >= len(ts)
        if fallback:
            idx = len(ts) - 1
        return float(ask[idx]), fallback

    print(f"\n{'delay':>5} {'n':>4} {'net':>9} {'maxDD':>7} {'DD/PnL':>7} {'PF':>5} {'win%':>6} "
          f"{'pos':>4} {'avgfill':>8} {'fallbk':>6}")
    results = {}
    for delay in [0.0, 1.0, 2.0, 3.0]:
        pnl, won, dates, fills, nfb = [], [], [], [], 0
        n_skip = 0
        for e in entries:
            side = e["side"]
            ask, fb = fill_at(e["market_slug"], side, e["recv_ts_ns"], delay)
            if ask is None:
                n_skip += 1
                continue
            nfb += int(fb)
            w = (side == 1 and e["label_up"] == 1) or (side == 0 and e["label_up"] == 0)
            p = NOTIONAL / ask * (1.0 if w else 0.0) - NOTIONAL
            pnl.append(p); won.append(int(w)); dates.append(e["date"]); fills.append(ask)
        m = metrics(pnl, won, dates)
        results[delay] = (m, float(np.mean(fills)), nfb)
        print(f"{delay:>4.0f}s {m['n']:>4} {m['net']:>+9.2f} {m['mdd']:>7.2f} {m['ddpnl']:>7.2f} "
              f"{m['pf']:>5.2f} {100*m['win']:>5.1f}% {m['pos_days']:>2}/7 {np.mean(fills):>8.3f} {nfb:>6}")

    # detail for 2s
    m2 = results[2.0][0]
    print(f"\n=== 2-second delay, per-day (worst-case committed fills) ===")
    for d in EVAL:
        tag = "SEL" if d in SELECT else "TEST"
        v = m2["per_day"].get(d, 0.0)
        print(f"  [{tag}] {d}: ${v:+8.2f}")
    base_net = results[0.0][0]["net"]
    print(f"\nPnL retention at 2s delay: ${m2['net']:+.2f} / ${base_net:+.2f} = "
          f"{100*m2['net']/base_net:.1f}% of frictionless")
    avg0 = results[0.0][1]; avg2 = results[2.0][1]
    print(f"avg fill paid: {avg0:.3f} (0s) -> {avg2:.3f} (2s)  "
          f"(+{10000*(avg2-avg0):.0f} bps of price)")


if __name__ == "__main__":
    main()
