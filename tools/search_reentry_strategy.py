"""Develop the bucket strategy with PRICE-DEPENDENT re-entry risk rules.

Fixed base: drop (50,60] bucket (keep buckets 0..3 = ttc 10..50), base EV
threshold 0.15, fill band [0.25, 0.60], isotonic calib on 04-22/04-30.

Searched risk rules (per market, candidates = one first-qualifying tick per
remaining bucket, processed in chronological order):
  - max_entries        : cap on entries per market {1,2,3,4}
  - reentry_thr         : edge floor for the 2nd+ entry (>= base) {0.15,0.20,0.25,0.30}
  - first_fill_gate F   : re-entries allowed only if the FIRST entry's fill >= F
                          {0.00(off),0.40,0.45,0.50,0.55}
  => cheap first entry (< F) locks the market to 1 entry; pricier first entry
     allows stacking, but only at a higher edge (bigger ROI to justify capital).

Selection: on SELECT days (05-07/15/21), require positive PnL every day,
n>=40, PF>=1.30; rank by Calmar (PnL/maxDD), tie-break lower DD/PnL.
Report winner on TEST (05-28/06-01/06-10/06-11) and full 7-day, dump trades.
"""
from __future__ import annotations
import itertools, json, sys
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import polars as pl

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from calib2d_ev_backtest import load_model, load_days, score_model_p, fit_iso  # noqa
from search_robust_strategy import base_frame, NOTIONAL  # noqa

REPO = TOOLS.parent
ART = REPO / "artifacts_cleaned" / "poly_l2_only_v2"
CALIB = ["2026-04-22", "2026-04-30"]
SELECT = ["2026-05-07", "2026-05-15", "2026-05-21"]
TEST = ["2026-05-28", "2026-06-01", "2026-06-10", "2026-06-11"]
EVAL = SELECT + TEST
SELSET = set(SELECT)
THR_BASE, LO, HI = 0.15, 0.25, 0.60

MAX_ENTRIES = [1, 2, 3, 4]
REENTRY_THR = [0.15, 0.20, 0.25, 0.30]
FIRST_GATE = [0.00, 0.40, 0.45, 0.50, 0.55]


def build_candidates(base):
    """One first-qualifying tick per (market,bucket), buckets 0..3 only."""
    q = base.filter(
        (pl.col("side_ev") >= THR_BASE) & (pl.col("fill") >= LO) &
        (pl.col("fill") <= HI) & (pl.col("ttc") >= 10.0) & (pl.col("bucket") <= 3))
    e = q.sort("recv_ts_ns").group_by(["market_slug", "bucket"]).first().sort("recv_ts_ns")
    by_market = defaultdict(list)
    for r in e.iter_rows(named=True):
        by_market[r["market_slug"]].append(r)
    for s in by_market:
        by_market[s].sort(key=lambda r: r["recv_ts_ns"])
    return by_market


def apply_policy(by_market, max_entries, reentry_thr, first_gate):
    rows = []
    for s, cand in by_market.items():
        accepted = []
        for c in cand:
            if not accepted:
                accepted.append(c); continue
            if len(accepted) >= max_entries:
                break
            if accepted[0]["fill"] < first_gate:      # cheap first -> no re-entry
                break
            if c["side_ev"] < reentry_thr:            # re-entry needs bigger edge
                continue
            accepted.append(c)
        rows.extend(accepted)
    rows.sort(key=lambda r: r["recv_ts_ns"])
    return rows


def pnl_of(rows):
    out = []
    for r in rows:
        won = (r["side"] == 1 and r["label_up"] == 1) or (r["side"] == 0 and r["label_up"] == 0)
        out.append(NOTIONAL / r["fill"] * (1.0 if won else 0.0) - NOTIONAL)
    return np.array(out)


def risk(rows, days=None):
    if days is not None:
        rows = [r for r in rows if r["date"] in days]
    if not rows:
        return {"n": 0, "pnl": 0.0, "maxdd": 0.0, "calmar": 0.0, "pf": 0.0, "win": float("nan"),
                "ddpnl": float("inf"), "per_day_pos": 0, "min_day": 0.0}
    pnl = pnl_of(rows)
    eq = np.cumsum(pnl); dd = np.maximum.accumulate(eq) - eq
    net = float(pnl.sum()); maxdd = float(dd.max())
    gw = float(pnl[pnl > 0].sum()); gl = float(-pnl[pnl < 0].sum())
    pf = gw / gl if gl > 0 else float("inf")
    won = np.array([(r["side"] == 1 and r["label_up"] == 1) or (r["side"] == 0 and r["label_up"] == 0)
                    for r in rows])
    pd = defaultdict(float)
    for r, p in zip(rows, pnl):
        pd[r["date"]] += p
    return {"n": len(rows), "pnl": net, "maxdd": maxdd,
            "calmar": net / maxdd if maxdd > 1e-9 else (float("inf") if net > 0 else 0.0),
            "pf": pf, "win": float(won.mean()),
            "ddpnl": maxdd / net if net > 0 else float("inf"),
            "per_day_pos": sum(1 for v in pd.values() if v > 0),
            "min_day": min(pd.values()) if pd else 0.0}


def main():
    model, feats, _ = load_model(ART)
    df_cal = load_days(CALIB)
    iso = fit_iso(score_model_p(model, feats, df_cal),
                  df_cal["label_up"].to_numpy().astype(np.float64))
    print("scoring 7 OOS days ...", flush=True)
    base = base_frame(model, feats, EVAL, iso)
    by_market = build_candidates(base)
    print(f"candidate markets={len(by_market)}, "
          f"candidate entries={sum(len(v) for v in by_market.values())}", flush=True)

    # entry-rank diagnostic on full candidate set (no cap)
    full = apply_policy(by_market, 99, 0.0, 0.0)
    rank_map = defaultdict(list)
    seen = Counter()
    for s, cand in by_market.items():
        for k, c in enumerate(cand, 1):
            won = (c["side"] == 1 and c["label_up"] == 1) or (c["side"] == 0 and c["label_up"] == 0)
            pnl = NOTIONAL / c["fill"] * (1.0 if won else 0.0) - NOTIONAL
            rank_map[k].append((won, pnl))
    print("\n=== ENTRY-RANK diagnostic (all candidates, buckets 10-50) ===")
    print(f"{'rank':>4} {'n':>4} {'win%':>6} {'pnl':>9} {'avg':>7}")
    for k in sorted(rank_map):
        rows = rank_map[k]; n = len(rows)
        w = sum(1 for x in rows if x[0]) / n; p = sum(x[1] for x in rows)
        print(f"{k:>4} {n:>4} {100*w:>5.1f}% {p:>+9.2f} {p/n:>+7.3f}")

    configs = list(itertools.product(MAX_ENTRIES, REENTRY_THR, FIRST_GATE))
    print(f"\nsearching {len(configs)} risk-rule configs ...", flush=True)
    scored = []
    for me, rt, fg in configs:
        rows = apply_policy(by_market, me, rt, fg)
        rs = risk(rows, SELSET)
        scored.append({"cfg": (me, rt, fg), "sel": rs, "rows": rows})

    elig = [x for x in scored if x["sel"]["n"] >= 40 and x["sel"]["pf"] >= 1.30
            and x["sel"]["pnl"] > 0 and x["sel"]["per_day_pos"] == 3]
    relaxed = False
    if not elig:
        relaxed = True
        elig = [x for x in scored if x["sel"]["n"] >= 40 and x["sel"]["pf"] >= 1.25 and x["sel"]["pnl"] > 0]
    elig.sort(key=lambda x: (x["sel"]["calmar"], -x["sel"]["ddpnl"]), reverse=True)

    print(f"\n{'(relaxed)' if relaxed else '(all-3-select-days-positive)'}")
    print("=== TOP risk-rule configs (select Calmar) — TEST transfer ===")
    print(f"{'max':>3} {'reThr':>5} {'gate':>4} | {'S_n':>4} {'S_pnl':>7} {'S_dd':>6} {'S_clmr':>6} {'S_pf':>5} | "
          f"{'T_n':>4} {'T_pnl':>7} {'T_dd':>6} {'T_ddp':>5} {'T_pf':>5} {'T_pos':>5} {'T_win':>5}")
    shown = []
    for x in elig[:16]:
        me, rt, fg = x["cfg"]
        rt_test = risk(x["rows"], set(TEST))
        shown.append((x, rt_test))
        print(f"{me:>3} {rt:>5.2f} {fg:>4.2f} | "
              f"{x['sel']['n']:>4} {x['sel']['pnl']:>+7.1f} {x['sel']['maxdd']:>6.1f} "
              f"{x['sel']['calmar']:>6.2f} {x['sel']['pf']:>5.2f} | "
              f"{rt_test['n']:>4} {rt_test['pnl']:>+7.1f} {rt_test['maxdd']:>6.1f} "
              f"{rt_test['ddpnl']:>5.2f} {rt_test['pf']:>5.2f} {rt_test['per_day_pos']:>4}/4 "
              f"{100*rt_test['win'] if rt_test['n'] else 0:>4.1f}")

    if not shown:
        print("nothing eligible"); return
    x, rt_test = shown[0]
    me, rt, fg = x["cfg"]
    rows = x["rows"]
    rfull = risk(rows, set(EVAL))
    print(f"\n>>> SELECTED: max_entries={me} reentry_thr={rt} first_gate={fg}")
    print(f"    SELECT: n={x['sel']['n']} pnl=${x['sel']['pnl']:+.1f} dd=${x['sel']['maxdd']:.1f} "
          f"calmar={x['sel']['calmar']:.2f} pf={x['sel']['pf']:.2f}")
    print(f"    TEST  : n={rt_test['n']} pnl=${rt_test['pnl']:+.1f} dd=${rt_test['maxdd']:.1f} "
          f"DD/PnL={rt_test['ddpnl']:.2f} pf={rt_test['pf']:.2f} pos={rt_test['per_day_pos']}/4 "
          f"win={100*rt_test['win']:.1f}%")
    print(f"    FULL7 : n={rfull['n']} pnl=${rfull['pnl']:+.1f} maxDD=${rfull['maxdd']:.1f} "
          f"DD/PnL={rfull['ddpnl']:.2f} calmar={rfull['calmar']:.2f} pf={rfull['pf']:.2f} "
          f"pos={rfull['per_day_pos']}/7")

    # per-day full
    pd = defaultdict(float); pdw = defaultdict(list)
    pnl = pnl_of(rows)
    for r, p in zip(rows, pnl):
        pd[r["date"]] += p
    print("    per-day 7d:")
    for d in EVAL:
        tag = "SEL" if d in SELSET else "TEST"
        print(f"      [{tag}] {d}: ${pd[d]:+8.2f}")

    # dump trades for interactive chart
    eq = np.cumsum(pnl)
    split = sum(1 for r in rows if r["date"] in SELSET)
    trades = []
    for i, (r, p) in enumerate(zip(rows, pnl)):
        dt = datetime.fromtimestamp(int(r["recv_ts_ns"]) / 1e9, tz=timezone.utc)
        won = (r["side"] == 1 and r["label_up"] == 1) or (r["side"] == 0 and r["label_up"] == 0)
        trades.append({"i": i, "dt": dt.strftime("%Y-%m-%d %H:%M:%S"),
                       "side": "UP" if r["side"] == 1 else "DOWN", "fill": round(float(r["fill"]), 3),
                       "bucket": int(r["bucket"]), "pnl": round(float(p), 2),
                       "eq": round(float(eq[i]), 2), "won": int(won), "sel": int(r["date"] in SELSET)})
    out = ART / "backtests" / "reentry_strategy_selected.json"
    out.write_text(json.dumps({
        "selected": {"max_entries": me, "reentry_thr": rt, "first_gate": fg,
                     "base_thr": THR_BASE, "fill": [LO, HI], "buckets": "10-50 (drop 50-60)",
                     "calib": CALIB, "select": SELECT, "test": TEST},
        "select": x["sel"], "test": rt_test, "full7": rfull,
        "per_day_7d": dict(pd), "split": split, "trades": trades}, indent=2, default=float))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
