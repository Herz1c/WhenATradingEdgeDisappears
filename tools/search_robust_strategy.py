"""Search for a strategy whose max drawdown is SMALL relative to its PnL
(Calmar = PnL / maxDD >> 1), selected honestly on held-out VAL days and
reported out-of-sample on TEST days.

Calibrator: global isotonic fit on CALIB days only.
Selection metric: VAL Calmar (net PnL / max drawdown), subject to
  n >= MIN_TRADES and profit_factor >= MIN_PF. TEST shown only for transfer.

Grid: EV threshold x fill band [lo,hi] x min_ev_frac.
After picking the winner on VAL, dumps its full TEST risk profile + equity curve.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from calib2d_ev_backtest import (  # noqa: E402
    load_model, load_days, score_model_p, fit_iso, ttc_bucket,
)

REPO = TOOLS.parent
ART = REPO / "artifacts_cleaned" / "poly_l2_only_v2"

CALIB = ["2026-04-22", "2026-04-30", "2026-05-07"]
VAL = ["2026-05-15", "2026-05-21"]
TEST = ["2026-05-28", "2026-06-01", "2026-06-10", "2026-06-11"]
NOTIONAL = 5.0

# selection constraints (on VAL)
MIN_TRADES = 60
MIN_PF = 1.30

# grid
THRS = [0.06, 0.08, 0.10, 0.12, 0.15]
FILL_LO = [0.30, 0.35, 0.40]
FILL_HI = [0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
EV_FRAC = [0.0, 0.10, 0.15, 0.20, 0.25]


def base_frame(model, feats, days, iso) -> pl.DataFrame:
    df = load_days(days)
    p = score_model_p(model, feats, df)
    cal_p = np.clip(iso.predict(p), 1e-6, 1 - 1e-6)
    up = df["up_best_ask"].to_numpy().astype(np.float64)
    dn = df["down_best_ask"].to_numpy().astype(np.float64)
    ttc = df["ttc_s"].to_numpy()
    ev_up = np.where((up > 0) & (up < 1), cal_p - up, -1.0)
    ev_dn = np.where((dn > 0) & (dn < 1), (1.0 - cal_p) - dn, -1.0)
    pick_up = ev_up >= ev_dn
    side = np.where(pick_up, 1, 0)
    fill = np.where(pick_up, up, dn)
    side_ev = np.where(pick_up, ev_up, ev_dn)
    safe = np.where(fill > 0, fill, 1.0)
    ev_frac = np.where(fill > 0, side_ev / safe, -1.0)
    return pl.DataFrame({
        "market_slug": df["market_slug"],
        "recv_ts_ns": df["recv_ts_ns"],
        "date": df["date"],
        "label_up": df["label_up"],
        "bucket": ttc_bucket(ttc),
        "ttc": ttc,
        "side": side,
        "fill": fill,
        "side_ev": side_ev,
        "ev_frac": ev_frac,
    })


def entries_for(base: pl.DataFrame, thr, fmin, fmax, evfrac) -> pl.DataFrame:
    q = base.filter(
        (pl.col("side_ev") >= thr) & (pl.col("ev_frac") >= evfrac) &
        (pl.col("fill") >= fmin) & (pl.col("fill") <= fmax) &
        (pl.col("ttc") >= 10.0)
    )
    if q.height == 0:
        return q
    e = q.sort("recv_ts_ns").group_by(["market_slug", "bucket"]).first()
    won = ((e["side"] == 1) & (e["label_up"] == 1)) | ((e["side"] == 0) & (e["label_up"] == 0))
    pnl = NOTIONAL / e["fill"].to_numpy() * won.to_numpy().astype(float) - NOTIONAL
    return e.with_columns([
        pl.Series("won", won.cast(pl.Int8)),
        pl.Series("pnl", pnl),
    ]).sort("recv_ts_ns")


def risk(e: pl.DataFrame) -> dict:
    if e.height == 0:
        return {"n": 0, "pnl": 0.0, "maxdd": 0.0, "calmar": 0.0, "pf": 0.0, "win": float("nan")}
    pnl = e["pnl"].to_numpy()
    eq = np.cumsum(pnl)
    dd = np.maximum.accumulate(eq) - eq
    maxdd = float(dd.max())
    net = float(pnl.sum())
    gw = float(pnl[pnl > 0].sum()); gl = float(-pnl[pnl < 0].sum())
    pf = gw / gl if gl > 0 else float("inf")
    calmar = net / maxdd if maxdd > 1e-9 else (float("inf") if net > 0 else 0.0)
    return {"n": e.height, "pnl": net, "maxdd": maxdd, "calmar": calmar,
            "pf": pf, "win": float(e["won"].mean())}


def main():
    model, feats, _ = load_model(ART)
    df_cal = load_days(CALIB)
    iso = fit_iso(score_model_p(model, feats, df_cal),
                  df_cal["label_up"].to_numpy().astype(np.float64))
    print("base frames ...", flush=True)
    bval = base_frame(model, feats, VAL, iso)
    btest = base_frame(model, feats, TEST, iso)
    print(f"val rows={bval.height:,}  test rows={btest.height:,}", flush=True)

    configs = list(itertools.product(THRS, FILL_LO, FILL_HI, EV_FRAC))
    configs = [c for c in configs if c[1] < c[2]]
    print(f"searching {len(configs)} configs on VAL ...", flush=True)

    rows = []
    for thr, lo, hi, ef in configs:
        rv = risk(entries_for(bval, thr, lo, hi, ef))
        rows.append((thr, lo, hi, ef, rv))

    # Rank by VAL calmar with constraints.
    elig = [(c, rv) for (*c4, rv) in
            [(thr, lo, hi, ef, rv) for thr, lo, hi, ef, rv in rows]
            for c in [tuple(c4[:4])]
            if rv["n"] >= MIN_TRADES and rv["pf"] >= MIN_PF and rv["pnl"] > 0]
    elig.sort(key=lambda x: x[1]["calmar"], reverse=True)

    print("\n=== TOP configs by VAL Calmar (PnL/maxDD) — with TEST transfer ===")
    print(f"{'thr':>4} {'band':>11} {'evf':>4} | "
          f"{'V_n':>4} {'V_pnl':>7} {'V_dd':>6} {'V_clmr':>6} {'V_pf':>5} | "
          f"{'T_n':>4} {'T_pnl':>7} {'T_dd':>6} {'T_clmr':>6} {'T_pf':>5} {'T_win':>5}")
    shown = []
    for (thr, lo, hi, ef), rv in elig[:20]:
        rt = risk(entries_for(btest, thr, lo, hi, ef))
        shown.append(((thr, lo, hi, ef), rv, rt))
        print(f"{thr:>4.2f} {f'{lo:.2f}-{hi:.2f}':>11} {ef:>4.2f} | "
              f"{rv['n']:>4} {rv['pnl']:>+7.1f} {rv['maxdd']:>6.1f} {rv['calmar']:>6.2f} {rv['pf']:>5.2f} | "
              f"{rt['n']:>4} {rt['pnl']:>+7.1f} {rt['maxdd']:>6.1f} {rt['calmar']:>6.2f} {rt['pf']:>5.2f} "
              f"{100*rt['win'] if rt['n'] else 0:>4.1f}")

    if not shown:
        print("no eligible config — relax constraints")
        return

    # Winner = best VAL calmar (top of list).
    (thr, lo, hi, ef), rv, rt = shown[0]
    print(f"\n>>> SELECTED (by VAL): thr={thr} band=[{lo},{hi}] ev_frac={ef}")
    print(f"    VAL : n={rv['n']} pnl=${rv['pnl']:+.1f} maxDD=${rv['maxdd']:.1f} "
          f"calmar={rv['calmar']:.2f} pf={rv['pf']:.2f}")
    print(f"    TEST: n={rt['n']} pnl=${rt['pnl']:+.1f} maxDD=${rt['maxdd']:.1f} "
          f"calmar={rt['calmar']:.2f} pf={rt['pf']:.2f} win={100*rt['win']:.1f}%")
    print(f"    DD/PnL on TEST = {rt['maxdd']/rt['pnl'] if rt['pnl']>0 else float('inf'):.2f}  "
          f"(want < 1.0)")

    # Full TEST risk profile of the winner.
    e = entries_for(btest, thr, lo, hi, ef)
    profile = full_profile(e)
    out = ART / "backtests" / "robust_strategy_selected.json"
    out.write_text(json.dumps({
        "selected": {"thr": thr, "fill_min": lo, "fill_max": hi, "ev_frac": ef,
                     "calib": CALIB, "val": VAL, "test": TEST, "notional": NOTIONAL},
        "val_metrics": rv, "test_metrics": rt,
        "test_profile": profile,
    }, indent=2, default=float))
    print(f"\nsaved -> {out}")


def full_profile(e: pl.DataFrame) -> dict:
    pnl = e["pnl"].to_numpy(); fill = e["fill"].to_numpy()
    won = e["won"].to_numpy().astype(int); side = e["side"].to_numpy()
    dates = e["date"].to_list()
    eq = np.cumsum(pnl)
    dd = np.maximum.accumulate(eq) - eq
    i_tr = int(np.argmax(dd))
    # streaks
    def streaks(w):
        runs = []; cur = w[0]; n = 1
        for x in w[1:]:
            if x == cur: n += 1
            else: runs.append((int(cur), n)); cur = x; n = 1
        runs.append((int(cur), n))
        lw = max((n for c, n in runs if c == 1), default=0)
        ll = max((n for c, n in runs if c == 0), default=0)
        return lw, ll
    lw, ll = streaks(won)
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    order = np.argsort(pnl)[::-1]
    net = float(pnl.sum())
    top5 = float(pnl[order][:5].sum())
    bins = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    hist = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (fill >= a) & (fill < b)
        if m.any():
            hist.append({"lo": a, "hi": b, "n": int(m.sum()),
                         "win": float(won[m].mean()), "sum_pnl": float(pnl[m].sum())})
    per_day = []
    for d in sorted(set(dates)):
        md = np.array([x == d for x in dates])
        per_day.append({"date": d, "n": int(md.sum()), "pnl": float(pnl[md].sum()),
                        "win": float(won[md].mean())})

    print("\n================ RISK PROFILE (TEST, selected strategy) ================")
    print(f"trades {len(pnl)}  net ${net:+.2f}  win {100*won.mean():.1f}%")
    print(f"max drawdown ${float(dd.max()):.2f}  ({dd.max()/NOTIONAL:.1f} stakes)  "
          f"DD/PnL {dd.max()/net if net>0 else float('inf'):.2f}")
    print(f"longest underwater {int(_uw(dd))} trades")
    print(f"streaks: win {lw} / loss {ll}")
    pf = (wins.sum() / -losses.sum()) if len(losses) else float('inf')
    print(f"profit factor {pf:.2f}  avg win ${wins.mean() if len(wins) else 0:+.3f}  "
          f"avg loss ${losses.mean() if len(losses) else 0:+.3f}")
    print(f"top 5 trades = {100*top5/net if net else float('nan'):.1f}% of net  "
          f"net w/o top5 ${net-top5:+.2f}")
    print(f"avg fill {fill.mean():.3f}  median {np.median(fill):.3f}  "
          f"%<0.5 {100*np.mean(fill<0.5):.1f}%")
    print("fill histogram:")
    for h in hist:
        print(f"  [{h['lo']:.2f},{h['hi']:.2f}): n={h['n']:>3} win={100*h['win']:5.1f}% sum=${h['sum_pnl']:+.2f}")
    print("per-day:")
    for p in per_day:
        print(f"  {p['date']}: n={p['n']:>3} pnl=${p['pnl']:+8.2f} win={100*p['win']:5.1f}%")

    return {"n": len(pnl), "net": net, "win": float(won.mean()),
            "maxdd": float(dd.max()), "dd_over_pnl": float(dd.max()/net) if net > 0 else None,
            "longest_underwater": int(_uw(dd)), "win_streak": lw, "loss_streak": ll,
            "profit_factor": float(pf), "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "top5_share_net": float(top5/net) if net else None,
            "avg_fill": float(fill.mean()), "median_fill": float(np.median(fill)),
            "pct_below_0.5": float(np.mean(fill < 0.5)),
            "fill_histogram": hist, "per_day": per_day,
            "equity_curve": [round(float(v), 3) for v in eq]}


def _uw(dd):
    luw = cur = 0
    for u in (dd > 1e-9):
        cur = cur + 1 if u else 0
        luw = max(luw, cur)
    return luw


if __name__ == "__main__":
    main()
