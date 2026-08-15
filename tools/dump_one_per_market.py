"""Same consistent config (thr=0.15, fill[0.25,0.60]) but ONE entry per market
(first qualifying tick across the whole 10-60s window), instead of one per TTC
bucket. Dumps per-trade timeline + prints risk metrics, for comparison.
"""
from __future__ import annotations
import json, sys
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
EVAL = ["2026-05-07", "2026-05-15", "2026-05-21",
        "2026-05-28", "2026-06-01", "2026-06-10", "2026-06-11"]
SELECT = set(EVAL[:3])
THR, LO, HI = 0.15, 0.25, 0.60


def entries_one_per_market(base, thr, lo, hi):
    q = base.filter(
        (pl.col("side_ev") >= thr) & (pl.col("fill") >= lo) &
        (pl.col("fill") <= hi) & (pl.col("ttc") >= 10.0))
    if q.height == 0:
        return q
    e = q.sort("recv_ts_ns").group_by("market_slug").first()  # ONE per market
    won = ((e["side"] == 1) & (e["label_up"] == 1)) | ((e["side"] == 0) & (e["label_up"] == 0))
    pnl = NOTIONAL / e["fill"].to_numpy() * won.to_numpy().astype(float) - NOTIONAL
    return e.with_columns([pl.Series("won", won.cast(pl.Int8)), pl.Series("pnl", pnl)]).sort("recv_ts_ns")


model, feats, _ = load_model(ART)
df_cal = load_days(CALIB)
iso = fit_iso(score_model_p(model, feats, df_cal), df_cal["label_up"].to_numpy().astype(np.float64))
base = base_frame(model, feats, EVAL, iso)
e = entries_one_per_market(base, THR, LO, HI)

ts = e["recv_ts_ns"].to_numpy(); pnl = e["pnl"].to_numpy(); fill = e["fill"].to_numpy()
side = e["side"].to_numpy(); won = e["won"].to_numpy(); date = e["date"].to_list()
eq = np.cumsum(pnl)
dd = np.maximum.accumulate(eq) - eq
split = sum(1 for d in date if d in SELECT)

# risk
net = float(pnl.sum()); maxdd = float(dd.max())
gw = float(pnl[pnl > 0].sum()); gl = float(-pnl[pnl < 0].sum())
pf = gw / gl if gl else float('inf')
pos_days = {}
for d, p in zip(date, pnl):
    pos_days[d] = pos_days.get(d, 0.0) + p
posn = sum(1 for v in pos_days.values() if v > 0)
print(f"ONE-PER-MARKET: n={len(pnl)}  net=${net:+.2f}  maxDD=${maxdd:.2f}  "
      f"DD/PnL={maxdd/net if net>0 else float('inf'):.2f}  PF={pf:.2f}  "
      f"win={100*won.mean():.1f}%  positive_days={posn}/{len(pos_days)}")
print("per-day:", {d: round(v, 1) for d, v in sorted(pos_days.items())})

rows = []
for i in range(len(pnl)):
    dt = datetime.fromtimestamp(int(ts[i]) / 1e9, tz=timezone.utc)
    rows.append({"i": i, "dt": dt.strftime("%Y-%m-%d %H:%M:%S"),
                 "side": "UP" if side[i] == 1 else "DOWN", "fill": round(float(fill[i]), 3),
                 "pnl": round(float(pnl[i]), 2), "eq": round(float(eq[i]), 2),
                 "won": int(won[i]), "sel": int(date[i] in SELECT)})
out = ART / "backtests" / "one_per_market_trades.json"
out.write_text(json.dumps({"n": len(rows), "split": split, "net": net, "maxdd": maxdd,
                           "pf": pf, "win": float(won.mean()), "positive_days": posn,
                           "n_days": len(pos_days), "trades": rows}))
print(f"saved -> {out}")
