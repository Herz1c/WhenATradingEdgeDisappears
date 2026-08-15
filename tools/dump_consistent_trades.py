"""Dump per-trade timeline (datetime, side, fill, pnl, cumulative equity) for the
consistent strategy (thr=0.15, fill[0.25,0.60]) so the equity curve can be made
interactive. Writes a compact JSON array.
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from calib2d_ev_backtest import load_model, load_days, score_model_p, fit_iso  # noqa: E402
from search_robust_strategy import base_frame, entries_for  # noqa: E402

REPO = TOOLS.parent
ART = REPO / "artifacts_cleaned" / "poly_l2_only_v2"
CALIB = ["2026-04-22", "2026-04-30"]
EVAL = ["2026-05-07", "2026-05-15", "2026-05-21",
        "2026-05-28", "2026-06-01", "2026-06-10", "2026-06-11"]
SELECT = set(EVAL[:3])
THR, LO, HI = 0.15, 0.25, 0.60

model, feats, _ = load_model(ART)
df_cal = load_days(CALIB)
iso = fit_iso(score_model_p(model, feats, df_cal),
              df_cal["label_up"].to_numpy().astype(np.float64))
base = base_frame(model, feats, EVAL, iso)
e = entries_for(base, THR, LO, HI, 0.0).sort("recv_ts_ns")

ts = e["recv_ts_ns"].to_numpy()
pnl = e["pnl"].to_numpy()
fill = e["fill"].to_numpy()
side = e["side"].to_numpy()
won = e["won"].to_numpy()
date = e["date"].to_list()
eq = np.cumsum(pnl)

rows = []
for i in range(len(pnl)):
    dt = datetime.fromtimestamp(int(ts[i]) / 1e9, tz=timezone.utc)
    rows.append({
        "i": i,
        "dt": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "side": "UP" if side[i] == 1 else "DOWN",
        "fill": round(float(fill[i]), 3),
        "pnl": round(float(pnl[i]), 2),
        "eq": round(float(eq[i]), 2),
        "won": int(won[i]),
        "sel": int(date[i] in SELECT),
    })

out = ART / "backtests" / "consistent_strategy_trades.json"
out.write_text(json.dumps({"n": len(rows), "split": sum(1 for r in rows if r["sel"]),
                           "trades": rows}))
print(f"dumped {len(rows)} trades, split at {sum(1 for r in rows if r['sel'])} -> {out}")
print("first:", rows[0])
print("last :", rows[-1])
