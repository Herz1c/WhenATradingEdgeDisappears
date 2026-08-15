"""Decisive book-parity test: replay the RECORDER PM-L2 up to each live ENTER's
exact decision instant and compute implied_p_up with the SAME extractor, then
compare to the live-logged implied_p_up. Isolates whether the live book matches
the recorder feed at the same instant (=> divergence vs DATASET is just the
builder's 200ms downsample/grid) or genuinely differs (=> live feed bug)."""
from __future__ import annotations
import glob, json, re, sys
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, orjson, zstandard
from poly_l2_only.extractor import MarketState, update_state as pm_update, state_to_features as pm_feat

_RECV = re.compile(rb'"recv_ts_ns":(\d+)')
_KEEP = (b'"event_type":"book"', b'"event_type":"best_bid_ask"',
         b'"event_type":"price_change"', b'"event_type":"last_trade_price"')
PM_BUCKET_NS = 200_000_000


def load_frames(path):
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as fh:
        data = dctx.stream_reader(fh).read()
    out = []
    for ln in data.split(b"\n"):
        if not ln:
            continue
        m = _RECV.search(ln)
        if not m or not any(k in ln for k in _KEEP):
            continue
        try:
            out.append((int(m.group(1)), orjson.loads(ln)))
        except orjson.JSONDecodeError:
            pass
    out.sort(key=lambda x: x[0])
    return out


def downsample_200ms(frames):
    """Mimic the builder: collapse price_change to 200ms buckets, keep others."""
    keep, pc = [], {}
    for ts, fr in frames:
        if fr.get("event_type") == "price_change":
            pc[ts // PM_BUCKET_NS] = (ts, fr)
        else:
            keep.append((ts, fr))
    keep.extend(pc.values())
    keep.sort(key=lambda x: x[0])
    return keep


def slug_file(slug, date):
    g = glob.glob(f"data/raw/polymarket/btc_updown_5m/{date}/*{slug}*.l2.jsonl.zst")
    return g[0] if g else None


def implied_at(frames, ts0):
    st = MarketState()
    last = 0
    for ts, fr in frames:
        if ts > ts0:
            break
        try:
            pm_update(st, fr)
            last = ts
        except Exception:
            pass
    try:
        f = pm_feat(st, ts0)
    except Exception:
        return None, last
    return f.get("implied_p_up"), last


def main():
    import pandas as pd
    rows = []
    for d in ("2026-06-26", "2026-06-27"):
        for l in open(f"logs/fair_value_bot/fv_decisions_{d}.jsonl"):
            r = json.loads(l)
            if r.get("decision") != "ENTER":
                continue
            f = slug_file(r["market_slug"], d)
            if not f:
                continue
            frames = load_frames(f)
            ts0 = r["snapshot_ts_ns"]
            imp_all, last = implied_at(frames, ts0)            # ALL frames (instantaneous)
            imp_ds, _ = implied_at(downsample_200ms(frames), ts0)  # builder 200ms cadence
            if imp_all is None:
                continue
            rows.append({"live": r.get("implied_p_up"), "rec_all": imp_all,
                         "rec_ds": imp_ds, "book_age_s": (ts0 - last) / 1e9})
    df = pd.DataFrame(rows)
    lv = df.live.astype(float); rv = df.rec_all.astype(float)
    df["adiff"] = (lv - rv).abs()
    print(f"replayed {len(df)} entered markets at the EXACT decision instant\n")
    print(f"  live vs recorder(ALL frames): mean|diff|={df.adiff.mean():.3f} "
          f"corr={np.corrcoef(lv,rv)[0,1]:.2f}")
    print(f"  book recv-age at decision: median={df.book_age_s.median():.2f}s mean={df.book_age_s.mean():.2f}s max={df.book_age_s.max():.1f}s\n")
    print("  divergence binned by recorder book freshness at decision:")
    for lo, hi, lbl in ((0, 1, "fresh  <1s"), (1, 3, "1-3s"), (3, 10, "3-10s"), (10, 1e9, ">10s stale")):
        s = df[(df.book_age_s >= lo) & (df.book_age_s < hi)]
        if len(s) == 0:
            continue
        c = np.corrcoef(s.live.astype(float), s.rec_all.astype(float))[0, 1] if len(s) > 2 else float("nan")
        print(f"     {lbl:12} n={len(s):>3}  mean|diff|={s.adiff.mean():.3f}  corr={c:.2f}")
    print(f"\n  live implied_p_up exactly 0.5 (one-sided/empty book): {(lv==0.5).mean()*100:.0f}%")
    print(f"  recorder implied_p_up exactly 0.5: {(rv==0.5).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
