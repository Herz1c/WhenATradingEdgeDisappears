"""Backtest fair_value_v1 with an execution delay.

Decision is made on features/quotes at snapshot t, but the simulated taker fill
uses the same side's ask at t + delay_s.  This is meant to answer whether a
wider/earlier TTC window survives realistic tick-to-trade latency.

Examples:
  py -3 tools/backtest_fair_value_delay_window.py --last-n 5 --ttc-min 15 --ttc-max 75 --delay-s 2
  py -3 tools/backtest_fair_value_delay_window.py --days 2026-06-25,2026-06-26 --delay-s 2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from build_fair_value_dataset import PM_BUCKET_NS, _iter_raw_lines, _RECV_RE

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "datasets" / "fair_value_v1"
ART = ROOT / "artifacts" / "fair_value_v1"

EV_THR = 0.10
PRICE_LO, PRICE_HI = 0.30, 0.70
BOOK_COHERENCE_TOL = 0.03
NS = 1_000_000_000
_PM_TS_NS_RE = re.compile(rb'"(?:timestamp_ns|message_timestamp_ns)":\s*(\d+)')
_PM_TS_MS_RE = re.compile(rb'"(?:timestamp_ms|message_timestamp_ms)":\s*(\d+)')


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def taker_fee(price: np.ndarray | float) -> np.ndarray | float:
    return 0.072 * price * (1.0 - price)


def available_days() -> list[str]:
    return [os.path.basename(p)[:10] for p in sorted(glob.glob(str(DS / "*.parquet")))]


def parse_days(args: argparse.Namespace) -> list[str]:
    days = available_days()
    if args.days:
        wanted = [d.strip() for d in args.days.split(",") if d.strip()]
        missing = [d for d in wanted if not (DS / f"{d}.parquet").exists()]
        if missing:
            raise SystemExit(f"missing parquet day(s): {missing}")
        return wanted
    return days[-int(args.last_n):]


def load_days(days: list[str]) -> pd.DataFrame:
    frames = []
    for d in days:
        f = DS / f"{d}.parquet"
        df = pd.read_parquet(f)
        df["date"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pm_source_ts_ns_from_line(line: bytes, fallback_recv_ns: int) -> int:
    m = _PM_TS_NS_RE.search(line)
    if m:
        return int(m.group(1))
    m = _PM_TS_MS_RE.search(line)
    if m:
        return int(m.group(1)) * 1_000_000
    return int(fallback_recv_ns)


def _read_pm_clock_frames(path: Path) -> list[tuple[int, int]]:
    """Return downsampled (recv_ts_ns, payload_source_ts_ns) for a PM L2 file."""
    keep: list[tuple[int, int]] = []
    pc_by_bucket: dict[int, tuple[int, int]] = {}
    try:
        for ln in _iter_raw_lines(path):
            m = _RECV_RE.search(ln)
            if not m:
                continue
            recv_ns = int(m.group(1))
            source_ns = _pm_source_ts_ns_from_line(ln, recv_ns)
            if b'"event_type":"price_change"' in ln:
                pc_by_bucket[recv_ns // PM_BUCKET_NS] = (recv_ns, source_ns)
            elif (b'"event_type":"book"' in ln
                  or b'"event_type":"best_bid_ask"' in ln
                  or b'"event_type":"last_trade_price"' in ln):
                keep.append((recv_ns, source_ns))
    except Exception:
        pass
    frames = keep + list(pc_by_bucket.values())
    frames.sort(key=lambda x: x[0])
    return frames


def _load_pm_clock_group(item: tuple[str, str, list[str]]):
    date_s, slug, paths_s = item
    frames: list[tuple[int, int]] = []
    for path_s in paths_s:
        frames.extend(_read_pm_clock_frames(Path(path_s)))
    if not frames:
        return date_s, slug, None, None
    frames.sort(key=lambda x: x[0])
    recv = np.asarray([int(ts) for ts, _ in frames], dtype=np.int64)
    src = np.asarray([int(source_ts or ts) for ts, source_ts in frames], dtype=np.int64)
    src = np.maximum.accumulate(src)
    return date_s, slug, recv, src


def _pm_l2_paths(raw_root: str, date_s: str, slug: str) -> list[Path]:
    root = Path(raw_root)
    bases = [root, root / "raw"]
    for base in bases:
        d = base / "polymarket" / "btc_updown_5m" / date_s
        if d.exists():
            paths = sorted(d.glob(f"*__{slug}__*.l2.jsonl.zst"))
            if paths:
                return paths
    return []


def add_pm_source_lag(df: pd.DataFrame, raw_root: str,
                      groups: set[tuple[str, str]] | None = None) -> pd.DataFrame:
    out = df.copy()
    out["pm_recv_ns"] = np.nan
    out["pm_source_ns"] = np.nan
    out["pm_recv_lag_s"] = np.nan
    out["pm_source_lag_s"] = np.nan

    missing = 0
    group_indices: dict[tuple[str, str], np.ndarray] = {}
    load_items: list[tuple[str, str, list[str]]] = []
    for (date_s, slug), idx in out.groupby(["date", "market_slug"], sort=False).groups.items():
        key = (str(date_s), str(slug))
        if groups is not None and (str(date_s), str(slug)) not in groups:
            continue
        paths = _pm_l2_paths(raw_root, str(date_s), str(slug))
        if not paths:
            missing += 1
            continue
        group_indices[key] = np.asarray(list(idx), dtype=np.int64)
        load_items.append((key[0], key[1], [str(p) for p in paths]))

    workers = min(
        len(load_items),
        max(1, int(os.getenv("FV_PM_SOURCE_LAG_WORKERS", str(min(8, os.cpu_count() or 1))))),
    )
    print(f"PM source-lag raw scan: {len(load_items)} market file group(s), workers={workers}")
    loaded = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_load_pm_clock_group, item) for item in load_items]
            for fut in as_completed(futures):
                loaded.append(fut.result())
    else:
        loaded = [_load_pm_clock_group(item) for item in load_items]

    for date_s, slug, recv, src in loaded:
        key = (str(date_s), str(slug))
        if recv is None or src is None or len(recv) == 0:
            missing += 1
            continue
        ii = group_indices[key]
        now = out.loc[ii, "now_ns"].to_numpy(dtype=np.int64)
        j = np.searchsorted(recv, now, side="right") - 1
        ok = j >= 0
        if not np.any(ok):
            continue
        dst = ii[ok]
        recv_at = recv[j[ok]]
        src_at = src[j[ok]]
        now_at = now[ok]
        out.loc[dst, "pm_recv_ns"] = recv_at.astype(float)
        out.loc[dst, "pm_source_ns"] = src_at.astype(float)
        out.loc[dst, "pm_recv_lag_s"] = (now_at - recv_at) / NS
        out.loc[dst, "pm_source_lag_s"] = (now_at - src_at) / NS
    if missing:
        print(f"warning: missing PM L2 raw files for {missing} market(s); their source lag is NaN")
    return out


def candidate_mask(df: pd.DataFrame, p: np.ndarray, *,
                   ttc_min: float, ttc_max: float,
                   use_book_quality: bool) -> np.ndarray:
    up_ask = df["up_best_ask"].to_numpy(dtype=float)
    dn_ask = df["down_best_ask"].to_numpy(dtype=float)
    p_up = p.astype(float)
    ev_up = p_up - up_ask
    ev_dn = (1.0 - p_up) - dn_ask
    m = (df["ttc_s"].to_numpy(dtype=float) > ttc_min) & (df["ttc_s"].to_numpy(dtype=float) <= ttc_max)
    edge = (
        ((ev_up >= EV_THR) & (ev_up >= ev_dn) & (up_ask > PRICE_LO) & (up_ask < PRICE_HI))
        | ((ev_dn >= EV_THR) & (ev_dn > ev_up) & (dn_ask > PRICE_LO) & (dn_ask < PRICE_HI))
    )
    m &= edge
    if use_book_quality:
        mid_sum = df["up_mid"].to_numpy(dtype=float) + df["down_mid"].to_numpy(dtype=float)
        m &= np.abs(1.0 - mid_sum) <= BOOK_COHERENCE_TOL
        m &= df["up_book_evts_5s"].to_numpy(dtype=float) > 0.0
        m &= df["down_book_evts_5s"].to_numpy(dtype=float) > 0.0
    return m


def add_delayed_quotes(df: pd.DataFrame, delay_s: float) -> pd.DataFrame:
    delay_ns = int(round(delay_s * NS))
    out = df.copy()
    out["fill_ns"] = np.nan
    out["fill_delay_actual_s"] = np.nan
    out["up_ask_delay"] = np.nan
    out["dn_ask_delay"] = np.nan
    if "pm_source_lag_s" in out.columns:
        out["pm_source_lag_delay_s"] = np.nan

    # Data is emitted on a regular grid, but use searchsorted per market so this
    # also behaves correctly when a row is missing.
    for _, idx in out.groupby("market_slug", sort=False).groups.items():
        ii = np.asarray(list(idx), dtype=np.int64)
        ns = out.loc[ii, "now_ns"].to_numpy(dtype=np.int64)
        order = np.argsort(ns)
        ii = ii[order]
        ns = ns[order]
        target = ns + delay_ns
        j = np.searchsorted(ns, target, side="left")
        ok = j < len(ns)
        if not np.any(ok):
            continue
        src = ii[ok]
        dst = ii[j[ok]]
        dst_ns = ns[j[ok]]
        out.loc[src, "fill_ns"] = dst_ns.astype(float)
        out.loc[src, "fill_delay_actual_s"] = (dst_ns - ns[ok]) / NS
        out.loc[src, "up_ask_delay"] = out.loc[dst, "up_best_ask"].to_numpy(dtype=float)
        out.loc[src, "dn_ask_delay"] = out.loc[dst, "down_best_ask"].to_numpy(dtype=float)
        if "pm_source_lag_s" in out.columns:
            out.loc[src, "pm_source_lag_delay_s"] = out.loc[dst, "pm_source_lag_s"].to_numpy(dtype=float)
    return out


def score_model(df: pd.DataFrame) -> np.ndarray:
    import lightgbm as lgb
    import joblib

    booster = lgb.Booster(model_file=str(ART / "model.txt"))
    feats = json.loads((ART / "features.json").read_text())
    iso = joblib.load(ART / "calibrator.pkl")
    x = df[feats].astype(float).values
    init = _logit(df["implied_p_up"].astype(float).values)
    raw = booster.predict(x, raw_score=True)
    p_raw = 1.0 / (1.0 + np.exp(-(init + raw)))
    return iso.transform(p_raw)


def backtest(
    df: pd.DataFrame,
    p: np.ndarray,
    *,
    ttc_min: float,
    ttc_max: float,
    delay_s: float,
    slippage_cap: float | None,
    fixed_shares: float,
    use_book_quality: bool,
    pm_source_lag_max: float | None,
) -> tuple[list[dict], dict]:
    d = df.copy()
    d["p_model"] = p
    d = add_delayed_quotes(d, delay_s)

    m = (d["ttc_s"].to_numpy() > ttc_min) & (d["ttc_s"].to_numpy() <= ttc_max)
    d = d.loc[m].copy()
    d = d.sort_values(["now_ns", "market_slug"], kind="mergesort")

    stats = defaultdict(int)
    trades: list[dict] = []
    seen: set[str] = set()

    for r in d.itertuples(index=False):
        slug = r.market_slug
        if slug in seen:
            continue
        up_ask = float(r.up_best_ask)
        dn_ask = float(r.down_best_ask)
        p_up = float(r.p_model)
        ev_up = p_up - up_ask
        ev_dn = (1.0 - p_up) - dn_ask

        side = None
        quoted = np.nan
        ev = np.nan
        if ev_up >= EV_THR and ev_up >= ev_dn and PRICE_LO < up_ask < PRICE_HI:
            side, quoted, ev = "UP", up_ask, ev_up
        elif ev_dn >= EV_THR and ev_dn > ev_up and PRICE_LO < dn_ask < PRICE_HI:
            side, quoted, ev = "DOWN", dn_ask, ev_dn
        else:
            stats["no_edge"] += 1
            continue

        if use_book_quality:
            mid_sum = float(r.up_mid) + float(r.down_mid)
            if abs(1.0 - mid_sum) > BOOK_COHERENCE_TOL:
                stats["book_incoherent"] += 1
                continue
            if float(r.up_book_evts_5s) <= 0.0 or float(r.down_book_evts_5s) <= 0.0:
                stats["book_inactive"] += 1
                continue

        if pm_source_lag_max is not None:
            lag = float(getattr(r, "pm_source_lag_s", np.nan))
            if not np.isfinite(lag) or lag > pm_source_lag_max:
                stats["pm_stale_decision"] += 1
                continue

        fill = float(r.up_ask_delay if side == "UP" else r.dn_ask_delay)
        if not np.isfinite(fill) or fill <= 0.0 or fill >= 1.0:
            stats["missing_delayed_quote"] += 1
            continue
        if pm_source_lag_max is not None:
            fill_lag = float(getattr(r, "pm_source_lag_delay_s", np.nan))
            if not np.isfinite(fill_lag) or fill_lag > pm_source_lag_max:
                stats["pm_stale_fill"] += 1
                continue
        if slippage_cap is not None and fill > quoted + slippage_cap:
            stats["slippage_block"] += 1
            continue

        won = bool(int(r.resolved_up) == (1 if side == "UP" else 0))
        pnl_per_share = (1.0 if won else 0.0) - fill - float(taker_fee(fill))
        trades.append({
            "date": r.date,
            "market_slug": slug,
            "now_ns": int(r.now_ns),
            "fill_ns": int(r.fill_ns),
            "ttc_s": float(r.ttc_s),
            "fill_delay_actual_s": float(r.fill_delay_actual_s),
            "side": side,
            "p_model": p_up,
            "quoted": float(quoted),
            "fill": fill,
            "price_move": fill - float(quoted),
            "pm_source_lag_s": float(getattr(r, "pm_source_lag_s", np.nan)),
            "pm_source_lag_delay_s": float(getattr(r, "pm_source_lag_delay_s", np.nan)),
            "ev_decision": float(ev),
            "won": won,
            "pnl_per_share": pnl_per_share,
            "pnl_fixed": pnl_per_share * fixed_shares,
        })
        stats["filled"] += 1
        seen.add(slug)
    stats["markets_seen"] = len(set(d["market_slug"]))
    stats["candidate_rows"] = (
        stats["filled"] + stats["book_incoherent"] + stats["book_inactive"]
        + stats["missing_delayed_quote"] + stats["slippage_block"]
        + stats["pm_stale_decision"] + stats["pm_stale_fill"]
    )
    return trades, dict(stats)


def summarize(label: str, trades: list[dict], stats: dict, fixed_shares: float) -> None:
    n = len(trades)
    print(f"\n=== {label} ===")
    print(f"filled trades: {n} | candidate rows blocked/slipped: "
          f"slip={stats.get('slippage_block', 0)} book_inactive={stats.get('book_inactive', 0)} "
          f"book_incoherent={stats.get('book_incoherent', 0)} missing_delay={stats.get('missing_delayed_quote', 0)} "
          f"pm_stale_decision={stats.get('pm_stale_decision', 0)} pm_stale_fill={stats.get('pm_stale_fill', 0)}")
    if n == 0:
        return
    pnl = np.array([t["pnl_per_share"] for t in trades], dtype=float)
    pnl_fixed = np.array([t["pnl_fixed"] for t in trades], dtype=float)
    won = np.array([t["won"] for t in trades], dtype=bool)
    moves = np.array([t["price_move"] for t in trades], dtype=float)
    delays = np.array([t["fill_delay_actual_s"] for t in trades], dtype=float)
    pm_lags = np.array([t["pm_source_lag_s"] for t in trades], dtype=float)
    print(f"win%={won.mean()*100:.1f}%  pnl/share=${pnl.sum():+.3f}  "
          f"$/trade={pnl.mean():+.4f}  fixed {fixed_shares:.1f}sh pnl=${pnl_fixed.sum():+.3f}")
    print(f"price move fill-quote: mean={moves.mean():+.4f}  "
          f"p50={np.median(moves):+.4f}  p95={np.quantile(moves, 0.95):+.4f}  max={moves.max():+.4f}")
    print(f"actual delay: mean={delays.mean():.3f}s  p95={np.quantile(delays, 0.95):.3f}s")
    if np.isfinite(pm_lags).any():
        print(f"PM source lag at decision: mean={np.nanmean(pm_lags):.3f}s  "
              f"p95={np.nanquantile(pm_lags, 0.95):.3f}s  max={np.nanmax(pm_lags):.3f}s")

    by_day = defaultdict(lambda: [0, 0, 0.0, 0.0])
    by_side = defaultdict(lambda: [0, 0, 0.0, 0.0])
    by_bucket = defaultdict(lambda: [0, 0, 0.0, 0.0])
    for t in trades:
        for bucket, key in ((by_day, t["date"]), (by_side, t["side"])):
            bucket[key][0] += 1
            bucket[key][1] += int(t["won"])
            bucket[key][2] += t["pnl_per_share"]
            bucket[key][3] += t["pnl_fixed"]
        lo = int((t["ttc_s"] - 1e-9) // 10 * 10)
        key = f"{lo:02d}-{lo+10:02d}"
        by_bucket[key][0] += 1
        by_bucket[key][1] += int(t["won"])
        by_bucket[key][2] += t["pnl_per_share"]
        by_bucket[key][3] += t["pnl_fixed"]

    print("\nby day:")
    print(f"{'day':12}{'n':>5}{'win%':>8}{'pnl/sh':>10}{f'pnl@{fixed_shares:.1f}':>12}")
    for day in sorted(by_day):
        n0, w0, p0, pf0 = by_day[day]
        print(f"{day:12}{n0:>5}{(w0/n0*100 if n0 else 0):>7.1f}%{p0:>+10.3f}{pf0:>+12.3f}")

    print("\nby side:")
    print(f"{'side':8}{'n':>5}{'win%':>8}{'pnl/sh':>10}{f'pnl@{fixed_shares:.1f}':>12}")
    for side in ("UP", "DOWN"):
        n0, w0, p0, pf0 = by_side[side]
        print(f"{side:8}{n0:>5}{(w0/n0*100 if n0 else 0):>7.1f}%{p0:>+10.3f}{pf0:>+12.3f}")

    print("\nby decision TTC bucket:")
    print(f"{'ttc':8}{'n':>5}{'win%':>8}{'pnl/sh':>10}{f'pnl@{fixed_shares:.1f}':>12}")
    for key in sorted(by_bucket, reverse=True):
        n0, w0, p0, pf0 = by_bucket[key]
        print(f"{key:8}{n0:>5}{(w0/n0*100 if n0 else 0):>7.1f}%{p0:>+10.3f}{pf0:>+12.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last-n", type=int, default=5)
    ap.add_argument("--days", default="")
    ap.add_argument("--ttc-min", type=float, default=15.0)
    ap.add_argument("--ttc-max", type=float, default=75.0)
    ap.add_argument("--delay-s", type=float, default=2.0)
    ap.add_argument("--slippage-cap", type=float, default=0.05)
    ap.add_argument("--fixed-shares", type=float, default=5.1)
    ap.add_argument("--no-book-quality", action="store_true")
    ap.add_argument("--raw-root", default="data/raw")
    ap.add_argument("--pm-source-lag-max", type=float, default=None,
                    help="Require PM payload/source timestamp age <= this many seconds at decision and delayed fill")
    args = ap.parse_args()

    days = parse_days(args)
    df = load_days(days)
    if df.empty:
        raise SystemExit("no rows loaded")
    p = score_model(df)
    if args.pm_source_lag_max is not None:
        cand = candidate_mask(
            df, p,
            ttc_min=args.ttc_min,
            ttc_max=args.ttc_max,
            use_book_quality=not args.no_book_quality,
        )
        groups = set(zip(df.loc[cand, "date"].astype(str), df.loc[cand, "market_slug"].astype(str)))
        print(f"PM source-lag scan: {int(cand.sum())} candidate rows across {len(groups)} market(s)")
        df = add_pm_source_lag(df, args.raw_root, groups=groups)

    print("fair_value_v1 delayed-fill backtest")
    print(f"days: {', '.join(days)}")
    print(f"rows={len(df):,} markets={df['market_slug'].nunique():,} "
          f"ttc({args.ttc_min:g},{args.ttc_max:g}] delay={args.delay_s:g}s "
          f"band[{PRICE_LO},{PRICE_HI}] EV>={EV_THR} book_quality={not args.no_book_quality}")
    if args.pm_source_lag_max is not None:
        lag = df.loc[cand, "pm_source_lag_s"].to_numpy(dtype=float)
        finite = np.isfinite(lag)
        if finite.any():
            print(f"PM source-lag filter: <= {args.pm_source_lag_max:g}s "
                  f"(candidate coverage {finite.mean()*100:.1f}%, p50={np.nanquantile(lag, 0.50):.3f}s "
                  f"p95={np.nanquantile(lag, 0.95):.3f}s max={np.nanmax(lag):.3f}s)")
        else:
            print(f"PM source-lag filter: <= {args.pm_source_lag_max:g}s (no candidate source-lag coverage)")

    trades_raw, stats_raw = backtest(
        df, p,
        ttc_min=args.ttc_min,
        ttc_max=args.ttc_max,
        delay_s=args.delay_s,
        slippage_cap=None,
        fixed_shares=args.fixed_shares,
        use_book_quality=not args.no_book_quality,
        pm_source_lag_max=args.pm_source_lag_max,
    )
    delay_label = f"{args.delay_s:g}s delayed fill"
    summarize(f"{delay_label}, no slippage cap", trades_raw, stats_raw, args.fixed_shares)

    trades_cap, stats_cap = backtest(
        df, p,
        ttc_min=args.ttc_min,
        ttc_max=args.ttc_max,
        delay_s=args.delay_s,
        slippage_cap=args.slippage_cap,
        fixed_shares=args.fixed_shares,
        use_book_quality=not args.no_book_quality,
        pm_source_lag_max=args.pm_source_lag_max,
    )
    summarize(f"{delay_label}, {args.slippage_cap:.2f} slippage cap", trades_cap, stats_cap, args.fixed_shares)


if __name__ == "__main__":
    main()
