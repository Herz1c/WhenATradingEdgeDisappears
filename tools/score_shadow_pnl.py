"""Score realized PnL of TCN shadow-bot entries against market resolutions.

Two entry sources:
  --source parity     entries = full_replay_expected_entries from
                      artifacts/tcn_shadow_parity/tcn_shadow_parity_<date>.json
                      (the canonical replay-expected set; requires the parity
                      verifier to have run for that day)
  --source decisions  entries = live ENTER lines from
                      logs/<decisions-dir>/tcn_decisions_<date>.jsonl
                      (available immediately, no parity run needed)

Resolutions come from the recorder archive
  data/raw/polymarket/resolution/btc_updown_5m/<date>/*.resolution.jsonl.zst
(`winning_outcome` field). PnL model matches the strategy lock: hold to
resolution, fee = 0.072 * p * (1-p) charged on the entry fill only, winner
redeems at $1.00.

Usage:
    py tools/score_shadow_pnl.py --date 2026-07-05
    py tools/score_shadow_pnl.py --date 2026-07-04 --source decisions \
        --decisions-dir logs/tcn_shadow_bot_direct_capture_v6
    py tools/score_shadow_pnl.py --all          # every parity day + summary

Output: artifacts/tcn_shadow_parity/shadow_pnl_<date>.json per day, and
shadow_pnl_summary.json when --all is used.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import sys
from pathlib import Path

import zstandard as zstd

REPO = Path(__file__).resolve().parents[1]
PARITY_DIR = REPO / "artifacts" / "tcn_shadow_parity"
RESOLUTION_ROOT = REPO / "data" / "raw" / "polymarket" / "resolution" / "btc_updown_5m"
DEFAULT_SHARES = 5.1
FEE_RATE = 0.072


def fee(price: float) -> float:
    return FEE_RATE * price * (1.0 - price)


def read_zst_lines(path: Path) -> list[str]:
    """Read a .jsonl.zst tolerantly: a still-open or truncated final frame is skipped."""
    dctx = zstd.ZstdDecompressor()
    data = b""
    try:
        with open(path, "rb") as f:
            with dctx.stream_reader(f, read_across_frames=True) as r:
                while True:
                    try:
                        chunk = r.read(1 << 20)
                    except zstd.ZstdError:
                        break
                    if not chunk:
                        break
                    data += chunk
    except (OSError, zstd.ZstdError):
        return []
    return data.decode("utf-8", errors="replace").splitlines()


def load_resolutions(date: str, extra_roots: list[Path] | None = None) -> dict[str, str]:
    """market_slug -> winning outcome ('UP'/'DOWN') for a UTC date."""
    winners: dict[str, str] = {}
    roots = [RESOLUTION_ROOT] + (extra_roots or [])
    for root in roots:
        for p in sorted(glob.glob(str(root / date / "*.resolution.jsonl.zst"))):
            for line in read_zst_lines(Path(p)):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                slug = d.get("market_slug")
                w = d.get("winning_outcome")
                if slug and w:
                    winners.setdefault(slug, str(w).upper())
    return winners


def entries_from_parity(date: str) -> list[dict]:
    path = PARITY_DIR / f"tcn_shadow_parity_{date}.json"
    if not path.exists():
        raise FileNotFoundError(f"parity report not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for e in doc.get("full_replay_expected_entries", []):
        out.append({
            "market_slug": e["market_slug"],
            "strategy_id": e.get("strategy_id", "unknown"),
            "side": str(e["side"]).upper(),
            "fill_quote": float(e["fill_quote"]),
            "ev": e.get("ev"),
            "ttc_s": e.get("ttc_s"),
            "entry_source": "parity_expected",
        })
    return out


def entries_from_decisions(date: str, decisions_dir: Path) -> list[dict]:
    path = decisions_dir / f"tcn_decisions_{date}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"decisions log not found: {path}")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("decision") != "ENTER":
                continue
            fill = d.get("fill_quote")
            side = d.get("side")
            if fill is None or side is None:
                continue
            out.append({
                "market_slug": d.get("market_slug"),
                "strategy_id": d.get("strategy_id", "unknown"),
                "side": str(side).upper(),
                "fill_quote": float(fill),
                "ev": d.get("ev"),
                "ttc_s": d.get("ttc_s"),
                "size_shares": d.get("size_shares"),
                "entry_source": "live_decision",
            })
    return out


def score(entries: list[dict], winners: dict[str, str], shares: float) -> dict:
    trades = []
    per_strategy: dict[str, dict] = {}
    total = {"trades": 0, "wins": 0, "pnl": 0.0, "unresolved": 0}
    for e in entries:
        w = winners.get(e["market_slug"])
        row = dict(e)
        if w is None:
            row["outcome"] = None
            row["pnl"] = None
            total["unresolved"] += 1
            trades.append(row)
            continue
        win = (w == e["side"])
        q = e["fill_quote"]
        trade_shares = float(e.get("size_shares") or shares)
        pnl = trade_shares * ((1.0 if win else 0.0) - q - fee(q))
        row["outcome"] = w
        row["win"] = win
        row["pnl"] = round(pnl, 4)
        trades.append(row)
        total["trades"] += 1
        total["wins"] += int(win)
        total["pnl"] += pnl
        s = per_strategy.setdefault(e["strategy_id"], {"trades": 0, "wins": 0, "pnl": 0.0})
        s["trades"] += 1
        s["wins"] += int(win)
        s["pnl"] += pnl
    total["pnl"] = round(total["pnl"], 4)
    total["win_rate"] = round(total["wins"] / total["trades"], 4) if total["trades"] else None
    for s in per_strategy.values():
        s["pnl"] = round(s["pnl"], 4)
        s["win_rate"] = round(s["wins"] / s["trades"], 4) if s["trades"] else None
    return {"total": total, "per_strategy": per_strategy, "trades": trades}


def score_day(date: str, source: str, decisions_dir: Path, shares: float,
              extra_resolution_roots: list[Path], label: str = "") -> dict:
    if source == "parity":
        entries = entries_from_parity(date)
    else:
        entries = entries_from_decisions(date, decisions_dir)
    winners = load_resolutions(date, extra_resolution_roots)
    result = {
        "date": date,
        "entry_source": source,
        "shares_per_entry": shares,
        "fee_model": f"{FEE_RATE} * p * (1-p) on entry fill",
        "n_resolutions_loaded": len(winners),
        **score(entries, winners, shares),
    }
    tag = f"_{label}" if label else ""
    out_path = PARITY_DIR / f"shadow_pnl{tag}_{date}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    t = result["total"]
    print(f"{date} [{source}]: {t['trades']} trades, {t['wins']} wins"
          f" ({(t['win_rate'] or 0) * 100:.0f}%), PnL {t['pnl']:+.2f} USD,"
          f" unresolved {t['unresolved']} -> {out_path.relative_to(REPO)}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="UTC date YYYY-MM-DD")
    ap.add_argument("--all", action="store_true",
                    help="score every day that has a parity report")
    ap.add_argument("--source", choices=["parity", "decisions"], default="parity")
    ap.add_argument("--decisions-dir", type=Path,
                    default=REPO / "logs" / "tcn_shadow_bot_direct_capture_v6")
    ap.add_argument("--shares", type=float, default=DEFAULT_SHARES)
    ap.add_argument("--label", default="",
                    help="strategy label; separates output files (shadow_pnl_<label>_*) "
                         "and restricts --all to the given --decisions-dir")
    ap.add_argument("--resolution-root", type=Path, action="append", default=[],
                    help="extra resolution roots to search (repeatable)")
    args = ap.parse_args()

    if not args.date and not args.all:
        ap.error("need --date or --all")

    plan: list[tuple[str, str]] = []   # (date, source)
    if args.all and args.label:
        # labeled strategy: decisions-source only, dates from its own dir
        for p in sorted(args.decisions_dir.glob("tcn_decisions_*.jsonl")):
            plan.append((p.stem.replace("tcn_decisions_", ""), "decisions"))
    elif args.all:
        parity_dates = {p.stem.replace("tcn_shadow_parity_", "")
                        for p in PARITY_DIR.glob("tcn_shadow_parity_*.json")}
        decision_dates = set()
        for ddir in sorted(REPO.glob("logs/tcn_shadow_bot_direct_capture*")):
            for p in ddir.glob("tcn_decisions_*.jsonl"):
                decision_dates.add(p.stem.replace("tcn_decisions_", ""))
        for d in sorted(parity_dates | decision_dates):
            plan.append((d, "parity" if d in parity_dates else "decisions"))
    else:
        plan = [(args.date, args.source)]

    results = []
    for d, source in plan:
        try:
            ddir = args.decisions_dir
            if not args.label and source == "decisions" and not (ddir / f"tcn_decisions_{d}.jsonl").exists():
                for cand in sorted(REPO.glob("logs/tcn_shadow_bot_direct_capture*")):
                    if (cand / f"tcn_decisions_{d}.jsonl").exists():
                        ddir = cand
                        break
            results.append(score_day(d, source, ddir, args.shares,
                                     args.resolution_root, label=args.label))
        except FileNotFoundError as e:
            print(f"{d}: SKIP ({e})", file=sys.stderr)

    if args.all and results:
        agg = {"trades": 0, "wins": 0, "pnl": 0.0, "unresolved": 0}
        per_strategy: dict[str, dict] = {}
        by_day = {}
        for r in results:
            t = r["total"]
            agg["trades"] += t["trades"]
            agg["wins"] += t["wins"]
            agg["pnl"] += t["pnl"]
            agg["unresolved"] += t["unresolved"]
            by_day[r["date"]] = {"trades": t["trades"], "pnl": t["pnl"]}
            for sid, s in r["per_strategy"].items():
                a = per_strategy.setdefault(sid, {"trades": 0, "wins": 0, "pnl": 0.0})
                a["trades"] += s["trades"]
                a["wins"] += s["wins"]
                a["pnl"] += s["pnl"]
        agg["pnl"] = round(agg["pnl"], 4)
        agg["win_rate"] = round(agg["wins"] / agg["trades"], 4) if agg["trades"] else None
        for s in per_strategy.values():
            s["pnl"] = round(s["pnl"], 4)
        # day-bootstrap CI on total PnL (gate criterion: 90% CI must exclude 0)
        import numpy as np
        daily = np.asarray([v["pnl"] for v in by_day.values()], dtype=np.float64)
        ci = None
        if daily.size >= 3:
            rng = np.random.default_rng(7)
            idx = rng.integers(0, daily.size, size=(10_000, daily.size))
            totals = daily[idx].sum(axis=1)
            ci = {
                "total_pnl_ci90": [round(float(np.percentile(totals, 5)), 2),
                                   round(float(np.percentile(totals, 95)), 2)],
                "p_total_leq_0": round(float(np.mean(totals <= 0.0)), 4),
                "gate_ci90_excludes_0": bool(np.percentile(totals, 5) > 0.0),
            }
        summary = {"days": len(results), "total": agg,
                   "bootstrap": ci,
                   "per_strategy": per_strategy, "by_day": by_day}
        tag = f"_{args.label}" if args.label else ""
        out = PARITY_DIR / f"shadow_pnl{tag}_summary.json"
        out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"SUMMARY: {agg['trades']} trades over {len(results)} days,"
              f" PnL {agg['pnl']:+.2f} USD -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
