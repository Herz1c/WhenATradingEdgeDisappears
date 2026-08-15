"""Recorder/shadow-bot liveness check.

The 2026-07-05 -> 2026-07-10 recording gap happened silently. This script makes
staleness loud: it looks at the newest file under each watched root and reports
WARN/OK per source. Exit code 1 if anything is stale, so it can gate scripts or
a scheduled task.

Usage:
    py tools/check_recorder_liveness.py                # human summary
    py tools/check_recorder_liveness.py --max-age-min 15
    py tools/check_recorder_liveness.py --json         # machine readable

Writes a one-line-per-run history to logs/liveness/liveness.jsonl so gaps are
visible in retrospect even if nobody watched the console.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (name, glob root, is_required)
WATCHED: list[tuple[str, str, bool]] = [
    ("polymarket_l2", "data/raw/polymarket/btc_updown_5m", True),
    ("polymarket_rtds", "data/raw/polymarket/rtds/crypto_prices_chainlink/btc_usd", True),
    ("polymarket_resolution", "data/raw/polymarket/resolution/btc_updown_5m", True),
    ("coinbase", "data/raw/coinbase/advanced/BTC-USD", True),
    ("binance", "data/raw/binance", False),
    ("chainlink_onchain", "data/raw/chainlink/onchain", False),
    ("tcn_shadow_decisions", "logs/tcn_shadow_bot_direct_capture_v6", True),
    ("tcn_v2_candidate_decisions", "logs/tcn_v2_candidate_shadow", True),
    ("tcn_v2_2_decisions", "logs/tcn_v2_2_shadow", True),
    ("tcn_direct_capture", "data/tcn_direct_capture_v6/raw", True),
]


def newest_mtime(root: Path) -> tuple[float | None, Path | None]:
    best: tuple[float, Path] | None = None
    if not root.exists():
        return None, None
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            for p in d.iterdir():
                if p.is_dir():
                    stack.append(p)
                else:
                    m = p.stat().st_mtime
                    if best is None or m > best[0]:
                        best = (m, p)
        except OSError:
            continue
    if best is None:
        return None, None
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-age-min", type=float, default=15.0,
                    help="staleness threshold in minutes (default 15)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    now = time.time()
    results = []
    any_stale = False
    for name, rel, required in WATCHED:
        mtime, path = newest_mtime(REPO / rel)
        if mtime is None:
            status = "MISSING" if required else "ABSENT"
            age_min = None
        else:
            age_min = (now - mtime) / 60.0
            status = "OK" if age_min <= args.max_age_min else "STALE"
        if required and status in ("MISSING", "STALE"):
            any_stale = True
        results.append({
            "source": name,
            "root": rel,
            "required": required,
            "status": status,
            "age_min": round(age_min, 2) if age_min is not None else None,
            "newest_file": str(path.relative_to(REPO)) if path else None,
        })

    record = {
        "checked_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "max_age_min": args.max_age_min,
        "ok": not any_stale,
        "results": results,
    }

    hist_dir = REPO / "logs" / "liveness"
    hist_dir.mkdir(parents=True, exist_ok=True)
    with open(hist_dir / "liveness.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    if args.json:
        print(json.dumps(record, indent=2))
    else:
        for r in results:
            age = f"{r['age_min']:.1f} min" if r["age_min"] is not None else "-"
            print(f"[{r['status']:7s}] {r['source']:24s} newest={age:>10s}  {r['newest_file'] or ''}")
        print("OVERALL:", "OK" if not any_stale else "STALE — check recorders/shadow bot!")
    return 0 if not any_stale else 1


if __name__ == "__main__":
    sys.exit(main())
