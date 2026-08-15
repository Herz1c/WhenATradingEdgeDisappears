"""Nightly shadow-evidence pipeline (Phase 0.4 + 3.2 ops).

For yesterday (UTC):
  1. build the daily episode shard from raw + direct capture
  2. run shadow parity verification against the shard
  3. score realized shadow PnL vs resolutions (parity-sourced)
  4. refresh the cumulative shadow summary (gate evidence)
  5. run the recorder liveness check

Run manually or via the scheduled task (see README in artifacts/audit_v1).
Exit code != 0 if any step fails, so failures are visible in the task history.

Usage:
    py tools/nightly_shadow_ops.py                # yesterday UTC
    py tools/nightly_shadow_ops.py --date 2026-07-10
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS_LOG = ROOT / "logs" / "liveness" / "nightly_ops.log"


def run(label: str, args: list[str]) -> bool:
    print(f"== {label}: {' '.join(args)}", flush=True)
    proc = subprocess.run([PY] + args, cwd=ROOT, capture_output=True, text=True)
    tail = "\n".join((proc.stdout + "\n" + proc.stderr).strip().splitlines()[-8:])
    print(tail, flush=True)
    ok = proc.returncode == 0
    # parity verifier exits 2 on strict-fail; that is a reportable result, not
    # a pipeline failure
    if label == "parity" and proc.returncode == 2:
        print("   (strict parity FAIL — recorded, pipeline continues)", flush=True)
        ok = True
    with open(STEPS_LOG, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now(dt.UTC).isoformat()} {label} rc={proc.returncode}\n")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None, help="UTC date; default = yesterday")
    ap.add_argument("--decisions-dir", default="logs/tcn_shadow_bot_direct_capture_v6")
    ap.add_argument("--capture-root", default="data/tcn_direct_capture_v6/raw")
    args = ap.parse_args()

    date = args.date or (dt.datetime.now(dt.UTC).date() - dt.timedelta(days=1)).isoformat()
    STEPS_LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"nightly ops for {date}")

    ok = True
    ok &= run("shard", ["tools/build_btc_5m_episode_dataset.py",
                        "--dates", date,
                        "--out", "data/datasets/btc_5m_episodes_v1_200ms",
                        "--cadence-ms", "200",
                        "--raw-roots", f"{args.capture_root};data/raw",
                        "--skip-splits"])
    ok &= run("parity", ["tools/verify_tcn_shadow_parity.py", date,
                         "--decisions-dir", args.decisions_dir])
    ok &= run("score", ["tools/score_shadow_pnl.py", "--date", date,
                        "--source", "parity"])
    ok &= run("summary", ["tools/score_shadow_pnl.py", "--all"])
    # parallel forward-test strategies (decisions-source, own labels)
    for label, ddir in (("v2cand", "logs/tcn_v2_candidate_shadow"),
                        ("v2_1", "logs/tcn_v2_1_shadow"),
                        ("v2_2", "logs/tcn_v2_2_shadow")):
        if (ROOT / ddir / f"tcn_decisions_{date}.jsonl").exists():
            ok &= run(f"score_{label}", ["tools/score_shadow_pnl.py", "--date", date,
                                         "--source", "decisions", "--label", label,
                                         "--decisions-dir", ddir])
            ok &= run(f"summary_{label}", ["tools/score_shadow_pnl.py", "--all",
                                           "--label", label, "--decisions-dir", ddir])
    ok &= run("liveness", ["tools/check_recorder_liveness.py"])
    print("nightly ops:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
