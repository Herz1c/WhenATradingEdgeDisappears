#!/usr/bin/env python3
"""Lightweight background loop that keeps today's canonical
`live_reference_events_v1` parquet from going more than ~30 min stale.

DESIGN — minimal CPU
====================

The full chainlink_recorder build-live-reference-events command rebuilds
the WHOLE day's parquet from raw recorder files.  It costs ~30-60s of
CPU per run.  We don't need that running continuously — bias is a
slowly-changing residual, the bot's MAX_BIAS_AGE_S is 2 days, and the
existing day's parquet is usable for the next pass.

So the daemon:

  1. Wakes up every `--interval` seconds (default 1800 = 30 min).
  2. Looks at today's canonical parquet's mtime.
     a. If it's younger than `--skip-if-fresh-min` (default 25 min):
        SKIP — go back to sleep.  No CPU spent.
     b. Otherwise: run the build (cost: ~30-60s with 2 workers), then
        sleep.
  3. On failure: retry every `--retry-interval` (default 60s) until
     success, then settle back into the slow cadence.

In steady state this means the daemon is CPU-idle for 29/30 minutes
and does one 30-60s burst per cycle.  Use `--max-workers 1` if you
want to lower the burst even further (slower but gentler).

Override anything via env / CLI:

    $env:CANONICAL_DAEMON_INTERVAL="1800"
    $env:CANONICAL_DAEMON_MAX_WORKERS="2"
    $env:CANONICAL_DAEMON_SKIP_IF_FRESH_MIN="25"

The bot's bias-freshness gate is 2 days, so even if this daemon is OFF
the bot still trades using whatever bias is on disk.  This daemon is
strictly for keeping things fresh during long sessions.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _env_or_default(var: str, default: int) -> int:
    return int(os.environ.get(var, str(default)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=int,
                    default=_env_or_default("CANONICAL_DAEMON_INTERVAL", 1800),
                    help="seconds between wake-ups (default 1800 = 30 min)")
    ap.add_argument("--max-workers", type=int,
                    default=_env_or_default("CANONICAL_DAEMON_MAX_WORKERS", 2),
                    help="workers for build-live-reference-events (default 2)")
    ap.add_argument("--skip-if-fresh-min", type=int,
                    default=_env_or_default("CANONICAL_DAEMON_SKIP_IF_FRESH_MIN", 25),
                    help="skip the build if today's parquet is younger than this (default 25 min)")
    ap.add_argument("--retry-interval", type=int,
                    default=_env_or_default("CANONICAL_DAEMON_RETRY_INTERVAL", 60),
                    help="seconds between retries after a failed build (default 60)")
    ap.add_argument("--log-level", default=os.environ.get("CANONICAL_DAEMON_LOG_LEVEL", "INFO"))
    args = ap.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-5s canonical_daemon :: %(message)s",
        level=getattr(logging, args.log_level.upper()),
    )
    log = logging.getLogger("canonical_daemon")
    log.info("starting; interval=%ds workers=%d skip_if_fresh=%dmin",
             args.interval, args.max_workers, args.skip_if_fresh_min)

    cycle = 0
    while True:
        cycle += 1
        today = datetime.now(UTC).date().isoformat()
        parquet = REPO / "data" / "canonical" / "live_reference_events_v1" / f"{today}.parquet"

        # SKIP path — no CPU spent
        if parquet.exists():
            age_min = (time.time() - parquet.stat().st_mtime) / 60
            if age_min < args.skip_if_fresh_min:
                log.info("[cycle %d] parquet is %.0fmin old (< %dmin) — skip rebuild, sleep %ds",
                         cycle, age_min, args.skip_if_fresh_min, args.interval)
                time.sleep(args.interval)
                continue

        # BUILD path — one burst of CPU
        t0 = time.monotonic()
        log.info("[cycle %d] rebuilding canonical for %s (workers=%d) ...",
                 cycle, today, args.max_workers)
        success = False
        try:
            proc = subprocess.run(
                [
                    sys.executable, "-u", "-m", "chainlink_recorder.cli",
                    "build-live-reference-events",
                    "--date-from", today, "--date-to", today,
                    "--max-workers", str(args.max_workers),
                ],
                cwd=str(REPO),
                env={**os.environ, "FEATURE_CLEANUP_ENABLED": "1", "PYTHONIOENCODING": "utf-8"},
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=max(180, args.interval),
            )
            elapsed = time.monotonic() - t0
            if proc.returncode == 0:
                import re
                m = re.search(r'"row_count":\s*(\d+)', proc.stdout or "")
                rows = m.group(1) if m else "?"
                log.info("[cycle %d] OK in %.1fs (rows=%s)", cycle, elapsed, rows)
                success = True
            else:
                log.error("[cycle %d] build failed rc=%d after %.1fs", cycle, proc.returncode, elapsed)
                log.error("stderr tail:\n%s", (proc.stderr or "")[-500:])
        except subprocess.TimeoutExpired:
            log.error("[cycle %d] build timed out", cycle)
        except Exception as exc:
            log.exception("[cycle %d] unexpected error: %r", cycle, exc)

        if success:
            sleep_for = max(10, args.interval - int(time.monotonic() - t0))
        else:
            sleep_for = max(10, args.retry_interval - int(time.monotonic() - t0))
            log.warning("[cycle %d] will retry in %ds", cycle, sleep_for)
        log.info("[cycle %d] sleeping %ds", cycle, sleep_for)
        time.sleep(sleep_for)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nshutdown requested", file=sys.stderr)
        sys.exit(0)
