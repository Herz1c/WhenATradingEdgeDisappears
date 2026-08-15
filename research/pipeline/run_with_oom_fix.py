#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper: set OOM-safe env vars + cleanup env vars, then exec a training script.

Usage: py -3 run_with_oom_fix.py <script.py> [--load-days N] [--max-train-rows N]
"""
import os, sys, runpy, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

script = sys.argv[1]
load_days = "18"   # default: load only most-recent 18 daily shards
max_train = "3000000"   # default: cap at 3M rows (was 8M)

# Parse optional args
i = 2
while i < len(sys.argv):
    if sys.argv[i] == "--load-days":
        load_days = sys.argv[i + 1]; i += 2
    elif sys.argv[i] == "--max-train-rows":
        max_train = sys.argv[i + 1]; i += 2
    else:
        i += 1

# Memory caps
os.environ["MAX_LOAD_DAYS_FROM_END"]   = load_days
os.environ["MAX_TRAIN_ROWS_OVERRIDE"]  = max_train
# Cleanup hooks
os.environ["FEATURE_CLEANUP_ENABLED"]  = "1"
os.environ["MODEL_ARTIFACT_ROOT_OVERRIDE"] = "artifacts_cleaned"

print(f"[wrapper] script={script}", file=sys.stderr)
print(f"[wrapper] MAX_LOAD_DAYS_FROM_END={load_days}", file=sys.stderr)
print(f"[wrapper] MAX_TRAIN_ROWS_OVERRIDE={max_train}", file=sys.stderr)
print(f"[wrapper] FEATURE_CLEANUP_ENABLED=1", file=sys.stderr)
print(f"[wrapper] MODEL_ARTIFACT_ROOT_OVERRIDE=artifacts_cleaned", file=sys.stderr)
print(f"[wrapper] running...", file=sys.stderr)

runpy.run_path(script, run_name="__main__")
