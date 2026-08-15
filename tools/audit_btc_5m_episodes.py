"""Strict audit of data/datasets/btc_5m_episodes_v1_200ms.

Trust nothing: recompute normalization from daily shards, re-verify masks,
clock policy, alignment, splits, labels, and the market baseline. Writes
artifacts/btc_5m_episodes_v1_200ms_audit/{audit_report.json,audit_summary.md}.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms"
OUT = ROOT / "artifacts" / "btc_5m_episodes_v1_200ms_audit"
OUT.mkdir(parents=True, exist_ok=True)
NS = 1_000_000_000

report: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": str(DS), "checks": {}, "blocking": [], "caveats": []}


def check(name: str, ok: bool, detail):
    report["checks"][name] = {"ok": bool(ok), "detail": detail}
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}: {str(detail)[:240]}")
    if not ok:
        report["blocking"].append(f"{name}: {str(detail)[:300]}")


def caveat(msg: str):
    report["caveats"].append(msg)
    print(f"[CAVEAT] {msg}")


def pctl(a: np.ndarray, qs=(50, 95, 99)) -> dict:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    out = {f"p{q}": round(float(np.percentile(a, q)), 4) for q in qs}
    out.update({"max": round(float(a.max()), 4), "n": int(a.size)})
    return out


manifest = json.loads((DS / "manifest.json").read_text(encoding="utf-8"))
norm = json.loads((DS / "normalization.json").read_text(encoding="utf-8"))
splits_json = json.loads((DS / "splits.json").read_text(encoding="utf-8"))
feature_names = json.loads((DS / "feature_names.json").read_text(encoding="utf-8"))
audit_names = json.loads((DS / "audit_names.json").read_text(encoding="utf-8"))
quote_names = json.loads((DS / "quote_names.json").read_text(encoding="utf-8"))
F = len(feature_names)
SEQ = manifest["episode"]["seq_len"]
CAD = manifest["episode"]["cadence_s"]
fp = manifest["freshness_policy"]

# ---------------------------------------------------------------- split npz
split_data: dict[str, dict] = {}
split_stats: dict[str, dict] = {}
slug_sets: dict[str, set] = {}
EXPECT_KEYS = {"X", "valid_mask", "row_present_mask", "source_valid_mask", "y",
               "p_market", "quotes", "audit", "now_ns", "market_slug", "date",
               "open_s", "close_s", "strike", "valid_count"}

for split in ("train", "val", "test"):
    z = np.load(DS / f"{split}.npz", allow_pickle=True)
    keys = set(z.files)
    check(f"{split}.keys", EXPECT_KEYS <= keys, f"missing={sorted(EXPECT_KEYS - keys)} extra={sorted(keys - EXPECT_KEYS)}")
    X = z["X"]
    vm = z["valid_mask"]
    rp = z["row_present_mask"]
    sv = z["source_valid_mask"]
    y = z["y"]
    pm = z["p_market"]
    aud = z["audit"]
    now_ns = z["now_ns"]
    slug = z["market_slug"].astype(str)
    date = z["date"].astype(str)
    open_s = z["open_s"]
    close_s = z["close_s"]
    n = X.shape[0]

    check(f"{split}.shape", X.shape == (n, SEQ, F) and vm.shape == (n, SEQ)
          and y.shape == (n,) and pm.shape == (n, SEQ) and aud.shape == (n, SEQ, len(audit_names)),
          f"X={X.shape} vm={vm.shape} y={y.shape} pm={pm.shape} aud={aud.shape}")
    check(f"{split}.market_count_vs_manifest", n == manifest["splits"][f"{split}_markets"],
          f"npz={n} manifest={manifest['splits'][f'{split}_markets']}")
    check(f"{split}.mask_dtype_bool", vm.dtype == bool and rp.dtype == bool and sv.dtype == bool,
          f"vm={vm.dtype} rp={rp.dtype} sv={sv.dtype}")
    check(f"{split}.mask_consistency", bool((vm == (rp & sv)).all()),
          "valid_mask == row_present & source_valid")
    check(f"{split}.valid_count_field", bool((vm.sum(axis=1) == z["valid_count"]).all()),
          "per-episode valid_count matches mask")

    # X finite + zeros outside mask (chunked)
    nan_bad = 0
    nz_invalid = 0
    for i in range(0, n, 400):
        xb = X[i:i + 400]
        mb = vm[i:i + 400]
        nan_bad += int((~np.isfinite(xb)).sum())
        nz_invalid += int((xb[~mb] != 0).sum())
    check(f"{split}.X_finite", nan_bad == 0, f"nonfinite={nan_bad}")
    check(f"{split}.X_zero_outside_mask", nz_invalid == 0, f"nonzero_at_invalid={nz_invalid}")

    # label + alignment
    check(f"{split}.y_binary", bool(np.isin(y, [0, 1]).all()), f"unique={np.unique(y).tolist()}")
    slug_epoch = np.array([int(s.split("-")[-1]) for s in slug], dtype=np.int64)
    check(f"{split}.slug_open_alignment", bool((slug_epoch == open_s).all()),
          f"mismatches={int((slug_epoch != open_s).sum())}")
    check(f"{split}.episode_length", bool(((close_s - open_s) == 300).all()),
          f"bad={int(((close_s - open_s) != 300).sum())}")
    date_from_open = np.array([datetime.fromtimestamp(int(s), tz=timezone.utc).date().isoformat()
                               for s in open_s])
    check(f"{split}.date_alignment", bool((date_from_open == date).all()),
          f"mismatches={int((date_from_open != date).sum())}")
    # builder derives the grid from a float32 t_s, so steps carry ~us-level
    # rounding jitter; anything beyond 1 ms would indicate a real clock bug.
    step_dev_ns = int(np.abs(np.diff(now_ns[0]).astype(np.int64) - int(CAD * NS)).max())
    step_ok = bool((now_ns[:, 0] == open_s.astype(np.int64) * NS).all()) and step_dev_ns < 1_000_000
    check(f"{split}.time_grid", step_ok,
          f"starts at open exactly; max step deviation {step_dev_ns / 1000:.1f} us (float32 grid rounding)")
    dup = n - len(set(slug.tolist()))
    check(f"{split}.no_duplicate_markets", dup == 0, f"duplicates={dup}")

    # p_market sanity on valid steps
    pm_valid = pm[vm]
    check(f"{split}.p_market_range", bool(np.isfinite(pm_valid).all()
          and (pm_valid > 0).all() and (pm_valid < 1).all()),
          f"min={np.nanmin(pm_valid):.4f} max={np.nanmax(pm_valid):.4f}")

    # clock policy on valid steps
    ai = {c: k for k, c in enumerate(audit_names)}
    pm_age_v = aud[:, :, ai["pm_source_age_s"]][vm]
    cex_age_v = aud[:, :, ai["cex_age_s"]][vm]
    rtds_age_v = aud[:, :, ai["rtds_source_age_s"]][vm]
    viol = {
        "pm_age_gt_max": int((~np.isfinite(pm_age_v) | (pm_age_v > fp["max_pm_source_age_s"])).sum()),
        "cex_age_gt_max": int((~np.isfinite(cex_age_v) | (cex_age_v > fp["max_cex_age_s"])).sum()),
        "rtds_age_gt_max": int((~np.isfinite(rtds_age_v) | (rtds_age_v > fp["max_rtds_source_age_s"])).sum()),
    }
    check(f"{split}.freshness_policy_on_valid", sum(viol.values()) == 0, viol)

    # audit distributions
    dist_valid = {c: pctl(aud[:, :, ai[c]][vm]) for c in audit_names}
    present_not_valid = rp & ~vm
    dist_pnv = {c: pctl(aud[:, :, ai[c]][present_not_valid]) for c in
                ("pm_source_age_s", "pm_delivery_lag_s", "cex_age_s", "rtds_source_age_s")}
    # why invalid (present rows failing policy)
    pnv_n = int(present_not_valid.sum())
    why = {}
    if pnv_n:
        pm_a = aud[:, :, ai["pm_source_age_s"]][present_not_valid]
        cx_a = aud[:, :, ai["cex_age_s"]][present_not_valid]
        rt_a = aud[:, :, ai["rtds_source_age_s"]][present_not_valid]
        why = {
            "pm_stale_or_missing": round(float(np.mean(~np.isfinite(pm_a) | (pm_a > fp["max_pm_source_age_s"]))), 4),
            "cex_stale_or_missing": round(float(np.mean(~np.isfinite(cx_a) | (cx_a > fp["max_cex_age_s"]))), 4),
            "rtds_stale_or_missing": round(float(np.mean(~np.isfinite(rt_a) | (rt_a > fp["max_rtds_source_age_s"]))), 4),
        }

    split_stats[split] = {
        "markets": n,
        "dates": sorted(set(date.tolist())),
        "n_dates": len(set(date.tolist())),
        "y_mean": round(float(y.mean()), 4),
        "valid_frac": round(float(vm.mean()), 4),
        "present_frac": round(float(rp.mean()), 4),
        "valid_steps_total": int(vm.sum()),
        "valid_steps_per_episode": {"p50": float(np.median(vm.sum(axis=1))),
                                    "p10": float(np.percentile(vm.sum(axis=1), 10)),
                                    "p90": float(np.percentile(vm.sum(axis=1), 90))},
        "audit_dist_valid": dist_valid,
        "audit_dist_present_not_valid": dist_pnv,
        "present_not_valid_steps": pnv_n,
        "why_invalid_frac": why,
    }
    slug_sets[split] = set(slug.tolist())
    split_data[split] = {"pm": pm, "vm": vm, "y": y, "date": date, "slug": slug}
    del X, aud, rp, sv, now_ns
    z.close()

# ---------------------------------------------------------------- split logic
tr_d, va_d, te_d = (split_stats[s]["dates"] for s in ("train", "val", "test"))
check("splits.chronological", max(tr_d) < min(va_d) <= max(va_d) < min(te_d),
      f"train<= {max(tr_d)} | val {min(va_d)}..{max(va_d)} | test>= {min(te_d)}")
check("splits.market_disjoint",
      not (slug_sets["train"] & slug_sets["val"]) and not (slug_sets["train"] & slug_sets["test"])
      and not (slug_sets["val"] & slug_sets["test"]),
      {"train∩val": len(slug_sets['train'] & slug_sets['val']),
       "train∩test": len(slug_sets['train'] & slug_sets['test']),
       "val∩test": len(slug_sets['val'] & slug_sets['test'])})
check("splits.match_splits_json",
      tr_d == splits_json["train"] and va_d == splits_json["val"] and te_d == splits_json["test"],
      "date lists equal splits.json")

# ---------------------------------------------------------------- normalization
print("recomputing normalization from daily shards (train dates) ...")
cnt = np.zeros(F, dtype=np.float64)
s1 = np.zeros(F, dtype=np.float64)
s2 = np.zeros(F, dtype=np.float64)
col_canary_bad_days = []
ipu = feature_names.index("implied_p_up")
for d in splits_json["train"]:
    z = np.load(DS / "daily" / f"{d}.npz", allow_pickle=True)
    xr = z["X_raw"]
    vm = z["valid_mask"]
    flat = xr[vm].astype(np.float64)  # float64 accumulation: float32 sums of
    fin = np.isfinite(flat)           # squares lose ~4 digits at these scales
    cnt += fin.sum(axis=0)
    s1 += np.where(fin, flat, 0).sum(axis=0)
    s2 += np.where(fin, flat ** 2, 0).sum(axis=0)
    ip = flat[:, ipu]
    ip = ip[np.isfinite(ip)]
    if ip.size and (ip.min() < 0 or ip.max() > 1):
        col_canary_bad_days.append(d)
    z.close()
mean_rc = np.where(cnt > 0, s1 / np.maximum(cnt, 1), 0.0)
var_rc = np.maximum(s2 / np.maximum(cnt, 1) - mean_rc ** 2, 0.0)
std_rc = np.sqrt(var_rc)
std_rc = np.where(std_rc > 1e-8, std_rc, 1.0)
mean_m = np.asarray(norm["mean"], dtype=np.float64)
std_m = np.asarray(norm["std"], dtype=np.float64)
scale = np.maximum(np.abs(mean_m), 1.0)
mean_err = float(np.max(np.abs(mean_rc - mean_m) / scale))
std_err = float(np.max(np.abs(std_rc - std_m) / np.maximum(std_m, 1e-6)))
check("normalization.fit_on_train_valid_only", mean_err < 1e-3 and std_err < 1e-3,
      f"max_rel_mean_err={mean_err:.2e} max_rel_std_err={std_err:.2e} "
      f"fit_valid_timesteps={norm['fit_valid_timesteps']} recomputed={int(cnt.max())}")
check("normalization.feature_order", norm["feature_names"] == feature_names, "orders match")
check("normalization.column_canary_daily", not col_canary_bad_days,
      f"days with implied_p_up outside [0,1]: {col_canary_bad_days}")

# verify X == (X_raw - mean)/std on a sample of markets
print("verifying split X vs daily X_raw on samples ...")
rng = np.random.default_rng(7)
max_dev = 0.0
for split, days in (("train", splits_json["train"][:2] + splits_json["train"][-2:]),
                    ("test", splits_json["test"][:2] + splits_json["test"][-2:])):
    z = np.load(DS / f"{split}.npz", allow_pickle=True)
    Xs = z["X"]
    vms = z["valid_mask"]
    slugs = z["market_slug"].astype(str)
    pos = {s: i for i, s in enumerate(slugs)}
    for d in days:
        dz = np.load(DS / "daily" / f"{d}.npz", allow_pickle=True)
        dslug = dz["market_slug"].astype(str)
        pick = rng.choice(len(dslug), size=min(4, len(dslug)), replace=False)
        for j in pick:
            i = pos.get(dslug[j])
            if i is None:
                continue
            expect = (dz["X_raw"][j] - mean_m.astype(np.float32)) / std_m.astype(np.float32)
            expect[~np.isfinite(expect)] = 0.0
            expect[~dz["valid_mask"][j]] = 0.0
            dev = float(np.max(np.abs(Xs[i] - expect)))
            max_dev = max(max_dev, dev)
        dz.close()
    z.close()
check("normalization.X_reproducible_from_raw", max_dev < 1e-4, f"max_abs_dev={max_dev:.2e}")

# ---------------------------------------------------------------- coverage
daily_info = {e["date"]: e for e in manifest["dates"]}
all_days = sorted(daily_info)
d0, d1 = datetime.fromisoformat(all_days[0]), datetime.fromisoformat(all_days[-1])
calendar = [(d0 + timedelta(days=k)).date().isoformat() for k in range((d1 - d0).days + 1)]
missing_days = [d for d in calendar if d not in daily_info]
low_valid_days = [(d, daily_info[d]["valid_frac"]) for d in all_days if daily_info[d]["valid_frac"] < 0.10]
small_days = [(d, daily_info[d]["episodes"]) for d in all_days if daily_info[d]["episodes"] < 50]
report["coverage"] = {
    "days_built": len(all_days), "calendar_span_days": len(calendar),
    "missing_calendar_days": missing_days,
    "low_valid_days_lt_0.10": low_valid_days,
    "small_days_lt_50_episodes": small_days,
    "per_day": [{"date": d, "episodes": daily_info[d]["episodes"],
                 "valid_frac": round(daily_info[d]["valid_frac"], 4),
                 "present_frac": round(daily_info[d]["present_frac"], 4)} for d in all_days],
}
print(f"coverage: {len(all_days)} days, {len(missing_days)} missing calendar days, "
      f"{len(low_valid_days)} low-valid days, {len(small_days)} small days")

# ---------------------------------------------------------------- baseline
print("market baseline (p_market) ...")
ttc_grid = 300.0 - np.arange(SEQ, dtype=np.float32) * CAD


def brier(p, y_):
    return float(np.mean((p - y_) ** 2))


def logloss(p, y_):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y_ * np.log(p) + (1 - y_) * np.log(1 - p)))


def auc(p, y_):
    order = np.argsort(p)
    ys = y_[order]
    n1 = int(ys.sum())
    n0 = len(ys) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.arange(1, len(ys) + 1)
    return float((ranks[ys == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


baseline = {}
for split in ("train", "val", "test"):
    d = split_data[split]
    pm, vm, y = d["pm"], d["vm"], d["y"]
    yb = np.broadcast_to(y[:, None], vm.shape)
    rows = {}
    for name, lo, hi in (("all_valid", 0.0, 300.0), ("ttc_15_90", 15.0, 90.0),
                         ("ttc_0_15", 0.0, 15.0), ("ttc_90_300", 90.0, 300.0)):
        band = (ttc_grid > lo) & (ttc_grid <= hi)
        m = vm & band[None, :]
        if int(m.sum()) < 100:
            continue
        p_, y_ = pm[m], yb[m].astype(int)
        rows[name] = {"n": int(m.sum()), "brier": round(brier(p_, y_), 5),
                      "logloss": round(logloss(p_, y_), 5), "auc": round(auc(p_, y_), 4)}
    # market-level: last valid step
    last_p = np.full(len(y), np.nan)
    for i in range(len(y)):
        idx = np.flatnonzero(vm[i])
        if idx.size:
            last_p[i] = pm[i, idx[-1]]
    ok = np.isfinite(last_p)
    rows["last_valid_step_market_level"] = {
        "n": int(ok.sum()), "brier": round(brier(last_p[ok], y[ok]), 5),
        "auc": round(auc(last_p[ok], y[ok]), 4)}
    baseline[split] = rows
report["baseline_p_market"] = baseline
for s, rows in baseline.items():
    print(f"  {s}: ", {k: v for k, v in rows.items() if k in ('ttc_15_90', 'last_valid_step_market_level')})
b_ok = baseline["test"]["ttc_15_90"]["brier"] < 0.16 and baseline["test"]["ttc_15_90"]["auc"] > 0.85
check("baseline.p_market_sane", b_ok, baseline["test"].get("ttc_15_90"))

# ---------------------------------------------------------------- leakage scan
overlap_audit = sorted(set(feature_names) & set(audit_names))
check("features.no_audit_columns", not overlap_audit, overlap_audit)
susp = [f for f in feature_names if any(t in f for t in
        ("recv", "delivery", "resolved", "close_price", "label", "outcome", "future"))]
check("features.no_suspicious_names", not susp, susp)
level_leak = [f for f in feature_names if f in ("cex_mid", "strike", "binance_mid", "synthetic_chainlink_nowcast")]
check("features.no_absolute_level_leak", not level_leak, level_leak)

# ---------------------------------------------------------------- summary
report["split_stats"] = split_stats
report["manifest_freshness_policy"] = fp
report["normalization_meta"] = {"fit_split": norm["fit_split"],
                                "fit_valid_timesteps": norm["fit_valid_timesteps"]}
n_fail = len(report["blocking"])
report["verdict"] = "ready" if n_fail == 0 else "not_ready"
(OUT / "audit_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(f"\nVERDICT: {report['verdict']} ({n_fail} blocking) -> {OUT / 'audit_report.json'}")
