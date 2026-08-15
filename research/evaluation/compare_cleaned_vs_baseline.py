#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare cleaned-feature retrains (artifacts_cleaned/) against the contaminated
baseline (artifacts/), side-by-side per model.

For each model that exists in BOTH directories, reports:
  - Test-day metrics (from each model's own metrics file)
  - Trojan-importance share (HARD%, from feature_importance.json)
  - Where applicable: top features changed

PRODUCES: docs/cleaned_vs_baseline_comparison.md
  + console-summary table

USAGE:
    py -3 compare_cleaned_vs_baseline.py
"""
from __future__ import annotations
import io, sys, json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, "src")
from feature_cleanup import TROJAN_HARD

BASELINE_ROOT = Path("artifacts")
CLEANED_ROOT  = Path("artifacts_cleaned")


def load_feature_importance(d: Path) -> dict[str, float] | None:
    """Try the common locations for feature importance per model dir."""
    for candidate in [d / "feature_importance.json", d / "metrics.json", d / "results.json"]:
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if candidate.name == "feature_importance.json":
            d_fi = data
        elif candidate.name == "metrics.json":
            d_fi = data.get("feature_importance", {})
        else:  # results.json
            d_fi = data.get("feature_importances") or data.get("feature_importance") or {}
            if isinstance(d_fi, list):
                feats = data.get("config", {}).get("features") or data.get("features")
                if feats and len(feats) == len(d_fi):
                    d_fi = dict(zip(feats, d_fi))
                else:
                    continue
        if d_fi:
            total = sum(d_fi.values())
            if total > 0:
                return {k: v / total for k, v in d_fi.items()}
    return None


def load_primary_metric(d: Path) -> dict | None:
    """Try to extract test-set headline metric from any of the standard files."""
    for candidate in [d / "metrics.json", d / "results.json", d / "eval_report.md"]:
        if not candidate.exists():
            continue
        if candidate.suffix == ".json":
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            # Common patterns
            for path in [
                ["test"],
                ["splits", "test"],
                ["metrics", "test"],
                ["test_metrics"],
            ]:
                cur = data
                ok = True
                for k in path:
                    if isinstance(cur, dict) and k in cur:
                        cur = cur[k]
                    else:
                        ok = False
                        break
                if ok and isinstance(cur, dict):
                    return cur
            # Maybe top-level metrics
            if any(k in data for k in ("auc", "log_loss", "mae", "rmse")):
                return {k: v for k, v in data.items() if isinstance(v, (int, float))}
        else:
            # eval_report.md — leave for future parser if needed
            continue
    return None


def find_model_dirs(root: Path) -> list[tuple[str, Path]]:
    """Return [(label, path), ...] for every leaf dir that contains model.pkl or feature_importance.json."""
    out = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if (p / "model.pkl").exists() or (p / "feature_importance.json").exists():
            label = str(p.relative_to(root)).replace("\\", "/")
            out.append((label, p))
    return sorted(out)


def trojan_share(fi: dict[str, float] | None) -> float | None:
    if not fi:
        return None
    return sum(v for k, v in fi.items() if k in TROJAN_HARD)


def format_metric(m: dict | None, primary_key: str = "auc") -> str:
    if not m:
        return "—"
    # Try a few canonical metric names
    for k in [primary_key, "auc", "roc_auc", "log_loss", "mae", "rmse", "ap"]:
        if k in m:
            try:
                return f"{m[k]:.4f}"
            except (TypeError, ValueError):
                continue
    return "—"


def main():
    baseline_models = dict(find_model_dirs(BASELINE_ROOT))
    cleaned_models  = dict(find_model_dirs(CLEANED_ROOT))

    print(f"Baseline ({BASELINE_ROOT}): {len(baseline_models)} model dirs", file=sys.stderr)
    print(f"Cleaned  ({CLEANED_ROOT}): {len(cleaned_models)} model dirs", file=sys.stderr)

    all_labels = sorted(set(baseline_models.keys()) | set(cleaned_models.keys()))

    rows = []
    for label in all_labels:
        bp = baseline_models.get(label)
        cp = cleaned_models.get(label)
        b_fi = load_feature_importance(bp) if bp else None
        c_fi = load_feature_importance(cp) if cp else None
        b_metric = load_primary_metric(bp) if bp else None
        c_metric = load_primary_metric(cp) if cp else None
        b_trojan = trojan_share(b_fi)
        c_trojan = trojan_share(c_fi)

        rows.append({
            "label":      label,
            "baseline_present": bp is not None,
            "cleaned_present":  cp is not None,
            "baseline_metric":  b_metric,
            "cleaned_metric":   c_metric,
            "baseline_trojan":  b_trojan,
            "cleaned_trojan":   c_trojan,
            "baseline_top5":    list(sorted(b_fi.items(), key=lambda kv: -kv[1])[:5]) if b_fi else None,
            "cleaned_top5":     list(sorted(c_fi.items(), key=lambda kv: -kv[1])[:5]) if c_fi else None,
        })

    # Console summary
    print()
    print("=" * 110)
    print("CLEANED vs CONTAMINATED — BASELINE COMPARISON")
    print("=" * 110)
    print(f"{'MODEL':<55} {'BASE PRES':>9} {'CLN PRES':>9} {'BASE M':>9} {'CLN M':>9} {'BASE T%':>9} {'CLN T%':>9}")
    print("-" * 110)
    for r in rows:
        bm = format_metric(r["baseline_metric"])
        cm = format_metric(r["cleaned_metric"])
        bt = f"{r['baseline_trojan']*100:.1f}%" if r["baseline_trojan"] is not None else "—"
        ct = f"{r['cleaned_trojan']*100:.1f}%" if r["cleaned_trojan"] is not None else "—"
        bp = "Y" if r["baseline_present"] else "·"
        cp = "Y" if r["cleaned_present"] else "·"
        print(f"{r['label'][:55]:<55} {bp:>9} {cp:>9} {bm:>9} {cm:>9} {bt:>9} {ct:>9}")
    print()

    # ── Build report ─────────────────────────────────────────────────────────
    lines = []
    lines.append("# Cleaned-Feature vs Contaminated Baseline Comparison\n")
    lines.append(f"Baseline dir: `{BASELINE_ROOT}`  ({len(baseline_models)} model dirs)\n")
    lines.append(f"Cleaned dir:  `{CLEANED_ROOT}` ({len(cleaned_models)} model dirs)\n\n")

    n_both = sum(1 for r in rows if r["baseline_present"] and r["cleaned_present"])
    n_b_only = sum(1 for r in rows if r["baseline_present"] and not r["cleaned_present"])
    n_c_only = sum(1 for r in rows if not r["baseline_present"] and r["cleaned_present"])
    lines.append(f"**Coverage**: {n_both} in both, {n_b_only} only in baseline, {n_c_only} only in cleaned\n\n")

    lines.append("## Side-by-side metrics & trojan importance\n\n")
    lines.append("| Model | In Base | In Clean | Base Metric | Clean Metric | Base Trojan% | Clean Trojan% |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for r in rows:
        bm = format_metric(r["baseline_metric"])
        cm = format_metric(r["cleaned_metric"])
        bt = f"{r['baseline_trojan']*100:.1f}%" if r["baseline_trojan"] is not None else "—"
        ct = f"{r['cleaned_trojan']*100:.1f}%" if r["cleaned_trojan"] is not None else "—"
        bp = "✓" if r["baseline_present"] else "—"
        cp = "✓" if r["cleaned_present"] else "—"
        lines.append(f"| `{r['label']}` | {bp} | {cp} | {bm} | {cm} | {bt} | {ct} |\n")
    lines.append("\n")

    lines.append("## Detail: Top-5 features per side\n\n")
    for r in rows:
        if not (r["baseline_present"] and r["cleaned_present"]):
            continue
        lines.append(f"### `{r['label']}`\n\n")
        b5 = r["baseline_top5"] or []
        c5 = r["cleaned_top5"] or []
        lines.append("| Rank | Baseline (top-5) | Cleaned (top-5) |\n|---|---|---|\n")
        for i in range(max(len(b5), len(c5))):
            b = f"`{b5[i][0]}` ({b5[i][1]*100:.1f}%)" if i < len(b5) else "—"
            c = f"`{c5[i][0]}` ({c5[i][1]*100:.1f}%)" if i < len(c5) else "—"
            lines.append(f"| {i+1} | {b} | {c} |\n")
        lines.append("\n")

    lines.append("## Next Steps After Reviewing\n\n")
    lines.append("For each model row, the decision criteria are:\n\n")
    lines.append("- **Cleaned metric ≥ baseline (within ±0.02)** AND **Clean Trojan% ≈ 0** → promote the cleaned model.\n")
    lines.append("- **Cleaned metric materially worse than baseline** → investigate; the model may genuinely have depended on the trojans (rare).\n")
    lines.append("- **Cleaned metric materially better** → trojans were actively harmful; definitely promote.\n")
    lines.append("\n")
    lines.append("To promote: move `artifacts/` -> `artifacts_contaminated/`, then `artifacts_cleaned/` -> `artifacts/`.\n")
    lines.append("To rerun OOS evaluation against the cleaned artifacts, point the eval scripts at `artifacts_cleaned/`.\n")

    out = Path("docs/cleaned_vs_baseline_comparison.md")
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Report saved to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
