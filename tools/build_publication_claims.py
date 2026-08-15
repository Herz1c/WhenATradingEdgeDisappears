"""Render the canonical public claim register to Markdown.

The JSON register is the single source of truth for verdict wording.  This tool
does not infer or recompute research results; claim-verification tests bind its
selected metrics to the committed evidence artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "publication_claims.json"
OUTPUT = ROOT / "docs" / "PUBLICATION_CLAIMS.md"


def render(register: dict) -> str:
    lines = [
        "# Canonical public claim register",
        "",
        "> This file is generated from `artifacts/publication_claims.json`. Edit the JSON",
        "> and run `py tools/build_publication_claims.py`; do not hand-edit this file.",
        "",
        register["evidence_boundary"],
        "",
        "| ID | Claim | Verdict |",
        "|---|---|---|",
    ]
    for item in register["claims"]:
        claim = item["claim"].replace("|", "\\|")
        verdict = item["verdict"].replace("|", "\\|")
        lines.append(f"| `{item['id']}` | {claim} | **{verdict}** |")

    for item in register["claims"]:
        lines.extend([
            "",
            f"## {item['id']}: {item['verdict']}",
            "",
            item["public_statement"],
            "",
            "Public evidence:",
            "",
        ])
        lines.extend(f"- `{path}`" for path in item["evidence"])
        if item.get("limitations"):
            lines.extend(["", "Limits:", ""])
            lines.extend(f"- {text}" for text in item["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the generated file is stale")
    args = parser.parse_args()
    register = json.loads(SOURCE.read_text(encoding="utf-8"))
    expected = render(register)
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != expected:
            raise SystemExit(f"stale generated claim register: {OUTPUT}")
        return 0
    OUTPUT.write_text(expected, encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
