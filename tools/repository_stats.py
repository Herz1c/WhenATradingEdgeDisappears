"""Print current repository counts without publishing stale hard-coded numbers."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def line_count(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            total += sum(1 for _ in stream)
    return total


def main() -> int:
    src_files = sorted((ROOT / "src").rglob("*.py"))
    test_modules = sorted((ROOT / "tests").glob("test_*.py"))
    tools = sorted((ROOT / "tools").glob("*.py"))
    print(json.dumps({
        "src_python_files": len(src_files),
        "src_python_lines": line_count(src_files),
        "test_modules": len(test_modules),
        "tool_scripts": len(tools),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
