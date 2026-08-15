"""Generate _cleaned.py variants of the OOS eval scripts pointing at artifacts_cleaned/."""
import re

for src in ["oos_eval_08bc.py", "oos_eval_may13_16.py"]:
    txt = open(src, encoding="utf-8").read()
    pattern = re.compile(r"""(["'])artifacts/""")
    n = len(pattern.findall(txt))
    new = pattern.sub(lambda m: m.group(1) + "artifacts_cleaned/", txt)
    new = new.replace("docs/oos_eval_08bc_report.md", "docs/oos_eval_08bc_cleaned_report.md")
    new = new.replace("docs/oos_eval_may13_16_report.md", "docs/oos_eval_may13_16_cleaned_report.md")
    dst = src.replace(".py", "_cleaned.py")
    open(dst, "w", encoding="utf-8").write(new)
    print(f"  {src} -> {dst}  ({n} paths swapped)")
