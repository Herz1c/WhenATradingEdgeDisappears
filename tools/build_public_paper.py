"""Build the portfolio paper from Markdown and quantitative public artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from reportlab import rl_config
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper/manuscript.md"
REGISTER = ROOT / "artifacts/publication_claims.json"
BRIER = ROOT / "artifacts/tcn_v2_eval/brier_summary.json"
SELECTION = ROOT / "artifacts/audit_v2/selection_audit.json"
NULL_DRAWS = ROOT / "artifacts/audit_v2/wrc_null_distribution.npz"
LIVE = ROOT / "artifacts/live_log_sample_v1/summary.json"
BUNDLE = ROOT / "artifacts/evaluation_repro_v2/manifest.json"
OUTPUT = ROOT / "paper/report.pdf"
HASH_INPUTS = (MANUSCRIPT, REGISTER, BRIER, SELECTION, NULL_DRAWS, LIVE, BUNDLE)
TOKEN = re.compile(r"\{\{([A-Z0-9_]+)\.(verdict|public_statement)\}\}")

INK = colors.HexColor("#17243b")
BLUE = colors.HexColor("#2878b5")
TEAL = colors.HexColor("#2a9d8f")
RED = colors.HexColor("#d1495b")
AMBER = colors.HexColor("#e9a23b")
MUTED = colors.HexColor("#64748b")
GRID = colors.HexColor("#d7dee8")


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in HASH_INPUTS:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve(text: str, register: dict) -> str:
    claims = {item["id"]: item for item in register["claims"]}

    def replace(match: re.Match[str]) -> str:
        claim_id, field = match.groups()
        if claim_id not in claims:
            raise ValueError(f"unknown claim token: {claim_id}")
        return str(claims[claim_id][field])

    resolved = TOKEN.sub(replace, text)
    if "{{" in resolved or "}}" in resolved:
        raise ValueError("unresolved manuscript token")
    return resolved


def _inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    return text


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PaperTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=22, leading=27, textColor=INK, alignment=TA_CENTER,
            spaceAfter=7 * mm,
        ),
        "meta": ParagraphStyle(
            "Meta", parent=base["Normal"], fontName="Helvetica", fontSize=9.5,
            leading=13, alignment=TA_CENTER, textColor=MUTED, spaceAfter=1.2 * mm,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=14, leading=17, textColor=INK, spaceBefore=5 * mm,
            spaceAfter=2.2 * mm, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Heading3"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=13, textColor=INK, spaceBefore=3 * mm,
            spaceAfter=1.5 * mm, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2,
            leading=13.3, alignment=TA_LEFT, textColor=colors.HexColor("#222936"),
            spaceAfter=2.7 * mm,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontName="Helvetica", fontSize=9,
            leading=12.5, leftIndent=5 * mm, firstLineIndent=-3 * mm,
            bulletIndent=1 * mm, spaceAfter=1.2 * mm,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.6,
            leading=9.7, textColor=colors.HexColor("#374151"),
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"], fontName="Helvetica-Oblique",
            fontSize=7.8, leading=10, textColor=MUTED, alignment=TA_CENTER,
            spaceBefore=1.2 * mm, spaceAfter=2.5 * mm,
        ),
    }


def _parse_table(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    rows = []
    for index, line in enumerate(lines):
        if index == 1:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append([Paragraph(_inline(cell), styles["small"]) for cell in cells])
    columns = len(rows[0])
    if columns == 2:
        widths = [102 * mm, 58 * mm]
    elif columns == 3:
        widths = [42 * mm, 68 * mm, 50 * mm]
    else:
        widths = [160 * mm / columns] * columns
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9f0f7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), INK),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aeb8c6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    return table


def _causal_clock_figure() -> Drawing:
    # The two timeline rows are positioned from the bottom, so the extra height
    # here becomes clear space between the figure title and the first row label.
    width, height = 160 * mm, 58 * mm
    drawing = Drawing(width, height)
    drawing.add(String(0, height - 12, "One late message, two historical realities", fontName="Helvetica-Bold", fontSize=10, fillColor=INK))
    x0, x1 = 25, width - 15
    for y, title, color in ((105, "Source-time replay (historical build)", RED), (42, "Receive-time replay (causal requirement)", TEAL)):
        drawing.add(String(0, y + 28, title, fontName="Helvetica-Bold", fontSize=8, fillColor=color))
        drawing.add(Line(x0, y, x1, y, strokeColor=GRID, strokeWidth=1.3))
        for fraction, label in ((0.05, "12:00 source"), (0.95, "12:02 received")):
            x = x0 + fraction * (x1 - x0)
            drawing.add(Line(x, y - 4, x, y + 4, strokeColor=INK, strokeWidth=0.8))
            drawing.add(String(x - 25, y - 16, label, fontName="Helvetica", fontSize=7, fillColor=MUTED))
        event_x = x0 + (0.05 if y == 105 else 0.95) * (x1 - x0)
        drawing.add(Rect(event_x - 4, y + 5, 8, 8, fillColor=color, strokeColor=color))
        text = "event visible too early" if y == 105 else "event visible when recorded"
        text_x = event_x + 8 if y == 105 else event_x - 106
        drawing.add(String(text_x, y + 7, text, fontName="Helvetica", fontSize=7, fillColor=color))
    return drawing


def _brier_figure() -> Drawing:
    summary = json.loads(BRIER.read_text(encoding="utf-8"))
    labels = {
        "identity": "Identity",
        "val_temperature": "Val temperature",
        "val_platt_l2": "Val Platt-L2",
        "val_isotonic": "Val isotonic",
        "val_residual_shrink": "Val residual",
        "train_dayblock_temperature": "Train temperature",
        "train_dayblock_residual_shrink": "Train residual",
    }
    values = [(labels[row["calibrator"]], row["test"]["delta_vs_market"] * 10_000) for row in summary["rows"]]
    width, height = 160 * mm, 70 * mm
    drawing = Drawing(width, height)
    drawing.add(String(0, height - 11, "Test Brier difference vs market (x10,000; lower is better)", fontName="Helvetica-Bold", fontSize=9.5, fillColor=INK))
    left, right, top, bottom = 102, width - 28, height - 28, 18
    low, high = -7.0, 11.0
    scale = (right - left) / (high - low)
    zero_x = left + (0 - low) * scale
    drawing.add(Line(zero_x, bottom, zero_x, top, strokeColor=INK, strokeWidth=0.8))
    for tick in (-5, 0, 5, 10):
        x = left + (tick - low) * scale
        drawing.add(Line(x, bottom - 2, x, top, strokeColor=GRID, strokeWidth=0.45))
        drawing.add(String(x - 5, 5, str(tick), fontName="Helvetica", fontSize=6.5, fillColor=MUTED))
    row_height = (top - bottom) / len(values)
    for index, (label, value) in enumerate(values):
        y = top - (index + 0.72) * row_height
        drawing.add(String(0, y + 1, label, fontName="Helvetica", fontSize=7, fillColor=INK))
        x = left + (value - low) * scale
        start = min(zero_x, x)
        color = TEAL if value < 0 else AMBER
        drawing.add(Rect(start, y, abs(x - zero_x), row_height * 0.48, fillColor=color, strokeColor=color))
        label_x = x - 27 if value < 0 else x + 3
        drawing.add(String(label_x, y + 1, f"{value:+.2f}", fontName="Helvetica", fontSize=6.5, fillColor=INK))
    return drawing


def _selection_figure() -> Drawing:
    null = np.load(NULL_DRAWS, allow_pickle=False)["null_max_mean_daily_pnl"]
    audit = json.loads(SELECTION.read_text(encoding="utf-8"))["corrected_public_reproduction"]
    observed = float(audit["observed_max_mean_daily_pnl"])
    width, height = 160 * mm, 67 * mm
    drawing = Drawing(width, height)
    drawing.add(String(0, height - 11, "Selection-adjusted null distribution of the best strategy", fontName="Helvetica-Bold", fontSize=9.5, fillColor=INK))
    left, right, bottom, top = 35, width - 15, 26, height - 28
    counts, _edges = np.histogram(null, bins=32, range=(0, 20))
    maximum = max(counts)
    for index, count in enumerate(counts):
        x0 = left + index / len(counts) * (right - left)
        x1 = left + (index + 1) / len(counts) * (right - left)
        bar_height = (count / maximum) * (top - bottom)
        drawing.add(Rect(x0, bottom, max(0.5, x1 - x0 - 0.5), bar_height, fillColor=BLUE, strokeColor=BLUE))
    drawing.add(Line(left, bottom, right, bottom, strokeColor=INK, strokeWidth=0.8))
    for tick in (0, 5, 10, 15, 20):
        x = left + tick / 20 * (right - left)
        drawing.add(Line(x, bottom - 2, x, bottom + 2, strokeColor=INK, strokeWidth=0.6))
        drawing.add(String(x - 4, 11, str(tick), fontName="Helvetica", fontSize=6.5, fillColor=MUTED))
    observed_x = left + observed / 20 * (right - left)
    drawing.add(Line(observed_x, bottom, observed_x, top + 3, strokeColor=RED, strokeWidth=2))
    drawing.add(String(observed_x + 4, top - 2, f"observed = {observed:.2f}", fontName="Helvetica-Bold", fontSize=7, fillColor=RED))
    median = float(np.median(null))
    drawing.add(String(left, 1, f"20,000 stationary-bootstrap draws; null median {median:.2f}; p = {audit['finite_sample_corrected_p_value']:.4f}", fontName="Helvetica", fontSize=7, fillColor=MUTED))
    return drawing


def _figure(marker: str, styles: dict[str, ParagraphStyle]) -> list:
    figures = {
        "[[CAUSAL_CLOCK_FIGURE]]": (_causal_clock_figure, "Figure 1. Source time can reveal a late message before local availability."),
        "[[BRIER_FIGURE]]": (_brier_figure, "Figure 2. Point estimates vary across seven inspected calibration maps; the primary Platt-L2 cluster interval still crosses zero."),
        "[[SELECTION_FIGURE]]": (_selection_figure, "Figure 3. The observed winner is not extreme relative to the best result expected across 847 searched candidates."),
    }
    factory, caption = figures[marker]
    return [
        Spacer(1, 1.5 * mm),
        factory(),
        Paragraph(caption, styles["caption"]),
        Spacer(1, 1.5 * mm),
    ]


def _story(markdown: str) -> list:
    styles = _styles()
    lines = markdown.splitlines()
    story: list = []
    paragraph: list[str] = []
    index = 0

    def flush() -> None:
        if paragraph:
            story.append(Paragraph(_inline(" ".join(paragraph)), styles["body"]))
            paragraph.clear()

    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            flush()
            index += 1
            continue
        if line in ("[[CAUSAL_CLOCK_FIGURE]]", "[[BRIER_FIGURE]]", "[[SELECTION_FIGURE]]"):
            flush()
            story.extend(_figure(line, styles))
            index += 1
            continue
        if line == "[[PAGEBREAK]]":
            flush()
            story.append(PageBreak())
            index += 1
            continue
        if line.startswith("| ") and index + 1 < len(lines) and lines[index + 1].startswith("|---"):
            flush()
            block = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].startswith("|"):
                block.append(lines[index])
                index += 1
            story.extend((_parse_table(block, styles), Spacer(1, 2.5 * mm)))
            continue
        if line.startswith("# "):
            flush()
            story.append(Spacer(1, 19 * mm))
            story.append(Paragraph(_inline(line[2:]), styles["title"]))
            index += 1
            while index < len(lines) and lines[index].strip() and not lines[index].startswith("##"):
                story.append(Paragraph(_inline(lines[index].replace("  ", "").strip()), styles["meta"]))
                index += 1
            story.append(Spacer(1, 9 * mm))
            continue
        if line.startswith("## "):
            flush()
            story.append(Paragraph(_inline(line[3:]), styles["h2"]))
            index += 1
            continue
        if line.startswith("### "):
            flush()
            story.append(Paragraph(_inline(line[4:]), styles["h3"]))
            index += 1
            continue
        if line.startswith("- "):
            flush()
            story.append(Paragraph(_inline(line[2:]), styles["bullet"], bulletText="-"))
            index += 1
            continue
        paragraph.append(line.strip())
        index += 1
    flush()
    return story


def _page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.4)
    canvas.line(25 * mm, 17 * mm, 185 * mm, 17 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.3)
    canvas.drawString(25 * mm, 11 * mm, "When a Trading Edge Disappears | Jan Herza | August 2026")
    canvas.drawRightString(185 * mm, 11 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build() -> None:
    rl_config.invariant = 1
    register = json.loads(REGISTER.read_text(encoding="utf-8"))
    markdown = _resolve(MANUSCRIPT.read_text(encoding="utf-8"), register)
    source_hash = _source_hash()
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=25 * mm, leftMargin=25 * mm,
        topMargin=20 * mm, bottomMargin=23 * mm,
        title="When a Trading Edge Disappears",
        author="Jan Herza",
        subject=f"source-sha256:{source_hash}",
        keywords="market microstructure, prediction markets, causal inference, data snooping",
    )
    document.build(_story(markdown), onFirstPage=_page, onLaterPages=_page)


def check() -> None:
    if not OUTPUT.is_file():
        raise SystemExit(f"missing generated paper: {OUTPUT}")
    reader = PdfReader(OUTPUT)
    metadata = reader.metadata or {}
    if str(metadata.get("/Subject", "")) != f"source-sha256:{_source_hash()}":
        raise SystemExit("paper is stale; run py tools/build_public_paper.py")
    if len(reader.pages) < 5:
        raise SystemExit("paper unexpectedly short")
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    required = (
        "When a Trading Edge Disappears",
        "0.5667",
        "[-0.002730, 0.001721]",
        "NO DEMONSTRATED TRADING EDGE",
        "Reproducibility and provenance",
    )
    for phrase in required:
        if phrase not in full_text:
            raise SystemExit(f"paper is missing required text: {phrase}")
    banned = (
        "execution passes comfortably",
        "all joins are causal",
        "pre-registered audit",
        "live forward test",
    )
    lower = full_text.lower()
    for phrase in banned:
        if phrase in lower:
            raise SystemExit(f"unsupported phrase in paper: {phrase}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    check() if args.check else build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
