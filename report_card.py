"""
report_card.py
--------------
Generates a professional PDF Speaker Fluency Report Card.

Layout:
  Page 1 — Header, score ring, key metrics, verdict
  Page 1 — Filler breakdown table, per-minute trend (embedded image)
  Page 1 — Word timeline (embedded image)
  Page 1 — Recommendations, footer

Uses reportlab for PDF generation (no external dependencies beyond pip).
"""

import os
from typing import List
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Circle, String, Rect, Line
from reportlab.graphics import renderPDF
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas as pdfcanvas

from classify import TaggedWord, Label
from metrics import FluentyMetrics


# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY      = colors.HexColor("#1A1A2E")
NAVY_MID  = colors.HexColor("#16213E")
PURPLE    = colors.HexColor("#0F3460")
CORAL     = colors.HexColor("#E94560")
AMBER     = colors.HexColor("#F5A623")
GREEN     = colors.HexColor("#1E8449")
BLUE      = colors.HexColor("#1565C0")
LIGHT_BG  = colors.HexColor("#F8F9FA")
BORDER    = colors.HexColor("#DEE2E6")
MUTED     = colors.HexColor("#6C757D")
WHITE     = colors.white
BLACK     = colors.black

GRADE_COLOURS = {
    "A": GREEN,
    "B": colors.HexColor("#27AE60"),
    "C": AMBER,
    "D": colors.HexColor("#E67E22"),
    "F": CORAL,
}

LABEL_COLOUR_MAP = {
    Label.FLUENT:       colors.HexColor("#1E8449"),
    Label.FILLED_PAUSE: CORAL,
    Label.FALSE_START:  AMBER,
    Label.REPETITION:   colors.HexColor("#6C3483"),
    Label.LONG_PAUSE:   colors.HexColor("#C87D10"),
}


# ── Score ring drawing ─────────────────────────────────────────────────────────
def _score_ring(score: int, grade: str) -> Drawing:
    """Draw a circular score indicator."""
    size = 130
    d = Drawing(size, size)
    cx, cy, r = size/2, size/2, 50

    # Background ring
    d.add(Circle(cx, cy, r, fillColor=LIGHT_BG, strokeColor=BORDER, strokeWidth=2))

    # Filled arc approximated by layered arcs (reportlab doesn't have arc fill)
    grade_colour = GRADE_COLOURS.get(grade, CORAL)
    # Inner circle (solid score colour)
    inner_r = r - 10
    d.add(Circle(cx, cy, inner_r, fillColor=grade_colour, strokeColor=None))

    # White centre
    d.add(Circle(cx, cy, inner_r - 14,
                 fillColor=WHITE, strokeColor=None))

    # Score text
    d.add(String(cx, cy + 6, str(score),
                 fontName="Helvetica-Bold", fontSize=28,
                 fillColor=NAVY, textAnchor="middle"))
    d.add(String(cx, cy - 14, "/100",
                 fontName="Helvetica", fontSize=11,
                 fillColor=MUTED, textAnchor="middle"))
    d.add(String(cx, cy - 28, f"Grade  {grade}",
                 fontName="Helvetica-Bold", fontSize=13,
                 fillColor=grade_colour, textAnchor="middle"))
    return d


# ── Recommendations ────────────────────────────────────────────────────────────
def _get_recommendations(m: FluentyMetrics) -> List[str]:
    recs = []
    if m.fpm > 5:
        recs.append(
            f"Your filler rate is {m.fpm:.1f}/min — significantly above the comfortable threshold "
            f"of ~2/min. Try pausing silently instead of filling the gap with 'um' or 'uh'. "
            f"Practise deliberate pauses during preparation."
        )
    elif m.fpm > 2:
        recs.append(
            f"Filler rate ({m.fpm:.1f}/min) is slightly elevated. Focus on replacing fillers "
            f"with a beat of silence — it reads as confidence, not hesitation."
        )
    else:
        recs.append(
            f"Excellent filler control ({m.fpm:.1f}/min). Keep it up!"
        )

    if m.wpm < 110:
        recs.append(
            f"Speaking rate ({m.wpm:.0f} WPM) is quite slow. While clarity is important, "
            f"try to aim for 120–160 WPM to maintain audience engagement."
        )
    elif m.wpm > 175:
        recs.append(
            f"Speaking rate ({m.wpm:.0f} WPM) is fast. Consider slowing down — "
            f"audience comprehension improves significantly in the 120–160 WPM range."
        )
    else:
        recs.append(
            f"Speaking rate ({m.wpm:.0f} WPM) is in the ideal range (120–160 WPM). Well done!"
        )

    if m.long_pauses > 5:
        recs.append(
            f"You had {m.long_pauses} long pauses totalling {m.total_pause_time_s:.1f}s. "
            f"Extended pauses (>0.8s) can disrupt flow. Preparation and topic familiarity "
            f"usually reduce these significantly."
        )

    if m.false_starts > 3:
        recs.append(
            f"{m.false_starts} false starts detected. These typically occur when sentence "
            f"planning lags behind delivery. Slowing down slightly at the start of each "
            f"new point can help."
        )

    top_filler = sorted(m.filler_breakdown.items(), key=lambda x: -x[1])
    if top_filler:
        word, count = top_filler[0]
        recs.append(
            f"Your most frequent filler is '{word}' ({count}×). "
            f"Being aware of your specific filler habit is the first step — "
            f"record yourself and listen back."
        )
    return recs


# ── Main PDF generator ─────────────────────────────────────────────────────────
def generate_report_card(metrics: FluentyMetrics,
                          tagged: List[TaggedWord],
                          audio_filename: str = "lecture.wav",
                          speaker_name: str = "Speaker",
                          out_path: str = "demo/report_card.pdf",
                          timeline_img: str = "demo/timeline.png",
                          trend_img: str = "demo/trend.png",
                          spectrogram_img: str = "demo/spectrogram.png"):

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm,
    )
    W, H = A4
    styles = getSampleStyleSheet()
    story = []

    def P(text, style="Normal", **kw):
        s = ParagraphStyle("_", parent=styles[style], **kw)
        return Paragraph(text, s)

    # ── HEADER ────────────────────────────────────────────────────────────────
    story.append(P(
        "🎙️  Speaker Fluency Report Card",
        "Heading1",
        fontSize=22, textColor=NAVY, spaceAfter=2,
        fontName="Helvetica-Bold",
    ))
    story.append(P(
        f"<b>{speaker_name}</b>  ·  {audio_filename}  ·  "
        f"Duration: {metrics.duration_s/60:.1f} min",
        fontSize=10, textColor=MUTED, spaceAfter=8,
    ))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=CORAL, spaceAfter=14))

    # ── SCORE + KEY METRICS (side by side) ────────────────────────────────────
    ring = _score_ring(metrics.fluency_score, metrics.grade)

    grade_colour = GRADE_COLOURS.get(metrics.grade, CORAL)
    metrics_data = [
        ["Metric", "Value", "Rating"],
        ["Filler Rate (FPM)",
         f"{metrics.fpm:.1f} / min",
         "✅ Good" if metrics.fpm <= 2 else ("⚠️ High" if metrics.fpm <= 5 else "❌ Very High")],
        ["Speaking Rate (WPM)",
         f"{metrics.wpm:.0f} WPM",
         "✅ Ideal" if 120 <= metrics.wpm <= 160 else "⚠️ Off-range"],
        ["Long Pauses",
         f"{metrics.long_pauses}  ({metrics.total_pause_time_s:.1f}s)",
         "✅ Good" if metrics.long_pauses <= 3 else "⚠️ Many"],
        ["False Starts",
         str(metrics.false_starts),
         "✅ Few" if metrics.false_starts <= 2 else "⚠️ Several"],
        ["Repetitions",
         str(metrics.repetitions),
         "✅ Few" if metrics.repetitions <= 3 else "⚠️ Several"],
        ["Total Words",
         str(metrics.total_words), ""],
    ]
    tbl = Table(metrics_data, colWidths=[5.5*cm, 3.5*cm, 3.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,0), 10),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT_BG, WHITE]),
        ("FONTSIZE",   (0,1), (-1,-1), 10),
        ("GRID",       (0,0), (-1,-1), 0.5, BORDER),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("TEXTCOLOR",  (2,1), (2,-1), NAVY),
    ]))

    combo = Table(
        [[Drawing(130, 130, *[ring]), tbl]],
        colWidths=[4*cm, None]
    )
    combo.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (0,-1), 0),
    ]))
    story.append(combo)
    story.append(Spacer(1, 10))

    # Verdict banner
    story.append(P(
        f"<b>Verdict:</b>  {metrics.verdict}",
        fontSize=11,
        textColor=WHITE,
        backColor=grade_colour,
        borderPadding=(6, 10, 6, 10),
        spaceAfter=14,
    ))

    # ── FILLER BREAKDOWN ──────────────────────────────────────────────────────
    if metrics.filler_breakdown:
        story.append(P("Filler Word Breakdown", "Heading2",
                       fontSize=13, textColor=NAVY, spaceAfter=6))
        fb_data = [["Filler Word", "Count", "% of fillers"]] + [
            [f"'{k}'", str(v),
             f"{v/max(metrics.filled_pauses,1)*100:.0f}%"]
            for k, v in sorted(metrics.filler_breakdown.items(),
                                key=lambda x: -x[1])
        ]
        fb_tbl = Table(fb_data, colWidths=[6*cm, 3*cm, 4*cm])
        fb_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), CORAL),
            ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT_BG, WHITE]),
            ("GRID",       (0,0), (-1,-1), 0.5, BORDER),
            ("FONTSIZE",   (0,0), (-1,-1), 10),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(fb_tbl)
        story.append(Spacer(1, 14))

    # ── VISUALISATIONS ────────────────────────────────────────────────────────
    page_w = W - 3.6*cm
    for img_path, caption in [
        (timeline_img,    "Word-level Disfluency Timeline"),
        (trend_img,       "Fluency Trend — Per Minute Breakdown"),
        (spectrogram_img, "Mel Spectrogram with Disfluency Overlay"),
    ]:
        if os.path.exists(img_path):
            story.append(P(caption, "Heading2",
                           fontSize=12, textColor=NAVY, spaceAfter=4))
            story.append(Image(img_path, width=page_w,
                               height=page_w * 0.28))
            story.append(Spacer(1, 12))

    # ── RECOMMENDATIONS ───────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1,
                             color=BORDER, spaceAfter=10))
    story.append(P("📋  Recommendations", "Heading2",
                   fontSize=14, textColor=NAVY, spaceAfter=8))

    for i, rec in enumerate(_get_recommendations(metrics), 1):
        story.append(P(
            f"<b>{i}.</b>  {rec}",
            fontSize=10, textColor="#212529",
            spaceAfter=8, leading=15,
            leftIndent=10,
        ))

    # ── FOOTER ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=BORDER, spaceAfter=6))
    story.append(P(
        "Generated by <b>Lecture Speech Disfluency Detector</b>  ·  "
        "Speech Processing Course Project  ·  2024–25  ·  "
        "github.com/nyrahealth/CrisperWhisper",
        fontSize=8, textColor=MUTED, alignment=TA_CENTER,
    ))

    doc.build(story)
    print(f"\n  Report card saved → {out_path}")


if __name__ == "__main__":
    import json, sys, dataclasses
    from classify import TaggedWord
    from metrics import FluentyMetrics

    with open("demo/tagged.json") as f:
        tagged = [TaggedWord(**w) for w in json.load(f)]
    with open("demo/metrics.json") as f:
        metrics = FluentyMetrics(**json.load(f))

    audio = sys.argv[1] if len(sys.argv) > 1 else "demo/sample.wav"
    generate_report_card(
        metrics, tagged,
        audio_filename=os.path.basename(audio),
        speaker_name="Test Speaker",
    )
