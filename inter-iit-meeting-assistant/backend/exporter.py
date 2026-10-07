"""Human-readable exports (plain text, PDF) of a finished meeting record."""

from io import BytesIO
from xml.sax.saxutils import escape

_PDF_SAFE = {"\u20b9": "Rs.", "\u2192": "->", "\u2022": "-"}  # glyphs the built-in PDF font lacks


def _disp(v: str) -> str:
    return "Unspecified" if v == "unspecified" else v


def to_text(rec: dict) -> str:
    L = ["MEETING RECORD", "=" * 60,
         f"Source file : {rec['source_file']}",
         f"Generated   : {rec['generated_at']}",
         f"Models      : speech-to-text={rec['models']['speech_to_text']}; "
         f"refinement={rec['models']['refinement']}; extraction={rec['models']['extraction']}",
         "", "SUMMARY", "-" * 60, rec["summary"] or "(none)", "", "MINUTES", "-" * 60]
    if rec["minutes"]:
        for i, m in enumerate(rec["minutes"], 1):
            L.append(f"{i}. {m['topic']}")
            L += [f"   - {p}" for p in m["points"]]
    else:
        L.append("(none)")
    L += ["", "KEY DECISIONS", "-" * 60]
    L += [f"{i}. {d}" for i, d in enumerate(rec["key_decisions"], 1)] or ["No decisions were stated."]
    L += ["", "ACTION ITEMS", "-" * 60]
    if rec["action_items"]:
        for i, a in enumerate(rec["action_items"], 1):
            L += [f"{i}. {a['task']}", f"   Owner   : {_disp(a['owner'])}", f"   Deadline: {_disp(a['deadline'])}"]
    else:
        L.append("No action items were stated.")
    L += ["", "REFINED TRANSCRIPT", "-" * 60, rec["refined_transcript"],
          "", "RAW TRANSCRIPT", "-" * 60, rec["raw_transcript"], ""]
    return "\n".join(L)


def _p(text: str) -> str:
    for k, v in _PDF_SAFE.items():
        text = text.replace(k, v)
    return escape(text).replace("\n", "<br/>")


def to_pdf(rec: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"))
    h1, h2 = ss["Title"], ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#0a5fc4"), spaceBefore=12)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title="Meeting Record")
    s = [Paragraph("Meeting Record", h1),
         Paragraph(_p(f"Source: {rec['source_file']}  |  Generated: {rec['generated_at']}"), small),
         Paragraph(_p(f"Models - speech-to-text: {rec['models']['speech_to_text']}; refinement: "
                      f"{rec['models']['refinement']}; extraction: {rec['models']['extraction']}"), small),
         Paragraph("Summary", h2), Paragraph(_p(rec["summary"] or "(none)"), body),
         Paragraph("Minutes", h2)]
    if rec["minutes"]:
        for i, m in enumerate(rec["minutes"], 1):
            s.append(Paragraph(f"<b>{i}. {_p(m['topic'])}</b>", body))
            for pt in m["points"]:
                s.append(Paragraph("&bull; " + _p(pt), ParagraphStyle("pt", parent=body, leftIndent=12)))
    else:
        s.append(Paragraph("(none)", body))

    s.append(Paragraph("Key decisions", h2))
    if rec["key_decisions"]:
        for i, d in enumerate(rec["key_decisions"], 1):
            s.append(Paragraph(f"{i}. {_p(d)}", body))
    else:
        s.append(Paragraph("No decisions were stated.", body))

    s.append(Paragraph("Action items", h2))
    if rec["action_items"]:
        rows = [[Paragraph("<b>Task</b>", body), Paragraph("<b>Owner</b>", body), Paragraph("<b>Deadline</b>", body)]]
        for a in rec["action_items"]:
            rows.append([Paragraph(_p(a["task"]), body), Paragraph(_p(_disp(a["owner"])), body),
                         Paragraph(_p(_disp(a["deadline"])), body)])
        t = Table(rows, colWidths=[95 * mm, 40 * mm, 35 * mm], repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bbbbbb")),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf2fc")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        s.append(t)
    else:
        s.append(Paragraph("No action items were stated.", body))

    s += [Paragraph("Refined transcript", h2), Paragraph(_p(rec["refined_transcript"]), body),
          Paragraph("Raw transcript", h2), Paragraph(_p(rec["raw_transcript"]), body)]
    doc.build(s)
    return buf.getvalue()
