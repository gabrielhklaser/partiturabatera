"""
Exportação: PDF (nativo, reportlab) e SVG (navegador), ambos desenhados pelo mesmo
layout (`drumscribe.layout`) — o que se vê na tela é exatamente o que sai no papel.

Sem dependência de LilyPond/Finale: as primitivas gráficas do layout são traduzidas
direto para o PDF, com fontes base-14 (Helvetica) e glifos desenhados por nós.
"""
from __future__ import annotations

import io
import os
from typing import Dict, List, Optional

from .layout import layout_score, render_svg_document, lane_of, y_of_step


# --------------------------------------------------------------------------- SVG
def to_svg(score: dict, **opts) -> str:
    return render_svg_document(score, None, **opts)


def to_svg_pages(score: dict, **opts) -> List[str]:
    lay = layout_score(score, **opts)
    from .layout import ops_to_svg
    return [ops_to_svg(pg, score) for pg in lay["pages"]]


# --------------------------------------------------------------------------- PDF
def to_pdf_bytes(score: dict, page: str = "a4_landscape", ls: float = 9.0,
                 bars_per_system: Optional[int] = None, max_systems: int = 96,
                 show_legend: bool = True, compact: bool = False) -> bytes:
    from reportlab.pdfgen import canvas as _cv
    from reportlab.lib.pagesizes import landscape, letter
    lay = layout_score(score, page=page, ls=ls, bars_per_system=bars_per_system,
                       max_systems=max_systems, show_legend=show_legend, compact=compact)
    pages = lay["pages"]
    buf = io.BytesIO()
    if not pages:
        c = _cv.Canvas(buf, pagesize=landscape(letter))
        c.drawString(48, 400, "sem eventos")
        c.save()
        return buf.getvalue()
    c = _cv.Canvas(buf, pagesize=(pages[0]["w"], pages[0]["h"]),
                   initialFontName="Helvetica", initialFontSize=9.0)
    c.setTitle(str(score.get("title") or "DrumScribe"))
    c.setAuthor("DrumScribe")
    c.setSubject("partitura de bateria gerada a partir de audio")
    for pg in pages:
        _paint(c, pg["ops"], float(pg["w"]), float(pg["h"]), pg.get("page_no", 1),
               len(pages))
        c.showPage()
    c.save()
    return buf.getvalue()


def _paint(c, ops: List[tuple], w: float, h: float, pno: int, npages: int) -> None:
    c.saveState()
    c.setFillColorRGB(0.07, 0.07, 0.07)
    c.setStrokeColorRGB(0.07, 0.07, 0.07)
    c.setLineWidth(0.7)
    c.setLineCap(1)
    c.setLineJoin(1)
    for op in ops:
        k = op[0]
        if k == "line":
            _, x1, y1, x2, y2, wd = op
            c.setLineWidth(float(wd))
            c.setStrokeColorRGB(0.07, 0.07, 0.07)
            c.line(float(x1), float(y1), float(x2), float(y2))
        elif k == "path":
            pts = op[1]
            wd = op[2]
            cap = op[3] if len(op) > 3 else "round"
            c.setLineWidth(float(wd))
            c.setLineCap(1 if cap == "round" else 0)
            p = c.beginPath()
            pts = list(pts)
            if not pts:
                continue
            p.moveTo(float(pts[0][0]), float(pts[0][1]))
            for a, b in pts[1:]:
                p.lineTo(float(a), float(b))
            c.drawPath(p, stroke=1, fill=0)
        elif k == "poly":
            pts, wd, close, filled = op[1], op[2], op[3], op[4]
            p = c.beginPath()
            pts = list(pts)
            if not pts:
                continue
            p.moveTo(float(pts[0][0]), float(pts[0][1]))
            for a, b in pts[1:]:
                p.lineTo(float(a), float(b))
            if close:
                p.close()
            c.setLineWidth(float(wd))
            c.drawPath(p, stroke=1, fill=1 if filled else 0)
        elif k == "ellipse":
            _, cx, cy, rx, ry, rot, filled = op
            rx, ry = float(rx), float(ry)
            c.saveState()
            c.translate(float(cx), float(cy))
            if rot:
                c.rotate(float(rot))
            p = c.beginPath()
            p.ellipse(-rx, -ry, rx, ry)
            c.drawPath(p, stroke=1, fill=1 if filled else 0)
            c.restoreState()
        elif k == "rect":
            _, x, y, rw, rh, filled = op
            c.rect(float(x), float(y), float(rw), float(rh), stroke=1, fill=1 if filled else 0)
        elif k == "curve":
            # quadrática -> cúbica (controles a 2/3 do caminho), como o SVG faz com Q
            _, x1, y1, cx, cy, x2, y2, wd = op
            c1x, c1y = x1 + 2.0 / 3.0 * (cx - x1), y1 + 2.0 / 3.0 * (cy - y1)
            c2x, c2y = x2 + 2.0 / 3.0 * (cx - x2), y2 + 2.0 / 3.0 * (cy - y2)
            c.setLineWidth(float(wd))
            p = c.beginPath()
            p.moveTo(float(x1), float(y1))
            p.curveTo(float(c1x), float(c1y), float(c2x), float(c2y), float(x2), float(y2))
            c.drawPath(p, stroke=1, fill=0)
        elif k == "text":
            x, y, s, size, weight, align = op[1], op[2], op[3], op[4], op[5], op[6]
            s = _ascii(str(s))
            fnt = "Helvetica"
            if weight == "bold":
                fnt = "Helvetica-Bold"
            elif weight == "italic":
                fnt = "Helvetica-Oblique"
            c.setFont(fnt, float(size))
            col = _colour(weight)
            c.setFillColorRGB(*col)
            if align == "center":
                c.drawCentredString(float(x), float(y), s)
            elif align == "right":
                c.drawRightString(float(x), float(y), s)
            else:
                c.drawString(float(x), float(y), s)
        elif k == "noteglyph":
            x, y, sc, kind = op[1], op[2], op[3], op[4]
            sc = float(sc)
            c.saveState()
            c.translate(float(x) + 3.2 * sc, float(y) + 1.6 * sc)
            c.rotate(-20.0)
            p = c.beginPath()
            p.ellipse(-2.4 * sc, -1.8 * sc, 2.4 * sc, 1.8 * sc)
            c.drawPath(p, stroke=0, fill=1)
            c.restoreState()
            c.setLineWidth(1.0 * sc)
            c.line(float(x) + 5.4 * sc, float(y) + 1.4 * sc, float(x) + 5.4 * sc,
                   float(y) + 9.2 * sc)
    if npages > 1:
        c.setFont("Helvetica", 7.5)
        c.setFillColorRGB(0.42, 0.42, 0.42)
        c.drawRightString(w - 20.0, 12.0, f"{pno} / {npages}")
    c.restoreState()


def _colour(weight: str):
    if weight == "note":
        return (0.15, 0.35, 0.75)
    if weight == "muted":
        return (0.45, 0.45, 0.45)
    return (0.07, 0.07, 0.07)


_MAP = {"\u266a": "= ", "\u2669": "= ", "x": "x", "·": "·", "—": "-", "≈": "~", "°": "o", "×": "x"}


def _ascii(s: str) -> str:
    """Latin-1 seguro para as fontes base-14; substitui glifos que a fonte não tem."""
    for k, v in _MAP.items():
        s = s.replace(k, v)
    return "".join(ch if ord(ch) < 256 else "?" for ch in s)


def write_pdf(score: dict, path: str, **opts) -> str:
    data = to_pdf_bytes(score, **opts)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path
