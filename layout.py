"""
Geometria da partitura — fonte única de verdade para o SVG (pré-visualização na web)
e para o PDF (exportação). Os renderizadores não decidem "onde fica a nota": apenas
traduzem as primitivas gráficas produzidas aqui.

Convenções aplicadas (Gould, "Behind Bars"; Read, "Music Notation", cap. 7):
  * pauta de percussão de 5 linhas com clave de percussão (caixa) e armadura de compasso;
  * voz 1 = pratos/chimel (hastes para cima), voz 2 = bumbo/caixa/toms (hastes para baixo);
  * pausas completam cada voz dentro do compasso; duração nunca cruza o beat, salvo
    quando começa exatamente no beat ou é sustain de prato;
  * vigas por divisao do tempo, com sub-vigas para semicolcheias/fusquinhas;
  * linhas suplementares (ledger) acima/abaixo da pauta;
  * compasso de repetição com barras diagonais + "%", acento (>), ghost (parênteses),
    flam (nota de adorno), roll/tremolo (barras na haste), "+" para prato aberto,
    sublinhado de virada (fill) e dinâmica por compasso (p/m/f).

Y cresce para CIMA (sistema do PDF); o SVG inverte. Unidades: pontos (1 pt = 1/72 in).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .kit import LANE_BY_ID
from .score_model import meter_info, score_visivel

PAGE_SIZES = {
    "a4_landscape": (841.89, 595.28),
    "a4_portrait": (595.28, 841.89),
    "letter_landscape": (792.0, 612.0),
}

Op = tuple


def lane_of(lane_id: str):
    return LANE_BY_ID.get(lane_id) or LANE_BY_ID["snare"]


def y_of_step(step: float, line1_y: float, ls: float) -> float:
    """Passo diatônico (1 = linha inferior da pauta; 1 unidade = meio espaço)."""
    return line1_y + (float(step) - 1.0) * (ls * 0.5)


def _dur_label(dur: int, tpb: int) -> str:
    from .rules import dur_label
    return dur_label(dur, tpb)[0]


def _beam_level(dur: int, tpb: int) -> int:
    """Número de vigas para uma duração (0 = haste livre com bandeira)."""
    if dur >= tpb:
        return 0
    if dur >= tpb // 2:
        return 1
    if dur >= max(1, (tpb // 4) or 1):
        return 2
    return 3


def _flag_count(dur: int, tpb: int) -> int:
    lvl = _beam_level(dur, tpb)
    if lvl == 0 and "eighth" in _dur_label(dur, tpb):
        return 1
    return lvl


# ------------------------------------------------------------------------------------
def layout_score(score: dict, page: str = "a4_landscape", ls: float = 9.0,
                 bars_per_system: Optional[int] = None, max_systems: int = 96,
                 show_legend: bool = True, compact: bool = False) -> dict:
    """
    Monta a partitura em páginas de primitivas gráficas.

    Retorna {"pages": [{"ops": [...], "w":, "h":}], "meta": {...}}.

    Pistas em `hide_lanes` são deixadas de fora da **gravura** (apresentação). Os dados enviados
    ficam intactos — auditoria, JSON e CSV continuam vendo todas as notas, de modo que ocultar
    um prato jamais pode esconder um defeito (docs/ERROS.md A16).
    """
    score = score_visivel(score)
    info = meter_info(score.get("meter", "4/4"))
    tpb, tpr = info["ticks_per_beat"], info["ticks_per_bar"]
    bars: List[dict] = list(score.get("bars") or [])
    if not bars:
        bars = [{"index": 0, "hits": [], "repeat_slash": False, "fill_marker": False}]
    pw, ph = PAGE_SIZES.get(page, PAGE_SIZES["a4_landscape"])
    m_left, m_right, m_bottom = 46.0, 40.0, (66.0 if show_legend else 34.0)
    ls = float(ls)

    # ----------------------------------------------------------- quebra em sistemas
    usable = pw - m_left - m_right
    # 1 sistema ocupa ~12 ls: 5,5 ls acima da linha inferior (pratos + viga) e
    # 4,5 ls abaixo (bumbo, hastes para baixo, dinâmica, número do compasso)
    need = 15.2 * ls        # 8.8 ls acima da linha inferior (pratos+viga) + 6.4 abaixo
    if bars_per_system is None:
        per = max(2, min(8, int(usable // (126.0 if not compact else 96.0))))
    else:
        per = max(1, int(bars_per_system))
    per = min(per, max(1, len(bars)))
    bar_w = min(usable / per, 176.0)
    per_sys_h = need
    head_h = (78.0 if not compact else 46.0)
    top_y = ph - head_h - 10.1 * ls
    systems = [bars[i:i + per] for i in range(0, len(bars), per)]
    systems = systems[:max_systems]

    pages: List[dict] = []
    ops: List[Op] = []
    bar_map: List[dict] = []          # geometria por compasso, para o cursor de reprodução
    y_first = top_y
    sys_in_page = 0
    max_sys_per_page = max(1, int((top_y - m_bottom - 5.8 * ls) // per_sys_h) + 1)

    def new_page(first: bool):
        nonlocal ops, y_first, sys_in_page
        ops = []
        y_first = top_y
        sys_in_page = 0
        pages.append({"ops": ops, "w": pw, "h": ph, "header": first})

    new_page(True)
    if not compact:
        draw_header(ops, score, info, m_left, ph, pw)
    sys_no = 0
    for chunk in systems:
        if sys_in_page >= max_sys_per_page:
            _flush(pages, ops, show_legend and not pages[-1].get("legend_done"), score, m_left, ph, ls)
            pages[-1]["legend_done"] = True
            new_page(False)
            if len(pages) > 1:
                draw_continuation(ops, score, m_left, ph)
        y1 = y_first - sys_in_page * 0.0            # posição do sistema corrente
        line1_y = y_first
        prev_b = systems[sys_no - 1][-1] if sys_no > 0 else None
        next_b = systems[sys_no + 1][0] if sys_no + 1 < len(systems) else None
        _draw_system(ops, chunk, line1_y, ls, tpb, tpr, score, info, m_left, pw - m_right,
                     bar_w, first_of_page=(sys_in_page == 0), prev_bar=prev_b, next_bar=next_b,
                     geom=bar_map, page_idx=len(pages) - 1, page_h=ph)
        y_first -= per_sys_h
        sys_in_page += 1
        sys_no += 1
    if show_legend and pages:
        add_legend(pages[-1]["ops"], ph, score, m_left)
    for i, pg in enumerate(pages):
        pg["n_pages"] = len(pages)
        pg["page_no"] = i + 1
    # páginas empilhadas: o documento SVG soma as alturas, e o mapa tem de falar a mesma língua
    for pg in bar_map:
        pg["y0"] = round(pg["y0"] + pg["page"] * ph, 2)
        pg["y1"] = round(pg["y1"] + pg["page"] * ph, 2)
        # `line1` ia junto com y0/y1? não — e ficava em espaço de página enquanto a caixa ia em
        # espaço do documento. Quem quisesse "a pauta deste compasso" erraria da 2ª página em diante
        # (595 pt). Todas as coordenadas do mapa falam, daqui em diante, a língua do documento.
        pg["line1"] = round(pg["line1"] + pg["page"] * ph, 2)
    # ordem de leitura: sistemas na ordem em que foram traçados (o cliente indexia por `i`)
    return {"pages": pages, "meta": {"page_w": pw, "page_h": ph, "systems": sys_no,
                                     "bars_per_system": per, "staff_space": ls,
                                     "n_bars": len(bars), "meter": info["name"],
                                     "ticks_per_bar": tpr, "n_pages": len(pages),
                                     "doc_h": round(ph * len(pages), 2),
                                     "bar_map": bar_map}}


def _flush(pages: List[dict], ops: List[Op], legend: bool, score: dict, m_left: float,
           ph: float, ls: float) -> None:
    pass


# ------------------------------------------------------------------------------------
def draw_header(ops: List[Op], score: dict, info: dict, x: float, ph: float, pw: float) -> None:
    title = str(score.get("title") or "Transcrição de Bateria")
    sub = str(score.get("subtitle") or "")
    rep = score.get("report") or {}
    ops.append(("text", x, ph - 34.0, title, 15.0, "bold", "left"))
    y = ph - 47.5
    if sub:
        ops.append(("text", x, y, sub, 8.4, "normal", "left"))
        y -= 10.5
    bpm = float(score.get("bpm") or 0.0)
    # "♩ = 100" desenhado (sem depender de glifo Unicode da fonte)
    ops.append(("noteglyph", x + 2.0, y + 2.6, 0.92, "quarter"))
    if bpm:
        lab = f"= {bpm:.0f} BPM" if abs(bpm - round(bpm)) < 0.51 else f"= {bpm:.1f} BPM"
        ops.append(("text", x + 12.0, y, lab, 10.0, "bold", "left"))
    meter = str(info.get("display") or score.get("meter") or "4/4")
    tx = x + 12.0 + 8.0 * len(f"{lab if bpm else ''}") + 14.0
    ops.append(("text", tx, y, f"compasso {meter}", 9.2, "normal", "left"))
    sw = float(score.get("swing") or 0.0)
    if sw > 0.02:
        ops.append(("text", tx + 76.0, y, f"swing {sw:.2f}", 9.2, "normal", "left"))
    y -= 11.5
    gq = rep.get("grid") or {}
    cls = rep.get("classification") or {}
    bits = []
    if gq.get("mode"):
        bits.append(f"grade {gq['mode']}")
    if gq.get("promoted_32nd"):
        bits.append(f"{gq['promoted_32nd']} notas em 32º")
    if isinstance(cls.get("mean_confidence"), (int, float)):
        bits.append(f"confiança média {cls['mean_confidence']:.2f}")
    if isinstance(rep.get("quality"), dict):
        q = rep["quality"]
        if q.get("hit_f1") is not None:
            bits.append(f"afinamento rítmico {100.0 * float(q.get('pos_accuracy', 0) or 0):.0f}%")
    if bits:
        ops.append(("text", x, y, "  ·  ".join(bits), 7.2, "normal", "left"))


def draw_continuation(ops: List[Op], score: dict, x: float, ph: float) -> None:
    title = str(score.get("title") or "Transcrição de Bateria")
    ops.append(("text", x, ph - 30.0, f"{title}  (continuação)", 10.0, "bold", "left"))


# ------------------------------------------------------------------------------------
def _extente_y(ops: List[Op], i0: int, i1: int) -> tuple:
    """ ymin, ymax (espaço de layout, +para cima) das primitivas de ops[i0:i1].

    Usado para o cursor de reprodução: a faixa vertical de um compasso é medida no que foi
    desenhado — feixe, hastes, número do compasso, dinâmicas — em vez de supor uma altura fixa,
    que descolaria da gravura assim que a notação mudasse.
    """
    ys: List[float] = []
    for op in ops[i0:i1]:
        k = op[0]
        if k == "line":
            ys += [float(op[2]), float(op[4])]
        elif k in ("path", "poly"):
            ys += [float(pt[1]) for pt in op[1]]
        elif k == "ellipse":
            ys += [float(op[2]) - float(op[4]), float(op[2]) + float(op[4])]
        elif k == "rect":
            ys += [float(op[2]), float(op[2]) + float(op[4])]
        elif k == "curve":
            ys += [float(op[2]), float(op[4]), float(op[6])]
        elif k == "text":
            ys += [float(op[2]), float(op[2]) + 0.75 * float(op[4])]
        elif k == "noteglyph":
            sc = float(op[3])
            ys += [float(op[2]), float(op[2]) + 9.2 * sc]
    if not ys:
        return (None, None)
    return (min(ys), max(ys))


def _draw_system(ops: List[Op], chunk: List[dict], line1_y: float, ls: float, tpb: int,
                 tpr: int, score: dict, info: dict, x_left: float, x_right: float,
                 bar_w: float, first_of_page: bool, prev_bar: Optional[dict] = None,
                 next_bar: Optional[dict] = None, geom: Optional[List[dict]] = None,
                 page_idx: int = 0, page_h: float = 0.0) -> None:
    for k in range(5):                              # linhas da pauta
        yy = line1_y + k * ls
        ops.append(("line", x_left, yy, x_right, yy, 0.7))
    x = x_left
    if first_of_page:
        x = _draw_clef_and_meter(ops, x_left, line1_y, ls, info)
    avail = x_right - x
    bw = min(bar_w, avail / max(1, len(chunk)))
    for bi, bar in enumerate(chunk):
        bx0, bx1 = x, x + bw
        pb = chunk[bi - 1] if bi > 0 else prev_bar
        nb = chunk[bi + 1] if bi + 1 < len(chunk) else next_bar
        i0 = len(ops)
        draw_bar(ops, bar, bx0, bx1, line1_y, ls, tpb, tpr, score, prev_bar=pb, next_bar=nb)
        if geom is not None:
            lo, hi = _extente_y(ops, i0, len(ops))
            # as 5 linhas do sistema valem para todos os compassos do sistema
            y_linhas = (line1_y, line1_y + 4 * ls)
            lo = min(lo, y_linhas[0]) if lo is not None else y_linhas[0] - 2.4 * ls
            hi = max(hi, y_linhas[1]) if hi is not None else y_linhas[1] + 2.4 * ls
            folga = 0.9 * ls
            # SVG inverte y (Y = h - y): o topo do bloco é o maior y de layout
            # `folga` cobre a tinta que passa da caixa do compasso (haste do 𝄄, nota fantasma,
            # vigas): a caixa x0/x1 continua exata para adjacência e hit-test, e quem desenha a
            # faixa de destaque soma a folga.
            geom.append({"i": int(bar.get("index", bi)), "page": int(page_idx),
                         "x0": round(bx0, 2), "x1": round(bx1, 2), "folga": 2.0,
                         "y0": round(page_h - (hi + folga), 2),
                         "y1": round(page_h - (lo - folga), 2),
                         "line1": round(page_h - line1_y, 2),
                         "beat0": round(bx0 + 5.0, 2), "beat1": round(bx1 - 5.0, 2)})
        x = bx1
    ops.append(("line", x + 1.1, line1_y, x + 1.1, line1_y + 4 * ls, 2.3))
    ops.append(("line", x + 4.2, line1_y, x + 4.2, line1_y + 4 * ls, 0.8))
    n0 = int(chunk[0].get("index", 0))
    ops.append(("text", x_left, line1_y + 10.0 * ls, str(n0 + 1), 7.4, "bold", "left"))


def _draw_clef_and_meter(ops: List[Op], x: float, line1_y: float, ls: float, info: dict) -> float:
    """Clave de percussão (caixa) + fórmula de compasso. Retorna o x seguinte."""
    cx = x + 6.0
    for off in (0.0, 3.2):
        ops.append(("rect", cx + off, line1_y + ls * 0.62, 2.2, ls * 2.76, True))
    tx = cx + 9.0
    ops.append(("text", tx, line1_y + ls * 3.2, str(info.get("beats", 4)), 12.0, "bold", "center"))
    ops.append(("text", tx, line1_y + ls * 1.35, "4", 12.0, "bold", "center"))
    return tx + 12.0


# ------------------------------------------------------------------------------------
def _tie_lanes(bar: Optional[dict], key: str) -> set:
    if not bar:
        return set()
    return {h.get("lane") for h in (bar.get("hits") or []) if h.get(key)}


def draw_bar(ops: List[Op], bar: dict, x0: float, x1: float, line1_y: float, ls: float,
             tpb: int, tpr: int, score: dict, prev_bar: Optional[dict] = None,
             next_bar: Optional[dict] = None) -> None:
    from .rules import fill_voice
    hits = list(bar.get("hits") or [])
    inner0, inner1 = x0 + 5.0, x1 - 5.0
    span = max(12.0, inner1 - inner0)

    def x_of(tick: float) -> float:
        return inner0 + (float(tick) / float(tpr)) * span

    # ---------------- compasso de repetição
    if bar.get("repeat_slash"):
        for k in range(2):
            yy = line1_y + ls * (1.5 + k)
            ops.append(("path", [(inner0 + 1.0, yy + 4.2), (inner0 + 15.0, yy - 4.2)], 1.4, "round"))
            ops.append(("ellipse", inner0 - 1.6, yy, 1.6, 1.25, 0.0, True))
            ops.append(("ellipse", inner0 + 17.6, yy, 1.6, 1.25, 0.0, True))
        n_rep = int(bar.get("repeat_count") or 0)
        if n_rep:
            ops.append(("text", (inner0 + inner1) / 2.0, line1_y + ls * 4.55, f"x{n_rep}",
                        8.0, "bold", "center"))
        ops.append(("line", x1, line1_y, x1, line1_y + 4 * ls, 0.9))
        return

    voices: Dict[int, List[dict]] = {}
    for v in (1, 2):
        hs = sorted([h for h in hits if lane_of(h["lane"]).voice == v],
                    key=lambda h: (int(h["tick"]), lane_of(h["lane"]).staff))
        voices[v] = fill_voice(hs, tpr, tpb)

    # ---------------- pausas primeiro (fica atrás), depois as notas
    for v in (1, 2):
        for it in voices[v]:
            if it["kind"] == "rest" and it.get("dur", 0) > 0:
                draw_rest(ops, x_of(it["tick"] + it["dur"] * 0.5), line1_y, ls,
                          int(it["dur"]), tpb)

    vel_sum, vel_n = 0, 0
    for h in hits:
        try:
            vel_sum += int(h.get("velocity", 88))
            vel_n += 1
        except Exception:
            pass
    if vel_n:
        mean_v = vel_sum / vel_n
        if mean_v < 58.0 or mean_v > 110.0:
            lab = "p" if mean_v < 58.0 else "f"
            ops.append(("text", x0 + 2.5, line1_y - ls * 5.0, lab, 8.6, "italic", "left"))

    ties_out = _tie_lanes(next_bar, "tie_stop")
    ties_in = _tie_lanes(prev_bar, "tie_start")
    for v in (1, 2):
        _draw_voice(ops, voices[v], v, x_of, inner0, inner1, line1_y, ls, tpb, tpr,
                    ties_out, ties_in)

    # ---------------- sublinhado de virada (fill)
    if bar.get("fill_marker"):
        fy = line1_y - ls * 3.6
        ops.append(("path", [(x0 + 1.5, fy + 1.6), (x0 + 1.5, fy), (x1 - 1.5, fy),
                            (x1 - 1.5, fy + 1.6)], 1.2, "round"))
    # ---------------- barra de compasso
    ops.append(("line", x1, line1_y, x1, line1_y + 4 * ls, 0.9))


def _draw_voice(ops: List[Op], items: List[dict], voice: int, x_of, inner0: float,
                inner1: float, line1_y: float, ls: float, tpb: int, tpr: int,
                ties_out: Optional[set] = None, ties_in: Optional[set] = None) -> None:
    stem_up = (voice == 1)
    notes: List[dict] = []
    for it in items:
        if it["kind"] != "note" or not it.get("hits"):
            continue
        nh = []
        for hh in it["hits"]:
            h = hh if isinstance(hh, dict) else hh.to_dict()
            ln = lane_of(h["lane"])
            step = float(h.get("staff", ln.staff))
            nh.append({"x": x_of(int(h.get("tick", it["tick"]))),
                       "y": y_of_step(step, line1_y, ls),
                       "head": h.get("head") or ln.head,
                       "artic": h.get("artic", "normal"),
                       "lane": h["lane"],
                       "step": step,
                       "tie_start": bool(h.get("tie_start")),
                       "tie_stop": bool(h.get("tie_stop")),
                       "vel": int(h.get("velocity", 88))})
        if not nh:
            continue
        notes.append({"tick": it["tick"], "dur": int(it["dur"]), "dots": int(it.get("dots") or 0),
                      "heads": nh, "level": _beam_level(int(it["dur"]), tpb),
                      "flags": _flag_count(int(it["dur"]), tpb)})
    if not notes:
        return
    # ---------------- grupos de viga: notas consecutivas curtas dentro da mesma divisao
    groups: List[List[int]] = []
    cur: List[int] = []
    for i, n in enumerate(notes):
        short = n["level"] >= 1
        if not short:
            if len(cur) > 1:
                groups.append(cur)
            cur = []
            continue
        if cur:
            p = notes[cur[-1]]
            gap_ok = (p["tick"] + p["dur"] + 1) >= n["tick"]
            same_beat = (p["tick"] // tpb) == (n["tick"] // tpb)
            if not (gap_ok and same_beat):
                if len(cur) > 1:
                    groups.append(cur)
                cur = []
        cur.append(i)
    if len(cur) > 1:
        groups.append(cur)
    in_group = {i for g in groups for i in g}

    for g in groups:
        xs = [notes[i]["heads"][0]["x"] for i in g]
        ys_ref = [(max(h["y"] for h in notes[i]["heads"]) if stem_up
                   else min(h["y"] for h in notes[i]["heads"])) for i in g]
        if stem_up:
            y_beam = max(ys_ref) + ls * 2.25
        else:
            y_beam = min(ys_ref) - ls * 2.25
        lvl = max(notes[i]["level"] for i in g)
        for j, i in enumerate(g):
            n = notes[i]
            for h in n["heads"]:
                sx = h["x"] + (2.85 if stem_up else -2.85)
                y_end = y_beam + (-2.2 if stem_up else 2.2)
                ops.append(("line", sx, h["y"] + (1.7 if stem_up else -1.7), sx, y_end, 1.0))
            _ledger(ops, n, ls, line1_y)
            _heads(ops, n, ls)
        for k in range(lvl):
            off = -(k * 2.5) if stem_up else (k * 2.5)
            ops.append(("path", [(xs[0], y_beam + off), (xs[-1], y_beam + off)], 2.5, "butt"))
        for k in range(2, lvl + 1):            # sub-vigas (semicolcheia/fusquinha)
            for j in range(len(g) - 1):
                a, b = notes[g[j]], notes[g[j + 1]]
                if a["level"] >= k and b["level"] >= k:
                    off = -(k * 2.5) if stem_up else (k * 2.5)
                    ops.append(("path", [(xs[j], y_beam + off), (xs[j + 1], y_beam + off)],
                                2.2, "butt"))
        for j, i in enumerate(g):              # ponto de aumento
            n = notes[i]
            if n["dots"]:
                for d in range(n["dots"]):
                    dx = xs[j] + 5.4 + d * 3.4
                    ops.append(("ellipse", dx, y_ref_of(n, stem_up) , 1.1, 1.1, 0.0, True))
    for i, n in enumerate(notes):              # notas fora de viga: haste + bandeira
        if i in in_group:
            continue
        y_ref = (max(h["y"] for h in n["heads"]) if stem_up else min(h["y"] for h in n["heads"]))
        sgn = 1.0 if stem_up else -1.0
        for h in n["heads"]:
            sx = h["x"] + sgn * 2.85
            ops.append(("line", sx, h["y"] + sgn * 1.7, sx, y_ref + sgn * ls * 2.9, 1.0))
        _ledger(ops, n, ls, line1_y)
        _heads(ops, n, ls)
        if n["dur"] >= 8:                       # semibreve/mínima: sem bandeira
            pass
        for f in range(n["flags"]):
            y0 = y_ref + sgn * ls * (2.9 - f * 0.95)
            x0 = n["heads"][0]["x"] + sgn * 2.85
            ops.append(("poly", [(x0, y0), (x0 + sgn * 3.7, y0 - sgn * 0.9),
                                (x0 + sgn * 1.3, y0 - sgn * 3.2), (x0, y0 - sgn * 2.1)],
                        0.8, True, True))
        if n["dots"]:
            for d in range(n["dots"]):
                ops.append(("ellipse", n["heads"][0]["x"] + 5.6 + d * 3.4, y_ref, 1.1, 1.1, 0.0, True))
        if n["flags"] == 0 and n["dur"] >= tpb:
            pass
    # ---------------- ligaduras e sustain de prato
    ties_out = ties_out or set()
    ties_in = ties_in or set()
    for n in notes:
        for h in n["heads"]:
            if h["tie_start"] and (not ties_out or h["lane"] in ties_out):
                d = -1.0 if stem_up else 1.0
                ops.append(("curve", h["x"] + 4.0, h["y"], (inner1 + h["x"]) / 2.0,
                            h["y"] + d * 6.2, inner1 + 3.0, h["y"] + d * 0.8, 0.85))
            if h["tie_stop"] and (not ties_in or h["lane"] in ties_in):
                d = -1.0 if stem_up else 1.0
                ops.append(("curve", inner0 - 3.0, h["y"] + d * 0.8, (inner0 + h["x"]) / 2.0,
                            h["y"] + d * 6.2, h["x"] - 4.0, h["y"], 0.85))


def y_ref_of(n: dict, stem_up: bool) -> float:
    ys = [h["y"] for h in n["heads"]]
    return (max(ys) if stem_up else min(ys))


def _heads(ops: List[Op], n: dict, ls: float) -> None:
    for h in n["heads"]:
        dl = _dur_label(n["dur"], 8)
        filled = not any(t in dl for t in ("half", "whole", "breve"))
        draw_notehead(ops, h["x"], h["y"], h["head"], filled, ls)
        art = h["artic"]
        up = lane_of(h["lane"]).stem == "up"
        ay = h["y"] + (ls * 1.05 if up else -ls * 1.05)
        if art in ("accent", "ghost_accent"):
            d = 1.0 if up else -1.0
            ops.append(("path", [(h["x"] - 3.4, ay + 2.3 * d), (h["x"] + 3.0, ay),
                                (h["x"] - 3.4, ay - 2.3 * d)], 1.0, "round"))
        if art in ("ghost", "ghost_accent"):
            for s in (-1.0, 1.0):
                ops.append(("path", [(h["x"] + s * 7.0, ay - 2.6), (h["x"] + s * 8.6, ay),
                                    (h["x"] + s * 7.0, ay + 2.6)], 0.7, "round"))
        if art == "flam":
            gx = h["x"] - 9.5
            ops.append(("ellipse", gx, h["y"] + (1.0 if up else -1.0), ls * 0.24, ls * 0.17, -18.0, True))
            ops.append(("line", gx + 1.9, h["y"] + (0.8 if up else -0.8),
                        gx + 1.9, h["y"] + (ls * 1.6 if up else -ls * 1.6), 0.85))
        if art in ("droll", "roll", "diddle"):
            sgn = 1.0 if up else -1.0
            for k in range(2):
                yy = h["y"] + sgn * (ls * 1.55 + k * 1.6)
                ops.append(("path", [(h["x"] - 3.2, yy), (h["x"] + 3.2, yy - sgn * 2.2)], 0.9, "round"))
        if h["lane"] == "hat_open":
            ops.append(("text", h["x"] + 8.0, h["y"] + (ls * 1.15 if up else -ls * 1.9), "+",
                        8.0, "bold", "center"))
        if h["lane"] in ("crash", "ride", "hat_open", "splash") and n["dur"] >= 8:
            sgn = 1.0 if up else -1.0
            yy = h["y"] + sgn * ls * 1.75
            ops.append(("path", [(h["x"] - 1.5, yy), (h["x"] + 9.5, yy)], 0.7, "round"))


def _ledger(ops: List[Op], n: dict, ls: float, line1_y: float) -> None:
    """Linhas suplementares para notas fora da pauta."""
    for h in n["heads"]:
        s = int(round(h["step"]))
        if s < 1:
            L = -1
            while L >= s - 1:
                yy = y_of_step(L, line1_y, ls)
                ops.append(("line", h["x"] - 5.6, yy, h["x"] + 5.6, yy, 0.75))
                L -= 2
        elif s > 9:
            L = 11
            while L <= s + 1:
                yy = y_of_step(L, line1_y, ls)
                ops.append(("line", h["x"] - 5.6, yy, h["x"] + 5.6, yy, 0.75))
                L += 2


def draw_notehead(ops: List[Op], x: float, y: float, head: str, filled: bool, ls: float) -> None:
    rx, ry = ls * 0.37, ls * 0.26
    if head == "x":
        d = rx * 0.98
        ops.append(("path", [(x - d, y - d * 0.66), (x + d, y + d * 0.66)], 1.45, "round"))
        ops.append(("path", [(x - d, y + d * 0.66), (x + d, y - d * 0.66)], 1.45, "round"))
    elif head == "minus":
        ops.append(("ellipse", x, y, rx * 0.86, ry * 0.86, -18.0, True))
        ops.append(("path", [(x - rx * 1.05, y + ry * 1.15), (x + rx * 1.05, y - ry * 1.15)],
                    0.85, "butt"))
    elif head == "diamond":
        d = rx * 1.12
        ops.append(("poly", [(x, y + d * 0.78), (x + d, y), (x, y - d * 0.78), (x - d, y)],
                    0.95, True, filled))
    elif head == "triangle":
        d = rx * 1.05
        ops.append(("poly", [(x - d, y - d * 0.7), (x + d, y - d * 0.7), (x, y + d * 0.85)],
                    0.95, True, filled))
    elif head == "slash":
        ops.append(("path", [(x - rx, y - ry * 1.4), (x + rx, y + ry * 1.4)], 1.7, "butt"))
    else:
        ops.append(("ellipse", x, y, rx, ry, -18.0, filled))


def draw_rest(ops: List[Op], x: float, line1_y: float, ls: float, dur: int, tpb: int) -> None:
    dl = _dur_label(dur, tpb)
    ymid = line1_y + ls * 2.0
    if "whole" in dl or dur >= 4 * tpb:
        ops.append(("rect", x - 3.0, ymid + ls * 0.55, 6.0, ls * 0.45, False))
    elif "half" in dl or dur >= 2 * tpb:
        ops.append(("rect", x - 3.0, ymid - ls * 0.05, 6.0, ls * 0.45, False))
    elif "quarter" in dl or dur >= tpb:
        ops.append(("poly", [(x + 1.5, ymid + ls * 1.05), (x - 1.3, ymid + ls * 0.4),
                            (x + 1.5, ymid - ls * 0.25), (x - 1.0, ymid - ls * 0.9)],
                    1.3, False, False))
        ops.append(("ellipse", x - 2.2, ymid - ls * 1.05, 1.5, 1.15, 32.0, True))
    else:
        n = max(1, _beam_level(dur, tpb))
        yy = ymid + ls * 0.15
        ops.append(("ellipse", x - 1.5, yy, 1.6, 1.2, 34.0, True))
        ops.append(("path", [(x - 0.5, yy + 0.5), (x + 2.3, yy + ls * 1.05)], 0.95, "round"))
        for k in range(n):
            yy2 = yy + ls * (1.05 - 0.36 * k)
            ops.append(("path", [(x + 2.3, yy2), (x + 5.2, yy2 - ls * 0.4)], 1.45, "round"))


# ------------------------------------------------------------------------------------
def add_legend(ops: List[Op], ph: float, score: dict, x: float, ls: float = 8.0) -> None:
    """Legenda do mapa de teclado, apenas com as pistas usadas nesta partitura."""
    used: List[str] = []
    for b in score.get("bars", []):
        for h in b.get("hits", []):
            if h.get("lane") and h["lane"] not in used:
                used.append(h["lane"])
    if not used:
        return
    order = ["kick", "snare", "rim", "hat_foot", "hat", "hat_open", "ride", "crash", "splash",
             "tom_hi", "tom_mid", "tom_low", "cowbell"]
    used = [u for u in order if u in used] + [u for u in used if u not in order]
    cols = min(4, max(1, len(used)))
    rows = math.ceil(len(used) / cols)
    y0 = 20.0 + (rows - 1) * 11.5
    colw = max(120.0, (560.0 / cols))
    for i, lid in enumerate(used):
        ln = lane_of(lid)
        c, r = i % cols, i // cols
        cx = x + c * colw
        cy = y0 + (rows - 1 - r) * 11.5
        draw_notehead(ops, cx + 4.0, cy + 2.6, ln.head, True, 7.6)
        ops.append(("text", cx + 13.0, cy, f"{ln.name} ({ln.short})", 7.4, "normal", "left"))


def ops_to_svg(page: dict, score: dict, opts: Optional[dict] = None) -> str:
    """Tradução das primitivas para SVG (usado pelo navegador e por testes visuais)."""
    w, h = page["w"], page["h"]
    out: List[str] = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.2f} {h:.2f}" '
                      f'width="{w:.2f}" height="{h:.2f}" font-family="Helvetica,Arial,sans-serif">',
                      f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>']
    out.extend(ops_to_svg_body(page["ops"], h))
    out.append("</svg>")
    return "".join(out)


def ops_to_svg_body(ops: List[Op], h: float) -> List[str]:
    def Y(y: float) -> float:
        return h - float(y)

    res: List[str] = []
    for op in ops:
        k = op[0]
        if k == "line":
            _, x1, y1, x2, y2, wd = op
            res.append(f'<line x1="{x1:.2f}" y1="{Y(y1):.2f}" x2="{x2:.2f}" y2="{Y(y2):.2f}" '
                       f'stroke="#111" stroke-width="{wd:.2f}"/>')
        elif k == "path":
            pts, wd, cap = op[1], op[2], (op[3] if len(op) > 3 else "round")
            p = " ".join(f"{a:.2f},{Y(b):.2f}" for a, b in pts)
            res.append(f'<polyline points="{p}" fill="none" stroke="#111" stroke-width="{wd:.2f}" '
                       f'stroke-linecap="{cap}" stroke-linejoin="round"/>')
        elif k == "poly":
            pts, wd, close, filled = op[1], op[2], op[3], op[4]
            p = " ".join(f"{a:.2f},{Y(b):.2f}" for a, b in pts)
            fill = "#111" if filled else "none"
            res.append(f'<polygon points="{p}" fill="{fill}" stroke="#111" stroke-width="{wd:.2f}"/>')
        elif k == "ellipse":
            _, cx, cy, rx, ry, rot, filled = op
            fill = "#111" if filled else "none"
            tr = f' transform="rotate({-rot:.1f} {cx:.2f} {Y(cy):.2f})"' if rot else ""
            res.append(f'<ellipse cx="{cx:.2f}" cy="{Y(cy):.2f}" rx="{rx:.2f}" ry="{ry:.2f}" '
                       f'fill="{fill}" stroke="#111" stroke-width="0.7"{tr}/>')
        elif k == "rect":
            _, x, y, rw, rh, filled = op
            fill = "#111" if filled else "none"
            res.append(f'<rect x="{x:.2f}" y="{Y(y + rh):.2f}" width="{rw:.2f}" height="{rh:.2f}" '
                       f'fill="{fill}" stroke="#111" stroke-width="0.6"/>')
        elif k == "curve":
            _, x1, y1, cx, cy, x2, y2, wd = op
            res.append(f'<path d="M{x1:.2f},{Y(y1):.2f} Q{cx:.2f},{Y(cy):.2f} {x2:.2f},{Y(y2):.2f}" '
                       f'fill="none" stroke="#111" stroke-width="{wd:.2f}" stroke-linecap="round"/>')
        elif k == "text":
            x, y, s, size, weight, align = op[1], op[2], op[3], op[4], op[5], op[6]
            anchor = {"left": "start", "center": "middle", "right": "end"}.get(align, "start")
            style = f'font-size="{size:.2f}" font-weight="{weight}" font-style="{"italic" if weight=="italic" else "normal"}"'
            res.append(f'<text x="{x:.2f}" y="{Y(y):.2f}" text-anchor="{anchor}" fill="#111" {style}>{_esc(s)}</text>')
        elif k == "noteglyph":
            x, y, sc, kind = op[1], op[2], op[3], op[4]
            res.append(f'<ellipse cx="{x + 3.2 * sc:.2f}" cy="{Y(y + 1.6 * sc):.2f}" '
                       f'rx="{2.4 * sc:.2f}" ry="{1.8 * sc:.2f}" fill="#111" transform="rotate(-20 '
                       f'{x + 3.2 * sc:.2f} {Y(y + 1.6 * sc):.2f})"/>')
            res.append(f'<line x1="{x + 5.4 * sc:.2f}" y1="{Y(y + 1.4 * sc):.2f}" '
                       f'x2="{x + 5.4 * sc:.2f}" y2="{Y(y + 9.2 * sc):.2f}" stroke="#111" stroke-width="{1.0 * sc:.2f}"/>')
    return res


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def render_svg_document(score: dict, layout: Optional[dict] = None, **opts) -> str:
    """Um SVG por página, empilhados num <div>-ish (SVG único com várias páginas)."""
    lay = layout or layout_score(score, **opts)
    pages = lay["pages"]
    if not pages:
        return '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
    w, h = pages[0]["w"], pages[0]["h"]
    gap = 0.0
    tot_h = h * len(pages) + gap * (len(pages) - 1)
    body: List[str] = []
    for i, pg in enumerate(pages):
        body.append(f'<g transform="translate(0 {i * (h + gap):.2f})">')
        body.append(f'<rect x="0" y="0" width="{w:.2f}" height="{h:.2f}" fill="#fff"/>')
        body.extend(ops_to_svg_body(pg["ops"], h))
        if len(pages) > 1:
            body.append(f'<text x="{w - 20:.2f}" y="{h - 12:.2f}" text-anchor="end" '
                        f'font-size="7.5" fill="#666">{i + 1} / {len(pages)}</text>')
        body.append("</g>")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.2f} {tot_h:.2f}" '
            f'width="{w:.2f}" height="{tot_h:.2f}" font-family="Helvetica,Arial,sans-serif">'
            + "".join(body) + "</svg>")
