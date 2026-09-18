"""
Exportação MusicXML 4.0 — partitura de percussão, pauta única de 5 linhas, duas vozes.

Mapeamento (pensado para abrir no MuseScore 4 / Finale / Dorico sem advertências):
  * `divisions = 8`: nossa unidade de tick é a semicolcheia da semínima → durações exatas;
  * clave de percussão (`sign="perc" line="2"`) e `<staff-position>` por nota, porque a
    posição das peças do kit não obedece à clave de sol "de altura";
  * voz 1 = pratos/chimel (hastes para cima), voz 2 = bumbo/caixa/toms, com `<backup>`;
  * notehead `x` (pratos), `circle-x` (chimel fechado/pedal), `minus` (fantasma),
    `diamond` (chimel aberto); acento em `<artics><accent/>`, rufola em `<artics><roll/>`,
    flam como nota de adorno `<grace/>` + `<cue/>`;
  * compasso de repetição (time-slash) escrito como pausa + `%` em `<words>` (o MusicXML não
    tem `bar-style` para "compass of repeats"), e `<repeat>` quando há repetição de seção.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

import math

from .kit import LANE_BY_ID
from .score_model import meter_info, score_visivel

DIVISIONS = 8
TYPE_BY_DUR = {1: "32nd", 2: "16th", 4: "eighth", 8: "quarter", 16: "half", 32: "whole",
               64: "breve"}


def _dur_parts(dur: int) -> Tuple[str, int]:
    d = max(1, int(dur))
    if d in TYPE_BY_DUR:
        return TYPE_BY_DUR[d], 0
    for base in sorted(TYPE_BY_DUR, reverse=True):
        if d == base + base // 2:
            return TYPE_BY_DUR[base], 1
        if d == base + base // 2 + base // 4:
            return TYPE_BY_DUR[base], 2
    for base in sorted(TYPE_BY_DUR):
        if d <= base:
            return TYPE_BY_DUR[base], 0
    return "32nd", 0


def _beam_level(dur: int, tpb: int) -> int:
    if dur >= tpb:
        return 0
    if dur >= max(1, tpb // 2):
        return 1
    if dur >= max(1, tpb // 4):
        return 2
    return 3


def _beam_plan(items: List[dict], tpb: int) -> Dict[int, List[Tuple[int, str]]]:
    groups: List[List[int]] = []
    cur: List[int] = []
    for i, it in enumerate(items):
        if it["rest"] or _beam_level(it["dur"], tpb) == 0:
            if len(cur) > 1:
                groups.append(cur)
            cur = []
            continue
        if cur:
            p = items[cur[-1]]
            same_beat = (p["tick"] // tpb) == (it["tick"] // tpb)
            if not same_beat or (p["tick"] + p["dur"] + 1) < it["tick"]:
                if len(cur) > 1:
                    groups.append(cur)
                cur = []
        cur.append(i)
    if len(cur) > 1:
        groups.append(cur)
    plan: Dict[int, List[Tuple[int, str]]] = {}
    for g in groups:
        lvl = max(_beam_level(items[i]["dur"], tpb) for i in g)
        for k in range(1, lvl + 1):
            mem = [i for i in g if _beam_level(items[i]["dur"], tpb) >= k]
            runs: List[List[int]] = []
            for j in mem:
                if runs and runs[-1][-1] + 1 == j:
                    runs[-1].append(j)
                else:
                    runs.append([j])
            for run in runs:
                if len(run) < 2:      # nota isolada = bandeira, não viga
                    continue
                for p, i in enumerate(run):
                    tag = "begin" if p == 0 else ("end" if p == len(run) - 1 else "continue")
                    plan.setdefault(i, []).append((k, tag))
    return plan


def _noteheads(lane: str, hit: dict) -> str:
    art = hit.get("artic", "normal")
    head = hit.get("head") or (LANE_BY_ID.get(lane) or LANE_BY_ID["snare"]).head
    if art in ("ghost", "ghost_accent") or head == "minus":
        return '<notehead>minus</notehead>'
    if head == "x":
        return '<notehead>circle-x</notehead>' if lane in ("hat", "hat_foot") else '<notehead>x</notehead>'
    if head == "diamond":
        return '<notehead>diamond</notehead>'
    if head == "triangle":
        return '<notehead>triangle</notehead>'
    return ""


def to_musicxml(score: dict, part_name: str = "Bateria") -> str:
    score = score_visivel(score)      # idem MIDI: XML segue a apresentação, os dados não
    info = meter_info(score.get("meter", "4/4"))
    tpb, tpr = info["ticks_per_beat"], info["ticks_per_bar"]
    beats = info["beats"]
    beat_type = 4 if info["kind"] == "simple" else 8
    bars = score.get("bars") or []
    out: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"'
        ' "http://www.musicxml.org/dtds/partwise.dtd">',
        '<score-partwise version="4.0">',
        f"  <work><work-title>{escape(str(score.get('title') or 'Transcrição de Bateria'))}"
        "</work-title></work>",
    ]
    if score.get("subtitle"):
        out.append(f"  <identification><encoding><software>DrumScribe"
                   f"</software></encoding><rights>{escape(str(score['subtitle']))}</rights>"
                   "</identification>")
    out += ["  <part-list>",
            f'    <score-part id="P1"><part-name>{escape(part_name)}</part-name>'
            "<part-abbreviation>Dr.</part-abbreviation></score-part>",
            "  </part-list>", '  <part id="P1">']
    n_rep = int(score.get("repeats", 1) or 1)
    ties = _valid_ties(score, tpr)
    for bi, bar in enumerate(bars):
        out.append(f'    <measure number="{int(bar.get("index", bi)) + 1}">')
        if bi == 0:
            out += ["      <attributes>", f"        <divisions>{DIVISIONS}</divisions>",
                    "        <key><fifths>0</fifths></key>",
                    f"        <time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time>",
                    "        <staves>1</staves>",
                    '        <clef number="1"><sign>perc</sign><line>2</line></clef>',
                    "      </attributes>"]
            bpm = float(score.get("bpm") or 0.0)
            if bpm:
                out += ['      <direction placement="above"><direction-type><metronome>',
                        "          <beat-unit>quarter</beat-unit>",
                        f"          <per-minute>{bpm:.0f}</per-minute>",
                        "        </metronome></direction-type></direction>",
                        f'      <sound tempo="{bpm:.0f}"/>']
            if float(score.get("swing") or 0.0) > 0.02:
                out.append('      <direction><direction-type><words>swing</words>'
                           "</direction-type></direction>")
        by_voice: Dict[int, List[dict]] = {1: [], 2: []}
        grouped: Dict[int, Dict[int, List[dict]]] = {1: {}, 2: {}}
        for h in sorted(bar.get("hits") or [], key=lambda x: (int(x["tick"]),
                                                              _step_of(x.get("lane", "snare"), x))):
            ln = LANE_BY_ID.get(h.get("lane", "snare")) or LANE_BY_ID["snare"]
            v = int(ln.voice)
            grouped[v].setdefault(int(h["tick"]), []).append(h)
        for v in (1, 2):
            for tk in sorted(grouped[v]):
                hs = grouped[v][tk]
                main = hs[0]
                dur = max(max(1, int(x.get("dur") or 2)) for x in hs)
                by_voice[v].append({"tick": tk, "dur": dur, "rest": False, "hit": main,
                                    "lane": main["lane"], "step": _step_of(main["lane"], main),
                                    "stem": LANE_BY_ID.get(main.get("lane"), ln).stem,
                                    "mates": hs[1:]})
        if bar.get("repeat_slash"):
            for v in (1, 2):
                if not by_voice[v]:
                    by_voice[v] = [{"tick": 0, "dur": tpr, "rest": True, "hit": None,
                                    "lane": None, "step": 5, "stem": None}]
            out.append('      <direction placement="above"><direction-type><words>%</words>'
                       "</direction-type></direction>")
        first = True
        bi_tie = int(bar.get("index", bi))
        for v in (1, 2):
            items = _fill_rests(by_voice[v], tpr)
            for it in items:
                it["_bar"] = bi_tie
            if not first:
                out.append(f"      <backup><duration>{tpr}</duration></backup>")
            first = False
            _emit(out, items, v, tpb, tpr, ties)
        out.append('      <barline location="right">')
        if bi == 0 and n_rep > 1:
            out.append('        <repeat direction="forward"/>')
        if bi == len(bars) - 1 and n_rep > 1:
            out.append('        <repeat direction="backward"/>')
        out.append("      </barline>")
        out.append("    </measure>")
    out += ["  </part>", "</score-partwise>"]
    return "\n".join(out)


def _step_of(lane: str, hit: dict) -> int:
    ln = LANE_BY_ID.get(lane) or LANE_BY_ID["snare"]
    return int(hit.get("staff", ln.staff))


def _fill_rests(notes: List[dict], tpr: int) -> List[dict]:
    """
    Timeline de uma voz: uma voz do MusicXML é uma sequência estrita de eventos, então a
    duração de cada um tem de parar no próximo ataque (a menos que a nota esteja ligada ao
    próximo golpe, caso em que ela *deve* encostar nele) e o compasso tem de fechar exatamente
    em `tpr` — se somar mais, o importador empurra as notas para a frente; se somar menos,
    ele acusa compasso incompleto.
    """
    items: List[dict] = []
    t = 0
    for n in sorted(notes, key=lambda x: int(x["tick"])):
        if n["tick"] > t:
            items.append({"tick": t, "dur": n["tick"] - t, "rest": True, "hit": None,
                          "lane": None, "step": 5, "stem": None})
        nxt = None
        h = n.get("hit")
        tied = bool(h is not None and h.get("tie_start"))
        items.append(n)
        end = int(n["tick"]) + max(1, int(n["dur"]))
        t = max(t, end)
    # recorta durações que invadem o ataque seguinte e completa o compasso
    real = [i for i in items if not i["rest"]]
    onsets = sorted({int(i["tick"]) for i in real})
    for i in real:
        tk = int(i["tick"])
        k = onsets.index(tk)
        limit = (onsets[k + 1] if k + 1 < len(onsets) else tpr) - tk
        dur = max(1, int(i["dur"]))
        if dur > limit:
            i["dur"] = max(1, limit)
    t = 0
    for i in items:
        t = max(t, int(i["tick"]) + max(1, int(i["dur"])))
    if t < tpr:
        items.append({"tick": t, "dur": tpr - t, "rest": True, "hit": None, "lane": None,
                      "step": 5, "stem": None})
    return sorted(items, key=lambda x: int(x["tick"]))


def _valid_ties(score: dict, tpr: int) -> Tuple[set, set]:
    """
    Ligaduras só podem ser emitidas quando têm parceiro: um `<tie type="start">` sem `<stop>`
    correspondente faz o Finale/MuseScore reclamar (e, pior, muda a duração soada). varremos
    por peça e aceitamos apenas cadeias em que o próximo golpe da mesma pista começa exatamente
    onde o anterior termina — dentro do compasso ou no início do seguinte.
    """
    per: Dict[int, List[dict]] = {}
    for bar in score.get("bars") or []:
        bi = int(bar.get("index", 0))
        for h in bar.get("hits") or []:
            g = dict(h)
            g["bar"] = bi
            ln = LANE_BY_ID.get(str(h.get("lane")))
            per.setdefault(int(ln.voice) if ln else 2, []).append(g)
    ok_start: set = set()
    ok_stop: set = set()
    for voice_i, lst in per.items():
        lst.sort(key=lambda x: (x["bar"], int(x.get("tick", 0))))
        # a corrente é conferida POR PISTA: `tie` no MusicXML pertence à nota (cada peça tem a
        # sua), enquanto a linha do tempo é por voz. Chave por voz+posição faria um acorde
        # herdar a ligadura do vizinho — um <tie start/> a mais, sem destino.
        for i, h in enumerate(lst):
            if not h.get("tie_start"):
                continue
            lane_h = str(h.get("lane"))
            end_abs = h["bar"] * tpr + int(round(_f(h.get("tick")))) + max(1, int(_f(h.get("dur"), 1)))
            nxt = next((x for x in lst[i + 1:]
                        if str(x.get("lane")) == lane_h and x.get("tie_stop")), None)
            if nxt is None:
                continue
            nxt_abs = nxt["bar"] * tpr + int(round(_f(nxt.get("tick"))))
            if abs(end_abs - nxt_abs) <= 1:                     # emenda (±1 tick de folga)
                ok_start.add((voice_i, h["bar"], int(round(_f(h.get("tick")))), lane_h))
                ok_stop.add((voice_i, nxt["bar"], int(round(_f(nxt.get("tick")))), lane_h))
    return ok_start, ok_stop


def _f(v, d: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else d
    except Exception:
        return d


def _emit(out: List[str], items: List[dict], voice: int, tpb: int, tpr: int,
          ties: Tuple[set, set] = (set(), set())) -> None:
    ok_start, ok_stop = ties
    plan = _beam_plan(items, tpb)

    def one(h: dict, lane: str, dx: int, ttype: str, dots: int, *, chord: bool = False,
            beam_i: Optional[int] = None, tie_ok: Tuple[bool, bool] = (False, False)) -> str:
        art = h.get("artic", "normal")
        pos = int(h.get("_pos", 0))
        bits: List[str] = ["      <note>"]
        if chord:
            bits.append("<chord/>")
        bits.append("<unpitched><display-step>C</display-step><display-octave>4</display-octave></unpitched>")
        bits.append(f"<duration>{dx}</duration>")
        if tie_ok[0]:
            bits.append('<tie type="start"/>')
        if tie_ok[1]:
            bits.append('<tie type="stop"/>')
        bits.append(f"<voice>{voice}</voice>")
        bits.append(f"<type>{ttype}</type>")
        bits += ["<dot/>" for _ in range(dots)]
        bits.append("<stem>up</stem>" if voice == 1 else "<stem>down</stem>")
        bits.append("<staff>1</staff>")
        if pos:
            bits.append(f"<staff-position>{pos}</staff-position>")
        nh = _noteheads(lane, h)
        if nh:
            bits.append(nh)
        notations: List[str] = []
        if art in ("accent", "ghost_accent"):
            notations.append("<artics><accent/></artics>")
        if art in ("droll", "roll", "diddle") or int(h.get("bars_n") or 1) > 1:
            notations.append("<artics><roll/></artics>")
        if art in ("ghost", "ghost_accent"):
            notations.append("<artics><staccato/></artics>")
        if lane == "hat_open":
            notations.append("<technical><other-technical>open</other-technical></technical>")
        if beam_i is not None:
            for k, tag in sorted(plan.get(beam_i) or []):
                notations.append(f'<beam number="{k}">{tag}</beam>')
        if tie_ok[0]:
            notations.append('<tied type="start"/>')
        if tie_ok[1]:
            notations.append('<tied type="stop"/>')
        if notations:
            bits.append("<notations>" + "".join(notations) + "</notations>")
        bits.append(f"<note-velocity>{max(1, int(h.get('velocity') or 88)) / 127.0:.3f}</note-velocity>")
        bits.append("</note>")
        return "".join(bits)

    for i, it in enumerate(items):
        dur = max(1, int(it["dur"]))
        ttype, dots = _dur_parts(dur)
        dx = dur
        if it["rest"]:
            if dur <= 0:
                continue
            out.append(f"      <note><rest/><duration>{dx}</duration><type>{ttype}</type>"
                       + "".join("<dot/>" for _ in range(dots))
                       + f"<voice>{voice}</voice><staff>1</staff></note>")
            continue
        h = it["hit"]
        lane = str(it.get("lane") or h.get("lane") or "snare")
        ln = LANE_BY_ID.get(lane) or LANE_BY_ID["snare"]
        key = (voice, int(it.get("_bar", 0)), int(it["tick"]), lane)
        tie_ok = (key in ok_start, key in ok_stop)
        h["_pos"] = int(it["step"]) - 5
        pre = ""
        if h.get("artic") == "flam":
            pre = ('      <note type="grace"><cue/><unpitched><display-step>C</display-step>'
                   "<display-octave>4</display-octave></unpitched>"
                   f"<staff-position>{h['_pos']}</staff-position><voice>{voice}</voice>"
                   "<stem>up</stem></note>\n")
        out.append(pre + one(h, lane, dx, ttype, dots, beam_i=i, tie_ok=tie_ok))
        for m in it.get("mates") or []:
            ml = str(m.get("lane") or "snare")
            m["_pos"] = int(_step_of(ml, m)) - 5
            mkey = (voice, int(it.get("_bar", 0)), int(it["tick"]), ml)
            out.append(one(m, ml, dx, ttype, dots, chord=True, tie_ok=(mkey in ok_start, mkey in ok_stop)))


def write_musicxml(score: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_musicxml(score))
    return path
