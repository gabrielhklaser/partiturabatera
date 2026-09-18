"""
Regras de escrita rítmica — a "engravadora lógica" compartilhada por SVG, PDF, MusicXML
e MIDI. Fonte única: dado um compasso e uma voz, produz a sequência exata de notas,
pausas, pontos e ligações que um baterista espera ler.

Regras aplicadas (Gould, "Behind Bars", cap. 5–6; Read, "Music Notation", cap. 7):
  * Duração = maior valor padrão que cabe no espaço até a próxima nota da mesma voz.
  * Nota/pausa não cruza a divisao do tempo (beat) — exceto quando começa exatamente
    no beat (aí pode valer o beat inteiro) ou quando é sustain de prato (usa-se ligadura).
  * Pausas curtas dentro do tempo são pontilhadas/represadas de forma canônica.
  * Em compasso composto (6/8, 12/8) a divisao de referência é a colcheia pontuada.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .kit import LANE_BY_ID
from .score_model import Bar, Hit, TICKS_PER_QUARTER, meter_info

# valores padrão de duração em ticks (tpq = 8 por semínima)
DURS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
REST_DURS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def dur_label(ticks: int, ticks_per_beat: int) -> Tuple[str, int]:
    """(nome, pontos) para um valor de duração em ticks."""
    per_beat = ticks_per_beat
    base_map = {1: "32nd", 2: "16th", 4: "eighth", 8: "quarter", 16: "half", 32: "whole", 64: "breve"}
    if per_beat == 12:  # composto: base = colcheia (4)
        base_map = {1: "32nd", 2: "16th", 4: "eighth", 6: "dotted-eighth", 8: "quarter",
                    12: "dotted-quarter", 16: "half", 24: "dotted-half", 32: "whole", 48: "dotted-half?"}
    if ticks in base_map:
        return base_map[ticks], 0
    for base in sorted(base_map):
        if base * 2 - base == base:
            pass
        if ticks == int(base * 1.5):
            return base_map[base].replace("dotted-", "") + "-dotted", 1
        if ticks == int(base * 1.75):
            return base_map[base].replace("dotted-", "") + "-double-dotted", 2
    return base_map.get(min(base_map, key=lambda b: abs(b - ticks)), "16th"), 0


def _allowed(ticks_per_beat: int) -> List[int]:
    """Durações candidatas, respeitando a divisao do beat (simple/composto)."""
    if ticks_per_beat == 12:
        return [1, 2, 3, 4, 6, 12, 8, 16, 24, 32, 48]
    return [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def fill_voice(hits: Sequence[Hit], ticks_per_bar: int, ticks_per_beat: int,
               sustain_lanes: Sequence[str] = ("crash", "ride", "hat_open", "splash"),
               allow_across_beat: bool = True) -> List[dict]:
    """
    Converte hits de uma voz em itens de escrita:
       {'kind':'note'|'rest', 'tick':t, 'dur':dticks, 'hits':[Hit...], 'dots':n,
        'tie_start':bool, 'tie_stop':bool, 'label':str}
    """
    if hits and isinstance(hits[0], dict):      # aceita o dicionário do Score.to_dict()
        from .score_model import Hit
        keys = set(Hit.__dataclass_fields__)
        hits = [Hit(**{k: v for k, v in h.items() if k in keys}) for h in hits]
    if not hits:
        return [{"kind": "rest", "tick": 0, "dur": ticks_per_bar, "hits": [],
                 "dots": 0, "tie_start": False, "tie_stop": False, "label": "whole-rest"}]
    allowed = [d for d in _allowed(ticks_per_beat) if d <= ticks_per_bar]
    boundary = ticks_per_beat if not allow_across_beat else 0
    pos = sorted({int(h.tick) for h in hits})
    by_tick: Dict[int, List[Hit]] = {t: [h for h in hits if int(h.tick) == t] for t in pos}

    items: List[dict] = []
    t = 0
    next_idx = 0
    while t < ticks_per_bar:
        if t in by_tick:
            hs = by_tick[t]
            nxt = pos[next_idx + 1] if next_idx + 1 < len(pos) else ticks_per_bar
            span = max(1, nxt - t)
            # sustain (prato aberto / crash): estende até o próximo evento, com ligação
            sustain = all(h.lane in sustain_lanes for h in hs) and len(hs) == 1
            cand = [d for d in allowed if d <= span] or [min(allowed, key=lambda d: abs(d - span))]
            if not sustain:
                if boundary:
                    cand = [d for d in cand if (t % boundary) + d <= boundary] or [1]
                else:
                    # pode cruzar o beat apenas se começar no beat e caber no resto da barra
                    cand = [d for d in cand if (t % boundary if boundary else 0) == 0 or t + d <= ticks_per_bar]
            dur = max(cand, key=lambda d: (d, -abs(d - span)))
            # não deixar 1 tick sobrando antes do próximo evento
            while dur > 1 and t + dur > nxt:
                dur = min(d for d in allowed if d < dur) if any(d < dur for d in allowed) else 1
            dots = 1 if dur in (3, 6, 12, 24, 48) else 0
            tie = bool(sustain and dur < span and nxt <= ticks_per_bar)
            items.append({"kind": "note", "tick": t, "dur": dur, "hits": hs, "dots": dots,
                          "tie_start": tie, "tie_stop": any(h.tie_stop for h in hs),
                          "label": dur_label(dur, ticks_per_beat)[0]})
            t += dur
            while next_idx < len(pos) and pos[next_idx] <= t:
                next_idx += 1
            continue
        nxt_note = pos[next_idx] if next_idx < len(pos) else ticks_per_bar
        span = max(1, nxt_note - t)
        cand = [d for d in REST_DURS if d <= span and d <= ticks_per_bar - t]
        if not cand:
            cand = [1]
        if boundary:
            ok = [d for d in cand if (t % boundary) + d <= boundary]
            cand = ok or [1]
        dur = max(cand)
        items.append({"kind": "rest", "tick": t, "dur": dur, "hits": [],
                      "dots": 1 if dur in (3, 6, 12, 24) else 0,
                      "tie_start": False, "tie_stop": False,
                      "label": dur_label(dur, ticks_per_beat)[0] + "-rest"})
        t += dur
        while next_idx < len(pos) and pos[next_idx] <= t:
            next_idx += 1
    return items


def bar_items(bar: Bar, ticks_per_bar: int, ticks_per_beat: int,
              voices: Sequence[int] = (1, 2)) -> Dict[int, List[dict]]:
    hs = [h for h in bar.sorted_hits()]
    out = {}
    for v in voices:
        out[v] = fill_voice([h for h in hs if LANE_BY_ID[h.lane].voice == v],
                            ticks_per_bar, ticks_per_beat)
    return out


def plan_beams(items: Sequence[dict], ticks_per_beat: int,
               ticks_per_quarter: int = TICKS_PER_QUARTER) -> List[dict]:
    """
    Agrupa notas em feixes (beams) por beat. Retorna, para cada item de nota, o nível de
    viga (1 = colcheia, 2 = semicolcheia, 3 = fusquinha) e a posição no grupo.
    """
    notes = [it for it in items if it["kind"] == "note"]
    plan = []
    cur: List[dict] = []

    def flush(force=False):
        nonlocal cur
        if not cur:
            return
        n = len(cur)
        for i, it in enumerate(cur):
            it["beam"] = {"group": id(cur) % 10**9, "index": i, "count": n,
                          "level": int(it.get("beam_level", 1))}
        cur = []

    for it in items:
        if it["kind"] != "note":
            flush()
            continue
        d = it["dur"]
        lvl = 1 if d <= ticks_per_beat / 2 else 0
        # nível de viga: nº de vigas = piso log2(beat/dur) limitado
        if d <= ticks_per_quarter / 2:      # ≤ colcheia
            levels = 1
            if d <= ticks_per_quarter / 4:
                levels = 2
            if d <= 1:
                levels = 3
        else:
            levels = 0
        it["beam_level"] = levels
        beat_of = it["tick"] // ticks_per_beat
        if cur and (cur[-1].get("beat") != beat_of or it["beam_level"] == 0):
            flush()
        if it["beam_level"] == 0:
            flush()
            it["beam"] = {"group": -1, "index": 0, "count": 1, "level": 0}
            continue
        it["beat"] = beat_of
        cur.append(it)
    flush()
    return [it for it in items if it["kind"] == "note"]


def is_repeat_of(prev: Bar, cur: Bar, ticks_per_beat: int, tol_vel: int = 22) -> bool:
    """Compasso idêntico ao anterior (posições, vozes e dinâmica aproximada)."""
    a = _sig(prev, tol_vel)
    b = _sig(cur, tol_vel)
    return a is not None and a == b


def _sig(bar: Bar, tol_vel: int = 22):
    """Assinatura rítmica do compasso (posições + peças + dinâmica arredondada)."""
    key = []
    for h in bar.sorted_hits():
        key.append((int(h.tick), h.lane, int(h.velocity) // max(1, int(tol_vel)), h.artic,
                    int(h.dur)))
    return tuple(sorted(key)) if key else None


def detect_repeats(bars: List[Bar], min_repeats: int = 2, tol_vel: int = 22) -> List[bool]:
    """Marca compassos repetidos a partir da segunda ocorrência consecutiva."""
    flags = [False] * len(bars)
    i = 1
    run = 0
    while i < len(bars):
        if _sig(bars[i]) and _sig(bars[i]) == _sig(bars[i - 1], tol_vel):
            run += 1
            if run >= min_repeats - 1:
                flags[i] = True
        else:
            run = 0
        i += 1
    return flags
