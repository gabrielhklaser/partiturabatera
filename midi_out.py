"""
Exportação MIDI (Standard MIDI File tipo 1) — canal 10 (índice 9) com as notas GM de
percussaria, uma trilha por voz é dispensável: o mapa de pistas já é monofônico por peça,
entao usamos uma única trilha de notas + uma trilha de mapa (tempo/armadura/marcadores).

  * PPQ = 480 → 1 tick nosso (1/8 da semínima) = 60 PPQ, sem arredondamento de acento;
  * duração = duração escrita (ticks) com piso de 30 PPQ, e pratos sustentados ganham o
    dobro (soam até o próximo golpe, como na partitura);
  * notas "offgrid" (não quantizadas) mantêm o tempo detectado, não o da grade.
"""
from __future__ import annotations

import struct
from typing import Dict, List, Sequence

from .kit import LANE_BY_ID
from .score_model import meter_info, score_visivel

PPQ = 480
TICK_PPU = PPQ // 8          # nossos ticks são semicolcheias da semínima (8 por semínima)


def _vlq(n: int) -> bytes:
    n = max(0, int(n))
    out = bytes([n & 0x7F])
    n >>= 7
    while n:
        out = bytes([(n & 0x7F) | 0x80]) + out
        n >>= 7
    return out


def _ev(tick: int, data: bytes) -> tuple:
    return (int(tick), data)


def _track(evs: Sequence[tuple]) -> bytes:
    evs = sorted(evs, key=lambda e: e[0])
    body = b""
    last = 0
    for t, d in list(evs) + [(None, b"\x00\xff\x2f\x00")]:      # end of track
        cur = 0 if t is None else int(t)
        body += _vlq(max(0, cur - last)) + d
        last = cur
    return b"MTrk" + struct.pack(">I", len(body)) + body


def to_midi(score: dict) -> bytes:
    score = score_visivel(score)          # o .mid que se ouve é o que se vê
    info = meter_info(score.get("meter", "4/4"))
    beats, kind = info["beats"], info["kind"]
    bpm = float(score.get("bpm") or 120.0)
    us = int(round(60000000.0 / max(1.0, bpm)))
    # ---- trilha 0: mapa
    map_ev: List[tuple] = [_ev(0, b"\xff\x51\x03" + struct.pack(">I", us)[1:]),
                          _ev(0, b"\xff\x58\x04" + bytes([beats, 4 if kind == "simple" else 8,
                                                          24, 8])),
                          _ev(0, b"\xff\x2d\x01\x00")]
    title = str(score.get("title") or "DrumScribe").encode("ascii", "replace")[:120]
    map_ev.append(_ev(0, b"\xff\x03" + _vlq(len(title)) + title))
    tpq = score.get("ticks_per_bar") or info["ticks_per_bar"]
    n_bars = len(score.get("bars") or [])
    for i in range(n_bars):
        mark = f"comp. {i + 1}".encode("ascii", "replace")
        map_ev.append(_ev(int(i * tpq * TICK_PPU), b"\xff\x06" + _vlq(len(mark)) + mark))

    # ---- trilha 1: notas (canal 10)
    notes: List[tuple] = []
    for bar in score.get("bars") or []:
        bidx = int(bar.get("index", 0))
        for h in bar.get("hits") or []:
            ln = LANE_BY_ID.get(h.get("lane", "snare")) or LANE_BY_ID["snare"]
            start = int((bidx * tpq + int(h.get("tick", 0))) * TICK_PPU)
            dur = max(1, int(h.get("dur") or 2))
            dl = dur * TICK_PPU
            if ln.tie:
                dl = int(dl * 2.0)
            dl = max(30, dl)
            vel = min(127, max(1, int(h.get("velocity") or 88)))
            note = int(h.get("gm", ln.gm)) if isinstance(h.get("gm"), int) else int(ln.gm)
            notes.append((start, note, vel, dl))
    notes.sort()
    track_ev: List[tuple] = []
    active: Dict[int, tuple] = {}
    for start, note, vel, dl in notes:
        if note in active:
            s0, v0, e0 = active.pop(note)
            track_ev.append(_ev(max(s0, e0 - 1), bytes([0x80 | 9, note, 0])))
        track_ev.append(_ev(start, bytes([0x90 | 9, note, vel])))
        active[note] = (start, vel, start + dl)
    for note, (s0, v0, e0) in sorted(active.items()):
        track_ev.append(_ev(e0, bytes([0x80 | 9, note, 0])))
    end = max([e[0] for e in track_ev] + [tpq * TICK_PPU])
    track_ev.append(_ev(end + TICK_PPU, b"\xff\x2f\x00"))

    header = b"MThd" + struct.pack(">I", 6) + struct.pack(">HHH", 1, 2, PPQ)
    return header + _track(map_ev) + _track(track_ev)


def write_midi(score: dict, path: str) -> str:
    with open(path, "wb") as f:
        f.write(to_midi(score))
    return path
