"""
Modelo de partitura (fonte única da verdade para SVG, PDF, MusicXML, MIDI e CSV).

Unidade de tempo: o TICK. 1 tick = semicolcheia (1/32 de pretinha)... na prática
adotamos `ticks_per_quarter = 32`? Não — adotamos 8 por semínima? Ver abaixo.

Optamos por TICKS_PER_QUARTER = 8  →  colcheia=4, semicolcheia=2, fusquinha(fffa)=1.
Isso dá resolução de 32ºs sem números gigantes. O MusicXML usa `divisions=8`.
Para meter compound (6/8), 1 beat = pontuada = 12 ticks.

Posição: `tick` dentro da barra (0 .. ticks_per_bar-1).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .kit import LANE_BY_ID

TICKS_PER_QUARTER = 8     # 8 → resolução de 32ºs (1 tick = fffa)


# --------------------------------------------------------------------------------------
# Métricas suportadas: (nome, beats_por_barra, simple|compound)
#   simple   → beat = semínima   (8 ticks)
#   compound → beat = colcheia pontuada (12 ticks)
# --------------------------------------------------------------------------------------
METERS: Dict[str, tuple] = {
    "2/4": (2, "simple"),
    "3/4": (3, "simple"),
    "4/4": (4, "simple"),
    "5/4": (5, "simple"),
    "6/8": (2, "compound"),
    "9/8": (3, "compound"),
    "12/8": (4, "compound"),
}


def ticks_per_beat(compound: bool) -> int:
    return 12 if compound else TICKS_PER_QUARTER


def meter_info(name: str) -> dict:
    beats, kind = METERS.get(name, (4, "simple"))
    tpb = ticks_per_beat(kind == "compound")
    return {"name": name, "beats": beats, "kind": kind,
            "ticks_per_beat": tpb, "ticks_per_bar": beats * tpb,
            "display": name if kind == "simple" else ("6/8" if beats == 2 else ("9/8" if beats == 3 else "12/8"))}


@dataclass
class Hit:
    lane: str
    tick: int                       # posição na barra
    dur: int = 2                    # duração em ticks
    velocity: int = 88              # 1..127
    artic: str = "normal"           # normal|accent|ghost|ghost_accent|flam|diddy|roll|open
    confidence: float = 0.5         # confiança da detecção 0..1
    head: str = "circle"            # override opcional de notahead
    tie_start: bool = False
    tie_stop: bool = False
    time: Optional[float] = 0.0     # segundos do golpe; None = sem medição (nota editada à mão)
    bars_n: int = 1                 # repetições (suspenso/tremolo)

    @property
    def voice(self) -> int:
        return LANE_BY_ID[self.lane].voice

    @property
    def stem(self) -> str:
        return LANE_BY_ID[self.lane].stem

    @property
    def staff(self) -> int:
        return LANE_BY_ID[self.lane].staff

    @property
    def notehead(self) -> str:
        if self.head and self.head != "circle":
            return self.head
        return "minus" if "ghost" in self.artic else LANE_BY_ID[self.lane].head

    def to_dict(self) -> dict:
        d = asdict(self)
        d["voice"] = self.voice
        d["stem"] = self.stem
        d["staff"] = self.staff
        d["notehead"] = self.notehead
        return d


@dataclass
class Bar:
    index: int                                   # 0-based
    hits: List[Hit] = field(default_factory=list)
    repeat_slash: bool = False                   # compasso de repetição (% / time-slash)
    fill_marker: bool = False                    # início de virada (sublinhado)
    cadence: bool = False
    crash_hit: bool = False                      # acento estrutural no início

    def sorted_hits(self) -> List[Hit]:
        return sorted(self.hits, key=lambda h: (h.tick, LANE_BY_ID[h.lane].staff))

    def to_dict(self) -> dict:
        return {"index": self.index, "repeat_slash": self.repeat_slash,
                "fill_marker": self.fill_marker,
                "hits": [h.to_dict() for h in self.sorted_hits()]}


@dataclass
class Score:
    title: str = "Transcrição de Bateria"
    subtitle: str = ""
    bpm: float = 120.0
    meter: str = "4/4"
    swing: float = 0.0                  # 0 = reto; 0.58..0.68 = swing (fração do beat)
    bars: List[Bar] = field(default_factory=list)
    lanes: List[str] = field(default_factory=list)
    #: pistas não gravadas (apresentação). Os hits continuam em `bars`, então auditoria,
    #: JSON e CSV vêm com tudo; só SVG/PDF/MIDI/MusicXML respeitam isto (docs/ERROS.md A16).
    hide_lanes: List[str] = field(default_factory=list)
    repeats: int = 1
    report: dict = field(default_factory=dict)

    @property
    def info(self) -> dict:
        return meter_info(self.meter)

    @property
    def ticks_per_bar(self) -> int:
        return self.info["ticks_per_bar"]

    @property
    def ticks_per_beat(self) -> int:
        return self.info["ticks_per_beat"]

    def to_dict(self) -> dict:
        return {
            "title": self.title, "subtitle": self.subtitle, "bpm": round(float(self.bpm), 2),
            "meter": self.meter, "swing": self.swing, "repeats": self.repeats,
            "ticks_per_quarter": TICKS_PER_QUARTER,
            "ticks_per_beat": self.ticks_per_beat, "ticks_per_bar": self.ticks_per_bar,
            "beats_per_bar": self.info["beats"], "compound": self.info["kind"] == "compound",
            "lanes": self.lanes, "hide_lanes": list(self.hide_lanes), "report": self.report,
            "n_bars": len(self.bars),
            "total_ticks": len(self.bars) * self.ticks_per_bar,
            "bars": [b.to_dict() for b in self.bars],
        }

    @staticmethod
    def from_dict(d: dict) -> "Score":
        s = Score(title=d.get("title", "Partitura"), subtitle=d.get("subtitle", ""),
                  bpm=float(d.get("bpm", 120.0)), meter=d.get("meter", "4/4"),
                  swing=float(d.get("swing", 0.0) or 0.0),
                  lanes=list(d.get("lanes", [])), repeats=int(d.get("repeats", 1) or 1), hide_lanes=[str(x) for x in (d.get("hide_lanes") or [])],
                  report=d.get("report", {}))
        s.bars = []
        for bd in d.get("bars", []):
            b = Bar(index=int(bd.get("index", 0)),
                    repeat_slash=bool(bd.get("repeat_slash", False)),
                    fill_marker=bool(bd.get("fill_marker", False)))
            for hd in bd.get("hits", []):
                b.hits.append(Hit(lane=hd.get("lane", "snare"), tick=int(hd.get("tick", 0)),
                                  dur=int(hd.get("dur", 2)), velocity=int(hd.get("velocity", 88)),
                                  artic=hd.get("artic", "normal"),
                                  confidence=float(hd.get("confidence", 0.5)),
                                  head=hd.get("head", "circle"),
                                  tie_start=bool(hd.get("tie_start", False)),
                                  tie_stop=bool(hd.get("tie_stop", False)),
                                  time=(float(hd["time"]) if hd.get("time") is not None else None),
                                  bars_n=int(hd.get("bars_n", 1) or 1)))
            s.bars.append(b)
        if not s.lanes:
            seen = []
            for b in s.bars:
                for h in b.hits:
                    if h.lane not in seen:
                        seen.append(h.lane)
            s.lanes = seen
        return s


def beat_of_tick(tick: int, ticks_per_beat: int) -> int:
    return tick // ticks_per_beat


def bar_time(index: int, tick: int, bpm: float, ticks_per_bar: int,
             ticks_per_beat: int) -> float:
    """Tempo em segundos de uma posição (barra, tick)."""
    beat_sec = 60.0 / max(1e-6, bpm)
    beats_per_bar = ticks_per_bar / ticks_per_beat
    return (index * beats_per_bar + tick / ticks_per_beat) * beat_sec


def score_visivel(score: dict) -> dict:
    """Cópia rasa com os hits das pistas ocultas removidos — para SVG/PDF/MIDI/MusicXML.

    De propósito não mexe em `score["bars"][i]["n_hits"]` nem nos dados: quem audita ou
    exporta JSON/CSV enxerga a partitura completa. Um `hide_lanes` que sumisse com notas dos
    dados seria a forma elegante de destruir informação lícita (regra do projeto).
    """
    ocl = set(str(x) for x in (score.get("hide_lanes") or []))
    if not ocl:
        return score
    out = dict(score)
    bars = []
    for b in score.get("bars") or []:
        hs = [h for h in (b.get("hits") or []) if str(h.get("lane")) not in ocl]
        nb = dict(b)
        nb["hits"] = hs
        bars.append(nb)
    out["bars"] = bars
    out["lanes"] = [l for l in (score.get("lanes") or []) if str(l) not in ocl]
    return out
