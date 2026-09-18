"""
Definição do "kit" de bateria: pistas (lanes), posição na pauta, notahead, haste,
nota GM (General MIDI) e banda espectral usada pelo detector.

Referência de convenções de notação:
  * E. Gould, "Behind Bars" (5-line percussion staff, stems up = cimbais/mão direita,
    stems down = caixa/bumbo/pé esquerdo).
  * G. M. drum map (GM Level 1, canal 10) para exportação MIDI.

Posição na pauta = índice de "step" diatônico em uma clave de sol, onde
  0 = D4 (espaço abaixo da 1ª linha), 1 = E4 (1ª linha), 2 = F4, ... 9 = F5 (5ª linha),
  10 = G5 (espaço acima), 11 = A5 (1ª linha suplementar superior).
Cada passo de 1 unidade = meio espaço de pauta.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List


@dataclass(frozen=True)
class Lane:
    id: str
    name: str            # nome em português (partitura)
    short: str           # rótulo curto usado na legenda
    staff: int           # posição na pauta (step diatônico)
    head: str            # 'circle' | 'x' | 'minus' | 'triangle' | 'diamond'
    stem: str            # 'up' | 'down'
    voice: int           # 1 = cima (cimbais), 2 = baixo (bumbo/caixa/toms)
    gm: int              # nota GM canal 10
    band: str            # chave da banda espectral dominante
    tie: bool = False    # sustenta (pratos abertos)
    ghostable: bool = True
    default: bool = True # entra no mapa de detecção padrão
    group: str = "core"


LANES: List[Lane] = [
    Lane("kick",      "Bumbo",              "Bx",   0, "circle",  "down", 2, 36, "low"),
    Lane("snare",     "Caixa",              "Cx",   6, "circle",  "down", 2, 38, "mid"),
    Lane("rim",       "Cross-stick",        "CS",   6, "x",       "down", 2, 37, "midhi", False),
    Lane("hat",       "Chimel (closed)",    "hh",  10, "x",       "up",   1, 42, "high"),
    Lane("hat_open",  "Chimel (open)",      "hhO", 10, "diamond", "up",   1, 46, "high", tie=True),
    Lane("hat_foot",  "Pedal de chimel",    "hh*", -2, "x",       "down", 2, 44, "high", ghostable=False),
    Lane("ride",      "Ride",               "rd",  11, "x",       "up",   1, 51, "high", tie=True),
    Lane("crash",     "Prato (crash)",      "cr",  12, "x",       "up",   1, 49, "high", ghostable=False),
    Lane("splash",    "Splash",             "sp",  12, "x",       "up",   1, 55, "high", False, False, "aux"),
    Lane("tom_hi",    "Tom 1",              "T1",   5, "circle",  "down", 2, 50, "mid"),
    Lane("tom_mid",   "Tom 2",              "T2",   4, "circle",  "down", 2, 48, "mid"),
    Lane("tom_low",   "Tom 3 / base",       "T3",   3, "circle",  "down", 2, 45, "midlow"),
    Lane("cowbell",   "Cowbell",            "cb",  10, "triangle","up",   1, 56, "high", False, False, "aux"),
]

LANE_BY_ID: Dict[str, Lane] = {l.id: l for l in LANES}

#: Famílias de prato (com o pedal do chimel): na **partitura simples** estas somem da gravura,
#: mas continuam nos dados e na auditoria — ocultar é escolha de apresentação, nunca de verdade.
#: `rim` fica de fora de propósito: cross-stick é uma articulação da caixa, não um prato.
PRATOS: List[str] = ["hat", "hat_open", "hat_foot", "ride", "crash", "splash", "cowbell"]
DEFAULT_LANES: List[str] = [l.id for l in LANES if l.default]

# Bandas espectrais (Hz) usadas pelo analisador multi-banda.
BANDS: Dict[str, tuple] = {
    "sub":     (20.0, 60.0),      # energia subsônica do bumbo
    "low":     (34.0, 120.0),     # corpo do bumbo
    "midlow":  (120.0, 260.0),    # tons graves / corpo da caixa
    "mid":     (180.0, 420.0),    # aro + tonal da caixa, toms
    "midhi":   (600.0, 2200.0),   # ataque da caixa, presence, cross-stick
    "high":    (3000.0, 7000.0),  # prato/chimel (banda metálica)
    "vhigh":   (7000.0, 16000.0), # chiado do chimel, crash
}

# Pesos de cada banda por pista — usados pelo classificador (score linear).
LANE_BAND_WEIGHTS: Dict[str, Dict[str, float]] = {
    "kick":     {"sub": 3.2, "low": 3.6, "midlow": 1.2, "mid": 0.2, "midhi": -0.9, "high": -2.2, "vhigh": -2.6},
    "snare":    {"sub": -0.6, "low": 0.7, "midlow": 2.0, "mid": 2.4, "midhi": 2.2, "high": 1.2, "vhigh": 0.5},
    "rim":      {"sub": -1.4, "low": -1.0, "midlow": -0.4, "mid": 0.4, "midhi": 2.3, "high": 0.9, "vhigh": -0.8},
    "hat":      {"sub": -2.0, "low": -1.6, "midlow": -0.8, "mid": -0.4, "midhi": 0.5, "high": 1.9, "vhigh": 3.0},
    "hat_open": {"sub": -1.8, "low": -1.4, "midlow": -0.6, "mid": 0.0, "midhi": 0.7, "high": 2.0, "vhigh": 2.6},
    "hat_foot": {"sub": -0.4, "low": -0.2, "midlow": -0.2, "mid": 0.2, "midhi": 0.6, "high": 1.2, "vhigh": 1.0},
    "ride":     {"sub": -1.4, "low": -1.0, "midlow": -0.4, "mid": 0.2, "midhi": 1.2, "high": 2.4, "vhigh": 1.2},
    "crash":    {"sub": -0.8, "low": -0.4, "midlow": 0.2, "mid": 0.8, "midhi": 1.6, "high": 2.0, "vhigh": 2.0},
    "splash":   {"sub": -1.4, "low": -1.0, "midlow": -0.4, "mid": 0.4, "midhi": 1.4, "high": 2.0, "vhigh": 1.8},
    "tom_hi":   {"sub": -0.6, "low": 0.6, "midlow": 2.4, "mid": 2.2, "midhi": -0.2, "high": -1.6, "vhigh": -2.2},
    "tom_mid":  {"sub": -0.4, "low": 1.0, "midlow": 2.4, "mid": 1.4, "midhi": -0.6, "high": -1.8, "vhigh": -2.4},
    "tom_low":  {"sub": 0.6, "low": 2.2, "midlow": 2.4, "mid": 0.8, "midhi": -0.9, "high": -2.0, "vhigh": -2.4},
    "cowbell":  {"sub": -1.6, "low": -1.2, "midlow": -0.4, "mid": 0.2, "midhi": 1.8, "high": 1.4, "vhigh": 0.2},
}

# Duração relativa esperada do decaimento (ms) — segundo critério do classificador.
LANE_DECAY = {
    "kick": (60, 320), "snare": (40, 190), "rim": (8, 90),
    "hat": (4, 95), "hat_open": (120, 2500), "hat_foot": (8, 120),
    "ride": (150, 4000), "crash": (250, 6000), "splash": (120, 2500),
    "tom_hi": (80, 600), "tom_mid": (100, 700), "tom_low": (120, 800),
    "cowbell": (60, 800),
}

# Faixa de frequência fundamental esperada (Hz) para pistas tonais.
LANE_PITCH = {
    "kick": (38, 95), "tom_hi": (120, 210), "tom_mid": (95, 165),
    "tom_low": (60, 130), "snare": (150, 260),
}


def lane(id_: str) -> Lane:
    return LANE_BY_ID[id_]


def to_dict() -> dict:
    return {l.id: asdict(l) for l in LANES}
