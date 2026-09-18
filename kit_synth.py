"""
Síntese física-minimalista das peças da bateria (síntese por modelos simples + ruído).

Não é um sampler: é um modelo reduzido suficiente para
  (a) gerar uma faixa de demonstração com ground-truth conhecido (testes A/B do
      transcritor, medição de recall/precisão), e
  (b) produzir o áudio de "preview" da partitura quando o usuário edita a grade.

Todas as vozes são mono, com ataque rápido (~0,5–6 ms) e decaimento exponencial,
de modo a reproduzirem as assinaturas tempo-frequência que o detector espera.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy import signal as sps


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def _env_exp(n: int, tau_samples: float, attack: int = 0) -> np.ndarray:
    e = np.exp(-np.arange(n, dtype=np.float32) / max(1.0, tau_samples)).astype(np.float32)
    if attack > 1:
        a = np.linspace(0.0, 1.0, attack, dtype=np.float32)
        e[:attack] *= a
    return e


def _bp(x: np.ndarray, lo: float, hi: float, sr: int, order: int = 4) -> np.ndarray:
    nyq = sr / 2.0
    lo = max(10.0, min(lo, nyq * 0.95))
    hi = max(lo + 20.0, min(hi, nyq * 0.98))
    b, a = sps.butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    return sps.filtfilt(b, a, x).astype(np.float32)


def _hp(x: np.ndarray, f: float, sr: int, order: int = 4) -> np.ndarray:
    nyq = sr / 2.0
    f = max(20.0, min(f, nyq * 0.98))
    b, a = sps.butter(order, f / nyq, btype="highpass")
    return sps.filtfilt(b, a, x).astype(np.float32)


def _osc(freq_from: float, freq_to: float, n: int, sr: int, expo: float = 3.0) -> np.ndarray:
    """Senoide com queda exponencial de pitch (modelo de membrana de caixa/bumbo)."""
    t = np.arange(n, dtype=np.float32) / sr
    dur = n / sr
    f = freq_to + (freq_from - freq_to) * np.exp(-expo * t / max(dur, 1e-6))
    ph = 2 * np.pi * np.cumsum(f) / sr
    return np.sin(ph).astype(np.float32)


# --------------------------------------------------------------------------------------
# vozes
# --------------------------------------------------------------------------------------

def voice_kick(sr: int, n: int, vel: float, rng: np.random.Generator,
               f0: float = 62.0) -> np.ndarray:
    body = _osc(f0 * 1.55, f0 * 0.62, n, sr, expo=6.0) * _env_exp(n, 0.16 * sr, 4)
    thump = _osc(f0, f0 * 0.85, n, sr, expo=10.0) * _env_exp(n, 0.30 * sr, 2)
    click = _bp(_noise(n, rng), 900, min(9000, sr / 2 * 0.95), sr) * _env_exp(n, 0.006 * sr)
    y = 0.85 * body + 1.05 * thump + 0.32 * click
    return (y * (0.35 + 0.65 * vel)).astype(np.float32)


def voice_snare(sr: int, n: int, vel: float, rng: np.random.Generator,
                f0: float = 190.0) -> np.ndarray:
    tone = _osc(f0 * 1.25, f0, n, sr, expo=8.0) * _env_exp(n, 0.055 * sr, 2)
    body = _osc(f0 * 0.62, f0 * 0.55, n, sr, expo=6.0) * _env_exp(n, 0.075 * sr, 2)
    buzz = _bp(_noise(n, rng), 1200, min(11000, sr / 2 * 0.95), sr) * _env_exp(n, 0.10 * sr, 3)
    snap = _hp(_noise(n, rng), 5000, sr) * _env_exp(n, 0.025 * sr, 1)
    y = 0.55 * tone + 0.42 * body + 0.95 * buzz + 0.35 * snap
    return (y * (0.30 + 0.70 * vel)).astype(np.float32)


def voice_hat(sr: int, n: int, vel: float, rng: np.random.Generator,
              open_: bool = False) -> np.ndarray:
    tau = (0.42 if open_ else 0.045) * sr
    metal = np.zeros(n, dtype=np.float32)
    for f in (3100.0, 4350.0, 5900.0, 7450.0, 9100.0):
        if f < sr / 2 * 0.95:
            metal += _osc(f, f * 0.995, n, sr, expo=1.0) * _env_exp(n, tau * (0.55 if open_ else 0.35))
    hiss = _hp(_noise(n, rng), 6200, sr) * _env_exp(n, tau, 1)
    y = 0.30 * metal / 5.0 + 1.0 * hiss
    y = y * (1.0 - 0.55 * (1.0 - vel)) if not open_ else y * (0.4 + 0.6 * vel)
    return y.astype(np.float32)


def voice_ride(sr: int, n: int, vel: float, rng: np.random.Generator) -> np.ndarray:
    ping = _osc(3650.0, 3600.0, n, sr, expo=1.0) * _env_exp(n, 0.30 * sr, 1)
    wash = _hp(_noise(n, rng), 4200, sr) * _env_exp(n, 0.62 * sr, 4)
    bell = _osc(5200.0, 5150.0, n, sr, expo=1.0) * _env_exp(n, 0.15 * sr, 1)
    y = 0.45 * ping + 0.85 * wash + 0.22 * bell
    return (y * (0.35 + 0.65 * vel)).astype(np.float32)


def voice_crash(sr: int, n: int, vel: float, rng: np.random.Generator) -> np.ndarray:
    body = _hp(_noise(n, rng), 1600, sr) * _env_exp(n, 1.15 * sr, int(0.004 * sr))
    shimmer = _bp(_noise(n, rng), 4500, min(13000, sr / 2 * 0.95), sr) * _env_exp(n, 0.8 * sr, 6)
    y = 0.9 * body + 0.6 * shimmer
    return (y * (0.45 + 0.55 * vel)).astype(np.float32)


def voice_tom(sr: int, n: int, vel: float, rng: np.random.Generator,
              f0: float = 140.0) -> np.ndarray:
    tone = _osc(f0 * 1.35, f0 * 0.88, n, sr, expo=5.0) * _env_exp(n, 0.24 * sr, 2)
    attack = _bp(_noise(n, rng), 700, min(6000, sr / 2 * 0.95), sr) * _env_exp(n, 0.012 * sr)
    y = 1.0 * tone + 0.30 * attack
    return (y * (0.35 + 0.65 * vel)).astype(np.float32)


def voice_rim(sr: int, n: int, vel: float, rng: np.random.Generator) -> np.ndarray:
    click = _bp(_noise(n, rng), 1200, 5200, sr) * _env_exp(n, 0.006 * sr)
    wood = _osc(950.0, 880.0, n, sr, expo=14.0) * _env_exp(n, 0.018 * sr)
    y = 0.9 * click + 0.7 * wood
    return (y * (0.45 + 0.55 * vel)).astype(np.float32)


def voice_cowbell(sr: int, n: int, vel: float, rng: np.random.Generator) -> np.ndarray:
    a = _osc(540.0, 538.0, n, sr, expo=1.0) * _env_exp(n, 0.22 * sr, 1)
    b = _osc(800.0, 798.0, n, sr, expo=1.0) * _env_exp(n, 0.18 * sr, 1)
    return ((a + b) * (0.4 + 0.6 * vel)).astype(np.float32)


def _v_hat_open(sr, n, vel, rng):
    return voice_hat(sr, n, vel, rng, open_=True)


def _v_tom(f0):
    def _f(sr, n, vel, rng):
        return voice_tom(sr, n, vel, rng, f0=f0)
    return _f


VOICE_FUNCS = {
    "kick": voice_kick, "snare": voice_snare, "rim": voice_rim, "hat": voice_hat,
    "hat_open": _v_hat_open, "hat_foot": voice_hat,
    "ride": voice_ride, "crash": voice_crash, "splash": voice_crash,
    "tom_hi": _v_tom(175.0), "tom_mid": _v_tom(135.0), "tom_low": _v_tom(95.0),
    "cowbell": voice_cowbell,
}
HIT_LEN = {"kick": 0.42, "snare": 0.26, "rim": 0.09, "hat": 0.11, "hat_open": 0.55,
           "hat_foot": 0.10, "ride": 0.9, "crash": 1.6, "splash": 0.7,
           "tom_hi": 0.4, "tom_mid": 0.45, "tom_low": 0.55, "cowbell": 0.4}


def render_hit(lane_id: str, sr: int, vel: float, rng: np.random.Generator,
               tail: Optional[float] = None) -> np.ndarray:
    fn = VOICE_FUNCS.get(lane_id, voice_snare)
    n = int(sr * (tail or HIT_LEN.get(lane_id, 0.3)))
    n = max(64, n)
    try:
        y = fn(sr, n, float(np.clip(vel, 0.05, 1.0)), rng)
    except TypeError:
        y = fn(sr, n, float(np.clip(vel, 0.05, 1.0)), rng)
    return np.asarray(y, dtype=np.float32)


def render_score(score: dict, sr: int = 44100, gain: float = 0.85,
                 seed: int = 7) -> np.ndarray:
    """Renderiza um Score-dict (see score_model.to_dict) como áudio mono."""
    rng = np.random.default_rng(seed)
    bpm = float(score.get("bpm", 120.0)) or 120.0
    spb = 60.0 / bpm
    tpq = int(score.get("ticks_per_quarter", 8))
    ticks_per_bar = int(score.get("ticks_per_bar", 32))
    total_ticks = int(score.get("total_ticks", 0) or 0)
    if not total_ticks and score.get("bars"):
        total_ticks = (max(int(b.get("index", 0)) for b in score["bars"]) + 1) * ticks_per_bar
    dur_sec = (total_ticks / tpq) * spb + 2.5
    out = np.zeros(int(dur_sec * sr) + sr, dtype=np.float32)

    for bar in score.get("bars", []):
        bar_idx = int(bar.get("index", 0))
        for h in bar.get("hits", []):
            lane_id = h.get("lane", "snare")
            tick = bar_idx * int(score.get("ticks_per_bar", 32)) + int(h.get("tick", 0))
            t0 = (tick / tpq) * spb
            i0 = int(t0 * sr)
            if i0 < 0 or i0 >= out.size:
                continue
            vel = float(h.get("velocity", 90)) / 127.0
            y = render_hit(lane_id, sr, vel, rng)
            if h.get("artic") in ("ghost", "ghost_accent"):
                y = y * 0.45
            seg = out[i0: i0 + y.size]
            k = min(seg.size, y.size)
            if k <= 0:
                continue
            peak = float(np.max(np.abs(y[:k]))) or 1e-6
            seg[:k] += (y[:k] / peak) * (0.10 + 0.9 * vel) * gain

    peak = float(np.max(np.abs(out))) or 1e-6
    return (out / peak * 0.9).astype(np.float32)
