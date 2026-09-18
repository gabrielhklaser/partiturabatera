"""
Gera uma faixa de demonstração (bateria isolada) com ground-truth exato.

Serve para (a) o usuário testar a plataforma sem ter arquivo à mão e (b) o teste
automatizado medir recall/precisão da transcrição em condições conhecidas.

Padrões (4 compassos por seção, 100 BPM, 4/4):
  A) rock básico          — chimel em 8º, caixa no backbeat, bumbo em 1/3 + "e de 3"
  B) funk leve            — 16º no chimel, caixas fantasma, bumbo sincopado
  C) meio-tempo com ride  — ride em 8º, hi-hat pedal nos tempos 2 e 4
  D) virada + prato       — 2 compassos de toms e crash no 1º tempo do último
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drumscribe import kit_synth
from drumscribe.score_model import TICKS_PER_QUARTER

TPB = TICKS_PER_QUARTER          # ticks por semínima (=8 → 16º = 2 ticks)
BEAT = TPB
BAR = 4 * BEAT                   # 32 ticks

HAT8 = [0, 4, 8, 12, 16, 20, 24, 28]
HAT16 = list(range(0, BAR, 2))


def _section_a(b):
    """rock básico"""
    ev = [("crash", 0, 118), ("kick", 0, 118), ("kick", 16, 108), ("kick", 22, 92),
          ("snare", 8, 116), ("snare", 24, 120), ("hat", 0, 92), ("hat", 4, 60),
          ("hat", 8, 88), ("hat", 12, 58), ("hat", 16, 90), ("hat", 20, 58),
          ("hat", 24, 92), ("hat", 28, 56)]
    return ev


def _section_b(b):
    """funk leve: 16ºs + ghosts"""
    ev = [("kick", 0, 112), ("kick", 6, 84), ("kick", 16, 104), ("kick", 20, 76),
          ("snare", 8, 114), ("snare", 14, 38), ("snare", 24, 118), ("snare", 27, 34),
          ("hat", 0, 90), ("hat", 2, 52), ("hat", 4, 62), ("hat", 6, 50),
          ("hat", 8, 86), ("hat", 10, 50), ("hat", 12, 60), ("hat", 14, 48),
          ("hat", 16, 88), ("hat", 18, 50), ("hat", 20, 62), ("hat", 22, 48),
          ("hat", 24, 90), ("hat", 26, 50), ("hat", 28, 60), ("hat_open", 30, 84)]
    return ev


def _section_c(b):
    """ride + pedal"""
    ev = [("ride", 0, 100), ("ride", 4, 74), ("ride", 8, 92), ("ride", 12, 72),
          ("ride", 16, 100), ("ride", 20, 74), ("ride", 24, 92),
          ("kick", 0, 110), ("kick", 18, 90), ("snare", 8, 112), ("snare", 24, 116),
          ("hat_foot", 8, 72), ("hat_foot", 24, 72)]
    return ev


def _section_d(b, last=False):
    if not last:
        ev = [("kick", 0, 108), ("snare", 8, 112), ("snare", 24, 114)]
        ev += [("hat", t, 80) for t in HAT8]
        ev += [("tom_hi", 30, 100)]
        return ev
    # virada de 1 compasso + crash
    ev = [("tom_hi", 0, 112), ("tom_hi", 2, 96), ("tom_mid", 4, 114), ("tom_mid", 6, 98),
          ("tom_low", 8, 116), ("tom_low", 10, 100), ("snare", 12, 96), ("snare", 14, 92),
          ("kick", 16, 122), ("snare", 18, 100), ("snare", 20, 104), ("snare", 22, 108),
          ("crash", 24, 124), ("hat_open", 24, 100)]
    return ev


def build_pattern() -> dict:
    bpm = 100.0
    hits = []
    layout = [_section_a, _section_a, _section_a, _section_a,
              _section_b, _section_b, _section_a, _section_a,
              _section_c, _section_c, _section_c, _section_c,
              _section_a, _section_a, _section_d, _section_d]
    for i, fn in enumerate(layout):
        for (lane, tick, vel) in fn(i):
            hits.append({"bar": i, "tick": int(tick), "lane": lane, "velocity": int(vel)})
    return {"bpm": bpm, "meter": "4/4", "ticks_per_quarter": TPB,
            "ticks_per_bar": BAR, "hits": hits, "bars": len(layout)}


def pattern_to_score_dict(pat: dict) -> dict:
    bars = {}
    for h in pat["hits"]:
        bars.setdefault(h["bar"], []).append({"lane": h["lane"], "tick": h["tick"],
                                               "velocity": h["velocity"], "dur": 2,
                                               "artic": "normal"})
    return {"title": "DEMO — Groove de referência", "bpm": pat["bpm"], "meter": pat["meter"],
            "ticks_per_quarter": pat["ticks_per_quarter"], "ticks_per_bar": pat["ticks_per_bar"],
            "bars": [{"index": k, "hits": sorted(v, key=lambda d: d["tick"])}
                     for k, v in sorted(bars.items())]}


def render(out_dir: str, sr: int = 44100, room: bool = True, mix_level: float = 0.8):
    pat = build_pattern()
    sc = pattern_to_score_dict(pat)
    x = kit_synth.render_score(sc, sr=sr, gain=0.95, seed=11)
    if room:
        from scipy import signal as sps
        # leve "sala": pré-echo + shelf, mantém os transientes nítidos
        nyq = sr / 2
        b, a = sps.butter(2, 7000 / nyq, btype="lowpass")
        x = sps.filtfilt(b, a, x).astype(np.float32)
        d = int(0.011 * sr)          # pré-echo curto, típico de microfone próximo
        y = x.copy()
        y[d:] += 0.07 * x[:-d]
        x = y
    x = x * (mix_level / (np.max(np.abs(x)) + 1e-9))
    n = int(x.size / sr * pat["bpm"] / 60)
    os.makedirs(out_dir, exist_ok=True)
    wav = os.path.join(out_dir, "demo_drums.wav")
    from drumscribe.audio_io import write_wav
    write_wav(wav, x, sr)
    mp3 = os.path.join(out_dir, "demo_drums.mp3")
    try:
        import imageio_ffmpeg, subprocess
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", wav, "-codec:a", "libmp3lame",
                        "-b:a", "192k", mp3], check=True, capture_output=True)
    except Exception as e:  # pragma: no cover
        mp3 = None
        print("mp3 não gerado:", e)
    gt = {"bpm": pat["bpm"], "meter": "4/4", "sr": sr, "sec_per_beat": 60 / pat["bpm"],
          "ticks_per_quarter": TPB, "ticks_per_bar": BAR,
          "events": [{"time": (h["bar"] * BAR + h["tick"]) / TPB * (60 / pat["bpm"]),
                      "lane": h["lane"], "velocity": h["velocity"],
                      "bar": h["bar"], "tick": h["tick"]} for h in pat["hits"]]}
    with open(os.path.join(out_dir, "demo_groundtruth.json"), "w") as f:
        json.dump(gt, f, indent=1)
    print("wav:", wav, os.path.getsize(wav), "bytes")
    if mp3:
        print("mp3:", mp3, os.path.getsize(mp3), "bytes")
    print("eventos GT:", len(gt["events"]), "| duração:", round(x.size / sr, 2), "s")
    return wav, gt


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
    render(out)
