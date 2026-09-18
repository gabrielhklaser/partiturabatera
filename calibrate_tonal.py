#!/usr/bin/env python3
"""
Calibra (por medida, não por palpite) o indicador `qa.mid_modulation`, usado no achado T18
"A faixa se comporta como bateria isolada?".

Estímulos: a faixa de bateria sozinha e a mesma faixa com um colchão sustentado somado em
três níveis, mais uma "voz" sintética (trem de harmônicos com formante em ~1 kHz). O
indicador é a profundidade de modulação P95/P25 do envelope da banda 300–3000 Hz.

Critério de aceitação (o script falha se não valer): dyn de bateria isolada deve ficar pelo
menos 40× acima do pior caso contaminado "audível" (pad a −20 dB), e o limiar escolhido deve
classificar todos os casos do quadro. Grava `samples/tonal_calibration.json`.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from drumscribe import qa                      # noqa: E402
from drumscribe.audio_io import decode_file    # noqa: E402


def pad(sr: int, n: int, db: float) -> np.ndarray:
    t = np.arange(n) / sr
    y = sum(np.sin(2 * np.pi * f * t + 0.7 * i) for i, f in enumerate((300.0, 500.0, 750.0)))
    y *= 0.25 + 0.12 * np.sin(2 * np.pi * 4.5 * t)
    return (y * 10 ** (db / 20.0)).astype(np.float32)


def voz(sr: int, n: int, db: float) -> np.ndarray:
    t = np.arange(n) / sr
    f0 = 180.0 * (1.0 + 0.03 * np.sin(2 * np.pi * 3.1 * t))
    y = np.zeros(n)
    for k in range(1, 14):
        amp = 1.0 / k * np.exp(-0.5 * ((k * f0 - 1000.0) / 600.0) ** 2)
        y += amp * np.sin(2 * np.pi * k * np.cumsum(f0) / sr)
    return (y * (0.55 + 0.45 * np.sin(2 * np.pi * 2.3 * t) ** 2) * 10 ** (db / 20.0)).astype(np.float32)


def main() -> int:
    a = decode_file(os.path.join(ROOT, "samples", "demo_drums.wav"))
    x, sr = np.asarray(a.x, np.float32), int(a.sr)
    n = x.size
    cases = {"drums_only": x,
             "pad_minus30": np.clip(x + pad(sr, n, -30.0), -1, 1),
             "pad_minus26": np.clip(x + pad(sr, n, -26.0), -1, 1),
             "pad_minus20": np.clip(x + pad(sr, n, -20.0), -1, 1),
             "voz_minus26": np.clip(x + voz(sr, n, -26.0), -1, 1),
             "voz_minus18": np.clip(x + voz(sr, n, -18.0), -1, 1)}
    dyn = {k: qa.mid_modulation(v, sr)[0] for k, v in cases.items()}
    print("P95/P25 da banda 300–3000 Hz:")
    for k in cases:
        print("   %-14s %10.2f" % (k, dyn[k]))
    # critério: colchão alto o bastante para gerar golpe espúrio (−20 dB e −18 dB relativo)
    # tem de cair abaixo do limiar, e a faixa limpa tem de ficar ≥ 40× acima dele.
    loud_bad = ("pad_minus20", "voz_minus18")
    clean, worst = dyn["drums_only"], max(dyn[k] for k in loud_bad)
    if not (worst > 0 and clean >= 40.0 * worst):
        print("FALHA: separação insuficiente (limpo %.1f vs pior %.1f = %.1f×)."
              % (clean, worst, clean / max(worst, 1e-9)))
        return 1
    thr = round(worst * 1.6, 1)
    if clean < thr:
        print("FALHA: limiar %.1f acusaria faixa limpa" % thr)
        return 1
    mis = [k for k in loud_bad if dyn[k] >= thr]
    if mis:
        print("FALHA: limiar %.1f não pega contaminação relevante em %s" % (thr, mis))
        return 1
    gray = [k for k in ("pad_minus30", "pad_minus26", "voz_minus26") if dyn[k] >= thr]
    print("zona cinzenta (colchões baixos, não sinalizados de propósito): %s" % (gray or "—"))
    out = dict(threshold=thr, separation=round(clean / worst, 1),
               gray_zone_below_threshold=gray,
               **{k: round(v, 2) for k, v in dyn.items()},
               note="limiar = 1,6× o pior caso de colchão audível (−20 dB relativo). "
                    "dyn medido em arquivo sintético com silêncio digital entre golpes é ordens de "
                    "grandeza maior que qualquer gravação real; por isso o limiar é deliberadamente "
                    "baixo e só acusa colchão forte o bastante para virar golpe espúrio.")
    path = os.path.join(ROOT, "samples", "tonal_calibration.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("OK: limiar %.1f (separação %.0f×) → %s" % (thr, clean / worst, os.path.relpath(path, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
