#!/usr/bin/env python3
"""Gera faixas longas (bateria isolada) para medir o custo de memória da análise.

    python3 tests/make_long.py --duracoes 167 335 586

Loop da faixa de demonstração — o conteúdo repetido é deliberado: o que se quer medir é o
custo por segundo de áudio, não a dificuldade musical. Sai em PCM16 mono no sample rate da
fonte, o formato mais barato de decodificar (assim o número atribui ao DSP o que é do DSP).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402


def main() -> int:
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonte", default=os.path.join(raiz, "samples", "demo_drums.wav"))
    ap.add_argument("--duracoes", type=float, nargs="*", default=[167.0, 335.0, 586.0])
    ap.add_argument("--para", default="/tmp")
    a = ap.parse_args()
    x, sr = sf.read(a.fonte, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    for dur in a.duracoes:
        n = max(1, int(round(dur * sr / x.size)))
        y = np.tile(x, n)
        out = os.path.join(a.para, f"long_{int(y.size / sr)}s.wav")
        sf.write(out, y, sr, subtype="PCM_16")
        print(f"{out}  {y.size / sr:.1f} s  {os.path.getsize(out)/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
