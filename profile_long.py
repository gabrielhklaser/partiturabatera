#!/usr/bin/env python3
"""Porta de verificação de memória para faixas longas (docs/ERROS.md A13/A14).

O que isto garante, e por que cada parte existe:

* **pico medido por processo** — cada duração roda num processo filho próprio. `ru_maxrss` é
  marca de alto-nível *acumulada* do processo: medir três durações em sequência no mesmo
  processo faz o pico da segunda incluir o da primeira, e o limite vira farsa (isto já enganou
  uma medição aqui — ver a nota em `docs/ERROS.md` A13).
* **`--limite MB`** — com o teto passado, o caso que passar do limite faz o script sair 1.
* **recusa legível é resultado válido** — se a faixa não cabe, o esperado é
  `MemoriaInsuficiente` com mensagem que diz o quê fazer; isso conta como ok, estourar a RAM
  não. O servidor traduz isso em HTTP 413 (ver `_erro_analise`).
* **a escada não move a grade** — todos os casos aceitos têm de chegar com o mesmo
  ms/quadro e o mesmo Hz/bin. Foi exatamente isto que a regra antiga de `hop = 512` violou,
  gravando toda nota um tick adiantada (A14).

    python3 tests/profile_long.py --limite 900
    python3 tests/profile_long.py                       # só mede
    python3 tests/make_long.py && python3 tests/profile_long.py --limite 900
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

DEFAULT = ("/tmp/long_167s.wav", "/tmp/long_335s.wav", "/tmp/long_586s.wav")


def pico_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def medir(caminho: str) -> dict:
    """Roda UMA faixa e devolve a medição. Chamado no processo filho."""
    from drumscribe.pipeline import MemoriaInsuficiente, PARAMS_DEFAULT, transcribe_bytes
    t0 = time.time()
    with open(caminho, "rb") as f:
        data = f.read()
    saida: dict = {"arquivo": os.path.basename(caminho), "recusa": None, "ok": False,
                   "eventos": 0, "compassos": 0, "ms_quadro": None, "hz_bin": None,
                   "sr": None, "hop": None, "seg": 0.0}
    try:
        r = transcribe_bytes(data, filename=os.path.basename(caminho),
                             params=dict(PARAMS_DEFAULT))
    except MemoriaInsuficiente as e:
        saida["recusa"] = str(e)
        saida["ok"] = "memória" in str(e).lower()
        saida["seg"] = round(time.time() - t0, 1)
        saida["pico_mb"] = round(pico_mb(), 1)
        return saida
    rep = r["report"]
    f = rep.get("file") or {}
    saida.update({"ok": True, "sr": f.get("analysis_sr"), "hop": f.get("hop"),
                  "n_fft": f.get("n_fft"), "ms_quadro": f.get("ms_por_quadro"),
                  "hz_bin": f.get("hz_por_bin"),
                  "ajustado": bool((rep.get("memory") or {}).get("ajustado_por_memoria")),
                  "custo_mb": (rep.get("memory") or {}).get("custo_mb"),
                  "orcamento_mb": (rep.get("memory") or {}).get("orcamento_mb"),
                  "compassos": len(r["score"].to_dict().get("bars") or []),
                  "eventos": len(r.get("detections") or []),
                  "dur": f.get("duration_sec"),
                  "seg": round(time.time() - t0, 1)})
    saida["pico_mb"] = round(pico_mb(), 1)
    return saida


def um_caso(caminho: str, limite: float) -> tuple:
    """Executa o caso num processo novo e lê o pico **dele**."""
    rodar = [sys.executable, "-u", os.path.abspath(__file__), "--_caso", caminho]
    t0 = time.time()
    p = subprocess.run(rodar, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600)
    lin = (p.stdout or b"").decode("utf-8", "ignore").strip().splitlines()
    if not lin:
        return (f"{os.path.basename(caminho)}: filho sem saída (código {p.returncode}) "
                f"{(p.stderr or b'').decode('utf-8','ignore')[-160:]}", float("inf"), True, {})
    m = json.loads(lin[-1])
    estourou = bool(limite) and m["pico_mb"] > limite
    if m.get("recusa"):
        msg = ("%s (%s s): RECUSA legível · %d chars · pico %.0f MB · %.0f s%s"
               % (m["arquivo"], ("%.0f" % m["dur"]) if m.get("dur") else "?",
                  len(m["recusa"]), m["pico_mb"], m["seg"],
                  "" if m["ok"] else " · MENSAGEM SEM A PALVRA 'memória'"))
        # um recado ilegível não é recusa graciosa: é o bug de novo
        return msg, m["pico_mb"], estourou or not m["ok"], m
    grades = "%.2f ms/quadro · %.2f Hz/bin" % (m["ms_quadro"] or 0, m["hz_bin"] or 0)
    msg = ("%s (%.0f s): pico %.0f MB previsto %.0f MB de %.0f MB · %d eventos · %d compassos"
           " · %d Hz hop %d · %s · %s%.0f s%s"
           % (m["arquivo"], m.get("dur") or 0, m["pico_mb"], m.get("custo_mb") or 0,
              m.get("orcamento_mb") or 0, m["eventos"], m["compassos"], m["sr"], m["hop"],
              grades, "rebaixado, " if m.get("ajustado") else "fiel, ", m["seg"],
              " · ACIMA do limite" if estourou else ""))
    return msg, m["pico_mb"], estourou, m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=float, default=0.0,
                    help="teto de pico de RSS por processo, em MB (0 = só medir)")
    ap.add_argument("--_caso", dest="caso", default=None, help=argparse.SUPPRESS)
    ap.add_argument("arquivos", nargs="*", default=None)
    a = ap.parse_args()

    if a.caso:
        print(json.dumps(medir(a.caso), ensure_ascii=False))
        return 0

    arqs = a.arquivos or [p for p in DEFAULT if os.path.exists(p)]
    if not arqs:
        print("sem faixas de longa duração (gere com `python3 tests/make_long.py`); nada a medir")
        return 0
    total = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES")) / 1e6
    print("%s: %d faixa(s), RAM da máquina %.0f MB, limite %.0f MB"
          % (os.path.basename(__file__), len(arqs), total, a.limite))
    ruins: list = []
    grades: list = []
    for p in arqs:
        msg, pico, ruim, m = um_caso(p, a.limite)
        print("  " + ("FAIL " if ruim else "ok   ") + msg)
        if ruim:
            ruins.append(p)
        if m.get("ok") and not m.get("recusa"):
            grades.append((m["arquivo"], m.get("ms_quadro"), m.get("hz_bin")))
    unicas = sorted({(g[1], g[2]) for g in grades})
    if len(unicas) > 1:
        print("  FAIL grade movida entre durações (A14): %s" % grades)
        ruins.append("grade")
    elif grades:
        print("  ok   grade física idêntica nos %d casos aceitos: %.2f ms/quadro, %.2f Hz/bin"
              % (len(grades), unicas[0][0] or 0, unicas[0][1] or 0))
    return 1 if ruins else 0


if __name__ == "__main__":
    raise SystemExit(main())
