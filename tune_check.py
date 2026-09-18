#!/usr/bin/env python3
"""Valida o recomendador de parâmetros contra gabarito — a lei é *nunca piorar*.

O recomendador (`drumscribe/tune.py`) escolhe `sensitivity` e `min_confidence` por faixa com
critérios que não usam gabarito (duplicatas, grade, densidade, piso de lacuna, auditoria). Como
o projeto tem gabarito para a demo, este teste confere a aposta: o par recomendado não pode
render f1/recall piores que o par vigente. Também confere o orçamento de varredura, o
empate declarado e a determinismo.

    python3 tests/tune_check.py

Demora ~2 min (13 análises da varredura + 2 de avaliação).
"""
from __future__ import annotations

import json
import os
import sys
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from drumscribe.pipeline import PARAMS_DEFAULT, transcribe_bytes  # noqa: E402
from drumscribe.tune import SENS, recomendar_arquivo, recomendar_bytes  # noqa: E402
from evaluate import evaluate  # noqa: E402

DEMO = os.path.join(ROOT, "samples", "demo_drums.wav")
GT = json.load(open(os.path.join(ROOT, "samples", "demo_groundtruth.json"), encoding="utf-8"))

ok, falhas = [], []


def checa(nome: str, cond: bool, dado: str = "") -> None:
    (ok if cond else falhas).append("%s%s" % (nome, (" — " + dado) if dado else ""))
    # imprime na hora: se uma perna mais abaixo quebrar, o que já foi verificado continua visível
    print("  %s %s" % ("✓" if cond else "✗", "%s%s" % (nome, (" — " + dado) if dado else "")))


def mede(par: dict) -> dict:
    with open(DEMO, "rb") as f:
        r = transcribe_bytes(f.read(), filename="demo_drums.wav", params=par)
    return evaluate(r["score"].to_dict(), r["detections"], GT)


rec = recomendar_arquivo(DEMO)
pad = {"sensitivity": float(PARAMS_DEFAULT["sensitivity"]),
       "min_confidence": float(PARAMS_DEFAULT["min_confidence"])}

checa("recomendado presente", bool(rec.get("recomendado")), json.dumps(rec.get("recomendado")))
_orc_demo = rec["referencia"]["orcamento"]
checa("varredura dentro do orçamento declarado",
      6 <= int(rec["referencia"]["celulas"]) <= int(_orc_demo["teto"]),
      "%d ≤ teto %d (%s)" % (rec["referencia"]["celulas"], _orc_demo["teto"], json.dumps(_orc_demo)))
checa("candidatos suficientes", len(rec["candidatos"]) >= 6, "%d" % len(rec["candidatos"]))
checa("motivos declarados", len(rec["motivos"]) >= 1, " || ".join(rec["motivos"])[:200])
checa("empate resolvido por escrito", ("calibração do projeto" in " ".join(rec["motivos"]))
      or bool(rec["candidatos"][0]["motivos"]), " | ".join(rec["motivos"])[:160])
_c_rec = rec["custo_recomendado"] if rec["custo_recomendado"] is not None else float("nan")
_c_atu = (rec["referencia"]["melhor_atual"] or {}).get("custo", float("nan"))
checa("custo do recomendado ≤ custo do vigente",
      rec["custo_recomendado"] is not None and rec["referencia"]["melhor_atual"] is not None
      and _c_rec <= _c_atu + 1e-9, "%.3f vs %.3f" % (_c_rec, _c_atu))
checa("par vigente medido (não estimado)", rec["referencia"]["melhor_atual"] is not None)
checa("nenhum candidato fora da grade declarada",
      all(abs(c["sensitivity"] - min(SENS, key=lambda v: abs(v - c["sensitivity"]))) < 1e-3
          or abs(c["sensitivity"] - float(PARAMS_DEFAULT["sensitivity"])) < 1e-3 for c in rec["candidatos"]),
      " ".join("%.2f" % c["sensitivity"] for c in rec["candidatos"]))

m_rec = mede({**pad, **{k: v for k, v in rec["recomendado"].items() if k in pad}})
m_pad = mede(pad)
checa("recomendado não piora f1 contra o gabarito", m_rec["hit_f1"] >= m_pad["hit_f1"] - 0.005,
      "hit_f1 %.3f (rec) vs %.3f (padrão)" % (m_rec["hit_f1"], m_pad["hit_f1"]))
checa("recomendado não piora recall de golpe", m_rec["hit_recall"] >= m_pad["hit_recall"] - 0.02,
      "hit_recall %.3f vs %.3f" % (m_rec["hit_recall"], m_pad["hit_recall"]))
checa("recomendado não piora precisão", m_rec["hit_precision"] >= m_pad["hit_precision"] - 0.02,
      "hit_precision %.3f vs %.3f" % (m_rec["hit_precision"], m_pad["hit_precision"]))
checa("recomendado não piora recall de onset", m_rec["onset_recall"] >= m_pad["onset_recall"] - 0.02,
      "onset_recall %.3f vs %.3f" % (m_rec["onset_recall"], m_pad["onset_recall"]))
custos = [c["custo"] for c in rec["candidatos"]]
checa("critério discrimina (não é constante)", max(custos) > 0.4,
      "custos %.2f…%.2f" % (min(custos), max(custos)))

# perna não-vácuo: partindo de um par deliberadamente ruim, a varredura tem de achar melhora
ruim = recomendar_arquivo(DEMO, params={"sensitivity": 1.45, "min_confidence": 0.55},
                          sens=(0.82, 0.95), conf=(0.20, 0.30))
checa("par ruim é rejeitado", ruim["recomendado"] != {"sensitivity": 1.45, "min_confidence": 0.55},
      json.dumps(ruim["recomendado"]))
checa("par ruim: custo do recomendado menor", ruim["custo_recomendado"] < (ruim["referencia"]["melhor_atual"] or {}).get("custo", 0.0),
      "%.3f vs %.3f" % (ruim["custo_recomendado"],
                        (ruim["referencia"]["melhor_atual"] or {}).get("custo", -1.0)))
m_ruim = mede({"sensitivity": 1.45, "min_confidence": 0.55})
m_novo = mede({**pad, **ruim["recomendado"]})
checa("mudar de um par ruim melhora o f1 medido", m_novo["hit_f1"] > m_ruim["hit_f1"],
      "hit_f1 %.3f → %.3f" % (m_ruim["hit_f1"], m_novo["hit_f1"]))
checa("veredito diz 'vale a pena mudar' quando muda",
      "nada a mudar" not in " ".join(ruim["motivos"]), " | ".join(ruim["motivos"])[:150])
checa("pos_accuracy continua 1,000", m_rec["pos_accuracy"] >= 0.999,
      "%.3f" % m_rec["pos_accuracy"])
checa("duração da varredura plausível (< 10 min)", rec["referencia"]["segundos"] < 600,
      "%.1f s" % rec["referencia"]["segundos"])

# determinismo: mesmos bytes → mesma escolha
# grade reduzida nas duas pernas: o que se testa é a determinismo da escolha, não o custo da busca
a = recomendar_bytes(open(DEMO, "rb").read(), "demo_drums.wav", sens=(0.95, 1.10), conf=(0.20, 0.30))
b = recomendar_bytes(open(DEMO, "rb").read(), "demo_drums.wav", sens=(0.95, 1.10), conf=(0.20, 0.30))
checa("determinístico", a["recomendado"] == b["recomendado"],
      "%s vs %s" % (a["recomendado"], b["recomendado"]))

# arquivo curto: orçamento limita as confiançaes a 2, sem travar
curto = "/tmp/tune_curto.wav"
with wave.open(curto, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(22050)
    n_amostras = 3 * 22050                       # 3 s: os picos têm de caber no vetor
    x = np.random.default_rng(7).standard_normal(n_amostras) * 0.05
    for i in range(6):
        p_ = int(n_amostras * (0.06 + 0.15 * i))
        x[p_:p_ + 400] += 0.6 * np.hanning(400)
    w.writeframes((x * 32767).astype("<i2").tobytes())
rc = recomendar_arquivo(curto)
checa("arquivo curto responde", bool(rc.get("recomendado")), json.dumps(rc["recomendado"]))
checa("grade reduzida em arquivo curto", len(rc["candidatos"]) >= 2, str(len(rc["candidatos"])))
orc = rc["referencia"]["orcamento"]
teto = int(orc["teto"])
checa("orçamento respeitado (células ≤ previsto)", int(rc["referencia"]["celulas"]) <= teto,
      "%d ≤ %d (orçamento %s)" % (rc["referencia"]["celulas"], teto, json.dumps(orc)))
from drumscribe.tune import _orcamento_celulas
curta, longa = _orcamento_celulas(3.0), _orcamento_celulas(300.0)
checa("orçamento aperta quando a faixa cresce", longa[0] <= curta[0] and longa[1] <= curta[1],
      "3 s → %s · 300 s → %s" % (curta, longa))
checa("o ponto vigente desceu o eixo da confiança junto",
      any(abs(c["sensitivity"] - float(rec["referencia"]["atual"]["sensitivity"])) < 1e-6
          and c["min_confidence"] > 0.2001 for c in rec["candidatos"]),
      " ".join("%.2f/%.2f" % (c["sensitivity"], c["min_confidence"]) for c in rec["candidatos"]))
checa("a verificação fim-a-fim é paga ou declarada como pulada",
      (rc["verificacao"] or {}).get("pulado") is True or int(rc["referencia"]["celulas"]) <= teto,
      "células %d · verif %s" % (rc["referencia"]["celulas"], json.dumps({
          k: rc["verificacao"][k] for k in ("custo", "pulado") if k in (rc["verificacao"] or {})})))

print()
print("\nhit_f1 rec %.3f · padrão %.3f · ruim %.3f → corrigido %.3f | "
      "recomendado %s/%s (varredura do ruim: %s/%s)"
      % (m_rec["hit_f1"], m_pad["hit_f1"], m_ruim["hit_f1"], m_novo["hit_f1"],
         rec["recomendado"]["sensitivity"], rec["recomendado"]["min_confidence"],
         ruim["recomendado"]["sensitivity"], ruim["recomendado"]["min_confidence"]))
print("  (%d ok, %d falha%s)" % (len(ok), len(falhas), "" if len(falhas) == 1 else "s"))
sys.exit(1 if falhas else 0)
