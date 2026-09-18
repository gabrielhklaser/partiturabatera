#!/usr/bin/env python3
"""
Validação dupla da plataforma inteira: cada estágio do pipeline é conferido contra um oráculo
independente, as leis de escrita da partitura são checadas, os exports são regerados e **relidos
por parsers próprios**, e o agente de correção é testado com defeitos injetados.

Uso:
    python3 tests/doublecheck.py                 # audit na faixa demo (wav + mp3) + injeção de defeitos
    python3 tests/doublecheck.py --fix           # idem, deixando o agente reparar o que for reparável
    python3 tests/doublecheck.py --file faixa.wav --out out/qa   # audit de qualquer arquivo (escreve .md/.json)

Sai com código ≠ 0 se algum achado `error` sobreviver à auditoria (é o gate de CI do projeto).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from drumscribe import qa                                    # noqa: E402
from drumscribe.audio_io import decode_file                  # noqa: E402
from drumscribe.pipeline import PARAMS_DEFAULT, transcribe_bytes   # noqa: E402

FAIL = "\033[31mFALHA\033[0m"
OK = "\033[32mok\033[0m"


def analyze(path: str, params: dict | None = None):
    res = transcribe_bytes(open(path, "rb").read(), filename=os.path.basename(path), params=params)
    try:
        a = decode_file(path)
        x, sr = np.asarray(a.x, dtype=np.float32), int(a.sr)
    except Exception:
        x, sr = None, 0
    return res["score"].to_dict(), res["report"], res.get("detections"), x, sr


def inject(score: dict) -> dict:
    """
    Defeitos sintéticos, um por lei, para provar que o verificador *vê* e o agente *conserta*:
    duplicata, tick fora do compasso, duração 0, velocidade fora, articulação inventada,
    invadindo o próximo ataque, 𝄄 sem repetição e pista ausente do cabeçalho.
    """
    s = copy.deepcopy(score)
    b0 = s["bars"][0]
    h0 = dict(b0["hits"][0])
    b0["hits"].append(dict(h0, velocity=int(h0.get("velocity", 90)) + 1))       # duplicata
    b0["hits"].append({"lane": "snare", "tick": 99, "dur": 2, "velocity": 90, "artic": "normal",
                       "confidence": 0.9})                                       # tick inválido
    b0["hits"].append({"lane": "kick", "tick": 4, "dur": 0, "velocity": 300, "artic": "blast",
                       "confidence": 0.9})                                       # dur/vel/artic errados
    b0["hits"].append({"lane": "hat", "tick": 6, "dur": 20, "velocity": 90, "artic": "normal",
                       "confidence": 0.9})                                       # invade o próximo ataque
    s["bars"][2]["hits"].append({"lane": "splash_qa", "tick": 0, "dur": 2, "velocity": 90,
                                 "artic": "normal", "confidence": 0.9})           # pista desconhecida
    s["bars"][3]["repeat_slash"] = True                                          # 𝄄 sem repetição
    s["bars"][1]["hits"][0]["tie_start"] = True                                   # ligadura pendurada
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="validação dupla + teste do agente de correção")
    ap.add_argument("--file", default=os.path.join(ROOT, "samples", "demo_drums.wav"))
    ap.add_argument("--gt", default=os.path.join(ROOT, "samples", "demo_groundtruth.json"))
    ap.add_argument("--sens", type=float, default=PARAMS_DEFAULT["sensitivity"])
    ap.add_argument("--fix", action="store_true", help="deixa o agente aplicar os reparos determinísticos")
    ap.add_argument("--out", default="", help="prefixo para gravar relatório (.md/.json)")
    ap.add_argument("--level", default="full", choices=("fast", "full"))
    args = ap.parse_args()

    p = dict(PARAMS_DEFAULT)
    p["sensitivity"] = args.sens
    n_fail = 0
    print("=" * 78)
    print("validação dupla — DrumScribe")
    print("=" * 78)

    # ---------------------------------------------------------------- 1. audit na(s) faixa(s) real(is)
    targets = [args.file]
    mp3 = os.path.join(ROOT, "samples", "demo_drums.mp3")
    if os.path.exists(mp3) and args.file.endswith(".wav"):
        targets.append(mp3)
    scores = {}
    for path in targets:
        print("\n--- %s" % os.path.basename(path))
        score, rep, dets, x, sr = analyze(path, p)
        if args.fix:
            r = qa.audit_and_repair(score, rep, x=x, sr=sr, dets=dets, level=args.level)
            rep_out = r["after"]
            print("    reparos do agente: %s" % (json.dumps(r["applied"], ensure_ascii=False) or "—"))
            if r.get("reverted"):
                print("    (um reparo foi revertido pela re-auditoria)")
            score = r["score_novo"] if r["changed"] else score
        else:
            rep_out = qa.audit(score, rep, x=x, sr=sr, dets=dets, level=args.level)
        errors = [f for f in rep_out["findings"] if f["severity"] == "error"]
        warns = [f for f in rep_out["findings"] if f["severity"] == "warn" and f["state"] != "ok"]
        for f in errors + warns:
            print("    %s [%s] %s — %s" % (FAIL, f["check"], f["title"], (f["detail"] or "")[:150]))
            n_fail += 1
        print("    %s  (%s)" % (OK if not errors else "", rep_out["summary"]))
        scores[os.path.basename(path)] = (score, rep, dets, x, sr, rep_out)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            stem = args.out + "." + os.path.splitext(os.path.basename(path))[0]
            with open(stem + ".md", "w", encoding="utf-8") as fh:
                fh.write(qa.to_markdown(rep_out, "Validação dupla — %s" % os.path.basename(path)))
            with open(stem + ".json", "w", encoding="utf-8") as fh:
                json.dump(rep_out, fh, ensure_ascii=False, indent=1, default=str)
            print("    relatório: %s{.md,.json}" % stem)

    # ---------------------------------------------------------------- 2. acurácia contra o gabarito
    if os.path.exists(args.gt):
        gt = json.load(open(args.gt, encoding="utf-8"))
        from tests.evaluate import evaluate
        for name, (score, rep, dets, x, sr, rep_out) in scores.items():
            m = evaluate(score, dets or [], gt)
            print("\n--- gabarito (%s)" % name)
            print("    recall %.3f · precisão %.3f · hit_f1 %.3f · pos_accuracy %.3f · tick_mae %.2f · BPM %.2f"
                  % (m["onset_recall"], m["onset_precision"], m["hit_f1"], m["pos_accuracy"],
                     m["tick_mae"], m["bpm"]))
            # as duas visões têm de concordar: o que a auditoria chama de "erro" não pode
            # conviver com métrica boa, nem o contrário
            checks = {f["check"]: f for f in rep_out["findings"]}
            for k in ("T3", "T4", "T5", "T6", "T7", "T8", "T9", "T11", "T12", "T13", "T14"):
                f = checks.get(k)
                if f is None:
                    print("    %s verificador %s ausente!" % (FAIL, k))
                    n_fail += 1
                elif f["severity"] == "error":
                    print("    %s %s ainda acusa erro: %s" % (FAIL, k, (f["detail"] or "")[:110]))
                    n_fail += 1
            if m["tick_mae"] > 1.0:
                print("    %s tick_mae alto (%.2f) mas a auditoria não acusou nada" % (FAIL, m["tick_mae"]))
                n_fail += 1

    # ---------------------------------------------------------------- 3. defeitos injetados
    print("\n--- injeção de defeitos (o verificador tem de ver; o agente tem de consertar)")
    name = os.path.basename(targets[0])
    score, rep, dets, x, sr, _ = scores[name]
    broken = inject(score)
    before = qa.audit(broken, rep, x=x, sr=sr, dets=dets, level="fast")
    seen = {f["check"]: f for f in before["findings"] if f["severity"] == "error"}
    expect = {"T4": "duplicata", "T6": "campos/faixas", "T5": "invasão de ataque",
              "T9": "𝄄 falso", "T8": "ligadura pendurada", "T11": "MusicXML inconsistente"}
    for k, what in expect.items():
        if k in seen:
            print("    %s visto: %-4s %s" % (OK, k, what))
        else:
            print("    %s não detectado: %-4s %s" % (FAIL, k, what))
            n_fail += 1
    fixed, applied = qa.repair(broken, before["findings"], rep)
    after = qa.audit(fixed, rep, x=x, sr=sr, dets=dets, level="fast")
    left = [f for f in after["findings"] if f["severity"] == "error"]
    print("    reparos aplicados: %s" % json.dumps(applied, ensure_ascii=False))
    if not applied:
        print("    %s agente não aplicou nada" % FAIL)
        n_fail += 1
    for f in left:
        print("    %s sobrou após reparo: %s — %s" % (FAIL, f["title"], (f["detail"] or "")[:120]))
        n_fail += 1
    if not left:
        print("    %s re-auditoria limpa: %d/%d achados de erro resolvidos"
              % (OK, len(expect), len(seen) or len(expect)))
    if not left:
        print("    %s re-auditoria limpa: %d/%d achados de erro resolvidos"
              % (OK, len(expect), len(seen) or len(expect)))
    # o agente pode mexer no que está errado; nota LICITA que sumir é perda de música
    from collections import Counter
    from drumscribe.kit import LANE_BY_ID

    def legal(hs, tpr):
        """Multiconjunto de (compasso,tick,peça) das notas que passam nas leis de campo."""
        out = Counter()
        for h in hs:
            if str(h.get("lane")) not in LANE_BY_ID:
                continue
            try:
                tk = int(round(float(h.get("tick"))))
                dur = int(round(float(h.get("dur") or 0)))
                vel = int(round(float(h.get("velocity") or 0)))
            except (TypeError, ValueError):
                continue
            if not (0 <= tk < tpr) or dur < 1 or not (1 <= vel <= 127):
                continue
            out[(int(h["bar"]), tk, str(h.get("lane")))] += 1
        return out

    tpr0 = int(score.get("ticks_per_bar") or 32)
    lb, laf = legal(qa._hits_of(broken), tpr0), legal(qa._hits_of(fixed), tpr0)
    # nota lícita única tem de sobreviver; se havia duplicata (ilegal por T4), o reparo tem de
    # deixá-la exatamente uma — perder a última cópia lícita é perder música
    lost = sorted(k for k, c in lb.items() if c == 1 and laf.get(k, 0) < 1)
    over = sorted((k, lb[k], laf.get(k, 0)) for k, c in lb.items() if c > 1 and laf.get(k, 0) != 1)
    if lost or over:
        print("    %s o reparo perdeu nota lícita: %s%s" % (FAIL, lost[:6],
              (" | duplicatas não resolvidas: %s" % over[:3]) if over else ""))
        n_fail += 1
    else:
        print("    %s nenhuma nota lícita perdida (%d posições preservadas, %d duplicata(s) colapsada(s))"
              % (OK, len(laf), sum(1 for c in lb.values() if c > 1)))

    # reparo tem de ser idempotente e não pode mudar a partitura boa
    again, applied2 = qa.repair(fixed, after["findings"], rep)
    if applied2:
        print("    %s reparo não é idempotente (segunda passada mudou %s)" % (FAIL, applied2))
        n_fail += 1
    else:
        print("    %s reparo idempotente" % OK)
    same, applied3 = qa.repair(score, qa.audit(score, rep, x=x, sr=sr, level="fast")["findings"], rep)
    if applied3:
        print("    %s agente mexeu numa partitura sem defeito: %s" % (FAIL, applied3))
        n_fail += 1
    else:
        print("    %s partitura íntegra: nenhuma alteração" % OK)

    print("\n" + "=" * 78)
    if n_fail:
        print("%d problema(s) encontrado(s)." % n_fail)
        return 1
    print("validação dupla completa: nenhum erro remanescente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
