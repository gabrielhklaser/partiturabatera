"""Leis do degrau de memória (docs/ERROS.md A13/A14).

Um degrau escolhido **pela máquina** nunca pode mover uma nota nem calar uma pista. Um degrau
escolhido **pelo usuário** (`analysis_sr`) pode custar recall, mas tem de rodar, tem de avisar
e tem de continuar gravando o instante certo — é o que este teste mede no demo com gabarito.

Cinco análises do demo (~25 s no total). Não está no `selfcheck` por causa do custo; rode
antes de mexer em `plano_dsp`, `hop`, `n_fft`, `resample_to`, `extract_features` ou em
qualquer envelope — todos eles aparecem aqui.
"""
import json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from drumscribe import pipeline as P                           # noqa: E402
from drumscribe.audio_io import decode_bytes                    # noqa: E402
from evaluate import evaluate                                   # noqa: E402

GT = json.load(open(os.path.join(ROOT, "samples", "demo_groundtruth.json")))
DATA = open(os.path.join(ROOT, "samples", "demo_drums.wav"), "rb").read()
FALHAS: list = []


def roda(**params):
    r = P.transcribe_bytes(DATA, filename="demo_drums.wav", params=dict(params))
    rep, m = r["report"], evaluate(r["score"].to_dict(), r["detections"], GT)
    f = rep["file"]
    return {"sr": f["analysis_sr"], "hop": f["hop"], "n_fft": f["n_fft"], "ms": f["ms_por_quadro"],
            "hz": f["hz_por_bin"], "nyq": f["nyquist_hz"], "warn": rep.get("warnings") or [],
            "ajustado": (rep.get("memory") or {}).get("ajustado_por_memoria"),
            "planos": (rep.get("memory") or {}).get("planos") or [],
            "recall": m.get("onset_recall"), "prec": m.get("onset_precision"), "f1": m.get("hit_f1"),
            "pos": m.get("pos_accuracy"), "tick": m.get("tick_mae"), "bias": m.get("time_bias_ms"),
            "mae": m.get("time_mae_ms"), "vel": m.get("vel_pearson"), "notas": m.get("n_notated")}


def chk(nome, cond, msg=""):
    print("  [%s] %s%s" % ("ok " if cond else "XX ", nome, (" — " + msg if not cond and msg else "")))
    if not cond:
        FALHAS.append(nome)


t = time.time()
base = roda()
print("\npadrão  sr=%d hop=%d n_fft=%d  %.2f ms/quadro  %.2f Hz/bin"
      "  recall=%.3f f1=%.3f pos=%.3f %+.2f tick  mae=%.2f vel=%.3f notas=%d"
      % (base["sr"], base["hop"], base["n_fft"], base["ms"], base["hz"], base["recall"],
         base["f1"], base["pos"], base["tick"], base["mae"], base["vel"], base["notas"]))
print("\n%-7s %-5s %-6s %-7s %-8s %-8s %-7s %-7s %-7s %-7s %-6s %-6s %s"
      % ("sr", "hop", "n_fft", "ms/q", "Hz/bin", "recall", "Δrecall", "f1", "pos", "tick",
         "mae", "notas", "aviso"))
for sr_req in (22050, 11025):
    v = roda(analysis_sr=sr_req)
    auto = sr_req >= P.SR_MINIMO_ESCADA
    mudas = [w for w in v["warn"] if "Nyquist" in w]
    print("%-7d %-5d %-6d %-7.2f %-8.2f %-8.3f %-7.3f %-7.3f %-7.3f %-8s %-7.2f %-6d %s"
          % (v["sr"], v["hop"], v["n_fft"], v["ms"], v["hz"], v["recall"],
             v["recall"] - base["recall"], v["f1"], v["pos"], "%+.2f" % v["tick"], v["mae"], v["notas"],
             ("muda: " + ",".join(sorted(k for (k, (f0, f1)) in P.BANDS.items()
                                        if f0 >= v["sr"] / 2.0))) if mudas else "—"))
    tag = "auto" if auto else "usuário"
    # lei universal: a grade física não muda quando a taxa muda
    chk("%s sr %d: ms/quadro do padrão" % (tag, sr_req), abs(v["ms"] - base["ms"]) < 0.05,
        "%.2f vs %.2f" % (v["ms"], base["ms"]))
    chk("%s sr %d: Hz/bin do padrão" % (tag, sr_req), abs(v["hz"] - base["hz"]) < 0.6,
        "%.2f vs %.2f" % (v["hz"], base["hz"]))
    # lei universal: nenhum degrau de taxa move nota escrita
    chk("%s sr %d: pos_accuracy intacto" % (tag, sr_req), v["pos"] >= base["pos"] - 1e-9,
        "%.3f vs %.3f" % (v["pos"], base["pos"]))
    chk("%s sr %d: |tick_mae| ≤ 0.05" % (tag, sr_req), abs(v["tick"]) <= 0.05, "%+.2f" % v["tick"])
    if auto:
        # lei do degrau automático: não pode custar detecções (só a metade do espectro, já dita)
        chk("auto sr %d: recall dentro de ±0.05" % sr_req,
            abs(v["recall"] - base["recall"]) <= 0.05,
            "%.3f vs %.3f" % (v["recall"], base["recall"]))
    else:
        # degrau só do usuário: tem de ser denunciado no relatório, senão é mentira
        chk("usuário sr %d: relatório avisa das pistas mudas" % sr_req, bool(mudas), "sem aviso")

# taxa fora da faixa: falha alta, não silenciosa (era o A14: o parâmetro era aceito e ignorado)
try:
    decode_bytes(DATA, "demo.wav", analysis_sr=5512)
    chk("analysis_sr=5512 recusa com erro claro", False, "aceitou em silêncio")
except ValueError as e:
    chk("analysis_sr=5512 recusa com erro claro", "fora da faixa" in str(e), str(e)[:70])

# a escada oferecida não pode conter degrau que mude a resolução temporal
# `planos` do relatório: (sr, hop, n_fft, custo) — é o que a UI de auditoria mostra.
planos = P.plano_dsp(3695580, 44100, 256, 1024, 10_000.0)[4]
ms = sorted({round(1000.0 * h / st, 3) for (st, h, nf, c) in planos})
hz = sorted({round(st / float(nf), 2) for (st, h, nf, c) in planos})
chk("planos: ms/quadro idêntico em todos os degraus", len(ms) == 1, "vistas: %s" % ms)
chk("planos: Hz/bin idêntico em todos os degraus", len(hz) == 1, "vistas: %s" % hz)
chk("planos: nenhum degrau abaixo do piso que cala pista",
    all(st >= P.SR_MINIMO_ESCADA for (st, h, nf, c) in planos), str(planos))
chk("planos: todos acima do limiar de 6 ms/quadro",
    all(1000.0 * h / st <= P.MS_MAX_POR_QUADRO + 1e-9 for (st, h, nf, c) in planos), str(ms))
custos = [round(c, 0) for (st, h, nf, c) in planos]
# O que tem de cair pela metade é a **parte variável** do custo, não o total: os 140 MB de
# interpretador + bibliotecas estão lá seja qual for a taxa (e o `custo_analise_mb` foi
# recalibrado por medição em A19, o que tornou a base proporcionalmente maior num trecho
# curto). Exigir `total2 < 0.62·total1` era exigir que a base caísse junto — fisicamente falso.
base = float(P.MB_BASE_INTERPRETADOR)
var = [max(0.0, c - base) for c in custos]
chk("planos: a parte variável do custo cai pela metade quando a taxa cai pela metade",
    len(var) < 2 or var[0] <= 0.0 or var[1] < 0.55 * var[0],
    "totais %s · variável %s (%.2f×)" % (custos, [round(v, 1) for v in var],
                                         (var[1] / var[0]) if var and var[0] > 0 else -1))

print("\n%.1f s, %d falhas" % (time.time() - t, len(FALHAS)))
sys.exit(1 if FALHAS else 0)
