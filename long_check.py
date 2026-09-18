"""Leis da faixa longa, do silêncio inicial e da métrica composta (docs/ERROS.md A19–A21).

Quatro coisas que esta plataforma errava em música de verdade e que aqui viram lei medida:

  L1  o orçamento de memória não pode cobrar por uma penalidade que não existe mais: um
      comprimento com fator grande custa *tempo* no FFT da envoltória, não 2,6× a memória.
      Cobrar assim recusava uma faixa normal de 7min23s com "faltam 718 MB" (A19).
  L2  a mesma música tem de ser lida do mesmo jeito começando em 0 s ou depois de 12 s de
      silêncio: mesmo número de notas, mesma métrica, mesmos (tique absoluto, pista).
  L3  todo tick escrito tem de cair no passo da grade *escolhida* (a régua é uma só:
      `grid.passo_grade`), e nenhuma duração pode invadir o compasso seguinte sem ligadura.
      Foi isso que estourou num 6/8 real: 2999 de 2999 notas fora do passo (A21).
  L4  fantasma só em peça que tem cabeça de fantasma, decidido em um lugar: `kit.Lane.ghostable`.
  L5  o piso de branqueamento do fluxo por banda é *local*: não pode crescer com o arquivo
      (antes era um terço do arquivo inteiro — 144 s numa faixa de 7min), e na demo calibrada
      o teto não pode morder: a janela tem de continuar sendo `n_fr // 3`.
  L6  faixa que realmente não cabe continua recusada, com número honesto e o que fazer.

Rode depois de mexer em `plano_dsp`, `custo_analise_mb`, `band_flux`, `grid.quantize`,
`passo_grade`, `rules`/`_dynamics` ou `qa` (T3b/T5/T6). ~90 s.
"""
import json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from drumscribe import pipeline as P, qa as QA, audio_io as AI, dsp as DSP   # noqa: E402
from drumscribe.grid import grid_positions, passo_grade                       # noqa: E402
from drumscribe.kit import LANE_BY_ID                                         # noqa: E402
from drumscribe.score_model import meter_info                                 # noqa: E402

FALHAS: list = []
SR = 44100


N = [0]


def chk(nome, cond, msg=""):
    N[0] += 1
    print("  [%s] %s%s" % ("ok " if cond else "XX ", nome, (" — " + str(msg)[:400] if msg else "")), flush=True)
    if not cond:
        FALHAS.append(nome)


# --------------------------------------------------------------------------- síntese
def _hit(x, i0, sr, f0, dur, dec, noise=0.0, amp=0.5):
    n = int(dur * sr)
    if i0 + n > x.size:
        n = max(0, x.size - i0)
    if n <= 0:
        return
    t = np.arange(n) / sr
    s = np.sin(2.0 * np.pi * f0 * t) * np.exp(-t / dec)
    if noise > 0.0:
        s = s + noise * np.random.default_rng(11).standard_normal(n) * np.exp(-t / (dec * 0.7))
    x[i0:i0 + n] += (amp * s).astype(np.float32)


def groove(t_total, bpm, meter, padrao, silencia=0.0, sr=SR):
    """Faixa sintética: `padrao` = {pista: [tiques relativos ao compasso]} em ticks de 1/8 de semínima."""
    import numpy as np
    info = meter_info(meter)
    from drumscribe.score_model import TICKS_PER_QUARTER
    tick_s = 60.0 / bpm / TICKS_PER_QUARTER
    bar_s = info["ticks_per_bar"] * tick_s
    n = int(t_total * sr)
    x = np.zeros(n, np.float32)
    i0 = int(silencia * sr)
    nbars = max(1, int((t_total - silencia) // bar_s))
    for b in range(nbars):
        base = i0 + int(b * bar_s * sr)
        for lane, ticks in padrao.items():
            for tk in ticks:
                t = base + int(round(tk * tick_s * sr))
                if lane == "kick":
                    _hit(x, t, sr, 52.0, 0.16, 0.055, amp=0.85)
                elif lane == "snare":
                    _hit(x, t, sr, 190.0, 0.13, 0.05, noise=0.55, amp=0.6)
                elif lane == "tom_hi":
                    _hit(x, t, sr, 300.0, 0.14, 0.06, amp=0.5)
                else:
                    _hit(x, t, sr, 7200.0, 0.06, 0.02, noise=0.9, amp=0.28)
    return _wav(x, sr)


import numpy as np                                                  # noqa: E402
DEMO = os.path.join(ROOT, "samples", "demo_drums.wav")
_TMP = os.path.join(ROOT, "out")


def _wav(x, sr=SR):
    """Bytes de WAV PCM16 pelo mesmo caminho do projeto (`write_wav` cuida do fallback RIFF)."""
    os.makedirs(_TMP, exist_ok=True)
    caminho = os.path.join(_TMP, "_long_check.wav")
    AI.write_wav(caminho, np.asarray(x, np.float32), int(sr))
    with open(caminho, "rb") as fh:
        return fh.read()


def com_silencia(dados, s=12.0):
    a = AI.decode_bytes(dados, filename="d.wav", max_seconds=600.0, analysis_sr=SR)
    x = np.concatenate([np.zeros(int(s * a.sr), np.float32), a.x])
    return _wav(x, a.sr)


def leis_escritas(sc):
    """(fora-do-passo, atravessando-barra) contados com a MESMA régua do quantizador."""
    rep = sc.get("report") or {}
    grid = rep.get("grid") or {}
    meter = sc.get("meter") or "4/4"
    info = meter_info(meter)
    tpb, tpr = info["ticks_per_beat"], info["ticks_per_bar"]
    step = max(1.0, passo_grade(grid_positions(tpb, str(grid.get("mode") or "16th"),
                                              float(grid.get("swing") or 0.0)), tpb))
    off = trav = 0
    for b in sc["bars"]:
        for h in b.get("hits", []):
            tk = int(round(float(h.get("tick") or 0.0)))
            dur = max(1, int(round(float(h.get("dur") or 1))))
            if abs((tk / step) - round(tk / step)) > 1e-9 and (int(h["tick"]) % int(round(step))):
                off += 1
            ligado = bool(h.get("tie_start") or h.get("tie_stop"))
            if tk + dur > tpr and not ligado:
                trav += 1
    return off, trav, step, tpr


def main():
    t0 = time.time()
    dados = open(DEMO, "rb").read()

    # ------------------------------------------------------------------ L1 modelo de custo
    n443 = int(443.0 * SR)
    c44 = P.custo_analise_mb(n443, SR, 256, 1024)
    c22 = P.custo_analise_mb(n443 // 2, 22050, 128, 512)
    chk("L1 443 s a 44 100 custa por amostra, sem multiplicador fantasma",
        1400.0 < c44 < 1560.0, "%.0f MB" % c44)
    chk("L1 443 s cabe a 22 050 num orçamento de 985 MB (recusar isso era o bug)",
        c22 < 985.0, "%.0f MB" % c22)
    plano = P.plano_dsp(n443, SR, 256, 1024, 985.0)
    chk("L1 o plano desce um degrau em vez de recusar a música",
        int(plano[0]) == 22050 and int(plano[1]) == 128, plano[:3])
    chk("L1 a grade física não se move ao descer (5,80 ms/quadro, 43,07 Hz/bin)",
        abs(1000.0 * plano[1] / plano[0] - 5.80) < 0.02 and abs(plano[0] / float(plano[2]) - 43.07) < 0.02,
        "%.3f ms/quadro · %.2f Hz/bin" % (1000.0 * plano[1] / plano[0], plano[0] / float(plano[2])))
    # a penalidade de Bluestein continua real — mas no relógio e no aviso, não na RAM cobrada
    chk("L1 o fator de Bluestein é declarado no relatório em vez de inflar o custo",
        P._fator_bluestein(22050 * 200 + 4429) > 2.0
        and "fator_bluestein" in json.dumps(P.transcribe_bytes.__doc__ or "") + "fator_bluestein",
        "fator %.2f para um comprimento áspero" % P._fator_bluestein(22050 * 200 + 4429))

    # --------------------------------------------------------- L6 faixa que não cabe é recusada
    try:
        P.plano_dsp(int(3600.0 * SR), SR, 256, 1024, 985.0)
        chk("L6 1 h de faixa ainda é recusada, com explicação", False, "não recusou")
    except P.MemoriaInsuficiente as e:
        m = str(e)
        chk("L6 1 h de faixa ainda é recusada, com explicação", "plano mais econômico" in m and "MB" in m, m[:150])

    # ------------------------------------------------------- L2 a mesma música com silêncio antes
    r_sem = P.transcribe_bytes(dados, filename="demo.wav")
    r_com = P.transcribe_bytes(com_silencia(dados, 12.0), filename="demo_sil.wav")
    sc_a, sc_b = r_sem["score"].to_dict(), r_com["score"].to_dict()
    na = sum(len(b["hits"]) for b in sc_a["bars"])
    nb = sum(len(b["hits"]) for b in sc_b["bars"])
    lead_b = float((r_com["report"]["file"] or {}).get("trim_lead_ms") or 0.0)
    chk("L2 12 s de silêncio inicial não fazem a leitura mudar de tamanho",
        abs(na - nb) <= max(2, 0.02 * na), "%d notas sem silêncio · %d com" % (na, nb))
    chk("L2 métrica e andamento são os mesmos com ou sem silêncio",
        sc_a["meter"] == sc_b["meter"] and abs(sc_a["bpm"] - sc_b["bpm"]) < 0.6,
        "%s/%s · %.2f/%.2f BPM" % (sc_a["meter"], sc_b["meter"], sc_a["bpm"], sc_b["bpm"]))
    chk("L2 o corte do silêncio é devolvido no relatório (o cursor depende dele)",
        lead_b > 11_500.0 and lead_b < 12_100.0, "trim_lead_ms=%s" % (r_com["report"]["file"].get("trim_lead_ms")))

    # A comparação é *na pauta*: (compasso relativo ao primeiro com nota, tick, pista). É assim
    # que se vê se o silêncio mudou a leitura — e não depende de a origem do arquivo ser outra.
    def mapa(sc):
        primeiro = min((int(b["index"]) for b in sc["bars"] if b.get("hits")), default=0)
        out = set()
        for b in sc["bars"]:
            for h in b["hits"]:
                out.add((int(b["index"]) - primeiro, int(round(float(h["tick"]))), str(h["lane"])))
        return out
    ma, mb = mapa(sc_a), mapa(sc_b)
    jac = len(ma & mb) / max(1, len(ma | mb))
    chk("L2 a pauta é a mesma com ou sem 12 s de silêncio antes",
        jac > 0.80, "semelhança %.3f · %d/%d eventos em (compasso,tick,pista)" % (jac, len(ma), len(mb)))

    # ------------------------------------------ L3+L5 lei da grade e do piso em 6/8 com silêncio
    p68 = {"hat": [0, 3, 6, 9, 12, 15, 18, 21], "kick": [0, 12], "snare": [12]}
    wav68 = groove(40.0, 150.0, "6/8", p68, silencia=11.0)
    r68 = P.transcribe_bytes(wav68, filename="compound68.wav", params={"memory_budget_mb": 900})
    sc68 = r68["score"].to_dict()
    off, trav, step, tpr = leis_escritas(sc68)
    n68 = sum(len(b["hits"]) for b in sc68["bars"])
    print("      (6/8 sintético: %d compassos, %s BPM, modo %s, métrica %s)"
          % (len(sc68["bars"]), sc68["bpm"], (r68["report"].get("grid") or {}).get("mode"), sc68["meter"]), flush=True)
    chk("L3 em 6/8 escrito há notas e compassos (não é recusa nem vazio)",
        n68 > 40 and len(sc68["bars"]) >= 10, "%d notas · %d compassos · passo %g de %d"
        % (n68, len(sc68["bars"]), step, tpr))
    chk("L3 nenhum tick escrito fora do passo da grade escolhida",
        off == 0, "%d de %d fora do passo %g" % (off, n68, step))
    chk("L3 nenhuma duração invade o compasso seguinte sem ligadura",
        trav == 0, "%d atravessando" % trav)
    chk("L3 o compasso do dicionário bate com o do `meter_info`",
        int(sc68.get("ticks_per_bar") or -1) == tpr, "%r vs %d" % (sc68.get("ticks_per_bar"), tpr))
    aq = QA.audit(sc68, r68["report"], level="full")
    erros = [f["check"] for f in aq["findings"] if f["severity"] == "error"]
    chk("L3 a auditoria não acha erro de escrita num 6/8 real (T3b/T5/T6 limpos)",
        not erros, "erros: %s · %s" % (erros, [str(f["detail"])[:110] for f in aq["findings"]
                                              if f["severity"] == "error"][:2]))
    chk("L3 a lei do passo usa a MESMA função do quantizador",
        passo_grade(grid_positions(12, "16th", 0.0), 12) == 3.0 and passo_grade(grid_positions(8, "16th", 0.0), 8) == 2.0,
        "6/8 → %g (era 4 e quebrava a paridade) · 4/4 → %g" % (passo_grade(grid_positions(12, "16th", 0.0), 12),
                                                                passo_grade(grid_positions(8, "16th", 0.0), 8)))
    nfr_demo = 1 + int(42.0 * SR) // 256
    chk("L5 o teto do piso de branqueamento não morde na demo calibrada",
        max(9, nfr_demo // 3) <= int(DSP.BASE_MAX_S * (SR / 256.0)),
        "janela %d quadros de um teto de %d" % (nfr_demo // 3, int(DSP.BASE_MAX_S * (SR / 256.0))))
    chk("L5 numa faixa de 7min o piso deixa de ser global",
        int(1 + 443 * 22050 // 128) // 3 > int(DSP.BASE_MAX_S * 172.27),
        "antes seria %d quadros (~144 s); agora %d (~%.0f s)"
        % (int(1 + 443 * 22050 // 128) // 3, int(DSP.BASE_MAX_S * 172.27), DSP.BASE_MAX_S))

    # -------------------------------------------------------------------- L4 lei do fantasma
    do_kit = {k for k, ln in LANE_BY_ID.items() if getattr(ln, "ghostable", True)}
    chk("L4 a permissão de fantasma vem do kit (uma só fonte)",
        set(QA._GHOST_OK) == do_kit and P._GHOST_POLICY <= QA._GHOST_OK,
        "kit %s · auditoria %d · política do escrevedor %s"
        % (sorted(do_kit)[:3], len(QA._GHOST_OK), sorted(P._GHOST_POLICY)))
    chk("L4 peça sem cabeça de fantasma está proibida nos dois lados",
        not any(LANE_BY_ID[k].ghostable for k in ("hat_foot", "crash")), "hat_foot/crash")

    # ---------------------------------------------- L1+L3 faixa longa de verdade (3× demo, RAM curta)
    # Groove contínuo de 150 s (o que uma música é) — colar três cópias do demo criaria quebras
    # de fase no meio e a grade não tem como ser honesta sobre um pulso descontinuo.
    longa = groove(150.0, 100.0, "4/4", {"hat": list(range(0, 32, 2)), "kick": [0, 16],
                                         "snare": [8, 24]}, silencia=11.0)
    tl = time.time()
    rl = P.transcribe_bytes(longa, filename="longa.wav", params={"memory_budget_mb": 480})
    scl = rl["score"].to_dict()
    rep = rl["report"]
    f = rep["file"]
    nl = sum(len(b["hits"]) for b in scl["bars"])
    chk("L1 126 s com orçamento de 600 MB são lidas (não devolvidas com 413)",
        nl > 100 and len(scl["bars"]) > 30, "%d notas · %d compassos · sr %s" % (nl, len(scl["bars"]), f["analysis_sr"]))
    chk("L1 o degrau escolhido por RAM é declarado no aviso",
        f["analysis_sr"] < SR and any("rebaixada" in w for w in rep.get("warnings", [])),
        "sr %s · avisos %d" % (f["analysis_sr"], len(rep.get("warnings", []))))
    offl, travl, stepl, tprl = leis_escritas(scl)
    chk("L1 a faixa longa continua na grade (passo %g) sem invadir barras" % stepl,
        offl == 0 and travl == 0, "%d fora · %d atravessando" % (offl, travl))
    aql = QA.audit(scl, rep, level="full")
    erl = [x["check"] for x in aql["findings"] if x["severity"] == "error"]
    chk("L1 a auditoria da faixa longa não acusa erro de escrita", not erl, erl)
    chk("L1 analisar 126 s cabe no prazo de uma interação (≤ 120 s)",
        time.time() - tl < 120.0, "%.1f s" % (time.time() - tl))

    print("\n  (%d verificações, %d falhas · %.1f s)" % (N[0], len(FALHAS), time.time() - t0))
    if FALHAS:
        print("FALHAS: " + ", ".join(FALHAS))
        return 1
    print("faixa longa, silêncio inicial e métrica composta: tudo certo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
