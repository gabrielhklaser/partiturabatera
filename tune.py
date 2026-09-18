"""Recomendador de `sensitivity` e `min_confidence` para a faixa concreta.

Por que isto existe: os dois parâmetros são os que mais mudam o resultado de uma transcrição,
e o usuário não tem como escolhe-los sem gabarito — quem ouve "pareceu faltar um prato" aumenta
a sensibilidade e ganha duplicata; quem acha "nota demais" corta confiança e perde o chimbal.
Este módulo mede a faixa e recomenda, com os números à mostra.

**O que é medido** (nada aqui exige gabarito; todos os limiares são leis que já existem no
projeto, reutilizadas — não constantes inventadas para esta função):

* `p1_gap_ms` — **piso** (1º percentil) do intervalo entre golpes consecutivos na mesma pista.
  A mediana não serve sozinha: com bumbo a 2 por compasso a mediana é ~1,2 s mesmo havendo pares
  a 8 ms. O `pipeline` deduplica a < 22 ms na mesma pista — a mesma lei, lida como número.
* `dup_ms` — mediana intra-pista, critério secundário (lei do aviso "golpes muito próximos" do
  `pipeline`, que usa mediana global < 60 ms).
* `resid_ms` / `offgrid` — resíduo mediano da rejanelagem e fração fora da grade (`report.grid`).
  Sensibilidade baixa espalha onsets fora do pulso; these two are the "the grid likes it" test.
* `densidade` — golpes por segundo. Acima de ~12/s, com 4/4 a 100 BPM, a conta não fecha com
  uma bateria humana (16ºs contínuos são 6,7/s por definição de 4 notas por batida).
* `share_chimel` — fração de golpes que caíram nas pistas de fallback (`hat*`). O classificador
  usa `hat` como rótulo de desempate: quando ele vira a maioria, a leitura está inventando
  chimbal em vez de achar a peça.
* `barras_sem_base`, `kick_por_comp`, `caixa_por_comp` — fração de compassos sem bumbo **nem**
  caixa e densidade por pista de base. É o lado "faltou nota": numa faixa de bateria isolada o
  pulso existe, e se ele sumiu do resultado a sensibilidade está baixa demais. (Uma régua por
  contagem de picos do envelope foi tentada e descartada: com proeminência fixa ela conta 94
  ataques contra 236 do gabarito no demo — chimbel em 16ºs fica abaixo, então a régua acusaria
  dupla detecção em partitura correta. Ela continua em `report.referencia_onsets`, como número
  informativo, sem lei anexada.)
* `erros_audit` — nº de achados `error` do `qa.audit(level="fast")` sobre a partitura daquele
  candidato. É a lei doméstica (T1–T19), não uma métrica nova.

**Empate**: se vários pontos têm custo zero, os critérios sem gabarito não os distinguem. O
empate é resolvido pelo ponto mais próximo da calibração do projeto (`SENS_CALIBRADA` /
`CONF_CALIBRADA` — é onde `merge_ms`, `refine_frac` e os pesos do classificador foram ajustados)
e depois por maior contagem de notas, e o fato de ter havido empate é dito nos `motivos`. Nada é
escolhido por ordem de varredura.

**Custo**: cada célula da grade é uma análise completa — `min_confidence` não é um corte
posterior, ele entra no ajuste da grade (`pipeline` rejanela a grade sobre os eventos que
sobreviveram), então não se pode extrapolar um degrau do outro. Por isso a busca é em duas
etapas com orçamento de células proporcional à duração: varre-se `sensitivity` com a confiança
padrão, e só as melhores sensibilidades descem no eixo da confiança. `celulas` devolvido no
relatório diz quantas análises foram pagas.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import qa as QA
from .audio_io import decode_file
from .pipeline import PARAMS_DEFAULT, transcribe_bytes

#: eixo de sensibilidade varrido (o padrão do projeto é 0.95)
SENS = (0.70, 0.82, 0.95, 1.10, 1.25, 1.45)
CONF = (0.20, 0.30, 0.40, 0.55)
#: mesma lei do aviso do pipeline, só que lida como número
DUP_MIN_MS = 60.0
#: piso de lacuna intra-pista aceito — o `pipeline` deduplica a < 22 ms; abaixo disso é dupla
GAP_PISO_MS = 22.0
#: ponto onde o resto do projeto foi calibrado (refine_frac, merge_ms, pesos do classificador);
#: em empate de custo, prefere-se o candidato mais próximo dele — não é preferência estética, é
#: que as *outras* constantes do pipeline foram ajustadas lá.
SENS_CALIBRADA = float(PARAMS_DEFAULT["sensitivity"])
CONF_CALIBRADA = float(PARAMS_DEFAULT["min_confidence"])
RESID_IDEAL_MS = 12.0
OFFGRID_IDEAL = 0.30
DENS_MAX_POR_SEG = 12.0
DENS_MIN_POR_SEG = 1.0
FALLBACK = ("hat", "hat_open", "hat_foot")


def _medidas(res: dict, dur_s: float) -> dict:
    """Extrai os números do resultado de uma análise. Nenhuma conta nova: só lê o que já foi."""
    sc = res["score"].to_dict()
    rep = res.get("report") or {}
    dets = res.get("detections") or []
    grade = rep.get("grid") or {}
    ts_por_pista: Dict[str, np.ndarray] = {}
    for d in dets:
        ts_por_pista.setdefault(str(d.get("lane")), []).append(float(d.get("time") or 0.0))
    lacunas = [np.diff(np.sort(v)) for v in ts_por_pista.values() if len(v) >= 3]
    lac = np.concatenate(lacunas) if lacunas else np.zeros(0)
    # a mediana intra-pista não serve de régua de duplicata: com kick a 2/compasso a mediana é
    # ~1,2 s mesmo com pares a 8 ms. O que denuncia dupla é o *piso* das lacunas (o próprio
    # `pipeline` deduplica a < 22 ms na mesma pista — mesma lei, lido como número).
    p1_gap = float(np.percentile(lac, 1)) if lac.size else 0.0
    barras = sc.get("bars") or []
    vazias = sum(1 for b in barras if not (b.get("hits") or []))
    n = len(dets)
    achados = 0
    avisos = 0
    try:
        tally = QA.audit(sc, rep, dets=dets, level="fast").get("tally") or {}
        achados = int(tally.get("error") or 0)
        avisos = int(tally.get("warn") or 0)
    except Exception:
        pass                                    # auditoria é reforço; sem ela a busca continua
    confs = np.array([float(d.get("conf") or 0.0) for d in dets], dtype=np.float64)
    ref = int(((rep.get("referencia_onsets") or {}) or {}).get("n") or 0)
    por_comp: Dict[str, float] = {}
    sem_base = 0
    for b_ in barras:
        lns = [str(h.get("lane")) for h in (b_.get("hits") or [])]
        for l_ in set(lns):
            por_comp[l_] = por_comp.get(l_, 0.0) + 1.0
        if "kick" not in lns and "snare" not in lns:
            sem_base += 1
    nb = max(1, len(barras))
    m = {"golpes": n,
            "por_seg": round(n / max(1e-6, dur_s), 3),
            "dup_ms": round(float(np.median(lac)) * 1000.0, 1) if lac.size else 0.0,
            "p1_gap_ms": round(p1_gap * 1000.0, 1),
            "min_gap_ms": round(float(lac.min()) * 1000.0, 1) if lac.size else 0.0,
            "resid_ms": round(float(grade.get("median_residual_ms") or 0.0), 2),
            "offgrid": round(float(grade.get("offgrid_ratio") or 0.0), 3),
            "share_chimel": round(sum(len(v) for k, v in ts_por_pista.items()
                                      if k in FALLBACK) / float(n), 3) if n else 0.0,
            "barras": len(barras),
            "compassos_vazios": round(vazias / float(len(barras)), 3) if barras else 0.0,
            "notas": sum(len(b.get("hits") or []) for b in barras),
            "erros_audit": achados, "avisos_audit": avisos,
            "bpm": round(float(sc.get("bpm") or 0.0), 2),
            "conf_mediana": round(float(np.median(confs)), 3) if confs.size else 0.0,
            "conf_p10": round(float(np.percentile(confs, 10)), 3) if confs.size else 0.0,
            "conf_p25": round(float(np.percentile(confs, 25)), 3) if confs.size else 0.0,
            "ref_onsets": ref,
            "barras_sem_base": round(sem_base / float(nb), 3),
            "kick_por_comp": round(por_comp.get("kick", 0.0) / nb, 3),
            "caixa_por_comp": round(por_comp.get("snare", 0.0) / nb, 3),
            "cobertura": round(n / float(ref), 3) if ref else 0.0}
    return m


def _pontuar(m: dict) -> Tuple[float, List[str]]:
    """Custo adimensional (menor = melhor) + os motivos que pesaram.

    Os pesos são escolhidos para a *ordem* dos critérios, não para um valor absoluto: duplicata é
    o defeito mais caro (destrói a leitura do pulso), virar partitura esparsa vem em seguida
    (engolir nota é pior que inventar nota — é o que a lei de não-destruir-notas-lícitas do
    projeto diz), e o resto é proporcional. Todos os limiares vêm de leis existentes no
    `pipeline`/`qa`; nenhum foi ajustado para um arquivo.
    """
    pen: List[Tuple[str, float]] = []
    # lado "faltou nota": uma bateria pop/rock sem pulso de bumbo+caixa em todo compasso é
    # sub-detecção, e isto não precisa de gabarito — a régua de proeminência do envelope sim
    # (media 94 "ataques fortes" contra 236 golpes do gabarito no demo, porque chimbel em 16ºs
    #  fica abaixo dela), então ela ficou só informativa no relatório e não é critério aqui.
    if m.get("barras"):
        if m["barras_sem_base"] > 0.15:
            pen.append(("%.0f%% dos compassos sem bumbo nem caixa" % (100 * m["barras_sem_base"]),
                        2.6 * min(3.0, (m["barras_sem_base"] - 0.15) / 0.15)))
        if m.get("kick_por_comp", 9.0) < 0.9:
            pen.append(("bumbo esparso (%.2f por compasso)" % m["kick_por_comp"],
                        1.8 * min(3.0, (0.9 - m["kick_por_comp"]) / 0.9)))
        if m.get("caixa_por_comp", 9.0) < 0.5:
            pen.append(("caixa esparsa (%.2f por compasso)" % m["caixa_por_comp"],
                        1.8 * min(3.0, (0.5 - m["caixa_por_comp"]) / 0.5)))
    if m.get("p1_gap_ms", 999.0) < GAP_PISO_MS:
        pen.append(("%.0f%% das lacunas intra-pista abaixo de %.0f ms (dupla detecção)"
                    % (100.0, GAP_PISO_MS), 3.0 * min(3.0, (GAP_PISO_MS - m["p1_gap_ms"]) / GAP_PISO_MS)))
    if m["dup_ms"] < 40.0:
        pen.append(("mediana de %.0f ms entre golpes da mesma pista" % m["dup_ms"],
                    1.5 * min(3.0, (40.0 - m["dup_ms"]) / 40.0)))
    if m["resid_ms"] > RESID_IDEAL_MS:
        pen.append(("resíduo de rejanelagem %.1f ms" % m["resid_ms"],
                    1.5 * min(3.0, (m["resid_ms"] - RESID_IDEAL_MS) / RESID_IDEAL_MS)))
    if m["offgrid"] > OFFGRID_IDEAL:
        pen.append(("%.0f%% dos golpes fora da grade" % (100 * m["offgrid"]),
                    2.0 * min(3.0, (m["offgrid"] - OFFGRID_IDEAL) / OFFGRID_IDEAL)))
    if m["por_seg"] > DENS_MAX_POR_SEG:
        pen.append(("densidade alta (%.1f golpes/s)" % m["por_seg"],
                    1.2 * min(3.0, (m["por_seg"] - DENS_MAX_POR_SEG) / DENS_MAX_POR_SEG)))
    elif m["por_seg"] < DENS_MIN_POR_SEG:
        pen.append(("densidade baixa (%.1f golpes/s)" % m["por_seg"],
                    1.2 * (DENS_MIN_POR_SEG - m["por_seg"]) / DENS_MIN_POR_SEG))
    if m["share_chimel"] > 0.55:
        pen.append(("%.0f%% das notas caíram no rótulo de desempate (chimbal)"
                    % (100 * m["share_chimel"]), 1.5 * min(3.0, (m["share_chimel"] - 0.55) / 0.30)))
    if m["compassos_vazios"] > 0.25:
        pen.append(("%.0f%% dos compassos ficaram sem nota" % (100 * m["compassos_vazios"]),
                    1.0 * min(3.0, (m["compassos_vazios"] - 0.25) / 0.25)))
    if m["erros_audit"]:
        pen.append(("%d erro(s) de auditoria" % m["erros_audit"], 0.45 * m["erros_audit"]))
    return round(sum(p for _, p in pen), 4), [k for k, _ in pen]


def _orcamento_celulas(dur_s: float) -> Tuple[int, int]:
    """(nº de sensibilidades na varredura, nº de confianças no refinamento).

    O teto é trabalho, não estética: cada célula custa uma análise completa da faixa.
    """
    if dur_s <= 70:
        return len(SENS), len(CONF)
    if dur_s <= 200:
        return len(SENS), 3
    if dur_s <= 420:
        return 5, 2
    return 3, 2


def recomendar_bytes(dados: bytes, filename: str = "audio.wav", *,
                     params: Optional[dict] = None,
                     sens: Optional[Sequence[float]] = None,
                     conf: Optional[Sequence[float]] = None, topo: int = 2) -> dict:
    """Mesma coisa a partir de bytes em memória (usado nos testes e em caminhos sem arquivo)."""
    # o parêntese é obrigatório: "...%s.wav" % x % y formata primeiro e depois tenta
    # "%" numa string (TypeError). Vício de `%`-format que já mordeu neste projeto.
    caminho = os.path.join("/tmp", "drumscribe-tune-%s.wav"
                           % (abs(hash((filename, len(dados)))) % 10 ** 9))
    with open(caminho, "wb") as f:
        f.write(dados)
    try:
        return recomendar_arquivo(caminho, params=params, sens=sens, conf=conf, topo=topo)
    finally:
        try:
            os.remove(caminho)
        except OSError:
            pass


def recomendar_arquivo(caminho: str, params: Optional[dict] = None,
                       sens: Optional[Sequence[float]] = None,
                       conf: Optional[Sequence[float]] = None,
                       topo: int = 2) -> dict:
    """Varredura de `sensitivity` × `min_confidence` com recomendação justificada.

    `topo` = quantas sensibilidades sobrevivem ao refinamento no eixo da confiança.
    """
    base = dict(PARAMS_DEFAULT)
    base.update({k: v for k, v in (params or {}).items() if k in PARAMS_DEFAULT})
    base["onda_fina"] = False                       # o zoom não é parte da varredura
    # Fotografa o ponto vigente ANTES do laço: `base` é mutado durante a varredura, e ler
    # `base["sensitivity"]` depois dela dava "ponto atual" = última célula testada (o veredito
    # "não vale mudar" sairia comparando com o número errado).
    sens_atual, conf_atual = float(base["sensitivity"]), float(base["min_confidence"])
    with open(caminho, "rb") as fh:
        dados = fh.read()
    dur = 0.0
    try:
        dur = float(decode_file(caminho, max_seconds=float(base["max_seconds"]) or None).duration)
    except Exception:
        pass
    grade_s = list(sens or SENS)
    grade_c = list(conf or CONF)
    n_s, n_c = _orcamento_celulas(dur)
    if dur > 70:
        grade_s = grade_s[:n_s]
    grade_c = grade_c[:max(2, n_c)]
    # o ponto vigente entra na grade: sem medi-lo, "já está bom" seria uma afirmação não medida
    for _lista, _chave in ((grade_s, "sensitivity"), (grade_c, "min_confidence")):
        _v = sens_atual if _chave == "sensitivity" else conf_atual
        if all(abs(float(x) - _v) > 1e-6 for x in _lista):
            _lista.append(_v)
            _lista.sort()
    grade_s = tuple(grade_s)
    grade_c = tuple(grade_c)

    # teto declarado de análises: a varredura + o ponto vigente descendo o eixo da confiança +
    # as duas verificações. É o número que o `referencia.orcamento` devolve e que o teste confere:
    # prometer orçamento e gastar mais é a mesma falta de honestidade de prometer resolução e não
    # ter gravado o envelope fino.
    teto = len(grade_s) + (max(1, topo) + 1) * max(1, len(grade_c) - 1) + 2
    t0 = time.time()
    células = 0
    linhas: List[dict] = []
    melhor_por_sens: List[Tuple[float, dict, int]] = []
    for sv in grade_s:
        base["sensitivity"] = float(sv)
        base["min_confidence"] = float(min(grade_c))
        res = transcribe_bytes(dados, filename=_nome(caminho), params=dict(base))
        células += 1
        m = _medidas(res, dur or res["report"].get("file", {}).get("duration_sec") or 0.0)
        custo, motivos = _pontuar(m)
        linhas.append({"sensitivity": round(float(sv), 3), "min_confidence": round(float(grade_c[0]), 3),
                       "custo": custo, "motivos": motivos, **m})
        melhor_por_sens.append((custo, {"sensitivity": float(sv), "res": res, "m": m}, células))

    # refinamento: só as melhores sensibilidades descem no eixo da confiança
    melhor_por_sens.sort(key=lambda t: t[0])
    escolhidos = melhor_por_sens[:max(1, topo)]
    # …e a sensibilidade *vigente* desce junto, mesmo fora dos dois primeiros: sem isso, o
    # "nada a mudar" seria comparado numa célula medida de passagem (1 confiança) contra candidatos
    # que desceram o eixo inteiro. Gasto extra só cabe se sobrar orçamento.
    if all(abs(ctx["sensitivity"] - sens_atual) > 1e-6 for _, ctx, _ in escolhidos):
        atual_sens = next((e for e in melhor_por_sens
                           if abs(e[1]["sensitivity"] - sens_atual) < 1e-6), None)
        # usa o `teto` declarado acima (uma segunda conta aqui sombreada a primeira fazia o
        # relatório prometer 14 e gastar 15 — conferido em tests/tune_check.py)
        if atual_sens is not None and células + len(grade_c) - 1 + 1 <= teto:
            escolhidos = escolhidos + [atual_sens]
    for _, ctx, _ in escolhidos:
        for cv in grade_c[1:]:
            par = dict(base)
            par["sensitivity"] = ctx["sensitivity"]
            par["min_confidence"] = float(cv)
            res = transcribe_bytes(dados, filename=_nome(caminho), params=par)
            células += 1
            m = _medidas(res, dur or res["report"].get("file", {}).get("duration_sec") or 0.0)
            custo, motivos = _pontuar(m)
            linhas.append({"sensitivity": round(ctx["sensitivity"], 3), "min_confidence": round(float(cv), 3),
                           "custo": custo, "motivos": motivos, **m})

    linhas.sort(key=_ordem)
    melhor = linhas[0]
    empatados = [r for r in linhas if r["custo"] <= melhor["custo"] + 1e-9]
    _sa, _ca = sens_atual, conf_atual
    atual = next((r for r in linhas if abs(r["sensitivity"] - _sa) < 1e-6
                  and abs(r["min_confidence"] - _ca) < 1e-6), None)
    if atual is None:                    # o par vigente não caiu na grade: mede ele à parte
        par = dict(base)
        par["sensitivity"], par["min_confidence"] = _sa, _ca
        _res = transcribe_bytes(dados, filename=_nome(caminho), params=par)
        células += 1
        _mm = _medidas(_res, dur or _res["report"].get("file", {}).get("duration_sec") or 0.0)
        _c, _mo = _pontuar(_mm)
        atual = {"sensitivity": round(_sa, 3), "min_confidence": round(_ca, 3), "custo": _c,
                 "motivos": _mo, **_mm}
        linhas.append(atual)
        linhas.sort(key=_ordem)
        melhor = linhas[0]
    verif = None
    if células + 1 > teto:                        # sem espaço para a verificação: pule, não minta
        verif = {"custo": melhor["custo"], "motivos": list(melhor["motivos"]),
                 "pulado": True, **melhor}
    # verificação fim-a-fim: o ponto recomendado é rodado de novo, com a auditoria do resultado
    if verif is None:
        par = dict(base)
        par["sensitivity"] = float(melhor["sensitivity"])
        par["min_confidence"] = float(melhor["min_confidence"])
        res = transcribe_bytes(dados, filename=_nome(caminho), params=par)
        células += 1
        mv = _medidas(res, dur or res["report"].get("file", {}).get("duration_sec") or 0.0)
        _vc, _vm = _pontuar(mv)
        verif = {"custo": _vc, "motivos": _vm, **mv}

    motivos_final = _motivos(melhor, atual, células, time.time() - t0, grade_s, grade_c, dur,
                             len(empatados))
    return {"ok": True,
            "recomendado": {"sensitivity": melhor["sensitivity"],
                            "min_confidence": melhor["min_confidence"]},
            "custo_recomendado": melhor["custo"],
            "custo_atual": None if atual is None else atual["custo"],
            "motivos": motivos_final,
            "candidatos": linhas,
            "verificacao": verif,
            "referencia": {"duracao_s": round(dur, 2), "celulas": células,
                           "grade_sens": [round(float(s), 3) for s in grade_s],
                           "grade_conf": [round(float(c), 3) for c in grade_c],
                           "atual": {"sensitivity": sens_atual, "min_confidence": conf_atual,
                                     "fonte": "par usado nesta análise"},
                           "orcamento": {"sens": int(n_s), "conf": int(n_c),
                                         "por_seg": (dur or 0.0) > 70, "teto": int(teto)},
                           "melhor_atual": atual, "segundos": round(time.time() - t0, 1),
                           "leis": {"dup_ms_min": DUP_MIN_MS, "resid_ms_ideal": RESID_IDEAL_MS,
                                    "offgrid_max": OFFGRID_IDEAL, "densidade_max": DENS_MAX_POR_SEG,
                                    "share_chimel_max": 0.55}}}


__all__ = ["recomendar_arquivo", "recomendar_bytes", "SENS", "CONF"]


def _ordem(r: dict) -> Tuple[float, float, int]:
    """Custo primeiro; empate resolvido pela calibração do projeto e depois por recall maior.

    Deixar `sorted` decidir o empate por ordem de varredura seria recomendação por acaso.
    """
    desvio = (abs(r["sensitivity"] - SENS_CALIBRADA) / max(1e-6, SENS_CALIBRADA)
              + abs(r["min_confidence"] - CONF_CALIBRADA) / max(1e-6, CONF_CALIBRADA))
    return (r["custo"], round(desvio, 4), -int(r["golpes"]))


def _nome(caminho: str) -> str:
    import os
    return os.path.basename(str(caminho or "faixa"))


def _motivos(melhor: dict, atual: Optional[dict], células: int, seg: float,
             grade_s, grade_c, dur: float, empatados: int = 1) -> List[str]:
    out = []
    if empatados > 1:
        # vem primeiro de propósito: é justamente no caso "nada a mudar" que o usuário tem de saber
        # que houve empate entre N pontos e que a escolha foi por calibração, não por medida.
        out.append("%d dos %d pontos testados ficaram com custo zero; entre eles foi escolhido "
                   "o mais próximo da calibração do projeto (sens %.2f / conf %.2f) — os critérios "
                   "sem gabarito não distinguem esses pontos, então a escolha é declarada, não "
                   "escondida." % (empatados, len(grade_s) * len(grade_c), SENS_CALIBRADA,
                                   CONF_CALIBRADA))
    if atual:
        d = atual["custo"] - melhor["custo"]
        if d <= 0.001:
            out.append("os valores padrão (sensibilidade %.2f, confiança %.2f) já são o melhor "
                       "ponto medido nesta varredura — nada a mudar."
                       % (atual["sensitivity"], atual["min_confidence"]))
            return out
        out.append("o ponto atual (sens %.2f · conf %.2f) custa %.2f de penalidade; o recomendado "
                   "(sens %.2f · conf %.2f) custa %.2f — %.0f%% a menos."
                   % (atual["sensitivity"], atual["min_confidence"], atual["custo"],
                      melhor["sensitivity"], melhor["min_confidence"], melhor["custo"],
                      100.0 * d / max(1e-6, atual["custo"])))
    if melhor["motivos"]:
        out.append("o que ainda limita este ponto: " + "; ".join(melhor["motivos"]) + ".")
    else:
        out.append("nenhum critério ficou abaixo do aceitável neste ponto (duplicatas, grade, "
                   "densidade, desempate e auditoria todos dentro da lei).")
    out.append("medido: %d golpes em %.0f s (%.1f/s), mediana de %.0f ms entre golpes da mesma "
               "pista, resíduo %.1f ms, %.0f%% fora da grade, %d/%d compassos, %d nota(s) escrita(s), "
               "%d erro(s) de auditoria."
               % (melhor["golpes"], dur, melhor["por_seg"], melhor["dup_ms"], melhor["resid_ms"],
                  100 * melhor["offgrid"], melhor["barras"] - melhor["compassos_vazios"] *
                  melhor["barras"], melhor["barras"], melhor["notas"], melhor["erros_audit"]))
    out.append("Custo pago: %d análises completas em %.0f s. A busca varreu sensibilidade de "
               "%.2f a %.2f e confiança de %.2f a %.2f; fora disso não há recomendação — o que "
               "existe é o direito de pedir."
               % (células, seg, min(grade_s), max(grade_s), min(grade_c), max(grade_c)))
    out.append("Depois de aplicar, confira na aba Áudio & batidas com zoom: a recomendação é "
               "feita com critérios sem gabarito, e nenhum critério sem gabarito substitui "
               "ouvir o resultado batendo com a partitura.")
    return out
