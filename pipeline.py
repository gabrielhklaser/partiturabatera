"""
Pipeline: arquivo de áudio (bateria isolada ou mix) → Score + relatório auditável.

Fluxo:

  decodificação → STFT → novelty por grupo de bandas → picos independentes por grupo
  → refinamento do ataque → features espectrais por evento → classificação **por grupo**
  (bumbo/tens, caixa/aro, pratos) → dedupe e flams → andamento (ACF + prior log-normal
  + pente) → regressão robusta tick→tempo → métrica/downbeat por template de backbeat
  → swing → quantização hierárquica (8º/16º/32º/tercinas) → dinâmica por peça
  (acentos/ghost notes) → durações e sustentações (rules.fill_voice) → partitura.

Decisão de arquitetura importante: eventos de bandas diferentes que coincidem no tempo
formam um **acorde** (bumbo + chimel no mesmo pulso), e não um único golpe classificado.
Fundir tudo em um evento faz uma peça "roubar" a outra — o erro mais comum de
transcritores de bateria "de uma etiqueta só".
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import classify as C
from . import dsp
from . import grid as G
from .audio_io import (decode_bytes, normalize_for_analysis, peaks_envelope,
                       peaks_finos, planes_finos, resample_to, trim_silence)
from .kit import BANDS, DEFAULT_LANES, LANE_BAND_WEIGHTS, LANE_BY_ID, PRATOS
from .score_model import Bar, Hit, Score, TICKS_PER_QUARTER, meter_info

PARAMS_DEFAULT = {
    "sensitivity": 0.95,         # 0.5 (conservador) .. 1.8 (agressivo)
    "grid_mode": "auto",         # auto|8th|16th|32nd+16th|triplet|16th+triplet
    "bpm_hint": None,
    "bpm_lock": None,
    "min_bpm": 45.0,
    "max_bpm": 220.0,
    "meter": "auto",
    "swing": "auto",             # auto|0|0.55..0.72
    "downbeat_shift_ticks": 0,
    "n_fft": 1024,
    "hop": 256,
    "lanes": DEFAULT_LANES,
    "dyn_mode": "blend",         # global|per_lane|blend
    "min_confidence": 0.30,
    "ghost_notes": True,
    "accents": True,
    "gate": True,
    "refine_frac": 0.62,
    "merge_ms": 13.0,
    "aux_lanes": False,          # reconhece pedal/cross-stick/cowbell/splash (mais FP)
    "fill_detect": True,
    "repeat_slash": True,        # compassos idênticos viram 𝄄 na gravura (dados preservados)
    "analysis_sr": 44100,
    "simples": False,              # partitura simples: só bumbo, caixa e tons na gravura
    "hide_lanes": None,            # lista explícita de pistas ocultas (sobrepõe `simples`)
    "onda_fina": True,             # envelope + novidade em alta resolução, para o zoom da onda
    "max_seconds": 60 * 10,
    "memory_budget_mb": None,          # None = 75 % da RAM disponível nesta máquina
}

# Cada grupo de bandas só pode ser atribuído a estas pistas.
# Cada grupo de bandas só pode ser atribuído a estas pistas. As pistas "auxiliares"
# (pedal de chimel, cowbell, splash, cross-stick) têm assinatura muito próxima de
# vizinhos comuns e por isso só entram se o usuário as ligar explicitamente.
GROUP_LANES: Dict[str, List[str]] = {
    "low": ["kick", "tom_low", "tom_mid", "tom_hi"],
    "mid": ["snare", "tom_mid", "tom_hi", "tom_low"],
    "high": ["hat", "hat_open", "ride", "crash"],
}
AUX_LANES: Dict[str, List[str]] = {"mid": ["rim"], "high": ["hat_foot", "cowbell", "splash"]}
# Rótulo de recurso quando nenhum portão passa mas o grupo tem evidência real de ataque.
GROUP_FALLBACK: Dict[str, str] = {"low": "kick", "mid": "snare", "high": "hat"}
GROUP_FALLBACK_ST: Dict[str, float] = {"low": 0.30, "mid": 0.26, "high": 0.18}


def _slice_feat(times: np.ndarray, feat, idx: np.ndarray):
    idx = np.asarray(idx, dtype=np.int64)
    kw = dict(n=int(idx.size), times=times[idx], band_db=feat.band_db[idx],
              band_rel=feat.band_rel[idx], centroid=feat.centroid[idx],
              rolloff=feat.rolloff[idx], flatness=feat.flatness[idx],
              pitch_low=feat.pitch_low[idx], attack_ms=feat.attack_ms[idx],
              decay_low_ms=feat.decay_low_ms[idx], decay_hi_ms=feat.decay_hi_ms[idx],
              rise_ratio=feat.rise_ratio[idx], loud=feat.loud[idx],
              hi_drop_db=None if feat.hi_drop_db is None else feat.hi_drop_db[idx],
              low_drop_db=None if feat.low_drop_db is None else feat.low_drop_db[idx],
              win_ms=None if feat.win_ms is None else feat.win_ms[idx],
              hi_slope=None if feat.hi_slope is None else feat.hi_slope[idx],
              mid_slope=None if getattr(feat, "mid_slope", None) is None else feat.mid_slope[idx],
              hf_centroid=None if getattr(feat, "hf_centroid", None) is None else feat.hf_centroid[idx],
              sub_ratio=None if getattr(feat, "sub_ratio", None) is None else feat.sub_ratio[idx],
              band_delta_db=None if getattr(feat, "band_delta_db", None) is None else feat.band_delta_db[idx],
              band_delta_rel=None if getattr(feat, "band_delta_rel", None) is None else feat.band_delta_rel[idx],
              low_slope=None if feat.low_slope is None else feat.low_slope[idx],
              band_peak_db=None if feat.band_peak_db is None else feat.band_peak_db[idx],
              novelty=None if feat.novelty is None else feat.novelty[idx],
              band_names=feat.band_names)
    return times[idx], dsp.OnsetFeatures(**kw)


def _smooth_cymbals(hits: List[dict], spb: float) -> List[dict]:
    """
    Suavização por coerência de padrão (filtro de maioria com contexto rítmico) para o
    grupo agudo, onde chimel/chimel-aberto/ride/prato se confundem por natureza.

    Justificativa: em uma bateria real o padrão de mão direita é estacionário — se 8 de 10
    eventos agudos consecutivos caem em múltiplos do beat e são 'hat', um 'crash' isolado
    no meio do padrão quase certamente é um chimel mal classificado (e vice-versa: crash
    legítimo aparece tipicamente 1× por seção, no downbeat). É um prior de 1ª ordem barato
    e honesto, não uma pretensão de HMM completo.
    """
    hi = [h for h in hits if h["group"] == "high"]
    if len(hi) < 6:
        return hits
    hi = sorted(hi, key=lambda h: h["time"])
    n = len(hi)
    for i, h in enumerate(hi):
        lo = max(0, i - 4)
        hiw = min(n, i + 5)
        ctx = hi[lo:hiw]
        near = [c for c in ctx if c is not h and abs(c["time"] - h["time"]) < 2.6 * spb]
        if len(near) < 3:
            continue
        counts: Dict[str, int] = {}
        for c in near:
            counts[c["lane"]] = counts.get(c["lane"], 0) + 1
        maj, cnt = max(counts.items(), key=lambda kv: kv[1])
        frac = cnt / float(len(near))
        # só rebaixa os rótulos instáveis (crash / prato aberto) para o padrão vizinho;
        # ride e chimel são decididos pela física (sustain), não pela maioria.
        if h["lane"] in ("crash", "hat_open") and frac >= 0.60 and maj in ("hat", "ride"):
            h["lane"] = maj
            h["conf"] = min(h["conf"], 0.45)
            h["smoothed"] = 1
    return hits


def _context(feat, times: np.ndarray, bpm: float, phase: float, spb: float) -> Dict[str, np.ndarray]:
    """
    Priors estruturais por evento (assumindo 4 por compasso — só um palpite suave, não
    uma decisão): downbeat, densidade local e nível relativo. Servem para separar
    crash (ataque isolado no início do compasso) de chimel (padrão denso e repetido).
    """
    n = times.size
    out = {"downbeat": np.zeros(n, np.float32), "dense": np.zeros(n, np.float32),
           "loud": np.zeros(n, np.float32)}
    if n == 0:
        return out
    spt = (60.0 / max(1e-6, bpm)) / TICKS_PER_QUARTER
    pos = np.rint((times - phase) / spt)
    tib = np.mod(pos, 32)
    out["downbeat"] = ((tib <= 1.0) | (tib >= 31.0)).astype(np.float32)
    for i in range(n):
        k = int(np.sum(np.abs(times - times[i]) <= 0.10)) - 1
        out["dense"][i] = float(np.clip(k / 4.0, 0.0, 1.0))
    if feat.loud is not None and int(np.asarray(feat.loud).size) == n:
        med = float(np.median(feat.loud))
        sd = float(np.std(feat.loud)) + 1e-6
        out["loud"] = np.clip((feat.loud - med) / (2.0 * sd), 0.0, 1.0).astype(np.float32)
    return out


def _lane_energy(feat, lanes: Sequence[str]) -> np.ndarray:
    """
    Energia (dB) relevante por pista: banda dominante da peça, suavizada com a energia
    total — dá dinâmica coerente quando bumbo e chimel tocam juntos.
    """
    n = feat.n
    out = np.zeros(n, dtype=np.float64)
    names = feat.band_names
    for i, ln in enumerate(lanes):
        b = LANE_BY_ID[ln].band if ln in LANE_BY_ID else "mid"
        j = names.index(b) if b in names else 0
        pk = (float(feat.band_peak_db[i, j]) if feat.band_peak_db is not None
              else float(feat.band_db[i, j]))
        dl = (float(feat.band_delta_db[i, j]) if getattr(feat, "band_delta_db", None) is not None else 0.0)
        out[i] = 0.40 * pk + 0.30 * float(feat.band_db[i, j]) + 0.30 * (float(feat.loud[i]) - max(0.0, dl) * 0.0 + dl)
    return out


class MemoriaInsuficiente(ValueError):
    """A faixa pedida não cabe na memória desta máquina nem no plano mais econômico.

    Existe para o oposto de um defeito: sem esta exceção o processo estourava o limite do kernel e
    era morto (SIGKILL) — a plataforma inteira caía no meio do upload do usuário. Aqui a análise
    recusa *antes* de alocar, com número e sugestão. Ver docs/ERROS.md A13.
    """


def mem_disponivel_mb() -> float:
    """RAM realmente disponível (MemAvailable), 0.0 quando não sabemos perguntar."""
    try:
        with open("/proc/meminfo", "r") as f:
            for ln in f:
                if ln.startswith("MemAvailable:"):
                    return float(ln.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


# Coeficientes do custo de memória, AJUSTADOS POR MÍNIMOS QUADRADOS sobre picos de RSS medidos
# processo por processo (tests/profile_long.py; base de import ~100 MB subtraída):
#   41,9 s @44100/256 → 278 MB   ·   83,8 s @44100/256 → 543 MB   ·   167,6 s @22050/512 → 561 MB
# O termo que manda é o de amostras de áudio (a envoltória de Hilbert do grave e o caminho do
# filtro andam na taxa de amostragem, e o custo por quadro×bin da grade é menor que 10 % disso).
# Deixamos 60 B/amostra contra os ~58 medidos e somamos a base do interpretador: fica ~15 %
# acima do medido nos três pontos, que é a margem de segurança do recuso-antes-de-alocar.
MB_POR_QUADRO_BIN = 4.0e-6          # ≈ 4 B por (quadro × bin): grade complex64 + fluxos por banda
MB_POR_AMOSTRA = 60.0 / 1e6         # ≈ 60 B por amostra de áudio na taxa de análise
MB_BASE_INTERPRETADOR = 140.0       # numpy/scipy/flask já mapeados antes da primeira alocação


def _fator_bluestein(n: int) -> float:
    """Penalidade de FFT para comprimentos com fator grande.

    Medido nesta máquina (docs/ERROS.md A13): o `hilbert` da envoltória do grave faz um FFT de
    comprimento igual ao nº de amostras; quando esse comprimento tem um fator maior que ~13, o
    pocketfft cai em Bluestein e aloca ~2,5× o tamanho em buffers de trabalho — 1,5 s e +444 MB
    contra 0,2 s e +55 MB para um comprimento "liso" vizinho. Um custo que depende do tamanho do
    arquivo *e* da sua fatoração não pode ser ignorado num orçamento de memória.
    """
    if n < 2:
        return 1.0
    m = int(n)
    for f in (2, 3, 5, 7, 11, 13):
        while m % f == 0:
            m //= f
    return 1.0 if m == 1 else 2.6


def custo_analise_mb(n_amostras: int, sr: int, hop: int, n_fft: int) -> float:
    """Pico estimado da análise, em MB — o número que decide entre “partitura” e “recusa”.

    Calibrado por medição nesta máquina: a inclinação marginal do pico em função do nº de
    amostras (44 100 Hz, hop 256) deu 61 B/amostra em 60→120 s, e o caminho completo de uma
    faixa de 443 s a 22 050 Hz mediu 718 MB de pico contra 804 MB previstos aqui — o modelo é
    conservador em ~12 %, que é a margem que se quer antes de um SIGKILL.

    A versão anterior multiplicava o termo por amostra por `_fator_bluestein`. **Rejeitado por
    medida (A19)**: com o caminho da envoltória reescrito em float32/complex64 (A13), o caso
    “comprimento com fator grande” passou a custar *tempo*, não memória — medido no único FFT de
    comprimento O(n) que sobrou (`energy_rise`): 2,7 s e +50 MB no comprimento áspero contra
    0,4 s e +179 MB no comprimento liso. Cobrar 2,6× a memória inteira por causa disso recusava
    músicas normais (uma faixa de 7min23s, 17,7 MB, dava 413 com “faltam 718 MB”). O fator fica
    onde ele dói de verdade: no prazo e no aviso de lentidão.
    """
    nfr = 1 + n_amostras // max(1, hop)
    nb = n_fft // 2 + 1
    return (MB_BASE_INTERPRETADOR
            + n_amostras * MB_POR_AMOSTRA
            + nfr * nb * MB_POR_QUADRO_BIN)


#: Máximo espaçamento de quadros que ainda grava o instante de uma nota no lugar.
#: Medido em `samples/demo_drums.wav` com gabarito: 5.80 ms → `pos_accuracy` 1.000 e
#: `tick_mae` 0.00; 11.61 ms → `tick_mae` −1.00 (toda nota um tick antes) e 23.2 ms chega a
#: engolir um compasso. Não é palpite de limiar: é a resolução abaixo da qual a rejanelagem
#: do `refine_frac` deixa de ter amostras suficientes para posicionar o ataque (A14).
MS_MAX_POR_QUADRO = 6.0

#: `hop`/`n_fft` são calibrados nesta taxa; quem pedir outra taxa recebe a mesma grade física.
SR_REF = 44100

#: Piso da escada automática de memória. Não é "o mínimo que roda" — é o mínimo que ainda
#: **ouve o kit**: abaixo de 22 050 Hz a banda `vhigh` (7–16 kHz) inteira cai acima de
#: Nyquist, e ela é a principal evidência de chimetal/crash (`hat` tem peso 3.0 nela, ver
#: `kit.LANE_BAND_WEIGHTS`). Uma escada que cala uma pista não é degradação de fidelidade, é
#: mutilação de entrada; isso só pode ser escolha do usuário, com aviso (`analysis_sr`).
SR_MINIMO_ESCADA = 22050


def plano_dsp(n_amostras: int, sr: int, hop0: int, n_fft0: int, orcamento_mb: float):
    """Escada de planos (taxa, hop, n_fft) do mais fiel ao mais econômico + custo de cada um.

    A escada só mexe na **taxa de amostragem**, e escala `hop` e `n_fft` na mesma proporção:
    assim o espaçamento de quadros em ms e a largura de bin em Hz ficam **idênticos** aos do
    plano pedido, e a única coisa sacrificada é a metade superior do espectro (Nyquist). O que
    se perde é audível e declarável; o que *não* se pode perder é o instante de uma nota.

    A versão anterior desta função também engrossava `hop` (512, 1024) achando que isso
    barateava o custo por quadro. Medido, o termo dominante é por amostra (não por quadro), e
    o hop grosso corrompia a gravura: `tick_mae` ia a −1.00 com 11.6 ms entre quadros
    (docs/ERROS.md A14). Por isso nenhum plano da escada muda a resolução temporal.
    """
    def custo_de(st: int, h: int, nf: int) -> float:
        n = int(round(n_amostras * st / sr))
        return custo_analise_mb(n, st, h, nf)

    # O degrau 0 é sempre o pedido do usuário, inclusive se ele pedir algo abaixo do piso da
    # escada: o piso limita o que a máquina pode *escolher sozinha*, não o que ela pode *fazer*
    # quando mandaram. Um plano vazio aqui era um IndexError no lugar de uma partitura.
    planos: List[tuple] = [(int(sr), int(hop0), int(n_fft0))]
    for f in (2, 4, 8, 16):
        st = int(round(sr / f))
        if st < SR_MINIMO_ESCADA:
            break
        h = max(16, int(round(hop0 / f)))
        nf = max(256, int(round(n_fft0 / f)))
        if nf * f < n_fft0:                  # n_fft encostou no piso: a grade espectral mudaria
            break
        if (st, h, nf) not in planos:
            planos.append((st, h, nf))
    for (st, h, nf) in planos:
        c = custo_de(st, h, nf)
        if c <= orcamento_mb:
            return st, h, nf, c, [(a, b, cc, custo_de(a, b, cc)) for (a, b, cc) in planos]
    st, h, nf = planos[-1]
    raise MemoriaInsuficiente(
        "a faixa é grande demais para a memória desta máquina: o plano mais econômico "
        "(%d Hz, hop %d, n_fft %d) ainda pede %.0f MB contra %.0f MB disponíveis. Divida o "
        "arquivo (a partitura é escrita por trecho) ou rode com mais RAM; um `analysis_sr` "
        "menor também ajuda — reduzir `hop` não economiza nada e desloca os ticks. Para "
        "assumir o risco de estourar mesmo assim, suba `memory_budget_mb` (agora %.0f)."
        % (st, h, nf, custo_de(st, h, nf), orcamento_mb, orcamento_mb))


# ======================================================================================
# entrada principal
# ======================================================================================

def transcribe_bytes(data: bytes, filename: str = "track", params: Optional[dict] = None) -> dict:
    p = dict(PARAMS_DEFAULT)
    p.update({k: v for k, v in (params or {}).items() if k in PARAMS_DEFAULT})
    t0 = time.time()
    tim: Dict[str, float] = {}
    warn: List[str] = []

    # ------------------------------------------------------------------- 1. decodificação
    t = time.time()
    audio = decode_bytes(data, filename=filename, max_seconds=float(p["max_seconds"]),
                         analysis_sr=int(p["analysis_sr"]))
    x_raw = audio.x
    x, lead = trim_silence(x_raw, audio.sr, thresh_db=-46.0, pad=0.12)
    if x.size < audio.sr * 1.2:
        x, lead = x_raw, 0
    sr = audio.sr
    tim["decode"] = time.time() - t

    # --------------------------------------------------------------------- 2. STFT + picos
    t = time.time()
    # taxa de análise escolhida pelo usuário: `hop`/`n_fft` são calibrados em `SR_REF`, então
    # quem pede outra taxa recebe a *mesma* grade física (ms/quadro e Hz/bin), só com menos
    # Nyquist — não uma grade mais grossa.
    n_fft = max(256, int(round(int(p["n_fft"]) * sr / SR_REF)))
    hop = max(16, int(round(int(p["hop"]) * sr / SR_REF)))
    # (regra removida: `if dur > 210 s: hop = 512` economizava ~0 MB e deslocava todo tick da
    #  partitura — ver docs/ERROS.md A14. A memória de uma faixa longa se resolve por taxa.)
    # Orçamento de memória, decidido ANTES de alocar (docs/ERROS.md A13): a faixa do usuário é
    # arbitrária, e estourar a RAM não produz um erro — produz um SIGKILL no processo que atende
    # todo mundo. O plano escolhe a taxa/hop mais fiel que cabe; se nenhum couber, recusa.
    # 95 % do que o kernel reporta como disponível — e não 75 %: o número já desconta o que está
    # em uso, e o trabalho pesado agora roda em processo filho isolado, então ficar no limite
    # custa uma recusa legível, não a plataforma inteira. Ver `_analise_isolada` em server.py.
    orcamento = float(p.get("memory_budget_mb") or 0.0)
    if orcamento <= 0:
        disponivel = mem_disponivel_mb()
        orcamento = max(256.0, 0.95 * disponivel) if disponivel else 1024.0
    lead_s = lead / float(sr)
    sr_dsp, hop, n_fft, custo_mb, escada = plano_dsp(x.size, sr, hop, n_fft, orcamento)
    if sr_dsp != sr:
        x, sr = resample_to(x, sr, sr_dsp)
        x_raw = x
        lead = int(round(lead_s * sr))
    custo_mb = custo_analise_mb(x.size, sr, hop, n_fft)
    # O fator de Bluestein não move mais a memória (ver `custo_analise_mb`); move o relógio. Vale
    # declará-lo, porque "a análise está lenta sem motivo" é exatamente o tipo de coisa que faz o
    # usuário achar que o arquivo não foi lido.
    f_fft = _fator_bluestein(x.size)
    ajustado = (sr_dsp != audio.sr) or (hop != int(p["hop"]))
    ms_quadro = 1000.0 * hop / sr
    if ms_quadro > MS_MAX_POR_QUADRO:
        warn.append("espaçamento de quadros de %.2f ms (hop %d a %d Hz) é grosseiro para a "
                    "colocação dos ticks: medido, a partitura sai com notas um tick adiantadas. "
                    "Use hop ≤ %d nesta taxa (ou deixe o plano de memória decidir)."
                    % (ms_quadro, hop, sr, int(MS_MAX_POR_QUADRO * sr / 1000.0)))
    mudas = sorted({k for (k, (f0, f1)) in BANDS.items() if f0 >= sr / 2.0})
    if mudas:
        pistas = sorted({ln for ln, w in LANE_BAND_WEIGHTS.items()
                         if any(abs(float(w.get(k, 0.0))) >= 1.0 for k in mudas)})
        warn.append("com a análise em %d Hz as bandas %s ficam acima de Nyquist (%.1f kHz) e não "
                    "há evidência espectral nelas — as pistas %s perdem a principal pista de "
                    "identificação. É escolha sua (parâmetro `analysis_sr`); o plano automático "
                    "nunca desce até aqui — ele para em %d Hz."
                    % (sr, "·".join(mudas), sr / 2000.0, ", ".join(pistas) or "nenhuma",
                       SR_MINIMO_ESCADA))
    if f_fft > 1.0 and x.size > 4 * 1024 * 1024:
        # Não é um erro e não custa memória (A19): é o único efeito real de um comprimento com
        # fator grande. Dito aqui, "a análise demorou" deixa de ser mistério.
        warn.append("a faixa tem %d amostras, cujo maior fator primo é > 13: o FFT da envoltória "
                    "do grave cai no caminho Bluestein e leva ~%.1f× mais tempo (medido nesta "
                    "máquina: 2,7 s contra 0,4 s por 4,4 M de amostras). Isso não muda a memória "
                    "necessária nem a partitura." % (x.size, f_fft))
    if ajustado:
        warn.append("análise rebaixada para caber na memória desta máquina: %d Hz (Nyquist "
                    "%.1f kHz), hop %d — a grade temporal não mudou (%.2f ms/quadro, %.2f Hz/bin). "
                    "Sons acima de %.1f kHz não foram ouvidos."
                    % (sr, sr / 2000.0, hop, ms_quadro, sr / float(n_fft), sr / 2000.0))
    if audio.truncated_sec:
        warn.append("o arquivo tem mais que o limite aceito: %d s do fim foram descartados e a "
                    "partitura cobre só %.1f s. Envie em partes (a escrita é por trecho)."
                    % (round(audio.truncated_sec), audio.duration))
    xa = normalize_for_analysis(x, -20.0)
    spec = dsp.stft(xa, sr, n_fft=n_fft, hop=hop)
    det = dsp.detect_events(spec, xa, sr, sens=float(p["sensitivity"]),
                            merge_ms=float(p["merge_ms"]), refine_frac=float(p["refine_frac"]))
    tim["onsets"] = time.time() - t
    t_rel = np.asarray(det["times"], dtype=np.float64)
    offset_s = lead / float(sr)
    if t_rel.size == 0:
        warn.append("Nenhum transiente encontrado. Aumente a sensibilidade e confirme que o "
                    "arquivo contém áudio audível.")
        return {"score": Score(), "report": {"warnings": warn, "n_events": 0,
                                             "file": {"name": filename},
                                             "params": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                                                        for k, v in p.items()}},
                "detections": [], "wave": peaks_envelope(x_raw, sr, 1200), "timings": tim,
                "fine": None}

    # ------------------------ 3. features + prévia de andamento (para priors estruturais)
    t = time.time()
    feat = dsp.extract_features(spec, xa, t_rel, BANDS, novelty_streams=det["streams"])
    env_pre, _s = dsp.onset_strength_envelope(spec, xa, sr, {"low": 1.15, "mid": 1.0, "high": 0.85})
    # Contagem de *referência* dos ataques proeminentes, com proeminência fixa — não depende de
    # `sensitivity`, então serve de régua para "a partitura deixou nota para trás?" tanto no
    # recomendador (`tune`) quanto no aviso abaixo. É um piso honesto, não um gabarito: picos
    # suaves de chimbel podem ficar abaixo dela, e isso é dito no texto do aviso.
    _ref_pk, _ref_pr = dsp.find_onset_frames(env_pre, spec.fps, sens=1.0, min_gap_ms=40.0,
                                             prom_frac=0.25, floor_frac=0.20)
    ref_onsets = {"n": int(_ref_pk.size),
                  "regra": "envelope global, proeminência 0,25, gap 40 ms — informativo: "
                           "ataques suaves de chimbel ficam abaixo disso, não é gabarito",
                  "por_seg": round(float(_ref_pk.size) / max(1e-6, float(t_rel[-1]) if t_rel.size else 1.0), 3)}
    if t_rel.size >= 8 and float(t_rel[-1]) >= 4.0:
        pre = G.estimate_tempo(env_pre, spec.fps, float(p["min_bpm"]), float(p["max_bpm"]),
                              bpm_hint=p["bpm_hint"])
        f0 = G.fit_grid(t_rel, float(pre["bpm"]), float(pre["phase_sec"]), 8,
                        mode="32nd+16th")
        bpm0, phase0 = (float(f0["bpm"]), float(f0["phase_sec"])) if f0.get("ok") else \
            (float(pre["bpm"]), float(pre["phase_sec"]))
    else:
        bpm0, phase0 = float(p["bpm_hint"] or 120.0), 0.0
    spb0 = 60.0 / bpm0
    lanes_enabled = set(str(l) for l in p["lanes"])
    hits_ev: List[dict] = []
    for gname, gidx in det["by_group"].items():
        if gidx.size == 0:
            continue
        gl = [l for l in GROUP_LANES.get(gname, []) if l in lanes_enabled]
        if p.get("aux_lanes"):
            gl += [l for l in AUX_LANES.get(gname, []) if l in lanes_enabled]
        if not gl:
            continue
        tg, fg = _slice_feat(t_rel, feat, gidx)
        # força relativa da novidade do grupo neste evento (0..1): quanto deste ataque
        # pertence de fato a esta banda — essencial para não chamar acorde de dois golpes
        # proeminência relativa do DESTE grupo no evento: mede se o grupo viu um ataque
        # próprio (acorde) ou apenas o vazamento espectral de outra peça.
        sbg = det.get("st_by_group", {}).get(gname)
        strength = np.asarray(sbg, dtype=np.float32)[gidx] if sbg is not None and len(sbg) == t_rel.size else None
        pp = dict(p)
        pp["_ctx"] = _context(fg, tg, bpm0, phase0, spb0)
        pp["_strength"] = strength
        pp["_fallback"] = GROUP_FALLBACK.get(gname)
        pp["_fallback_min_st"] = GROUP_FALLBACK_ST.get(gname, 0.62)
        lab, cf, _dom = C.classify_onsets(fg, lanes=gl, params=pp, gate=bool(p.get("gate", True)))
        en = _lane_energy(fg, [gl[int(x)] for x in lab])
        for k in range(gidx.size):
            li = int(lab[k])
            if li < 0:              # nenhum portão físico válido → não é golpe deste grupo
                continue
            hits_ev.append({"i": int(gidx[k]), "lane": gl[li], "conf": float(cf[k]),
                            "energy": float(en[k]), "time": float(tg[k]), "group": gname})
    hits_ev = _smooth_cymbals(hits_ev, spb0)
    # dedupe: mesma pista a < 22 ms (disparo duplo entre grupos) → mantém mais confiável
    hits_ev.sort(key=lambda d: (d["time"], d["lane"]))
    # dedupe POR PISTA com última posição mantida (o "anterior" na lista ordenada por tempo
    # pode ser de outra peça — comparar só com o vizinho imediato deixa duplicatas).
    ded: List[dict] = []
    last_by_lane: Dict[str, int] = {}
    for h in hits_ev:
        j = last_by_lane.get(h["lane"], -1)
        if j >= 0 and h["time"] - ded[j]["time"] < 0.030:
            if h["conf"] > ded[j]["conf"]:
                h["merged_from"] = 1
                ded[j] = h
            continue
        ded.append(h)
        last_by_lane[h["lane"]] = len(ded) - 1
    # filtro de confiança
    keep_hits = [h for h in ded if h["conf"] >= float(p["min_confidence"])]
    ev_keep = sorted({h["i"] for h in keep_hits})
    idx_all = np.array(ev_keep, dtype=np.int64) if ev_keep else np.zeros(0, np.int64)
    t_k, feat_k = _slice_feat(t_rel, feat, idx_all)
    lab_arr = np.array([h["lane"] for h in keep_hits], dtype=object)
    pos_of_ev = {int(v): j for j, v in enumerate(idx_all)}
    for h in keep_hits:
        h["ki"] = pos_of_ev[h["i"]]
    tim["features"] = time.time() - t
    n = t_k.size
    if n == 0:
        warn.append("Todos os eventos ficaram abaixo da confiança mínima. Baixe o limiar de "
                    "confiança ou aumente a sensibilidade.")
        return {"score": Score(), "report": {"warnings": warn, "n_events": 0,
                                             "params": dict(p)}, "detections": [],
                "wave": peaks_envelope(x_raw, sr, 1200), "timings": tim, "fine": None}

    # ----------------------------------------------------------------------- 4. andamento
    t = time.time()
    # `env_pre` acima é exatamente esta chamada (mesmas bandas, mesmos pesos): recomputá-la
    # custava um segundo passe completo do envelope de Hilbert sobre a faixa toda — a maior
    # alocação por amostra da cadeia. Uma só passada, dois usos.
    env, _str = env_pre, _s
    if float(t_k[-1]) < 4.0 or n < 8:
        tempo = {"bpm": float(p["bpm_hint"] or 120.0), "phase_sec": 0.0, "confidence": 0.0,
                 "candidates": [], "method": "fallback-faixa-curta"}
    else:
        tempo = G.estimate_tempo(env, spec.fps, float(p["min_bpm"]), float(p["max_bpm"]),
                                bpm_hint=p["bpm_hint"])
    bpm, phase = float(tempo["bpm"]), float(tempo["phase_sec"])
    if p["bpm_lock"]:
        bpm = float(p["bpm_lock"])
        tempo["method"] += "+lock"
        tempo["confidence"] = 1.0
    else:
        pre = "32nd+16th" if str(p["grid_mode"]) == "auto" else str(p["grid_mode"])
        fit = G.fit_grid(t_k, bpm, phase, 8, mode=pre)
        if fit.get("ok"):
            fb = float(fit["bpm"])
            tempo["fit"] = {k: fit.get(k) for k in ("bpm", "rms_ms", "n_used", "drift_ms_end", "iters")}
            # proteção contra o ajuste ser puxado por outliers: se divergir >3,5% do
            # máximo do pente, mantém-se o BPM do pente e aproveita-se só a fase.
            if abs(fb - bpm) / bpm <= 0.035:
                bpm = fb
                tempo["method"] += "+robust-fit"
            else:
                tempo["method"] += "+robust-fit(rejeitado)"
            tempo["fit"]["bpm_comb"] = round(bpm, 3)
            phase = float(fit["phase_sec"])
    # refinamento fino: maximiza o ajuste à grade (objetivo certo para notação)
    ev_conf0 = np.zeros(n, dtype=np.float64)
    for h in hits_ev:
        if 0 <= h["i"] < n:
            ev_conf0[h["i"]] = max(ev_conf0[h["i"]], h["conf"])
    # peso de *decisão de escrita*: eventos fracos não podem definir a granularidade
    w_dec = np.clip(ev_conf0 - 0.30, 0.02, 1.0)
    rb = G.refine_bpm(t_k, bpm, phase, 8, weights=w_dec)
    if rb.get("refined") and rb["fitness"] >= 0.5:
        tempo["bpm_before_refine"] = round(bpm, 3)
        bpm, phase = float(rb["bpm"]), float(rb["phase_sec"])
        tempo["refine"] = rb
        tempo["method"] += "+grid-residual"
    spb = 60.0 / bpm
    sec_per_tick = spb / TICKS_PER_QUARTER
    n_beats = max(4.0, (float(t_k[-1]) - phase) / spb)
    tim["tempo"] = time.time() - t

    # --------------------------------------------------- 5. swing + métrica / downbeat
    t = time.time()
    auto_swing = str(p["swing"]).lower() == "auto"
    swing_meas = G.estimate_swing(t_k, bpm, phase, 8) if auto_swing else float(p["swing"] or 0.0)
    pos_ticks = (t_k - phase) / sec_per_tick
    # para a métrica interessa o "papel estrutural" de cada evento: bumbo > caixa > prato
    prio = {"kick": 0, "snare": 1, "rim": 1, "crash": 2, "splash": 2, "ride": 3, "hat": 3,
            "hat_open": 3, "hat_foot": 3, "tom_hi": 4, "tom_mid": 4, "tom_low": 4, "cowbell": 5}
    ev_lane: Dict[int, Tuple[int, str]] = {}
    for h in keep_hits:
        k = prio.get(h["lane"], 9)
        if h["ki"] not in ev_lane or k < ev_lane[h["ki"]][0]:
            ev_lane[h["ki"]] = (k, h["lane"])
    ev_lanes = [ev_lane.get(i, (9, "hat"))[1] for i in range(n)]
    ev_energy = np.zeros(n, dtype=np.float64)
    for h in keep_hits:
        ev_energy[h["ki"]] = max(ev_energy[h["ki"]], h["energy"])
    if str(p["meter"]).lower() == "auto":
        met = G.estimate_meter(pos_ticks, ev_lanes, ev_energy, n_beats=n_beats)
        meter, tick_offset = met["meter"], int(met["tick_offset"])
    else:
        meter = str(p["meter"])
        met = {"meter": meter, "tick_offset": 0, "score": 0.0, "table": [], "n_bars": 0}
        tick_offset = 0
    user_shift = int(p.get("downbeat_shift_ticks", 0) or 0)
    tick_offset += user_shift
    info = meter_info(meter)
    tim["meter"] = time.time() - t

    # --------------------------------------------------------------------- 6. quantização
    t = time.time()
    grid_mode = str(p["grid_mode"])
    ev_conf = np.zeros(n, dtype=np.float64)
    for h in keep_hits:
        ev_conf[h["ki"]] = max(ev_conf[h["ki"]], h["conf"])
    wq = np.clip(ev_conf - 0.30, 0.02, 1.0)         # pesos = confiança da detecção
    q = G.quantize(t_k, bpm, phase, meter, mode=grid_mode,
                   swing=(None if auto_swing else float(swing_meas or 0.0)),
                   tick_offset=tick_offset, weights=wq)
    swing = float(q.get("swing", 0.0) or 0.0)
    # polimento do downbeat: com os ticks já projetados na grade, o template de backbeat
    # resolve a ambiguidade de ±1 tick que a otimização de fase não enxerga
    if n >= 12 and str(p["meter"]).lower() == "auto":
        ad = G.align_downbeat(q["ticks_abs"], ev_lanes, wq, meter)
        # A mesma régua do `quantize`: o offset só pode se mover em passos da grade escolhida.
        step_off = G.passo_grade(G.grid_positions(info["ticks_per_beat"], q["mode"], swing),
                                 info["ticks_per_beat"])
        off2 = int(round(int(ad["tick_offset"]) / step_off) * step_off) + user_shift
        if off2 != tick_offset:
            q2 = G.quantize(t_k, bpm, phase, meter, mode=grid_mode if grid_mode != "auto" else q["mode"],
                            swing=(None if auto_swing else swing), tick_offset=off2, weights=wq)
            if q2["fitness"] >= q["fitness"] - 1e-9:
                q, tick_offset = q2, off2
                met = dict(met)
                met.update({"tick_offset_prev": int(tick_offset), "downbeat_align": ad})
                swing = float(q.get("swing", 0.0) or 0.0)
        else:
            met = dict(met)
            met["downbeat_align"] = ad
            met["tick_offset_prev"] = int(met.get("tick_offset", tick_offset))
    if n >= 12:
        fit2 = G.fit_grid(t_k, bpm, phase, info["ticks_per_beat"], mode=q["mode"])
        if fit2.get("ok") and abs(float(fit2["bpm"]) - bpm) > 0.02:
            bpm = float(fit2["bpm"])
            spb = 60.0 / bpm
            sec_per_tick = spb / TICKS_PER_QUARTER
            q = G.quantize(t_k, bpm, phase, meter, mode=grid_mode if grid_mode != "auto" else q["mode"],
                           swing=(None if auto_swing else swing), tick_offset=tick_offset,
                           weights=wq)
            swing = float(q.get("swing", 0.0) or 0.0)
    ticks, bars_idx = q["ticks"], q["bars"]
    b0 = int(np.min(bars_idx)) if bars_idx.size else 0
    bars_idx = bars_idx - b0
    tim["quantize"] = time.time() - t

    # ----------------------------------------------------------------------- 7. dinâmica
    t = time.time()
    lane_list = [h["lane"] for h in keep_hits]
    e_lane = np.asarray([h["energy"] for h in keep_hits], dtype=np.float64)
    conf_h = np.asarray([h["conf"] for h in keep_hits], dtype=np.float32)
    vel, artic = _dynamics(e_lane, lane_list, p)
    flams = _detect_flams(np.asarray([h["time"] for h in keep_hits]), lane_list)
    for i in flams:
        if artic[i] == "normal":
            artic[i] = "flam"
    tim["dyn"] = time.time() - t

    # ---------------------------------------------------------------------- 8. partitura
    t = time.time()
    n_bars = int(bars_idx.max(initial=0)) + 1
    score = Score(bpm=round(bpm, 2), meter=meter, swing=swing,
                  title="Transcrição de Bateria", subtitle=filename)
    per_bar: Dict[int, List[Hit]] = {i: [] for i in range(max(n_bars, 1))}
    buckets: Dict[int, Dict[Tuple[int, str], Hit]] = {i: {} for i in range(max(n_bars, 1))}
    used_lanes: List[str] = []
    for i, h in enumerate(keep_hits):
        b = int(bars_idx[h["ki"]])
        if not (0 <= b < n_bars):
            continue
        _merge_hit(buckets[b], Hit(lane=h["lane"], tick=int(ticks[h["ki"]]), dur=2,
                                   velocity=int(vel[i]), artic=artic[i],
                                   confidence=float(conf_h[i]),
                                   time=float(t_k[h["ki"]] + offset_s)))
        if h["lane"] not in used_lanes:
            used_lanes.append(h["lane"])
    for b, bk in buckets.items():
        per_bar[b] = sorted(bk.values(), key=lambda x: (x.tick, x.lane))
    tmp_bars = [Bar(index=bi, hits=per_bar[bi]) for bi in range(n_bars)]
    for bi, bar in enumerate(tmp_bars):
        _mark_sustain(bar, info, tmp_bars[bi + 1] if bi + 1 < n_bars else None)
        score.bars.append(bar)
    score.lanes = [l for l in DEFAULT_LANES if l in used_lanes] + \
                  [l for l in used_lanes if l not in DEFAULT_LANES]
    if p.get("fill_detect"):
        _mark_fills(score)
    n_slash = _mark_repeats(score) if p.get("repeat_slash", True) else 0
    tim["assemble"] = time.time() - t

    # ----------------------------------------------------------------------- 9. relatório
    # Os números por peça são contados NA PARTITURA FINAL, não na lista de detecções:
    # `_merge_hit` funde dois golpes do mesmo slot numa nota só (é o comportamento certo na
    # pauta), e um relatório contado antes da fusão prometeteria mais notas do que as que o
    # usuário vê impressas. `n_detections` mantém o outro número, para que a diferença entre
    # os dois seja visível em vez de silenciosa.
    per_lane: Dict[str, dict] = {}
    arr_lane = np.asarray(lane_list, dtype=object)
    total_not = sum(len(b.hits) for b in score.bars) or 1
    for ln in score.lanes:
        rows = [h for b in score.bars for h in b.hits if h.lane == ln]
        if not rows:
            continue
        m = arr_lane == ln
        ki = np.array([h["ki"] for h, mm in zip(keep_hits, m) if mm], dtype=np.int64)
        v = np.array([h.velocity for h in rows], dtype=np.float64)
        cf = np.array([h.confidence for h in rows], dtype=np.float64)
        per_lane[ln] = {"name": LANE_BY_ID[ln].name if ln in LANE_BY_ID else ln,
                        "count": len(rows),
                        "share_pct": round(100.0 * len(rows) / total_not, 1),
                        "mean_velocity": int(np.mean(v)), "min_velocity": int(np.min(v)),
                        "max_velocity": int(np.max(v)),
                        "mean_confidence": round(float(np.mean(cf)), 3),
                        "first_bar": 1 + min(bb.index for bb in score.bars
                                             if any(h.lane == ln for h in bb.hits)),
                        "n_detections": int(m.sum()),
                        "mean_offset_ms": round(float(np.mean(q["resid"][ki])) * sec_per_tick * 1000.0, 1)
                        if ki.size else 0.0}
    hit_idx = np.array([h["ki"] for h in keep_hits], dtype=np.int64)
    off = np.asarray(q["offgrid"], dtype=bool)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "file": {"name": audio.filename, "sr_source": audio.source_sr, "channels": audio.channels,
                 "duration_sec": round(audio.duration, 3),
                 "peak_dbfs": round(20 * math.log10(max(audio.peak, 1e-9)), 2),
                 "rms_dbfs": round(20 * math.log10(max(audio.rms, 1e-9)), 2),
                 "crest_db": round(20 * math.log10((audio.peak + 1e-9) / (audio.rms + 1e-9)), 2),
                 "clipped_ratio": round(audio.clipped_ratio, 5), "analysis_sr": sr,
                 "n_fft": n_fft, "hop": hop, "fps": round(spec.fps, 2),
                 "ms_por_quadro": round(1000.0 * hop / sr, 3),
                 "hz_por_bin": round(sr / float(n_fft), 3),
                 "nyquist_hz": int(sr // 2),
                 "trim_lead_ms": round(1000.0 * lead / sr, 1)},
        "referencia_onsets": ref_onsets,
        "memory": {"orcamento_mb": round(orcamento), "custo_mb": round(custo_mb),
                   "disponivel_mb": round(mem_disponivel_mb()) or None,
                   "ajustado_por_memoria": bool(ajustado),
                   "fator_bluestein": round(f_fft, 2),
                   # cada degrau expõe a grade que produz: se `ms_por_quadro` differs entre
                   # degraus, a escada voltou a mover notas (A14) — isto é o que o teste lê.
                   "planos": [{"sr": int(a), "hop": int(b), "n_fft": int(c),
                               "ms_por_quadro": round(1000.0 * b / a, 3),
                               "hz_por_bin": round(a / float(c), 3),
                               "custo_mb": round(d)}
                              for (a, b, c, d) in escada]},
        "tempo": {"bpm": round(bpm, 2), "confidence": round(float(tempo.get("confidence", 0.0)), 3),
                  "method": tempo.get("method", ""), "candidates": tempo.get("candidates", []),
                  "fit": tempo.get("fit", {}), "beat_period_ms": round(1000.0 * spb, 2),
                  "phase_ms": round(1000.0 * phase, 2),
                  "bpm_range": [p["min_bpm"], p["max_bpm"]]},
        "meter": {"value": meter, "beats_per_bar": info["beats"], "kind": info["kind"],
                  "ticks_per_bar": info["ticks_per_bar"], "tick_offset": int(tick_offset),
                  "downbeat_align": (met.get("downbeat_align") or {}),
                  "score": round(float(met.get("score", 0.0)), 4), "table": met.get("table", []),
                  "n_beats": round(float(n_beats), 2), "n_bars": int(n_bars)},
        "grid": {"mode": q["mode"], "mode_requested": grid_mode, "swing": float(swing),
                 "swing_detected": float(q.get("swing_detected", 0.0) or 0.0),
                 "fitness": round(float(q["fitness"]), 4), "offgrid": int(np.sum(off)),
                 "offgrid_ratio": round(float(off.mean()) if off.size else 0.0, 3),
                 "tol_ms": round(float(q["tol_ticks"]) * sec_per_tick * 1000.0, 1),
                 "mean_residual_ms": round(float(np.mean(q["resid"][hit_idx])) * sec_per_tick * 1000.0, 1)
                 if hit_idx.size else 0.0,
                 "median_residual_ms": round(float(np.median(q["resid"])) * sec_per_tick * 1000.0, 1),
                 "swing_allowed": bool(q.get("swing_allowed", False)),
                 "promoted_32nd": int(q.get("promoted_32nd", 0)),
                 # declarado sobre O QUE FOI ESCRITO, não sobre o array pré-escrita: `q["ticks"]`
                 # tem um item por *evento* e a pauta pode ter dois golpes no mesmo instante (bumbo +
                 # caixa) — em 4/4 isso não aparecia (todos os ticks são pares) e num 6/8 real a
                 # auditoria recontava 1474 ímpares contra 1008 declarados (A21).
                 "odd_ticks": int(sum(1 for b in score.bars for h in b.hits if int(h.tick) % 2)),
                 "fits_by_grid": q.get("fits", {}),
                 "fit_target": G.FIT_TARGET},
        "events": {"clusters": int(det["n"]), "per_group": {k: int(len(v)) for k, v in det["by_group"].items()},
                   "candidate_hits": len(ded), "notated": len(keep_hits), "events_used": int(n),
                   "mean_confidence": round(float(np.mean(conf_h)) if conf_h.size else 0.0, 3),
                   "min_confidence_used": float(p["min_confidence"]), "flams": len(flams),
                   "per_lane": per_lane},
        "params": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in p.items()},
        "engraving": {"repeat_slash_bars": int(n_slash),
                      "fill_bars": int(sum(1 for b in score.bars if b.fill_marker)),
                      "n_bars": len(score.bars), "lanes": list(score.lanes)},
        "timings_sec": {k: round(v, 3) for k, v in tim.items()},
        "total_sec": round(time.time() - t0, 3),
        "warnings": warn + _warnings(audio, bpm, info, q, hit_idx, sec_per_tick, conf_h,
                                     np.asarray([h["time"] for h in keep_hits]), t_k, e_lane),
    }
    score.report = report
    _aud = report.get("file") or {}
    _parts = [str(filename)] if filename else []
    _dur = _aud.get("duration_sec")
    if _dur:
        _parts.append("%.1f s" % float(_dur))
    _sr = _aud.get("analysis_sr")
    if _sr:
        _parts.append("%.1f kHz" % (float(_sr) / 1000.0))
    _parts.append("%d notas em %d compassos" % (len(keep_hits), len(score.bars)))
    score.subtitle = " · ".join(_parts)

    # ---- apresentação (não toca nos dados): partitura simples / pistas ocultas
    ocultas = [str(x) for x in (p.get("hide_lanes") or (PRATOS if p.get("simples") else []))]
    if ocultas:
        score.hide_lanes = ocultas
        report["warnings"].append(
            "parte das pistas está oculta na gravura (%s): a partitura foi impressa sem elas, "
            "mas os golpes continuam nos dados, no JSON/CSV e na auditoria — ocultar é "
            "apresentação, nunca apagamento." % "·".join(ocultas))
    report["apresentacao"] = {"hide_lanes": list(score.hide_lanes),
                              "simples": bool(p.get("simples")),
                              "pratos": list(PRATOS)}

    # ---- envelope fino (zoom sem re-decodificar): onda em ~1 ms + novidade por grupo
    fine = None
    if p.get("onda_fina"):
        try:
            onda, bs = peaks_finos(x_raw, sr)
            n = int(onda.shape[1])
            curvas = {"global": env_pre}
            for k, v in (det.get("streams") or {}).items():
                if v is not None and len(v):
                    curvas[str(k)] = v
            nomes = sorted(curvas)
            pl = planes_finos({k: curvas[k] for k in nomes}, n, bs, sr,
                              desvio_amostras=int(lead), fps=float(spec.fps))
            fine = {"onda": onda, "curvas": pl, "nomes": nomes,
                    "bucket_sec": float(bs), "n": n, "sr": int(sr),
                    "fps": round(float(spec.fps), 3), "lead_ms": round(1000.0 * lead / sr, 2)}
        except Exception as e:                                   # o zoom é acessório: não derruba
            report["warnings"].append("envelope de alta resolução indisponível (%s) — o zoom "
                                      "mostra a visão geral de 1800 baldes." % e)

    dets = [{"time": round(float(t_k[h["ki"]] + offset_s), 4), "lane": h["lane"],
             "conf": round(float(h["conf"]), 3), "velocity": int(vel[i]),
             "tick": int(ticks[h["ki"]]), "bar": int(bars_idx[h["ki"]]) if bars_idx.size else 0,
             "resid_ms": round(float(q["resid"][h["ki"]]) * sec_per_tick * 1000.0, 1),
             "artic": artic[i], "group": h["group"],
             "nov": ([round(float(v), 3) for v in feat.novelty[h["i"]]]
                     if feat.novelty is not None and h["i"] < feat.novelty.shape[0] else [])}
            for i, h in enumerate(keep_hits)]
    return {"score": score, "report": report, "detections": dets,
            "wave": peaks_envelope(x_raw, sr, 1800), "timings": tim, "fine": fine,
            "audio": {"sr": sr, "duration": round(audio.duration, 3),
                      "sample_rate": audio.source_sr, "channels": audio.channels}}


# ======================================================================================
# reconstrução de partitura a partir de hits editados (usado pelo servidor/editor)
# ======================================================================================

def rebuild_score(hits: list, bpm: float, meter: str = "4/4", swing: float = 0.0,
                  title: str = "Transcrição de Bateria", subtitle: str = "",
                  report: Optional[dict] = None, grid_mode: str = "16th",
                  ticks_per_beat: Optional[int] = None) -> dict:
    """
    Dado a lista de golpes (dicts com lane/tick/bar/dur/velocity/artic/confidence/time),
    refaz a escrita: durações, pausas, vigas, sustenus, acentos, viradas, repetições.
    É o mesmo final de `transcribe_bytes`, exposto para o editor interativo.
    """
    info = meter_info(meter)
    tpb = int(ticks_per_beat or info["ticks_per_beat"])
    tpr = info["ticks_per_bar"]
    rows = []
    for h in hits:
        lane = str(h.get("lane", "snare"))
        if lane not in LANE_BY_ID:
            continue
        tk = int(round(float(h.get("tick", 0))))
        bar = int(round(float(h.get("bar", 0))))
        rows.append({"lane": lane, "tick": int(np.clip(tk, 0, tpr - 1)), "bar": max(0, bar),
                     "dur": int(max(1, round(float(h.get("dur", 2) or 2)))),
                     "velocity": int(np.clip(int(round(float(h.get("velocity", 88)))), 1, 127)),
                     "artic": str(h.get("artic", "normal")),
                     "confidence": float(h.get("confidence", 0.9)),
                     # sem medição não significa "segundo zero": preservar None evita que a
                     # auditoria leia uma nota editada à mão como se o áudio dissesse 0,000 s
                     "time": (float(h["time"]) if h.get("time") is not None else None)})
    n_bars = max([r["bar"] for r in rows], default=0) + 1
    score = Score(bpm=round(float(bpm), 2), meter=meter, swing=float(swing or 0.0),
                  title=title, subtitle=subtitle)
    per_bar: Dict[int, List[Hit]] = {i: [] for i in range(n_bars)}
    buckets: Dict[int, Dict[Tuple[int, str], Hit]] = {i: {} for i in range(n_bars)}
    used_lanes: List[str] = []
    for r in rows:
        _merge_hit(buckets[r["bar"]], Hit(lane=r["lane"], tick=r["tick"], dur=r["dur"],
                                         velocity=r["velocity"], artic=r["artic"],
                                         confidence=r["confidence"], time=r["time"]))
        if r["lane"] not in used_lanes:
            used_lanes.append(r["lane"])
    for b, bk in buckets.items():
        per_bar[b] = sorted(bk.values(), key=lambda x: (x.tick, x.lane))
    tmp_bars = [Bar(index=bi, hits=per_bar[bi]) for bi in range(n_bars)]
    for bi, bar in enumerate(tmp_bars):
        _mark_sustain(bar, info, tmp_bars[bi + 1] if bi + 1 < n_bars else None)
        score.bars.append(bar)
    score.lanes = [l for l in DEFAULT_LANES if l in used_lanes] + \
                  [l for l in used_lanes if l not in DEFAULT_LANES]
    _mark_fills(score)
    _mark_repeats(score)
    score.repeats = 1
    if report:
        score.report = report
    return score.to_dict()


def transcribe_file(path: str, params: Optional[dict] = None) -> dict:
    with open(path, "rb") as f:
        return transcribe_bytes(f.read(), filename=path.replace("\\", "/").split("/")[-1],
                                params=params)


# ======================================================================================
# helpers
# ======================================================================================

def _detect_flams(times: np.ndarray, lanes: Sequence[str]) -> set:
    """Flam/drag: dois golpes da mesma peça a 22–65 ms → o segundo é a nota principal."""
    out = set()
    order = np.argsort(times)
    for k in range(len(order) - 1):
        i, j = int(order[k]), int(order[k + 1])
        if lanes[i] == lanes[j] and lanes[i] in ("snare", "tom_hi", "tom_mid", "tom_low", "kick"):
            dt = float(times[j] - times[i])
            if 0.022 <= dt <= 0.065:
                out.add(j)
    return out


def _dynamics(e_db: np.ndarray, lanes: List[str], p: dict) -> Tuple[np.ndarray, List[str]]:
    """
    Velocidade MIDI a partir do nível em dB relativo à mediana global, com correção
    parcial pela mediana da própria peça (senão um chimel em pianíssimo seria notado
    fortíssimo só por ser o mais forte do chimel).
    """
    e = np.asarray(e_db, dtype=np.float64)
    n = e.size
    if n == 0:
        return np.zeros(0, dtype=np.int64), []
    med_all = float(np.median(e))
    arr = np.asarray(lanes, dtype=object)
    r = e - med_all
    mode = p.get("dyn_mode", "blend")
    if mode in ("per_lane", "blend"):
        for ln in set(lanes):
            m = arr == ln
            if int(m.sum()) < 3:
                continue
            d = e[m] - float(np.median(e[m]))
            r[m] = d if mode == "per_lane" else (0.55 * r[m] + 0.45 * d)
    scale = float(p.get("dyn_scale", 4.0))
    vel = np.rint(np.clip(78.0 + scale * r, 20.0, 127.0)).astype(np.int64)
    artic: List[str] = []
    for i, ln in enumerate(lanes):
        v = int(vel[i])
        a = "normal"
        # A política diz onde se *espera* um fantasma; a permissão vem do `kit` (uma só fonte,
        # ver `_GHOST_OK` em qa.py): peça sem cabeça própria de fantasma não recebe o sinal.
        if (p.get("ghost_notes", True) and v <= 45 and ln in _GHOST_POLICY
                and getattr(LANE_BY_ID.get(ln), "ghostable", True)):
            a = "ghost"
        if a == "normal" and p.get("accents", True) and v >= 108 and ln in (
                "snare", "kick", "tom_hi", "tom_mid", "tom_low", "crash"):
            a = "accent"
        artic.append(a)
    return vel, artic


def _mark_sustain(bar: Bar, info: dict, next_bar: Optional[Bar] = None) -> None:
    """
    Sustain de prato/chimel: a nota é esticada até o próximo ataque da voz 1 e só recebe
    ligadura se existir um **destino na MESMA peça** — o próximo golpe daquele prato, encostado
    no fim da nota (dentro do compasso, ou no tempo 1 do compasso seguinte).

    A restrição à mesma pista é o ponto: ligadura é atributo da *nota*, e a voz só governa a
    linha do tempo. Ligar um ride a um chimbal escreve `<tie start>` no ride e `<tie stop>` no
    chimbal — o notador emenda dois instrumentos diferentes, a duração soada do ride cresce e o
    contador de ligaduras do export fica assimétrico. Sem destino na mesma peça, a nota fica
    apenas com o valor longo (que já preenche a voz), sem ligadura.
    """
    tpr = int(info["ticks_per_bar"])
    tpb = int(info["ticks_per_beat"])
    hits = bar.sorted_hits()
    # Lei de escrita (A22): sem ligadura, uma nota não pode terminar depois do fim da barra — a
    # impressora não tem onde pendurá-la e o MusicXML sairia com uma nota a mais. Corta-se o
    # sustento, nunca a nota: `dur` é encurtado até a barra e o ataque permanece onde o áudio pôs.
    for h in hits:
        if not (h.tie_start or h.tie_stop) and int(h.dur) > tpr - int(h.tick):
            h.dur = max(1, tpr - int(h.tick))
    v1 = [h for h in hits if LANE_BY_ID[h.lane].voice == 1]
    for i, h in enumerate(v1):
        if not LANE_BY_ID[h.lane].tie:
            continue
        if i + 1 < len(v1):
            nxt, base = v1[i + 1], h.tick
            gap = nxt.tick - h.tick
            lands_here = True
        else:
            nb = [k for k in (next_bar.sorted_hits() if next_bar is not None else [])
                  if LANE_BY_ID[k.lane].voice == 1]
            nxt = nb[0] if nb else None
            gap = (tpr - h.tick) + (nxt.tick if nxt is not None else 0)
            lands_here = nxt is not None and int(nxt.tick) == 0
        if gap < tpb:
            continue                                  # ataque seguinte cedo demais: sem sustain
        h.dur = max(int(h.dur), min(gap if lands_here else (tpr - h.tick), tpr - h.tick))
        same_lane = nxt is not None and str(nxt.lane) == str(h.lane)
        if lands_here and nxt is not None and same_lane:
            h.tie_start = True
            nxt.tie_stop = True


#: Onde a escrita de bateria costuma usar nota fantasma (a *política*; a permissão é `kit`).
_GHOST_POLICY = frozenset(("snare", "hat", "ride", "kick"))

_ARTIC_RANK = {"normal": 0, "ghost": 1, "ghost_accent": 2, "flam": 3, "droll": 4, "accent": 5}


def _merge_hit(bucket: Dict[Tuple[int, str], Hit], h: Hit) -> None:
    """
    Duas detecções do mesmo golpe (ex.: chimel que dispara na banda alta e no ataque do
    mid) não podem virar duas notas no mesmo pulso: fundimos em uma, ficando com a
    velocidade maior, a melhor articulação e a maior confiança.
    """
    key = (int(h.tick), h.lane)
    old = bucket.get(key)
    if old is None:
        bucket[key] = h
        return
    keep = old
    if h.velocity > old.velocity or _ARTIC_RANK.get(h.artic, 0) > _ARTIC_RANK.get(old.artic, 0):
        keep, other = h, old
    else:
        keep, other = old, h
    keep.velocity = max(int(old.velocity), int(h.velocity))
    keep.confidence = max(float(old.confidence), float(h.confidence))
    keep.artic = max((old.artic, h.artic), key=lambda a: _ARTIC_RANK.get(a, 0))
    keep.tie_start = bool(old.tie_start or h.tie_start)
    keep.tie_stop = bool(old.tie_stop or h.tie_stop)
    keep.dur = max(int(old.dur), int(h.dur))
    keep.time = min(float(old.time or 0.0), float(h.time or 0.0))
    bucket[key] = keep



def _mark_repeats(score: Score, min_repeats: int = 3) -> int:
    """
    Compassos idênticos a partir da 3ª ocorrência viram "compasso de repetição" (𝄄 / time
    slash) — o padrão em lead sheet de bateria. As notas continuam no JSON (o editor e o
    MIDI/MusicXML mantêm tudo); só a gravura troca por barras de repetição.
    """
    from . import rules as R
    if len(score.bars) < 3:
        return 0
    flags = R.detect_repeats(score.bars, min_repeats=min_repeats)
    n = 0
    for i, f in enumerate(flags):
        if f and len(score.bars[i].hits) >= 3:
            score.bars[i].repeat_slash = True
            n += 1
    return n


def _mark_fills(score: Score) -> None:
    for i, b in enumerate(score.bars):
        tones = [h for h in b.hits if h.lane in ("tom_hi", "tom_mid", "tom_low")]
        kicks = [h for h in b.hits if h.lane == "kick"]
        snares = [h for h in b.hits if h.lane == "snare"]
        if len(tones) >= 3 or (len(tones) >= 2 and not kicks and len(snares) <= 2):
            prev = [h for h in score.bars[i - 1].hits if h.lane.startswith("tom")] if i else []
            if not prev:
                b.fill_marker = True
        if b.fill_marker and i + 1 < len(score.bars):
            if any(h.lane in ("crash", "kick") and h.tick < 4 for h in score.bars[i + 1].hits):
                b.cadence = True


def _warnings(audio, bpm, info, q, hit_idx, sec_per_tick, conf, times, t_k, e_lane) -> List[str]:
    w: List[str] = []
    if audio.clipped_ratio > 0.002:
        w.append("Sinal com clipping (~%.2f%% das amostras acima de -0.1 dBFS): transientes "
                 "achatados reduzem a resolução da dinâmica exportada." % (100 * audio.clipped_ratio))
    if audio.peak < 0.06:
        w.append("Entrada muito baixa (%.1f dBFS peak): ruído de piso pode estar limitando "
                 "chimel e ghost notes." % (20 * math.log10(max(audio.peak, 1e-9))))
    if hit_idx.size and float(np.mean(q["resid"][hit_idx])) * sec_per_tick * 1000.0 > 24.0:
        w.append("Resíduo médio de quantização alto: pulso possivelmente não uniforme "
                 "(rubato/viradas) ou BPM em oitava errada.")
    if conf.size and float(np.mean(conf)) < 0.42:
        w.append("Confiança média baixa na classificação das peças — revise a legenda e "
                 "edite na grade. Para mix cheio, aumente a sensibilidade e confie menos "
                 "nos pratos.")
    if bpm < 55 or bpm > 190:
        w.append("Andamento extremo (%.1f BPM): confirme a oitava (÷2 / ×2)." % bpm)
    if info["kind"] == "compound":
        w.append("Métrica composta (%s) detectada: confira se não é a mesma música escrita "
                 "em quaternário simples com tercinas." % info["name"])
    if times.size >= 8:
        gap = float(np.median(np.diff(times)))
        if gap < 0.06:
            w.append("Golpes muito próximos (mediana %.0f ms): há provável dupla detecção; "
                     "aumente merge_ms ou reduza a sensibilidade." % (1000 * gap))
    return w


def score_to_csv(score: dict) -> str:
    """
    CSV de correção (o formato que o editor lê de volta). Uma linha por golpe, com o nome da
    peça e a nota GM, para poder ser aberto no Excel/LibreOffice sem traduzir nada à mão.
    """
    lines = ["compasso,tick,tempo_s,peca,duracao_ticks,nota_midi,velocidade,articulacao,confiança"]
    for b in score.get("bars", []):
        for h in b.get("hits", []):
            ln = LANE_BY_ID.get(str(h.get("lane")))
            lines.append("%d,%d,%.3f,%s,%d,%d,%d,%s,%.2f" % (
                int(b.get("index") or 0), int(h.get("tick") or 0), float(h.get("time") or 0.0),
                (ln.name if ln else str(h.get("lane"))), int(h.get("dur") or 2),
                int(ln.gm if ln else 38), int(h.get("velocity") or 88),
                str(h.get("artic", "normal")), float(h.get("confidence") or 0.0)))
    return "\n".join(lines)


def score_to_csv_bom(score: dict) -> str:
    return "\ufeff" + score_to_csv(score)
