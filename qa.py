"""
Validação dupla e agente de correção (auditoria de cada estágio do pipeline).
=========================================================================================

Nenhum estágio é aceito pela própria palavra. Cada grandeza é medida **duas vezes por
caminhos independentes** — o caminho do pipeline (A) e um oráculo escrito aqui, com outra
cadeia de sinal (B) — e a divergência vira um *achado* com severidade, evidência numérica
e, quando existir, um reparo determinístico.

Camadas
-------
1. **Medição dupla.** Envelope de RMS + autocorrelação por FFT, sem STFT, sem bandas e sem
   prior de andamento, para o BPM; lei do copista (bumbo nos tempos fortes, caixa nos de
   baixo, prato em todos) aplicada à partitura *já impressa* para a posição da barra; e
   re-cálculo de `tick = tempo/secpertick` para a grade. Coincidir duas implementações que
   não compartilham código é evidência; divergir é bug em alguma delas.
2. **Leis de escrita.** invariantes que não dependem de áudio: tick no intervalo, duração > 0,
   velocidade em 1..127, articulação no alfabeto, fantasma com cabeça "–", voz monoconsistente
   (sem sobreposição na mesma voz), ligaduras alternando start/stop por peça, 𝄄 só sobre
   repetição fiel, nenhum par (compasso,tick,peça) duplicado, soma das durações de uma voz =
   um compasso cheio no MusicXML.
3. **Compilação dupla dos exports.** MusicXML/MIDI/PDF são regerados e **relidos por leitores
   independentes** (`xml.etree` e um parser SMF próprio, não o nosso `midi_out`) e confrontados
   com a partitura: nº de notas, soma de durações em ticks, divisões, canal 10, notas GM e o
   meta-evento de andamento.
4. **Oráculo sem gabarito (ida e volta).** A partitura é sintetizada (`kit_synth`) e o áudio
   sintético é reanalisado: um golpe escrito que não volta como ataque é suspeito. Depois, os
   ataques evidentes do áudio original que não têm nota escrita entram numa lista — isso
   funciona em arquivo de usuário, que não tem ground truth.
5. **Premissas do sinal.** clipping, nível, duração e um teste heurístico de "isto é uma track
   isolada de bateria?" — porque todo o resto assume que sim.

`repair()` aplica só o que é determinístico e depois **re-audita**: se o quadro piorar, o
reparo é revertido. Nada aqui altera o pipeline; é leitura e reconstrução pela mesma porta
que o editor usa (`rebuild_score`).
"""
from __future__ import annotations

import json
import math
import os
import re
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .kit import LANE_BY_ID
from .score_model import TICKS_PER_QUARTER, meter_info, score_visivel

SEVERITIES = ("error", "warn", "info")
_ARTIC = ("normal", "accent", "ghost", "ghost_accent", "flam", "droll")
_CYMBAL = {"hat", "hat_foot", "hat_open", "ride", "ride_bell", "crash", "splash", "china"}
#: Quem pode receber cabeça de fantasma é decidido em UM lugar: `kit.Lane.ghostable`. A lista
#: anterior era paralela (e incluía `cross_stick`, que não é pista, e excluía `kick`, que o
#: `kit` admite) — assim o escrevedor podia emitir um fantasma que o auditor rejeitava, e isso
#: apareceu como erro T6 numa faixa real de 239 compassos (A21).
_GHOST_OK = frozenset(k for k, ln in LANE_BY_ID.items() if getattr(ln, "ghostable", True))


# =========================================================================================
# utilidades
# =========================================================================================
def _f(v: Any, d: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else d
    except Exception:
        return d


def _find(check: str, stage: str, title: str, severity: str, *, a: Any = None, b: Any = None,
          detail: str = "", fix: Optional[str] = None, state: str = "ok", data: Any = None) -> dict:
    """`state`: ok (passou) | manual (real, exige decisão) | fixed | reverted | skipped."""
    return {"check": check, "stage": stage, "title": title, "severity": severity, "a": a, "b": b,
            "detail": detail, "fix": fix, "state": state, "data": data}


def _hits_of(score: dict) -> List[dict]:
    out: List[dict] = []
    for bar in score.get("bars", []) or []:
        bi = int(_f(bar.get("index")))
        for h in bar.get("hits", []) or []:
            g = dict(h)
            g["bar"] = bi
            out.append(g)
    return out


def _spt(score: dict) -> float:
    """segundos por tick"""
    bpm = _f(score.get("bpm"), 120.0) or 120.0
    tpq = _f(score.get("ticks_per_quarter", TICKS_PER_QUARTER), 8.0) or 8.0
    return (60.0 / bpm) / tpq


# =========================================================================================
# oráculos
# =========================================================================================
def oracle_envelope(x: np.ndarray, sr: int, win_ms: float = 10.0) -> Tuple[np.ndarray, float]:
    """Envelope de energia por RMS temporal. Sem STFT, sem bandas, sem normalização alguma."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = int(max(1, round(0.001 * win_ms * sr)))
    m = x.size // n
    if m < 8:
        return np.zeros(0, np.float64), 0.0
    y = np.sqrt((x[:m * n].reshape(m, n) ** 2).mean(axis=1))
    return y, 1000.0 / win_ms


def oracle_bpm(env: np.ndarray, fps: float, min_bpm: float = 45.0, max_bpm: float = 220.0,
               n_max: int = 14) -> dict:
    """
    Andamento por autocorrelação via FFT do envelope, escolhendo o lag mais alto cujo pico
    ainda divide bem os picos anteriores (harmonics). Diferente do pipeline: sem prior de
    BPM, sem varredura de fase, sem fluxo por banda.
    """
    out = {"bpm": 0.0, "confidence": 0.0, "peaks": [], "method": "acf-envelope-rms"}
    if env.size < int(fps * 2.0):
        out["method"] = "envelope-curto"
        return out
    e = env - float(np.mean(env))
    nfft = 1 << int(math.ceil(math.log2(2 * e.size)))
    ac = np.fft.irfft(np.abs(np.fft.rfft(e, nfft)) ** 2, nfft)[:e.size]
    ac = ac / (ac[0] + 1e-12)
    lo = max(2, int(round(60.0 / max_bpm * fps)))
    hi = min(e.size - 1, int(round(60.0 / min_bpm * fps)), int(round(n_max * fps)))
    if hi <= lo:
        out["method"] = "faixa-invalida"
        return out
    seg = ac[lo:hi + 1]
    lags = np.arange(lo, hi + 1)
    # picos locais da autocorrelação
    pk = [i for i in range(1, seg.size - 1) if seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]]
    if not pk:
        pk = [int(np.argmax(seg))]
    peaks = sorted(((float(seg[i]), int(lags[i])) for i in pk), reverse=True)
    out["peaks"] = [(round(v, 3), round(60.0 * fps / lg, 2)) for v, lg in peaks[:6]]
    bestv, bestlag = peaks[0]
    # preferimos o menor lag (andamento mais rápido) entre picos quase empatados: é o
    # desempate clássico para não escorregar para ½×
    for v, lg in peaks[:6]:
        if v >= 0.97 * bestv and lg < bestlag:
            bestlag = lg
    out["bpm"] = round(60.0 * fps / bestlag, 3)
    out["confidence"] = round(max(0.0, min(1.0, bestv)), 3)
    return out


def octave_dev(bpm_a: float, bpm_b: float, ratios: Sequence[float] = (0.25, 1 / 3.0, 0.5, 1.0, 2.0, 3.0, 4.0)) -> Tuple[float, float]:
    """
    Desvio relativo de duas estimativas de andamento **módulo oitava/submúltiplo** — a
    ambiguidade de oitava é inerente a qualquer método espectral, então um oráculo não pode
    acusar o pipeline de errar por 2×. Devolve (desvio %, razão escolhida).
    """
    if bpm_a <= 0 or bpm_b <= 0:
        return 100.0, 0.0
    best, br = 1e9, 1.0
    for r in ratios:
        d = abs(bpm_a - bpm_b * r) / bpm_a * 100.0
        if d < best:
            best, br = d, r
    return round(float(best), 3), float(round(br, 4))


def grid_fit_pure(times: np.ndarray, bpm: float, sub: float = 1.0, sharp: float = 0.30) -> float:
    """Ajuste (0..1) de um pulso puro: média de exp −(d/σσ)², σ = sharp·período·sub."""
    if times.size == 0 or bpm <= 0:
        return 0.0
    per = (60.0 / bpm) * sub
    r = np.mod(times, per)
    d = np.minimum(r, per - r)
    return float(np.mean(np.exp(-(d / (sharp * per)) ** 2)))


def oracle_onsets(x: np.ndarray, sr: int, gap_ms: float = 30.0, k: float = 1.6,
                  win_ms: float = 10.0, return_prominence: bool = False):
    """
    Ataques por diferença positiva de energia em janelas de 10 ms, limiar mediana + k·MAD e
    gap mínimo. Não usa STFT nem bandas nem o refinamento do pipeline: se o pipeline perdeu
    um ataque que isto vê, é um achado honesto (e vice-versa).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = int(max(1, round(0.001 * win_ms * sr)))
    m = x.size // n
    if m < 16:
        return (np.zeros(0), np.zeros(0)) if return_prominence else np.zeros(0)
    e = (x[:m * n].reshape(m, n) ** 2).mean(axis=1)
    d = np.maximum(np.diff(e, prepend=e[0]), 0.0)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) or 1e-9
    thr = med + k * 1.4826 * mad
    fps = 1000.0 / win_ms
    gap = max(1, int(round(0.001 * gap_ms * fps)))
    pk: List[int] = []
    last = -10 ** 9
    for i in range(1, d.size - 1):
        if d[i] >= thr and d[i] >= d[i - 1] and d[i] >= d[i + 1] and i - last >= gap:
            pk.append(i)
            last = i
    times = np.array([p / fps for p in pk], dtype=np.float64)
    prom = np.array([float(d[p]) for p in pk], dtype=np.float64)
    return (times, prom) if return_prominence else times


def mid_modulation(x: np.ndarray, sr: int, max_sec: float = 90.0) -> Tuple[float, float]:
    """
    Mede o quanto a energia da faixa média (300–3000 Hz) está **amarrada aos ataques**:
    `dyn` = P95/P25 (profundidade de modulação) e `cv` = coeficiente de variação do envelope.
    Bateria isolada é um trem de transientes com vales fundos → dyn alto (milhares em arquivo
    limpo, dezenas a centenas em gravação com ruído de sala). Um colchão sustentado (órgão,
    voz, baixo) preenche os vales e derruba dyn; foi medido em `tests/calibrate_tonal.py`
    (ver `samples/tonal_calibration.json`): −30 dB de pad já corta dyn de ~6·10⁴ para ~17.
    Não é classificador: é uma estatística com limiar conservador, para avisar que a
    premissa "faixa isolada" pode não valer.
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size < 4096 or sr < 8000:
        return 0.0, 0.0
    x = x[: int(max_sec * sr)]
    n_fft, hop = 2048, 512
    frames = 1 + max(0, (x.size - n_fft) // hop)
    if frames < 32:
        return 0.0, 0.0
    w = np.hanning(n_fft).astype(np.float32)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    seg = x[np.clip(idx, 0, x.size - 1)] * w[None, :]
    P = np.abs(np.fft.rfft(seg, axis=1)) ** 2 + 1e-12
    fr = np.fft.rfftfreq(n_fft, 1.0 / sr)
    e = P[:, (fr >= 300.0) & (fr <= 3000.0)].sum(axis=1)
    p95, p25 = float(np.percentile(e, 95)), float(np.percentile(e, 25))
    dyn = p95 / max(p25, 1e-9)
    cv = float(np.std(e) / (np.mean(e) + 1e-9))
    return round(dyn, 2), round(cv, 3)


# =========================================================================================
# lei do copista: a partitura impressa lê como backbeat?
# =========================================================================================
def _beat_weights(ticks_per_bar: int, ticks_per_beat: int, beats: int) -> Dict[str, np.ndarray]:
    """Perfil de plausibilidade por (pista, posição). Escrito à mão, sem ver o `align_downbeat`."""
    def pos(b: int) -> int:
        return b * ticks_per_beat

    def kick(t: int) -> float:
        return {4: {0: 1.0, 8: 0.15, 16: 0.85, 24: 0.15},
                3: {0: 1.0, 8: 0.25, 16: 0.25},
                2: {0: 1.0, 8: 0.4},
                5: {0: 1.0, 8: 0.35, 16: 0.6, 24: 0.6},
                6: {0: 1.0, 12: 0.4, 24: 0.5, 36: 0.5}}.get(beats, {}).get(t, 0.20)

    def snare(t: int) -> float:
        return {4: {8: 1.0, 24: 1.0, 0: 0.1, 16: 0.1},
                3: {8: 1.0, 16: 0.4, 0: 0.15},
                2: {8: 1.0, 0: 0.2},
                5: {8: 1.0, 16: 0.5, 24: 0.3, 32: 0.5},
                6: {12: 1.0, 36: 1.0, 0: 0.2, 24: 0.3}}.get(beats, {}).get(t, 0.35)

    prof = {"kick": kick, "snare": snare,
            "cymbal": (lambda t: 1.0 if t % ticks_per_beat == 0 else 0.72)}
    fine: Dict[str, np.ndarray] = {}
    for name, fn in prof.items():
        g = np.array([fn(pos(b)) if pos(b) < ticks_per_bar else 0.12 for b in range(beats)],
                     dtype=np.float64)
        # cada tick herda o peso do seu beat (é a lei do copista, não uma curva ajustada)
        fine[name] = np.array([max(0.12, float(g[min(t // ticks_per_beat, beats - 1)]))
                               for t in range(ticks_per_bar)])
    return fine


def backbeat_score(hits: Sequence[dict], ticks_per_bar: int, ticks_per_beat: int, beats: int,
                   shift: int = 0) -> Tuple[float, int]:
    """Média de plausibilidade, ponderada por confiança e força; (score, nº de golpes contados)."""
    if not hits:
        return 0.0, 0
    w = _beat_weights(ticks_per_bar, ticks_per_beat, beats)
    num = den = 0.0
    n = 0
    for h in hits:
        lane = str(h.get("lane", ""))
        arr = w.get("cymbal") if lane in _CYMBAL else w.get(lane)
        if arr is None:
            continue
        tk = (int(round(_f(h.get("tick")))) + shift) % ticks_per_bar
        wt = max(0.05, _f(h.get("confidence"), 0.8)) * (0.5 + 0.5 * _f(h.get("velocity"), 90.0) / 127.0)
        num += wt * float(arr[tk])
        den += wt
        n += 1
    return (float(num / den) if den else 0.0), n


# =========================================================================================
# verificadores
# =========================================================================================
def check_tempo(score: dict, report: dict, x: Optional[np.ndarray], sr: int) -> List[dict]:
    bpm = _f(score.get("bpm"))
    tm = (report or {}).get("tempo", {}) or {}
    if bpm <= 0:
        return [_find("T1", "andamento", "BPM ausente", "error", a=bpm, state="manual")]
    out: List[dict] = []
    if x is not None and x.size > 8192 and sr > 4000:
        env, fps = oracle_envelope(x, sr)
        ob = oracle_bpm(env, fps, _f((tm.get("bpm_range") or [45])[0], 45.0),
                        _f((tm.get("bpm_range") or [0, 220])[1], 220.0))
        t_all = np.array([_f(h.get("time")) for h in _hits_of(score) if h.get("time") is not None],
                         dtype=np.float64)
        cand = {lbl: round(grid_fit_pure(t_all, b), 4)
                for lbl, b in (("½×", bpm / 2.0), ("1×", bpm), ("2×", bpm * 2.0))}
        dev, ratio = octave_dev(bpm, ob["bpm"])
        # a ambiguidade de oitava é inerente a qualquer método espectral: o oráculo só pode
        # acusar erro quando nenhuma razão ½×/2×/3× explica a diferença
        sev = "info" if dev <= 2.0 else ("warn" if dev <= 6.0 else "error")
        out.append(_find(
            "T1", "andamento", "BPM: pipeline × oráculo de envelope RMS", sev,
            a=round(bpm, 2), b=round(ob["bpm"], 2),
            detail="desvio módulo oitava %.2f %% (razão %.3f) · ajuste de grade ½×/1×/2× = "
                   "%.3f/%.3f/%.3f · picos do oráculo %s" % (dev, ratio, cand["½×"], cand["1×"],
                                                             cand["2×"], ob["peaks"][:3]),
            state="ok" if sev == "info" else "manual",
            data={"dev_pct": dev, "grid_fit": cand, "method": ob["method"],
                  "confidence": ob["confidence"]}))
    else:
        out.append(_find("T1", "andamento", "BPM: oráculo indisponível (sem áudio decodificado)",
                         "info", a=round(bpm, 2), b=None, state="skipped"))
    conf = _f(tm.get("confidence"))
    if conf and conf < 0.35:
        out.append(_find("T1b", "andamento", "Confiança de andamento baixa (%.2f)" % conf, "warn",
                         a=conf, detail="a periodicidade ficou ambígua. Confira o BPM e reanalise, "
                                        "ou trave com `bpm_lock`.", state="manual"))
    return out


def check_meter(score: dict, report: dict) -> List[dict]:
    hits = _hits_of(score)
    tpb = int(_f(score.get("ticks_per_beat", 8), 8.0)) or 8
    tpr = int(_f(score.get("ticks_per_bar", 32), 32.0)) or 32
    beats = int(_f(score.get("beats_per_bar", 4), 4.0)) or 4
    out: List[dict] = []
    try:
        info = meter_info(str(score.get("meter", "4/4")))
        if int(info["ticks_per_bar"]) != tpr or int(info["beats"]) != beats:
            out.append(_find("T2a", "compasso", "Metadado de compasso inconsistente", "error",
                             a={"ticks_per_bar": tpr, "beats_per_bar": beats},
                             b={"ticks_per_bar": int(info["ticks_per_bar"]),
                                "beats_per_bar": int(info["beats"])},
                             detail="o dicionário da partitura não bate com `score_model.METERS`",
                             state="manual"))
    except Exception as e:
        out.append(_find("T2a", "compasso", "Fórmula de compasso desconhecida", "error",
                         a=str(score.get("meter")), detail=repr(e), state="manual"))
    base, n = backbeat_score(hits, tpr, tpb, beats, 0)
    best_s, best_r = base, 0
    for r in range(1, beats):
        s, _ = backbeat_score(hits, tpr, tpb, beats, r * tpb)
        if s > best_s:
            best_s, best_r = s, r
    margin = best_s - base
    off = int(_f((report or {}).get("meter", {}).get("tick_offset")))
    if best_r:
        sev = "warn" if margin > 0.06 else "info"
        out.append(_find(
            "T2", "compasso", "Posição da barra × lei do backbeat", sev,
            a={"ajuste": round(base, 4), "tick_offset": off},
            b={"ajuste": round(best_s, 4), "mover_ticks": int(best_r * tpb)},
            detail="girar o ponto de barra %d tempo(s) (%+d ticks) melhora a leitura de backbeat em "
                   "%.3f (%.4f → %.4f), sobre %d golpes pontuados. Em 4/4 com bumbo em 1–3 e caixa em "
                   "2–4 a evidência pode ser simétrica de verdade: por isso é alerta, não erro."
                   % (best_r, best_r * tpb, margin, base, best_s, n),
            fix="rotate_downbeat" if margin > 0.06 else None,
            state="ok" if sev == "info" else "manual",
            data={"shift_ticks": int(best_r * tpb), "margin": round(float(margin), 4)}))
    else:
        out.append(_find("T2", "compasso", "Barra coerente com a lei do backbeat", "info",
                         a=round(base, 4), b=round(base, 4),
                         detail="nenhuma rotação de 1..%d tempo(s) melhora a leitura (ajuste %.4f, "
                                "%d golpes)" % (beats - 1, base, n), state="ok"))
    nb = len(score.get("bars", []) or [])
    if nb < 2:
        out.append(_find("T2b", "compasso", "Partitura com %d compasso(s)" % nb, "warn",
                         detail="abaixo de ~8 s de música o andamento e a fórmula são pouco confiáveis",
                         state="manual"))
    return out


def check_grid(score: dict, report: dict, dets: Optional[Sequence[dict]] = None) -> List[dict]:
    """
    Confere a malha sem confiar em nada do que o pipeline gravou: refaz o caminho
    `tempo do golpe → posição na grade` a partir do BPM, da fase e do offset de downbeat
    declarados no relatório e exige que `(compasso, tick)` da partitura saia do mesmo.

    Duas leis separadas, porque são falhas diferentes:
      * **rigidez** — um golpe longe demais do slot escrito só pode ser nota no lugar errado;
        resíduo humano de ±1,5 tick é normal (o pipeline anuncia `tol_ms`), logo o limite é
        derivado dessa tolerância e não de um número inventado;
      * **deriva** — resíduo crescendo linearmente com o compasso = BPM ligeiramente errado
        (o erro acumula); nenhum dos dois aparece no teste de rigidez, e é o modo como um
        andamento "quase certo" estraga a partitura no fim da música.
    """
    grid = (report or {}).get("grid", {}) or {}
    met = (report or {}).get("meter", {}) or {}
    spt = _spt(score)
    tpb = int(_f(score.get("ticks_per_beat", 8), 8.0)) or 8
    tpr = int(_f(score.get("ticks_per_bar", 32), 32.0)) or 32
    lead = _f((report or {}).get("file", {}).get("trim_lead_ms")) / 1000.0
    hits = _hits_of(score)
    tol_ticks = max(1.0, _f(grid.get("tol_ms"), 0.0) / (1000.0 * spt)) if spt > 0 else 1.5
    clk = clock_map(hits, _f(score.get("bpm"), 120.0), tpr, report)
    out: List[dict] = []

    # ------------------------------------------------------------------------ T3  malha × relógio
    if clk:
        rows = clk["rows"]                                  # (desvio já sem origem, golpe)
        r = np.array([x for x, _h in rows])
        lim = grid_bound_ticks(score, report)
        bad = [{"bar": int(h["bar"]) + 1, "tick": int(round(_f(h.get("tick")))), "lane": h.get("lane"),
                 "desvio_ticks": round(float(x), 3)} for x, h in rows if abs(float(x)) > lim]
        # deriva: resíduo crescendo com o número do compasso = BPM ligeiramente errado.
        # Este é o único modo de pegar "andamento quase certo": a rigidez ponto a ponto
        # continua linda nos primeiros compassos e o erro só aparece no fim da música.
        bars_i = np.array([int(h["bar"]) for _x, h in rows], dtype=np.float64)
        slope = 0.0
        if bars_i.size > 8 and float(np.std(bars_i)) > 1e-6:
            slope = float(np.polyfit(bars_i, r, 1)[0])
        nbars = int(np.ptp(bars_i)) + 1
        drift_ticks = abs(slope) * max(1, nbars)
        drift_bad = drift_ticks > max(0.6, tol_ticks * 0.5)
        extra = "" if not drift_bad else (
            f" E a malha ESCOREGA: o resíduo cresce {slope:.3f} tick por compasso "
            f"({drift_ticks:.2f} ticks em {nbars} compassos) — assinatura de BPM levemente "
            f"errado, não de execução humana.")
        out.append(_find(
            "T3", "grade", "Malha escrita × relógio do áudio (rigidez e deriva)",
            "error" if (bad or drift_bad) else "info",
            a={"dispersão_ticks": round(float(np.std(r)), 3), "máx": round(float(np.max(np.abs(r))), 3),
               "deriva_tick_por_compasso": round(slope, 4), "n": int(r.size)},
            b={"tol_declarada_ticks": round(tol_ticks, 2), "fora": len(bad),
               "limite_ticks": round(lim, 2), "deriva_max_ticks": round(max(0.6, tol_ticks * 0.5), 2)},
            detail=(f"refaz pos = (t − lead − fase)/spt {int(clk['sign']):+d}·tick_offset e compara com "
                    f"compasso·{tpr} + tick, para cada golpe. A origem da numeração (mediana "
                    f"{clk['origin']:.3f} ticks ≈ {clk['origin'] * spt * 1000.0:.0f} ms) é removida de propósito: "
                    f"o primeiro compasso gravado pode começar em qualquer barra, o que vale é a malha ser "
                    f"rígida. Dispersão {float(np.std(r)):.3f} ticks, máx {float(np.max(np.abs(r))):.3f}, limite "
                    f"{lim:.2f} ticks (= {lim / tol_ticks:.2f}× os {tol_ticks:.2f} ticks de tolerância que você "
                    f"anunciou em grid.tol_ms, para não marcar timing humano como erro).{extra}"
                    + (f"; ofensores: {bad[:3]}" if bad else "")),
            state="ok" if (not bad and not drift_bad) else "manual",
            fix="requantize_from_clock" if (bad or drift_bad) else None,
            data={"disp": round(float(np.std(r)), 3), "slope": round(slope, 5),
                  "drift_ticks": round(drift_ticks, 3), "offenders": bad[:40],
                  "limite_ticks": round(lim, 3)}))
    else:
        out.append(_find("T3", "grade", "Malha escrita × relógio do áudio", "info",
                         detail="menos de 8 notas com `time` na partitura — não há como re-derivar "
                                "a grade de forma estatística; nada a conferir",
                         state="skipped"))

    # ---------------------------------- T3e: o `time` gravado precisa existir dentro do arquivo
    dur = _f((report or {}).get("file", {}).get("duration_sec"))
    if dur > 0:
        out_of = [{"bar": int(h["bar"]) + 1, "tick": int(round(_f(h.get("tick")))), "lane": h.get("lane"),
                   "time": round(_f(h.get("time")), 4)}
                  for h in hits if h.get("time") is not None and not (-0.02 <= _f(h["time"]) <= dur + 0.02)]
        out.append(_find(
            "T3e", "grade", "Tempo declarado dentro do arquivo", "error" if out_of else "info",
            a=len(out_of), b=round(dur, 2),
            detail=(f"cada nota carrega o instante do golpe que a gerou; fora de [0, {dur:.2f} s] esse "
                    f"instante não descreve nada no áudio e o par (som ↔ nota) perde o sentido — "
                    f"desalinha o playhead, a sobreposição de forma de onda e a audição de conferência. "
                    f"O reparo reescreve o tempo a partir da posição gravada (a nota não se move)."
                    + (f"; exemplos: {out_of[:3]}" if out_of else "")),
            state="ok" if not out_of else "auto",
            fix="resync_time" if out_of else None,
            data={"fora": out_of[:20], "dur": dur}))

    # ---------------------- T3d: o resíduo que o pipeline anunciou tem de caber na tolerância dele
    tol = _f(grid.get("tol_ms"), -1.0)
    rs = [abs(_f(d.get("resid_ms"))) for d in (dets or []) if d.get("resid_ms") is not None]
    if tol > 0 and rs:
        n_out = int(sum(1 for v in rs if v > tol + 0.6))
        out.append(_find("T3d", "grade", "Resíduo de quantização × tolerância anunciada",
                         "error" if n_out else "info",
                         a={"max_ms": round(max(rs), 1), "tol_ms": round(tol, 1)}, b={"fora": n_out},
                         detail=f"{n_out} de {len(rs)} eventos com resíduo acima da tolerância que o próprio "
                                f"pipeline anunciou ({tol:.1f} ms; máx medido {max(rs):.1f} ms) — fora dela, "
                                f"a nota foi escrita num slot que o áudio não sustenta",
                         state="ok" if not n_out else "manual",
                         fix="requantize" if n_out else None))

    # ------------------------------- T3b: posições fora do passo do modo só se forem anunciadas
    mode = str(grid.get("mode") or "")
    # A régua vem da MESMA função que o quantizador usa para arredondar o offset de downbeat:
    # enquanto houve duas contas, um 6/8 real escreveu 2999 notas a 2 ticks do passo (A21).
    try:
        from .grid import grid_positions, passo_grade
        _sw = _f(grid.get("swing"), 0.0)
        step = int(round(passo_grade(grid_positions(tpb, mode, _sw), tpb))) if mode else None
        if step is not None and step < 1:
            step = 1
    except Exception:
        step = None
    if step is None:
        step = {"4th": tpb, "8th": max(1, tpb // 2), "16th": max(1, tpb // 4),
                "32nd": max(1, tpb // 8), "32nd+16th": max(1, tpb // 8)}.get(mode)
    ticks = np.array([int(_f(h.get("tick"))) for h in hits], dtype=np.int64)
    if step and ticks.size:
        off_grid = int(np.sum(ticks % step != 0))
        prom = int(_f(grid.get("promoted_32nd"), 0.0))
        declared_odd = int(_f(grid.get("odd_ticks"), -1.0))
        swing = _f(grid.get("swing"), 0.0)
        odd_real = int(np.sum(ticks % 2 != 0))
        ok = (off_grid <= prom) or (swing > 0.02 and off_grid <= prom + int(0.5 * ticks.size))
        consistent = declared_odd < 0 or declared_odd == odd_real
        out.append(_find("T3b", "grade", f"Golpes fora do passo de {mode} explicados por promoção",
                         "error" if not (ok and consistent) else "info", a=off_grid, b=prom,
                         detail=f"passo do modo {mode} = {step} ticks; {off_grid} de {ticks.size} golpes fora "
                                f"dele ({100.0 * off_grid / max(1, ticks.size):.1f} %); o relatório declara "
                                f"{prom} promoções a 32º e {declared_odd} ticks ímpares (recontado aqui: "
                                f"{odd_real}), swing {swing:.2f}. Nota fora do passo só é lícita como promoção "
                                f"anunciada ou posição de swing — senão é escrita irregular",
                         state="ok" if (ok and consistent) else "manual",
                         fix="requantize" if not ok else None,
                         data={"off_grid": off_grid, "promoted": prom,
                               "ratio": round(off_grid / max(1, ticks.size), 3)}))

    # --------------------------------------------- T3c: ajuste reportado × ajuste de pulso puro
    fit = _f(grid.get("fitness"), -1.0)
    if fit >= 0:
        pure = round(grid_fit_pure(
            np.array([_f(h.get("time")) - lead for h in hits if h.get("time") is not None]),
            _f(score.get("bpm"), 120.0), sub=0.5 if "16" in mode else 1.0), 4)
        out.append(_find("T3c", "grade", "Ajuste reportado × ajuste de pulso puro", "info", a=fit, b=pure,
                         detail="medidas diferentes por construção: a sua pondera a grade inteira do modo "
                                "escolhido, a do auditor só a distância ao pulso. Divergir não é erro; valor "
                                "baixo nas duas é que significa grade explicando pouco do áudio",
                         state="ok" if min(fit, pure) > 0.5 else "manual"))
    return out


# =========================================================================================
# leis de escrita da partitura (não dependem de áudio)
# =========================================================================================
def _lane_ids(score: dict) -> set:
    return set(score.get("lanes") or []) | set(LANE_BY_ID.keys())


def _note_values_ok(dur: int, tpb: int, tpr: int) -> bool:
    """`dur` tem de ser um valor imprimível (nota ± ponto) que caiba no compasso."""
    from .rules import dur_label
    if dur < 1 or dur > tpr:
        return False
    name, dots = dur_label(int(dur), tpb)
    base = {8: 8, 4: 4, 2: 2, 1: 1, 12: 12, 6: 6, 3: 3}.get(tpb, tpb)
    # aceita qualquer potência de 2 (ou 3×/1,5× dela) e o modo composto do tpb
    for b in ({1, 2, 4, 8, 16, 32} if tpb % 8 == 0 else {3, 6, 12, 24}):
        if dur in (b, int(b * 1.5), int(b * 1.75)):
            return True
    return bool(name) and dots <= 2 and dur <= tpr


def check_notation(score: dict, report: Optional[dict] = None) -> List[dict]:
    """
    Leis de escrita — invariantes estruturais da partitura. Cada uma tem uma justificativa de
    impressão: um valor que não existe, uma voz que se sobrepõe, uma ligadura sem destino ou um
    𝄄 sobre compasso diferente é erro de notação, independentemente de estar ou não "certo"
    no áudio.
    """
    out: List[dict] = []
    hits = _hits_of(score)
    tpr = int(_f(score.get("ticks_per_bar", 32), 32.0)) or 32
    tpb = int(_f(score.get("ticks_per_beat", 8), 8.0)) or 8
    n = len(hits)
    if n == 0:
        return [_find("T4", "notação", "Partitura vazia", "warn",
                      detail="nenhuma nota para conferir", state="manual")]

    # ------------------------------------------------------------------- T4  duplicata por posição
    seen: Dict[Tuple[int, int, str], int] = {}
    for h in hits:
        k = (int(h["bar"]), int(round(_f(h.get("tick")))), str(h.get("lane")))
        seen[k] = seen.get(k, 0) + 1
    dup = [{"bar": b + 1, "tick": t, "lane": ln, "n": c} for (b, t, ln), c in seen.items() if c > 1]
    out.append(_find("T4", "notação", "Duplicatas (compasso,tick,peça)", "error" if dup else "info",
                     a=len(dup), b=0,
                     detail="duas notas da mesma peça no mesmo slot são uma nota só com a cabeça errada: "
                            "o MusicXML vira acorde impossível e a reprodução toca o golpe duas vezes"
                            + (f"; {dup[:3]}" if dup else ""),
                     state="ok" if not dup else "auto",
                     fix="dedupe" if dup else None, data={"dup": dup[:20]}))

    # ---------------------------------------- T5  voz monofônica: nada invade o ataque seguinte
    over: List[dict] = []
    by_voice: Dict[Tuple[int, int], List[dict]] = {}
    for h in hits:
        v = int(_f(LANE_BY_ID[h["lane"]].voice, 2)) if h.get("lane") in LANE_BY_ID else 2
        by_voice.setdefault((int(h["bar"]), v), []).append(h)
    for (b, v), lst in by_voice.items():
        lst = sorted(lst, key=lambda x: (_f(x.get("tick")), str(x.get("lane"))))
        onsets = [int(round(_f(x.get("tick")))) for x in lst]
        uniq = sorted({int(round(_f(x.get("tick")))) for x in lst})
        for i, h in enumerate(lst):
            tk = onsets[i]
            dur = max(1, int(_f(h.get("dur"), 1)))
            after = [u for u in uniq if u > tk]
            nxt = after[0] if after else tpr          # simultâneo (acorde) não é invasão
            if tk + dur > nxt and not h.get("tie_start"):
                other = next((x for x in lst if int(round(_f(x.get("tick")))) == (after[0] if after else tpr)),
                             None)
                over.append({"bar": b + 1, "voz": v, "nota": f"{h.get('lane')}@{tk}", "dur": dur,
                             "invade": (f"{other.get('lane')} tick {onsets[i + 1]}" if other
                                        else "a barra seguinte")})
    out.append(_find("T5", "notação", "Sobreposição na mesma voz", "error" if over else "info",
                     a=len(over), b=0,
                     detail="na pauta de bateria a voz 1 (pratos) é monofônica e nenhum ataque pode cair "
                            "dentro da duração escrita da nota anterior na mesma voz — a impressora teria de "
                            "escrever duas hastes no mesmo lugar" + (f"; {over[:3]}" if over else ""),
                     state="ok" if not over else "auto",
                     fix="clip_durations" if over else None, data={"over": over[:20]}))

    stop_at = {(int(h["bar"]), int(round(_f(h.get("tick"))))) for h in hits if h.get("tie_stop")}
    start_at = {(int(h["bar"]), int(round(_f(h.get("tick"))))) for h in hits if h.get("tie_start")}

    # --------------------------------------------------------- T6  campos obrigatórios e faixas
    lanes_ok = _lane_ids(score)
    bad: List[dict] = []
    for h in hits:
        tk = _f(h.get("tick"), -1)
        dur = _f(h.get("dur"), 0)
        vel = _f(h.get("velocity"), 0)
        art = str(h.get("artic", "normal"))
        lane = str(h.get("lane"))
        why = []
        if lane not in LANE_BY_ID:
            why.append("pista desconhecida")
        elif lane not in set(score.get("lanes") or lanes_ok):
            why.append("pista fora do cabeçalho da parte")
        if tk < 0 or tk >= tpr or abs(tk - round(tk)) > 1e-6:
            why.append("tick fora do compasso")
        if dur < 1:
            why.append("duração ≤ 0")
        if vel < 1 or vel > 127:
            why.append("velocidade fora de 1..127")
        if art not in _ARTIC:
            why.append(f"articulação inválida: {art}")
        if art in ("ghost", "ghost_accent") and lane not in _GHOST_OK:
            why.append("fantasma em peça que não admite cabeça −")
        if why:
            bad.append({"bar": int(h["bar"]) + 1, "tick": int(tk), "lane": lane, "motivo": ", ".join(why)})
    out.append(_find("T6", "notação", "Campos obrigatórios e faixas", "error" if bad else "info",
                     a=len(bad), b=0,
                     detail="tick ∈ [0, compasso), duração ≥ 1, velocidade 1..127, articulação do alfabeto "
                            "({}) e pista conhecida — fora disso o export é indefinido".format("·".join(_ARTIC))
                            + (f"; {bad[:3]}" if bad else ""),
                     state="ok" if not bad else "auto",
                     fix="normalize_fields" if bad else None, data={"bad": bad[:30]}))

    # ------------------------------------------------------- T7  o valor escrito existe na notação
    bad_dur = [{"bar": int(h["bar"]) + 1, "tick": int(round(_f(h.get("tick")))), "dur": int(_f(h.get("dur"), 0)),
                "lane": h.get("lane")}
               for h in hits if not _note_values_ok(int(round(_f(h.get("dur"), 1))), tpb, tpr)]
    out.append(_find("T7", "notação", "Valores de duração imprimíveis", "error" if bad_dur else "info",
                     a=len(bad_dur), b=0,
                     detail="toda duração precisa ser nota + pontos que caibam no compasso (com tpq = "
                            f"{TICKS_PER_QUARTER}, os valores base são {sorted({1, 2, 4, 8, 16})} ticks e "
                            f"{tpb} por tempo); valor não representável vira figura errada no Sibelius"
                            + (f"; {bad_dur[:3]}" if bad_dur else ""),
                     state="ok" if not bad_dur else "auto",
                     fix="requantize" if bad_dur else None, data={"bad": bad_dur[:20]}))

    # --------------------------------- T8  ligaduras: start/stop aos pares, dentro da mesma voz
    #     A duração que importa é a da VOZ: uma ligadura só existe se houver, na mesma voz, a
    #     nota de destino exatamente no tick em que a origem termina (ou no tick 0 do compasso
    #     seguinte, quando ela atravessa a barra).
    tie: List[dict] = []
    for (b, v), lst in by_voice.items():
        ends = {(int(round(_f(x.get("tick")))), True) for x in lst if x.get("tie_stop")}
        starts_by_bar = {}
        for x in lst:
            if x.get("tie_stop"):
                starts_by_bar.setdefault(b, set()).add(int(round(_f(x.get("tick")))))
        for nb2, l2 in by_voice.items():
            if nb2[1] != v:
                continue
            for x in l2:
                if x.get("tie_stop"):
                    starts_by_bar.setdefault(nb2[0], set()).add(int(round(_f(x.get("tick")))))
        ord_abs = sorted(lst, key=lambda x: (int(x["bar"]), int(round(_f(x.get("tick"))))))
        for i2, h in enumerate(ord_abs):
            tk = int(round(_f(h.get("tick"))))
            dur = max(1, int(_f(h.get("dur"), 1)))
            end = tk + dur
            if h.get("tie_start"):
                # destino: um stop exatamente onde a nota termina, OU a nota seguinte da mesma
                # voz abrindo corrente (sustenus de prato atravessam barras assim)
                nxt = ord_abs[i2 + 1] if i2 + 1 < len(ord_abs) else None
                ok = ((b, end) in stop_at if end < tpr else
                      (b + 1 + (end - tpr) // tpr, (end - tpr) % tpr) in stop_at)
                if not ok and nxt is not None:
                    na = int(nxt["bar"]) * tpr + int(round(_f(nxt.get("tick"))))
                    ha = int(h["bar"]) * tpr + tk
                    ok = bool(nxt.get("tie_stop") or nxt.get("tie_start")) and na >= ha + dur
                    if nxt.get("tie_stop") and na != ha + dur:
                        ok = False
                if not ok:
                    tie.append({"voz": v, "início": f"c{b + 1} tick {tk} ({h.get('lane')})",
                                "motivo": f"início de ligadura sem destino na voz {v} (a nota termina em "
                                          f"{end} ticks e não há stop ali)"})
            if h.get("tie_stop"):
                prev = ((b, tk) in start_at) or ((b - 1, tk + tpr) in start_at)
                if not prev:          # pode vir de uma corrente: nota anterior da voz com tie_start
                    j2 = ord_abs.index(h)
                    if j2 > 0 and ord_abs[j2 - 1].get("tie_start"):
                        pa = int(ord_abs[j2 - 1]["bar"]) * tpr + int(round(_f(ord_abs[j2 - 1].get("tick"))))
                        pae = pa + max(1, int(_f(ord_abs[j2 - 1].get("dur"), 1)))
                        prev = pae >= b * tpr + tk
                if not prev:
                    tie.append({"voz": v, "fim": f"c{b + 1} tick {tk} ({h.get('lane')})",
                                "motivo": f"fim de ligadura sem início correspondente na voz {v}"})
    out.append(_find("T8", "notação", "Ligaduras alternam e têm destino", "error" if tie else "info",
                     a=len(tie), b=0,
                     detail="start sem stop (ou stop sem start) faz o notador ignorar a ligadura ou acusar "
                            "erro na importação — e a duração soante da nota deixa de bater com a pauta"
                            + (f"; {tie[:3]}" if tie else ""),
                     state="ok" if not tie else "auto", fix="drop_dangling_ties" if tie else None,
                     data={"tie": tie[:20]}))

    # ------------------------------------------------- T9  𝄄 (repeat slash) só sobre repetição fiel
    slash_bad: List[dict] = []
    bars = score.get("bars", []) or []
    def _sig(bar: dict) -> List[Tuple[int, str, int]]:
        return sorted((int(round(_f(h.get("tick")))), str(h.get("lane")), int(_f(h.get("velocity"), 90)))
                      for h in (bar.get("hits") or []))
    for i, bar in enumerate(bars):
        if not bar.get("repeat_slash"):
            continue
        if i == 0:
            slash_bad.append({"bar": int(_f(bar.get("index"))) + 1, "motivo": "𝄄 no primeiro compasso"})
            continue
        cur, prev = _sig(bar), _sig(bars[i - 1])
        if len(cur) != len(prev):
            slash_bad.append({"bar": int(_f(bar.get("index"))) + 1,
                              "motivo": f"não é repetição fiel ({len(cur)} vs {len(prev)} notas)"})
            continue
        far = [1 for (t1, l1, v1), (t2, l2, v2) in zip(cur, prev)
               if (t1, l1) != (t2, l2) or abs(v1 - v2) > 12]
        if far:
            slash_bad.append({"bar": int(_f(bar.get("index"))) + 1,
                              "motivo": f"{len(far)} notas diferem da barra anterior em posição/peça/força"})
    out.append(_find("T9", "notação", "𝄄 só sobre repetição fiel", "error" if slash_bad else "info",
                     a=len(slash_bad), b=0,
                     detail="o sinal de repetição de compasso manda o baterista tocar de novo o que está "
                            "escrito na barra anterior; marcar 𝄄 onde as notas mudam apaga informação real"
                            + (f"; {slash_bad[:3]}" if slash_bad else ""),
                     state="ok" if not slash_bad else "auto",
                     fix="unmark_slash" if slash_bad else None,
                     data={"bars": sorted({int(x["bar"]) - 1 for x in slash_bad})}))

    # ---------------------- T10  compasso completo por voz: notas + pausas = exatamente a barra
    from .rules import fill_voice
    incomplete: List[dict] = []
    for bar in bars:
        bi = int(_f(bar.get("index")))
        hs = list(bar.get("hits") or [])
        for v in (1, 2):
            lst2 = [x for x in hs if (LANE_BY_ID[str(x.get("lane"))].voice
                                      if str(x.get("lane")) in LANE_BY_ID else 2) == v]
            try:
                items = fill_voice(lst2, tpr, tpb)
                tot = sum(int(_f(it.get("dur"), 0)) for it in items)
            except Exception:
                continue
            if tot != tpr:
                incomplete.append({"bar": bi + 1, "voz": v, "soma_ticks": tot})
    out.append(_find("T10", "notação", "Todo compasso preenchido por voz", "error" if incomplete else "info",
                     a=len(incomplete), b=tpr,
                     detail="nota + pausa pontuada têm de somar exatamente um compasso por voz; se sobrarem "
                            "ticks a impressora insere pausa invisível e o tempo desloca para o resto da peça"
                            + (f"; {incomplete[:3]}" if incomplete else ""),
                     state="ok" if not incomplete else "auto",
                     fix="requantize" if incomplete else None, data={"bars": incomplete[:20]}))
    return out


# =========================================================================================
# leitura independente dos exports (compilação dupla)
# =========================================================================================
def _xml_measures(xml: str) -> dict:
    """
    Relê o MusicXML com `xml.etree` (nada do nosso escritor) e devolve o que um notador veria:
    nº de compassos, divisões, soma de ticks por voz em cada compasso, notas, acordes,
    ligaduras e o andamento declarado em <sound>.
    """
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    attrs = {"n_measures": 0, "divisions": 0, "notes": 0, "chords": 0, "rests": 0, "graces": 0,
             "ties": {"start": 0, "stop": 0}, "tuplets": 0, "tempo": 0.0, "voice_max": 0,
             "bad_measures": [], "pos": {}, "metronome": 0.0, "beat_unit": "", "voice_sum": {}}
    part = root.find("part") if root.tag == "score-partwise" else root.find("score-partwise/part")
    if part is None:                                   # id="P1" e afins: procura com namespaces
        for el in root.iter():
            if el.tag.endswith("}part") or el.tag == "part":
                part = el
                break
    if part is None:
        return attrs
    div = part.find("measure/attributes/divisions")
    attrs["divisions"] = int(_f(div.text if div is not None else 0))
    divs = max(1, attrs["divisions"])
    tpq = TICKS_PER_QUARTER
    scale = divs / float(tpq)
    for m in part.findall("measure"):
        attrs["n_measures"] += 1
        per: Dict[int, int] = {}
        pos: Dict[int, List[int]] = {}
        for nd in m.findall("note"):
            # Adorno de flam é `<note type="grace">`: não é nota tocada nem slot de dados. Sem
            # contá-lo à parte, o total do arquivo discordava da partitura por causa dos flans
            # (medido: 2163 notas no XML contra 2160 slots numa faixa com 3 flans — A23).
            if (nd.get("type") or "") == "grace":
                attrs["graces"] += 1
                continue
            is_chord = nd.find("chord") is not None
            if is_chord:
                attrs["chords"] += 1
            if nd.find("rest") is not None:
                attrs["rests"] += 1
            elif not is_chord:
                attrs["notes"] += 1
            is_chord = nd.find("chord") is not None
            vt = nd.find("voice")
            v = int(_f(vt.text if vt is not None else 1, 1.0))
            dt = nd.find("duration")
            dx = int(_f(dt.text if dt is not None else 0))
            if is_chord:
                continue                      # <chord/> não avança o cursor da voz
            per[v] = per.get(v, 0) + dx
            if nd.find("rest") is None:
                pos.setdefault(v, []).append((per[v] - dx) / scale)   # pausa não é nota
            for t in nd.findall("tie"):
                ty = t.get("type")
                if ty in attrs["ties"]:
                    attrs["ties"][ty] += 1
            if nd.find("time-modification") is not None:
                attrs["tuplets"] += 1
        attrs["voice_max"] = max([attrs["voice_max"]] + list(per.values()))
        bi = int(_f(m.get("number"), 0)) - 1
        for v, tot in per.items():
            attrs["voice_sum"][f"{bi + 1}:{v}"] = int(round(tot / scale))
        for v, lst in pos.items():
            attrs["pos"].setdefault((bi, v), []).extend([float(p) for p in lst])
        for v, tot in per.items():
            if abs(tot / scale - round(tot / scale)) > 1e-9:
                attrs["bad_measures"].append({"bar": int(_f(m.get("number"), 0)), "voz": v,
                                               "ticks": round(tot / scale, 3)})
    for el in root.iter("sound"):
        if el.get("tempo"):
            attrs["tempo"] = _f(el.get("tempo"))
            break
    pm = root.find(".//metronome/per-minute")          # a indicação métronômica é outra fonte
    attrs["metronome"] = _f(pm.text if pm is not None else 0.0)
    bu = root.find(".//metronome/beat-unit")
    attrs["beat_unit"] = (bu.text.strip() if bu is not None and bu.text else "")
    return attrs


def _read_smf(data: bytes, notas: bool = False) -> dict:
    """
    Parser SMF mínimo e independente (não usa `midi_out`): cabeçalho, trilhas, divisão,
    notas no canal 10, tempo e marcadores.

    Com `notas=True` acumula também `[(tick_on, pitch, vel, dur_tick), …]` — é o que permite à UI
    mostrar o MIDI real do arquivo (não o nosso dicionário). As contagens que T12 julga são
    as mesmas de antes: a lista extra não toca em `notes`/`ch10`/`offs_missing`.
    """
    if data[:4] != b"MThd":
        raise ValueError("sem cabeçalho MThd")
    ln, fmt, ntrk, div = struct.unpack(">IHHH", data[4:14])
    out = {"format": fmt, "ntracks": ntrk, "division": div, "header_len": ln,
           "notes": 0, "ch10": 0, "tempo_us": 0, "markers": 0, "programs": 0,
           "max_tick": 0, "offs_missing": 0, "gm": set()}
    pend: dict = {}
    notas_l: list = []
    p, end = 14, len(data)
    tracks = []
    while p + 8 <= end:
        if data[p:p + 4] != b"MTrk":
            break
        tl = struct.unpack(">I", data[p + 4:p + 8])[0]
        tracks.append(data[p + 8:p + 8 + tl])
        p += 8 + tl
    out["ntracks_read"] = len(tracks)
    for ti, tr in enumerate(tracks):
        i, running, tt = 0, 0, 0
        while i < len(tr):
            d = 0
            while True:                            # delta VLQ
                b = tr[i]; i += 1
                d = (d << 7) | (b & 0x7F)
                if not b & 0x80:
                    break
            tt += d
            if i >= len(tr):
                break
            st = tr[i]
            if st & 0x80:
                i += 1
                running = st
            else:
                st = running
            typ, ch = st & 0xF0, st & 0x0F
            if st == 0xFF:
                mt = tr[i]; i += 1
                d2 = 0
                while True:
                    b = tr[i]; i += 1
                    d2 = (d2 << 7) | (b & 0x7F)
                    if not b & 0x80:
                        break
                payload = tr[i:i + d2]; i += d2
                if mt == 0x51:
                    out["tempo_us"] = int.from_bytes(payload[:3], "big")
                elif mt == 0x06:
                    out["markers"] += 1
            elif st in (0xF0, 0xF7):
                d2 = 0
                while True:
                    b = tr[i]; i += 1
                    d2 = (d2 << 7) | (b & 0x7F)
                    if not b & 0x80:
                        break
                i += d2
            elif typ in (0x90, 0x80, 0xA0, 0xB0, 0xE0):
                n = 2 if typ != 0xE0 else 2
                if typ == 0x90 and ch == 9:
                    k, v = tr[i], tr[i + 1]
                    if v > 0:
                        out["notes"] += 1
                        out["ch10"] += 1
                        out["gm"].add(int(k))
                        out["max_tick"] = max(out["max_tick"], tt)
                        if notas:
                            pend[(ti, k)] = (tt, v)
                    else:
                        out["offs_missing"] += 1          # tocar com nota-on de velocidade 0 é lícito
                        if notas and (ti, k) in pend:
                            t_on, v_on = pend.pop((ti, k))
                            notas_l.append((t_on, int(k), int(v_on), max(0, tt - t_on)))
                elif typ == 0x80 and ch == 9 and notas:
                    k = tr[i]
                    if (ti, k) in pend:
                        t_on, v_on = pend.pop((ti, k))
                        notas_l.append((t_on, int(k), int(v_on), max(0, tt - t_on)))
                elif typ == 0xC0:
                    pass
                else:
                    out["programs"] += 1
                i += n
            elif typ in (0xC0, 0xD0):
                if ti == 1:
                    out["programs"] += 1
                i += 1
            else:
                break
    out["gm"] = sorted(out["gm"])
    out["n_note_off_missing"] = out["offs_missing"]
    if notas:
        for (ti, k), (t_on, v_on) in sorted(pend.items()):
            notas_l.append((t_on, int(k), int(v_on), 0))    # sem note-off: duração 0, dito na UI
        notas_l.sort()
        out["notas"] = notas_l
        out["sem_off"] = sum(1 for x in notas_l if x[3] == 0)
    return out


def _pdf_pages(data: bytes) -> Tuple[int, int]:
    """(nº de páginas declarado, nº de objetos /Page) lidos do dicionário do PDF."""
    if data[:5] != b"%PDF-":
        return -1, -1
    declared = -1
    m = re.search(rb"/Type\s*/Pages[^>]*?/Count\s+(\d+)", data, re.S)
    if m:
        declared = int(m.group(1))
    else:
        m2 = re.search(rb"/Count\s+(\d+)", data)
        if m2:
            declared = int(m2.group(1))
    n_page = len(re.findall(rb"/Type\s*/Page\b(?!s)", data))
    return declared, n_page


def check_exports(score: dict, level: str = "full", lead: Optional[float] = None) -> List[dict]:
    """
    Regera cada export e lê o resultado com um parser que não é o do escritor. Uma partitura
    que só existe no nosso dicionário não vale nada: o que o baterista usa é o arquivo.
    """
    out: List[dict] = []
    hits = _hits_of(score)
    tpr = int(_f(score.get("ticks_per_bar", 32), 32.0)) or 32
    n = len(hits)
    nb = len(score.get("bars", []) or [])
    def _voice(h):
        return int(_f(LANE_BY_ID[str(h.get("lane"))].voice, 2.0)) if str(h.get("lane")) in LANE_BY_ID else 2
    # O que se compara com o arquivo é o que o escritor escreve. `hide_lanes` é apresentação (A16):
    # MIDI e MusicXML saem filtrados, o dicionário não. Se a expectativa viesse dos dados, ocultar
    # os pratos "quebrava" a auditoria de um resultado correto — medido: 2 erros falsos (T11 102/225
    # e T12 169/292) no mesmo resultado que, sem ocultação, dava 0. As leis de escrita (T1–T10)
    # continuam julgando TODAS as notas: é o que A16 promete.
    ocultas = [str(x) for x in (score.get("hide_lanes") or [])]
    arq_hits = _hits_of(score_visivel(score)) if ocultas else hits
    if ocultas and len(arq_hits) != n:
        out.append(_find("T10p", "export", "apresentação filtra o arquivo, não o resultado", "info",
                         detail="%d nota(s) nos dados, %d no que é gravado (pistas ocultas: %s). As "
                                "leis de escrita foram checadas sobre as %d; a releitura de arquivo, "
                                "sobre as %d — é assim que ocultar uma pista não pode virar erro de "
                                "transcrição." % (n, len(arq_hits), ", ".join(ocultas), n, len(arq_hits)),
                         a={"dados": n, "arquivo": len(arq_hits), "ocultas": ocultas},
                         b={"checks_de_dados": "T1–T10 sobre as %d" % n}, state="ok"))
    slots = {(int(h["bar"]), int(round(_f(h.get("tick")))), _voice(h)) for h in arq_hits}
    exp_notes = len(slots)      # uma nota primária por (compasso, tick, pauta)
    chords = len(arq_hits) - exp_notes      # as simultâneas da MESMA pauta entram como <chord/>

    # --------------------------------------------------------------------------- T11 MusicXML
    try:
        from .musicxml import to_musicxml
        xml = to_musicxml(score)
        r = _xml_measures(xml)
        probs: List[str] = []
        if r["divisions"] != TICKS_PER_QUARTER:
            probs.append(f"divisions {r['divisions']} ≠ {TICKS_PER_QUARTER} ticks/semínima")
        if r["n_measures"] != nb:
            probs.append(f"{r['n_measures']} compassos no XML vs {nb} na partitura")
        short = {k: v for k, v in (r.get("voice_sum") or {}).items() if v != tpr}
        if short:
            probs.append(f"{len(short)} voz(es) de compasso sem soma igual a {tpr} ticks: "
                         f"{list(short.items())[:3]}")
        if r["bad_measures"]:
            probs.append(f"voz com soma ≠ compasso cheio em {len(r['bad_measures'])} lugar(es): "
                         f"{r['bad_measures'][:3]}")
        if r["notes"] != exp_notes:
            probs.append(f"{r['notes']} notas no XML, esperado {exp_notes} (+{chords} de acorde)")
        _flams_decl = int(_f(((score.get("report") or {}).get("events") or {}).get("flams"), -1.0))
        # Direção da lei: o escrevedor não pode *inventar* adorno — cada `<note type="grace">` tem
        # de ter um flam nos dados. O contrário é lícito e acontece: um par de flam detectado pode
        # deixar de ser adorno na rejanela do editor (`rebuild_score`), e aí o arquivo tem menos
        # adornos que o relatório flans. Cobrar igualdade era cobrar do editor algo que ele não faz.
        if _flams_decl >= 0 and r["graces"] > _flams_decl:
            probs.append(f"{r['graces']} notas de adorno no arquivo contra {_flams_decl} flans nos "
                         f"dados — adorno sem flam declarado é escrita que o áudio não sustenta")
        if r["ties"]["start"] != r["ties"]["stop"]:
            probs.append(f"ligaduras {r['ties']['start']} início vs {r['ties']['stop']} fim")
        # posição: a linha do tempo reconstruída pelo leitor tem de dar os mesmos
        # (compasso, tick) da partitura — é assim que se vê nota deslocada/engolida
        def _voice_of(h):
            return int(_f(LANE_BY_ID[str(h.get("lane"))].voice, 2.0)) if str(h.get("lane")) in LANE_BY_ID else 2
        exp_pos: Dict[Tuple[int, int], List[float]] = {}
        for h in arq_hits:
            exp_pos.setdefault((int(h["bar"]), _voice_of(h)), []).append(_f(h.get("tick")))
        misplaced, exemplos = 0, []
        for k, lst in r["pos"].items():
            e = sorted({int(round(x)) for x in exp_pos.get(k, [])})
            g = sorted({int(round(x)) for x in lst})
            d = sorted(set(e) ^ set(g))
            if d:
                misplaced += len(d)
                exemplos.append({"bar": k[0] + 1, "voz": k[1], "ticks_divergentes": d[:6],
                                 "pauta": e[:8], "xml": g[:8]})
        if misplaced:
            probs.append(f"{misplaced} posição(ões) divergem entre o XML lido e a partitura (nota "
                         f"realocada, engolida por pausa ou fora da barra): {exemplos[:2]}")
        declared_t = int(sum(max(1, int(_f(h.get("dur"), 1))) for h in arq_hits))
        bpm_s = _f(score.get("bpm"))
        if bpm_s and abs(r["tempo"] - bpm_s) > 0.51:
            probs.append(f"<sound tempo> {r['tempo']} ≠ BPM {score.get('bpm')}")
        if r.get("metronome") and bpm_s and abs(r["metronome"] - bpm_s) > 0.51:
            probs.append(f"<per-minute> {r['metronome']} ≠ BPM {score.get('bpm')} (o metrônomo impresso "
                         f"diria outro andamento ao baterista)")
        if r["notes"] and r.get("beat_unit") and r["beat_unit"] != "quarter":
            probs.append(f"unidade do metrônomo = {r['beat_unit']!r}, mas tpq = {TICKS_PER_QUARTER} "
                         f"(seminima) — a indicacao precisa bater com a divisao")
        out.append(_find("T11", "export", "XML relido: notas, divisões e compassos completos por voz",
                         "error" if probs else "info",
                         a={"notas": r["notes"], "pausas": r["rests"], "acordes": r["chords"],
                            "adornos": r["graces"],
                            "compassos": r["n_measures"], "div": r["divisions"], "tupletos": r["tuplets"],
                            "ligaduras": r["ties"], "soma_máxima_voz": r["voice_max"], "tempo": r["tempo"]},
                         b={"notas_esperadas": exp_notes, "compassos": nb, "div": TICKS_PER_QUARTER,
                            "ticks_por_compasso": tpr, "durações_somadas": declared_t},
                         detail=("o leitor independente conta outra coisa que a partitura: "
                                 + "; ".join(probs) if probs else
                                 f"XML reescrito e relido: {r['n_measures']} compassos, {r['notes']} notas "
                                 f"(+{r['rests']} pausas, {r['chords']} em acorde, {r['graces']} adornos de "
                                 f"flam), divisões {r['divisions']}, "
                                 f"toda voz somando exatamente {tpr} ticks, {declared_t} ticks de duração no "
                                 f"total, tupletos {r['tuplets']} (só existem com swing), "
                                 f"andamento {r['tempo']:.1f} BPM, ligaduras pareadas"),
                         state="ok" if not probs else "manual", data={"probs": probs[:6]}))
    except Exception as e:
        out.append(_find("T11", "export", "MusicXML não releu", "error", detail=repr(e), state="manual"))

    if level != "full":
        out.append(_find("T12", "export", "MIDI/PDF/CSV na auditoria rápida", "info",
                         detail="releitura de SMF/PDF é cara; rode --level full (ou o servidor, que "
                                "audita rápido e oferece o full sob demanda)", state="skipped"))
        return out

    # --------------------------------------------------------------------------------- T12 MIDI
    try:
        from .midi_out import to_midi
        data = to_midi(score)
        m = _read_smf(data)
        probs = []
        if m["format"] not in (0, 1):
            probs.append(f"formato {m['format']}")
        if m["header_len"] != 6:
            probs.append(f"tamanho do bloco de dados do cabeçalho = {m['header_len']} (deve ser 6)")
        if m["division"] != 480:
            probs.append(f"divisão {m['division']} ≠ 480 por semínima")
        if m["ntracks_read"] != m["ntracks"]:
            probs.append(f"{m['ntracks_read']} trilhas lidas de {m['ntracks']} declaradas")
        if m["notes"] != len(arq_hits):
            probs.append("%d notas no canal 10 vs %d no que a gravura escreve (%d nos dados)"
                         % (m["notes"], len(arq_hits), n))
        bpm = _f(score.get("bpm"), 120.0)
        want_us = int(round(60.0e6 / max(1e-6, bpm)))
        if m["tempo_us"] and abs(m["tempo_us"] - want_us) > 2000:
            probs.append(f"meta-evento de tempo {m['tempo_us']} µs ≠ {want_us} µs (BPM {bpm:.2f})")
        scale = 480.0 / TICKS_PER_QUARTER
        # A régua do fim do arquivo é a GEOMETRIA ESCRITA, não os segundos medidos. A fórmula
        # antiga (`t_last × bpm / 60 × 480`) pressupunha que a barra 1 do arquivo cai no instante
        # `lead` exato e ignorava a fase da grade e o deslocamento de downbeat — por isso acusava
        # 8050 ticks numa faixa com 10,98 s de silêncio e 1208 ticks num groove onde o .mid estava
        # *exatamente* em 404 × 60, o último (compasso·tpr + tick) escrito (A22). Medir o arquivo
        # contra o que se gravou é a lei; medir contra relógio é outra coisa.
        tpr_x = int(_f(score.get("ticks_per_bar", 32), 32.0)) or 32
        if arq_hits:
            ult_escrito = max(int(h["bar"]) * tpr_x + int(round(_f(h.get("tick")))) for h in arq_hits)
            maior_dur = max(max(1, int(_f(h.get("dur"), 1))) for h in arq_hits)
        else:
            ult_escrito, maior_dur = 0, 0
        last_abs = int(round(ult_escrito * scale))
        fim_abs = int(round((ult_escrito + maior_dur) * scale))
        if m["notes"] and not (last_abs - 2 <= m["max_tick"] <= fim_abs + 2):
            probs.append("último evento MIDI no tick %d · a gravura escreve o último ataque em %d "
                         "(%d ticks absolutos × escala %g) e a nota mais longa dele dura %d ticks, "
                         "então o arquivo pode terminar no máximo em %d"
                         % (m["max_tick"], last_abs, ult_escrito, scale, maior_dur, fim_abs))
        out.append(_find("T12", "export", "Standard MIDI File relido", "error" if probs else "info",
                         a={"format": m["format"], "trilhas": m["ntracks_read"], "divisão": m["division"],
                            "notas_ch10": m["notes"], "tempo_us": m["tempo_us"], "marcadores": m["markers"],
                            "gm_usadas": len(m["gm"]), "fim": m["max_tick"]},
                         b={"notas": len(arq_hits), "notas_na_partitura": n, "compassos": nb,
                            "divisão": 480, "tempo_us": want_us,
                            "gm_esperadas": sorted({LANE_BY_ID[h["lane"]].gm for h in arq_hits
                                                     if h.get("lane") in LANE_BY_ID})},
                         detail=("regravado e relido por um parser SMF próprio: " + "; ".join(probs)
                                 + ("" if not (lead and lead > 0.5) else
                                    " — lembrete para quem for alinhar com o arquivo no DAW: o .mid "
                                    "começa no primeiro tempo da partitura, e o áudio tem %.1f s de "
                                    "silêncio antes dele (%.0f ms de deslocamento a aplicar)"
                                    % (lead, 1000.0 * lead)))
                         if probs else
                         (f"SMF tipo {m['format']}, {m['ntracks_read']} trilhas, divisão {m['division']}, "
                          f"{m['notes']} notas no canal 10 (GM {sorted(m['gm'])}), tempo {m['tempo_us']} µs, "
                          f"{m['markers']} marcadores de compasso, fim no tick {m['max_tick']}"),
                         state="ok" if not probs else "manual", data={"probs": probs[:6], "gm": m["gm"]}))
    except Exception as e:
        out.append(_find("T12", "export", "MIDI não releu", "error", detail=repr(e), state="manual"))

    # ------------------------------------------------------------------- T13 PDF/SVG e T14 CSV/JSON
    try:
        from .engrave import to_pdf_bytes, to_svg_pages
        pages = to_svg_pages(score)
        pdf = to_pdf_bytes(score)
        decl, npage = _pdf_pages(pdf)
        probs = []
        if pdf[:5] != b"%PDF-":
            probs.append("sem cabeçalho %PDF")
        if not pdf.rstrip().endswith(b"%%EOF"):
            probs.append("sem %%EOF no fim")
        if decl >= 0 and decl != len(pages):
            probs.append(f"PDF declara {decl} páginas, gravador produziu {len(pages)}")
        if npage and npage != len(pages):
            probs.append(f"{npage} objetos /Page vs {len(pages)} páginas")
        if len(pdf) < 1200 * max(1, len(pages)):
            probs.append(f"PDF com {len(pdf)} bytes — suspeita de estar vazio")
        out.append(_find("T13", "export", "PDF: cabeçalho, páginas e conteúdo vetorial",
                         "error" if probs else "info",
                         a={"bytes": len(pdf), "páginas_pdf": decl, "objetos_page": npage,
                            "páginas_svg": len(pages)},
                         b={"páginas_esperadas": len(pages), "compassos": nb},
                         detail="; ".join(probs) if probs else
                         f"{len(pages)} página(s) de SVG vetorial viraram PDF de {len(pdf)} bytes com "
                         f"{decl} páginas declaradas e {npage} objetos de página (sem raster, então "
                         f"imprime nítido a 1200 dpi)",
                         state="ok" if not probs else "manual", data={"probs": probs[:6]}))
    except Exception as e:
        out.append(_find("T13", "export", "PDF não gerou/relê", "error", detail=repr(e), state="manual"))

    try:
        js = json.loads(json.dumps(score))
        nh = len(_hits_of(js))
        rows = sum(1 for _ in str(_csv_of(score)).splitlines()) - 1
        probs = []
        if nh != n:
            probs.append(f"JSON volta com {nh} golpes, tinham {n}")
        if rows != n:
            probs.append(f"CSV com {rows} linhas vs {n} notas")
        out.append(_find("T14", "export", "JSON/CSV de correção (ida e volta)", "error" if probs else "info",
                         a={"json_golpes": nh, "csv_linhas": rows}, b={"golpes": n},
                         detail="o editor corrige por esse CSV/JSON: se a serialização perder um campo, a "
                                "correção se perde silenciosamente" + ("; " + "; ".join(probs) if probs else ""),
                         state="ok" if not probs else "manual"))
    except Exception as e:
        out.append(_find("T14", "export", "JSON/CSV falharam", "error", detail=repr(e), state="manual"))
    return out


def _csv_of(score: dict) -> str:
    """o mesmo CSV que o servidor entrega para o editor — nada de segunda implementação aqui"""
    from .pipeline import score_to_csv
    return score_to_csv(score)


# =========================================================================================
# premissas do arquivo de áudio
# =========================================================================================
_CAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "samples", "tonal_calibration.json")
_CAL: Dict[str, Any] = {}
_SUSTAIN_THR = 7.5


def calibration() -> dict:
    """Lê a tabela medida por `tests/calibrate_tonal.py` (limiar de tonalidade)."""
    global _CAL, _SUSTAIN_THR
    if _CAL:
        return _CAL
    try:
        with open(_CAL_PATH, encoding="utf-8") as fh:
            _CAL = json.load(fh)
        _SUSTAIN_THR = float(_CAL.get("threshold", _SUSTAIN_THR))
        if "measured" not in _CAL:            # a tabela medida é escrita em chaves planas
            labels = {"drums_only": "só bateria", "pad_minus30": "pad −30 dB",
                      "pad_minus26": "pad −26 dB", "pad_minus20": "pad −20 dB",
                      "voz_minus26": "voz −26 dB", "voz_minus18": "voz −18 dB"}
            _CAL["measured"] = {labels[k]: v for k, v in _CAL.items() if k in labels}
    except Exception:
        _CAL = {}
    return _CAL


def check_audio_assumptions(report: dict, tonal: Optional[Tuple[float, float]]) -> List[dict]:
    fl = (report or {}).get("file", {}) or {}
    out: List[dict] = []
    clip = _f(fl.get("clipped_ratio"))
    out.append(_find("T16", "premissas", "Clipping do arquivo", "warn" if clip > 0.001 else "info",
                     a=round(clip, 5), b=0.001,
                     detail="picos estourados achatam o ataque e prejudicam a detecção" if clip > 0.001
                            else "%.4f %% de amostras no teto" % (100 * clip),
                     state="manual" if clip > 0.001 else "ok"))
    peak, dur = _f(fl.get("peak_dbfs"), -99.0), _f(fl.get("duration_sec"))
    if peak < -30.0:
        out.append(_find("T17", "premissas", "Nível muito baixo (%.1f dBFS)" % peak, "warn",
                         detail="o normalizador interno compensa, mas a relação sinal/ruído pesa na "
                                "classificação de peças.", state="manual"))
    if dur and dur < 6.0:
        out.append(_find("T17", "premissas", "Trecho curto (%.1f s)" % dur, "warn", a=dur,
                         detail="abaixo de ~6 s andamento e fórmula ficam ambíguos", state="manual"))
    if not (peak < -30.0) and not (dur and dur < 6.0):
        out.append(_find("T17", "premissas", "Nível e duração adequados", "info", a=peak, b=dur,
                         detail="%.1f dBFS pico, %.1f s, %d canais, sr-fonte %d Hz"
                                % (peak, dur, int(_f(fl.get("channels"), 1)), int(_f(fl.get("sr_source")))),
                         state="ok"))
    if tonal is not None:
        dyn, cv = tonal
        ok = dyn >= _SUSTAIN_THR
        src = ("limiar medido em %s" % ", ".join("%s %.2f" % (k, v) for k, v in
                                                  sorted(_CAL.get("measured", {}).items(),
                                                         key=lambda kv: -kv[1]))
                if _CAL.get("measured") else "sem tabela de calibração — rodar tests/calibrate_tonal.py")
        out.append(_find("T18", "premissas", "A faixa se comporta como bateria isolada?",
                         "info" if ok else "warn", a=dyn, b=_SUSTAIN_THR,
                         detail="profundidade de modulação da banda 300–3000 Hz: P95/P25 = %.1f e CV = %.2f. "
                                "Percussão isolada tem vales fundos entre ataques (dyn alto); colchão "
                                "sustentado preenche os vales. %s — zona cinzenta deliberada abaixo do limiar: "
                                "colchão −30/−26 dB e voz −26 dB. Se houver instrumento por cima, a "
                                "classificação de peças passa a errar de forma sistemática; o aviso não é "
                                "erro porque o ritmo ainda pode estar certo."
                                % (dyn, cv, src),
                         state="ok" if ok else "manual"))
    return out


# =========================================================================================
# cobertura: o que o áudio tem e a partitura não tem (e vice-versa)
# =========================================================================================
def check_coverage(x: Optional[np.ndarray], sr: int, score: dict, report: dict,
                   dets: Optional[Sequence[dict]] = None) -> List[dict]:
    """
    Sem gabarito humano, a única pergunta que sobra é de *conjunto*: cada golpe que o detector
    aceitou tem de estar escrito, e cada ataque evidente do áudio tem de ter explicação. O
    teste é tolerante por desenho — fusões (`_merge_hit`) e 𝄄 apagam deliberately notações que
    existiam como detecções — por isso a lei é "existe um (compasso,tick) para onde isso vai",
    não "existe uma nota com a mesma pista".
    """
    out: List[dict] = []
    hits = _hits_of(score)
    slots = {(int(h["bar"]), int(round(_f(h.get("tick"))))) for h in hits}
    lanes_by_slot: Dict[Tuple[int, int], set] = {}
    for h in hits:
        lanes_by_slot.setdefault((int(h["bar"]), int(round(_f(h.get("tick"))))), set()).add(str(h.get("lane")))
    min_conf = _f((report or {}).get("params", {}).get("min_confidence"), 0.3)

    # ----------------------------------------------------------------- T15a detecções não escritas
    if dets:
        miss = []
        for d in dets:
            if _f(d.get("conf"), 1.0) < min_conf - 1e-6:
                continue                                  # rejeitada por confiança: certo não escrever
            k = (int(_f(d.get("bar"))), int(round(_f(d.get("tick")))))
            if k in slots:
                continue
            miss.append({"bar": k[0] + 1, "tick": k[1], "lane": d.get("lane"),
                         "conf": round(_f(d.get("conf")), 2)})
        ratio = len(miss) / max(1, len(dets))
        out.append(_find("T15a", "cobertura", "Golpes aceitos que não chegaram à pauta",
                         "error" if ratio > 0.03 else ("warn" if miss else "info"),
                         a=len(miss), b={"detecções": len(dets), "limiar_conf": min_conf},
                         detail="uma detecção com confiança acima do limiar que não foi escrita e nem foi "
                                "fundida com a vizinha é um golpe perdido — o caminho A (classificação → "
                                "grade → pauta) sumiu com ele sem avisar"
                                + (f"; exemplos: {miss[:3]}" if miss else ""),
                         state="ok" if not miss else "manual", data={"miss": miss[:30]}))

    # -------------------------------------------------- T15b ataques fortes do áudio sem explicação
    if x is not None and x.size > 8192 and sr > 8000:
        try:
            t_or, prom = oracle_onsets(x, sr, return_prominence=True)
        except Exception:
            t_or, prom = np.zeros(0), np.zeros(0)
        if t_or.size:
            thr = float(np.percentile(prom, 90)) if prom.size else 0.0
            strong = t_or[prom >= thr] if thr > 0 else t_or
            lead = _f((report or {}).get("file", {}).get("trim_lead_ms")) / 1000.0
            spt = _spt(score)
            bpm = _f(score.get("bpm"), 120.0) or 120.0
            tpr = int(_f(score.get("ticks_per_bar", 32), 32.0)) or 32
            times = np.array([_f(h.get("time")) for h in hits if h.get("time") is not None])
            tol = max(0.045, 0.6 * spt)
            unexplained = int(np.sum([not np.any(np.abs(times - float(t)) <= tol) for t in strong]))
            ratio = unexplained / max(1, strong.size)
            out.append(_find("T15b", "cobertura", "Ataques evidentes sem nota (oráculo sem gabarito)",
                             "warn" if ratio > 0.30 else "info", a=f"{unexplained}/{int(strong.size)}",
                             b={"tol_ms": round(tol * 1000.0, 1), "p90_prom": round(thr, 3)},
                             detail="oráculo independente dos 10 %% de ataques mais proeminentes do arquivo "
                                    "(envelope de energia, sem as bandas nem os pesos do pipeline): %d deles "
                                    "não têm nenhuma nota a ±%.0f ms. Percentual alto = a partitura está "
                                    "muda onde o áudio bate; pode ser também o 𝄄 encurtando a pauta, por "
                                    "isso é aviso e não erro." % (unexplained, tol * 1000.0),
                             state="ok" if ratio <= 0.30 else "manual",
                             data={"ratio": round(ratio, 3), "n_oracle": int(t_or.size)}))
        # ---------------------------- T15c contagem por peça (A = relatório, B = pauta recontada)
        per = (report or {}).get("events", {}).get("per_lane") or {}
        real: Dict[str, int] = {}
        for h in hits:
            real[str(h.get("lane"))] = real.get(str(h.get("lane")), 0) + 1
        mism, det_ok = [], True
        for ln, d in per.items():
            c, nd = int(_f(d.get("count"), -1)), int(_f(d.get("n_detections"), _f(d.get("count"), 0)))
            if c != real.get(ln, 0):
                mism.append({"lane": ln, "relatorio": c, "pauta": real.get(ln, 0)})
            if nd < real.get(ln, 0):
                det_ok = False
        extra = "" if det_ok else " — `n_detections` é MENOR que as notas escritas, impossível por construção"
        grave = bool(mism) or not det_ok
        q_out = sum(int(_f(d.get("count"), 0)) for d in per.values())
        out.append(_find("T15c", "cobertura", "Contagem por peça: relatório × pauta",
                         "error" if grave else "info", a={"soma_relatorio": q_out, "divergem": len(mism)},
                         b={"notas_pauta": len(hits)},
                         detail="o painel mostra estes números ao usuário. `count` é a partitura final e "
                                "`n_detections` o total de golpes aceitos naquela pista: a fusão de dois "
                                "golpes do mesmo slot (_merge_hit) faz o segundo ser maior, e é assim que "
                                "um acorde nasce — divergência entre os dois é explicada, divergência de "
                                "`count` com a pauta é bug" + (f"; {mism[:4]}" if mism else "") + extra,
                         state="ok" if not grave else "manual",
                         data={"mism": mism[:20], "det_ok": det_ok}))
    return out


# =========================================================================================
# ida e volta: reconstrução e renderização determinísticas
# =========================================================================================
def check_roundtrip(score: dict) -> List[dict]:
    """
    A partitura passa pelo *mesmo* caminho do editor interativo (rebuild_score) e pela
    impressão. Se uma dessas portas não for estável, toda correção manual é areia movediça.
    """
    from .pipeline import rebuild_score
    out: List[dict] = []
    hits = _hits_of(score)
    tpr = int(_f(score.get("ticks_per_bar", 32), 32.0)) or 32
    try:
        key0 = sorted((int(h["bar"]), int(round(_f(h.get("tick")))), str(h.get("lane"))) for h in hits)
        rebuilt = rebuild_score(hits, _f(score.get("bpm"), 120.0), str(score.get("meter", "4/4")),
                               _f(score.get("swing"), 0.0), title=str(score.get("title", "")),
                               subtitle="", report=None,
                               grid_mode=str((score.get("report") or {}).get("grid", {}).get("mode", "16th")))
        key1 = sorted((int(h["bar"]), int(round(_f(h.get("tick")))), str(h.get("lane")))
                      for h in _hits_of(rebuilt))
        diffs = sum(1 for a, b in zip(key0, key1) if a != b) if len(key0) == len(key1) else len(key0) + len(key1)
        out.append(_find("T19a", "ida-e-volta", "Reconstruir pelo caminho do editor não muda nada",
                         "error" if diffs else "info", a=len(key1), b={"esperado": len(key0), "diferem": diffs},
                         detail="o editor grava a lista de golpes e refaz durações, vigas, sustenus e 𝄄 com "
                                "as mesmas regras da transcrição; posição de nota é a única coisa que ele tem "
                                "de preservar integralmente",
                         state="ok" if not diffs else "manual", data={"diff": diffs}))
    except Exception as e:
        out.append(_find("T19a", "ida-e-volta", "Rebuild lançou exceção", "error", detail=repr(e),
                         state="manual"))
    try:
        from .engrave import to_svg, to_pdf_bytes
        s1, s2 = to_svg(score), to_svg(score)
        p1, p2 = to_pdf_bytes(score), to_pdf_bytes(score)
        # o gerador de PDF escreve CreationDate e o /ID do trailer: normaliza essas duas
        # coisas e exige igualdade byte a byte no resto — é o conteúdo desenhado que tem de
        # ser reproduzível, não a hora em que o arquivo foi feito
        norm = lambda b: re.sub(rb"/(CreationDate|ModID|ModDate|ID)\s*\[?.*?\]?\s*(?=/|>>)", b"", b, flags=re.S)
        det_ok = (s1 == s2) and (norm(p1) == norm(p2))
        n_draw = len(re.findall(r"<(line|ellipse|polyline|polygon|path|rect)\b", s1))
        empty = len(s1) < 400 or len(p1) < 1200 or n_draw < 20
        out.append(_find("T19b", "ida-e-volta", "Gravação determinística e não vazia",
                         "error" if (empty or not det_ok) else "info",
                         a={"svg_bytes": len(s1), "pdf_bytes": len(p1), "estável": bool(det_ok),
                            "primitivas": n_draw},
                         b={"primitivas_mínimas": 20},
                         detail="renderizar duas vezes tem de dar byte a byte o mesmo arquivo (senão a "
                                "correção do usuário não é reproduzível) e o resultado tem de ter conteúdo: "
                                "%d primitivas vetoriais no SVG, %d bytes de PDF idênticos exceto as "
                                "metadades de data do gerador" % (n_draw, len(p1)),
                         state="ok" if (det_ok and not empty) else "manual"))
    except Exception as e:
        out.append(_find("T19b", "ida-e-volta", "Gravação falhou", "error", detail=repr(e), state="manual"))
    try:
        from .kit_synth import render_score
        from .audio_io import Audio
        sr2 = 22050
        y = render_score(score, sr=sr2)
        t_wr, prom = oracle_onsets(np.asarray(y, dtype=np.float32), sr2, return_prominence=True)
        spt = _spt(score)
        lead = 0.0
        times = np.array([_f(h.get("time")) - lead for h in hits if h.get("time") is not None])
        if times.size and t_wr.size:
            tol = max(0.05, 0.9 * spt)
            # lei principal: toda nota escrita tem de produzir um ataque audível na síntese
            heard = int(sum(1 for t in times if np.any(np.abs(t_wr - float(t)) <= tol)))
            frac = heard / max(1, times.size)
            # direção inversa, só informativA: ataques "extras" do oráculo sobre o áudio
            # sintético são decays de prato/rimshot lidos como transiente, não nota fantasma
            spurious = int(sum(1 for t in t_wr if not np.any(np.abs(times - float(t)) <= tol)))
            out.append(_find("T19c", "ida-e-volta", "Toda nota escrita soa na síntese da partitura",
                             "warn" if frac < 0.85 else "info", a=round(frac, 3),
                             b={"notas": int(times.size), "ouvidas": heard,
                                "ataques_sintéticos": int(t_wr.size)},
                             detail="a pauta foi sintetizada peça por peça pelo mesmo `kit_synth` que o "
                                    "botão de audição toca e o oráculo de ataque procurou os transientes de "
                                    "volta: %.0f %% das notas escritas (%d/%d) produzem um ataque a ±%.0f ms. "
                                    "É o teste de que a notação é tocável, não só legível — nota que não soa "
                                    "é pausa invisível na audição de conferência. %d ataques sintéticos não "
                                    "têm nota correspondente: são caudas de prato/rimshot lidas como "
                                    "transiente pelo detector (por isso o teste é no sentido das notas)."
                                    % (100 * frac, heard, times.size, tol * 1000.0, spurious),
                             state="ok" if frac >= 0.85 else "manual",
                             data={"frac": round(frac, 3), "heard": heard, "spurious": spurious}))
    except Exception as e:
        out.append(_find("T19c", "ida-e-volta", "Síntese de conferência não rodou", "info",
                         detail=repr(e), state="skipped"))
    return out


# =========================================================================================
# o agente de correção
# =========================================================================================
def clock_map(hits: Sequence[dict], bpm: float, tpr: int,
              report: Optional[dict] = None) -> Optional[dict]:
    """
    Única definição da lei `tempo ↔ grade`, compartilhada pelo auditor e pelo agente.

        pos = (t − lead − fase)/spt + sinal·tick_offset      e      pos − origem = compasso·tpr + tick

    A `origem` é a mediana dos desvios: a numeração pode começar em qualquer barra, então o que
    é julgável é a rigidez da malha, não o valor absoluto. O sinal do offset de downbeat é detalhe
    interno do pipeline — provam-se os dois e fica o mais rígido. `rows` traz (desvio, golpe).
    """
    report = report or {}
    spt = 60.0 / max(1e-6, float(bpm)) / TICKS_PER_QUARTER
    if spt <= 0 or tpr <= 0:
        return None
    lead = _f(report.get("file", {}).get("trim_lead_ms")) / 1000.0
    ph = _f(report.get("tempo", {}).get("phase_ms")) / 1000.0
    off = _f(report.get("meter", {}).get("tick_offset"))
    ok = [h for h in hits if h.get("time") is not None and h.get("bar") is not None
          and 0 <= int(round(_f(h.get("tick")))) < tpr]
    if len(ok) < 8:
        return None
    best = None
    for sign in (+1.0, -1.0):
        vals = [((float(h["time"]) - lead - ph) / spt + sign * off
                 - (int(h["bar"]) * tpr + int(round(_f(h.get("tick"))))), h) for h in ok]
        arr = np.array([v for v, _h in vals])
        med = float(np.median(arr))
        spread = float(np.std(arr - med))
        if best is None or spread < best[0]:
            best = (spread, med, sign, [(float(v - med), h) for v, h in vals])
    spread, med, sign, rows = best
    return {"spt": spt, "lead": lead, "phase": ph, "off": off, "sign": sign, "origin": med,
            "tpr": tpr, "spread": spread, "rows": rows,
            "max": float(np.max(np.abs([r[0] for r in rows])))}


def grid_bound_ticks(score: dict, report: Optional[dict]) -> float:
    """
    Acima disto um desvio tempo↔grade não é mais resíduo humano e sim malha quebrada. É a MESMA
    régua usada pelo verificador T3 e pelo reparo — se as duas divergirem, o agente fica oscilando
    na fronteira em vez de convergir.
    """
    spt = _spt(score)
    tol = (max(1.0, _f((report or {}).get("grid", {}).get("tol_ms"), 0.0) / (1000.0 * spt))
           if spt > 0 else 1.5)
    return tol * 1.35 + 0.15


def project_time(t: float, clk: dict, step: int) -> Tuple[int, int]:
    """segundos → (compasso, tick) na grade, pelo mapa derivado do próprio score."""
    pos = ((float(t) - clk["lead"] - clk["phase"]) / clk["spt"] + clk["sign"] * clk["off"]
            - clk["origin"])
    abs_tk = int(round(pos / step) * step)
    if abs_tk < 0:
        abs_tk = 0
    return abs_tk // clk["tpr"], abs_tk % clk["tpr"]


def time_of(bar: int, tick: int, clk: dict) -> float:
    """(compasso, tick) → segundos (inverso de `project_time`)."""
    return ((int(bar) * clk["tpr"] + int(tick) + clk["origin"] - clk["sign"] * clk["off"])
            * clk["spt"] + clk["phase"] + clk["lead"])


def _rotate(score: dict, shift_ticks: int) -> List[dict]:
    """Desloca a barra inicial: mesma música, outro ponto de compasso (lei do backbeat)."""
    tpr = int(_f(score.get("ticks_per_bar", 32), 32.0)) or 32
    out: List[dict] = []
    for h in _hits_of(score):
        g = dict(h)
        abs_tk = int(h["bar"]) * tpr + int(round(_f(h.get("tick")))) + int(shift_ticks)
        if abs_tk < 0:
            continue
        g["bar"], g["tick"] = abs_tk // tpr, abs_tk % tpr
        out.append(g)
    return out


def _clean_hits(hits: List[dict], tpr: int, do: Dict[str, bool], step: int = 2,
                clk: Optional[dict] = None, dur_sec: float = 0.0) -> Tuple[List[dict], Dict[str, int]]:
    """
    Reparo determinístico direto sobre os golpes. Nada aqui é heurística de bom senso: cada
    linha corresponde a uma lei do verificador, e o `fix` do achado é o que autoriza a ação.
    """
    import bisect
    n_dedupe = n_clip = n_bound = n_art = n_lane = n_tie = n_resync = 0
    lanes_ok = set(LANE_BY_ID.keys())
    if do.get("normalize_fields"):
        kept = []
        for h in hits:
            if str(h.get("lane")) not in lanes_ok:
                n_lane += 1
                continue
            art = str(h.get("artic", "normal"))
            if art not in _ARTIC:
                h["artic"] = "normal"
                n_art += 1
            elif art in ("ghost", "ghost_accent") and str(h.get("lane")) not in _GHOST_OK:
                h["artic"] = "accent" if "accent" in art else "normal"
                n_art += 1
            tk = int(round(_f(h.get("tick"))))
            if not (0 <= tk < tpr):
                if clk and clk.get("spt") and h.get("time") is not None:
                    b_q, tk_q = project_time(_f(h["time"]), clk, step)
                    h["bar"], h["tick"] = b_q, tk_q
                else:
                    h["tick"] = int(min(max(0, tk % tpr), tpr - 1))
                    if h.get("time") is None and clk and clk.get("spt"):
                        h["time"] = round(time_of(int(_f(h.get("bar"))), int(h["tick"]), clk), 4)
                n_bound += 1
            if _f(h.get("dur"), 1) < 1:
                h["dur"] = max(1, min(tpr, abs(int(_f(h.get("dur")))) or 1))
                n_clip += 1
            h["velocity"] = int(min(127, max(1, round(_f(h.get("velocity"), 90)))))
            kept.append(h)
        hits = kept
    if do.get("dedupe"):
        best: Dict[Tuple[int, int, str], dict] = {}
        for h in hits:
            k = (int(h["bar"]), int(round(_f(h.get("tick")))), str(h.get("lane")))
            cur = best.get(k)
            if cur is None:
                best[k] = h
                continue
            n_dedupe += 1
            pref = lambda d: (_f(d.get("confidence")), _f(d.get("velocity")))   # noqa: E731
            if pref(h) > pref(cur):
                best[k] = h
        hits = list(best.values())
    if do.get("clip_durations") or do.get("requantize"):
        taken = {(int(h["bar"]), int(round(_f(h.get("tick")))), str(h.get("lane"))) for h in hits}
        for h in hits:                                  # 1º: alinha o tick com o passo da grade
            tk = int(round(_f(h.get("tick"))))
            if do.get("requantize") and tk % step:
                cand = int(np.clip(round(tk / step) * step, 0, tpr - (tpr % step or step)))
                if (int(h["bar"]), cand, str(h.get("lane"))) in taken:
                    cand = tk                               # destino ocupado: mantém a posição
                else:
                    taken.discard((int(h["bar"]), tk, str(h.get("lane"))))
                    taken.add((int(h["bar"]), cand, str(h.get("lane"))))
                    tk, n_bound = cand, n_bound + 1
            if tk + max(1, int(_f(h.get("dur"), 1))) > tpr and not h.get("tie_start"):
                h["dur"] = max(1, tpr - tk)
                n_clip += 1
            h["tick"] = tk
        by: Dict[Tuple[int, int], List[dict]] = {}
        for h in hits:
            v = int(_f(LANE_BY_ID[str(h["lane"])].voice, 2)) if str(h.get("lane")) in LANE_BY_ID else 2
            by.setdefault((int(h["bar"]), v), []).append(h)
        for (_b, v), lst in by.items():
            lst.sort(key=lambda x: _f(x.get("tick")))
            for h in lst:
                # A voz 1 é monofônica, mas a posição de uma nota NÃO é negociável: ela é o
                # instante em que o baterista bateu. O que se corrige é a duração (recortada até
                # o próximo ataque da voz); mover a nota criava colisão de slot e a deduplicação
                # seguinte engolia notas lícitas — perda de música por causa de uma nota ruim.
                onsets = sorted({int(round(_f(x.get("tick")))) for x in lst})
                tk = int(round(_f(h.get("tick"))))
                dur = max(1, int(_f(h.get("dur"), 1)))
                k = bisect.bisect_right(onsets, tk) - 1
                nxt = onsets[k + 1] if 0 <= k + 1 < len(onsets) else tpr
                limit = max(1, nxt - tk)                 # e para no próximo ataque da mesma voz
                if not h.get("tie_start") and dur > limit:
                    dur, n_clip = max(1, limit), n_clip + 1
                if tk + dur > tpr and not h.get("tie_start"):
                    dur, n_clip = max(1, tpr - tk), n_clip + 1
                h["tick"], h["dur"] = tk, max(1, dur)
    if do.get("drop_dangling_ties"):
        # mesma lei do T8, ao contrário: derruba apenas o lado que não tem par (a nota fica,
        # só deixa de ser ligada — sumir com a nota seria destruir informação musical)
        stop_at = {(int(h["bar"]), int(round(_f(h.get("tick"))))) for h in hits if h.get("tie_stop")}
        start_at = {(int(h["bar"]), int(round(_f(h.get("tick"))))) for h in hits if h.get("tie_start")}
        for h in hits:
            tk = int(round(_f(h.get("tick"))))
            dur = max(1, int(_f(h.get("dur"), 1)))
            end = tk + dur
            b = int(h["bar"])
            if h.get("tie_start"):
                ok = (b, end) in stop_at if end < tpr else \
                    (b + 1 + (end - tpr) // tpr, (end - tpr) % tpr) in stop_at
                if not ok:
                    nb = sorted([x for x in hits if int(x["bar"]) * tpr + int(round(_f(x.get("tick"))))
                                 > b * tpr + tk],
                                key=lambda x: (int(x["bar"]), _f(x.get("tick"))))
                    if nb:
                        x = nb[0]
                        xa = int(x["bar"]) * tpr + int(round(_f(x.get("tick"))))
                        ok = (x.get("tie_stop") and xa == b * tpr + tk + dur) or \
                             (x.get("tie_start") and xa >= b * tpr + tk + dur)
                if not ok:
                    h["tie_start"] = False
                    n_tie += 1
            if h.get("tie_stop"):
                pb = [x for x in hits if x.get("tie_start")
                      and int(x["bar"]) * tpr + int(round(_f(x.get("tick")))) + max(1, int(_f(x.get("dur"), 1)))
                      == b * tpr + tk]
                if not pb and not ((b, tk) in start_at or (b - 1, tk + tpr) in start_at):
                    h["tie_stop"] = False
                    n_tie += 1
    pode_resync = bool(do.get("normalize_fields")) and bool(clk and clk.get("spt"))
    for h in hits:
        if pode_resync and h.get("time") is not None:
            t0 = float(h["time"])
            hi = dur_sec if dur_sec > 0 else 1e9
            if not (-0.02 <= t0 <= hi + 0.02):
                # um tempo fora do arquivo não descreve nada: reescreve-se a partir da posição
                # gravada — a nota não se move, mas o par (som ↔ nota) volta a fazer sentido
                h["time"] = round(time_of(int(_f(h.get("bar"))), int(round(_f(h.get("tick")))), clk), 4)
                n_resync += 1
    return hits, {"dedupe": n_dedupe, "clip_durations": n_clip, "clamp_tick": n_bound,
                  "normalize_artic": n_art, "drop_unknown_lane": n_lane, "drop_dangling_ties": n_tie,
                  "resync_time": n_resync}


def repair(score: dict, findings: List[dict], report: Optional[dict] = None,
           allow: Optional[Sequence[str]] = None, rounds: int = 3,
           verify: bool = True) -> Tuple[dict, List[dict]]:
    """
    Executa só o que é determinístico e **re-conferir faz parte do reparo**: limpa, re-encaixa
    na malha, reconstrói pela mesma porta do editor (`rebuild_score`) e roda o auditor de novo
    sobre o resultado, repetindo enquanto aparecer coisa consertável. O laço interno é o que
    garante que o agente não devolve uma partitura que ele mesmo reprovaria.
    """
    from .pipeline import rebuild_score
    report = report or {}
    tpr = int(_f(score.get("ticks_per_bar", 32), 32.0)) or 32
    tpb = int(_f(score.get("ticks_per_beat", 8), 8.0)) or 8
    proposed = {f["fix"]: f for f in findings if f.get("fix")}
    if allow is not None:
        proposed = {k: v for k, v in proposed.items() if k in allow}
    if not proposed:
        return score, []
    hits = _hits_of(score)
    log: List[dict] = []
    if "rotate_downbeat" in proposed:
        shift = int(_f((proposed["rotate_downbeat"].get("data") or {}).get("shift_ticks")))
        if shift:
            hits = _rotate(score, shift)
            log.append({"fix": "rotate_downbeat", "shift_ticks": shift,
                        "motivo": "lei do backbeat (margem %.3f)"
                                  % _f((proposed["rotate_downbeat"].get("data") or {}).get("margin"))})
    do = {"dedupe": "dedupe" in proposed,
          "clip_durations": ("clip_durations" in proposed or "requantize" in proposed),
          "normalize_fields": ("normalize_fields" in proposed or "resync_time" in proposed
                               or "drop_dangling_ties" in proposed),
          "requantize": "requantize" in proposed,
          "drop_dangling_ties": "drop_dangling_ties" in proposed}
    mode = str((report.get("grid") or {}).get("mode", "16th"))
    step = {"8th": max(1, tpb // 2), "16th": max(1, tpb // 4), "32nd": max(1, tpb // 8)}.get(
        mode, max(1, tpb // 4))
    lim = grid_bound_ticks(score, report)
    dur_sec = _f(report.get("file", {}).get("duration_sec"))
    reanchor = "requantize_from_clock" in proposed
    tot: Dict[str, int] = {}
    moved_total = 0
    retimed_total = 0
    for _it in range(4):
        # PONTO FIXO. Uma passada não chega: re-encaixar uma nota pode fazê-la colidir com outra
        # no mesmo slot (aí entra a deduplicação) e recortar duração pode deslocar uma nota da
        # voz 1, mudando o slot dela de novo. Limpar → re-encaixar → limpar até estabilizar é o
        # que torna a saída auto-consistente. As contagens são ACUMULADAS — a última passada é a
        # vazia, e logar só ela apagaria o histórico do que foi feito.
        clk = clock_map(hits, _f(score.get("bpm"), 120.0), tpr, report) or {}
        hits, cnt = _clean_hits(hits, tpr, do, step, clk or None, dur_sec)
        for k, v in cnt.items():
            if v:
                tot[k] = tot.get(k, 0) + v
        if clk.get("spt"):
            occ = {}
            for h in hits:
                occ.setdefault((int(h["bar"]), int(round(_f(h.get("tick"))))), set()).add(str(h.get("lane")))
            for h in hits:
                if h.get("time") is None:
                    continue
                tk = int(round(_f(h.get("tick"))))
                if not (0 <= tk < tpr):
                    continue
                pos = (_f(h["time"]) - clk["lead"] - clk["phase"]) / clk["spt"] + clk["sign"] * clk["off"]
                if abs(pos - clk["origin"] - (int(h["bar"]) * tpr + tk)) > lim:
                    b_q, tk_q = project_time(_f(h["time"]), clk, step)
                    others = occ.get((b_q, tk_q), set()) - {str(h.get("lane"))}
                    same_lane = str(h.get("lane")) in occ.get((b_q, tk_q), set()) and (b_q, tk_q) != \
                        (int(h["bar"]), tk)
                    if (b_q, tk_q) != (int(h["bar"]), tk) and not same_lane:
                        # o áudio está livre nesse ponto: a nota vai para onde soa
                        occ.setdefault((int(h["bar"]), tk), set()).discard(str(h.get("lane")))
                        occ.setdefault((b_q, tk_q), set()).add(str(h.get("lane")))
                        h["bar"], h["tick"] = b_q, tk_q
                        moved_total += 1
                    elif others or same_lane:
                        # o instante alegado pertence a outra nota: o `time` é que está errado,
                        # e reescrevê-lo a partir da posição preserva a música escrita
                        h["time"] = round(time_of(int(h["bar"]), tk, clk), 4)
                        retimed_total += 1
        if not any(cnt.values()) and _it:
            break
    for k, v in sorted(tot.items()):
        if v:
            log.append({"fix": k, "itens": v})
    if moved_total:
        log.append({"fix": "requantize_from_clock", "notas_reencaixadas": moved_total,
                    "motivo": "o (compasso,tick) escrito não batia com o tempo do golpe (T3)"})
    if retimed_total:
        log.append({"fix": "resync_time", "itens": retimed_total,
                    "motivo": "o instante alegado pela nota pertencia a outra nota; o tempo foi "
                              "reescrito a partir da posição gravada (nenhuma nota foi movida nem perdida)"})
    if "unmark_slash" in proposed:
        bad_bars = {int(_f(x)) for x in ((proposed["unmark_slash"].get("data") or {}).get("bars") or [])}
        if bad_bars:
            log.append({"fix": "unmark_slash", "compassos": len(bad_bars),
                        "motivo": "repisado sem repetição fiel — desmarcado após a reconstrução"})
    if not log:
        return score, []
    new = rebuild_score(hits, _f(score.get("bpm"), 120.0), str(score.get("meter", "4/4")),
                        _f(score.get("swing"), 0.0),
                        title=str(score.get("title", "Transcrição de Bateria")),
                        subtitle=(str(score.get("subtitle", "")) + " · corrigido").strip(" ·"),
                        report=report, grid_mode=mode)
    if "unmark_slash" in proposed:
        raw = ((proposed["unmark_slash"].get("data") or {}).get("bars") or [])
        bad_bars = {int(_f(x)) for x in raw}
        n = 0
        for b in new.get("bars", []) or []:
            if int(_f(b.get("index"))) in bad_bars and b.get("repeat_slash"):
                b["repeat_slash"] = False
                n += 1
        if n:
            log = [x for x in log if x.get("fix") != "unmark_slash"]
            log.append({"fix": "unmark_slash", "compassos": n})
    if verify and rounds > 0:
        for _ in range(rounds):
            try:
                recheck = audit(new, report, level="fast")
            except Exception:
                break
            left = [f for f in recheck.get("findings", []) if f.get("fix")
                    and f.get("severity") in ("error", "warn")]
            if not left:
                break
            new2, log2 = repair(new, left, report, allow=allow, rounds=0, verify=False)
            if not log2:
                break
            new, log = new2, log + log2
    return new, log


# =========================================================================================
# entrada única de auditoria
# =========================================================================================
def audit(score: dict, report: Optional[dict] = None, *, x: Optional[np.ndarray] = None, sr: int = 0,
          dets: Optional[Sequence[dict]] = None, level: str = "full") -> dict:
    """
    Roda todos os verificadores e compila o quadro. `level="fast"` pula os caros (releitura de
    MIDI/PDF, ida-e-volta de síntese e cobertura por oráculo) para poder rodar automaticamente
    depois de cada análise. Cada verificador é chamado por nome, para que uma exceção vire um
    achado atribuído — nunca um resultado ausente.
    """
    report = report or {}
    calibration()
    tonal: Optional[Tuple[float, float]] = None
    if x is not None and x.size > 8192 and sr > 8000:
        try:
            tonal = mid_modulation(x, sr)
        except Exception:
            tonal = None
    findings: List[dict] = []
    calls = [("check_tempo", lambda: check_tempo(score, report, x, sr)),
             ("check_meter", lambda: check_meter(score, report)),
             ("check_grid", lambda: check_grid(score, report, dets)),
             ("check_notation", lambda: check_notation(score, report)),
             ("check_exports", lambda: check_exports(
                 score, level, lead=_f((report or {}).get("file", {}).get("trim_lead_ms")) / 1000.0)),
             ("check_audio", lambda: check_audio_assumptions(report, tonal))]
    if level == "full":
        calls += [("check_roundtrip", lambda: check_roundtrip(score)),
                  ("check_coverage", lambda: check_coverage(x, sr, score, report, dets))]
    for name, fn in calls:
        try:
            findings += fn()
        except Exception as e:
            findings.append(_find(name, "auditoria", "Verificador lançou exceção", "error",
                                  detail="%s: %s" % (type(e).__name__, e), state="manual"))
    order = {s: i for i, s in enumerate(SEVERITIES)}
    rank = {"manual": 0, "reverted": 1, "fixed": 2, "skipped": 3, "auto": 4, "ok": 5}
    findings.sort(key=lambda f: (order.get(f["severity"], 3), rank.get(f["state"], 6), f["check"]))
    tally = {s: sum(1 for f in findings if f["severity"] == s) for s in SEVERITIES}
    tally["manual"] = sum(1 for f in findings if f["state"] == "manual")
    tally["auto"] = sum(1 for f in findings if f["state"] == "auto")
    tally["checks"] = len(findings)
    return {"findings": findings, "tally": tally, "level": level,
            "summary": "%d erro(s), %d aviso(s), %d informação(ns) · %d pedem decisão · %d corrigíveis"
                       % (tally["error"], tally["warn"], tally["info"], tally["manual"], tally["auto"]),
            "score_digest": _digest(score)}


def _digest(score: dict) -> dict:
    hits = _hits_of(score)
    per: Dict[str, int] = {}
    for h in hits:
        per[str(h.get("lane", "?"))] = per.get(str(h.get("lane", "?")), 0) + 1
    return {"n_bars": len(score.get("bars", []) or []), "n_hits": len(hits),
            "bpm": round(_f(score.get("bpm")), 2), "meter": str(score.get("meter", "?")),
            "swing": round(_f(score.get("swing")), 3),
            "per_lane": dict(sorted(per.items(), key=lambda kv: -kv[1]))}


def audit_and_repair(score: dict, report: Optional[dict] = None, *, x: Optional[np.ndarray] = None,
                     sr: int = 0, dets: Optional[Sequence[dict]] = None, level: str = "full",
                     allow: Optional[Sequence[str]] = None, max_rounds: int = 2) -> dict:
    """
    Laço do agente: audita → repara → **re-audita**, e só aceita um reparo que reduza o peso
    (4·erros + avisos + 0.5·corrigíveis restantes); senão reverte. Nada é escondido: o quadro
    devolve `before` e `after` lado a lado, com o log do que foi feito.
    """
    def weight(t: dict) -> float:
        return 4.0 * int(t.get("error", 0)) + int(t.get("warn", 0)) + 0.5 * int(t.get("auto", 0))
    cur_score, cur = score, audit(score, report, x=x, sr=sr, dets=dets, level=level)
    first = cur
    accepted: List[dict] = []
    reverted, rounds = False, 0
    while rounds < max_rounds:
        try:
            nxt, log = repair(cur_score, cur["findings"], report, allow)
        except Exception as e:
            accepted.append({"erro": "%s: %s" % (type(e).__name__, e)})
            break
        if not log:
            break
        rounds += 1
        after = audit(nxt, report, x=x, sr=sr, dets=dets, level=level)
        if weight(after["tally"]) >= weight(cur["tally"]):
            reverted = True
            accepted = accepted[:-1] + [{"revertido": log, "peso_antes": weight(cur["tally"]),
                                        "peso_depois": weight(after["tally"])}]
            break
        cur_score, cur, accepted = nxt, after, accepted + log
    out = dict(cur)
    out["before"] = first
    out["after"] = cur
    out["applied"] = [] if reverted else accepted
    out["reverted"] = reverted
    out["rounds"] = rounds
    out["changed"] = bool(accepted) and not reverted and cur is not first
    out["score_novo"] = cur_score if out["changed"] else score
    out["pesos"] = {"inicial": weight(first["tally"]), "final": weight(cur["tally"])}
    return out


# --------------------------------------------------------------------------------- relatório
_ICON = {"ok": "ok", "manual": "DECIDIR", "auto": "corrigível", "fixed": "corrigido",
         "reverted": "revertido", "skipped": "n/d", "nd": "n/d"}
_SEV = {"error": "ERRO", "warn": "aviso", "info": "info"}


def to_markdown(rep: dict, title: str = "Validação dupla — quadro de achados") -> str:
    L: List[str] = ["# " + title, ""]
    dig = rep.get("score_digest") or {}
    if dig:
        L.append("**Partitura:** compasso %s · %s BPM · swing %s · %d compassos · %d golpes  "
                 % (dig.get("meter"), dig.get("bpm"), dig.get("swing"), dig.get("n_bars"),
                    dig.get("n_hits")))
        L.append("")
        L.append("**Peças:** " + " · ".join("%s %d" % (k, v) for k, v in (dig.get("per_lane") or {}).items()))
        L.append("")
    L.append("**Quadro:** %s  " % rep.get("summary", ""))
    L.append("")
    L.append("| estado | sev | teste | A (pipeline) | B (oráculo/lei) | evidência |")
    L.append("|---|---|---|---|---|---|")
    for f in rep.get("findings", []):
        L.append("| %s | %s | **%s** %s | %s | %s | %s |" % (
            _ICON.get(f.get("state"), f.get("state")), _SEV.get(f["severity"], f["severity"]),
            f["check"], f["title"], _md(f.get("a"), 26), _md(f.get("b"), 26), _md(f.get("detail"), 400)))
    if rep.get("applied"):
        L += ["", "## Reparos aplicados pelo agente", ""]
        for a in rep["applied"]:
            L.append("- " + _md(a, 300))
    if rep.get("reverted"):
        L += ["", "> Um reparo proposto foi **revertido** na re-auditoria: o quadro não melhorou e a "
              "partitura ficou como estava. Nada foi escondido — veja o log acima."]
    if rep.get("rounds"):
        L += ["", "*%d rodada(s) de audit→repair→audit.*" % rep["rounds"]]
    L += ["", "---", "*`drumscribe.qa` — cada linha tem duas medidas de origens distintas; só há "
          "reclamação quando divergem ou quando uma lei de escrita é violada.*"]
    return "\n".join(L) + "\n"


def _md(v: Any, n: int = 40) -> str:
    if isinstance(v, bool):
        return "sim" if v else "não"
    if isinstance(v, (int, float)):
        return ("%g" % v)
    if v is None:
        return "—"
    s = json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else str(v)
    s = s.replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


def run_file(path: str, params: Optional[dict] = None, *, fix: bool = False, level: str = "full",
             md_out: Optional[str] = None, json_out: Optional[str] = None,
             score_out: Optional[str] = None) -> dict:
    """CLI: analisa, audita (e repara se `fix`), e grava o relatório compilado em disco."""
    from .pipeline import transcribe_bytes
    from .audio_io import decode_file
    with open(path, "rb") as fh:
        data = fh.read()
    res = transcribe_bytes(data, filename=os.path.basename(path), params=params)
    score, rep = res["score"].to_dict(), res["report"]
    try:
        a = decode_file(path)
        x, sr = np.asarray(a.x, dtype=np.float32), int(a.sr)
    except Exception:
        x, sr = None, 0
    if fix:
        out = audit_and_repair(score, rep, x=x, sr=sr, dets=res.get("detections"), level=level)
        body = dict(out["after"] if out.get("changed") else out["before"])
        body["applied"] = out["applied"]
        body["reverted"] = out.get("reverted", False)
        body["rounds"] = out.get("rounds", 0)
        body["pesos"] = out.get("pesos", {})
        new_score = out.get("score_novo")
    else:
        body = audit(score, rep, x=x, sr=sr, dets=res.get("detections"), level=level)
        new_score = None
    if md_out:
        os.makedirs(os.path.dirname(os.path.abspath(md_out)), exist_ok=True)
        with open(md_out, "w", encoding="utf-8") as fh:
            fh.write(to_markdown(body, title="Validação dupla — %s" % os.path.basename(path)))
    if json_out:
        os.makedirs(os.path.dirname(os.path.abspath(json_out)), exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in body.items() if k not in ("score_novo", "before", "after")},
                      fh, ensure_ascii=False, indent=1, default=str)
    if fix and score_out and new_score is not None and body.get("applied"):
        from .engrave import write_pdf
        with open(score_out, "w", encoding="utf-8") as fh:
            json.dump(new_score, fh, ensure_ascii=False, indent=1)
        write_pdf(new_score, os.path.splitext(score_out)[0] + ".pdf")
        body["pdf_corrigido"] = os.path.splitext(score_out)[0] + ".pdf"
        body["json_corrigido"] = score_out
    return body
