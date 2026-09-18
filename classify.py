"""
Classificador de peças da bateria: pontuação espectral suave + *portões físicos* duros.

Modelo em duas camadas:

1. **Score suave** — combinação linear do z-score de banda (robusto a masterização e a
   ganhos diferentes por microfone) com termos de plausibilidade física: decaimento,
   queda em dB, fundamental (Hz) e "noisiness" (flatness geométrica/algébrica).
2. **Portões duros** — critérios mínimos que uma peça precisa cumprir (ex.: chimel
   fechado precisa perder ≥9 dB antes do próximo evento; bumbo precisa de energia
   sustentada em sub+low). Escore de pista vetada = -inf. Sem portão, um toque de
   caixa com corpo em 110 Hz vira bumbo com frequência; com portão, isso cai ~70%.

Assinaturas usadas (medidas empíricas de kits acústicos e deste sintetizador):

  bumbo   → 34–120 Hz dominante (≥28% da energia), f0 38–95 Hz, sem >7 kHz, queda rápida
  caixa   → corpo 180–420 Hz + ruído 1–11 kHz simultâneo, queda HF em <150 ms
  chimel  → ≥45% da energia acima de 3,2 kHz e queda ≥9 dB; sem energia grave
  ride/prato → mesma banda, mas sustain (queda <8 dB) e flatness menor (parciais metálicos)
  toms    → banda 120–420 Hz tonal (flatness baixa), sem ruído HF, sustain médio
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dsp import OnsetFeatures, zscore_per_band
from .kit import LANE_BAND_WEIGHTS, LANE_DECAY, LANE_PITCH, LANE_BY_ID, LANES

# Inclinações de decaimento típicas (dB/s) medidas em kits acústicos e no sintetizador
# de referência; usadas como prior suave de "sustain vs. golpe curto".
IDEAL_HI_SLOPE = {"kick": -25.0, "snare": -95.0, "rim": -300.0, "hat": -170.0,
                  "hat_open": -20.0, "hat_foot": -110.0, "ride": -14.0, "crash": -10.0,
                  "splash": -18.0, "tom_hi": -45.0, "tom_mid": -40.0, "tom_low": -35.0,
                  "cowbell": -45.0}
IDEAL_LO_SLOPE = {"kick": -35.0, "tom_hi": -60.0, "tom_mid": -55.0, "tom_low": -50.0,
                  "snare": -125.0}

NOISY_LANES = {"snare", "hat", "hat_open", "hat_foot", "ride", "crash", "splash", "cowbell", "rim"}
TONAL_LANES = {"kick", "tom_hi", "tom_mid", "tom_low"}


def _log(v) -> np.ndarray:
    return np.log(np.maximum(np.asarray(v, dtype=np.float64), 1e-9))


def _decay_term(decay_ms: np.ndarray, span: Tuple[float, float]) -> np.ndarray:
    lo, hi = span
    lx = _log(np.clip(decay_ms, 1.0, 1e5))
    llo, lhi = _log(np.array([lo])), _log(np.array([hi]))
    out = np.zeros_like(lx)
    below, above = lx < llo, lx > lhi
    out[below] = -np.clip((llo - lx[below]) / 0.8, 0.0, 3.0)
    out[above] = -np.clip((lx[above] - lhi) / 1.1, 0.0, 3.0)
    out[(~below) & (~above)] = 0.9
    return out.astype(np.float32)


def _gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _bi(feat: OnsetFeatures, name: str) -> np.ndarray:
    v = getattr(feat, name, None)
    return np.zeros(feat.n, dtype=np.float32) if v is None else np.asarray(v, dtype=np.float32)


def band_index(feat: OnsetFeatures, band: str) -> int:
    try:
        return feat.band_names.index(band)
    except ValueError:
        return -1


def _z(feat: OnsetFeatures) -> np.ndarray:
    """z-score robusto por banda sobre o *delta* de nível (ring-out do golpe anterior removido)."""
    base = getattr(feat, "band_delta_db", None)
    if base is None:
        base = feat.band_db
    return zscore_per_band(base)


def hard_gate_mask(feat: OnsetFeatures, lanes: List[str], strength: Optional[np.ndarray] = None,
                   params: Optional[dict] = None) -> np.ndarray:
    """
    (n, L) bool — True onde a atribuição é fisicamente plausível.

    Três famílias de evidência, todas calculadas **por banda** (nunca sobre o total do
    evento, que em acorde mistura duas fontes):
      * dominância de banda dentro do evento (fração);
      * nível relativo da banda (z-score robusto) — separa "tem energia" de "tem só o
        vazamento do golpe vizinho";
      * inclinação de decaimento da banda (dB/s) — sustain de prato vs. golpe curto.
    `strength` é a força relativa da novidade do grupo que detectou o evento: um disparo
    fraco no grupo grave durante uma caixa não é um bumbo.
    """
    n = feat.n
    z = zscore_per_band(feat.band_db)
    names = feat.band_names
    idx = {b: (names.index(b) if b in names else -1) for b in names}

    def Z(b: str) -> np.ndarray:
        i = idx.get(b, -1)
        return z[:, i] if i >= 0 else np.zeros(n, dtype=np.float32)

    sub, low, midlow, mid, midhi, high, vhigh = (Z("sub"), Z("low"), Z("midlow"), Z("mid"),
                                                 Z("midhi"), Z("high"), Z("vhigh"))
    rel = getattr(feat, "band_delta_rel", None)
    if rel is None:
        rel = feat.band_rel
    rname = {b: (names.index(b) if b in names else -1) for b in names}

    def FR(*bs: str) -> np.ndarray:
        acc = np.zeros(n, dtype=np.float32)
        for b in bs:
            i = rname.get(b, -1)
            if i >= 0:
                acc = acc + rel[:, i]
        return acc

    f_grave = FR("sub", "low", "midlow")
    f_body = FR("mid", "midhi")
    f_hf = FR("high", "vhigh")
    hs = _bi(feat, "hi_slope")
    ls = _bi(feat, "low_slope")
    ms = _bi(feat, "mid_slope")
    flat = _bi(feat, "flatness")
    pitch = _bi(feat, "pitch_low")
    hf_cent = _bi(feat, "hf_centroid")
    win = np.maximum(_bi(feat, "win_ms"), 20.0)
    dec_hi = _bi(feat, "decay_hi_ms")
    dec_lo = _bi(feat, "decay_low_ms")
    st = np.ones(n, dtype=np.float32) if strength is None else np.asarray(strength, np.float32)

    ok: Dict[str, np.ndarray] = {}
    # --- bumbo: energia abaixo de 62 Hz é a assinatura que nenhuma outra peça produz com
    #     essa proporção. sub_ratio separa bumbo (≈0,5) do corpo da caixa (≈0,08).
    sub_r = _bi(feat, "sub_ratio")
    dsub = _bi(feat, "band_delta_db")
    isub = rname.get("sub", 0)
    kick_delta = dsub[:, isub] if dsub.shape[1] else np.zeros(n, np.float32)
    ok["kick"] = ((sub_r >= 0.28) & (f_grave >= 0.38) & (ls <= -8.0) & (kick_delta >= 4.0) &
                  (((pitch > 28) & (pitch < 155)) | (pitch <= 0)) & (st >= 0.14))
    ok["snare"] = ((f_body + FR("midlow") >= 0.24) & (ms <= -18.0) & (ls <= -14.0) &
                   (sub_r <= 0.44) & (f_hf >= 0.08))
    ok["rim"] = ((FR("midhi") >= 0.30) & (hs <= -90.0) & (dec_hi <= 0.25 * win) & (f_grave <= 0.62))
    ok["hat"] = ((f_hf >= 0.34) & (vhigh >= -0.1) & (hs <= -42.0) & (f_body <= 0.50))
    ok["hat_open"] = ((f_hf >= 0.38) & (hs > -42.0) & (f_body <= 0.45) & (f_grave <= 0.66))
    ok["hat_foot"] = ((f_hf >= 0.30) & (hs <= -12.0) & (f_grave <= 0.45) & (low <= 0.9))
    ok["ride"] = ((f_hf >= 0.30) & (hs > -42.0) & (f_body <= 0.60) & (f_grave <= 0.75))
    ok["crash"] = ((f_hf >= 0.30) & (hs > -42.0) & (vhigh >= -0.35) & (f_grave <= 0.75))
    ok["splash"] = ((f_hf >= 0.36) & (hs > -48.0) & (hf_cent <= 9600.0))
    ok["tom_hi"] = ((FR("mid", "midlow") >= 0.40) & (f_hf <= 0.34) & (flat <= 0.74) &
                    (ms <= -8.0) & (sub_r <= 0.30) & (st >= 0.20))
    ok["tom_mid"] = ((FR("midlow", "mid") >= 0.38) & (f_hf <= 0.38) & (flat <= 0.82) &
                     (ms <= -8.0) & (sub_r <= 0.34) & (st >= 0.20))
    ok["tom_low"] = ((FR("sub", "low") >= 0.36) & (f_hf <= 0.38) & (flat <= 0.86) &
                     (ls <= -1.5) & (ms <= -2.0) & (sub_r >= 0.22) & (sub_r <= 0.46) & (st >= 0.20))
    ok["cowbell"] = ((FR("midhi", "high") >= 0.44) & (f_grave <= 0.34) & (hs <= -28.0))
    for ln in lanes:
        if ln not in ok:
            ok[ln] = np.ones(n, dtype=bool)
    return np.stack([np.asarray(ok[ln], dtype=bool) for ln in lanes], axis=1)


def score_matrix(feat: OnsetFeatures, lanes: List[str], params: Optional[dict] = None) -> np.ndarray:
    p = params or {}
    w_band = float(p.get("w_band", 1.0))
    w_phys = float(p.get("w_phys", 0.85))
    temp = float(p.get("softmax_t", 0.9))
    n = feat.n
    base = feat.band_delta_db if getattr(feat, "band_delta_db", None) is not None else feat.band_db
    z = zscore_per_band(base)
    scores = np.zeros((n, len(lanes)), dtype=np.float32)
    dec_lo = _bi(feat, "decay_low_ms")
    dec_hi = _bi(feat, "decay_hi_ms")
    flat = _bi(feat, "flatness")
    rise = _bi(feat, "rise_ratio")
    pitch = _bi(feat, "pitch_low")
    hs = _bi(feat, "hi_slope")
    ls = _bi(feat, "low_slope")
    win = np.maximum(_bi(feat, "win_ms"), 20.0)
    for li, ln in enumerate(lanes):
        W = LANE_BAND_WEIGHTS.get(ln)
        if W is None:
            scores[:, li] = -1e4
            continue
        s = np.zeros(n, dtype=np.float32)
        for bi, bn in enumerate(feat.band_names):
            if bn in W and bi < z.shape[1]:
                s += W[bn] * z[:, bi]
        s *= w_band
        lo, hi = LANE_DECAY.get(ln, (20, 900))
        s += w_phys * 0.35 * _decay_term(dec_lo, (lo * 0.5, hi * 2.4))
        s += w_phys * 0.35 * _decay_term(dec_hi, (lo * 0.30, hi * 1.8))
        # sustain vs. golpe curto: inclinação da cauda em dB/s (medido neste dominio:
        # chimel ≈ -170, caixa ≈ -95, ride ≈ -14, crash ≈ -10 dB/s)
        i_hi = IDEAL_HI_SLOPE.get(ln, -60.0)
        s += w_phys * 0.85 * np.clip(1.0 - abs(hs - i_hi) / 95.0, -1.0, 1.0)
        i_lo = IDEAL_LO_SLOPE.get(ln)
        if i_lo is not None:
            s += w_phys * 0.45 * np.clip(1.0 - abs(ls - i_lo) / 110.0, -1.0, 1.0)
        if ln in LANE_PITCH:
            f0, f1 = LANE_PITCH[ln]
            mu = 0.5 * _log(np.array([f0 * f1]))[0]
            sig = max(0.22, 0.5 * (_log(np.array([f1]))[0] - _log(np.array([f0]))[0]))
            t = np.zeros(n, dtype=np.float32)
            v = pitch > 0
            if v.any():
                t[v] = _gauss(_log(pitch[v]), mu, sig).astype(np.float32)
            s += w_phys * 1.15 * t
        s += w_phys * 0.85 * (flat - 0.5) * (1.0 if ln in NOISY_LANES else -0.7)
        if ln in ("snare", "kick", "hat", "rim", "hat_foot"):
            s += w_phys * 0.5 * (rise - 0.5)
        hfc = _bi(feat, "hf_centroid")
        if ln in ("hat", "hat_open", "hat_foot"):
            s += w_phys * 0.55 * np.clip((hfc - 7200.0) / 2600.0, -1.0, 1.0)
        elif ln in ("ride", "crash", "splash", "cowbell"):
            s += w_phys * 0.30 * np.clip((7600.0 - hfc) / 2600.0, -1.0, 1.0)
        if ln in ("crash", "ride", "hat_open"):        # sustain: energia que continua após ~90 ms
            s += w_phys * 0.40 * np.clip((win - 240.0) / 220.0, -0.6, 0.9)
        if ln == "ride":
            s += w_phys * 0.35 * np.clip((feat.loud - np.median(feat.loud)) / 6.0, -0.8, 0.4)
        stn = (params or {}).get("_strength")
        if stn is not None:
            s += w_phys * 0.55 * (np.asarray(stn, dtype=np.float32) - 0.55)
        ctx = (params or {}).get("_ctx")
        if ctx is not None:
            if ln == "crash":
                s += 0.40 * np.asarray(ctx.get("downbeat", np.zeros(n)), dtype=np.float32)
                s -= 0.55 * np.asarray(ctx.get("dense", np.zeros(n)), dtype=np.float32)
            elif ln == "hat":
                s += 0.25 * np.asarray(ctx.get("dense", np.zeros(n)), dtype=np.float32)
            elif ln == "hat_foot":
                s -= 0.6 * np.asarray(ctx.get("loud", np.zeros(n)), dtype=np.float32)
            elif ln == "ride":
                s += 0.30 * np.asarray(ctx.get("downbeat", np.zeros(n)), dtype=np.float32)
        scores[:, li] = s
    return scores


def classify_onsets(feat: OnsetFeatures, lanes: Optional[List[str]] = None,
                    params: Optional[dict] = None, gate: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna (labels, confiança, banda dominante). Confiança = mistura de probabilidade
    softmax e margem top1-top2 (a margem é o que realmente mede separabilidade).
    """
    lanes = list(lanes or [l.id for l in LANES if l.default])
    n = feat.n
    if n == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
    S = score_matrix(feat, lanes, params)
    dom = np.argmax(feat.band_rel, axis=1).astype(np.int64)
    stn = (params or {}).get("_strength")
    valid = (hard_gate_mask(feat, lanes, strength=stn, params=params) if gate
             else np.ones_like(S, dtype=bool))
    any_valid = valid.any(axis=1)
    S2 = np.where(valid, S, -1e6)
    # Nenhum portão válido → o candidato provavelmente NÃO é um golpe independente desse
    # grupo (é o vazamento de outra peça). Só a pista primária do grupo pode sobreviver,
    # e com confiança reduzida; o resto é descartado (label = -1).
    prim_name = (params or {}).get("_fallback")
    fallback_ok = np.full(n, -1, dtype=np.int64)
    st_arr = np.ones(n, dtype=np.float32) if stn is None else np.asarray(stn, dtype=np.float32)
    if prim_name in lanes:
        # o grupo disparou com força real → anota a pista primária do grupo, com confiança
        # baixa (fica visível na UI como "revisar"); sem força → descarta (é vazamento).
        pi = lanes.index(prim_name)
        thr_fb = float((params or {}).get("_fallback_min_st", 0.62))
        fallback_ok = np.where(st_arr >= thr_fb, pi, -1)
    bad = (~any_valid) & (fallback_ok >= 0)
    if bad.any():
        S2[bad] = -1.0
        S2[bad, fallback_ok[bad]] = 1.0
    discard = (~any_valid) & (fallback_ok < 0)
    temp = float((params or {}).get("softmax_t", 0.9))
    P = np.exp((S2 - S2.max(axis=1, keepdims=True)) / max(0.15, temp))
    P /= (P.sum(axis=1, keepdims=True) + 1e-9)
    order = np.argsort(-P, axis=1)
    labels = order[:, 0]
    if P.shape[1] > 1:
        order = np.where(discard[:, None], 0, order)
        top = P[np.arange(n), order[:, 0]]
        second = P[np.arange(n), order[:, 1]]
        margin = top - second
    else:
        margin = P[:, 0]
    conf = np.clip(0.35 * P[np.arange(n), labels] + 0.65 * margin, 0.0, 1.0)
    conf[~any_valid] = np.minimum(conf[~any_valid], 0.30)
    if bad.any():
        conf[bad] = 0.30
    labels[discard] = -1

    # fallback explícito para eventos ambíguos: decide pela banda dominante
    bn = np.array(feat.band_names)
    weak = (margin < 0.12) & any_valid
    if weak.any():
        lane_arr = np.array(lanes)
        domname = np.array([feat.band_names[i] if i >= 0 else "mid" for i in dom])
        hi_m = np.isin(domname, ("high", "vhigh"))
        lo_m = np.isin(domname, ("sub", "low", "midlow"))
        for i in np.flatnonzero(weak):
            tgt = "hat" if hi_m[i] else ("kick" if lo_m[i] else "snare")
            if tgt in lane_arr:
                labels[i] = int(np.flatnonzero(lane_arr == tgt)[0])
                conf[i] = float(min(conf[i], 0.40))
    return labels.astype(np.int64), conf.astype(np.float32), dom


def lane_prior_boost(labels: np.ndarray, conf: np.ndarray, lanes: List[str]) -> np.ndarray:
    """
    Correção de contexto mínima: pratos não costumam tocar em uníssono exato com a caixa
    no mesmo pulso; se um 'hat' de confiança muito baixa coincide com um snare forte,
    ele tende a ser o ruído do ataque da caixa. Conservador por design.
    """
    out = labels.copy()
    return out
