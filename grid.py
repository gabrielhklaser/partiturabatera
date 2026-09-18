"""
Refinamento contínuo de andamento/fase por regressão robusta + estimativa de métrica
em "espaço de ticks" (template de backbeat com busca de fase do downbeat).

Por que regressão: o ACF/pente dá BPM com erro de ~0.2–0.5%. Numa faixa de 4 minutos
isso acumula mais de 1/3 de beat de deriva no fim — o suficiente para mover a partitura
um tempo inteiro. Um ajuste linear t_i = a·tick_i + b sobre os eventos já casados
(M-estimator com corte por MAD) zera a deriva e é, na prática, o que fazem os
beat-trackers com DP + pós-PROCESSAMENTO de suavização.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .score_model import TICKS_PER_QUARTER, meter_info, METERS

PRIOR_CENTER_BPM = 118.0
PRIOR_SIGMA_OCT = 0.62


# --------------------------------------------------------------------------------------
# autocorrelação + prior + pente
# --------------------------------------------------------------------------------------

def _autocorr(x: np.ndarray) -> np.ndarray:
    n = x.size
    if n < 8:
        return np.zeros(n, dtype=np.float32)
    N = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(x - x.mean(), N)
    r = np.fft.irfft(np.abs(F) ** 2, N)[:n].astype(np.float64)
    return (r / (r[0] + 1e-12)).astype(np.float32)


def _prior(bpm) -> np.ndarray:
    b = np.asarray(bpm, dtype=np.float64)
    return np.exp(-0.5 * (np.log2(np.maximum(b, 1e-3) / PRIOR_CENTER_BPM) / PRIOR_SIGMA_OCT) ** 2)


def comb_matrix(o: np.ndarray, fps: float, bpm: np.ndarray, phase_frac: np.ndarray,
                tol_s: float = 0.045, empty_pen: float = 0.30) -> np.ndarray:
    from scipy import ndimage
    n = o.size
    w = max(1, int(tol_s * fps))
    omax = ndimage.maximum_filter1d(o, size=2 * w + 1, mode="nearest")
    bpm = np.atleast_1d(np.asarray(bpm, dtype=np.float64))
    ph = np.atleast_1d(np.asarray(phase_frac, dtype=np.float64))
    out = np.full((bpm.size, ph.size), -1e9, dtype=np.float32)
    ks = None
    for i, B in enumerate(bpm):
        period = 60.0 / B * fps
        if period < 2.0:
            continue
        kmax = int((n - 2) // period)
        if kmax < 4:
            continue
        ks = np.arange(kmax, dtype=np.float64)
        for j, p in enumerate(ph):
            idx = np.rint(p * period + ks * period).astype(np.int64)
            idx = np.clip(idx, 0, n - 1)
            v = omax[idx]
            cov = float(np.mean(v > 0.10))
            out[i, j] = float(v.mean() - empty_pen * (1.0 - cov))
    return out


def estimate_tempo(o: np.ndarray, fps: float, min_bpm: float = 45.0, max_bpm: float = 220.0,
                   bpm_hint: Optional[float] = None) -> dict:
    """ACF ponderado por prior → busca fina (período × fase) → teste de oitava ×2, ÷2."""
    fallback = {"bpm": float(bpm_hint or 120.0), "phase_sec": 0.0, "confidence": 0.0,
                 "candidates": [], "method": "fallback"}
    if o.size < int(fps * 2.5):
        return fallback
    r = _autocorr(o)
    lo = max(2, int(60.0 / max_bpm * fps))
    hi = min(r.size - 1, int(60.0 / min_bpm * fps))
    lags = np.arange(lo, hi + 1)
    if lags.size < 4:
        return fallback
    sc = r[lags] * _prior(60.0 * fps / lags)
    best_i = int(np.argmax(sc))
    b0 = float(60.0 * fps / lags[best_i])
    if bpm_hint:
        near = np.abs(np.log2(60.0 * fps / lags / float(bpm_hint))) < 0.25
        if near.any():
            b0 = float(60.0 * fps / lags[np.argmax(np.where(near, sc, -1e9))])
    bpm_grid = np.arange(b0 * 0.86, b0 * 1.16, 0.02)
    ph_grid = np.linspace(0.0, 1.0, 64, endpoint=False)
    M = comb_matrix(o, fps, bpm_grid, ph_grid)
    if not np.isfinite(M).any():
        return fallback
    i, j = np.unravel_index(int(np.nanargmax(np.where(np.isfinite(M), M, -np.inf))), M.shape)
    bpm_fine = float(bpm_grid[i])
    score = float(M[i, j])

    cands = []
    for mult, name in ((1.0, "1x"), (2.0, "2x"), (0.5, "0.5x")):
        bb = bpm_fine * mult
        if not (min_bpm <= bb <= max_bpm):
            continue
        g = np.arange(bb * 0.985, bb * 1.015, 0.02)
        mm = comb_matrix(o, fps, g, ph_grid)
        k, l = np.unravel_index(int(np.nanargmax(np.where(np.isfinite(mm), mm, -np.inf))), mm.shape)
        cands.append({"bpm": float(g[k]), "score": float(mm[k, l]), "label": name,
                      "phase_frac": float(ph_grid[l])})
    for c in cands:
        c["adj"] = c["score"] * float(_prior(np.array([c["bpm"]]))[0])
    base = next((c for c in cands if c["label"] == "1x"), cands[0])
    pick = max(cands, key=lambda c: c["adj"])
    if pick["adj"] <= base["adj"] * 1.06:
        pick = base
    period_s = 60.0 / pick["bpm"]
    scores = [c["score"] for c in cands]
    spread = (max(scores) - min(scores)) / (abs(np.mean(scores)) + 1e-6)
    conf = float(np.clip(0.5 + 0.5 * (1.0 - min(1.0, spread * 3.0)) * (0.5 + pick["score"]), 0.05, 0.99))
    return {"bpm": round(pick["bpm"], 3), "phase_sec": float(pick["phase_frac"] * period_s),
            "score": float(pick["score"]), "confidence": conf,
            "period_frames": float(period_s * fps), "method": "acf+prior+comb",
            "candidates": [{"bpm": round(c["bpm"], 2), "score": round(c["score"], 5),
                            "label": c["label"]} for c in cands],
            "envelope_fps": fps}


def fit_grid(times: np.ndarray, bpm: float, phase_sec: float, ticks_per_beat: int,
             mode: str = "16th", iters: int = 4) -> dict:
    """
    Regressão robusta tick→tempo. Retorna bpm/fase refinados, r² e resíduos.
    `times` em segundos; posição absoluta em ticks = (t - phase)/sec_per_tick.
    """
    times = np.asarray(times, dtype=np.float64)
    out = {"bpm": float(bpm), "phase_sec": float(phase_sec), "ok": False,
            "n_used": 0, "rms_ms": 0.0, "drift_ms_end": 0.0, "iters": 0}
    if times.size < 8:
        return out
    sec_per_tick = 60.0 / max(1e-6, bpm) / TICKS_PER_QUARTER
    g = grid_positions(ticks_per_beat, mode, 0.0)
    keep = np.ones(times.size, bool)
    a = sec_per_tick
    b = float(phase_sec)
    for it in range(iters):
        pos = (times - b) / a
        rel = pos[:, None] % ticks_per_beat
        d = np.abs(rel - g[None, :])
        d = np.minimum(d, ticks_per_beat - d)
        slot = np.argmin(np.abs(rel - g[None, :]), axis=1)
        resid = d[np.arange(pos.size), slot]
        ticks = np.floor(pos / ticks_per_beat) * ticks_per_beat + np.rint(g[slot])
        # refaz com o tick absoluto "livre" (sem mod) para a regressão
        pos_raw = (times - b) / a
        ticks = np.where(resid <= ticks_per_beat * 0.30, np.rint(pos_raw), np.nan)
        m = keep & np.isfinite(ticks)
        if int(m.sum()) < 8:
            break
        X = np.stack([ticks[m], np.ones(int(m.sum()))], axis=1)
        y = times[m]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a_new, b_new = float(coef[0]), float(coef[1])
        if not (0.0005 < a_new < 0.5):
            break
        r = y - (X @ coef)
        mad = float(np.median(np.abs(r - np.median(r)))) * 1.4826 + 1e-9
        kk = (3.5, 2.6, 2.0, 1.8)[min(it, 3)]
        k2 = np.abs(r) <= kk * mad
        keep2 = keep.copy()
        keep2[np.flatnonzero(m)] = k2
        if int(keep2.sum()) >= 8:
            keep = keep2
        a, b = a_new, b_new
        out["iters"] = it + 1
        bpm_new = 60.0 / (TICKS_PER_QUARTER * a)
        if abs(bpm_new - out["bpm"]) < 0.004 and it > 0:
            out["bpm"] = round(bpm_new, 3)
            out["phase_sec"] = b
            break
        out["bpm"] = round(bpm_new, 3)
        out["phase_sec"] = b
    if times.size and np.isfinite(a):
        pos = (times - b) / a
        rel = pos[:, None] % ticks_per_beat
        d = np.abs(rel - g[None, :])
        d = np.minimum(d, ticks_per_beat - d)
        resid = d.min(axis=1)
        out["rms_ms"] = round(float(np.sqrt(np.mean(resid ** 2))) * a * 1000.0, 2)
        out["n_used"] = int(keep.sum())
        out["ok"] = True
        out["drift_ms_end"] = round(float((times[-1] - b) / a - (times[-1] - phase_sec) /
                                          (60.0 / max(1e-6, bpm) / TICKS_PER_QUARTER)) * a * 1000.0, 1)
        out["sec_per_tick"] = a
    return out


# --------------------------------------------------------------------------------------
# métrica / downbeat em espaço de ticks
# --------------------------------------------------------------------------------------

# templates em ticks relativos ao início do compasso (tpq=8 → semínima=8, colcheia=4)
METER_TEMPLATES: Dict[str, dict] = {
    "4/4": {"tpr": 32, "kick": {0: 1.35, 16: 1.15}, "snare": {8: 1.75, 24: 1.75},
            "crash": {0: 0.55}, "anti": {"snare": {0: 0.6, 16: 0.6}, "kick": {8: 0.25, 24: 0.25}},
            "sig": 0.32},
    "2/4": {"tpr": 16, "kick": {0: 1.35}, "snare": {8: 1.6}, "crash": {0: 0.5},
            "anti": {"snare": {0: 0.55}, "kick": {8: 0.3}}, "sig": 0.32},
    "3/4": {"tpr": 24, "kick": {0: 1.3}, "snare": {16: 1.25, 8: 0.6}, "crash": {0: 0.5},
            "anti": {"snare": {0: 0.5}}, "sig": 0.32},
    "5/4": {"tpr": 40, "kick": {0: 1.2, 24: 0.75}, "snare": {16: 1.2, 32: 1.25},
            "crash": {0: 0.45}, "anti": {}, "sig": 0.32},
    "6/8": {"tpr": 24, "kick": {0: 1.3, 12: 0.85}, "snare": {12: 1.5}, "crash": {0: 0.5},
            "anti": {"snare": {0: 0.6}}, "sig": 0.6},
    "9/8": {"tpr": 36, "kick": {0: 1.2, 24: 0.6}, "snare": {12: 1.15}, "crash": {0: 0.45},
            "anti": {}, "sig": 0.6},
    "12/8": {"tpr": 48, "kick": {0: 1.3, 24: 0.9}, "snare": {24: 1.4}, "crash": {0: 0.5},
             "anti": {"snare": {0: 0.6}}, "sig": 0.6},
}


def estimate_meter(pos_ticks: np.ndarray, lane_of: Sequence[str], loud: np.ndarray,
                   candidates: Sequence[str] = ("4/4", "3/4", "2/4", "6/8", "5/4", "12/8", "9/8"),
                   n_beats: float = 0.0, sig_ticks: Optional[float] = None) -> dict:
    """
    Para cada métrica candidata e cada fase de downbeat, pontua o ajuste ao template de
    backbeat com um kernel gaussiano circular (tolerante a jitter de ±~20 ms) e dá bônus
    para energia alta no início do compasso. Normalizado por número de eventos.
    """
    pos = np.asarray(pos_ticks, dtype=np.float64)
    n = pos.size
    if n < 8:
        return {"meter": "4/4", "tick_offset": 0, "score": 0.0, "table": [], "n_bars": 1}
    lanes = np.asarray(list(lane_of))
    e = np.power(10.0, (np.asarray(loud, dtype=np.float64) - np.median(loud)) / 20.0)
    e = e / (np.percentile(e, 99) + 1e-9)
    grp = np.array([("kick" if l == "kick" else
                     "snare" if l in ("snare", "rim", "tom_hi", "tom_mid", "tom_low") else
                     "crash" if l in ("crash", "splash") else "hat") for l in lanes])
    table = []
    for name in candidates:
        T = METER_TEMPLATES.get(name)
        if not T:
            continue
        tpr = T["tpr"]
        sig = float(sig_ticks) if sig_ticks else T["sig"] * 8.0
        best = (-1e9, 0)
        masks = {g: (grp == g) for g in ("kick", "snare", "crash")}
        counts = {g: int(m.sum()) for g, m in masks.items()}
        for off in range(0, tpr):
            x = (pos - off) % tpr
            tot = 0.0
            for g in ("kick", "snare", "crash"):
                if not counts[g]:
                    continue
                for tp, w in T.get(g, {}).items():
                    d = np.abs(x - tp)
                    d = np.minimum(d, tpr - d)
                    k = np.exp(-0.5 * (d / sig) ** 2)
                    tot += w * float(np.sum(k[masks[g]]) / counts[g]) * counts[g] / max(1, n)
            for g, dd in T.get("anti", {}).items():
                if not counts.get(g):
                    continue
                for tp, w in dd.items():
                    d = np.abs(x - tp)
                    d = np.minimum(d, tpr - d)
                    k = np.exp(-0.5 * (d / sig) ** 2)
                    tot -= w * float(np.sum(k[masks[g]]) / counts[g]) * counts[g] / max(1, n)
            # bônus downbeat: energia média nos inícios de compasso vs. média geral
            near0 = (x < sig) | (x > tpr - sig)
            if near0.any():
                tot += 0.9 * (float(np.mean(e[near0])) / (float(np.mean(e)) + 1e-6) - 1.0)
            sc = tot / (n_beats if n_beats else max(1.0, n / 6.0))
            if sc > best[0]:
                best = (sc, off)
        nbars = max(1, int(round(float(np.max(pos + 1)) / tpr)))
        table.append({"meter": name, "score": round(float(best[0]), 4), "tick_offset": int(best[1]),
                      "beats_per_bar": tpr // (12 if name in ("6/8", "9/8", "12/8") else 8),
                      "n_bars": nbars})
    table.sort(key=lambda d: -d["score"])
    top = table[0]
    return {"meter": top["meter"], "tick_offset": top["tick_offset"], "score": top["score"],
            "table": table, "n_bars": top["n_bars"]}


# --------------------------------------------------------------------------------------
# grade e quantização
# --------------------------------------------------------------------------------------

def grid_positions(ticks_per_beat: int, mode: str, swing: float = 0.0) -> np.ndarray:
    """Posições permitidas (ticks, podem ser fracionárias) dentro de um beat."""
    t = float(ticks_per_beat)
    sw = float(swing or 0.0)
    if mode in ("8th",):
        return np.array([0.0, sw * t if sw > 0.52 else t / 2.0], dtype=np.float64)
    if mode in ("16th",):
        if sw > 0.52:
            return np.array([0.0, sw * t / 2.0, t / 2.0, t * (1.0 + sw) / 2.0], dtype=np.float64)
        return np.array([0.0, t / 4.0, t / 2.0, 3.0 * t / 4.0], dtype=np.float64)
    if mode in ("32nd+16th", "32nd"):
        base = grid_positions(ticks_per_beat, "16th", sw)
        if mode == "32nd":
            return np.unique(np.round(np.arange(0.0, t, t / 8.0), 4))
        extra = np.concatenate([base + t / 8.0, base - t / 8.0])
        extra = extra[(extra >= 0.0) & (extra < t)]
        return np.unique(np.round(np.concatenate([base, extra]), 4))
    if mode in ("triplet", "16th+triplet"):
        base = grid_positions(ticks_per_beat, "16th", 0.0) if mode == "16th+triplet" else np.array([0.0])
        tri = np.arange(3) * t / 3.0
        six = np.arange(6) * t / 6.0
        allp = np.concatenate([base, tri, six]) if mode == "16th+triplet" else np.concatenate([base, tri])
        allp = allp[(allp >= 0.0) & (allp < t)]
        return np.unique(np.round(allp, 4))
    return grid_positions(ticks_per_beat, "16th", sw)


def passo_grade(g: np.ndarray, tpb: float) -> float:
    """Espaçamento, em ticks, da grade que foi escolhida — a única régua de "está no passo".

    Existiam três contas separadas para isso (`tpb/4 if tpb == 8 else tpb/3` no `quantize`, o
    mesmo `tpb/3` no polimento de downbeat do `pipeline`, e `tpb // subdivisão` no auditor
    `T3b`). Para 4/4 (tpb = 8) as três coincidiam em 2 ticks e nada aparecia; em 6/8 a grade de
    16ºs é [0,3,6,9] (passo 3) enquanto o offset de downbeat era arredondado a múltiplos de 4 —
    e o efeito era o pior possível: **todas** as 2999 notas de uma faixa real ficavam a 2 ticks do
    passo declarado, empilhadas no fim do compasso, invadindo a barra seguinte (docs/ERROS.md A21).

    Uma grade uniforme devolve o passo; uma rede não uniforme (swing, tercinas misturadas) devolve
    `tpb/4`, que é a régua que o projeto já usa nesses casos — assim a função não muda o
    comportamento em swing nem em 4/4: ela só para de discordar da grade nos andamentos compostos.
    """
    pos = np.unique(np.rint(np.asarray(g, dtype=np.float64) % float(tpb)))
    if pos.size < 2:
        return float(tpb)
    gaps = np.diff(np.concatenate([pos, [pos[0] + float(tpb)]]))
    u = np.unique(gaps[np.abs(gaps) > 1e-9])
    if u.size == 1:
        return float(u[0])
    return float(tpb) / 4.0


def _grid_fit(pos: np.ndarray, tpb: int, g: np.ndarray, w: Optional[np.ndarray] = None,
              sigma: Optional[float] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Ajuste dos eventos a uma grade: (fitness ponderado, resíduo em ticks, slot).
    `w` é o peso por evento (confiança da detecção): meia-dúzia de falsos positivos não
    pode forçar a partitura inteira a escrever fusquinhas.
    """
    rel = pos[:, None] % tpb
    d = np.abs(rel - g[None, :])
    d = np.minimum(d, tpb - d)
    slot = np.argmin(d, axis=1)
    resid = d[np.arange(pos.size), slot]
    sig = float(sigma) if sigma else max(0.9, tpb / 8.0)
    q = np.exp(-((resid / sig) ** 2))
    if w is None:
        fit = float(q.mean())
    else:
        w = np.clip(np.asarray(w, dtype=np.float64), 1e-3, None)
        fit = float(np.sum(w * q) / (np.sum(w) + 1e-12))
    return fit, resid, slot


BASE_MODES = ["8th", "16th", "32nd+16th", "16th+triplet"]
FIT_TARGET = 0.90          # grade mínima suficiente: menos símbolos, mesma música


def quantize(times: np.ndarray, bpm: float, phase_sec: float, meter: str,
             mode: str = "auto", swing: Optional[float] = None, tick_offset: int = 0,
             tol_beat: float = 0.16, weights: Optional[np.ndarray] = None) -> dict:
    """
    Projeta os instantes na grade métrica.

    Regras (deliberadamente determinísticas — "grade mínima suficiente"):
      1. A grade de referência é a de SEMICOLCHEIAS (16ºs): é o padrão de leitura de
         bateria e dá fase estável (8 pontos por beat não deixam o ajuste escapar).
      2. Colcheias (8ºs) são aceitas se toda a massa sustentada couber nelas.
      3. Fusquinhas (32ºs) só entram por *promoção individual*: o evento precisa ser
         confiante, estar a > 0,6 tick da grade de 16ºs e o slot de 32ºs precisa reduzir
         o resíduo em > 0,25 tick. Um punhado de detecções incertas nunca reescreve a
         partitura inteira.
      4. Swing exige evidência: só troca a grade se o fitness melhorar > 4%.
      5. O offset de downbeat é quantizado ao beat (um compasso começa num tempo, não
         numa fração dele) — senão a partitura inteira desloca 1 tick.
    """
    info = meter_info(meter)
    tpb = info["ticks_per_beat"]
    tpr = info["ticks_per_bar"]
    sec_per_tick = 60.0 / max(1e-6, bpm) / TICKS_PER_QUARTER
    times = np.asarray(times, dtype=np.float64)
    empty = {k: np.zeros(0, np.int64) for k in ("ticks", "bars")}
    empty.update({"resid": np.zeros(0), "ticks_abs": np.zeros(0), "offgrid": np.zeros(0, bool),
                  "mode": mode, "swing": 0.0, "fitness": 0.0, "tol_ticks": 1.0,
                  "sec_per_tick": sec_per_tick, "positions": np.array([0.0]),
                  "swing_detected": 0.0, "tick_offset": int(tick_offset), "meter": meter,
                  "best_fit_all": 0.0, "n_candidates": 0, "swing_allowed": False,
                  "promoted_32nd": 0, "fits": {}})
    if times.size == 0:
        return empty
    pos = (times - phase_sec) / sec_per_tick
    w = np.ones(times.size) if weights is None else np.asarray(weights, dtype=np.float64)
    g16 = grid_positions(tpb, "16th", 0.0)
    g8 = grid_positions(tpb, "8th", 0.0)
    fits: Dict[str, float] = {}
    fit16, r16, s16 = _grid_fit(pos, tpb, g16, w)
    fit8, r8, s8 = _grid_fit(pos, tpb, g8, w)
    fits["16th@0.00"] = round(fit16, 4)
    fits["8th@0.00"] = round(fit8, 4)
    swing_cands = []
    if swing is None:
        for sw in (0.60, 0.67, 0.72):
            for gm in ("16th", "8th"):
                g = grid_positions(tpb, gm, sw)
                f_, r_, s_ = _grid_fit(pos, tpb, g, w)
                swing_cands.append({"fit": f_, "mode": gm, "swing": sw, "g": g,
                                    "resid": r_, "slot": s_})
                fits[f"{gm}@{sw:.2f}"] = round(f_, 4)
    best_sw = max(swing_cands, key=lambda c: c["fit"]) if swing_cands else None
    use_swing = bool(best_sw and best_sw["fit"] > fit16 * 1.04)
    sel = {"fit": fit16, "mode": "16th", "swing": 0.0, "g": g16, "resid": r16, "slot": s16}
    if str(mode) == "8th" or (not use_swing and fit8 >= max(0.965, fit16 - 0.005)):
        sel = {"fit": fit8, "mode": "8th", "swing": 0.0, "g": g8, "resid": r8, "slot": s8}
    if str(mode) in ("8th", "16th", "32nd", "32nd+16th", "triplet", "16th+triplet") and mode != "auto":
        g = grid_positions(tpb, str(mode), 0.0)
        f_, r_, s_ = _grid_fit(pos, tpb, g, w)
        sel = {"fit": f_, "mode": str(mode), "swing": 0.0, "g": g, "resid": r_, "slot": s_}
        fits[f"forced:{mode}"] = round(f_, 4)
    if use_swing and str(mode) in ("auto", "8th", "16th"):
        sel = {"fit": best_sw["fit"], "mode": best_sw["mode"], "swing": best_sw["swing"],
               "g": best_sw["g"], "resid": best_sw["resid"], "slot": best_sw["slot"]}
    if str(mode) in ("triplet", "16th+triplet"):
        g = grid_positions(tpb, str(mode), 0.0)
        f_, r_, s_ = _grid_fit(pos, tpb, g, w)
        sel = {"fit": f_, "mode": str(mode), "swing": 0.0, "g": g, "resid": r_, "slot": s_}

    g, slot = sel["g"], sel["slot"]
    tol = tol_beat * tpb
    # ------------------------------------------------------------------------------ (BUG FIX)
    # Encaixe na grade em coordenada ABSOLUTA. A reconstrução antiga
    # (`floor(pos/tpb)*tpb + g[slot]`) partia do pressuposto de que o melhor slot da grade
    # está sempre dentro do mesmo beat — mas quando o evento cai perto da borda do beat o
    # melhor slot é o 0 do beat *seguinte* (ou o último do anterior), e o índice `slot` não
    # carrega essa informação. O resultado era um erro exato de um tempo na nota escrita
    # (visto em ~6 % das notas da faixa de teste, sempre as que caem entre duas posições).
    # Varremos as posições do beat atual e dos dois vizinhos e escolhemos a mais próxima;
    # assim `ticks_abs` e `resid` vivem no mesmo espaço e a partitura fica auto-consistente.
    base = np.floor(pos / tpb) * tpb
    cand = base[:, None] + np.rint(np.asarray(g))[None, :]
    cand = np.concatenate((cand - tpb, cand, cand + tpb), axis=1)
    dabs = np.abs(pos[:, None] - cand)
    jbest = np.argmin(dabs, axis=1)
    ticks_abs = cand[np.arange(pos.size), jbest]
    resid = dabs[np.arange(pos.size), jbest]
    # promoção individual a 32ºs (só com evidência)
    promoted = 0
    allow32 = str(mode) in ("auto", "16th", "32nd+16th", "32nd") and sel["swing"] == 0.0
    if allow32:
        g32 = grid_positions(tpb, "32nd", 0.0)
        f32, _r32, _s32 = _grid_fit(pos, tpb, g32, w)
        fits["32nd@0.00"] = round(f32, 4)
        cand32 = base[:, None] + np.rint(np.asarray(g32))[None, :]
        cand32 = np.concatenate((cand32 - tpb, cand32, cand32 + tpb), axis=1)
        d32 = np.abs(pos[:, None] - cand32)
        j32 = np.argmin(d32, axis=1)
        t32 = cand32[np.arange(pos.size), j32]
        r32 = d32[np.arange(pos.size), j32]
        melhora = (resid - r32) > 0.25
        take = melhora & (resid > 0.58) & (w > 0.20) & ~np.isclose(np.mod(t32, 2.0), 0.0)
        if int(take.sum()) >= 2:
            ticks_abs = np.where(take, t32, ticks_abs)
            resid = np.where(take, r32, resid)
            promoted = int(take.sum())
    ticks_abs = np.where(resid > tol, np.rint(pos), ticks_abs)
    resid = np.where(resid > tol, np.abs(pos - np.rint(pos)), resid)
    # offset de downbeat: mantém a paridade da grade escolhida (um deslocamento ímpar
    # jogaria todas as notas em 32ºs e estragaria a escrita)
    step_off = passo_grade(g, tpb)          # a régua é a grade escolhida, não uma fórmula paralela
    off = int(round(float(tick_offset) / step_off) * step_off)
    tb = ticks_abs - off
    bar = np.floor(tb / tpr)
    tick_in_bar = np.clip((tb - bar * tpr).astype(np.int64), 0, tpr - 1)
    r1 = pos % tpb
    near = np.abs(r1 - tpb / 2.0) < tpb * 0.24
    sw_est = 0.0
    if int(near.sum()) >= 6:
        med = float(np.median(r1[near]))
        if tpb * 0.53 < med < tpb * 0.76:
            sw_est = round(med / tpb, 3)
    return {"ticks": tick_in_bar, "bars": bar.astype(np.int64), "resid": resid,
            "ticks_abs": ticks_abs, "mode": sel["mode"], "swing": float(sel["swing"]),
            "swing_detected": float(sw_est), "fitness": float(sel["fit"]),
            "offgrid": resid > tol, "tol_ticks": float(tol), "sec_per_tick": sec_per_tick,
            "positions": np.asarray(g), "tick_offset": int(off), "meter": meter,
            "best_fit_all": float(max(fits.values()) if fits else sel["fit"]),
            "n_candidates": len(fits), "swing_allowed": use_swing,
            "promoted_32nd": promoted, "fits": fits}


def refine_bpm(times: np.ndarray, bpm: float, phase: float, tpb: int, mode: str = "16th",
               weights: Optional[np.ndarray] = None, span_pct: float = 1.6,
               step: float = 0.02) -> dict:
    """
    Varre BPM (e fase) maximizando o *ajuste dos eventos à grade* — o objetivo que importa
    para escrever partitura (o pente/ACF maximiza correlação do envelope, que é mais plano
    e aceita ~0,3% de erro; isso aqui chega a <0,05%).
    """
    times = np.asarray(times, dtype=np.float64)
    if times.size < 8:
        return {"bpm": bpm, "phase_sec": phase, "fitness": 0.0, "refined": False}
    mode = "16th"          # a fase é sempre ancorada na grade-base de semicolcheias
    g = grid_positions(tpb, mode, 0.0)
    w = np.ones(times.size) if weights is None else np.asarray(weights, dtype=np.float64)
    best = (-1.0, bpm, phase)
    for k in range(1, 4):                            # coarse-to-fine
        lo = best[1] - span_pct / k
        hi = best[1] + span_pct / k
        st = step * k
        grid_bpm = np.arange(lo, hi, st)
        for B in grid_bpm:
            sec_tick = (60.0 / B) / TICKS_PER_QUARTER
            pos0 = (times - best[2]) / sec_tick
            for ph in np.linspace(0.0, tpb, 48, endpoint=False):
                fit, _r, _s = _grid_fit(pos0 - ph, tpb, g, w)
                if fit > best[0]:
                    best = (float(fit), float(B), float(best[2] + ph * sec_tick))
    # polimento de fase: remove o resíduo sistemático (mediana com sinal) — é o que
    # garante que "bumbo no tempo 1" saia no tick 0 e não no 1
    for _ in range(3):
        sec_tick = (60.0 / best[1]) / TICKS_PER_QUARTER
        pos0 = (times - best[2]) / sec_tick
        rel = pos0 % tpb
        d = rel[:, None] - g[None, :]
        wrap = np.where(np.abs(d) > tpb / 2.0, d - np.sign(d) * tpb, d)
        slot = np.argmin(np.abs(wrap), axis=1)
        sres = wrap[np.arange(pos0.size), slot]
        shift = float(np.median(sres[w > 0.05])) if int(np.sum(w > 0.05)) >= 4 else float(np.median(sres))
        if abs(shift) < 0.012:
            break
        best = (best[0], best[1], best[2] + shift * sec_tick)
    return {"bpm": round(best[1], 3), "phase_sec": best[2], "fitness": round(best[0], 4),
            "refined": abs(best[1] - bpm) > 1e-6}


def align_downbeat(ticks_abs: np.ndarray, lane_of: Sequence[str], weights: np.ndarray,
                   meter: str, sig: float = 0.60, k_open: int = 10) -> dict:
    """
    Escolhe o *offset* de início de compasso (em ticks) para os ticks já projetados na grade.

    Não decide a fórmula de compasso — só onde cai a barra. Isso importa porque a
    otimização de fase é cega a deslocamentos de um múltiplo do período da grade, e porque
    um backbeat simples de 4/4 (bumbo em 1 e 3, caixa em 2 e 4) é matematicamente
    invariante a um deslocamento de dois tempos: a partitura tem de escolher entre
    "bumbo no tempo 1" e "bumbo no tempo 3" com outra evidência. Usamos:

      1. ajuste ao template de backbeat (bumbo/caixa/prato), como média POR PEÇA
         ponderada por confiança — senão um offset que pesca mais falsos positivos ganha;
      2. a abertura da peça: os k primeiros golpes (ponderados) devem cair perto do
         início do compasso, porque gravações de bateria quase sempre começam no 1.

    Combinação: 0,72 · (1) + 0,28 · (2). Assim um encaixe ruim do padrão nunca é
    comprado pela abertura, e um empate exato do padrão é resolvido pela abertura.
    """
    T = METER_TEMPLATES.get(meter)
    t = np.asarray(ticks_abs, dtype=np.float64)
    if T is None or t.size == 0:
        return {"tick_offset": 0, "score": 0.0, "table": [], "note": "sem template"}
    tpr = T["tpr"]
    lanes = np.asarray(list(lane_of))
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if w.size != t.size:
        w = np.ones(t.size)
    masks = {}
    for g, ids in (("kick", ("kick",)),
                   ("snare", ("snare", "rim")),
                   ("crash", ("crash", "splash"))):
        m = np.zeros(t.size, dtype=bool)
        for i in ids:
            m |= (lanes == i)
        masks[g] = m
    gw = {"kick": 1.0, "snare": 1.0, "crash": 0.6}
    # evidência de abertura: os primeiros golpes *sustentados* (FPs de início de faixa
    # não podem dizer onde fica a barra), e o contraste de energia nos inícios de compasso
    k = min(int(k_open), t.size)
    strong = np.flatnonzero(w >= max(np.median(w), 1e-6))
    if strong.size < 4:
        strong = np.arange(t.size)
    io_ = strong[np.argsort(t[strong])][:3]
    raw: List[tuple] = []
    for off in range(0, int(tpr)):
        x = (t - off) % tpr
        num = den = 0.0
        for g in ("kick", "snare", "crash"):
            pos = T.get(g) or {}
            m = masks[g]
            if not int(m.sum()) or not pos:
                continue
            best = np.zeros(int(m.sum()))
            for tp in pos:
                d = np.abs(x[m] - float(tp))
                d = np.minimum(d, tpr - d)
                best = np.maximum(best, np.exp(-0.5 * (d / sig) ** 2) * float(pos[tp]))
            num += gw[g] * float(np.sum(best * w[m]))
            den += gw[g] * float(np.sum(w[m]))
        anti = 0.0
        for g, dd in (T.get("anti") or {}).items():
            m = masks.get(g)
            if m is None or not int(m.sum()):
                continue
            pen = np.zeros(int(m.sum()))
            for tp, wt in dd.items():
                d = np.abs(x[m] - float(tp))
                d = np.minimum(d, tpr - d)
                pen = np.maximum(pen, np.exp(-0.5 * (d / sig) ** 2) * float(wt))
            anti += float(np.sum(pen * w[m])) / (float(np.sum(w[m])) + 1e-9)
        tpl = (num / den if den > 0 else 0.0) - 0.35 * anti
        d0 = np.minimum(x[io_], tpr - x[io_])
        # "pelo menos um dos primeiros golpes fortes está na barra" — a média puniria os
        # golpes que caem em 2, 3 e 4 do MESMO compasso (que é o que se quer!)
        opening = float(np.max(np.exp(-0.5 * (d0 / 1.8) ** 2)))
        raw.append((0.72 * tpl + 0.28 * opening, off, float(tpl), float(opening)))
    raw.sort(key=lambda z: -z[0])
    tab = [{"tick_offset": int(o), "score": round(float(v), 4), "template": round(float(t_), 4),
            "opening": round(float(op), 4)} for v, o, t_, op in raw[:8]]
    best = raw[0]
    return {"tick_offset": int(best[1]), "score": round(float(best[0]), 4), "table": tab,
            "template": round(float(best[2]), 4), "opening": round(float(best[3]), 4)}


def estimate_swing(times: np.ndarray, bpm: float, phase_sec: float, tpb: int) -> float:
    sec_per_tick = 60.0 / max(1e-6, bpm) / TICKS_PER_QUARTER
    pos = (np.asarray(times) - phase_sec) / sec_per_tick
    if pos.size < 8:
        return 0.0
    rel = pos % tpb
    off = rel[(rel > tpb * 0.30) & (rel < tpb * 0.74)]
    if off.size < 6:
        return 0.0
    med = float(np.median(off))
    return round(med / tpb, 3) if tpb * 0.53 < med < tpb * 0.76 else 0.0
