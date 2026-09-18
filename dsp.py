"""
Núcleo de DSP para detecção de transientes de bateria e extração de features.

Bibliografia das técnicas usadas (implementadas de forma transparente, sem caixas-pretas):

  * Dixon, S. (2006) "Extensive evaluation of the spectral flux method for onset
    detection" — flux espectral com retificação de meia onda e limiar adaptativo.
  * Masuda (1998) / Sumandani & Kurozumi (2004) — HFC e wHFC: ponderar a diferença por
    magnitude realça transientes sobre sustain tonal.
  * Dähne, Meinhard & Zölner (2012) — complex-domain novelty: usa fase para rejeitar
    modulação de amplitude sustentada (crucial em pratos longos e em mix com baixo).
  * Schreiber et al. (2010) — prior log-normal de andamento, resolve a ambiguidade
    de oitava do autocorrelator.
  * Ellis (2007) — onset strength + alinhamento de grade por programação dinâmica
    (aqui: busca conjunta período×fase com pente de máximos).
  * Peeters (2004) — descritores espectrais (centroid, rolloff, flatness, bandas).

Arquitetura: detecção **por banda** (bumbo / caixa / pratos) com limiar e proeminência
próprios, seguida de **fusão** em eventos e classificação por impressão digital espectral.
Isso é essencial em bateria: o chimel em pianíssimo e o bumbo em fortíssimo estão em
ordens de grandeza diferentes e não podem compartilhar um limiar global.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy import signal as sps

EPS = 1e-10

# Agrupamentos usados na detecção multi-banda
DETECT_BANDS: Dict[str, dict] = {
    "low":   {"range": (28.0, 170.0),  "min_gap_ms": 62.0, "prom": 0.22, "weight": 1.00,
              "energy_rise": True},
    "mid":   {"range": (150.0, 3200.0), "min_gap_ms": 34.0, "prom": 0.20, "weight": 1.00,
              "energy_rise": False},
    "high":  {"range": (3200.0, 20000.0), "min_gap_ms": 28.0, "prom": 0.20, "weight": 1.00,
              "energy_rise": False},
}


# --------------------------------------------------------------------------------------
# STFT
# --------------------------------------------------------------------------------------

@dataclass
class Spectrogram:
    S: np.ndarray            # (frames, bins) complex64, um lado
    freqs: np.ndarray        # (bins,) Hz
    sr: int
    hop: int
    n_fft: int

    @property
    def fps(self) -> float:
        return self.sr / self.hop

    @property
    def n_frames(self) -> int:
        return self.S.shape[0]

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) * self.hop / self.sr

    def bin_range(self, f0: float, f1: float) -> slice:
        i0 = int(np.searchsorted(self.freqs, f0, side="left"))
        i1 = int(np.searchsorted(self.freqs, f1, side="right"))
        i1 = max(i1, i0 + 1)
        return slice(max(0, i0), min(len(self.freqs), i1))

    def frame_range(self, t0: float, t1: float) -> slice:
        a = int(round(t0 * self.fps))
        b = int(round(t1 * self.fps))
        return slice(max(0, a), min(self.n_frames, max(a + 1, b)))

    def times_for(self, frames: np.ndarray) -> np.ndarray:
        return np.asarray(frames, dtype=np.float64) * self.hop / self.sr


def _window(kind: str, n: int) -> np.ndarray:
    if kind == "hann":
        return np.hanning(n + 1)[:-1].astype(np.float32)
    if kind == "hamming":
        return np.hamming(n + 1)[:-1].astype(np.float32)
    if kind == "blackman":
        return np.blackman(n + 1)[:-1].astype(np.float32)
    return np.ones(n, dtype=np.float32)


# Custo de memória do STFT é o que limita a duração analisável: a grade tem n_quadros × n_fft
# elementos e *qualquer* cópiasela custa dezenas de MB por minuto de áudio. A versão anterior
# materializava (i) uma matriz de índices int64, (ii) "frames", (iii) a janela aplicada, (iv) um
# .astype() redundante e (v) o resultado da FFT em complex128 — picos de 1,0 GB para 160 s e OOM
# (processo morto pelo kernel) com faixas reais de 3-5 min. Aqui a grade é lida por vista
# estrideada e transformada em blocos, sempre em precisão simples: só o `S` final é retido.
BLOCK_FRAMES = 4096


def stft(x: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256,
         win: str = "hann", block: int = BLOCK_FRAMES) -> Spectrogram:
    """STFT de precisão simples, em blocos (pico de memória ≈ S + um bloco)."""
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy.fft import rfft as _rfft          # scipy respeita float32 → complex64; np.fft não

    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if n_fft % 2:
        n_fft += 1
    pad = n_fft // 2
    nfr = 1 + max(0, (x.size + 2 * pad - n_fft) // hop)
    w = _window(win, n_fft)
    nb = n_fft // 2 + 1
    S = np.empty((nfr, nb), dtype=np.complex64)
    if nfr:
        xp = np.pad(x, (pad, pad))
        jan = sliding_window_view(xp, n_fft)[:(nfr - 1) * hop + n_fft][::hop]
        blk = min(int(block), nfr)
        for i0 in range(0, nfr, blk):
            n = min(blk, nfr - i0)
            S[i0:i0 + n] = _rfft(jan[i0:i0 + n] * w[None, :], n=n_fft, axis=1, workers=1)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float32)
    return Spectrogram(S=S, freqs=freqs, sr=sr, hop=hop, n_fft=n_fft)


def _mag(S: np.ndarray, compression: float = 0.5) -> np.ndarray:
    m = np.abs(S).astype(np.float32)
    if compression and compression != 1.0:
        m = np.power(m + 1e-7, compression)
    return m


# --------------------------------------------------------------------------------------
# Novelties por banda
# --------------------------------------------------------------------------------------

#: piso aplicado a features de banda inexistente (10·log10(1e-16)); o classificador lê como
#: "nenhuma energia ali", que é o que há — ver o ramo `sl.stop <= sl.start` em extract_features.
_PISO_DB = -160.0

#: Piso da janela de branqueamento de `band_flux`, em segundos de música (ver comentário no uso).
BASE_MAX_S = 15.0


def band_flux(spec: Spectrogram, f0: float, f1: float, kind: str = "hfc",
              whitening: bool = True, use_phase: bool = True,
              block: int = 8 * BLOCK_FRAMES) -> np.ndarray:
    """
    Função de novidade restrita a uma faixa de frequência.
      kind: 'flux' | 'hfc' | 'complex' | 'mix'  ('mix' = hfc + complex, o padrão)

    Reescrita por causa de memória (docs/ERROS.md A13): a formulação antiga materializava
    `pred` em complex128 (o `np.exp` de um ângulo float64 arrasta a conta toda para dupla),
    mais a subtração e dois `np.abs` — ~40 B por bin por quadro, 5× o próprio espectrograma,
    e isso por banda. Com faixas de 3 min isso passava de 1 GB e o kernel matava o processo.
    Aqui tudo fica em float32/complex64 e o termo de fase é somado em blocos de quadros, então
    o transitório máximo é do tamanho de um bloco. A aritmética (ordem dos fatores incluída) é
    a mesma de antes; o que muda é a precisão do termo de fase, ~1e-7 relativo.
    """
    sl = spec.bin_range(f0, f1)
    if sl.stop <= sl.start:
        return np.zeros(spec.n_frames, dtype=np.float32)
    S = spec.S[:, sl]
    nfr = S.shape[0]
    m = np.abs(S)                                   # float32 (complex64 → |·| float32)
    m += 1e-7
    np.power(m, 0.5, out=m)                         # compressão de meio, no lugar
    if whitening:
        # A janela do piso de branqueamento era `nfr // 3` — um terço do *arquivo inteiro*. Numa
        # faixa de 42 s isso são 14 s (razoável); numa de 7 min são 144 s, e "normalizar local"
        # vira normalizar global: a introdução silenciosa de uma gravação comprime o piso de tudo
        # que vem depois e os ataques suaves desaparecem do detector. Limita-se a janela por
        # tempo (A20); acima de 15 s de música ela já não é local de qualquer forma, e o
        # limite não toca a demo calibrada (14,0 s), que continua bit a bit a mesma.
        base = ndimage.uniform_filter1d(m, size=max(9, min(nfr // 3, int(BASE_MAX_S * spec.fps))),
                                        axis=0, mode="nearest")
        base += 1e-8
        m /= base
        del base
    pos = np.diff(m, n=1, axis=0)
    np.maximum(pos, 0.0, out=pos)                    # d e pos eram o mesmo array em dobro
    if kind == "flux":
        o = pos.sum(axis=1)
    elif kind == "hfc":
        mm = m[1:] + m[:-1]
        mm *= pos
        mm *= 0.5
        o = mm.sum(axis=1)
    elif kind in ("complex", "mix"):
        bins = spec.freqs[sl]
        pred_ang = (2.0 * np.pi * bins * spec.hop / spec.n_fft).astype(np.float32)
        fac = np.exp(1j * pred_ang).astype(np.complex64)     # complex64: metade do bytes/valor
        c = np.empty(nfr - 1, dtype=np.float32)
        blk = max(256, min(int(block), nfr - 1))
        for i0 in range(0, nfr - 1, blk):
            i1 = min(i0 + blk, nfr - 1)
            a = S[i0:i1]
            t = np.abs(S[i0 + 1:i1 + 1] - a * fac[None, :])   # |S[t] − pred|, float32
            t -= np.abs(a)
            np.maximum(t, 0.0, out=t)
            c[i0:i1] = t.sum(axis=1)
        if kind == "complex":
            o = c
        else:
            h = m[1:] + m[:-1]
            h *= pos
            h *= 0.5
            h = h.sum(axis=1)
            o = _unit(h) + _unit(c)
    else:
        raise ValueError(kind)
    o = np.asarray(o, dtype=np.float32)
    if o.size == spec.n_frames - 1:          # alinha d[t] = f[t]-f[t-1] com o frame t
        o = np.concatenate([[np.float32(0.0)], o])
    return o


def envelope_hilbert(x: np.ndarray) -> np.ndarray:
    """Envoltória |sinal analítico| em precisão simples, sem passar por float64.

    `scipy.signal.hilbert` devolve complex64 quando o input é float32, mas por dentro chama
    `np.fft.fft`, que trabalha em complex128: medido, um vetor de 6,8 M amostras custava 55 MB
    de saída e 520 MB de pico — era isso que matava a faixa longa (docs/ERROS.md A13). Aqui o
    FFT é o do SciPy (respeita float32 → complex64) e o filtro de Hilbert é aplicado em place.
    """
    from scipy.fft import fft as _fft, ifft as _ifft
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    if n < 2:
        return np.abs(x).astype(np.float32)
    X = _fft(x, workers=1)                      # complex64 (float32 → complex64 no SciPy)
    # Sinal analítico: dobra as frequências positivas, zera as negativas, preserva DC e Nyquist.
    h = np.zeros(n, dtype=np.float32)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    X *= h                                       # em place: nenhuma cópia extra
    y = _ifft(X, workers=1)
    del X, h
    return np.abs(y).astype(np.float32, copy=False)



def energy_rise(x: np.ndarray, sr: int, band: Tuple[float, float], times: np.ndarray,
                smooth_ms: float = 5.0, causal: bool = False) -> np.ndarray:
    """
    "Energy rise" de banda filtrada, amostrado nos tempos dos frames do STFT.
    Em 28–170 Hz um frame de 23 ms cobre menos de um ciclo: o flux espectral é pobre
    aqui e o envelope da banda é o detector correto para bumbo (Müller, cap. 8).
    """
    nyq = sr / 2.0
    lo = max(15.0, band[0]) / nyq
    hi = min(band[1], nyq * 0.98) / nyq
    if hi <= lo + 0.005:
        hi = min(0.98, lo + 0.02)
    b, a = sps.butter(2, [lo, hi], btype="bandpass")
    # O envelope era o maior alocação da cadeia inteira: com coeficientes float64 o `filtfilt`
    # sobe o sinal para float64 e o `hilbert` devolve complex128 — 405 MB para 41 s, ~1,6 GB
    # para 3 min, e o kernel matava o processo (docs/ERROS.md A13). Aqui o caminho inteiro é
    # float32/complex64: 2× menos memória e o `xf` é liberado antes de qualquer outra cópia.
    # O filtro fica em float64 de propósito: com coeficientes float32 este butter de 2ª ordem
    # com polo em ~21 Hz é instável (medido: NaN na saída). O que dá para cortar sem mover nada é
    # o resto do caminho — o envelope de Hilbert era a maior alocação da cadeia inteira (16 B por
    # amostra em complex128, ~405 MB para 41 s, ~1,6 GB para 3 min, e o kernel matava o processo:
    # docs/ERROS.md A13). Reduzido a float32/complex64 são 2× menos, com o `xf` liberado antes da
    # próxima cópia. Chegou-se a testar dizimar a envoltória (mais 8×), e foi **rejeitado por
    # medida**: mudava máximos locais do detector em ~4e-3 relativo. Precisão da partitura não se
    # troca por RAM — quem precisa de mais folga sobe `hop`/`analysis_sr`, e o orçamento abaixo
    # recusa a faixa com aviso em vez de degradar a escrita.
    padlen = min(3 * max(len(a), len(b)), max(1, x.size - 1))
    if causal:
        xf = sps.lfilter(b, a, x, axis=0)
    else:
        xf = sps.filtfilt(b, a, x, padlen=padlen)
    env = envelope_hilbert(xf)
    del xf
    k = max(1, int(smooth_ms * 1e-3 * sr))
    env = ndimage.uniform_filter1d(env, size=k, mode="nearest")
    idx = np.clip(np.rint(np.asarray(times) * sr).astype(np.int64), 0, env.size - 1)
    e = env[idx]
    d = np.diff(e, prepend=e[0])
    return np.maximum(d, 0.0).astype(np.float32)


def _unit(o: np.ndarray) -> np.ndarray:
    o = np.asarray(o, dtype=np.float32)
    return o / (np.percentile(o, 99.5) + EPS) if o.size > 30 else o / (o.max() + EPS)


def detrend(o: np.ndarray, fps: float, win_ms: float = 170.0) -> np.ndarray:
    """
    Remove a linha de base local por mediana móvel (robusta a outliers) e devolve a
    grandeza relativa ao espalhamento local — é isso que permite enxergar um chimel
    pianíssimo no meio de um padrão denso, onde um limiar global falharia.
    """
    o = np.asarray(o, dtype=np.float32)
    if o.size < 5:
        return np.zeros_like(o)
    w = max(3, int(win_ms * 1e-3 * fps)) | 1
    base = ndimage.median_filter(o, size=w, mode="nearest")
    dev = o - base
    scale = np.percentile(dev, 99) if dev.size > 40 else (dev.max() + EPS)
    if scale <= EPS:
        scale = float(np.std(dev) * 3.0) + EPS
    out = np.maximum(dev, 0.0) / (scale + EPS)
    return out.astype(np.float32)


def moving_floor(o: np.ndarray, fps: float, win_s: float, frac: float) -> np.ndarray:
    w = max(9, int(win_s * fps)) | 1
    p = ndimage.percentile_filter(o, percentile=int(frac * 100), size=w, mode="nearest")
    return p


def find_onset_frames(o: np.ndarray, fps: float, sens: float = 1.0, min_gap_ms: float = 34.0,
                      prom_frac: float = 0.20, win_s: float = 2.2, floor_frac: float = 0.18,
                      extra_height: float = 0.0):
    """
    Picos locais com (a) altura acima de um piso percentil-móvel e (b) proeminência
    mínima relativa — os dois critérios em conjunto são bem mais estáveis que o
    clássico "média + k·σ" quando o padrão é denso.

    Devolve (frames, proeminência_relativa). A proeminência normalizada pela escala da
    própria banda é a grandeza que diz se aquele grupo really viu um ataque seu, ou só
    o vazamento espectral do golpe de outra peça — a informação que resolve acorde.
    """
    if o.size < 8:
        return np.zeros(0, dtype=np.int64)
    d = max(1, int(min_gap_ms * 1e-3 * fps))
    floor = moving_floor(o, fps, win_s, floor_frac) * sens
    height = np.maximum(floor, extra_height)
    hi = np.percentile(o, 99) if o.size > 50 else o.max()
    prom = max(1e-4, prom_frac * hi / max(0.35, sens))
    pk, props = sps.find_peaks(o, height=height, distance=d, prominence=prom)
    if pk.size == 0:
        pk, props = sps.find_peaks(o, height=np.percentile(o, 88), distance=d)
    scale = (np.percentile(o, 99) if o.size > 50 else (o.max() + EPS)) + EPS
    pr = np.asarray(props.get("prominences", np.zeros(pk.size)), dtype=np.float32) / scale
    return pk.astype(np.int64), np.clip(pr, 0.0, 4.0)


def refine_onset_frames(o: np.ndarray, frames: np.ndarray, fps: float,
                        search_ms: float = 22.0, frac: float = 0.45) -> np.ndarray:
    """
    Refina para a borda de subida: último ponto antes do pico que cruza 20% do valor
    de pico. Remove o atraso sistemático do filtragem/suavização (≈1–2 frames) e
    reduz o erro de quantização — importa a partir de ~150 BPM, onde a semicolcheia
    vale <100 ms.
    """
    if frames.size == 0:
        return frames.astype(np.float64)
    w = max(2, int(search_ms * 1e-3 * fps))
    out = np.array(frames, dtype=np.float64)
    for j, fr in enumerate(frames):
        a, b = max(0, int(fr) - w), min(o.size, int(fr) + 2)
        if b - a < 3:
            continue
        seg = o[a:b]
        pk = int(np.argmax(seg))
        pkv = float(seg[pk])
        if pkv <= EPS or pk == 0:
            continue
        thr = float(np.clip(frac, 0.05, 0.9)) * pkv
        i = pk
        while i > 0 and seg[i] > thr:
            i -= 1
        frac = 0.0
        if i + 1 < seg.size and seg[i + 1] > seg[i]:
            frac = (thr - seg[i]) / (seg[i + 1] - seg[i] + EPS)
        out[j] = a + i + float(np.clip(frac, 0.0, 1.0))
    return out


# --------------------------------------------------------------------------------------
# Features por evento
# --------------------------------------------------------------------------------------

@dataclass
class OnsetFeatures:
    n: int
    times: np.ndarray                  # segundos
    band_db: np.ndarray                # (n, nbands)
    band_rel: np.ndarray               # (n, nbands) fração de energia
    centroid: np.ndarray = None        # type: ignore[assignment]
    rolloff: np.ndarray = None         # type: ignore[assignment]
    flatness: np.ndarray = None        # type: ignore[assignment]
    pitch_low: np.ndarray = None       # type: ignore[assignment]
    attack_ms: np.ndarray = None       # type: ignore[assignment]
    decay_low_ms: np.ndarray = None    # type: ignore[assignment]
    decay_hi_ms: np.ndarray = None     # type: ignore[assignment]
    rise_ratio: np.ndarray = None      # type: ignore[assignment]
    novelty: np.ndarray = None         # type: ignore[assignment] (n, ngroups) proeminência por banda
    loud: np.ndarray = None            # type: ignore[assignment] energia total (dB)
    hi_drop_db: np.ndarray = None      # type: ignore[assignment] queda da banda aguda até o próximo evento
    low_drop_db: np.ndarray = None     # type: ignore[assignment] queda da banda grave
    win_ms: np.ndarray = None          # type: ignore[assignment] janela efetiva de medição (ms)
    hi_slope: np.ndarray = None        # type: ignore[assignment] dB/s da banda aguda (sustain vs. curto)
    hf_centroid: np.ndarray = None     # type: ignore[assignment] centróide (Hz) restrito a > 2,2 kHz
    sub_ratio: np.ndarray = None       # type: ignore[assignment] E(20–62Hz) / E(20–170Hz): assinatura do bumbo
    band_delta_db: np.ndarray = None   # type: ignore[assignment] (n, nb) dB a mais que o pré-evento (ring-out removido)
    band_delta_rel: np.ndarray = None  # type: ignore[assignment] (n, nb) fração do delta positivo (separação suave de fontes)
    mid_slope: np.ndarray = None       # type: ignore[assignment] dB/s da banda 180–1500 Hz (corpo)
    low_slope: np.ndarray = None       # type: ignore[assignment] dB/s da banda grave
    band_peak_db: np.ndarray = None    # type: ignore[assignment] (n, nbands) nível de ataque por banda
    band_names: List[str] = field(default_factory=list)


def extract_features(spec: Spectrogram, x: np.ndarray, times: np.ndarray,
                     bands: Dict[str, Tuple[float, float]],
                     novelty_streams: Optional[Dict[str, np.ndarray]] = None,
                     attack_ms_win: float = 55.0, full_ms_win: float = 300.0) -> OnsetFeatures:
    """
    'Impressão digital' de cada evento: energia por banda nos 55 ms seguintes ao ataque,
    descritores espectrais, tempo de ataque, decaimento por banda e fundamental do corpo.
    """
    names = list(bands.keys())
    nb = len(names)
    n = times.size
    zero = lambda: np.zeros(max(n, 0), dtype=np.float32)
    f = OnsetFeatures(n=n, times=np.asarray(times, dtype=np.float64),
                      band_db=np.zeros((max(n, 0), nb), np.float32),
                      band_rel=np.zeros((max(n, 0), nb), np.float32),
                      centroid=zero(), rolloff=zero(), flatness=zero(), pitch_low=zero(),
                      attack_ms=zero(), decay_low_ms=zero(), decay_hi_ms=zero(),
                      rise_ratio=zero(), loud=zero(), hi_drop_db=zero(), low_drop_db=zero(),
                      win_ms=zero(), hi_slope=zero(), low_slope=zero(), mid_slope=zero(), hf_centroid=zero(),
                      sub_ratio=np.full(max(n, 0), 0.5, np.float32),
                      band_delta_db=np.zeros((max(n, 0), nb), np.float32),
                      band_delta_rel=np.zeros((max(n, 0), nb), np.float32),
                      band_peak_db=np.zeros((max(n, 0), nb), np.float32), band_names=names)
    if n == 0:
        if novelty_streams:
            f.novelty = np.zeros((0, len(novelty_streams)), np.float32)
            f.band_names = names
        return f

    m2 = _mag(spec.S, compression=2.0)                     # potência linear
    fps = spec.fps
    awin = max(2, int(attack_ms_win * 1e-3 * fps))
    fwin = max(awin + 2, int(full_ms_win * 1e-3 * fps))
    band_sl = {k: spec.bin_range(*b) for k, b in bands.items()}
    hi_cut = min(len(spec.freqs), max(4, int((spec.sr / 2 * 0.96))))
    freqs = spec.freqs[:hi_cut]
    lo_sl = spec.bin_range(26.0, 340.0)
    nz_sl = spec.bin_range(1400.0, min(16000.0, spec.sr / 2 * 0.95))
    frames = np.rint(times * fps).astype(np.int64)

    for j in range(n):
        fr = int(frames[j])
        a = max(0, min(fr, spec.n_frames - 2))
        b = min(spec.n_frames, a + awin)
        nx = int(frames[j + 1]) if j + 1 < n else spec.n_frames
        floor_d = min(spec.n_frames, a + max(awin, int(0.085 * fps)))
        d = min(spec.n_frames, a + fwin, max(floor_d, nx))
        dl = min(spec.n_frames, a + int(0.42 * fps))          # janela longa p/ sustain
        f.win_ms[j] = 1000.0 * (d - a) / fps
        if b <= a + 1:
            b = min(spec.n_frames, a + 2)
        seg = m2[a:b]
        pre_a = max(0, a - max(2, int(0.040 * fps)))
        seg_pre = m2[pre_a:a] if a - pre_a >= 1 else seg[:1]
        for i, k in enumerate(names):
            sl = band_sl[k]
            if sl.stop <= sl.start:
                # A banda inteira está acima de Nyquist (taxa de análise baixa demais para
                # ela): não há bins ali. Tratar como "energia ausente" é a resposta honesta —
                # reassinar bins de outra faixa para preencher faria o classificador ouvir
                # chimbal onde só existe caixa. O aviso de quem ficou mudo sai no relatório.
                f.band_db[j, i] = _PISO_DB
                f.band_delta_db[j, i] = 0.0
                f.band_peak_db[j, i] = _PISO_DB
                continue
            f.band_db[j, i] = float(10.0 * np.log10(seg[:, sl].sum() + 1e-9))
            lv = float(10.0 * np.log10(seg[:, sl].mean() + 1e-12))
            pv = float(10.0 * np.log10(seg_pre[:, sl].mean() + 1e-12))
            f.band_delta_db[j, i] = lv - pv
            f.band_peak_db[j, i] = float(10.0 * np.log10(seg[:, sl].max(axis=1).max() + 1e-9))
        tot = np.power(10.0, f.band_db[j] / 10.0)
        f.band_rel[j] = tot / (tot.sum() + EPS)
        dlt = np.maximum(f.band_delta_db[j], 0.0)
        f.band_delta_rel[j] = dlt / (dlt.sum() + EPS)
        e_sub = seg[:, spec.bin_range(18.0, 62.0)].sum()
        e_bass = seg[:, spec.bin_range(18.0, 170.0)].sum()
        f.sub_ratio[j] = float(e_sub / (e_bass + EPS))
        prof = seg[:, :hi_cut].sum(axis=0) + 1e-12
        hfc_sl = spec.bin_range(2200.0, min(16000.0, spec.sr / 2 * 0.95))
        p_hf = seg[:, hfc_sl].sum(axis=0) + 1e-12
        fr_hf = spec.freqs[hfc_sl]
        f.hf_centroid[j] = float(np.sum(p_hf * fr_hf) / (p_hf.sum() + EPS))
        s_all = prof.sum()
        f.centroid[j] = float(np.sum(prof * freqs) / (s_all + EPS))
        csum = np.cumsum(prof)
        f.rolloff[j] = float(freqs[min(int(np.searchsorted(csum, 0.85 * csum[-1])), len(freqs) - 1)])
        nzm = prof[nz_sl] + 1e-12
        f.flatness[j] = float(np.exp(np.mean(np.log(nzm))) / (np.mean(nzm) + EPS))
        f.loud[j] = float(10.0 * np.log10(m2[a:d].sum() + 1e-9))
        # envelope total → ataque, rise ratio, decaimentos
        e_all = m2[a:d, :].sum(axis=1)
        if e_all.size > 4:
            head = float(e_all[: min(3, e_all.size)].sum())
            tail = float(e_all[3:].sum())
            f.rise_ratio[j] = float(np.clip(head / (head + tail + EPS) * 3.0, 0, 3))
            pk = int(np.argmax(e_all[:6]))
            pkv = float(e_all[pk]) + EPS
            i90 = int(np.argmax(e_all >= 0.9 * pkv))
            i10 = int(np.argmax(e_all >= 0.1 * pkv))
            f.attack_ms[j] = float(max(0, i90 - i10) * 1000.0 / fps)
            f.decay_low_ms[j] = _decay_frames(e_all, pk) * 1000.0 / fps
            e_lo = m2[a:d, spec.bin_range(26.0, 260.0)].sum(axis=1)
            f.low_drop_db[j] = _drop_db(e_lo)
            e_all_long = m2[a:dl, :].sum(axis=1)
            f.low_slope[j] = _decay_slope(e_all_long, min(pk, e_all_long.size - 2), fps)
            e_hi = m2[a:d, spec.bin_range(2500.0, min(16000.0, spec.sr / 2 * 0.95))].sum(axis=1)
            if e_hi.size > 4:
                f.decay_hi_ms[j] = _decay_frames(e_hi, int(np.argmax(e_hi[:4]))) * 1000.0 / fps
            f.hi_drop_db[j] = _drop_db(e_hi)
            e_hi_long = m2[a:dl, spec.bin_range(2500.0, min(16000.0, spec.sr / 2 * 0.95))].sum(axis=1)
            f.hi_slope[j] = _decay_slope(e_hi_long, int(np.argmax(e_hi_long[:4])), fps)
            e_mid = m2[a:dl, spec.bin_range(170.0, 1500.0)].sum(axis=1)
            if e_mid.size > 4:
                f.mid_slope[j] = _decay_slope(e_mid, int(np.argmax(e_mid[:4])), fps)
        avg = m2[a:b, lo_sl].mean(axis=0)
        if avg.size > 3:
            k = int(np.argmax(avg))
            if 0 < k < avg.size - 1 and avg[k] > 1e-8:
                y0, y1, y2 = (np.log(avg[k - 1] + EPS), np.log(avg[k] + EPS), np.log(avg[k + 1] + EPS))
                den = (y0 - 2 * y1 + y2)
                delta = 0.5 * (y0 - y2) / (den if abs(den) > 1e-9 else EPS)
                fb = spec.freqs[lo_sl.start] + (k + np.clip(delta, -1, 1)) * (spec.sr / spec.n_fft)
                f.pitch_low[j] = float(fb) if 26 < fb < 420 else 0.0

    f.rise_ratio = _center01(f.rise_ratio)
    f.flatness = np.clip(f.flatness * 1.6, 0.0, 1.0)
    if novelty_streams:
        keys = list(novelty_streams.keys())
        nm = np.zeros((n, len(keys)), dtype=np.float32)
        for gi, gk in enumerate(keys):
            s = novelty_streams[gk]
            idx = np.clip(frames, 0, s.size - 1)
            lo = np.clip(frames - max(1, int(0.028 * fps)), 0, s.size - 1)
            hi = np.clip(frames + max(1, int(0.028 * fps)), 0, s.size - 1)
            for j in range(n):
                nm[j, gi] = float(np.nanmax(s[lo[j]: hi[j] + 1]) if hi[j] > lo[j] else s[idx[j]])
        f.novelty = nm
    return f


def _drop_db(env: np.ndarray, frac_win: float = 0.34) -> float:
    """
    Queda (dB) entre o pico da envoltória e o *piso* da cauda da janela.
    Usar o mínimo da cauda (e não o máximo) torna a medida imune à contaminação
    pelo evento seguinte — que em chimel denso chega 60–100 ms depois.
    """
    if env.size < 4:
        return 0.0
    pk_i = int(np.argmax(env[: min(5, env.size)]))
    pk = float(env[pk_i])
    if pk <= 1e-9:
        return 0.0
    tail = env[pk_i:]
    k = max(2, int(tail.size * frac_win))
    end = float(np.min(tail[-k:])) + 1e-9
    return float(max(0.0, 20.0 * math.log10(pk / end)))


def _decay_slope(env: np.ndarray, peak_i: int, fps: float) -> float:
    """
    Taxa de decaimento em dB/s, estimada como

        slope = -20·log10( pico / percentil25(cauda) ) / duração_da_janela

    Usar o percentil 25 da cauda (e não o valor instantâneo nem o mínimo) dá a *cauda
    real* do som sem ser enganado nem pelo ataque do golpe seguinte nem por um único frame
    de ruído: é o "sustain" que separa chimel (curto) de ride/crash (longo) mesmo num
    padrão denso de 16ºs.
    """
    if env.size < 5 or peak_i >= env.size - 3:
        return 0.0
    tail = np.asarray(env[peak_i:], dtype=np.float64)
    pk = float(np.max(tail[: max(2, min(4, tail.size))])) + 1e-9
    q = float(np.percentile(tail, 25)) + 1e-9
    if pk <= q:
        return 0.0
    drop = 20.0 * math.log10(pk / q)
    dur = max(0.02, tail.size / max(1e-6, fps))
    return float(-drop / dur)


def _decay_frames(env: np.ndarray, peak_i: int, frac: float = 0.25) -> float:
    """Frames até a envolver cair a `frac` do pico (~ -12 dB)."""
    if peak_i >= env.size - 2:
        return float(max(0, env.size - 1 - peak_i))
    pk = float(env[peak_i])
    if pk <= 1e-9:
        return 0.0
    tail = env[peak_i:]
    below = np.nonzero(tail < frac * pk)[0]
    return float(below[0]) if below.size else float(tail.size)


def _center01(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    lo, hi = np.percentile(v, 3), np.percentile(v, 97)
    if hi - lo < EPS:
        return np.full_like(v, 0.5)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def zscore_per_band(band_db: np.ndarray) -> np.ndarray:
    if band_db.size == 0:
        return band_db
    mu = np.median(band_db, axis=0, keepdims=True)
    sd = (np.percentile(band_db, 84, axis=0, keepdims=True) -
          np.percentile(band_db, 16, axis=0, keepdims=True)) / 2.0
    sd = np.maximum(sd, 1.2)
    return np.clip((band_db - mu) / sd, -4.0, 4.0)


def onset_strength_envelope(spec: Spectrogram, x: np.ndarray, sr: int,
                            weights: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Envelope global de novidade (soma ponderada por banda, cada uma já 'detrendada'
    e normalizada pela sua própria escala) — é o sinal usado para a busca de andamento.
    """
    streams: Dict[str, np.ndarray] = {}
    for k, cfg in DETECT_BANDS.items():
        f0, f1 = cfg["range"]
        s = detrend(band_flux(spec, f0, f1, kind="mix"), spec.fps)
        if cfg["energy_rise"]:
            s = np.maximum(s, detrend(energy_rise(x, sr, (f0 * 0.75, f1 * 0.55),
                                                   spec.times), spec.fps))
        streams[k] = s.astype(np.float32)
    n = min(s.size for s in streams.values())
    env = np.zeros(n, dtype=np.float32)
    for k, s in streams.items():
        env += float(weights.get(k, 1.0)) * s[:n]
    env = ndimage.uniform_filter1d(env, size=max(1, int(5e-3 * spec.fps)), mode="nearest")
    env = env / (np.percentile(env, 99.5) + EPS)
    return np.clip(env, 0.0, 1.6).astype(np.float32), streams


def detect_events(spec: Spectrogram, x: np.ndarray, sr: int, sens: float = 1.0,
                  merge_ms: float = 13.0, bands: Optional[Dict[str, dict]] = None,
                  refine_frac: float = 0.45) -> dict:
    """
    Detecção por grupo de bandas e *fusão apenas do instante de referência*.

    Cada grupo (grave / médio / agudo) produz sua própria lista de picos, com limiar,
    gap mínimo e proeminência próprios. Eventos de grupos diferentes que caem a menos
    de `merge_ms` compartilham o mesmo tempo de ataque (é um acorde: bumbo + chimel),
    mas **não** são colapsados em um único evento: cada grupo mantém seu candidato,
    que será classificado dentro do seu próprio subconjunto de pistas.
    """
    cfgs = bands or DETECT_BANDS
    per_band: Dict[str, np.ndarray] = {}
    prom_band: Dict[str, np.ndarray] = {}
    streams: Dict[str, np.ndarray] = {}
    for k, cfg in cfgs.items():
        f0, f1 = cfg["range"]
        s_ = detrend(band_flux(spec, f0, f1, kind="mix"), spec.fps)
        if cfg.get("energy_rise"):
            s_ = np.maximum(s_, detrend(energy_rise(x, sr, (f0 * 0.75, f1 * 0.55), spec.times,
                                                  smooth_ms=13.0), spec.fps, win_ms=220.0))
        s_ = ndimage.uniform_filter1d(s_, size=max(1, int(4e-3 * spec.fps)), mode="nearest")
        streams[k] = s_.astype(np.float32)
        pk, pr = find_onset_frames(
            s_, spec.fps, sens=sens * float(cfg.get("weight", 1.0)),
            min_gap_ms=cfg["min_gap_ms"], prom_frac=cfg["prom"],
            floor_frac=cfg.get("floor", 0.18))
        per_band[k] = pk
        prom_band[k] = pr

    keys = list(cfgs.keys())
    pairs: List[Tuple[int, str, float]] = []
    for k in keys:
        for fr, pm in zip(per_band[k], prom_band[k]):
            pairs.append((int(fr), k, float(pm)))
    pairs.sort()
    gap = max(1, int(merge_ms * 1e-3 * spec.fps))
    clusters: List[dict] = []
    for fr, k, pm in pairs:
        if clusters and fr - clusters[-1]["frames"][-1] <= gap:
            if k not in clusters[-1]["groups"]:
                clusters[-1]["groups"].append(k)
            clusters[-1]["frames"].append(fr)
            clusters[-1]["st"][k] = max(clusters[-1]["st"].get(k, 0.0), pm)
        else:
            clusters.append({"frames": [fr], "groups": [k], "st": {k: pm}})
    if not clusters:
        return {"n": 0, "events": [], "times": np.zeros(0), "streams": streams,
                "per_group": {k: np.zeros(0, np.int64) for k in keys}, "frames": np.zeros(0, np.int64)}
    times_s = np.array([float(np.mean(c["frames"])) for c in clusters], dtype=np.float64)
    frames_ref = refine_onset_frames(np.max(np.stack([streams[k] for k in keys]), axis=0),
                                    times_s, spec.fps, frac=refine_frac)
    times_s = np.rint(frames_ref) * spec.hop / spec.sr
    for i, c in enumerate(clusters):
        c["frame"] = int(np.rint(frames_ref[i]))
        c["time"] = float(times_s[i])
    # índice por grupo → eventos
    ev_index_of: Dict[str, List[int]] = {k: [] for k in keys}
    st_of: Dict[str, List[float]] = {k: [] for k in keys}
    for i, c in enumerate(clusters):
        for g in c["groups"]:
            if g in ev_index_of:
                ev_index_of[g].append(i)
                st_of[g].append(float(c["st"].get(g, 0.0)))
    return {"n": len(clusters), "events": clusters, "times": times_s,
            "st_by_group": {k: np.array(v, dtype=np.float32) for k, v in st_of.items()},
            "by_group": {k: np.array(v, dtype=np.int64) for k, v in ev_index_of.items()},
            "groups": [c["groups"] for c in clusters],
            "streams": streams, "per_group": {k: per_band[k] for k in keys},
            "frames": np.rint(frames_ref).astype(np.int64)}
