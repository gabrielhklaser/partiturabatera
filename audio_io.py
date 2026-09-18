"""Entrada e saída de áudio: decodificação robusta e preparação para análise.

Formatos aceitos (na ordem das rotas): **miniaudio** → **soundfile/libsndfile** → **ffmpeg**.
Cada rota devolve ``(amostras, sr, canais)`` em float32 ou ``None``; a primeira que funcionar
vence. Se nenhuma funcionar, o erro sobe como ``ValueError`` com as três tentativas — nunca um
áudio silencioso, que é o jeito mais caro de falhar (a transcrição "de sucesso" de um vazio).

O resto do módulo mantém o contrato que o resto do pacote usa:

* :class:`Audio` — mono float32 + metadados medidos (pico, RMS, clipagem, taxa de origem);
* :func:`normalize_for_analysis` — ganho por RMS para a janela dinâmica do classificador;
* :func:`trim_silence` — corta o silêncio inicial e devolve o deslocamento em amostras;
* :func:`peaks_envelope` — mínimo/máximo real por balde, para desenhar a forma de onda;
* :func:`resample_to` / :func:`write_wav` — reamostragem polifase e gravação WAV.

Por que "em float32 mono" e por que a taxa de análise é um parâmetro: a cadeia inteira custa
memória proporcional ao nº de amostras (ver ``docs/ERROS.md`` A13/A14), e o plano de memória do
``pipeline`` usa :func:`resample_to` para caber em máquinas pequenas.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from math import gcd
from typing import Optional, Tuple

import numpy as np

__all__ = ["Audio", "decode_bytes", "decode_file", "normalize_for_analysis", "trim_silence",
           "peaks_envelope", "peaks_finos", "planes_finos", "resample_to", "write_wav",
           "ffmpeg_available"]


@dataclass
class Audio:
    x: np.ndarray          # float32 (n,) mono, [-1, 1] típico
    sr: int                # taxa com que a análise vai rodar
    source_sr: int         # taxa do arquivo como veio (a de cima pode ser reamostrada)
    channels: int
    duration: float
    peak: float
    rms: float
    clipped_ratio: float
    filename: str = ""
    route: str = ""        # qual decodificador produziu isto
    truncated_sec: float = 0.0   # quanto foi cortado por `max_seconds` (0 = nada)


# -------------------------------------------------------------------------------------
# decodificadores
# -------------------------------------------------------------------------------------

def _decode_miniaudio(data: bytes) -> Optional[Tuple[np.ndarray, int, int]]:
    try:
        import miniaudio
    except Exception:
        return None
    try:
        d = miniaudio.decode(data, output_format=miniaudio.SampleFormat.FLOAT32)
    except Exception:
        return None
    try:
        nch = int(d.nchannels) or 1
        a = np.asarray(d.samples, dtype=np.float32)
        if nch > 1:
            a = a[: (a.size // nch) * nch].reshape(-1, nch)
        else:
            a = a.reshape(-1, 1)
        return a, int(d.sample_rate), nch
    except Exception:
        return None


def _decode_soundfile(data: bytes) -> Optional[Tuple[np.ndarray, int, int]]:
    try:
        import soundfile as sf
    except Exception:
        return None
    try:
        a, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        return None
    return np.ascontiguousarray(a, dtype=np.float32), int(sr), int(a.shape[1])


def _ffmpeg_exe() -> Optional[str]:
    """Executável do ffmpeg: o da imagem ``imageio-ffmpeg`` primeiro, depois o do PATH."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def ffmpeg_available() -> bool:
    return _ffmpeg_exe() is not None


def _probe(exe: str, path: str) -> Tuple[Optional[int], int]:
    """Lê taxa e nº de canais do cabeçalho, sem decodificar nada (para pedir o layout certo)."""
    sr, ch = _probe_sr(exe, path), 0
    try:
        p = subprocess.run([exe, "-hide_banner", "-i", path], stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=20)
        for ln in (p.stderr or b"").decode("utf-8", "ignore").splitlines():
            if "Stream #" in ln and "Audio" in ln:
                import re
                m = re.search(r"(\d+)\s+channels", ln)
                if m:
                    ch = int(m.group(1))
                break
    except Exception:
        pass
    return sr, ch


def _probe_sr(exe: str, path: str) -> Optional[int]:
    try:
        p = subprocess.run([exe, "-hide_banner", "-i", path], stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=20)
        import re
        for ln in (p.stderr or b"").decode("utf-8", "ignore").splitlines():
            m = re.search(r"(\d+)\s+Hz", ln)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return None


def _decode_ffmpeg(data: bytes, filename: str = "") -> Optional[Tuple[np.ndarray, int, int]]:
    exe = _ffmpeg_exe()
    if exe is None:
        return None
    ext = os.path.splitext(filename or "")[1].lower() or ".bin"
    tmp_in = tmp_out = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(data)
            tmp_in = f.name
        sr0, ch0 = _probe(exe, tmp_in)
        out_sr = int(sr0) if sr0 and 4000 <= int(sr0) <= 192000 else 44100
        nch = int(ch0) if ch0 and ch0 <= 8 else 2
        tmp_out = tempfile.NamedTemporaryFile(suffix=".f32", delete=False).name
        cmd = [exe, "-v", "error", "-y", "-i", tmp_in, "-f", "f32le", "-acodec", "pcm_f32le",
               "-ac", str(nch), "-ar", str(out_sr), tmp_out]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600,
                       check=True)
        a = np.fromfile(tmp_out, dtype=np.float32)
        if a.size == 0:
            return None
        n = a.size // nch
        a = a[: n * nch].reshape(n, nch)
        return a, int(out_sr), int(nch)
    except Exception:
        return None
    finally:
        for p in (tmp_in, tmp_out):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# -------------------------------------------------------------------------------------
# entrada
# -------------------------------------------------------------------------------------

def _rms_blocos(x: np.ndarray, bloco: int = 1 << 16) -> float:
    """RMS em float64 acumulado por blocos.

    ``np.mean(x.astype(np.float64) ** 2)`` aloca duas vias do tamanho do áudio (58 MB por minuto
    a 44,1 kHz) só para medir um número. Mesma aritmética, memória constante.
    """
    x = np.asarray(x)
    if not x.size:
        return 0.0
    soma = 0.0
    for i0 in range(0, x.size, bloco):
        b = x[i0:i0 + bloco].astype(np.float64, copy=False)
        soma += float(np.dot(b, b))
    return float(np.sqrt(soma / x.size))


def decode_bytes(data: bytes, filename: str = "", max_seconds: float = 60 * 12,
                 analysis_sr: Optional[int] = None) -> Audio:
    """Decodifica e devolve mono float32 pronto para análise.

    ``analysis_sr`` é aplicado de verdade (com anti-alias): foi durante um tempo um parâmetro
    decorativo — chamava um ``_gcd`` que não existia e o ``except Exception: pass`` engolia o
    NameError, de modo que toda faixa era analisada na taxa de origem (``docs/ERROS.md`` A14).
    Erro de reamostragem agora sobe, porque engolir isso troca um falho audível por um mudo.
    """
    rotas = (("miniaudio", _decode_miniaudio(data)),
             ("soundfile", _decode_soundfile(data)),
             ("ffmpeg", _decode_ffmpeg(data, filename)))
    a = sr = nch = route = None
    for nome, res in rotas:
        if res is not None:
            a, sr, nch = res
            route = nome
            break
    if a is None or not len(a):
        falta = "" if ffmpeg_available() else " — nem ffmpeg nem soundfile/miniaudio decodificaram"
        raise ValueError(
            "não foi possível decodificar %s%s. Formatos suportados: wav, aiff, flac, ogg, mp3, "
            "m4a/aac (requer ffmpeg).%s" % (filename or "o arquivo", falta, ""))

    a = np.ascontiguousarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
        nch = 1
    # downmix (soma coerente; preserve energia de fase somando canais)
    x = a.mean(axis=1) if nch > 1 else a[:, 0]

    # decima para a taxa de análise (anti-alias via resample_poly)
    sr_src = int(sr)
    if analysis_sr and not (6000 <= int(analysis_sr) <= 192000):
        raise ValueError("analysis_sr=%s está fora da faixa suportada (6000–192000 Hz); "
                         "omita o parâmetro para usar a taxa do arquivo" % (analysis_sr,))
    if analysis_sr and int(analysis_sr) != int(sr):
        g = int(gcd(int(analysis_sr), int(sr)))
        x = resample_poly_public(x, int(sr), int(analysis_sr))[0]
        sr = int(analysis_sr)
    elif sr > 96000:                        # evita custo desnecessário em 176/192 kHz
        x = resample_poly_public(x, int(sr), 44100)[0]
        sr = 44100

    truncado = 0.0
    if max_seconds:
        n = int(float(max_seconds) * sr)
        if x.size > n:
            truncado = (x.size - n) / float(sr)
            x = x[:n]

    x = np.ascontiguousarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = _rms_blocos(x)
    clipped = float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0

    return Audio(x=x, sr=int(sr), source_sr=sr_src, channels=int(nch),
                 duration=x.size / float(sr), peak=peak, rms=rms, clipped_ratio=clipped,
                 filename=os.path.basename(str(filename or "")), route=route,
                 truncated_sec=round(truncado, 3))


def decode_file(path: str, **kw) -> Audio:
    with open(path, "rb") as f:
        data = f.read()
    return decode_bytes(data, filename=os.path.basename(path), **kw)


# -------------------------------------------------------------------------------------
# preparação
# -------------------------------------------------------------------------------------

def resample_poly_public(x: np.ndarray, sr_in: int, sr_out: int) -> Tuple[np.ndarray, int]:
    from scipy.signal import resample_poly
    g = int(gcd(int(sr_in), int(sr_out)))
    return resample_poly(x, int(sr_out) // g, int(sr_in) // g).astype(np.float32), int(sr_out)


def resample_to(x: np.ndarray, sr_in: int, sr_out: int) -> Tuple[np.ndarray, int]:
    """Reamostragem mono float32 (polifase com anti-alias). Usada pelo plano de memória do
    ``pipeline``, que só pode decidir a taxa depois de saber o tamanho real da faixa."""
    x = np.asarray(x, dtype=np.float32)
    if not sr_out or int(sr_out) == int(sr_in) or x.size == 0:
        return x, int(sr_in)
    return resample_poly_public(x, int(sr_in), int(sr_out))


def normalize_for_analysis(x: np.ndarray, target_rms_db: float = -20.0,
                           max_gain_db: float = 30.0) -> np.ndarray:
    """Ganho por RMS até ``target_rms_db`` — a janela onde o classificador foi ajustado.

    Limitado (``max_gain_db``) de propósito: sem teto, um arquivo gravado a −55 dBFS subiria
    35 dB e o ruído de piso viraria "energia de chimel". Silêncio absoluto passa sem ganho.
    """
    x = x.astype(np.float32, copy=True)
    rms = _rms_blocos(x)
    if not x.size or rms <= 1e-7:
        return x
    alvo = float(target_rms_db)
    atual = 20.0 * np.log10(max(rms, 1e-12))
    ganho_db = float(np.clip(alvo - atual, -max_gain_db, max_gain_db))
    return (x * float(10.0 ** (ganho_db / 20.0))).astype(np.float32)


def trim_silence(x: np.ndarray, sr: int, thresh_db: float = -48.0,
                 pad: float = 0.12) -> Tuple[np.ndarray, int]:
    """Descarta o silêncio inicial e devolve quantas amostras foram cortadas.

    A janela de medição é curta (30 ms) e o corte recua ``pad`` segundos, para não comer o
    ataque do primeiro golpe — é esse deslocamento que depois vira a fase da grade.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x, 0
    w = max(1, int(0.030 * sr))
    n = x.size // w
    if n < 2:
        return x, 0
    env = np.sqrt(np.mean(np.square(x[: n * w].reshape(n, w), dtype=np.float64), axis=1))
    lim = 10.0 ** (float(thresh_db) / 20.0)
    acima = np.nonzero(env > lim)[0]
    if not acima.size:
        return x, 0
    i0 = int(acima[0]) * w
    i0 = max(0, i0 - int(round(pad * sr)))
    if i0 == 0:
        return x, 0
    return x[i0:], i0


def peaks_envelope(x: np.ndarray, sr: int, n_points: int = 2400) -> dict:
    """Mínimo e máximo reais por balde (não RMS): é o que faz o desenho mostrar picos de ataque.

    Devolve também ``rms`` por balde, que o navegador usa como sombra, e a escala em segundos
    (``dt``) para o cursor saber em que instante está cada balde.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return {"min": [], "max": [], "rms": [], "n": 0, "dt": 0.0, "sr": int(sr)}
    n = max(1, min(int(n_points), x.size))
    w = int(np.ceil(x.size / n))
    n_usado = x.size // w
    if n_usado < 1:
        w, n_usado = x.size, 1
    bloco = x[: n_usado * w].reshape(n_usado, w)
    mn = bloco.min(axis=1).astype(np.float32)
    mx = bloco.max(axis=1).astype(np.float32)
    rms = np.sqrt(np.mean(np.square(bloco, dtype=np.float32), axis=1)).astype(np.float32)
    if n_usado < n:                       # completa com zeros para o desenho não esticar o fim
        k = n - n_usado
        mn = np.concatenate([mn, np.zeros(k, np.float32)])
        mx = np.concatenate([mx, np.zeros(k, np.float32)])
        rms = np.concatenate([rms, np.zeros(k, np.float32)])
    return {"min": [round(float(v), 4) for v in mn],
            "max": [round(float(v), 4) for v in mx],
            "rms": [round(float(v), 4) for v in rms],
            "n": int(n), "dt": round(float(w) / float(sr), 6), "sr": int(sr),
            "bucket_sec": round(float(w) / float(sr), 6)}


def peaks_finos(x: np.ndarray, sr: int, n_max: int = 300000,
                bucket_min: float = 0.001) -> Tuple[np.ndarray, float]:
    """Envelope (mín, máx) em float32 com o maior nº de baldes que cabe em `n_max`.

    É o que permite dar zoom na forma de onda sem re-decodificar nada: o arquivo é gravado
    cru, plano a plano, e a rota `/api/wave` lê só o trecho pedido com `np.fromfile(offset=…)`.
    A lista de 1200 baldes do `peaks_envelope` serve à visão geral; aqui o teto é de ~300 000
    baldes, o que a 44,1 kHz dá 1 ms de resolução para faixas de até 5 min.

    Retorna ``(arr, bucket_sec)`` com `arr` de shape ``(2, n)`` — mínimo na linha 0, máximo na 1,
    contíguos de propósito: o leitor fatia com um `offset` em bytes e nada mais.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros((2, 0), np.float32), float(bucket_min)
    n = int(min(n_max, max(8, x.size // max(1, int(bucket_min * sr)))))
    w = max(1, int(np.ceil(x.size / n)))
    n_usado = x.size // w
    arr = np.zeros((2, max(n_usado, n)), dtype=np.float32)
    if n_usado >= 1:
        bloco = x[: n_usado * w].reshape(n_usado, w)
        arr[0, :n_usado] = bloco.min(axis=1)
        arr[1, :n_usado] = bloco.max(axis=1)
        if n_usado < n:                       # completa com zeros: o fim não é esticado
            arr[:, n_usado:] = 0.0
    return arr, float(w) / float(sr)


def planes_finos(mapas: dict, n: int, bucket_sec: float, sr: int,
                 desvio_amostras: int = 0, fps: float = 0.0) -> np.ndarray:
    """Reamostra séries por quadro (envelope de novidade) para a grade de baldes do desenho.

    Alinhamento é feito por *tempo*, não por índice: o envelope é calculado sobre o áudio já
    cortado (o `trim_silence` comeu `desvio_amostras` amostras), e é isso que faz a curva de
    novidade bater com o pico que ela explica no gráfico.
    """
    ch = len(mapas)
    out = np.zeros((ch, n), dtype=np.float32)
    if not n or not ch:
        return out
    meiodo = (np.arange(n, dtype=np.float64) + 0.5) * bucket_sec
    if fps <= 0:
        fps = 1.0 / max(1e-6, bucket_sec)
    j = np.rint((meiodo * sr - desvio_amostras) / (sr / fps)).astype(np.int64)
    j = np.clip(j, 0, None)
    for i, (k, v) in enumerate(sorted(mapas.items())):
        v = np.asarray(v, dtype=np.float32).ravel()
        if v.size:
            out[i] = v[np.minimum(j, v.size - 1)]
    return out


def write_wav(path: str, x: np.ndarray, sr: int) -> None:
    """Escreve WAV mono. Prefere soundfile (PCM_16); se não houver, cabeçalho RIFF à mão."""
    x = np.asarray(x, dtype=np.float32).ravel()
    try:
        import soundfile as sf
        sf.write(path, x, int(sr), subtype="PCM_16")
        return
    except Exception:
        pass
    import struct
    i16 = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")
    data = i16.tobytes()
    nch, bits = 1, 16
    byte_rate = int(sr) * nch * bits // 8
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, nch, int(sr), byte_rate, nch * bits // 8, bits))
        f.write(b"data" + struct.pack("<I", len(data)))
        f.write(data)
