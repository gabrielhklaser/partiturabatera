"""
Servidor do DrumScribe — Flask.

Rotas (todas com caminho relativo; o front-end nunca aponta para localhost):

  GET  /                      página (upload + partitura + editor)
  GET  /static/<arquivo>      assets
  GET  /api/health            status / versões
  GET  /api/demo              analisa a faixa de demonstração (em cache)
  POST /api/analyze           multipart 'file' (+ campo 'params' em JSON) → resultado
  GET  /api/result/<id>.json  resultado guardado (partitura + relatório + forma de onda)
  GET  /api/audio/<id>        o áudio enviado, para o player do navegador
  POST /api/svg               {score, opts} → SVG atualizado (pré-visualização)
  POST /api/rebuild           {score, hits, ...} → reescreve a partitura após edição
  POST /api/export/<fmt>      fmt ∈ pdf|svg|musicxml|mid|json|csv → download

O resultado da análise fica em memória (dict) e o áudio em out/uploads/<id>, para que o
navegador possa tocar o arquivo original sincronizado com a partitura.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from flask import (Flask, Response, abort, jsonify, request, send_file,
                  send_from_directory)

from . import __version__ as _ver
from .audio_io import decode_bytes
from .engrave import to_pdf_bytes, to_svg, write_pdf
from .layout import layout_score
from .midi_out import to_midi
from .musicxml import to_musicxml
from .pipeline import PARAMS_DEFAULT, rebuild_score, transcribe_bytes, MemoriaInsuficiente
from .kit import LANES as _LANES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
UPLOADS = os.path.join(OUT, "uploads")
STATIC = os.path.join(ROOT, "static")
os.makedirs(UPLOADS, exist_ok=True)

app = Flask(__name__, static_folder=STATIC, static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
#: Limite próprio, medido durante o streaming para o disco: dá mensagem em JSON legível
#: (o 413 do Werkzeug é HTML e a interface não sabe mostrar isso sem travar a tela).
LIMITE_BYTES = 200 * 1024 * 1024

RESULTS: Dict[str, dict] = {}


# -------------------------------------------------------------------------------------
def _safe_name(name: str) -> str:
    name = os.path.basename(str(name or "audio")).replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._\- ]+", "_", name)
    return name[:120] or "audio.wav"


def _params_from_request() -> dict:
    p = dict(PARAMS_DEFAULT)
    raw = request.form.get("params") if request.form else None
    if raw:
        try:
            p.update(json.loads(raw))
        except Exception:
            pass
    if request.is_json and isinstance(request.get_json(silent=True), dict):
        p.update(request.get_json(silent=True).get("params") or {})
    for k in ("sensitivity", "min_confidence", "refine_frac", "bpm_hint", "swing", "meter",
              "grid_mode", "aux_lanes", "show_aux", "fill_detect", "repeat_slash",
              "downbeat_shift_ticks"):
        if request.form and k in request.form:
            v = request.form.get(k)
            key = "aux_lanes" if k == "show_aux" else k
            if k in ("show_aux", "aux_lanes", "fill_detect", "repeat_slash"):
                p[key] = str(v).lower() in ("1", "true", "on", "sim")
            elif k in ("sensitivity", "min_confidence", "refine_frac", "bpm_hint",
                       "downbeat_shift_ticks"):
                try:
                    p[key] = float(v) if k != "downbeat_shift_ticks" else int(round(float(v)))
                except Exception:
                    pass
            else:
                p[key] = v
    return p


def _store(result: dict, audio_bytes: bytes, filename: str, params: dict) -> str:
    rid = uuid.uuid4().hex[:12]
    d = os.path.join(UPLOADS, rid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, filename), "wb") as f:
        f.write(audio_bytes)
    result["id"] = rid
    result["dir"] = d
    result["path"] = os.path.join(d, filename)
    result["params"] = params
    result["created"] = time.time()
    RESULTS[rid] = result
    _gc()
    return rid


def _varre_wave() -> int:
    """Apaga envelopes finos órfãos na subida.

    `out/wave/` é pasta desta aplicação e nada fora de `RESULTS` a usa: um plano sem dono é
    resíduo de uma sessão anterior (o `RESULTS` é memória de processo — B10). Sem esta varredura,
    cada reinício deixaria até 24 × ~1 MB de arquivos que nenhum coletor alcança, porque o `rmtree`
    do `_gc` só cobre o que está dentro de `out/uploads/`.
    """
    apagados = 0
    try:
        nomes = os.listdir(WAVE_DIR)
    except OSError:
        return 0
    donos = {str(k) for k in RESULTS}
    for nome in nomes:
        if not nome.endswith(".f32"):
            continue
        if nome.split(".", 1)[0] in donos:
            continue
        try:
            os.remove(os.path.join(WAVE_DIR, nome))
            apagados += 1
        except OSError:
            pass
    return apagados


def _gc(max_keep: int = 24) -> None:
    if len(RESULTS) <= max_keep:
        return
    old = sorted(RESULTS.items(), key=lambda kv: kv[1].get("created", 0))[: len(RESULTS) - max_keep]
    for rid, r in old:
        RESULTS.pop(rid, None)
        try:
            d = os.path.abspath(r.get("dir") or "")
            # Lei do coletor: só se apaga o que ele mesmo criou — um diretório dentro de
            # `out/uploads/<id>`. Sem esta guarda, um resultado apontando para `samples/`
            # (a demo, por exemplo) fazia o `rmtree` levar a faixa de demonstração e o
            # gabarito junto. Já esteve a um `len(RESULTS) > 24` de acontecer.
            if d.startswith(os.path.abspath(UPLOADS) + os.sep):
                import shutil
                shutil.rmtree(d, ignore_errors=True)
            # `out/wave/<rid>.*` não está dentro de `dir` quando o áudio é a demo (que mora em
            # `samples/`, que o coletor não pode tocar) — então sai à parte, sempre.
            for suf in (".onda.f32", ".curvas.f32"):
                try:
                    os.remove(os.path.join(WAVE_DIR, rid + suf))
                except OSError:
                    pass
        except Exception:
            pass


def _cursor_clock(score: dict, report: Optional[dict] = None) -> Optional[dict]:
    """Origem e duração do compasso no tempo do arquivo, pela MESMA lei que a auditoria julga.

    `report.tempo.phase_ms` não é a origem da grade gravada: no demo ele difere em 1,95 s (3,25
    batidas) da mediana dos 290 golpes medidos. Quem marca compasso a partir dele acerta o áudio
    errado — por isso o cursor, o clique na pauta e o loop do compasso passam a usar `qa.clock_map`,
    o estimador que T3/T3d já validam (mediana dos desvios tempo↔grade, com o sinal do offset de
    downbeat provado nos dois sentidos).
    """
    try:
        from . import qa as QA
        hits = [dict(h, bar=b.get("index")) for b in (score.get("bars") or [])
                for h in (b.get("hits") or [])]
        bpm = float(score.get("bpm") or 120.0)
        tpr = int(score.get("ticks_per_bar") or 32)
        clk = QA.clock_map(hits, bpm, tpr, report)
        tpb = int(score.get("ticks_per_beat") or 8)
        if not clk:
            spt = 60.0 / bpm / tpb
            ph = (float(((report or {}).get("tempo") or {}).get("phase_ms") or 0.0)
                  + float(((report or {}).get("file") or {}).get("trim_lead_ms") or 0.0)) / 1000.0
            return {"t0": round(ph, 4), "bar_len": round(tpr * spt, 4), "fonte": "relatório",
                    "n": 0, "spread_ms": None, "max_ms": None}
        return {"t0": round(QA.time_of(0, 0, clk), 4),
                "bar_len": round(QA.time_of(1, 0, clk) - QA.time_of(0, 0, clk), 4),
                "fonte": "grade medida", "n": len(clk.get("rows") or []),
                "spread_ms": round(1000.0 * clk["spread"] * clk["spt"], 1),
                "max_ms": round(1000.0 * clk["max"] * clk["spt"], 1)}
    except Exception:                                              # noqa: BLE001
        return None


def _svg_pair(score: dict, report: Optional[dict] = None, **opts):
    """(svg, meta) — o documento e a geometria por compasso, da mesma passada de layout.

    O cursor de reprodução do navegador precisa saber onde cada compasso foi traçado. Exigir isso
    do `layout_score` (e não chutar uma grade de posições no front) mantém o cursor colado na
    gravura: qualquer mudança de pauta, quebra de sistema ou legenda move o mapa junto.
    """
    from .layout import layout_score, render_svg_document
    lay = layout_score(score, **opts)
    meta = dict(lay["meta"])
    meta["page"] = str(opts.get("page") or "a4_landscape")
    meta["clock"] = _cursor_clock(score, report)
    return render_svg_document(score, lay), meta


def _public(result: dict, with_svg: bool = True) -> dict:
    out = {"id": result["id"], "score": result["score"], "report": result["report"],
           "wave": result.get("wave"), "audio_url": "/api/audio/%s" % result["id"],
           "params": result.get("params"), "filename": result.get("filename")}
    if with_svg:
        out["svg"] = result.get("svg")
        out["layout"] = result.get("layout")     # geometria por compasso, para o cursor
    wf = result.get("wave_fine")
    out["wave_fine"] = None if not wf else {k: wf[k] for k in
                                            ("bucket_sec", "n", "nc", "nomes", "sr", "fps",
                                             "lead_ms", "bytes") if k in wf}
    out["tem_tune"] = bool(result.get("tune"))
    return out


# -------------------------------------------------------------------------------------
_VER_TOKEN = re.compile(r"\{\{VER\}\}")


def _asset_stamp() -> str:
    """Carimbo dos estáticos, usado nas URLs (`style.css?v=…`).

    O `#busy` é um véu full-screen: se o navegador servir um `style.css` de antes da regra
    `[hidden]{display:none!important}`, a página abre coberta e inclicável — e nenhuma requisição
    nova do HTML resolve isso, porque o sub-recurso vem do cache. O carimbo muda com
    (mtime, tamanho) de cada arquivo, então HTML e o CSS que o protege chegam sempre juntos.
    """
    h = hashlib.sha1()
    for nome in ("style.css", "app.js", "index.html"):
        try:
            st = os.stat(os.path.join(STATIC, nome))
            h.update(("%s:%d:%d" % (nome, int(st.st_mtime), st.st_size)).encode())
        except OSError:
            h.update(("%s:-:-" % nome).encode())
    return h.hexdigest()[:10]


@app.get("/")
def index():
    try:
        with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as fh:
            html = _VER_TOKEN.sub(_asset_stamp(), fh.read())
    except OSError:
        return send_from_directory(STATIC, "index.html")
    resp = Response(html, mimetype="text/html; charset=utf-8")
    resp.headers["Cache-Control"] = "no-store"      # o carimbo só vale para este HTML
    return resp


@app.errorhandler(413)
def _grande_demais_werkzeug(e):
    return jsonify({"ok": False, "codigo": "grande_demais",
                    "error": "arquivo acima de %.0f MB, o teto de envio desta instalação. "
                             "Envie um trecho menor." % (app.config["MAX_CONTENT_LENGTH"] / 1e6)}), 413


@app.get("/api/health")
def health():
    """Sonda barata para o front (e para humanos): responde mesmo com dependência faltando.

    Um ambiente onde o `pip install` não rodou é o modo de falha mais comum desta
    plataforma — e o pior de diagnosticar, porque o navegador mostra apenas "carregando".
    """
    def tem(mod):
        try:
            __import__(mod)
            return True
        except Exception:                                        # noqa: BLE001
            return False
    deps = {m: tem(m) for m in ("numpy", "scipy", "soundfile", "miniaudio",
                               "reportlab", "pypdfium2", "imageio_ffmpeg")}
    faltam = [m for m, v in deps.items() if not v]
    return jsonify({
        "ok": not faltam, "app": "DrumScribe", "version": _ver, "results": len(RESULTS),
        "python": __import__("sys").version.split()[0],
        "static": os.path.isfile(os.path.join(STATIC, "index.html")),
        "demo": os.path.isfile(os.path.join(ROOT, "samples", "demo_drums.wav")),
        "deps": deps, "faltando": faltam,
        "aviso": ("alguma dependência está ausente — rode ./run.sh ou "
                  "python3 -m pip install -r requirements.txt") if faltam else None,
        "pasta": ROOT,
    })


# =====================================================================================
# isolamento do processo de análise
# =====================================================================================
#
# Por que isto existe (docs/ERROS.md A13): antes, a análise rodava dentro do processo que
# atende todo mundo. Uma faixa mais longa que a RAM não devolvia um erro — o kernel matava o
# servidor inteiro (`oom-kill ... total-vm:2239564kB, anon-rss:1655428kB`), e o usuário via
# apenas a aba morta. Com o trabalho pesado em um filho `spawn`, o pior caso passa a ser uma
# resposta 500 legível e a plataforma no ar. O filho é morto por *nosotros* quando estoura o
# prazo, e é o único jeito de limitar picos transitórios de alocadores que não devolvem RAM.
#
# Injeção de falha para os testes: DRUMSCRIBE_FALHAR_FILHO=oom|memoria|trava faz o filho
# simularem SIGKILL, MemoriaInsuficiente e travamento, nesta ordem de severidade.

def _salva_upload(f, limite: int = LIMITE_BYTES):
    """Copia o multipart para `out/uploads/<id>/<nome>` em blocos de 1 MB, com teto próprio.

    Não usa `f.read()`: o corpo inteiro na memória do servidor é justamente o que não pode
    acontecer quando a faixa é longa. O limite estourado devolve `GrandeDemais`, que vira 413
    em JSON — e apaga o parcial, para não deixar lixo no disco.
    """
    rid = uuid.uuid4().hex[:12]
    nome = _safe_name(f.filename)
    d = os.path.join(UPLOADS, rid)
    os.makedirs(d, exist_ok=True)
    caminho = os.path.join(d, nome)
    total = 0
    try:
        with open(caminho, "wb") as out:
            while True:
                pedaco = f.read(1 << 20)
                if not pedaco:
                    break
                total += len(pedaco)
                if total > limite:
                    raise GrandeDemais(
                        "o arquivo passou de %.0f MB durante o envio (%.1f MB recebidos). Ele é "
                        "grande demais para o upload desta instalação; envie um trecho menor."
                        % (limite / 1e6, total / 1e6))
                out.write(pedaco)
    except GrandeDemais:
        try:
            os.unlink(caminho)
            os.rmdir(d)
        except OSError:
            pass
        raise
    except Exception as e:
        try:
            os.unlink(caminho)
        except OSError:
            pass
        raise ValueError("falha ao receber o arquivo: %s" % e)
    if total == 0:
        try:
            os.unlink(caminho)
            os.rmdir(d)
        except OSError:
            pass
        raise ValueError("arquivo vazio")
    return rid, caminho, nome, total


def _store_pre_salvo(result: dict, rid: str, caminho: str, nome: str, params: dict) -> str:
    """Como `_store`, mas o áudio já está em disco — não reescrevemos 100 MB por nada."""
    result["id"] = rid
    result["path"] = caminho
    d = os.path.dirname(caminho)
    if os.path.abspath(d).startswith(os.path.abspath(UPLOADS) + os.sep):
        result["dir"] = d
    result["params"] = params
    result["created"] = time.time()
    result.setdefault("filename", nome)
    RESULTS[rid] = result
    _gc()
    return rid



class GrandeDemais(RuntimeError):
    """Upload acima de `LIMITE_BYTES`, descoberto durante o streaming."""


class FilhoMorto(RuntimeError):
    """O processo filho saiu sem entregar resultado (quase sempre: OOM killer)."""


class PrazoEsgotado(RuntimeError):
    """A análise não terminou dentro do prazo — o filho foi encerrado de propósito."""


WAVE_DIR = os.path.join(OUT, "wave")


def _grava_finos(result: dict, rid: str) -> None:
    """Despeja o envelope fino em dois `.f32` planos sob `out/wave/<rid>.*` e aponta os caminhos.

    Por que arquivo e não JSON: a 1 ms de resolução a demo tem 42 000 baldes × (onda + 4 curvas)
    ≈ 0,7 MB de float32; a mesma coisa em JSON daria ~15 MB por resultado, e o `RESULTS` guarda 24
    — seria 360 MB de memória no meio de uma máquina sem swap. Em disco, o zoom lê só o trecho
    pedido por `seek`, e um resultado a mais não custa RAM. É também a razão de o zoom não
    re-decodificar o MP3 do usuário: quem decodifica arquivo de usuário é o filho (A13), não o
    navegador nem o pai a cada clique.
    """
    meta = result.get("wave_fine")
    if not meta or "onda" not in meta:
        return
    buf_onda, buf_curvas = meta.pop("onda"), meta.pop("curvas")
    n = int(meta["n"])
    try:
        os.makedirs(WAVE_DIR, exist_ok=True)
        pa = os.path.join(WAVE_DIR, "%s.onda.f32" % rid)
        pc = os.path.join(WAVE_DIR, "%s.curvas.f32" % rid)
        for caminho_fino, buf in ((pa, buf_onda), (pc, buf_curvas)):
            with open(caminho_fino, "wb") as fh:          # grava em um tempo, sem re-decodificar
                fh.write(buf)
        meta["onda"], meta["curvas"] = pa, pc
        meta["bytes"] = len(buf_onda) + len(buf_curvas)
        result["wave_fine"] = meta
    except Exception as e:
        result["report"].setdefault("warnings", []).append(
            "envelope fino não gravado (%s) — o zoom mostra a visão geral." % e)
        result.pop("wave_fine", None)


def _trabalha(caminho: str, nome: str, params: dict, acao: str = "analise") -> dict:
    """A análise completa, do arquivo ao SVG+relatório. Roda no filho.

    Recebe o *caminho*, não os bytes: um upload de 150 MB lido no pai seria 150 MB lá dentro
    e mais 150 MB atravessando a fila do filho. O arquivo já está em disco (gravado em
    streaming por `_salva_upload`), então só o processo que precisa dele o carrega.
    """
    t0 = time.time()
    if acao == "sintonia":
        from .tune import recomendar_arquivo
        return {"acao": "sintonia", "tune": recomendar_arquivo(caminho, params=params)}
    with open(caminho, "rb") as fh:
        dados = fh.read()
    res = transcribe_bytes(dados, filename=nome, params=params)
    del dados
    score = res["score"].to_dict()
    result = {"score": score, "report": res["report"], "wave": res["wave"],
              "filename": nome, "audio_info": res.get("audio"),
              "detections": res.get("detections"),
              "elapsed_sec": round(time.time() - t0, 2)}
    fine = res.get("fine")
    if fine is not None:
        try:
            onda = np.ascontiguousarray(fine["onda"], dtype="<f4")
            curvas = np.ascontiguousarray(fine["curvas"], dtype="<f4")
            result["wave_fine"] = {"bucket_sec": float(fine["bucket_sec"]), "n": int(fine["n"]),
                                   "nc": int(curvas.shape[0]), "nomes": [str(x) for x in fine["nomes"]],
                                   "sr": int(fine["sr"]), "fps": float(fine["fps"]),
                                   "lead_ms": float(fine["lead_ms"]),
                                   "onda": onda.tobytes(order="C"), "curvas": curvas.tobytes(order="C")}
        except Exception as e:                         # o zoom é acessório: não derruba a análise
            result["report"].setdefault("warnings", []).append(
                "envelope fino indisponível (%s) — o zoom mostra a visão geral de 1800 baldes." % e)
    result["svg"], result["layout"] = _svg_pair(score, res["report"], page="a4_landscape")
    return result


def _filho_worker(fila, caminho: str, nome: str, params: dict, acao: str = "analise") -> None:
    falha = os.environ.get("DRUMSCRIBE_FALHAR_FILHO", "").strip().lower()
    try:
        if falha == "oom":
            import signal
            os.kill(os.getpid(), signal.SIGKILL)         # morre sem entregar nada
            return
        if falha == "trava":
            time.sleep(10_000)
        if falha == "memoria":
            raise MemoriaInsuficiente("falha injetada para teste (DRUMSCRIBE_FALHAR_FILHO=memoria)")
        fila.put(("ok", _trabalha(caminho, nome, params, acao)))
    except MemoriaInsuficiente as e:
        fila.put(("memoria", str(e)))
    except Exception as e:                               # noqa: BLE001 — o pai precisa saber
        fila.put(("erro", "%s: %s" % (type(e).__name__, e)))
    finally:
        try:
            fila.close()
        except Exception:
            pass


_CTX = None


def _contexto():
    """`spawn` sempre: fork de um processo com numpy/FFT em aberto herda memória e locks."""
    global _CTX
    if _CTX is None:
        if os.environ.get("DRUMSCRIBE_INLINE"):
            _CTX = "inline"
        else:
            import multiprocessing as mp
            _CTX = mp.get_context("spawn")
    return _CTX


def _prazo_s() -> float:
    try:
        return float(os.environ.get("DRUMSCRIBE_PRAZO_ANALISE") or 0.0)
    except ValueError:
        return 0.0


def _analise_isolada(caminho: str, nome: str, params: dict, acao: str = "analise",
                     limite: Optional[float] = None) -> dict:
    """Roda `_trabalha` em outro processo e devolve o resultado — ou um erro explicável.

    A fila é lida pelo pai enquanto o filho escreve: o resultado (SVG + relatório) passa de
    longe do buffer de um `Pipe`, e um pai que espera o filho *fechar* antes de ler morre de
    deadlock. Queue com leitor ativo resolve isso.
    """
    ctx = _contexto()
    if ctx == "inline":                                  # escape hatch: testes unitários, debug
        return _trabalha(caminho, nome, params, acao)
    import queue as _q
    fila = ctx.Queue()
    proc = ctx.Process(target=_filho_worker,
                       args=(fila, caminho, nome, params, acao), daemon=True)
    proc.start()
    t0 = time.time()
    limite = float(limite or 0.0) or (_prazo_s() or 900.0)
    tipo = carga = None
    while True:
        restante = limite - (time.time() - t0)
        if restante <= 0:
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join(1)
            raise PrazoEsgotado(
                "a análise não terminou em %.0f s e o processo foi encerrado. Faixas longas "
                "custam caro: envie um trecho menor, ou aumente DRUMSCRIBE_PRAZO_ANALISE "
                "(agora %.0f s)." % (limite, limite))
        try:
            tipo, carga = fila.get(timeout=min(0.5, restante))
            break
        except _q.Empty:
            if not proc.is_alive() and fila.empty():
                proc.join(1)
                cod = proc.exitcode
                if cod is not None and cod < 0:
                    raise FilhoMorto(
                        "o sistema encerrou a análise com o sinal %d (código %d)%s. Sem swap, "
                        "isso quer dizer falta de memória: a faixa pedida não coube. A "
                        "plataforma continua no ar — envie um trecho mais curto ou rode com "
                        "mais RAM." % (-cod, cod, " — OOM killer" if cod == -9 else ""))
                raise FilhoMorto("o processo de análise saiu com código %s sem entregar "
                                 "resultado (a plataforma continua no ar)." % cod)
    if tipo == "memoria":
        proc.join(3)
        raise MemoriaInsuficiente(carga)
    if tipo == "erro":
        proc.join(3)
        raise RuntimeError(carga)
    proc.join(3)
    return carga


def _erro_analise(e: Exception) -> Any:
    """Traduz a falha em resposta: nunca 200 falso, nunca stack, nunca tela travada."""
    if isinstance(e, MemoriaInsuficiente):
        return jsonify({"ok": False, "codigo": "memoria", "error": str(e)}), 413
    if isinstance(e, FilhoMorto):
        return jsonify({"ok": False, "codigo": "memoria_nucleo", "error": str(e)}), 500
    if isinstance(e, PrazoEsgotado):
        return jsonify({"ok": False, "codigo": "prazo", "error": str(e)}), 504
    if isinstance(e, ValueError):
        return jsonify({"ok": False, "codigo": "arquivo", "error": str(e)}), 422
    app.logger.warning("falha na análise: %s\n%s", e, traceback.format_exc())
    return jsonify({"ok": False, "codigo": "interno",
                    "error": "a análise falhou neste arquivo: %s" % e}), 422


@app.post("/api/analyze")
def analyze():
    f = request.files.get("file") or request.files.get("audio")
    if f is None:
        return jsonify({"ok": False, "error": "nenhum arquivo recebido (campo 'file')"}), 400
    params = _params_from_request()
    t0 = time.time()
    try:
        rid, caminho, filename, _bytes = _salva_upload(f)
    except GrandeDemais as e:
        return jsonify({"ok": False, "codigo": "grande_demais", "error": str(e)}), 413
    except ValueError as e:
        return jsonify({"ok": False, "codigo": "arquivo", "error": str(e)}), 400
    try:
        result = _analise_isolada(caminho, filename, params)
    except Exception as e:                                   # corrupção, memória, prazo, filho morto
        return _erro_analise(e)
    result["elapsed_sec"] = round(time.time() - t0, 2)
    rid = _store_pre_salvo(result, rid, caminho, filename, params)
    _grava_finos(result, rid)
    return jsonify({"ok": True, **_public(result)})


@app.get("/api/demo")
def demo():
    path = os.path.join(ROOT, "samples", "demo_drums.mp3")
    if not os.path.exists(path):
        path = os.path.join(ROOT, "samples", "demo_drums.wav")
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": "faixa de demonstração ausente"}), 404
    try:
        result = _analise_isolada(path, os.path.basename(path), dict(PARAMS_DEFAULT))
    except Exception as e:
        return _erro_analise(e)
    result["elapsed_sec"] = None
    result["demo"] = True
    rid = uuid.uuid4().hex[:12]
    _store_pre_salvo(result, rid, path, os.path.basename(path), dict(PARAMS_DEFAULT))
    _grava_finos(result, rid)
    return jsonify({"ok": True, **_public(result)})


@app.get("/api/result/<rid>.json")
@app.get("/api/result/<rid>")
def result_json(rid: str):
    r = RESULTS.get(rid)
    if not r:
        return jsonify({"ok": False, "error": "resultado não encontrado (o servidor reiniciou?)"}), 404
    return jsonify({"ok": True, **_public(r, with_svg=request.args.get("svg", "1") != "0")})


@app.get("/api/audio/<rid>")
def audio(rid: str):
    r = RESULTS.get(rid)
    if not r or not os.path.exists(r["path"]):
        abort(404)
    return send_file(r["path"], as_attachment=False, download_name=os.path.basename(r["path"]))


@app.post("/api/svg")
def svg_route():
    body = request.get_json(silent=True) or {}
    score = body.get("score") or (RESULTS.get(body.get("id", ""), {}) or {}).get("score")
    if not score:
        return jsonify({"ok": False, "error": "faltou a partitura"}), 400
    opts = body.get("opts") or {}
    keep = {k: v for k, v in opts.items() if k in ("page", "ls", "compact", "show_legend",
                                                  "bars_per_system")}
    rep = (RESULTS.get(body.get("id", ""), {}) or {}).get("report")
    svg, meta = _svg_pair(score, rep, **keep)
    return jsonify({"ok": True, "svg": svg, "layout": meta})


@app.post("/api/rebuild")
def rebuild():
    body = request.get_json(silent=True) or {}
    rid = body.get("id")
    base = RESULTS.get(rid or "", {})
    score = body.get("score") or base.get("score")
    if not score:
        return jsonify({"ok": False, "error": "faltou a partitura base"}), 400
    hits = body.get("hits")
    if hits is None:
        hits = [dict(h, bar=b["index"]) for b in score.get("bars", []) for h in b.get("hits", [])]
    new = rebuild_score(hits, bpm=float(body.get("bpm") or score.get("bpm") or 120.0),
                        meter=str(body.get("meter") or score.get("meter") or "4/4"),
                        swing=float(body.get("swing") if body.get("swing") is not None
                                    else (score.get("swing") or 0.0)),
                        title=str(body.get("title") or score.get("title") or "Transcrição"),
                        subtitle=str(body.get("subtitle", score.get("subtitle") or "")),
                        report=base.get("report"))
    svg, meta = _svg_pair(new, base.get("report"), page="a4_landscape")
    if base:
        base["score"] = new
        base["svg"], base["layout"] = svg, meta
    return jsonify({"ok": True, "score": new, "svg": svg, "layout": meta})


@app.post("/api/synth")
def synth():
    """Renderiza a partitura (síntese do kit) como WAV — para conferir a escrita ouvindo."""
    body = request.get_json(silent=True) or {}
    score = _score_from(body)
    if score is None:
        return jsonify({"ok": False, "error": "partitura não encontrada"}), 400
    try:
        import numpy as np
        from .kit_synth import render_score
        from .audio_io import write_wav
        x = render_score(score, sr=int(body.get("sr") or 44100),
                         gain=float(body.get("gain") or 0.85))
        buf = io.BytesIO()
        import soundfile as sf
        sf.write(buf, np.asarray(x, dtype=np.float32), 44100, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return send_file(buf, mimetype="audio/wav", as_attachment=False,
                         download_name="transcricao.wav")
    except Exception as e:
        app.logger.warning("synth falhou: %s\n%s", e, traceback.format_exc())
        return jsonify({"ok": False, "error": "não foi possível gerar o áudio: %s" % e}), 422


# --------------------------------------------------------------------------- exportações
def _score_from(body: dict) -> Optional[dict]:
    if not isinstance(body, dict):
        return None
    if body.get("score"):
        return body["score"]
    r = RESULTS.get(str(body.get("id") or ""))
    return r["score"] if r else None


@app.post("/api/export/<fmt>")
def export(fmt: str):
    body = request.get_json(silent=True) or {}
    score = _score_from(body)
    if score is None:
        return jsonify({"ok": False, "error": "partitura não encontrada"}), 400
    title = re.sub(r"[^A-Za-z0-9._ -]+", "_", str(score.get("title") or "drumscribe"))[:60]
    if fmt == "pdf":
        data = to_pdf_bytes(score, page=str(body.get("page") or "a4_landscape"))
        return send_file(io.BytesIO(data), mimetype="application/pdf", as_attachment=True,
                         download_name=title + ".pdf")
    if fmt == "svg":
        data = to_svg(score, page=str(body.get("page") or "a4_landscape")).encode("utf-8")
        return send_file(io.BytesIO(data), mimetype="image/svg+xml", as_attachment=True,
                         download_name=title + ".svg")
    if fmt in ("musicxml", "xml", "mxl"):
        data = to_musicxml(score).encode("utf-8")
        return send_file(io.BytesIO(data), mimetype="application/vnd.recordare.musicxml+xml",
                         as_attachment=True, download_name=title + ".musicxml")
    if fmt in ("mid", "midi"):
        data = to_midi(score)
        return send_file(io.BytesIO(data), mimetype="audio/midi", as_attachment=True,
                         download_name=title + ".mid")
    if fmt == "json":
        data = json.dumps({"score": score, "report": (body.get("report") or score.get("report")
                                                       or {})}, ensure_ascii=False, indent=1)
        return send_file(io.BytesIO(data.encode("utf-8")), mimetype="application/json",
                         as_attachment=True, download_name=title + ".json")
    if fmt == "csv":
        from .pipeline import score_to_csv_bom
        return send_file(io.BytesIO(score_to_csv_bom(score).encode("utf-8")),
                         mimetype="text/csv", as_attachment=True, download_name=title + ".csv")
    abort(404)


# ------------------------------------------------------------------- zoom de waveform
def _le_baldes(caminho: str, n_cols: int, i0: int, m: int, k: int, linhas) -> List[list]:
    """Lê só as linhas/baldes pedidos, por deslocamento de bytes, e agrupa por máximo/mínimo.

    `onda` é gravado como float32 (2, n) contíguo: linha 0 = mínimo do balde, linha 1 = máximo.
    Um `seek` de 4·(linha·n + i₀) e `m`·4 bytes lidos bastam — o arquivo inteiro (0,7 MB na demo,
    até ~10 MB em 180 s) nunca entra no processo.
    """
    with open(caminho, "rb") as fh:
        out = []
        for r in linhas:
            fh.seek(4 * (r * n_cols + i0))
            v = np.frombuffer(fh.read(4 * m), dtype="<f4")
            # (blocos, k) sempre — inclusive com k = 1. Um ramo separado que rearrumasse para
            # (1, N) devolvia um único ponto por linha quando o zoom já não agrupa nada, que foi
            # exatamente o sintoma medido na verificação (344 baldes lidos, 1 desenhado).
            if k > 1 and v.size >= k:
                v = v[: (v.size // k) * k]
            out.append(v.reshape(-1, k))
        return out


@app.get("/api/wave/<rid>")
def wave_fino(rid: str):
    """Envelope de ~1 ms + novidade por grupo, na janela [t0, t1] — o zoom da aba Áudio.

    Não re-decodifica nada: os dois planos float32 foram gravados pelo filho durante a análise.
    É o desenho e a leitura no mesmo eixo de tempo, para o usuário conferir que o que a plataforma
    *ouviu* bate com o que ela *escreveu*.
    """
    r = RESULTS.get(str(rid))
    if r is None:
        return jsonify({"ok": False, "codigo": "sem_analise",
                        "error": "essa análise não está mais na memória desta instância "
                                 "(saiu no descarte de resultados). Reenvie a faixa."}), 404
    meta = r.get("wave_fine")
    if not meta or not os.path.exists(meta["onda"]):
        return jsonify({"ok": False, "codigo": "sem_envelope_fino",
                        "error": "esta análise não gerou envelope fino (faixa degradada ou "
                                 "gravado antes de o recurso existir). A visão geral continua "
                                 "valendo; reenvie a faixa para ter o zoom em ~1 ms.",
                        "wave": r.get("wave")}), 200
    try:
        t0 = max(0.0, float(request.args.get("t0") or 0.0))
        t1 = float(request.args.get("t1") or 0.0)
        n_out = max(64, min(4000, int(request.args.get("n") or 1600)))
        quer_curvas = str(request.args.get("curvas") or "1") not in ("0", "false", "nao", "não")
    except ValueError:
        return jsonify({"ok": False, "codigo": "parametro",
                        "error": "t0/t1/n precisam de número"}), 400
    bs = float(meta["bucket_sec"])
    nb = int(meta["n"])
    dur = nb * bs
    if t1 <= t0:
        t0, t1 = 0.0, dur
    t1 = min(t1, dur)
    t0 = min(t0, max(0.0, t1 - bs))
    i0 = int(math.floor(t0 / bs))
    i1 = int(math.ceil(t1 / bs))
    i0 = max(0, min(nb - 1, i0))
    i1 = max(i0 + 1, min(nb, i1))
    span = i1 - i0
    k = max(1, int(math.ceil(span / float(n_out))))
    m = (span // k) * k                      # só blocos inteiros: o resto é descartado, não estimado
    if m < 1:
        k, m = 1, span
    onda = _le_baldes(meta["onda"], nb, i0, m, k, (0, 1))
    res = {"ok": True, "id": r["id"], "bucket_sec": round(bs, 6), "n": int(m // k),
           "dt": round(k * bs, 6), "t0": round(i0 * bs, 4), "t1": round((i0 * bs) + m * bs, 4),
           "baldes_por_ponto": int(k), "recortado_ms": round((span - m) * bs * 1000.0, 2),
           "sr": int(meta["sr"]), "fps": float(meta["fps"]), "lead_ms": float(meta["lead_ms"]),
           "min": [round(float(v), 4) for v in onda[0].max(axis=1)],
           "max": [round(float(v), 4) for v in onda[1].max(axis=1)]}
    if quer_curvas and meta.get("curvas") and os.path.exists(meta["curvas"]):
        cur = _le_baldes(meta["curvas"], nb, i0, m, k, tuple(range(int(meta["nc"]))))
        res["curvas"] = {str(nome): [round(float(v), 4) for v in c.max(axis=1)]
                         for nome, c in zip(meta["nomes"], cur)}
        topo = float(np.max([c.max() for c in cur])) if cur else 1.0
        res["curvas_topo"] = round(topo, 4)
    dets = r.get("detections") or []
    wa, wb = res["t0"], res["t1"]
    ev = [d for d in dets if wa - 0.25 <= float(d.get("time") or 0.0) <= wb + 0.25][:4000]
    # casa cada ataque com a pauta *visível*: "e" virou nota, "m" é o segundo ataque que coube
    # na mesma nota (fusão do gravador), "n" é ataque detectado que não tem nota. Comparar com o
    # que foi *gravado* (não com os dados) é o que responde a pergunta do usuário — "o espectro
    # disse isto, a partitura escreveu aquilo".
    from collections import Counter
    from .score_model import score_visivel
    notas = Counter()
    for b_ in (score_visivel(r["score"]).get("bars") or []):
        for h_ in (b_.get("hits") or []):
            notas[(b_.get("index"), h_.get("tick", 0), h_.get("lane"))] += 1
    vistos: Counter = Counter()
    saida = []
    for d in ev:
        key = (d.get("bar"), d.get("tick", 0), d.get("lane"))
        i = vistos[key]
        vistos[key] += 1
        st = "e" if i < notas.get(key, 0) else ("m" if notas.get(key, 0) > 0 else "n")
        saida.append({"t": d["time"], "lane": d["lane"], "conf": d["conf"], "vel": d["velocity"],
                      "bar": d["bar"], "tick": d["tick"], "resid_ms": d["resid_ms"], "st": st})
    res["eventos"] = saida
    vis = score_visivel(r["score"])
    tot = 0
    na_janela = 0
    for b_ in (vis.get("bars") or []):
        for h_ in (b_.get("hits") or []):
            tot += 1
            tt_ = h_.get("time")
            if tt_ is not None and wa - 0.25 <= float(tt_) <= wb + 0.25:
                na_janela += 1
    res["n_notas_janela"] = na_janela
    res["n_notas_partitura"] = tot
    res["eventos_total"] = len([d for d in dets if wa - 0.25 <= float(d.get("time") or 0.0) <= wb + 0.25])
    res["estados"] = {k: sum(1 for e in saida if e["st"] == k) for k in ("e", "m", "n")}
    res["limitado"] = len(res["eventos"]) < res["eventos_total"]
    return jsonify(_jsonable(res))


# ------------------------------------------------------------------ apresentação (ocultar pistas)
@app.post("/api/view")
def view():
    """Mostra/oculta pistas **sem tocar nos dados**: regravura e devolve o mesmo par svg+layout.

    Ocultar é apresentação. O `score` guardado continua com todas as notas, o `/api/export/json`
    e o `.csv` continuam entregando tudo, e a auditoria continua julgando tudo — só o que é
    desenhado (e o MIDI/MusicXML exportado) respeita `hide_lanes`. Documentado em A16.
    """
    body = request.get_json(silent=True) or {}
    r = RESULTS.get(str(body.get("id") or ""))
    if r is None:
        return jsonify({"ok": False, "error": "análise não encontrada — reenvie a faixa"}), 404
    from .kit import PRATOS
    sc = r["score"]
    if body.get("simples"):
        hide = list(PRATOS)
    elif body.get("hide_lanes") is not None:
        hl = body.get("hide_lanes")
        hide = [str(x) for x in hl] if isinstance(hl, (list, tuple)) else []
    else:
        hide = []
    permitido = {l.id for l in _LANES}
    desconhecidas = [x for x in hide if x not in permitido]
    if desconhecidas:
        return jsonify({"ok": False, "codigo": "pista_desconhecida",
                        "error": "não existe pista %s — as válidas são %s"
                                 % (", ".join(desconhecidas), ", ".join(sorted(permitido)))}), 400
    sc["hide_lanes"] = hide
    r["params"] = dict(r.get("params") or {})
    r["params"]["hide_lanes"] = hide
    r["params"]["simples"] = bool(body.get("simples"))
    try:
        r["svg"], r["layout"] = _svg_pair(sc, r.get("report"), page="a4_landscape")
    except Exception as e:                                   # nunca deixar a aba no escuro
        return jsonify({"ok": False, "codigo": "regravura",
                        "error": "a regravura falhou (%s: %s) — a partitura anterior continua "
                                 "íntegra na tela." % (type(e).__name__, e)}), 500
    rep = r.setdefault("report", {})
    rep["apresentacao"] = {"hide_lanes": hide, "simples": bool(body.get("simples")),
                           "pratos": list(PRATOS),
                           "n_dados": sum(len(b.get("hits") or []) for b in (sc.get("bars") or []))}
    return jsonify(_jsonable(_public(r)))


# ------------------------------------------------------------------ recomendador de parâmetros
TUNE_DUR_MAX = 300.0


@app.post("/api/tune")
def tune():
    """Mede a faixa e recomenda `sensitivity` + `min_confidence`, com os números à mostra.

    Roda no filho isolado (até ~17 análises completas, orçamento proporcional à duração) e é
    cacheado por resultado: um segundo
    clique não paga de novo. Acima de `TUNE_DUR_MAX` recusamos com explicação — varredura em
    faixa longa custaria mais que o prazo do filho, e "esperar 20 min sem dizer nada" é
    exatamente a tela presa que este projeto proíbe.
    """
    body = request.get_json(silent=True) or {}
    r = RESULTS.get(str(body.get("id") or ""))
    if r is None:
        return jsonify({"ok": False, "error": "análise não encontrada — reenvie a faixa"}), 404
    if r.get("tune"):
        return jsonify(_jsonable({"ok": True, "tune": r["tune"], "cache": True,
                                  "id": r["id"]}))
    caminho = r.get("path")
    if not caminho or not os.path.exists(caminho):
        return jsonify({"ok": False, "codigo": "sem_arquivo",
                        "error": "o arquivo original não está mais em disco nesta instância, "
                                 "então não há o que remarcar. Reenvie a faixa."}), 409
    rep = r.get("report") or {}
    dur = float(((rep.get("file") or {}).get("duration_sec")) or 0.0)
    if dur > TUNE_DUR_MAX:
        return jsonify({"ok": False, "codigo": "faixa_longa",
                        "error": "a varredura re-analisa a faixa ~16 vezes; para %.0f s isso "
                                 "passaria do prazo de uma análise. Marque um trecho de até %.0f s "
                                 "e envie-o, ou ajuste `sensitivity`/`min_confidence` à mão na aba "
                                 "Parâmetros." % (dur, TUNE_DUR_MAX)}), 413
    params = dict(r.get("params") or {})
    # sem `body["sens"]` aqui de propósito: uma rota que aceita um parâmetro e o ignora é o defeito
    # que A14 registrou para `analysis_sr`. A grade é do `tune`, e o usuário não precisa escolhe-la.
    limite = min(3600.0, max(300.0, 17.0 * max(1.0, dur) * 0.30))
    try:
        out = _analise_isolada(caminho, r.get("filename") or "audio.wav", params,
                               acao="sintonia", limite=limite)
    except Exception as e:
        resp = _erro_analise(e)
        return resp if isinstance(resp, tuple) else (jsonify({"ok": False, "error": str(e)}), 500)
    rec = out.get("tune") or {}
    r["tune"] = rec
    return jsonify(_jsonable({"ok": True, "tune": rec, "cache": False, "id": r["id"],
                              "duracao_s": round(dur, 2)}))


# ------------------------------------------------------------------ MIDI como leitura, não só download
@app.post("/api/midi_view")
def midi_view():
    """Gera o MIDI da partitura atual e o relê com o parser do `qa` — a aba mostra o arquivo.

    Importante: as notas vêm da *leitura do .mid*, tick a tick, não do nosso dicionário. É o que
    faz esta aba provar que o export existe e bate com a pauta (mesma parser que T12 usa, com
    duração recuperada dos note-off). O que o usuário vê é o que um sequenciador veria.
    """
    body = request.get_json(silent=True) or {}
    r = RESULTS.get(str(body.get("id") or ""))
    sc = _score_from(body)
    if sc is None:
        return jsonify({"ok": False, "error": "partitura não encontrada"}), 400
    from . import qa as QA
    from .kit import LANES
    from .score_model import score_visivel
    try:
        dados = to_midi(sc)
        m = QA._read_smf(dados, notas=True)
    except Exception as e:
        return jsonify({"ok": False, "codigo": "midi",
                        "error": "a geração/leitura do MIDI falhou (%s: %s) — o download do .mid "
                                 "pode ainda assim funcionar a partir da aba Exportar."
                                 % (type(e).__name__, e)}), 500
    div = int(m.get("division") or 480) or 480
    tempo = int(m.get("tempo_us") or 600000) or 600000
    gm2lane: Dict[int, str] = {}
    for l in LANES:
        gm2lane.setdefault(int(l.gm), str(l.id))
    escala = float(tempo) / 1e6 / float(div)                  # segundos por tick, do próprio arquivo
    ns = m.get("notas") or []
    LIMITE = 6000
    notas = [[round(t * escala, 4), round(max(du, 1) * escala, 4), int(p), int(v),
              gm2lane.get(int(p), "outro")] for (t, p, v, du) in ns[:LIMITE]]
    vis = score_visivel(sc)
    n_vis = sum(len(b.get("hits") or []) for b in (vis.get("bars") or []))
    n_dados = sum(len(b.get("hits") or []) for b in (sc.get("bars") or []))
    alt = [p for _, p, _, _ in ns] or [36]
    return jsonify(_jsonable({
        "ok": True, "id": (r or {}).get("id"),
        "notas": notas, "n_arquivo": len(ns), "n_escritas_visiveis": n_vis,
        "n_na_partitura": n_dados, "truncado": len(ns) > LIMITE, "limite": LIMITE,
        "sem_off": int(m.get("sem_off") or 0),
        "division": div, "tempo_us": tempo, "formato": m.get("format"),
        "trilhas": m.get("ntracks_read"), "marcadores": m.get("markers"),
        "bytes": len(dados), "ocultas": list(sc.get("hide_lanes") or []),
        "pitch_faixa": [int(min(alt)), int(max(alt))],
        "por_pista": {k: sum(1 for x in notas if x[4] == k) for k in sorted({x[4] for x in notas})},
        "bpm_arquivo": round(60e6 / float(tempo), 3),
        "bpm_partitura": sc.get("bpm"),
        # o .mid começa no primeiro tempo da partitura; o arquivo do usuário pode ter silêncio
        # antes disso, e quem for montar na DAW precisa do número — escondê-lo é que gera a
        # impressão de que o MIDI "não bate" (A22).
        "lead_ms": float(((sc.get("report") or {}).get("file") or {}).get("trim_lead_ms") or 0.0),
        "dur_audio_s": float(((sc.get("report") or {}).get("file") or {}).get("duration_sec") or 0.0)}))


def _jsonable(obj):
    """numpy → tipos que o `jsonify` aceita (o auditor devolve int/float de arrays)."""
    def dec(o):
        try:
            f = float(o)
            return int(f) if float(f).is_integer() and abs(f) < 1e15 else f
        except Exception:
            return str(o)
    return json.loads(json.dumps(obj, default=dec, ensure_ascii=False))


def _qa_audio(base: dict):
    """áudio original para os oráculos, com cache (a auditoria roda 2× por sessão)."""
    if base.get("_qa_aud") is not None:
        return base["_qa_aud"]
    x, sr = None, 0
    try:
        import numpy as np
        from .audio_io import decode_file
        a = decode_file(base["path"])
        x, sr = np.asarray(a.x, dtype="float32"), int(a.sr)
    except Exception:
        app.logger.info("auditoria sem áudio decodificado")
    base["_qa_aud"] = (x, sr)
    return x, sr


@app.post("/api/qa")
def qa_audit():
    """
    Auditoria dupla do resultado: cada estágio medido duas vezes por caminhos independentes,
    mais as leis de escrita da partitura e a releitura dos exports. `level="full"` acrescenta
    a releitura de MIDI/PDF, a ida-e-volta da síntese e a cobertura por oráculo.
    """
    body = request.get_json(silent=True) or {}
    base = RESULTS.get(body.get("id") or "", {})
    if not base:
        return jsonify({"ok": False, "error": "resultado não encontrado"}), 404
    from . import qa as QA
    level = "full" if str(body.get("level")) == "full" else "fast"
    x, sr = _qa_audio(base)
    try:
        out = QA.audit(base["score"], base.get("report"), x=x, sr=sr,
                       dets=base.get("detections"), level=level)
    except Exception as e:
        app.logger.warning("auditoria falhou: %s\n%s", e, traceback.format_exc())
        return jsonify({"ok": False, "error": "a auditoria não rodou: %s" % e}), 500
    md = QA.to_markdown(out, "Validação dupla — %s" % (base.get("filename") or "faixa"))
    base["qa"] = out
    return jsonify(_jsonable({"ok": True, "qa": out, "markdown": md, "level": level}))


@app.post("/api/qa/fix")
def qa_fix():
    """
    Agente de correção: aplica só o que é determinístico, reconstrói pela mesma porta do editor
    e re-audita. Se o quadro não melhorar, o reparo é revertido e a partitura fica como estava —
    o log diz o que aconteceu.
    """
    body = request.get_json(silent=True) or {}
    base = RESULTS.get(body.get("id") or "", {})
    if not base:
        return jsonify({"ok": False, "error": "resultado não encontrado"}), 404
    from . import qa as QA
    level = "full" if str(body.get("level")) == "full" else "fast"
    allow = body.get("allow")
    x, sr = _qa_audio(base)
    try:
        res = QA.audit_and_repair(base["score"], base.get("report"), x=x, sr=sr,
                                  dets=base.get("detections"), level=level,
                                  allow=list(allow) if allow else None)
    except Exception as e:
        app.logger.warning("reparo falhou: %s\n%s", e, traceback.format_exc())
        return jsonify({"ok": False, "error": "o agente não conseguiu reparar: %s" % e}), 500
    if res.get("changed"):
        base["score"] = res["score_novo"]
        base["svg"], base["layout"] = _svg_pair(base["score"], base.get("report"),
                                                 page="a4_landscape")
    after = res.get("after") or res.get("before") or {}
    md = QA.to_markdown(after, "Validação dupla (após correções) — %s" % (base.get("filename") or "faixa"))
    out = {"ok": True, "changed": bool(res.get("changed")), "applied": res.get("applied") or [],
           "reverted": bool(res.get("reverted")), "rounds": res.get("rounds", 0),
           "pesos": res.get("pesos") or {}, "qa": after, "markdown": md,
           "score": base["score"], "svg": base["svg"], "layout": base.get("layout")}
    return jsonify(_jsonable(out))


@app.get("/api/lanes")
def lanes():
    from .kit import LANES
    return jsonify({"ok": True, "lanes": [{"id": l.id, "name": l.name, "short": l.short,
                                           "head": l.head, "stem": l.stem, "voice": l.voice,
                                           "gm": l.gm, "staff": l.staff, "group": l.group}
                                          for l in LANES]})


@app.errorhandler(413)
def too_large(e):
    return jsonify({"ok": False, "error": "arquivo grande demais (limite 80 MB)"}), 413


@app.errorhandler(500)
def internal(e):
    return jsonify({"ok": False, "error": "erro interno: %s" % e}), 500


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="DrumScribe — servidor local")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    try:
        limpo = _varre_wave()
        if limpo:
            print("  · %d envelope(s) fino(s) órfão(s) removidos de out/wave/ (a sessão anterior "
                  "terminou com eles; os resultados vivem em memória — B10)" % limpo)
    except Exception:                                      # a limpeza nunca impede o servidor de subir
        pass
    print("DrumScribe em http://%s:%d/  (Ctrl+C para sair)" % (a.host if a.host != "0.0.0.0"
                                                                else "localhost", a.port))
    app.run(host=a.host, port=a.port, debug=a.debug, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
