"""
Autoteste de ponta a ponta do DrumScribe (sem dependências de teste externas).

    python3 tests/selfcheck.py            # roda tudo e imprime o resumo
    python3 tests/selfcheck.py --quick    # pula a renderização de imagem do PDF

Checa, na ordem:
  1. import do pacote e contrato do dicionário-Score (round-trip `Score.from_dict`);
  2. análise do demo: métricas mínimas contra o ground-truth (pos_accuracy, hit_f1, BPM,
     fórmula de compasso) — o objetivo é pegar regressão de grade/alimentação;
  3. gravura: layout produz primitivas, SVG é bem formado e o PDF tem páginas;
  4. exportações: MusicXML parses + soma de notas, SMF parseia e o número de notas
     confere com a partitura, CSV tem uma linha por nota;
  5. edição: `rebuild_score` não duplica notas no mesmo pulso e preserva o total.
"""
from __future__ import annotations

import io
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.evaluate import evaluate                                    # noqa: E402

from drumscribe.engrave import to_pdf_bytes, to_svg                   # noqa: E402
from drumscribe.layout import layout_score                            # noqa: E402
from drumscribe.midi_out import to_midi                               # noqa: E402
from drumscribe.musicxml import to_musicxml                           # noqa: E402
from drumscribe.pipeline import rebuild_score, transcribe_bytes       # noqa: E402
from drumscribe.score_model import Score                              # noqa: E402

WAV = os.path.join(ROOT, "samples", "demo_drums.wav")
GT = os.path.join(ROOT, "samples", "demo_groundtruth.json")

fails: list = []
oks: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (oks if cond else fails).append("%s%s" % (name, (" — " + detail) if detail else ""))
    print(("  ok   " if cond else "  FALHA") + "  " + name + ((" — " + detail) if detail else ""))


# ------------------------------------------------------------------ parseadores
def parse_smf(data: bytes):
    assert data[:4] == b"MThd", "sem cabeçalho MThd"
    clen, fmt, ntr, div = struct.unpack(">IHHH", data[4:14])
    assert clen == 6, "tamanho do cabeçalho != 6"
    off = 14
    tracks = []
    for _ in range(ntr):
        typ = data[off:off + 4]
        ln = struct.unpack(">I", data[off + 4:off + 8])[0]
        assert typ == b"MTrk", "chunk != MTrk"
        tracks.append(data[off + 8:off + 8 + ln])
        off += 8 + ln
    assert off == len(data), "bytes sobrando no fim do arquivo"
    out = []
    for body in tracks:
        t = 0
        run = 0
        k = 0
        ev = []
        while k < len(body):
            while True:
                b = body[k]
                k += 1
                run = (run << 7) | (b & 0x7F)
                if not (b & 0x80):
                    break
            t += run
            run = 0
            if k >= len(body):
                break
            if body[k] == 0xFF:
                typ2 = body[k + 1]
                k2 = k + 2
                l = 0
                while True:
                    b = body[k2]
                    k2 += 1
                    l = (l << 7) | (b & 0x7F)
                    if not (b & 0x80):
                        break
                ev.append(("meta", t, typ2, body[k2:k2 + l]))
                k = k2 + l
                if typ2 == 0x2F:
                    break
                continue
            st = body[k] & 0xF0
            ch = body[k] & 0x0F
            if st in (0x90, 0x80):
                ev.append(("note", t, st, ch, body[k + 1], body[k + 2]))
                k += 3
            elif st in (0xC0, 0xD0):
                k += 2
            else:
                k += 3
        out.append(ev)
    return fmt, div, out


def main() -> int:
    quick = "--quick" in sys.argv
    print("DrumScribe · selfcheck")
    if not os.path.exists(WAV):
        print("amostra ausente: rode  python3 samples/make_demo.py")
        return 2

    # 1 ------------------------------------------------------------- import + contrato
    print("\n[1] pacote e contrato da partitura")
    with open(WAV, "rb") as f:
        audio = f.read()
    res = transcribe_bytes(audio, "demo_drums.wav", params={"sensitivity": 0.95})
    sc = res["score"]
    d = sc.to_dict()
    for key in ("title", "bpm", "meter", "swing", "ticks_per_quarter", "ticks_per_bar",
                "bars", "lanes", "report"):
        check("campo %s no dicionário" % key, key in d)
    rt = Score.from_dict(json.loads(json.dumps(d)))
    d2 = rt.to_dict()
    n1 = sum(len(b["hits"]) for b in d["bars"])
    n2 = sum(len(b["hits"]) for b in d2["bars"])
    check("round-trip preserva as notas", n1 == n2, "%d vs %d" % (n1, n2))
    check("round-trip preserva bpm/compasso",
          abs(float(d2["bpm"]) - float(d["bpm"])) < 1e-6 and d2["meter"] == d["meter"])

    # 2 ------------------------------------------------------------- métricas
    print("\n[2] exatidão contra o ground-truth")
    gt = json.load(open(GT))
    m = evaluate(d, res["detections"], gt)
    for k, v in (("n_gt", m["n_gt"]), ("n_notated", m["n_notated"]),
                 ("onset_recall", m["onset_recall"]), ("onset_precision", m["onset_precision"]),
                 ("hit_f1", m["hit_f1"]), ("pos_accuracy", m["pos_accuracy"]),
                 ("tick_mae", m["tick_mae"]), ("bpm_err_pct", m["bpm_err_pct"]),
                 ("vel_pearson", m.get("vel_pearson"))):
        print("     %-15s %s" % (k, v))
    check("recall de onsets ≥ 0.85", m["onset_recall"] >= 0.85, str(m["onset_recall"]))
    check("precision de onsets ≥ 0.85", m["onset_precision"] >= 0.85, str(m["onset_precision"]))
    check("hit_f1 ≥ 0.50", m["hit_f1"] >= 0.50, str(m["hit_f1"]))
    check("pos_accuracy ≥ 0.90", m["pos_accuracy"] >= 0.90, str(m["pos_accuracy"]))
    check("tick_mae ≤ 0.5", 0 <= m["tick_mae"] <= 0.5, str(m["tick_mae"]))
    check("erro de BPM < 0,5 %", m["bpm_err_pct"] < 0.5, "%.2f%%" % m["bpm_err_pct"])
    check("fórmula de compasso = 4/4", m["meter"] == gt["meter"], str(m["meter"]))
    check("bumbo com recall total", m["per_lane"]["kick"]["recall"] >= 0.95,
          str(m["per_lane"]["kick"]["recall"]))
    check("chimel com recall ≥ 0.75", m["per_lane"]["hat"]["recall"] >= 0.75,
          str(m["per_lane"]["hat"]["recall"]))
    grid = res["report"]["grid"]
    check("grade escolhida sem 32ºs indevidos", grid["odd_ticks"] == 0,
          "mode=%s odd=%d" % (grid["mode"], grid["odd_ticks"]))
    check("tempo da partitura coerente com o áudio", abs(d["bpm"] - gt["bpm"]) / gt["bpm"] < 0.006)

    # 3 ------------------------------------------------------------- gravura
    print("\n[3] gravura (layout / SVG / PDF)")
    lay = layout_score(d, page="a4_landscape")
    nops = sum(len(p["ops"]) for p in lay["pages"])
    check("layout gerou primitivas", nops > 400, "%d ops em %d página(s)" % (nops, len(lay["pages"])))
    check("sistemas com compassos", lay["meta"]["systems"] >= 1 and lay["meta"]["bars_per_system"] >= 2,
          "%d sistemas × %d compassos" % (lay["meta"]["systems"], lay["meta"]["bars_per_system"]))
    svg = to_svg(d)
    import xml.etree.ElementTree as ET
    try:
        ET.fromstring(svg)
        check("SVG é XML válido", True, "%d KB" % (len(svg) // 1024))
    except Exception as e:
        check("SVG é XML válido", False, str(e))
    # ------------------------------------------- mapa de geometria (cursor de reprodução)
    bm = lay["meta"].get("bar_map") or []
    nb = len(d["bars"])
    check("bar_map cobre todo compasso, uma vez, na ordem",
          len(bm) == nb and [g["i"] for g in bm] == list(range(nb)),
          "%d entradas × %d compassos" % (len(bm), nb))
    ok_geom = all(0 < g["x0"] < g["x1"] <= lay["meta"]["page_w"] + 0.01
                  and 0 <= g["y0"] < g["y1"] <= lay["meta"]["doc_h"] + 0.01 for g in bm)
    check("bar_map dentro da página", ok_geom,
          "x máx %.1f · y máx %.1f" % (max(g["x1"] for g in bm), max(g["y1"] for g in bm)))
    vizinhos = [(a, b) for a, b in zip(bm, bm[1:])
                if a["page"] == b["page"] and abs(a["y0"] - b["y0"]) < 0.01]
    check("compassos do mesmo sistema se tocam, sem sobrepor",
          all(abs(a["x1"] - b["x0"]) < 0.01 for a, b in vizinhos), "%d pares" % len(vizinhos))
    check("o batimento 0 fica dentro do compasso (o cursor nunca sai da caixa)",
          all(g["x0"] <= g["beat0"] < g["beat1"] <= g["x1"] for g in bm))
    # a caixa do mapa tem de cobrir a tinta desenhada — medida nos próprios ops, não presumida
    import drumscribe.layout as _L
    medida: list = []
    _orig = _L.draw_bar

    def _sonda(ops, bar, x0, x1, line1_y, lsz, tpbx, tprx, score, **kw):
        i0 = len(ops)
        _orig(ops, bar, x0, x1, line1_y, lsz, tpbx, tprx, score, **kw)
        xs = []
        for op in ops[i0:]:
            k = op[0]
            if k == "line":
                xs += [op[1], op[3]]
            elif k in ("path", "poly"):
                xs += [pt[0] for pt in op[1]]
            elif k in ("ellipse", "text", "noteglyph"):
                xs.append(op[1])
            elif k == "rect":
                xs += [op[1], op[1] + op[3]]
            elif k == "curve":
                xs += [op[1], op[5]]
        if xs:
            medida.append((int(bar.get("index", 0)), min(xs), max(xs)))
        return None

    _L.draw_bar = _sonda
    try:
        lay_s = layout_score(d, page="a4_landscape")
    finally:
        _L.draw_bar = _orig
    caixas = {g["i"]: g for g in lay_s["meta"]["bar_map"]}
    fora = [(i, round(x0 - (caixas[i]["x0"] - caixas[i]["folga"]), 2),
             round((caixas[i]["x1"] + caixas[i]["folga"]) - x1, 2))
            for i, x0, x1 in medida
            if x0 < caixas[i]["x0"] - caixas[i]["folga"] - 0.01
            or x1 > caixas[i]["x1"] + caixas[i]["folga"] + 0.01]
    check("a faixa do cursor cobre a tinta desenhada de cada compasso",
          bool(medida) and not fora, "%d compassos medidos · fora %s" % (len(medida), fora[:2]))

    check("bar_map é determinístico",
          bm == layout_score(d, page="a4_landscape")["meta"]["bar_map"])
    import re as _re
    vb = _re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    check("o mapa cabe no viewBox do SVG",
          bool(vb) and float(vb.group(2)) >= lay["meta"]["doc_h"] - 0.02
          and max(g["x1"] for g in bm) <= float(vb.group(1)) + 0.01,
          vb.group(0) if vb else "sem viewBox")
    check("a altura total do documento bate com as páginas",
          abs(lay["meta"]["doc_h"] - lay["meta"]["page_h"] * len(lay["pages"])) < 0.02,
          "%.1f = %.1f × %d" % (lay["meta"]["doc_h"], lay["meta"]["page_h"], len(lay["pages"])))

    pdf = to_pdf_bytes(d)
    check("PDF com assinatura e %%EOF", pdf[:5] == b"%PDF-" and b"%%EOF" in pdf[-32:],
          "%d bytes" % len(pdf))
    if not quick:
        try:
            import pypdfium2 as pdfium
            doc = pdfium.PdfDocument(pdf)
            npg = len(doc)
            check("PDF renderiza as páginas", npg == len(lay["pages"]),
                  "%d páginas" % npg)
            png = doc[0].render(scale=1.0).to_pil()
            a = np.asarray(png.convert("L"))
            ink = float((a < 128).mean())
            check("página 1 tem tinta suficiente", 0.002 < ink < 0.30, "=%.3f do pixel" % ink)
        except ImportError:
            print("     (pypdfium2 ausente — checagem visual pulada)")

    # 4 ------------------------------------------------------------- exportações
    print("\n[4] exportações")
    xml = to_musicxml(d)
    root = ET.fromstring(xml)
    nnotes = len(root.findall(".//{*}part/{*}measure/{*}note"))
    nrest = len(root.findall(".//{*}note/{*}rest"))
    check("MusicXML: notas escritas (com pausas) ≥ hits", nnotes >= n1,
          "%d <note> p/ %d hits, %d pausas" % (nnotes, n1, nrest))
    check("MusicXML: clave de percussão", "perc" in ET.tostring(root, encoding="unicode")[:4000]
          or "<sign>perc</sign>" in xml)
    check("MusicXML: compasso presente", "<beat-type>4</beat-type>" in xml)
    check("MusicXML: tempo em <sound>", 'tempo="100"' in xml or "tempo=\"99" in xml or "tempo=\"10" in xml)
    fmt, div, tracks = parse_smf(to_midi(d))
    ons = [e for tr in tracks for e in tr if e[0] == "note" and e[2] == 0x90 and e[5] > 0]
    check("MIDI: tipo 1, PPQ 480", fmt == 1 and div == 480, "fmt=%d div=%d" % (fmt, div))
    check("MIDI: nº de notas = nº de notas da partitura", len(ons) == n1,
          "%d vs %d" % (len(ons), n1))
    check("MIDI: tudo no canal 10 (percussão)", all(e[4 - 1] == 9 for e in ons))
    gm = set(e[4] for e in ons)
    check("MIDI: apenas notas GM de bateria (35–81)", all(35 <= g <= 81 for g in gm),
          "notas %s" % sorted(gm)[:8])

    # 5 ------------------------------------------------------------- edição
    print("\n[5] reconstrução após edição manual")
    hits = [dict(h, bar=b["index"]) for b in d["bars"] for h in b["hits"]]
    for h in hits[:3]:
        h["lane"] = "tom_mid"
    d3 = rebuild_score(hits, bpm=d["bpm"], meter=d["meter"], swing=d["swing"])
    n3 = sum(len(b["hits"]) for b in d3["bars"])
    dup = 0
    for b in d3["bars"]:
        seen = set()
        for h in b["hits"]:
            k = (h["tick"], h["lane"])
            dup += 1 if k in seen else 0
            seen.add(k)
    check("rebuild sem notas duplicadas no mesmo pulso", dup == 0, "%d duplicidades" % dup)
    check("rebuild preserva as notas (±3 por fusões)", abs(n3 - n1) <= 3, "%d vs %d" % (n3, n1))
    check("rebuild mantém o total do kit", len(d3["lanes"]) >= 3, str(d3["lanes"]))
    lay3 = layout_score(d3)
    check("rebuild volta a gravurar", sum(len(p["ops"]) for p in lay3["pages"]) > 200)

    # ------------------------------------------------------- entrada, taxa e orçamento (A14)
    from drumscribe import audio_io as AI
    from drumscribe import dsp as _dsp
    from drumscribe import pipeline as P
    bruto = open(os.path.join(ROOT, "samples", "demo_drums.wav"), "rb").read()
    a_full = AI.decode_bytes(bruto, "demo.wav")
    a_dn = AI.decode_bytes(bruto, "demo.wav", analysis_sr=22050)
    check("analysis_sr reamostra de fato (metade das amostras)",
          abs(a_dn.x.size * 2 - a_full.x.size) <= 2, "%d vs %d" % (a_dn.x.size, a_full.x.size))
    check("source_sr preserva a taxa do arquivo", a_dn.source_sr == a_full.sr and
          a_dn.sr == 22050, "%s/%s" % (a_dn.source_sr, a_dn.sr))
    try:
        AI.decode_bytes(bruto, "demo.wav", analysis_sr=5512)
        check("analysis_sr inválido falha alto", False, "aceitou em silêncio")
    except ValueError as e:
        check("analysis_sr inválido falha alto", "fora da faixa" in str(e), str(e)[:60])
    check("trim de silêncio devolve deslocamento e não engole o início",
          AI.trim_silence(np.zeros(44100, np.float32), 44100)[1] == 0)
    # a escada não pode mover a grade física nem descer até calar uma pista (A14)
    pl = P.plano_dsp(3695580, 44100, 256, 1024, 10_000.0)[4]
    check("escada preserva ms/quadro", len({round(1000.0 * h / sr2, 3) for (sr2, h, nf, c) in pl}) == 1,
          str(pl))
    check("escada preserva Hz/bin", len({round(sr2 / float(nf), 2) for (sr2, h, nf, c) in pl}) == 1)
    check("escada não desce abaixo do piso que cala pista",
          all(sr2 >= P.SR_MINIMO_ESCADA for (sr2, h, nf, c) in pl), str([x[0] for x in pl]))
    check("Bluestein medido: n com fator >13 paga 2,6×", P._fator_bluestein(6839028) > 2.0 and
          P._fator_bluestein(2 ** 20) == 1.0)
    # banda acima de Nyquist = ausência de evidência, nunca exceção (achado de hoje)
    sp_peq = _dsp.stft(np.zeros(11025, np.float32), 11025, n_fft=256, hop=128)
    bf = _dsp.band_flux(sp_peq, 7000.0, 16000.0)
    check("banda acima de Nyquist devolve fluxo nulo", bf.shape == (sp_peq.n_frames,) and
          float(np.max(np.abs(bf))) == 0.0, str(bf.shape))

    print("\n%d verificações ok, %d falhas" % (len(oks), len(fails)))
    if fails:
        for f in fails:
            print("  ✗ " + f)
        return 1
    print("tudo certo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
