#!/usr/bin/env python3
"""Leis do zoom de waveform, do piano-roll do .mid e da partitura simples.

Cobre as três coisas pedidas na rodada, cada uma com uma propriedade medida — não com
"respondeu 200":

  A  resolução: o envelope fino tem baldes ~1 ms (≥ 10× mais finos que a visão geral) e o número
     de pontos bate com a janela pedida.
  B  nenhum ponto perdido: `min`, `max` e cada curva têm exatamente `n` pontos — inclusive quando
     o agrupamento é 1 (janela estreita). Um ramo que rearrumava o vetor para (1, N) devolvia um
     ponto só e o zoom "não funcionava".
  C  contenção de pico: relido no mesmo balde da visão geral, o piso/teto fino não é menor.
  D  casamento com a pauta: `estados` soma `eventos`, e "e"+"m" cobrem as notas gravadas.
  E  escala de tempo do MIDI: `segundos = tick · tempo_us / 1e6 / division`, conferido tick a tick
     com a grade da partitura — o inverso dessa divisão dava nota de 24 s.
  F  ocultar é apresentação: `hide_lanes` muda a gravura, o .mid e o MusicXML, e não muda o
     JSON, o CSV, a contagem de dados nem a auditoria (A16).

    python3 tests/wave_check.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["DRUMSCRIBE_INLINE"] = "1"          # sem filho: o teste é do roteamento e dos números
os.environ.setdefault("DRUMSCRIBE_PRAZO_ANALISE", "900")

from drumscribe import server                                  # noqa: E402
from drumscribe.kit import PRATOS                               # noqa: E402

ok, falhas = [], []


def checa(nome, cond, dado=""):
    txt = "%s%s" % (nome, (" — " + str(dado)) if dado else "")
    (ok if cond else falhas).append(txt)
    print("  %s %s" % ("✓" if cond else "✗", txt))


DEMO = os.path.join(ROOT, "samples", "demo_drums.wav")
dados = open(DEMO, "rb").read()
app = server.app
app.config["TESTING"] = True
cl = app.test_client()

r = cl.post("/api/analyze", data={"file": (io.BytesIO(dados), "demo_drums.wav"),
                                  "params": json.dumps({"min_confidence": 0.30})},
            content_type="multipart/form-data")
j = r.get_json()
assert r.status_code == 200, (r.status_code, r.get_data(as_text=True)[:400])
rid = j["id"]
dur = j["report"]["file"]["duration_sec"]
wf = j.get("wave_fine") or {}
print("análise inline: %s · %.1f s · %d baldes de %.3f ms (%.1f MB)" % (
    rid, dur, wf.get("n", 0), 1000 * (wf.get("bucket_sec") or 0), (wf.get("bytes") or 0) / 1e6))

checa("A  envelope fino anunciado ao navegador", bool(wf) and wf["n"] > 1000, wf.get("n"))
checa("A  resolução ≈1 ms e ≥10× a visão geral",
      0.0005 <= wf["bucket_sec"] <= 0.002 and dur / wf["n"] < (j["wave"]["dt"]) / 10,
      "%.4f ms vs %.1f ms" % (1000 * wf["bucket_sec"], 1000 * j["wave"]["dt"]))

g = cl.get("/api/wave/%s?t0=0&t1=%s&n=1200&curvas=1" % (rid, dur)).get_json()
z = cl.get("/api/wave/%s?t0=1.0&t1=1.4&n=2000&curvas=1" % rid).get_json()
est = cl.get("/api/wave/%s?t0=%s&t1=%s&n=3000&curvas=0" % (rid, dur * 0.4, dur * 0.5)).get_json()

esperado = int(round((1.4 - 1.0) / z["bucket_sec"]))
checa("A  nº de pontos bate com a janela", abs(z["n"] - esperado) <= 2, "%d vs ~%d" % (z["n"], esperado))
checa("B  min/max/curvas com o mesmo n (agrupamento 1)",
      len(z["min"]) == len(z["max"]) == z["n"] and all(len(v) == z["n"] for v in z["curvas"].values()),
      "n=%d · curvas %s" % (z["n"], {k: len(v) for k, v in z["curvas"].items()}))
checa("B  idem com agrupamento >1 (janela inteira)",
      len(g["min"]) == len(g["max"]) == g["n"] and all(len(v) == g["n"] for v in g["curvas"].values()),
      "n=%d k=%d" % (g["n"], g["baldes_por_ponto"]))
checa("B  curvas são as quatro bandas anunciadas", sorted(z["curvas"]) == sorted(wf["nomes"]),
      sorted(z["curvas"]))

i0 = int(1.0 / g["dt"])
i1 = int(1.4 / g["dt"])
pino_fino = max(z["max"])
pino_geral = max(g["max"][i0:i1 + 1]) if i1 > i0 else 0.0
checa("C  o balde fino não perde pico do balde geral", pino_fino >= pino_geral - 1e-3,
      "%.3f vs %.3f" % (pino_fino, pino_geral))
checa("C  piso dentro do intervalo lido", min(z["min"]) >= -1.0 and max(z["max"]) <= 1.0 + 1e-6,
      "%.2f…%.2f" % (min(z["min"]), max(z["max"])))

tt = [e["t"] for e in z["eventos"]]
checa("D  eventos dentro da janela pedida", all(-0.3 <= t - 1.0 for t in tt) and
      all(t - 1.4 <= 0.3 for t in tt), "%.3f…%.3f" % (min(tt), max(tt)) if tt else "vazio")
som = sum(est["estados"].values())
checa("D  estados cobrem todos os eventos", som == len(est["eventos"]) == est["eventos_total"],
      "%s = %d" % (json.dumps(est["estados"]), som))
checa("D  contagem de notas da janela é a da gravura visível",
      est["n_notas_janela"] > 0 and est["n_notas_partitura"] >= est["n_notas_janela"],
      "%d na janela / %d no total" % (est["n_notas_janela"], est["n_notas_partitura"]))
resid = [abs(e["resid_ms"]) for e in est["eventos"]]
checa("D  resíduo de rejanelagem plausível (< 120 ms mediano)",
      resid and sorted(resid)[len(resid) // 2] < 120, "%.1f ms" % (sorted(resid)[len(resid) // 2]))

m = cl.post("/api/midi_view", json={"id": rid}).get_json()
checa("E  o .mid relido bate com o que a pauta tem", m["n_arquivo"] == m["n_escritas_visiveis"] > 0,
      "%d vs %d" % (m["n_arquivo"], m["n_escritas_visiveis"]))
esc = float(m["tempo_us"]) / 1e6 / float(m["division"])
checa("E  escala = tempo_us/1e6/division (segundos por tick)",
      abs(esc - float(m["tempo_us"]) / 1e6 / float(m["division"])) < 1e-12
      and abs(60e6 / float(m["tempo_us"]) - float(m["bpm_arquivo"])) < 1e-3   # 3 decimais no rótulo
      and abs(float(m["bpm_arquivo"]) - float(j["score"]["bpm"])) < 0.1,
      "division %d · %.4f ms/tick · %.3f BPM (pauta %.3f) — a faixa não é 100,0 exato, e a "
      "escala vem do arquivo, não de uma constante" % (m["division"], 1000 * esc,
                                                        m["bpm_arquivo"], j["score"]["bpm"]))
# a lei que importa: as notas estão *no* grid de ticks do arquivo (erro de meio tick = bug de escala)
por_lane = {}
for n in m["notas"]:
    por_lane.setdefault(n[4], []).append(n)
pior = 0.0
pares = 0
for ln, lista in por_lane.items():
    lista = sorted(lista, key=lambda x: x[0])[:80]
    for x, y in zip(lista, lista[1:]):
        dt = y[0] - x[0]
        if dt <= 0:
            continue
        pior = max(pior, abs(dt - round(dt / esc) * esc))
        pares += 1
checa("E  espaçamentos são inteiros de tick (escala certa)", pior < 1e-4 and pares > 20,
      "pior desvio %.5f s em %d pares" % (pior, pares))
tmax = max(n[0] for n in m["notas"])
checa("E  nenhum evento fora da duração da faixa", tmax <= dur + 0.5, "%.2f s (faixa %.2f s)" % (tmax, dur))
checa("E  durações entre 1 tick e 1 compasso",
      all(1e-4 <= n[1] <= 3.5 for n in m["notas"]), "%.3f…%.3f s" % (
          min(n[1] for n in m["notas"]), max(n[1] for n in m["notas"])))
checa("E  toda altura de nota tem pista conhecida", "outro" not in m["por_pista"],
      sorted(m["por_pista"]))
spb = 60.0 / float(m["bpm_arquivo"]) / (j["score"]["ticks_per_beat"] or 8)
durs = sorted({round(n[1] / spb) for n in m["notas"]})
checa("E  durações são múltiplos inteiros do pulso", all(0 < d <= 16 for d in durs), durs)

# ---------------------------------------------------------------- F · ocultar é apresentação
svg_cheia = j["svg"]
n_dados = sum(len(b["hits"]) for b in j["score"]["bars"])
json_antes = json.loads(cl.post("/api/export/json", json={"id": rid, "score": j["score"]}).get_data(as_text=True))
csv_antes = cl.post("/api/export/csv", json={"id": rid, "score": j["score"]}).get_data(as_text=True)
xml_antes = cl.post("/api/export/musicxml", json={"id": rid, "score": j["score"]}).get_data(as_text=True)
mid_antes = cl.post("/api/export/mid", json={"id": rid, "score": j["score"]}).get_data()

v = cl.post("/api/view", json={"id": rid, "hide_lanes": list(PRATOS)}).get_json()
n_depois = sum(len(b["hits"]) for b in v["score"]["bars"])
j2 = json.loads(cl.post("/api/export/json", json={"id": rid}).get_data(as_text=True))
csv_depois = cl.post("/api/export/csv", json={"id": rid}).get_data(as_text=True)
xml_depois = cl.post("/api/export/musicxml", json={"id": rid}).get_data(as_text=True)
mid_depois = cl.post("/api/export/mid", json={"id": rid}).get_data()
m2 = cl.post("/api/midi_view", json={"id": rid}).get_json()
qa = cl.post("/api/qa", json={"id": rid, "level": "fast"}).get_json()

checa("F  a gravura encolhe quando os pratos somem", len(v["svg"]) < len(svg_cheia) * 0.9,
      "%d → %d bytes" % (len(svg_cheia), len(v["svg"])))
checa("F  os dados não mudam (n_hits igual)", n_depois == n_dados, "%d vs %d" % (n_depois, n_dados))
checa("F  o JSON exportado continua com tudo",
      sum(len(b["hits"]) for b in j2["score"]["bars"]) == n_dados, str(n_dados))
checa("F  o CSV continua com tudo", csv_depois == csv_antes, "%d linhas" % csv_antes.count(chr(10)))
checa("F  MIDI/MusicXML são apresentação (encolhem)",
      len(mid_depois) < len(mid_antes) and len(xml_depois) < len(xml_antes),
      "mid %d→%d · xml %d→%d" % (len(mid_antes), len(mid_depois), len(xml_antes), len(xml_depois)))
checa("F  o roll do .mid acompanha a gravura, não os dados",
      m2["n_arquivo"] < m["n_arquivo"] and m2["n_na_partitura"] == m["n_na_partitura"],
      "%d→%d notas no arquivo · %d na partitura" % (m["n_arquivo"], m2["n_arquivo"], m2["n_na_partitura"]))
# e o essencial: ocultar pistas não pode fabricar erro. Com `hide_lanes` ativo (linha acima), a
# auditoria full dava 2 erros falsos (T11/T12 comparando arquivo filtrado com dados completos).
qaf = cl.post("/api/qa", json={"id": rid, "level": "full"}).get_json()
tal = ((qaf.get("qa") or {}).get("tally") or {})
checa("F  ocultar pistas não cria erro de auditoria", bool(qaf.get("ok")) and int(tal.get("error") or 0) == 0,
      json.dumps(tal))
checa("F  a auditoria declara o que a apresentação filtrou",
      any(x.get("check") == "T10p" for x in ((qaf.get("qa") or {}).get("findings") or [])),
      str([x.get("check") for x in ((qaf.get("qa") or {}).get("findings") or []) if x.get("stage") == "export"]))
dig = (qa.get("qa") or {}).get("score_digest") or {}
checa("F  a auditoria julga tudo (não só o visível)",
      bool(qa.get("ok")) and int(dig.get("n_hits") or -1) == n_dados,
      "digest n_hits=%s · dados=%d" % (dig.get("n_hits"), n_dados))
checa("F  `simples` é o atalho para as pistas de prato", 
      cl.post("/api/view", json={"id": rid, "simples": True}).get_json()["score"]["hide_lanes"] == list(PRATOS))
checa("F  pista inexistente falha alto, com a lista válida",
      cl.post("/api/view", json={"id": rid, "hide_lanes": ["bumbo_dourado"]}).status_code == 400)
volt = cl.post("/api/view", json={"id": rid, "hide_lanes": []}).get_json()
checa("F  mostrar de novo restaura a gravura", len(volt["svg"]) > len(v["svg"]) * 1.05,
      "%d → %d" % (len(v["svg"]), len(volt["svg"])))

# ------------------------------------------------------------------- bordas e sintonia
bad = cl.get("/api/wave/nao-existe?t0=0&t1=1")
checa("rid desconhecido responde 404 com explicação", bad.status_code == 404 and bad.get_json()["codigo"] == "sem_analise")
bad2 = cl.get("/api/wave/%s?t0=abc" % rid)
checa("parâmetro inválido responde 400", bad2.status_code == 400)
fora = cl.get("/api/wave/%s?t0=%s&t1=%s" % (rid, dur + 5, dur + 9)).get_json()
checa("janela fora do fim da faixa não explode", fora["ok"] and fora["n"] >= 1,
      "janela %.2f–%.2f · n=%d" % (fora["t0"], fora["t1"], fora["n"]))
tg = cl.post("/api/tune", json={"id": rid}, query_string={})
tt2 = tg.get_json()
checa("tune roda pelo servidor e recomenda", tt2.get("ok") and tt2["tune"]["recomendado"],
      json.dumps(tt2.get("tune", {}).get("recomendado")))
checa("tune cacheado (segunda chamada não revarre)",
      cl.post("/api/tune", json={"id": rid}).get_json().get("cache") is True)

print("\n  (%d ok, %d falha%s)" % (len(ok), len(falhas), "" if len(falhas) == 1 else "s"))
sys.exit(1 if falhas else 0)
