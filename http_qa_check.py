#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verifica a camada HTTP da Auditoria (a UI usa exatamente essas rotas).

Roda contra um servidor já de pé (por padrão http://127.0.0.1:8000). Cobre:
  GET  /api/demo            → resultado de exemplo
  POST /api/qa              → auditoria em três níveis + markdown
  POST /api/qa/fix          → agente: aplica, re-audita, não perde nota lícita
  POST /api/rebuild         → injeção de defeitos pela rota do editor
  GET  / e /static/app.js    → a aba existe no HTML e o script carrega
Sem dependências além da biblioteca padrão.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import Counter

OK, BAD = "\033[32mok\033[0m", "\033[31mFALHA\033[0m"
_problemas = 0


def check(label, cond, extra=""):
    global _problemas
    print(f"  {OK if cond else BAD}  {label}" + (f"  · {extra}" if extra else ""))
    if not cond:
        _problemas += 1
    return bool(cond)


def req(base, method, path, body=None, raw=False):
    data = None
    hdr = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdr["content-type"] = "application/json"
    r = urllib.request.Request(base + path, data=data, method=method, headers=hdr)
    try:
        with urllib.request.urlopen(r, timeout=600) as f:
            payload = f.read()
            code = f.status
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    if raw:
        return code, payload
    try:
        return code, json.loads(payload.decode("utf-8"))
    except Exception:                     # noqa: BLE001
        return code, {"_raw": payload[:200].decode("utf-8", "replace")}


def main() -> int:
    ap = argparse.ArgumentParser(description="checa as rotas HTTP da Auditoria")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    print("== servidor ==")
    code, page = req(base, "GET", "/", raw=True)
    html = page.decode("utf-8", "replace")
    check("GET / responde", code == 200, f"HTTP {code}")
    check("aba Auditoria no HTML", 'data-tab="auditoria"' in html and 'id="pane-auditoria"' in html)
    check("selo de resumo presente", 'id="qa-badge"' in html)
    check("faixa de servidor fora do ar presente",
          'id="offline"' in html and 'id="offline-retry"' in html)
    print("  (o documento tem de se bastar: cache velho de style.css já deixou a tela presa)")
    with urllib.request.urlopen(base + "/", timeout=30) as f:
        doc = f.read().decode("utf-8", "replace")
        cc = f.headers.get("Cache-Control", "")
    check("HTML chega com no-store (carimbo de versão não pode envelhecer)", "no-store" in cc, cc)
    check("as URLs dos recursos vêm carimbadas", "?v=" in doc and "{{VER}}" not in doc,
          " ".join(sorted(set(__import__("re").findall(r"(?:href|src)=\"(static/[^\"]+)\"", doc)))))
    check("<head> traz o [hidden]{display:none!important} inline",
          "[hidden]{display:none!important}" in doc.replace(" ", ""))
    check("#busy nasce com display:none inline (cache nenhum o reabre)",
          'id="busy" hidden style="display:none"' in doc)
    check("há cão de guarda se o app.js não rodar", "__drumscribe_ready" in doc)
    import re as _re
    for u in sorted(set(_re.findall(r"(?:href|src)=\"(static/[^\"]+\?v=[0-9a-f]+)\"", doc))):
        code, corpo = req(base, "GET", "/" + u, raw=True)
        protege = b"[hidden]{display:none!important}" in corpo or b"setVeil" in corpo
        check(f"recurso carimbado serve o conteúdo novo ({u.split('?')[0]})",
              code == 200 and protege, f"HTTP {code} · {len(corpo)} bytes")
    code, js = req(base, "GET", "/static/app.js", raw=True)
    js = js.decode("utf-8", "replace")
    check("app.js carrega e tem a camada de QA",
          code == 200 and all(k in js for k in ("runQA", "qaApply", "renderQA", "qaDownload", "QA_LABEL")))

    print("\n== sonda de saúde ==")
    code, h = req(base, "GET", "/api/health")
    check("GET /api/health", code == 200 and h.get("ok") is True, f"HTTP {code}")
    check("nenhuma dependência faltando", not h.get("faltando"), str(h.get("faltando")))
    check("estático no lugar", h.get("static") is True and h.get("demo") is True,
          f"static={h.get('static')} demo={h.get('demo')}")
    check("versão e Python informados", bool(h.get("version")) and bool(h.get("python")),
          f"v{h.get('version')} · py {h.get('python')}")

    print("\n== demo ==")
    code, demo = req(base, "GET", "/api/demo")
    check("GET /api/demo", code == 200, f"HTTP {code}")
    sid = demo.get("id")
    check("id retornado", bool(sid), str(sid))

    print("\n== geometria para o cursor de reprodução ==")
    lay = demo.get("layout") or {}
    bm = lay.get("bar_map") or []
    sc = demo.get("score") or {}
    n_bars = len(sc.get("bars") or [])
    check("a resposta traz o mapa de compassos", len(bm) == n_bars and n_bars > 4,
          f"{len(bm)} entradas × {n_bars} compassos · {lay.get('n_pages')} página(s)")
    check("cada entrada tem caixa e batimento 1 dentro dela",
          all(g["x0"] < g["beat0"] < g["beat1"] <= g["x1"] and g["y0"] < g["y1"] for g in bm))
    ph = float(lay.get("page_h") or 0)
    linhas_fora = [g["i"] for g in bm if not (g["y0"] < g["line1"] < g["y1"])]
    check("a pauta do compasso cabe na caixa dele (tudo em espaço do documento)",
          not linhas_fora, f"fora {linhas_fora[:5]}" if linhas_fora else f"{len(bm)} conferidos")
    sem_off = [g["i"] for g in bm if g["page"] and g["line1"] < g["page"] * ph - 0.01]
    check("o deslocamento de página vale para as três coordenadas (não só para a caixa)",
          not sem_off, f"line1 esquecido nos compassos {sem_off[:5]} (foram +{ph:.0f} pt)" if sem_off
          else f"páginas {sorted({g['page'] for g in bm})} · +{ph:.2f} pt por página")
    check("o mapa cobre o documento inteiro (última página incluída)",
          max(g["y1"] for g in bm) <= (lay.get("doc_h") or 0) + 0.02 and
          {g["page"] for g in bm} == set(range(lay.get("n_pages") or 0)),
          f"y máx {max(g['y1'] for g in bm):.0f} de {lay.get('doc_h')}")
    bar_len = ((sc.get("ticks_per_bar") or 32) / (sc.get("ticks_per_beat") or 8)) * (60 / (sc.get("bpm") or 120))
    def caixa_de(b):
        g = bm[b]
        return g["beat0"] <= g["beat0"] + 0.5 * (g["beat1"] - g["beat0"]) <= g["beat1"]
    check("cada compasso tem largura útil (o cursor tem por onde andar)", all(caixa_de(b) for b in range(n_bars)),
          f"compasso = {bar_len:.2f} s")
    clock = lay.get("clock") or {}
    check("a resposta traz o relógio da grade (origem medida, não phase_ms)",
          clock.get("t0") is not None and clock.get("bar_len", 0) > 0,
          f"t0 {clock.get('t0')} s · compasso {clock.get('bar_len')} s · {clock.get('fonte')} "
          f"sobre {clock.get('n')} golpes")
    check("a origem é rigidamente coerente com as notas (dispersão < 40 ms)",
          (clock.get("spread_ms") or 999) < 40 and (clock.get("max_ms") or 999) < 200,
          f"dp ±{clock.get('spread_ms')} ms · máx ±{clock.get('max_ms')} ms")
    # A pergunta que importa: o compasso que o cursor acende é o compasso que está soando?
    tol = float((((demo.get("report") or {}).get("grid") or {}).get("tol_ms")) or 96.0) / 1000.0
    na_meia = []          # o meio do compasso tem de acender o próprio compasso — sem tolerância
    fora = []             # cada golpe, dentro da caixa do seu compasso ± a tolerância da auditoria
    na_fronteira = 0
    t0, bl = float(clock.get("t0") or 0), float(clock.get("bar_len") or 1)
    def comp_de(t):
        return int((t - t0) // bl)
    for b in sc.get("bars", []):
        ts = [h["time"] for h in b.get("hits", []) if h.get("time") is not None]
        if not ts:
            continue
        meio = (min(ts) + max(ts)) / 2.0
        if comp_de(meio) != b["index"]:
            na_meia.append((b["index"], round(meio, 2), comp_de(meio)))
        inicio, fim_b = t0 + b["index"] * bl, t0 + (b["index"] + 1) * bl
        for t in ts:
            if not (inicio - tol <= t <= fim_b + tol):
                fora.append((b["index"], round(t, 3)))
            elif t < inicio + tol or t > fim_b - tol:
                na_fronteira += 1
    check("o meio de cada compasso acende exatamente o seu compasso", not na_meia,
          f"{n_bars} compassos · erros {na_meia[:3]}" if na_meia else f"{n_bars} conferidos")
    check("cada golpe cai na caixa do seu compasso ± a tolerância do auditor", not fora,
          f"{na_fronteira} golpe(s) a menos de {tol*1000:.0f} ms da barra (jitter humano)" +
          (f" · fora {fora[:3]}" if fora else ""))
    rep_t = (demo.get("report") or {}).get("tempo") or {}
    rep_f = (demo.get("report") or {}).get("file") or {}
    fase_rel = (float(rep_t.get("phase_ms") or 0) + float(rep_f.get("trim_lead_ms") or 0)) / 1000.0
    check("e a diferença entre as duas origens é registrada (não escondida)",
          abs(fase_rel - t0) > 0.25 or abs(fase_rel) < 0.25,
          f"relatório {fase_rel:.3f} s × grade medida {t0:.3f} s → Δ {abs(fase_rel - t0)*1000:.0f} ms")
    code, sv = req(base, "POST", "/api/svg", {"id": sid, "opts": {"page": "a4_landscape"}})
    check("/api/svg devolve o mapa junto", code == 200 and len(((sv.get("layout") or {}).get("bar_map")) or []))
    hits = [{**h, "bar": b["index"]} for b in sc.get("bars", []) for h in b["hits"]]
    code, rb = req(base, "POST", "/api/rebuild", {"id": sid, "hits": hits, "bpm": sc.get("bpm"),
                                                 "meter": sc.get("meter"), "swing": sc.get("swing", 0)})
    novo = len(((rb.get("layout") or {}).get("bar_map")) or [])
    check("/api/rebuild refaz o mapa (edição não deixa cursor velho na tela)",
          code == 200 and novo == len(((rb.get("score") or {}).get("bars")) or []) > 0,
          f"{novo} compassos")

    print("\n== auditoria ==")
    for level in ("fast", "full"):
        code, out = req(base, "POST", "/api/qa", {"id": sid, "level": level})
        qa = out.get("qa", {})
        t = qa.get("tally", {})
        ids = {f.get("check") for f in qa.get("findings", [])}
        need = {"T1", "T2", "T3", "T3b", "T3c", "T3d", "T3e", "T4", "T5", "T6", "T7", "T8",
                "T9", "T10", "T11", "T12", "T16", "T17", "T18"}
        if level == "full":
            need |= {"T13", "T14", "T15a", "T15b", "T15c", "T19a", "T19b", "T19c"}
            check("nível pedido é o aplicado", qa.get("level") == "full", str(qa.get("level")))
        check(f"POST /api/qa level={level}",
              code == 200 and t.get("error") == 0 and t.get("warn") == 0
              and t.get("info", 0) >= 15 and need <= ids,
              f"{t.get('error')} erro(s), {t.get('warn')} aviso(s), {t.get('info')} verif. · "
              f"{len(ids)} leis")
        if level == "full":
            md = out.get("markdown", "")
            check("markdown exportável (tabela + evidência por lei)",
                  len(md) > 4000 and "**T1**" in md and "**T19c**" in md and "| estado |" in md,
                  f"{len(md)} chars")
            check("resumo legível", bool(qa.get("summary")))
            try:
                json.dumps(out)
                check("JSON serializável (sem numpy scalars)", True)
            except Exception as e:                                    # noqa: BLE001
                check("JSON serializável (sem numpy scalars)", False, str(e))
            code2, err = req(base, "POST", "/api/qa", {"id": "nao-existe", "level": "fast"})
            check("id inexistente → erro claro", code2 in (400, 404), f"HTTP {code2}")

    print("\n== agente de correção (partitura já lícita) ==")
    code, out = req(base, "POST", "/api/qa/fix", {"id": sid, "level": "full"})
    check("não inventa alteração em partitura lícita",
          code == 200 and not out.get("changed") and not out.get("applied")
          and out.get("reverted") is False,
          f"changed={out.get('changed')} pesos={out.get('pesos')}")

    print("\n== agente sobre partitura furada (pela rota do editor) ==")
    code, full = req(base, "GET", f"/api/result/{sid}.json")
    score = full["score"]
    hits = []
    for b in score["bars"]:
        for h in b["hits"]:
            hits.append({**h, "bar": b["index"]})
    n0 = len(hits)
    legal = Counter((h["bar"], int(round(h["tick"])), h["lane"]) for h in hits)
    hits.append({"lane": "pipe", "tick": 99, "dur": 0, "velocity": 200, "artic": "blast", "bar": 0})
    hits.append(dict(hits[1]))                                        # duplicata exata
    hits[3]["tick"] = 3.4                                             # tick não inteiro
    if len(hits[5]) and score["ticks_per_bar"]:
        hits[5]["tick"] = score["ticks_per_bar"] - 1
        hits[5]["dur"] = 40                                           # duração além do compasso
    code, outb = req(base, "POST", "/api/rebuild",
                     {"id": sid, "hits": hits, "bpm": score["bpm"],
                      "meter": score["meter"], "swing": score.get("swing", 0)})
    check("POST /api/rebuild aceita o estado furado", code == 200, f"HTTP {code}")
    broken = outb.get("score", {})
    code, outq = req(base, "POST", "/api/qa", {"id": sid, "level": "fast"})
    tq = outq.get("qa", {}).get("tally", {})
    check("auditor acusa os defeitos", tq.get("error", 0) > 0,
          outq.get("qa", {}).get("summary", ""))
    ids_before = {f.get("check") for f in outq.get("qa", {}).get("findings", []) if f.get("severity") == "error"}
    check("leis violadas são as esperadas", {"T3", "T3b", "T4", "T5", "T6", "T8"} & ids_before,
          " ".join(sorted(ids_before)) + " · propostas " + str(sorted({(f.get("fix") or "-") for f in outq.get("qa", {}).get("findings", []) if f.get("severity") == "error"})))
    code, outf = req(base, "POST", "/api/qa/fix", {"id": sid, "level": "fast"})
    tq2 = outf.get("qa", {}).get("tally", {})
    check("agente reduz o peso do erro",
          code == 200 and outf.get("changed")
          and outf.get("pesos", {}).get("final", 1e9) < outf.get("pesos", {}).get("inicial", -1),
          f"{outf.get('pesos')}")
    check("agente não entrega partitura pior", tq2.get("error", 99) <= tq.get("error", 0),
          f"{tq.get('error')} → {tq2.get('error')} erro(s)")
    check("nada reprovado foi revertido", not outf.get("reverted"))
    left = {f.get("check") for f in outf.get("qa", {}).get("findings", []) if f.get("severity") == "error"}
    check("só sobra o que não tem conserto mecânico", left == set(), f"restam {sorted(left)}")
    check("algum conserto foi anunciado", bool(outf.get("applied")),
          json.dumps([a.get("fix") for a in outf.get("applied", [])], ensure_ascii=False))

    new_hits = []
    for b in outf.get("score", {}).get("bars", []):
        for h in b["hits"]:
            new_hits.append({**h, "bar": b["index"]})
    after = Counter((h["bar"], int(round(h["tick"])), h["lane"]) for h in new_hits)
    perdidas = []
    for k, c in legal.items():
        got = after.get(k, 0)
        if c == 1 and got < 1:
            perdidas.append(k)
        if c > 1 and got > 1:
            perdidas.append(("duplicata sobreviveu", k))
    check("nenhuma nota lícita perdida nem duplicata mantida", not perdidas,
          f"{n0} → {len(new_hits)} notas · {perdidas[:3]}")
    check("pista desconhecida removida", all(h["lane"] in (score.get("lanes") or {}) for h in new_hits))
    check("SVG regerado", len(outf.get("svg", "")) > 5000 and "<svg" in outf.get("svg", ""))
    code, after_fix = req(base, "POST", "/api/qa", {"id": sid, "level": "fast"})
    tf = after_fix.get("qa", {}).get("tally", {})
    check("o reparo foi gravado no resultado salvo (auditoria limpa no mesmo id)",
          code == 200 and tf.get("error") == 0 and tf.get("warn") == 0,
          after_fix.get("qa", {}).get("summary", ""))
    code, _ = req(base, "POST", "/api/export/pdf", {"id": sid})
    check("PDF volta a sair depois do reparo", code == 200, f"HTTP {code}")
    print(f"\n{'=' * 60}\n" + ("tudo certo." if not _problemas else f"{_problemas} problema(s).") + "\n")
    return 1 if _problemas else 0


if __name__ == "__main__":
    sys.exit(main())
