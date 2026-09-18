#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invariante de CSS/DOM — a classe de bug que prende a interface inteira.

Aprendido da pior forma: `.busy{display:grid}` numa folha do autor **anula** o `[hidden]` do
navegador (mesma especificidade, mas a folha do autor ganha), então o véu de "analisando…" fica
pintado para sempre, `position:fixed;inset:0;z-index:60` cobrindo tudo e engolindo clique. O
`doublecheck` do áudio não vê isso; só uma regra estática sobre HTML+CSS+JS vê.

Regras:
 1. todo elemento escondido pelo **atributo** `hidden` (na marca ou via `$("id").hidden = …`) não
    pode ter classe que declare `display:` sem um par `.classe[hidden]` ou a regra global
    `[hidden]{…!important}`;
 2. o mesmo para as abas: `.tabpane` é alternada pela **classe** `hidden`, que precisa existir;
 3. todo `data-tab="x"` tem de ter `#pane-x` e vice-versa (aba morta = usuário perdido);
 4. toda função que liga o véu (`busy(true`) tem de desligá-lo (`busy(false` ou `hideBusy(`);
 5. o véu precisa de via de escape (botão + Esc + clique no fundo);
 6. chaves do CSS balanceadas e nenhum `position:fixed` com `z-index` alto sem proteção de `hidden`.

    python3 tests/ui_css_check.py
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "static")
_problemas = 0


def check(rot, cond, extra=""):
    global _problemas
    print(f"  {'\033[32mok\033[0m  ' if cond else '\033[31mFALHA\033[0m '}{rot}" + (f"  · {extra}" if extra else ""))
    if not cond:
        _problemas += 0 if cond else 1
    return bool(cond)


def ler(nome):
    with open(os.path.join(STATIC, nome), encoding="utf-8") as fh:
        return fh.read()


def regras_css(css):
    """[(seletores, corpo)] — suficiente para as perguntas que fazemos.

    Comentários são riscados antes: `/* texto */\n[hidden]{…}` faria o seletor da regra ser lido
    como "/* texto */ [hidden]", e o teste deixaria de ver a proteção que existe (falso negativo).
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out = []
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        sel, corpo = m.group(1).strip(), m.group(2)
        if sel.startswith("@"):
            continue
        out.append((sel, corpo))
    return out


def main() -> int:
    html = ler("index.html")
    css = ler("style.css")
    js = ler("app.js")
    regras = regras_css(css)

    print("== atributo `hidden` × classes que declaram display ==")
    def protege_atributo():
        """(regra global do atributo, classes com `.x[hidden]`) — o que vale para `hidden` booleano."""
        global_ok, por_classe = False, set()
        for sel, corpo in regras:
            for part in sel.split(","):
                part = part.strip()
                if re.fullmatch(r"\[hidden\]", part):
                    if "!important" in corpo and "display:none" in corpo.replace(" ", ""):
                        global_ok = True
                m = re.fullmatch(r"\.(?P<c>[\w-]+)\[hidden\]", part)
                if m:
                    por_classe.add(m.group("c"))
        return global_ok, por_classe

    def utilitario_classe_hidden():
        for sel, corpo in regras:
            if any(p.strip() == ".hidden" for p in sel.split(",")):
                if "!important" in corpo and "display:none" in corpo.replace(" ", ""):
                    return True
        return False

    global_ok, proteg = protege_atributo()
    util_ok = utilitario_classe_hidden()
    check("existe [hidden]{display:none!important} (protege o atributo)", global_ok,
          "sem ela, qualquer classe com `display` vence o atributo do navegador")

    def exibe_display(classe):
        for sel, corpo in regras:
            for part in sel.split(","):
                if part.strip() == "." + classe and re.search(r"(^|;)\s*display\s*:", corpo):
                    return True
        return False

    def em_risco(classes, via):
        """via = "attr": vale o atributo;  via = "classe": vale o utilitário .hidden"""
        ruim = []
        for c in classes:
            if not exibe_display(c):
                continue
            if via == "classe" and util_ok:
                continue
            if via == "attr" and (global_ok or c in proteg):
                continue
            if c in ("hidden", "tabpane") and via == "classe":
                continue
            if c == "hidden":                       # é o próprio utilitário, não corre nada
                continue
            ruim.append(c)
        return ruim

    def protegido_inline(eid):
        """`style="display:none"` no próprio elemento: nenhuma folha externa, velha ou nova, vence."""
        m2 = re.search(r'<[^>]*id="%s"[^>]*>' % re.escape(eid), html)
        return bool(m2 and 'style="display:none"' in m2.group(0))

    # 1a) elementos escondidos pelo ATRIBUTO na marca; os que só usam a CLASSE `hidden` são da aba
    alvos_attr, alvos_classe = [], []
    for m in re.finditer(r"<(\w+)([^>]*\bhidden\b[^>]*)>", html):
        attrs = m.group(2)
        if re.search(r"\bhidden(?!=)", attrs):
            alvo_attr = True
        else:
            alvo_attr = False
        cls = re.search(r'class="([^"]+)"', attrs)
        idm = re.search(r'id="([^"]+)"', attrs)
        eid, classes = (idm.group(1) if idm else "?"), (cls.group(1).split() if cls else [])
        if alvo_attr:
            alvos_attr.append((eid, classes))
        elif "hidden" in classes:
            alvos_classe.append((eid, classes))
    viol = [f"#{eid}.{c}" for eid, classes in alvos_attr if not protegido_inline(eid)
            for c in em_risco(classes, "attr")]
    viol_c = [f"#{eid}.{c}" for eid, classes in alvos_classe for c in em_risco(classes, "classe")]
    check("nenhum elemento com o atributo `hidden` fica à vista", not viol,
          f"{len(alvos_attr)} por atributo + {len(alvos_classe)} por classe" +
          (f" · em risco {viol}" if viol else ""))

    # 1b) elementos que o JS alterna com `.hidden =` (atributo, sempre)
    corpos = re.split(r"\n(?=(?:async )?function )", js)
    def liga_hidden_no_js(c):
        return re.search(r"\.hidden\s*=|setVeil\(", c) is not None

    ids_js = set()
    for c in corpos:
        if liga_hidden_no_js(c):
            ids_js |= set(re.findall(r'\$\("([\w-]+)"\)', c))
    ids_js |= set(re.findall(r'setVeil\(\$\("([\w-]+)"\)', js))
    via_js = sorted(ids_js)

    viol2 = []
    for eid in via_js:
        m = re.search(r'id="%s"[^>]*class="([^"]+)"' % re.escape(eid), html) or \
            re.search(r'class="([^"]+)"[^>]*id="%s"' % re.escape(eid), html)
        if not m or protegido_inline(eid):
            continue
        viol2 += [f"#{eid}.{c}" for c in em_risco(m.group(1).split(), "attr")]
    check("nada escondido por JS fica à vista", not viol2,
          f"{len(via_js)} ids alternados no JS ({', '.join(via_js)})" +
          (f" · em risco {viol2}" if viol2 else ""))
    check("o utilitário .hidden{display:none!important} existe (as abas dependem dele)", util_ok)

    print("\n== o estado inicial não depende do stylesheet externo (cache!) ==")
    head = html.split("</head>")[0]
    check("o <head> traz o próprio [hidden]{display:none!important}",
          "[hidden]" in head and "display:none!important" in head.replace(" ", ""))
    check("o <head> traz o #busy{display:none} inicial",
          re.search(r"#busy\s*\{[^}]*display:\s*none", head) is not None)
    m = re.search(r'<div class="busy"[^>]*>', html)
    check("#busy nasce com display:none inline (nenhuma folha externa o vence)",
          m is not None and 'style="display:none"' in m.group(0), (m.group(0) if m else "ausente"))
    linkados = re.findall(r'<(?:link[^>]*href|script[^>]*src)="(static/[^"]+)"', html)
    sem_carimbo = [u for u in linkados if "?v=" not in u]
    check("todo recurso externo tem carimbo de versão", not sem_carimbo,
          " ".join(linkados) + (f" · sem carimbo {sem_carimbo}" if sem_carimbo else ""))
    check("o JS assume o véu por estilo inline, não por classe",
          'b.style.display = on ? "grid" : "none"' in js)
    check("hideBusy age mesmo com o estado incoerente (hidden=true e véu na tela)",
          'if (!b.hidden || b.style.display !== "none")' in js.replace("\r", ""))
    check("há cão de guarda no documento (app.js morto não prende a tela)",
          "__drumscribe_ready" in html and "b.style.display = \"none\"" in html)

    print("\n== abas ==")
    tabs = set(re.findall(r'data-tab="([\w-]+)"', html))
    panes = set(re.findall(r'id="pane-([\w-]+)"', html))
    check("toda aba tem painel e todo painel tem aba", tabs == panes,
          f"abas {sorted(tabs)}")
    check("a troca de abas é genérica (data-tab → #pane-<nome>)",
          'b.dataset.tab' in js and '"pane-" + ' in js)
    paineis = set(re.findall(r'id="([\w-]+)"[^>]*class="[^"]*\btabpane\b', html)) | \
              set(re.findall(r'class="[^"]*\btabpane\b[^"]*" id="([\w-]+)"', html))
    fora = sorted(p for p in paineis if not p.startswith("pane-"))
    check("todo painel segue a convenção pane-<aba>", paineis and not fora,
          f"{len(paineis)} painéis" + (f" · fora {fora}" if fora else ""))

    print("\n== o véu nunca pode ficar preso ==")
    corpo_funcao = re.split(r"\n(?=(?:async )?function )", js)
    presas = [c.split("\n")[0].strip() for c in corpo_funcao
              if re.search(r"busy\(\s*true", c) and not re.search(r"busy\(\s*false|hideBusy\(", c)]
    check("toda função que liga o véu o desliga", not presas, "; ".join(presas[:3]))
    check("o véu tem botão de sair", 'id="busy-hide"' in html and 'addEventListener("click", () => hideBusy())' in js)
    check("Esc também solta", 'ev.key === "Escape"' in js)
    check("clique no fundo do véu solta", "ev.target === veil" in js)
    check("o texto padrão não é um aviso eterno", "analisando…" not in html.split('<div class="busy"')[1][:200]
          or 'id="busy-msg"' in html, "dentro de #busy só o <p> mutável")

    print("\n== camadas fixas ==")
    css_bal = css.count("{") == css.count("}")
    check("chaves balanceadas", css_bal, f"{css.count('{')}/{css.count('}')}")
    altas = []
    for sel, corpo in regras:
        if "position:fixed" in corpo and re.search(r"z-index\s*:\s*([5-9]\d{2,}|\d{4,})", corpo):
            for part in sel.split(","):
                c = part.strip().lstrip(".")
                if c and exibe_display(c) and c not in proteg and not global_ok:
                    altas.append(part.strip())
    check("nenhuma camada full-screen sem proteção de oclusão", not altas, " ".join(altas[:3]))

    print("\n" + "=" * 60)
    print("invariantes de interface ok." if not _problemas else f"{_problemas} problema(s).")
    return 1 if _problemas else 0


if __name__ == "__main__":
    sys.exit(main())
