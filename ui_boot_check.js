/* Reproduz o sintoma "a página fica carregando para sempre" contra o app.js real.
 *
 * O navegador não está disponível aqui, então o teste carrega static/app.js num VM do Node com
 * um DOM mínimo e um `fetch` encenado. As quatro cenas abaixo são os modos de falha que o
 * usuário vê quando o servidor não está de pé — e o que a interface deve fazer em cada um:
 *   A  conexão recusada        → faixa "não consigo falar com o servidor", sem spinner preso
 *   B  servidor no ar,deps fora → faixa dizendo QUAIS bibliotecas faltam
 *   C  servidor saudável        → faixa Some, /api/lanes alimenta a legenda
 *   D  servidor pendurado       → o prazo estoura e vira erro legível (não espera infinita)
 * Mais: o handler de rejeição solta o spinner, e a UI é montada ANTES de qualquer rede.
 *
 *   node tests/ui_boot_check.js
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const SRC = fs.readFileSync(path.join(__dirname, "..", "static", "app.js"), "utf8");
let problemas = 0;
const ok = (rot, cond, extra) => {
  console.log("  " + (cond ? "\x1b[32mok\x1b[0m  " : "\x1b[31mFALHA\x1b[0m ") + rot +
              (extra ? "  · " + extra : ""));
  if (!cond) problemas++;
};

function novoDOM() {
  const el = (id) => ({
    id, hidden: true, textContent: "", value: "", checked: false, disabled: false,
    className: "", scrollTop: 0, scrollWidth: 0, innerHTML: "", dataset: {}, style: {},
    _on: {}, files: [],
    addEventListener(t, fn) { (this._on[t] = this._on[t] || []).push(fn); },
    appendChild() {}, remove() {}, click() {}, focus() {}, setPointerCapture() {},
    querySelectorAll() { return []; }, querySelector() { return null; },
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 20 }; },
    setAttribute() {}, getAttribute() { return null; }, add() {},
  });
  const reg = new Map();
  const win = { _on: {}, addEventListener(t, fn) { (this._on[t] = this._on[t] || []).push(fn); } };
  const doc = {
    getElementById(id) { if (!reg.has(id)) reg.set(id, el(id)); return reg.get(id); },
    querySelectorAll() { return []; }, querySelector() { return null; },
    createElement(t) { return el("created:" + t); },
    createTextNode() { return {}; },
    addEventListener() {}, body: el("body"), documentElement: el("html"),
    registerEventListener() {},
  };
  return { doc, win, reg };
}

// fetch encenado: `responde(ctx)` devolve {ok,status,json} | lança | nunca resolve
function contexto(cena) {
  const { doc, win, reg } = novoDOM();
  let pedidos = [];
  const ctx = {
    document: doc, window: win, navigator: { userAgent: "node", language: "pt-BR" },
    console, setTimeout, clearTimeout, setInterval: () => 0, clearInterval: () => {},
    URL: { createObjectURL: () => "blob:x", revokeObjectURL: () => {} },
    Blob: function () {}, AbortController, DOMException, Date, Math, JSON, Number, String,
    Array, Object, Promise, Error, RegExp, Intl: { DateTimeFormat: function () { return { format: () => "" }; } },
    requestAnimationFrame: () => 0, cancelAnimationFrame: () => {},
    Image: function () {}, localStorage: { getItem: () => null, setItem: () => {} },
    fetch: async (url, opt) => {
      pedidos.push(url);
      if (opt && opt.signal) opt.signal.addEventListener("abort", () => {});
      return cena(url, opt, pedidos);
    },
    _pedidos: () => pedidos, _reg: reg, _win: win,
  };
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  // o script termina chamando boot(); trocamos por uma promessa guardada para poder esperá-la
  vm.runInContext(SRC.replace(/\nboot\(\);\s*$/, "\nglobalThis._boot = boot();"), ctx,
                  { filename: "static/app.js" });
  if (!ctx._boot) throw new Error("o app.js não chama boot() no fim — o teste precisa ser revisto");
  return ctx;
}
const corpo = (ctx, cod) => vm.runInContext(cod, ctx);
// o DOM falso cria elementos sob demanda: "nunca tocado" == "não está travado na tela"
const livre = (ctx, id) => !ctx._reg.has(id) || ctx._reg.get(id).hidden === true;
const RES = { ok: true, status: 200 };

(async () => {
  console.log("== cena A: conexão recusada (servidor não está rodando) ==");
  let ctx = contexto(async () => { throw new TypeError("Failed to fetch"); });
  await corpo(ctx, "_boot");
  ok("a UI foi montada antes da rede (listeners registrados)",
     (ctx._reg.get("offline-retry")._on.click || []).length > 0);
  ok("faixa de servidor fora do ar aparece", ctx._reg.get("offline").hidden === false);
  ok("o texto diz o que fazer",
     /rode \.\/run\.sh/.test(ctx._reg.get("offline-why").textContent),
     ctx._reg.get("offline-why").textContent);
  ok("nenhum spinner preso", livre(ctx, "busy"));
  ok("estado não depende do servidor (peças de reserva)", corpo(ctx, "state.lanes.length") === 8);
  ok("nenhum pedido além da sonda", ctx._pedidos().length === 1, ctx._pedidos().join(" "));

  console.log("\n== cena B: servidor no ar, bibliotecas ausentes ==");
  ctx = contexto(async (url) => {
    if (url.indexOf("health") >= 0) {
      return Object.assign({}, RES, { headers: { get: () => "application/json" },
        json: async () => Object.assign({}, RES, { faltando: ["reportlab", "miniaudio"] }) });
    }
    return Object.assign({}, RES, { headers: { get: () => "application/json" },
                                    json: async () => ({ ok: true, lanes: [] }) });
  });
  await corpo(ctx, "_boot");
  ok("faixa mostra QUAIS dependências faltam",
     ctx._reg.get("offline").hidden === false &&
     /reportlab/.test(ctx._reg.get("offline-why").textContent) &&
     /requirements\.txt/.test(ctx._reg.get("offline-why").textContent),
     ctx._reg.get("offline-why").textContent);

  console.log("\n== cena C: servidor saudável ==");
  const LANES9 = Array.from({ length: 9 }, (_, i) => ({ id: "l" + i, name: "P" + i }));
  ctx = contexto(async (url) => {
    const body = url.indexOf("health") >= 0
      ? Object.assign({}, RES, { ok: true, faltando: [], deps: {}, demo: true })
      : { ok: true, lanes: LANES9 };
    return Object.assign({}, RES, { headers: { get: () => "application/json" },
                                    json: async () => body });
  });
  await corpo(ctx, "_boot");
  ok("faixa sumiu", ctx._reg.get("offline").hidden === true);
  ok("as peças vieram do servidor", corpo(ctx, "state.lanes.length") === 9);
  ok("sonda + lista de peças na ordem certa",
     ctx._pedidos()[0] === "api/health" && ctx._pedidos()[1] === "api/lanes",
     ctx._pedidos().join(" "));

  console.log("\n== cena D: servidor pendurado (aceita a conexão e não responde) ==");
  ctx = contexto((url, opt) => new Promise((_res, rej) => {
    if (opt && opt.signal) opt.signal.addEventListener("abort", () => rej(new Error("aborted")));
  }));
  const t0 = Date.now();
  await corpo(ctx, "_boot");
  const dur = Date.now() - t0;
  ok("o prazo da sonda estourou em vez de esperar para sempre", dur < 15000 && dur >= 3000,
     (dur / 1000).toFixed(1) + " s");
  ok("a mensagem diz que o servidor não respondeu",
     ctx._reg.get("offline").hidden === false &&
     /sem resposta do servidor em 4 s/.test(ctx._reg.get("offline-why").textContent),
     ctx._reg.get("offline-why").textContent);
  ok("sem spinner preso", livre(ctx, "busy"));

  console.log("\n== rede de segurança: rejeição solta o spinner ==");
  ctx = contexto(async () => { throw new TypeError("Failed to fetch"); });
  await corpo(ctx, "_boot");
  corpo(ctx, 'busy(true, "analisando…")');
  ok("spinner ligado pela simulação", ctx._reg.get("busy").hidden === false,
     "busy existe e está visível");
  const handler = (ctx._win._on.unhandledrejection || [])[0];
  ok("o handler está registrado na janela", typeof handler === "function");
  if (handler) {
    handler({ reason: new Error("estouro de pilha no oráculo") });
    ok("rejeição solta o spinner", ctx._reg.get("busy").hidden === true);
    ok("rejeição vira mensagem no status",
       /estouro de pilha no or/.test(ctx._reg.get("status").textContent),
       ctx._reg.get("status").textContent);
  }

  console.log("\n== o arquivo não tem URL absoluta (o iframe do preview não tem rede externa) ==");
  const abs = (SRC.match(/https?:\/\/[^\s"')]+/g) || []).filter((u) => !/^https?:\/\/localhost/.test(u));
  // URIs de namespace (svg, xml, xlink) são identificadores: não geram pedido de rede nenhum
  const NS = /^https?:\/\/www\.w3\.org\/(2000|2001|2003|1999|xlink)/;
  const rede = abs.filter((u) => !NS.test(u));
  ok("nenhum recurso externo a buscar na rede (o iframe do preview não tem internet)",
     rede.length === 0, rede.length ? rede.slice(0, 3).join(" ")
       : (abs.length ? abs.length + " URI(s) de namespace, inofensivos: " + abs[0] : "nenhum"));

  console.log("\n== cena E: stylesheet velho em cache (o caso que deixou a tela presa) ==");
  // o que a folha antiga produz: #busy com o atributo `hidden` presente E o véu pintado na tela.
  ctx = contexto(async () => { throw new TypeError("Failed to fetch"); });
  await corpo(ctx, "_boot");
  ok("boot() força display:none no véu por estilo inline (independe do CSS)",
     corpo(ctx, '$("busy").style.display') === "none",
     "display=" + corpo(ctx, '$("busy").style.display'));
  corpo(ctx, '$("busy").style.display = "grid"');            // simula o cache velho pintando o véu
  const vivo = corpo(ctx, '(() => { const b = $("busy"); return !b.hidden || b.style.display !== "none"; })()');
  ok("o estado incoerente é detectável", vivo === true);
  corpo(ctx, "hideBusy()");
  ok("o botão ocultar funciona mesmo nesse estado",
     corpo(ctx, '$("busy").style.display') === "none" && corpo(ctx, '$("busy").hidden') === true);
  ok("o sinal de prontidão existe para o cão de guarda do documento",
     corpo(ctx, "window.__drumscribe_ready") === true);
  ok("busy(true) escreve display inline, não só o atributo",
     (corpo(ctx, '(() => { busy(true, "x"); return $("busy").style.display; })()')) === "grid");
  corpo(ctx, "busy(false)");
  ok("busy(false) devolve display:none", corpo(ctx, '$("busy").style.display') === "none");

  console.log("\n== cena F: cursor de reprodução sobre a partitura (matemática, sem DOM) ==");
  ctx = contexto(async () => { throw new TypeError("Failed to fetch"); });
  await corpo(ctx, "_boot");
  corpo(ctx, `
    state.score = { ticks_per_bar: 32, ticks_per_beat: 8, bpm: 100, meter: "4/4" };
    state.phase = 0.74; state.mode = "original";
    globalThis.MAPA = [
      { i: 0, page: 0, x0: 73, x1: 218, beat0: 78,  beat1: 213, y0: 90,  y1: 208 },
      { i: 1, page: 0, x0: 218, x1: 363, beat0: 223, beat1: 358, y0: 90,  y1: 208 },
      { i: 2, page: 1, x0: 46, x1: 191, beat0: 51,  beat1: 186, y0: 690, y1: 808 }];
  `);
  ok("duração do compasso vem do BPM e da métrica do desenho",
     Math.abs(corpo(ctx, "durBarro()") - 2.4) < 1e-9, corpo(ctx, "durBarro()") + " s");
  let r = corpo(ctx, "posicaoCursor(0, MAPA, durBarro(), 0)");
  ok("t=0 senta no batimento 1 do compasso 1",
     r && r.k === 0 && r.i === 0 && Math.abs(r.x - 78) < 1e-6, JSON.stringify(r));
  r = corpo(ctx, "posicaoCursor(2.4 * 1.5, MAPA, durBarro(), 0)");
  ok("meio do compasso 2 = meio do vão desenhado daquele compasso",
     r && r.i === 1 && Math.abs(r.fx - 0.5) < 1e-9 && Math.abs(r.x - 290.5) < 1e-6,
     JSON.stringify(r));
  r = corpo(ctx, "posicaoCursor(2.4 * 2 + 1.2, MAPA, durBarro(), 0)");
  ok("depois do último compasso o cursor encosta no fim (não salta pra fora)",
     r && r.i === 2 && r.x <= 186.0001 && r.x >= 51, JSON.stringify(r));
  ok("t<0 e mapa vazio devolvem nada (sem cursor, sem erro)",
     corpo(ctx, "posicaoCursor(-1, MAPA, 2.4, 0)") === null &&
     corpo(ctx, "posicaoCursor(3, [], 2.4, 0)") === null);
  ok("o compasso é achado na página 2 também",
     corpo(ctx, "posicaoCursor(2.4 * 2, MAPA, durBarro(), 0).g.y0") === 690);
  ok("a fase do arquivo desloca o compasso 1",
     Math.abs(corpo(ctx, "posicaoCursor(0.74, MAPA, durBarro(), 0.74).x") - 78) < 1e-6 &&
     Math.abs(corpo(ctx, "posicaoCursor(0.74, MAPA, durBarro(), 0).k") - 0) === 0);
  ok("na transcrição sintetizada a base é zero, no arquivo é a fase medida",
     Math.abs(corpo(ctx, "faseAtual()") - 0.74) < 1e-9 &&
     (corpo(ctx, 'state.mode = "synth"; Math.abs(t0DoBaro(3) - 7.2)') < 1e-9) &&
     (corpo(ctx, 'state.mode = "original"; Math.abs(t0DoBaro(3) - 7.94)') < 1e-9),
     "sintético 3×2,4 = 7,20 s · arquivo = fase 0,74 + 7,20 s");
  corpo(ctx, `
    state.layout = { clock: { t0: -0.0067, bar_len: 2.4, fonte: "grade medida", n: 290,
                              spread_ms: 3.1, max_ms: 71.7 } };
    state._gradePara = null; state._gradeOrigem = undefined;
  `);
  ok("gradeBase() obedece ao relógio medido pelo servidor (não ao phase_ms)",
     Math.abs(corpo(ctx, "gradeBase().t0") + 0.0067) < 1e-9 &&
     Math.abs(corpo(ctx, "gradeBase().len") - 2.4) < 1e-9 &&
     corpo(ctx, "gradeBase().n") === 290,
     "t0 " + corpo(ctx, "gradeBase().t0") + " s · compasso " + corpo(ctx, "gradeBase().len") + " s");
  ok("o t0 do compasso 5 vem do relógio, batida a batida",
     Math.abs(corpo(ctx, "t0DoBaro(4)") - (-0.0067 + 9.6)) < 1e-6,
     "t0DoBaro(4) = " + Number(corpo(ctx, "t0DoBaro(4)")).toFixed(4) + " s");
  corpo(ctx, 'state.mode = "synth";');
  ok("na síntese a origem é zero e o compasso vem da grade do desenho",
     Math.abs(corpo(ctx, "gradeBase().t0")) < 1e-9 &&
     Math.abs(corpo(ctx, "t0DoBaro(2)") - 4.8) < 1e-9);
  corpo(ctx, 'state.mode = "original"; state.layout = { clock: null };');
  // 8 notas é o mínimo de evidência para a mediana (mesma guarda do auditor: `len(ok) < 8`)
  const SN = 'state.score = { bpm: 100, ticks_per_beat: 8, ticks_per_bar: 32, bars: [' +
    '{ index: 0, hits: [' + [0, 8, 16, 24].map((t) => '{ tick: ' + t + ', time: ' +
      (0.02 + t * 0.075).toFixed(3) + ' }').join(', ') + '] },' +
    '{ index: 1, hits: [' + [0, 8, 16, 24].map((t) => '{ tick: ' + t + ', time: ' +
      (2.42 + t * 0.075).toFixed(3) + ' }').join(', ') + '] }] };';
  const med = Number(corpo(ctx, `(() => {
       ${SN}
       state._gradePara = null; state._gradeOrigem = undefined;
       return gradeBase().t0; })()`));
  ok("sem relógio do servidor, a mediana local assume e acha a mesma origem",
     Math.abs(med - 0.02) < 1e-9 && corpo(ctx, "gradeBase().fonte") === "mediana local",
     "mediana = " + med + " s");
  ok("com pouca evidência (<8 notas medidas) a mediana local recusa, em vez de chutar",
     corpo(ctx, `(() => { state.score = { bpm: 100, bars: [{ index: 0, hits: [
        { tick: 0, time: 0.02 }, { tick: 8, time: 0.6 } ] }] };
       state._gradePara = null; state._gradeOrigem = undefined;
       return String(origemMedida()); })()`) === "null");
  const notaTxt = String(corpo(ctx, `
    state.layout = { clock: { t0: -0.0067, bar_len: 2.4, fonte: "grade medida", n: 290,
                              spread_ms: 3.1, max_ms: 71.7 } };
    state._gradePara = null; state._gradeOrigem = undefined;
    notaDaGrade();`));
  ok("a precisão da origem vai junto no título do leitor",
     /grade medida/.test(notaTxt) && /290 golpes/.test(notaTxt) && /±3\.1 ms/.test(notaTxt),
     notaTxt.slice(0, 96));

  ok("o clique na partitura busca pelo relógio medido, nunca pelo phase_ms",
     /gradeBase\(\)/.test(corpo(ctx, "seekNaPartitura")) &&
     !/state\.phase/.test(corpo(ctx, "seekNaPartitura")));
  ok("a faixa de loop na forma de onda segue o cursor (mesma origem)",
     !/state\.phase/.test(corpo(ctx, "drawWave")));
  ok("fimDeSistema reconhece a última caixa de cada linha (adjacência x1→x0)",
     corpo(ctx, `(() => {
        const m = [{i:0,page:0,x0:73,x1:219,y0:90,y1:208},{i:1,page:0,x0:219,x1:365,y0:90,y1:208},
                   {i:2,page:0,x0:365,x1:511,y0:90,y1:208},{i:3,page:1,x0:46,x1:192,y0:689,y1:803}];
        return [fimDeSistema(m,0), fimDeSistema(m,1), fimDeSistema(m,2), fimDeSistema(m,3)].join();
      })()`) === "false,false,true,true");
  ok("a faixa de destaque do compasso inclui a barra de fim de sistema",
     /fimDeSistema\(mapa, p\.i\)/.test(SRC));
  ok("o leitor escreve em português (rótulo 'compasso N · batida/M')",
     SRC.includes('"compasso "') && !/compassso|compasss/.test(SRC));
  ok("drawCursor sem SVG desenhado é no-op (não estoura antes da 1ª partitura)",
     corpo(ctx, "(() => { try { drawCursor(1.2); return true; } catch (e) { return String(e); } })()") === true);

  console.log("\n== cena G: o zoom da aba Áudio move a janela (e pede o envelope fino dela) ==");
  const pedidosWave = [];
  const respJson = (o) => Object.assign({}, RES, {
    headers: { get: () => "application/json" }, json: async () => o,
  });
  ctx = contexto(async (url) => {
    if (url.indexOf("wave/") >= 0) {
      const q = Object.fromEntries((url.split("?")[1] || "").split("&").filter(Boolean)
        .map((kv) => kv.split("=")));
      pedidosWave.push({ t0: +q.t0, t1: +q.t1, n: +q.n, curvas: q.curvas });
      const N = 120;
      const arr = (v) => Array.from({ length: N }, () => v);
      return respJson({ ok: true, n: N, dt: 0.001, t0: +q.t0, t1: +q.t1, bucket_sec: 0.001,
                       min: arr(-0.3), max: arr(0.6), curvas: { global: arr(0.4), low: arr(0.2),
                                                                 mid: arr(0.3), high: arr(0.5) },
                       curvas_topo: 0.5, baldes_por_ponto: 1, recortado_ms: 0, sr: 44100,
                       fps: 172.266, lead_ms: 0, eventos: [{ t: +q.t0 + 0.01, lane: "kick", conf: 0.8,
                         vel: 100, bar: 0, tick: 0, resid_ms: 1.2, st: "e" }],
                       eventos_total: 1, n_notas_janela: 1, n_notas_partitura: 40,
                       estados: { e: 1, m: 0, n: 0 }, limitado: false });
    }
    if (url.indexOf("health") >= 0) {
      return respJson(Object.assign({}, RES, { ok: true, faltando: [], deps: {}, demo: true }));
    }
    return respJson({ ok: true, lanes: [] });
  });
  await corpo(ctx, "_boot");
  // o DOM falso cria elementos com `checked:false`; no navegador o HTML traz as caixas marcadas
  corpo(ctx, `["w-fino","w-curvas","w-notas"].forEach((i) => { document.getElementById(i).checked = true; });`);
  corpo(ctx, `state.id = "tst"; state.duration = 40; state.wave_fine = { n: 40000,
      bucket_sec: 0.001, nomes: ["global","high","low","mid"], nc: 4, sr: 44100, fps: 172.266,
      lead_ms: 0 };
    state.wave = { min: [-0.2], max: [0.4], rms: [0.1], n: 1800, dt: 0.0222, sr: 44100 };
    state.score = { bpm: 100, ticks_per_beat: 8, ticks_per_bar: 32, bars: [
      { index: 0, hits: [{ time: 0.4, tick: 0, lane: "kick", velocity: 100 }] }] };
    state.layout = { clock: { t0: 0, bar_len: 2.4, fonte: "teste", n: 10, spread_ms: 2, max_ms: 5 },
                     bar_map: [] };
    state.hits = [{ time: 0.4, tick: 0, lane: "kick", bar: 0, confidence: 0.8 }];
    state.view = { a: null, b: null }; state.zoom = 1;
    drawWave(); refreshWave(true);`);
  await corpo(ctx, "new Promise((r) => setTimeout(r, 0))");
  ok("zoom 1 = janela inteira",
     corpo(ctx, "(() => { const j = janelaAtual(); return (j[0] === 0 && j[1] === 40) ? 'ok' : j; })()") === "ok");
  const antes = pedidosWave.length;
  corpo(ctx, "setZoom(8)");
  await corpo(ctx, "new Promise((r) => setTimeout(r, 0))");
  const w = pedidosWave[pedidosWave.length - 1] || {};
  ok("o zoom encolhe a janela (40 s / 8 = 5 s)",
     Math.abs(corpo(ctx, "janelaAtual()[1] - janelaAtual()[0]") - 5) < 0.02,
     String(corpo(ctx, "janelaAtual()[1] - janelaAtual()[0]")));
  ok("e o pedido leva a janela, não a faixa toda", pedidosWave.length > antes &&
     Math.abs((w.t1 - w.t0) - 5) < 0.05, JSON.stringify(w));
  ok("o envelope fino chegou ao desenho (344→120 pontos por linha)",
     corpo(ctx, "state.finoOk === true && state.fino.n === 120 && state.fino.curvas.global.length === 120"),
     String(corpo(ctx, "[state.finoOk, state.fino && state.fino.n]")));
  ok("desenhar com o canvas sem 2d não estoura",
     corpo(ctx, "(() => { try { desenhaOndas(); desenhaRoll(); return true; } catch (e) { return String(e); } })()") === true);
  const spanG = corpo(ctx, "janelaAtual()[1] - janelaAtual()[0]");
  const zoomG = corpo(ctx, "state.zoom");
  ok("a alavanca da barra e a janela ficam de acordo",
     Math.abs(40 / spanG - zoomG) < 0.05, "zoom " + zoomG + " · janela " + spanG + " s");
  corpo(ctx, "setZoom(1)");
  await corpo(ctx, "new Promise((r) => setTimeout(r, 0))");
  ok("voltar a 1 devolve a faixa toda",
     corpo(ctx, "(() => { const j = janelaAtual(); return j[0] === 0 && j[1] === 40; })()"));
  corpo(ctx, `setJanela(10, 10.5); state.sel = { a: 10.5, b: 10.2 }; aplicarSelecao();`);
  await corpo(ctx, "new Promise((r) => setTimeout(r, 0))");
  ok("a seleção (shift+arraste) vira janela ordenada",
     Math.abs(corpo(ctx, "janelaAtual()[0]") - 10.2) < 1e-6 &&
     Math.abs(corpo(ctx, "janelaAtual()[1]") - 10.5) < 1e-6,
     JSON.stringify(corpo(ctx, "janelaAtual()")));
  ok("rodar a roda do mouse aproxima em volta do ponteiro (a âncora não pula)",
     (() => {
       corpo(ctx, "setJanela(10, 20); zoomNaJanela(12, 0.5);");
       const j = corpo(ctx, "janelaAtual()");
       // t = 12 estava a 20% da tela; ao cortar a janela pela metade ela tem de continuar a 20%
       return Math.abs((j[1] - j[0]) - 5) < 1e-6 && Math.abs(j[0] - 11) < 1e-6;
     })(), JSON.stringify(corpo(ctx, "janelaAtual()")));
  ok("o ponteiro fica onde estava na tela",
     (() => {
       corpo(ctx, "setJanela(0, 40);");
       const antesX = 0.25 * 900;
       corpo(ctx, "zoomNaJanela(10, 0.5);");
       const j = corpo(ctx, "janelaAtual()");
       return Math.abs(j[0] - 5) < 1e-6;
     })(), JSON.stringify(corpo(ctx, "janelaAtual()")));
  ok("o botão “só este compasso” enquadra um compasso da grade medida",
     (() => {
       corpo(ctx, "state.offset = 5.0; state.view = {a:null,b:null}; enquadrarCompasso();");
       return Math.abs(corpo(ctx, "janelaAtual()[1] - janelaAtual()[0]") - 2.4) < 1e-6;
     })(), String(corpo(ctx, "janelaAtual()")));
  ok("com a faixa toda visível, “seguir” não fica reposicionando",
     corpo(ctx, "(() => { state.view = {a:null,b:null}; segueJanela(39); return janelaAtual()[1] === 40; })()"));
  corpo(ctx, `state.wave_fine = null; state.finoChave = ""; state.sel = null; setJanela(1, 2);
              refreshWave(true);`);
  await corpo(ctx, "new Promise((r) => setTimeout(r, 0))");
  ok("sem envelope fino: nenhuma rede é gasta e a falta é dita",
     ctx._reg.get("w-note").hidden === false &&
     /não gerou envelope fino/.test(ctx._reg.get("w-note").textContent),
     String(ctx._reg.get("w-note").textContent).slice(0, 70));

  console.log("\n== cena H: recusa da análise anunciada na própria folha (A22) ==");
  let nAn = 0;
  ctx = contexto(async (url) => {
    if (url.indexOf("health") >= 0) {
      return respJson({ ok: true, faltando: [], deps: {}, demo: true });
    }
    if (url.indexOf("analyze") >= 0) {
      nAn++;
      return { ok: false, status: 413,
               headers: { get: () => "application/json" },
               json: async () => ({ ok: false, codigo: "memoria",
                 error: "a faixa é grande demais para a memória desta máquina: o plano mais " +
                        "econômico ainda pede 1703 MB" }) };
    }
    if (url.indexOf("midi_view") >= 0) {
      return respJson({ ok: true, n_arquivo: 3, n_escritas_visiveis: 3, n_na_partitura: 3,
                        notas: [[0, 0.1, 36, 100, "kick"], [0.1, 0.1, 38, 90, "snare"],
                                [0.2, 0.1, 42, 70, "hat"]],
                        division: 480, tempo_us: 600000, bpm_arquivo: 100, bpm_partitura: 100,
                        bytes: 900, ocultas: [], truncado: false, limite: 6000, sem_off: 0,
                        lead_ms: 10980, marcadores: 2, pitches: [36, 42], por_pista: {} });
    }
    return respJson({ ok: true, lanes: [] });
  });
  ctx.FormData = function () { this.append = () => {}; };
  await corpo(ctx, "_boot");
  corpo(ctx, `state.id = "resultado-anterior"; state.file = { name: "musica-do-usuario.mp3" };`);
  await corpo(ctx, "analyze()");
  const folha = ctx._reg.get("score-erro"), porq = ctx._reg.get("score-erro-why");
  ok("a recusa aparece DENTRO da folha, não só na linha de status",
     !!folha && folha.hidden === false, String(folha && folha.hidden));
  ok("e diz o motivo com o número que o servidor deu",
     !!porq && /1703 MB/.test(porq.textContent), String(porq && porq.textContent).slice(0, 70));
  ok("a linha de status acompanha", /falha:/.test(ctx._reg.get("status").textContent),
     String(ctx._reg.get("status").textContent).slice(0, 40));
  ok("o resultado anterior não é destruído (continua consultável)",
     corpo(ctx, "state.id") === "resultado-anterior" && nAn === 1, "pedido " + nAn);
  ok("o aviso some quando um resultado é adotado",
     (() => { corpo(ctx, "falhaNaFolha(null)"); return ctx._reg.get("score-erro").hidden === true; })());
  corpo(ctx, `state.score = { hide_lanes: [], bpm: 100, bars: [{ index: 0, hits: [{ tick: 0 }] }] };
              state.duration = 40; state.hits = [];`);
  await corpo(ctx, "pedeMidi(true)");
  const nt = ctx._reg.get("m-note");
  ok("a aba MIDI diz quanto deslocar o .mid para bater com o áudio (intro silenciosa)",
     !!nt && /DAW/.test(nt.textContent) && /em 10980 ms/.test(nt.textContent),
     String(nt && nt.textContent).slice(-96));

  console.log("\n" + "=".repeat(60));
  console.log(problemas ? problemas + " problema(s)." : "boot robusto: nenhum problema.");
  process.exit(problemas ? 1 : 0);
})();
