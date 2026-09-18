/* DrumScribe — interface de transcrição de bateria. Sem dependências externas. */
"use strict";

const API = (p) => "api/" + p;
const $ = (id) => document.getElementById(id);
const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

const LANE_COLORS = {
  kick: "#e8543f", snare: "#2f7bd6", rim: "#7a5cd0", hat: "#31a06a", hat_open: "#1f8f6a",
  hat_foot: "#8a8f98", ride: "#d99f2b", crash: "#c8541f", splash: "#b0701f",
  tom_hi: "#9c5fb0", tom_mid: "#7a4f96", tom_low: "#5b3f78", cowbell: "#a08b2b",
};

const state = {
  id: null, score: null, report: null, wave: null, detections: [], file: null,
  lanes: [], hits: [], zoom: 1, playing: false, src: null, ctx: null, buffer: null,
  startedAt: 0, offset: 0, raf: 0, mode: "original", loopBar: null, duration: 0,
  layout: null, curBar: -1,
  phase: 0, busy: false,
  /* janela de visualização partilhada entre a onda e o piano-roll (segundos) */
  view: { a: null, b: null }, zoomCentro: null,
  fino: null, finoChave: "", finoOk: null, finoPromise: null, finoDepois: null,
  sel: null, arrasto: null,
  mid: null, midChave: "", midPedindo: false,
  tune: null, tunePedindo: false,
  pres: { simples: false, pratos: true },
  wave_fine: null,
};
const ZOOM_MAX = 240, ONDA_H = 320;

/* ----------------------------------------------------------------- utilidades */
function setStatus(msg, kind) {
  const el = $("status");
  el.textContent = msg;
  el.className = "status" + (kind ? " " + kind : "");
}
/* O véu é um layer full-screen: quem o manda aparecer/sumir tem de ser dono do `display`, e não
   depender de `[hidden]` + folha de estilo externa (que pode estar em cache, e estava quando a
   plataforma inteira ficou inclicável). `hidden` vai junto por semântica/AC. */
function setVeil(b, on) {
  b.hidden = !on;
  b.style.display = on ? "grid" : "none";     // inline: só perde para outro inline, nunca para o cache
}

function busy(on, msg) {
  state.busy = on;
  const b = $("busy");
  if (b) setVeil(b, !!on);
  if (msg && $("busy-msg")) $("busy-msg").textContent = msg;
}

/* Via de escape: nenhuma operação demorada — legítima ou pendurada — pode prender o usuário atrás
   de um véu, e o botão funciona em qualquer estado (inclusive "escondido" mas visível). */
function hideBusy(msg) {
  const b = $("busy");
  if (!b) return;
  if (!b.hidden || b.style.display !== "none") {
    setVeil(b, false);
    state.busy = false;
    if (msg) setStatus(msg);
  }
}
function fmt(n, d) { return Number(n).toFixed(d === undefined ? 2 : d).replace(".", ","); }

/* Prazos. A falha de operação mais comum desta plataforma é o servidor não estar de pé
   (ou o `pip install` não ter rodado): sem prazo, o `fetch` pendurado deixa a interface
   girando para sempre, que é indistinguível de "a análise é lenta". Então toda chamada tem
   um teto explícito, proporcional ao trabalho — e o `boot()` não espera a rede para montar a UI. */
const TMO = { sonda: 4000, rapido: 10000, medio: 180000, pesado: 900000 };

async function fetchT(url, opt, ms) {
  const prazo = ms || TMO.rapido;
  const ctl = new AbortController();
  const rel = setTimeout(() => ctl.abort(), prazo);
  let r;
  try {
    r = await fetch(url, Object.assign({ signal: ctl.signal }, opt || {}));
  } catch (e) {
    clearTimeout(rel);
    if (ctl.signal.aborted) {
      throw new Error("sem resposta do servidor em " + Math.round(prazo / 1000) +
                      " s — ele pode ter caído (./run.sh; diagnóstico em /api/health)");
    }
    throw new Error("não consegui falar com o servidor — rode ./run.sh e recarregue");
  }
  clearTimeout(rel);
  return r;
}

async function jget(url, opt, ms) {
  const r = await fetchT(url, opt, ms);
  const ct = (r.headers.get("content-type") || "");
  if (!r.ok) {
    let msg = "erro " + r.status;
    try { const j = await r.json(); msg = j.error || msg; } catch (e) {}
    throw new Error(msg);
  }
  if (ct.indexOf("json") >= 0) return r.json();
  return r;
}

/* ---------------------------------------------------------------------- setup */
const LANES_FALLBACK = [{ id: "kick", name: "Bumbo" }, { id: "snare", name: "Caixa" },
  { id: "hat", name: "Chimel (closed)" }, { id: "ride", name: "Ride" },
  { id: "crash", name: "Prato (crash)" }, { id: "tom_hi", name: "Tom 1" },
  { id: "tom_mid", name: "Tom 2" }, { id: "tom_low", name: "Tom 3 / base" }];

async function boot() {
  const veu = $("busy");
  if (veu) setVeil(veu, false);   // nada de véu herdado de HTML/CSS: começa limpo
  window.__drumscribe_ready = true;
  wire();                       // a página tem de responder antes de qualquer rede
  state.lanes = LANES_FALLBACK.slice();
  const h = await probe();
  if (!h) return;                 // sem servidor não há o que perguntar; a faixa explica
  try {
    const r = await jget(API("lanes"), null, TMO.rapido);
    if (r && r.lanes && r.lanes.length) { state.lanes = r.lanes; renderLegend(); }
  } catch (e) { /* a sonda já avisou */ }
  setInterval(() => { if ($("offline") && !$("offline").hidden) probe(); }, 10000);
}

/* Sonda de saúde: revela o que falta (dependência ausente é bem diferente de porta errada) */
async function probe() {
  const b = $("offline"), why = $("offline-why");
  let j = null;
  try {
    j = await jget(API("health"), null, TMO.sonda);
  } catch (e) {
    if (why) why.textContent = (e.message || "sem resposta") + " · última tentativa " + new Date().toLocaleTimeString();
    if (b) b.hidden = false;
    return null;
  }
  if (b) {
    const falta = (j && j.faltando) || [];
    if (falta.length) {
      b.hidden = false;
      if (why) why.textContent = "servidor no ar, mas faltam bibliotecas: " + falta.join(", ") +
        " · rode python3 -m pip install -r requirements.txt";
    } else {
      b.hidden = true;
    }
  }
  return j;
}

function wire() {
  /* abas */
  $$("#steps .step").forEach((b) => b.addEventListener("click", () => {
    $$("#steps .step").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $$(".tabpane").forEach((p) => p.classList.add("hidden"));
    $("pane-" + b.dataset.tab).classList.remove("hidden");
    if (b.dataset.tab === "audio") { drawWave(); refreshWave(); }
    if (b.dataset.tab === "midi") { desenhaRoll(); pedeMidi(false); }
    if (b.dataset.tab === "auditoria" && state.id && !state.qa) runQA(true);
  }));

  /* sair do véu de "analisando…" sem cancelar o que está rodando */
  const bh = $("busy-hide"); if (bh) bh.addEventListener("click", () => hideBusy());
  const veil = $("busy");
  if (veil) veil.addEventListener("click", (ev) => { if (ev.target === veil) hideBusy(); });
  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") hideBusy(); });

  /* cursor na partitura */
  const sw = $("svg-wrap") || $("score-wrap");
  if (sw) sw.addEventListener("click", (ev) => { if (!ev.target.closest || !ev.target.closest(".no-seek")) seekNaPartitura(ev); });

  /* servidor fora do ar */
  const or_ = $("offline-retry"); if (or_) or_.addEventListener("click", async () => {
    setStatus("testando o servidor…");
    const j = await probe();
    setStatus(j ? "servidor respondeu — pronto." : "ainda sem resposta do servidor.",
              j ? "ok" : "err");
  });

  /* auditoria */
  const qr = $("qa-run"); if (qr) qr.addEventListener("click", () => runQA(false));
  const qf = $("qa-fix"); if (qf) qf.addEventListener("click", () => qaApply());
  const qd = $("qa-dl"); if (qd) qd.addEventListener("click", () => qaDownload());
  const qfull = $("qa-full"); if (qfull) qfull.addEventListener("change", () => { if (state.qa_level) renderQA(); });

  /* arquivo */
  const drop = $("drop");
  $("file").addEventListener("change", (e) => takeFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => {
    e.preventDefault(); drop.classList.remove("over");
    if (ev === "drop" && e.dataTransfer.files.length) takeFile(e.dataTransfer.files[0]);
  }));

  $("btn-demo").addEventListener("click", demo);
  $("btn-run").addEventListener("click", analyze);
  $$("input,select", document.querySelector(".controls")).forEach((el) => {
    el.addEventListener("change", syncLabels);
    if (el.type === "range") el.addEventListener("input", syncLabels);
  });
  syncLabels();

  /* player */
  $("btn-play").addEventListener("click", togglePlay);
  $("btn-synth").addEventListener("click", () => playSynth());
  $("met").addEventListener("change", () => drawWave());
  $("vol").addEventListener("input", () => { if (state.gain) state.gain.gain.value = +$("vol").value; });
  document.addEventListener("keydown", (e) => {
    if (/input|select|textarea/i.test((e.target.tagName || ""))) return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  });

  /* zoom — a mesma alavanca move a largura da pauta e a janela do áudio/MIDI.
     Foi exatamente aqui que o zoom estava quebrado: ele só mexia no `width` do SVG, então na aba
     Áudio & batidas nada acontecia (a onda era desenhada na resolução fixa de 1800 baldes).
     Agora `state.zoom` define uma janela em segundos e `/api/wave` devolve o envelope fino dela. */
  $("z-in").addEventListener("click", () => setZoom(state.zoom * 1.35));
  $("z-out").addEventListener("click", () => setZoom(state.zoom / 1.35));
  $("fit").addEventListener("click", () => { setZoom(1); state.sel = null; });
  if ($("w-in")) $("w-in").addEventListener("click", () => { const j = janelaAtual(); zoomNaJanela((j[0] + j[1]) / 2, 1 / 1.6); });
  if ($("w-out")) $("w-out").addEventListener("click", () => { const j = janelaAtual(); zoomNaJanela((j[0] + j[1]) / 2, 1.6); });
  if ($("w-fit")) $("w-fit").addEventListener("click", () => { state.sel = null; setJanela(0, duracaoTotal()); state.zoom = 1; aplicaLarguraSVG(); });
  if ($("w-sel")) $("w-sel").addEventListener("click", aplicarSelecao);
  if ($("w-bar")) $("w-bar").addEventListener("click", enquadrarCompasso);
  ["w-fino", "w-curvas", "w-notas", "w-follow"].forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener("change", () => {
      if (id === "w-fino") { state.finoChave = ""; refreshWave(true); }
      else if (id === "w-curvas") refreshWave(true);
      else desenhaOndas();
    });
  });
  const cv = $("wave");
  if (cv) {
    cv.addEventListener("mousedown", onOndaDown);
    cv.addEventListener("wheel", onRoda, { passive: false });
    cv.addEventListener("dblclick", () => { const j = janelaAtual(); zoomNaJanela((j[0] + j[1]) / 2, 1 / 4); });
  }
  window.addEventListener("mousemove", onOndaMove);
  window.addEventListener("mouseup", onOndaUp);
  document.addEventListener("keydown", teclasZoom);
  const rl = $("roll");
  if (rl) rl.addEventListener("click", (ev) => {
    const m = state.mid, rb = rl.getBoundingClientRect();
    if (!m || !m.notas || !state._roll) return;
    const t = state._roll.x2t(ev.clientX - rb.left);
    let melhor = null, md = 1e9;
    m.notas.forEach((n) => { const d = Math.abs(n[0] - t); if (d < md) { md = d; melhor = n; } });
    if (!melhor || md > 0.25) return;
    state.offset = Math.max(0, melhor[0] - 0.02);
    if (state.playing) startPlay(state.offset);
    drawCursor(state.offset); drawWave();
    setStatus("nota do .mid em " + fmt(melhor[0], 3) + " s · altura " + melhor[2] +
              " · velocidade " + melhor[3] + " · duração " + fmt(melhor[1] * 1000, 0) + " ms.");
  });

  /* exportações */
  $$("#sec-export button").forEach((b) => b.addEventListener("click", () => exportAs(b.dataset.fmt)));

  /* editor */
  $("ed-add").addEventListener("click", addRow);
  $("ed-apply").addEventListener("click", rebuild);
  // o clique na onda passou por `onOndaUp` (precisa distinguir arrastar de clicar): ver wireZoom
  if ($("btn-tune")) $("btn-tune").addEventListener("click", pedirTune);
  if ($("pres-full")) $("pres-full").addEventListener("click", () => {
    state.pres.simples = false; atualizaPres(); aplicaView("partitura completa: tudo à mostra.");
  });
  if ($("pres-simple")) $("pres-simple").addEventListener("click", () => {
    state.pres.simples = true; atualizaPres();
    aplicaView("partitura simples: bumbo, caixa e tons — pratos ocultos na gravura.");
  });
  if ($("pres-pratos")) $("pres-pratos").addEventListener("change", () => {
    state.pres.pratos = $("pres-pratos").checked; atualizaPres();
    aplicaView(state.pres.pratos ? "pratos de volta à gravura." : "pratos ocultos na gravura (os dados seguem completos).");
  });
  if ($("m-refresh")) $("m-refresh").addEventListener("click", () => pedeMidi(true));
  if ($("m-dl")) $("m-dl").addEventListener("click", () => exportAs("mid"));
  window.addEventListener("resize", () => { drawWave(); desenhaRoll(); });
}

function syncLabels() {
  $("v-sens").textContent = fmt(+$("sens").value, 2);
  $("v-minc").textContent = fmt(+$("minc").value, 2);
  const db = +$("dbeat").value;
  $("v-dbeat").textContent = db === 0 ? "auto" : (db > 0 ? "+" : "") + db + " ticks";
}

function takeFile(f) {
  if (!f) return;
  state.file = f;
  $("fname").textContent = f.name + " · " + fmt(f.size / 1048576, 1) + " MB";
  $("btn-run").disabled = false;
  setStatus("arquivo pronto — pode ajustar a análise e clicar em transcrever.");
}

/* ------------------------------------------------------------------ análise */
async function demo() {
  busy(true, "analisando a faixa de demonstração…");
  try {
    const r = await jget(API("demo"), null, TMO.pesado);
    falhaNaFolha(null);
    adopt(r);
    setStatus("demonstração carregada (42 s, 100 BPM).", "ok");
  } catch (e) { setStatus("falha: " + e.message, "err"); }
  finally { busy(false); }
}

function falhaNaFolha(msg) {
  /* O estado "falhou" tem de aparecer onde o usuário olha (a folha), não só na linha de status:
     sem isto a tela mostra a partitura anterior como se fosse a nova e nada parece acontecer. */
  const bd = $("score-erro"), why = $("score-erro-why");
  if (!bd) return;
  if (!msg) { bd.hidden = true; return; }
  bd.hidden = false;
  if (why) why.textContent = " · " + msg;
}

async function analyze() {
  if (!state.file) { setStatus("escolha um arquivo primeiro.", "err"); return; }
  const fd = new FormData();
  fd.append("file", state.file, state.file.name);
  fd.append("params", JSON.stringify(currentParams()));
  busy(true, "decodificando áudio e detectando golpes…");
  setStatus("analisando…");
  try {
    const r = await jget(API("analyze"), { method: "POST", body: fd }, TMO.pesado);
    falhaNaFolha(null);
    adopt(r);
    /* O degrau automático de memória tem de aparecer na primeira linha que o usuário lê, não
       só dentro da aba de auditoria: análise feita a 22 050 Hz é outra análise. */
    const mem = (r.report && r.report.memory) || {};
    const f = (r.report && r.report.file) || {};
    setStatus("análise em " + fmt(r.report.total_sec || 0, 1) + " s · " +
              (r.score.bars || []).length + " compassos" +
              (mem.ajustado_por_memoria
                ? " · rebaixada a " + f.analysis_sr + " Hz para caber na memória (Nyquist " +
                  fmt((f.nyquist_hz || 0) / 1000, 1) + " kHz — ver avisos)"
                : "."),
              mem.ajustado_por_memoria ? "warn" : "ok");
  } catch (e) {
    setStatus("falha: " + e.message, "err");
    falhaNaFolha(e.message);
  } finally { busy(false); }
}

function currentParams() {
  return {
    sensitivity: +$("sens").value,
    min_confidence: +$("minc").value,
    grid_mode: $("grid").value,
    meter: $("meter").value,
    bpm_hint: +$("bpm").value || 0,
    swing: $("swing").value,
    downbeat_shift_ticks: +$("dbeat").value || 0,
    aux_lanes: $("aux").checked,
    fill_detect: $("fill").checked,
    // `simples` é o pedido; quem sabe o que é prato é o servidor (kit.PRATOS). O quadro
    // “pratos” da barra é um `hide_lanes` explícito e vale só até a próxima transcrição.
    simples: !!state.pres.simples,
  };
}

function adopt(r, opt) {
  opt = opt || {};
  state.layout = r.layout || null;
  state._gradePara = null; state._gradeOrigem = undefined;   // partitura nova, origem nova
  if (!opt.manterJanela) {
    state.view = { a: null, b: null }; state.zoomCentro = null; state.zoom = 1;
    state.fino = null; state.finoChave = ""; state.finoOk = null; state.finoPromise = null;
    state.finoDepois = null; state.sel = null;
    state.mid = null; state.midChave = ""; state.midPedindo = false;
  }
  if (r.wave_fine) state.wave_fine = r.wave_fine;
  state.id = r.id;
  state.score = r.score;
  state.report = r.report;
  state.wave = r.wave;
  state.audio_url = r.audio_url;
  state.hits = currentHits();
  state.duration = (r.report.file && r.report.file.duration_sec) || 0;
  const lead = (r.report.file && r.report.file.trim_lead_ms) || 0;
  state.phase = (((r.report.tempo && r.report.tempo.phase_ms) || 0) + lead) / 1000.0;
  renderScore(r.svg || "");
  renderLegend();
  renderEditor();
  renderReport();
  drawWave();
  atualizaPres();
  invalidarMidi();
  stopPlay();
  state.qa = null; state.qa_applied = []; state.qa_pesos = null; state.qa_reverted = false;
  if (!opt.manterJanela) refreshWave();
  if (!opt.semQA) runQA(true);       // auditoria silenciosa: o resultado já nasce conferido
}

/* ---------------------------------------------------------------- partitura */
function renderScore(svg) {
  const wrap = $("score-wrap");
  if (!svg) return;
  const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
  const node = doc.documentElement;
  node.removeAttribute("width");
  node.removeAttribute("height");
  node.style.width = "100%";
  node.style.height = "auto";
  node.style.display = "block";
  node.style.background = "#fff";
  node.style.boxShadow = "0 1px 12px rgba(0,0,0,.35)";
  wrap.innerHTML = "";
  $("empty-score") && $("empty-score").remove();
  const holder = document.createElement("div");
  holder.id = "svg-holder";
  holder.appendChild(document.importNode(node, true));
  wrap.appendChild(holder);
  applyZoom();
  cursorReset();
}

function aplicaLarguraSVG() {
  const h = $("svg-holder");
  // a pauta é vetorial: 320% já é grande demais para ler; o teto é só dela, o do zoom de áudio
  // (ZOOM_MAX) é outro porque lá o que cresce é a resolução temporal, não o desenho.
  if (h) h.style.width = (100 * Math.min(3.2, Math.max(0.35, state.zoom))).toFixed(1) + "%";
  const z = $("zval");
  if (z) z.textContent = Math.round(state.zoom * 100) + "%";
}
function applyZoom() { aplicaLarguraSVG(); }
function setZoom(z) {
  state.zoom = Math.min(ZOOM_MAX, Math.max(0.35, z));
  state.zoomCentro = centroJanela();
  aplicaZoom();
}
/* ------------------------------------------------------------------ janela (zoom) */
function duracaoTotal() {
  return Math.max(0.001, state.duration || (state.wave && state.wave.n && state.wave.dt
                 ? state.wave.n * state.wave.dt : 0.001));
}
function janelaAtual() {
  const d = duracaoTotal();
  let a = state.view.a, b = state.view.b;
  if (a == null || b == null || !(b > a)) return [0, d];
  return [Math.max(0, Math.min(a, d - 0.001)), Math.min(d, b)];
}
function centroJanela() {
  const t = state.playing ? currentTime() : (state.offset || 0);
  const [a, b] = janelaAtual();
  if (t != null && t >= a && t <= b) return t;
  return (a + b) / 2;
}
function aplicaZoom() {
  aplicaLarguraSVG();
  const d = duracaoTotal();
  const span = Math.min(d, Math.max(0.05, d / Math.max(1, state.zoom)));
  const c = state.zoomCentro == null ? d / 2 : state.zoomCentro;
  let a = c - span / 2, b = c + span / 2;
  if (a < 0) { b -= a; a = 0; }
  if (b > d) { a -= (b - d); b = d; }
  setJanela(Math.max(0, a), b, { viaZoom: true });
}
function setJanela(a, b, opt) {
  opt = opt || {};
  const d = duracaoTotal();
  b = Math.min(d, Math.max(a + 0.0005, b));
  a = Math.max(0, Math.min(a, b - 0.0005));
  state.view = { a: a, b: b };
  if (!opt.viaZoom) state.zoom = Math.min(ZOOM_MAX, Math.max(0.35, d / Math.max(0.0005, b - a)));
  state.zoomCentro = (a + b) / 2;
  aplicaLarguraSVG();
  etiquetaJanela();
  if (opt.semOnda) { if (abaAtiva() === "midi") desenhaRoll(); return; }
  refreshWave();
  if (abaAtiva() === "midi") desenhaRoll();
}
function etiquetaJanela() {
  const el = $("w-span"); if (!el) return;
  const [a, b] = janelaAtual();
  const d = duracaoTotal(), span = b - a;
  const res = state.finoOk && state.fino ? state.fino.bucket_sec * 1000 : null;
  el.textContent = "janela " + fmt(a, 2) + "–" + fmt(b, 2) + " s (" +
    (span < 1 ? fmt(span * 1000, 1) + " ms" : fmt(span, 2) + " s") +
    " · " + Math.round(d / span) + "×" + (res ? " · " + fmt(res, 2) + " ms/balde" : "") + ")";
  const w = $("w-sel");
  if (w) w.disabled = !(state.sel && Math.abs(state.sel.b - state.sel.a) > 0.004);
}
function abaAtiva() {
  const b = document.querySelector("#steps .step.active");
  return b ? b.dataset.tab : "score";
}
function zoomNaJanela(t, fator) {
  const [a, b] = janelaAtual();
  const d = duracaoTotal(), span = b - a;
  let ns = span * fator;
  ns = Math.min(d, Math.max(0.05, ns));
  const fx = (t - a) / span;
  let na = t - fx * ns;
  if (na < 0) na = 0;
  setJanela(na, Math.min(d, na + ns));
}
function enquadrarCompasso() {
  if (!state.score) return;
  const g = gradeBase();
  const t = state.offset || 0;
  const b = Math.max(0, Math.floor((t - g.t0) / (g.len || 1)));
  setJanela(g.t0 + b * g.len, g.t0 + (b + 1) * g.len);
  setStatus("janela no compasso " + (b + 1) + " (" + fmt(g.t0 + b * g.len, 2) + "–" +
            fmt(g.t0 + (b + 1) * g.len, 2) + " s).");
}

/* ------------------------------------------------------------------- ondas */
function laneColor(l) { return LANE_COLORS[l] || "#888"; }

async function refreshWave(forcar) {
  const [a, b] = janelaAtual();
  desenhaOndas();                          // pinta primeiro o que já tem; a rede vem depois
  if (!state.id) { state.finoOk = null; return; }
  const finoDesligado = $("w-fino") && !$("w-fino").checked;
  if (finoDesligado) { state.finoOk = false; notaOnda("envelope fino desligado — esta é a visão geral de 1800 baldes da faixa inteira."); return; }
  if (!state.wave_fine) {
    state.finoOk = false;
    notaOnda("esta análise não gerou envelope fino (faixa rebaixada por memória ou gravada antes " +
             "do recurso) — o zoom continua andando, mas sobre a visão geral. Reenvie a faixa para " +
             "ter ~1 ms por balde.");
    return;
  }
  const cv = $("wave");
  const W = Math.max(320, Math.floor(cv.clientWidth || 900));
  const chave = a.toFixed(4) + "|" + b.toFixed(4) + "|" + W + "|" + (($("w-curvas") && $("w-curvas").checked) ? "c" : "-");
  if (!forcar && chave === state.finoChave) return;
  if (state.finoPromise) { state.finoDepois = chave; return; }   // uma leitura por vez; a última janela ganha
  const url = API("wave/" + state.id + "?t0=" + a.toFixed(4) + "&t1=" + b.toFixed(4) +
                  "&n=" + W + "&curvas=" + (($("w-curvas") && $("w-curvas").checked) ? "1" : "0"));
  state.finoPromise = fetchT(url, null, TMO.rapido).then((r) => r.json()).then((j) => {
    state.finoPromise = null;
    if (!j || !j.ok) {
      state.finoOk = false; state.fino = null;
      notaOnda((j && j.error) || "envelope indisponível — mostrando a visão geral.");
    } else {
      state.finoOk = true; state.fino = j; state.finoChave = chave;
      notaOnda(j.recortado_ms > 0.5 ? ("últimos " + fmt(j.recortado_ms, 1) + " ms da janela ficaram de " +
               "fora do agrupamento (nº de baldes não divisível por " + j.baldes_por_ponto + ").") : "");
    }
    desenhaOndas(); etiquetaJanela();
    if (state.finoDepois) { const k = state.finoDepois; state.finoDepois = null; refreshWave(true); }
  }).catch((e) => {
    state.finoPromise = null; state.finoOk = false;
    notaOnda("a leitura do envelope falhou (" + e.message + ") — visão geral no lugar.");
    desenhaOndas();
  });
}

function notaOnda(txt) {
  const el = $("w-note"); if (!el) return;
  el.textContent = txt || "";
  el.hidden = !txt;
}
function estadoOnda(txt) { const el = $("w-read"); if (el && txt != null) el.dataset.busy = txt; }

/* a única passada de desenho da aba: onda + novidade + o que foi gravado, na MESMA janela */
/* medida do contêedor com reserva: sem `parentElement` ou sem `getContext` (navegador sem 2d,
   DOM de teste) a função sai cedo em vez de estourar — a aba tem de continuar legível. */
function caixaDe(cv) {
  const pe = cv && cv.parentElement;
  if (pe && pe.getBoundingClientRect) return pe.getBoundingClientRect();
  return { width: (cv && cv.clientWidth) || 900, height: 320 };
}
function contexto2d(cv) {
  if (!cv || !cv.getContext) return null;
  try { return cv.getContext("2d"); } catch (e) { return null; }
}

function desenhaOndas() {
  const cv = $("wave");
  if (!cv) return;
  const box = caixaDe(cv);
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.max(600, Math.floor(box.width * dpr));
  cv.height = Math.floor(ONDA_H * dpr);
  const g = contexto2d(cv);
  if (!g) return;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.scale(dpr, dpr);
  const W = cv.width / dpr, H = cv.height / dpr;
  g.clearRect(0, 0, W, H);
  g.fillStyle = "#12161c"; g.fillRect(0, 0, W, H);
  const R = { onda: [8, 116], curvas: [124, 198], notas: [206, 252], regua: [258, 282], comp: [286, 314] };
  const [a, b] = janelaAtual();
  const span = Math.max(1e-6, b - a);
  const t2x = (t) => ((t - a) / span) * W;
  const x2t = (x) => a + (x / W) * span;
  state._x2t = x2t; state._t2x = t2x; state._ondaW = W;
  if (!state.duration) {
    g.fillStyle = "#5d6875"; g.font = "13px system-ui";
    g.fillText("sem áudio carregado — envie uma faixa ou use a demonstração", 12, (R.onda[0] + R.onda[1]) / 2);
    return;
  }
  const f = (state.finoOk && state.fino) ? state.fino : null;
  const src = f ? { min: f.min, max: f.max, dt: f.dt, t0: f.t0 }
                : (state.wave ? { min: state.wave.min, max: state.wave.max, dt: state.wave.dt, t0: 0 } : null);
  const rot = (y, txt) => { g.fillStyle = "#5d6875"; g.font = "10px ui-monospace,monospace";
                            g.textAlign = "left"; g.fillText(txt, 4, y); };
  rot(R.onda[0] + 10, f ? "forma de onda · " + fmt(f.bucket_sec * 1000, 2) + " ms/balde"
                        : "forma de onda (visão geral · " + (state.wave ? fmt(state.wave.dt * 1000, 1) : "—") + " ms/balde)");
  if (f) rot(R.curvas[0] + 10, "novidade por banda (o que o detector ouviu) · topo " + fmt(f.curvas_topo || 1, 2));
  rot(R.notas[0] + 9, "o que a partitura gravou · cheio = nota · meio pino = dois ataques numa nota" +
      (f && f.estados ? " · vazado = " + (f.estados.n || 0) + " ataque(s) sem nota · " +
       f.n_notas_janela + " notas na janela" : " · vazado = ataque sem nota"));

  /* grade de batidas e compassos — a mesma origem do cursor (grade medida, não phase_ms) */
  const gb = gradeBase();
  const bpm = (state.score && state.score.bpm) || 0;
  let barNum = null;
  if (bpm && gb.len > 0) {
    const tpb = state.score.ticks_per_beat || 8, tpr = state.score.ticks_per_bar || tpb * 4;
    const spb = gb.len / (tpr / tpb);
    const k0 = Math.floor((a - gb.t0) / spb) - 1, k1 = Math.ceil((b - gb.t0) / spb) + 1;
    const querBatidas = !$("met") || $("met").checked;      // a caixa da barra vale aqui também
    const mostraBatida = (k1 - k0) < 400 && querBatidas;
    for (let k = k0; k <= k1; k++) {
      const t = gb.t0 + k * spb;
      if (t < a - spb || t > b + spb) continue;
      const x = t2x(t);
      const isBar = ((k % (tpr / tpb)) + (tpr / tpb)) % (tpr / tpb) === 0;
      g.strokeStyle = isBar ? "rgba(255,255,255,.30)" : "rgba(255,255,255,.12)";
      g.lineWidth = isBar ? 1.1 : 0.7;
      g.beginPath(); g.moveTo(x, R.onda[0]); g.lineTo(x, R.notas[1]); g.stroke();
      if (isBar && mostraBatida) {
        const bi = Math.floor((t - gb.t0) / gb.len);
        g.fillStyle = "rgba(255,255,255,.42)"; g.font = "9.5px ui-monospace,monospace";
        g.textAlign = "left";
        g.fillText(String(bi + 1), x + 3, R.comp[0] + 11);
        const tnx = gb.t0 + (bi + 1) * gb.len, larg = t2x(tnx) - x;
        if (larg > 30) {
          g.fillStyle = "rgba(255,255,255,.05)";
          g.fillRect(x, R.onda[0], larg, R.notas[1] - R.onda[0]);
        }
        barNum = bi;
      }
    }
  }
  /* forma de onda */
  if (src && src.min && src.min.length) {
    const mid = (R.onda[0] + R.onda[1]) / 2, amp = (R.onda[1] - R.onda[0]) / 2 - 6;
    const passo = (src.dt / span) * W;
    g.fillStyle = f ? "#3f86d8" : "#2d6cb5";
    for (let i = 0; i < src.min.length; i++) {
      const t = src.t0 + (i + 0.5) * src.dt;
      const x = t2x(t);
      if (x < -3 || x > W + 3) continue;
      const y0 = mid - Math.min(1, src.max[i]) * amp;
      const y1 = mid - Math.max(-1, src.min[i]) * amp;
      g.fillRect(x - Math.max(0.35, passo * 0.45), y0, Math.max(0.7, passo * 0.9),
                 Math.max(0.8, y1 - y0));
    }
    g.strokeStyle = "rgba(255,255,255,.16)"; g.lineWidth = 0.6;
    g.beginPath(); g.moveTo(0, mid); g.lineTo(W, mid); g.stroke();
  }
  /* novidade por banda */
  if (f && f.curvas && $("w-curvas") && $("w-curvas").checked) {
    const cores = { global: "#e6edf5", low: LANE_COLORS.kick, mid: LANE_COLORS.snare,
                    high: LANE_COLORS.hat };
    const nomes = ["global", "low", "mid", "high"].filter((k) => f.curvas[k]);
    const alt = R.curvas[1] - R.curvas[0] - 8;
    let topeGlobal = 1e-6;
    nomes.forEach((k) => { const v = f.curvas[k]; for (let i = 0; i < v.length; i++) if (v[i] > topeGlobal) topeGlobal = v[i]; });
    nomes.forEach((k, li) => {
      const v = f.curvas[k];
      g.strokeStyle = cores[k] || "#9ab"; g.lineWidth = k === "global" ? 1.3 : 1.0;
      g.globalAlpha = k === "global" ? 0.95 : 0.85;
      g.beginPath();
      for (let i = 0; i < v.length; i++) {
        const t = f.t0 + (i + 0.5) * f.dt, x = t2x(t);
        if (x < -2 || x > W + 2) continue;
        const y = R.curvas[1] - Math.max(0, Math.min(1, v[i] / topeGlobal)) * alt;
        if (li === 0 || k === "global") { if (i === 0) g.moveTo(x, y); else g.lineTo(x, y); }
        else { g.lineTo(x, y); }
      }
      g.stroke();
      g.globalAlpha = 1;
    });
    g.font = "9.5px ui-monospace,monospace"; g.textAlign = "left";
    nomes.forEach((k, i) => {
      const rot2 = { global: "global", low: "grave", mid: "médio", high: "agudo" }[k] || k;
      g.fillStyle = cores[k] || "#9ab";
      g.fillText(rot2, 6 + i * 54, R.curvas[1] + 8);
    });
  } else if (!f) {
    g.fillStyle = "#3b4450"; g.font = "11px system-ui"; g.textAlign = "left";
    g.fillText("curvas de novidade só com o envelope fino", 8, (R.curvas[0] + R.curvas[1]) / 2);
  }
  /* o que foi gravado — detecções do próprio trecho, marcadas contra a pauta */
  const mostraNotas = !($("w-notas") && !$("w-notas").checked);
  const escritos = novosEscritos();
  const evs = (f && f.eventos) ? f.eventos
            : ((state.hits || []).filter((h) => h.time != null).map((h) => ({
                 t: +h.time, lane: h.lane, conf: h.confidence == null ? 1 : h.confidence,
                 bar: h.bar, tick: h.tick, escrito: true })));
  const base = R.notas[1], topo = R.notas[0];
  (mostraNotas ? evs : []).forEach((ev) => {
    const x = t2x(ev.t);
    if (x < -2 || x > W + 2) return;
    const st = ev.st || (ev.escrito === false ? "n"
                          : (escritos.has(ev.bar + "|" + ev.tick + "|" + ev.lane) ? "e" : "n"));
    const cor = laneColor(ev.lane);
    g.globalAlpha = 0.5 + 0.5 * Math.min(1, ev.conf || 0.6);
    g.fillStyle = cor;
    if (st === "e") {
      g.fillRect(x - 1, topo, 2, base - topo);
    } else if (st === "m") {                       // dois ataques, uma nota: meio traço + pino
      g.fillRect(x - 1, (topo + base) / 2, 2, base - (topo + base) / 2);
      g.fillRect(x - 2.6, topo + 2, 5.2, 3);
    } else {                                       // ataque sem nota: caixa vazada
      g.globalAlpha = 0.9;
      g.strokeStyle = cor; g.lineWidth = 1;
      g.strokeRect(x - 2.2, topo + 5, 4.4, 9);
    }
    g.globalAlpha = 1;
    if (ev.resid_ms != null && Math.abs(ev.resid_ms) > 25) {
      g.fillStyle = "#f0b43c"; g.beginPath(); g.arc(x, base + 5, 1.7, 0, 6.2832); g.fill();
    }
  });
  /* loop do compasso */
  if (state.loopBar !== null && state.score) {
    g.fillStyle = "rgba(240,180,60,.14)";
    g.fillRect(t2x(gb.t0 + state.loopBar * gb.len), R.onda[0],
               Math.max(2, (gb.len / span) * W), R.notas[1] - R.onda[0]);
  }
  /* seleção (shift+arraste) */
  if (state.sel) {
    const x0 = t2x(Math.min(state.sel.a, state.sel.b)), x1 = t2x(Math.max(state.sel.a, state.sel.b));
    g.fillStyle = "rgba(120,200,255,.16)"; g.fillRect(x0, R.onda[0], Math.max(1, x1 - x0), R.notas[1] - R.onda[0]);
    g.strokeStyle = "rgba(120,200,255,.8)"; g.lineWidth = 1;
    g.strokeRect(x0 + 0.5, R.onda[0] + 0.5, Math.max(1, x1 - x0), R.notas[1] - R.onda[0]);
  }
  /* régua de tempo */
  const passos = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60];
  let ps = passos[passos.length - 1];
  for (let i = 0; i < passos.length; i++) { if (span / passos[i] <= 11) { ps = passos[i]; break; } }
  g.strokeStyle = "rgba(255,255,255,.30)"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, R.regua[0]); g.lineTo(W, R.regua[0]); g.stroke();
  g.font = "10px ui-monospace,monospace"; g.textAlign = "center";
  for (let k = Math.ceil(a / ps); k * ps <= b; k++) {
    const t = k * ps, x = t2x(t);
    g.strokeStyle = "rgba(255,255,255,.30)";
    g.beginPath(); g.moveTo(x, R.regua[0]); g.lineTo(x, R.regua[0] + 5); g.stroke();
    g.fillStyle = "#8b97a6";
    g.fillText(ps < 1 ? fmt(t * 1000, ps < 0.01 ? 1 : 0) + " ms" : fmt(t, ps < 5 ? 1 : 0) + " s", x, R.regua[1]);
  }
  /* cursor */
  const t = currentTime();
  if (t !== null && t >= a - span * 0.02 && t <= b + span * 0.02) {
    const x = t2x(t);
    g.strokeStyle = "#f5d97a"; g.lineWidth = 1.4;
    g.beginPath(); g.moveTo(x, R.onda[0] - 4); g.lineTo(x, R.notas[1] + 8); g.stroke();
    g.fillStyle = "#f5d97a"; g.beginPath();
    g.moveTo(x - 4, R.onda[0] - 6); g.lineTo(x + 4, R.onda[0] - 6); g.lineTo(x, R.onda[0] - 1); g.fill();
  }
  g.textAlign = "left";
}

/* conjunto (bar|tick|pista) do que de fato virou nota na pauta — para o "vazado" ser verdade */
function novosEscritos() {
  const st = new Set();
  ((state.score && state.score.bars) || []).forEach((b) => (b.hits || []).forEach((h) => {
    st.add((b.index != null ? b.index : h.bar) + "|" + (h.tick == null ? 0 : h.tick) + "|" + h.lane);
  }));
  return st;
}

function drawWave() { desenhaOndas(); }

/* --------------------------------------------------------------- player */
async function ensureCtx() {
  if (!state.ctx) {
    state.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const g = state.ctx.createGain();
    g.gain.value = +$("vol").value;
    g.connect(state.ctx.destination);
    state.gain = g;
  }
  if (state.ctx.state === "suspended") await state.ctx.resume();
}

async function loadBuffer(url) {
  const r = await fetchT(url, null, TMO.medio);
  if (!r.ok) throw new Error("áudio indisponível");
  const data = await r.arrayBuffer();
  await ensureCtx();
  return state.ctx.decodeAudioData(data);
}

function currentTime() {
  if (!state.buffer) return state.duration ? 0 : null;
  if (!state.playing) return state.offset;
  return Math.min(state.duration, state.offset + (state.ctx.currentTime - state.startedAt));
}

async function togglePlay() {
  if (state.playing) { stopPlay(); return; }
  try {
    if (!state.buffer) {
      if (!state.audio_url) { setStatus("toque primeiro uma faixa (ou use a demonstração).", "err"); return; }
      busy(true, "preparando o áudio…");
      state.buffer = await loadBuffer(state.audio_url);
      state.duration = state.buffer.duration;
      busy(false);
    }
    await ensureCtx();
    state.mode = "original";
    startPlay(state.offset || 0);
  } catch (e) {
    busy(false); setStatus("não consegui tocar: " + e.message, "err");
  }
}

function startPlay(off) {
  if (state.src) { try { state.src.stop(); } catch (e) {} }
  const s = state.ctx.createBufferSource();
  s.buffer = state.buffer;
  s.connect(state.gain);
  s.start(0, Math.max(0, Math.min(off, state.buffer.duration - 0.02)));
  state.src = s;
  state.offset = off;
  state.startedAt = state.ctx.currentTime;
  state.playing = true;
  $("btn-play").textContent = "❚❚";
  s.onended = () => {
    if (state.loopBar !== null && state.playing) {
      startPlay(t0DoBaro(state.loopBar));
    } else { stopPlay(); drawCursor(state.duration || 0); }
  };
  tick();
}

function stopPlay() {
  if (state.src) { state.offset = currentTime() || 0; try { state.src.stop(); } catch (e) {} state.src = null; }
  state.playing = false;
  drawCursor(state.offset || 0);          // o cursor marca onde a audição parou
  $("btn-play").textContent = "▶";
  cancelAnimationFrame(state.raf);
  drawWave();
}

function tick() {
  const t = currentTime();
  if (t === null) return;
  $("timecode").textContent = fmt(t, 2) + " s";
  segueJanela(t);
  drawWave();
  drawCursor(t);
  if (state.playing) state.raf = requestAnimationFrame(tick);
}

function seekFromClick(e) {
  if (!state.duration) return;
  const r = e.target.getBoundingClientRect();
  // com a janela ampliada, o x do mouse é lido pela JANELA (state._x2t), não pela faixa toda:
  // mapear por `duração` inteira era o zoom mentir para o clique.
  let t;
  if (state._x2t) {
    const escala = (state._ondaW || r.width) / (r.width || 1);
    t = state._x2t((e.clientX - r.left) * escala);
  } else {
    t = ((e.clientX - r.left) / r.width) * state.duration;
  }
  if (e.shiftKey && state.score) {
    const b = Math.max(0, Math.floor((t - faseAtual()) / durBarro()));
    state.loopBar = (state.loopBar === b) ? null : b;
    setStatus(state.loopBar === null ? "loop desligado." : "repetindo o compasso " + (b + 1) + " (shift+clique de novo desliga).");
    drawWave();
    return;
  }
  state.offset = t;
  if (state.playing) startPlay(t);
  drawWave();
}

async function playSynth() {
  if (!state.score) { setStatus("sem partitura para sintetizar.", "err"); return; }
  try {
    busy(true, "sintetizando a bateria da partitura…");
    const r = await fetchT(API("synth"), {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: state.id, score: state.score }),
    }, TMO.medio);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || "erro " + r.status);
    const buf = await r.arrayBuffer();
    await ensureCtx();
    state.buffer = await state.ctx.decodeAudioData(buf);
    state.duration = state.buffer.duration;
    state.offset = 0;
    state.mode = "synth";                 // o sintético começa no tique zero, não na fase do arquivo
    busy(false);
    startPlay(0);
    setStatus("tocando a *transcrição* (não o arquivo original).", "ok");
  } catch (e) { busy(false); setStatus("síntese falhou: " + e.message, "err"); }
}

/* ----------------------------------------------------------------- legenda */
function renderLegend() {
  const el = $("legend");
  if (!el) return;
  const used = [];
  ((state.score && state.score.bars) || []).forEach((b) => (b.hits || []).forEach((h) => {
    if (used.indexOf(h.lane) < 0) used.push(h.lane);
  }));
  el.innerHTML = used.map((l) => {
    const ln = state.lanes.find((x) => x.id === l) || { name: l, short: l };
    const n = ((state.score.bars || []).reduce((a, b) => a + (b.hits || []).filter((h) => h.lane === l).length, 0));
    return '<span class="lg"><i style="background:' + laneColor(l) + '"></i>' + ln.name +
           ' <b>' + n + '</b></span>';
  }).join("");
}

/* ------------------------------------------------------------------ editor */
function tickLabel(t) {
  const tpb = (state.score && state.score.ticks_per_beat) || 8;
  const beat = Math.floor(t / tpb) + 1;
  const sub = ["1", "e", "&", "a"][Math.floor((t % tpb) / (tpb / 4))] || "+";
  return beat + (sub === "1" ? "" : " " + sub);
}

function currentHits() {
  const out = [];
  ((state.score && state.score.bars) || []).forEach((b) => {
    (b.hits || []).forEach((h) => {
      const o = {};
      Object.keys(h).forEach((k) => (o[k] = h[k]));
      o.bar = b.index;
      out.push(o);
    });
  });
  out.sort((a, b) => (a.bar - b.bar) || (a.tick - b.tick) || (a.lane < b.lane ? -1 : 1));
  return out;
}

function renderEditor() {
  const grid = $("ed-grid");
  state.hits = currentHits();
  const nBars = ((state.score && state.score.bars) || []).length;
  const tpb = (state.score && state.score.ticks_per_beat) || 8;
  const tpr = (state.score && state.score.ticks_per_bar) || tpb * 4;
  const step = tpb / 4;
  const opts = (sel, list) => list.map(([v, t]) =>
    '<option value="' + v + '"' + (String(v) === String(sel) ? " selected" : "") + ">" + t + "</option>").join("");
  const laneOpts = state.lanes.map((l) => [l.id, l.name]);
  const artOpts = [["normal", "normal"], ["accent", "acento >"], ["ghost", "ghost ()"],
                   ["ghost_accent", "ghost+acento"], ["flam", "flam"], ["droll", "paradiddle"]];
  let html = '<div class="ed-row ed-head"><span>comp.</span><span>pulso</span><span>peça</span>' +
             '<span>vel.</span><span>artic.</span><span>conf.</span><span></span></div>';
  state.hits.forEach((h, i) => {
    const ticks = [];
    for (let t = 0; t < tpr; t += step) ticks.push([t, tickLabel(t)]);
    html += '<div class="ed-row" data-i="' + i + '">' +
      '<input type="number" data-k="bar" min="1" max="' + nBars + '" value="' + (h.bar + 1) + '">' +
      '<select data-k="tick">' + opts(h.tick, ticks) + "</select>" +
      '<select data-k="lane">' + opts(h.lane, laneOpts) + "</select>" +
      '<input type="number" data-k="velocity" min="1" max="127" value="' + (h.velocity || 88) + '">' +
      '<select data-k="artic">' + opts(h.artic || "normal", artOpts) + "</select>" +
      '<span class="conf"><i style="width:' + Math.round((h.confidence || 0) * 100) + '%"></i>' +
      fmt(h.confidence || 0, 2) + "</span>" +
      '<button class="del" title="remover nota">✕</button></div>';
  });
  grid.innerHTML = html;
  $("ed-info").textContent = state.hits.length + " notas em " + nBars + " compassos";
  $$(".ed-row input, .ed-row select", grid).forEach((el) => {
    el.addEventListener("change", (e) => {
      const row = el.closest(".ed-row");
      const i = +row.dataset.i, k = el.dataset.k;
      let v = el.value;
      if (k === "bar") v = Math.max(0, +v - 1);
      else if (k === "tick" || k === "velocity" || k === "dur") v = +v;
      state.hits[i][k] = v;
      row.classList.add("dirty");
      paintHitsFromEditor();
    });
  });
  $$(".ed-row .del", grid).forEach((b) => b.addEventListener("click", (e) => {
    const i = +e.target.closest(".ed-row").dataset.i;
    state.hits.splice(i, 1);
    applyHitsToScore();
    renderEditor();
    markDirty("nota removida — clique em reescrever partitura.");
  }));
}

function paintHitsFromEditor() {
  applyHitsToScore();
  drawWave();
}

function applyHitsToScore() {
  if (!state.score) return;
  const byBar = {};
  state.score.bars.forEach((b) => (byBar[b.index] = []));
  state.hits.forEach((h) => {
    const b = Math.max(0, Math.min(state.score.bars.length - 1, h.bar | 0));
    if (!byBar[b]) byBar[b] = [];
    const o = {};
    Object.keys(h).forEach((k) => { if (k !== "bar") o[k] = h[k]; });
    byBar[b].push(o);
  });
  state.score.bars.forEach((b) => { b.hits = (byBar[b.index] || []).sort((x, y) => x.tick - y.tick); });
}

function addRow() {
  if (!state.score) { setStatus("analise um áudio primeiro.", "err"); return; }
  const tpb = state.score.ticks_per_beat || 8;
  state.hits.push({ bar: 0, tick: tpb * 2, lane: "snare", dur: 2, velocity: 96,
                    artic: "normal", confidence: 1.0, time: 0, head: "circle" });
  applyHitsToScore();
  renderEditor();
  markDirty("nota adicionada no compasso 1 — reescreva a partitura.");
  $("ed-grid").scrollTop = $("ed-grid").scrollWidth;
}

async function rebuild() {
  if (!state.score) return;
  busy(true, "reescrevendo a partitura…");
  try {
    const r = await jget(API("rebuild"), {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: state.id, hits: state.hits, bpm: state.score.bpm,
                             meter: state.score.meter, swing: state.score.swing,
                             title: state.score.title, subtitle: state.score.subtitle }),
    }, TMO.medio);
    state.score = r.score;
    if (r.layout) state.layout = r.layout;      // geometria nova com a partitura nova: o cursor
                                                 // não pode ficar no mapa da versão anterior
    state.hits = currentHits();
    renderScore(r.svg);
    renderEditor();
    renderLegend();
    drawWave();
    refreshWave(true);                           // a faixa "o que foi gravado" é o resultado desta edição
    invalidarMidi();
    if (abaAtiva() === "midi") pedeMidi(true);
    markDirty("");
    setStatus("partitura reescrita com as suas correções.", "ok");
  } catch (e) { setStatus("falha ao reescrever: " + e.message, "err"); }
  finally { busy(false); }
}

/* o roll é lido do .mid gerado a partir da partitura: qualquer coisa que mude a pauta muda ele */
function invalidarMidi() { state.midChave = ""; }

function markDirty(msg) {
  const b = $("ed-apply");
  b.classList.toggle("warn", !!msg);
  if (msg) setStatus(msg);
}

/* ---------------------------------------------------------------- relatório */
function renderReport() {
  const el = $("report");
  const rep = state.report || {};
  const sc = state.score || {};
  const tempo = rep.tempo || {}, met = rep.meter || {}, grid = rep.grid || {};
  const evs = rep.events || {};
  const cls = { mean_confidence: evs.mean_confidence, per_lane: evs.per_lane }, eng = rep.engraving || {};
  const cards = [
    ["andamento", (tempo.bpm ? fmt(tempo.bpm, 1) + " BPM" : "—"), tempo.method || ""],
    ["compasso", met.value || sc.meter || "—", (met.score ? "ajuste " + fmt(met.score, 3) : "")],
    ["grade", grid.mode || "—", (grid.fitness ? "ajuste " + fmt(grid.fitness, 3) : "")],
    ["swing", grid.swing ? fmt(grid.swing, 2) : "reto", grid.swing_allowed ? "detectado" : ""],
    ["compassos", String(sc.bars ? sc.bars.length : 0), (eng.fill_bars ? eng.fill_bars + " com virada" : "")],
    ["notas", String(currentHits().length), cls.mean_confidence ? "confiança " + fmt(cls.mean_confidence, 2) : ""],
    ["análise", rep.total_sec ? fmt(rep.total_sec, 1) + " s" : "—",
      (rep.timings_sec ? "ataques " + fmt(rep.timings_sec.onsets || 0, 1) + " s · total " +
        fmt(rep.total_sec || 0, 1) + " s" : "")],
    ["áudio", rep.file ? fmt(rep.file.duration_sec || 0, 1) + " s" : "—",
      rep.file ? (rep.file.peak_dbfs + " dBFS pico") : ""],
  ];
  let html = '<div class="cards">' + cards.map((c) =>
    '<div class="card"><span>' + c[0] + "</span><b>" + c[1] + "</b><small>" + (c[2] || "&nbsp;") +
    "</small></div>").join("") + "</div>";

  const per = (cls.per_lane) || {};
  const rows = Object.keys(per).map((k) =>
    '<tr><td><i class="sw" style="background:' + laneColor(k) + '"></i>' +
    ((state.lanes.find((l) => l.id === k) || {}).name || k) + "</td><td>" + per[k].hits +
    "</td><td>" + fmt((per[k].mean_confidence || 0), 2) + "</td><td>" + fmt(per[k].mean_vel || 0, 0) + "</td></tr>");
  if (rows.length) {
    html += '<h3>peças detectadas</h3><table class="tbl"><thead><tr><td>peça</td><td>notas</td>' +
      '<td>confiança</td><td>vel. média</td></tr></thead><tbody>' + rows.join("") + "</tbody></table>";
  }
  const fits = grid.fits_by_grid;
  if (fits) {
    const ks = Object.keys(fits).sort((a, b) => fits[b] - fits[a]);
    html += '<h3>candidatos de grade (quanto maior, melhor o encaixe)</h3><div class="fits">' +
      ks.map((k) => '<span class="' + (k === (grid.mode + "@" + (grid.swing || 0).toFixed(2)) ? "pick" : "") +
        '"><code>' + k + "</code><b>" + fmt(fits[k], 3) + "</b></span>").join("") + "</div>";
  }
  const tab = (met.table || []).slice(0, 5);
  if (tab.length) {
    html += '<h3>métricas candidatas</h3><div class="fits">' + tab.map((t) =>
      '<span><code>' + t.meter + "</code><b>" + fmt(t.score, 4) + "</b> <small>off " + t.tick_offset +
      "</small></span>").join("") + "</div>";
  }
  if ((rep.warnings || []).length) {
    html += '<h3>avisos</h3><ul class="warn-list">' + rep.warnings.map((w) => "<li>" + w + "</li>").join("") + "</ul>";
  }
  html += '<h3>ajuste da grade (ms por nota)</h3><p class="mono small">' +
    "mediana " + fmt(grid.median_residual_ms || 0, 1) + " ms · fora da grade " +
    fmt(100 * (grid.offgrid_ratio || 0), 1) + "% · alvo de fitness " + fmt(grid.fit_target || 0.9, 2) +
    " · 32ºs promovidos " + (grid.promoted_32nd || 0) + "</p>";
  html += '<h3>parâmetros usados</h3><pre class="json">' +
    JSON.stringify(rep.params || {}, null, 1).slice(0, 2600) + "</pre>";
  el.innerHTML = html;
}

/* ------------------------------------------------------------------ exports */
async function exportAs(fmtName) {
  if (!state.score) { setStatus("não há partitura para exportar.", "err"); return; }
  busy(true, "gerando " + fmtName.toUpperCase() + "…");
  try {
    const r = await fetchT(API("export/" + fmtName), {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: state.id, score: state.score, page: "a4_landscape" }),
    }, TMO.medio);
    if (!r.ok) {
      let m = "erro " + r.status;
      try { m = (await r.json()).error || m; } catch (e) {}
      throw new Error(m);
    }
    const blob = await r.blob();
    const cd = r.headers.get("content-disposition") || "";
    const m = /filename="?([^"]+)"?/.exec(cd);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = m ? m[1] : ("drumscribe." + fmtName);
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    setStatus(fmtName.toUpperCase() + " gerado (" + Math.round(blob.size / 1024) + " KB).", "ok");
  } catch (e) { setStatus("falha ao exportar: " + e.message, "err"); }
  finally { busy(false); }
}

/* ------------------------------------------------ auditoria dupla e agente de correção */
const QA_LABEL = { ok: "ok", manual: "DECIDIR", auto: "corrigível", fixed: "corrigido",
                   reverted: "revertido", skipped: "n/d", nd: "n/d" };
const QA_SEV = { error: "erro", warn: "aviso", info: "info" };

function qaEsc(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Math.abs(v) >= 100 ? Math.round(v) : Math.round(v * 1000) / 1000;
  if (typeof v === "boolean") return v ? "sim" : "não";
  if (typeof v === "object") {
    const s = JSON.stringify(v);
    return s.length > 320 ? s.slice(0, 319) + "…" : s;
  }
  return String(v);
}

function qaBadge() {
  const b = $("qa-badge");
  if (!b) return;
  const t = state.qa && state.qa.tally;
  if (!t) { b.hidden = true; return; }
  b.hidden = false;
  b.textContent = t.error ? (t.error + (t.error > 1 ? " erros" : " erro"))
                          : (t.warn ? (t.warn + (t.warn > 1 ? " avisos" : " aviso")) : "limpo");
  b.className = "badge " + (t.error ? "bad" : (t.warn ? "warn" : "good"));
}

async function runQA(quiet) {
  if (!state.id) { setStatus("analise uma faixa antes de auditar.", "err"); return; }
  const box = $("qa-full");
  const level = (box && box.checked) ? "full" : "fast";
  if (!quiet) busy(true, "auditando: oráculos independentes, leis de pauta e releitura dos exports…");
  try {
    const r = await fetchT(API("qa"), {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: state.id, level: level }),
    }, level === "full" ? TMO.pesado : TMO.medio);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || "erro " + r.status);
    const j = await r.json();
    state.qa = j.qa; state.qa_md = j.markdown; state.qa_level = level;
    renderQA(); qaBadge();
    if (!quiet) setStatus("auditoria (" + level + "): " + j.qa.summary,
                          j.qa.tally.error ? "err" : (j.qa.tally.warn ? "" : "ok"));
  } catch (e) {
    setStatus("auditoria falhou: " + e.message, "err");
  } finally {
    if (!quiet) busy(false);
  }
}

async function qaApply() {
  if (!state.id) return;
  busy(true, "o agente corrige o que é determinístico e re-audita o resultado…");
  try {
    const r = await fetchT(API("qa/fix"), {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: state.id, level: state.qa_level || "fast" }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || "erro " + r.status);
    const j = await r.json();
    state.qa_applied = j.applied || [];
    state.qa_pesos = j.pesos || null;
    state.qa_reverted = !!j.reverted;
    if (j.changed && j.score) {
      state.score = j.score;
      state.hits = currentHits();
      renderScore(j.svg || "");
      renderLegend(); renderEditor(); renderReport(); drawWave();
    }
    state.qa = j.qa; state.qa_md = j.markdown;
    renderQA(); qaBadge();
    const n = state.qa_applied.filter((a) => a && !a.revertido).length;
    setStatus(j.changed ? ("agente: " + n + " correção(ões) aplicada(s), re-auditoria aprovada — " + j.qa.summary)
                        : (j.reverted ? "o agente propôs correções e a re-auditoria não aprovou — nada foi alterado"
                                      : "nada havia para corrigir automaticamente"),
              j.changed ? "ok" : "");
  } catch (e) {
    setStatus("o reparo falhou: " + e.message, "err");
  } finally {
    busy(false);
  }
}

function renderQA() {
  const host = $("qa-body");
  if (!host) return;
  const qa = state.qa;
  const fix = $("qa-fix"), dl = $("qa-dl");
  if (fix) fix.disabled = !(qa && qa.tally && (qa.tally.auto > 0 || qa.tally.error > 0 || qa.tally.warn > 0));
  if (dl) dl.disabled = !qa;
  const sum = $("qa-sum");
  const marca = (txt, cls) => { if (sum) { sum.textContent = txt; sum.className = cls; } };
  if (!qa) {
    marca("sem auditoria ainda", "qa-sum mono");
    host.innerHTML = '<p class="qa-note">Rode a auditoria para ver o quadro.</p>';
    return;
  }
  const tt = qa.tally || {};
  marca((qa.summary || "") + " · nível " + (state.qa_level || "fast"),
        "qa-sum mono " + (tt.error ? "bad" : (tt.warn ? "warn" : "good")));
  const t = qa.tally || {};
  const cards = [
    ["erros", t.error || 0, "lei violada ou os dois caminhos divergindo"],
    ["avisos", t.warn || 0, "suspeita medida — a decisão é sua"],
    ["informações", t.info || 0, "conferido e dentro da lei"],
    ["pedem decisão", t.manual || 0, "sem reparo automático possível"],
    ["corrigíveis", t.auto || 0, "o agente tem reparo determinístico"],
    ["verificações", t.checks || 0, (state.qa_level || "fast") + " · dupla medição"],
  ];
  let h = '<div class="cards">' + cards.map((c) =>
    '<div class="card"><span>' + c[0] + "</span><b>" + c[1] + "</b><small>" + c[2] + "</small></div>"
  ).join("") + "</div>";
  if (state.qa_pesos) {
    h += '<p class="qa-sum mono">peso do quadro: ' + qaEsc(state.qa_pesos.inicial) + " → " +
         qaEsc(state.qa_pesos.final) + (state.qa_reverted ? " · última proposta revertida pela re-auditoria" : "") + "</p>";
  }
  if (state.qa_applied && state.qa_applied.length) {
    h += "<h3>reparos aplicados pelo agente</h3><ul class=\"qa-log\">" +
         state.qa_applied.map((a) => {
           if (typeof a === "string") return "<li>" + qaEsc(a) + "</li>";
           const rest = Object.keys(a).filter((k) => k !== "fix")
             .map((k) => k + " " + qaEsc(a[k])).join(" · ");
           return "<li><b>" + qaEsc(a.fix || "?") + "</b>" + (rest ? " — " + rest : "") + "</li>";
         }).join("") + "</ul>";
  }
  const rows = (qa.findings || []).map((f) => {
    const cls = f.severity === "error" ? "err" : (f.severity === "warn" ? "wrn" : "inf");
    const fx = f.fix ? '<span class="tag fix">→ ' + f.fix + "</span>" : "";
    return '<tr class="qa-row ' + cls + '">' +
      '<td><span class="tag st ' + (f.state || "ok") + '">' + (QA_LABEL[f.state] || f.state || "ok") + "</span></td>" +
      '<td class="mono"><b>' + f.check + "</b><br><small>" + (f.stage || "") + " · " +
      (QA_SEV[f.severity] || f.severity) + "</small></td>" +
      "<td>" + (f.title || "") + (fx ? "<br>" + fx : "") + "</td>" +
      '<td class="mono">' + qaEsc(f.a) + "</td>" +
      '<td class="mono">' + qaEsc(f.b) + "</td>" +
      "<td><details><summary>evidência</summary><p>" + qaEsc(f.detail) + "</p></details></td></tr>";
  }).join("");
  h += "<h3>quadro de achados</h3><table class=\"tbl qa-tbl\"><thead><tr><th>estado</th><th>teste</th>" +
       "<th>o que confere</th><th>A (pipeline)</th><th>B (oráculo/lei)</th><th>evidência</th></tr></thead>" +
       "<tbody>" + rows + "</tbody></table>";
  host.innerHTML = h;
}

function qaDownload() {
  if (!state.qa_md) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([state.qa_md], { type: "text/markdown;charset=utf-8" }));
  a.download = "auditoria-" + (state.id || "faixa") + ".md";
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

/* ------------------------------------------------- cursor de reprodução na partitura */
/* O mapa de geometria vem do layout (`layout.bar_map`), então o cursor é desenhado NAS MESMAS
   coordenadas da gravura: zoom, quebra de sistema e legenda movem tudo junto. Medir em px do
   navegador (getBoundingClientRect) daria duas fontes de verdade — e a segunda erraria no
   primeiro zoom. */
const SVGNS = "http://www.w3.org/2000/svg";

function durBarro() {
  const sc = state.score;
  if (!sc) return 0;
  const tpb = sc.ticks_per_beat || 8, tpr = sc.ticks_per_bar || 32;
  return (tpr / tpb) * (60 / (sc.bpm || 120));
}
/* Origem da grade no tempo do arquivo.

   `report.tempo.phase_ms` NÃO é essa origem (no demo difere 1,95 s — 3,25 batidas — da posição
   real dos golpes), então usamos a origem que o auditoria valida: `layout.clock.t0`, medida por
   mediana dos desvios tempo↔grade sobre as notas da própria partitura. Sem ela, caímos na mesma
   mediana calculada aqui; e a transcrição sintetizada começa no tique zero, sem fase nenhuma. */
function origemMedida() {
  const sc = state.score;
  if (!sc) return null;
  if (state._gradeOrigem !== undefined && state._gradePara === sc) return state._gradeOrigem;
  const tpr = sc.ticks_per_bar || 32, tpb = sc.ticks_per_beat || 8;
  const spt = (60 / (sc.bpm || 120)) / tpb;
  const v = [];
  (sc.bars || []).forEach((b) => (b.hits || []).forEach((h) => {
    if (h.time == null || !isFinite(h.time)) return;
    v.push(+h.time - (((b.index || 0) * tpr) + (h.tick || 0)) * spt);
  }));
  let out = null;
  if (v.length >= 8) {
    v.sort((a, b) => a - b);
    out = v[Math.floor(v.length / 2)];
  }
  state._gradePara = sc; state._gradeOrigem = out;
  return out;
}

function gradeBase() {
  const len = durBarro();
  if (state.mode === "synth") return { t0: 0, len: len, fonte: "síntese (tique 0)" };
  const c = state.layout && state.layout.clock;
  if (c && c.t0 != null && c.bar_len > 0) {
    return { t0: +c.t0, len: +c.bar_len, fonte: c.fonte || "grade medida",
             n: c.n, spread: c.spread_ms, max: c.max_ms };
  }
  const m = origemMedida();
  if (m != null) return { t0: m, len: len, fonte: "mediana local" };
  return { t0: state.phase || 0, len: len, fonte: "relatório (pouca evidência)" };
}

/* kept: origem usada por todo cálculo de compasso (cursor, clique, loop, grade da forma de onda) */
function faseAtual() { return gradeBase().t0; }
function t0DoBaro(b) { return gradeBase().t0 + (b || 0) * gradeBase().len; }
function notaDaGrade() {
  const g = gradeBase();
  if (g.n) return " · origem: " + g.fonte + " sobre " + g.n + " golpes (±" + g.spread +
                  " ms dp, máx ±" + g.max + " ms)";
  return " · origem: " + g.fonte;
}

/* pura de propósito: tempo → compasso + fração, sem tocar em DOM, para poder ser testada à parte */
function posicaoCursor(t, mapa, len, off) {
  if (!mapa || !mapa.length || !(len > 0) || !(t >= 0)) return null;
  const u = (t - (off || 0)) / len;
  let k = Math.floor(u);
  if (k < 0) k = 0;
  if (k >= mapa.length) k = mapa.length - 1;
  const g = mapa[k];
  const fx = Math.max(0, Math.min(1, u - k));
  const a = (g.beat0 == null ? g.x0 : g.beat0), b = (g.beat1 == null ? g.x1 : g.beat1);
  return { k: k, i: g.i, g: g, fx: fx, x: a + fx * (b - a), noFim: u >= mapa.length };
}

function svgEl(n, attrs) {
  const e = document.createElementNS(SVGNS, n);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function cursorCamada(criar) {
  const holder = $("svg-holder");
  if (!holder) return null;
  const svg = holder.querySelector("svg");
  if (!svg) return null;
  let g = svg.querySelector("#cur-layer");
  if (!g && criar) {
    g = svgEl("g", { id: "cur-layer", "pointer-events": "none" });
    g.appendChild(svgEl("rect", { id: "cur-band", fill: "rgba(240,180,41,.14)",
                                  stroke: "rgba(240,180,41,.55)", "stroke-width": "0.8",
                                  rx: "2" }));
    g.appendChild(svgEl("line", { id: "cur-line", stroke: "#e8543f", "stroke-width": "1.7",
                                 "stroke-linecap": "round" }));
    const cab = svgEl("g", { id: "cur-cap" });
    cab.appendChild(svgEl("rect", { id: "cur-capbox", fill: "#e8543f", rx: "2", height: "11" }));
    cab.appendChild(svgEl("text", { id: "cur-tag", fill: "#fff", "font-size": "8",
                                   "font-weight": "bold", x: "0", y: "0" }));
    g.appendChild(cab);
    svg.appendChild(g);
  }
  return g;
}

/* O último compasso de um sistema tem tinta além da caixa: a barra dupla de fim de sistema é
   desenhada 4,1 pt à direita do `x1` (`layout._draw_system`). Sem isso a faixa de destaque
   "corta" a barra do compasso que está tocando — e a medição no PDF rasterizado confirmou o Δ. */
function fimDeSistema(mapa, i) {
  const g = mapa[i], nx = mapa[i + 1];
  if (!nx) return true;
  return nx.page !== g.page || Math.abs(nx.x0 - g.x1) > 0.5;
}

function drawCursor(t) {
  const g = cursorCamada(false);
  const read = $("cur-read");
  const mapa = state.layout && state.layout.bar_map;
  if (!g || !mapa || !mapa.length) {
    if (read) read.textContent = "";
    return;
  }
  const gb = gradeBase();
  const p = posicaoCursor(t, mapa, gb.len, gb.t0);
  const band = g.querySelector("#cur-band"), line = g.querySelector("#cur-line");
  const cap = g.querySelector("#cur-cap"), box = g.querySelector("#cur-capbox"), tag = g.querySelector("#cur-tag");
  if (!p) {
    band.setAttribute("width", "0"); line.setAttribute("stroke-opacity", "0");
    cap.setAttribute("opacity", "0");
    if (read) read.textContent = "";
    return;
  }
  const vivo = !!state.playing;
  g.setAttribute("opacity", vivo ? "1" : "0.55");
  const y0 = p.g.y0, y1 = p.g.y1;
  const folga = p.g.folga || 0;
  const cauda = fimDeSistema(mapa, p.i) ? Math.max(folga, 5.0) : folga;
  band.setAttribute("x", (p.g.x0 - folga).toFixed(2)); band.setAttribute("y", y0.toFixed(2));
  band.setAttribute("width", Math.max(4, p.g.x1 - p.g.x0 + folga + cauda).toFixed(2));
  band.setAttribute("height", Math.max(8, y1 - y0).toFixed(2));
  line.setAttribute("x1", p.x.toFixed(2)); line.setAttribute("x2", p.x.toFixed(2));
  line.setAttribute("y1", (y0 + 1).toFixed(2)); line.setAttribute("y2", (y1 - 1).toFixed(2));
  line.setAttribute("stroke-opacity", "1");

  const sc = state.score || {};
  const beatPerBar = Math.max(1, Math.round((sc.ticks_per_bar || 32) / (sc.ticks_per_beat || 8)));
  const beat = Math.min(beatPerBar, Math.floor(p.fx * beatPerBar) + 1);
  const rot = (sc.meter || "4/4").split("/")[0];
  const txt = "compasso " + (p.i + 1) + " · " + beat + "/" + rot;
  tag.textContent = txt;
  cap.setAttribute("opacity", "1");
  const larg = 4.6 + txt.length * 4.4;
  let cx = p.g.x0, cy = y0 - 12;
  const pw = (state.layout.page_w || 842);
  if (cx + larg > pw - 6) cx = pw - 6 - larg;
  if (cy < 2) cy = y1 + 2;
  cap.setAttribute("transform", "translate(" + cx.toFixed(1) + " " + cy.toFixed(1) + ")");
  box.setAttribute("x", "0"); box.setAttribute("y", "0"); box.setAttribute("width", larg.toFixed(1));
  tag.setAttribute("x", "2.6"); tag.setAttribute("y", "8");

  if (read) {
    read.textContent = txt + " · " + fmt(t, 2) + " s" + (p.noFim ? " · fim" : "");
    read.title = "cursor sobre o compasso desenhado" + notaDaGrade() +
                 (state.mode === "synth" ? " · tocando a transcrição sintetizada"
                                        : " · tocando o arquivo");
  }
  if (p.k !== state.curBar) {
    state.curBar = p.k;
    const seguir = $("follow");
    if (vivo && seguir && seguir.checked) {
      const wrap = $("score-wrap");
      if (wrap && $("pane-score") && !$("pane-score").classList.contains("hidden")) {
        const escala = (wrap.clientWidth || 1) / (state.layout.page_w || 842);
        const topo = cy * escala, base = (y1 + 6) * escala;
        if (topo < wrap.scrollTop + 8 || base > wrap.scrollTop + wrap.clientHeight - 8) {
          wrap.scrollTo({ top: Math.max(0, topo - wrap.clientHeight * 0.35), behavior: "smooth" });
        }
      }
    }
  }
}

function cursorReset() {
  state.curBar = -1;
  cursorCamada(true);
  drawCursor(currentTime() == null ? (state.offset || 0) : currentTime());
}

/* clique na pauta move o ouvinte: a pergunta "onde estou?" funciona nos dois sentidos */
function seekNaPartitura(ev) {
  const mapa = state.layout && state.layout.bar_map;
  const holder = $("svg-holder");
  if (!mapa || !mapa.length || !state.duration || !holder) return;
  const svg = holder.querySelector("svg");
  if (!svg) return;
  let x = null, y = null;
  if (svg.getScreenCTM) {
    const m = svg.getScreenCTM();
    if (m) {
      const pt = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(m.inverse());
      x = pt.x; y = pt.y;
    }
  }
  if (x == null) {
    const r = svg.getBoundingClientRect();
    const escala = r.width / (state.layout.page_w || r.width || 1);
    x = (ev.clientX - r.left) / escala; y = (ev.clientY - r.top) / escala;
  }
  const achado = mapa.findIndex((e) => x >= e.x0 && x <= e.x1 && y >= e.y0 - 14 && y <= e.y1 + 2);
  if (achado < 0) return;
  const bx = mapa[achado], gb = gradeBase();
  const b = bx.i;
  // dentro do compasso também: a posição x escolhe a batida, não só a caixa
  const fx = (bx.beat1 > bx.beat0) ? Math.min(1, Math.max(0, (x - bx.beat0) / (bx.beat1 - bx.beat0))) : 0;
  const t = Math.max(0, gb.t0 + (b + fx) * gb.len);
  if (ev.shiftKey) {
    state.loopBar = (state.loopBar === b) ? null : b;
    setStatus(state.loopBar === null ? "loop desligado."
            : "repetindo o compasso " + (b + 1) + " (shift+clique de novo desliga).");
    drawWave();
    return;
  }
  state.offset = Math.min(state.duration, t);
  if (state.playing) startPlay(state.offset);
  drawCursor(state.offset); drawWave();
  setStatus("indo ao compasso " + (b + 1) + (fx > 0.02 ? " · batida " + (1 + Math.floor(fx * 4)) : "") +
            " (" + fmt(state.offset, 2) + " s).");
}

/* ------------------------------------------------------- a janela acompanha o cursor */
function segueJanela(t) {
  const cb = $("w-follow");
  if (!cb || !cb.checked || !t) return;
  const [a, b] = janelaAtual();
  const span = b - a, d = duracaoTotal();
  if (span >= d - 1e-6) return;                     // já mostra a faixa inteira
  if (t > b - span * 0.12 || t < a + span * 0.02) {
    let na = t - span * 0.15;
    na = Math.max(0, Math.min(na, d - span));
    setJanela(na, na + span);
  }
}

/* ------------------------------------------------------------------ piano-roll do .mid */
async function pedeMidi(forcar) {
  const sum = $("m-sum");
  if (!state.id) { if (sum) sum.textContent = "sem análise ainda — transcreva uma faixa primeiro."; return; }
  const chave = state.id + "|" + JSON.stringify((state.score.hide_lanes) || []) +
                "|" + (state.score.bars || []).length + "|" +
                ((state.score.bars || [])[0] || { hits: [] }).hits.length;
  if (!forcar && state.mid && chave === state.midChave) { desenhaRoll(); return; }
  if (state.midPedindo) return;
  state.midPedindo = true;
  if (sum) sum.textContent = "gerando o .mid e relendo nota a nota…";
  try {
    const r = await fetchT(API("midi_view"), {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: state.id }),
    }, TMO.medio);
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.error || ("erro " + r.status));
    state.mid = j; state.midChave = chave;
    const bd = $("midi-badge");
    if (bd) { bd.hidden = false; bd.textContent = String(j.n_arquivo); }
    if (sum) sum.textContent = j.n_arquivo + " notas lidas do arquivo · " + j.n_escritas_visiveis +
      " escritas na gravura · " + j.n_na_partitura + " na partitura (dados) · divisão " +
      j.division + " · " + fmt(j.bpm_arquivo, 1) + " BPM · " + fmt(j.bytes / 1024, 1) + " kB";
    const nt = $("m-note");
    if (nt) nt.textContent = (j.ocultas && j.ocultas.length
        ? "pistas ocultas na gravura: " + j.ocultas.join(", ") : "nenhuma pista oculta") +
      (j.truncado ? " · exibindo as primeiras " + j.limite + " de " + j.n_arquivo : "") +
      (j.sem_off ? " · " + j.sem_off + " nota(s) sem note-off (duração assumida)" : "") +
      (j.lead_ms > 500 ? " · o áudio tem " + fmt(j.lead_ms / 1000, 1) +
       " s antes do primeiro tempo: desloque o .mid em " + Math.round(j.lead_ms) +
       " ms na DAW para bater com a gravação" : "");
    desenhaRoll();
  } catch (e) {
    if (sum) sum.textContent = "não consegui ler o MIDI: " + e.message;
    if (e && /timeout|abort/i.test(e.message || "")) setStatus("a leitura do MIDI demorou; tente de novo.", "warn");
  } finally { state.midPedindo = false; }
}

function desenhaRoll() {
  const cv = $("roll");
  if (!cv) return;
  const box = caixaDe(cv);
  const dpr = window.devicePixelRatio || 1;
  const H = 380;
  cv.width = Math.max(600, Math.floor(box.width * dpr));
  cv.height = Math.floor(H * dpr);
  const g = contexto2d(cv);
  if (!g) return;
  g.setTransform(1, 0, 0, 1, 0, 0); g.scale(dpr, dpr);
  const W = cv.width / dpr;
  g.fillStyle = "#12161c"; g.fillRect(0, 0, W, H);
  const m = state.mid;
  const GU = 64;                                       // sangria com os nomes das peças
  if (!m || !m.notas) {
    g.fillStyle = "#5d6875"; g.font = "13px system-ui";
    g.fillText(m ? "sem notas no arquivo" : "gere o MIDI — a leitura aparece aqui", 12, H / 2);
    return;
  }
  const [a, b] = janelaAtual();
  const span = Math.max(1e-6, b - a);
  const t2x = (t) => ((t - a) / span) * (W - GU);
  const p0 = m.pitch_faixa[0], p1 = m.pitch_faixa[1];
  const np = Math.max(1, p1 - p0 + 1);
  const topo = 18, alt = H - topo - 16;
  const passo = alt / np;
  /* linhas por altura (com o nome da peça que soa ali) */
  const porPitch = {};
  m.notas.forEach((n) => { porPitch[n[2]] = n[4]; });
  g.font = "9.5px ui-monospace,monospace";
  for (let p = p0; p <= p1; p++) {
    const y = topo + alt - (p - p0 + 1) * passo;
    const lane = porPitch[p];
    g.fillStyle = lane ? hexA(laneColor(lane), 0.09) : (p % 2 ? "#171c23" : "#141920");
    g.fillRect(GU, y, W - GU, Math.max(1, passo));
    if (lane) {
      g.fillStyle = hexA(laneColor(lane), 0.9);
      g.textAlign = "right";
      g.fillText((state.lanes.find((x) => x.id === lane) || { short: lane }).short + " " + p,
                 GU - 6, y + Math.max(4, passo * 0.78));
    }
  }
  g.textAlign = "left";
  /* grade do pulso, na mesma janela da aba Áudio */
  const gb = gradeBase();
  if (gb.len > 0) {
    const tpb = state.score.ticks_per_beat || 8, tpr = state.score.ticks_per_bar || tpb * 4;
    const spb = gb.len / (tpr / tpb);
    for (let k = Math.floor((a - gb.t0) / spb) - 1; k <= Math.ceil((b - gb.t0) / spb) + 1; k++) {
      const t = gb.t0 + k * spb;
      if (t < a - spb || t > b + spb) continue;
      const x = GU + t2x(t);
      const isBar = ((k % (tpr / tpb)) + (tpr / tpb)) % (tpr / tpb) === 0;
      g.strokeStyle = isBar ? "rgba(255,255,255,.26)" : "rgba(255,255,255,.09)";
      g.lineWidth = isBar ? 1 : 0.6;
      g.beginPath(); g.moveTo(x, topo - 8); g.lineTo(x, topo + alt); g.stroke();
      if (isBar) {
        g.fillStyle = "rgba(255,255,255,.4)"; g.textAlign = "center";
        g.fillText(String(Math.floor((t - gb.t0) / gb.len) + 1), x, 11); g.textAlign = "left";
      }
    }
  }
  /* as notas, como o arquivo as tem */
  m.notas.forEach((n) => {
    const x = GU + t2x(n[0]);
    if (x < GU - 40 || x > W + 20) return;
    const y = topo + alt - (n[2] - p0 + 1) * passo;
    const w = Math.max(2, (n[1] / span) * (W - GU));
    const cor = laneColor(n[4]) || "#9ab";
    g.fillStyle = hexA(cor, 0.45 + 0.55 * Math.min(1, (n[3] || 0) / 127));
    g.fillRect(x, y + 0.5, w, Math.max(2, passo - 1));
    g.strokeStyle = hexA(cor, 0.95); g.lineWidth = 0.7;
    g.strokeRect(x + 0.5, y + 0.5, Math.max(1, w - 1), Math.max(1, passo - 2));
  });
  /* cursor */
  const t = currentTime();
  if (t !== null) {
    const x = GU + t2x(t);
    if (x >= GU && x <= W) {
      g.strokeStyle = "#f5d97a"; g.lineWidth = 1.4;
      g.beginPath(); g.moveTo(x, topo - 10); g.lineTo(x, topo + alt); g.stroke();
    }
  }
  state._roll = { GU: GU, topo: topo, alt: alt, passo: passo, p0: p0, t2x: t2x, x2t: (x) => a + ((x - GU) / (W - GU)) * span };
}
function hexA(hex, alfa) {
  const h = String(hex || "#888").replace("#", "");
  const n = h.length === 3 ? h.split("").map((c) => c + c).join("") : h;
  const r = parseInt(n.slice(0, 2), 16), g = parseInt(n.slice(2, 4), 16), b = parseInt(n.slice(4, 6), 16);
  return "rgba(" + r + "," + g + "," + b + "," + alfa + ")";
}

/* ------------------------------------------------------------------ apresentação */
const PRATOS_PADRAO = ["hat", "hat_open", "hat_foot", "ride", "crash", "splash", "cowbell"];

function pistasOcultas() {
  const pr = (state.report && state.report.apresentacao && state.report.apresentacao.pratos) || PRATOS_PADRAO;
  if (!state.pres.simples) return [];
  return state.pres.pratos ? [] : pr.slice();
}
function atualizaPres() {
  const a = $("pres-simple"), f = $("pres-full"), c = $("pres-pratos"), n = $("pres-note");
  if (a) a.classList.toggle("active", !!state.pres.simples);
  if (f) f.classList.toggle("active", !state.pres.simples);
  if (c) c.checked = !!state.pres.pratos;
  const oc = pistasOcultas();
  if (n) n.textContent = oc.length ? (oc.length + " pistas ocultas na gravura")
                                     : (state.pres.simples ? "simples, com pratos" : "tudo à mostra");
  if (c) c.title = "mostrar/ocultar " + PRATOS_PADRAO.join(", ") + " — só na gravura, MIDI e MusicXML";
}
async function aplicaView(msg) {
  if (!state.id) {
    setStatus("rode uma análise primeiro — ocultar pistas é uma escolha sobre o resultado.", "err");
    atualizaPres();
    return;
  }
  const oc = pistasOcultas();
  try {
    const r = await fetchT(API("view"), {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: state.id, hide_lanes: oc }),
    }, TMO.medio);
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.error || ("erro " + r.status));
    adopt(j, { manterJanela: true, semQA: true });          // a janela e o zoom ficam onde estavam
    invalidarMidi();
    if (abaAtiva() === "midi") pedeMidi(false);
    const ap = (j.report && j.report.apresentacao) || {};
    const nd = (ap.n_dados || 0);
    setStatus(msg || ("gravura reescrita — " + (oc.length ? "ocultas " + oc.join(", ") + "; " : "") +
              "os dados seguem com " + nd + " notas (JSON/CSV e auditoria não mudam)."), "ok");
  } catch (e) {
    setStatus("não consegui regravurar: " + e.message, "err");
  }
}

/* ------------------------------------------------------------------ sintonia medida */
async function pedirTune() {
  if (state.tunePedindo) return;
  if (!state.id) {
    $("tune-out").hidden = false;
    $("tune-out").innerHTML = "<p class='hint'>ainda não há faixa analisada — a recomendação é " +
      "medida <b>na sua faixa</b>, não tabelada. Transcreva (ou use a demonstração) e clique de novo.</p>";
    return;
  }
  state.tunePedindo = true;
  const dur = (state.report && state.report.file && state.report.file.duration_sec) || 0;
  $("tune-st").textContent = "medindo… (" + Math.round(17 * Math.max(1, dur) * 0.25) + " s previstos, até 17 análises)";
  busy(true, "varredura de sensibilidade × confiança na sua faixa (até ~17 análises)…");
  try {
    const r = await fetchT(API("tune"), {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: state.id }),
    }, TMO.pesado);
    const j = await r.json();
    if (!r.ok || !j.ok) {
      $("tune-out").hidden = false;
      $("tune-out").innerHTML = "<p class='hint warn'>" + esc(j.error || ("erro " + r.status)) + "</p>";
      setStatus("a recomendação não rodou: " + esc(j.error || ("erro " + r.status)), "warn");
    } else {
      state.tune = j.tune;
      renderTune(j.tune, j.cache);
      setStatus("varredura pronta: " + j.tune.candidatos.length + " pontos medidos em " +
                fmt(j.tune.referencia.segundos, 1) + " s.", "ok");
    }
  } catch (e) {
    $("tune-out").hidden = false;
    $("tune-out").innerHTML = "<p class='hint warn'>a varredura não terminou (" + esc(e.message) +
      ") — o servidor continua de pé; tente com um trecho mais curto.</p>";
  } finally {
    state.tunePedindo = false; $("tune-st").textContent = ""; busy(false);
  }
}
function esc(x) {
  return String(x == null ? "" : x).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function renderTune(t, cache) {
  const el = $("tune-out");
  if (!el) return;
  el.hidden = false;
  const rec = t.recomendado || {};
  const atual = (t.referencia && t.referencia.atual) || {};
  const muda = String(rec.sensitivity) !== String(atual.sensitivity) ||
               String(rec.min_confidence) !== String(atual.min_confidence);
  const linhas = (t.candidatos || []).slice(0, 8).map((c) => {
    const escolhido = c.sensitivity === rec.sensitivity && c.min_confidence === rec.min_confidence;
    return "<tr class='" + (escolhido ? "sel" : "") + (c.custo > 0.4 ? " ruim" : "") + "'>" +
      "<td class='mono'>" + fmt(c.sensitivity, 2) + "</td><td class='mono'>" + fmt(c.min_confidence, 2) +
      "</td><td class='mono'>" + (c.custo <= 1e-9 ? "zero" : fmt(c.custo, 2)) + "</td>" +
      "<td class='mono'>" + c.golpes + "</td><td class='mono'>" + fmt(c.p1_gap_ms, 0) +
      "</td><td class='mono'>" + fmt(c.resid_ms, 1) + "</td><td class='mono'>" +
      fmt(100 * (c.offgrid || 0), 0) + "%</td><td class='mono'>" + (c.erros_audit || 0) + "</td></tr>";
  }).join("");
  el.innerHTML =
    "<p class='mono small'>varredura: " + t.referencia.celulas + " análises em " +
      fmt(t.referencia.segundos, 1) + " s" + (cache ? " (resultado em cache)" : "") +
      " · grade " + t.referencia.grade_sens.map((x) => fmt(x, 2)).join("/") + " × " +
      t.referencia.grade_conf.map((x) => fmt(x, 2)).join("/") + "</p>" +
    "<p class='tune-rec'>" + (muda
      ? "<b>recomendado:</b> sensibilidade <span class='mono'>" + fmt(rec.sensitivity, 2) +
        "</span> · confiança <span class='mono'>" + fmt(rec.min_confidence, 2) + "</span> " +
        "(hoje " + fmt(atual.sensitivity, 2) + " / " + fmt(atual.min_confidence, 2) + ")"
      : "o par de hoje (<span class='mono'>" + fmt(atual.sensitivity, 2) + " / " +
        fmt(atual.min_confidence, 2) + "</span>) já é o melhor ponto medido nesta faixa") + "</p>" +
    "<ul class='tune-motivos'>" + (t.motivos || []).map((m) => "<li>" + esc(m) + "</li>").join("") + "</ul>" +
    "<table class='tune-tabela'><thead><tr><th>sens</th><th>conf</th><th>custo</th><th>golpes</th>" +
    "<th>lacuna ms</th><th>resíduo ms</th><th>fora grade</th><th>erros</th></tr></thead>" +
    "<tbody>" + linhas + "</tbody></table>" +
    "<p class='hint'>" + (muda
      ? "Custo é penalidade adimensional (duplicata, grade, densidade, auditoria) — sem gabarito." +
        " Aplicar re-analisa a faixa inteira."
      : "Nada aplicado: manteria a mesma análise.") + "</p>" +
    (muda ? "<div class='row gap'><button class='primary small' id='tune-apply' type='button'>aplicar " +
            fmt(rec.sensitivity, 2) + " / " + fmt(rec.min_confidence, 2) + " e retranscrever</button>" +
            "<button class='ghost small' id='tune-keep' type='button'>manter como está</button></div>" : "");
  const ap = $("tune-apply");
  if (ap) ap.addEventListener("click", () => {
    $("sens").value = String(rec.sensitivity); $("minc").value = String(rec.min_confidence);
    syncLabels();
    setStatus("aplicado — retranscrevendo com " + fmt(rec.sensitivity, 2) + " / " +
              fmt(rec.min_confidence, 2) + "…");
    analyze();
  });
  const kp = $("tune-keep");
  if (kp) kp.addEventListener("click", () => { el.hidden = true; });
}

/* ------------------------------------------------------------------ interação da onda */
function onOndaDown(ev) {
  if (!state.duration) return;
  const cv = $("wave");
  const r = cv.getBoundingClientRect();
  if (!state._x2t) return;
  const escala = (state._ondaW || r.width) / (r.width || 1);
  const x = (ev.clientX - r.left) * escala;
  const t = state._x2t(x);
  state.arrasto = { x0: ev.clientX, t0: t, a: janelaAtual()[0], b: janelaAtual()[1],
                    modo: ev.shiftKey ? "sel" : "pan", r: r, escala: escala, mexeu: false };
  if (ev.shiftKey) state.sel = { a: t, b: t };
  ev.preventDefault();
}
function onOndaMove(ev) {
  const ar = state.arrasto;
  if (!ar) return;
  const dx = ev.clientX - ar.x0;
  if (Math.abs(dx) > 3) ar.mexeu = true;
  if (!ar.mexeu) return;
  const W = state._ondaW || 900;
  const span = ar.b - ar.a;
  if (ar.modo === "sel") {
    const r = $("wave").getBoundingClientRect();
    const x = (ev.clientX - r.left) * ((state._ondaW || r.width) / (r.width || 1));
    state.sel = { a: ar.t0, b: state._x2t(x) };
    desenhaOndas();
    return;
  }
  const dt = -(dx / W) * span;
  let na = ar.a + dt, nb = ar.b + dt;
  const d = duracaoTotal();
  if (na < 0) { nb -= na; na = 0; }
  if (nb > d) { na -= (nb - d); nb = d; }
  setJanela(Math.max(0, na), nb);
}
function onOndaUp(ev) {
  const ar = state.arrasto;
  state.arrasto = null;
  if (!ar) return;
  if (!ar.mexeu) { seekFromClick({ clientX: ev.clientX, shiftKey: ev.shiftKey, target: $("wave") }); return; }
  if (ar.modo === "sel") {
    if (state.sel && Math.abs(state.sel.b - state.sel.a) > 0.004) aplicarSelecao();
    else { state.sel = null; desenhaOndas(); }
  }
}
function aplicarSelecao() {
  if (!state.sel) return;
  const a = Math.min(state.sel.a, state.sel.b), b = Math.max(state.sel.a, state.sel.b);
  state.sel = null;
  setJanela(a, b);
  setStatus("zoom na seleção: " + fmt(a, 3) + "–" + fmt(b, 3) + " s (" + fmt((b - a) * 1000, 1) +
            " ms), " + Math.round(duracaoTotal() / (b - a)) + "×.");
}
function teclasZoom(ev) {
  const aba = abaAtiva();
  if (aba !== "audio" && aba !== "midi") return;
  if (/input|select|textarea/i.test((ev.target && ev.target.tagName) || "")) return;
  const [a, b] = janelaAtual();
  const c = (a + b) / 2, span = b - a, d = duracaoTotal();
  if (ev.key === "+" || ev.key === "=") { zoomNaJanela(c, 1 / 1.6); ev.preventDefault(); }
  else if (ev.key === "-" || ev.key === "_") { zoomNaJanela(c, 1.6); ev.preventDefault(); }
  else if (ev.key === "0") { setJanela(0, d); state.zoom = 1; aplicaLarguraSVG(); ev.preventDefault(); }
  else if (ev.key === "[") { setJanela(Math.max(0, a - span * 0.5), Math.max(span, b - span * 0.5)); ev.preventDefault(); }
  else if (ev.key === "]") { setJanela(a + span * 0.5, Math.min(d, b + span * 0.5)); ev.preventDefault(); }
  else if (ev.key === "b" || ev.key === "B") { enquadrarCompasso(); ev.preventDefault(); }
}
function onRoda(ev) {
  if (!state.duration) return;
  ev.preventDefault();
  const cv = $("wave"), r = cv.getBoundingClientRect();
  if (!state._x2t) return;
  const x = (ev.clientX - r.left) * ((state._ondaW || r.width) / (r.width || 1));
  const t = state._x2t(x);
  if (ev.shiftKey) {
    const [a, b] = janelaAtual(), span = b - a;
    const dx = (ev.deltaY !== 0 ? ev.deltaY : ev.deltaX) / 4 * (span / (r.width || 900));
    setJanela(a + dx, b + dx);
    return;
  }
  zoomNaJanela(t, ev.deltaY < 0 ? 1 / 1.35 : 1.35);
}

/* rede de segurança: promessa rejeitada sem catch não pode deixar "analisando…" na tela */
window.addEventListener("unhandledrejection", (ev) => {
  try {
    busy(false);
    setStatus("erro inesperado: " + ((ev.reason && ev.reason.message) || ev.reason || "?"), "err");
    probe();
  } catch (e) { /* nada mais a fazer */ }
});
window.addEventListener("error", () => { try { busy(false); } catch (e) {} });

boot();
