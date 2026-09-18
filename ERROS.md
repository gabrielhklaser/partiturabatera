# Erros confirmados, causas raiz e limitações assumidas

Registro do que a **validação dupla** (`drumscribe/qa.py` + `tests/doublecheck.py`) encontrou de
verdade neste projeto — defeitos do produto, defeitos do próprio auditor e limitações que ficam
visíveis. A regra de trabalho: **todo achado ou é consertado na fonte ou é registrado aqui como
limitação; nada é silenciado** (um verificador que só acusa o que o produto já sabe não serve
para nada).

Como reproduzir:

```bash
python3 tests/doublecheck.py              # audit no demo (wav + mp3) + injeção de 7 defeitos
python3 tests/doublecheck.py --fix        # idem, deixando o agente reparar
python3 tests/doublecheck.py --file sua.faixa.wav --level full --out out/qa
python3 tests/selfcheck.py                # 40 verificações de ponta a ponta
python3 tests/evaluate.py                 # acurácia contra o gabarito
python3 tests/calibrate_tonal.py          # mede o limiar de "é bateria isolada?"
```

Estado atual: `doublecheck` termina com *validação dupla completa: nenhum erro remanescente*
(0 erros, 0 avisos, 27 verificações por arquivo, reparo idempotente, partitura lícita preservada);
`selfcheck` 40/0; `evaluate` recall 0.919 · precisão 0.935 · **pos_accuracy 1.000** ·
tick_mae 0.00 · erro de BPM 0.01 %.

---

## A. Defeitos do produto encontrados pela auditoria (consertados na fonte)

### A1. Nota escrita um tempo inteiro longe do áudio (6 % das notas)
- **Sintoma:** `time` do golpe e o par `(compasso, tick)` gravado discordavam em exatamente
  8 ticks (um semínima) num subconjunto de eventos — a nota era escrita no lugar errado e a
  sobreposição na forma de onda / o playhead da audição ficavam deslocados.
- **Causa raiz:** `grid.quantize()` reconstruía a posição como
  `floor(pos/tpb)·tpb + grade[slot]`. O índice `slot` diz *qual* posição da grade venceu, mas não
  diz *em qual beat*; quando o melhor slot caía na borda do beat (o vencedor era o `0` do beat
  seguinte ou o último do anterior), o beat se perdia — erro de um tempo, sistemático, em todos os
  eventos entre duas posições da grade.
- **Como a auditoria viu:** T3 (rigidez) medindo `pos ↔ compasso·tpr + tick` por golpe, sem
  confiar em nada do que o pipeline gravou. O `tick_mae` do gabarito **não** pegava isso (os
  eventos afetados eram mostly rótulos de fallback, fora do conjunto casado) — mais um argumento
  para uma lei estrutural além da métrica de acurácia.
- **Conserto:** encaixe em coordenada **absoluta** — varre as posições da grade nos beats
  anterior, atual e seguinte e escolhe a mais próxima; `ticks_abs` e `resid` passam a viver no
  mesmo espaço. Promoção a 32º reescrita sobre a mesma base.
- **Prova:** `pos_accuracy` 0.969 → **1.000**; T3 com dispersão 0,183 tick e máx 0,985 (limite
  1,88 tick); histograma de ofensores vazio em wav e mp3.

### A2. Ligadura emendando peças diferentes
- **Sintoma:** o MusicXML saía com 3 `<tie type="start">` e 2 `<tie type="stop">`.
- **Causa raiz:** `_mark_sustain()` esticava o ride até o próximo ataque **da voz 1** e ligava a
  nota seguinte — que era o chimbal. No MusicXML `tie` é atributo da *nota*; a voz só governa a
  linha do tempo. Ligar ride→chimbal emenda dois instrumentos, aumenta a duração soada do ride e
  quebra a simetria start/stop (o notador reclama na importação).
- **Como a auditoria viu:** T11, que regrava e **relê** o XML com `xml.etree` e conta as ligaduras.
- **Conserto:** em `_mark_sustain`, só liga se o destino é a **mesma pista**; `_valid_ties` passou
  a chavear por `(voz, compasso, tick, pista)` para o par não poder ser herdado por um acorde.
- **Prova:** T8 e T11 em `info/ok`; contagem de `tie` simétrica nos dois arquivos; PDF/MIDI
  inalterados (duração vem do tick, não da ligadura).

### A3. Relatório por peça contando a lista errada
- **Sintoma:** `report.events.per_lane[lane].count` ≠ notas gravadas na pauta (63 vs 65 no bumbo).
- **Causa raiz:** a contagem era feita sobre `keep_hits`, **antes** de `_merge_hit` fundir dois
  golpes do mesmo slot numa nota só com `<chord/>`. O painel mostrava ao usuário mais golpes do que
  a partitura tinha.
- **Como a auditoria viu:** T15c (relatório × pauta recontada).
- **Conserto:** `per_lane` passa a ser contado **na partitura final**, mantendo `n_detections`
  separado — assim a diferença entre os dois números existe, mas fica *explicada* em vez de
  silenciosa.
- **Prova:** T15c `info/ok` com `soma_relatorio == notas_pauta`.

### A4. `rebuild_score` fabricava evidência (`time = 0.0`)
- **Sintoma:** toda nota editada pelo editor (ou injetada) ganhava `time = 0.0`, como se o áudio
  tivesse um golpe no segundo zero.
- **Consequência grave:** o agente de correção, lendo esse tempo inventado, *movia* uma nota lícita
  para o tick 0; a deduplicação seguinte colapsava as duas e a música era perdida — perda silenciosa
  causada pelo reparo, não pela transcrição.
- **Como a auditoria viu:** lei nova no banco de testes — comparar o **multiconjunto** de
  `(compasso, tick, peça)` lícitos antes e depois do reparo: nota lícita única tem de sobreviver;
  duplicata (ilegal por T4) tem de colapsar exatamente em uma.
- **Conserto:** `rebuild_score` preserva `time = None` quando não houve medição; `Hit.time` é
  `Optional[float]`; as leis que usam o instante (`T3`, `T3e`, `T3d`, projeção do reparo) pulam
  `None` em vez de lê-lo como zero. A UI já tratava ausência (`if (!h.time) return`).
- **Prova:** `nenhuma nota lícita perdida (293 posições preservadas, 1 duplicata colapsada)`.

### A5. O CSV do editor existia só dentro do servidor
- **Sintoma:** nada verificava o formato que o editor lê de volta, porque o CSV era montado no
  meio da rota HTTP.
- **Conserto:** extraído para `pipeline.score_to_csv()` / `score_to_csv_bom()` — servidor e
  auditoria (`T14`) usam a mesma função; se a serialização perder um campo, T14 acusa.

### A6. T3 era cego a andamento levemente errado
- **Sintoma:** BPM 0,3 % errado passava limpo: a rigidez ponto a ponto continua ótima nos primeiros
  compassos e o resíduo só explode no fim da música.
- **Conserto:** lei de **deriva** no T3 — regressão linear do resíduo contra o número do compasso;
  erro quando `|inclinação| × nº de compassos > max(0,6; 0,5 × tol)`.
- **Calibração (medida, não chutada):** perturbando só o BPM da faixa demo: +0,1 % → deriva 0,15
  tick; +0,3 % → inclinação 0,073 tick/compasso, deriva 1,24; +0,6 % → deriva 2,86. Ou seja, o
  verificador sente 0,3 % de erro de andamento.

### A7. O reparo podia criar o defeito que consertava
- **Sintoma 1:** o `do` do `repair()` não incluía `normalize_fields` → a proposta
  `fix="normalize_fields"` era anunciada e nunca executada (articulação `blast` sobrevivendo).
- **Sintoma 2:** o cursor de voz, ao recortar durações, **empurrava** a nota para frente, criando
  colisão de slot → a deduplicação engolia notas vizinhas lícitas.
- **Sintoma 3:** o laço de ponto fixo logava só a **última** passada (a vazia) e descartava o
  log inteiro — o agente dizia que não havia feito nada depois de ter feito tudo.
- **Sintoma 4:** o limite do reparo (`max(2, tol)`) era outro do limite do verificador
  (`1,35 × tol`), então um na fronteira ficava acusando erro para sempre e o reparo oscilava
  (1 nota → 2 notas → …).
- **Conserto:** um único `grid_bound_ticks()` compartilhado pelo auditor e pelo agente; recorte só
  de **duração** (posição de nota não é negociável — é o instante em que o baterista bateu);
  projeção na malha **condicionada** a `fix="requantize_from_clock"` proposta; contagens acumuladas
  no laço; ponto fixo `limpar → re-encaixar → limpar` com teto de 4 iterações; e re-auditoria
  dentro do próprio `repair()` (roda o auditor sobre o resultado e repassa o que ainda for
  consertável) — é isso que torna a saída auto-consistente por construção.
- **Prova:** `reparo idempotente` + `re-auditoria limpa` + `partitura íntegra: nenhuma alteração`
  no `tests/doublecheck.py`.

### A8. Falsos positivos do próprio T15/T3 já corrigidos
Vale estar registrado porque são as armadilhas deste domínio:
- marcar **toda** detecção sem nota como golpe perdido: `_merge_hit` e o 𝄄 apagam notações
  deliberadamente — a lei passou a ser "*existe* um `(compasso,tick)` para onde isso vai".
- acusar o `𝄄` de "repetição não fiel" sem contar as pausas/excluir os acordes do XML, e comparar
  fitness de grade entre BPMs diferentes (a grade mais fina sempre ganha).
- exigir soma de compasso por voz **contando** `<chord/>`: acorde não avança o cursor — a soma e a
  posição só contam notas primárias.
- julgar a *posição absoluta* da malha: a numeração pode começar em qualquer barra, então o T3
  remove a mediana (origem) e julga só a rigidez; e o **sinal** do `tick_offset` é detalhe interno
  do pipeline — o auditor prova os dois e fica com o mais rígido (com sinal errado, todos os 289
  eventos saíam ~12 ticks off: artefato puro).
- `1 − flatness espectral` classifica faixa só-de-bateria como tonal (0,852), e razão
  cauda/ataque "sustain" separava só 1,46×: a medida que funciona é a **profundidade de modulação**
  da banda 300–3000 Hz (`mid_modulation`), calibrada por `tests/calibrate_tonal.py`.

### A10. O véu de "analisando…" nunca saía — a interface inteira ficava inerte (o pior defeito do projeto)
- **Sintoma relatado:** "abre e fica na tela *analisando…* e nunca abre. Atrás dá pra ver o layout,
  mas não consigo sair dessa tela".
- **Causa raiz:** `static/style.css` tinha `.busy{position:fixed;inset:0;display:grid;z-index:60}` e
  `index.html` escondia o véu com o **atributo** `hidden`. A folha do autor vence a folha do
  navegador, e a regra do navegador é `[hidden]{display:none}` — mesma especificidade, origem
  inferior. `display:grid` na classe ⇒ `hidden` não fazia nada. O véu era pintado **sempre**, e como
  é uma camada full-screen com `z-index:60`, ela **interceptava todos os cliques**: nem trocar de aba
  era possível. Mesmo motivo para o selo `#qa-badge` aparecer como pílula vazia (`.badge{display:inline-block}`).
- **Por que nenhum teste viu:** o `doublecheck` mede áudio, pauta, XML, MIDI, CSV e PDF — nada disso
  toca em caixa de modelo. HTTP devolve 200, o DOM stub do `ui_boot_check` não resolve CSS. A falha
  era 100% visual e 100% paralisante: o exemplo clássico de que "validar os dados" não é "validar o
  produto".
- **Conserto (em camadas):**
  1. regra global `[hidden]{display:none!important}` no topo do CSS — vale para os três elementos de
     hoje e para qualquer um que vier;
  2. o véu ganhou via de escape: botão **ocultar**, clique no fundo e **Esc** chamam `hideBusy()` — a
     operação continua em voo, mas nenhuma tela pode ficar presa de novo;
  3. `busy()` passou a ignorar elemento ausente (nada de `TypeError` silencioso no caminho de erro);
  4. teste de invariante `tests/ui_css_check.py`: para cada elemento escondido por atributo, nenhuma
     classe com `display:` pode existir sem um `.classe[hidden]` ou a regra global; abas/`data-tab`
     conferidas; toda função que liga o véu tem de desligá-lo; e o teste é **provado por regressão** —
     sem a regra global ele falha citando `#busy.busy` e `#qa-badge.badge`.
- **Segundo round (o usuário continuou preso):** o conserto acima estava certo no arquivo, mas o
  navegador dele trouxe o `style.css` **de antes** do cache. Sinais que entregaram isso: o botão
  *ocultar* aparecia (é HTML estático, não JS) com o texto padrão *"analisando…"* (o JS nunca o
  escreveu) e não fazia nada — porque `hideBusy()` só agia `if (!b.hidden)`, e naquele estado o
  atributo dizia "escondido" enquanto a folha velha pintava o véu. Lição de projeto: **o estado
  inicial de uma camada que bloqueia a página não pode depender de um recurso externo cacheável.**
- **Conserto definitivo:**
  1. `<style>` **dentro do `<head>`** com `[hidden]{display:none!important}` e `#busy{display:none}`
     — viaja com o HTML, não pode divergir dele;
  2. `#busy` nasce com `style="display:none"` inline — nenhum stylesheet, velho ou novo, vence isso;
  3. `busy()/hideBusy()` passaram a escrever `style.display` (`setVeil`) em vez de mexer só no
     atributo, e `hideBusy()` age em **qualquer** estado incoerido (antes ele se recusava a agir
     justamente no estado em que o usuário estava);
  4. `boot()` força o véu para `display:none` como primeiro ato e marca `window.__drumscribe_ready`;
  5. cão de guarda inline no fim do `<body>`: se o `app.js` não rodar em 2,5 s (cache velho, arquivo
     bloqueado, erro de parse), ele mesmo esconde o véu e escreve no status o que fazer — a página
     nunca fica inclicável por causa de script;
  6. **carimbo de versão nas URLs** (`static/style.css?v=f84c67c8c1`), gerado de (mtime, tamanho) por
     `_asset_stamp()` em `server.py`, com o HTML em `Cache-Control: no-store`. Assim o cache deixa
     de ser uma variável: mudou o arquivo, mudou a URL.
- **O que passa a ser impossível:** (a) véu cobrindo a página no primeiro paint — o HTML sozinho o
  esconde; (b) correção visual presa atrás de cache — URL nova por construção; (c) botão de sair
  inútil — `hideBusy()` não depende mais do atributo estar coerente; (d) tela presa sem JS — o cão
  de guarda do documento resolve.

---

### A9. Servidor indisponível virava "carregando para sempre" na tela
- **Sintoma relatado:** a plataforma "fica somente carregando".
- **Causa 1 (ambiente):** o processo do servidor não sobrevive ao reinício do ambiente, e o
  `pip install` também não (pacotes ficam fora do snapshot) — o `run.sh` cobre os dois, mas nada
  disso era visível para quem estava do outro lado da tela.
- **Causa 2 (defeito do produto):** `boot()` fazia `await fetch("api/lanes")` **antes** de montar a
  interface e nenhum `fetch` tinha prazo. Com o backend morto, a promessa pode ficar pendurada:
  `wire()` nunca rodava, nenhum botão funcionava e o overlay de "analisando…" girava indefinidamente
  — indistinguível de "a análise é lenta".
- **Conserto:** `wire()` primeiro, rede depois; `fetchT()` com orçamento por chamada
  (sonda 4 s · rápido 10 s · médio 3 min · pesado 15 min) que distingue *conexão recusada* de
  *sem resposta*; faixa de estado `#offline` dizendo o que fazer (`./run.sh`) ou **quais**
  bibliotecas faltam, com re-tentativa automática a cada 10 s; `GET /api/health` como sonda barata
  que responde mesmo com dependência ausente (ela mesma reporta `faltando`); e um handler de
  `unhandledrejection` que solta o spinner — nenhum caminho pode deixar a tela travada.
- **Descartado por medida (para não culpar o que não é):** o servidor não é o gargalo nem bloqueia a
  interface — com três análises concorrentes (21 s cada, 63 s de CPU no total, logo realmente
  paralelas) a sonda e o estático responderam entre 0,002 s e 1,0 s; `threaded=True` e `0.0.0.0`
  confirmados na rota de `main()`. Também não há recurso externo no front (o iframe do preview não
  tem rede): `tests/ui_boot_check.js` verifica que `app.js` não referencia nenhuma URL absoluta.
  E a URL pública do preview exige o token de tráfego do ambiente — um `curl` cru de dentro do
  sandbox recebe `403` do proxy, o que **não** é falha da aplicação.
- **Prova:** `node tests/ui_boot_check.js` carrega o `app.js` real num DOM mínimo e encena os quatro
  modos de falha (recusa · deps ausentes · saudável · pendurado) mais o prazo estourando em 4,0 s;
  `python3 tests/http_qa_check.py` cobre a sonda e as rotas de auditoria contra o servidor de pé.
- **Regressão dirigida (o teste precisa gritar quando o conserto some):** com a regra global riscada
  do `style.css`, `ui_css_check.py` aponta `#qa-badge.badge` e `#busy.busy`; com a regra riscada
  também do `<head>` e do atributo inline, ele aponta os dois blocos de invariantes. Rodado e
  confirmado nas duas direções — é assim que se sabe que o teste mede o que diz medir.

### A11. O cursor de reprodução mostrava o compasso errado — `phase_ms` não é a origem da grade
- **Sintoma:** tocando o arquivo original, a partitura não acompanhava o áudio. O leitor de compasso
  e o *loop* (shift+clique) chegavam a um compasso que já tinha passado; a grade de batidas pintada na
  forma de onda também não batia com nenhuma nota.
- **Causa raiz:** a frente do app convertia tempo ↔ grade usando `report.tempo.phase_ms`
  (mais `file.trim_lead_ms`), isto é, a **fase da autocorrelação** do estimador de andamento — que não
  é a origem da grade *quantizada*. A origem verdadeira é `origem = mediana(t_n − (compasso·tpr + tick)·spt)`
  sobre as notas medidas, que `qa.clock_map` já ajusta (e a auditoria T3 já confere). No demo:
  `phase_ms = 1944,71 ms` contra origem medida **−6,7 ms** → **1.951 ms = 3,25 batidas** de erro,
  iguais para todos os 17 compassos. Não era "um compasso fora": era um deslocamento sistemático.
- **Por que nenhum teste via:** `evaluate.py` mede *acerto de eventos* (recall, precisão, `tick_mae`)
  e as leis de `qa.py` conferem a partitura contra si mesma e contra os *instantes medidos* — nenhuma
  delas compara o **desenho** (onde o compasso foi traçado) com o **tempo do arquivo**. Sincronismo de
  reprodução simplesmente não tinha oráculo. Um `curl` na resposta também não mostra nada: o `phase_ms`
  é um número plausível e o relatório é legítimo; só a *interpretação* dele como origem estava errada.
- **Como apareceu:** ao exigir do código um cross-check — para cada nota, o índice de compasso que o
  cursor calcula tem de ser o índice em que a partitura a escreve. Deu 17/17 "desvio"; e aí se viu que
  a **expectativa do script** também estava errada (ela assumia `t = phase + b·bar_len`), o que só
  reforça a lição: derivar a expectativa das notas medidas, não de uma fórmula copiada do código testado.
- **Conserto (a origem passa a vir de um só lugar, auditado):**
  1. `server._cursor_clock(score, report)` ajusta a origem com `qa.clock_map` e devolve
     `layout.clock = {t0, bar_len, fonte, n, spread_ms, max_ms}` — junto do `layout.bar_map`, na mesma
     passada de `layout_score`, em `/api/analyze`, `/api/demo`, `/api/svg`, `/api/rebuild`, `/api/qa/fix`;
  2. o navegador nunca mais chuta posição: `gradeBase()` escolhe a base (síntese → `t0 = 0`, porque o
     `kit_synth` usa a própria partitura como relógio; arquivo → `layout.clock`; sem relógio →
     `origemMedida()`, a mediana local sobre as notas; e só então o relatório, rotulado
     `"relatório (pouca evidência)"`); cursor, clique, *loop*, leitura do bússola e grade da forma de
     onda todos saem de `gradeBase()`;
  3. dentro do compasso, o cursor interpola na **área útil desenhada** (`beat0 = x0+5 … beat1 = x1−5`
     da caixa daquele compasso, já com `página·page_h`), então zoom, duas páginas e quebra de sistema
     estão certos por construção — e clicar na partitura busca para a batida sob o cursor, não só para a
     caixa;
  4. recusa honesta: `qa.clock_map` exige ≥ 8 notas medidas e `origemMedida()` idem — abaixo disso
     devolvem `null`/rótulo explícito em vez de inventar uma origem;
  5. a divergência não foi escondida: `Δ 1951 ms` entre as duas origens continua medido e exposto
     (teste próprio), porque um dia ela é a pista de que o estimador de andamento escorregou.
- **Prova:** `tests/http_qa_check.py` — "a resposta traz o relógio da grade", "a origem é rigidamente
  coerente com as notas (dp ±12,5 ms · máx ±71,7 ms < 40 ms de dispersão exigida)", "o meio de cada
  compasso acende exatamente o seu compasso" (17/17), "cada golpe cai na caixa do seu compasso ± a
  tolerância do auditor" (51 golpes a < 96 ms da barra são *jitter* humano, não erro — a régua da
  auditoria, não uma tolerância escolhida para passar), "a diferença entre as duas origens é
  registrada", `/api/svg` e `/api/rebuild` trazem o mapa refeito; `tests/ui_boot_check.js` cena F —
  16 verificações (o relógio do servidor manda, a síntese parte do zero, a mediana local acha a mesma
  origem, pouca evidência recusa, clique e faixa de *loop* não usam `phase_ms`).
- **Regressão dirigida:** com a fórmula velha de volta na pintura do *loop*, a verificação "a faixa de
  loop na forma de onda segue o cursor" falha (rodado e confirmado); riscar `layout.clock` da resposta
  faz o front cair em `mediana local` sem quebrar — é o caminho que o teste da mediana cobre.
- **Regra que fica:** nenhuma conta de tempo ↔ grade parte de `phase_ms`; usam-se `qa.clock_map` e
  `qa.time_of`. O mapa de compassos e o relógio moram em `layout.meta` e nunca no dicionário da
  partitura: `T19a` exige que reconstruir não altere nada, e o `score` é o que vai para
  MusicXML/MIDI/CSV — geometria de tela não pode contaminar o arquivo.

### A12. `line1` do mapa de compassos estava em outra língua de coordenadas (e dois detalhes do mesmo assento)
- **Como apareceu:** ao *escrever* a lei que falta, não ao olhar a tela. Passamos a exigir do mapa
  entregue ao navegador que `y0 < line1 < y1` e que o deslocamento de página valha para **todas** as
  coordenadas. Falhou na hora: `fora [15, 16]` — os dois compassos da 2ª página.
- **Causa raiz:** em `layout_score`, o pós-passo que empilha as páginas somava `página·page_h` em
  `y0` e `y1`, mas não em `line1`. O campo continuou em **espaço de página** enquanto a caixa ia para
  **espaço do documento**. Como `ops_to_svg` empilha com `translate(0, i·h)`, o documento é um só
  sistema de coordenadas — quem lesse `line1` para posicionar algo na pauta do compasso erraria
  595,28 pt da 2ª página em diante. Hoje o navegador não usa `line1`, então nada aparecia errado na
  tela: é um defeito **dormente**, do tipo que só cobra caro no dia em que alguém usar o campo.
- **Decisão:** consertar o campo, e não apagá-lo. Ele é a única coordenada que diz *onde está a 1ª
  linha da pauta daquele compasso* — sem ela o front teria de reconstruir pauta a partir de
  `staff_space` e da ordem dos sistemas, que é exatamente o tipo de chutação que produziu A11.
- **Dois detalhes do mesmo assento, achados na mesma varredura:**
  1. a faixa de destaque do compasso ia de `x0 − folga` a `x1 + folga` (2,0 pt), mas o traço de fim de
     sistema é desenhado a 1,1–4,2 pt **à direita** do `x1` — no compasso que fecha a linha, a barra
     dupla ficava fora do destaque medido no PDF rasterizado (Δ 4,11 pt). Agora a banda usa
     `fimDeSistema(mapa, i)` (adjacência `x1 → próximo.x0`, que também flagua quebra de página) e
     estende a cauda para 5 pt nesses casos;
  2. o rótulo do leitor dizia `compassso`. Copia revisada e protegida por verificação, porque o
     usuário lê isso: `!/compassso/` sobre o texto-fonte de `app.js`.
- **Prova:** `tests/http_qa_check.py` ("a pauta do compasso cabe na caixa dele", "o deslocamento de
  página vale para as três coordenadas · páginas [0, 1] · +595,28 pt por página") e
  `tests/ui_boot_check.js` cena F (adjacência de `fimDeSistema`, a cauda de 5 pt no desenho da banda,
  ortografia do rótulo). As duas verificações de HTTP **falharam antes** do conserto em `layout.py` e
  passaram depois — em particular contra o processo velho, o que é a prova de que medem o mapa servido
  e não o arquivo no disco.

---

### A13. Subir a própria faixa derrubava a plataforma inteira — OOM sem erro, sem rastro

**Relato do usuário:** “Eu subi a track, e ao carregar o servidor fechou.” Não havia 500, não
havia traceback no log: o Werkzeug só registra a resposta *depois* de ela existir, e a resposta
nunca existiu. O que havia era `dmesg`:

```
oom-kill:constraint=CONSTRAINT_NONE,…,global_oom,task=python3,pid=4276,…
Killed process 4276 (python3) total-vm:2239564kB, anon-rss:1655428kB
```

Máquina: 1984 MB de RAM, **sem swap**, 2 núcleos. A faixa do usuário era longa; o pico da
cadeia é proporcional ao comprimento, e nada na arquitetura limitava esse crescimento.

**Quatro causas, cada uma medida (não deduzida):**

1. **Nenhum isolamento.** `analyze()`/`demo()` chamavam `transcribe_bytes` dentro do processo
   que atende todo mundo. O OOM killer escolhe o maior RSS — que era o próprio servidor.
2. **Vias inteiras materializadas.** `dsp.stft` alocava a matriz de janelas (`x.shape × n_fft`)
   e o `rfft` do NumPy sobe para complex128; `band_flux` materializava `pred` em complex128
   mais dois `np.abs`; `energy_rise`/`hilbert` filtravam em float64 sobre a faixa toda.
   Medido no STFT de 160 s: **1017 MB → 294 MB** só com blocos + `scipy.fft` float32.
3. **Bluestein.** O envelope de Hilbert é um FFT do comprimento `n` da faixa inteira. Quando `n`
   tem fator primo > 13, o `scipy.fft` cai no algoritmo de Bluestein, que aloca ~2,6× o tamanho e
   roda várias passadas. Medido aos 155 s (`n = 6 839 028`, 26 715 quadros):

   | etapa | pico RSS do processo |
   |---|---|
   | depois do STFT | 310 MB |
   | `sps.filtfilt` | 399 MB |
   | `scipy.fft.fft(x)` sem padding | **843 MB (+444) e 1,5 s** |
   | `X *= h` + `ifft` | **1156 MB** |

   Varredura de arrays vivos no fim da banda: **0 MB** contra 1105 MB de alto-nível — ou seja,
   era estouro transitório de *um* chamamento, não dado retido. `next_fast_len(6 839 028)` =
   6 842 880 (+3 852) **não** escapa de Bluestein, e preencher mudaria o resultado (a envoltória
   analítica é circular no comprimento dado): o conserto honesto é planejar a memória, não
   trocar o comprimento da FFT.
4. **O orçamento previsto errava para menos.** A primeira fórmula de custo errou o pico medido
   em −26 % — um orçamento que *subestima* é pior que nenhum, porque autoriza o estouro.

**O que foi feito:**

* `custo_analise_mb` com coeficientes **ajustados por mínimos quadrados sobre picos medidos**
   (após subtrair a importação base de ~100 MB): `60 B/amostra × fator_bluestein(n) +
   4 B/(quadro·bin) + 140 MB`. Erro atual: 11–16 % **para mais** (conservador) nos três pontos
   usados no ajuste e ~60 % no degrau de 22 050 Hz — superestimar recusa uma faixa que caberia;
   subestimar mata a plataforma. Preferi o primeiro defeito.
* `plano_dsp` roda **antes** de qualquer alocação grande e escolhe o degrau mais fiel que cabe;
  se nenhum couber, `MemoriaInsuficiente` com mensagem que diz o que fazer.
* Análise isolada em processo filho `spawn` (`_analise_isolada`), resultado por `Queue` com o
  pai lendo enquanto o filho escreve (um `Pipe` de 64 KB deadlockaria no SVG de ~100 KB), prazo
  configurável, e mapeamento honesto de falha em `_erro_analise`: 413 memória, 500 filho morto
  (com o sinal e o nome do OOM killer), 504 prazo, 422 arquivo. Injeção de falha para teste:
  `DRUMSCRIBE_FALHAR_FILHO=oom|memoria|trava`.
* Upload copiado para o disco **em streaming de 1 MB** (`_salva_upload`); o pai nunca segura os
  bytes e só o caminho atravessa a fila. `MAX_CONTENT_LENGTH` 80 → 200 MB, com limite próprio e
  413 em JSON (o 413 do Werkzeug é HTML e a interface não sabia exibilo).
* `_gc` agora só apaga diretório **dentro de `out/uploads`** — sem essa guarda, o `rmtree` do
  resultado da demo apagaria `samples/` (faixa + gabarito) no 25º upload.

**Porta de verificação:** `tests/profile_long.py --limite 900`, uma duração por processo (medir
três durações no mesmo processo faria o pico da segunda incluir o da primeira — `ru_maxrss` é
alto-nível acumulado; foi assim que eu mesmo me enganei uma vez). Medido hoje no servidor:

| arquivo | resultado | pico do filho | previsto |
|---|---|---|---|
| 168 s | 200 ok, rebaixado para 22 050 Hz | 464 MB | 746 MB |
| 335 s | 413 em 1 s (recusa legível) | 544 MB (só decodificar) | — |
| 586 s | 413 em 2 s (recusa legível) | 872 MB (só decodificar) | — |

`/api/health` respondeu 200 antes, entre e depois de cada caso. **Custo que reste:** para
*recusar* 586 s o processo ainda chega a ~870 MB, porque decodificar e reamostrar a faixa toda
vem antes do plano. É o preço de medir o tamanho real antes de alocar; cortar isso exigiria
ler o cabeçalho sem decodificar (e o cabeçalho de MP3 não dá a duração sem varrer o arquivo).


### A14. `analysis_sr` era decorativo, e a escada de memória movia as notas — os dois se cobriam

Dois defeitos que se escondiam:

1. **O parâmetro de taxa de análise nunca funcionou.** `audio_io.decode_bytes` fazia

   ```python
   g = _gcd(int(analysis_sr), int(sr))      # `_gcd` não existia neste módulo
   ```

   dentro de um `try: … except Exception: pass`. O `NameError` virava *no-op*: o valor era aceito,
   ecoado no relatório (`file.analysis_sr`) e ignorado. Toda faixa era analisada na taxa de
   origem. Consequência imediata, e a pior: **a tabela de “qualidade por degrau” que eu tinha
   medido eram três execuções idênticas** rotuladas como taxas diferentes. O sinal delator foi
   `ms/quadro` não bater com `hop/sr`. Li o módulo e confirmei: `hasattr(audio_io, "_gcd") ==
   False` e `decode_bytes(..., analysis_sr=22050|11025)` devolvendo `sr=44100` nos três casos.

2. **A escada automática degradava pelo parâmetro errado.** Havia

   ```python
   if x.size / sr > 210 and hop < 512:      # “faixa longa → quadros mais grossos”
       hop = 512
   ```

   Medido contra o gabarito (`tests/evaluate.py`), o efeito não é de fidelidade menor — é de
   gravura errada:

   | tier | ms/quadro | onset_recall | hit_f1 | pos_accuracy | tick_mae | compassos |
   |---|---|---|---|---|---|---|
   | 44 100 / 256 (padrão) | 5,80 | 0,919 | 0,585 | **1,000** | 0,00 | 17 |
   | 44 100 / 512 (a regra) | 11,61 | 0,881 | 0,614 | **0,000** | **−1,00** | 17 |
   | 44 100 / 1024 | 23,21 | 0,797 | 0,558 | 1,000 | 0,01 | **16** |

   Com 11,6 ms entre quadros **toda nota saía um tick adiantada**; com 23,2 ms um compasso
   desaparecia. E a economia era real de ~0 MB (o termo dominante do custo é por amostra, não por
   quadro). Regra removida; `MS_MAX_POR_QUADRO = 6.0` virou lei com o motivo medido ao lado.

**Como ficou.** A escada só altera a **taxa de amostragem** e escala `hop` e `n_fft` na mesma
proporção, de modo que *ms/quadro* e *Hz/bin* fiquem **idênticos** em todo degrau: a única coisa
sacrificada é a metade superior do espectro, que é degradante de forma declarável. `hop`/`n_fft`
passaram a ser interpretados na taxa de referência `SR_REF = 44100` e reescalados para a taxa
efetiva, então pedir outra taxa não engrossa a grade. Piso da escada automática:
`SR_MINIMO_ESCADA = 22050` — abaixo disso a banda `vhigh` (7–16 kHz), que tem peso 3.0 no chimbal
(`kit.LANE_BAND_WEIGHTS`), cai inteira acima de Nyquist e a pista ficaria muda; só o usuário pode
escolher isso, e o relatório avisa. Efeito colateral encontrado ao testar o caminho: com banda
acima de Nyquist, `dsp.extract_features` arrebentava em `ValueError: zero-size array to
reduction operation maximum which has no identity` — hoje `band_flux` devolve fluxo nulo e a
feature da banda vazia vai ao piso (`_PISO_DB`), que é “não há evidência ali”, e não bin de outra
faixa reassignado para parecer que há.

**Medido depois do conserto** (`tests/tiers_check.py`, a mesma porta):

| tier | ms/quadro | Hz/bin | onset_recall | Δrecall | hit_f1 | pos_accuracy | tick_mae |
|---|---|---|---|---|---|---|---|
| 44 100 / 256 / 1024 (padrão) | 5,80 | 43,07 | 0,928 | — | 0,579 | 1,000 | +0,00 |
| 22 050 / 128 / 512 (degrau automático) | 5,80 | 43,07 | 0,915 | −0,013 | 0,573 | **1,000** | **+0,00** |
| 11 025 / 64 / 256 (só a pedido do usuário) | 5,80 | 43,07 | 0,818 | −0,110 | 0,447 | 1,000 | +0,01 |

Ou seja: o degrau que a máquina pode escolher sozinha custa 1,3 ponto de recall e **não move
nota**; o degrau que cala uma pista custa 11 pontos — por isso ele não está na escada.
`analysis_sr` fora de 6–192 kHz agora sobe `ValueError` em vez de ser engolido.

**Nota de processo (a parte que dói).** Ao escrever o conserto do item 1, abri
`audio_io.py` para escrita antes de ter a string pronta; a exceção veio depois do truncamento e
o arquivo ficou **vazio** — e o `.pyc` bom já tinha sido sobrescrito pelo módulo vazio da minha
própria tentativa de import. Não há git neste repositório. Reconstruí o módulo pelo contrato
(o que os consumidores exigem) e validei com as portas existentes: `selfcheck` 48/48,
`doublecheck` limpo, `evaluate` nos números do código atual. Para saber se a reconstrução era
fiel onde ela *poderia* mentir — a normalização de análise e o corte de silêncio — medi cinco
variantes de cada (`noop`, RMS com/de sem banda morta, pico 0,95, `trim` desligado/estrito/1 ms):
todas deram `hit_f1` 0,578–0,579 e `pos_accuracy` 1,000, isto é, **a forma exata dessas duas
funções não é crítica** para o resultado medido. A lei é mensurável; a forma, não.


### A15. Deriva numérica das reescritas de memória — registrada em vez de negada

O baseline de acurácia gravado neste projeto antes do trabalho de memória era:

```
n_notated 308 · onset_recall 0,919 · onset_precision 0,935 · hit_f1 0,585
time_bias −6,01 ms · time_mae 6,82 ms · vel_pearson 0,756 · bpm 99,99
```

Hoje, mesma faixa, mesmo gabarito, mesmo CLI (`tests/evaluate.py`):

```
n_notated 313 · onset_recall 0,928 · onset_precision 0,920 · hit_f1 0,579
time_bias −7,67 ms · time_mae 8,29 ms · vel_pearson 0,710 · bpm 99,97
```

As invariantes duras **não** mudaram: `pos_accuracy 1,000`, `tick_mae 0,00`, 4/4, 17 compassos,
auditoria `doublecheck` limpa e reparos idempotentes. O que se moveu é a zona de limiar: +5 notas
escritas, 1,5 ms de MAE de tempo, 0,05 de correlação de acento.

Causa: as reescritas por memória trocaram acumulação float64 por float32 e somas por bloco —
`stft` (1,46e-7 relativo), `envelope_hilbert` (4,4e-7), `band_flux` (max|Δ| = 0,0), RMS em
blocos. Num sinal com ~1 400 eventos e limiares absolutos (`thr_lo` do detector, saldo de banda
que vira dinâmica MIDI), 1e-7 relativo é suficiente para virar decisão em algumas notas. Não é
bug aritmético: é o preço de operar em precisão simples, e o preço foi escolhido de propósito.

Por que não reverti: voltar ao caminho float64 do `stft`/`band_flux`/`hilbert` é devolver o pico
que matava o processo (1,02 GB → 0,29 GB só no STFT de 160 s). Prefiro 0,579 de F1 com a
plataforma de pé a 0,585 com SIGKILL — mas o número fica escrito, medido, em vez de declarado
“inalterado” por quem não refez a medição. Nenhum limiar de auditoria (T11, T13–T19) foi tocado
para acomodar isso, e a tabela de precisão do README foi atualizada com os valores de hoje.


### A16. Pediram “partitura simples com opção de mostrar/ocultar pratos” — o caminho fácil destruía dados

O pedido é de apresentação: a pauta de bateria “simples” mostra bumbo, caixa e tons, e o usuário
quer poder ligar os pratos. O caminho fácil era mexer em `params["lanes"]`, que filtra o
classificador. Ele foi rejeitado por três razões medidas:

1. `lanes_enabled` corta antes da classificação, e o **rótulo de queda-de-braço** do classificador
   (grupo com energia, peça incerta → `hat*`, ver B2) remanejava os golpes de prato para outras
   pistas. “Ocultar o chimbel” produziria *bumbo* inventado — destruição de nota lícita, a lei que o
   projeto não quebra.
2. `min_confidence` não é corte posterior (ele entra no ajuste da grade), então qualquer
   filtragem por pista muda andamento e rejanelagem: a “pauta simples” seria outra transcrição, não
   a mesma leitura com menos linhas.
3. Os oráculos (`qa.audit`, `n_hits`, T1–T19) julgam o áudio; se o filtro entrasse pelos dados, a
   auditoria passaria a julgar uma parte do sinal e o resultado deixaria de ser verificável.

A implementação é portanto uma chave de **gravura**: `Score.hide_lanes` (lista de pistas) é lida só
por `layout_score` → `score_visivel`, `to_midi`, `to_musicxml`. Medido no demo, com
`hide_lanes = kit.PRATOS` (7 pistas):

Medidas pelo `tests/wave_check.py` sobre `samples/demo_drums.wav` reproduzido em memória (é por
isso que aparecem 296 notas aqui e 292 nas chamadas HTTP de `/api/demo`, que prefere o MP3 — o
`decode` do MP3 não é bit a bit o do WAV; nenhuma das duas contagens é "a" contagem do projeto).

| o que muda | antes | depois |
|---|---|---|
| SVG | 100 261 B | 61 249 B |
| MusicXML | 111 619 B | 65 209 B |
| .mid | 2 690 B (296 notas) | 1 694 B (168 notas) |
| operações de layout | 821 | 501 (mapa de compassos idêntico) |
| **notas no dicionário/JSON/CSV** | **296** | **296** |
| `score_digest.n_hits` da auditoria | 296 | 296 |

Ou seja: some da folha e do arquivo que se toca, não do resultado. O `mapa de compassos`
(`bar_map`) não se move, então o cursor de reprodução continua certo com pratos ocultos — isso é
verificado, não presumido. A UI expõe “simples / completa” + a caixa “pratos” (rota
`POST /api/view`) e o `report.apresentacao` grava no resultado o que estava oculto e quantas notas
haviam nos dados, para que a folha impressa por outra pessoa não pareça ter perdido notas.

*Congelado em:* `tests/wave_check.py` (leis F, oito verificações).


### A17. O zoom “não funcionava” de novo — e as duas aritméticas que vieram junto

O sintoma reportado foi outra vez literal e exato: *o zoom na aba Áudio & batidas não está
funcionando*. Causa: `applyZoom` só alterava a largura do SVG da partitura; a onda era desenhada de
um `state.wave` de resolução fixa (1800 baldes para a faixa inteira), mapeado sempre em
`t → x = t/duração·W`. Mover a alavanca não tinha efeito nenhum naquele canvas. Corrigido com uma
**janela partilhada** (`state.view` em segundos): a alavanca define `span = duração/zoom`, a aba de
áudio e o piano-roll do MIDI desenham a mesma janela, e o dado vem de `GET /api/wave/<id>?t0&t1`,
que lê os planos float32 gravados pelo filho durante a análise por deslocamento de bytes — nada é
re-decodificado no pai (A13).

Três aritméticas erradas apareceram na verificação e valem registro porque nenhuma delas dava erro:

* **Forma do agrupamento.** O leitor de baldes tinha um ramo separado para “não há agrupamento”
  (`k = 1`) que rearrumava o vetor para `(1, N)`. O consumidor faz `max(axis=1)`, então a janela
  estreita devolvia **um ponto** em vez de 344: zoom alto = linha reta. A lei agora é uma só
  forma `(blocos, k)` em todo ramo, com `k = 1` incluso. Foi pega porque a verificação conferiu
  `len(min) == len(max) == len(curvas[…]) == n`, não só `ok: true`.
* **Segundos por tick.** `60/bpm·1e6/division` em vez de `tempo_us/1e6/division` — o inverso. No
  arquivo da demo (99,97 BPM, divisão 480) isso transformava uma semicolcheia de 149 ms em
  **24,8 s** e um ataque em 0,6 s em 100 s: o piano-roll existiria, desenhado, completamente
  errado. O teste agora confere uma propriedade, não uma constante: os espaçamentos entre notas do
  mesmo instrumento são inteiros de tick (pior desvio medido 0,00009 s em 244 pares) e a duração
  está entre 1 tick e 1 compasso. Constante errada é o que produziu a primeira versão desta lei,
  que supunha 100,0 BPM exatos — a faixa não é, e a escala tem de vir do arquivo.
* **Critério sem lado de “faltou nota”.** A primeira versão do recomendador pontuava duplicata,
  grade, densidade e auditoria, mas nada recompensava *detectar*. Resultado medido: ele recomendava
  o canto inferior esquerdo da grade (sens 0,70 / conf 0,20, 276 notas) contra os 313 do ponto de
  calibração — menos notas, custo zero. Também lia `base["sensitivity"]` *depois* do laço que o
  mutou, então “o ponto atual” era a última célula testada (1,45/0,20) e a comparação
  “vale mudar / não vale” saía contra um número que não existia. Corrigido com (i) critérios de
  cobertura — fração de compassos sem bumbo nem caixa, golpes por compasso das pistas de base —
  e (ii) fotografia do par vigente antes da varredura, com o ponto vigente sempre dentro da grade.
  Descartei de propósito uma régua por contagem de picos do envelope: com proeminência fixa ela
  contava 94 “ataques fortes” contra 236 do gabarito no demo (chimbel em 16ºs fica abaixo dela) e
  acusaria de dupla detecção uma partitura correta; o número continua em
  `report.referencia_onsets`, informativo, sem lei anexada.

Depois da correção, na demo a varredura (16 análises, teto declarado 17) recomenda **0,95 / 0,30
— os valores correntes** — e o veredito diz “já é o melhor ponto medido nesta varredura”. O que os números
gabaritados mostram, e que precisa ficar escrito porque contraria a expectativa de que “recomendar”
significa “melhorar o f1”:

| par | notas | custo | hit_f1 | hit_recall | hit_precision | pos_accuracy |
|---|---|---|---|---|---|---|
| 0,95 / 0,30 (corrente, recomendado) | 313 | 0,000 | 0,579 | **0,674** | 0,508 | 1,000 |
| 0,70 / 0,20 (o que a primeira versão escolhia) | 276 | 0,000 | **0,586** | 0,636 | 0,543 | 1,000 |

Os dois têm custo zero — os critérios sem gabarito **não distinguem** esses pontos, e foi isso que
a primeira versão tentou esconder escolhendo o primeiro da varredura. A recusa explícita do
0,70/0,20 é uma escolha de produto declarada: a 0,7% de f1 trocam-se 3,8 pontos de recall, e uma
nota que não foi escrita é invisível para quem toca, enquanto uma nota a mais é editável. Por isso o
empate é resolvido pelo ponto mais próximo da calibração do projeto e, depois, por maior contagem
de notas — e o fato de ter havido empate sai impresso nos `motivos` da UI. *Congelado em:*
`tests/tune_check.py` (verificações incluindo
“recomendado nunca piora f1/recall/precisão contra o gabarito”, “o ponto vigente é medido, não
estimado” e “empate é declarado, não decidido por ordem de varredura”).

### A18. “Ocultar os pratos” fabricava 2 erros na auditoria — a expectativa estava do lado errado

Depois de A16, com `hide_lanes` ativo na demo enviada como MP3, `POST /api/qa level=full` devolvia
`{"error": 2, "manual": 2}` num resultado que, sem ocultação nenhuma, dá 0 erros:

* **T11** — “102 notas no XML, esperado 225 (+67 de acorde)”
* **T12** — “169 notas no canal 10 vs 292 na partitura”

Causa: `qa.check_exports` regera o export (que respeita a apresentação) e comparava com expectativas
derivadas do **dicionário completo**. A releitura estava certa; a régua é que era outra. O efeito
prático seria o pior possível: o usuário marcava “partitura simples” e a plataforma passava a dizer
que a transcrição tem erro — e o agente de correção receberia um quadro para “consertar” apagando
notas lícitas, exatamente o que a lei do projeto não permite.

Correção na origem, não no limiar: `check_exports` passa a calcular as expectativas de arquivo sobre
`score_visivel(score)` (o mesmo corte que `to_midi`/`to_musicxml` fazem), mantendo as leis de escrita
T1–T10 sobre **todas** as notas. E o filtro deixa de ser silencioso: nasce o achado
`T10p · apresentação filtra o arquivo, não o resultado`, com as duas contagens
(“296 nota(s) nos dados, 168 no que é gravado”). Medido depois da correção, com pratos ocultos:
`tally = {error: 0, warn: 0, info: 23, manual: 0}` — e sem ocultação, `{error: 0, …, info: 22}`.

*Congelado em:* `tests/wave_check.py` (duas verificações F: “ocultar pistas não cria erro de
auditoria” e “a auditoria declara o que a apresentação filtrou”).

### A19. A faixa era analisável e mesmo assim vinha 413 — o modelo de custo não conhecia o próprio custo

**Relato do usuário:** “algumas músicas parecem não estar lendo, principalmente as que têm um
silêncio no início.” O arquivo real (`samples/exemplo.mp3`, 17,7 MB, 443 s) devolvia

```
413 {"kind":"memoria","msg":"…o plano mais econômico pede 1915 MB contra 1096 MB disponíveis"}
```

A guarda fez o que devia (o servidor continuou de pé), mas **recusou um trabalho que cabia**: medido
no filho isolado, o pico real da cadeia inteira a 22 050 Hz foi **718 MB** contra 1096 MB de
orçamento. A recusa virava o defeito.

**O que estava errado na conta.** Dois enganos no mesmo lugar: (i) as grades por banda eram
cobradas como se tivessem o comprimento do *arquivo*, quando o `n_fft` é fixo — com seis pistas,
seis grades de `n_fft/2+1` bins pesam mais que o áudio de uma faixa curta; (ii) o custo do Bluestein
estava **descrito** em A13 e não estava **na fórmula**. Faltava ainda o piso das caixas de
`hilbert`/filtros, que não depende do comprimento.

**Como foi recalibrado, por medição e não por palpite:** a inclinação marginal do pico em função do
número de amostras (60→120 s, 44 100 Hz, hop 256) deu **61 B/amostra**; o caminho completo no
arquivo de 443 s mediu 718 MB contra 804 MB previstos — o modelo ficou **conservador em ~12 %**,
que é exatamente a margem que se quer antes de um SIGKILL. A recusa passou a declarar a margem.

*Efeito medido:* o mesmo upload virou **200 em 26 s** (239 compassos, 2999 notas, 6/8, 99,11 BPM),
com `ajustado_por_memoria: true` — a escada desceu de 44 100 para 22 050 em vez de recusar.

*Congelado em:* `tests/long_check.py` L1 — recusa só quando **realmente** não cabe (1 h pede 5540
MB no plano mais econômico contra 985 MB), previsão ≥ medido, margem declarada, e a faixa de 126 s
dentro do prazo de uma interação (13,7 s).

*Corolário no portão da escada:* `tiers_check` exigia `custo[degrau 2] < 0.62·custo[degrau 1]` sobre
os **totais**. Com a base recalibrada (140 MB de interpretador + bibliotecas, que não caem com a
taxa), um trecho curto passou a ter razão 0,68 **sem que nada estivesse errado**. A asserção foi
reescrita para a grandeza que a lei de fato afirma: a **parte variável** do custo cai pela metade
quando a taxa cai pela metade (medido 0,50×). Endureceu, não afrouxou: 0,55 no lugar de 0,62, e no
termo certo.

---

### A20. Branqueamento com memória do arquivo inteiro — o ataque depois do silêncio ficava sob a base

O que impedia a leitura não era o silêncio em si: `audio_io.trim_silence` cortou os 11,00 s de
lead-in do usuário (`lead 242 585` amostras a 22 050). O que matava os ataques era o **piso da
base local** de `dsp.band_flux`, que crescia com o comprimento do arquivo:

```
piso = 0,02 × duração          →  8,8 s de janela num arquivo de 443 s
                                  0,5 s na demo de 10,4 s  (por isso nunca apareceu)
```

Depois de um silêncio longo, a mediana da janela continuava cobrindo o ruído *pré*-silêncio: 24 dos
30 primeiros ataques após o corte ficavam abaixo do limiar de disparo — a pauta começava vários
compassos depois do que se ouve.

**Correção:** `BASE_MAX_S = 15.0` — uma estatística local não pode lembrar o arquivo inteiro. Em
faixas curtas o teto não se aplica e nada muda: o fixture da demo ficou **bit a bit idêntico** e
`tests/evaluate.py` reproduziu a linha de base exatamente (f1 0,579 · pos 1,000 · viés −7,67 ms).

*Congelado em:* `long_check` L5 (piso do branqueamento local: 2584 quadros ≈ 15 s, e não 25 438 ≈
144 s) e L2 (a mesma faixa com 12 s de silêncio gravados na frente produz 296 vs 293 notas, mesmo
andamento e a mesma métrica — semelhança 0,900).

---

### A21. Fórmulas paralelas para a mesma grandeza — três para o passo da grade, três para o fantasma

No arquivo do usuário a auditoria acusava, com números que não batiam entre si:

```
T3b  2999 de 2999 golpes fora do passo (100,0 %)   mas declarava  odd_ticks 3333
T6   fantasma em kick — proibido                   e o escrevedor tinha escrito mesmo assim
```

**Diagnóstico.** A mesma grandeza era calculada em três lugares, cada um com sua tabela:
`grid.fit_grid`, `pipeline` (a linha que escolhe o passo) e `qa.T3b` (a lei). Em 4/4 as três
concordavam, então a divergência era invisível. Numa 6/8 real (tpb 12, 24 ticks por compasso) o
passo de semicolcheia é **3**, não 4: a pauta escrita ficou toda em ticks que a régua do auditor
chamava de fora do passo. A permissão de cabeça de fantasma tinha a mesma doença em três listas
(`kit.Lane.ghostable`, `pipeline._GHOST_POLICY`, `qa._GHOST_OK`).

**Correção na fonte, uma régua por grandeza:** `grid.passo_grade(g, tpb)` virou a única tábua e os
três chamadores a usam; `kit.Lane.ghostable` é a única permissão, `qa._GHOST_OK` é derivado dela e
a política do escrevedor é subconjunto declarado (verificado no `selfcheck` #25). O `odd_ticks`
**declarado** no relatório passou a ser contado sobre o que foi **escrito**
(`score.bars[*].hits[*].tick`) em vez do array pré-gravação: `q["ticks"]` tem um item por *evento*,
mas um instante pode carregar bumbo e caixa juntos — eram 1008 declarados contra 1474 recontados.

---

### A22. Nota atravessando a barra de compasso, e a régua de segundos que acusava um MIDI perfeito

Dois defeitos de escrita medidos no mesmo arquivo:

1. **`dur` sem ligadura que ultrapassa a barra.** `crash` no tick 24 com dur 18 num compasso de 24
   ticks, `kick` no 30 com dur 10 — impresso, a cabeça da nota caía depois da barra seguinte.
   Corrigido onde o `dur` é escrito (`pipeline._mark_sustain`, nas duas rotas: análise e
   `rebuild_score` do editor): nota sem ligadura tem `dur ≤ tpr − tick` (24+12→8, 30+4→2).
2. **T12 media a lei errada, na unidade errada.** A régua era `t_último × bpm/60 × 480` segundos e
   comparava com o fim do SMF — misturava fase de downbeat e o *lead* cortado numa lei estrutural.
   No arquivo real reclamava 8050 ticks; num 6/8 com MIDI **perfeito** reclamava 1208. Subtrair só
   o *lead* não bastava. Agora a régua é a **geometria escrita** `(bar·tpr + tick) × 60` contra o fim
   lido do `.mid` (escala 480/8): no exemplo, último instante escrito 404 × 60 = 24 240 = `max_tick`
   lido, com os 1000 ticks de folga de escrita declarados.

Efeito colateral útil, e honesto: o `.mid` não carrega áudio, então o deslocamento entre ele e o MP3
**tem** de ser divulgado. O `lead` é medido e exposto (`/api/midi_view` devolve `lead_ms` e
`dur_audio_s`; o painel da DAW avisa quando é preciso correr a importação de 10,98 s).

---

### A23. O auditor contava mal as duas pontas — e por isso acusava a partitura

```
T11  notas_xml 2163 · slots 2160 · dados 2999 · acordes 839 · ligaduras 0/0
```

Um regex meu contando `<note>` por `<measure>` dizia outra coisa e quase me fez culpar `_fill_rests`
e `_mark_sustain`, que estavam certos. A verdade, lida no `musicxml.py`: para `artic == "flam"` o
escrevedor emite, **antes** da nota principal,

```xml
<note type="grace"><cue/><duration>0</duration>…</note>
```

Uma nota de adorno é um `<note>` legítimo, sem `<chord/>` e que não abre slot de dados: 2160 slots
+ 3 adornos = 2163 ✓. Os 839 `<chord/>` e todos os `pos` já batiam exatamente.

**Correção na lei, não no gravador:** o leitor independente do `qa` passou a classificar cada
elemento (`notas` / `acordes` / `pausas` / `adornos`) e a confrontar a contagem de notas com os slots
de dados; os adornos têm lei própria, na direção honesta — **adorno não pode ser inventado**
(`adornos ≤ flans nos dados`). O contrário é lícito: um par de flam detectado pode deixar de virar
adorno na rejanela do editor, e cobrar igualdade era cobrar do editor algo que ele não faz.

*Lição gravada:* contar `<note>` sem separar `<chord/>`, `<rest/>` e `type="grace"` mente. E o
número da lei tem de vir do **arquivo relido**, não de um diagnóstico caseiro.

---

### A24. Um erro que só pintava a linha de status deixava a partitura antiga na tela com os botões vivos

Com “não está lendo”, o caminho de erro de `analyze()` escrevia uma etiqueta vermelha no rodapé e
**não tocava a folha**: se já havia uma partitura na tela (a demo, ou o arquivo anterior), o PDF, o
`.mid` e o MusicXML continuavam apontando para um documento que não era o que o usuário subiu — na
prática, “o arquivo não foi lido e não apareceu PDF”.

**Correção:** nasce `falhaNaFolha()` no `app.js`, com o cartão `#score-erro` acima do `#score-wrap`
e a classe `.falha` no CSS — análise que falha **explica o motivo na própria folha** e limpa a
partitura; adotar um resultado novo limpa o aviso. O servidor não tem caminho que devolva meia
partitura, e o contrato é cobrado por teste: `http_qa_check` exige que `/api/analyze` responda ou
com escore + PDF ou com erro HTTP explícito (`kind`/`msg`/`hint`).

*Congelado em:* cena H de `tests/ui_boot_check.js` (6 asserções: o cartão aparece, o motivo aparece,
a folha é limpa, os botões de exportação ficam inerte, o aviso some ao adotar, e nada disso
depende de rede).

---

## B. Limitações conhecidas — ficam visíveis, não são suprimidas

| # | Limitação | Onde aparece | Por que é assim |
|---|---|---|---|
| A13 | Faixa que **nem no degrau mais econômico** cabe na memória é recusada com 413 (medido: 1 h pede 5540 MB contra ~985 MB disponíveis). Faixa que cabe no degrau de baixo é transcrito nele, com `ajustado_por_memoria: true` | aviso vermelho no topo, com a conta de memória e a margem declaradas | o plano escolhe o degrau mais fiel que cabe; mentir a respeito mata o processo que atende todo mundo (ver A13/A14/A19) |
| B1 | Em 4/4 com bumbo em 1–3 e caixa em 2–4 a posição da barra é **genuinamente ambígua**; a transcrição pode começar um tempo antes/depois | T2 (`aviso`, margem medida) + slider **Deslocar a barra** (`#dbeat`) na UI | a evidência espectral é simétrica; girar a barra não muda o áudio, só a leitura. Resolver isso exigiria referência harmônica, que o produto deliberadamente não usa |
| B2 | Rótulos com `confidence = 0.30` são queda-de-braço do classificador por grupo (fallback). Subir `min_confidence` para 0,31 **apaga** a parte inteira (recall 0,686 → 0,30) | relatório `mean_confidence`, T15a/T15b | o portão físico não decidiu a peça; o grupo sabia que tinha energia ali. Preferimos escrever e marcar a baixa confiança a calar o golpe |
| B3 | Peças de prato raro (`ride` 0,036 de recall no gabarito; `crash`/`hat_foot`/`hat_open` 0) | `evaluate.py`, `lane_confusion` | o alfabeto de faixas do demo é intencionalmente pobre: sem amostra real por peça, o classificador só tem a banda espectral. O produto é honesto: o número fica no relatório |
| B4 | 32º só entra por promoção *individual* e anunciada (`promoted_32nd`); nenhum punhado de detecções incertas reescreve a grade inteira | T3b (error se houver tick fora do passo sem promoção declarada) | grade mínima suficiente: leitura estável vale mais que micro-precisão |
| B5 | PDF não é byte-a-byte idêntico entre execuções (o gerador escreve data/ID) | T19b `info` — o teste compara o conteúdo **após normalizar** metadades de data e exige igualdade total no SVG | é o desenho que tem de ser reproduzível, não o carimbo da hora |
| B6 | A síntese de conferência tem envelopes próprios: `T19c` é **aviso**, medido no sentido "toda nota escrita soa?", não "todo ataque do sintético é nota" | T19c (51 % → 100 % após corrigir o caminho) | prato com decay longo é lido como transiente extra pelo detector; a pergunta útil é se a pauta é tocável |
| B7 | Colchão tonal audível por cima da bateria (> limiar medido) degrada a classificação de peças | T18 `aviso`, com a tabela medida e a zona cinzenta declarada | o produto assume faixa isolada — é premissa de projeto, não bug |
| B8 | Faixa com clipping forte, nível < −30 dBFS ou < 6 s: aviso, e andamento/fórmula ficam pouco confiáveis | T16/T17/T2b | pouca informação para estimar período e métrica |
| B9 | Sem buzz roll / tremolo notado por número de strokes; `droll` sai como texto de articulação | `rules.py`, `musicxml.py` | manter a escrita legível em notação simples de bateria; o campo existe no modelo para crescer |
| B10 | `out/uploads/` é memória de processo (24 resultados, GC por idade); reiniciar o servidor perde os IDs | `server.py` | escolha deliberada para manter a plataforma sem banco; o download é sempre regerado do `score` |
| B11 | Flans são gravados como nota de adorno (`<note type="grace"><cue/>`) antes da nota principal; importadores que não implementam adornos podem ler a nota seguinte adiantada | `musicxml.py`, `qa.T11` (lei dos adornos) | é a notação correta para o gesto, e a auditoria sabe contá-los à parte; preferimos o símbolo certo a omitir o flam |
| B12 | `T19c` (a pauta sintetizada soa?) fica em **aviso** em faixas densas reais: medido 84 % das notas com ataque a ±68 ms numa faixa de 443 s | achado `T19c` no relatório de auditoria | em texto denso o kit sintético tem sobreposição de decaimento que o detector lê como ataque extra; a pergunta que a lei responde — “toda nota escrita soa?” — continua respondida |

---

## C. Risco operacional conhecido deste repositório

Edits por *substituição de região* já apagaram código duas vezes (`qa.py` perdeu metade dos
verificadores; `server.py` foi destruído por um `str.replace('')` com fatia invertida). Ambos foram
reconstruídos. Mitigação adotada: nenhum patch por expressão solta — `ast.parse` no arquivo
alterado **e** `python3 tests/doublecheck.py` + `tests/selfcheck.py` antes de dizer que está pronto.

Terceira ocorrência do mesmo modo de falha, nesta rodada: um patch em `qa.py` feito com
`re.search` de corpo delimitado por lookahead duplicou o fim do arquivo (o âncora final não foi
repetida dentro do padrão) e `ast.parse` **passou** no arquivo corrompido, porque a cópia era
sintaticamente válida. Só `grep -c "def NOME"` por função revelou o estrago. A mitigação ganhou um
passo: antes de gravar, âncora tem de ser **string fixa com contagem de ocorrências** (`count == 1`),
e depois de gravar, o `grep -c` de cada função tocada volta a ser conferido — aliado ao
`doublecheck`/`selfcheck`, que pegaram o problema em minutos.

Segundo risco, este externo ao código: **este ambiente é efêmero** — o processo do servidor e os
pacotes instalados não sobrevivem ao reinício do sandbox, e a URL do preview fica girando sem nada
atras dela. Por isso `run.sh` reinstala o que falta e *confere* depois, recusa porta ocupada, e o
front tem sonda + prazo + faixa de erro (A9) em vez de espera infinita.
