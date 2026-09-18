# DrumScribe

Suba um áudio de **bateria isolada** (MP3/WAV/FLAC) → o sistema ouve, mede e **escreve a
partitura do kit** numa pauta de percussão, com **exportação em PDF**, MusicXML, MIDI,
SVG, JSON e CSV. Tudo local, sem serviço externo.

```bash
./run.sh                    # instala o que faltar e abre em http://localhost:8000
```

Se a página abrir e ficar girando (ou ficar em branco): o servidor é o único estado do sistema, e
ele não sobrevive a um reinício do ambiente. Verifique `curl http://localhost:8000/api/health` — a
sonda responde `{"ok": true, "faltando": [], …}` quando está tudo certo e diz **quais bibliotecas**
faltam quando não está. (Uma análise de verdade leva ~20 s para 40 s de áudio: ali o spinner é
informação, não travamento — a diferença é que ele agora tem prazo e a faixa de erro aparece se o
servidor não responder.) A própria página mostra uma faixa "não consigo falar com o servidor" em vez
de esperar para sempre, e tenta de novo sozinha a cada 10 s. Durante uma análise o véu de
"analisando…" aparece por cima de tudo; se você quiser continuar mexendo enquanto ela roda, aperte
**Esc** (ou clique fora do cartão / *ocultar*) — a operação segue em segundo plano e o resultado
aparece sozinho. O estado inicial do véu mora **dentro do HTML** (`<style>` inline + `display:none`
no próprio elemento) e os recursos chegam com carimbo de versão (`style.css?v=…`), então nem um
`style.css` velho no cache, nem um `app.js` que não rode, conseguem deixar a tela inclicável; se o
script não der sinal de vida em 2,5 s, um cão de guarda no documento esconde o véu e diz o que fazer.

Depois de abrir a página: **solte o arquivo** (ou clique em *usar demonstração*), ajuste
*sensibilidade / grade / compasso / swing* se quiser, e exporte. A aba **Corrigir** permite
trocar peça, pulso, velocidade e articulação de cada nota; *reescrever partitura* refaz a
gravura (durações, pausas, vigas, acentos) mantendo suas mudanças. A aba **Áudio & batidas**
toca o arquivo com a grade de batidas marcada sobre a forma de onda (shift+clique isola um
compasso em loop) e o botão 🎛 toca a **transcrição** sintetizada — a maneira mais rápida de
conferir se a escrita está certa. A aba **Auditoria** confere a partitura por fora (dupla validação)
e tem um botão de **correções automáticas**.

---

## Como funciona (cadeia visível, sem caixa-preta)

| etapa | módulo | o que faz |
|---|---|---|
| 1 · decodificação | `audio_io.py` | miniaudio → soundfile → ffmpeg; normalização, corte de silêncio, envelope para o desenho |
| 2 · espectrograma | `dsp.py` | STFT 1024/256 (hop 512 acima de 210 s), fluxo por banda, *novelty*, onsets **por grupo espectral** |
| 3 · features | `dsp.py` | por golpe: delta de banda (40 ms antes × janela do evento), sub-razão, centroide, flatness, slopes de decaimento, ataque |
| 4 · classificação | `classify.py` | escore + **portões físicos** por peça (bumbo exige sub-corpo e queda do agudo; chimel exige inclinação espectral alta; pratos exigem sustento) |
| 5 · andamento | `grid.py` | autocorrelação + prior + pente harmônico; refino por resíduo de grade (erro típico <0,05 %) |
| 6 · grade rítmica | `grid.py` | **grade mínima suficiente** (8ºs se bastam), swing com histerese, downbeat por template de backbeat |
| 7 · escrita | `rules.py` | durações, pausas, pontos, vigas, sustenus de prato, acentos, ghost, flam, viradas, compassos de repetição |
| 8 · gravura | `layout.py` + `engrave.py` | geometria única → primitivas → **SVG** (navegador) e **PDF** (reportlab) |
| 9 · exportação | `musicxml.py`, `midi_out.py` | MusicXML 4.0 e SMF tipo 1 (canal 10, PPQ 480) |
| 10 · auditoria | `qa.py` | confere tudo isso por fora (relê XML/SMF/PDF/PNG e remede o áudio), aponta o que está errado e propõe o conserto |

Detecção **multi-rota por grupo** (grave / médio / agudo): um bumbo e um chimel no mesmo
pulso são dois eventos, não um rótulo ambíguo — é o que permite escrever acordes de bateria.
As durações nunca cruzam a divisão do tempo, salvo sustain de prato (aí vira liga).

## Precisão medida (faixa de demonstração, 41,9 s, 100 BPM, 236 eventos)

```
python3 tests/evaluate.py            # roda a cadeia completa e compara com o ground-truth
python3 tests/selfcheck.py           # 48 verificações: métricas + gravura + exportações
python3 tests/tiers_check.py         # nenhum degrau automático pode mover nota (18,5 s)
```

| métrica | valor |
|---|---|
| recall / precisão de onsets (±40 ms, qualquer peça) | 0,93 / 0,92 |
| F1 de acerto com peça correta | 0,58 |
| posição na grade (compassos/tick) dos acertos | **1,000** (MAE de tick 0,00) |
| viés / MAE de tempo dos acertos | −7,7 ms / 8,3 ms |
| erro de BPM | 0,03 % |
| fórmula de compasso | 4/4 ✓ (17 compassos, 2 páginas) |
| correlação de velocidade com o ground-truth | 0,71 (MAE ≈ 24 em MIDI) |
| recall por peça | bumbo 1,00 · chimel 0,85 · caixa 0,61 · tom agudo 1,00 · ride 0,00 · crash 0,00 |

Estes são os números de hoje, medidos depois das reescritas por memória (float32 e somas em
bloco): a colocação não se moveu, e o F1 perdeu 0,006 com +5 notas escritas. A comparação com o
estado anterior, com os dois lados medidos, está em `docs/ERROS.md` A15 — registrar a deriva foi
o combinado, não "está igual".

Os números são do áudio sintético do repositório (audível e reprodutível): ele existe para
detectar **regressão**, não para prometer desempenho em gravações reais — em estúdio, com
sangria de microfone e sala, ride/crash e caixa dependem muito mais da mixagem.

## Validação dupla / auto-correção

A transcrição tem um segundo par de olhos: `drumscribe/qa.py` confere o resultado **sem confiar em
nada do pipeline** — regrava e relê o MusicXML com ElementTree, abre o SMF byte a byte, re-analisa o
PNG do gráfico, reconstrói a matriz de amostras do PDF com `pypdfium2` e volta ao áudio para remensurar
onset, espectro, energia e ritmo em coordenadas próprias. Cada achado traz a lei, a evidência, os
números, o que é consenso de prática e a fonte externa, e um conserto proponível — a mesma rotina serve
de auditoria, de diagnóstico para humanos e de motor do agente.

Dezenove famílias de leis, entre elas:

| lei | o que garante |
|---|---|
| T3 / T3b / T3c | cada golpe escrito está a `tol` de um ataque real; `(t, compasso, tick)` é rígido (sem deriva de andamento); a malha mínima anunciada é respeitada |
| T3d / T3e | a metragem do compasso fecha contra o áudio e o BPM bate dentro de ±0,5 %; nenhum `time` aponta para outro lugar |
| T4–T8 | sem choque de slot nem acorde de fato, pausas e ligados coerentes, campos legais, vozes e linhas de tempo |
| T9–T12 | compassos com numeração e repetição certas; XML fecha em cada voz, `divisions=8`, 4/4 com um `<time>`, tempo só em `<sound>`; MIDI com 480 PPQ, `t × bpm/60 × 480` e 9 chaves GM conferidas |
| T14 | o CSV que o editor lê de volta reconstrói exatamente o mesmo estado |
| T15a–T15d | laudo sem auto-contradição, o que o usuário lê é a lista que ele vê, e o **ouvido é o último recurso**: os 10 % de ataques mais fortes do arquivo têm de estar notados em ±45 ms |
| T16–T19 | pré-condições (clipping, silêncio, duração, campo estéreo, faixa tonal **calibrada por medida**) e paridade SVG↔PDF↔páginas↔total de notas, com página vazia = erro |

A aba **Auditoria** roda as leis (`rápida` / `completa`), lista cada achado com a evidência, exporta o
relatório e aplica **Correções automáticas**: o agente conserta, re-audita, itera até o ponto fixo e
**não entrega** uma partitura que audite pior do que entrou (cada rodada é revertida se for o caso);
ele nunca move a posição de uma nota — o que está errado é o valor de duração/campo, não o instante em
que o baterista bateu. A auditoria roda sozinha depois de cada adoção de sugestão e o resumo aparece no
selo da aba.

```bash
python3 tests/doublecheck.py                # demo (wav + mp3): auditoria + injeção de 7 defeitos
python3 tests/doublecheck.py --fix          # idem, deixando o agente consertar
python3 tests/doublecheck.py --file x.wav --gt x.json --level full --out out/qa
python3 tests/http_qa_check.py                # as rotas HTTP que a aba usa, contra o servidor de pé
python3 tests/long_check.py                  # o contrato de upload: custo, silêncio inicial, 6/8 (25; ~40 s)
```

O `doublecheck` é o teste honesto do verificador: parte de um estado que **sabemos** lícito (as mesmas
leis, medindo o mesmo objeto, sem ruído em excesso) e depois fura a partitura — `tick` ilegal, nota
duplicada, pista inexistente, `dur=0`, `artic="blast"`, compasso 0 `repeat=5`, `time` trocado por
engano — exigindo que cada furo seja **visto** e, com `--fix`, **reparado**, sem perder nota lícita e
sem ganhar problema novo. Causas raiz e provas de cada erro confirmado estão em
[`docs/ERROS.md`](docs/ERROS.md).

## Onde o áudio está (cursor na partitura)

Tocar o arquivo e olhar a pauta só serve se a pauta disser **onde você está**. Então o cursor não é
adivinhação: ele é derivado da mesma geometria que produziu a gravura.

| peça | o que garante |
|---|---|
| `layout.bar_map` (servidor) | a caixa de cada compasso **no espaço do documento** — `x0/x1`, `y0/y1`, `line1` (a 1ª linha da pauta, com a página somada, como as demais) e a área útil `beat0…beat1` — medida na passada de `layout_score`, nunca reconstruída no front |
| `layout.clock` (servidor) | `t0` = origem da grade **medida** sobre as notas (`qa.clock_map`: mediana de `t − (compasso·tpr + tick)·spt`), `bar_len`, dispersão e nº de golpes que a sustentam. `phase_ms` **não** é a origem — ver [`docs/ERROS.md`](docs/ERROS.md) A11 |
| `gradeBase()` (navegador) | decide a base numa ordem honesta: síntese → `t0 = 0` (o `kit_synth` toma a partitura como relógio) · arquivo → `layout.clock` · sem relógio → mediana local sobre ≥ 8 notas medidas · e só então o relatório, com o rótulo `"relatório (pouca evidência)"` |
| `posicaoCursor()` | `u = (t − t0)/bar_len` → compasso `floor(u)` → fração dentro da área útil. Função pura, sem DOM: é testada à parte |

Do que depende: cursor + banda do compasso tocando (com a barra dupla de fim de sistema dentro da
banda), leitura "compasso N · batida M" no `#cur-read`
(com a precisão da origem no `title`), rolagem automática até a linha do compasso (o "seguir o compasso"
liga/desliga), clique na pauta para buscar (a posição x escolhe a **batida**, não só a caixa),
shift+clique para tocar o compasso em *loop* — e a grade de batidas pintada na forma de onda. Todos os
seis saem de `gradeBase()`, logo não há como o cursor de um lado mentir enquanto o do outro acerta.

Honestades incluídas: com poucos golpes medidos a mediana local **recusa** (`null`) em vez de chutar; a
síntese e o arquivo têm origens diferentes e isso é dito na interface; a divergência entre `phase_ms` e
a origem medida (1.951 ms no demo) é medida e exposta no `title` do leitor — não escondida.

```bash
python3 tests/http_qa_check.py     # "o meio de cada compasso acende exatamente o seu compasso" (17/17)
node tests/ui_boot_check.js        # cena F: 16 verificações do relógio, do clique e do fallback
```


---

## Conferir a leitura: zoom na onda, MIDI na tela e partitura simples

Três coisas que servem à mesma pergunta — *o que a plataforma ouviu é o que ela escreveu?*

**Zoom na aba “Áudio & batidas”.** A alavanca da barra (e `+`/`−`/`0`, roda do mouse,
`shift`+arraste para marcar uma região, `b` para enquadrar o compasso do cursor) define uma
**janela em segundos** que é partilhada com o piano-roll do MIDI. Dentro da janela, o canvas pinta
três faixas do mesmo instante: a forma de onda a ~1 ms/balde, a **novidade espectral por banda**
(grave/média/aguda e o máximo global — o sinal que o detector usou) e **o que a partitura gravou**:
tique cheio = virou nota, meio pino = dois ataques que couberam numa nota, caixa vazada = ataque
detectado sem nota. Resíduo de rejanelagem acima de 25 ms ganha ponto âmbar. É assim que se vê, sem
interpretar relatório, que o pico do espectro e a nota escrita batem.

O dado vem de `GET /api/wave/<id>`, que lê dois planos `float32` gravados durante a análise
(`out/wave/<id>.onda.f32`, `.curvas.f32`) por **deslocamento de bytes**: a demo cabe em 42 000 baldes
= 1,0 MB, e uma chamada de zoom custa ~18 kB de JSON. Nada é re-decodificado no processo que atende
o navegador — foi o preço da lei A13 (quem decodifica arquivo do usuário é o filho isolado). Se o
envelope fino não existe (faixa rebaixada por memória), a aba avisa e mostra a visão geral, em vez de
fingir resolução.

**MIDI na tela, não só no download.** A aba “MIDI” gera o `.mid` da partitura e o **relê nota a nota**
com o parser próprio do auditor (`qa._read_smf`, o mesmo que o T12 usa): as barras do piano-roll são
os `note on`/`note off` do arquivo, com o tempo vindo de `tempo_us / division` do cabeçalho. Por isso
a soma fecha com a pauta: na demo, 296 notas no arquivo = 296 escritas = 296 nos dados; com pratos
ocultos, 168 = 168, e os dados seguem 296. Clicar numa nota move o cursor de reprodução.

A auditoria acompanha isso sem se confundir: `qa.check_exports` regera o arquivo e compara com o
que o **escritor** escreve (o corte visível), enquanto as leis de escrita T1–T10 continuam julgando
todas as notas — e o achado `T10p` declara as duas contagens. Sem isso, ocultar pratos fabricava 2
erros de auditoria num resultado correto (A18).

**Partitura simples com pratos opcionais.** `simples` (ou o botão da barra) esconde as sete pistas de
prato `kit.PRATOS` da gravura, do PDF, do MIDI e do MusicXML. É **apresentação**: o dicionário, o
JSON, o CSV e a auditoria continuam com todas as notas, o `mapa de compassos` não se move (o cursor
continua certo) e o `report.apresentacao` grava quantas notas havia. Filtrar por `lanes` foi recusado
porque remanejava rótulos e mudava a transcrição — ver A16.

**Recomendador de parâmetros** (`POST /api/tune`, botão em “2 · Análise”). `sensitivity` e
`min_confidence` são os dois parâmetros que mais mudam o resultado, e sem gabarito não há como
escolhê-los: baixar a confiança apaga chimbel, subir a sensibilidade ganha duplicata. A plataforma
mede a sua faixa numa varredura de até ~17 análises (6 sensibilidades × até 4 confianças, com
orçamento proporcional à duração) e pontua cada ponto por duplicata (piso de lacuna
intra-pista), resíduo e fração fora da grade, densidade, presença de bumbo+caixa por compasso,
fração de queda-de-braço em `hat*` e nº de erros da auditoria — tudo lei que já existia aqui,
nenhuma constante ajustada para um arquivo. O resultado é a tabela de candidatos, o custo, os motivos
por extenso e um botão “aplicar”. Onde os critérios não distinguem dois pontos, **isso é dito**
(“N dos pontos testados ficaram com custo zero; entre eles escolheu-se o mais próximo da calibração
do projeto”), e o f1 do gabarito acompanha a decisão no `tests/tune_check.py`.

## Faixa longa: o que a plataforma faz quando não cabe

O pico de memória da cadeia cresce com o comprimento da faixa, e a máquina deste preview tem
2 GB e nenhum swap. Duas atitudes, e nenhuma delas é “tentar assim mesmo”:

1. **Medir antes de alocar.** `plano_dsp` estima o pico (`60 B` por amostra × penalidade de
   Bluestein + `4 B` por quadro·bin + 140 MB de interpretador — coeficientes ajustados por
   mínimos quadrados sobre picos reais), escolhe a **taxa de análise** mais alta que cabe e
   declara o degrau em `report.memory`, com aviso no topo da partitura. Reduzir a taxa não move
   uma nota: `hop` e `n_fft` descem junto, então **5,80 ms por quadro e 43,07 Hz por bin
   continuam os mesmos** — o que se perde é o topo do espectro (Nyquist), e isso é dito no
   relatório. Se nem o degrau mais econômico couber, recusa-se com a conta feita.
2. **Rodar fora do servidor.** A análise inteira acontece num processo filho `spawn`, com
   resultado entregue por fila. O pior caso deixou de ser “a aba morre”:

   | falha | resposta |
   |---|---|
   | o kernel matou o filho por memória | 500 `memoria_nucleo` — diz o sinal, o motivo e que a plataforma segue no ar |
   | não cabe em nenhum degrau | 413 `memoria` — com MB pedidos, MB disponíveis e o que fazer |
   | passou do prazo | 504 `prazo` — o filho é encerrado de propósito, o servidor não |
   | arquivo indecodificável / vazio | 422 `arquivo` / 400 |
   | upload acima de 200 MB | 413 `grande_demais` em JSON (não o HTML do Werkzeug) |

Medido no servidor deste preview:

| arquivo | o que aconteceu |
|---|---|
| 10,4 s (demo) | 200 em ~9,8 s, 17 compassos, 292 notas, **sem** degrau automático |
| 126 s com orçamento forçado de 480 MB | 200 em 14,1 s, degrau declarado no relatório, auditoria sem erro |
| 443 s — faixa real do usuário, 17,7 MB, 11 s de silêncio no início | **200 em 26,5 s**: 239 compassos, 2999 notas, 6/8, 99,11 BPM, rebaixada para 22 050 Hz (`ajustado_por_memoria`), pico 718 MB contra 804 previstos. Antes desta rodada devolvia 413 — ver `docs/ERROS.md` A19/A20 |
| 1 h (3600 s) | 413 em <2 s, com a conta na mensagem: “o plano mais econômico pede 5540 MB contra ~985 MB disponíveis” |

A recusa por comprimento **não é** uma política de tamanho: é o resultado da conta de memória, e a
mensagem diz os dois números. Um arquivo que cabe no degrau de baixo é transcrito no degrau de baixo
(com `ajustado_por_memoria: true` no relatório); um que não cabe em nenhum é recusado em <2 s, sem
decodificar, e a plataforma continua de pé. O silêncio do começo é cortado e **declarado**
(`report.file.trim_lead_ms` = 10 980 ms no caso acima), porque é ele que alinha o `.mid` com o MP3.

Portas: `python3 tests/profile_long.py --limite 900` (uma duração **por processo**: `ru_maxrss`
é alto-nível acumulado, e medir durações seguidas no mesmo processo faz um pico engolir o outro)
e `python3 tests/tiers_check.py` (nenhum degrau que a máquina pode escolher sozinha move
`pos_accuracy`/`tick_mae`; degrau que cala pista não entra na escada).

Variáveis do servidor: `PORT`, `DRUMSCRIBE_PRAZO_ANALISE` (segundos, padrão 900),
`DRUMSCRIBE_INLINE` (analysis no próprio processo — depuração; abre mão do isolamento),
`DRUMSCRIBE_FALHAR_FILHO=oom|memoria|trava` (injeção para testar os três caminhos acima).

## Limitações assumidas

* **Ride e crash** ainda são pobres: quando o chimel está tocando junto, a energia metálica se
  parece demais; o portão usa a inclinação espectral do decaimento, que a sangria borra.
* Falsos positivos de **bumbo** aparecem quando o caixa tem corpo grave ou há eco curto: por isso
  existe `sensitivity` e o editor.
* **Downbeat ambíguo**: um backbeat simples de 4/4 (bumbo em 1 e 3, caixa em 2 e 4) é invariante
  a um deslocamento de dois tempos. O sistema resolve com o template + a abertura da peça, mas
  se você discordar, mexa em *deslocamento da barra* (ticks) — a partitura inteira reencadeia.
* Compensação de tempo humano (push/lateness) não é notada como "atrasado": tudo é escrito na
  grade; o desvio fica no relatório (`median_residual_ms`) e no JSON.
* Sem cifra de violão/melodia: é um transcritor de **bateria**.

## API (usada pelo front-end, serve para script também)

```
POST /api/analyze      multipart: file, params (JSON)  → {id, score, report, wave, svg, layout, audio_url}
    `layout` = geometria do cursor: {bar_map[{i,page,x0,x1,y0,y1,line1,beat0,beat1,folga}],
    clock{t0,bar_len,fonte,n,spread_ms,max_ms}, page_w, page_h, n_pages, doc_h, staff_space,
    ticks_per_bar, bars_per_system} — ver A11 em docs/ERROS.md

GET  /api/demo         analisa a faixa do repositório
GET  /api/result/<id>.json
POST /api/svg          {score, opts}                    → SVG atualizado
POST /api/rebuild      {id, hits, bpm, meter, swing}    → partitura reescrita após edição
POST /api/synth        {id|score}                       → WAV da transcrição (síntese do kit)
POST /api/export/<fmt> fmt ∈ pdf|svg|musicxml|mid|json|csv
GET  /api/audio/<id>   o arquivo enviado, para o player
GET  /api/health       sonda: versão, dependências presentes, pasta, nº de resultados
GET  /api/lanes        mapa de peças (posição na pauta, GM, cabeça de nota)
GET  /api/wave/<id>    ?t0&t1&n≤4000&curvas=0|1             → envelope fino da janela (lê o
    plano float32 gravado na análise, por deslocamento de bytes: nada é re-decodificado no pai).
    Devolve min/max por balde (~1 ms), as quatro curvas de novidade, os ataques da janela com o
    estado `e`/`m`/`n` (virou nota / fundidos / sem nota) e as contagens das duas contagens.
POST /api/view         {id, hide_lanes|[simples]}           → regravura com pistas ocultas — só
    gravura, MIDI e MusicXML; JSON/CSV/auditoria continuam com todos os dados (A16)
POST /api/tune         {id}                                 → varredura de `sensitivity` ×
    `min_confidence` na sua faixa, com custo, motivos e a tabela de candidatos (cacheada por id)
POST /api/midi_view    {id}                                 → gera o .mid e o relê nota a nota com
    o parser do `qa` (tick→segundos pelo `tempo_us`/`division` do próprio arquivo)
POST /api/qa           {id, level:fast|full}                → {qa, markdown} da partitura salva
POST /api/qa/fix       {id, level}                          → re-auditoria + partitura corrigida + SVG
```

Exemplo por linha de comando:

```bash
curl -F "file=@meu_kit.mp3" -F 'params={"sensitivity":0.95,"meter":"auto"}' \
     http://localhost:8000/api/analyze | jq -r .id
curl -X POST -H 'content-type: application/json' -d '{"id":"<ID>"}' \
     http://localhost:8000/api/export/pdf -o partitura.pdf
```

Biblioteca, sem servidor:

```python
from drumscribe.pipeline import transcribe_file
r = transcribe_file("meu_kit.mp3", params={"sensitivity": 0.95, "grid_mode": "auto"})
print(r["report"]["tempo"]["bpm"], r["report"]["meter"]["value"], r["report"]["grid"]["mode"])
r["score"].to_dict()                                  # JSON único: pauta, vozes, report
from drumscribe.engrave import write_pdf
write_pdf(r["score"].to_dict(), "partitura.pdf")
```

## Parâmetros

| chave | padrão | efeito |
|---|---|---|
| `sensitivity` | 0,95 | limiar de proeminência dos onsets em cada banda (menor = mais notas) |
| `min_confidence` | 0,30 | corta golpes cuja classificação é incerta |
| `refine_frac` | 0,62 | onde na subida de energia marcar o ataque (0,5 = meio; maior = mais cedo) |
| `grid_mode` | `auto` | `8th`, `16th`, `32nd+16th`, `16th+triplet` forçam a grade |
| `meter` | `4/4` | `auto` testa 2/4, 3/4, 4/4, 5/4, 6/8, 9/8, 12/8 |
| `swing` | `auto` | 0 = reto; 0,60–0,72 = grau de tercinização |
| `bpm_hint` | 0 | força a busca de andamento ao redor do valor |
| `downbeat_shift_ticks` | 0 | move a linha de compasso (correção do ambíguo 1×3) |
| `show_aux` | falso | inclui rim shot, pedal de chimel, splash, cowbell |
| `fill_detect` | verdadeiro | sublinha viradas e marca compassos de repetição (𝄄) |
| `page` (na exportação) | `a4_landscape` | também `a4_portrait` / `letter_landscape` |
| `analysis_sr` | 44 100 | taxa com que a faixa é reamostrada para análise; a escada de memória pode descê-la até 22 050 (abaixo disso o chimbal fica sem banda — A14) |
| `n_fft` / `hop` | 1024 / 256 | janelas **na taxa de referência de 44 100 Hz** (23,2 ms de janela, 5,80 ms por quadro); reescaladas junto com `analysis_sr`, de propósito — mudar `hop` sozinho move ticks (A14) |
| `memory_budget_mb` | auto | teto de pico da análise; auto = 95 % do que o kernel reporta disponível. Subir isso é assumir o risco do OOM |
| `max_seconds` | 600 | o que passar disso é descartado, **com aviso** no relatório (a partitura cobre só o trecho analisado) |
| `simples` | falso | partitura “simples”: esconde `kit.PRATOS` **na gravura** (bumbo, caixa e tons ficam). Não toca em classificação, grade nem dados — A16 |
| `hide_lanes` | — | lista explícita de pistas ocultas; sobrepõe `simples`. Nome inválido devolve 400 com a lista das válidas, em vez de silenciar |
| `onda_fina` | verdadeiro | grava o envelope a ~1 ms e as curvas de novidade por banda (≈1 MB por minuto de áudio, em `out/wave/`), para o zoom da aba Áudio não re-decodificar nada |

## Arquivos

```
drumscribe/  audio_io dsp grid classify rules kit kit_synth score_model layout
             engrave musicxml midi_out pipeline qa tune server
static/      index.html app.js style.css          (sem framework, sem CDN)
out/wave/    <id>.onda.f32 <id>.curvas.f32        envelope a ~1 ms do zoom (cache; limpo no boot)
samples/     make_demo.py demo_drums.{wav,mp3} demo_groundtruth.json
             tonal_calibration.json (limiares medidos pelo auditor)
tests/       evaluate.py (gabarito) selfcheck.py (ponta a ponta)
             doublecheck.py (auditoria + injeção de defeitos) calibrate_tonal.py
             http_qa_check.py (rotas HTTP, contra o servidor de pé)
             tiers_check.py (leis do degrau de memória: grade e colocação)
             profile_long.py (pico de RSS por duração, 1 filho por caso; --limite MB vira lei)
             make_long.py (gera as faixas longas em /tmp para os dois acima)
             ui_boot_check.js (boot do front sob rede morta/pendurada — node; a cena G
               verifica que o zoom da aba Áudio encolhe a janela e que o pedido leva t0/t1, e a
               cena H que uma análise que falha limpa a folha em vez de deixar a partitura antiga)
             ui_css_check.py (invariante: nada que se esconde com `hidden` pode ficar pintado)
             wave_check.py (leis do zoom: resolução, nº de pontos, contenção de pico, casamento
               com a pauta, escala de tempo do .mid e "ocultar é apresentação" — 36 verificações)
             long_check.py (contrato de upload: custo previsto ≥ pico medido, recusa com margem,
               126 s dentro do prazo, ataque após 12 s de silêncio, grade 6/8 e duração na barra)
             tune_check.py (o recomendador nunca piora f1/recall/precisão contra o gabarito; ~4 min)
docs/        plan.md (arquitetura, fórmulas, decisões)
             ERROS.md (erros confirmados, causas raiz, limitações)
requirements.txt  run.sh
```
