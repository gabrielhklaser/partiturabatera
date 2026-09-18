# DrumScribe — plano técnico e decisões

Documento de arquitetura. Destinado a quem vai manter o código: o *porquê* de cada escolha,
as fórmulas usadas e o que foi tentado e **não** funcionou (para ninguém reintroduzir).

## 1. Problema e recorte

Entrada: um arquivo de áudio contendo **bateria isolada** (stem de estúdio, ou gravação de
ensaio). Saída: **partitura de pauta de percussão** legível por baterista, com durações,
acentos, fantasmas, ligações de prato e a métrica correta; exportável em PDF/MusicXML/MIDI.

Recorte deliberado: não há tentativa de separar fonte (não é um *source separator*); a
suposição é que o sinal de entrada já é (aproximadamente) só a bateria. Tudo o que existe de
aprendizado de máquina aqui é estatística física por banda — nada de rede neural — porque a
exigência do projeto é *transcrição auditável*: cada nota na partitura tem um rastro de
medições no relatório.

## 2. Cadeia

```
áudio → STFT → fluxo por banda → onsets por GRUPO → features por evento
      → classificação (escore + portões) → suave de pratos → dedupe por rota
      → andamento (ACF/prior/pente) → refino por resíduo → métrica/downbeat
      → quantização (grade mínima suficiente + histerese de swing)
      → re-classificação com contexto → regras de escrita → gravura → PDF/SVG/XML/MIDI
```

### 2.1 Espectrograma e onsets (`dsp.py`)

* STFT `n_fft=1024`, `hop=256` → 172 quadros/s; `hop=512` acima de 210 s (memória/tempo).
* `band_flux(spec, f0, f1, kind="energy_rise"|"half")`: fluxo de meia-ondinha por banda, com
  realce espectral leve; para a banda grave usa `energy_rise(smooth_ms=13)`, que mede a subida
  (não a diferença) — o bumbo tem corpo oscilante e `diff` puro duplica eventos.
* `find_onset_frames(...) → (quadros, proeminência relativa)`: pico local acima de
  `piso + kσ` **e** com proeminência relativa (pico ÷ p99 do fluxo da banda). A proeminência
  relativa substituiu o "strength = novelty/máximo", que não separava nada (chimel e bumbo
  chegavam ambos a 1,0).
* `refine_onset_frames(o, frames, frac=0.62)`: recua no tempo até `frac` do pico de subida, o
  que alinha o ataque com a transição perceptiva; `refine_frac` é exposto na API.
* **Multi-rota por grupo**: `low` (28–170 Hz), `mid` (150–3200 Hz), `high` (3200–20000 Hz) são
  detectados **separadamente**, cada um com seu `min_gap`/limiar, e cada grupo tem seu próprio
  conjunto de rotas candidatas (`GROUP_LANES`). Um bumbo + chimel no mesmo pulso viram dois
  eventos. *Decisão-chave: nunca fundir bandas em um único evento — foi a causa do colapso de
  acordes na primeira versão.*
* `min_gap`: 62/34/28 ms (grave/médio/agudo). O valor grave é alto porque o ringing do bumbo e
  o slap-back da sala criam candidatos; 34 ms gerava ~112 candidatos falsos na faixa de teste.

### 2.2 Features físicas por evento (`dsp.extract_features`)

| feição | definição | para que serve |
|---|---|---|
| `band_db[j,i]` | 10·log10(mean energia dos quadros do evento, banda j) | nível absoluto (contaminado pelo ring anterior) |
| `band_delta_db[j,i]` | banda do evento − média dos **40 ms anteriores** | **a** grandeza discriminativa: remove o sustain herdado |
| `band_delta_rel` | delta normalizado em [0,1] por banda | portões ("separação suave de fontes") |
| `sub_ratio` | E(18–62 Hz) / E(18–170 Hz) | assinatura do bumbo; caixa fica em ~0,08, bumbo ~0,50 |
| `hi_slope` | dB/s da banda alta na janela longa (420 ms) | chimel decai rápido (−90 dB/s), prato de condução devagar (−20) |
| `low_slope` | idem na banda grave | tom/caixa mantêm, bumbo despenca |
| `hf_centroid`, `flatness`, `centroid`, `rolloff` | timbre do ataque | rim shot vs baqueta, splash vs crash |
| `attack_ms`, `decay_*_ms`, `rise_ratio` | envelope | sustento de prato, ghost de caixa |
| `pitch_low` | estimativa por autofaseamento da componente grave | bumbo entre 28–155 Hz |

A janela do evento é cortada no onset seguinte (piso de 85 ms) — senão, em chimel a 16ºs, todas
as durações/queda saem zeradas.

### 2.3 Classificação (`classify.py`)

`score_matrix` = Σ_banda w[lane,banda]·z(`band_delta_db`) + termo de sustain (para
crash/ride/hat_open) + bônus de contexto (`downbeat`/`dense`/`loud`) + prior de proeminência do
grupo. `hard_gate_mask` aplica portões de física, por grupo:

* **bumbo**: `sub_ratio ≥ 0,28`, `band_delta_db[sub] ≥ 4 dB`, energia grave ≥ 0,38 da referência,
  `low_slope ≤ −8 dB/s`, `pitch_low ∈ [28,155]`;
* **caixa**: delta em `midlow`/`mid`/`midhi` positivos e dominantes, `flatness` média, `sub_ratio`
  baixo (senão é bumbo), `hi_slope` intermediário;
* **chimel**: `band_delta_rel[high] ≥ 0,34` e `hi_slope ≤ −42` (decaimento curto);
* **ride/crash/hat_open**: `hi_slope > −42` (sustento); crash ainda exige `vhigh` ≥ −0,35;
* **tons**: `midlow`/`mid` com `pitch_low` em janelas próprias; tom agudo ≠ caixa pela razão
  `mid/midhi`.

Quando o portão reprova mas o grupo tem proeminência real (`st ≥ GROUP_FALLBACK_ST[grupo]`), o
evento recebe a **rota primária do grupo** com confiança 0,30 (bumbo/caixa/chimel) em vez de
ser jogado fora — recall > pureza. Rejeitar em vez de rebaixar destruiu o recall na primeira
tentativa.

Suavização de pratos (`_smooth_cymbals`): crash/hat_open só são rebaixados para a maioria de
hat/ride vizinha; **ride nunca é rebaixado** (a regra anterior engolia o ride inteiro).

### 2.4 Andamento e grade (`grid.py`)

* `estimate_tempo`: ACF do envelope composto + prior log-normal em 76–186 BPM + pente harmônico
  (½×, 2×) → deixa ±0,3 % de erro.
* `fit_grid`: mínimos quadrados robustos de `tempo = a·tick + b` sobre eventos aceitos, com
  rejeição por MAD apertando 3,5→2,6→2,0→1,8 por iteração (4 iterações). Rejeita-se o `bpm` do
  `fit_grid` se diferir do ótimo do pente em > 3,5 % (proteção contra oitava).
* `refine_bpm`: varredura de BPM (0,02 de passo, 3 varreduras coarse-to-fine) e de fase
  (48 fases por beat) maximizando o **ajuste ponderado à grade**, seguida de polimento de fase
  pela mediana do resíduo com sinal → erro de BPM 0,01 % na faixa de teste (contra 0,3 % do
  pente puro).
* `quantize` — **grade mínima suficiente**: parte da grade de 16ºs (sempre a referência de
  fase), aceita 8ºs se o fitness não cair, e só promove um evento individual aos 32ºs se ele for
  **confiante** (`w > 0,20`), estiver a > 0,58 tick dos 16ºs e o slot de 32ºs reduzir o resíduo
  em > 0,25 tick. Sem essa regra, meia dúzia de falsos positivos faz o `argmax` de fitness
  escolher 32ºs e a partitura inteira fica ilegível (erro real observado: pos_accuracy 1,0 → 0,72).
* swing: só troca para grade swingada (0,60/0,67/0,72) se o fitness melhorar **> 4 %**
  (histerese); a decisão muda o feel da leitura, não pode ser um empate.
* `METER_TEMPLATES` em **espaço de ticks** (não em índice de semicolcheia), com anti-templates,
  para que 12/8 ≠ 4/4; `estimate_meter` escolhe fórmula e offset de downbeat por kernel gaussiano.
* `align_downbeat`: os ticks já projetados + template de backbeat (média **por peça**, ponderada
  por confiança — a soma favorecia offsets com mais falsos positivos) e a evidência de abertura
  da peça, combinados 0,72/0,28. Necessário porque o backbeat 4/4 é **invariante a ±2 tempos**:
  nenhuma métrica de encaixe resolve sozinha "bumbo no 1" vs "bumbo no 3". O usuário corrige com
  `downbeat_shift_ticks`.

### 2.5 Regras de escrita (`rules.py`)

* `fill_voice`: para cada voz, a duração de cada nota é o maior valor padrão que cabe até o
  próximo ataque da **mesma voz**; nunca cruza a divisão do tempo, exceto quando começa exatamente
  no beat (aí vale o beat inteiro) ou quando é sustain de prato (usa liga). Pausas preenchem o
  compasso na forma canônica (pontilhada/represada). Em compasso composto a divisão de referência
  é a colcheia pontuada.
* acento (`>`) para velocidade alta, `notehead="minus"` + parênteses para ghost (≤ 46), flam para
  dois golpes da mesma peça a 22–65 ms (o segundo é a nota principal), roll para repetição no
  mesmo pulso, `bars_n`/`tie` para prato sustentado, sublinhado para virada, 𝄄 para compasso
  idêntico ao anterior a partir da 3ª ocorrência (`detect_repeats`).
* velocidade MIDI: `v = clip(round(78 + 4,2·(0,55·Δglobal + 0,45·Δpeça)), 24, 127)` onde Δ é em
  dB relativo à mediana (global e da própria rota, para que um chimel `pp` não vire `ff`).

### 2.6 Gravura (`layout.py` → `engrave.py`)

**Uma única geometria** gera SVG (navegador) e PDF (reportlab): o layout devolve primitivas
(`line, path, poly, ellipse, rect, curve, text, noteglyph`) num sistema com Y para cima; cada
renderizador só traduz. Assim é impossível a pré-visualização discordar do PDF.

* pauta de 5 linhas, `ls = 9 pt`; passo diatônico (0 = D4 … 11 = A5) → `y = linha1 + (passo−1)·ls/2`;
  linhas suplementares acima/abaixo; clave de percussão + armadura só no 1º sistema da página.
* voz 1 (pratos/chimel) com hastes para cima, voz 2 (bumbo/caixa/tons) para baixo; vigas por
  divisão do tempo com **sub-vigas** por par (nível = log2 da duração), nota isolada ganha
  bandeira; pausas no centro da pauta.
* cabeças: círculo cheio/branco (duração ≥ mínima), `x`, `circle-x` (chimel), `minus` (ghost),
  losango (aberto), triângulo (cowbell); `+` sobre chimel aberto; linha de sustain sobre prato;
  ligadura em curva quadrática só quando existe parceira no compasso vizinho.
* quebra em sistemas: `largura útil // 126 pt` compassos por sistema (mín. 2, máx. 8); um sistema
  ocupa 15,2 `ls` (10,1 acima da linha inferior, 5,1 abaixo); legenda do mapa de teclado só com
  as peças efetivamente usadas; cabeçalho com ♩ desenhado (a fonte base-14 não tem o glifo — o
  SVG usa vetor também, para as duas saídas ficarem idênticas).

### 2.7 Exportação

* **MusicXML 4.0** partwise: `divisions=8` (= 1 tick), `<clef sign="perc" line="2"/>`,
  `<unpitched>` + `<staff-position>` por nota (a posição no kit não é altura), duas vozes com
  `<backup>`, `notehead` `x|circle-x|minus|diamond`, `<artics><accent/><roll/></artics>`,
  `<grace/><cue/>` para flam, `<tie>`/`<tied>` para sustain, `<sound tempo>`, `<repeat>` quando a
  partitura tem seção repetida.
* **MIDI** SMF tipo 1, PPQ 480 (1 tick = 60 PPQ, exato), canal 10, trilha de mapa com
  tempo/armadura/marcadores por compasso; pratos ganham duração dobrada.
* **CSV** com BOM (Excel pt-BR), **JSON** = contrato da pauta (abaixo).

### Contrato do dicionário-Score (fonte única)

```
{title, subtitle, bpm, meter, swing, repeats, ticks_per_quarter:8, ticks_per_beat,
 ticks_per_bar, beats_per_bar, compound, lanes:[ids], report, n_bars, total_ticks,
 bars:[{index, repeat_slash, fill_marker,
        hits:[{lane, tick, dur, velocity, artic, confidence, head, tie_start, tie_stop,
               time, bars_n, voice, stem, staff, notehead}]}]}
```

## 3. Validação

* `samples/make_demo.py` sintetiza 10 seções (groove, viradas, chimel aberto, ghost, flam,
  ride, 3/4, swing) com **ground-truth** em ticks; o eco de sala é 0,07 @ 11 ms (16 ms de slap
  já quebravam a detecção de bumbo — ver §4).
* `tests/evaluate.py` pareia por (tempo, rota) com tolerância de 40 ms e atribuição gulosa, e
  mede recall/precisão de onset, F1 de acerto, `pos_accuracy` (alinhamento de compasso por
  **moda**, para que um FP inicial não desloque a métrica), MAE de tick, viés/MAE de tempo,
  Pearson de velocidade, confusão de rotas e BPM/compasso.
* `tests/selfcheck.py` roda a cadeia inteira e exige 40 invariantes: contrato do dicionário,
  limiares de métrica, SVG/XML válidos, PDF renderizável (via pypdfium2), contagem de notas no
  SMF igual à da partitura, e `rebuild_score` sem duplicar nota no mesmo pulso.

## 4. O que foi tentado e falhou (não reintroduzir)

1. **Fundir as bandas em um evento rotulado** → acordes colapsam e o corpo grave da caixa gera
   bumbos fantasmas. Multi-rota por grupo é obrigatório.
2. Comparar só o vizinho temporal no dedupe → chimel denso intercalado deixa duplicatas; tem de
   ser **por rota** (`last_by_lane`). E a rota `kick` não pode estar no grupo `mid`.
3. `strength = novelty/máximo` como confiança → inútil; usar proeminência relativa por banda.
4. Portões sobre o **nível absoluto** da banda → o sustain do golpe anterior contamina; usar
   `band_delta_db` (evento − 40 ms anteriores).
5. Portão de bumbo usando `f_hf` baixo como válvula de escape → reintroduziu os fantasmas de
   ringing. A exigência é `sub_ratio` + `low_slope`, não a ausência de agudo.
6. Rejeitar quando o portão falha → recall despenca; rebaixar para a rota primária do grupo.
7. Prior de crash forte no downbeat (+0,85) → todo chimel de início de compasso virou crash.
8. Limiar global no novelty composto → perde chimel baixinho e ghost de caixa.
9. Templates de métrica em índice de semicolcheia → 12/8 confundido com 4/4.
10. Confiar no ACF para o BPM → 0,2–0,3 % de erro (visível em 40 compassos); sempre
    `fit_grid`/`refine_bpm`.
11. Preferir a grade mais grossa por "argmax de fitness" com σ relativo ao espaçamento → a grade
    fina sempre vence; σ **fixo** (1 tick) + promoção individual de 32ºs resolveu.
12. Quantizar o offset de downbeat ao beat → desloca tudo em 1–2 ticks e a métrica de posição
    zera; o offset fino (22 ticks na faixa de teste) é o correto.
13. `min_confidence` alto (> 0,4) → corta as rebaixadas e derruba o recall para 0,27.

## 4.1 Rejeitados nesta rodada (não reintroduzir)

* **Re-decodificar o áudio a cada passo de zoom.** A alavanca de zoom é arrastada quadro a quadro;
  reabrir o arquivo a cada chamada de roda é o que derrubou a plataforma antes. O caminho é um só:
  gravar os planos finos uma vez por análise (`out/wave/`) e ler só a janela pedida.
* **Uma pirâmide de picos em JSON (máx/min por bloco, cada nível metade do anterior).** São ~4 MB de
  JSON por música, com a desvantagem de a geometria ter de ser reprojetada no cliente. Os dois
  arquivos binários com `dtype`/`n`/`nc`/`bucket_sec` no header são menores, mais diretos e já
  respondem com `dt = bucket_sec·k` quando a janela precisa ser agrupada.
* **Régua de contagem de picos para escolher a sensibilidade** (ver A17): contaria 94 ataques contra
  236 eventos de gabarito na demo, porque o chimel em semicolcheia cai abaixo de proeminência fixa.
  Uma régua *discriminante* mas *errada* é pior que nenhuma.
* **Filtrar pratos por `params["lanes"]`** (renomearia as pistas) ou **no cliente** (o `index.html`
  não é dono de `kit.PRATOS`) — ver A16.

## 5. Próximos passos honestos

1. **Ride/crash**: medir `hi_slope` apenas no vão **antes** do próximo onset (o sustento atual é
   poluído pelo chimel seguinte) e re-ajustar o portão; hoje ride tem recall 0,04.
2. Falsos positivos de bumbo: exigir *subida* na sub-banda (não apenas nível) — devem sobrar
   ~29 fantasmas hoje.
3. Atraso/adiantamento humano (push/late feel): publicar `time_bias_ms` por peça no
   relatório do app. Na faixa de teste o viés de medição é ~ −6 ms e é absorvido pela fase, mas
   em gravação real essa informação vale ouro para um baterista.
4. Compasso de Pickup (anacruse) e mudanças de andamento/métrica no meio da música — hoje assume
   andamento único.
5. Um importador de áudio com `ffmpeg` direto (o fallback já existe) e suporte a estéreo com
   dois microfones (a caixa no centro, os pratos nos lados).

## 6. Rodada 2026-09-17 — zoom no áudio, roll de MIDI, auto-calibração, “simples”

O pedido: zoom na aba **Áudio & batidas** com as marcações visíveis, uma **visão MIDI** além da
partitura, um **analisador que recomenda `sensitivity`/`min_confidence` para a música** e uma
**partitura simples** com pratos mostrando/ocultando. Módulos e decisões, um parágrafo cada.

**Plano fino (`audio_io.peaks_finos` / `audio_io.planes_finos` → `out/wave/<id>.{onda,curvas}.f32`).**
`peaks_envelope` é fixo em 1800 baldes (~23 ms), que é a resolução em que o zoom morre — a 240 % a
janela tem 250 ms e os 1800 baldes seriam 1,4 ponto na tela. `peaks_finos` é a mesma aritmética de
máx/min com `dt_alvo = 1/1000 s` (mantém `hop`/`n_fft` do plano, logo `bin_range` por banda é
idêntico e as curvas de novidade continuam falando a língua dos eventos). `planes_finos` empacota as
duas coisas num contêiner bruto com header textual (`dtype|n|nc|bucket_sec|sr|origem`), `float32`,
`(2,n)` para a envoltória e `(nc,n)` para as curvas; o `sr` é do arquivo, não da análise, porque a
envoltória é lida em amostras do arquivo.

**`GET /api/wave/<rid>?t0&t1&n`.** Lê só o trecho pedido (memória mapeada), agrupa em `n` pontos,
devolve envelope + as quatro curvas de novidade **já escaladas pelo máximo da própria janela** (para
caber, como a envoltória) e os eventos classificados da janela com `estados`/`st`/`n_notas_janela`
calculados no servidor, onde estão as regras. Consequência inevitável e escrita na UI: a régua de
novidade é *relativa* e muda de sentido conforme a janela. O `dt` real devolvido é o contrato do zoom.

**`qa._read_smf(data, notas=True)`.** O leitor que já existia passa a devolver os eventos, e o roll
do MIDI usa a **grade do arquivo** (`tempo_us/division`) para converter tick em segundo — nunca a
grade da partitura. Assim a frase “o que o arquivo tem é o que está no rolo” é exata por construção e
a divergência entre arquivo e partitura deixa de ser escondida (é um aviso na barra).

**`score_model.score_visivel` + `hide_lanes` (apresentação).** Uma chave por resultado. A filtragem
acontece depois de `apply_rules`, no caminho de cada gravador — `layout.layout()`, `to_svg_pages`,
`midi_out.to_midi`, `musicxml.to_musicxml` (o roll via `to_midi`, então vale para ele também) — e
nunca em `build_score` nem nos dados. `qa` julga as leis de escrita sobre todas as notas e compara os
arquivos com `score_visivel` (A18). `POST /api/view {"simples":true}` troca a gravura em 34 ms com um
aviso de que a grade não foi recalculada.

**`tune.py`.** Varredura `sens × conf` com orçamento de células por RAM e tempo, critérios derivados
só das leis que já existem aqui (duplicata ≤ 60 ms, piso de lacuna 22 ms, resíduo > 25 ms, fora da
grade > ¼ de pulso, barras sem bumbo+caixa, densidade, fração de queda-de-braço, erros da auditoria),
cobertura explícita nos dois lados, desempate pela calibração do projeto com o empate **declarado**.
`POST /api/tune` roda no mesmo filho isolado da análise e o melhor par entra no cache do resultado.
A escolha registrada: `0.95/0.30` vence a varredura sem-gabarito e mede f1 0,579 contra 0,586 de
`0.70/0.20` ao custo de +13 % de notas e +51 % de falsos positivos — está em A17, não escondido.
