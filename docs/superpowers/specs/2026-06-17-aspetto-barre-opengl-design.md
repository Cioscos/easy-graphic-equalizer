# Blocco 1 — Personalizzazione aspetto barre (renderer OpenGL)

**Data:** 2026-06-17
**Stato:** design approvato, pronto per il piano di implementazione
**Scope:** solo renderer fullscreen OpenGL (`thread/opengl_thread.py`) + GUI (`gui/`)

## Contesto e roadmap

Brainstorming sulle prossime feature della finestra fullscreen OpenGL. Sono emerse 5 direzioni
desiderate (colori, forma/layout barre, effetti reattivi, spettro radiale, forma d'onda/linea),
troppe per un singolo spec. Sono state raggruppate in 3 blocchi coerenti rispetto a come si
innestano sull'architettura attuale (barre instanced + shader + control-queue), da affrontare in
sequenza, **uno spec ciascuno**:

- **Blocco 1 — Aspetto barre (questo documento):** colori + forma/disposizione. Tutto nello shader
  delle barre + nuovi controlli GUI. Rischio basso; costruisce l'impianto (uniform, messaggi di
  controllo, controlli GUI) riusato dai blocchi successivi.
- **Blocco 2 — Effetti reattivi (futuro):** peak-cap che ricade, glow/bloom, pulsazione dello
  sfondo sul beat.
- **Blocco 3 — Nuove modalità (futuro):** selettore di modalità + spettro radiale e forma
  d'onda/linea (nuovi render path).

Questo spec copre **solo il Blocco 1**.

## Obiettivo

Rendere personalizzabili **colore** e **forma/disposizione** delle barre nel renderer OpenGL,
oggi fissati da costanti di modulo e dallo shader. Otto funzionalità in totale.

## Principi guida

1. **Solo OpenGL.** L'anteprima in-window (`EqualizerPreviewWidget`) resta invariata, coerentemente
   con `bars_alpha` e i parametri DSP del tab Avanzate. Le callback GUI inoltrano esclusivamente
   all'`EqualizerOpenGLThread` (pattern `if self.equalizer_opengl_thread: ...put(...)`).
2. **Default = look attuale.** Con i valori di default il fullscreen rende *esattamente* come oggi.
   È il principale criterio di non-regressione.
3. **Renderer "stupido" sui colori.** La GUI converte i colori scelti in tuple RGB normalizzate
   `(r, g, b)` ∈ [0,1] e le passa sulla control-queue; il renderer le usa come uniform senza logica
   di parsing.

## Non-goal

- Nessun preset salvabile/caricabile (eventuale blocco futuro).
- Nessun effetto reattivo (Blocco 2), nessuna modalità radiale/onda (Blocco 3).
- Nessuna modifica alla pipeline DSP né all'anteprima in-window.

## Decisioni prese durante il brainstorming

- **Modalità colore:** tutte e quattro (Classico con soglie regolabili, Tinta unica, Gradiente
  verticale, Spettro per frequenza).
- **Scelta colori:** `QColorDialog` (widget dedicato `ColorPickerFrame`) + 4–5 preset rapidi.
- **Forma/disposizione:** tutte e quattro (larghezza/spaziatura, cime arrotondate, ancoraggio
  specchiato dal centro, ordine bande simmetrico).
- **Gradiente:** sfumatura **lungo la singola barra** (base→cima), non ancorata alla finestra.
- **Ordine simmetrico:** bassi al centro, specchiati verso i due bordi → **raddoppia** le barre a
  schermo (ogni banda disegnata su entrambi i lati).
- **GUI:** nuovo tab dedicato **"🎛️ Barre"** (Visualizzazione resta pulito), gruppi *Colore* e
  *Forma*, con controlli colore **condizionali** alla modalità scelta.

## Design

### A. Sistema colore (fragment shader delle barre)

Aggiungere al fragment shader `uniform int uColorMode` con quattro rami:

| Modalità | Valore | Come colora | Dati extra |
|---|---|---|---|
| Classico | 0 | verde/giallo/rosso per **livello assoluto** (comportamento attuale) | `uGreenSplit`, `uYellowSplit` (già esistono, ora pilotate da GUI) |
| Tinta unica | 1 | colore uniforme | `uColorA` (vec3) |
| Gradiente | 2 | sfumatura `uColorA`→`uColorB` lungo la barra | `uColorA`, `uColorB` (vec3) |
| Spettro | 3 | tinta da indice barra, via `hsv2rgb` in GLSL | nessuno |

Nuovi *varying* esportati dal vertex shader (i valori sono già calcolati lì, vanno solo passati):

- `vAmp = aUnitPos.y * aHeight` — frazione di altezza-finestra (ancoraggio dal basso). Usata da
  **Classico** per lo split, così il colore resta legato alla loudness reale **anche** in modalità
  specchiata (il colore è disaccoppiato dalla geometria).
- `vBarY = aUnitPos.y` — frazione lungo la barra [0 base, 1 cima]. Usata dal **Gradiente** (per-barra).
- `vBarT = float(gl_InstanceID) / max(uNumBars - 1, 1)` — indice barra normalizzato. Usata dallo **Spettro**.
- `vBarX = aUnitPos.x` — frazione sulla larghezza barra [0,1]. Usata dalle **cime arrotondate** (sezione B).

Lo spettro mappa l'hue su un arco che evita l'avvolgimento rosso→rosso (es. hue ∈ [0, 0.83] dai
bassi agli acuti), saturazione/valore pieni.

**Scelta colori (GUI):** nuovo widget `ColorPickerFrame` modellato su `BackgroundFilepickerFrame`
(etichetta di intestazione + bottone-swatch che mostra il colore corrente e apre `QColorDialog`,
più una riga di 4–5 preset rapidi). Espone `get_color()` e una `command(rgb)`; la callback GUI invia
la tupla RGB normalizzata sulla control-queue.

### B. Sistema forma/disposizione

| Feature | Dove agisce | Meccanismo |
|---|---|---|
| Larghezza/spaziatura | `uBarWidthFrac` (uniform già esistente) | oggi impostata dalla costante `BAR_WIDTH_FRAC`; diventa attributo `self._bar_width_frac` pilotabile a runtime. |
| Cime arrotondate | fragment shader | `uniform bool uRounded`: cappuccio semicircolare in cima alla barra, **corretto per aspect ratio** (raggio = mezza larghezza barra in pixel), bordo anti-aliased (smoothstep). Usa `vBarX`/`vBarY` + uniform con la larghezza barra in pixel. |
| Ancoraggio specchiato | vertex shader | `uniform bool uMirror`: la coordinata y passa da `[0, h]` (dal basso) a `[0.5 − h/2, 0.5 + h/2]` (centrata). Il colore resta corretto grazie a `vAmp`. |
| Ordine simmetrico | CPU (shader invariato) | funzione pura `arrange_symmetric(heights) -> heights`: bassi al centro, specchiati ai bordi; **raddoppia** le barre visibili. La stessa draw call instanced le disegna; la capacità del VBO cresce automaticamente (`_ensure_heights_capacity`). Applicata alle ampiezze smoothed prima dell'upload, dietro un flag `self._band_order`. |

Ancoraggio verticale (`uMirror`) e ordine orizzontale (`band_order`) sono **ortogonali** e
combinabili.

### C. Protocollo control-queue + parametri costruttore

Nuovi messaggi, drenati in cima al loop come gli esistenti (`process_control_message`):

| Messaggio | Value | Effetto |
|---|---|---|
| `set_color_mode` | int 0–3 | imposta `uColorMode` (la GUI mappa l'etichetta → int) |
| `set_color_a` | `(r,g,b)` ∈ [0,1] | tinta / colore base gradiente |
| `set_color_b` | `(r,g,b)` ∈ [0,1] | colore cima gradiente |
| `set_green_split` | float [0,1] | soglia giallo (Classico) |
| `set_yellow_split` | float [0,1] | soglia rosso (Classico) |
| `set_bar_width` | float [0,1] | `uBarWidthFrac` |
| `set_rounded` | bool | `uRounded` |
| `set_bar_anchor` | bool (`uMirror`: False = dal basso, True = specchiato) | geometria vertex shader |
| `set_band_order` | bool (False = standard, True = simmetrico) | flag riordino CPU |

Parametri speculari aggiunti al costruttore di `EqualizerOpenGLThread` (come `db_floor` & co.), con
default pari al look attuale; `fullscreen_command` li popola dai valori correnti dei widget all'apertura.

**Clamp difensivo:** garantire `green_split ≤ yellow_split` (riordino/clamp lato renderer) per non
invertire le bande colore.

### D. GUI — nuovo tab "🎛️ Barre"

`_build_bars_tab()` aggiunto in `_build_settings_tabs`.

- **Gruppo Colore:** `OptionMenuCustomFrame` "Modalità colore" + controlli **condizionali**
  (mostra/nascondi nella callback di cambio modalità):
  - Classico → 2 `SliderCustomFrame` ("Soglia giallo" → `green_split`, "Soglia rosso" → `yellow_split`).
  - Tinta unica → 1 `ColorPickerFrame`.
  - Gradiente → 2 `ColorPickerFrame` (base, cima).
  - Spettro → nessun controllo colore.
- **Gruppo Forma:** `SliderCustomFrame` "Larghezza barre"; `OptionMenuCustomFrame` "Cime
  arrotondate" (Sì/No), "Ancoraggio" (Dal basso/Centro), "Ordine bande" (Standard/Simmetrico).

Riuso dei widget esistenti (`SliderCustomFrame`, `OptionMenuCustomFrame`). **L'unico widget nuovo è
`ColorPickerFrame`.** Nuove callback GUI sul modello di `update_db_floor` (inoltro alla sola
control-queue OpenGL).

## Default (non-regressione)

| Parametro | Default | = comportamento attuale |
|---|---|---|
| color_mode | Classico (0) | sì |
| green_split / yellow_split | 0.5 / 0.8 | sì (`GREEN_SPLIT`/`YELLOW_SPLIT`) |
| color_a (tinta) | verde `#22d36a` | n/a (non usato in Classico) |
| color_a/b (gradiente) | blu→magenta | n/a |
| bar_width | 0.8 | sì (`BAR_WIDTH_FRAC`) |
| rounded | off | sì (squadrate) |
| bar_anchor | dal basso | sì |
| band_order | standard | sì |

## Errori / edge case

- I messaggi colore/forma **non** effettuano chiamate GL: salvano stato, usato come uniform nel
  successivo `_render`. Nessun nuovo thread, nessun nuovo rischio di lifecycle/teardown.
- Conversioni colore robuste: hex/`QColor` non validi vengono ignorati (nessun crash).
- `green_split ≤ yellow_split` forzato lato renderer.
- Ordine simmetrico: il raddoppio delle barre è gestito dalla crescita automatica del VBO altezze.
- Cambi di numero bande (`set_frequency_bands`) e di ordine/ancoraggio devono restare coerenti: il
  riordino lavora sull'array di ampiezze corrente, di lunghezza variabile.

## Verifica

Non esiste test suite (vedi CLAUDE.md): si verifica eseguendo l'app.

- **Lato Windows (monitor reale):** `uv run python main.py`, selezionare un device, aprire il
  fullscreen e provare ogni modalità colore e ogni opzione di forma, incluse le combinazioni
  (es. specchiato + spettro, simmetrico + gradiente). Verificare che i default rendano come prima.
- **Lato WSL2:** almeno avvio senza errori d'import e smoke test EGL (nessuna regressione di import
  degli shader); la verifica visiva piena richiede un display GL reale.

## File toccati (previsti)

- `thread/opengl_thread.py` — shader (vertex+fragment), uniform/varying, attributi di stato,
  `process_control_message`, costruttore, `arrange_symmetric`, applicazione del riordino e degli
  uniform in `_render`.
- `gui/color_picker_frame.py` — **nuovo** widget `ColorPickerFrame`.
- `gui/main_window_gui.py` — `_build_bars_tab`, nuove callback di inoltro, popolamento del
  costruttore in `fullscreen_command`.
- (eventuale) `gui/help_window.py` — testo di aiuto per i nuovi controlli.
- `CLAUDE.md` — aggiornare la sezione rendering OpenGL con le nuove uniform e i messaggi di controllo.
