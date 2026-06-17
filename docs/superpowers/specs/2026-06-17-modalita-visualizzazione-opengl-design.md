# Blocco 3 — Nuove modalità di visualizzazione (renderer OpenGL)

**Data:** 2026-06-17
**Stato:** design approvato, pronto per il piano di implementazione
**Scope:** solo renderer fullscreen OpenGL (`thread/opengl_thread.py`) + GUI (`gui/`)

## Contesto e roadmap

Terzo e ultimo blocco della roadmap nata dal brainstorming sulle feature della finestra fullscreen
OpenGL (5 direzioni → 3 blocchi, uno spec ciascuno):

- **Blocco 1 — Aspetto barre** ✅ fatto e rilasciato (release `2.1.0`): colori + forma/disposizione.
- **Blocco 2 — Effetti reattivi** ✅ fatto e rilasciato (release `2.2.0`): peak-cap, bloom, beat pulse.
- **Blocco 3 — Nuove modalità (questo documento):** un **selettore di modalità** che affianca alle barre
  attuali altre tre modalità di visualizzazione (spettro radiale, oscilloscopio, spettro linea/area).

Questo spec copre **solo il Blocco 3**. È il blocco più impegnativo: introduce il concetto di
"modalità" e nuovi render path nel renderer OpenGL.

## Obiettivo

Trasformare il renderer da "solo barre" a multi-modalità, con un selettore che sceglie una fra
**4 modalità** mutuamente esclusive. Tre sono nuove; le barre attuali diventano una delle quattro.

| # | Modalità | Cos'è | Dati |
|---|---|---|---|
| 0 | **Barre** | l'attuale (default) | ampiezze per banda |
| 1 | **Radiale** | barre disposte in cerchio che si irradiano dal centro | ampiezze per banda |
| 2 | **Oscilloscopio** | forma d'onda nel tempo (dominio del tempo) | waveform mono grezza |
| 3 | **Linea/Area** | spettro come linea continua / area riempita | ampiezze per banda |

## Principi guida

1. **Solo OpenGL.** L'anteprima in-window (`EqualizerPreviewWidget`) resta invariata. Le callback GUI
   inoltrano esclusivamente all'`EqualizerOpenGLThread` (pattern `if self.equalizer_opengl_thread: …put(…)`).
2. **Default = look attuale.** Con `mode = Barre` e i nuovi parametri ai default, il fullscreen rende
   *esattamente* come post-Blocco 2. È il criterio principale di non-regressione.
3. **Massimo riuso.** Si estende l'impianto esistente (programma barre instanced, control-queue,
   modalità colore del Blocco 1, pipeline bloom) invece di duplicarlo. Una sola classe renderer, un
   solo nuovo programma shader.
4. **L'FFT gira sempre.** Anche in Oscilloscopio l'FFT è calcolata ogni frame (≈0.1 ms): serve alla
   beat-detection e rende istantaneo il cambio di modalità.

## Non-goal

- Nessun preset salvabile/caricabile.
- Nessuna modifica alla pipeline DSP delle bande né all'anteprima in-window.
- Niente modalità oltre alle quattro elencate (no stereo oscilloscope, no particellari, ecc.).
- Nessun allineamento con la DSP/anteprima Qt (diverge per scelta, vedi CLAUDE.md).

## Decisioni prese durante il brainstorming

- **Set modalità:** quattro (Barre, Radiale, Oscilloscopio, Linea/Area). Default all'avvio = **Barre**.
- **Radiale:** **cerchio pieno** come default, con **simmetria** come interruttore (riusa il concetto
  "ordine bande simmetrico" delle barre) e **raggio del foro centrale** su slider. Il semicerchio è
  stato scartato (YAGNI).
- **Oscilloscopio:** **mono** (media dei canali) come default, con **specchiato** (nastro simmetrico
  alto/basso) come interruttore. Niente traccia stereo.
- **Linea/Area:** **un'unica modalità** con interruttore **riempimento** (default acceso, effetto
  "skyline") e **spessore** su slider — non due voci separate.
- **Effetti (Blocco 2) × modalità:** **bloom in tutte e quattro** le modalità; **peak-cap solo Barre**
  (nascosto/ignorato altrove); **beat pulse sempre** (zoom sfondo, agnostico alla modalità).
- **Colore (Blocco 1) × modalità:** le 4 modalità colore valgono ovunque. Radiale le eredita tutte; le
  modalità a traccia (Oscilloscopio, Linea/Area) le replicano nel proprio shader.
- **GUI:** selettore "Modalità" in cima al tab **"🎨 Visualizzazione"**; il tab **"🎛️ Barre" diventa
  "🎛️ Forma"** con controlli di forma **condizionali alla modalità attiva**.

## Design

### A. Modello "modalità" nel renderer

- Nuovo stato `self._mode ∈ {0,1,2,3}` (default 0 = Barre), pilotato dal control message `set_viz_mode`.
- `_render(...)` diventa: `glClear` → **sfondo** (invariato, con `uBgZoom` del beat) → **dispatch**
  `_draw_current_mode` → **peak-cap** (solo se `mode == Barre`) → **composito bloom** (se abilitato).
- **`_draw_current_mode(viewport_px)`** — nuovo metodo unico che smista alla draw della modalità
  corrente. Lo chiamano **sia** il pass su schermo **sia** il prepass bloom: così il bloom funziona in
  ogni modalità riusando la stessa draw (niente codice duplicato).
- **Waveform per l'oscilloscopio:** il main loop legge già `fft_window` dal ring buffer
  (`get_window_for_fft()`). Quando `mode == Oscilloscopio`, deriva un **waveform mono** = media dei
  canali, normalizzato in `[-1, 1]` (i campioni `soundcard` sono già float in quel range), e lo passa
  alla draw. L'FFT continua a girare comunque (beat + cambio modalità immediato).

### B. Modalità Barre (mode 0) — invariata

Render attuale (instanced quad + per-instance height). Nessuna modifica funzionale; solo incapsulata
dietro il dispatch e il flag `mode`.

### C. Modalità Radiale (mode 1) — estensione del programma barre

Il programma instanced delle barre viene **esteso** con un ramo polare, così Radiale riusa *tutta* la
logica colore/forma del Blocco 1.

- **Vertex shader:** nuova uniform `uRadial` (0/1). Con `uRadial == 1`:
  - angolo centrale del raggio: `θ = (gl_InstanceID + 0.5) / uNumBars · 2π`;
  - estensione angolare della barra: `(aUnitPos.x − 0.5) · slotAngolare · uBarWidthFrac`;
  - raggio: `r = uInnerRadius + aUnitPos.y · aHeight · radialScale`;
  - posizione: `centro + r · (sin θ', cos θ')`, con **correzione aspect** (`uAspect = fbW/fbH`) sulla
    componente x per avere cerchi tondi su schermi 16:9.
  - I varying `vAmp`, `vBarY`, `vBarX`, `vBarT` restano calcolati → il fragment shader colora come per
    le barre (Classico/Tinta/Gradiente/**Spettro** per angolo) senza modifiche.
- **Nuove uniform:** `uRadial`, `uInnerRadius`, `uAspect`.
- **Simmetria:** riuso di `arrange_symmetric` (raddoppia le bande → spettro specchiato sull'asse
  verticale, niente "cucitura" bassi/acuti). Stesso flag `self._band_order_symmetric` delle barre.
- **Larghezza** (`uBarWidthFrac`): riusata come larghezza angolare.
- **Cime arrotondate** (`uRounded`): la math attuale è cartesiana in px → in Radiale è **ignorata**
  (trattata come 0). L'ancoraggio `uMirror` non si applica (in Radiale l'ancoraggio è il foro centrale).

### D. Modalità a traccia (mode 2 Oscilloscopio, mode 3 Linea/Area) — nuovo programma "strip"

Le curve continue con spessore non si prestano all'instancing; `glLineWidth > 1` non è affidabile in
core profile 3.3. Quindi: **un solo nuovo programma "strip"** alimentato da un VBO di vertici costruito
su **CPU ogni frame** (N ≤ poche centinaia di vertici → costo in microsecondi, trascurabile).

- **Geometria (triangle strip), builder puri su CPU:**
  - **Linea/Area da `heights`:** punti `(x_i = i/(N−1), y_i = height_i)`.
    - *Riempito:* triangle strip tra la **base** (y=0) e la curva.
    - *Linea:* **nastro** di spessore `uLineThickness` lungo la curva.
    - Simmetria: riuso `arrange_symmetric` (spettro specchiato in orizzontale).
  - **Oscilloscopio da waveform:** campioni mappati a `(x_i, y = 0.5 + sample_i · gain)`, **downsample**
    a ~larghezza schermo per limitare i vertici.
    - *Mono:* nastro lungo l'onda.
    - *Specchiato:* area simmetrica tra `0.5 ± |sample_i| · gain`.
    - **Trigger su zero-crossing:** la finestra parte dal primo attraversamento dello zero in salita,
      per stabilizzare l'onda (evita lo "scorrimento" frame-su-frame). Funzione pura, testabile.
- **Fragment shader strip:** **replica i 4 rami colore** del Blocco 1 con i propri varying:
  `vY` (frazione di altezza, per Classico/Gradiente) e `vX` (posizione lungo l'asse, per Spettro). Per
  l'Oscilloscopio il "Classico" mappa il colore su `|sample|`.
- **Parametri:** `uLineThickness` (spessore), `uFill` (linea vs area — solo Linea/Area), `uOscMirror`
  (mono vs specchiato — solo Oscilloscopio). Quello che varia per-frame è la geometria CPU; le uniform
  gestiscono colore/alpha.

### E. Effetti (Blocco 2) × modalità

- **Bloom:** in **tutte** le modalità. `_render_bloom` disegna `_draw_current_mode` nell'FBO half-res,
  poi blur separabile + composito additivo (pipeline invariata). Il glow su linee/cerchi al neon è dove
  rende meglio.
- **Peak-cap:** **solo Barre**. Negli altri modi il path è saltato e il controllo è **nascosto** in GUI.
- **Beat pulse:** **sempre** (zoom dello sfondo via `uBgZoom`, indipendente dalla modalità).

### F. Colore/forma del Blocco 1 × modalità (riuso)

| Feature Blocco 1 | Barre | Radiale | Oscilloscopio | Linea/Area |
|---|:---:|:---:|:---:|:---:|
| Modalità colore (Classico/Tinta/Gradiente/Spettro) | ✓ | ✓ (varying invariati) | ✓ (replicate nello strip shader) | ✓ (replicate) |
| Larghezza (`uBarWidthFrac`) | ✓ | ✓ (angolare) | — | — |
| Simmetria (`arrange_symmetric`) | ✓ | ✓ | — | ✓ |
| Cime arrotondate / ancoraggio centro (`uRounded`/`uMirror`) | ✓ | — | — | — |

### G. Protocollo control-queue + parametri costruttore

Nuovi messaggi, drenati come gli esistenti in `process_control_message`; parametri speculari nel
costruttore di `EqualizerOpenGLThread` (come i blocchi precedenti), popolati da `fullscreen_command`.

| Messaggio | Value | Effetto |
|---|---|---|
| `set_viz_mode` | int 0–3 | imposta `self._mode` (la GUI mappa l'etichetta → int) |
| `set_inner_radius` | float [0,1] (frazione del raggio max) | `uInnerRadius` (Radiale) |
| `set_line_thickness` | float (frazione altezza finestra) | spessore del nastro (Oscilloscopio, Linea/Area) |
| `set_fill` | bool | linea vs area riempita (Linea/Area) |
| `set_osc_mirror` | bool | mono vs specchiato (Oscilloscopio) |

### H. GUI

- **🎨 Visualizzazione:** in cima, nuovo `OptionMenuCustomFrame` **"Modalità"**
  (Barre/Radiale/Oscilloscopio/Linea-Area) → callback `on_viz_mode_changed` (mappa etichetta→int,
  inoltra `set_viz_mode`, aggiorna la visibilità condizionale dei controlli di forma).
- **🎛️ Forma** (ex "🎛️ Barre"): gruppo *Colore* invariato (universale) + gruppo *Forma* con controlli
  **condizionali alla modalità** (stesso pattern mostra/nascondi di `on_color_mode_changed`):
  - **Barre:** larghezza, arrotondate, ancoraggio, ordine bande (esistenti).
  - **Radiale:** raggio foro (slider), simmetria (riuso "Ordine bande"), larghezza.
  - **Oscilloscopio:** spessore (slider), specchiato (toggle).
  - **Linea/Area:** spessore (slider), riempimento (toggle), simmetria.
- **✨ Effetti:** invariato, ma il gruppo **Peak-cap** è mostrato solo in modalità Barre.
- Riuso dei widget esistenti (`SliderCustomFrame`, `OptionMenuCustomFrame`). **Nessun widget nuovo.**

## Default (non-regressione)

| Parametro | Default | = comportamento attuale |
|---|---|---|
| mode | Barre (0) | sì |
| inner_radius | 0.18 (frazione del raggio max, da tarare a vista) | n/a |
| line_thickness | 0.012 (frazione altezza finestra, da tarare a vista) | n/a |
| fill | True (area) | n/a |
| osc_mirror | False (mono) | n/a |

Con `mode = Barre` il rendering è identico a post-Blocco 2 (gli altri parametri sono inerti).

## Errori / edge case

- I nuovi messaggi (eccetto cambi sfondo, già gestiti) **non** fanno chiamate GL: salvano stato, usato
  nel successivo `_render`. Nessun nuovo thread, nessun nuovo rischio di lifecycle/teardown.
- I builder di geometria (strip linea/area, strip oscilloscopio, trigger zero-crossing) sono **funzioni
  pure** numpy → testabili headless senza contesto GL.
- VBO/VAO del programma strip creati una volta in `init_gl_objects`, capacità cresciuta on-demand come
  per le altezze; **liberati in `_cleanup_gl`**.
- Cambî runtime di numero bande / modalità / sotto-opzioni restano coerenti: la geometria CPU lavora
  sull'array corrente di lunghezza variabile; il bloom riusa `_draw_current_mode`.
- Waveform mono: gestire finestra vuota/silenzio (nessun crash; nastro piatto al centro).
- Correzione aspect del Radiale: usa la dimensione reale del framebuffer (già nota in `run()`).

## Verifica

Nessuna test suite (CLAUDE.md): si verifica eseguendo l'app.

- **Headless (WSL2, `.venv-linux`):** import senza errori + smoke test EGL (shader compilano);
  asserzioni sulle **funzioni pure** (builder geometria strip, trigger zero-crossing, e gli handler dei
  control message che mutano lo stato atteso). Costruzione GUI offscreen (con argomento fittizio per
  `soundcard` — vedi memoria `soundcard-headless-argv`). *Vedi memoria `wsl2-venv-and-gl-verification`.*
- **Windows (display reale):** ogni modalità + sotto-opzioni (radiale simmetrico/foro, oscilloscopio
  mono/specchiato, linea vs area); bloom in tutte; combinazioni con colori/simmetria del Blocco 1;
  sfondo immagine/video ancora ok; `mode = Barre` = nessuna regressione. La verifica visiva la fa l'utente.

## File toccati (previsti)

- `thread/opengl_thread.py` — `self._mode` + dispatch in `_render`; `_draw_current_mode`; estensione
  shader barre (ramo polare + `uRadial`/`uInnerRadius`/`uAspect`); **nuovo programma/VAO/VBO "strip"** +
  builder geometria su CPU + trigger zero-crossing; derivazione waveform mono nel loop `run()`;
  `_render_bloom` mode-aware; nuovi control message + parametri costruttore; `_cleanup_gl`.
- `gui/main_window_gui.py` — selettore "Modalità", `on_viz_mode_changed`, rinomina tab "Barre"→"Forma",
  controlli di forma condizionali, peak-cap condizionale, callback di inoltro, popolamento del
  costruttore in `fullscreen_command`.
- `gui/help_window.py` — testo d'aiuto per le modalità.
- `CLAUDE.md` — documentare il modello modalità, le nuove uniform e i messaggi di controllo (file
  locale, **gitignored** in questo repo: aggiornato ma non committato — vedi memoria `claude-md-untracked`).
