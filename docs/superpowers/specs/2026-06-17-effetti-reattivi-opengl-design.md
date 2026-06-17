# Blocco 2 — Effetti reattivi (renderer OpenGL)

**Data:** 2026-06-17
**Stato:** design approvato, pronto per il piano di implementazione
**Scope:** solo renderer fullscreen OpenGL (`thread/opengl_thread.py`) + GUI (`gui/`)

## Contesto e roadmap

Secondo blocco della roadmap nata dal brainstorming sulle feature della finestra fullscreen OpenGL
(5 direzioni → 3 blocchi, uno spec ciascuno):

- **Blocco 1 — Aspetto barre** ✅ fatto e rilasciato (release `2.1.0`): colori + forma/disposizione.
- **Blocco 2 — Effetti reattivi (questo documento):** peak-cap che ricade, glow (bloom), pulsazione
  dello sfondo sul beat.
- **Blocco 3 — Nuove modalità (futuro):** spettro radiale e forma d'onda/linea.

Questo spec copre **solo il Blocco 2**.

## Obiettivo

Aggiungere tre effetti reattivi al renderer OpenGL: **peak-cap**, **glow/bloom**, **pulsazione sul
beat**. Ognuno attivabile e parametrizzabile dalla GUI.

## Principi guida

1. **Solo OpenGL.** L'anteprima in-window resta invariata. Le callback GUI inoltrano solo
   all'`EqualizerOpenGLThread`.
2. **Tutti gli effetti default OFF.** Aprendo il fullscreen senza abilitarli, l'aspetto è identico a
   post-Blocco 1: nessuna regressione e **nessun costo extra** (i path degli effetti disattivi sono
   saltati, incluso l'intero pipeline bloom).
3. **Coerenza coi pattern esistenti:** stato a runtime modificato via control-queue, parametri
   speculari nel costruttore, controlli GUI in un tab dedicato.

## Non-goal

- Nessuna nuova modalità di visualizzazione (radiale/onda → Blocco 3).
- Nessun preset salvabile. Nessuna modifica alla pipeline DSP delle bande né all'anteprima.

## Decisioni prese durante il brainstorming

- **Effetti:** tutti e tre (peak-cap, glow, beat pulse).
- **Peak-cap:** colore **a scelta dell'utente** (riuso `ColorPickerFrame`, default bianco);
  comportamento hold + caduta con slider.
- **Glow:** **vero bloom** (FBO + blur gaussiano + composito additivo), non l'alone additivo
  economico.
- **Beat pulse:** stile **zoom punch** sullo sfondo; rilevamento beat dall'energia dei bassi.

## Design

### A. Peak-cap che ricade

- **Stato per-barra** `self._peaks` (np.float32, una voce per barra **visualizzata**): ogni frame,
  se l'altezza barra supera il picco → il picco scatta al nuovo valore e si resetta il timer di
  hold; altrimenti, trascorso l'hold, il picco **cade** in modo frame-rate-independent
  (`peak -= fall_rate * dt`, clamp a 0). Calcolato sulle altezze **dopo** `_display_heights`, così è
  coerente con ordine simmetrico/ancoraggio specchiato (in modalità specchiata il cap segue
  l'estremità superiore).
- **Render:** seconda draw call **instanced** dedicata — un VBO di altezze-picco (una float per
  barra) + uno shader "caps" che disegna un trattino sottile (pochi px) nello slot di ogni barra
  all'altezza del picco. Colore = uniform `uCapColor`.
- **Parametri:** `enabled` (bool), `color` (rgb), `hold_ms`, `fall_rate` (frazione/secondo). Slider
  in GUI.

### B. Glow — vero bloom (FBO + blur)

Pipeline separabile a **risoluzione dimezzata** (perf):

1. **Pass A — estrazione:** render delle barre in un **FBO bloom** (half-res, clear nero). Sorgente
   luminosa = le barre stesse.
2. **Pass B/C — blur:** **blur gaussiano separabile** (orizzontale, poi verticale) con **ping-pong**
   tra due texture/FBO; 1–2 iterazioni per un alone morbido. Shader blur dedicato (campiona la
   texture, somma N tap pesati).
3. **Composito:** sul framebuffer reale, dopo sfondo + barre, si disegna un quad fullscreen che
   campiona la texture sfocata in **blending additivo** (`GL_ONE, GL_ONE`) scalato da
   `uBloomIntensity`.

- FBO/texture creati una volta nota la dimensione del framebuffer (fullscreen = risoluzione fissa →
  nessun resize dinamico), con guardia di completezza (`glCheckFramebufferStatus`), e **liberati in
  `_cleanup_gl`**. Creazione **lazy** alla prima attivazione.
- Quando `bloom_enabled` è False, l'**intera pipeline è saltata** (nessun costo).
- **Parametri:** `enabled` (bool), `intensity` (slider).
- *Nota:* è la parte più pesante; numero di iterazioni/tap e risoluzione sono candidati al tuning
  dopo la verifica visiva.

### C. Beat pulse (zoom punch)

- **Beat detection:** ogni frame calcolo l'**energia dei bassi** come somma della potenza FFT sotto
  un cutoff fisso (~150 Hz, indipendente dal numero di bande — più robusto del sommare bande
  variabili), mantengo una **media mobile** dell'energia (EMA). Un beat scatta quando
  `energia > media × fattore_sensibilità` e se è trascorso un **periodo refrattario** (~0.15 s,
  anti-doppio-trigger). La decisione è isolata in `_detect_beat(energy) -> bool` (puro/testabile);
  lo stato della media mobile vive nell'istanza.
- **Effetto:** a ogni beat un **inviluppo di pulsazione** va a 1.0 e **decade** esponenzialmente
  (frame-rate-independent). Lo sfondo fa **zoom**: nuova uniform `uBgZoom` nel vertex shader dello
  sfondo che scala il quad attorno al centro di `1 + pulse * intensity`. Le barre non sono toccate.
- **Parametri:** `enabled` (bool), `intensity` (zoom max), `sensitivity` (soglia rilevamento).

### D. GUI — nuovo tab "✨ Effetti"

`_build_effects_tab()` aggiunto a `_build_settings_tabs`. Tre gruppi:

- **Peak-cap:** `OptionMenuCustomFrame` On/Off · `ColorPickerFrame` Colore · `SliderCustomFrame`
  Hold (ms) · `SliderCustomFrame` Caduta.
- **Glow:** On/Off · Intensità.
- **Beat:** On/Off · Intensità · Sensibilità.

Riuso dei widget esistenti (`SliderCustomFrame`, `OptionMenuCustomFrame`, `ColorPickerFrame`).
**Nessun widget nuovo.** Callback di inoltro sul modello di `update_db_floor` (solo OpenGL).

### E. Protocollo control-queue + costruttore

Nuovi messaggi (drenati come gli esistenti) e parametri costruttore speculari:

| Messaggio | Value | Effetto |
|---|---|---|
| `set_peakcap_enabled` | bool | attiva/disattiva il peak-cap |
| `set_peakcap_color` | `(r,g,b)` ∈ [0,1] | colore del trattino |
| `set_peakcap_hold` | float (s) | tempo di hold prima della caduta |
| `set_peakcap_fall` | float (frazione/s) | velocità di caduta |
| `set_bloom_enabled` | bool | attiva/disattiva il bloom |
| `set_bloom_intensity` | float | forza del composito additivo |
| `set_beat_enabled` | bool | attiva/disattiva la pulsazione |
| `set_beat_intensity` | float | zoom massimo |
| `set_beat_sensitivity` | float | soglia (fattore sulla media mobile) |

### F. Default (non-regressione)

| Parametro | Default (di partenza, da tarare a vista) |
|---|---|
| peakcap_enabled / bloom_enabled / beat_enabled | **False** |
| peakcap_color | bianco `(1, 1, 1)` |
| peakcap_hold | `0.5` s |
| peakcap_fall | `0.8` (frazione di altezza al secondo) |
| bloom_intensity | `1.0` |
| beat_intensity | `0.08` (≈ 8% di zoom) |
| beat_sensitivity | `1.5` (energia > 1.5× media → beat) |

Con tutti gli `enabled = False` il rendering è identico a post-Blocco 1 (i valori sopra sono usati
solo quando l'effetto è abilitato).

## Errori / edge case

- Peak state e beat detection: pura logica numpy/stato (nessuna chiamata GL) → testabili headless.
  `_detect_beat` isolato per asserzioni.
- Bloom: creazione FBO con `glCheckFramebufferStatus`; risorse liberate in `_cleanup_gl`; pipeline
  saltata se disabilitata. Il render delle barre nel FBO riusa geometria/programma esistenti.
- `_peaks` ridimensionato quando cambia il numero di barre visualizzate (come `current_amplitudes`).
- Cambi runtime degli effetti non fanno chiamate GL fuori dal thread (salvano stato; il GL avviene
  nel `_render`/loop sul thread OpenGL).

## Verifica

Nessuna test suite (CLAUDE.md): si verifica eseguendo l'app.

- **Headless (WSL2, `.venv-linux`):** asserzioni su `_detect_beat`, sull'evoluzione di `_peaks`
  (snap/hold/fall) e sui handler dei messaggi (mutano lo stato atteso). Costruzione GUI offscreen
  (con argomento fittizio per `soundcard`, vedi nota ambiente). *Vedi memoria
  `soundcard-headless-argv`.*
- **Windows (display reale):** abilitare ogni effetto e verificarne il comportamento; bloom
  (qualità/perf), peak-cap (hold/caduta, colore), beat (zoom a tempo, sensibilità); combinazioni con
  le opzioni del Blocco 1; default OFF = nessuna regressione; sfondo immagine/video ancora ok.

## File toccati (previsti)

- `thread/opengl_thread.py` — stato peak/beat/bloom + parametri costruttore; shader caps, blur,
  composito bloom; uniform `uBgZoom` nel bg vertex shader; seconda draw instanced per i cap;
  pipeline FBO bloom; `_detect_beat`; aggiornamento `process_control_message`, `_render`, `run()`
  (creazione FBO), `_cleanup_gl`.
- `gui/main_window_gui.py` — `_build_effects_tab`, callback di inoltro, stato GUI, popolamento del
  costruttore in `fullscreen_command`.
- `CLAUDE.md` — documentare i nuovi effetti/uniform/messaggi (file locale, **gitignored** in questo
  repo: aggiornato ma non committato — vedi memoria `claude-md-untracked`).
