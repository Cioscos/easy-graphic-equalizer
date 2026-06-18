# Performance & windowing — renderer OpenGL

**Data:** 2026-06-18
**Stato:** design approvato, pronto per il piano di implementazione
**Scope:** solo renderer fullscreen OpenGL (`thread/opengl_thread.py`); eventuale ritocco a
`thread/video_decode_thread.py` solo se la misura lo richiede. Nessun cambio a GUI, code di
controllo, algoritmo DSP o shader (output visivo invariato).

## Contesto

Dopo i blocchi feature (aspetto barre, effetti reattivi, modalità di visualizzazione) il focus si
sposta sulle **performance** della finestra OpenGL. Osservazioni dell'utente su monitor 165Hz:

- Senza video la GPU "dorme" (30-40% di utilizzo, clock ~300MHz): regge i 165fps con ampio margine.
- Con il **video di sfondo** la GPU sale poco (~400MHz) ma gli fps **scendono a 140-150**: non
  riesce più a tenere i 165.

**Diagnosi (dalla lettura del codice).** La bassa occupazione GPU con VSync attivo
(`glfw.swap_interval(1)`) è il comportamento *atteso e sano*: il carico (poche migliaia di quad in
instanced rendering) è banale, la GPU lo finisce in molto meno di 6ms e si addormenta fino al
vblank. **Il collo di bottiglia non è la GPU ma la CPU/Python sul thread GL**: a 165Hz ci sono ~6ms
a frame, spesi in drain delle code, FFT+numpy, smoothing e decine di chiamate PyOpenGL (overhead
per-chiamata). Col video si aggiungono (a) la contesa GIL del thread di decodifica e — soprattutto —
(b) l'**upload texture sincrono** `glTexSubImage2D` da puntatore client (`_upload_video_frame`), che
stalla il thread GL finché il driver non ha consumato i ~6MB del frame. Questo sfora il budget di
6ms → vblank persi → 140-150 invece di 165.

Inoltre la finestra è creata in **fullscreen esclusivo** (`glfw.create_window(..., selected_monitor,
None)`), non in borderless windowed.

## Obiettivo (criteri di successo misurabili)

1. **165fps stabili con il video di sfondo attivo**: frame time costante < 6.06ms, nessun vblank
   perso (verificato con `PERF_LOGS`).
2. **Finestra borderless windowed**: nessun cambio di modalità video, alt-tab istantaneo, copre il
   monitor selezionato.
3. **Margine di frame time sul thread GL** documentato con misura `PERF_LOGS` prima/dopo.
4. **Resta GL 3.3 core**, zero regressioni visive in ogni modalità/effetto, nessun taglio di
   compatibilità hardware.

## Principi guida

1. **Misura prima di scalare.** Gli interventi del *core* (sotto) sono noti, isolati e a basso
   rischio. Tecniche più invasive (UBO, GL 4.x, DSP off-thread) si fanno **solo se** `PERF_LOGS`
   mostra il thread GL ancora CPU-bound dopo il core.
2. **Interventi indipendenti e revertibili.** Borderless, upload async e limatura CPU sono
   ortogonali: ciascuno si verifica e si annulla da solo.
3. **Nessuna regressione percepita.** Output visivo, modalità, effetti e semantica dei messaggi di
   controllo restano identici. Si cambia *come* si rende, non *cosa*.

## Non-goal

- **Nessun uso della GPU "libera" per qualità** (no supersampling/SSAA, no MSAA più alto, no più
  barre, no effetti più pesanti): è una leva di qualità, non di performance, ed è fuori scope.
- **Niente DSP off-thread** di default: l'FFT è ~0.1ms su 6ms e il GIL limita l'overlap reale;
  spostarla non aiuta il budget. Solo se la misura lo smentisce.
- **Niente GL 4.x né riscrittura a UBO nel core**: confinati al tier di escalation.
- **Nessuna opzione fullscreen esclusivo** mantenuta accanto alla borderless: la borderless la
  sostituisce (su Windows 10/11 la latenza è ~equivalente grazie a *independent flip*).
- Nessun cambio a GUI, algoritmo DSP, shader, o semantica della control-queue.

## Decisioni prese durante il brainstorming

- **Profilo:** GL **3.3 + escalation su misura** (non rework profondo a prescindere). Si prende ~tutto
  il beneficio pratico restando su 3.3; il 4.x solo se i numeri lo impongono.
- **Borderless** sostituisce il fullscreen esclusivo (rispetta il selettore monitor esistente).
- **Fix dei 165fps col video** = upload asincrono via **PBO** (core da GL 2.1, nessun bump di
  versione).
- **UBO / skip-uniform-ridondanti** spostati dal core al **tier di escalation** (riscrivere le
  uniform di tutti gli shader è ampio e a ROI basso → non si fa a tappeto).

## Design

### A. Borderless windowed

In `run()`, sostituire la creazione della finestra fullscreen esclusiva con una finestra **windowed
senza decorazioni** dimensionata e posizionata sul monitor selezionato:

- Mantenere gli hint di contesto attuali (3.3 core, forward-compat) e `glfw.SAMPLES = MSAA_SAMPLES`.
- Aggiungere `glfw.window_hint(glfw.DECORATED, glfw.FALSE)` (niente bordo/titolo). Opzionale
  `glfw.window_hint(glfw.FLOATING, glfw.TRUE)` per tenerla sopra.
- Risolvere il monitor come ora; leggere `video_mode` (risoluzione + refresh) **e** la posizione del
  monitor con `glfw.get_monitor_pos(selected_monitor)`.
- Creare la finestra **windowed**: `glfw.create_window(w, h, "Audio Visualizer", None, None)`, poi
  `glfw.set_window_pos(window, mon_x, mon_y)` così copre esattamente il monitor.
- Restano invariati: `swap_interval(1)`, viewport dal framebuffer reale, ESC/key callback,
  `glViewport`.

**Comportamento atteso.** Windows 10/11: finestra borderless a tutto schermo → DWM concede
*independent flip* → latenza ~pari all'esclusivo, niente tearing, alt-tab istantaneo (niente cambio
di modalità video). **Caveat Linux:** il bypass del compositor dipende dal WM (X11/Wayland);
accettabile per un visualizer.

### B. Upload video asincrono — PBO double-buffered  ← fix dei 165fps

Sostituire l'upload sincrono in `_upload_video_frame` con **2 Pixel Buffer Object in ring**:

1. Allocare 2 PBO (`glGenBuffers(2)`) e un indice corrente; (ri)allocarli a
   `width*height*3` byte al primo frame o al cambio dimensione, insieme alla texture
   (`glTexImage2D(... NULL)`).
2. Ogni frame, con PBO[idx] legato a `GL_PIXEL_UNPACK_BUFFER`:
   - **orphaning**: `glBufferData(GL_PIXEL_UNPACK_BUFFER, size, None, GL_STREAM_DRAW)` (il driver
     restituisce storage fresco → nessun fence manuale, nessuna attesa sul buffer in uso dalla GPU);
   - copiare i byte del frame nel PBO (`glBufferSubData`; in alternativa `glMapBufferRange` con
     `GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT` + memcpy, se la
     misura lo giustifica);
   - `glTexSubImage2D(..., GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))` → legge **dal PBO** e
     ritorna subito (trasferimento DMA asincrono, **niente stallo** del thread GL).
3. Sbindare il PBO (`bind 0`) e avanzare l'indice (frame N+1 usa l'altro PBO → la copia CPU non
   aspetta la DMA del frame N).

Dettagli: `glPixelStorei(GL_UNPACK_ALIGNMENT, 1)` impostato **una volta** (vedi C), non per-frame; il
flip verticale resta sullo shader (`uFlipV`). PBO liberati in `_cleanup_gl`. Tutto **3.3-safe** (PBO
core da 2.1, map-range da 3.0). Il rischio di corruzione è basso grazie all'orphaning (niente
sincronizzazione manuale).

### C. Limatura CPU (basso rischio) + strumentazione di misura

- **Hoist** di `glPixelStorei(GL_UNPACK_ALIGNMENT, 1)` e di ogni stato GL invariante fuori dal loop
  (in init / a contesto creato).
- **Preallocazione** degli scratch per-frame **solo dove è pulito** (path osc/line `build_*`); nel
  core FFT le allocazioni sono già piccole e in parte con `out=` → guadagno modesto, niente
  riscritture rischiose.
- **`PERF_LOGS` aggregato**: invece dello spam per-frame, statistiche su finestra di N frame
  (avg/min/max frame time, % di frame sopra il budget di 6.06ms), così il confronto prima/dopo è
  leggibile. È il **gate** che decide se serve l'escalation.

### D. Tier di escalation — *solo se la misura lo impone*

Si procede a uno step solo se `PERF_LOGS` mostra il thread GL ancora CPU-bound dopo il tier
precedente.

- **Step 1 (resta 3.3):** UBO per batchare le uniform; skip degli upload-uniform ridondanti
  (impostarle solo al cambio valore).
- **Step 2 (richiede 4.x):** buffer **persistent-mapped** (`GL_ARB_buffer_storage`, 4.4) per
  heights/strip/PBO; eventuale **DSA** (4.5) per pulizia. Comporta bump della versione minima del
  contesto + fencing manuale → da fare consapevolmente.
- **DSP off-thread:** solo se il DSP risulta un costo reale (improbabile).

## Errori / edge case

- **Borderless cross-platform:** Windows → independent flip; Linux → dipende dal compositor.
  Verifica su entrambi se possibile; il fallback peggiore (composito attivo) resta giocabile.
- **PBO:** allocazione/riallocazione al primo frame o al cambio dimensione del video; `bind 0` dopo
  l'uso per non lasciare il `GL_PIXEL_UNPACK_BUFFER` legato (altre `glTexImage2D`/`glTexSubImage2D`
  — sfondo immagine — interpreterebbero il puntatore come offset nel PBO). Risorse liberate in
  `_cleanup_gl`; `_stop_video`/`clear_video` smettono di caricare ma non devono lasciare stato GL
  incoerente.
- **Nessuna test suite** (CLAUDE.md) → ogni cambiamento va verificato eseguendo l'app; gli interventi
  sono piccoli e indipendenti per limitare il raggio di una regressione.

## Verifica

Nessuna test suite: si verifica eseguendo l'app + `PERF_LOGS`.

- **Headless (WSL2, `.venv-linux`):** smoke test EGL che l'import OpenGL/creazione contesto regga; le
  parti GL vere richiedono un display. *Vedi memorie `wsl2-venv-and-gl-verification` e
  `soundcard-headless-argv`.*
- **Windows (display reale, 165Hz) — la verifica che conta:**
  - **Baseline:** `PERF_LOGS` con e senza video → registrare avg/min/max frame time e % frame sopra
    budget (riproduce il 140-150 col video).
  - **Dopo A:** borderless copre il monitor, alt-tab istantaneo, 165fps invariati **senza** video.
  - **Dopo B:** col video il frame time torna < 6ms e gli fps risalgono a 165 stabili.
  - **Non-regressione:** ogni modalità (Barre/Radiale/Oscilloscopio/Linea), effetti on/off, sfondo
    immagine **e** video, cambio bande a runtime.
- **Decisione escalation:** se dopo A+B+C la misura mostra ancora il thread GL CPU-bound, aprire lo
  Step 1; altrimenti chiudere qui.

## File toccati (previsti)

- `thread/opengl_thread.py` — `run()` (creazione finestra borderless: hint `DECORATED`, windowed +
  `set_window_pos` sul monitor); `_upload_video_frame` (PBO double-buffered); hoist `glPixelStorei` e
  init dei PBO; `_cleanup_gl` (free PBO); eventuale `PERF_LOGS` aggregato nel loop.
- `thread/video_decode_thread.py` — **non previsto**; candidato solo se la misura mostra contesa/
  back-pressure da rivedere.
- `CLAUDE.md` — documentare borderless + path PBO video (file locale, **gitignored** in questo repo:
  aggiornato ma non committato — vedi memoria `claude-md-untracked`).
