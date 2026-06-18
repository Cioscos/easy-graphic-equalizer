# Rimozione anteprima in-window — solo fullscreen OpenGL

Data: 2026-06-18

## Contesto e obiettivo

L'app ha due backend di visualizzazione mutuamente esclusivi: l'anteprima Qt in-window
(`EqualizerPreviewWidget` + `EqualizerPreviewWorker`) e il renderer fullscreen GLFW/OpenGL
(`EqualizerOpenGLThread`). L'anteprima ha una pipeline DSP propria, di qualità/latenza inferiori,
e duplica concetti (sfondo, alpha, bande) col renderer OpenGL.

Obiettivo: **eliminare completamente l'anteprima in-window**. La finestra diventa un *pannello di
controllo* che configura e lancia il renderer OpenGL a schermo intero, unico modo di visualizzare.
Esito: meno codice, un solo percorso DSP/renderer, GUI più chiara e focalizzata.

## Architettura

### Prima
- `AudioCaptureThread` (producer) → due consumer mutuamente esclusivi:
  - `EqualizerPreviewWorker` (QThread) → emette `bands_ready` → `EqualizerPreviewWidget` (in-window).
  - `EqualizerOpenGLThread` (fullscreen) [+ `VideoDecodeThread` opzionale].
- Pulsante **Start/Stop** avvia/ferma l'anteprima; **Fullscreen** lancia OpenGL.

### Dopo
- `AudioCaptureThread` (producer) → **unico** consumer `EqualizerOpenGLThread` [+ `VideoDecodeThread` opzionale].
- Nessuna anteprima. Azione primaria unica: **Avvia a schermo intero** (toggle avvia/ferma).

## Layout GUI (pattern "Preferenze classico")

- **Menu bar** (invariata): *Tema* (Light/Dark/System), *Aiuto* (Apri guida).
- **Riga in alto** a tutta larghezza: etichetta `Seleziona dispositivo:` + lista dispositivi compatta.
- **Corpo**: `QSplitter` orizzontale — **nav** (`QListWidget#navList`) a sinistra | **pagina impostazioni**
  (`QStackedWidget`, ogni pagina in `QScrollArea`) a destra. Le 5 pagine e i `QGroupBox` restano identici
  (Visualizzazione, Forma, Colore, Audio, Effetti), con la stessa visibilità condizionale.
- **Footer** a tutta larghezza: pulsante primario **"Avvia a schermo intero"** (icona `maximize.png`).
  - Disabilitato finché non è selezionato un dispositivo.
  - Mentre il fullscreen è attivo: testo "Ferma", colore rosso (`COLOR_STOP`).
  - Alla chiusura del fullscreen (anche via Esc → `_on_opengl_closed`): torna "Avvia", colore accento/verde.

## Modifiche al codice

### File eliminati
- `gui/equalizer_preview_widget.py`
- `thread/equalizer_preview_worker.py`

### `gui/main_window_gui.py`
**Rimozioni**
- Import: `EqualizerPreviewWorker`, `EqualizerPreviewWidget`.
- Stato: `preview_thread`, `preview_worker`, `is_capturing` (e `start_stop_button`).
- Metodi: `start_capture`, `stop_capture`, `start_stop_capture` (e relativi helper di stato del pulsante).
- Chiamate all'anteprima: `set_background_image`, `set_background_alpha`, `show_video_placeholder`, `clear_bars`.

**Adattamenti**
- Costruzione finestra: un'unica colonna logica = dispositivo in alto + `QSplitter(nav | pagina)` + footer Avvia.
  Riuso di `_build_menu_bar`, `_page`, `_group` e dei 5 builder di pagina (immutati). Il pulsante Avvia è
  l'ex `fullscreen_btn`, rinominato/spostato nel footer; nessun `_build_right_panel`/anteprima.
- `on_device_selected`: abilita il pulsante Avvia (non più Start/Stop); rimuove il ramo `is_capturing`/`stop_capture`.
- `fullscreen_command`: resta toggle. Aggiorna testo/colore del pulsante a "Ferma"/rosso all'avvio; rimuove il
  blocco iniziale che fermava l'anteprima.
- `_on_opengl_closed`: oltre ad azzerare il thread, ripristina il pulsante a "Avvia"/accento.
- `update_noise_threshold`, `update_frequency_bands`: condizione `if self.equalizer_opengl_thread` (solo OpenGL).
- `update_alpha`: niente aggiornamento anteprima; memorizza `bg_alpha` e inoltra a OpenGL (`set_alpha`).
  Unificabile con `update_alpha_opengl`.
- `_apply_image_background` / `_apply_video_background`: rimuovono le righe sull'anteprima; mantengono
  `self.bg_img` / `self.bg_video_path` (usati da `fullscreen_command` al lancio e via control queue se attivo).
- `closeEvent`: rimuove `stop_capture`; continua a fermare `audio_thread` e `equalizer_opengl_thread`.

### `gui/help_window.py`
- Aggiorna il testo che cita l'anteprima in-window (es. la frase "have no effect on the in-window preview"),
  riformulandolo per il solo renderer fullscreen.

## Punti fermi / invarianti
- `self.bg_img` (immagine di default) resta caricata: serve a `fullscreen_command`.
- Tutti i controlli (sfondo, trasparenza, opacità barre, soglia rumore, bande, avanzate DSP, forma, colore,
  effetti, monitor) restano: `fullscreen_command` legge i valori correnti al lancio, e la control queue
  li applica a caldo mentre il fullscreen è attivo. Nessun comportamento di configurazione perso.
- `AudioCaptureThread` resta avviato alla selezione del dispositivo (serve a OpenGL).
- La control queue e i tipi di messaggio restano invariati (erano già il protocollo condiviso; i tipi
  "solo anteprima" non esistevano — preview e OpenGL gestivano gli stessi `set_noise_threshold`/`set_frequency_bands`).

## Casi limite
- Avvio fullscreen senza dispositivo: impossibile, il pulsante è disabilitato (e `fullscreen_command`
  mantiene comunque il controllo `if not self.audio_thread`).
- Chiusura del fullscreen via Esc: `opengl_window_closed` (thread GLFW) → segnale `_opengl_closed`
  (queued) → `_on_opengl_closed` sul thread GUI ripristina il pulsante. Nessun widget toccato fuori dal thread GUI.
- Cambio dispositivo mentre il fullscreen è attivo: `on_device_selected` ricrea l'`audio_queue`/thread;
  comportamento invariato rispetto a oggi a parte la rimozione del ramo anteprima.

## Verifica
Avvio (WSL2: venv Linux + display GL):
```
UV_PROJECT_ENVIRONMENT=.venv-linux uv run python main.py
```
Checklist:
1. La finestra si apre senza pannello anteprima: dispositivo in alto, nav | pagina, footer Avvia.
2. Avvia è disabilitato; selezionando un dispositivo si abilita (accento/verde).
3. Avvia apre il fullscreen OpenGL; il pulsante diventa "Ferma"/rosso.
4. Chiusura via Esc o via "Ferma": il pulsante torna "Avvia"/accento, nessun thread appeso.
5. Le 5 pagine e la visibilità condizionale (modalità/colore) funzionano come prima.
6. Cambio tema dal menu mantiene accento+QSS.
7. Sfondo immagine/video, alpha, bande, ecc. configurano il fullscreen (verifica a video).
8. Chiusura finestra: nessun thread residuo.
9. Nessun riferimento residuo all'anteprima nel codice (grep).
```
