# Menù impostazioni in-window (overlay ImGui sulla finestra OpenGL)

- **Data:** 2026-06-19
- **Stato:** Design approvato — pronto per il piano di implementazione
- **Branch:** `in-window-imgui-settings-menu`

## Contesto e motivazione

La visualizzazione gira in una finestra **GLFW/OpenGL borderless** dimensionata sul monitor scelto. Il pannello di controllo **PySide6** configura e lancia quella finestra, ma una volta a schermo intero il pannello finisce **dietro** la visualizzazione: un utente con **un solo monitor** non può più cambiare le impostazioni senza chiudere il fullscreen.

Obiettivo: portare gli **stessi comandi** della GUI **dentro** la finestra OpenGL, così da poter regolare tutto a schermo intero anche con un solo monitor.

## Goals / Non-goals

**Goals**
- Menù in-window che rispecchia tutti i controlli **live** del pannello PySide6 (tutte le 5 sezioni).
- Le modifiche fatte in-window **tornano** al pannello PySide6 (sync bidirezionale) → sopravvivono al riavvio del fullscreen e finiscono nei profili (#34).
- Apertura/chiusura con **tasto toggle configurabile** (default `F1`).

**Non-goals (fuori scope per questa iterazione)**
- **Selettore monitor**: non cambiabile a runtime senza ricreare la finestra → resta solo nel pannello PySide6.
- **File-picker sfondo** (`set_image`/`set_video`/`clear_video`): richiede un dialog file → resta solo nel pannello PySide6.
- Ridisegno/temizzazione spinta del look ImGui (oltre il dark di default + accento del brand come rifinitura).
- Salvataggio profili *dal* menù in-window: i profili continuano a essere gestiti dal pannello PySide6 (il sync bidirezionale fa sì che lo stato in-window vi confluisca comunque).

## Decisioni di design (esito del brainstorming)

1. **Libreria:** `imgui-bundle` (Dear ImGui) con il **backend GLFW in puro Python** (`GlfwRenderer` da `imgui_bundle.python_backends`), agganciato alla finestra GLFW **esistente**.
   - Si usa l'API di basso livello `imgui` + `python_backends`, **NON** `immapp`/`hello_imgui` (creerebbero una loro finestra).
   - Scartate: implementazione "da zero" (non giustificata: richiederebbe font atlas, text rendering, layout, hit-testing, focus, theming); `pyimgui` (fermo a ImGui 1.82, supporto 3.12+ non garantito).
   - Verifica vincoli progetto: wheel binari `cp312`/`cp313` disponibili per Windows (AMD64) e Linux manylinux (glibc ≥2.28) → compatibili con `requires-python >=3.12` e con la CI (`windows-latest` / `ubuntu-22.04`, glibc 2.35). Dipende già da `glfw` e `PyOpenGL`.
2. **Ambito:** mirror di tutti i controlli **live**, esclusi i 2 di setup (monitor, file-picker sfondo).
3. **Sync:** **bidirezionale** — canale inverso renderer→GUI tramite segnale Qt.
4. **Attivazione:** tasto toggle **configurabile** (default `F1`).
5. **Struttura:** logica del menù isolata in una **classe dedicata** (`thread/imgui_overlay.py`), non inline in `opengl_thread.py` (modulo già molto grande).

## Architettura

### Componenti (nuovi / modificati)

| Componente | Tipo | Ruolo |
|---|---|---|
| `thread/imgui_overlay.py` → `ImGuiOverlay` | **nuovo** | Inizializza il contesto ImGui + `GlfwRenderer` sulla finestra esistente. Ogni frame: `process_inputs()` → `new_frame()` → (se aperto) costruisce i widget → `render()` → `impl.render(get_draw_data())`. Legge lo stato corrente dal renderer, scrive i messaggi su `control_queue`, notifica le modifiche via callback `on_change`. Espone `toggle()`, `is_open()`, `set_toggle_key()`, `wants_keyboard()`/`wants_mouse()` (proxy su `io.want_capture_*`), `shutdown()`. |
| `thread/opengl_thread.py` → `EqualizerOpenGLThread` | modificato | Istanzia `ImGuiOverlay` **dopo** `make_context_current`. Nel loop: disegna il menù **dopo `_render()` e prima di `glfw.swap_buffers()`**. Gestisce toggle-key + cursore. Nuovi param costruttore: `menu_toggle_key: str = "F1"`, `on_setting_changed: Callable[[dict], None] = None`. Nuovo messaggio di controllo: `set_menu_toggle_key`. |
| `gui/main_window_gui.py` → `AudioCaptureGUI` | modificato | Nuovo segnale `_setting_changed = Signal(dict)`; la callback `on_setting_changed` (invocata sul thread GL) **emette** il segnale; slot `_on_setting_changed(msg)` (thread GUI) aggiorna il widget **a segnali bloccati**. Registry `message_type → (widget, to_widget_value_fn)`. Nuovo controllo "Tasto menù" (pagina Visualizzazione). |
| Widget custom in `gui/` (`SliderCustomFrame`, `OptionMenuCustomFrame`, `ColorPickerFrame`) | modificato | Metodo per **impostare il valore senza ri-emettere** il callback (via `QSignalBlocker` sul widget Qt sottostante), così il sync inverso non rimbalza in coda. |
| `pyproject.toml` / `uv.lock` | modificato | `uv add imgui-bundle` + `uv lock`. |
| `sound-wave.spec` | modificato | `collect_all('imgui_bundle')` (estensione nativa + dati) per le build frozen Windows/Linux. |

### Flusso dati

```
                       (tasto toggle)
USER ───► key_callback ───────────────► ImGuiOverlay.toggle()  ──► cursore NORMAL/HIDDEN
                                              │
 apertura: menù visibile, legge stato corrente da  renderer._*  (stesso thread GL → lettura diretta)
                                              │
 modifica widget ImGui ──┬─► control_queue.put({"type":"set_…","value":…})  ──► drenata in cima al loop
                         │        (STESSO percorso degli slider PySide6)         ──► renderer applica
                         │
                         └─► on_setting_changed({"type":"set_…","value":…})
                                   │  (sul thread GL)
                                   └─► AudioCaptureGUI._setting_changed.emit(msg)   (segnale Qt, queued)
                                            │
                                            └─► [thread GUI] _on_setting_changed(msg)
                                                     └─► registry[type] → widget.set_silently(to_widget_value(value))
```

- **Lettura stato:** l'overlay gira **sul thread GL**, lo stesso che possiede gli attributi `self._*` del renderer → lettura diretta, nessun problema di cross-thread (i tipi sono primitivi/tuple, letti atomicamente).
- **Scrittura:** identica alla GUI — `control_queue.put(...)`, drenata da `process_control_message` in cima al loop.
- **Sync inverso:** riusa **esattamente** il pattern di `window_close_callback` → `_opengl_closed` (callback sul thread GLFW che emette un segnale Qt consegnato in coda allo slot GUI). Nessun widget toccato direttamente dal thread GL.
- **Riavvio fullscreen:** la GUI ricostruisce `EqualizerOpenGLThread` leggendo i valori dei widget — ora **aggiornati dal sync** → le modifiche in-window persistono e sono salvabili come profilo.

## Inventario controlli da rispecchiare

Tutti emettono gli **stessi** messaggi di controllo della GUI. Widget ImGui suggeriti: `slider_int`/`slider_float`, `combo`, `color_edit3`, `checkbox`. Replicare le **conversioni di unità** della GUI (ms→s per attack/release/peakcap_hold; nome→indice per le combo; on/off→bool).

### Scheda **Visualizzazione**
| Controllo | Widget ImGui | Range / valori | Messaggio | Attr. renderer |
|---|---|---|---|---|
| Modalità | combo | Barre / Radiale / Oscilloscopio / Linea-Area (→ int 0–3) | `set_viz_mode` | `_mode` |
| Trasparenza sfondo | slider_float | 0.0–1.0 | `set_alpha` | `_bg_alpha` |
| Opacità barre | slider_float | 0.0–1.0 | `set_bars_alpha` | `_bars_alpha` |
| Tasto menù *(nuovo)* | combo/selettore tasto | set di tasti supportati (default `F1`) | `set_menu_toggle_key` | (toggle key) |

### Scheda **Forma**
| Controllo | Widget | Range | Messaggio | Visibilità |
|---|---|---|---|---|
| Larghezza barre | slider_float | 0.1–1.0 | `set_bar_width` | sempre (Barre) |
| Cime arrotondate | checkbox | bool | `set_rounded` | Barre |
| Ancoraggio | combo | Dal basso / Centro (→ bool `_mirror`) | `set_bar_anchor` | Barre |
| Ordine bande | combo | Standard / Simmetrico (→ bool) | `set_band_order` | Barre/Radiale/Linea |
| Raggio foro | slider_float | 0.0–0.8 | `set_inner_radius` | **Radiale** |
| Spessore linea | slider_float | 0.002–0.06 | `set_line_thickness` | **Linea/Oscilloscopio** |
| Riempimento | combo | Linea / Area (→ bool) | `set_fill` | **Linea/Area** |
| Specchiato | checkbox | bool | `set_osc_mirror` | **Oscilloscopio** |
| Fluidità (oscill.) | slider_float | 0.0–1.0 | `set_osc_smoothing` | **Oscilloscopio** |

### Scheda **Colore**
| Controllo | Widget | Range | Messaggio | Visibilità |
|---|---|---|---|---|
| Modalità colore | combo | Classico / Tinta unica / Gradiente / Spettro (→ int 0–3) | `set_color_mode` | sempre |
| Soglia giallo | slider_float | 0.0–1.0 | `set_green_split` | **Classico** |
| Soglia rosso | slider_float | 0.0–1.0 | `set_yellow_split` | **Classico** |
| Colore barre / base | color_edit3 | RGB 0–1 | `set_color_a` | **Tinta unica / Gradiente** |
| Colore cima | color_edit3 | RGB 0–1 | `set_color_b` | **Gradiente** |

### Scheda **Audio**
| Controllo | Widget | Range | Messaggio | Note |
|---|---|---|---|---|
| Soglia rumore | slider_float | 0.0–1.0 | `set_noise_threshold` | |
| Bande di frequenza | slider_int | 1–100 | `set_frequency_bands` | ricalcola le bande |
| Pavimento dB | slider_int | −90…−25 | `set_db_floor` | |
| Tetto dB | slider_int | −20…6 | `set_db_ceiling` | |
| Attacco (ms) | slider_int | 1–100 ms | `set_attack_tau` | valore **/1000** nel messaggio |
| Rilascio (ms) | slider_int | 20–1000 ms | `set_release_tau` | valore **/1000** nel messaggio |
| Tilt (dB/ottava) | slider_float | −6…6 | `set_tilt_db_per_oct` | |

### Scheda **Effetti**
| Controllo | Widget | Range | Messaggio | Visibilità |
|---|---|---|---|---|
| Peak-cap | checkbox | bool | `set_peakcap_enabled` | **solo Barre** |
| Colore cap | color_edit3 | RGB 0–1 | `set_peakcap_color` | **solo Barre** |
| Hold (ms) | slider_int | 0–2000 ms | `set_peakcap_hold` | **solo Barre**, **/1000** |
| Caduta (/s) | slider_float | 0.1–3.0 | `set_peakcap_fall` | **solo Barre** |
| Glow | checkbox | bool | `set_bloom_enabled` | sempre |
| Intensità (glow) | slider_float | 0.0–3.0 | `set_bloom_intensity` | sempre |
| Beat pulse | checkbox | bool | `set_beat_enabled` | sempre |
| Intensità zoom | slider_float | 0.0–0.3 | `set_beat_intensity` | sempre |
| Sensibilità | slider_float | 1.05–3.0 | `set_beat_sensitivity` | sempre |

La **visibilità condizionale** replica `_refresh_group_visibility` della GUI: con ImGui (immediate mode) è naturale — basta racchiudere i widget in blocchi `if`.

## Interazione & input

- **Toggle configurabile:** default `F1`. Mappa nome-tasto → codice GLFW (es. `F1..F12`, `M`, `Tab` sconsigliato perché ImGui lo usa per la navigazione campi). Passato al costruttore e aggiornabile via `set_menu_toggle_key`. Esposto come controllo "Tasto menù" nel pannello e quindi salvabile nei profili.
- **ESC:** menù **aperto** → chiude il menù; menù **chiuso** → esce dal fullscreen (comportamento attuale preservato).
- **Cursore:** `GLFW_CURSOR_NORMAL` quando il menù è aperto, `GLFW_CURSOR_HIDDEN` quando chiuso.
- **Callback GLFW:** il `GlfwRenderer` installa proprie callback (key/char/mouse/scroll). La `key_callback` esistente va **concatenata**: gestire toggle/ESC solo quando ImGui non sta catturando la tastiera (`io.want_capture_keyboard`), e inoltrare gli eventi al backend ImGui. Vedi Rischio #1.

## Layout del menù

Una finestra ImGui con **`TabBar` di 5 schede** che rispecchiano le pagine attuali: **Visualizzazione · Forma · Colore · Audio · Effetti**. Stessi controlli e stessa visibilità condizionale. Stile: dark di default, con un tocco di accento del brand (`gui/theme.py::ACCENT`) come rifinitura.

## Modifiche lato GUI (sync inverso)

1. Segnale `_setting_changed = Signal(dict)` su `AudioCaptureGUI`.
2. Callback `on_setting_changed(msg)` passata al costruttore del thread; il suo unico compito è `self._setting_changed.emit(msg)` (thread-safe, consegnato in coda).
3. Slot `_on_setting_changed(msg)` (thread GUI): cerca nel registry, converte il valore in unità-widget e imposta il widget **a segnali bloccati**.
4. Registry `message_type → (widget, to_widget_value_fn)`: `to_widget_value_fn` è l'**inverso** della conversione forward del controllo (identità per la maggior parte; `×1000` per attack/release/peakcap_hold; valore→indice per le combo). Le conversioni forward sono già in `main_window_gui.py`.
5. Ogni widget custom riusato dal sync espone un "set valore senza emettere" (`QSignalBlocker`).

## Dipendenze & build

- `uv add imgui-bundle` → `uv lock`; eseguire `uv sync` su **entrambe** le venv (Windows `.venv`, WSL2 `.venv-linux`).
- `sound-wave.spec`: aggiungere `collect_all('imgui_bundle')`. Le wheel cp312/cp313 esistono per Windows e manylinux ≥2.28 → la CI (`windows-latest`/`ubuntu-22.04`) resta verde.

## Verifica (il progetto non ha test suite)

- **Unit test logica pura** (stile `tests/test_frame_time_stats.py`): (a) mappa nome-tasto → codice GLFW; (b) registry/conversione `message_type → valore-widget` (round-trip con le conversioni forward della GUI).
- **Smoke test import** headless (EGL) di `imgui_bundle` + creazione di un `GlfwRenderer` su una finestra GLFW invisibile, se l'ambiente lo consente.
- **Verifica reale** su Windows con display: aprire il menù col toggle, modificare almeno un controllo per ciascuna delle 5 schede, verificare che (1) la visualizzazione reagisca, (2) il widget Qt corrispondente si aggiorni, (3) le modifiche persistano dopo chiusura/riavvio del fullscreen, (4) ESC chiuda il menù se aperto ed esca se chiuso, (5) il cursore compaia/sparisca correttamente.

## Rischi & mitigazioni

1. **Concatenazione callback GLFW (rischio principale):** il backend Python potrebbe sovrascrivere la `key_callback` esistente invece di concatenarla. *Mitigazione:* prototipo minimo all'inizio dell'implementazione; installare le proprie callback **dopo** il `GlfwRenderer` e inoltrare manualmente gli eventi ai handler del backend, oppure leggere lo stato tasti da `io`. Gestire toggle/ESC solo se `not io.want_capture_keyboard`.
2. **Stato GL condiviso:** disegnare ImGui **per ultimo**, sul framebuffer di default (dopo il composite del bloom). Il backend ImGui salva/ripristina lo stato GL; il frame successivo `_render` ri-binda comunque VAO/shader/uniform.
3. **Costo a menù chiuso:** quando chiuso, non costruire widget (solo `new_frame`/`render` su draw-list vuota → trascurabile). Opzionalmente saltare del tutto il ciclo ImGui quando chiuso, mantenendo `process_inputs` per coerenza dell'input.
4. **Build frozen:** `imgui_bundle` ha un'estensione nativa; senza `collect_all` la build frozen fallirebbe a runtime. Validare il bundle eseguendo l'artefatto (la CI builda ma non esegue).

## Out of scope (ribadito)
- Selettore monitor e file-picker sfondo in-window.
- Persistenza/profili gestiti dal menù in-window (restano nel pannello PySide6; il sync vi fa confluire lo stato).
