# Audit bug della codebase — 2026-07-02

Risultato dell'audit completo del codice applicativo di "Sound Wave" (perimetro:
`main.py`, `gui/`, `thread/`, `profiles.py`, `resource_manager.py`, `tests/`;
esclusi packaging/CI). Oggetto: bug concreti e rischi latenti (race condition,
leak di risorse, edge case pericolosi); esclusi stile e refactoring.

**Metodo:** 4 agenti di analisi in parallelo (renderer OpenGL; thread ausiliari
e DSP; GUI principale e contratti; profili e widget), ogni segnalazione poi
riverificata indipendentemente sul codice. Lo stato di verifica è indicato per
ciascun finding:

- **[V]** = verificato direttamente sul codice durante il consolidamento
- **[A]** = verificato dall'agente (citazioni file:riga ricontrollate a campione)

Le righe citate si riferiscono allo stato del repo al commit `b507f36`.

> **Esito remediation (branch `bugfix/audit-2026-07`):** tutti i finding
> CRITICI e MEDI sono stati risolti; dei MINORI sono risolti tutti tranne
> m8 (quantizzazione slider float — innocua e convergente), m13 (join decoder
> con timeout — richiede I/O patologico, mitigato dal daemon flag), m17 (API
> GLFW su thread worker — limite architetturale documentato in CLAUDE.md),
> m18 (contratto di get_window_for_fft — documentato nel docstring) e m19
> (`clear_video` senza sender — affordance UI fuori scope). m24 è risolto
> dalla suite di test estesa. Dettagli per-fix nel piano
> `docs/superpowers/plans/2026-07-02-bonifica-bug-audit.md`.

---

## CRITICI — crash, freeze della GUI, app che non parte

### C1. Uno `stop()` durante l'init del renderer viene inghiottito → GUI congelata nel join [V]

- **Dove:** `thread/opengl_thread.py:1537-1539`; siti di freeze `gui/main_window_gui.py:946-948` e `closeEvent`
- **Confidenza:** alta
- `run()` esegue incondizionatamente `stop_event.clear()` dopo la creazione della
  finestra. Una richiesta di stop arrivata nella finestra di init (init GLFW +
  enumerazione monitor + finestra MSAA: decine/centinaia di ms) viene cancellata:
  il main loop parte e non ha più motivo di uscire, mentre la GUI è bloccata in
  `while is_alive(): join(timeout=0.1)` (stesso pattern in `closeEvent` → l'app
  non si chiude). Il clear è un residuo senza casi legittimi: un
  `threading.Thread` non è riavviabile.
- **Scenario:** doppio click su «Avvia» / «Ferma», oppure chiusura dell'app
  subito dopo l'avvio del fullscreen → pannello di controllo congelato; si
  recupera solo con ESC nella finestra GL o uccidendo il processo.

### C2. Scritture profilo non atomiche + auto-load all'avvio non protetto → un profilo corrotto impedisce l'avvio [V]

- **Dove:** `profiles.py:179-181` (`write_envelope`), `profiles.py:202-205` (`set_last`); consumo `gui/main_window_gui.py:188-191`
- **Confidenza:** alta (verificato empiricamente: file troncato → `JSONDecodeError`)
- `write_envelope`/`set_last` scrivono il JSON in-place con `write_text` (niente
  file temporaneo + `os.replace`); l'auto-load in `AudioCaptureGUI.__init__` non
  ha try/except, quindi l'eccezione esce dal costruttore e da `main()`.
- **Scenario:** kill/blackout a metà salvataggio → JSON troncato su disco → a
  ogni lancio successivo l'app muore con traceback prima di mostrare la
  finestra, finché l'utente non cancella a mano il file in
  `~/.config/sound-wave/` (o `%APPDATA%\sound-wave\`).

### C3. `deserialize` non valida i tipi + `_apply_profile` fuori dai try → crash a metà applicazione, profilo rotto persistito [V]

- **Dove:** `profiles.py:82`; siti di crash `gui/main_window_gui.py:1171`, `gui/color_picker_frame.py:22`, `gui/slider_frame.py:122`; chiamate non protette `gui/main_window_gui.py:1309` e `:1353`
- **Confidenza:** alta (verificato empiricamente con file di test)
- Un utente può importare qualunque file; `deserialize` riempie solo le chiavi
  mancanti, i valori del tipo sbagliato passano intatti: `"viz_mode": null` →
  `TypeError` in `int(s["viz_mode"])`; `"peakcap_color": [1,2]` → `ValueError`
  in `rgb01_to_hex`; `"n_bands": "x"` → `TypeError` nello slider. Sia
  `_on_profile_selected` sia `_on_profile_import` chiamano `_apply_profile`
  **fuori** dal loro blocco try, e `import_file` salva il profilo nello store
  **prima** del crash.
- **Scenario:** import di un JSON malformato → traceback nello slot Qt, GUI
  aggiornata a metà, messaggi parziali in coda al renderer, e il profilo rotto
  resta nella combo: ogni nuova selezione ri-crasha.

### C4. Race al riavvio del renderer: il `glfw.terminate()` del vecchio thread distrugge lo stato GLFW del nuovo [V]

- **Dove:** `thread/opengl_thread.py:1730-1733` (callback su ESC prima del teardown), `:1685-1696` (`finally`); `gui/main_window_gui.py:1007-1009` (`_on_opengl_closed` azzera il ref senza join)
- **Confidenza:** media (meccanica certa; innesco dipendente dal timing)
- Su ESC (o su eccezione del renderer) la GUI è notificata subito, mentre il
  thread GL è ancora nel `finally` (join del decoder video fino a 1 s, shutdown
  ImGui, cleanup GL). `_on_opengl_closed` azzera il riferimento senza join, il
  bottone torna «Avvia». GLFW è stato globale di processo e non ref-counted: il
  `glfw.init()` del nuovo thread è un no-op e il `glfw.terminate()` del vecchio
  invalida **tutte** le finestre, compresa quella del nuovo renderer.
- **Scenario:** ESC → riclick rapido su «Avvia» (specie con video di sfondo
  attivo, che allunga il teardown) → crash / `GLFWError` / renderer che muore
  subito. Conflitto analogo per il doppio `imgui.create_context()`.
- Segnalato indipendentemente da 3 agenti su 4.

### C5. `stop()` del thread audio non interrompe un `record()` bloccato → freeze alla chiusura o al cambio device (Linux) [V]

- **Dove:** `thread/audioCaptureThread.py:38-42` (`record()` bloccante), `:68-69` (`stop()` = solo flag); join illimitati `gui/main_window_gui.py:666-668` e `closeEvent`
- **Confidenza:** media (loop bloccante verificato nel backend `soundcard`
  installato; la frequenza dipende dal comportamento di sospensione di PulseAudio)
- `stop()` alza un flag controllato solo *tra* le chiamate a `record()`. Sul
  backend PulseAudio un source «monitor» non produce dati durante il silenzio
  (`_record_chunk` attende dati indefinitamente); il thread non è daemon.
- **Scenario (Linux):** nessun audio in riproduzione → l'utente chiude la
  finestra o cambia device → `record()` non ritorna → join infinito → GUI
  congelata e processo che non esce. Su Windows WASAPI sintetizza zeri dopo ~4
  block length, quindi il problema non si presenta.

---

## MEDI — comportamento visibilmente errato, leak, desync

### M1. Alt-F4 sulla finestra GL non notifica la GUI → stato desincronizzato e coda che accumula [V]
- **Dove:** `thread/opengl_thread.py:1608` (condizione del loop), `:1683-1696` (`finally` senza callback)
- `window_close_callback` è invocato solo dal ramo ESC di `key_callback` e dal
  ramo `except`; l'uscita normale per `window_should_close` (Alt-F4/chiusura WM)
  termina il thread in silenzio. La GUI mantiene `equalizer_opengl_thread` ≠
  None, il bottone resta «Ferma», e ogni modifica successiva accumula messaggi
  (inclusi `set_image` con immagini PIL) nella `equalizer_control_queue`
  persistente, riapplicati in blocco al prossimo avvio. Confidenza: alta.

### M2. Cambio device audio con renderer attivo → renderer orfano sulla coda vecchia, barre congelate [V]
- **Dove:** `gui/main_window_gui.py:650-657`
- `on_device_selected` sostituisce `self.audio_queue` con una nuova `queue.Queue`,
  ma il renderer in esecuzione tiene il riferimento ricevuto alla costruzione, e
  non esiste un messaggio di controllo per sostituirlo. Il nuovo capture thread
  riempie la coda nuova; il renderer drena per sempre quella vecchia e vuota →
  FFT ricalcolata all'infinito sulla stessa finestra stantia → barre congelate
  fino a stop/riavvio manuale del fullscreen. Confidenza: alta.

### M3. Morte silenziosa del thread di cattura; riselezionare lo stesso device è un no-op [V]
- **Dove:** `thread/audioCaptureThread.py:61-63` (`break` silenzioso), `:37` (eccezione di `recorder()` fuori dal try); `gui/main_window_gui.py:650` (guardia `!= device`)
- Su errore di cattura (device scollegato: errore COM WASAPI / `PA_STREAM_FAILED`)
  il producer stampa e muore; nessun segnale alla GUI (`audio_thread` resta
  impostato, «Avvia» resta abilitato, barre congelate). La guardia
  `last_device_selected != device` rende no-op il click sullo stesso device,
  quindi non esiste percorso di recovery senza selezionare un device diverso e
  tornare indietro. Se è `recorder()` stesso a sollevare all'apertura,
  l'eccezione esce da `run()` con traceback grezzo. Confidenza: alta.

### M4. `menu_toggle_key` mai persistito: manca da `DEFAULTS` → sempre resettato a F1 [V]
- **Dove:** `profiles.py:24-59`; prodotto in `gui/main_window_gui.py:1155`, consumato `:1198`, push live `:1203`
- `serialize`/`deserialize` iterano solo su `DEFAULTS`, che non è stato
  aggiornato quando il menù in-window ha introdotto la chiave (commit `b507f36`):
  `_collect_profile` la scrive, `serialize` la scarta, al load
  `s.get("menu_toggle_key", "F1")` restituisce sempre F1 — e a renderer attivo
  viene anche pushato `set_menu_toggle_key: "F1"`, revocando un'impostazione live
  che il profilo non doveva toccare. Verificato empiricamente. Confidenza: alta.

### M5. Importare qualunque JSON dict-shaped crea in silenzio un profilo tutto-default [V]
- **Dove:** `profiles.py:183-193` (in particolare `:185`)
- `import_file` non valida l'envelope: `raw.get("settings", {})` su un file
  estraneo (es. `package.json`) → `{}` → `deserialize` riempie coi default →
  profilo fantasma creato e applicato senza alcun errore, sostituendo le
  impostazioni correnti a schermo. `APP_TAG`/`CURRENT_VERSION`, scritti
  all'export, non vengono mai controllati all'import. Confidenza: alta.

### M6. Nessuna validazione range al load: stato e renderer ricevono valori fuori range che lo slider mostra clampati [V]
- **Dove:** `profiles.py:64-72` (dichiara «No validazione range»); `gui/slider_frame.py:118-125`; `gui/main_window_gui.py:1195-1249`
- `_apply_profile` salva `float(s[...])` in `self.*` e lo pusha al renderer,
  mentre `QSlider.setValue` clampa silenziosamente solo la visualizzazione.
  Esempio: `"bar_width": 5.0` (range 0.1–1.0) → slider a 1.0 ma renderer a 5.0
  (barre sovrapposte); «Salva» ri-persiste 5.0 rendendo il valore sticky. Vale
  per tutti gli slider (`db_floor`, `inner_radius`, `beat_sensitivity`, …).
  Confidenza: alta.

### M7. «Salva come…» sovrascrive senza conferma; nomi diversi collidono sullo stesso file [V]
- **Dove:** `gui/main_window_gui.py:1320-1327`; `profiles.py:85-88` (`sanitize_name`), `:158-159`
- Nessun controllo di esistenza né conferma; inoltre `sanitize_name("a/b")` ==
  `sanitize_name("a_b")` == `"a_b"`, quindi nomi visivamente diversi distruggono
  lo stesso profilo. Perdita di dati di una configurazione salvata. Confidenza: alta.

### M8. La combo profili può mostrare un profilo mai caricato → «Salva» lo sovrascrive con impostazioni non sue [A]
- **Dove:** `gui/profile_bar_frame.py:58-68`; call site `gui/main_window_gui.py:265`, `:1339`
- `set_profiles(names, selected=None)` ripopola a segnali bloccati: senza
  `selected` (avvio senza `last_profile` valido, o dopo «Elimina») la combo
  atterra su `names[0]` senza far scattare `on_select`, quindi il profilo
  mostrato NON è attivo. Un successivo «Salva» (`current_name()`) lo sovrascrive
  con le impostazioni correnti, che non gli appartengono. Confidenza: alta.

### M9. Clamp ordine-dipendente di `set_green_split`/`set_yellow_split` → desync GUI/renderer al load di un profilo [V]
- **Dove:** `thread/opengl_thread.py:1325-1333`; sender `gui/main_window_gui.py:1216-1217`
- Ogni messaggio clampa contro il valore *corrente* dell'altra soglia. Con
  renderer a green=0.5/yellow=0.8, un profilo con green=0.85/yellow=0.9 produce:
  green→min(0.85, 0.8)=0.8, poi yellow→0.9. Il renderer resta a 0.8 mentre la
  GUI mostra 0.85, finché non si ritocca lo slider. Il clamp non è nemmeno
  replicato lato GUI (`update_green_split`/`update_yellow_split`). Confidenza: alta.

### M10. Framebuffer bloom incompleto → leak di texture/FBO ritentato ogni frame [V]
- **Dove:** `thread/opengl_thread.py:972-1000` (ramo `:993-995`); retry per-frame via `_render_bloom`
- Sul fallimento di `glCheckFramebufferStatus` la funzione ritorna `False` senza
  eliminare texture/FBO appena generati e senza marcare «non riprovare»
  (`_bloom_size` resta None): con il bloom attivo viene rieseguita a ogni frame,
  leakando 1–4 oggetti GL per frame (~60–144/s). Innesco realistico:
  `GL_RGB16F` non è un formato color-renderable obbligatorio in GL 3.3 (lo è
  `GL_RGBA16F`) → dipende dal driver. Confidenza: alta sulla meccanica, media
  sull'innesco.

### M11. Un'eccezione in `process_control_message` uccide l'intero renderer (es. immagine corrotta) [V]
- **Dove:** `thread/opengl_thread.py:1614-1617` (drain nel try principale), `:1400-1404`, `:1761-1762` (`load_texture`)
- Il drain della control queue gira dentro il `try` di `run()`: qualunque
  eccezione in un handler chiude la finestra. Caso realistico: la GUI fa
  `Image.open()` (valida solo l'header, decodifica lazy) e passa l'oggetto PIL;
  la decodifica reale (`transpose`/`tobytes`) avviene sul thread GL e su un file
  troncato/cancellato solleva → «OpenGL thread error», visualizzatore chiuso di
  colpo. Confidenza: alta.

### M12. Toggle key ed ESC gestiti senza consultare `want_text_input` di ImGui [A]
- **Dove:** `thread/opengl_thread.py:1714-1733`; `thread/menu_keys.py:9-13`
- `key_callback` inoltra il tasto a ImGui e poi agisce *incondizionatamente* su
  toggle ed ESC. `TOGGLE_KEY_NAMES` esclude Tab «perché lo usa ImGui» ma include
  M/N/P, altrettanto digitabili nei campi di testo (Ctrl+click su uno slider
  ImGui lo trasforma in input testuale): digitare il tasto-toggle mentre si
  edita chiude il menù a metà editing; ESC per annullare un edit chiude anche il
  menù (doppia gestione). L'agente ha verificato che il backend `imgui_bundle`
  non consuma gli eventi. Confidenza: alta.

### M13. `AnimatedButton` accetta `hover_color` e lo ignora [A]
- **Dove:** `gui/animated_button.py:58`, `:109-110`, `:211`; caller `gui/main_window_gui.py:280`, `:662`, `:933`, `:936`
- `_hover_brush()` ritorna sempre `base.lighter(115)`; `_hover_color_override` è
  scritto in due punti e letto in nessuno (verificato con grep). Le tinte hover
  semantiche del tema (`COLOR_STOP_HOVER` ecc.) sono ignorate: l'hover di
  «Ferma» schiarisce il rosso invece di scurirlo. Confidenza: alta.

### M14. Swatch «Tinta unica» incoerente con lo stato; i due picker su `color_a` non si sincronizzano [V]
- **Dove:** `gui/main_window_gui.py:104` (stato `(0.231, 0.420, 1.0)` = `#3b6bff`) vs `:482` (`initial_color='#22d36a'`); `:480-490`
- All'avvio in modalità «Tinta unica» le barre sono blu ma lo swatch è verde.
  Inoltre `bars_solid_color` e `bars_grad_base` pilotano entrambi
  `update_color_a` ma non si aggiornano a vicenda (solo l'echo del menù
  in-window li sincronizza): cambiare il colore in una modalità lascia l'altra
  con lo swatch vecchio. Confidenza: alta.

### M15. Video leggibile ma senza frame → loop stretto che rigenera il subprocess ffmpeg senza backoff [V]
- **Dove:** `thread/video_decode_thread.py:48-71`
- Se `get_reader()` riesce ma lo stream non produce frame (header del container
  valido, nessun frame decodificabile), il `for` termina subito e il `while`
  esterno ricrea reader e subprocess ffmpeg alla massima velocità, finché lo
  sfondo video resta attivo. Nessuna rilevazione «zero frame prodotti» né
  backoff. (Un video da 1 frame è throttlato solo dalla back-pressure della
  coda: comunque ~fps spawn/s.) Confidenza: media (molti file corrotti sollevano
  invece un'eccezione, che è gestita).

---

## MINORI — edge case improbabili, incoerenze contenute

### Profili / store
- **m1.** `resolve_bg_ref` calcola il default eagerly e non guardato:
  `_default_bg_path` → `get_image_path` solleva `FileNotFoundError` se manca
  l'asset del bundle, trasformando il resolver «safe fallback» in un crash (e
  bloccando l'avvio via `gui/main_window_gui.py:147`). `profiles.py:116`,
  `resource_manager.py:31`. Probabilità bassa. [A]
- **m2.** `get_last` cattura solo `JSONDecodeError`/`OSError`: un
  `settings.json` valido-ma-non-dict (es. lista) → `AttributeError` non gestito
  all'avvio. `profiles.py:195-200`. [V]
- **m3.** `sanitize_name` lascia passare i nomi riservati Windows
  (`CON`, `NUL`, `COM1`…) → `CON.json` fallisce la scrittura o punta a un
  device. `profiles.py:61,85-88`. [A]
- **m4.** Profili creati esternamente con caratteri fuori whitelist sono
  listati (`list_names` ritorna gli stem raw) ma mai caricabili/eliminabili
  (`_path_for` risanitizza → file diverso). `profiles.py:158-162`. [A]

### Widget
- **m5.** `AnimatedButton.update_base_colors` applica il nuovo colore solo se
  `not underMouse()`: invocato dal click stesso del bottone footer, lascia il
  verde-hover con testo «Ferma» finché il mouse non esce. `gui/animated_button.py:212-215`. [A]
- **m6.** `SliderCustomFrame` ignora `steps` per gli slider int (lo slider
  «Bande di frequenza» chiede `steps=32` e ottiene un 1–100 continuo).
  `gui/slider_frame.py:46`. [A]
- **m7.** `warning_trigger_value=0` disabilita il warning (check di
  truthiness invece di `is not None`). `gui/slider_frame.py:38,82,107,127`.
  Latente (l'unico uso attuale è 15). [A]
- **m8.** Round-trip dei float slider quantizzato di un tick: `set_value(0.012)`
  → `get_value()` = 0.011976 → è ciò che «Salva» persiste. Convergente, nessuna
  deriva cumulativa. `gui/slider_frame.py:91-99`. [A]
- **m9.** `OptionMenuCustomFrame.set_value` è un no-op silenzioso per valori
  sconosciuti → divergenza widget/stato (es. `menu_toggle_key` spazzatura da
  profilo). `gui/optionmenu_frame.py:54-60`. [A]
- **m10.** Il file dialog degli sfondi parte da `"resources/bg"` relativo alla
  CWD: sbagliato nel frozen build (le risorse stanno in `sys._MEIPASS`) e con
  CWD ≠ root del repo; dovrebbe passare da `ResourceManager`.
  `gui/background_filepicker_frame.py:50`. [A]

### Renderer / thread
- **m11.** Viewport e `_fb_width/_fb_height` catturati una sola volta
  all'avvio; nessuna framebuffer-size callback (solo la resize di ImGui è
  registrata) → cambio risoluzione/DPI a runtime lascia il rendering sul
  viewport vecchio e i FBO bloom alla vecchia half-res. `thread/opengl_thread.py:1549-1551`. [A]
- **m12.** La maschera dei bassi per la beat detection include il bin DC
  (0 Hz): un offset DC nel segnale gonfia stabilmente l'energia dei bassi e
  desensibilizza il beat. `thread/opengl_thread.py:556-557`. [A]
- **m13.** `_stop_video` fa `join(timeout=1.0)` e azzera il riferimento: un
  decoder bloccato in I/O (share di rete, disco lento) resta orfano con il
  subprocess ffmpeg vivo. `thread/opengl_thread.py:1804-1807`. Probabilità bassa. [A]
- **m14.** Metadati video con fps non finito (`inf`): il filtro
  `if fps and fps > 0` lo lascia passare → `_video_frame_interval = 0.0` →
  `ZeroDivisionError` in `_advance_video` sul thread GL → il renderer si chiude
  per colpa di un metadato. `thread/video_decode_thread.py:52-54`,
  `thread/opengl_thread.py:1846`. Probabilità bassa, meccanica certa. [V]
- **m15.** `imgui.create_context()` e `GlfwRenderer(...)` non sono
  exception-paired: se il backend solleva, il contesto ImGui resta orfano e si
  accumula a ogni retry di avvio. `thread/imgui_overlay.py:24-28`. [A]
- **m16.** L'overlay mostra `bg_alpha` 0.0 quando `None` (che il renderer
  disegna come 1.0): incoerenza `float(r._bg_alpha or 0.0)` vs
  `1.0 if None`. Non raggiungibile dalla GUI attuale (passa sempre 0.5).
  `thread/imgui_overlay.py:92` vs `thread/opengl_thread.py:1078`. [A]
- **m17.** Tutta l'API GLFW (init/create_window/poll_events/terminate) gira su
  un thread worker: contratto GLFW violato (documentato «main thread only»).
  Funziona di fatto su Windows/X11 (piattaforme target), fallirebbe su macOS.
  **Da documentare come limite architetturale, non fixabile a basso costo.**
  `thread/opengl_thread.py:1487,1530,1673,1696`. [A]
- **m18.** `AudioBufferAccumulator.get_window_for_fft()` ritorna
  `np.array([])` (shape `(0,)`, float64) a buffer non pieno, contro il
  contratto `(N, C)` float32 del percorso normale; e quando `write_pos == 0`
  ritorna il buffer live aliasato (oggi sicuro: single-thread verificato).
  Trappola latente per usi futuri. `thread/AudioBufferAccumulator.py:84-86`. [A]
- **m19.** `clear_video` è gestito dal renderer ma nessuno lo invia mai (né
  GUI né overlay): ramo morto; non esiste percorso UI per tornare da video a
  immagine se non applicando un file immagine. `thread/opengl_thread.py:1410-1413`. [A]

### GUI
- **m20.** `_apply_image_background` azzera `bg_video_path` *prima* che
  `Image.open` possa fallire: su errore il renderer continua col video ma lo
  stato GUI dice «immagine» → un «Salva» in quel momento persiste lo sfondo
  sbagliato. `gui/main_window_gui.py:689-693`. [A]
- **m21.** `_apply_background_from_profile` assegna `bg_image_path = path`
  prima dell'`open`: su fallimento persiste un path inaccessibile mentre
  l'avviso dichiara di usare lo sfondo predefinito. `gui/main_window_gui.py:1284-1289`. [A]
- **m22.** Avvio senza messaggio d'errore in due probe d'ambiente non
  guardati nel costruttore: `raise Exception("GLFW initialization failed")`
  (`gui/main_window_gui.py:633-634` via `get_available_monitors`) e
  `asyncio.run(self.load_devices())` (`:225`, `sc.all_microphones` solleva su
  Linux senza PulseAudio/PipeWire) → traceback senza finestra né dialog. [A]

### Test / igiene
- **m23.** `tests/test_menu_keys.py:29-31` non può rilevare il fallback
  silenzioso a F1 che il commento dichiara di testare: `toggle_key_to_glfw`
  ritorna `KEY_F1` (=290 ≥ 0) anche per nomi non mappati, quindi
  `assert ... >= 0` passa sempre. Il check corretto è l'appartenenza a
  `_NAME_TO_GLFW`. [A]
- **m24.** Gap in `tests/test_profiles.py`: i percorsi pericolosi sono
  esattamente quelli non testati — tipi errati in `deserialize` (C3), envelope
  estraneo accettato (M5), round-trip di `menu_toggle_key` (M4), file
  corrotto/troncato in `load` (C2), `get_last` non-dict (m2). Igiene: i test
  usano `mkdtemp` senza cleanup. [A]
- **m25.** `imgui.ini` viene scritto nella CWD dall'overlay ImGui e non è in
  `.gitignore` (oggi appare come file untracked nella root del repo). [V]

---

## Aree verificate e risultate pulite

- Contratto completo dei **36 messaggi di controllo** GUI→renderer: tutti
  gestiti in `process_control_message`, tipi/unità coerenti (conversioni ms↔s
  corrette in entrambe le direzioni, echo del menù in-window compreso); i
  kwargs del costruttore di `EqualizerOpenGLThread` combaciano con la firma.
- Matematica del ring buffer (`AudioBufferAccumulator`: wrap, code, snapshot) e
  della mappa bin→banda (nessuna banda vuota, nessun double-count); `log10`
  protetto, pacing video (accumulatore + `MAX_CATCHUP_FRAMES` + back-pressure)
  corretto; PBO sbindato dopo l'upload video.
- Teardown in `closeEvent`: ordine di join audio→renderer corretto e non
  deadlockabile (il renderer legge l'audio con `get_nowait`).
- `_cleanup_gl` idempotente e sicuro su init parziale; pairing begin/end
  dell'overlay ImGui corretto; registrazione callback GLFW completa
  (`attach_callbacks=False` + chaining manuale, firme verificate sul backend
  installato).
- `gui/theme.py` (QSS valido, font correttamente deferiti), `gui/help_window.py`,
  `FrameTimeStats` (asserzioni dei test ri-derivate a mano, corrette),
  `resource_manager` coerente con `sound-wave.spec` per il frozen build.
