# Design — Bonifica bug della codebase (audit 2026-07-02)

## Contesto e obiettivo

Un audit completo del codice applicativo (4 agenti in parallelo per sottosistema,
ogni segnalazione riverificata indipendentemente sul codice) ha prodotto
**5 finding critici, 15 medi e ~25 minori**. Il report completo, con file:riga,
scenari di innesco, confidenza e stato di verifica per ciascun finding, è in
**`docs/superpowers/audits/2026-07-02-audit-bug-codebase.md`** — questo design
non lo duplica: definisce *come* risolverli.

Obiettivo: eliminare tutti i critici e i medi, e la quasi totalità dei minori,
mantenendo invariato il comportamento visibile dell'app per l'utente che non
incappa nei bug.

## Decisioni di perimetro (concordate)

- **Cosa conta come bug:** difetti concreti + rischi latenti (race, leak,
  teardown, edge case). Esclusi stile e refactoring non funzionali.
- **Ambito file:** solo codice applicativo (`main.py`, `gui/`, `thread/`,
  `profiles.py`, `resource_manager.py`, `tests/`). Packaging/CI fuori scope.
- **Organizzazione:** un unico branch `bugfix/audit-2026-07` da `master`,
  un commit per unità di fix, in ordine di gravità; PR unica verso `master`.
- **Test:** regressione automatica per la logica pura; verifica manuale
  (checklist) per GL/audio/GUI.

## Architettura della remediation

Un'*unità di fix* raggruppa i finding che condividono la causa radice: si
committa e si verifica come un tutto. Gli ID (C=critico, M=medio, m=minore)
rimandano al report dell'audit.

### Fase 1 — Critici (5 unità)

**F1.1 — Stop non inghiottibile durante l'init del renderer** _(risolve C1)_
Rimuovere il blocco `if self.stop_event.is_set(): self.stop_event.clear()` in
`EqualizerOpenGLThread.run()` (`thread/opengl_thread.py:1537-1539`). Un
`threading.Thread` non è riavviabile: non esiste un caso legittimo per il clear.
Uno stop richiesto durante l'init fa uscire il main loop al primo check.

**F1.2 — Persistenza profili atomica + avvio resiliente** _(risolve C2)_
`write_envelope` e `set_last` scrivono su file temporaneo nella stessa directory
e concludono con `os.replace` (atomico sullo stesso filesystem). L'auto-load in
`AudioCaptureGUI.__init__` viene protetto: su qualunque eccezione l'app parte
con i default e mostra un warning non bloccante (statusbar o `QMessageBox`
dopo `show()`), senza cancellare il file corrotto (l'utente può ispezionarlo).

**F1.3 — `deserialize` valida tipi e range; `_apply_profile` protetto** _(risolve C3 + M6)_
`deserialize` diventa lo strangolatoio di validazione: per ogni chiave di
`DEFAULTS` un validatore per famiglia (int, float, bool, str, colore = sequenza
di 3 numeri clampati in [0,1], background = dict con shape nota) con **range
per chiave** allineati ai limiti degli slider GUI. Valore di tipo invalido →
default; fuori range → clamp. Le chiamate a `_apply_profile` in
`_on_profile_selected` e `_on_profile_import` entrano nei rispettivi try (difesa
in profondità: dopo la validazione non dovrebbero più poter sollevare).

**F1.4 — Chiusura del renderer: notifica unica a teardown completato** _(risolve C4 + M1)_
`window_close_callback` viene invocato **solo** alla fine del `finally` di
`run()`, dopo `glfw.terminate()`: ogni percorso di uscita (ESC, Alt-F4, stop
da GUI, eccezione) converge lì. Il ramo ESC di `key_callback` si limita a
`set_window_should_close`; il ramo `except` non chiama più il callback
direttamente. `_on_opengl_closed` fa un `join()` (ormai quasi istantaneo:
il thread ha già finito il teardown) prima di azzerare il riferimento.
Nota: quando lo stop parte dalla GUI (`fullscreen_command`/`closeEvent`) il
join è già lì; il callback arriverà a riferimento già azzerato → lo slot deve
essere idempotente (`if self.equalizer_opengl_thread is None: reset bottone e basta`).

**F1.5 — Thread audio non bloccante alla chiusura** _(risolve C5 + parte di M3)_
`AudioCaptureThread` diventa `daemon=True`; i due siti GUI sostituiscono il
`while is_alive(): join(0.1)` con un join a timeout limitato (~2 s) e poi
proseguono (il daemon non tiene vivo il processo). `self.device.recorder(...)`
entra nel try di `run()` così un device scomparso all'apertura non produce un
traceback grezzo.

### Fase 2 — Medi (11 unità)

**F2.1 — Cambio device senza orfanare il renderer** _(M2)_ — `on_device_selected`
non ricrea `self.audio_queue`: riusa la stessa istanza (drenandola con
`get_nowait` in loop) così il renderer in esecuzione continua a leggere la coda
giusta.

**F2.2 — Morte del capture thread visibile e recuperabile** _(resto di M3)_ —
callback d'errore marshalled sul thread GUI via segnale Qt (stesso pattern di
`_opengl_closed`): avviso all'utente e stato riabilitato. La guardia
`last_device_selected != device` viene rimossa o bypassata quando il thread è
morto, così riselezionare lo stesso device riavvia la cattura.

**F2.3 — `menu_toggle_key` persistito** _(M4)_ — aggiunto a `DEFAULTS`
(`"menu_toggle_key": "F1"`); serialize/deserialize lo trattano come le altre
chiavi (validatore: stringa in `TOGGLE_KEY_NAMES`, altrimenti default).

**F2.4 — Import: envelope validato** _(M5)_ — `import_file` richiede
`raw["app"] == APP_TAG` e `settings` dict; altrimenti `ValueError` con
messaggio chiaro (la GUI lo mostra già). `version` letto ma non bloccante
(tolleranza in avanti invariata).

**F2.5 — «Salva come…» con conferma di sovrascrittura** _(M7)_ — se
`sanitize_name(nome)` è già in `list_names()`, `QMessageBox.question` di
conferma prima di salvare (copre anche le collisioni da sanitizzazione).

**F2.6 — Niente selezione fantasma nella combo profili** _(M8)_ —
`ProfileBarFrame.set_profiles` senza `selected` valido imposta
`setCurrentIndex(-1)`: nessun profilo mostrato → `current_name()` vuoto →
«Salva» ricade su «Salva come…» (comportamento già esistente).

**F2.7 — Soglie verde/giallo senza clamp ordine-dipendente** _(M9)_ — il
renderer memorizza i valori raw ricevuti; il vincolo `green ≤ yellow` è
applicato al momento dell'uso con una funzione pura
(`effective_splits(green, yellow) -> (lo, hi)`) richiamata in `_draw_*`.
L'ordine dei messaggi diventa irrilevante e la funzione è unit-testabile.

**F2.8 — Bloom: fallimento senza leak né retry** _(M10)_ — su
`glCheckFramebufferStatus != COMPLETE`: delete di texture/FBO creati fino a
quel punto e flag `_bloom_unsupported = True` consultato da `_render_bloom`
(un solo tentativo, log una tantum).

**F2.9 — Control queue resiliente** _(M11)_ — ogni `process_control_message`
avvolto in try/except con log dell'errore e del tipo di messaggio: un handler
che fallisce scarta il messaggio, il renderer sopravvive.

**F2.10 — Input del menù ImGui non ambiguo** _(M12)_ — in `key_callback`,
toggle ed ESC vengono ignorati quando `imgui.get_io().want_text_input` è vero
(l'utente sta scrivendo in un campo del menù).

**F2.11 — Colori GUI coerenti** _(M13 + M14)_ — `_hover_brush` usa
`_hover_color_override` quando presente; gli swatch (`bars_solid_color`,
`bars_grad_base`, `bars_grad_top`) sono inizializzati dallo stato `self.*`
(un'unica fonte di verità); `update_color_a`/`update_color_b` sincronizzano i
picker gemelli (con segnali bloccati, per non ri-innescare le callback).

**F2.12 — Decoder video: guardia zero-frame** _(M15)_ — se un ciclo completo
del reader produce 0 frame, il thread termina con un log esplicativo invece di
ricreare il subprocess ffmpeg in loop.

### Fase 3 — Minori (6 commit per area)

- **F3.1 Store profili** _(m1, m2, m3, m4)_: `resolve_bg_ref` calcola il
  default pigramente e dentro try — se anche l'asset del bundle manca ritorna
  `("", False)` e i chiamanti trattano il path vuoto come «nessuno sfondo»
  (niente crash all'avvio); `get_last` valida che il JSON sia un dict;
  `sanitize_name` rifiuta i nomi riservati Windows (suffisso `_`);
  `list_names` espone solo stem stabili al round-trip di sanitizzazione.
- **F3.2 Widget** _(m5, m6, m7, m9, m10)_: `update_base_colors` applica sempre
  il colore (animando verso hover se `underMouse`); `steps` rispettato anche
  per gli slider int; check `is not None` per `warning_trigger_value`;
  `OptionMenuCustomFrame.set_value` su valore sconosciuto logga un warning e
  lascia la selezione invariata (difesa in profondità: dopo F1.3 i valori da
  profilo sono sempre validi); start-dir del file dialog via `ResourceManager`.
- **F3.3 Renderer** _(m11, m12, m14, m15, m16)_: framebuffer-size callback che
  aggiorna viewport/`_fb_*` e invalida `_bloom_size`; maschera bassi del beat
  esclude il bin DC (`_freqs > 0`); fps del video accettato solo se finito
  (`math.isfinite`) e positivo; `imgui.create_context()`/`GlfwRenderer`
  exception-paired (destroy del contesto se il backend solleva); overlay
  allineato alla semantica `None → 1.0` per `bg_alpha`.
- **F3.4 Sfondi GUI** _(m20, m21)_: lo stato (`bg_video_path`,
  `bg_image_path`, `bg_img`) viene mutato **solo dopo** che `Image.open` è
  riuscita, in entrambi i percorsi (apply manuale e da profilo).
- **F3.5 Avvio con errori leggibili** _(m22)_: i probe d'ambiente del
  costruttore (init GLFW per i monitor, enumerazione device audio) avvolti in
  try/except → `QMessageBox.critical` con messaggio chiaro e degradazione
  (lista vuota) invece del traceback senza finestra.
- **F3.6 Igiene test e repo** _(m23, m25)_: fix dell'asserzione di
  `test_menu_keys` (membership in `_NAME_TO_GLFW`); `imgui.ini` aggiunto a
  `.gitignore` e path dell'ini ImGui reindirizzato alla config dir dell'app
  (accanto ai profili), così la CWD resta pulita ovunque.

Il finding **m24** (gap nei test dei profili) non ha un'unità di fix propria:
è risolto dalla suite estesa descritta in «Test e verifica», che copre
esattamente i percorsi elencati nel report.

### Non-obiettivi (documentati, non fixati)

- **m8** — quantizzazione di un tick nel round-trip dei float slider: innocua,
  convergente, il fix (memorizzare il float esatto a lato) non ripaga.
- **m13** — decoder bloccato in I/O oltre il timeout di join: richiede I/O
  patologico; il daemon flag ne limita già l'impatto alla vita del processo.
- **m17** — API GLFW su thread worker: contratto GLFW violato ma funzionante
  sulle piattaforme target (Windows/X11); portare GLFW sul main thread
  stravolgerebbe l'architettura Qt-main/GL-worker. Annotato in CLAUDE.md.
- **m19** — `clear_video` senza sender: il ramo resta (innocuo); un'affordance
  UI «rimuovi video» è una feature, fuori scope di questa bonifica.
- **m18** — contratto di ritorno di `get_window_for_fft` su buffer non pieno:
  documentato nel docstring (nessun chiamante attuale ne è affetto).

## Test e verifica

**Automatici** (`uv run -m pytest`, logica pura, girano anche in WSL2):

- `tests/test_profiles.py` esteso: tipi sbagliati per ogni famiglia di chiavi →
  default; fuori-range → clamp; envelope estraneo → rifiutato; round-trip
  `menu_toggle_key`; file troncato → eccezione dove prevista; `get_last`
  non-dict; nomi riservati Windows; cleanup delle dir temporanee (fixture
  `tmp_path`).
- `tests/test_menu_keys.py`: asserzione di membership (rileva il fallback F1).
- Nuovi test per gli helper puri estratti dai fix: `effective_splits` (F2.7) e
  validazione fps (F3.3).

**Manuali** (checklist su Windows; ogni voce mappa un finding):

1. Avvia/Ferma in rapida successione, più volte → nessun freeze (C1)
2. ESC → riclick immediato su «Avvia» → riavvio pulito (C4)
3. Alt-F4 sulla finestra GL → bottone torna «Avvia» (M1)
4. Cambio device con renderer attivo → barre vive (M2)
5. Import di `package.json` → errore chiaro, nessun profilo creato (M5)
6. Import di profilo con tipi rotti → default al posto degli invalidi (C3)
7. Profilo «last» troncato a mano → app parte coi default + warning (C2)
8. «Salva come…» su nome esistente → conferma richiesta (M7)
9. Digitare il tasto-toggle in un campo del menù ImGui → menù resta aperto (M12)
10. Hover «Ferma» più scuro; swatch «Tinta unica» blu all'avvio (M13, M14)
11. Video corrotto come sfondo → niente respawn ffmpeg, log chiaro (M15)

**Criterio di completamento:** test automatici verdi + checklist manuale
passata + report audit aggiornato con l'esito per finding (fixed/documentato)
+ CLAUDE.md aggiornato per i limiti documentati (m17) e le nuove invarianti
(scritture profilo atomiche, notifica chiusura renderer nel `finally`).
