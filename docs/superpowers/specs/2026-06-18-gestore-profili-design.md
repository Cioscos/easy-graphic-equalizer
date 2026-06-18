# Gestore profili (import/export configurazioni) — Design

- **Data:** 2026-06-18
- **Stato:** approvato (in attesa di review dello spec)
- **Branch:** `feature/gestore-profili`

## Contesto e obiettivo

L'utente deve poter salvare le proprie configurazioni preferite come **profili con nome** e ricaricarle senza reimpostare tutto da zero a ogni avvio, oltre a poterle **condividere** come file. Per gli sfondi (immagine/video) referenziati da un profilo serve un **safe fallback** quando il file non esiste più.

Oggi tutte le impostazioni vivono come attributi `self.*` di `AudioCaptureGUI` (colore, forma, effetti, modalità di visualizzazione, alpha) **più** alcuni valori letti direttamente dagli slider (soglia rumore, bande, DSP avanzato). Esiste già il canale `equalizer_control_queue` (GUI→renderer) con tutti i messaggi `set_*`, quindi applicare un profilo "a caldo" mentre il fullscreen gira è naturale.

## Decisioni di design (fissate in brainstorming)

- **Modello:** gestore profili con **nomi** persistiti su disco (non solo import/export una-tantum).
- **Portata di un profilo:** solo la configurazione **visiva + audio/DSP + sfondo**. **Esclusi**: dispositivo audio (specifico della macchina, scelto all'avvio), monitor fullscreen, tema dell'app.
- **UI:** una **barra profili dedicata in alto**, tra il selettore dispositivo e il corpo nav/pagine: `QComboBox` dei profili + pulsanti *Salva*, *Salva come…*, *Elimina*, *Importa…*, *Esporta…*.
- **Persistenza:** profili in una cartella di config cross-platform; all'avvio si ricarica l'ultimo profilo selezionato.
- **Architettura collect/apply:** metodi **espliciti** sulla GUI + un modulo `profiles.py` puro (senza Qt). Niente registry dichiarativo (refactor troppo invasivo in un codebase senza test), niente replay dei control message (non aggiornerebbe i widget).
- **Niente dirty-tracking** della combo: le modifiche manuali dopo il caricamento non cambiano la selezione; *Salva* sovrascrive il profilo selezionato (YAGNI).

## Componenti

### `profiles.py` (nuovo, top-level, senza Qt — testabile)

- **`ProfileStore`** — gestisce la cartella di config e i file:
  - Cartella di config (dependency-free):
    - Windows: `%APPDATA%/sound-wave` (fallback `~`)
    - macOS: `~/Library/Application Support/sound-wave`
    - Linux/altro: `$XDG_CONFIG_HOME/sound-wave` (fallback `~/.config/sound-wave`)
  - Layout: `<config>/profiles/<nome>.json` (un file per profilo) + `<config>/settings.json` con `{"last_profile": "<nome>"}`.
  - API: `list_names()`, `load(name) -> dict`, `save(name, settings)`, `delete(name)`, `export(name, dest_path)`, `import_file(src_path) -> name`, `get_last()`, `set_last(name)`.
  - Sanitizzazione del nome → nome-file sicuro (caratteri non validi rimossi/sostituiti); collisioni risolte con suffisso numerico.
- **Serializzazione / validazione:**
  - `CURRENT_VERSION = 1`, `DEFAULTS` (mappa chiave→valore di default), `serialize(settings) -> dict`, `deserialize(raw) -> dict`.
  - `deserialize` **tollerante**: chiavi mancanti → default, chiavi sconosciute ignorate, tipi coerciti dove sensato. Un profilo salvato da una versione diversa carica sempre senza crash. Se `raw` non è un JSON valido o manca `settings` → `ValueError` (gestito dalla GUI con avviso).
- **Riferimenti sfondo portabili:**
  - `encode_bg_ref(path) -> dict`: se `path` è dentro `resources/bg` → `{"kind":"resource","name":"bg (19).png"}` (portabile fra macchine e build frozen); altrimenti `{"kind":"path","value":"<path assoluto>"}`.
  - `resolve_bg_ref(ref, resource_manager) -> (resolved_path: str, ok: bool)`: risorsa mancante o path inesistente → ritorna il path dell'immagine **predefinita bundle** (`bg (19).png`) con `ok=False`.

### `gui/profile_bar_frame.py` (nuovo)

`QWidget` con `QHBoxLayout`: etichetta «Profilo:», `QComboBox`, e i cinque pulsanti. Espone callback/segnali verso la GUI (`on_select`, `on_save`, `on_save_as`, `on_delete`, `on_import`, `on_export`) e metodi `set_profiles(names, selected)` / `current_name()`. Pulsanti `QPushButton` standard (stilizzati da qdarktheme), coerenti con le azioni secondarie.

### `gui/slider_frame.py` (modifica minima)

Aggiunta di `set_value(value)` che imposta lo slider con `blockSignals(True/False)` attorno a `setValue` (niente rilancio della `command`), coerente con `OptionMenuCustomFrame.set_value`. Riusa `_to_ticks`/`_refresh_value_label`.

### `gui/main_window_gui.py` (modifiche)

- Nuovo attributo `self.bg_image_path` che traccia il path dell'immagine attiva (default = path di `bg (19).png`); aggiornato in `_apply_image_background`.
- Costruzione e inserimento della profile-bar nel layout (sotto il device selector).
- `_collect_profile() -> dict`, `_apply_profile(settings: dict)`, gli handler dei pulsanti, e il caricamento dell'ultimo profilo all'avvio (dopo la costruzione dei widget).
- `self._profile_store = ProfileStore()`.

### `resource_manager.py` (modifica — punto off-topic 2)

`base_path` diventa `sys._MEIPASS`-aware:
```python
if getattr(sys, "frozen", False):
    self.base_path = Path(sys._MEIPASS) / "resources"
else:
    self.base_path = Path(__file__).parent / "resources"
```
Risoluzione robusta nel frozen build; ne beneficiano anche i riferimenti-risorsa dei profili.

## Modello dati (JSON)

```json
{ "app": "sound-wave", "version": 1, "settings": { ... } }
```

Chiavi di `settings` (valori in **unità native del renderer**), con sorgente GUI e control message di apply:

| Chiave | Tipo | Sorgente (widget / stato) | Control message | Note apply |
|---|---|---|---|---|
| `noise_threshold` | float | `noise_slider` | `set_noise_threshold` | |
| `n_bands` | int | `frequency_slider` | `set_frequency_bands` | |
| `db_floor` | int | `adv_db_floor_slider` | `set_db_floor` | |
| `db_ceiling` | int | `adv_db_ceiling_slider` | `set_db_ceiling` | |
| `attack_tau` | float (s) | `adv_attack_slider` (ms) | `set_attack_tau` (s) | slider = s·1000 |
| `release_tau` | float (s) | `adv_release_slider` (ms) | `set_release_tau` (s) | slider = s·1000 |
| `tilt_db_per_oct` | float | `adv_tilt_slider` | `set_tilt_db_per_oct` | |
| `viz_mode` | int 0–3 | `viz_mode_option` + `self.viz_mode` | `set_viz_mode` | reverse map label↔int + `on_viz_mode_changed` per visibilità |
| `bg_alpha` | float | `alpha_slider` + `self.bg_alpha` | `set_alpha` | |
| `bars_alpha` | float | `bars_alpha_slider` + `self.bars_alpha` | `set_bars_alpha` | |
| `background` | object | filepicker + `self.bg_img`/`bg_image_path`/`bg_video_path` | `set_image` / `set_video` | vedi sotto |
| `color_mode` | int 0–3 | `bars_mode_option` + `self.bars_color_mode` | `set_color_mode` | reverse map + `on_color_mode_changed` |
| `color_a` | [r,g,b] | `bars_solid_color` **e** `bars_grad_base` + `self.bars_color_a` | `set_color_a` | impostare **entrambi** i picker |
| `color_b` | [r,g,b] | `bars_grad_top` + `self.bars_color_b` | `set_color_b` | |
| `green_split` | float | `bars_yellow_thr` + `self.bars_green_split` | `set_green_split` | |
| `yellow_split` | float | `bars_red_thr` + `self.bars_yellow_split` | `set_yellow_split` | |
| `bar_width` | float | `bars_width_slider` + `self.bars_width` | `set_bar_width` | |
| `rounded` | bool | `bars_rounded_option` (No/Sì) + `self.bars_rounded` | `set_rounded` | bool↔label |
| `anchor_center` | bool | `bars_anchor_option` (Dal basso/Centro) + `self.bars_anchor_center` | `set_bar_anchor` | bool↔label |
| `band_symmetric` | bool | `bars_order_option` (Standard/Simmetrico) + `self.bars_band_symmetric` | `set_band_order` | bool↔label |
| `inner_radius` | float | `viz_inner_radius_slider` + `self.viz_inner_radius` | `set_inner_radius` | |
| `line_thickness` | float | `viz_thickness_slider` + `self.viz_line_thickness` | `set_line_thickness` | |
| `fill` | bool | `viz_fill_option` (Linea/Area) + `self.viz_fill` | `set_fill` | bool↔label |
| `osc_mirror` | bool | `viz_osc_mirror_option` (No/Sì) + `self.viz_osc_mirror` | `set_osc_mirror` | bool↔label |
| `osc_smoothing` | float | `viz_osc_smooth_slider` + `self.viz_osc_smoothing` | `set_osc_smoothing` | |
| `peakcap_enabled` | bool | `fx_peakcap_option` (Off/On) + `self.fx_peakcap_enabled` | `set_peakcap_enabled` | bool↔label |
| `peakcap_color` | [r,g,b] | `fx_peakcap_color_picker` + `self.fx_peakcap_color` | `set_peakcap_color` | |
| `peakcap_hold` | float (s) | `fx_peakcap_hold_slider` (ms) + `self.fx_peakcap_hold` (s) | `set_peakcap_hold` (s) | slider = s·1000 |
| `peakcap_fall` | float | `fx_peakcap_fall_slider` + `self.fx_peakcap_fall` | `set_peakcap_fall` | |
| `bloom_enabled` | bool | `fx_bloom_option` + `self.fx_bloom_enabled` | `set_bloom_enabled` | bool↔label |
| `bloom_intensity` | float | `fx_bloom_slider` + `self.fx_bloom_intensity` | `set_bloom_intensity` | |
| `beat_enabled` | bool | `fx_beat_option` + `self.fx_beat_enabled` | `set_beat_enabled` | bool↔label |
| `beat_intensity` | float | `fx_beat_intensity_slider` + `self.fx_beat_intensity` | `set_beat_intensity` | |
| `beat_sensitivity` | float | `fx_beat_sens_slider` + `self.fx_beat_sensitivity` | `set_beat_sensitivity` | |

`background`:
```json
{ "type": "image" | "video", "ref": { "kind": "resource", "name": "bg (19).png" }
                                  /* oppure */ { "kind": "path", "value": "/abs/file.mp4" } }
```
La trasparenza dello sfondo è `bg_alpha` (non duplicata dentro `background`).

**Conversioni di unità (ms↔s):** per `attack_tau`/`release_tau`/`peakcap_hold` lo slider lavora in ms ma il profilo memorizza secondi. *Collect* = `slider.get_value() / 1000`; *apply* = `slider.set_value(s * 1000)` e control message in secondi.

## Flussi

- **Avvio:** `ProfileStore` popola la combo con `list_names()`; se `get_last()` esiste e il file è caricabile → `_apply_profile`; altrimenti si resta sui default attuali e la combo mostra un segnaposto «— Nessun profilo —».
- **Seleziona dalla combo** → `_apply_profile(load(name))` + `set_last(name)`.
- **Salva:** se un profilo è selezionato → `save(selected, _collect_profile())`; altrimenti si comporta come *Salva come…*.
- **Salva come…:** `QInputDialog` per il nome → `save(name, _collect_profile())` → aggiorna combo, seleziona, `set_last`.
- **Elimina:** conferma → `delete(selected)` → aggiorna combo (seleziona il primo rimasto o il segnaposto).
- **Esporta…:** `QFileDialog` (salva) → `export(selected, dest)` (scrive il JSON del profilo). Se nessun profilo selezionato → esporta lo stato corrente (`_collect_profile()`).
- **Importa…:** `QFileDialog` (apri) → `import_file(src)` valida e salva nello store (nome derivato dal filename, con anti-collisione) → seleziona e applica. JSON non valido → `QMessageBox.warning`.

### `_apply_profile(settings)`

Per ogni impostazione, in quest'ordine: (1) imposta il valore del/dei widget (via `set_value`/`set_color`/`setText`, senza rilanciare le `command`); (2) aggiorna lo stato `self.*`; (3) se `self.equalizer_opengl_thread` è attivo, invia il control message corrispondente. Poi richiama esplicitamente `on_color_mode_changed(label)` e `on_viz_mode_changed(label)` per aggiornare la **visibilità condizionale** dei gruppi (le loro callback non scattano sul set programmatico). Lo sfondo è gestito a parte (vedi sotto). Così sia il **prossimo avvio** (che legge widget/`self.*` in `fullscreen_command`) sia il **live update** (control queue) restano corretti.

## Safe fallback sfondo

`_apply_profile`, per `background`:
- Risolve il `ref` con `resolve_bg_ref(ref, self.resource_manager) -> (path, ok)`.
- `type == "image"`: `self.bg_image_path = path`, `self.bg_video_path = None`, `self.bg_img = Image.open(path)`; se il renderer gira → `set_image`. Aggiorna il testo del filepicker.
- `type == "video"`: se il file esiste → `self.bg_video_path = path` e (se gira) `set_video`; se **manca** → niente video di default ⇒ fallback all'immagine predefinita (come sopra) e `ok=False`.
- Se `ok is False` → `QMessageBox.warning` **non bloccante**: «Sfondo non trovato: `<ref>`. Uso lo sfondo predefinito.». Un solo avviso aggregato per applicazione di profilo.

## Punti off-topic (stesso branch/PR)

1. **`PERF_LOGS`**: `opengl_thread.py:26` `True` → `False`.
2. **Resources nel bundle**: lo spec `sound-wave.spec` le include già (`datas.append((resources,'resources'))`); la fix vera è rendere `ResourceManager` `sys._MEIPASS`-aware (sopra).
3. **File morti**: cancellare **`py-to-exe-settings.json`** (non referenziato; "stale" per CLAUDE.md; superato da `sound-wave.spec`). Tutti i moduli `.py` risultano referenziati; `resources/bg/*` **non** si toccano (asset-contenuto intenzionali, anche bersaglio dei riferimenti-risorsa dei profili).

## Testing / verifica

- **Unit test puri** in `tests/` per `profiles.py` (unica parte senza Qt):
  - round-trip `serialize`→`deserialize` preserva i valori;
  - `deserialize` tollerante: chiavi mancanti → default, chiavi sconosciute ignorate, JSON malformato → `ValueError`;
  - `encode_bg_ref` (risorsa vs path) e `resolve_bg_ref` con fallback su file mancante;
  - sanitizzazione nome e anti-collisione.
- **Verifica manuale** eseguendo l'app (`uv run python main.py`): salva → riavvia (auto-load) → carica/elimina/importa/esporta; profilo con sfondo mancante (avviso + fallback); applicazione live con fullscreen aperto.

## File: nuovi / modificati / cancellati

- **Nuovi:** `profiles.py`, `gui/profile_bar_frame.py`, `tests/test_profiles.py`.
- **Modificati:** `gui/main_window_gui.py`, `gui/slider_frame.py`, `resource_manager.py`, `thread/opengl_thread.py` (PERF_LOGS).
- **Cancellati:** `py-to-exe-settings.json`.

## Fuori scope

- Inclusione di tema/monitor/dispositivo nei profili (eventuale estensione futura).
- Dirty-tracking della combo / indicatore "modificato".
- Preset di fabbrica pre-inclusi.
- Migrazione automatica da versioni di profilo > 1 (oggi solo tolleranza in lettura).
