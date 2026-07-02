# Bonifica bug della codebase — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** eliminare i 5 bug critici, i 15 medi e 19 minori individuati dall'audit
2026-07-02 (report: `docs/superpowers/audits/2026-07-02-audit-bug-codebase.md`,
design: `docs/superpowers/specs/2026-07-02-bonifica-bug-audit-design.md`), con
test di regressione per la logica pura.

**Architecture:** l'app è una pipeline producer/consumer: `AudioCaptureThread`
(producer audio) → `queue.Queue` → `EqualizerOpenGLThread` (renderer GLFW/OpenGL
su thread worker), coordinati da `AudioCaptureGUI` (PySide6) tramite una
control-queue di dict `{"type":..., "value":...}`. I fix non cambiano questa
architettura: la rendono robusta (teardown, validazione input, notifiche
cross-thread via segnali Qt).

**Tech Stack:** Python 3.12, PySide6, GLFW/PyOpenGL, imgui-bundle, imageio,
soundcard, numpy. Dipendenze gestite con `uv`.

## Global Constraints

- **Git:** utente `Cioscos`; **MAI aggiungere `Co-Authored-By` o altri trailer** ai commit (regola CLAUDE.md).
- **Branch:** tutto il lavoro su `bugfix/audit-2026-07` (da `master`), un commit per task.
- **Test:** il repo NON usa pytest. I test sono script headless con `main()` che stampa `OK`; si eseguono con `uv run python tests/test_<nome>.py` dalla radice del repo. I nuovi test seguono lo stesso pattern.
- **Ambiente WSL2:** `UV_PROJECT_ENVIRONMENT=.venv-linux` è già settato nel bashrc; `uv run` funziona direttamente. Non toccare `.venv` (Windows).
- **Non toccare:** `sound-wave.spec`, `.github/`, `pyproject.toml`/`uv.lock` (packaging fuori scope).
- **Non importare moduli GL/Qt nei test:** i test devono restare headless (per questo la logica da testare viene estratta in moduli puri).
- I numeri di riga citati si riferiscono al commit `b507f36`; usare sempre il contenuto mostrato (old_string) per localizzare il punto, non il numero di riga.

---

### Task 1: Branch + docs + fix C1 (stop inghiottito durante l'init del renderer)

**Files:**
- Modify: `thread/opengl_thread.py` (righe ~1537-1539)
- Docs: `docs/superpowers/audits/2026-07-02-audit-bug-codebase.md`, `docs/superpowers/specs/2026-07-02-bonifica-bug-audit-design.md`, `docs/superpowers/plans/2026-07-02-bonifica-bug-audit.md` (già scritti, da committare)

**Interfaces:**
- Consumes: `EqualizerOpenGLThread.stop()` setta `self.stop_event` (threading.Event); il main loop esce quando `stop_event.is_set()`.
- Produces: garanzia che uno `stop()` chiamato in QUALSIASI momento dopo `start()` termini il thread.

- [ ] **Step 1: Crea il branch e committa i documenti**

```bash
git checkout -b bugfix/audit-2026-07
git add docs/superpowers/audits/2026-07-02-audit-bug-codebase.md \
        docs/superpowers/specs/2026-07-02-bonifica-bug-audit-design.md \
        docs/superpowers/plans/2026-07-02-bonifica-bug-audit.md
git commit -m "Docs: report audit bug, design e piano della bonifica"
```

- [ ] **Step 2: Rimuovi il clear dello stop event**

In `thread/opengl_thread.py`, dentro `run()`, subito dopo `glfw.set_window_pos(window, monitor_x, monitor_y)`, elimina queste tre righe:

```python
            # check if the stop event is set
            if self.stop_event.is_set():
                self.stop_event.clear()
```

Non va sostituito con nulla: un `threading.Thread` non è riavviabile, quindi non esiste alcun caso in cui l'evento vada ripulito. Con la rimozione, uno stop arrivato durante l'init fa uscire il main loop al primo check di `while not glfw.window_should_close(window) and not self.stop_event.is_set():`.

- [ ] **Step 3: Verifica**

```bash
grep -n "stop_event.clear" thread/opengl_thread.py
```
Expected: nessun output (exit code 1).

```bash
uv run python -c "from thread.opengl_thread import EqualizerOpenGLThread; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Fix critico: stop del renderer non piu' inghiottito durante l'init (C1)"
```

---

### Task 2: F1.2 — Scrittura profili atomica + auto-load protetto (C2)

**Files:**
- Modify: `profiles.py` (`write_envelope` ~179, `set_last` ~202, nuovo helper `_atomic_write_text`)
- Modify: `gui/main_window_gui.py` (auto-load in `__init__`, righe ~187-191; import `QTimer`)
- Test: `tests/test_profiles.py`

**Interfaces:**
- Consumes: `ProfileStore.write_envelope(settings, dest_path)`, `ProfileStore.set_last(name)`, `ProfileStore.load(name)` (firme invariate).
- Produces: `profiles._atomic_write_text(dest: Path, text: str) -> None` (helper interno riusato da entrambe le scritture).

- [ ] **Step 1: Scrivi i test che falliscono**

In `tests/test_profiles.py`, sostituisci `_fresh_store` e aggiungi i nuovi test prima di `main()` (e le nuove chiamate in `main()`):

```python
_TMPDIRS: list = []


def _fresh_store():
    tmp = tempfile.TemporaryDirectory(prefix="swprof_")
    _TMPDIRS.append(tmp)
    return profiles.ProfileStore(config_dir=tmp.name), tmp.name


def test_write_envelope_atomic_no_tmp_leftovers():
    store, tmp = _fresh_store()
    store.save("alfa", dict(profiles.DEFAULTS))
    store.set_last("alfa")
    leftovers = list(Path(tmp).rglob("*.tmp"))
    assert leftovers == [], leftovers
    assert store.load("alfa")["n_bands"] == profiles.DEFAULTS["n_bands"]
    assert store.get_last() == "alfa"


def test_load_truncated_profile_raises():
    store, _ = _fresh_store()
    store.save("alfa", dict(profiles.DEFAULTS))
    path = store._path_for("alfa")
    path.write_text(path.read_text(encoding="utf-8")[:20], encoding="utf-8")
    try:
        store.load("alfa")
    except (ValueError, json.JSONDecodeError):
        return
    raise AssertionError("load doveva fallire su JSON troncato")
```

In `main()` aggiungi le due chiamate e la pulizia finale:

```python
def main():
    try:
        test_serialize_roundtrip_preserves_values()
        test_deserialize_fills_missing_and_ignores_unknown()
        test_deserialize_rejects_non_dict()
        test_sanitize_name()
        test_encode_bg_ref_resource_vs_path()
        test_resolve_bg_ref_ok_and_fallback()
        test_store_save_list_load_delete()
        test_store_last_profile()
        test_store_export_import_roundtrip_and_collision()
        test_store_import_invalid_raises()
        test_write_envelope_atomic_no_tmp_leftovers()
        test_load_truncated_profile_raises()
        print("OK")
    finally:
        for t in _TMPDIRS:
            t.cleanup()
```

Aggiorna anche `test_store_export_import_roundtrip_and_collision`: sostituisci `dest = Path(tempfile.mkdtemp(prefix="swexp_")) / "alfa.json"` con:

```python
    exp = tempfile.TemporaryDirectory(prefix="swexp_")
    _TMPDIRS.append(exp)
    dest = Path(exp.name) / "alfa.json"
```

- [ ] **Step 2: Esegui i test — devono passare già in parte**

```bash
uv run python tests/test_profiles.py
```
Expected: `test_write_envelope_atomic_no_tmp_leftovers` PASSA già (oggi non ci sono file .tmp perché la scrittura è diretta) — è il guardiano della regressione futura; `test_load_truncated_profile_raises` PASSA già (documenta il contratto). L'output deve essere `OK`. Se fallisce qualcosa, fermati e indaga.

- [ ] **Step 3: Implementa la scrittura atomica in `profiles.py`**

Aggiungi l'helper sopra la classe `ProfileStore`:

```python
def _atomic_write_text(dest: Path, text: str) -> None:
    """Scrittura atomica: file temporaneo nella stessa directory + os.replace.

    Un kill/blackout a metà scrittura non può lasciare il file di destinazione
    troncato (os.replace è atomico sullo stesso filesystem)."""
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, dest)
```

Sostituisci il corpo di `write_envelope`:

```python
    def write_envelope(self, settings: dict, dest_path) -> None:
        envelope = {"app": APP_TAG, "version": CURRENT_VERSION, "settings": serialize(settings)}
        _atomic_write_text(
            Path(dest_path), json.dumps(envelope, indent=2, ensure_ascii=False)
        )
```

Sostituisci il corpo di `set_last`:

```python
    def set_last(self, name: str) -> None:
        _atomic_write_text(
            self._settings_file, json.dumps({"last_profile": name}, ensure_ascii=False)
        )
```

- [ ] **Step 4: Proteggi l'auto-load all'avvio in `gui/main_window_gui.py`**

Aggiungi `QTimer` all'import: `from PySide6.QtCore import Qt, QTimer, Signal`.

Sostituisci il blocco in `__init__`:

```python
        # Auto-load dell'ultimo profilo usato (se ancora presente)
        last = self._profile_store.get_last()
        if last and last in self._profile_store.list_names():
            self.profile_bar.set_profiles(self._profile_store.list_names(), last)
            self._apply_profile(self._profile_store.load(last))
```

con:

```python
        # Auto-load dell'ultimo profilo usato (se ancora presente). Protetto:
        # un profilo corrotto su disco non deve impedire l'avvio — l'app parte
        # coi default e segnala il problema senza cancellare il file.
        last = None
        try:
            last = self._profile_store.get_last()
            if last and last in self._profile_store.list_names():
                self.profile_bar.set_profiles(self._profile_store.list_names(), last)
                self._apply_profile(self._profile_store.load(last))
        except Exception as e:
            msg = (f"Impossibile caricare l'ultimo profilo «{last}»: {e}\n"
                   "Avvio con le impostazioni predefinite.")
            QTimer.singleShot(0, lambda: QMessageBox.warning(self, "Profilo", msg))
```

(`QTimer.singleShot(0, ...)`: il box appare appena l'event loop parte, dopo `show()` — mostrarlo dentro `__init__` bloccherebbe la costruzione.)

- [ ] **Step 5: Esegui i test**

```bash
uv run python tests/test_profiles.py
uv run python -c "import gui.main_window_gui; print('OK')"
```
Expected: `OK` da entrambi.

- [ ] **Step 6: Commit**

```bash
git add profiles.py gui/main_window_gui.py tests/test_profiles.py
git commit -m "Fix critico: scrittura profili atomica e auto-load protetto (C2)"
```

---

### Task 3: F1.3 — `deserialize` valida tipi e range; `_apply_profile` nei try (C3, M6)

**Files:**
- Modify: `profiles.py` (nuovi validatori + `deserialize`)
- Modify: `gui/main_window_gui.py` (`_on_profile_selected` ~1301, `_on_profile_import` ~1341)
- Test: `tests/test_profiles.py`

**Interfaces:**
- Consumes: `profiles.DEFAULTS` (tabella chiave→default).
- Produces: `deserialize(raw: dict) -> dict` (firma invariata, ora valida: tipo sbagliato → default, fuori range → clamp). Tabella interna `_VALIDATORS: dict[str, Callable[[value, default], value]]` — Task 8 vi aggiungerà `menu_toggle_key`.

- [ ] **Step 1: Scrivi i test che falliscono**

In `tests/test_profiles.py` aggiungi (e registra in `main()` dopo `test_load_truncated_profile_raises()`):

```python
def test_deserialize_wrong_types_fall_back_to_defaults():
    out = profiles.deserialize({
        "viz_mode": None,          # int atteso
        "n_bands": "x",            # int atteso
        "bg_alpha": "0.5",         # float atteso (stringa non accettata)
        "rounded": "sì",           # bool atteso
        "color_a": [1, 2],         # rgb: servono 3 componenti
        "peakcap_color": "bianco", # rgb atteso
        "background": "non-dict",  # dict atteso
        "attack_tau": float("nan"),  # float non finito
    })
    for key in ("viz_mode", "n_bands", "bg_alpha", "rounded", "color_a",
                "peakcap_color", "background", "attack_tau"):
        assert out[key] == profiles.DEFAULTS[key], (key, out[key])


def test_deserialize_out_of_range_clamped():
    out = profiles.deserialize({
        "bar_width": 5.0,           # range [0.1, 1.0]
        "n_bands": 1000,            # range [1, 100]
        "db_floor": -200,           # range [-90, -25]
        "color_b": [2.0, -1.0, 0.5],
        "beat_sensitivity": 0.0,    # range [1.05, 3.0]
    })
    assert out["bar_width"] == 1.0
    assert out["n_bands"] == 100
    assert out["db_floor"] == -90.0
    assert out["color_b"] == [1.0, 0.0, 0.5]
    assert out["beat_sensitivity"] == 1.05
```

- [ ] **Step 2: Esegui — devono FALLIRE**

```bash
uv run python tests/test_profiles.py
```
Expected: `AssertionError` su `viz_mode` (oggi i valori passano intatti).

- [ ] **Step 3: Implementa i validatori in `profiles.py`**

Aggiungi `import math` in testa (dopo `import json`). Sotto `DEFAULTS` aggiungi:

```python
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _valid_float(lo: float, hi: float):
    def check(v, default):
        return _clamp(float(v), lo, hi) if _is_number(v) else default
    return check


def _valid_int(lo: int, hi: int):
    def check(v, default):
        return int(_clamp(int(round(v)), lo, hi)) if _is_number(v) else default
    return check


def _valid_bool(v, default):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def _valid_rgb(v, default):
    if isinstance(v, (list, tuple)) and len(v) == 3 and all(_is_number(c) for c in v):
        return [_clamp(float(c), 0.0, 1.0) for c in v]
    return default


def _valid_background(v, default):
    if isinstance(v, dict) and v.get("type") in ("image", "video") and isinstance(v.get("ref"), dict):
        return v
    return default


def _valid_str(v, default):
    return v if isinstance(v, str) and v else default


# Validatori per chiave: tipo sbagliato → default, fuori range → clamp.
# I range combaciano con gli slider della GUI (gui/main_window_gui.py) e del
# menù in-window (thread/imgui_overlay.py).
_VALIDATORS = {
    "noise_threshold": _valid_float(0.0, 1.0),
    "n_bands": _valid_int(1, 100),
    "db_floor": _valid_float(-90.0, -25.0),
    "db_ceiling": _valid_float(-20.0, 6.0),
    "attack_tau": _valid_float(0.001, 0.1),
    "release_tau": _valid_float(0.02, 1.0),
    "tilt_db_per_oct": _valid_float(-6.0, 6.0),
    "viz_mode": _valid_int(0, 3),
    "bg_alpha": _valid_float(0.0, 1.0),
    "bars_alpha": _valid_float(0.0, 1.0),
    "background": _valid_background,
    "color_mode": _valid_int(0, 3),
    "color_a": _valid_rgb,
    "color_b": _valid_rgb,
    "green_split": _valid_float(0.0, 1.0),
    "yellow_split": _valid_float(0.0, 1.0),
    "bar_width": _valid_float(0.1, 1.0),
    "rounded": _valid_bool,
    "anchor_center": _valid_bool,
    "band_symmetric": _valid_bool,
    "inner_radius": _valid_float(0.0, 0.8),
    "line_thickness": _valid_float(0.002, 0.06),
    "fill": _valid_bool,
    "osc_mirror": _valid_bool,
    "osc_smoothing": _valid_float(0.0, 1.0),
    "peakcap_enabled": _valid_bool,
    "peakcap_color": _valid_rgb,
    "peakcap_hold": _valid_float(0.0, 2.0),
    "peakcap_fall": _valid_float(0.1, 3.0),
    "bloom_enabled": _valid_bool,
    "bloom_intensity": _valid_float(0.0, 3.0),
    "beat_enabled": _valid_bool,
    "beat_intensity": _valid_float(0.0, 0.3),
    "beat_sensitivity": _valid_float(1.05, 3.0),
}
```

Sostituisci `deserialize`:

```python
def deserialize(raw: dict) -> dict:
    """Riempie le chiavi mancanti con i default, valida tipo e range di quelle
    presenti (tipo sbagliato → default; fuori range → clamp) e ignora le
    sconosciute. Tollerante fra versioni. Solleva ValueError se raw non è dict.
    """
    if not isinstance(raw, dict):
        raise ValueError("settings non è un oggetto JSON")
    out = {}
    for key, default in DEFAULTS.items():
        if key in raw:
            validator = _VALIDATORS.get(key)
            out[key] = validator(raw[key], default) if validator else raw[key]
        else:
            out[key] = default
    return out
```

- [ ] **Step 4: Esegui i test — devono passare tutti**

```bash
uv run python tests/test_profiles.py
```
Expected: `OK` (anche i test pre-esistenti: il roundtrip con valori validi resta identico).

- [ ] **Step 5: Sposta `_apply_profile` nei try (difesa in profondità)**

In `gui/main_window_gui.py`, `_on_profile_selected` diventa:

```python
    def _on_profile_selected(self, name) -> None:
        if not name:
            return
        try:
            settings = self._profile_store.load(name)
            self._apply_profile(settings)
        except Exception as e:
            QMessageBox.warning(self, "Profilo", f"Impossibile caricare il profilo: {e}")
            return
        self._profile_store.set_last(name)
```

`_on_profile_import` diventa (solo il blocco try in fondo cambia):

```python
        try:
            name = self._profile_store.import_file(path)
            self._refresh_profile_bar(selected=name)
            self._apply_profile(self._profile_store.load(name))
        except Exception as e:
            QMessageBox.warning(self, "Importa profilo", f"File non valido: {e}")
            return
        self._profile_store.set_last(name)
```

- [ ] **Step 6: Verifica import e commit**

```bash
uv run python -c "import gui.main_window_gui; print('OK')" && uv run python tests/test_profiles.py
git add profiles.py gui/main_window_gui.py tests/test_profiles.py
git commit -m "Fix critico: deserialize valida tipi e range, apply-profile protetto (C3, M6)"
```

---

### Task 4: F1.4 — Notifica di chiusura del renderer unificata nel finally (C4, M1)

**Files:**
- Modify: `thread/opengl_thread.py` (ramo ESC di `key_callback` ~1723-1733, ramo `except` ~1679-1684, `finally` ~1685-1696)
- Modify: `gui/main_window_gui.py` (`_on_opengl_closed` ~1007-1009, ramo stop di `fullscreen_command` ~945-950, `closeEvent` ~1381-1384)

**Interfaces:**
- Consumes: `self.window_close_callback` (callable settato dalla GUI = `opengl_window_closed`, che emette il segnale Qt `_opengl_closed`).
- Produces: garanzia che `window_close_callback` venga invocato **esattamente una volta, a teardown GLFW completato**, per ogni percorso di uscita (ESC, Alt-F4, stop da GUI, eccezione).

- [ ] **Step 1: ESC non chiama più il callback direttamente**

In `key_callback`, sostituisci:

```python
            # Menù chiuso → comportamento storico: esci dal fullscreen.
            glfw.set_window_should_close(window, True)
            self.stop()
            if self.window_close_callback:
                self.window_close_callback()
```

con:

```python
            # Menù chiuso → comportamento storico: esci dal fullscreen.
            # La notifica alla GUI parte dal finally di run(), a teardown
            # completato: un riavvio non può sovrapporsi a init/terminate GLFW.
            glfw.set_window_should_close(window, True)
            self.stop()
```

- [ ] **Step 2: Il ramo except non chiama più il callback; il finally sì, per ultimo**

Sostituisci la coda di `run()`:

```python
        except Exception as exc:
            # Errore non gestito: notifica la GUI affinché ripristini lo stato
            # (equalizer_opengl_thread = None) invece di lasciare un thread morto.
            print(f"OpenGL thread error: {exc}")
            if self.window_close_callback:
                self.window_close_callback()
        finally:
            # Ferma il decoder video e il subprocess ffmpeg prima di liberare il GL.
            self._stop_video()
            # Chiudi l'overlay ImGui mentre il contesto GL è ancora valido.
            if self._imgui_overlay is not None:
                self._imgui_overlay.shutdown()
                self._imgui_overlay = None
            # Cleanup garantito: risorse GL, finestra e GLFW (sicuro anche su init parziale).
            self._cleanup_gl()
            if window is not None:
                glfw.destroy_window(window)
            glfw.terminate()
```

con:

```python
        except Exception as exc:
            print(f"OpenGL thread error: {exc}")
        finally:
            # Ferma il decoder video e il subprocess ffmpeg prima di liberare il GL.
            self._stop_video()
            # Chiudi l'overlay ImGui mentre il contesto GL è ancora valido.
            if self._imgui_overlay is not None:
                self._imgui_overlay.shutdown()
                self._imgui_overlay = None
            # Cleanup garantito: risorse GL, finestra e GLFW (sicuro anche su init parziale).
            self._cleanup_gl()
            if window is not None:
                glfw.destroy_window(window)
            glfw.terminate()
            # Notifica UNICA alla GUI, a teardown completato: ogni percorso di
            # uscita (ESC, Alt-F4, stop da GUI, eccezione) converge qui. Copre
            # anche la chiusura dal window manager, che prima moriva in silenzio.
            if self.window_close_callback:
                self.window_close_callback()
```

- [ ] **Step 3: Lo slot GUI joina (ormai quasi istantaneo) ed è idempotente**

In `gui/main_window_gui.py`, sostituisci `_on_opengl_closed`:

```python
    def _on_opengl_closed(self):
        # Il callback parte a teardown GLFW completato, quindi il join è quasi
        # istantaneo. Idempotente: se lo stop è partito dalla GUI (fullscreen_
        # command/closeEvent) il riferimento è già None e resta solo il bottone.
        thread = self.equalizer_opengl_thread
        if thread is not None:
            thread.join(timeout=2.0)
            self.equalizer_opengl_thread = None
        self._set_fullscreen_button_running(False)
```

- [ ] **Step 4: Join limitati (niente loop infiniti) nei due siti GUI**

In `fullscreen_command`, sostituisci il ramo stop:

```python
        if self.equalizer_opengl_thread:
            self.equalizer_opengl_thread.stop()
            while self.equalizer_opengl_thread.is_alive():
                self.equalizer_opengl_thread.join(timeout=0.1)
            self.equalizer_opengl_thread = None
            self._set_fullscreen_button_running(False)
```

con:

```python
        if self.equalizer_opengl_thread:
            self.equalizer_opengl_thread.stop()
            # Join limitato: con lo stop non più inghiottibile (Task 1) il
            # teardown è rapido; il limite protegge comunque la GUI.
            self.equalizer_opengl_thread.join(timeout=5.0)
            self.equalizer_opengl_thread = None
            self._set_fullscreen_button_running(False)
```

In `closeEvent`, sostituisci il blocco renderer:

```python
        if self.equalizer_opengl_thread:
            self.equalizer_opengl_thread.stop()
            while self.equalizer_opengl_thread.is_alive():
                self.equalizer_opengl_thread.join(timeout=0.1)
```

con:

```python
        if self.equalizer_opengl_thread:
            self.equalizer_opengl_thread.stop()
            self.equalizer_opengl_thread.join(timeout=5.0)
```

- [ ] **Step 5: Verifica import e commit**

```bash
uv run python -c "from thread.opengl_thread import EqualizerOpenGLThread; import gui.main_window_gui; print('OK')"
git add thread/opengl_thread.py gui/main_window_gui.py
git commit -m "Fix critico: notifica chiusura renderer unificata nel finally (C4, M1)"
```

---

### Task 5: F1.5 — Thread audio daemon, join limitati, recorder nel try (C5)

**Files:**
- Modify: `thread/audioCaptureThread.py`
- Modify: `gui/main_window_gui.py` (`stop_audio_thread` ~664-668, `closeEvent` blocco audio ~1376-1379)

**Interfaces:**
- Consumes: `AudioCaptureThread(audio_queue, device, debug=False, channels=2)` (firma invariata in questo task; Task 7 aggiunge `on_error`).
- Produces: thread daemon → un `record()` bloccato non impedisce mai l'uscita del processo; `stop_audio_thread()` ritorna entro ~2 s in ogni caso.

- [ ] **Step 1: Rendi il thread daemon e proteggi l'apertura del recorder**

In `thread/audioCaptureThread.py`, `__init__`:

```python
    def __init__(self, audio_queue, device, debug=False, channels=2):
        # daemon=True: su Linux/PulseAudio un source "monitor" può non produrre
        # dati durante il silenzio → record() blocca e stop() non può
        # interromperlo. Il daemon flag garantisce che il processo esca comunque.
        threading.Thread.__init__(self, daemon=True)
```

In `run()`, avvolgi l'apertura del recorder (oggi un device scomparso fa uscire l'eccezione da `run()` con traceback grezzo). Il metodo completo, dopo la modifica (cambiano solo il `try` esterno e l'indentazione del `with`; il filtro warning e il while interno sono identici a prima):

```python
    def run(self):
        if self._debug:
            print("Audio capture thread run() called")

        # Le catture in loopback su WASAPI marcano discontinuita in modo
        # fisiologico (riprese dopo il silenzio, glitch del flusso di rendering):
        # soundcard le emette come warning a ogni occorrenza tramite
        # simplefilter('always'). Sono benigni — i dati restano validi — quindi
        # silenziamo solo questo messaggio. Va registrato qui (a thread avviato,
        # dopo l'import di soundcard) perche' i filtri inseriti per ultimi hanno
        # precedenza: cosi' il nostro 'ignore' vince sull''always' di soundcard.
        # Filtro per messaggio (non per categoria) cosi' altri SoundcardRuntimeWarning
        # restano visibili, ed e' cross-platform (niente import della classe).
        warnings.filterwarnings("ignore", message="data discontinuity in recording")

        try:
            with self.device.recorder(samplerate=RATE, channels=self.channels) as recorder:
                while self._running:
                    try:
                        if self._debug:
                            print("Capturing audio data...")
                        data = recorder.record(numframes=CHUNK)
                        if self._debug:
                            print(f"Captured {len(data)} samples of audio data")
                        # Put non bloccante: se la coda e' piena (consumer momentaneamente
                        # fermo, es. init OpenGL) scartiamo il frame piu' vecchio e inseriamo
                        # il piu' recente, invece di bloccare record() — un blocco starverebbe
                        # il buffer WASAPI e fabbricherebbe discontinuita' reali. Coerente con
                        # i consumer che tengono comunque solo l'ultimo frame.
                        try:
                            self.audio_queue.put_nowait(data)
                        except queue.Full:
                            try:
                                self.audio_queue.get_nowait()   # scarta il frame stantio
                            except queue.Empty:
                                pass
                            try:
                                self.audio_queue.put_nowait(data)
                            except queue.Full:
                                pass
                    except Exception as exception:
                        print(f"Error capturing audio: {exception}")
                        break
        except Exception as exception:
            # recorder() puo' sollevare gia' all'apertura (device scomparso):
            # senza questo try l'eccezione uscirebbe da run() con traceback grezzo.
            print(f"Error opening audio device: {exception}")

        if self._debug:
            print("Loop exited")
```

- [ ] **Step 2: Join limitati nella GUI**

`stop_audio_thread` diventa:

```python
    def stop_audio_thread(self):
        if self.audio_thread:
            self.audio_thread.stop()
            # Join limitato: un record() bloccato (Linux, silenzio di sistema)
            # non deve congelare la GUI. Il thread è daemon: se non esce qui,
            # muore comunque col processo.
            self.audio_thread.join(timeout=2.0)
            self.audio_thread = None
```

In `closeEvent`, sostituisci il blocco audio:

```python
        if self.audio_thread:
            self.audio_thread.stop()
            while self.audio_thread.is_alive():
                self.audio_thread.join(timeout=0.1)
```

con:

```python
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.join(timeout=2.0)
```

- [ ] **Step 3: Verifica import e commit**

```bash
uv run python -c "from thread.audioCaptureThread import AudioCaptureThread; import gui.main_window_gui; print('OK')"
git add thread/audioCaptureThread.py gui/main_window_gui.py
git commit -m "Fix critico: thread audio daemon e join limitati (C5)"
```

---

### Task 6: F2.1 — Cambio device senza orfanare il renderer (M2)

**Files:**
- Modify: `gui/main_window_gui.py` (`on_device_selected` ~650-657)

**Interfaces:**
- Consumes: `self.audio_queue` è il riferimento che `EqualizerOpenGLThread` riceve alla costruzione e tiene per sempre.
- Produces: `self.audio_queue` non viene MAI ricreata dopo la prima costruzione: il renderer resta agganciato alla coda giusta.

- [ ] **Step 1: Riusa la stessa coda**

Sostituisci in `on_device_selected`:

```python
        if self.last_device_selected != device:
            self.audio_queue = None
            self.stop_audio_thread()

            self.audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
            self.audio_thread = AudioCaptureThread(self.audio_queue, device=device)
            self.audio_thread.start()
            self.last_device_selected = device
```

con:

```python
        if self.last_device_selected != device:
            self.stop_audio_thread()

            # Riusa la STESSA coda: il renderer in esecuzione ne tiene il
            # riferimento ricevuto alla costruzione — ricrearla lo lascerebbe
            # per sempre su una coda morta (barre congelate). Drena il residuo
            # del vecchio device.
            if self.audio_queue is None:
                self.audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
            else:
                while True:
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        break
            self.audio_thread = AudioCaptureThread(self.audio_queue, device=device)
            self.audio_thread.start()
            self.last_device_selected = device
```

- [ ] **Step 2: Verifica import e commit**

```bash
uv run python -c "import gui.main_window_gui; print('OK')"
git add gui/main_window_gui.py
git commit -m "Fix: cambio device senza orfanare il renderer sulla coda vecchia (M2)"
```

---

### Task 7: F2.2 — Morte del capture thread visibile e recuperabile (M3)

**Files:**
- Modify: `thread/audioCaptureThread.py` (nuovo param `on_error`)
- Modify: `gui/main_window_gui.py` (nuovo segnale `_audio_error`, slot `_on_audio_error`, passaggio `on_error`)

**Interfaces:**
- Produces: `AudioCaptureThread(audio_queue, device, debug=False, channels=2, on_error: Optional[Callable[[str], None]] = None)`. `on_error(message)` è invocato SUL THREAD DI CATTURA quando la cattura muore per errore (mai su stop volontario) — chi lo riceve deve marshallare (la GUI usa un segnale Qt).

- [ ] **Step 1: Callback d'errore nel thread di cattura**

In `thread/audioCaptureThread.py`:

```python
    def __init__(self, audio_queue, device, debug=False, channels=2, on_error=None):
        ...
        self._on_error = on_error
```

Aggiungi il metodo:

```python
    def _notify_error(self, message: str) -> None:
        # Solo per morti impreviste: uno stop volontario non è un errore.
        if self._running and self._on_error is not None:
            try:
                self._on_error(message)
            except Exception:
                pass
```

Nel ramo `except` interno al while (quello con `break`), prima del `break`:

```python
                except Exception as exception:
                    print(f"Error capturing audio: {exception}")
                    self._notify_error(str(exception))
                    break
```

Nel ramo `except` esterno aggiunto al Task 5:

```python
        except Exception as exception:
            print(f"Error opening audio device: {exception}")
            self._notify_error(str(exception))
```

- [ ] **Step 2: Segnale e slot nella GUI**

In `AudioCaptureGUI`, accanto agli altri segnali:

```python
    _opengl_closed = Signal()
    _setting_changed = Signal(dict)
    _audio_error = Signal(str)
```

In `__init__`, accanto a `self._opengl_closed.connect(...)`:

```python
        self._audio_error.connect(self._on_audio_error)
```

Nuovo slot (mettilo accanto a `_on_opengl_closed`):

```python
    def _on_audio_error(self, message: str) -> None:
        """Slot (thread GUI): la cattura audio è morta. Azzera lo stato così
        riselezionare lo STESSO device fa ripartire la cattura (senza questo
        reset la guardia last_device_selected renderebbe il click un no-op)."""
        self.audio_thread = None
        self.last_device_selected = None
        QMessageBox.warning(
            self, "Audio",
            f"La cattura audio si è interrotta: {message}\n"
            "Riseleziona il dispositivo per riprovare.",
        )
```

In `on_device_selected`, passa il callback alla costruzione:

```python
            self.audio_thread = AudioCaptureThread(
                self.audio_queue, device=device,
                on_error=lambda msg: self._audio_error.emit(msg),
            )
```

- [ ] **Step 3: Verifica import e commit**

```bash
uv run python -c "import gui.main_window_gui; print('OK')"
git add thread/audioCaptureThread.py gui/main_window_gui.py
git commit -m "Fix: errore di cattura audio visibile e recuperabile dalla GUI (M3)"
```

---

### Task 8: F2.3 — `menu_toggle_key` persistito nei profili (M4)

**Files:**
- Modify: `profiles.py` (`DEFAULTS` + `_VALIDATORS`)
- Test: `tests/test_profiles.py`

**Interfaces:**
- Consumes: `_VALIDATORS` e `_valid_str` dal Task 3.
- Produces: la chiave `"menu_toggle_key"` fa parte dello schema profilo (default `"F1"`).

- [ ] **Step 1: Test che fallisce**

```python
def test_menu_toggle_key_roundtrip():
    s = dict(profiles.DEFAULTS)
    s["menu_toggle_key"] = "M"
    out = profiles.deserialize(profiles.serialize(s))
    assert out["menu_toggle_key"] == "M", out.get("menu_toggle_key")
    # tipo sbagliato → default
    assert profiles.deserialize({"menu_toggle_key": 42})["menu_toggle_key"] == "F1"
```

Registralo in `main()`.

```bash
uv run python tests/test_profiles.py
```
Expected: FAIL — `serialize` oggi scarta la chiave (`out.get(...)` = `None`... l'assert fallisce con KeyError/None).

- [ ] **Step 2: Implementa**

In `profiles.py`, dentro `DEFAULTS`, dopo `"bars_alpha": 1.0,`:

```python
    "menu_toggle_key": "F1",
```

Dentro `_VALIDATORS`, dopo `"bars_alpha": ...,`:

```python
    "menu_toggle_key": _valid_str,
```

(La validazione di appartenenza a `TOGGLE_KEY_NAMES` resta al renderer — `toggle_key_to_glfw` ha già il fallback F1 — perché `profiles.py` deve restare puro, senza importare glfw.)

- [ ] **Step 3: Esegui e committa**

```bash
uv run python tests/test_profiles.py
```
Expected: `OK`.

```bash
git add profiles.py tests/test_profiles.py
git commit -m "Fix: menu_toggle_key persistito nei profili (M4)"
```

---

### Task 9: F2.4 — Import: envelope validato (M5)

**Files:**
- Modify: `profiles.py` (`import_file` ~183-193)
- Test: `tests/test_profiles.py`

**Interfaces:**
- Produces: `import_file(src_path) -> str` solleva `ValueError` per file JSON che non sono profili Sound Wave (manca `app == "sound-wave"` o `settings` non-dict). Nessun profilo viene creato in quel caso.

- [ ] **Step 1: Test che fallisce**

```python
def test_store_import_foreign_json_rejected():
    store, tmp = _fresh_store()
    foreign = Path(tmp) / "package.json"
    foreign.write_text('{"name": "x", "version": "1.0.0"}', encoding="utf-8")
    try:
        store.import_file(foreign)
    except ValueError:
        assert store.list_names() == []   # niente profilo fantasma
        return
    raise AssertionError("import_file doveva rifiutare un JSON estraneo")
```

Registralo in `main()`. Expected al run: FAIL (`import_file` oggi crea un profilo tutto-default).

- [ ] **Step 2: Implementa**

Sostituisci l'inizio di `import_file`:

```python
    def import_file(self, src_path) -> str:
        raw = json.loads(Path(src_path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("app") != APP_TAG:
            raise ValueError("il file non è un profilo Sound Wave")
        if not isinstance(raw.get("settings"), dict):
            raise ValueError("profilo senza sezione settings valida")
        settings = deserialize(raw["settings"])
```

(il resto — anti-collisione nome e `save` — resta invariato. `version` viene letto ma non bloccato: tolleranza in avanti come da design.)

- [ ] **Step 3: Esegui e committa**

```bash
uv run python tests/test_profiles.py
```
Expected: `OK` (il test `test_store_export_import_roundtrip_and_collision` continua a passare: l'export scrive l'envelope completo con `app`).

```bash
git add profiles.py tests/test_profiles.py
git commit -m "Fix: import profili valida l'envelope Sound Wave (M5)"
```

---

### Task 10: F2.5 — «Salva come…» con conferma di sovrascrittura (M7)

**Files:**
- Modify: `gui/main_window_gui.py` (`_on_profile_save_as` ~1320-1327)

- [ ] **Step 1: Implementa**

Sostituisci `_on_profile_save_as`:

```python
    def _on_profile_save_as(self) -> None:
        name, accepted = QInputDialog.getText(self, "Salva profilo", "Nome del profilo:")
        if not accepted or not name.strip():
            return
        saved = profiles.sanitize_name(name)
        # Il confronto va fatto sul nome SANITIZZATO: nomi diversi possono
        # collidere sullo stesso file (es. "a/b" → "a_b").
        if saved in self._profile_store.list_names():
            confirm = QMessageBox.question(
                self, "Salva profilo",
                f"Il profilo «{saved}» esiste già. Sovrascriverlo?",
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
        self._profile_store.save(name, self._collect_profile())
        self._refresh_profile_bar(selected=saved)
        self._profile_store.set_last(saved)
```

- [ ] **Step 2: Verifica import e commit**

```bash
uv run python -c "import gui.main_window_gui; print('OK')"
git add gui/main_window_gui.py
git commit -m "Fix: conferma sovrascrittura in Salva come... (M7)"
```

---

### Task 11: F2.6 — Nessuna selezione fantasma nella combo profili (M8)

**Files:**
- Modify: `gui/profile_bar_frame.py` (`set_profiles` ~58-68)

**Interfaces:**
- Produces: `set_profiles(names, selected)` senza `selected` valido lascia la combo a indice -1 (testo vuoto) → `current_name()` ritorna `None` → «Salva» ricade su «Salva come…» (comportamento già esistente in `_on_profile_save`).

- [ ] **Step 1: Implementa**

Sostituisci `set_profiles`:

```python
    def set_profiles(self, names: List[str], selected: Optional[str]) -> None:
        """Ripopola la combo. Blocca i segnali: NON scatena on_select. Senza un
        `selected` valido nessuna voce risulta scelta (indice -1): il profilo
        mostrato è sempre e solo uno realmente caricato — mai un profilo
        "fantasma" che «Salva» sovrascriverebbe con impostazioni non sue."""
        self._combo.blockSignals(True)
        self._combo.clear()
        if names:
            self._combo.addItems(names)
            if selected and selected in names:
                self._combo.setCurrentText(selected)
            else:
                self._combo.setCurrentIndex(-1)
        else:
            self._combo.addItem(PLACEHOLDER)
        self._combo.blockSignals(False)
```

- [ ] **Step 2: Verifica import e commit**

```bash
uv run python -c "import gui.profile_bar_frame; print('OK')"
git add gui/profile_bar_frame.py
git commit -m "Fix: nessuna selezione fantasma nella combo profili (M8)"
```

---

### Task 12: F2.7 — Soglie verde/giallo senza clamp ordine-dipendente (M9)

**Files:**
- Create: `thread/render_params.py`
- Modify: `thread/opengl_thread.py` (handler `set_green_split`/`set_yellow_split` ~1325-1333; uso uniform in `_draw_bars_instanced` ~1009-1010 e nel draw dello strip ~953-954; import)
- Test (create): `tests/test_render_params.py`

**Interfaces:**
- Produces: `thread.render_params.effective_splits(green: float, yellow: float) -> tuple[float, float]` — clampa entrambi in [0,1] e li ritorna ordinati `(lo, hi)`. Modulo puro (nessun import GL) così il test resta headless.

- [ ] **Step 1: Scrivi il test che fallisce (modulo inesistente)**

Crea `tests/test_render_params.py`:

```python
"""
Verifica headless (niente pytest nel repo): asserzioni pure su render_params.
Eseguire dalla radice del repo con:  uv run python tests/test_render_params.py
Uscita 0 + "OK" = passato.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thread.render_params import effective_splits


def test_order_independent():
    assert effective_splits(0.5, 0.8) == (0.5, 0.8)
    # green oltre yellow: il vincolo si applica all'uso, ordinando — non
    # azzoppando uno dei due in base all'ordine di arrivo dei messaggi
    assert effective_splits(0.85, 0.3) == (0.3, 0.85)


def test_clamped():
    assert effective_splits(-1.0, 2.0) == (0.0, 1.0)
    assert effective_splits(1.5, 0.2) == (0.2, 1.0)


def main():
    test_order_independent()
    test_clamped()
    print("OK")


if __name__ == "__main__":
    main()
```

```bash
uv run python tests/test_render_params.py
```
Expected: FAIL con `ModuleNotFoundError: No module named 'thread.render_params'`.

- [ ] **Step 2: Crea il modulo**

Crea `thread/render_params.py`:

```python
"""Helper puri per i parametri di rendering (testabili senza contesto GL)."""


def effective_splits(green: float, yellow: float) -> tuple[float, float]:
    """Soglie effettive (verde→giallo, giallo→rosso) del color-mode classico.

    I due valori arrivano da messaggi di controllo indipendenti: il vincolo
    green <= yellow va applicato all'USO, non alla ricezione, altrimenti il
    risultato dipende dall'ordine dei messaggi (caricare un profilo poteva
    lasciare il renderer con una soglia diversa da quella mostrata in GUI).
    Clampa entrambi in [0, 1] e li ritorna ordinati.
    """
    g = min(max(float(green), 0.0), 1.0)
    y = min(max(float(yellow), 0.0), 1.0)
    return (g, y) if g <= y else (y, g)
```

```bash
uv run python tests/test_render_params.py
```
Expected: `OK`.

- [ ] **Step 3: Usa l'helper nel renderer**

In `thread/opengl_thread.py`, aggiungi all'import block (dopo `from thread.menu_keys import toggle_key_to_glfw`):

```python
from thread.render_params import effective_splits
```

Sostituisci i due handler:

```python
        elif message_type == 'set_green_split':
            # green_split è la soglia inferiore: clamp [0,1] e <= yellow_split.
            v = min(max(float(message['value']), 0.0), 1.0)
            self._green_split = min(v, self._yellow_split)

        elif message_type == 'set_yellow_split':
            # yellow_split è la soglia superiore: clamp [0,1] e >= green_split.
            v = min(max(float(message['value']), 0.0), 1.0)
            self._yellow_split = max(v, self._green_split)
```

con:

```python
        elif message_type == 'set_green_split':
            # Valore raw (solo clamp [0,1]): il vincolo green <= yellow è
            # applicato all'uso (effective_splits), così l'ordine dei
            # messaggi è irrilevante.
            self._green_split = min(max(float(message['value']), 0.0), 1.0)

        elif message_type == 'set_yellow_split':
            self._yellow_split = min(max(float(message['value']), 0.0), 1.0)
```

In `_draw_bars_instanced`, sostituisci:

```python
        glUniform1f(self._bars_uniforms["uGreenSplit"], self._green_split)
        glUniform1f(self._bars_uniforms["uYellowSplit"], self._yellow_split)
```

con:

```python
        g_split, y_split = effective_splits(self._green_split, self._yellow_split)
        glUniform1f(self._bars_uniforms["uGreenSplit"], g_split)
        glUniform1f(self._bars_uniforms["uYellowSplit"], y_split)
```

Stessa sostituzione nel punto di draw dello strip (cerca le altre due righe `self._strip_uniforms["uGreenSplit"]` / `["uYellowSplit"]`, ~riga 953):

```python
        g_split, y_split = effective_splits(self._green_split, self._yellow_split)
        glUniform1f(self._strip_uniforms["uGreenSplit"], g_split)
        glUniform1f(self._strip_uniforms["uYellowSplit"], y_split)
```

- [ ] **Step 4: Esegui e committa**

```bash
uv run python tests/test_render_params.py && uv run python -c "from thread.opengl_thread import EqualizerOpenGLThread; print('OK')"
git add thread/render_params.py thread/opengl_thread.py tests/test_render_params.py
git commit -m "Fix: soglie verde/giallo senza clamp ordine-dipendente (M9)"
```

---

### Task 13: F2.8 — Bloom: fallimento senza leak né retry (M10)

**Files:**
- Modify: `thread/opengl_thread.py` (`__init__` sezione bloom ~594-596, `_ensure_bloom_fbos` ~972-1000)

- [ ] **Step 1: Aggiungi il flag di stato**

In `__init__`, dove sono inizializzati `self._bloom_fbos = []` / `self._bloom_texs = []` / `self._bloom_size = None`, aggiungi:

```python
        self._bloom_unsupported = False  # True se il driver rifiuta l'FBO RGB16F: non ritentare
```

- [ ] **Step 2: Ripulisci sul fallimento e non ritentare**

Sostituisci `_ensure_bloom_fbos`:

```python
    def _ensure_bloom_fbos(self) -> bool:
        """
        Crea (lazy) i due FBO ping-pong half-res (GL_RGB16F) per il bloom.
        Ritorna True se disponibili. Richiede dimensione framebuffer nota.
        Su framebuffer incompleto (GL_RGB16F non è color-renderable obbligatorio
        in GL 3.3) libera il creato e marca il bloom come non disponibile: senza
        il flag il tentativo — e il leak — si ripeterebbe a ogni frame.
        """
        if self._bloom_unsupported:
            return False
        if self._bloom_size is not None:
            return True
        w = max(1, int(self._fb_width) // 2)
        h = max(1, int(self._fb_height) // 2)
        fbos, texs = [], []
        for _ in range(2):
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, w, h, 0, GL_RGB, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                all_fbos, all_texs = fbos + [fbo], texs + [tex]
                glDeleteFramebuffers(len(all_fbos), all_fbos)
                glDeleteTextures(len(all_texs), all_texs)
                self._bloom_unsupported = True
                print("[GL] Bloom non disponibile: framebuffer RGB16F incompleto")
                return False
            fbos.append(fbo)
            texs.append(tex)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._bloom_fbos, self._bloom_texs, self._bloom_size = fbos, texs, (w, h)
        return True
```

- [ ] **Step 3: Verifica import e commit**

```bash
uv run python -c "from thread.opengl_thread import EqualizerOpenGLThread; print('OK')"
git add thread/opengl_thread.py
git commit -m "Fix: bloom senza leak ne' retry su framebuffer incompleto (M10)"
```

---

### Task 14: F2.9 — Control queue resiliente alle eccezioni (M11)

**Files:**
- Modify: `thread/opengl_thread.py` (drain della control queue nel main loop, ~1614-1617)

- [ ] **Step 1: Implementa**

Sostituisci nel main loop:

```python
                # Process messages from the control queue
                while not self._control_queue.empty():
                    message = self._control_queue.get(block=False)
                    self.process_control_message(message)
```

con:

```python
                # Process messages from the control queue
                while not self._control_queue.empty():
                    message = self._control_queue.get(block=False)
                    try:
                        self.process_control_message(message)
                    except Exception as exc:
                        # Un messaggio malformato (es. set_image con immagine
                        # corrotta: PIL decodifica lazy e il file esplode QUI,
                        # non alla Image.open della GUI) non deve uccidere il
                        # renderer: scarta il messaggio e continua.
                        mtype = message.get('type') if isinstance(message, dict) else repr(message)
                        print(f"Errore nel messaggio di controllo {mtype}: {exc}")
```

- [ ] **Step 2: Verifica import e commit**

```bash
uv run python -c "from thread.opengl_thread import EqualizerOpenGLThread; print('OK')"
git add thread/opengl_thread.py
git commit -m "Fix: control queue resiliente alle eccezioni degli handler (M11)"
```

---

### Task 15: F2.10 — Toggle/ESC ignorati mentre ImGui ha l'input testuale (M12)

**Files:**
- Modify: `thread/opengl_thread.py` (import + `key_callback` ~1704-1733)

- [ ] **Step 1: Implementa**

Aggiungi all'import block di `thread/opengl_thread.py`:

```python
from imgui_bundle import imgui
```

In `key_callback`, dopo il blocco `if action != glfw.PRESS: return`, inserisci:

```python
        # Se ImGui sta usando la tastiera per un campo di testo (es. Ctrl+click
        # su uno slider del menù), toggle ed ESC appartengono a lui: gestirli
        # qui chiuderebbe il menù a metà editing.
        if self._imgui_overlay is not None and imgui.get_io().want_text_input:
            return
```

(`want_text_input` e non `want_capture_keyboard`: il secondo è vero ogni volta che il menù è aperto sotto il mouse, e renderebbe impossibile chiudere il menù col toggle. Il gate serve solo durante l'editing testuale.)

- [ ] **Step 2: Verifica import e commit**

```bash
uv run python -c "from thread.opengl_thread import EqualizerOpenGLThread; print('OK')"
git add thread/opengl_thread.py
git commit -m "Fix: toggle/ESC ignorati durante l'editing testuale ImGui (M12)"
```

---

### Task 16: F2.11 — Hover color rispettato e swatch allineati allo stato (M13, M14, m5)

**Files:**
- Modify: `gui/animated_button.py` (`_hover_brush` ~109-110, `update_base_colors` ~208-215)
- Modify: `gui/main_window_gui.py` (costruzione swatch ~480-495, `update_color_a` ~793-796; import `rgb01_to_hex`)

**Interfaces:**
- Consumes: `ColorPickerFrame.set_color(rgb)` non invoca la callback (verificato); `rgb01_to_hex(rgb: tuple[float,float,float]) -> str` da `gui/color_picker_frame.py`.

- [ ] **Step 1: `AnimatedButton` usa l'override e applica il colore anche in hover**

Sostituisci `_hover_brush`:

```python
    def _hover_brush(self) -> QColor:
        if self._hover_color_override is not None:
            return QColor(self._hover_color_override)
        return self._base_color.lighter(115)
```

Sostituisci `update_base_colors`:

```python
    def update_base_colors(self, fg_color: str, hover_color: Optional[str] = None) -> None:
        """Aggiorna i colori base del pulsante mantenendo le animazioni."""
        self._base_color = QColor(fg_color)
        self._hover_color_override = QColor(hover_color) if hover_color else None
        # Applica SUBITO il nuovo colore: verso l'hover se il mouse è sopra
        # (caso tipico: il click sul footer che alterna Avvia/Ferma avviene
        # col mouse sul bottone), altrimenti verso il base.
        self._color_anim.stop()
        self._set_bg(QColor(self._hover_brush() if self.underMouse() else self._base_color))
```

- [ ] **Step 2: Swatch inizializzati dallo stato + sync dei gemelli**

In `gui/main_window_gui.py` aggiorna l'import:

```python
from gui.color_picker_frame import ColorPickerFrame, rgb01_to_hex
```

Nella costruzione dei picker, sostituisci:

```python
        # Tinta unica: un colore
        self.bars_solid_color = ColorPickerFrame(
            header_name='Colore barre:',
            initial_color='#22d36a',
            command=self.update_color_a,
        )
        # Gradiente: due colori
        self.bars_grad_base = ColorPickerFrame(
            header_name='Colore base:',
            initial_color='#3b6bff',
            command=self.update_color_a,
        )
        self.bars_grad_top = ColorPickerFrame(
            header_name='Colore cima:',
            initial_color='#ff3bd0',
            command=self.update_color_b,
        )
```

con:

```python
        # Tinta unica e Gradiente condividono color_a: gli swatch partono
        # dallo stato self.* (unica fonte di verità), non da esadecimali fissi.
        self.bars_solid_color = ColorPickerFrame(
            header_name='Colore barre:',
            initial_color=rgb01_to_hex(self.bars_color_a),
            command=self.update_color_a,
        )
        self.bars_grad_base = ColorPickerFrame(
            header_name='Colore base:',
            initial_color=rgb01_to_hex(self.bars_color_a),
            command=self.update_color_a,
        )
        self.bars_grad_top = ColorPickerFrame(
            header_name='Colore cima:',
            initial_color=rgb01_to_hex(self.bars_color_b),
            command=self.update_color_b,
        )
```

Sostituisci `update_color_a`:

```python
    def update_color_a(self, rgb):
        self.bars_color_a = rgb
        # Tinta unica e Gradiente condividono color_a: tieni allineati entrambi
        # gli swatch (set_color non riscatena la callback → niente loop).
        self.bars_solid_color.set_color(rgb)
        self.bars_grad_base.set_color(rgb)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_color_a", "value": rgb})
```

- [ ] **Step 3: Verifica import e commit**

```bash
uv run python -c "import gui.main_window_gui, gui.animated_button; print('OK')"
git add gui/animated_button.py gui/main_window_gui.py
git commit -m "Fix: hover color rispettato e swatch colore allineati allo stato (M13, M14, m5)"
```

---

### Task 17: F2.12 — Decoder video: guardia zero-frame (M15)

**Files:**
- Modify: `thread/video_decode_thread.py` (`run()` ~47-71)

- [ ] **Step 1: Implementa**

Nel `while` esterno di `run()`, aggiungi il contatore e la guardia:

```python
            while not self.stop_event.is_set():
                reader = imageio.get_reader(self._path)
                produced = 0
                try:
                    meta = reader.get_meta_data()
                    fps = meta.get('fps')
                    if fps and fps > 0:
                        self.fps = float(fps)
                    if self._debug:
                        print(f"Video fps={self.fps}, meta={meta}")

                    for frame in reader:
                        if self.stop_event.is_set():
                            break
                        # imageio/ffmpeg restituisce già RGB uint8; garantiamo
                        # contiguità per un upload GL diretto (glTexImage2D).
                        if frame.dtype != np.uint8 or not frame.flags["C_CONTIGUOUS"]:
                            frame = np.ascontiguousarray(frame, dtype=np.uint8)
                        self._put_frame(frame)
                        produced += 1
                finally:
                    try:
                        reader.close()
                    except Exception:
                        pass
                if produced == 0:
                    # Header leggibile ma zero frame decodificati: senza questa
                    # guardia il while ricreerebbe reader e subprocess ffmpeg in
                    # loop stretto per tutta la vita dello sfondo video.
                    print(f"Video senza frame decodificabili, decoder fermato: {self._path}")
                    break
                # Fine stream: il while esterno ricrea il reader e riparte (loop).
```

- [ ] **Step 2: Verifica import e commit**

```bash
uv run python -c "from thread.video_decode_thread import VideoDecodeThread; print('OK')"
git add thread/video_decode_thread.py
git commit -m "Fix: decoder video senza respawn ffmpeg su stream vuoto (M15)"
```

---

### Task 18: F3.1 — Robustezza store profili (m1, m2, m3, m4)

**Files:**
- Modify: `profiles.py` (`resolve_bg_ref` ~111-131, `get_last` ~195-200, `sanitize_name` ~85-88, `list_names` ~161-162)
- Test: `tests/test_profiles.py`

**Interfaces:**
- Produces: `resolve_bg_ref` può ora ritornare `("", False)` quando manca anche l'asset predefinito del bundle — i chiamanti (Task 21) trattano il path vuoto come «nessuno sfondo».

- [ ] **Step 1: Test che falliscono**

Aggiungi a `tests/test_profiles.py` (e registra in `main()`):

```python
def test_get_last_non_dict_settings_returns_none():
    store, tmp = _fresh_store()
    (Path(tmp) / "settings.json").write_text('["x"]', encoding="utf-8")
    assert store.get_last() is None


def test_sanitize_name_windows_reserved():
    assert profiles.sanitize_name("CON") == "CON_"
    assert profiles.sanitize_name("con") == "con_"
    assert profiles.sanitize_name("COM1") == "COM1_"
    assert profiles.sanitize_name("Console") == "Console"  # non riservato


def test_list_names_skips_unresolvable_stems():
    store, tmp = _fresh_store()
    store.save("buono", dict(profiles.DEFAULTS))
    # File creato esternamente con un carattere fuori whitelist: verrebbe
    # listato ma _path_for lo risanitizzerebbe su un file diverso.
    (Path(tmp) / "profiles" / "foo?bar.json").write_text("{}", encoding="utf-8")
    assert store.list_names() == ["buono"]
```

Expected al run: FAIL (`get_last` → `AttributeError`; `sanitize_name("CON")` → `"CON"`; `list_names` include `foo?bar`).
Nota: `test_list_names_skips_unresolvable_stems` su Windows non può creare `foo?bar.json` — i test girano in WSL2, dove il nome è legale.

- [ ] **Step 2: Implementa in `profiles.py`**

`get_last`:

```python
    def get_last(self) -> Optional[str]:
        try:
            data = json.loads(self._settings_file.read_text(encoding="utf-8"))
            return data.get("last_profile") if isinstance(data, dict) else None
        except (json.JSONDecodeError, OSError):
            return None
```

`sanitize_name` (aggiungi la regex a livello di modulo, sotto `_INVALID_NAME`):

```python
_WINDOWS_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$", re.IGNORECASE)


def sanitize_name(name: str) -> str:
    """Nome profilo → nome-file sicuro (preserva lettere accentate, rimuove
    separatori, evita i nomi-device riservati di Windows)."""
    cleaned = _INVALID_NAME.sub("_", (name or "").strip()).strip(". ")
    if not cleaned:
        return "profilo"
    if _WINDOWS_RESERVED.match(cleaned):
        return cleaned + "_"
    return cleaned
```

`list_names`:

```python
    def list_names(self) -> List[str]:
        # Espone solo stem stabili al round-trip di sanitizzazione: un file
        # creato esternamente con caratteri fuori whitelist verrebbe listato
        # ma _path_for lo risolverebbe su un ALTRO file → load/delete/export
        # fallirebbero per sempre su quella voce.
        return sorted(
            p.stem for p in self.profiles_dir.glob("*.json")
            if sanitize_name(p.stem) == p.stem
        )
```

`resolve_bg_ref`:

```python
def resolve_bg_ref(ref: dict, rm: ResourceManager) -> Tuple[str, bool]:
    """Risolve un riferimento-sfondo in un path assoluto.

    Ritorna (path, ok). Se il file non esiste → (path predefinito bundle, False);
    se manca anche l'asset predefinito → ("", False): il chiamante tratta il
    path vuoto come "nessuno sfondo" (il resolver non deve MAI sollevare).
    """
    def _default() -> str:
        try:
            return _default_bg_path(rm)
        except FileNotFoundError:
            return ""

    if not isinstance(ref, dict):
        return _default(), False
    kind = ref.get("kind")
    if kind == "resource":
        name = ref.get("name") or DEFAULT_BG_NAME
        try:
            return str(rm.get_image_path(name, "bg")), True
        except FileNotFoundError:
            return _default(), False
    if kind == "path":
        value = ref.get("value") or ""
        if value and os.path.isfile(value):
            return value, True
    return _default(), False
```

- [ ] **Step 3: Proteggi anche il caricamento del default in `__init__` della GUI**

In `gui/main_window_gui.py`, sostituisci:

```python
        # Immagine di sfondo di default (PIL): usata dal renderer OpenGL.
        self._default_bg_path = str(self.resource_manager.get_image_path(profiles.DEFAULT_BG_NAME, 'bg'))
        self.bg_image_path = self._default_bg_path  # path immagine attiva (per i profili)
        self.bg_img = Image.open(self._default_bg_path)
```

con:

```python
        # Immagine di sfondo di default (PIL): usata dal renderer OpenGL.
        # Un'installazione con asset mancante non deve impedire l'avvio: il
        # renderer gestisce bg_image=None (nessuno sfondo).
        try:
            self._default_bg_path = str(self.resource_manager.get_image_path(profiles.DEFAULT_BG_NAME, 'bg'))
            self.bg_img = Image.open(self._default_bg_path)
        except Exception:
            self._default_bg_path = ""
            self.bg_img = None
        self.bg_image_path = self._default_bg_path  # path immagine attiva (per i profili)
```

- [ ] **Step 4: Esegui e committa**

```bash
uv run python tests/test_profiles.py && uv run python -c "import gui.main_window_gui; print('OK')"
git add profiles.py gui/main_window_gui.py tests/test_profiles.py
git commit -m "Fix minori: robustezza store profili (m1-m4)"
```

---

### Task 19: F3.2 — Widget: slider, option menu, file picker (m6, m7, m9, m10)

**Files:**
- Modify: `gui/slider_frame.py` (costruttore ~38-46 e 56-61, `_on_changed` ~107, `set_value` ~127)
- Modify: `gui/optionmenu_frame.py` (`set_value` ~54-60)
- Modify: `gui/background_filepicker_frame.py` (costruttore, `open_file_dialog` ~49-50)
- Modify: `gui/main_window_gui.py` (costruzione `file_picker` ~354-358)

**Interfaces:**
- Produces: `BackgroundFilepickerFrame(..., start_dir: Optional[str] = None)` — nuova kwarg; la GUI passa la dir dei background del bundle via `ResourceManager.base_path`.

- [ ] **Step 1: `slider_frame.py` — warning con `is not None`, steps per gli int**

Nel costruttore, sostituisci:

```python
        if warning_trigger_value and not warning_text:
            raise ValueError("No warning text when warning_trigger_value set")
```

con:

```python
        if warning_trigger_value is not None and not warning_text:
            raise ValueError("No warning text when warning_trigger_value set")
```

Sostituisci il ramo int del costruttore:

```python
        if self._is_int:
            self._slider.setMinimum(int(from_))
            self._slider.setMaximum(int(to))
            self._slider.setSingleStep(1)
            self._slider.setValue(int(round(initial_value)))
```

con:

```python
        if self._is_int:
            self._slider.setMinimum(int(from_))
            self._slider.setMaximum(int(to))
            # steps, se dato, definisce la granularità (frecce/PgUp) anche per
            # gli slider int: prima era silenziosamente ignorato.
            step = max(1, int(round((to - from_) / steps))) if steps else 1
            self._slider.setSingleStep(step)
            self._slider.setValue(int(round(initial_value)))
```

Sostituisci la visibilità iniziale del warning:

```python
        self.warning_label.setVisible(
            bool(warning_trigger_value and self.get_value() >= warning_trigger_value)
        )
```

con:

```python
        self.warning_label.setVisible(
            warning_trigger_value is not None and self.get_value() >= warning_trigger_value
        )
```

In `_on_changed` e in `set_value`, sostituisci (in entrambi):

```python
        if self._warning_trigger_value:
```

con:

```python
        if self._warning_trigger_value is not None:
```

- [ ] **Step 2: `optionmenu_frame.py` — niente no-op silenzioso**

Sostituisci `set_value`:

```python
    def set_value(self, value: str) -> None:
        """Imposta il valore del menu senza scatenare la callback. Un valore
        sconosciuto lascia la selezione invariata (con log): dopo la
        validazione dei profili non dovrebbe più accadere — se accade è un
        bug a monte da rendere visibile, non da nascondere."""
        idx = self._combo.findText(str(value))
        if idx < 0:
            print(f"OptionMenu: valore sconosciuto {value!r}, selezione invariata")
            return
        self._combo.blockSignals(True)
        self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)
```

- [ ] **Step 3: `background_filepicker_frame.py` — start dir iniettata**

Aggiungi la kwarg al costruttore e salvala:

```python
    def __init__(
        self,
        parent=None,
        *,
        header_name: str = "BackgroundFilepickerFrame",
        placeholder_text: Optional[str] = None,
        apply_command: Optional[Callable] = None,
        start_dir: Optional[str] = None,
    ):
        super().__init__(parent)
        self._start_dir = start_dir
```

Sostituisci la prima riga di `open_file_dialog`:

```python
        start_dir = "resources/bg" if os.path.isdir("resources/bg") else ""
```

con:

```python
        # Dir iniettata dal chiamante (ResourceManager): il path relativo alla
        # CWD era sbagliato nel frozen build e con CWD ≠ radice del repo.
        start_dir = self._start_dir if self._start_dir and os.path.isdir(self._start_dir) else ""
```

- [ ] **Step 4: La GUI passa la dir del bundle**

In `gui/main_window_gui.py`, costruzione del file picker:

```python
        self.file_picker = BackgroundFilepickerFrame(
            header_name='Sfondo (immagine o video):',
            placeholder_text='Percorso file...',
            apply_command=self.apply_bg_command,
            start_dir=str(self.resource_manager.base_path / "bg"),
        )
```

- [ ] **Step 5: Verifica import e commit**

```bash
uv run python -c "import gui.main_window_gui; print('OK')"
git add gui/slider_frame.py gui/optionmenu_frame.py gui/background_filepicker_frame.py gui/main_window_gui.py
git commit -m "Fix minori: widget slider/option/filepicker (m6, m7, m9, m10)"
```

---

### Task 20: F3.3 — Renderer: viewport, bin DC, fps, contesto ImGui, alpha overlay (m11, m12, m14, m15, m16)

**Files:**
- Modify: `thread/render_params.py` (nuovo helper `sanitize_fps`)
- Modify: `thread/opengl_thread.py` (callback framebuffer, `_bass_mask` ~556-557, `_advance_video` ~1837-1838)
- Modify: `thread/video_decode_thread.py` (validazione fps ~52-54)
- Modify: `thread/imgui_overlay.py` (`__init__` ~24-28, slider alpha ~92)
- Test: `tests/test_render_params.py`

**Interfaces:**
- Produces: `thread.render_params.sanitize_fps(value) -> Optional[float]` — ritorna il valore come float solo se numero finito e positivo, altrimenti `None`. Usato dal decoder video e riusabile ovunque servano fps «sicuri».

- [ ] **Step 1: Callback framebuffer-size (m11)**

In `thread/opengl_thread.py`, in `run()`, dopo `glfw.set_window_size_callback(window, impl.resize_callback)` aggiungi:

```python
            glfw.set_framebuffer_size_callback(window, self._on_framebuffer_size)
```

Aggiungi i due metodi (mettili accanto a `_apply_cursor_mode`):

```python
    def _on_framebuffer_size(self, window, width, height):
        """Callback GLFW (gira sul thread GL durante poll_events): tiene
        viewport e FBO del bloom allineati alla dimensione reale del
        framebuffer (cambio risoluzione/scala DPI a runtime)."""
        if width <= 0 or height <= 0:
            return  # finestra minimizzata: non toccare nulla
        self._fb_width, self._fb_height = width, height
        glViewport(0, 0, width, height)
        self._invalidate_bloom_fbos()

    def _invalidate_bloom_fbos(self) -> None:
        """Libera i FBO bloom (se esistono): verranno ricreati lazy alla
        nuova dimensione dal prossimo _ensure_bloom_fbos."""
        if self._bloom_size is None:
            return
        try:
            glDeleteFramebuffers(len(self._bloom_fbos), self._bloom_fbos)
            glDeleteTextures(len(self._bloom_texs), self._bloom_texs)
        except Exception:
            pass
        self._bloom_fbos, self._bloom_texs, self._bloom_size = [], [], None
```

(Il `_cleanup_gl` esistente resta corretto: dopo l'invalidazione le liste sono vuote, quindi nessun double-free.)

- [ ] **Step 2: Maschera bassi senza bin DC (m12)**

Sostituisci in `__init__`:

```python
        _freqs = np.fft.rfftfreq(N_FFT, 1.0 / RATE)
        self._bass_mask = _freqs < BASS_CUTOFF_HZ
```

con:

```python
        _freqs = np.fft.rfftfreq(N_FFT, 1.0 / RATE)
        # Escludi il bin DC (0 Hz): un offset DC nel segnale catturato gonfia
        # stabilmente l'energia dei bassi e desensibilizza la beat detection.
        self._bass_mask = (_freqs > 0) & (_freqs < BASS_CUTOFF_HZ)
```

- [ ] **Step 3: fps validato tramite helper puro testato (m14)**

Prima il test: in `tests/test_render_params.py` aggiungi (e registra in `main()`):

```python
from thread.render_params import effective_splits, sanitize_fps


def test_sanitize_fps():
    assert sanitize_fps(30) == 30.0
    assert sanitize_fps(29.97) == 29.97
    # inf/nan/zero/negativi/non-numeri → None: con interval = 1/inf = 0.0
    # il pacing video sul thread GL dividerebbe per zero (ZeroDivisionError
    # → renderer chiuso per colpa di un metadato).
    assert sanitize_fps(float("inf")) is None
    assert sanitize_fps(float("nan")) is None
    assert sanitize_fps(0) is None
    assert sanitize_fps(-25) is None
    assert sanitize_fps("30") is None
    assert sanitize_fps(None) is None
    assert sanitize_fps(True) is None
```

```bash
uv run python tests/test_render_params.py
```
Expected: FAIL con `ImportError: cannot import name 'sanitize_fps'`.

Poi l'helper: in `thread/render_params.py` aggiungi `import math` in testa e la funzione:

```python
def sanitize_fps(value) -> float | None:
    """fps dai metadati video: accetta solo numeri finiti e positivi.

    ffmpeg può riportare inf/nan/0 su file con metadati rotti; un fps non
    finito produrrebbe un frame-interval pari a 0 e una divisione per zero
    nel pacing video del thread GL."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value) and value > 0:
        return float(value)
    return None
```

```bash
uv run python tests/test_render_params.py
```
Expected: `OK`.

In `thread/video_decode_thread.py`, aggiungi l'import in testa (sotto `import numpy as np`):

```python
from thread.render_params import sanitize_fps
```

e sostituisci:

```python
                    fps = meta.get('fps')
                    if fps and fps > 0:
                        self.fps = float(fps)
```

con:

```python
                    fps = sanitize_fps(meta.get('fps'))
                    if fps is not None:
                        self.fps = fps
```

In `thread/opengl_thread.py`, `_advance_video`, sostituisci:

```python
        if self._video_thread is not None and self._video_thread.fps > 0:
            self._video_frame_interval = 1.0 / self._video_thread.fps
```

con:

```python
        fps = sanitize_fps(self._video_thread.fps) if self._video_thread is not None else None
        if fps is not None:
            self._video_frame_interval = 1.0 / fps
```

e aggiorna l'import già presente dal Task 12:

```python
from thread.render_params import effective_splits, sanitize_fps
```

- [ ] **Step 4: Contesto ImGui exception-paired (m15) e alpha coerente (m16)**

In `thread/imgui_overlay.py`, sostituisci in `__init__`:

```python
        # Il contesto ImGui va creato PRIMA del backend (altrimenti RuntimeError).
        imgui.create_context()
        # attach_callbacks=False: registriamo noi le callback GLFW in run() (il backend
        # le sovrascriverebbe senza concatenarle alla nostra key_callback).
        self.impl = GlfwRenderer(window, attach_callbacks=False)
```

con:

```python
        # Il contesto ImGui va creato PRIMA del backend (altrimenti RuntimeError).
        imgui.create_context()
        # attach_callbacks=False: registriamo noi le callback GLFW in run() (il backend
        # le sovrascriverebbe senza concatenarle alla nostra key_callback).
        try:
            self.impl = GlfwRenderer(window, attach_callbacks=False)
        except Exception:
            # Senza questa coppia il contesto resterebbe orfano (self._imgui_overlay
            # resta None nel renderer → shutdown() mai chiamato) e si accumulerebbe
            # a ogni tentativo di avvio.
            imgui.destroy_context()
            raise
```

Nella scheda Visualizzazione (`_build_ui`), sostituisci:

```python
                changed, val = imgui.slider_float("Trasparenza sfondo", float(r._bg_alpha or 0.0), 0.0, 1.0)
```

con:

```python
                # None = opaco per _draw_background: mostra 1.0, non 0.0
                changed, val = imgui.slider_float("Trasparenza sfondo", float(1.0 if r._bg_alpha is None else r._bg_alpha), 0.0, 1.0)
```

- [ ] **Step 5: Verifica ed esegui i test**

```bash
uv run python tests/test_render_params.py && \
uv run python -c "from thread.opengl_thread import EqualizerOpenGLThread; from thread.video_decode_thread import VideoDecodeThread; print('OK')"
```
Expected: `OK` due volte.

- [ ] **Step 6: Commit**

```bash
git add thread/render_params.py thread/opengl_thread.py thread/video_decode_thread.py \
        thread/imgui_overlay.py tests/test_render_params.py
git commit -m "Fix minori renderer: viewport, bin DC, fps, contesto ImGui, alpha overlay"
```

---

### Task 21: F3.4 — Stato sfondi GUI aggiornato solo a caricamento riuscito (m20, m21)

**Files:**
- Modify: `gui/main_window_gui.py` (`_apply_image_background` ~687-697, `_apply_background_from_profile` ~1271-1299)

- [ ] **Step 1: `_apply_image_background`**

Sostituisci:

```python
    def _apply_image_background(self, filename: str):
        """Carica un'immagine come sfondo del renderer OpenGL."""
        try:
            self.bg_video_path = None  # torna alla modalità immagine
            self.bg_img = Image.open(filename)
            self.bg_image_path = filename
            self.update_bg_opengl(self.bg_img)
        except FileNotFoundError:
            QMessageBox.critical(self, 'Errore', 'File non trovato!')
        except Exception as e:
            QMessageBox.critical(self, 'Errore', f"Errore nel caricamento dell'immagine: {e}")
```

con:

```python
    def _apply_image_background(self, filename: str):
        """Carica un'immagine come sfondo del renderer OpenGL. Lo stato viene
        mutato solo a caricamento riuscito: un fallimento non deve cancellare
        un eventuale sfondo video attivo (che il renderer continua a mostrare,
        e che «Salva» deve continuare a persistere)."""
        try:
            img = Image.open(filename)
        except FileNotFoundError:
            QMessageBox.critical(self, 'Errore', 'File non trovato!')
            return
        except Exception as e:
            QMessageBox.critical(self, 'Errore', f"Errore nel caricamento dell'immagine: {e}")
            return
        self.bg_video_path = None  # torna alla modalità immagine
        self.bg_img = img
        self.bg_image_path = filename
        self.update_bg_opengl(self.bg_img)
```

- [ ] **Step 2: `_apply_background_from_profile` (ramo immagine + path vuoto dal Task 18)**

Sostituisci il ramo `else` (immagine):

```python
        else:
            # immagine, oppure video mancante → fallback immagine
            self.bg_video_path = None
            self.bg_image_path = path
            try:
                self.bg_img = Image.open(path)
            except Exception:
                ok = False
            self.file_picker.set_filename(path)
            if self.equalizer_opengl_thread:
                self.equalizer_control_queue.put({"type": "set_image", "value": self.bg_img})
```

con:

```python
        else:
            # immagine, oppure video mancante → fallback immagine. Stato mutato
            # solo se l'apertura riesce: mai persistere un path che non si apre.
            img = None
            if path:
                try:
                    img = Image.open(path)
                except Exception:
                    ok = False
            else:
                ok = False   # resolve_bg_ref: manca anche l'asset predefinito
            if img is not None:
                self.bg_video_path = None
                self.bg_img = img
                self.bg_image_path = path
                self.file_picker.set_filename(path)
                if self.equalizer_opengl_thread:
                    self.equalizer_control_queue.put({"type": "set_image", "value": self.bg_img})
```

- [ ] **Step 3: Verifica import e commit**

```bash
uv run python -c "import gui.main_window_gui; print('OK')"
git add gui/main_window_gui.py
git commit -m "Fix minori: stato sfondi mutato solo a caricamento riuscito (m20, m21)"
```

---

### Task 22: F3.5 — Avvio con messaggi d'errore leggibili (m22)

**Files:**
- Modify: `gui/main_window_gui.py` (`get_available_monitors` ~632-641, chiamata `asyncio.run(self.load_devices())` ~225)

- [ ] **Step 1: `get_available_monitors` senza raise**

Sostituisci:

```python
    def get_available_monitors(self) -> list[str]:
        if not glfw.init():
            raise Exception("GLFW initialization failed")
```

con:

```python
    def get_available_monitors(self) -> list[str]:
        if not glfw.init():
            # Niente raise: l'app deve mostrarsi comunque (magari l'utente vuole
            # solo gestire i profili). Il fullscreen resterà non avviabile.
            QMessageBox.critical(
                self, "Errore",
                "Inizializzazione GLFW fallita: nessun display OpenGL disponibile.\n"
                "La visualizzazione fullscreen non sarà utilizzabile.",
            )
            return []
```

- [ ] **Step 2: Enumerazione device protetta**

In `_build_device_selector`, sostituisci:

```python
        asyncio.run(self.load_devices())
```

con:

```python
        try:
            asyncio.run(self.load_devices())
        except Exception as e:
            # es. Linux senza PulseAudio/PipeWire: soundcard solleva qui.
            QMessageBox.critical(
                self, "Errore audio",
                f"Impossibile enumerare i dispositivi audio: {e}\n"
                "Su Linux servono PulseAudio o PipeWire attivi.",
            )
```

- [ ] **Step 3: Verifica import e commit**

```bash
uv run python -c "import gui.main_window_gui; print('OK')"
git add gui/main_window_gui.py
git commit -m "Fix minori: avvio con messaggi d'errore leggibili (m22)"
```

---

### Task 23: F3.6 — Igiene: test menu_keys, imgui.ini in config dir, .gitignore (m23, m25)

**Files:**
- Modify: `tests/test_menu_keys.py` (~29-31)
- Modify: `thread/imgui_overlay.py` (`__init__`)
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `profiles._default_config_dir() -> Path` (dir di config per-utente, già usata da `ProfileStore`).

- [ ] **Step 1: Fix dell'asserzione del test (prima il test, che oggi non rileva il bug)**

In `tests/test_menu_keys.py`, sostituisci l'import e il test:

```python
from thread.menu_keys import TOGGLE_KEY_NAMES, _NAME_TO_GLFW, toggle_key_to_glfw
```

```python
def test_names_list_is_nonempty_and_default_present():
    assert "F1" in TOGGLE_KEY_NAMES
    assert all(isinstance(n, str) for n in TOGGLE_KEY_NAMES)
    # Ogni nome esposto deve avere una voce ESPLICITA nella mappa: il vecchio
    # check `>= 0` passava anche per nomi non mappati, perché il fallback F1
    # di toggle_key_to_glfw è a sua volta un codice valido.
    for name in TOGGLE_KEY_NAMES:
        assert name in _NAME_TO_GLFW, name
        assert toggle_key_to_glfw(name) == _NAME_TO_GLFW[name]
```

```bash
uv run python tests/test_menu_keys.py
```
Expected: `OK` (la mappa è oggi completa; il test ora rileverebbe una voce mancante).

- [ ] **Step 2: `imgui.ini` nella config dir**

In `thread/imgui_overlay.py`, aggiungi l'import in testa (dopo `from thread.menu_keys import TOGGLE_KEY_NAMES`):

```python
from profiles import _default_config_dir
```

In `__init__`, subito dopo `imgui.create_context()` (e prima del blocco `try` del backend aggiunto al Task 20):

```python
        # Persisti il layout del menù nella config dir dell'app: di default
        # ImGui scrive "imgui.ini" nella CWD, sporcando la dir di lavoro.
        try:
            ini_dir = _default_config_dir()
            ini_dir.mkdir(parents=True, exist_ok=True)
            imgui.get_io().set_ini_filename(str(ini_dir / "imgui.ini"))
        except Exception:
            imgui.get_io().set_ini_filename(None)  # meglio nessuna persistenza che una CWD sporca
```

- [ ] **Step 3: `.gitignore` + rimozione del file vagante**

Aggiungi in fondo a `.gitignore`:

```
# Layout ImGui (a runtime è reindirizzato alla config dir; questo copre
# eventuali file creati da build vecchie)
imgui.ini
```

```bash
rm -f imgui.ini
```

- [ ] **Step 4: Esegui e committa**

```bash
uv run python tests/test_menu_keys.py && uv run python -c "import thread.imgui_overlay; print('OK')"
git add tests/test_menu_keys.py thread/imgui_overlay.py .gitignore
git commit -m "Igiene: assert reale in test_menu_keys, imgui.ini in config dir (m23, m25)"
```

---

### Task 24: Chiusura — suite completa, esiti nel report, note in CLAUDE.md

**Files:**
- Modify: `docs/superpowers/audits/2026-07-02-audit-bug-codebase.md` (sezione esiti in testa)
- Modify: `CLAUDE.md` (nuove invarianti + limite documentato m17)

- [ ] **Step 1: Esegui TUTTA la suite**

```bash
uv run python tests/test_profiles.py && \
uv run python tests/test_menu_keys.py && \
uv run python tests/test_render_params.py && \
uv run python tests/test_frame_time_stats.py
```
Expected: quattro `OK`.

- [ ] **Step 2: Esiti nel report audit**

In `docs/superpowers/audits/2026-07-02-audit-bug-codebase.md`, subito dopo il paragrafo introduttivo, aggiungi:

```markdown
> **Esito remediation (branch `bugfix/audit-2026-07`):** tutti i finding
> CRITICI e MEDI sono stati risolti; dei MINORI sono risolti tutti tranne
> m8 (quantizzazione slider float — innocua e convergente), m13 (join decoder
> con timeout — richiede I/O patologico, mitigato dal daemon flag), m17 (API
> GLFW su thread worker — limite architetturale documentato in CLAUDE.md),
> m18 (contratto di get_window_for_fft — documentato nel docstring) e m19
> (`clear_video` senza sender — affordance UI fuori scope). m24 è risolto
> dalla suite di test estesa. Dettagli per-fix nel piano
> `docs/superpowers/plans/2026-07-02-bonifica-bug-audit.md`.
```

- [ ] **Step 3: Docstring del contratto di `get_window_for_fft` (m18)**

In `thread/AudioBufferAccumulator.py`, aggiungi al docstring di `get_window_for_fft` (senza cambiare il codice):

```
        NB (contratto): prima che il buffer sia pieno ritorna np.array([])
        (shape (0,), float64), NON una matrice (0, channels) float32; e quando
        write_pos == 0 ritorna il buffer VIVO aliasato, valido solo fino alla
        prossima add_samples. Sicuro nell'uso attuale (lettura e scrittura sul
        solo thread GL, snapshot consumato subito); qualunque nuovo chiamante
        deve rispettare entrambe le cose.
```

- [ ] **Step 4: Note in CLAUDE.md**

In CLAUDE.md, nella sezione «OpenGL-specific concerns», aggiungi in fondo:

```markdown
- **Invarianti post-bonifica (2026-07):** (1) le scritture dei profili sono
  atomiche (`profiles._atomic_write_text`: tmp + `os.replace`) e `deserialize`
  valida tipi e range (`_VALIDATORS`) — nuove chiavi profilo vanno aggiunte a
  ENTRAMBI `DEFAULTS` e `_VALIDATORS`; (2) `window_close_callback` del renderer
  è invocato SOLO dal `finally` di `run()`, a teardown GLFW completato — mai
  chiamarlo da altri punti (il riavvio si affida a questo per non sovrapporre
  init/terminate GLFW); (3) `self.audio_queue` della GUI non va MAI ricreata
  (il renderer ne tiene il riferimento); (4) tutta l'API GLFW gira su un thread
  worker: funziona su Windows/X11 ma viola il contratto "main thread only" di
  GLFW — su macOS non funzionerebbe; è un limite architetturale accettato.
```

- [ ] **Step 5: Commit finale**

```bash
git add docs/superpowers/audits/2026-07-02-audit-bug-codebase.md CLAUDE.md thread/AudioBufferAccumulator.py
git commit -m "Docs: esito bonifica nel report audit, invarianti in CLAUDE.md"
```

- [ ] **Step 6: Checklist manuale (su Windows — richiede GL/audio reali)**

Da eseguire manualmente prima della PR (ogni voce mappa un finding):

1. Avvia/Ferma in rapida successione, più volte → nessun freeze (C1)
2. ESC nella finestra GL → riclick immediato su «Avvia» → riavvio pulito (C4)
3. Alt-F4 sulla finestra GL → il bottone torna «Avvia» da solo (M1)
4. Cambio device con renderer attivo → barre vive (M2)
5. Import di un `package.json` → errore chiaro, nessun profilo creato (M5)
6. Import di un profilo con tipi rotti → default al posto degli invalidi, nessun crash (C3)
7. Profilo «last» troncato a mano su disco → l'app parte coi default + warning (C2)
8. «Salva come…» su nome esistente → chiede conferma (M7)
9. Toggle key digitata in un campo di testo del menù ImGui → il menù resta aperto (M12)
10. Hover su «Ferma» → rosso più scuro; swatch «Tinta unica» blu all'avvio (M13, M14)
11. Video corrotto come sfondo → niente respawn ffmpeg a raffica, log chiaro (M15)
12. `imgui.ini` non ricompare nella radice del repo dopo l'uso del menù (m25)
