# In-window ImGui Settings Menu — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overlay a Dear ImGui settings menu on the fullscreen GLFW/OpenGL window so single-monitor users can change every live setting in-window, with changes synced back to the PySide6 control panel.

**Architecture:** A dedicated `ImGuiOverlay` class (new file) creates the ImGui context + GLFW backend on the renderer's existing window and draws the menu each frame after the visualization. Widgets read the renderer's live `self._*` state and push the **same** `set_*` control-queue messages the PySide6 sliders use; each change also fires an `on_change` callback that the GUI marshals onto its thread (reusing the existing close-signal pattern) to update the matching widget — silently, so no echo loops.

**Tech Stack:** Python ≥3.12, `imgui-bundle` (Dear ImGui), `glfw` (already present), PyOpenGL (already present), PySide6, `uv`.

## Global Constraints

- **Python:** `requires-python >=3.12`; deps via **`uv`** only (no `pip`, no `requirements.txt`). `imgui-bundle` ships cp312/cp313 wheels for Windows AMD64 and Linux manylinux (glibc ≥2.28) → compatible with CI (`windows-latest` / `ubuntu-22.04`).
- **Two venvs:** WSL2 uses `.venv-linux` (via `UV_PROJECT_ENVIRONMENT`), Windows uses `.venv`. After changing deps run `uv sync` on **each** platform.
- **Git:** commit as the `Cioscos` user; **never** add a `Co-Authored-By` (or any attribution) trailer.
- **Tests:** the repo has **no pytest**. Tests are standalone scripts run with `uv run python tests/test_*.py` that print `OK` on success. GL/Qt code has no headless harness → verified by **running the app** (the user has a Windows display; WSL2 is headless).
- **imgui-bundle API facts** (verified against pthom/imgui_bundle `main`), used throughout:
  - Import: `from imgui_bundle import imgui` and `from imgui_bundle.python_backends.glfw_backend import GlfwRenderer`. Import the system `glfw` **before** `imgui_bundle`.
  - `imgui.create_context()` is **required before** `GlfwRenderer(window)` (raises otherwise). Window must be created + current first.
  - Teardown: `impl.shutdown()` then `imgui.destroy_context()`. **`refresh_font_texture()` does not exist** — never call it.
  - `GlfwRenderer(window, attach_callbacks=True)` **OVERWRITES** GLFW callbacks and does **not** chain. We use `attach_callbacks=False` and forward events to its public handlers: `keyboard_callback(window,key,scancode,action,mods)`, `char_callback`, `mouse_callback` (cursor-pos), `mouse_button_callback`, `scroll_callback`, `resize_callback`.
  - Widgets return tuples — **reassign the value**: `slider_float/slider_int/combo/checkbox` → `(changed, value)`; `combo(label, current_int, items_list)`; `color_edit3(label, [r,g,b])` → `(changed, col)` where `col` is indexable; `begin(name)` → `(visible, p_open)` (always call `imgui.end()`); `begin_tab_bar(id)` → `bool`; `begin_tab_item(label)` → `(selected, p_open)`.
  - Per-frame order: `impl.process_inputs()` → `imgui.new_frame()` → UI → `imgui.render()` → `impl.render(imgui.get_draw_data())`.

---

### Task 1: Add the `imgui-bundle` dependency

**Files:**
- Modify: `pyproject.toml` (via `uv add`), `uv.lock` (regenerated)

- [ ] **Step 1: Add the dependency (WSL2)**

Run from the repo root:
```bash
uv add imgui-bundle
```
Expected: `pyproject.toml` gains `imgui-bundle` under `[project] dependencies`; `uv.lock` is updated; `.venv-linux` is synced.

- [ ] **Step 2: Smoke-test the import headless**

Run:
```bash
uv run python -c "from imgui_bundle import imgui; from imgui_bundle.python_backends.glfw_backend import GlfwRenderer; print('OK', imgui.get_version())"
```
Expected: prints `OK` and a Dear ImGui version (e.g. `1.92.x`). No traceback. (Importing does not need a GL context.)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add imgui-bundle dependency for in-window menu"
```

- [ ] **Step 4: Note for the Windows env**

After merging/pulling on Windows, run `uv sync` there to add `imgui-bundle` to `.venv` before running the app. (Documentation step — no code.)

---

### Task 2: Toggle-key mapping module + unit test

Pure, headless-testable logic: map a human key name (e.g. `"F1"`) to a GLFW key code, and expose the list of selectable names. Kept in its own tiny module (only imports `glfw`) so it's importable without `imgui-bundle` and unit-testable.

**Files:**
- Create: `thread/menu_keys.py`
- Test: `tests/test_menu_keys.py`

**Interfaces:**
- Produces: `TOGGLE_KEY_NAMES: list[str]`; `toggle_key_to_glfw(name: str) -> int` (returns `glfw.KEY_F1` for unknown names).

- [ ] **Step 1: Write the failing test**

Create `tests/test_menu_keys.py`:
```python
"""
Verifica headless (niente pytest nel repo): asserzioni pure su menu_keys.
Eseguire dalla radice del repo con:  uv run python tests/test_menu_keys.py
Uscita 0 + "OK" = passato.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glfw
from thread.menu_keys import TOGGLE_KEY_NAMES, toggle_key_to_glfw


def test_known_keys_map_to_glfw_codes():
    assert toggle_key_to_glfw("F1") == glfw.KEY_F1
    assert toggle_key_to_glfw("F12") == glfw.KEY_F12
    assert toggle_key_to_glfw("M") == glfw.KEY_M


def test_unknown_key_defaults_to_f1():
    assert toggle_key_to_glfw("not-a-key") == glfw.KEY_F1
    assert toggle_key_to_glfw("") == glfw.KEY_F1


def test_names_list_is_nonempty_and_default_present():
    assert "F1" in TOGGLE_KEY_NAMES
    assert all(isinstance(n, str) for n in TOGGLE_KEY_NAMES)
    # ogni nome esposto deve mappare a un codice valido (no fallback silenzioso)
    for name in TOGGLE_KEY_NAMES:
        assert toggle_key_to_glfw(name) >= 0


def main():
    test_known_keys_map_to_glfw_codes()
    test_unknown_key_defaults_to_f1()
    test_names_list_is_nonempty_and_default_present()
    print("OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python tests/test_menu_keys.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'thread.menu_keys'`.

- [ ] **Step 3: Write the implementation**

Create `thread/menu_keys.py`:
```python
"""
Mappa nome-tasto → codice GLFW per il tasto-toggle del menù in-window.
Modulo minimale (importa solo glfw) così è testabile headless senza imgui-bundle.
"""
import glfw

# Nomi selezionabili dalla GUI (combo "Tasto menù"). Tab è escluso di proposito:
# ImGui lo usa per la navigazione tra i campi.
TOGGLE_KEY_NAMES = [
    "F1", "F2", "F3", "F4", "F5", "F6",
    "F7", "F8", "F9", "F10", "F11", "F12",
    "M", "N", "P",
]

_NAME_TO_GLFW = {
    "F1": glfw.KEY_F1, "F2": glfw.KEY_F2, "F3": glfw.KEY_F3, "F4": glfw.KEY_F4,
    "F5": glfw.KEY_F5, "F6": glfw.KEY_F6, "F7": glfw.KEY_F7, "F8": glfw.KEY_F8,
    "F9": glfw.KEY_F9, "F10": glfw.KEY_F10, "F11": glfw.KEY_F11, "F12": glfw.KEY_F12,
    "M": glfw.KEY_M, "N": glfw.KEY_N, "P": glfw.KEY_P,
}


def toggle_key_to_glfw(name: str) -> int:
    """Codice GLFW per il nome dato; F1 come fallback per nomi sconosciuti."""
    return _NAME_TO_GLFW.get(str(name), glfw.KEY_F1)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run python tests/test_menu_keys.py`
Expected: prints `OK`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add thread/menu_keys.py tests/test_menu_keys.py
git commit -m "feat: toggle-key name->glfw mapping for in-window menu"
```

---

### Task 3: `ImGuiOverlay` class (construction, open/close state, empty UI, shutdown)

Encapsulates all ImGui concerns. This task produces a working overlay that shows an (almost) empty window when open; the full controls come in Task 5. GL-bound → no unit test; verified in Task 4's run.

**Files:**
- Create: `thread/imgui_overlay.py`

**Interfaces:**
- Consumes: a GLFW `window`; a `renderer` (the `EqualizerOpenGLThread`, read for live state via its `_*` attrs); a `control_queue` (`queue.Queue`); an `on_change` callable `(dict) -> None` (may be `None`).
- Produces: `ImGuiOverlay(window, renderer, control_queue, on_change)`; `.impl` (the `GlfwRenderer`, for callback wiring); `.is_open() -> bool`; `.set_open(bool)`; `.toggle()`; `.render_frame()`; `.shutdown()`; `._emit(msg_type, value)`.

- [ ] **Step 1: Write the class**

Create `thread/imgui_overlay.py`:
```python
"""
Menù impostazioni in-window: overlay Dear ImGui sulla finestra GLFW/OpenGL del
renderer. Vive interamente sul thread OpenGL (lo stesso che possiede il contesto
GL e gli attributi del renderer), quindi legge lo stato live del renderer
direttamente e scrive i messaggi di controllo sulla stessa coda della GUI.
"""
# NB: importare il glfw di sistema PRIMA di imgui_bundle (così si usa quello standard).
import glfw
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

from thread.menu_keys import TOGGLE_KEY_NAMES


class ImGuiOverlay:
    def __init__(self, window, renderer, control_queue=None, on_change=None):
        self._window = window
        self._renderer = renderer
        self._control_queue = control_queue
        self._on_change = on_change
        self._open = False
        self._first_ui = True   # dimensiona la finestra solo alla prima apparizione

        # Il contesto ImGui va creato PRIMA del backend (altrimenti RuntimeError).
        imgui.create_context()
        # attach_callbacks=False: registriamo noi le callback GLFW in run() (il backend
        # le sovrascriverebbe senza concatenarle alla nostra key_callback).
        self.impl = GlfwRenderer(window, attach_callbacks=False)

    # --- stato apertura ---
    def is_open(self) -> bool:
        return self._open

    def set_open(self, value: bool) -> None:
        self._open = bool(value)

    def toggle(self) -> None:
        self._open = not self._open

    # --- ciclo per-frame ---
    def render_frame(self) -> None:
        """Da chiamare ogni frame, dopo il render della visualizzazione e prima dello
        swap. Quando il menù è chiuso non costruisce widget (costo trascurabile)."""
        self.impl.process_inputs()
        imgui.new_frame()
        if self._open:
            self._build_ui()
        imgui.render()
        self.impl.render(imgui.get_draw_data())

    def shutdown(self) -> None:
        """Libera le risorse GL del backend e distrugge il contesto ImGui. Richiede
        contesto GL ancora attivo (chiamare prima di destroy_window/terminate)."""
        try:
            self.impl.shutdown()
        except Exception:
            pass
        try:
            imgui.destroy_context()
        except Exception:
            pass

    # --- invio modifiche ---
    def _emit(self, msg_type: str, value) -> None:
        """Applica la modifica al renderer (control queue) e notifica la GUI (on_change),
        esattamente come uno slider del pannello PySide6."""
        msg = {"type": msg_type, "value": value}
        if self._control_queue is not None:
            self._control_queue.put(msg)
        if self._on_change is not None:
            self._on_change(msg)

    # --- UI (riempita in Task 5) ---
    def _build_ui(self) -> None:
        if self._first_ui:
            imgui.set_next_window_size((440, 600))
            self._first_ui = False
        imgui.begin("Impostazioni")
        imgui.text("Menu in costruzione")
        imgui.end()
```

- [ ] **Step 2: Smoke-test the import headless**

Run: `uv run python -c "import thread.imgui_overlay; print('OK')"`
Expected: prints `OK`, no traceback. (Constructing the class needs a GL window — not done here; only the import is checked.)

- [ ] **Step 3: Commit**

```bash
git add thread/imgui_overlay.py
git commit -m "feat: ImGuiOverlay scaffold (context, open/close, empty UI)"
```

---

### Task 4: Integrate the overlay into `EqualizerOpenGLThread`

Wire the overlay into the renderer: constructor params, creation in `run()`, callback forwarding, per-frame draw, cursor handling, ESC/toggle logic, the `set_menu_toggle_key` message, and shutdown. After this task the toggle key opens an (empty) in-window menu and ESC/cursor behave correctly.

**Files:**
- Modify: `thread/opengl_thread.py`

**Interfaces:**
- Consumes: `ImGuiOverlay` (Task 3), `toggle_key_to_glfw` (Task 2).
- Produces: constructor params `menu_toggle_key: str = "F1"`, `on_setting_changed: Callable = None`; instance attrs `self._imgui_overlay`, `self._menu_toggle_key`, `self._menu_toggle_glfw_key`, `self._on_setting_changed`; method `_apply_cursor_mode(window)`; message type `set_menu_toggle_key`.

- [ ] **Step 1: Add the imports**

In `thread/opengl_thread.py`, after the existing thread imports (currently lines 15-17):
```python
from thread.AudioBufferAccumulator import AudioBufferAccumulator
from thread.video_decode_thread import VideoDecodeThread
from thread.frame_time_stats import FrameTimeStats
from thread.imgui_overlay import ImGuiOverlay
from thread.menu_keys import toggle_key_to_glfw
```

- [ ] **Step 2: Add constructor parameters**

In `__init__`, change the signature tail (currently line 488) from:
```python
                 osc_smoothing: float = 0.5,
                 window_close_callback: Callable = None):
```
to:
```python
                 osc_smoothing: float = 0.5,
                 menu_toggle_key: str = "F1",
                 on_setting_changed: Callable = None,
                 window_close_callback: Callable = None):
```

- [ ] **Step 3: Store the new state**

In `__init__`, immediately after `self.window_close_callback = window_close_callback` (currently line 499), insert:
```python
        # --- Menù in-window (overlay ImGui) ---
        self._on_setting_changed = on_setting_changed   # callback verso la GUI (sync inverso)
        self._menu_toggle_key = str(menu_toggle_key)     # nome tasto (per profili / menù)
        self._menu_toggle_glfw_key = toggle_key_to_glfw(menu_toggle_key)  # codice GLFW
        self._imgui_overlay = None                       # creato in run(), a contesto GL attivo
```

- [ ] **Step 4: Create the overlay and wire callbacks in `run()`**

In `run()`, replace the single line (currently 1555):
```python
            glfw.set_key_callback(window, self.key_callback)
```
with:
```python
            # Overlay ImGui (menù in-window). Creabile solo a contesto GL attivo.
            self._imgui_overlay = ImGuiOverlay(
                window,
                renderer=self,
                control_queue=self._control_queue,
                on_change=self._on_setting_changed,
            )
            impl = self._imgui_overlay.impl
            # La NOSTRA key_callback gestisce toggle/ESC e inoltra a ImGui; le altre
            # callback vanno direttamente agli handler del backend (attach_callbacks=False
            # → il backend NON le concatena, le registriamo noi).
            glfw.set_key_callback(window, self.key_callback)
            glfw.set_char_callback(window, impl.char_callback)
            glfw.set_cursor_pos_callback(window, impl.mouse_callback)
            glfw.set_mouse_button_callback(window, impl.mouse_button_callback)
            glfw.set_scroll_callback(window, impl.scroll_callback)
            glfw.set_window_size_callback(window, impl.resize_callback)
            # Menù chiuso all'avvio → cursore nascosto (visualizzazione pulita).
            glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
```

- [ ] **Step 5: Draw the overlay each frame**

In the main loop, immediately after `self._render(smoothed_amplitudes, waveform, frame_delta)` (currently line 1633) and before `glfw.swap_buffers(window)`:
```python
                # 5) Overlay ImGui (menù in-window): disegnato per ultimo, sul
                #    framebuffer di default, sopra barre/sfondo.
                self._imgui_overlay.render_frame()
```

- [ ] **Step 6: Shut the overlay down in `finally`**

In `run()`'s `finally` (currently lines 1648-1655), insert the overlay shutdown after `self._stop_video()` and before `self._cleanup_gl()`:
```python
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

- [ ] **Step 7: Rewrite `key_callback` and add `_apply_cursor_mode`**

Replace the whole `key_callback` method (currently lines 1663-1678) with:
```python
    def key_callback(self, window, key, scancode, action, mods):
        """
        Callback tasti. Inoltra prima l'evento al backend ImGui (per testo/navigazione
        del menù), poi gestisce il tasto-toggle del menù ed ESC. ESC: chiude il menù se
        aperto, altrimenti esce dal fullscreen (comportamento storico).
        """
        # Inoltra al backend ImGui (registrato con attach_callbacks=False).
        if self._imgui_overlay is not None:
            self._imgui_overlay.impl.keyboard_callback(window, key, scancode, action, mods)

        if action != glfw.PRESS:
            return

        # Tasto-toggle configurabile: apre/chiude il menù.
        if key == self._menu_toggle_glfw_key and self._imgui_overlay is not None:
            self._imgui_overlay.toggle()
            self._apply_cursor_mode(window)
            return

        if key == glfw.KEY_ESCAPE:
            # Menù aperto → ESC lo chiude (non esce dal fullscreen).
            if self._imgui_overlay is not None and self._imgui_overlay.is_open():
                self._imgui_overlay.set_open(False)
                self._apply_cursor_mode(window)
                return
            # Menù chiuso → comportamento storico: esci dal fullscreen.
            glfw.set_window_should_close(window, True)
            self.stop()
            if self.window_close_callback:
                self.window_close_callback()

    def _apply_cursor_mode(self, window):
        """Cursore visibile col menù aperto, nascosto col menù chiuso."""
        is_open = self._imgui_overlay is not None and self._imgui_overlay.is_open()
        mode = glfw.CURSOR_NORMAL if is_open else glfw.CURSOR_HIDDEN
        glfw.set_input_mode(window, glfw.CURSOR, mode)
```

- [ ] **Step 8: Handle the `set_menu_toggle_key` message**

In `process_control_message`, insert a new branch right before the `elif message_type == 'set_image':` branch (currently line 1385):
```python
        elif message_type == 'set_menu_toggle_key':
            # Cambia il tasto che apre/chiude il menù in-window (nome → codice GLFW).
            self._menu_toggle_key = str(message['value'])
            self._menu_toggle_glfw_key = toggle_key_to_glfw(message['value'])

```

- [ ] **Step 9: Verify by running the app (Windows display)**

Run `uv run python main.py`, configure a monitor, and start the fullscreen renderer. Then:
- Press **F1** → an "Impostazioni" window appears with "Menu in costruzione"; the **cursor becomes visible**.
- Press **F1** again → the menu hides; the **cursor hides**.
- With the menu **open**, press **ESC** → the menu closes (app stays running).
- With the menu **closed**, press **ESC** → the fullscreen renderer closes and the control panel button returns to "Avvia" (historic behavior intact).
- Audio bars keep animating normally throughout.

Expected: all of the above hold; no crash on open/close or on shutdown.

- [ ] **Step 10: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "feat: mount ImGui overlay in OpenGL renderer (toggle key, cursor, ESC)"
```

---

### Task 5: Fill the menu UI (5 tabs, all live controls)

Replace `ImGuiOverlay._build_ui` with the full menu: a tab bar mirroring the panel's 5 pages, each control reading the renderer's live `_*` state and emitting the matching `set_*` message. After this task the in-window menu changes the visualization one-way (sync-back is Task 6). The toggle-key combo is added in Task 7.

**Files:**
- Modify: `thread/imgui_overlay.py`

- [ ] **Step 1: Replace `_build_ui`**

Replace the placeholder `_build_ui` (from Task 3) with:
```python
    def _build_ui(self) -> None:
        r = self._renderer
        mode = int(r._mode)        # 0 Barre, 1 Radiale, 2 Oscilloscopio, 3 Linea/Area
        cmode = int(r._color_mode)  # 0 Classico, 1 Tinta, 2 Gradiente, 3 Spettro

        if self._first_ui:
            imgui.set_next_window_size((440, 600))
            self._first_ui = False
        imgui.begin("Impostazioni")

        if imgui.begin_tab_bar("sw_tabs"):
            # ---------------- Visualizzazione ----------------
            sel, _ = imgui.begin_tab_item("Visualizzazione")
            if sel:
                changed, idx = imgui.combo("Modalità", mode,
                                           ["Barre", "Radiale", "Oscilloscopio", "Linea/Area"])
                if changed:
                    self._emit("set_viz_mode", int(idx))
                changed, val = imgui.slider_float("Trasparenza sfondo", float(r._bg_alpha or 0.0), 0.0, 1.0)
                if changed:
                    self._emit("set_alpha", float(val))
                changed, val = imgui.slider_float("Opacità barre", float(r._bars_alpha), 0.0, 1.0)
                if changed:
                    self._emit("set_bars_alpha", float(val))
                imgui.end_tab_item()

            # ---------------- Forma ----------------
            sel, _ = imgui.begin_tab_item("Forma")
            if sel:
                if mode in (0, 1):
                    changed, val = imgui.slider_float("Larghezza barre", float(r._bar_width_frac), 0.1, 1.0)
                    if changed:
                        self._emit("set_bar_width", float(val))
                if mode == 0:
                    changed, val = imgui.checkbox("Cime arrotondate", bool(r._rounded))
                    if changed:
                        self._emit("set_rounded", bool(val))
                    changed, val = imgui.checkbox("Ancoraggio dal centro", bool(r._mirror))
                    if changed:
                        self._emit("set_bar_anchor", bool(val))
                if mode in (0, 1, 3):
                    changed, val = imgui.checkbox("Ordine simmetrico (bassi al centro)", bool(r._band_order_symmetric))
                    if changed:
                        self._emit("set_band_order", bool(val))
                if mode == 1:
                    changed, val = imgui.slider_float("Raggio foro", float(r._inner_radius), 0.0, 0.8)
                    if changed:
                        self._emit("set_inner_radius", float(val))
                if mode in (2, 3):
                    changed, val = imgui.slider_float("Spessore linea", float(r._line_thickness), 0.002, 0.06)
                    if changed:
                        self._emit("set_line_thickness", float(val))
                if mode == 3:
                    changed, val = imgui.checkbox("Riempi area", bool(r._fill))
                    if changed:
                        self._emit("set_fill", bool(val))
                if mode == 2:
                    changed, val = imgui.checkbox("Specchiato", bool(r._osc_mirror))
                    if changed:
                        self._emit("set_osc_mirror", bool(val))
                    changed, val = imgui.slider_float("Fluidità", float(r._osc_smoothing), 0.0, 1.0)
                    if changed:
                        self._emit("set_osc_smoothing", float(val))
                imgui.end_tab_item()

            # ---------------- Colore ----------------
            sel, _ = imgui.begin_tab_item("Colore")
            if sel:
                changed, idx = imgui.combo("Modalità colore", cmode,
                                           ["Classico", "Tinta unica", "Gradiente", "Spettro"])
                if changed:
                    self._emit("set_color_mode", int(idx))
                if cmode == 0:
                    changed, val = imgui.slider_float("Soglia giallo", float(r._green_split), 0.0, 1.0)
                    if changed:
                        self._emit("set_green_split", float(val))
                    changed, val = imgui.slider_float("Soglia rosso", float(r._yellow_split), 0.0, 1.0)
                    if changed:
                        self._emit("set_yellow_split", float(val))
                if cmode in (1, 2):
                    label_a = "Colore barre" if cmode == 1 else "Colore base"
                    changed, col = imgui.color_edit3(label_a, [float(c) for c in r._color_a])
                    if changed:
                        self._emit("set_color_a", (float(col[0]), float(col[1]), float(col[2])))
                if cmode == 2:
                    changed, col = imgui.color_edit3("Colore cima", [float(c) for c in r._color_b])
                    if changed:
                        self._emit("set_color_b", (float(col[0]), float(col[1]), float(col[2])))
                imgui.end_tab_item()

            # ---------------- Audio ----------------
            sel, _ = imgui.begin_tab_item("Audio")
            if sel:
                changed, val = imgui.slider_float("Soglia rumore", float(r._noise_threshold), 0.0, 1.0)
                if changed:
                    self._emit("set_noise_threshold", float(val))
                changed, val = imgui.slider_int("Bande di frequenza", int(r._n_bands), 1, 100)
                if changed:
                    self._emit("set_frequency_bands", int(val))
                changed, val = imgui.slider_int("Pavimento dB", int(round(r._db_floor)), -90, -25)
                if changed:
                    self._emit("set_db_floor", float(val))
                changed, val = imgui.slider_int("Tetto dB", int(round(r._db_ceiling)), -20, 6)
                if changed:
                    self._emit("set_db_ceiling", float(val))
                changed, val = imgui.slider_int("Attacco (ms)", int(round(r._attack_tau * 1000)), 1, 100)
                if changed:
                    self._emit("set_attack_tau", val / 1000.0)
                changed, val = imgui.slider_int("Rilascio (ms)", int(round(r._release_tau * 1000)), 20, 1000)
                if changed:
                    self._emit("set_release_tau", val / 1000.0)
                changed, val = imgui.slider_float("Tilt (dB/ottava)", float(r._tilt_db_per_oct), -6.0, 6.0)
                if changed:
                    self._emit("set_tilt_db_per_oct", float(val))
                imgui.end_tab_item()

            # ---------------- Effetti ----------------
            sel, _ = imgui.begin_tab_item("Effetti")
            if sel:
                if mode == 0:  # peak-cap solo in modalità Barre
                    changed, val = imgui.checkbox("Peak-cap", bool(r._peakcap_enabled))
                    if changed:
                        self._emit("set_peakcap_enabled", bool(val))
                    changed, col = imgui.color_edit3("Colore cap", [float(c) for c in r._peakcap_color])
                    if changed:
                        self._emit("set_peakcap_color", (float(col[0]), float(col[1]), float(col[2])))
                    changed, val = imgui.slider_int("Hold (ms)", int(round(r._peakcap_hold * 1000)), 0, 2000)
                    if changed:
                        self._emit("set_peakcap_hold", val / 1000.0)
                    changed, val = imgui.slider_float("Caduta (/s)", float(r._peakcap_fall), 0.1, 3.0)
                    if changed:
                        self._emit("set_peakcap_fall", float(val))
                changed, val = imgui.checkbox("Glow", bool(r._bloom_enabled))
                if changed:
                    self._emit("set_bloom_enabled", bool(val))
                changed, val = imgui.slider_float("Intensità glow", float(r._bloom_intensity), 0.0, 3.0)
                if changed:
                    self._emit("set_bloom_intensity", float(val))
                changed, val = imgui.checkbox("Beat pulse", bool(r._beat_enabled))
                if changed:
                    self._emit("set_beat_enabled", bool(val))
                changed, val = imgui.slider_float("Intensità zoom", float(r._beat_intensity), 0.0, 0.3)
                if changed:
                    self._emit("set_beat_intensity", float(val))
                changed, val = imgui.slider_float("Sensibilità", float(r._beat_sensitivity), 1.05, 3.0)
                if changed:
                    self._emit("set_beat_sensitivity", float(val))
                imgui.end_tab_item()

            imgui.end_tab_bar()

        imgui.end()
```

Note: slider ranges mirror the PySide6 panel (see the spec's control inventory). If any differ from the panel's `SliderCustomFrame` `from_`/`to`, align them to the panel's values.

- [ ] **Step 2: Verify by running the app (Windows display)**

Run `uv run python main.py`, start fullscreen, press **F1**. For **each tab**:
- Move a slider / toggle a checkbox / change a combo → the visualization reacts immediately and identically to the equivalent panel control.
- Switch **Modalità** to Radiale / Oscilloscopio / Linea-Area and confirm the **Forma** tab shows only the relevant controls (Raggio foro for Radiale; Spessore/Fluidità/Specchiato for Osc; Spessore/Riempi for Linea); **Effetti** shows Peak-cap only in Barre.
- Switch **Modalità colore** and confirm Colore shows the right sub-controls.

Expected: every control works; conditional visibility matches the panel; no crash.

- [ ] **Step 3: Commit**

```bash
git add thread/imgui_overlay.py
git commit -m "feat: full in-window menu UI (5 tabs mirroring the panel)"
```

---

### Task 6: Bidirectional sync — push in-window changes back to the PySide6 panel

Add the reverse channel: the overlay's `on_change` callback (already passed in Task 4) is wired to a GUI callback that emits a `Signal(dict)` (reusing the close-signal pattern), whose slot updates the matching widget + state **silently** (the widgets' `set_value`/`set_color` already block signals, so no echo loop). After this task, changing a setting in-window updates the panel — so it survives fullscreen relaunch and feeds profile save.

**Files:**
- Modify: `gui/main_window_gui.py`

**Interfaces:**
- Consumes: `EqualizerOpenGLThread(..., on_setting_changed=...)` (Task 4).
- Produces: `AudioCaptureGUI._setting_changed = Signal(dict)`; methods `inwindow_setting_changed(message)` (GL-thread callback → emits) and `_on_setting_changed(message)` (GUI-thread slot → applies).

- [ ] **Step 1: Declare the signal**

In `gui/main_window_gui.py`, next to the existing `_opengl_closed = Signal()` (line 85), add:
```python
    _opengl_closed = Signal()
    _setting_changed = Signal(dict)
```

- [ ] **Step 2: Connect the slot**

In `__init__`, next to `self._opengl_closed.connect(self._on_opengl_closed)` (line 181), add:
```python
        self._opengl_closed.connect(self._on_opengl_closed)
        self._setting_changed.connect(self._on_setting_changed)
```

- [ ] **Step 3: Pass the callback to the renderer**

In the `EqualizerOpenGLThread(...)` construction (line 936), add `on_setting_changed=self.inwindow_setting_changed,` right before `window_close_callback=self.opengl_window_closed,`:
```python
                osc_smoothing=self.viz_osc_smoothing,
                monitor=self.selected_monitor,
                on_setting_changed=self.inwindow_setting_changed,
                window_close_callback=self.opengl_window_closed,
            )
```

- [ ] **Step 4: Add the GL-thread callback (marshals to the GUI thread)**

Next to `opengl_window_closed` (lines 980-982), add:
```python
    def inwindow_setting_changed(self, message: dict):
        # Chiamato sul thread GL dall'overlay ImGui: marshalla sul thread GUI via
        # segnale (consegnato in coda), come opengl_window_closed → _opengl_closed.
        self._setting_changed.emit(message)
```

- [ ] **Step 5: Add the GUI-thread slot (applies the change silently)**

Add this method (e.g. right after `_on_opengl_closed`, after line 986):
```python
    def _on_setting_changed(self, message: dict) -> None:
        """
        Slot (thread GUI): una modifica fatta dal menù in-window torna qui e aggiorna
        il widget + lo stato self.* corrispondente, SENZA ri-mettere il messaggio sul
        control queue (il renderer ha già applicato il valore → nessun loop). I setter
        dei widget (set_value/set_color) bloccano i segnali, quindi non riscattano le
        callback update_*. Per i due combo con visibilità (viz/color) richiamiamo le
        on_*_changed: rimettono una sola volta lo stesso messaggio in coda (echo innocuo,
        il renderer non genera on_change → niente loop).
        """
        t = message["type"]
        v = message["value"]

        # --- Audio (solo widget: il costruttore li rilegge dai widget) ---
        if t == "set_noise_threshold":
            self.noise_slider.set_value(v)
        elif t == "set_frequency_bands":
            self.frequency_slider.set_value(v)
        elif t == "set_db_floor":
            self.adv_db_floor_slider.set_value(v)
        elif t == "set_db_ceiling":
            self.adv_db_ceiling_slider.set_value(v)
        elif t == "set_attack_tau":
            self.adv_attack_slider.set_value(v * 1000.0)   # s → ms
        elif t == "set_release_tau":
            self.adv_release_slider.set_value(v * 1000.0)  # s → ms
        elif t == "set_tilt_db_per_oct":
            self.adv_tilt_slider.set_value(v)

        # --- Visualizzazione ---
        elif t == "set_alpha":
            self.bg_alpha = float(v); self.alpha_slider.set_value(self.bg_alpha)
        elif t == "set_bars_alpha":
            self.bars_alpha = float(v); self.bars_alpha_slider.set_value(self.bars_alpha)
        elif t == "set_viz_mode":
            label = _INT_TO_VIZ_MODE.get(int(v), "Barre")
            self.viz_mode_option.set_value(label)   # silenzioso (blockSignals)
            self.on_viz_mode_changed(label)          # visibilità + self.viz_mode (+ echo)

        # --- Colore ---
        elif t == "set_color_mode":
            label = _INT_TO_COLOR_MODE.get(int(v), "Classico")
            self.bars_mode_option.set_value(label)
            self.on_color_mode_changed(label)        # visibilità + self.bars_color_mode (+ echo)
        elif t == "set_color_a":
            self.bars_color_a = tuple(v)
            self.bars_solid_color.set_color(self.bars_color_a)
            self.bars_grad_base.set_color(self.bars_color_a)
        elif t == "set_color_b":
            self.bars_color_b = tuple(v)
            self.bars_grad_top.set_color(self.bars_color_b)
        elif t == "set_green_split":
            self.bars_green_split = float(v); self.bars_yellow_thr.set_value(self.bars_green_split)
        elif t == "set_yellow_split":
            self.bars_yellow_split = float(v); self.bars_red_thr.set_value(self.bars_yellow_split)

        # --- Forma ---
        elif t == "set_bar_width":
            self.bars_width = float(v); self.bars_width_slider.set_value(self.bars_width)
        elif t == "set_rounded":
            self.bars_rounded = bool(v); self.bars_rounded_option.set_value("Sì" if self.bars_rounded else "No")
        elif t == "set_bar_anchor":
            self.bars_anchor_center = bool(v)
            self.bars_anchor_option.set_value("Centro" if self.bars_anchor_center else "Dal basso")
        elif t == "set_band_order":
            self.bars_band_symmetric = bool(v)
            self.bars_order_option.set_value("Simmetrico" if self.bars_band_symmetric else "Standard")
        elif t == "set_inner_radius":
            self.viz_inner_radius = float(v); self.viz_inner_radius_slider.set_value(self.viz_inner_radius)
        elif t == "set_line_thickness":
            self.viz_line_thickness = float(v); self.viz_thickness_slider.set_value(self.viz_line_thickness)
        elif t == "set_fill":
            self.viz_fill = bool(v); self.viz_fill_option.set_value("Area" if self.viz_fill else "Linea")
        elif t == "set_osc_mirror":
            self.viz_osc_mirror = bool(v); self.viz_osc_mirror_option.set_value("Sì" if self.viz_osc_mirror else "No")
        elif t == "set_osc_smoothing":
            self.viz_osc_smoothing = float(v); self.viz_osc_smooth_slider.set_value(self.viz_osc_smoothing)

        # --- Effetti ---
        elif t == "set_peakcap_enabled":
            self.fx_peakcap_enabled = bool(v); self.fx_peakcap_option.set_value("On" if self.fx_peakcap_enabled else "Off")
        elif t == "set_peakcap_color":
            self.fx_peakcap_color = tuple(v); self.fx_peakcap_color_picker.set_color(self.fx_peakcap_color)
        elif t == "set_peakcap_hold":
            self.fx_peakcap_hold = float(v); self.fx_peakcap_hold_slider.set_value(self.fx_peakcap_hold * 1000.0)  # s → ms
        elif t == "set_peakcap_fall":
            self.fx_peakcap_fall = float(v); self.fx_peakcap_fall_slider.set_value(self.fx_peakcap_fall)
        elif t == "set_bloom_enabled":
            self.fx_bloom_enabled = bool(v); self.fx_bloom_option.set_value("On" if self.fx_bloom_enabled else "Off")
        elif t == "set_bloom_intensity":
            self.fx_bloom_intensity = float(v); self.fx_bloom_slider.set_value(self.fx_bloom_intensity)
        elif t == "set_beat_enabled":
            self.fx_beat_enabled = bool(v); self.fx_beat_option.set_value("On" if self.fx_beat_enabled else "Off")
        elif t == "set_beat_intensity":
            self.fx_beat_intensity = float(v); self.fx_beat_intensity_slider.set_value(self.fx_beat_intensity)
        elif t == "set_beat_sensitivity":
            self.fx_beat_sensitivity = float(v); self.fx_beat_sens_slider.set_value(self.fx_beat_sensitivity)
```

- [ ] **Step 6: Verify by running the app (Windows display)**

Run `uv run python main.py`, start fullscreen, press **F1**, and for a control in **each** tab:
- Change it in the menu → alt-tab to the panel → the matching panel widget shows the **new** value.
- Change **Modalità** in the menu → the panel's Modalità combo and the conditionally-shown groups update.
- Close fullscreen (ESC), then start it again → the in-window changes are **still applied** (the renderer is rebuilt from the now-updated panel).

Expected: panel mirrors in-window changes; no infinite loop / flicker; values survive relaunch.

- [ ] **Step 7: Commit**

```bash
git add gui/main_window_gui.py
git commit -m "feat: sync in-window menu changes back to the PySide6 panel"
```

---

### Task 7: Configurable toggle key — panel control, in-window control, profiles

Expose the toggle key as a panel control (so it's discoverable and saved in profiles), add it to the in-window menu, and include it in profile save/load.

**Files:**
- Modify: `gui/main_window_gui.py`, `thread/imgui_overlay.py`

- [ ] **Step 1: Import the key names and init the state (GUI)**

In `gui/main_window_gui.py`, add the import near the other `thread`/`gui` imports:
```python
from thread.menu_keys import TOGGLE_KEY_NAMES
```
Then, next to `self.bars_color_mode = 0` (line ~96, with the other GUI state defaults), add:
```python
        self.menu_toggle_key = "F1"   # tasto che apre/chiude il menù in-window
```

- [ ] **Step 2: Add the panel control + its callback**

In `_build_visualization_page` (lines 343-380), after the `self.bars_alpha_slider = SliderCustomFrame(...)` block (line 360-364) and before `resa_widgets = [...]`, create the control:
```python
        self.menu_toggle_option = OptionMenuCustomFrame(
            header_name='Tasto menù in-window:',
            values=TOGGLE_KEY_NAMES,
            initial_value=self.menu_toggle_key,
            command=self.update_menu_toggle_key,
        )
```
Then add a group for it in the returned `_page(...)`:
```python
        return self._page(
            self._group("Modalità", self.viz_mode_option),
            self._group("Sfondo", self.file_picker, self.alpha_slider),
            self._group("Resa", *resa_widgets),
            self._group("Menù in-window", self.menu_toggle_option),
        )
```
Add the callback method (near the other `update_*` methods, e.g. after `update_bars_alpha`):
```python
    def update_menu_toggle_key(self, label: str):
        self.menu_toggle_key = label
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_menu_toggle_key", "value": label})
```

- [ ] **Step 3: Pass the toggle key to the renderer**

In the `EqualizerOpenGLThread(...)` construction, add `menu_toggle_key=self.menu_toggle_key,` (e.g. right before `on_setting_changed=...` added in Task 6):
```python
                monitor=self.selected_monitor,
                menu_toggle_key=self.menu_toggle_key,
                on_setting_changed=self.inwindow_setting_changed,
                window_close_callback=self.opengl_window_closed,
            )
```

- [ ] **Step 4: Reverse-sync branch for the toggle key**

In `_on_setting_changed` (Task 6), add a branch (e.g. after `set_beat_sensitivity`):
```python
        elif t == "set_menu_toggle_key":
            self.menu_toggle_key = str(v); self.menu_toggle_option.set_value(self.menu_toggle_key)
```

- [ ] **Step 5: Profile save**

In `_collect_profile` (lines 1000-1041), add to the returned dict (e.g. after the `"osc_smoothing"` entry):
```python
            "menu_toggle_key": self.menu_toggle_key,
```

- [ ] **Step 6: Profile load**

In `_apply_profile`, in the Visualizzazione section (near `self.viz_mode_option.set_value(viz_label)`, line 1073), add (using `.get` for backward compatibility with older profiles):
```python
        self.menu_toggle_key = str(s.get("menu_toggle_key", "F1"))
        self.menu_toggle_option.set_value(self.menu_toggle_key)
        if push:
            q.put({"type": "set_menu_toggle_key", "value": self.menu_toggle_key})
```

- [ ] **Step 7: Add the toggle-key combo to the in-window menu**

In `thread/imgui_overlay.py`, add the import at the top (it's already imported in Task 3 — confirm `from thread.menu_keys import TOGGLE_KEY_NAMES` is present). In `_build_ui`, in the **Visualizzazione** tab (after the "Opacità barre" slider, before `imgui.end_tab_item()`), add:
```python
                try:
                    key_idx = TOGGLE_KEY_NAMES.index(r._menu_toggle_key)
                except ValueError:
                    key_idx = 0
                changed, key_idx = imgui.combo("Tasto menù", key_idx, TOGGLE_KEY_NAMES)
                if changed:
                    self._emit("set_menu_toggle_key", TOGGLE_KEY_NAMES[key_idx])
```

- [ ] **Step 8: Verify profile round-trip headless**

Run: `uv run python tests/test_profiles.py`
Expected: prints `OK`. If it fails because of the new `menu_toggle_key` key (e.g. an exact-keys assertion or a sample profile dict), add `menu_toggle_key` (value `"F1"`) to that test's expected keys / sample profile, then re-run until it prints `OK`.

- [ ] **Step 9: Verify by running the app (Windows display)**

Run `uv run python main.py`:
- In the panel, set **Tasto menù in-window** to e.g. `F2`, start fullscreen → **F2** now toggles the menu (F1 no longer does).
- Open the menu → its Visualizzazione tab shows a **Tasto menù** combo; change it → the panel's combo updates (sync) and the new key toggles the menu.
- Save a profile, change the key, load the profile back → the key (and all other settings) restore; the panel combo reflects it.

Expected: configurable key works from both panel and menu, survives save/load.

- [ ] **Step 10: Commit**

```bash
git add gui/main_window_gui.py thread/imgui_overlay.py tests/test_profiles.py
git commit -m "feat: configurable in-window menu toggle key (panel, menu, profiles)"
```

---

### Task 8: Bundle `imgui_bundle` in the PyInstaller spec

`imgui_bundle` has a native extension + data files PyInstaller's static analysis misses; `collect_all` pulls them in so frozen Windows/Linux builds run.

**Files:**
- Modify: `sound-wave.spec`

- [ ] **Step 1: Add `imgui_bundle` to the collect_all loop**

In `sound-wave.spec`, change the collect loop tuple from:
```python
for _pkg in ('qdarktheme', 'imageio_ffmpeg', 'glfw'):
```
to:
```python
for _pkg in ('qdarktheme', 'imageio_ffmpeg', 'glfw', 'imgui_bundle'):
```

- [ ] **Step 2: Verify the spec parses (and, if possible, builds)**

Run (Windows or Linux with the synced env):
```bash
uv run --with pyinstaller pyinstaller --noconfirm sound-wave.spec
```
Expected: the build completes and `dist/sound wave/` is produced with no `imgui_bundle`-related analysis errors. (Running the frozen artifact to confirm the menu opens is the ultimate check — CI builds but never runs it.)

- [ ] **Step 3: Commit**

```bash
git add sound-wave.spec
git commit -m "build: bundle imgui_bundle in PyInstaller spec for frozen builds"
```

---

## Final verification

- [ ] On a Windows display, run `uv run python main.py` and exercise the full flow end-to-end: launch fullscreen, toggle the menu, change at least one control per tab, confirm the panel mirrors them, confirm relaunch preserves them, confirm profile save/load, confirm ESC semantics (close menu vs exit) and cursor show/hide.
- [ ] Run all headless tests: `uv run python tests/test_menu_keys.py && uv run python tests/test_frame_time_stats.py && uv run python tests/test_profiles.py` → each prints `OK`.
- [ ] (Optional) Produce the frozen build and run the artifact to confirm the menu works bundled.

## Notes / non-obvious points

- **No widget changes needed:** `SliderCustomFrame.set_value`, `OptionMenuCustomFrame.set_value` already block signals; `ColorPickerFrame.set_color` does not fire its command. So the reverse-sync setters are inherently silent — the spec's "add a silent setter" item is already satisfied by the existing API.
- **Why the echo on viz/color mode is safe:** `process_control_message` never calls `on_change`; only the overlay's `_emit` does. So a re-pushed `set_viz_mode`/`set_color_mode` is applied by the renderer with no further reverse emit → no loop.
- **Draw order:** the overlay draws after `_render` (which composites bloom to the default framebuffer, leaving FBO 0 bound) and before `swap_buffers`. The ImGui backend saves/restores GL state; the next frame's `_render` rebinds everything anyway.
- **Excluded by design:** monitor selector (can't change without recreating the window) and background file-picker (needs a file dialog) stay panel-only.
