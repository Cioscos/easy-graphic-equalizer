# Aspetto barre OpenGL (Blocco 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rendere personalizzabili colore (4 modalità) e forma/disposizione (larghezza, cime arrotondate, ancoraggio specchiato, ordine simmetrico) delle barre nel solo renderer fullscreen OpenGL.

**Architecture:** Tutto si innesta sullo shader instanced delle barre esistente in `thread/opengl_thread.py`: nuove `uniform`/`varying` nel programma barre, nuovi messaggi sulla control-queue `GUI→renderer`, e un nuovo tab "🎛️ Barre" nella GUI PySide6. Una sola funzione pura (`arrange_symmetric`) gestisce l'ordine simmetrico lato CPU. I default preservano esattamente il look attuale.

**Tech Stack:** Python 3.12, PyOpenGL (GLSL 3.3 core), GLFW, NumPy, PySide6 (`QColorDialog`), `uv`.

> **Convenzioni di progetto (vincolanti):**
> - **Nessuna test suite.** Verifica della logica pura con `uv run python` (headless, WSL2 usa `.venv-linux` via `UV_PROJECT_ENVIRONMENT`); verifica visiva eseguendo l'app **su Windows** (display GL reale).
> - **Commit:** utente `Cioscos`, **mai** trailer `Co-Authored-By`.
> - Branch di lavoro: `feature/opengl-bars-appearance` (già creato; lo spec è già committato lì).

**Spec di riferimento:** `docs/superpowers/specs/2026-06-17-aspetto-barre-opengl-design.md`

---

## File Structure

- **Modify** `thread/opengl_thread.py` — shader (vertex+fragment), uniform/varying, attributi di stato + default, parametri costruttore, `process_control_message`, `_render`, `run()` (salva dimensione framebuffer), nuove funzioni `arrange_symmetric` (module-level) e `_display_heights`.
- **Create** `gui/color_picker_frame.py` — widget `ColorPickerFrame` + funzioni pure `hex_to_rgb01` / `rgb01_to_hex`.
- **Modify** `gui/main_window_gui.py` — `_build_bars_tab`, callback di inoltro, attributi di stato GUI, popolamento costruttore in `fullscreen_command`.
- **Modify** `CLAUDE.md` — documentare nuove uniform/messaggi; (e una riga di help nel testo della GUI).

---

## Task 1: Funzione pura `arrange_symmetric`

Riordino dell'array di altezze per la disposizione simmetrica: bassi al centro, acuti specchiati ai bordi (raddoppia le barre).

**Files:**
- Modify: `thread/opengl_thread.py` (aggiungere funzione a livello di modulo, dopo le costanti, prima della classe)

- [ ] **Step 1: Scrivere la verifica (fallisce: funzione assente)**

```bash
uv run python - <<'PY'
import numpy as np
from thread.opengl_thread import arrange_symmetric
out = arrange_symmetric(np.array([1, 2, 3], dtype=np.float32))
assert out.tolist() == [3, 2, 1, 1, 2, 3], out.tolist()
assert out.dtype == np.float32
# array vuoto e singolo elemento non devono esplodere
assert arrange_symmetric(np.array([], dtype=np.float32)).shape[0] == 0
assert arrange_symmetric(np.array([5], dtype=np.float32)).tolist() == [5, 5]
print("OK")
PY
```
Expected: `ImportError` / `AttributeError` (funzione non ancora definita).

- [ ] **Step 2: Implementare la funzione**

In `thread/opengl_thread.py`, subito dopo il blocco delle costanti `MAX_CATCHUP_FRAMES` (prima degli shader), aggiungere:

```python
def arrange_symmetric(heights: np.ndarray) -> np.ndarray:
    """
    Riordina le altezze per la disposizione simmetrica "bassi al centro":
    specchia lo spettro attorno al centro, così la banda più bassa finisce nei
    due bar centrali e la più alta ai due bordi. Raddoppia il numero di barre.

    Esempio: [b0, b1, b2] -> [b2, b1, b0, b0, b1, b2].

    Funzione pura (nessuno stato GL): testabile a sé.
    """
    if heights.shape[0] == 0:
        return heights
    return np.concatenate([heights[::-1], heights])
```

- [ ] **Step 3: Eseguire la verifica (deve passare)**

```bash
uv run python - <<'PY'
import numpy as np
from thread.opengl_thread import arrange_symmetric
out = arrange_symmetric(np.array([1, 2, 3], dtype=np.float32))
assert out.tolist() == [3, 2, 1, 1, 2, 3], out.tolist()
assert out.dtype == np.float32
assert arrange_symmetric(np.array([], dtype=np.float32)).shape[0] == 0
assert arrange_symmetric(np.array([5], dtype=np.float32)).tolist() == [5, 5]
print("OK")
PY
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Add arrange_symmetric helper for symmetric bar order"
```

---

## Task 2: Widget `ColorPickerFrame` + conversioni pure

**Files:**
- Create: `gui/color_picker_frame.py`

- [ ] **Step 1: Scrivere la verifica delle conversioni (fallisce: modulo assente)**

```bash
uv run python - <<'PY'
from gui.color_picker_frame import hex_to_rgb01, rgb01_to_hex
r, g, b = hex_to_rgb01("#ff8000")
assert abs(r - 1.0) < 1e-6 and abs(g - 0.50196) < 1e-3 and b == 0.0, (r, g, b)
assert hex_to_rgb01("3b6bff") == hex_to_rgb01("#3b6bff")   # tollera senza '#'
assert rgb01_to_hex((1.0, 0.0, 0.0)) == "#ff0000"
assert rgb01_to_hex((0.0, 0.50196, 1.0)) == "#0080ff"
print("OK")
PY
```
Expected: `ImportError` (modulo non ancora creato).

- [ ] **Step 2: Creare il widget e le funzioni pure**

Creare `gui/color_picker_frame.py`:

```python
from typing import Callable, Optional, Sequence, Tuple

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gui.theme import font_body

Rgb01 = Tuple[float, float, float]


def hex_to_rgb01(value: str) -> Rgb01:
    """'#rrggbb' (o 'rrggbb') -> (r, g, b) in [0,1]."""
    h = value.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)


def rgb01_to_hex(rgb: Rgb01) -> str:
    """(r, g, b) in [0,1] -> '#rrggbb'."""
    r, g, b = (max(0, min(255, int(round(c * 255.0)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


# Preset rapidi mostrati come piccoli swatch sotto il bottone principale.
_PRESETS = ["#22d36a", "#3bffd0", "#3b6bff", "#ff3bd0", "#ffae3b"]


class ColorPickerFrame(QWidget):
    """
    Etichetta + bottone-swatch che apre QColorDialog, più una riga di preset
    rapidi. Espone get_color()/set_color() in RGB normalizzato e invoca
    command(rgb) a ogni cambio (modello: BackgroundFilepickerFrame).
    """

    def __init__(
        self,
        parent=None,
        *,
        header_name: str = "ColorPickerFrame",
        initial_color: str = "#3b6bff",
        command: Optional[Callable[[Rgb01], None]] = None,
        presets: Optional[Sequence[str]] = None,
    ):
        super().__init__(parent)
        self._command = command
        self._rgb: Rgb01 = hex_to_rgb01(initial_color)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(3)

        header = QLabel(header_name)
        header.setFont(font_body())
        layout.addWidget(header)

        self._swatch = QPushButton()
        self._swatch.setFixedHeight(26)
        self._swatch.clicked.connect(self._open_dialog)
        layout.addWidget(self._swatch)

        presets_row = QHBoxLayout()
        presets_row.setSpacing(4)
        for hex_value in (presets or _PRESETS):
            btn = QPushButton()
            btn.setFixedSize(22, 18)
            btn.setStyleSheet(f"background-color: {hex_value}; border: 1px solid #555; border-radius: 4px;")
            btn.clicked.connect(lambda _checked=False, h=hex_value: self._apply(hex_to_rgb01(h)))
            presets_row.addWidget(btn)
        presets_row.addStretch(1)
        layout.addLayout(presets_row)

        self._refresh_swatch()

    def _open_dialog(self) -> None:
        r, g, b = self._rgb
        initial = QColor.fromRgbF(r, g, b)
        chosen = QColorDialog.getColor(initial, self, "Scegli un colore")
        if chosen.isValid():
            self._apply((chosen.redF(), chosen.greenF(), chosen.blueF()))

    def _apply(self, rgb: Rgb01) -> None:
        self._rgb = rgb
        self._refresh_swatch()
        if self._command is not None:
            self._command(self._rgb)

    def _refresh_swatch(self) -> None:
        self._swatch.setStyleSheet(
            f"background-color: {rgb01_to_hex(self._rgb)}; border: 1px solid #555; border-radius: 4px;"
        )

    def get_color(self) -> Rgb01:
        """Colore corrente in RGB normalizzato [0,1]."""
        return self._rgb

    def set_color(self, rgb: Rgb01) -> None:
        """Imposta il colore senza invocare la callback."""
        self._rgb = rgb
        self._refresh_swatch()
```

- [ ] **Step 3: Eseguire la verifica delle conversioni (deve passare)**

```bash
uv run python - <<'PY'
from gui.color_picker_frame import hex_to_rgb01, rgb01_to_hex
r, g, b = hex_to_rgb01("#ff8000")
assert abs(r - 1.0) < 1e-6 and abs(g - 0.50196) < 1e-3 and b == 0.0, (r, g, b)
assert hex_to_rgb01("3b6bff") == hex_to_rgb01("#3b6bff")
assert rgb01_to_hex((1.0, 0.0, 0.0)) == "#ff0000"
assert rgb01_to_hex((0.0, 0.50196, 1.0)) == "#0080ff"
print("OK")
PY
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add gui/color_picker_frame.py
git commit -m "Add ColorPickerFrame widget with hex<->rgb01 helpers"
```

---

## Task 3: Shader colore (4 modalità) — default invariato

Aggiunge le 4 modalità colore allo shader delle barre, i parametri costruttore/attributi relativi al colore e l'invio delle uniform. Con i default (`color_mode=0`, soglie 0.5/0.8) il rendering resta identico a oggi.

**Files:**
- Modify: `thread/opengl_thread.py` (shader sources, costruttore, `init_gl_objects`, `_render`)

- [ ] **Step 1: Sostituire il vertex shader delle barre**

Sostituire l'intero blocco `_BARS_VERTEX_SRC = """..."""` con:

```python
_BARS_VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec2 aUnitPos;   // quad unitario: x,y in [0,1]
layout(location = 1) in float aHeight;   // altezza normalizzata per-istanza [0,1]
uniform float uNumBars;
uniform float uBarWidthFrac;
uniform float uMirror;       // 0 = dal basso, 1 = specchiato dal centro
out float vAmp;     // frazione altezza-finestra dal basso (Classico)
out float vBarY;    // frazione lungo la barra [0,1] (Gradiente / cap)
out float vBarX;    // frazione sulla larghezza [0,1] (cap arrotondato)
out float vBarT;    // indice barra normalizzato [0,1] (Spettro)
out float vBarH;    // altezza barra normalizzata (cap arrotondato)
void main() {
    float slot = 1.0 / uNumBars;
    float xLeft = (float(gl_InstanceID) + 0.5 * (1.0 - uBarWidthFrac)) * slot;
    float x = xLeft + aUnitPos.x * (slot * uBarWidthFrac);
    float yBottom = aUnitPos.y * aHeight;
    float yCenter = 0.5 + (aUnitPos.y - 0.5) * aHeight;
    float y = mix(yBottom, yCenter, uMirror);
    vAmp = yBottom;
    vBarY = aUnitPos.y;
    vBarX = aUnitPos.x;
    vBarT = (uNumBars > 1.0) ? float(gl_InstanceID) / (uNumBars - 1.0) : 0.0;
    vBarH = aHeight;
    gl_Position = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
"""
```
(`uMirror` non viene ancora impostato: il valore di default GL è 0 → comportamento "dal basso" invariato.)

- [ ] **Step 2: Sostituire il fragment shader delle barre**

Sostituire l'intero blocco `_BARS_FRAGMENT_SRC = """..."""` con:

```python
_BARS_FRAGMENT_SRC = """
#version 330 core
in float vAmp;
in float vBarY;
in float vBarT;
out vec4 FragColor;
uniform int uColorMode;       // 0 classico, 1 tinta, 2 gradiente, 3 spettro
uniform vec3 uColorA;         // tinta / base gradiente
uniform vec3 uColorB;         // cima gradiente
uniform float uGreenSplit;
uniform float uYellowSplit;
uniform float uBarsAlpha;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec3 col;
    if (uColorMode == 0) {                 // Classico: livello assoluto
        if (vAmp < uGreenSplit)            col = vec3(0.0, 1.0, 0.0);
        else if (vAmp < uYellowSplit)      col = vec3(1.0, 1.0, 0.0);
        else                               col = vec3(1.0, 0.0, 0.0);
    } else if (uColorMode == 1) {          // Tinta unica
        col = uColorA;
    } else if (uColorMode == 2) {          // Gradiente lungo la barra
        col = mix(uColorA, uColorB, vBarY);
    } else {                               // Spettro per frequenza
        col = hsv2rgb(vec3(vBarT * 0.83, 0.9, 1.0));
    }
    FragColor = vec4(col, uBarsAlpha);
}
"""
```

- [ ] **Step 3: Aggiungere le costanti di default colore**

Subito sotto `TILT_DB_PER_OCT = 0.0` (blocco di taratura), aggiungere:

```python
# --- Colore barre (Blocco 1). Default usati solo nelle modalità tinta/gradiente. ---
DEFAULT_COLOR_A = (0.231, 0.420, 1.0)   # blu  (#3b6bff): tinta / base gradiente
DEFAULT_COLOR_B = (1.0, 0.231, 0.816)   # magenta (#ff3bd0): cima gradiente
```

- [ ] **Step 4: Aggiungere i parametri colore al costruttore**

In `__init__`, nella firma, aggiungere (subito dopo `tilt_db_per_oct: float = TILT_DB_PER_OCT,` e prima di `window_close_callback`):

```python
                 color_mode: int = 0,
                 color_a: tuple = DEFAULT_COLOR_A,
                 color_b: tuple = DEFAULT_COLOR_B,
                 green_split: float = GREEN_SPLIT,
                 yellow_split: float = YELLOW_SPLIT,
```

E nel corpo di `__init__`, subito dopo `self._tilt_db_per_oct = tilt_db_per_oct`, aggiungere:

```python
        # --- Stato colore barre (Blocco 1). Default = look classico attuale. ---
        self._color_mode = int(color_mode)
        self._color_a = tuple(color_a)
        self._color_b = tuple(color_b)
        self._green_split = float(green_split)
        self._yellow_split = float(yellow_split)
```

- [ ] **Step 5: Registrare le uniform colore in `init_gl_objects`**

Sostituire il dizionario `self._bars_uniforms = {...}` con:

```python
        self._bars_uniforms = {
            name: glGetUniformLocation(self._bars_program, name)
            for name in ("uNumBars", "uBarWidthFrac", "uGreenSplit", "uYellowSplit",
                         "uBarsAlpha", "uMirror", "uColorMode", "uColorA", "uColorB")
        }
```

- [ ] **Step 6: Inviare le uniform colore in `_render`**

Sostituire il blocco che parte da `glUseProgram(self._bars_program)` fino a `glUniform1f(self._bars_uniforms["uBarsAlpha"], ...)` con:

```python
        glUseProgram(self._bars_program)
        glUniform1f(self._bars_uniforms["uNumBars"], float(num_bars))
        glUniform1f(self._bars_uniforms["uBarWidthFrac"], BAR_WIDTH_FRAC)
        glUniform1f(self._bars_uniforms["uGreenSplit"], self._green_split)
        glUniform1f(self._bars_uniforms["uYellowSplit"], self._yellow_split)
        glUniform1f(self._bars_uniforms["uBarsAlpha"], float(self._bars_alpha))
        glUniform1i(self._bars_uniforms["uColorMode"], int(self._color_mode))
        glUniform3f(self._bars_uniforms["uColorA"], *self._color_a)
        glUniform3f(self._bars_uniforms["uColorB"], *self._color_b)
```

- [ ] **Step 7: Verifica import + non-regressione headless**

```bash
uv run python - <<'PY'
import queue
from thread.opengl_thread import EqualizerOpenGLThread
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
assert t._color_mode == 0 and t._green_split == 0.5 and t._yellow_split == 0.8
print("OK")
PY
```
Expected: `OK` (gli shader compilano solo a contesto attivo, ma l'import e la costruzione non devono rompersi).

- [ ] **Step 8: Verifica visiva (Windows, display reale)**

Run: `uv run python main.py` → seleziona un device → **Fullscreen**.
Expected: le barre rendono **identiche a prima** (verde/giallo/rosso per livello). Chiudi con ESC.

- [ ] **Step 9: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Add 4 bar color modes to OpenGL shader (default unchanged)"
```

---

## Task 4: Messaggi di controllo colore

**Files:**
- Modify: `thread/opengl_thread.py` (`process_control_message`)

- [ ] **Step 1: Verifica (fallisce: i messaggi non mutano ancora lo stato)**

```bash
uv run python - <<'PY'
import queue
from thread.opengl_thread import EqualizerOpenGLThread
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
t.process_control_message({"type": "set_color_mode", "value": 2})
assert t._color_mode == 2, t._color_mode
t.process_control_message({"type": "set_color_a", "value": (0.1, 0.2, 0.3)})
assert t._color_a == (0.1, 0.2, 0.3)
t.process_control_message({"type": "set_color_b", "value": (0.4, 0.5, 0.6)})
assert t._color_b == (0.4, 0.5, 0.6)
# clamp soglie: green <= yellow sempre
t.process_control_message({"type": "set_green_split", "value": 0.9})
t.process_control_message({"type": "set_yellow_split", "value": 0.3})
assert t._green_split <= t._yellow_split, (t._green_split, t._yellow_split)
print("OK")
PY
```
Expected: `AssertionError`/`KeyError` (rami non gestiti → `_color_mode` resta 0).

- [ ] **Step 2: Aggiungere i rami in `process_control_message`**

Subito prima del ramo `elif message_type == 'set_image':`, aggiungere:

```python
        elif message_type == 'set_color_mode':
            self._color_mode = int(message['value'])

        elif message_type == 'set_color_a':
            self._color_a = tuple(message['value'])

        elif message_type == 'set_color_b':
            self._color_b = tuple(message['value'])

        elif message_type == 'set_green_split':
            # green_split è la soglia inferiore: clamp [0,1] e <= yellow_split.
            v = min(max(float(message['value']), 0.0), 1.0)
            self._green_split = min(v, self._yellow_split)

        elif message_type == 'set_yellow_split':
            # yellow_split è la soglia superiore: clamp [0,1] e >= green_split.
            v = min(max(float(message['value']), 0.0), 1.0)
            self._yellow_split = max(v, self._green_split)
```

- [ ] **Step 3: Eseguire la verifica (deve passare)**

```bash
uv run python - <<'PY'
import queue
from thread.opengl_thread import EqualizerOpenGLThread
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
t.process_control_message({"type": "set_color_mode", "value": 2})
assert t._color_mode == 2
t.process_control_message({"type": "set_color_a", "value": (0.1, 0.2, 0.3)})
t.process_control_message({"type": "set_color_b", "value": (0.4, 0.5, 0.6)})
assert t._color_a == (0.1, 0.2, 0.3) and t._color_b == (0.4, 0.5, 0.6)
t.process_control_message({"type": "set_green_split", "value": 0.9})
t.process_control_message({"type": "set_yellow_split", "value": 0.3})
assert t._green_split <= t._yellow_split, (t._green_split, t._yellow_split)
print("OK")
PY
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Handle color control messages in OpenGL renderer"
```

---

## Task 5: GUI — tab "🎛️ Barre" (colore) end-to-end

**Files:**
- Modify: `gui/main_window_gui.py`

- [ ] **Step 1: Import del nuovo widget**

Dopo `from gui.equalizer_preview_widget import EqualizerPreviewWidget`, aggiungere:

```python
from gui.color_picker_frame import ColorPickerFrame
```

- [ ] **Step 2: Mappa modalità + stato GUI colore**

Sotto le costanti di modulo (dopo `_THEME_MODE = {...}`), aggiungere:

```python
# Etichette del menu "Modalità colore" -> intero uColorMode del renderer.
_COLOR_MODE_TO_INT = {"Classico": 0, "Tinta unica": 1, "Gradiente": 2, "Spettro": 3}
```

In `__init__`, dopo `self.bars_alpha = 1.0  # ...`, aggiungere lo stato GUI (sorgente di verità passata al costruttore del thread, come `bg_alpha`):

```python
        # Stato GUI dell'aspetto barre (Blocco 1). Default = look attuale.
        self.bars_color_mode = 0
        self.bars_color_a = (0.231, 0.420, 1.0)
        self.bars_color_b = (1.0, 0.231, 0.816)
        self.bars_green_split = 0.5
        self.bars_yellow_split = 0.8
```

- [ ] **Step 3: Registrare il tab "Barre"**

In `_build_settings_tabs`, dopo `self.tabview.addTab(self._build_advanced_tab(), "⚙️ Avanzate")`, aggiungere:

```python
        self.tabview.addTab(self._build_bars_tab(), "🎛️ Barre")
```

- [ ] **Step 4: Costruire il tab (sezione Colore)**

Aggiungere il metodo `_build_bars_tab` alla classe (lo estenderemo con la sezione Forma nel Task 9):

```python
    def _build_bars_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        v.addWidget(self._make_label("Colore", font_label()))

        self.bars_mode_option = OptionMenuCustomFrame(
            header_name='Modalità colore:',
            values=["Classico", "Tinta unica", "Gradiente", "Spettro"],
            initial_value="Classico",
            command=self.on_color_mode_changed,
        )
        v.addWidget(self.bars_mode_option)

        # Classico: due soglie regolabili
        self.bars_yellow_thr = SliderCustomFrame(
            header_name='Soglia giallo:',
            command=self.update_green_split,
            initial_value=self.bars_green_split,
        )
        v.addWidget(self.bars_yellow_thr)

        self.bars_red_thr = SliderCustomFrame(
            header_name='Soglia rosso:',
            command=self.update_yellow_split,
            initial_value=self.bars_yellow_split,
        )
        v.addWidget(self.bars_red_thr)

        # Tinta unica: un colore
        self.bars_solid_color = ColorPickerFrame(
            header_name='Colore barre:',
            initial_color='#22d36a',
            command=self.update_color_a,
        )
        v.addWidget(self.bars_solid_color)

        # Gradiente: due colori
        self.bars_grad_base = ColorPickerFrame(
            header_name='Colore base:',
            initial_color='#3b6bff',
            command=self.update_color_a,
        )
        v.addWidget(self.bars_grad_base)

        self.bars_grad_top = ColorPickerFrame(
            header_name='Colore cima:',
            initial_color='#ff3bd0',
            command=self.update_color_b,
        )
        v.addWidget(self.bars_grad_top)

        v.addStretch(1)

        # Imposta la visibilità condizionale iniziale (modalità = Classico)
        self.on_color_mode_changed("Classico")
        return tab
```

- [ ] **Step 5: Callback di visibilità condizionale + inoltro colore**

Aggiungere i metodi (vicino a `update_bars_alpha`):

```python
    def on_color_mode_changed(self, label: str):
        """Mostra solo i controlli colore pertinenti e inoltra la modalità."""
        classico = label == "Classico"
        tinta = label == "Tinta unica"
        grad = label == "Gradiente"
        self.bars_yellow_thr.setVisible(classico)
        self.bars_red_thr.setVisible(classico)
        self.bars_solid_color.setVisible(tinta)
        self.bars_grad_base.setVisible(grad)
        self.bars_grad_top.setVisible(grad)
        self.bars_color_mode = _COLOR_MODE_TO_INT.get(label, 0)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_color_mode", "value": self.bars_color_mode})

    def update_color_a(self, rgb):
        self.bars_color_a = rgb
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_color_a", "value": rgb})

    def update_color_b(self, rgb):
        self.bars_color_b = rgb
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_color_b", "value": rgb})

    def update_green_split(self, value):
        self.bars_green_split = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_green_split", "value": float(value)})

    def update_yellow_split(self, value):
        self.bars_yellow_split = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_yellow_split", "value": float(value)})
```

- [ ] **Step 6: Popolare il costruttore in `fullscreen_command`**

Nel ramo `else:` di `fullscreen_command`, nella chiamata `EqualizerOpenGLThread(...)`, aggiungere (subito dopo `tilt_db_per_oct=...,`):

```python
                color_mode=self.bars_color_mode,
                color_a=self.bars_color_a,
                color_b=self.bars_color_b,
                green_split=self.bars_green_split,
                yellow_split=self.bars_yellow_split,
```

- [ ] **Step 7: Smoke test import GUI (headless)**

```bash
QT_QPA_PLATFORM=offscreen uv run python -c "import gui.main_window_gui as m; assert m._COLOR_MODE_TO_INT['Spettro'] == 3; print('OK')" smoketest
```
Expected: `OK` — verifica l'import del modulo e la mappa modalità senza costruire la finestra (quindi senza enumerare i device audio). Nota: l'argomento finale `smoketest` serve come `sys.argv[1]`: su Linux `soundcard` lo legge all'import (`_infer_program_name`) e con `python -c` senza argomenti darebbe `IndexError`. Richiede comunque PySide6/soundcard importabili; in un ambiente senza backend audio (PulseAudio/PipeWire), salta questo step e affidati alla verifica visiva su Windows (Step 8).

- [ ] **Step 8: Verifica visiva (Windows)**

Run: `uv run python main.py` → device → **Fullscreen** → tab **🎛️ Barre**.
Expected: cambiando "Modalità colore" compaiono solo i controlli giusti; **Tinta unica** colora tutte le barre del colore scelto (e dal picker/preset cambia live); **Gradiente** sfuma base→cima lungo ogni barra; **Spettro** fa l'arcobaleno bassi→acuti; **Classico** + soglie sposta i confini verde/giallo/rosso.

- [ ] **Step 9: Commit**

```bash
git add gui/main_window_gui.py
git commit -m "Add Barre tab with live color controls (OpenGL)"
```

---

## Task 6: Shader forma (larghezza, cime arrotondate, ancoraggio) — default invariato

**Files:**
- Modify: `thread/opengl_thread.py` (fragment shader, costruttore, `init_gl_objects`, `_render`, `run()`)

- [ ] **Step 1: Aggiungere i parametri forma al costruttore**

In `__init__`, nella firma, dopo `yellow_split: float = YELLOW_SPLIT,` (aggiunti al Task 3) aggiungere:

```python
                 bar_width: float = BAR_WIDTH_FRAC,
                 rounded: bool = False,
                 mirror: bool = False,
                 band_order_symmetric: bool = False,
```

E nel corpo, dopo `self._yellow_split = float(yellow_split)`:

```python
        # --- Stato forma/disposizione barre (Blocco 1). Default = look attuale. ---
        self._bar_width_frac = float(bar_width)
        self._rounded = bool(rounded)
        self._mirror = bool(mirror)
        self._band_order_symmetric = bool(band_order_symmetric)
        # Dimensione framebuffer (px), serve al cap arrotondato; reale in run().
        self._fb_width = 1.0
        self._fb_height = 1.0
```

- [ ] **Step 2: Sostituire il fragment shader (aggiunge cap arrotondato)**

Sostituire l'intero `_BARS_FRAGMENT_SRC = """..."""` con la versione completa:

```python
_BARS_FRAGMENT_SRC = """
#version 330 core
in float vAmp;
in float vBarY;
in float vBarT;
in float vBarX;
in float vBarH;
out vec4 FragColor;
uniform int uColorMode;
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform float uGreenSplit;
uniform float uYellowSplit;
uniform float uBarsAlpha;
uniform float uRounded;       // 0/1: cima arrotondata
uniform float uNumBars;
uniform float uBarWidthFrac;
uniform vec2 uViewportPx;     // dimensione framebuffer in pixel

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec3 col;
    if (uColorMode == 0) {
        if (vAmp < uGreenSplit)            col = vec3(0.0, 1.0, 0.0);
        else if (vAmp < uYellowSplit)      col = vec3(1.0, 1.0, 0.0);
        else                               col = vec3(1.0, 0.0, 0.0);
    } else if (uColorMode == 1) {
        col = uColorA;
    } else if (uColorMode == 2) {
        col = mix(uColorA, uColorB, vBarY);
    } else {
        col = hsv2rgb(vec3(vBarT * 0.83, 0.9, 1.0));
    }

    float a = uBarsAlpha;
    if (uRounded > 0.5) {
        // Cappuccio semicircolare alla cima, corretto per aspect ratio.
        float wpx = (1.0 / uNumBars) * uBarWidthFrac * uViewportPx.x;  // larghezza barra (px)
        float hpx = max(vBarH * uViewportPx.y, 1.0);                   // altezza barra (px)
        float r = 0.5 * wpx;
        float dyTop = (1.0 - vBarY) * hpx;        // px sotto la cima (0 = cima)
        if (dyTop < r) {
            float dx = (vBarX - 0.5) * wpx;       // offset orizzontale dal centro (px)
            float dyc = r - dyTop;                // offset verticale dal centro del cappuccio
            float dist = sqrt(dx * dx + dyc * dyc);
            float aa = max(fwidth(dist), 1e-3);
            a *= 1.0 - smoothstep(r - aa, r + aa, dist);
        }
    }
    FragColor = vec4(col, a);
}
"""
```

- [ ] **Step 3: Registrare le uniform forma in `init_gl_objects`**

Sostituire il dizionario `self._bars_uniforms = {...}` con:

```python
        self._bars_uniforms = {
            name: glGetUniformLocation(self._bars_program, name)
            for name in ("uNumBars", "uBarWidthFrac", "uGreenSplit", "uYellowSplit",
                         "uBarsAlpha", "uMirror", "uColorMode", "uColorA", "uColorB",
                         "uRounded", "uViewportPx")
        }
```

- [ ] **Step 4: Salvare la dimensione del framebuffer in `run()`**

Subito dopo `glViewport(0, 0, fb_width, fb_height)` in `run()`, aggiungere:

```python
            self._fb_width, self._fb_height = fb_width, fb_height
```

- [ ] **Step 5: Inviare le uniform forma in `_render`**

Sostituire il blocco uniform delle barre (quello modificato al Task 3) con:

```python
        glUseProgram(self._bars_program)
        glUniform1f(self._bars_uniforms["uNumBars"], float(num_bars))
        glUniform1f(self._bars_uniforms["uBarWidthFrac"], float(self._bar_width_frac))
        glUniform1f(self._bars_uniforms["uGreenSplit"], self._green_split)
        glUniform1f(self._bars_uniforms["uYellowSplit"], self._yellow_split)
        glUniform1f(self._bars_uniforms["uBarsAlpha"], float(self._bars_alpha))
        glUniform1i(self._bars_uniforms["uColorMode"], int(self._color_mode))
        glUniform3f(self._bars_uniforms["uColorA"], *self._color_a)
        glUniform3f(self._bars_uniforms["uColorB"], *self._color_b)
        glUniform1f(self._bars_uniforms["uMirror"], 1.0 if self._mirror else 0.0)
        glUniform1f(self._bars_uniforms["uRounded"], 1.0 if self._rounded else 0.0)
        glUniform2f(self._bars_uniforms["uViewportPx"], float(self._fb_width), float(self._fb_height))
```

- [ ] **Step 6: Verifica costruzione headless**

```bash
uv run python - <<'PY'
import queue
from thread.opengl_thread import EqualizerOpenGLThread
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
assert t._bar_width_frac == 0.8 and t._rounded is False and t._mirror is False
assert t._band_order_symmetric is False
print("OK")
PY
```
Expected: `OK`

- [ ] **Step 7: Verifica visiva (Windows)**

Run: `uv run python main.py` → device → **Fullscreen**.
Expected: nessun cambiamento visivo rispetto a prima (default: larghezza 0.8, squadrate, dal basso).

- [ ] **Step 8: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Add rounded caps, mirror anchor and live width uniforms (default unchanged)"
```

---

## Task 7: Messaggi di controllo forma

**Files:**
- Modify: `thread/opengl_thread.py` (`process_control_message`)

- [ ] **Step 1: Verifica (fallisce: rami assenti)**

```bash
uv run python - <<'PY'
import queue
from thread.opengl_thread import EqualizerOpenGLThread
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
t.process_control_message({"type": "set_bar_width", "value": 0.4})
assert abs(t._bar_width_frac - 0.4) < 1e-6
t.process_control_message({"type": "set_rounded", "value": True}); assert t._rounded is True
t.process_control_message({"type": "set_bar_anchor", "value": True}); assert t._mirror is True
t.process_control_message({"type": "set_band_order", "value": True}); assert t._band_order_symmetric is True
print("OK")
PY
```
Expected: `AssertionError` (rami non gestiti).

- [ ] **Step 2: Aggiungere i rami in `process_control_message`**

Subito dopo il ramo `set_yellow_split` (Task 4), aggiungere:

```python
        elif message_type == 'set_bar_width':
            self._bar_width_frac = float(message['value'])

        elif message_type == 'set_rounded':
            self._rounded = bool(message['value'])

        elif message_type == 'set_bar_anchor':
            # True = specchiato dal centro, False = dal basso.
            self._mirror = bool(message['value'])

        elif message_type == 'set_band_order':
            # True = simmetrico (bassi al centro), False = standard.
            self._band_order_symmetric = bool(message['value'])
```

- [ ] **Step 3: Eseguire la verifica (deve passare)**

```bash
uv run python - <<'PY'
import queue
from thread.opengl_thread import EqualizerOpenGLThread
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
t.process_control_message({"type": "set_bar_width", "value": 0.4})
assert abs(t._bar_width_frac - 0.4) < 1e-6
t.process_control_message({"type": "set_rounded", "value": True}); assert t._rounded is True
t.process_control_message({"type": "set_bar_anchor", "value": True}); assert t._mirror is True
t.process_control_message({"type": "set_band_order", "value": True}); assert t._band_order_symmetric is True
print("OK")
PY
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Handle shape control messages in OpenGL renderer"
```

---

## Task 8: Applicare l'ordine simmetrico nel rendering

**Files:**
- Modify: `thread/opengl_thread.py` (nuovo `_display_heights`, chiamato in `_render`)

- [ ] **Step 1: Verifica del metodo (fallisce: metodo assente)**

```bash
uv run python - <<'PY'
import queue, numpy as np
from thread.opengl_thread import EqualizerOpenGLThread
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
h = np.array([1, 2, 3], dtype=np.float32)
assert t._display_heights(h).tolist() == [1, 2, 3]            # standard: identità
t._band_order_symmetric = True
assert t._display_heights(h).tolist() == [3, 2, 1, 1, 2, 3]   # simmetrico: raddoppia
print("OK")
PY
```
Expected: `AttributeError` (metodo non definito).

- [ ] **Step 2: Aggiungere `_display_heights` e chiamarlo in `_render`**

Aggiungere il metodo (subito prima di `_render`):

```python
    def _display_heights(self, heights: np.ndarray) -> np.ndarray:
        """
        Applica la disposizione delle barre prima dell'upload: identità in ordine
        standard, riordino simmetrico (bassi al centro, raddoppia) altrimenti.
        """
        if self._band_order_symmetric:
            return arrange_symmetric(heights)
        return heights
```

In `_render`, subito dopo `glClear(GL_COLOR_BUFFER_BIT)` e prima di `num_bars = int(heights.shape[0])`, inserire:

```python
        heights = self._display_heights(heights)
```

- [ ] **Step 3: Eseguire la verifica (deve passare)**

```bash
uv run python - <<'PY'
import queue, numpy as np
from thread.opengl_thread import EqualizerOpenGLThread
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
h = np.array([1, 2, 3], dtype=np.float32)
assert t._display_heights(h).tolist() == [1, 2, 3]
t._band_order_symmetric = True
assert t._display_heights(h).tolist() == [3, 2, 1, 1, 2, 3]
print("OK")
PY
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Apply symmetric band order in the OpenGL render path"
```

---

## Task 9: GUI — sezione Forma del tab "Barre" end-to-end

**Files:**
- Modify: `gui/main_window_gui.py`

- [ ] **Step 1: Stato GUI forma**

In `__init__`, dopo `self.bars_yellow_split = 0.8` (Task 5), aggiungere:

```python
        self.bars_width = 0.8
        self.bars_rounded = False
        self.bars_anchor_center = False
        self.bars_band_symmetric = False
```

- [ ] **Step 2: Aggiungere i controlli Forma in `_build_bars_tab`**

In `_build_bars_tab`, sostituire la riga finale `v.addStretch(1)` (e la chiamata successiva) con il blocco seguente, poi rimettere lo stretch e la chiamata di visibilità:

```python
        v.addWidget(self._make_label("Forma", font_label()))

        self.bars_width_slider = SliderCustomFrame(
            header_name='Larghezza barre:',
            command=self.update_bar_width,
            from_=0.1, to=1.0,
            initial_value=self.bars_width,
        )
        v.addWidget(self.bars_width_slider)

        self.bars_rounded_option = OptionMenuCustomFrame(
            header_name='Cime arrotondate:',
            values=["No", "Sì"],
            initial_value="No",
            command=self.update_rounded,
        )
        v.addWidget(self.bars_rounded_option)

        self.bars_anchor_option = OptionMenuCustomFrame(
            header_name='Ancoraggio:',
            values=["Dal basso", "Centro"],
            initial_value="Dal basso",
            command=self.update_anchor,
        )
        v.addWidget(self.bars_anchor_option)

        self.bars_order_option = OptionMenuCustomFrame(
            header_name='Ordine bande:',
            values=["Standard", "Simmetrico"],
            initial_value="Standard",
            command=self.update_band_order,
        )
        v.addWidget(self.bars_order_option)

        v.addStretch(1)
        self.on_color_mode_changed("Classico")
        return tab
```
(Eliminare il vecchio `v.addStretch(1)` / `self.on_color_mode_changed("Classico")` / `return tab` lasciati dal Task 5: devono comparire una sola volta, in fondo.)

- [ ] **Step 3: Callback di inoltro forma**

Aggiungere i metodi (vicino a `update_yellow_split`):

```python
    def update_bar_width(self, value):
        self.bars_width = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_bar_width", "value": float(value)})

    def update_rounded(self, label):
        self.bars_rounded = (label == "Sì")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_rounded", "value": self.bars_rounded})

    def update_anchor(self, label):
        self.bars_anchor_center = (label == "Centro")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_bar_anchor", "value": self.bars_anchor_center})

    def update_band_order(self, label):
        self.bars_band_symmetric = (label == "Simmetrico")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_band_order", "value": self.bars_band_symmetric})
```

- [ ] **Step 4: Popolare il costruttore in `fullscreen_command`**

Nella chiamata `EqualizerOpenGLThread(...)`, dopo `yellow_split=self.bars_yellow_split,` (Task 5), aggiungere:

```python
                bar_width=self.bars_width,
                rounded=self.bars_rounded,
                mirror=self.bars_anchor_center,
                band_order_symmetric=self.bars_band_symmetric,
```

- [ ] **Step 5: Smoke test import GUI (headless)**

```bash
QT_QPA_PLATFORM=offscreen uv run python -c "import gui.main_window_gui; print('OK')" smoketest
```
Expected: `OK` — cattura errori di sintassi/nome nel nuovo codice del tab. Nota: l'argomento `smoketest` serve come `sys.argv[1]` (vedi Task 5 Step 7). Richiede gli import disponibili; altrimenti verifica su Windows (Step 6).

- [ ] **Step 6: Verifica visiva (Windows)**

Run: `uv run python main.py` → device → **Fullscreen** → tab **🎛️ Barre** → sezione Forma.
Expected: **Larghezza** assottiglia/ingrossa le barre live; **Cime arrotondate = Sì** smussa la cima; **Ancoraggio = Centro** fa crescere le barre su+giù dal centro; **Ordine bande = Simmetrico** mette i bassi al centro specchiando verso i bordi (le barre raddoppiano). Provare combinazioni (es. Centro + Spettro, Simmetrico + Gradiente + arrotondate).

- [ ] **Step 7: Commit**

```bash
git add gui/main_window_gui.py
git commit -m "Add Barre tab shape controls (width, rounded, anchor, order)"
```

---

## Task 10: Documentazione e verifica finale

**Files:**
- Modify: `CLAUDE.md`
- Modify: `gui/help_window.py` (testo di aiuto)

- [ ] **Step 1: Aggiornare `CLAUDE.md`**

Nella sezione "Rendering (OpenGL 3.3 core profile + shaders)", aggiungere una frase che documenti le nuove uniform e i messaggi:

```markdown
The bars fragment shader supports four **color modes** (`uColorMode`: classic green/yellow/red with adjustable `uGreenSplit`/`uYellowSplit`, solid `uColorA`, vertical gradient `uColorA`→`uColorB`, frequency spectrum) and **shape** options: live `uBarWidthFrac`, rounded caps (`uRounded`, aspect-corrected via `uViewportPx`), and a center-mirror anchor (`uMirror`). Symmetric band order ("bass-centered", which doubles the on-screen bars) is a CPU reorder in `_display_heights`/`arrange_symmetric`, not a shader concern. All are OpenGL-only, driven by control messages `set_color_mode`/`set_color_a`/`set_color_b`/`set_green_split`/`set_yellow_split`/`set_bar_width`/`set_rounded`/`set_bar_anchor`/`set_band_order` and matching constructor params; defaults preserve the prior look. GUI controls live in the "🎛️ Barre" tab.
```

- [ ] **Step 2: Aggiungere una riga di help nella GUI**

In `gui/help_window.py`, individuare la stringa del testo di aiuto e aggiungere il paragrafo:

```
🎛️ Barre (solo fullscreen OpenGL): scegli la modalità colore (Classico con soglie regolabili, Tinta unica, Gradiente, Spettro per frequenza) e la forma (larghezza, cime arrotondate, ancoraggio dal basso o specchiato dal centro, ordine bande standard o simmetrico con i bassi al centro).
```

- [ ] **Step 3: Verifica finale headless completa**

```bash
uv run python - <<'PY'
import queue, numpy as np
from thread.opengl_thread import EqualizerOpenGLThread, arrange_symmetric
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
for msg in [
    {"type": "set_color_mode", "value": 3},
    {"type": "set_bar_width", "value": 0.5},
    {"type": "set_rounded", "value": True},
    {"type": "set_bar_anchor", "value": True},
    {"type": "set_band_order", "value": True},
]:
    t.process_control_message(msg)
assert (t._color_mode, t._rounded, t._mirror, t._band_order_symmetric) == (3, True, True, True)
assert arrange_symmetric(np.array([1, 2], dtype=np.float32)).tolist() == [2, 1, 1, 2]
print("OK")
PY
```
Expected: `OK`

- [ ] **Step 4: Verifica visiva finale (Windows)**

Run: `uv run python main.py`. Per ogni modalità colore e ogni opzione di forma, confermare il comportamento atteso e che i **default** rendano come prima del Blocco 1 (non-regressione). Verificare anche `set_image`/`set_video` ancora funzionanti (sfondo + barre personalizzate).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md gui/help_window.py
git commit -m "Document OpenGL bars appearance controls (Block 1)"
```

---

## Note finali

- **Non-regressione:** ogni task che tocca lo shader verifica che i default rendano come prima. È il criterio di accettazione principale dato l'assenza di test automatici visivi.
- **Interazioni:** ancoraggio (`uMirror`) e ordine bande (`_band_order_symmetric`) sono ortogonali e combinabili; il colore Classico usa `vAmp` (livello assoluto) così resta corretto anche in modalità specchiata.
- **WSL2:** le verifiche `uv run python - <<PY` girano headless (usano `.venv-linux`); le verifiche visive richiedono un display GL reale (Windows). `QT_QPA_PLATFORM=offscreen` serve solo alle verifiche di costruzione GUI.
- Al termine, valutare il merge del branch `feature/opengl-bars-appearance` (skill `superpowers:finishing-a-development-branch`).
