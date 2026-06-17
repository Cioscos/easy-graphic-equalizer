# Modalità di visualizzazione OpenGL (Blocco 3) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trasformare il renderer fullscreen OpenGL da "solo barre" a multi-modalità con un selettore: Barre (attuale, default), Spettro radiale, Oscilloscopio, Linea/Area.

**Architecture:** Tutto in `thread/opengl_thread.py`. Un attributo `self._mode` smista il disegno in `_draw_current_mode`, chiamato sia dal pass su schermo sia dal prepass bloom (→ bloom in ogni modalità). Il **Radiale riusa il programma instanced delle barre** con un ramo polare (`uRadial`/`uInnerRadius`/`uAspect`). Oscilloscopio e Linea/Area usano **un solo nuovo programma "strip"** alimentato da triangle-strip costruiti su CPU da **funzioni pure** (testabili headless). GUI: selettore in "🎨 Visualizzazione", tab "🎛️ Barre" → "🎛️ Forma" con controlli condizionali. Default `mode=Barre` ⇒ rendering identico a post-Blocco 2.

**Tech Stack:** Python 3.12, PyOpenGL (GLSL 3.3 core), GLFW, NumPy, PySide6, `uv`.

## Global Constraints

- **Solo renderer OpenGL.** L'anteprima in-window (`EqualizerPreviewWidget`) NON va toccata. Le callback GUI inoltrano solo all'`EqualizerOpenGLThread` (pattern `if self.equalizer_opengl_thread: …put(…)`).
- **Non-regressione:** con `mode=Barre` e i nuovi parametri ai default, il fullscreen rende **esattamente** come post-Blocco 2.
- **Nessuna test suite.** Logica pura → snippet `uv run python` headless (WSL2 `.venv-linux`); shader/GL/GUI → verifica visiva su Windows (la fa l'utente).
- **`uv` su WSL2:** prefissare **ogni** comando `uv` con `export UV_PROJECT_ENVIRONMENT=.venv-linux &&`. Il Bash non-interattivo può non sorgere `~/.bashrc`; un `uv` "nudo" rigenererebbe il `.venv` di Windows come venv Linux (memoria `wsl2-venv-and-gl-verification`).
- **`soundcard`/`sys.argv[1]`:** gli snippet che importano la GUI passano un arg fittizio (`… smoketest`) e usano `QT_QPA_PLATFORM=offscreen` (memoria `soundcard-headless-argv`).
- **Commit:** utente `Cioscos`, **mai** trailer `Co-Authored-By` né altre attribuzioni.
- **`CLAUDE.md` è gitignored** in questo repo: aggiornalo ma **non committarlo** (memoria `claude-md-untracked`).
- **Branch di lavoro:** `opengl-visualization-modes` (già creato; lo spec è già committato lì).

**Spec di riferimento:** `docs/superpowers/specs/2026-06-17-modalita-visualizzazione-opengl-design.md`

---

## File Structure

- **Modify** `thread/opengl_thread.py` — costanti + enum modalità; stato/param di mode, inner_radius, line_thickness, fill, osc_mirror; nuovi rami `process_control_message`; **funzioni pure** `trigger_align`/`build_spectrum_strip`/`build_waveform_strip`; estensione vertex shader barre (ramo polare); nuovo programma "strip" (shader + VAO/VBO + helper); `_draw_bars_instanced(radial=)`, `_draw_strip`, `_upload_strip`, `_upload_heights`, `_prepare_mode_geometry`, `_draw_current_mode`, `_draw_background`; `_render(heights, waveform, dt)`; `_render_bloom` mode-aware; waveform mono in `run()`; `_cleanup_gl`.
- **Modify** `gui/main_window_gui.py` — selettore "Modalità" in Visualizzazione; `on_viz_mode_changed`; rinomina `_build_bars_tab`→`_build_shape_tab` ("🎛️ Forma"); nuovi controlli di forma condizionali; peak-cap condizionale; callback di inoltro; popolamento costruttore in `fullscreen_command`.
- **Modify** `CLAUDE.md` — doc (locale, non committato).

---

## Task 1: Stato modalità, costanti, parametri costruttore e control message

**Files:** Modify `thread/opengl_thread.py` (costanti, costruttore, `process_control_message`)

**Interfaces:**
- Produces: costanti `MODE_BARS=0`, `MODE_RADIAL=1`, `MODE_OSCILLOSCOPE=2`, `MODE_LINE=3`; attributi `self._mode`, `self._inner_radius`, `self._line_thickness`, `self._fill`, `self._osc_mirror`; parametri costruttore `mode/inner_radius/line_thickness/fill/osc_mirror`; control message `set_viz_mode`/`set_inner_radius`/`set_line_thickness`/`set_fill`/`set_osc_mirror`.

- [ ] **Step 1: Costanti modalità + default**

In testa al file, dopo il blocco `BAR_WIDTH_FRAC/GREEN_SPLIT/YELLOW_SPLIT` (sezione "Aspetto delle barre") aggiungere:

```python
# --- Modalità di visualizzazione (Blocco 3) ---
MODE_BARS = 0          # barre verticali (attuale, default)
MODE_RADIAL = 1        # barre disposte in cerchio
MODE_OSCILLOSCOPE = 2  # forma d'onda nel tempo
MODE_LINE = 3          # spettro a linea / area riempita
OSC_GAIN = 2.5         # guadagno display dell'oscilloscopio (campione → frazione schermo)
DEFAULT_INNER_RADIUS = 0.18      # raggio del foro centrale (radiale), frazione del raggio max
DEFAULT_LINE_THICKNESS = 0.012   # spessore nastro (osc/linea), frazione altezza finestra
```

- [ ] **Step 2: Parametri costruttore**

Nella firma di `__init__`, dopo `bloom_intensity: float = BLOOM_INTENSITY,` e prima di `window_close_callback: Callable = None`:

```python
                 mode: int = MODE_BARS,
                 inner_radius: float = DEFAULT_INNER_RADIUS,
                 line_thickness: float = DEFAULT_LINE_THICKNESS,
                 fill: bool = True,
                 osc_mirror: bool = False,
```

Nel corpo di `__init__`, dopo il blocco "--- Bloom (Blocco 2) ---" (`self._bloom_intensity = ...`) aggiungere:

```python
        # --- Modalità di visualizzazione (Blocco 3). Default = Barre (look attuale). ---
        self._mode = int(mode)
        self._inner_radius = float(inner_radius)
        self._line_thickness = float(line_thickness)
        self._fill = bool(fill)
        self._osc_mirror = bool(osc_mirror)
        self._draw_count = 0          # vertici (strip) o istanze (barre/radiale) da disegnare
        self._bar_heights = np.zeros(self._n_bands, dtype=np.float32)  # altezze disposte (peak-cap)
```

- [ ] **Step 3: Attributi GL del programma strip (placeholder)**

Nel blocco "Oggetti OpenGL" di `__init__` (dove sono dichiarati `self._bloom_size = None` ecc.), aggiungere:

```python
        self._strip_program = None
        self._strip_vao = None
        self._strip_vbo = None
        self._strip_capacity = 0      # n. di vertici allocati nel VBO strip
        self._strip_uniforms = {}
```

- [ ] **Step 4: Verifica (fallisce: stato/rami assenti)**

```bash
export UV_PROJECT_ENVIRONMENT=.venv-linux && uv run python - <<'PY'
import queue
from thread.opengl_thread import EqualizerOpenGLThread, MODE_BARS
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
assert t._mode == MODE_BARS
t.process_control_message({"type":"set_viz_mode","value":2}); assert t._mode == 2
t.process_control_message({"type":"set_inner_radius","value":0.3}); assert abs(t._inner_radius-0.3) < 1e-6
t.process_control_message({"type":"set_line_thickness","value":0.02}); assert abs(t._line_thickness-0.02) < 1e-6
t.process_control_message({"type":"set_fill","value":False}); assert t._fill is False
t.process_control_message({"type":"set_osc_mirror","value":True}); assert t._osc_mirror is True
t2 = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue(),
                           mode=3, inner_radius=0.25, line_thickness=0.03, fill=False, osc_mirror=True)
assert t2._mode == 3 and abs(t2._inner_radius-0.25) < 1e-6 and t2._fill is False and t2._osc_mirror is True
print("OK")
PY
```
Expected: `AttributeError`/`KeyError` (stato o rami non ancora presenti).

- [ ] **Step 5: Aggiungere i rami in `process_control_message`**

Dopo il ramo `set_bloom_intensity` (Blocco 2) e prima di `set_image`:

```python
        elif message_type == 'set_viz_mode':
            self._mode = int(message['value'])

        elif message_type == 'set_inner_radius':
            self._inner_radius = float(message['value'])

        elif message_type == 'set_line_thickness':
            self._line_thickness = float(message['value'])

        elif message_type == 'set_fill':
            self._fill = bool(message['value'])

        elif message_type == 'set_osc_mirror':
            self._osc_mirror = bool(message['value'])
```

- [ ] **Step 6: Eseguire la verifica (deve passare)** — rilanciare lo Step 4. Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Add visualization-mode state, constructor params and control messages"
```

---

## Task 2: Funzioni pure di geometria (trigger + strip linea/area + strip oscilloscopio)

**Files:** Modify `thread/opengl_thread.py` (funzioni a livello di modulo, vicino a `arrange_symmetric`)

**Interfaces:**
- Produces: `trigger_align(waveform)->np.ndarray`; `build_spectrum_strip(heights, fill, thickness)->np.ndarray (2N,3)`; `build_waveform_strip(waveform, mirror, thickness, gain)->np.ndarray (2M,3)`. Tutte float32, layout `(x, y, mag)` per vertice, ordine adatto a `GL_TRIANGLE_STRIP`.

- [ ] **Step 1: Scrivere la verifica (fallisce: funzioni assenti)**

```bash
export UV_PROJECT_ENVIRONMENT=.venv-linux && uv run python - <<'PY'
import numpy as np
from thread.opengl_thread import trigger_align, build_spectrum_strip, build_waveform_strip

# trigger_align: parte dal primo zero-crossing in salita entro il primo quarto
w = np.array([-0.2,-0.1,0.3,0.5,0.2,-0.4,-0.1,0.2], dtype=np.float32)
a = trigger_align(w); assert abs(a[0]+0.1) < 1e-6 and a.shape[0] == 7
# nessun crossing → invariato
flat = np.linspace(-1,-0.5,8).astype(np.float32)
assert trigger_align(flat).shape[0] == 8

# build_spectrum_strip riempito: base/curva alternati, x in [0,1], mag=altezza
v = build_spectrum_strip(np.array([0.2,0.8],np.float32), True, 0.01)
assert v.shape == (4,3)
assert abs(v[0,1]-0.0) < 1e-6 and abs(v[1,1]-0.2) < 1e-6 and abs(v[3,1]-0.8) < 1e-6
assert abs(v[0,0]-0.0) < 1e-6 and abs(v[2,0]-1.0) < 1e-6
# linea (non riempito): nastro attorno alla curva
vl = build_spectrum_strip(np.array([0.5],np.float32), False, 0.2)
assert vl.shape == (2,3) and abs(vl[1,1]-0.6) < 1e-6 and abs(vl[0,1]-0.4) < 1e-6

# build_waveform_strip mono: mag=|sample|*gain*2, clamp
v2 = build_waveform_strip(np.array([0.0,0.2],np.float32), False, 0.02, 2.5)
assert v2.shape == (4,3) and abs(v2[0,2]-0.0) < 1e-6 and abs(v2[3,2]-1.0) < 1e-6
# specchiato: area tra 0.5±|s|
v3 = build_waveform_strip(np.array([0.2],np.float32), True, 0.02, 2.5)
assert v3.shape == (2,3) and abs(v3[0,1]-0.0) < 1e-6 and abs(v3[1,1]-1.0) < 1e-6
print("OK")
PY
```
Expected: `ImportError`/`AttributeError` (funzioni non definite).

- [ ] **Step 2: Implementare le tre funzioni**

Subito dopo `arrange_symmetric(...)` (prima del blocco "--- Shader GLSL ---") aggiungere:

```python
def trigger_align(waveform: np.ndarray) -> np.ndarray:
    """
    Allinea la finestra al primo zero-crossing in salita entro il primo quarto, per
    stabilizzare l'oscilloscopio (riduce lo scorrimento frame-su-frame). Se non trova
    un crossing, restituisce la finestra invariata. Funzione pura (nessuno stato GL).
    """
    n = int(waveform.shape[0])
    if n < 3:
        return waveform
    limit = max(1, n // 4)
    seg = waveform[:limit + 1]
    rising = (seg[:-1] <= 0.0) & (seg[1:] > 0.0)
    idx = np.nonzero(rising)[0]
    if idx.size == 0:
        return waveform
    return waveform[int(idx[0]):]


def build_spectrum_strip(heights: np.ndarray, fill: bool, thickness: float) -> np.ndarray:
    """
    Triangle strip per la modalità Linea/Area dalle ampiezze per banda.
    fill=True: area tra base (y=0) e curva. fill=False: nastro di spessore verticale
    'thickness' centrato sulla curva. Restituisce (2N, 3) float32 con (x, y, mag) per
    vertice: x in [0,1] (Spettro), mag = altezza banda (Classico/Gradiente). Ordine
    adatto a GL_TRIANGLE_STRIP. Funzione pura.
    """
    n = int(heights.shape[0])
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    xs = np.linspace(0.0, 1.0, n, dtype=np.float32) if n > 1 else np.array([0.5], dtype=np.float32)
    h = np.clip(heights.astype(np.float32), 0.0, 1.0)
    if fill:
        bot = np.zeros(n, dtype=np.float32)
        top = h
    else:
        half = 0.5 * float(thickness)
        top = np.clip(h + half, 0.0, 1.0)
        bot = np.clip(h - half, 0.0, 1.0)
    verts = np.empty((2 * n, 3), dtype=np.float32)
    verts[0::2, 0] = xs;  verts[0::2, 1] = bot;  verts[0::2, 2] = h
    verts[1::2, 0] = xs;  verts[1::2, 1] = top;  verts[1::2, 2] = h
    return verts


def build_waveform_strip(waveform: np.ndarray, mirror: bool,
                         thickness: float, gain: float) -> np.ndarray:
    """
    Triangle strip per l'oscilloscopio. mono: nastro di spessore 'thickness' attorno
    alla curva y = 0.5 + sample*gain. specchiato: area simmetrica tra 0.5 ± |sample|*gain.
    Restituisce (2M, 3) float32 (x, y, mag): mag = |sample|*gain*2 in [0,1]
    (Classico/Gradiente), x in [0,1] (Spettro). Funzione pura.
    """
    m = int(waveform.shape[0])
    if m == 0:
        return np.zeros((0, 3), dtype=np.float32)
    xs = np.linspace(0.0, 1.0, m, dtype=np.float32) if m > 1 else np.array([0.5], dtype=np.float32)
    s = np.clip(waveform.astype(np.float32) * float(gain), -0.5, 0.5)
    mag = np.clip(np.abs(s) * 2.0, 0.0, 1.0)
    if mirror:
        a = np.abs(s)
        top = np.clip(0.5 + a, 0.0, 1.0)
        bot = np.clip(0.5 - a, 0.0, 1.0)
    else:
        half = 0.5 * float(thickness)
        top = np.clip(0.5 + s + half, 0.0, 1.0)
        bot = np.clip(0.5 + s - half, 0.0, 1.0)
    verts = np.empty((2 * m, 3), dtype=np.float32)
    verts[0::2, 0] = xs;  verts[0::2, 1] = bot;  verts[0::2, 2] = mag
    verts[1::2, 0] = xs;  verts[1::2, 1] = top;  verts[1::2, 2] = mag
    return verts
```

- [ ] **Step 3: Eseguire la verifica (deve passare)** — rilanciare lo Step 1. Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Add pure geometry builders for strip modes (trigger, line/area, waveform)"
```

---

## Task 3: Estensione del programma barre per il Radiale

**Files:** Modify `thread/opengl_thread.py` (`_BARS_VERTEX_SRC`, `init_gl_objects` uniforms, `_draw_bars_instanced`)

**Interfaces:**
- Consumes: stato `self._inner_radius`, `self._fb_width/_fb_height`, `self._mirror`, `self._rounded` (Task 1 + esistenti).
- Produces: `_draw_bars_instanced(num_bars, viewport_px, radial=False)`; uniform `uRadial`/`uInnerRadius`/`uAspect` nel programma barre.

- [ ] **Step 1: Estendere il vertex shader delle barre**

Sostituire interamente `_BARS_VERTEX_SRC = """ … """` con (aggiunge il ramo polare; il resto è invariato):

```python
_BARS_VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec2 aUnitPos;   // quad unitario: x,y in [0,1]
layout(location = 1) in float aHeight;   // altezza normalizzata per-istanza [0,1]
uniform float uNumBars;
uniform float uBarWidthFrac;
uniform float uMirror;       // 0 = dal basso, 1 = specchiato dal centro
uniform float uRadial;       // 0 = cartesiano (barre), 1 = polare (radiale)
uniform float uInnerRadius;  // raggio del foro centrale [0,1] (radiale)
uniform float uAspect;       // fbW/fbH: corregge la x per cerchi tondi
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
    if (uRadial > 0.5) {
        // angolo centrale del raggio (parte dall'alto, senso orario) + larghezza angolare
        float ang = (float(gl_InstanceID) + 0.5) * slot * 6.28318530718;
        ang += (aUnitPos.x - 0.5) * slot * uBarWidthFrac * 6.28318530718;
        float r = uInnerRadius + aUnitPos.y * aHeight * (1.0 - uInnerRadius);
        float px = (r * sin(ang)) / uAspect;   // correzione aspect: cerchio tondo
        float py = r * cos(ang);
        gl_Position = vec4(px, py, 0.0, 1.0);
    } else {
        gl_Position = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    }
}
"""
```

- [ ] **Step 2: Registrare le nuove uniform nel dict**

In `init_gl_objects`, nella creazione di `self._bars_uniforms`, aggiungere i tre nomi alla tupla:

```python
        self._bars_uniforms = {
            name: glGetUniformLocation(self._bars_program, name)
            for name in ("uNumBars", "uBarWidthFrac", "uGreenSplit", "uYellowSplit",
                         "uBarsAlpha", "uMirror", "uColorMode", "uColorA", "uColorB",
                         "uRounded", "uViewportPx", "uRadial", "uInnerRadius", "uAspect")
        }
```

- [ ] **Step 3: Aggiornare `_draw_bars_instanced` con il parametro `radial`**

Sostituire interamente il metodo `_draw_bars_instanced`:

```python
    def _draw_bars_instanced(self, num_bars: int, viewport_px, radial: bool = False) -> None:
        """Imposta le uniform delle barre e disegna la passata instanced.
        radial=True attiva il ramo polare (uRadial) per la modalità Radiale; in radiale
        ancoraggio specchiato e cime arrotondate sono disattivati (math cartesiana)."""
        glUseProgram(self._bars_program)
        glUniform1f(self._bars_uniforms["uNumBars"], float(num_bars))
        glUniform1f(self._bars_uniforms["uBarWidthFrac"], float(self._bar_width_frac))
        glUniform1f(self._bars_uniforms["uGreenSplit"], self._green_split)
        glUniform1f(self._bars_uniforms["uYellowSplit"], self._yellow_split)
        glUniform1f(self._bars_uniforms["uBarsAlpha"], float(self._bars_alpha))
        glUniform1i(self._bars_uniforms["uColorMode"], int(self._color_mode))
        glUniform3f(self._bars_uniforms["uColorA"], *self._color_a)
        glUniform3f(self._bars_uniforms["uColorB"], *self._color_b)
        glUniform1f(self._bars_uniforms["uMirror"], 0.0 if radial else (1.0 if self._mirror else 0.0))
        glUniform1f(self._bars_uniforms["uRounded"], 0.0 if radial else (1.0 if self._rounded else 0.0))
        glUniform2f(self._bars_uniforms["uViewportPx"], float(viewport_px[0]), float(viewport_px[1]))
        glUniform1f(self._bars_uniforms["uRadial"], 1.0 if radial else 0.0)
        glUniform1f(self._bars_uniforms["uInnerRadius"], float(self._inner_radius))
        aspect = float(self._fb_width) / max(float(self._fb_height), 1.0)
        glUniform1f(self._bars_uniforms["uAspect"], aspect)
        glBindVertexArray(self._bars_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, num_bars)
        glBindVertexArray(0)
```

- [ ] **Step 4: Verifica costruzione headless (import + `__init__`)**

```bash
export UV_PROJECT_ENVIRONMENT=.venv-linux && uv run python -c "import queue; from thread.opengl_thread import EqualizerOpenGLThread; EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue()); print('construct OK')" 
```
Expected: `construct OK` (import del modulo riuscito; la compilazione shader avviene solo con contesto GL — verifica visiva su Windows o smoke EGL opzionale, vedi Note finali).

- [ ] **Step 5: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Extend bars shader with polar branch for radial mode"
```

---

## Task 4: Nuovo programma "strip" (shader + VAO/VBO + helper draw/upload)

**Files:** Modify `thread/opengl_thread.py` (shader sources, `init_gl_objects`, helper, `_cleanup_gl`)

**Interfaces:**
- Consumes: stato colore (`self._color_mode/_color_a/_color_b/_green_split/_yellow_split/_bars_alpha`), `self._strip_*` (Task 1).
- Produces: `_ensure_strip_capacity(n)`; `_upload_strip(verts)` (imposta `self._draw_count`); `_draw_strip()`.

- [ ] **Step 1: Sorgenti shader "strip"**

Dopo il blocco `_BG_FRAGMENT_SRC = """ … """` aggiungere:

```python
# --- Shader "strip": curve continue (oscilloscopio, linea/area) via triangle strip ---
# Vertici (x, y, mag): x,y in [0,1] (posizione); mag in [0,1] (magnitudine per la
# colorazione). Replica le 4 modalità colore del Blocco 1: Classico per mag, Gradiente
# verticale per y, Spettro per x, Tinta unica.
_STRIP_VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;    // (x, y) in [0,1]
layout(location = 1) in float aMag;   // magnitudine per la colorazione [0,1]
out float vX;
out float vY;
out float vMag;
void main() {
    vX = aPos.x;
    vY = aPos.y;
    vMag = aMag;
    gl_Position = vec4(aPos.x * 2.0 - 1.0, aPos.y * 2.0 - 1.0, 0.0, 1.0);
}
"""

_STRIP_FRAGMENT_SRC = """
#version 330 core
in float vX;
in float vY;
in float vMag;
out vec4 FragColor;
uniform int uColorMode;
uniform vec3 uColorA;
uniform vec3 uColorB;
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
    if (uColorMode == 0) {
        if (vMag < uGreenSplit)            col = vec3(0.0, 1.0, 0.0);
        else if (vMag < uYellowSplit)      col = vec3(1.0, 1.0, 0.0);
        else                               col = vec3(1.0, 0.0, 0.0);
    } else if (uColorMode == 1) {
        col = uColorA;
    } else if (uColorMode == 2) {
        col = mix(uColorA, uColorB, vY);
    } else {
        col = hsv2rgb(vec3(vX * 0.83, 0.9, 1.0));
    }
    FragColor = vec4(col, uBarsAlpha);
}
"""
```

- [ ] **Step 2: Compilare il programma e creare VAO/VBO in `init_gl_objects`**

In fondo a `init_gl_objects` (dopo i programmi bloom) aggiungere:

```python
        # --- Programma "strip" (oscilloscopio + linea/area): VBO (x,y,mag) interleaved ---
        self._strip_program = compileProgram(
            compileShader(_STRIP_VERTEX_SRC, GL_VERTEX_SHADER),
            compileShader(_STRIP_FRAGMENT_SRC, GL_FRAGMENT_SHADER),
        )
        self._strip_uniforms = {
            name: glGetUniformLocation(self._strip_program, name)
            for name in ("uColorMode", "uColorA", "uColorB",
                         "uGreenSplit", "uYellowSplit", "uBarsAlpha")
        }
        self._strip_vao = glGenVertexArrays(1)
        glBindVertexArray(self._strip_vao)
        self._strip_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._strip_vbo)
        stride = 3 * 4  # x, y, mag
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * 4))
        glVertexAttribDivisor(1, 0)
        self._strip_capacity = 0
        glBindVertexArray(0)
```

- [ ] **Step 3: Helper capacità + upload + draw**

Dopo `_ensure_peaks_capacity` aggiungere:

```python
    def _ensure_strip_capacity(self, n_verts: int) -> None:
        """Garantisce che il VBO strip contenga almeno n_verts vertici (3 float l'uno)."""
        if n_verts <= self._strip_capacity:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self._strip_vbo)
        glBufferData(GL_ARRAY_BUFFER, n_verts * 3 * 4, None, GL_STREAM_DRAW)
        self._strip_capacity = n_verts

    def _upload_strip(self, verts: np.ndarray) -> None:
        """Carica i vertici (M, 3) float32 nel VBO strip e imposta self._draw_count."""
        if verts.dtype != np.float32 or not verts.flags["C_CONTIGUOUS"]:
            verts = np.ascontiguousarray(verts, dtype=np.float32)
        n = int(verts.shape[0])
        self._draw_count = n
        if n == 0:
            return
        self._ensure_strip_capacity(n)
        glBindBuffer(GL_ARRAY_BUFFER, self._strip_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, verts.nbytes, verts)

    def _draw_strip(self) -> None:
        """Disegna la geometria strip corrente (triangle strip) col programma strip."""
        glUseProgram(self._strip_program)
        glUniform1i(self._strip_uniforms["uColorMode"], int(self._color_mode))
        glUniform3f(self._strip_uniforms["uColorA"], *self._color_a)
        glUniform3f(self._strip_uniforms["uColorB"], *self._color_b)
        glUniform1f(self._strip_uniforms["uGreenSplit"], self._green_split)
        glUniform1f(self._strip_uniforms["uYellowSplit"], self._yellow_split)
        glUniform1f(self._strip_uniforms["uBarsAlpha"], float(self._bars_alpha))
        glBindVertexArray(self._strip_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, int(self._draw_count))
        glBindVertexArray(0)
```

- [ ] **Step 4: Liberare le risorse strip in `_cleanup_gl`**

In `_cleanup_gl`: aggiungere `self._strip_vbo` all'elenco dei VBO da eliminare, `self._strip_vao` ai VAO, `self._strip_program` ai programmi; e azzerarli a fine metodo. Concretamente:

- nella riga `for vbo in (self._bars_quad_vbo, self._heights_vbo, self._bg_vbo, self._peaks_vbo):` aggiungere `, self._strip_vbo`;
- nella riga `for vao in (self._bars_vao, self._bg_vao, self._caps_vao):` aggiungere `, self._strip_vao`;
- nell'elenco `for prog in (...):` aggiungere `self._strip_program`;
- in fondo aggiungere: `self._strip_vbo = self._strip_vao = self._strip_program = None` e `self._strip_capacity = 0`.

- [ ] **Step 5: Verifica costruzione headless**

```bash
export UV_PROJECT_ENVIRONMENT=.venv-linux && uv run python -c "import queue; from thread.opengl_thread import EqualizerOpenGLThread; t=EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue()); assert hasattr(t,'_strip_program'); print('construct OK')"
```
Expected: `construct OK`

- [ ] **Step 6: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Add strip shader program and upload/draw helpers for continuous modes"
```

---

## Task 5: Integrazione — dispatch per modalità, render, bloom mode-aware, waveform nel loop

**Files:** Modify `thread/opengl_thread.py` (`_draw_background`, `_prepare_mode_geometry`, `_upload_heights`, `_draw_current_mode`, `_render`, `_render_bloom`, `run()`)

**Interfaces:**
- Consumes: tutto da Task 1–4 (`MODE_*`, builders, `_draw_bars_instanced(radial=)`, `_draw_strip`, `_upload_strip`).
- Produces: `_render(heights, waveform, dt)`; dispatch `_draw_current_mode(viewport_px)`; bloom mode-aware.

- [ ] **Step 1: Helper sfondo e upload altezze**

Subito **prima** di `_render` aggiungere due metodi (il primo è il blocco sfondo estratto da `_render`, invariato; il secondo fattorizza l'upload delle altezze):

```python
    def _draw_background(self) -> None:
        """Disegna lo sfondo (video se attivo, altrimenti immagine) con lo zoom del beat."""
        if self._video_active and self._video_texture:
            bg_tex, flip = self._video_texture, 1.0
        elif self._background_texture:
            bg_tex, flip = self._background_texture, 0.0
        else:
            return
        glUseProgram(self._bg_program)
        alpha = 1.0 if self._bg_alpha is None else float(self._bg_alpha)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, bg_tex)
        glUniform1i(self._bg_uniforms["uTexture"], 0)
        glUniform1f(self._bg_uniforms["uAlpha"], alpha)
        glUniform1f(self._bg_uniforms["uFlipV"], flip)
        glUniform1f(self._bg_uniforms["uBgZoom"], float(self._beat_pulse * self._beat_intensity))
        glBindVertexArray(self._bg_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

    def _upload_heights(self, heights: np.ndarray) -> None:
        """Carica le altezze per-istanza nel VBO barre e imposta self._draw_count."""
        if heights.dtype != np.float32 or not heights.flags["C_CONTIGUOUS"]:
            heights = np.ascontiguousarray(heights, dtype=np.float32)
        n = int(heights.shape[0])
        self._draw_count = n
        self._ensure_heights_capacity(n)
        glBindBuffer(GL_ARRAY_BUFFER, self._heights_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, heights.nbytes, heights)
```

- [ ] **Step 2: `_prepare_mode_geometry` e `_draw_current_mode`**

Subito dopo `_upload_heights` aggiungere:

```python
    def _prepare_mode_geometry(self, heights: np.ndarray, waveform) -> bool:
        """
        Costruisce e carica la geometria della modalità corrente; imposta self._draw_count
        e (per le modalità a banda) self._bar_heights. Ritorna False se non c'è nulla da
        disegnare (audio/bande vuoti).
        """
        if self._mode == MODE_OSCILLOSCOPE:
            if waveform is None or waveform.size == 0:
                self._draw_count = 0
                return False
            wf = trigger_align(np.ascontiguousarray(waveform, dtype=np.float32))
            verts = build_waveform_strip(wf, self._osc_mirror, self._line_thickness, OSC_GAIN)
            self._upload_strip(verts)
            return self._draw_count > 0

        # Modalità basate sulle bande: applica la disposizione (es. simmetrica).
        heights = self._display_heights(heights)
        if int(heights.shape[0]) < 1:
            self._draw_count = 0
            return False
        self._bar_heights = heights  # per il peak-cap (solo Barre)
        if self._mode == MODE_LINE:
            verts = build_spectrum_strip(np.ascontiguousarray(heights, dtype=np.float32),
                                         self._fill, self._line_thickness)
            self._upload_strip(verts)
        else:  # MODE_BARS o MODE_RADIAL: VBO altezze per-istanza
            self._upload_heights(heights)
        return self._draw_count > 0

    def _draw_current_mode(self, viewport_px) -> None:
        """Disegna la visualizzazione della modalità corrente (riusato da schermo e bloom)."""
        if self._mode == MODE_RADIAL:
            self._draw_bars_instanced(int(self._draw_count), viewport_px, radial=True)
        elif self._mode in (MODE_OSCILLOSCOPE, MODE_LINE):
            self._draw_strip()
        else:  # MODE_BARS
            self._draw_bars_instanced(int(self._draw_count), viewport_px, radial=False)
```

- [ ] **Step 3: Riscrivere `_render` con il dispatch**

Sostituire interamente il metodo `_render` (dalla firma fino al `return` finale) con:

```python
    def _render(self, heights: np.ndarray, waveform, dt: float) -> None:
        """
        Disegna un frame: pulisce, prepara la geometria della modalità corrente, disegna
        lo sfondo, poi la visualizzazione (dispatch per modalità), il peak-cap (solo Barre)
        e il composito bloom.

        Args:
            heights: ampiezze per banda [0,1] (float32).
            waveform: campioni mono [-1,1] per l'oscilloscopio (None nelle altre modalità).
            dt: tempo trascorso dall'ultimo frame (s).
        """
        glClear(GL_COLOR_BUFFER_BIT)

        # Geometria della modalità corrente (upload VBO + self._draw_count).
        if not self._prepare_mode_geometry(heights, waveform):
            self._draw_background()
            return

        self._draw_background()

        # Bloom prepass (mode-aware): la modalità corrente nell'FBO half-res.
        if self._bloom_enabled:
            self._render_bloom()

        # Visualizzazione corrente.
        self._draw_current_mode((self._fb_width, self._fb_height))

        # Peak-cap: solo in modalità Barre.
        if self._peakcap_enabled and self._mode == MODE_BARS:
            peaks = self._update_peaks(self._bar_heights, dt)
            if peaks.dtype != np.float32 or not peaks.flags["C_CONTIGUOUS"]:
                peaks = np.ascontiguousarray(peaks, dtype=np.float32)
            num_bars = int(self._draw_count)
            self._ensure_peaks_capacity(num_bars)
            glBindBuffer(GL_ARRAY_BUFFER, self._peaks_vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, peaks.nbytes, peaks)
            glUseProgram(self._caps_program)
            glUniform1f(self._caps_uniforms["uNumBars"], float(num_bars))
            glUniform1f(self._caps_uniforms["uBarWidthFrac"], float(self._bar_width_frac))
            glUniform1f(self._caps_uniforms["uMirror"], 1.0 if self._mirror else 0.0)
            glUniform1f(self._caps_uniforms["uCapThickness"], CAP_THICKNESS)
            glUniform3f(self._caps_uniforms["uCapColor"], *self._peakcap_color)
            glBindVertexArray(self._caps_vao)
            glDrawArraysInstanced(GL_TRIANGLES, 0, 6, num_bars)
            glBindVertexArray(0)

        # Composito bloom additivo sopra la scena.
        if self._bloom_enabled and self._bloom_size is not None:
            self._composite_bloom()
        return
```

- [ ] **Step 4: Rendere `_render_bloom` mode-aware**

Cambiare la firma `def _render_bloom(self, num_bars: int) -> None:` in `def _render_bloom(self) -> None:` e sostituire la riga `self._draw_bars_instanced(num_bars, (w, h))` con:

```python
        self._draw_current_mode((w, h))
```

(Il resto di `_render_bloom` — FBO, blur ping-pong, ripristino viewport — resta invariato.)

- [ ] **Step 5: Derivare il waveform e aggiornare la chiamata in `run()`**

In `run()`, subito dopo il blocco che calcola l'FFT (dopo `self.target_amplitudes = self.create_equalizer(...)` e il suo `if PERF_LOGS`), aggiungere:

```python
                # Waveform mono per l'oscilloscopio (solo se serve): media dei canali.
                waveform = None
                if self._mode == MODE_OSCILLOSCOPE and fft_window.size > 0:
                    waveform = fft_window.mean(axis=1).astype(np.float32)
```

E cambiare la chiamata `self._render(smoothed_amplitudes, frame_delta)` in:

```python
                self._render(smoothed_amplitudes, waveform, frame_delta)
```

- [ ] **Step 6: Verifica costruzione headless**

```bash
export UV_PROJECT_ENVIRONMENT=.venv-linux && uv run python -c "import queue; from thread.opengl_thread import EqualizerOpenGLThread; t=EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue()); assert hasattr(t,'_draw_current_mode') and hasattr(t,'_prepare_mode_geometry'); print('construct OK')"
```
Expected: `construct OK`

- [ ] **Step 7: Verifica logica geometria per-modalità (headless, senza GL)**

`_prepare_mode_geometry` chiama metodi GL (`_upload_*`) → non eseguibile headless. Verificare invece che i builder producano i conteggi attesi che il dispatch userà:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-linux && uv run python - <<'PY'
import numpy as np
from thread.opengl_thread import build_spectrum_strip, build_waveform_strip, arrange_symmetric
h = np.array([0.1,0.5,0.9], dtype=np.float32)
assert build_spectrum_strip(h, True, 0.01).shape == (6,3)       # 2N vertici
assert build_spectrum_strip(arrange_symmetric(h), True, 0.01).shape == (12,3)  # simmetrico raddoppia
assert build_waveform_strip(np.zeros(100,np.float32), False, 0.01, 2.5).shape == (200,3)
print("OK")
PY
```
Expected: `OK`

- [ ] **Step 8: Verifica visiva (Windows)**

Run: `export UV_PROJECT_ENVIRONMENT=.venv-linux && uv run python main.py` (su Windows: `uv run python main.py`) → seleziona device → Fullscreen. Default = Barre → **identico a prima**. Per provare le altre modalità prima della GUI (Task 6), impostare temporaneamente `mode=…` nel costruttore in `fullscreen_command`, oppure attendere il Task 6. Verificare: nessun crash, barre invariate di default, bloom ancora funzionante in modalità Barre.

- [ ] **Step 9: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Add per-mode render dispatch, mode-aware bloom and oscilloscope waveform"
```

---

## Task 6: GUI — selettore modalità + tab "🎛️ Forma" condizionale

**Files:** Modify `gui/main_window_gui.py`

**Interfaces:**
- Consumes: control message `set_viz_mode/set_inner_radius/set_line_thickness/set_fill/set_osc_mirror`; parametri costruttore `mode/inner_radius/line_thickness/fill/osc_mirror`.
- Produces: selettore "Modalità", `on_viz_mode_changed`, controlli di forma condizionali, peak-cap condizionale.

- [ ] **Step 1: Mappa etichetta→int e stato GUI**

In testa al modulo, dopo `_COLOR_MODE_TO_INT = {...}`:

```python
# Etichette del menu "Modalità" -> intero self._mode del renderer.
_VIZ_MODE_TO_INT = {"Barre": 0, "Radiale": 1, "Oscilloscopio": 2, "Linea/Area": 3}
```

In `__init__`, dopo lo stato effetti del Blocco 2 (`self.fx_beat_sensitivity = 1.5`) e prima di `self.bg_video_path = None`:

```python
        # Stato GUI modalità di visualizzazione (Blocco 3). Default = Barre.
        self.viz_mode = 0
        self.viz_inner_radius = 0.18
        self.viz_line_thickness = 0.012
        self.viz_fill = True
        self.viz_osc_mirror = False
```

- [ ] **Step 2: Selettore "Modalità" nel tab Visualizzazione**

In `_build_visualization_tab`, subito dopo `v = QVBoxLayout(tab)` e prima di `self.appearance_mode = ...`:

```python
        self.viz_mode_option = OptionMenuCustomFrame(
            header_name='Modalità:',
            values=["Barre", "Radiale", "Oscilloscopio", "Linea/Area"],
            initial_value="Barre",
            command=self.on_viz_mode_changed,
        )
        v.addWidget(self.viz_mode_option)
```

- [ ] **Step 3: Rinominare il tab "Barre"→"Forma" e aggiungere i nuovi controlli di forma**

In `_build_settings_tabs`, cambiare:

```python
        self.tabview.addTab(self._build_bars_tab(), "🎛️ Barre")
```
in:
```python
        self.tabview.addTab(self._build_shape_tab(), "🎛️ Forma")
```

Rinominare la definizione `def _build_bars_tab(self) -> QWidget:` in `def _build_shape_tab(self) -> QWidget:`. Dentro, **prima** di `v.addStretch(1)` e della riga finale `self.on_color_mode_changed("Classico")`, aggiungere i nuovi controlli di forma:

```python
        # Radiale
        self.viz_inner_radius_slider = SliderCustomFrame(
            header_name='Raggio foro:',
            command=self.update_inner_radius,
            from_=0.0, to=0.8,
            initial_value=self.viz_inner_radius,
        )
        v.addWidget(self.viz_inner_radius_slider)

        # Oscilloscopio / Linea-Area
        self.viz_thickness_slider = SliderCustomFrame(
            header_name='Spessore linea:',
            command=self.update_line_thickness,
            from_=0.002, to=0.06,
            initial_value=self.viz_line_thickness,
        )
        v.addWidget(self.viz_thickness_slider)

        # Linea/Area
        self.viz_fill_option = OptionMenuCustomFrame(
            header_name='Riempimento:', values=["Linea", "Area"], initial_value="Area",
            command=self.update_fill)
        v.addWidget(self.viz_fill_option)

        # Oscilloscopio
        self.viz_osc_mirror_option = OptionMenuCustomFrame(
            header_name='Specchiato:', values=["No", "Sì"], initial_value="No",
            command=self.update_osc_mirror)
        v.addWidget(self.viz_osc_mirror_option)
```

- [ ] **Step 4: Etichetta peak-cap con handle (per nasconderla)**

In `_build_effects_tab`, sostituire `v.addWidget(self._make_label("Peak-cap", font_label()))` con:

```python
        self.fx_peakcap_label = self._make_label("Peak-cap", font_label())
        v.addWidget(self.fx_peakcap_label)
```

- [ ] **Step 5: Callback `on_viz_mode_changed` + inoltri**

Aggiungere i metodi (vicino a `on_color_mode_changed`):

```python
    def on_viz_mode_changed(self, label: str):
        """Mostra solo i controlli di forma pertinenti alla modalità e inoltra la scelta."""
        bars = label == "Barre"
        radial = label == "Radiale"
        osc = label == "Oscilloscopio"
        line = label == "Linea/Area"
        # Forma (tab "🎛️ Forma")
        self.bars_width_slider.setVisible(bars or radial)
        self.bars_rounded_option.setVisible(bars)
        self.bars_anchor_option.setVisible(bars)
        self.bars_order_option.setVisible(bars or radial or line)
        self.viz_inner_radius_slider.setVisible(radial)
        self.viz_thickness_slider.setVisible(osc or line)
        self.viz_fill_option.setVisible(line)
        self.viz_osc_mirror_option.setVisible(osc)
        # Peak-cap: solo in modalità Barre (tab "✨ Effetti")
        self.fx_peakcap_label.setVisible(bars)
        self.fx_peakcap_option.setVisible(bars)
        self.fx_peakcap_color_picker.setVisible(bars)
        self.fx_peakcap_hold_slider.setVisible(bars)
        self.fx_peakcap_fall_slider.setVisible(bars)
        self.viz_mode = _VIZ_MODE_TO_INT.get(label, 0)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_viz_mode", "value": self.viz_mode})

    def update_inner_radius(self, value):
        self.viz_inner_radius = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_inner_radius", "value": float(value)})

    def update_line_thickness(self, value):
        self.viz_line_thickness = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_line_thickness", "value": float(value)})

    def update_fill(self, label):
        self.viz_fill = (label == "Area")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_fill", "value": self.viz_fill})

    def update_osc_mirror(self, label):
        self.viz_osc_mirror = (label == "Sì")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_osc_mirror", "value": self.viz_osc_mirror})
```

- [ ] **Step 6: Visibilità iniziale dopo la costruzione dei tab**

In `_build_settings_tabs`, **dopo** `parent_layout.addWidget(self.tabview, 1)`, aggiungere:

```python
        # Visibilità condizionale iniziale dei controlli di forma/effetti (modalità Barre).
        self.on_viz_mode_changed("Barre")
```

- [ ] **Step 7: Popolare il costruttore in `fullscreen_command`**

Nella chiamata `EqualizerOpenGLThread(...)`, dopo `bloom_intensity=self.fx_bloom_intensity,`:

```python
                mode=self.viz_mode,
                inner_radius=self.viz_inner_radius,
                line_thickness=self.viz_line_thickness,
                fill=self.viz_fill,
                osc_mirror=self.viz_osc_mirror,
```

- [ ] **Step 8: Smoke test import GUI (headless)**

```bash
QT_QPA_PLATFORM=offscreen UV_PROJECT_ENVIRONMENT=.venv-linux uv run python -c "import gui.main_window_gui; print('OK')" smoketest
```
Expected: `OK`

- [ ] **Step 9: Verifica costruzione GUI + visibilità condizionale (headless)**

```bash
QT_QPA_PLATFORM=offscreen UV_PROJECT_ENVIRONMENT=.venv-linux uv run python - smoketest <<'PY'
from PySide6.QtWidgets import QApplication
app = QApplication([])
from gui.main_window_gui import AudioCaptureGUI
w = AudioCaptureGUI()
# default = Barre
assert w.viz_mode_option.get_value() == "Barre"
assert w.bars_width_slider.isVisible() and not w.viz_inner_radius_slider.isVisible()
assert w.fx_peakcap_option.isVisible()
# passa a Linea/Area
w.on_viz_mode_changed("Linea/Area")
assert w.viz_mode == 3 and w.viz_thickness_slider.isVisible() and w.viz_fill_option.isVisible()
assert not w.fx_peakcap_option.isVisible() and not w.bars_width_slider.isVisible()
# passa a Radiale
w.on_viz_mode_changed("Radiale")
assert w.viz_mode == 1 and w.viz_inner_radius_slider.isVisible() and w.bars_width_slider.isVisible()
print("OK")
PY
```
Expected: `OK`
(Nota: `isVisible()` su una finestra non mostrata può risultare False per tutti; se così, sostituire con `not w.<widget>.isHidden()` che riflette il flag esplicito impostato da `setVisible`.)

- [ ] **Step 10: Verifica visiva (Windows)**

Run: `uv run python main.py` → Fullscreen → tab **🎨 Visualizzazione** selettore **Modalità**. Provare le 4 modalità live; per ciascuna i controlli del tab **🎛️ Forma** pertinenti compaiono/scompaiono; peak-cap visibile solo in Barre. Provare: Radiale (raggio foro, simmetria, larghezza, spettro colore), Oscilloscopio (spessore, specchiato), Linea/Area (spessore, riempimento Linea/Area, simmetria); bloom in ogni modalità; combinazioni con i colori del Blocco 1. Default Barre = nessuna regressione.

- [ ] **Step 11: Commit**

```bash
git add gui/main_window_gui.py
git commit -m "Add visualization-mode selector and conditional shape controls in GUI"
```

---

## Task 7: Documentazione e verifica finale

**Files:** Modify `CLAUDE.md` (locale, non committato)

- [ ] **Step 1: Aggiornare `CLAUDE.md`**

Nella sezione "OpenGL-specific concerns", aggiungere un punto:

```markdown
- **Visualization modes (Block 3, OpenGL-only):** a `set_viz_mode` selector switches `self._mode` between **Bars** (0, default), **Radial** (1), **Oscilloscope** (2), **Line/Area** (3). `_render` calls `_prepare_mode_geometry` (uploads heights for bars/radial, or CPU-built triangle-strip verts for the continuous modes) then `_draw_current_mode(viewport)` — the single dispatch reused by both the screen pass and the bloom prepass, so **bloom works in every mode**. Radial reuses the bars instanced program via a polar branch (`uRadial`/`uInnerRadius`/`uAspect`, aspect-corrected for round circles; rounded caps & center-anchor disabled there). Oscilloscope/Line-Area use a dedicated `_strip_program` (vertices `(x,y,mag)`, replicating the 4 Block-1 colour modes) fed by pure builders `build_spectrum_strip`/`build_waveform_strip` (+ `trigger_align` zero-crossing for a stable scope). The oscilloscope waveform is the channel-averaged `fft_window` (mono), derived in `run()` only when that mode is active (FFT still runs every frame for beat detection). **Peak-cap is Bars-only**; symmetric band order (`arrange_symmetric`) applies to Bars/Radial/Line. Control messages `set_viz_mode`/`set_inner_radius`/`set_line_thickness`/`set_fill`/`set_osc_mirror` + constructor params. GUI: mode selector in "🎨 Visualizzazione", the "🎛️ Barre" tab renamed "🎛️ Forma" with controls shown conditionally per mode. Default `mode=Bars` → identical to post-Block 2.
```

- [ ] **Step 2: Verifica finale headless (logica + costruzione)**

```bash
export UV_PROJECT_ENVIRONMENT=.venv-linux && uv run python - <<'PY'
import queue, numpy as np
from thread.opengl_thread import (EqualizerOpenGLThread, trigger_align,
    build_spectrum_strip, build_waveform_strip, arrange_symmetric,
    MODE_BARS, MODE_RADIAL, MODE_OSCILLOSCOPE, MODE_LINE)
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
for m, attr, val in [("set_viz_mode", "_mode", 1), ("set_inner_radius","_inner_radius",0.3),
                     ("set_line_thickness","_line_thickness",0.02), ("set_fill","_fill",False),
                     ("set_osc_mirror","_osc_mirror",True)]:
    t.process_control_message({"type": m, "value": val})
assert t._mode == 1 and abs(t._inner_radius-0.3) < 1e-6 and t._fill is False and t._osc_mirror is True
# builder coerenti coi conteggi del dispatch
h = np.array([0.2,0.6,0.9], np.float32)
assert build_spectrum_strip(h, True, 0.01).shape == (6,3)
assert build_spectrum_strip(arrange_symmetric(h), False, 0.01).shape == (12,3)
assert build_waveform_strip(np.zeros(64,np.float32), True, 0.01, 2.5).shape == (128,3)
assert trigger_align(np.array([-0.1,0.2,0.3],np.float32)).shape[0] == 3
print("OK")
PY
```
Expected: `OK`

- [ ] **Step 3: Verifica visiva finale (Windows)**

Run: `uv run python main.py`. Esercitare tutte e 4 le modalità e le loro sotto-opzioni, le combinazioni con i colori/simmetria del Blocco 1 e con gli effetti (bloom in tutte, beat in tutte, peak-cap solo Barre); sfondo immagine **e** video ok in ogni modalità; con `mode=Barre` e default → rendering identico a post-Blocco 2 (non-regressione).

- [ ] **Step 4: Commit (solo file tracciati)**

`CLAUDE.md` è gitignored: **non** committarlo. Il codice è già committato nei task precedenti. Nessun commit necessario qui se non ci sono altri file tracciati modificati.

---

## Self-Review (compilata in fase di scrittura)

- **Spec coverage:** §A modello modalità → Task 1+5; §B Barre invariate → Task 5 (dispatch); §C Radiale → Task 3+5; §D modalità a traccia + builder + trigger → Task 2+4+5; §E effetti×modalità (bloom ovunque, peak-cap solo barre, beat sempre) → Task 5 (`_render_bloom` mode-aware, guard `mode==MODE_BARS`); §F colore/forma×modalità → shader Task 3 (radiale) + Task 4 (strip) + `_display_heights`/`arrange_symmetric` riusati in Task 5; §G control-queue+costruttore → Task 1; §H GUI → Task 6; Default/non-regressione → default `mode=Barre` ovunque; Verifica → snippet headless + visivo in ogni task.
- **Placeholder scan:** nessun TBD/TODO; ogni step mostra codice o comando concreto.
- **Type consistency:** `MODE_*` costanti usate coerentemente; builder restituiscono sempre `(M,3) float32`; `_draw_count` impostato da `_upload_heights`/`_upload_strip` e letto da `_draw_current_mode`/peak-cap; `_render(heights, waveform, dt)` con chiamata aggiornata in `run()` e `_render_bloom()` senza argomenti.

---

## Note finali

- **Ordine in `_render`:** clear → `_prepare_mode_geometry` (upload + `_draw_count`) → sfondo → (bloom prepass mode-aware) → `_draw_current_mode` → peak-cap (solo Barre) → composito bloom.
- **Non-regressione:** il check chiave è "modalità Barre + default = identico a post-Blocco 2". Le modalità a traccia non chiamano `_update_peaks`; il radiale forza `uMirror=0`/`uRounded=0`.
- **Smoke EGL opzionale (de-risk shader, WSL2):** per verificare che i nuovi shader (radiale, strip) **compilino** senza Windows, si può creare un contesto offscreen sotto WSLg: `PYOPENGL_PLATFORM=egl` prima di importare OpenGL e `glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)` + finestra nascosta (`glfw.window_hint(glfw.VISIBLE, glfw.FALSE)`), poi `init_gl_objects()`. **Non** aggiungere l'hint EGL al `run()` dell'app (Windows usa WGL di default). È opzionale e fragile: la verifica autorevale resta visiva su Windows.
- **Performance:** i builder strip sono vettoriali numpy su N ≤ poche centinaia di vertici (oscilloscopio fino a ~2·N_FFT) → microsecondi; nessun impatto sul frame budget. Se l'oscilloscopio risultasse troppo "denso", aggiungere un downsample del waveform (candidato al tuning, non in scope).
- **Tuning a vista (Windows):** `OSC_GAIN`, `DEFAULT_INNER_RADIUS`, `DEFAULT_LINE_THICKNESS` sono costanti di modulo regolabili dopo la verifica visiva.
- **Al termine:** PR + release `2.3.0` via `superpowers:finishing-a-development-branch`.
```
