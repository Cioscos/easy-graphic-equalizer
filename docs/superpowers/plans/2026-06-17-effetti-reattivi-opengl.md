# Effetti reattivi OpenGL (Blocco 2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggiungere tre effetti reattivi al renderer fullscreen OpenGL — peak-cap che ricade, glow via vero bloom (FBO + blur), pulsazione dello sfondo sul beat — tutti attivabili dalla GUI e **default OFF**.

**Architecture:** Tutto in `thread/opengl_thread.py`: stato per-barra dei picchi + seconda draw instanced per i cap; pipeline bloom a risoluzione dimezzata (FBO ping-pong + blur gaussiano separabile + composito additivo); beat detection dall'energia sotto ~150 Hz (media mobile + refrattario) che pilota uno zoom dello sfondo (`uBgZoom`). GUI in un nuovo tab "✨ Effetti". Effetti disattivi = path saltati, nessun costo né cambiamento visivo.

**Tech Stack:** Python 3.12, PyOpenGL (GLSL 3.3 core, FBO/`GL_RGB16F`), GLFW, NumPy, PySide6, `uv`.

> **Convenzioni di progetto (vincolanti):**
> - **Nessuna test suite.** Logica pura → `uv run python` headless (WSL2 `.venv-linux`); shader/FBO/GUI → verifica visiva su Windows.
> - **Commit:** utente `Cioscos`, **mai** trailer `Co-Authored-By`.
> - **`CLAUDE.md` è gitignored** in questo repo: aggiornalo ma **non committarlo** (vedi memoria `claude-md-untracked`).
> - Smoke test GUI headless: aggiungere un argomento fittizio (`… smoketest`) perché `soundcard` legge `sys.argv[1]` all'import (memoria `soundcard-headless-argv`).
> - Branch di lavoro: `feature/opengl-reactive-effects` (già creato; lo spec è già committato lì).

**Spec di riferimento:** `docs/superpowers/specs/2026-06-17-effetti-reattivi-opengl-design.md`

---

## File Structure

- **Modify** `thread/opengl_thread.py` — stato/param di peak-cap, beat, bloom; metodi puri `_update_peaks`, `_detect_beat`; calcolo energia bassi in `create_equalizer`; shader caps/blur/composite; `uBgZoom` nel bg vertex shader; pipeline FBO bloom; integrazione in `_render`/`run()`/`_cleanup_gl`; nuovi rami in `process_control_message`; nuovi parametri costruttore.
- **Modify** `gui/main_window_gui.py` — `_build_effects_tab`, callback di inoltro, stato GUI, popolamento costruttore in `fullscreen_command`.
- **Modify** `CLAUDE.md` — doc (locale, non committato).

Default di tutti gli effetti = **OFF** → con i default il rendering è identico a post-Blocco 1.

---

## Task 1: Logica picchi `_update_peaks` (puro) + stato/param peak-cap

**Files:** Modify `thread/opengl_thread.py` (costanti, costruttore, nuovo metodo)

- [ ] **Step 1: Costanti default peak-cap**

Dopo `DEFAULT_COLOR_B = (...)` (aggiunto nel Blocco 1) aggiungere:

```python
# --- Effetti reattivi (Blocco 2). Default OFF: nessun cambiamento visivo. ---
DEFAULT_PEAKCAP_COLOR = (1.0, 1.0, 1.0)   # bianco
PEAKCAP_HOLD = 0.5        # s di hold prima della caduta
PEAKCAP_FALL = 0.8        # frazione di altezza al secondo
CAP_THICKNESS = 0.012     # spessore del trattino (frazione di altezza finestra)
```

- [ ] **Step 2: Parametri costruttore peak-cap**

Nella firma di `__init__`, dopo `band_order_symmetric: bool = False,` (Blocco 1) e prima di `window_close_callback`:

```python
                 peakcap_enabled: bool = False,
                 peakcap_color: tuple = DEFAULT_PEAKCAP_COLOR,
                 peakcap_hold: float = PEAKCAP_HOLD,
                 peakcap_fall: float = PEAKCAP_FALL,
```

Nel corpo di `__init__`, dopo il blocco forma del Blocco 1 (`self._band_order_symmetric = ...` e le righe `self._fb_width/_fb_height`), aggiungere:

```python
        # --- Peak-cap (Blocco 2) ---
        self._peakcap_enabled = bool(peakcap_enabled)
        self._peakcap_color = tuple(peakcap_color)
        self._peakcap_hold = float(peakcap_hold)
        self._peakcap_fall = float(peakcap_fall)
        # self._n_bands è già impostato; self.frequency_bands viene creato solo più
        # tardi da _setup_bands. _update_peaks ridimensiona comunque se il numero di
        # barre visualizzate cambia (es. ordine simmetrico).
        self._peaks = np.zeros(self._n_bands, dtype=np.float32)
        self._peak_age = np.zeros(self._n_bands, dtype=np.float32)
```

- [ ] **Step 3: Verifica (fallisce: metodo assente)**

```bash
uv run python - <<'PY'
import queue, numpy as np
from thread.opengl_thread import EqualizerOpenGLThread
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
t._peakcap_hold = 0.1; t._peakcap_fall = 1.0
p = t._update_peaks(np.array([0.6], dtype=np.float32), 0.05); assert abs(p[0]-0.6) < 1e-6   # snap su
p = t._update_peaks(np.array([0.2], dtype=np.float32), 0.05); assert abs(p[0]-0.6) < 1e-6   # hold (age 0.05<0.1)
p = t._update_peaks(np.array([0.2], dtype=np.float32), 0.10); assert abs(p[0]-0.5) < 1e-6   # cade 1.0*0.1
p = t._update_peaks(np.array([0.2, 0.3], dtype=np.float32), 0.01); assert p.shape[0] == 2   # resize
print("OK")
PY
```
Expected: `AttributeError` (metodo non definito).

- [ ] **Step 4: Implementare `_update_peaks`**

Aggiungere il metodo (subito prima di `smooth_amplitudes`):

```python
    def _update_peaks(self, heights: np.ndarray, dt: float) -> np.ndarray:
        """
        Aggiorna i picchi per-barra (peak-cap): scatto al nuovo massimo, hold, poi
        caduta frame-rate-independent. Lavora sulle altezze già disposte.
        """
        n = heights.shape[0]
        if self._peaks.shape[0] != n:
            self._peaks = np.zeros(n, dtype=np.float32)
            self._peak_age = np.zeros(n, dtype=np.float32)
        rising = heights >= self._peaks
        self._peaks[rising] = heights[rising]
        self._peak_age[rising] = 0.0
        held = ~rising
        self._peak_age[held] += dt
        falling = held & (self._peak_age > self._peakcap_hold)
        self._peaks[falling] -= self._peakcap_fall * dt
        np.clip(self._peaks, 0.0, 1.0, out=self._peaks)
        return self._peaks
```

- [ ] **Step 5: Eseguire la verifica (deve passare)**

Rilanciare lo snippet dello Step 3. Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Add per-bar peak tracking (_update_peaks) for peak-cap effect"
```

---

## Task 2: Messaggi di controllo peak-cap

**Files:** Modify `thread/opengl_thread.py` (`process_control_message`)

- [ ] **Step 1: Verifica (fallisce: rami assenti)**

```bash
uv run python - <<'PY'
import queue
from thread.opengl_thread import EqualizerOpenGLThread
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
t.process_control_message({"type": "set_peakcap_enabled", "value": True}); assert t._peakcap_enabled is True
t.process_control_message({"type": "set_peakcap_color", "value": (0.1, 0.2, 0.3)}); assert t._peakcap_color == (0.1, 0.2, 0.3)
t.process_control_message({"type": "set_peakcap_hold", "value": 0.3}); assert abs(t._peakcap_hold-0.3) < 1e-6
t.process_control_message({"type": "set_peakcap_fall", "value": 1.2}); assert abs(t._peakcap_fall-1.2) < 1e-6
print("OK")
PY
```
Expected: `AssertionError`/`KeyError`.

- [ ] **Step 2: Aggiungere i rami in `process_control_message`**

Dopo il ramo `set_band_order` (Blocco 1) e prima di `set_image`:

```python
        elif message_type == 'set_peakcap_enabled':
            self._peakcap_enabled = bool(message['value'])

        elif message_type == 'set_peakcap_color':
            self._peakcap_color = tuple(message['value'])

        elif message_type == 'set_peakcap_hold':
            self._peakcap_hold = float(message['value'])

        elif message_type == 'set_peakcap_fall':
            self._peakcap_fall = float(message['value'])
```

- [ ] **Step 3: Eseguire la verifica (deve passare)**

Rilanciare lo snippet dello Step 1. Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Handle peak-cap control messages"
```

---

## Task 3: Rendering peak-cap (shader caps + draw instanced)

**Files:** Modify `thread/opengl_thread.py` (shader sources, `init_gl_objects`, `_render`, `run()`)

- [ ] **Step 1: Sorgenti shader dei cap**

Dopo il blocco `_BARS_FRAGMENT_SRC = """..."""` aggiungere:

```python
# --- Shader peak-cap: trattino sottile instanced all'altezza del picco ---
_CAPS_VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec2 aUnitPos;   // quad unitario [0,1]^2 (condiviso con le barre)
layout(location = 1) in float aPeak;     // picco normalizzato per-istanza [0,1]
uniform float uNumBars;
uniform float uBarWidthFrac;
uniform float uMirror;
uniform float uCapThickness;
void main() {
    float slot = 1.0 / uNumBars;
    float xLeft = (float(gl_InstanceID) + 0.5 * (1.0 - uBarWidthFrac)) * slot;
    float x = xLeft + aUnitPos.x * (slot * uBarWidthFrac);
    float yTopBottom = aPeak;              // estremità superiore, ancoraggio dal basso
    float yTopCenter = 0.5 + 0.5 * aPeak;  // estremità superiore, ancoraggio specchiato
    float yc = mix(yTopBottom, yTopCenter, uMirror);
    float y = yc + (aUnitPos.y - 0.5) * uCapThickness;  // trattino sottile attorno a yc
    gl_Position = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
"""

_CAPS_FRAGMENT_SRC = """
#version 330 core
out vec4 FragColor;
uniform vec3 uCapColor;
void main() { FragColor = vec4(uCapColor, 1.0); }
"""
```

- [ ] **Step 2: Aggiungere attributi GL per i cap (costruttore)**

Nel blocco "Oggetti OpenGL" di `__init__` (dove sono dichiarati `self._bars_program = None` ecc.), aggiungere:

```python
        self._caps_program = None
        self._caps_vao = None
        self._peaks_vbo = None
        self._peaks_capacity = 0
        self._caps_uniforms = {}
```

- [ ] **Step 3: Compilare il programma cap e creare VAO/VBO in `init_gl_objects`**

In fondo a `init_gl_objects` (dopo il setup dello sfondo) aggiungere:

```python
        # --- Programma + VAO/VBO peak-cap (riusa il quad unitario delle barre) ---
        self._caps_program = compileProgram(
            compileShader(_CAPS_VERTEX_SRC, GL_VERTEX_SHADER),
            compileShader(_CAPS_FRAGMENT_SRC, GL_FRAGMENT_SHADER),
        )
        self._caps_uniforms = {
            name: glGetUniformLocation(self._caps_program, name)
            for name in ("uNumBars", "uBarWidthFrac", "uMirror", "uCapThickness", "uCapColor")
        }
        self._caps_vao = glGenVertexArrays(1)
        glBindVertexArray(self._caps_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._bars_quad_vbo)   # quad unitario condiviso
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        self._peaks_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._peaks_vbo)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)
        self._peaks_capacity = 0
        glBindVertexArray(0)
```

- [ ] **Step 4: Helper capacità VBO picchi**

Dopo `_ensure_heights_capacity` aggiungere:

```python
    def _ensure_peaks_capacity(self, n: int) -> None:
        """Garantisce che il VBO dei picchi contenga almeno n float."""
        if n <= self._peaks_capacity:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self._peaks_vbo)
        glBufferData(GL_ARRAY_BUFFER, n * 4, None, GL_STREAM_DRAW)
        self._peaks_capacity = n
```

- [ ] **Step 5: Passare `dt` a `_render` e disegnare i cap**

Cambiare la firma `def _render(self, heights: np.ndarray) -> None:` in:

```python
    def _render(self, heights: np.ndarray, dt: float) -> None:
```

In `run()`, la chiamata `self._render(smoothed_amplitudes)` diventa:

```python
                self._render(smoothed_amplitudes, frame_delta)
```

In `_render`, **subito dopo** la draw delle barre (`glDrawArraysInstanced(GL_TRIANGLES, 0, 6, num_bars)` e prima di `glBindVertexArray(0)`), aggiungere il blocco cap:

```python
        glBindVertexArray(0)

        # --- Peak-cap (Blocco 2): aggiorna i picchi e disegna i trattini ---
        if self._peakcap_enabled:
            peaks = self._update_peaks(heights, dt)
            if peaks.dtype != np.float32 or not peaks.flags["C_CONTIGUOUS"]:
                peaks = np.ascontiguousarray(peaks, dtype=np.float32)
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
        return
```

(Nota: la riga `glBindVertexArray(0)` esistente alla fine di `_render` viene così assorbita; assicurarsi che non resti duplicata.)

- [ ] **Step 6: Liberare le risorse cap in `_cleanup_gl`**

In `_cleanup_gl`, aggiungere `self._peaks_vbo` all'elenco dei VBO, `self._caps_vao` ai VAO e `self._caps_program` ai programmi da eliminare, e azzerarli a fine metodo:

```python
        for vbo in (self._bars_quad_vbo, self._heights_vbo, self._bg_vbo, self._peaks_vbo):
            if vbo:
                _safe(lambda v=vbo: glDeleteBuffers(1, [v]))
        for vao in (self._bars_vao, self._bg_vao, self._caps_vao):
            if vao:
                _safe(lambda v=vao: glDeleteVertexArrays(1, [v]))
        for prog in (self._bars_program, self._bg_program, self._caps_program):
            if prog:
                _safe(lambda p=prog: glDeleteProgram(p))
```

E nelle righe di azzeramento finali aggiungere `self._peaks_vbo = None`, `self._caps_vao = None`, `self._caps_program = None`, `self._peaks_capacity = 0`.

- [ ] **Step 7: Verifica costruzione headless**

```bash
uv run python -c "import queue; from thread.opengl_thread import EqualizerOpenGLThread; EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue()); print('construct OK')"
```
Expected: `construct OK`

- [ ] **Step 8: Verifica visiva (Windows)**

Run: `uv run python main.py` → Fullscreen. Default = nessun cap (peakcap OFF) → invariato. Per un'anteprima prima della GUI: impostare temporaneamente `peakcap_enabled: bool = True` nel costruttore, eseguire, osservare i trattini che tengono il picco e ricadono, poi ripristinare `False`.

- [ ] **Step 9: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Render peak-cap markers via instanced caps shader"
```

---

## Task 4: Beat detection `_detect_beat` (puro) + energia bassi + stato/param

**Files:** Modify `thread/opengl_thread.py` (costanti, costruttore, `create_equalizer`, nuovo metodo)

- [ ] **Step 1: Costanti default beat**

Dopo le costanti peak-cap (Task 1):

```python
BEAT_INTENSITY = 0.08         # zoom massimo dello sfondo (8%)
BEAT_SENSITIVITY = 1.5        # energia > 1.5x media -> beat
BEAT_EMA_ALPHA = 0.08         # reattività della media mobile dell'energia
BEAT_REFRACTORY = 0.15        # s minimi tra due beat
BEAT_DECAY_TAU = 0.18         # s, decadimento dell'inviluppo di pulsazione
BASS_CUTOFF_HZ = 150.0        # energia bassi: potenza FFT sotto questa frequenza
```

- [ ] **Step 2: Parametri costruttore beat + stato**

Nella firma di `__init__`, dopo i parametri peak-cap (Task 1):

```python
                 beat_enabled: bool = False,
                 beat_intensity: float = BEAT_INTENSITY,
                 beat_sensitivity: float = BEAT_SENSITIVITY,
```

Nel corpo, dopo il blocco peak-cap:

```python
        # --- Beat pulse (Blocco 2) ---
        self._beat_enabled = bool(beat_enabled)
        self._beat_intensity = float(beat_intensity)
        self._beat_sensitivity = float(beat_sensitivity)
        self._beat_ema = 0.0          # media mobile dell'energia bassi
        self._beat_refractory = 0.0   # timer refrattario (s)
        self._beat_pulse = 0.0        # inviluppo di pulsazione [0,1]
        self._bass_energy = 0.0       # energia bassi dell'ultimo frame (da create_equalizer)
        # Maschera dei bin FFT sotto il cutoff bassi (per l'energia beat)
        _freqs = np.fft.rfftfreq(N_FFT, 1.0 / RATE)
        self._bass_mask = _freqs < BASS_CUTOFF_HZ
```

- [ ] **Step 3: Calcolare l'energia bassi in `create_equalizer`**

In `create_equalizer`, subito dopo la riga `power = power.mean(axis=0)` (potenza media per bin tra i canali), aggiungere:

```python
        # Energia bassi (per la beat detection): somma potenza sotto il cutoff.
        self._bass_energy = float(np.sum(power[self._bass_mask]))
```

- [ ] **Step 4: Verifica (fallisce: metodo assente)**

```bash
uv run python - <<'PY'
import queue
from thread.opengl_thread import EqualizerOpenGLThread
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
t._beat_sensitivity = 1.5
for _ in range(30):  # energia stabile -> nessun beat
    assert t._detect_beat(1.0, 0.016) is False
assert t._detect_beat(5.0, 0.016) is True     # spike -> beat
assert t._detect_beat(5.0, 0.016) is False    # entro il refrattario -> no
print("OK")
PY
```
Expected: `AttributeError`.

- [ ] **Step 5: Implementare `_detect_beat`**

Aggiungere il metodo (vicino a `_update_peaks`):

```python
    def _detect_beat(self, energy: float, dt: float) -> bool:
        """
        Onset detection sull'energia dei bassi: beat quando l'energia supera
        media_mobile * sensibilità, fuori dal periodo refrattario. Aggiorna la
        media mobile (EMA) e il timer refrattario. Puro/testabile.
        """
        self._beat_refractory = max(0.0, self._beat_refractory - dt)
        if self._beat_ema <= 1e-12:
            self._beat_ema = energy
        is_beat = (energy > self._beat_ema * self._beat_sensitivity) and (self._beat_refractory <= 0.0)
        self._beat_ema += (energy - self._beat_ema) * BEAT_EMA_ALPHA
        if is_beat:
            self._beat_refractory = BEAT_REFRACTORY
        return is_beat
```

- [ ] **Step 6: Eseguire la verifica (deve passare)**

Rilanciare lo snippet dello Step 4. Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Add bass-energy beat detection (_detect_beat)"
```

---

## Task 5: Messaggi di controllo beat

**Files:** Modify `thread/opengl_thread.py` (`process_control_message`)

- [ ] **Step 1: Verifica (fallisce: rami assenti)**

```bash
uv run python - <<'PY'
import queue
from thread.opengl_thread import EqualizerOpenGLThread
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
t.process_control_message({"type": "set_beat_enabled", "value": True}); assert t._beat_enabled is True
t.process_control_message({"type": "set_beat_intensity", "value": 0.2}); assert abs(t._beat_intensity-0.2) < 1e-6
t.process_control_message({"type": "set_beat_sensitivity", "value": 2.0}); assert abs(t._beat_sensitivity-2.0) < 1e-6
print("OK")
PY
```
Expected: `AssertionError`/`KeyError`.

- [ ] **Step 2: Aggiungere i rami**

Dopo i rami peak-cap (Task 2):

```python
        elif message_type == 'set_beat_enabled':
            self._beat_enabled = bool(message['value'])

        elif message_type == 'set_beat_intensity':
            self._beat_intensity = float(message['value'])

        elif message_type == 'set_beat_sensitivity':
            self._beat_sensitivity = float(message['value'])
```

- [ ] **Step 3: Eseguire la verifica (deve passare)** — rilanciare lo Step 1. Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Handle beat control messages"
```

---

## Task 6: Beat pulse — zoom dello sfondo

**Files:** Modify `thread/opengl_thread.py` (bg vertex shader, `init_gl_objects` uniforms, `run()`, `_render`)

- [ ] **Step 1: Aggiungere `uBgZoom` al bg vertex shader**

Sostituire `_BG_VERTEX_SRC = """..."""` con (aggiunge lo zoom attorno al centro):

```python
_BG_VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;        // NDC [-1,1]
layout(location = 1) in vec2 aTexCoord;
uniform float uFlipV;                     // 0 = immagine, 1 = frame video
uniform float uBgZoom;                    // zoom additivo sul beat (0 = nessuno)
out vec2 vTexCoord;
void main() {
    vTexCoord = vec2(aTexCoord.x, mix(aTexCoord.y, 1.0 - aTexCoord.y, uFlipV));
    gl_Position = vec4(aPos * (1.0 + uBgZoom), 0.0, 1.0);
}
"""
```

- [ ] **Step 2: Registrare la uniform `uBgZoom`**

In `init_gl_objects`, nel dizionario `self._bg_uniforms = {...}`, aggiungere `"uBgZoom"`:

```python
        self._bg_uniforms = {
            name: glGetUniformLocation(self._bg_program, name)
            for name in ("uTexture", "uAlpha", "uFlipV", "uBgZoom")
        }
```

- [ ] **Step 3: Aggiornare beat + inviluppo nel loop di `run()`**

In `run()`, subito dopo il calcolo dell'FFT (`self.target_amplitudes = self.create_equalizer(...)`, che ora popola anche `self._bass_energy`) e prima della ballistica, aggiungere:

```python
                # Beat detection + inviluppo di pulsazione (Blocco 2)
                if self._beat_enabled:
                    if self._detect_beat(self._bass_energy, frame_delta):
                        self._beat_pulse = 1.0
                    # decadimento esponenziale frame-rate-independent
                    self._beat_pulse *= math.exp(-frame_delta / BEAT_DECAY_TAU)
                else:
                    self._beat_pulse = 0.0
```

- [ ] **Step 4: Applicare lo zoom nello sfondo (`_render`)**

Nel blocco sfondo di `_render`, dove si impostano le uniform del bg (`glUniform1f(self._bg_uniforms["uFlipV"], flip)`), aggiungere subito dopo:

```python
            glUniform1f(self._bg_uniforms["uBgZoom"], float(self._beat_pulse * self._beat_intensity))
```

- [ ] **Step 5: Verifica costruzione headless**

```bash
uv run python -c "import queue; from thread.opengl_thread import EqualizerOpenGLThread; EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue()); print('construct OK')"
```
Expected: `construct OK`

- [ ] **Step 6: Verifica visiva (Windows)**

Run: `uv run python main.py` → Fullscreen con uno sfondo immagine/video. Default = nessuna pulsazione. Anteprima: impostare temporaneamente `beat_enabled: bool = True`, mettere musica ritmica, osservare lo sfondo "pulsare" sul beat; ripristinare `False`.

- [ ] **Step 7: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Add beat-driven background zoom pulse"
```

---

## Task 7: Parametri e messaggi bloom (stato)

**Files:** Modify `thread/opengl_thread.py` (costanti, costruttore, `process_control_message`)

- [ ] **Step 1: Costanti default bloom**

Dopo le costanti beat (Task 4):

```python
BLOOM_INTENSITY = 1.0     # forza del composito additivo
BLOOM_BLUR_PASSES = 1     # iterazioni di blur (H+V) per il bloom
```

- [ ] **Step 2: Parametri costruttore + stato**

Nella firma di `__init__`, dopo i parametri beat:

```python
                 bloom_enabled: bool = False,
                 bloom_intensity: float = BLOOM_INTENSITY,
```

Nel corpo, dopo il blocco beat:

```python
        # --- Bloom (Blocco 2) ---
        self._bloom_enabled = bool(bloom_enabled)
        self._bloom_intensity = float(bloom_intensity)
```

E nel blocco "Oggetti OpenGL" di `__init__`:

```python
        self._blur_program = None
        self._bloom_composite_program = None
        self._blur_uniforms = {}
        self._bloom_uniforms = {}
        self._bloom_fbos = []      # 2 FBO ping-pong
        self._bloom_texs = []      # 2 texture half-res (GL_RGB16F)
        self._bloom_size = None    # (w, h) half-res; None = non ancora creati
```

- [ ] **Step 3: Verifica (fallisce: rami assenti)**

```bash
uv run python - <<'PY'
import queue
from thread.opengl_thread import EqualizerOpenGLThread
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
t.process_control_message({"type": "set_bloom_enabled", "value": True}); assert t._bloom_enabled is True
t.process_control_message({"type": "set_bloom_intensity", "value": 1.6}); assert abs(t._bloom_intensity-1.6) < 1e-6
print("OK")
PY
```
Expected: `AssertionError`/`KeyError`.

- [ ] **Step 4: Aggiungere i rami**

Dopo i rami beat (Task 5):

```python
        elif message_type == 'set_bloom_enabled':
            self._bloom_enabled = bool(message['value'])

        elif message_type == 'set_bloom_intensity':
            self._bloom_intensity = float(message['value'])
```

- [ ] **Step 5: Eseguire la verifica (deve passare)** — rilanciare lo Step 3. Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Add bloom state and control messages"
```

---

## Task 8: Pipeline bloom (FBO + blur separabile + composito)

**Files:** Modify `thread/opengl_thread.py` (shader blur/composite, helper FBO/draw, `init_gl_objects`, `_render`, `_cleanup_gl`)

- [ ] **Step 1: Shader blur e composite**

Dopo gli shader dei cap (Task 3) aggiungere:

```python
# --- Shader bloom: blur gaussiano separabile + composito additivo (quad fullscreen) ---
_FULLSCREEN_VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 vUV;
void main() { vUV = aTexCoord; gl_Position = vec4(aPos, 0.0, 1.0); }
"""

_BLUR_FRAGMENT_SRC = """
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
uniform vec2 uDir;   // (1/w, 0) orizzontale oppure (0, 1/h) verticale
void main() {
    float w[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec3 c = texture(uTex, vUV).rgb * w[0];
    for (int i = 1; i < 5; i++) {
        c += texture(uTex, vUV + uDir * float(i)).rgb * w[i];
        c += texture(uTex, vUV - uDir * float(i)).rgb * w[i];
    }
    FragColor = vec4(c, 1.0);
}
"""

_BLOOM_COMPOSITE_FRAGMENT_SRC = """
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
uniform float uIntensity;
void main() { FragColor = vec4(texture(uTex, vUV).rgb * uIntensity, 1.0); }
"""
```

- [ ] **Step 2: Compilare i programmi bloom in `init_gl_objects`**

In fondo a `init_gl_objects` (dopo il programma cap) aggiungere:

```python
        # --- Programmi bloom (riusano il VAO dello sfondo come quad fullscreen) ---
        self._blur_program = compileProgram(
            compileShader(_FULLSCREEN_VERTEX_SRC, GL_VERTEX_SHADER),
            compileShader(_BLUR_FRAGMENT_SRC, GL_FRAGMENT_SHADER),
        )
        self._bloom_composite_program = compileProgram(
            compileShader(_FULLSCREEN_VERTEX_SRC, GL_VERTEX_SHADER),
            compileShader(_BLOOM_COMPOSITE_FRAGMENT_SRC, GL_FRAGMENT_SHADER),
        )
        self._blur_uniforms = {
            name: glGetUniformLocation(self._blur_program, name) for name in ("uTex", "uDir")
        }
        self._bloom_uniforms = {
            name: glGetUniformLocation(self._bloom_composite_program, name)
            for name in ("uTex", "uIntensity")
        }
```

- [ ] **Step 3: Helper creazione FBO + draw barre + pipeline bloom**

Aggiungere questi metodi (vicino a `_render`):

```python
    def _ensure_bloom_fbos(self) -> bool:
        """
        Crea (lazy) i due FBO ping-pong half-res (GL_RGB16F) per il bloom.
        Ritorna True se disponibili. Richiede dimensione framebuffer nota.
        """
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
                return False
            fbos.append(fbo)
            texs.append(tex)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._bloom_fbos, self._bloom_texs, self._bloom_size = fbos, texs, (w, h)
        return True

    def _draw_bars_instanced(self, num_bars: int, viewport_px) -> None:
        """Imposta le uniform delle barre e disegna la passata instanced (riuso)."""
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
        glUniform2f(self._bars_uniforms["uViewportPx"], float(viewport_px[0]), float(viewport_px[1]))
        glBindVertexArray(self._bars_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, num_bars)
        glBindVertexArray(0)

    def _render_bloom(self, num_bars: int) -> None:
        """Renderizza le barre in un FBO half-res, le sfoca (H+V) e lascia il
        risultato in _bloom_texs[0], pronto per il composito additivo."""
        if not self._ensure_bloom_fbos():
            return
        w, h = self._bloom_size
        # Pass A: barre nel FBO 0
        glBindFramebuffer(GL_FRAMEBUFFER, self._bloom_fbos[0])
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT)
        self._draw_bars_instanced(num_bars, (w, h))
        # Pass B/C: blur ping-pong (orizzontale 0->1, verticale 1->0), N iterazioni
        glDisable(GL_BLEND)
        glUseProgram(self._blur_program)
        glBindVertexArray(self._bg_vao)
        for _ in range(BLOOM_BLUR_PASSES):
            for dir_vec, src, dst in ((( 1.0 / w, 0.0), 0, 1), ((0.0, 1.0 / h), 1, 0)):
                glBindFramebuffer(GL_FRAMEBUFFER, self._bloom_fbos[dst])
                glClear(GL_COLOR_BUFFER_BIT)
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, self._bloom_texs[src])
                glUniform1i(self._blur_uniforms["uTex"], 0)
                glUniform2f(self._blur_uniforms["uDir"], dir_vec[0], dir_vec[1])
                glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glEnable(GL_BLEND)
        # Ripristina framebuffer e viewport reali
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self._fb_width, self._fb_height)

    def _composite_bloom(self) -> None:
        """Disegna la texture sfocata in additivo sopra la scena."""
        glUseProgram(self._bloom_composite_program)
        glBlendFunc(GL_ONE, GL_ONE)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._bloom_texs[0])
        glUniform1i(self._bloom_uniforms["uTex"], 0)
        glUniform1f(self._bloom_uniforms["uIntensity"], float(self._bloom_intensity))
        glBindVertexArray(self._bg_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)   # ripristina blending standard
```

- [ ] **Step 4: Integrare il bloom in `_render`**

Il bloom va calcolato **prima** di disegnare la scena (prepass+blur, ripristina il framebuffer reale) e composto **dopo** barre e cap.

In `_render`, subito dopo `glBufferSubData(...)` delle altezze (cioè dopo l'upload del VBO altezze, prima di `glUseProgram(self._bars_program)`), inserire il prepass:

```python
        if self._bloom_enabled:
            self._render_bloom(num_bars)
```

E in fondo a `_render`, **dopo** il blocco dei peak-cap (subito prima del `return`), aggiungere il composito:

```python
        if self._bloom_enabled and self._bloom_size is not None:
            self._composite_bloom()
        return
```

- [ ] **Step 5: Liberare FBO/texture/programmi bloom in `_cleanup_gl`**

In `_cleanup_gl`, dopo l'eliminazione delle texture sfondo/video, aggiungere:

```python
        for tex in self._bloom_texs:
            _safe(lambda t=tex: glDeleteTextures(1, [t]))
        for fbo in self._bloom_fbos:
            _safe(lambda f=fbo: glDeleteFramebuffers(1, [f]))
        self._bloom_texs = []
        self._bloom_fbos = []
        self._bloom_size = None
```

E aggiungere `self._blur_program` e `self._bloom_composite_program` all'elenco dei programmi da eliminare, azzerandoli a fine metodo (`self._blur_program = self._bloom_composite_program = None`).

- [ ] **Step 6: Verifica costruzione headless**

```bash
uv run python -c "import queue; from thread.opengl_thread import EqualizerOpenGLThread; EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue()); print('construct OK')"
```
Expected: `construct OK`

- [ ] **Step 7: Verifica visiva (Windows)**

Run: `uv run python main.py` → Fullscreen. Default = bloom OFF → invariato. Anteprima: impostare temporaneamente `bloom_enabled: bool = True`, osservare l'alone luminoso attorno alle barre; provare `bloom_intensity` 0.5 / 1.5; verificare nessun crash e framerate accettabile; ripristinare `False`.

- [ ] **Step 8: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Add true bloom glow (half-res FBO + separable blur + additive composite)"
```

---

## Task 9: GUI — tab "✨ Effetti" end-to-end

**Files:** Modify `gui/main_window_gui.py`

- [ ] **Step 1: Stato GUI degli effetti**

In `__init__`, dopo lo stato forma del Blocco 1 (`self.bars_band_symmetric = False`):

```python
        # Stato GUI effetti reattivi (Blocco 2). Default = OFF.
        self.fx_peakcap_enabled = False
        self.fx_peakcap_color = (1.0, 1.0, 1.0)
        self.fx_peakcap_hold = 0.5
        self.fx_peakcap_fall = 0.8
        self.fx_bloom_enabled = False
        self.fx_bloom_intensity = 1.0
        self.fx_beat_enabled = False
        self.fx_beat_intensity = 0.08
        self.fx_beat_sensitivity = 1.5
```

- [ ] **Step 2: Registrare il tab**

In `_build_settings_tabs`, dopo `self.tabview.addTab(self._build_bars_tab(), "🎛️ Barre")`:

```python
        self.tabview.addTab(self._build_effects_tab(), "✨ Effetti")
```

- [ ] **Step 3: Costruire il tab effetti**

Aggiungere il metodo (dopo `_build_bars_tab`):

```python
    def _build_effects_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        v.addWidget(self._make_label("Peak-cap", font_label()))
        self.fx_peakcap_option = OptionMenuCustomFrame(
            header_name='Peak-cap:', values=["Off", "On"], initial_value="Off",
            command=self.update_peakcap_enabled)
        v.addWidget(self.fx_peakcap_option)
        self.fx_peakcap_color_picker = ColorPickerFrame(
            header_name='Colore cap:', initial_color='#ffffff', command=self.update_peakcap_color)
        v.addWidget(self.fx_peakcap_color_picker)
        self.fx_peakcap_hold_slider = SliderCustomFrame(
            header_name='Hold (ms):', command=self.update_peakcap_hold,
            from_=0, to=2000, initial_value=int(self.fx_peakcap_hold * 1000),
            value_type=SliderCustomFrame.ValueType.INT)
        v.addWidget(self.fx_peakcap_hold_slider)
        self.fx_peakcap_fall_slider = SliderCustomFrame(
            header_name='Caduta (/s):', command=self.update_peakcap_fall,
            from_=0.1, to=3.0, initial_value=self.fx_peakcap_fall,
            value_type=SliderCustomFrame.ValueType.DOUBLE)
        v.addWidget(self.fx_peakcap_fall_slider)

        v.addWidget(self._make_label("Glow (bloom)", font_label()))
        self.fx_bloom_option = OptionMenuCustomFrame(
            header_name='Glow:', values=["Off", "On"], initial_value="Off",
            command=self.update_bloom_enabled)
        v.addWidget(self.fx_bloom_option)
        self.fx_bloom_slider = SliderCustomFrame(
            header_name='Intensità:', command=self.update_bloom_intensity,
            from_=0.0, to=3.0, initial_value=self.fx_bloom_intensity,
            value_type=SliderCustomFrame.ValueType.DOUBLE)
        v.addWidget(self.fx_bloom_slider)

        v.addWidget(self._make_label("Beat pulse", font_label()))
        self.fx_beat_option = OptionMenuCustomFrame(
            header_name='Beat pulse:', values=["Off", "On"], initial_value="Off",
            command=self.update_beat_enabled)
        v.addWidget(self.fx_beat_option)
        self.fx_beat_intensity_slider = SliderCustomFrame(
            header_name='Intensità zoom:', command=self.update_beat_intensity,
            from_=0.0, to=0.3, initial_value=self.fx_beat_intensity,
            value_type=SliderCustomFrame.ValueType.DOUBLE)
        v.addWidget(self.fx_beat_intensity_slider)
        self.fx_beat_sens_slider = SliderCustomFrame(
            header_name='Sensibilità:', command=self.update_beat_sensitivity,
            from_=1.05, to=3.0, initial_value=self.fx_beat_sensitivity,
            value_type=SliderCustomFrame.ValueType.DOUBLE)
        v.addWidget(self.fx_beat_sens_slider)

        v.addStretch(1)
        return tab
```

- [ ] **Step 4: Callback di inoltro**

Aggiungere i metodi (vicino a `update_band_order`):

```python
    def update_peakcap_enabled(self, label):
        self.fx_peakcap_enabled = (label == "On")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_peakcap_enabled", "value": self.fx_peakcap_enabled})

    def update_peakcap_color(self, rgb):
        self.fx_peakcap_color = rgb
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_peakcap_color", "value": rgb})

    def update_peakcap_hold(self, value):
        self.fx_peakcap_hold = float(value) / 1000.0
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_peakcap_hold", "value": float(value) / 1000.0})

    def update_peakcap_fall(self, value):
        self.fx_peakcap_fall = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_peakcap_fall", "value": float(value)})

    def update_bloom_enabled(self, label):
        self.fx_bloom_enabled = (label == "On")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_bloom_enabled", "value": self.fx_bloom_enabled})

    def update_bloom_intensity(self, value):
        self.fx_bloom_intensity = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_bloom_intensity", "value": float(value)})

    def update_beat_enabled(self, label):
        self.fx_beat_enabled = (label == "On")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_beat_enabled", "value": self.fx_beat_enabled})

    def update_beat_intensity(self, value):
        self.fx_beat_intensity = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_beat_intensity", "value": float(value)})

    def update_beat_sensitivity(self, value):
        self.fx_beat_sensitivity = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_beat_sensitivity", "value": float(value)})
```

- [ ] **Step 5: Popolare il costruttore in `fullscreen_command`**

Nella chiamata `EqualizerOpenGLThread(...)`, dopo i parametri forma del Blocco 1 (`band_order_symmetric=...`):

```python
                peakcap_enabled=self.fx_peakcap_enabled,
                peakcap_color=self.fx_peakcap_color,
                peakcap_hold=self.fx_peakcap_hold,
                peakcap_fall=self.fx_peakcap_fall,
                beat_enabled=self.fx_beat_enabled,
                beat_intensity=self.fx_beat_intensity,
                beat_sensitivity=self.fx_beat_sensitivity,
                bloom_enabled=self.fx_bloom_enabled,
                bloom_intensity=self.fx_bloom_intensity,
```

- [ ] **Step 6: Smoke test import GUI (headless)**

```bash
QT_QPA_PLATFORM=offscreen uv run python -c "import gui.main_window_gui; print('OK')" smoketest
```
Expected: `OK` (l'argomento `smoketest` serve come `sys.argv[1]` per `soundcard`).

- [ ] **Step 7: Verifica costruzione GUI (headless)**

```bash
QT_QPA_PLATFORM=offscreen uv run python - smoketest <<'PY'
from PySide6.QtWidgets import QApplication
app = QApplication([])
from gui.main_window_gui import AudioCaptureGUI
w = AudioCaptureGUI()
assert w.fx_peakcap_option.get_value() == "Off"
assert w.fx_bloom_option.get_value() == "Off"
assert w.fx_beat_option.get_value() == "Off"
print("OK")
PY
```
Expected: `OK`

- [ ] **Step 8: Verifica visiva (Windows)**

Run: `uv run python main.py` → Fullscreen → tab **✨ Effetti**. Provare ogni effetto On/Off e i relativi slider, live: peak-cap (colore/hold/caduta), glow (intensità), beat pulse (intensità/sensibilità). Combinazioni con le opzioni del tab Barre. Con tutti Off, identico a prima.

- [ ] **Step 9: Commit**

```bash
git add gui/main_window_gui.py
git commit -m "Add Effetti tab with peak-cap, glow and beat-pulse controls"
```

---

## Task 10: Documentazione e verifica finale

**Files:** Modify `CLAUDE.md` (locale, non committato)

- [ ] **Step 1: Aggiornare `CLAUDE.md`**

Nella sezione "OpenGL-specific concerns", aggiungere un punto:

```markdown
- **Reactive effects (Block 2, OpenGL-only, default OFF):** per-bar **peak-cap** (`_update_peaks` + a dedicated instanced caps shader, colour/hold/fall configurable); **bloom** glow (half-res ping-pong FBOs `GL_RGB16F`, separable gaussian `_blur_program`, additive `_bloom_composite_program`; bars drawn into the FBO via `_draw_bars_instanced`, pipeline skipped when disabled); **beat pulse** (`_detect_beat` on FFT bass energy below `BASS_CUTOFF_HZ` with EMA + refractory, driving a `uBgZoom` background zoom envelope). Control messages `set_peakcap_*`/`set_bloom_*`/`set_beat_*` + constructor params; GUI in the "✨ Effetti" tab. All default OFF → no change/cost unless enabled.
```

- [ ] **Step 2: Verifica finale headless**

```bash
uv run python - <<'PY'
import queue
from thread.opengl_thread import EqualizerOpenGLThread
t = EqualizerOpenGLThread(queue.Queue(), control_queue=queue.Queue())
for m in [
    {"type": "set_peakcap_enabled", "value": True},
    {"type": "set_bloom_enabled", "value": True},
    {"type": "set_beat_enabled", "value": True},
    {"type": "set_bloom_intensity", "value": 1.4},
    {"type": "set_beat_sensitivity", "value": 1.8},
]:
    t.process_control_message(m)
assert t._peakcap_enabled and t._bloom_enabled and t._beat_enabled
assert abs(t._bloom_intensity - 1.4) < 1e-6 and abs(t._beat_sensitivity - 1.8) < 1e-6
# beat: spike dopo energia stabile
for _ in range(20):
    t._detect_beat(1.0, 0.016)
assert t._detect_beat(6.0, 0.016) is True
print("OK")
PY
```
Expected: `OK`

- [ ] **Step 3: Verifica visiva finale (Windows)**

Run: `uv run python main.py`. Esercitare i tre effetti (singoli e combinati) e le opzioni del Blocco 1; verificare che con tutti gli effetti OFF il rendering sia identico a post-Blocco 1 (non-regressione), e che sfondo immagine/video restino ok (incluso lo zoom del beat sullo sfondo).

- [ ] **Step 4: Commit (solo doc tracciati, se presenti)**

`CLAUDE.md` è gitignored: **non** committarlo. Se non ci sono altri file tracciati modificati in questo task, non serve commit. (Il codice è già committato nei task precedenti.)

---

## Note finali

- **Non-regressione:** ogni effetto è default OFF e il suo path è saltato quando disabilitato; il check chiave è "tutto OFF = identico a post-Blocco 1".
- **`_render(heights, dt)`:** la firma cambia nel Task 3 (serve `dt` per i picchi); aggiornare la chiamata in `run()`.
- **Ordine in `_render`:** upload altezze → (bloom prepass+blur, ripristina FBO/viewport) → sfondo (+`uBgZoom`) → barre → peak-cap → composito bloom.
- **Performance bloom:** half-res + `BLOOM_BLUR_PASSES` iterazioni; se a schermo risulta pesante o troppo tenue, tarare passes/risoluzione (candidato dichiarato al tuning).
- **WSL2/Windows:** verifiche logiche headless (`.venv-linux`); shader/FBO/GUI visivi su Windows. Smoke GUI con argomento fittizio (`soundcard`/`sys.argv[1]`).
- Al termine: PR + (se vorrai) release `2.2.0` via `superpowers:finishing-a-development-branch`.
