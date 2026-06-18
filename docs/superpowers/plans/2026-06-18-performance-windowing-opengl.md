# Performance & windowing OpenGL — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Far reggere al renderer OpenGL i 165fps anche con il video di sfondo, passare a finestra borderless windowed, e dotarlo di una misura aggregata del frame time — restando su GL 3.3.

**Architecture:** Tre interventi ortogonali su `thread/opengl_thread.py` più un piccolo helper puro nuovo. (1) Un `FrameTimeStats` puro Python che aggrega il frame time sotto `PERF_LOGS` (lo strumento di misura, costruito per primo). (2) La finestra viene creata **windowed senza decorazioni** dimensionata sul monitor scelto invece che fullscreen esclusiva. (3) L'upload del frame video passa da `glTexSubImage2D` sincrono (che stalla il thread GL) a **PBO double-buffered** (copia CPU→PBO + DMA PBO→texture asincrone).

**Tech Stack:** Python 3.12+, GLFW, PyOpenGL (+accelerate) GL 3.3 core, numpy, gestione dipendenze con `uv`.

## Global Constraints

Ogni task eredita implicitamente questi vincoli (valori copiati dallo spec
`docs/superpowers/specs/2026-06-18-performance-windowing-opengl-design.md`):

- **Resta GL 3.3 core.** Nessun bump di versione del contesto in questo piano (PBO è core da GL 2.1, map-range da 3.0 → niente 4.x).
- **Output visivo invariato.** Nessun cambio a GUI, algoritmo DSP, shader o semantica della control-queue. Si cambia *come* si rende, non *cosa*.
- **Borderless sostituisce il fullscreen esclusivo** (nessuna doppia opzione in GUI).
- **Niente test suite nel repo** (CLAUDE.md). Verifica per tipo: *assert headless* per la logica pura (no pytest — script con `assert` eseguito da `uv run python`), *smoke di import headless* per i cambi GL, e **verifica manuale su Windows (display reale, 165Hz) con `PERF_LOGS`** per il comportamento di rendering reale. **Non introdurre pytest né altre dipendenze.**
- **Ambiente WSL2:** comandi headless via `uv run` (usa `.venv-linux`, vedi memoria `wsl2-venv-and-gl-verification`).
- **Git:** committare come utente **Cioscos**; **NESSUN** trailer `Co-Authored-By` (regola del repo). Lavorare sul branch `feature/opengl-performance-windowing` (già creato).
- **CLAUDE.md è gitignored** in questo repo: aggiornarlo localmente ma **non committarlo** (memoria `claude-md-untracked`).

---

## File Structure

- **Create `thread/frame_time_stats.py`** — helper puro `FrameTimeStats` (nessuna dipendenza GL/numpy). Unica responsabilità: aggregare i delta-tempo dei frame e produrre una riga di riepilogo ogni N frame. Isolato → importabile e verificabile headless.
- **Create `tests/test_frame_time_stats.py`** — verifica headless ad asserzioni pure del helper (eseguibile con `uv run python`, niente pytest).
- **Modify `thread/opengl_thread.py`** — (a) import + costruzione/uso di `FrameTimeStats` nel loop sotto `PERF_LOGS`; (b) `run()`: creazione finestra borderless windowed + hoist di `glPixelStorei`; (c) `__init__`: attributi di stato dei PBO; (d) `_upload_video_frame`: upload via PBO double-buffered; (e) `_cleanup_gl`: free dei PBO.
- **Modify `CLAUDE.md`** (locale, **non committato**) — documentare borderless + path PBO video.

---

### Task 1: Helper di misura `FrameTimeStats` + cablaggio in `PERF_LOGS`

Costruiamo per primo lo strumento di misura: è il gate che convalida il Task 3 e decide l'eventuale escalation. È anche l'unica unità a logica pura → vera TDD headless.

**Files:**
- Create: `thread/frame_time_stats.py`
- Create: `tests/test_frame_time_stats.py`
- Modify: `thread/opengl_thread.py` (import dopo `:16`; init dopo `:1540`; sostituzione blocco `:1608-1610`)

**Interfaces:**
- Consumes: nulla (prima unità).
- Produces: `class FrameTimeStats` con:
  - `__init__(self, report_every: int = 120, budget_s: float = 1.0 / 60.0) -> None`
  - `add(self, dt: float) -> str | None` — accumula un delta-tempo di frame (secondi); ritorna la stringa di riepilogo ogni `report_every` frame (poi azzera la finestra), altrimenti `None`.

- [ ] **Step 1: Scrivi il test che fallisce**

Crea `tests/test_frame_time_stats.py`:

```python
"""
Verifica headless (niente pytest nel repo): asserzioni pure su FrameTimeStats.
Eseguire dalla radice del repo con:  uv run python tests/test_frame_time_stats.py
Uscita 0 + "OK" = passato.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thread.frame_time_stats import FrameTimeStats


def test_no_report_before_window():
    stats = FrameTimeStats(report_every=4, budget_s=0.01)
    assert stats.add(0.005) is None
    assert stats.add(0.005) is None
    assert stats.add(0.005) is None


def test_report_at_window_boundary():
    stats = FrameTimeStats(report_every=4, budget_s=0.01)
    stats.add(0.005)
    stats.add(0.005)
    stats.add(0.015)  # oltre budget
    report = stats.add(0.005)
    assert report is not None, "il 4° frame deve produrre un report"
    # avg = (0.005+0.005+0.015+0.005)/4 = 0.0075 s = 7.50 ms
    assert "avg=7.50ms" in report, report
    assert "min=5.00ms" in report, report
    assert "max=15.00ms" in report, report
    # 1 frame su 4 oltre budget = 25%
    assert "oltre-budget=25%" in report, report


def test_resets_after_report():
    stats = FrameTimeStats(report_every=2, budget_s=0.01)
    assert stats.add(0.005) is None
    assert stats.add(0.005) is not None   # primo report
    assert stats.add(0.005) is None       # finestra azzerata
    assert stats.add(0.005) is not None   # secondo report


def main():
    test_no_report_before_window()
    test_report_at_window_boundary()
    test_resets_after_report()
    print("OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Esegui il test e verifica che fallisce**

Run: `uv run python tests/test_frame_time_stats.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'thread.frame_time_stats'`

- [ ] **Step 3: Implementa il helper**

Crea `thread/frame_time_stats.py`:

```python
"""
Statistiche aggregate sul frame time per la diagnosi di performance del renderer
OpenGL. Puro Python, nessuna dipendenza GL → importabile e verificabile headless.

Accumula i delta-tempo dei frame e, ogni `report_every` frame, produce una riga
di riepilogo (avg/min/max + percentuale di frame oltre il budget) usata sotto il
flag PERF_LOGS per confrontare le performance prima/dopo un intervento.
"""
from __future__ import annotations


class FrameTimeStats:
    def __init__(self, report_every: int = 120, budget_s: float = 1.0 / 60.0) -> None:
        """
        Args:
            report_every: numero di frame per ogni riga di riepilogo (>= 1).
            budget_s: budget di frame time in secondi; i frame più lenti di questo
                contano come "oltre budget" (tipicamente 1/refresh_rate).
        """
        if report_every < 1:
            raise ValueError("report_every must be >= 1")
        self._report_every = int(report_every)
        self._budget_s = float(budget_s)
        self._reset()

    def _reset(self) -> None:
        self._count = 0
        self._sum = 0.0
        self._min = float("inf")
        self._max = 0.0
        self._over_budget = 0

    def add(self, dt: float) -> str | None:
        """
        Accumula un delta-tempo di frame (secondi). Ritorna una stringa di
        riepilogo ogni `report_every` frame (e poi azzera la finestra), altrimenti
        None.
        """
        self._count += 1
        self._sum += dt
        if dt < self._min:
            self._min = dt
        if dt > self._max:
            self._max = dt
        if dt > self._budget_s:
            self._over_budget += 1

        if self._count < self._report_every:
            return None

        report = self._format()
        self._reset()
        return report

    def _format(self) -> str:
        avg = self._sum / self._count
        over_pct = 100.0 * self._over_budget / self._count
        avg_fps = 1.0 / avg if avg > 0 else 0.0
        return (
            f"[PERF] {self._count} frame: "
            f"avg={avg * 1e3:.2f}ms min={self._min * 1e3:.2f}ms max={self._max * 1e3:.2f}ms "
            f"oltre-budget={over_pct:.0f}% "
            f"(budget={self._budget_s * 1e3:.2f}ms, ~{avg_fps:.0f}fps medi)"
        )
```

- [ ] **Step 4: Esegui il test e verifica che passa**

Run: `uv run python tests/test_frame_time_stats.py`
Expected: stampa `OK`, exit code 0

- [ ] **Step 5: Cabla l'helper in `opengl_thread.py` — import**

In `thread/opengl_thread.py`, dopo la riga 16 (`from thread.video_decode_thread import VideoDecodeThread`), aggiungi:

```python
from thread.frame_time_stats import FrameTimeStats
```

- [ ] **Step 6: Cabla — costruzione dello stats object nel `run()`**

In `run()`, subito dopo `last_frame_time = glfw.get_time()` (riga ~1540), aggiungi:

```python
            # Statistiche aggregate del frame time (solo sotto PERF_LOGS): budget
            # = 1/refresh del monitor → "oltre-budget" conta i frame che mancano il vblank.
            frame_stats = FrameTimeStats(report_every=120, budget_s=1.0 / max(self.frame_rate, 1)) if PERF_LOGS else None
```

- [ ] **Step 7: Cabla — sostituisci la stampa per-frame nel loop**

In `run()`, sostituisci il blocco esistente (righe ~1608-1610):

```python
                if PERF_LOGS and frame_delta > 0:
                    fps = 1.0 / frame_delta
                    print(f"PERF LOG: FrameTime = {frame_delta * 1000:.3f}ms (FPS ~ {fps:.1f})")
```

con:

```python
                if frame_stats is not None and frame_delta > 0:
                    report = frame_stats.add(frame_delta)
                    if report:
                        print(report)
```

- [ ] **Step 8: Smoke di import headless**

Run (dalla radice del repo): `uv run python -c "import thread.opengl_thread; print('import OK')"`
Expected: stampa `import OK` senza traceback (conferma che import + cablaggio non rompono il modulo).

- [ ] **Step 9: Commit**

```bash
git add thread/frame_time_stats.py tests/test_frame_time_stats.py thread/opengl_thread.py
git commit -m "Add aggregated frame-time PERF_LOGS stats helper"
```

---

### ⏱️ Checkpoint di misura (MANUALE — utente, Windows 165Hz)

Prima di applicare il fix, cattura la **baseline** con lo strumento appena aggiunto, così il prima/dopo è quantificato:

1. In `thread/opengl_thread.py` metti `PERF_LOGS = True` (riga 25). *(Modifica locale temporanea, da rimettere a `False` a fine lavoro — non committare il `True`.)*
2. `uv run python main.py`, avvia il fullscreen **con un video di sfondo**.
3. Annota alcune righe `[PERF] ...` (atteso: `avg` vicino/oltre il budget, `oltre-budget%` alto, `~fps medi` ~140-150 → riproduce il sintomo).
4. Ripeti **senza** video (atteso: ~165fps, `oltre-budget≈0%`).

Questi numeri sono il riferimento per validare il Task 3.

---

### Task 2: Finestra borderless windowed

Sostituisce la creazione fullscreen esclusiva con una finestra senza decorazioni dimensionata e posizionata sul monitor scelto. Intervento indipendente e isolato al solo `run()`.

**Files:**
- Modify: `thread/opengl_thread.py` (blocco creazione finestra `:1489-1499`)

**Interfaces:**
- Consumes: `selected_monitor` (oggetto monitor GLFW già risolto sopra nel `run()`), `self.window_width`, `self.window_height` (già impostati dal video mode).
- Produces: nessuna nuova interfaccia (cambia solo il modo di creare `window`).

- [ ] **Step 1: Sostituisci il blocco di creazione finestra**

In `run()`, sostituisci le righe ~1489-1499:

```python
            # Richiedi esplicitamente un contesto OpenGL 3.3 core (shader + instancing).
            glfw.default_window_hints()
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
            # MSAA: richiedi un framebuffer di default multisample (antialiasing hardware).
            glfw.window_hint(glfw.SAMPLES, MSAA_SAMPLES)

            window = glfw.create_window(self.window_width, self.window_height, "Audio Visualizer", selected_monitor, None)
            if not window:
                raise Exception("GLFW window creation failed")
```

con:

```python
            # Richiedi esplicitamente un contesto OpenGL 3.3 core (shader + instancing).
            glfw.default_window_hints()
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
            # MSAA: richiedi un framebuffer di default multisample (antialiasing hardware).
            glfw.window_hint(glfw.SAMPLES, MSAA_SAMPLES)
            # Borderless windowed (NON fullscreen esclusivo): finestra senza bordo/titolo,
            # dimensionata e posizionata sul monitor scelto. Niente cambio di modalità video
            # → alt-tab istantaneo e (su Windows) present al refresh del monitor senza il
            # mode-switch dell'esclusivo. NON FLOATING: come le borderless dei giochi
            # moderni, l'alt-tab la manda dietro.
            glfw.window_hint(glfw.DECORATED, glfw.FALSE)

            # Posizione del monitor scelto nello spazio virtuale multi-monitor.
            monitor_x, monitor_y = glfw.get_monitor_pos(selected_monitor)

            # Finestra WINDOWED (monitor=None): è la borderless, non l'esclusiva.
            window = glfw.create_window(self.window_width, self.window_height, "Audio Visualizer", None, None)
            if not window:
                raise Exception("GLFW window creation failed")

            # Posizionala esattamente sul monitor scelto così lo copre tutto.
            glfw.set_window_pos(window, monitor_x, monitor_y)
```

- [ ] **Step 2: Smoke di import headless**

Run: `uv run python -c "import thread.opengl_thread; print('import OK')"`
Expected: `import OK` senza traceback.

- [ ] **Step 3: Verifica manuale (MANUALE — utente, Windows)**

Avvia `uv run python main.py` e apri il fullscreen. Verifica:
- la finestra copre **tutto** il monitor selezionato, **senza bordo né barra del titolo**;
- **alt-tab** è istantaneo (niente flash/cambio risoluzione) e manda la finestra dietro;
- **ESC** chiude la finestra;
- con il selettore monitor su un secondo schermo, la finestra appare **su quel** monitor;
- con `PERF_LOGS = True`, **senza** video gli fps medi restano ~165 (`oltre-budget≈0%`) come prima.

- [ ] **Step 4: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Open the OpenGL window as borderless windowed instead of exclusive fullscreen"
```

---

### Task 3: Upload video asincrono via PBO double-buffered (+ hoist `glPixelStorei`)

Il fix dei 165fps col video: rimuove lo stallo sincrono di `glTexSubImage2D` da puntatore client usando 2 PBO in ring (copia CPU→PBO + DMA PBO→texture asincrone).

**Files:**
- Modify: `thread/opengl_thread.py` (`__init__` dopo `:603`; `run()` dopo `:1520`; `_upload_video_frame` `:1775-1804`; `_cleanup_gl` dopo `:1219`)

**Interfaces:**
- Consumes: `self._video_texture`, `self._video_tex_size`, `frame: np.ndarray` (HxWx3 uint8 RGB), `ctypes` (già importato), `FrameTimeStats` del Task 1 per la verifica.
- Produces: nuovi attributi di stato `self._video_pbos: list | None`, `self._video_pbo_index: int`, `self._video_pbo_size: int` (consumati solo internamente da `_upload_video_frame`/`_cleanup_gl`).

- [ ] **Step 1: Aggiungi gli attributi di stato dei PBO in `__init__`**

In `__init__`, subito dopo la riga 603 (`self._video_time_acc = 0.0 ...`), aggiungi:

```python
        self._video_pbos = None             # 2 PBO (ping-pong) per upload async; None = non creati
        self._video_pbo_index = 0           # indice del PBO corrente nel ring
        self._video_pbo_size = 0            # byte allocati per ciascun PBO; 0 = non allocati
```

- [ ] **Step 2: Hoist di `glPixelStorei` nel `run()` (una volta sola)**

In `run()`, subito dopo `glClearColor(0.0, 0.0, 0.0, 1.0)` (riga ~1520), aggiungi:

```python
            # RGB a 3 byte/pixel con larghezza non multipla di 4: forziamo una volta
            # per tutte l'allineamento di unpack a 1 byte (lo usa l'upload video via
            # PBO; sicuro anche per le texture RGBA, sempre 4-allineate).
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
```

- [ ] **Step 3: Riscrivi `_upload_video_frame` per usare i PBO**

Sostituisci l'intero metodo `_upload_video_frame` (righe ~1775-1804) con:

```python
    def _upload_video_frame(self, frame: np.ndarray) -> None:
        """
        Carica un frame video (np.ndarray HxWx3 uint8 RGB, riga 0 in alto) nella
        texture video tramite PBO double-buffered: la copia CPU→PBO e il
        trasferimento PBO→texture (DMA) sono asincroni, così l'upload NON stalla il
        thread GL (a differenza di glTexSubImage2D da puntatore client). Alloca al
        primo frame o quando cambiano le dimensioni. Il flip verticale è demandato
        allo shader (uFlipV=1), non copiato in CPU. glPixelStorei(UNPACK_ALIGNMENT,1)
        è impostato una volta in run().
        """
        h, w = int(frame.shape[0]), int(frame.shape[1])
        size = w * h * 3  # RGB, 3 byte/pixel
        frame = np.ascontiguousarray(frame)  # il PBO copia memoria contigua

        # Crea (lazy) la texture al primo frame.
        if self._video_texture is None:
            self._video_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self._video_texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            self._video_tex_size = None

        # Crea (lazy) i 2 PBO del ring.
        if self._video_pbos is None:
            self._video_pbos = list(glGenBuffers(2))
            self._video_pbo_index = 0
            self._video_pbo_size = 0

        # (Ri)alloca la texture al primo frame o al cambio dimensione (storage vuoto).
        if self._video_tex_size != (w, h):
            glBindTexture(GL_TEXTURE_2D, self._video_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB,
                         GL_UNSIGNED_BYTE, None)
            self._video_tex_size = (w, h)

        # (Ri)alloca i PBO al cambio dimensione.
        if self._video_pbo_size != size:
            for pbo in self._video_pbos:
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
                glBufferData(GL_PIXEL_UNPACK_BUFFER, size, None, GL_STREAM_DRAW)
            self._video_pbo_size = size

        idx = self._video_pbo_index

        # 1) Copia CPU → PBO[idx]: orphaning (storage fresco → niente attesa sulla
        #    DMA del frame precedente) + scrittura dei byte del frame.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._video_pbos[idx])
        glBufferData(GL_PIXEL_UNPACK_BUFFER, size, None, GL_STREAM_DRAW)
        glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, size, frame)

        # 2) PBO[idx] → texture: glTexSubImage2D legge dal PBO bound e ritorna subito
        #    (DMA asincrona). NB: con un PBO bound l'ultimo argomento è un OFFSET nel
        #    buffer, non un puntatore client → ctypes.c_void_p(0).
        glBindTexture(GL_TEXTURE_2D, self._video_texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB,
                        GL_UNSIGNED_BYTE, ctypes.c_void_p(0))

        # 3) Sbinda il PBO: indispensabile, altrimenti la successiva glTexImage2D/
        #    glTexSubImage2D dell'immagine di sfondo interpreterebbe il suo puntatore
        #    come offset in questo PBO.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        # Alterna i due PBO: il frame N+1 usa l'altro buffer, così la copia CPU non
        # aspetta la DMA del frame N.
        self._video_pbo_index = 1 - idx
```

- [ ] **Step 4: Libera i PBO in `_cleanup_gl`**

In `_cleanup_gl`, subito dopo il blocco che libera `self._video_texture` (righe ~1217-1219), aggiungi:

```python
        if self._video_pbos:
            _safe(lambda: glDeleteBuffers(len(self._video_pbos), self._video_pbos))
            self._video_pbos = None
            self._video_pbo_size = 0
```

- [ ] **Step 5: Smoke di import headless**

Run: `uv run python -c "import thread.opengl_thread; print('import OK')"`
Expected: `import OK` senza traceback.

- [ ] **Step 6: Verifica manuale + misura (MANUALE — utente, Windows, `PERF_LOGS = True`)**

Avvia `uv run python main.py`, apri il fullscreen **con video di sfondo** e confronta con la baseline:
- il video si vede correttamente (orientamento giusto, niente righe sfalsate/colori storti su larghezze non multiple di 4);
- `[PERF]` mostra `avg` **sotto** il budget e gli `~fps medi` risaliti a **~165**, `oltre-budget%` crollato vs baseline;
- **switch immagine ↔ video** a runtime (via GUI): entrambi gli sfondi si caricano correttamente (verifica che il rebind del PBO a 0 non abbia rotto l'upload dell'immagine);
- nessun glitch al **cambio di video** di dimensione diversa (riallocazione PBO/texture);
- chiusura pulita (niente errori GL allo stop).

- [ ] **Step 7: Commit**

```bash
git add thread/opengl_thread.py
git commit -m "Upload video frames via double-buffered PBO to avoid GL-thread stalls"
```

---

### Task 4: Verifica d'integrazione, documentazione e decisione di escalation

Chiude il lavoro: regressione cross-modalità, aggiornamento doc locale, e il gate che decide se serve l'escalation (UBO/4.x) — che resta **fuori** da questo piano.

**Files:**
- Modify: `CLAUDE.md` (locale, **NON committato**)

- [ ] **Step 1: Regressione manuale cross-modalità (MANUALE — utente, Windows)**

Con il fullscreen attivo, scorri tutte le combinazioni e verifica che nulla sia regredito rispetto a prima del lavoro:
- modalità **Barre / Radiale / Oscilloscopio / Linea** (con e senza ordine simmetrico);
- effetti **peak-cap / bloom / beat** on/off;
- sfondo **immagine** e **video**;
- cambio **numero di bande** a runtime;
- con `PERF_LOGS = True`: in ogni combinazione gli fps medi restano ~165 (il video non fa più scendere a 140-150).

- [ ] **Step 2: Aggiorna `CLAUDE.md` (locale, non committato)**

Nella sezione "OpenGL-specific concerns" / "Video background" documenta in una frase ciascuno:
- la finestra è ora **borderless windowed** (no fullscreen esclusivo): creata `monitor=None` + `DECORATED=FALSE` + `set_window_pos` sul monitor scelto;
- l'upload del frame video usa **PBO double-buffered** in `_upload_video_frame` (async, niente stallo); `glPixelStorei(GL_UNPACK_ALIGNMENT,1)` è hoisted in `run()`; PBO liberati in `_cleanup_gl`;
- sotto `PERF_LOGS` il frame time è riassunto da `FrameTimeStats` (`thread/frame_time_stats.py`).

**Non eseguire `git add CLAUDE.md`** (gitignored, vedi memoria `claude-md-untracked`).

- [ ] **Step 3: Rimetti `PERF_LOGS = False` (se era stato messo a True per le misure)**

Verifica che la riga 25 di `thread/opengl_thread.py` sia `PERF_LOGS = False`. Se l'hai cambiata per misurare, riportala a `False` e includila nell'ultimo commit se necessario:

Run: `uv run python -c "import thread.opengl_thread as o; print('PERF_LOGS =', o.PERF_LOGS)"`
Expected: `PERF_LOGS = False`

- [ ] **Step 4: Decisione di escalation (gate)**

Sulla base delle misure del Task 3/Step 1:
- **Se i 165fps col video sono stabili** (`oltre-budget≈0%`): **chiudi qui.** Gli interventi del piano bastano.
- **Se il thread GL è ancora CPU-bound** (avg vicino al budget anche senza video, o `oltre-budget%` non trascurabile): apri lo **Step 1 dell'escalation** dallo spec (UBO + skip uniform ridondanti, ancora su 3.3) come **nuovo** spec/piano dedicato. Il 4.x (persistent-mapped/DSA) e il DSP off-thread restano l'ultima risorsa, solo se la misura lo impone. **Non** sono parte di questo piano.

---

## Note di scope

- **Preallocazione scratch numpy (intervento C dello spec):** valutata e **rinviata**. L'unica preallocazione "pulita" sarebbe sui path osc/line `build_*`, ma cambierebbe la firma dei builder (non pulita) e quei path **non** sono il collo di bottiglia riportato (barre + video). Coerente col qualificatore dello spec "solo dove è pulito" e col principio "misura prima": se la misura lo richiede, rientra nell'escalation. Gli altri due item dell'intervento C — hoist `glPixelStorei` (Task 3) e `PERF_LOGS` aggregato (Task 1) — sono implementati.
- **Escalation (UBO, GL 4.x, DSP off-thread):** fuori da questo piano per design; gated sulla misura (Task 4/Step 4).

## Self-Review

**1. Copertura spec:**
- Obiettivo "165fps col video" → Task 3 (PBO) + verifica misurata Task 3/Step 6. ✓
- Obiettivo "borderless windowed" → Task 2. ✓
- Obiettivo "margine frame time misurato prima/dopo" → Task 1 (`FrameTimeStats`) + checkpoint baseline + Task 3/Step 6. ✓
- "Resta GL 3.3" → vincolo globale; PBO/map-range sono ≤3.0. ✓
- Intervento C: hoist `glPixelStorei` (Task 3/Step 2) ✓, `PERF_LOGS` aggregato (Task 1) ✓, prealloc → rinviata con motivazione (Note di scope) ✓.
- Edge case "rebind PBO a 0" → Task 3/Step 3 (commento + codice) + verifica switch immagine↔video Task 3/Step 6. ✓
- Tier di escalation → Task 4/Step 4 + Note di scope. ✓
- Verifica cross-modalità → Task 4/Step 1. ✓
- Doc CLAUDE.md locale non committata → Task 4/Step 2. ✓

**2. Scansione placeholder:** nessun TBD/TODO; ogni step ha codice/comando completo. ✓

**3. Consistenza dei tipi:** `FrameTimeStats(report_every, budget_s)` e `add(dt) -> str | None` usati in modo identico in helper, test e cablaggio (Task 1/Step 6-7). Attributi `_video_pbos`/`_video_pbo_index`/`_video_pbo_size` definiti in `__init__` (Task 3/Step 1) e usati con gli stessi nomi in `_upload_video_frame` e `_cleanup_gl`. ✓
