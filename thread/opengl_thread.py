import threading
import math
import time
import queue
import ctypes
from typing import Optional, Callable

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
from PIL import Image
from PIL.Image import Transpose

from thread.AudioBufferAccumulator import AudioBufferAccumulator

RATE = 44100
N_FFT = 4096
MIN_FREQ = 20
MAX_FREQ = 20000

GENERAL_LOG = False
FFT_LOGS = False
PERF_LOGS = False

# --- Taratura fedeltà/latenza (regolabili a vista) ---
# Scala dB assoluta: l'altezza barra riflette il livello reale del segnale.
DB_FLOOR = -60.0        # dB: sotto questo livello la barra è a 0
DB_CEILING = 0.0        # dB: fondo scala (barra piena). Abbassare (es. -10) per riempire prima.
# Ballistica delle barre (costanti di tempo, frame-rate-indipendenti):
ATTACK_TAU = 0.012      # s: salita rapida → cattura i transienti (kick/snare) quasi all'istante
RELEASE_TAU = 0.220     # s: discesa morbida → falloff leggibile in stile peak-meter
# Compensazione spettrale opzionale (0 = off). +3.0 ≈ tilt "pink" (+3 dB/ottava).
TILT_DB_PER_OCT = 0.0

# --- Aspetto delle barre (ex magic number) ---
BAR_WIDTH_FRAC = 0.8    # frazione dello slot occupata dalla barra (il resto è spazio)
GREEN_SPLIT = 0.5       # sotto questa frazione d'altezza finestra → verde
YELLOW_SPLIT = 0.8      # tra GREEN_SPLIT e questa → giallo; oltre → rosso

# --- Shader GLSL (OpenGL 3.3 core) ---
# Barre: rendering instanced. Il quad unitario [0,1]² viene espanso per ogni barra
# usando gl_InstanceID e l'altezza per-istanza; nessuna geometria generata dalla CPU.
_BARS_VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec2 aUnitPos;   // quad unitario: x,y in [0,1]
layout(location = 1) in float aHeight;   // altezza normalizzata per-istanza [0,1]
uniform float uNumBars;
uniform float uBarWidthFrac;
out float vFill;                          // frazione di altezza finestra del frammento
void main() {
    float slot = 1.0 / uNumBars;
    float xLeft = (float(gl_InstanceID) + 0.5 * (1.0 - uBarWidthFrac)) * slot;
    float x = xLeft + aUnitPos.x * (slot * uBarWidthFrac);
    float y = aUnitPos.y * aHeight;
    vFill = y;
    gl_Position = vec4(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
"""

# Lo split verde/giallo/rosso è calcolato per-fragment sulla GPU (replica esatta dei
# tagli usati dalla pipeline fissa precedente, ma netto al pixel).
_BARS_FRAGMENT_SRC = """
#version 330 core
in float vFill;
out vec4 FragColor;
uniform float uGreenSplit;
uniform float uYellowSplit;
void main() {
    vec3 col;
    if (vFill < uGreenSplit)        col = vec3(0.0, 1.0, 0.0);
    else if (vFill < uYellowSplit)  col = vec3(1.0, 1.0, 0.0);
    else                            col = vec3(1.0, 0.0, 0.0);
    FragColor = vec4(col, 1.0);
}
"""

# Sfondo: quad fullscreen testurizzato. L'alpha è una uniform → nessuna ricreazione
# della texture quando cambia (niente più re-upload né leak).
_BG_VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;        // NDC [-1,1]
layout(location = 1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main() {
    vTexCoord = aTexCoord;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

_BG_FRAGMENT_SRC = """
#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform float uAlpha;
void main() {
    vec4 c = texture(uTexture, vTexCoord);
    FragColor = vec4(c.rgb, c.a * uAlpha);
}
"""


class EqualizerOpenGLThread(threading.Thread):
    """
    Thread per la visualizzazione OpenGL dell'equalizzatore audio.

    Pipeline DSP (ottimizzata per fedeltà e latenza, NON più allineata a Tkinter):
        finestra Blackman → rfft (una sola volta, vettoriale sui canali) →
        potenza per bin normalizzata sul guadagno finestra → somma per banda
        logaritmica via bincount → conversione in dB con riferimento FISSO →
        mappa [DB_FLOOR, DB_CEILING] in [0,1] → noise gate → ballistica
        attacco/rilascio. L'FFT viene ricalcolata a ogni frame sulla finestra
        più fresca fornita dal ring buffer.

    Attributes:
        audio_queue: Coda per ricevere i dati audio
        control_queue: Coda per i messaggi di controllo
        noise_threshold: Soglia di rumore normalizzata (0-1)
        channels: Numero di canali audio (default 2 per stereo)
        n_bands: Numero di bande di frequenza
        monitor: Monitor per fullscreen
        bg_image: Immagine di sfondo
        bg_alpha: Trasparenza dello sfondo
        window_close_callback: Callback alla chiusura della finestra
    """

    def __init__(self,
                 audio_queue,
                 noise_threshold=0.1,
                 channels=2,
                 n_bands=9,
                 control_queue=None,
                 workers: int = 5,
                 auto_workers: bool = False,
                 monitor=None,
                 bg_image: Optional[Image.Image] = None,
                 bg_alpha: Optional[float] = None,
                 window_close_callback: Callable = None):
        super().__init__()
        self._audio_queue = audio_queue
        self._control_queue = control_queue
        self._noise_threshold = noise_threshold
        self._channels = channels
        self._n_bands = n_bands
        self._monitor = monitor
        self._bg_image = bg_image
        self._bg_alpha = bg_alpha
        self.window_close_callback = window_close_callback
        # NOTA: `workers`/`auto_workers` sono accettati per compatibilità con il
        # chiamante ma ora inerti: l'FFT è single-pass vettoriale (rfft) e non usa
        # più ProcessPoolExecutor, che aggiungeva latenza (pickling/IPC) ricalcolando
        # l'intera FFT per ogni worker.

        # Oggetti OpenGL (creati in init_gl_objects una volta attivo il contesto)
        self._bars_program = None
        self._bg_program = None
        self._bars_vao = None
        self._bars_quad_vbo = None
        self._heights_vbo = None      # VBO delle altezze per-istanza (N float)
        self._heights_capacity = 0    # n. di float attualmente allocati nel VBO
        self._bg_vao = None
        self._bg_vbo = None
        self._bars_uniforms = {}
        self._bg_uniforms = {}

        self.stop_event = threading.Event()
        self._background_texture = None

        self.window_width = self.window_height = None
        self.frame_rate = None

        # Finestra Blackman e guadagno coerente (precalcolati: lunghezza fissa N_FFT).
        # Dividere |FFT| per window_gain calibra un tono a fondo scala a ~0 dB.
        self._window = np.blackman(N_FFT).astype(np.float32)
        self._window_gain = float(np.sum(self._window)) / 2.0

        # Ring buffer snapshot: fornisce sempre gli ultimi N_FFT frame (latenza minima)
        self.audio_accumulator = AudioBufferAccumulator(window_size=N_FFT,
                                                        channels=self._channels)

        # Bande di frequenza + mappa bin→banda + buffer ampiezze
        self._setup_bands(self._n_bands)

    def _setup_bands(self, num_bands: int) -> None:
        """
        (Ri)calcola le bande di frequenza, la mappa bin→banda per l'aggregazione
        vettoriale e azzera i buffer delle ampiezze. Da chiamare in __init__ e a
        ogni cambio del numero di bande.
        """
        self.frequency_bands = self.generate_frequency_bands(num_bands)
        self._compute_band_bins()

        n = len(self.frequency_bands)
        self.target_amplitudes = np.zeros(n, dtype=np.float32)
        self.current_amplitudes = np.zeros(n, dtype=np.float32)

    def _compute_band_bins(self) -> None:
        """
        Precalcola, per ogni bin della rfft, l'indice della banda di appartenenza
        (-1 se fuori da ogni banda) così l'aggregazione per banda si fa in O(bin)
        con np.bincount invece di un loop di maschere booleane O(bande×bin).
        Calcola anche l'eventuale tilt percettivo per banda.
        """
        freqs = np.fft.rfftfreq(N_FFT, 1.0 / RATE)  # len N_FFT//2 + 1
        bin_band = np.full(freqs.shape, -1, dtype=np.int64)
        for b, (low_freq, high_freq) in enumerate(self.frequency_bands):
            bin_band[(freqs >= low_freq) & (freqs <= high_freq)] = b
        self._bin_band = bin_band
        self._valid_bins = bin_band >= 0

        # Tilt spettrale opzionale (dB per banda), basato sulla freq. centrale geometrica
        if TILT_DB_PER_OCT != 0.0:
            centers = np.array([
                math.sqrt(max(low, 1e-9) * max(high, 1e-9))
                for low, high in self.frequency_bands
            ])
            self._tilt_db = TILT_DB_PER_OCT * np.log2(np.maximum(centers, 1e-9) / 1000.0)
        else:
            self._tilt_db = 0.0

    def generate_frequency_bands(self, num_bands: int):
        """
        Genera num_bands bande in scala (quasi) logaritmica tra MIN_FREQ e MAX_FREQ,
        garantendo bordi strettamente crescenti e larghezza >= un bin FFT: nessuna
        banda degenere (es. (MAX, MAX)) nemmeno a conteggi elevati. Restituisce
        esattamente num_bands tuple (low, high).

        Args:
            num_bands (int): Numero di bande desiderate.

        Returns:
            list[tuple[float, float]]: Lista di tuple (freq_min, freq_max).
        """
        if num_bands < 1:
            return []
        if num_bands == 1:
            return [(float(MIN_FREQ), float(MAX_FREQ))]

        bin_width = RATE / N_FFT  # larghezza minima di una banda (un bin)

        # Bordi logaritmici ideali: edges[0]=MIN_FREQ, edges[num_bands]=MAX_FREQ
        ratio = (MAX_FREQ / MIN_FREQ) ** (1.0 / num_bands)
        edges = [MIN_FREQ * (ratio ** i) for i in range(num_bands + 1)]
        edges[0] = float(MIN_FREQ)
        edges[num_bands] = float(MAX_FREQ)

        # Passata in avanti: impone la spaziatura minima (alza i bordi troppo vicini,
        # tipico alle basse frequenze dove il log è più fitto del bin width).
        for i in range(1, num_bands + 1):
            if edges[i] < edges[i - 1] + bin_width:
                edges[i] = edges[i - 1] + bin_width

        # Passata all'indietro: riancora l'ultimo bordo a MAX_FREQ e, se la passata in
        # avanti ha compresso la coda, ridistribuisce mantenendo la spaziatura minima.
        # (Per num_bands <= 100 lo spazio disponibile è ampiamente sufficiente.)
        edges[num_bands] = float(MAX_FREQ)
        for i in range(num_bands - 1, 0, -1):
            if edges[i] > edges[i + 1] - bin_width:
                edges[i] = edges[i + 1] - bin_width

        return [(edges[i], edges[i + 1]) for i in range(num_bands)]

    def create_equalizer(self,
                         audio_data: np.ndarray,
                         frequency_bands: list[tuple[float, float]],
                         noise_threshold: float = 0.1,
                         channels: int = 2) -> np.ndarray:
        """
        Calcola le ampiezze per-banda (livello normalizzato in [0,1]) per una
        finestra audio multi-canale, su scala dB con riferimento fisso.

        Args:
            audio_data (np.ndarray): shape (N_FFT, channels).
            frequency_bands (list): lista di tuple (low_freq, high_freq).
            noise_threshold (float): soglia di rumore normalizzata in [0,1].
            channels (int): numero di canali (2 per stereo).

        Returns:
            np.ndarray: livello per ogni banda in [0..1].
        """
        # (N_FFT, channels) → (channels, N_FFT)
        samples = audio_data.T

        # Finestra Blackman (broadcast su tutti i canali) + rfft (solo freq. positive)
        windowed = samples * self._window                       # (channels, N_FFT)
        spectrum = np.fft.rfft(windowed, axis=1)                # (channels, N_FFT//2 + 1)

        # Potenza per bin, normalizzata sul guadagno finestra (tono a fondo scala ~ 0 dB)
        power = (np.abs(spectrum) / self._window_gain) ** 2
        # Media della potenza tra i canali
        power = power.mean(axis=0)                              # (N_FFT//2 + 1,)

        # Somma dell'energia per banda in O(bin) (niente loop di maschere)
        band_power = np.bincount(
            self._bin_band[self._valid_bins],
            weights=power[self._valid_bins],
            minlength=len(frequency_bands)
        )

        # Conversione in dB (+ tilt percettivo opzionale)
        band_db = 10.0 * np.log10(band_power + 1e-12) + self._tilt_db

        # Mappa [DB_FLOOR, DB_CEILING] → [0, 1] con riferimento FISSO (no per-frame max)
        level = (band_db - DB_FLOOR) / (DB_CEILING - DB_FLOOR)
        np.clip(level, 0.0, 1.0, out=level)

        # Noise gate in termini normalizzati
        level[level < noise_threshold] = 0.0

        if FFT_LOGS:
            print('LOGGING NORMALIZED LEVELS (dB scale)')
            for i, lvl in enumerate(level):
                print(f"Band {i} level = {lvl:.4f}")

        return level

    def init_gl_objects(self):
        """
        Compila gli shader e crea VAO/VBO (una sola volta, a contesto GL attivo).
        Le barre usano rendering instanced: un quad unitario statico condiviso + un
        VBO di altezze per-istanza aggiornato ogni frame. Lo sfondo è un quad
        fullscreen testurizzato con alpha via uniform.
        """
        # --- Programmi shader ---
        self._bars_program = compileProgram(
            compileShader(_BARS_VERTEX_SRC, GL_VERTEX_SHADER),
            compileShader(_BARS_FRAGMENT_SRC, GL_FRAGMENT_SHADER),
        )
        self._bg_program = compileProgram(
            compileShader(_BG_VERTEX_SRC, GL_VERTEX_SHADER),
            compileShader(_BG_FRAGMENT_SRC, GL_FRAGMENT_SHADER),
        )
        self._bars_uniforms = {
            name: glGetUniformLocation(self._bars_program, name)
            for name in ("uNumBars", "uBarWidthFrac", "uGreenSplit", "uYellowSplit")
        }
        self._bg_uniforms = {
            name: glGetUniformLocation(self._bg_program, name)
            for name in ("uTexture", "uAlpha")
        }

        # --- VAO/VBO barre ---
        # Quad unitario [0,1]² come due triangoli (geometria statica condivisa).
        quad = np.array([
            0.0, 0.0,  1.0, 0.0,  1.0, 1.0,
            0.0, 0.0,  1.0, 1.0,  0.0, 1.0,
        ], dtype=np.float32)

        self._bars_vao = glGenVertexArrays(1)
        glBindVertexArray(self._bars_vao)

        self._bars_quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._bars_quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        # VBO altezze per-istanza (una float per barra, aggiornata ogni frame).
        self._heights_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._heights_vbo)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glVertexAttribDivisor(1, 1)
        self._heights_capacity = 0
        self._ensure_heights_capacity(len(self.frequency_bands))

        glBindVertexArray(0)

        # --- VAO/VBO sfondo (quad fullscreen NDC + texcoord interleaved) ---
        # pos.xy, tex.uv; bottom-left tex=(0,0) per combaciare col FLIP in load_texture.
        bg = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0, -1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 1.0,
        ], dtype=np.float32)

        self._bg_vao = glGenVertexArrays(1)
        glBindVertexArray(self._bg_vao)
        self._bg_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._bg_vbo)
        glBufferData(GL_ARRAY_BUFFER, bg.nbytes, bg, GL_STATIC_DRAW)
        stride = 4 * 4  # 4 float per vertice
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(2 * 4))
        glBindVertexArray(0)

    def _ensure_heights_capacity(self, n: int) -> None:
        """
        Garantisce che il VBO delle altezze possa contenere almeno n float.
        Ridimensiona (ri)allocando lo storage solo quando serve più spazio.
        """
        if n <= self._heights_capacity:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self._heights_vbo)
        glBufferData(GL_ARRAY_BUFFER, n * 4, None, GL_STREAM_DRAW)
        self._heights_capacity = n

    def _render(self, heights: np.ndarray) -> None:
        """
        Disegna un frame: pulisce, disegna lo sfondo (se presente) e poi le barre
        con una sola draw call instanced.

        Args:
            heights (np.ndarray): altezze normalizzate [0,1], una per barra (float32).
        """
        glClear(GL_COLOR_BUFFER_BIT)

        num_bars = int(heights.shape[0])
        if num_bars < 1:
            return

        # --- Sfondo ---
        if self._background_texture:
            glUseProgram(self._bg_program)
            alpha = 1.0 if self._bg_alpha is None else float(self._bg_alpha)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._background_texture)
            glUniform1i(self._bg_uniforms["uTexture"], 0)
            glUniform1f(self._bg_uniforms["uAlpha"], alpha)
            glBindVertexArray(self._bg_vao)
            glDrawArrays(GL_TRIANGLES, 0, 6)

        # --- Barre ---
        if heights.dtype != np.float32 or not heights.flags["C_CONTIGUOUS"]:
            heights = np.ascontiguousarray(heights, dtype=np.float32)
        self._ensure_heights_capacity(num_bars)
        glBindBuffer(GL_ARRAY_BUFFER, self._heights_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, heights.nbytes, heights)

        glUseProgram(self._bars_program)
        glUniform1f(self._bars_uniforms["uNumBars"], float(num_bars))
        glUniform1f(self._bars_uniforms["uBarWidthFrac"], BAR_WIDTH_FRAC)
        glUniform1f(self._bars_uniforms["uGreenSplit"], GREEN_SPLIT)
        glUniform1f(self._bars_uniforms["uYellowSplit"], YELLOW_SPLIT)
        glBindVertexArray(self._bars_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, num_bars)
        glBindVertexArray(0)

    def _cleanup_gl(self) -> None:
        """
        Elimina tutte le risorse OpenGL (programmi, VAO, VBO, texture). Idempotente
        e tollerante agli errori, così può girare anche dopo un'inizializzazione
        parziale (es. shader falliti).
        """
        def _safe(fn):
            try:
                fn()
            except Exception:
                pass

        if self._background_texture:
            _safe(lambda: glDeleteTextures(1, [self._background_texture]))
            self._background_texture = None
        for vbo in (self._bars_quad_vbo, self._heights_vbo, self._bg_vbo):
            if vbo:
                _safe(lambda v=vbo: glDeleteBuffers(1, [v]))
        for vao in (self._bars_vao, self._bg_vao):
            if vao:
                _safe(lambda v=vao: glDeleteVertexArrays(1, [v]))
        for prog in (self._bars_program, self._bg_program):
            if prog:
                _safe(lambda p=prog: glDeleteProgram(p))

        self._bars_quad_vbo = self._heights_vbo = self._bg_vbo = None
        self._bars_vao = self._bg_vao = None
        self._bars_program = self._bg_program = None
        self._heights_capacity = 0

    def process_control_message(self, message):
        """
        Processa i messaggi di controllo dalla coda.

        Args:
            message (dict): Messaggio con 'type' e 'value'
        """
        message_type = message["type"]

        if message_type == "set_noise_threshold":
            self._noise_threshold = message["value"]

        elif message_type == 'set_frequency_bands':
            self._n_bands = message["value"]
            # Ricalcola bande, mappa bin→banda e buffer ampiezze
            self._setup_bands(message["value"])
            # Adegua la capacità del VBO delle altezze al nuovo numero di barre
            self._ensure_heights_capacity(len(self.frequency_bands))

        elif message_type == 'set_alpha':
            # L'alpha è una uniform dello shader di sfondo: basta memorizzarla.
            self._bg_alpha = message['value']

        elif message_type == 'set_image':
            self._bg_image = message['value']
            self._background_texture = self.load_texture()

        elif message_type in ('set_workers', 'set_auto_workers'):
            # Inerti: la multiprocessing è stata rimossa dal percorso FFT.
            pass

    def smooth_amplitudes(self, targets, delta_time: float) -> np.ndarray:
        """
        Ballistica attacco-rapido / rilascio-lento (stile VU/peak-meter),
        indipendente dal frame rate. La salita rapida cattura i transienti
        (bassa latenza percepita), la discesa morbida rende il falloff leggibile.

        Args:
            targets: livelli target dall'FFT (array in [0,1])
            delta_time: tempo trascorso dall'ultimo frame (s)

        Returns:
            np.ndarray: livelli interpolati correnti
        """
        delta_time = max(delta_time, 1e-6)
        targets = np.asarray(targets, dtype=np.float32)

        # Riallinea se è cambiato il numero di bande
        if targets.shape[0] != self.current_amplitudes.shape[0]:
            self.current_amplitudes = np.zeros(targets.shape[0], dtype=np.float32)

        attack = 1.0 - math.exp(-delta_time / ATTACK_TAU)
        release = 1.0 - math.exp(-delta_time / RELEASE_TAU)

        coeff = np.where(targets > self.current_amplitudes, attack, release)
        self.current_amplitudes = self.current_amplitudes + (targets - self.current_amplitudes) * coeff

        return self.current_amplitudes

    def run(self):
        """
        Main loop del thread OpenGL. Crea finestra/contesto OpenGL 3.3 core, inizializza
        gli oggetti GL (shader, VAO/VBO), esegue il loop di rendering e garantisce il
        cleanup delle risorse anche in caso di errore (try/finally). Un'eccezione non
        gestita viene loggata e notificata alla GUI via window_close_callback, così non
        resta un thread morto in silenzio.
        """
        window = None
        try:
            if not glfw.init():
                raise Exception("GLFW initialization failed")

            # retrieve the monitor object by name
            selected_monitor = None
            if self._monitor:
                monitors = glfw.get_monitors()
                monitor_idx = int(self._monitor.split(".")[0]) - 1  # Extract the index from the monitor name
                if 0 <= monitor_idx < len(monitors):
                    selected_monitor = monitors[monitor_idx]
            else:
                selected_monitor = glfw.get_primary_monitor()

            if not selected_monitor:
                raise Exception('Not actual monitor instance has been found')

            # Get the primary monitor's video mode
            video_mode = glfw.get_video_mode(selected_monitor)
            self.window_width = video_mode.size.width
            self.window_height = video_mode.size.height
            self.frame_rate = video_mode.refresh_rate

            print(f"Monitor refresh rate: {self.frame_rate}Hz (FFT ricalcolata ogni frame)")

            # Richiedi esplicitamente un contesto OpenGL 3.3 core (shader + instancing).
            glfw.default_window_hints()
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

            window = glfw.create_window(self.window_width, self.window_height, "Audio Visualizer", selected_monitor, None)
            if not window:
                raise Exception("GLFW window creation failed")

            # check if the stop event is set
            if self.stop_event.is_set():
                self.stop_event.clear()

            glfw.make_context_current(window)
            gl_version = glGetString(GL_VERSION)
            print(f"OpenGL version: {gl_version.decode() if gl_version else '?'}")

            # Enable VSync per sincronizzazione con il refresh del monitor
            glfw.swap_interval(1)

            # Viewport sulla dimensione reale del framebuffer (corretto anche su hi-DPI)
            fb_width, fb_height = glfw.get_framebuffer_size(window)
            glViewport(0, 0, fb_width, fb_height)

            # Enable blending
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glClearColor(0.0, 0.0, 0.0, 1.0)

            # Disable auto iconify
            glfw.set_window_attrib(window, glfw.AUTO_ICONIFY, glfw.FALSE)
            glfw.set_key_callback(window, self.key_callback)

            # Compila shader e crea VAO/VBO, poi carica la texture di sfondo (a contesto attivo)
            self.init_gl_objects()
            self._background_texture = self.load_texture()

            # Timing preciso
            last_frame_time = glfw.get_time()

            # -------------------------
            # MAIN LOOP
            # -------------------------
            while not glfw.window_should_close(window) and not self.stop_event.is_set():
                # Timing preciso usando GLFW
                current_frame_time = glfw.get_time()
                frame_delta = current_frame_time - last_frame_time
                last_frame_time = current_frame_time

                # Process messages from the control queue
                while not self._control_queue.empty():
                    message = self._control_queue.get(block=False)
                    self.process_control_message(message)

                # 1) Svuota TUTTA la coda audio nel ring buffer: tenendo solo gli ultimi
                #    N_FFT frame, analizziamo sempre l'audio più recente (niente backlog).
                while True:
                    try:
                        self.audio_accumulator.add_samples(self._audio_queue.get_nowait())
                    except queue.Empty:
                        break

                # 2) Calcola l'FFT a OGNI frame sulla finestra più fresca disponibile
                fft_window = self.audio_accumulator.get_window_for_fft()
                if fft_window.size > 0:
                    if GENERAL_LOG:
                        print('LOG: Start FFT computation')

                    t0 = time.perf_counter()
                    self.target_amplitudes = self.create_equalizer(
                        audio_data=fft_window,
                        frequency_bands=self.frequency_bands,
                        noise_threshold=self._noise_threshold,
                        channels=self._channels
                    )
                    if PERF_LOGS:
                        print(f"PERF LOG: FFT took {(time.perf_counter() - t0) * 1000:.3f} ms")

                # 3) Ballistica attacco/rilascio
                smoothed_amplitudes = self.smooth_amplitudes(self.target_amplitudes, frame_delta)

                # 4) Disegna il frame (sfondo + barre instanced) con i valori smoothed
                self._render(smoothed_amplitudes)

                glfw.swap_buffers(window)
                glfw.poll_events()

                if PERF_LOGS and frame_delta > 0:
                    fps = 1.0 / frame_delta
                    print(f"PERF LOG: FrameTime = {frame_delta * 1000:.3f}ms (FPS ~ {fps:.1f})")
        except Exception as exc:
            # Errore non gestito: notifica la GUI affinché ripristini lo stato
            # (equalizer_opengl_thread = None) invece di lasciare un thread morto.
            print(f"OpenGL thread error: {exc}")
            if self.window_close_callback:
                self.window_close_callback()
        finally:
            # Cleanup garantito: risorse GL, finestra e GLFW (sicuro anche su init parziale).
            self._cleanup_gl()
            if window is not None:
                glfw.destroy_window(window)
            glfw.terminate()

    def stop(self):
        """
        Ferma il thread impostando l'evento di stop.
        """
        self.stop_event.set()

    def key_callback(self, window, key, scancode, action, mods):
        """
        Callback per i tasti. ESC chiude la finestra.

        Args:
            window: Window GLFW
            key: Codice del tasto
            scancode: Scancode del tasto
            action: Azione (PRESS/RELEASE)
            mods: Modificatori
        """
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            self.stop()
            if self.window_close_callback:
                self.window_close_callback()

    def load_texture(self):
        """
        Carica la texture di sfondo da self._bg_image, eliminando prima l'eventuale
        texture precedente (evita leak di memoria GPU). L'alpha NON viene applicato
        qui: è gestito come uniform dallo shader di sfondo.

        Returns:
            int: ID della nuova texture OpenGL, oppure 0 se non c'è immagine.
        """
        # Elimina la texture precedente per evitare leak di VRAM
        if self._background_texture:
            try:
                glDeleteTextures(1, [self._background_texture])
            except Exception:
                pass
            self._background_texture = None

        if not self._bg_image:
            return 0

        im = self._bg_image.transpose(Transpose.FLIP_TOP_BOTTOM)
        im_data = im.convert('RGBA').tobytes()
        width, height = im.size

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, im_data)
        return texture

    # Il rendering è ora interamente in _render() (shader + instancing, core 3.3):
    # i vecchi draw_background/draw_bars (pipeline fixed-function) sono stati rimossi.
