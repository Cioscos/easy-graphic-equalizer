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
from thread.video_decode_thread import VideoDecodeThread
from thread.frame_time_stats import FrameTimeStats

RATE = 44100
N_FFT = 4096
MIN_FREQ = 20
MAX_FREQ = 20000

GENERAL_LOG = False
FFT_LOGS = False
PERF_LOGS = True

# --- Taratura fedeltà/latenza (regolabili a vista) ---
# Scala dB assoluta: l'altezza barra riflette il livello reale del segnale.
DB_FLOOR = -60.0        # dB: sotto questo livello la barra è a 0
DB_CEILING = 0.0        # dB: fondo scala (barra piena). Abbassare (es. -10) per riempire prima.
# Ballistica delle barre (costanti di tempo, frame-rate-indipendenti):
ATTACK_TAU = 0.012      # s: salita rapida → cattura i transienti (kick/snare) quasi all'istante
RELEASE_TAU = 0.220     # s: discesa morbida → falloff leggibile in stile peak-meter
# Compensazione spettrale opzionale (0 = off). +3.0 ≈ tilt "pink" (+3 dB/ottava).
TILT_DB_PER_OCT = 0.0

# --- Colore barre (Blocco 1). Default usati solo nelle modalità tinta/gradiente. ---
DEFAULT_COLOR_A = (0.231, 0.420, 1.0)   # blu  (#3b6bff): tinta / base gradiente
DEFAULT_COLOR_B = (1.0, 0.231, 0.816)   # magenta (#ff3bd0): cima gradiente

# --- Effetti reattivi (Blocco 2). Default OFF: nessun cambiamento visivo. ---
DEFAULT_PEAKCAP_COLOR = (1.0, 1.0, 1.0)   # bianco
PEAKCAP_HOLD = 0.5        # s di hold prima della caduta
PEAKCAP_FALL = 0.8        # frazione di altezza al secondo
CAP_THICKNESS = 0.012     # spessore del trattino (frazione di altezza finestra)
BEAT_INTENSITY = 0.08         # zoom massimo dello sfondo (8%)
BEAT_SENSITIVITY = 1.5        # energia > 1.5x media -> beat
BEAT_EMA_ALPHA = 0.08         # reattività della media mobile dell'energia
BEAT_REFRACTORY = 0.15        # s minimi tra due beat
BEAT_DECAY_TAU = 0.18         # s, decadimento dell'inviluppo di pulsazione
BASS_CUTOFF_HZ = 150.0        # energia bassi: potenza FFT sotto questa frequenza
BLOOM_INTENSITY = 1.0     # forza del composito additivo
BLOOM_BLUR_PASSES = 1     # iterazioni di blur (H+V) per il bloom

# --- Aspetto delle barre (ex magic number) ---
BAR_WIDTH_FRAC = 0.8    # frazione dello slot occupata dalla barra (il resto è spazio)
GREEN_SPLIT = 0.5       # sotto questa frazione d'altezza finestra → verde
YELLOW_SPLIT = 0.8      # tra GREEN_SPLIT e questa → giallo; oltre → rosso

# --- Antialiasing ---
# Campioni MSAA richiesti sul framebuffer di default (smussa i bordi di tutte le
# modalità). Fissato alla creazione della finestra: non modificabile a caldo.
MSAA_SAMPLES = 4

# --- Modalità di visualizzazione (Blocco 3) ---
MODE_BARS = 0          # barre verticali (attuale, default)
MODE_RADIAL = 1        # barre disposte in cerchio
MODE_OSCILLOSCOPE = 2  # forma d'onda nel tempo
MODE_LINE = 3          # spettro a linea / area riempita
OSC_GAIN = 1.4         # guadagno display dell'oscilloscopio (campione → frazione mezza-altezza)
OSC_MAX_AMP = 0.45     # ampiezza max attorno al centro (clamp, frazione mezza-altezza)
OSC_DISPLAY_POINTS = 512  # campioni mostrati dopo il trigger (finestra breve = onda leggibile)
OSC_SMOOTH_MAX_TAU = 0.12  # s: a smoothing=1 l'onda si assesta in ~questo tempo (0 = nessuno)
DEFAULT_INNER_RADIUS = 0.18      # raggio del foro centrale (radiale), frazione del raggio max
DEFAULT_LINE_THICKNESS = 0.012   # spessore nastro (osc/linea), frazione altezza finestra

# --- Video di sfondo (solo OpenGL) ---
# Massimo numero di frame video "recuperabili" in un singolo tick di rendering:
# evita che, dopo uno stallo (resize, breakpoint), il video parta in fast-forward.
MAX_CATCHUP_FRAMES = 4


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


def build_band_strip(xs: np.ndarray, ybot: np.ndarray, ytop: np.ndarray) -> np.ndarray:
    """
    Triangle strip che riempie la banda tra le curve ybot..ytop alle ascisse xs. Usato
    dall'Area (ybot=0) e dall'oscilloscopio specchiato (banda simmetrica attorno al centro).
    Restituisce (2N, 2) float32 (x, y) per GL_TRIANGLE_STRIP. Funzione pura.
    """
    n = int(xs.shape[0])
    if n < 2:
        return np.zeros((0, 2), dtype=np.float32)
    xsf = xs.astype(np.float32)
    verts = np.empty((2 * n, 2), dtype=np.float32)
    verts[0::2, 0] = xsf;  verts[0::2, 1] = np.clip(ybot, 0.0, 1.0)
    verts[1::2, 0] = xsf;  verts[1::2, 1] = np.clip(ytop, 0.0, 1.0)
    return verts


def build_line_ribbon(xs: np.ndarray, ys: np.ndarray,
                      half_thickness: float, aspect: float) -> np.ndarray:
    """
    Nastro (triangle strip) di spessore COSTANTE in pixel lungo la polilinea (xs, ys) in
    [0,1]^2, con offset perpendicolare alla direzione della curva e correzione dell'aspect
    ratio: la larghezza resta costante anche sui tratti ripidi (niente "spike" verticali
    sugli zero-crossing). Usato dallo spettro a linea e dall'oscilloscopio mono.
    Restituisce (2N, 2) float32 (x, y) per GL_TRIANGLE_STRIP. Funzione pura.
    """
    n = int(xs.shape[0])
    if n < 2:
        return np.zeros((0, 2), dtype=np.float32)
    xsf = xs.astype(np.float32)
    ysf = ys.astype(np.float32)
    asp = max(float(aspect), 1e-6)
    X = xsf * asp
    Y = ysf
    # Tangente per-vertice (differenze centrali; estremi unilaterali) in spazio aspect-corretto
    tx = np.empty(n, dtype=np.float32)
    ty = np.empty(n, dtype=np.float32)
    tx[1:-1] = X[2:] - X[:-2];  ty[1:-1] = Y[2:] - Y[:-2]
    tx[0] = X[1] - X[0];        ty[0] = Y[1] - Y[0]
    tx[-1] = X[-1] - X[-2];     ty[-1] = Y[-1] - Y[-2]
    L = np.sqrt(tx * tx + ty * ty)
    L[L < 1e-9] = 1e-9
    nax = -ty / L            # normale (spazio aspect-corretto)
    nay = tx / L
    ox = (half_thickness * nax / asp).astype(np.float32)   # ritorno a x in [0,1]
    oy = (half_thickness * nay).astype(np.float32)
    verts = np.empty((2 * n, 2), dtype=np.float32)
    verts[0::2, 0] = np.clip(xsf - ox, 0.0, 1.0);  verts[0::2, 1] = np.clip(ysf - oy, 0.0, 1.0)
    verts[1::2, 0] = np.clip(xsf + ox, 0.0, 1.0);  verts[1::2, 1] = np.clip(ysf + oy, 0.0, 1.0)
    return verts


# --- Shader GLSL (OpenGL 3.3 core) ---
# Barre: rendering instanced. Il quad unitario [0,1]² viene espanso per ogni barra
# usando gl_InstanceID e l'altezza per-istanza; nessuna geometria generata dalla CPU.
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

# Lo split verde/giallo/rosso è calcolato per-fragment sulla GPU (replica esatta dei
# tagli usati dalla pipeline fissa precedente, ma netto al pixel).
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
uniform float uMirror;        // 0 = dal basso (arrotonda solo la cima), 1 = specchiato (entrambe)

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
        // Cappuccio semicircolare, corretto per aspect ratio. Dal basso si arrotonda
        // solo la cima (la base poggia sul bordo schermo); specchiato si arrotondano
        // entrambe le estremità (effetto capsula), prendendo la più vicina.
        float wpx = (1.0 / uNumBars) * uBarWidthFrac * uViewportPx.x;  // larghezza barra (px)
        float hpx = max(vBarH * uViewportPx.y, 1.0);                   // altezza barra (px)
        float r = 0.5 * wpx;
        float dyTop = (1.0 - vBarY) * hpx;        // px sotto la cima (0 = cima)
        float dyBot = vBarY * hpx;                // px sopra la base (0 = base)
        float d = (uMirror > 0.5) ? min(dyTop, dyBot) : dyTop;  // estremità arrotondata più vicina
        if (d < r) {
            float dx = (vBarX - 0.5) * wpx;       // offset orizzontale dal centro (px)
            float dyc = r - d;                    // offset verticale dal centro del cappuccio
            float dist = sqrt(dx * dx + dyc * dyc);
            float aa = max(fwidth(dist), 1e-3);
            a *= 1.0 - smoothstep(r - aa, r + aa, dist);
        }
    }
    FragColor = vec4(col, a);
}
"""

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

# Sfondo: quad fullscreen testurizzato. L'alpha è una uniform → nessuna ricreazione
# della texture quando cambia (niente più re-upload né leak).
# uFlipV capovolge la coordinata v a costo zero: le immagini sono già flippate in CPU
# da load_texture (uFlipV=0), mentre i frame video hanno la riga 0 in alto e vanno
# capovolti qui (uFlipV=1) invece di copiarli flippati a ogni frame.
_BG_VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;        // NDC [-1,1]
layout(location = 1) in vec2 aTexCoord;
uniform float uFlipV;                     // 0 = immagine (già flippata), 1 = frame video
uniform float uBgZoom;                    // zoom additivo sul beat (0 = nessuno)
out vec2 vTexCoord;
void main() {
    vTexCoord = vec2(aTexCoord.x, mix(aTexCoord.y, 1.0 - aTexCoord.y, uFlipV));
    gl_Position = vec4(aPos * (1.0 + uBgZoom), 0.0, 1.0);
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

# --- Shader "strip": curve continue (oscilloscopio, linea/area) via triangle strip ---
# Vertici (x, y, mag): x,y in [0,1] (posizione); mag in [0,1] (magnitudine per la
# colorazione). Replica le 4 modalità colore del Blocco 1: Classico per mag, Gradiente
# verticale per y, Spettro per x, Tinta unica.
_STRIP_VERTEX_SRC = """
#version 330 core
layout(location = 0) in vec2 aPos;    // (x, y) in [0,1]
out float vX;
out float vY;
void main() {
    vX = aPos.x;
    vY = aPos.y;
    gl_Position = vec4(aPos.x * 2.0 - 1.0, aPos.y * 2.0 - 1.0, 0.0, 1.0);
}
"""

_STRIP_FRAGMENT_SRC = """
#version 330 core
in float vX;
in float vY;
out vec4 FragColor;
uniform int uColorMode;
uniform vec3 uColorA;
uniform vec3 uColorB;
uniform float uGreenSplit;
uniform float uYellowSplit;
uniform float uBarsAlpha;
uniform float uVerticalFade;  // 1 = alpha sfuma con vY (area morbida), 0 = piena

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec3 col;
    if (uColorMode == 0) {
        if (vY < uGreenSplit)              col = vec3(0.0, 1.0, 0.0);
        else if (vY < uYellowSplit)        col = vec3(1.0, 1.0, 0.0);
        else                               col = vec3(1.0, 0.0, 0.0);
    } else if (uColorMode == 1) {
        col = uColorA;
    } else if (uColorMode == 2) {
        col = mix(uColorA, uColorB, vY);
    } else {
        col = hsv2rgb(vec3(vX * 0.83, 0.9, 1.0));
    }
    float a = uBarsAlpha * mix(1.0, vY, uVerticalFade);
    FragColor = vec4(col, a);
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
                 monitor=None,
                 bg_image: Optional[Image.Image] = None,
                 bg_alpha: Optional[float] = None,
                 bars_alpha: float = 1.0,
                 bg_video: Optional[str] = None,
                 db_floor: float = DB_FLOOR,
                 db_ceiling: float = DB_CEILING,
                 attack_tau: float = ATTACK_TAU,
                 release_tau: float = RELEASE_TAU,
                 tilt_db_per_oct: float = TILT_DB_PER_OCT,
                 color_mode: int = 0,
                 color_a: tuple = DEFAULT_COLOR_A,
                 color_b: tuple = DEFAULT_COLOR_B,
                 green_split: float = GREEN_SPLIT,
                 yellow_split: float = YELLOW_SPLIT,
                 bar_width: float = BAR_WIDTH_FRAC,
                 rounded: bool = False,
                 mirror: bool = False,
                 band_order_symmetric: bool = False,
                 peakcap_enabled: bool = False,
                 peakcap_color: tuple = DEFAULT_PEAKCAP_COLOR,
                 peakcap_hold: float = PEAKCAP_HOLD,
                 peakcap_fall: float = PEAKCAP_FALL,
                 beat_enabled: bool = False,
                 beat_intensity: float = BEAT_INTENSITY,
                 beat_sensitivity: float = BEAT_SENSITIVITY,
                 bloom_enabled: bool = False,
                 bloom_intensity: float = BLOOM_INTENSITY,
                 mode: int = MODE_BARS,
                 inner_radius: float = DEFAULT_INNER_RADIUS,
                 line_thickness: float = DEFAULT_LINE_THICKNESS,
                 fill: bool = True,
                 osc_mirror: bool = False,
                 osc_smoothing: float = 0.5,
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
        self._bars_alpha = bars_alpha
        self.window_close_callback = window_close_callback

        # --- Taratura DSP regolabile a runtime (default = costanti di modulo). ---
        # I metodi leggono questi attributi, non le costanti globali, così la GUI
        # può modificarli via process_control_message (vedi set_db_floor & co.).
        self._db_floor = db_floor
        self._db_ceiling = db_ceiling
        self._attack_tau = attack_tau
        self._release_tau = release_tau
        self._tilt_db_per_oct = tilt_db_per_oct

        # --- Stato colore barre (Blocco 1). Default = look classico attuale. ---
        self._color_mode = int(color_mode)
        self._color_a = tuple(color_a)
        self._color_b = tuple(color_b)
        self._green_split = float(green_split)
        self._yellow_split = float(yellow_split)

        # --- Stato forma/disposizione barre (Blocco 1). Default = look attuale. ---
        self._bar_width_frac = float(bar_width)
        self._rounded = bool(rounded)
        self._mirror = bool(mirror)
        self._band_order_symmetric = bool(band_order_symmetric)
        # Dimensione framebuffer (px), serve al cap arrotondato; reale in run().
        self._fb_width = 1.0
        self._fb_height = 1.0

        # --- Peak-cap (Blocco 2) ---
        self._peakcap_enabled = bool(peakcap_enabled)
        self._peakcap_color = tuple(peakcap_color)
        self._peakcap_hold = float(peakcap_hold)
        self._peakcap_fall = float(peakcap_fall)
        # self._n_bands è già impostato; self.frequency_bands viene creato solo più
        # tardi da _setup_bands. _update_peaks ridimensiona comunque se cambia il
        # numero di barre visualizzate (es. ordine simmetrico).
        self._peaks = np.zeros(self._n_bands, dtype=np.float32)
        self._peak_age = np.zeros(self._n_bands, dtype=np.float32)

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

        # --- Bloom (Blocco 2) ---
        self._bloom_enabled = bool(bloom_enabled)
        self._bloom_intensity = float(bloom_intensity)

        # --- Modalità di visualizzazione (Blocco 3). Default = Barre (look attuale). ---
        self._mode = int(mode)
        self._inner_radius = float(inner_radius)
        self._line_thickness = float(line_thickness)
        self._fill = bool(fill)
        self._osc_mirror = bool(osc_mirror)
        self._osc_smoothing = float(osc_smoothing)   # 0 = nessuno, 1 = max (vedi OSC_SMOOTH_MAX_TAU)
        self._osc_prev = None         # ultima onda visualizzata (smoothing temporale)
        self._draw_count = 0          # vertici (strip) o istanze (barre/radiale) da disegnare
        self._bar_heights = np.zeros(self._n_bands, dtype=np.float32)  # altezze disposte (peak-cap)

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
        self._caps_program = None
        self._caps_vao = None
        self._peaks_vbo = None
        self._peaks_capacity = 0
        self._caps_uniforms = {}
        self._blur_program = None
        self._bloom_composite_program = None
        self._blur_uniforms = {}
        self._bloom_uniforms = {}
        self._bloom_fbos = []      # 2 FBO ping-pong
        self._bloom_texs = []      # 2 texture half-res (GL_RGB16F)
        self._bloom_size = None    # (w, h) half-res; None = non ancora creati
        self._strip_program = None
        self._strip_vao = None
        self._strip_vbo = None
        self._strip_capacity = 0      # n. di vertici allocati nel VBO strip
        self._strip_uniforms = {}

        self.stop_event = threading.Event()
        self._background_texture = None

        # --- Stato video di sfondo (solo OpenGL): vedi _start_video/_advance_video ---
        self._bg_video = bg_video           # path video iniziale (opzionale)
        self._video_thread = None           # VideoDecodeThread produttore di frame (CPU)
        self._video_queue = None            # coda corta dei frame decodificati
        self._video_texture = None          # texture GL dedicata al video (separata dall'immagine)
        self._video_active = False          # True quando il video sostituisce l'immagine
        self._video_tex_size = None         # (w, h) texture allocata; None = non ancora creata
        self._video_frame_interval = 1.0 / 30.0  # 1/fps (aggiornato dai metadati del video)
        self._video_time_acc = 0.0          # accumulatore di tempo per il pacing del video
        self._video_pbos = None             # 2 PBO (ping-pong) per upload async; None = non creati
        self._video_pbo_index = 0           # indice del PBO corrente nel ring
        self._video_pbo_size = 0            # byte allocati per ciascun PBO; 0 = non allocati

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
        if self._tilt_db_per_oct != 0.0:
            centers = np.array([
                math.sqrt(max(low, 1e-9) * max(high, 1e-9))
                for low, high in self.frequency_bands
            ])
            self._tilt_db = self._tilt_db_per_oct * np.log2(np.maximum(centers, 1e-9) / 1000.0)
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

        # Energia bassi (per la beat detection): somma potenza sotto il cutoff.
        self._bass_energy = float(np.sum(power[self._bass_mask]))

        # Somma dell'energia per banda in O(bin) (niente loop di maschere)
        band_power = np.bincount(
            self._bin_band[self._valid_bins],
            weights=power[self._valid_bins],
            minlength=len(frequency_bands)
        )

        # Conversione in dB (+ tilt percettivo opzionale)
        band_db = 10.0 * np.log10(band_power + 1e-12) + self._tilt_db

        # Mappa [db_floor, db_ceiling] → [0, 1] con riferimento FISSO (no per-frame max).
        # max(..., 1e-6) protegge da una divisione degenere se i due estremi
        # venissero impostati troppo vicini/invertiti via control message.
        denom = max(self._db_ceiling - self._db_floor, 1e-6)
        level = (band_db - self._db_floor) / denom
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
            for name in ("uNumBars", "uBarWidthFrac", "uGreenSplit", "uYellowSplit",
                         "uBarsAlpha", "uMirror", "uColorMode", "uColorA", "uColorB",
                         "uRounded", "uViewportPx", "uRadial", "uInnerRadius", "uAspect")
        }
        self._bg_uniforms = {
            name: glGetUniformLocation(self._bg_program, name)
            for name in ("uTexture", "uAlpha", "uFlipV", "uBgZoom")
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

        # --- Programma "strip" (oscilloscopio + linea/area): VBO (x,y,mag) interleaved ---
        self._strip_program = compileProgram(
            compileShader(_STRIP_VERTEX_SRC, GL_VERTEX_SHADER),
            compileShader(_STRIP_FRAGMENT_SRC, GL_FRAGMENT_SHADER),
        )
        self._strip_uniforms = {
            name: glGetUniformLocation(self._strip_program, name)
            for name in ("uColorMode", "uColorA", "uColorB",
                         "uGreenSplit", "uYellowSplit", "uBarsAlpha", "uVerticalFade")
        }
        self._strip_vao = glGenVertexArrays(1)
        glBindVertexArray(self._strip_vao)
        self._strip_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._strip_vbo)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * 4, ctypes.c_void_p(0))
        self._strip_capacity = 0
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

    def _ensure_peaks_capacity(self, n: int) -> None:
        """Garantisce che il VBO dei picchi contenga almeno n float."""
        if n <= self._peaks_capacity:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self._peaks_vbo)
        glBufferData(GL_ARRAY_BUFFER, n * 4, None, GL_STREAM_DRAW)
        self._peaks_capacity = n

    def _ensure_strip_capacity(self, n_verts: int) -> None:
        """Garantisce che il VBO strip contenga almeno n_verts vertici (2 float l'uno: x, y)."""
        if n_verts <= self._strip_capacity:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self._strip_vbo)
        glBufferData(GL_ARRAY_BUFFER, n_verts * 2 * 4, None, GL_STREAM_DRAW)
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
        # Area (Linea/Area con riempimento): alpha che sfuma verso la base → niente "muro".
        fade = 1.0 if (self._mode == MODE_LINE and self._fill) else 0.0
        glUniform1f(self._strip_uniforms["uVerticalFade"], fade)
        glBindVertexArray(self._strip_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, int(self._draw_count))
        glBindVertexArray(0)

    def _display_heights(self, heights: np.ndarray) -> np.ndarray:
        """
        Applica la disposizione delle barre prima dell'upload: identità in ordine
        standard, riordino simmetrico (bassi al centro, raddoppia) altrimenti.
        """
        if self._band_order_symmetric:
            return arrange_symmetric(heights)
        return heights

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

    def _render_bloom(self) -> None:
        """Renderizza la modalità corrente in un FBO half-res, la sfoca (H+V) e lascia il
        risultato in _bloom_texs[0], pronto per il composito additivo."""
        if not self._ensure_bloom_fbos():
            return
        w, h = self._bloom_size
        # Pass A: modalità corrente nel FBO 0
        glBindFramebuffer(GL_FRAMEBUFFER, self._bloom_fbos[0])
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT)
        self._draw_current_mode((w, h))
        # Pass B/C: blur ping-pong (orizzontale 0->1, verticale 1->0), N iterazioni
        glDisable(GL_BLEND)
        glUseProgram(self._blur_program)
        glBindVertexArray(self._bg_vao)
        for _ in range(BLOOM_BLUR_PASSES):
            for dir_vec, src, dst in (((1.0 / w, 0.0), 0, 1), ((0.0, 1.0 / h), 1, 0)):
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

    def _prepare_mode_geometry(self, heights: np.ndarray, waveform, dt: float) -> bool:
        """
        Costruisce e carica la geometria della modalità corrente; imposta self._draw_count
        e (per le modalità a banda) self._bar_heights. Ritorna False se non c'è nulla da
        disegnare (audio/bande vuoti).
        """
        aspect = float(self._fb_width) / max(float(self._fb_height), 1.0)

        if self._mode == MODE_OSCILLOSCOPE:
            if waveform is None or waveform.size == 0:
                self._draw_count = 0
                return False
            # Finestra breve sui campioni più RECENTI (bassa latenza), allineata allo
            # zero-crossing: poche oscillazioni, onda leggibile e reattiva.
            recent = np.ascontiguousarray(waveform[-2 * OSC_DISPLAY_POINTS:], dtype=np.float32)
            wf = trigger_align(recent)[:OSC_DISPLAY_POINTS]
            if wf.shape[0] < 2:
                self._draw_count = 0
                return False
            xs = np.linspace(0.0, 1.0, wf.shape[0], dtype=np.float32)
            s = np.clip(wf * OSC_GAIN, -OSC_MAX_AMP, OSC_MAX_AMP).astype(np.float32)
            # Smoothing temporale (EMA frame-rate-independent): onda meno nervosa.
            if (self._osc_smoothing > 0.0 and self._osc_prev is not None
                    and self._osc_prev.shape == s.shape):
                tau = self._osc_smoothing * OSC_SMOOTH_MAX_TAU
                a = 1.0 - math.exp(-max(dt, 1e-6) / tau) if tau > 1e-6 else 1.0
                s = (self._osc_prev + (s - self._osc_prev) * a).astype(np.float32)
            self._osc_prev = s
            if self._osc_mirror:
                verts = build_band_strip(xs, 0.5 - np.abs(s), 0.5 + np.abs(s))
            else:
                verts = build_line_ribbon(xs, 0.5 + s, 0.5 * self._line_thickness, aspect)
            self._upload_strip(verts)
            return self._draw_count > 0

        # Modalità basate sulle bande: applica la disposizione (es. simmetrica).
        heights = self._display_heights(heights)
        n = int(heights.shape[0])
        if n < 1:
            self._draw_count = 0
            return False
        self._bar_heights = heights  # per il peak-cap (solo Barre)
        if self._mode == MODE_LINE:
            if n < 2:
                self._draw_count = 0
                return False
            h = np.clip(np.ascontiguousarray(heights, dtype=np.float32), 0.0, 1.0)
            xs = np.linspace(0.0, 1.0, n, dtype=np.float32)
            if self._fill:
                verts = build_band_strip(xs, np.zeros(n, dtype=np.float32), h)
            else:
                verts = build_line_ribbon(xs, h, 0.5 * self._line_thickness, aspect)
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
        if not self._prepare_mode_geometry(heights, waveform, dt):
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
        if self._video_texture:
            _safe(lambda: glDeleteTextures(1, [self._video_texture]))
            self._video_texture = None
        if self._video_pbos:
            _safe(lambda: glDeleteBuffers(len(self._video_pbos), self._video_pbos))
            self._video_pbos = None
            self._video_pbo_size = 0
        for tex in self._bloom_texs:
            _safe(lambda t=tex: glDeleteTextures(1, [t]))
        for fbo in self._bloom_fbos:
            _safe(lambda f=fbo: glDeleteFramebuffers(1, [f]))
        self._bloom_texs = []
        self._bloom_fbos = []
        self._bloom_size = None
        for vbo in (self._bars_quad_vbo, self._heights_vbo, self._bg_vbo, self._peaks_vbo, self._strip_vbo):
            if vbo:
                _safe(lambda v=vbo: glDeleteBuffers(1, [v]))
        for vao in (self._bars_vao, self._bg_vao, self._caps_vao, self._strip_vao):
            if vao:
                _safe(lambda v=vao: glDeleteVertexArrays(1, [v]))
        for prog in (self._bars_program, self._bg_program, self._caps_program,
                     self._blur_program, self._bloom_composite_program, self._strip_program):
            if prog:
                _safe(lambda p=prog: glDeleteProgram(p))

        self._bars_quad_vbo = self._heights_vbo = self._bg_vbo = None
        self._bars_vao = self._bg_vao = None
        self._bars_program = self._bg_program = None
        self._heights_capacity = 0
        self._peaks_vbo = self._caps_vao = self._caps_program = None
        self._peaks_capacity = 0
        self._blur_program = self._bloom_composite_program = None
        self._strip_vbo = self._strip_vao = self._strip_program = None
        self._strip_capacity = 0

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

        elif message_type == 'set_bars_alpha':
            # L'opacità delle barre è una uniform dello shader delle barre.
            self._bars_alpha = message['value']

        elif message_type == 'set_db_floor':
            # dB sotto cui la barra è a 0. Letto ogni frame in create_equalizer.
            self._db_floor = message['value']

        elif message_type == 'set_db_ceiling':
            # dB di fondo scala (barra piena).
            self._db_ceiling = message['value']

        elif message_type == 'set_attack_tau':
            # Costante di tempo di salita (s) della ballistica delle barre.
            self._attack_tau = message['value']

        elif message_type == 'set_release_tau':
            # Costante di tempo di discesa (s) della ballistica delle barre.
            self._release_tau = message['value']

        elif message_type == 'set_tilt_db_per_oct':
            # Il tilt è un array dB per-banda: va ricalcolato. _compute_band_bins
            # rifà tilt e mappa bin→banda (costo trascurabile) senza azzerare i
            # buffer ampiezze (a differenza di _setup_bands), evitando glitch.
            self._tilt_db_per_oct = message['value']
            self._compute_band_bins()

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

        elif message_type == 'set_peakcap_enabled':
            self._peakcap_enabled = bool(message['value'])

        elif message_type == 'set_peakcap_color':
            self._peakcap_color = tuple(message['value'])

        elif message_type == 'set_peakcap_hold':
            self._peakcap_hold = float(message['value'])

        elif message_type == 'set_peakcap_fall':
            self._peakcap_fall = float(message['value'])

        elif message_type == 'set_beat_enabled':
            self._beat_enabled = bool(message['value'])

        elif message_type == 'set_beat_intensity':
            self._beat_intensity = float(message['value'])

        elif message_type == 'set_beat_sensitivity':
            self._beat_sensitivity = float(message['value'])

        elif message_type == 'set_bloom_enabled':
            self._bloom_enabled = bool(message['value'])

        elif message_type == 'set_bloom_intensity':
            self._bloom_intensity = float(message['value'])

        elif message_type == 'set_viz_mode':
            self._mode = int(message['value'])
            self._osc_prev = None   # azzera lo storico smoothing al cambio modalità

        elif message_type == 'set_osc_smoothing':
            self._osc_smoothing = float(message['value'])

        elif message_type == 'set_inner_radius':
            self._inner_radius = float(message['value'])

        elif message_type == 'set_line_thickness':
            self._line_thickness = float(message['value'])

        elif message_type == 'set_fill':
            self._fill = bool(message['value'])

        elif message_type == 'set_osc_mirror':
            self._osc_mirror = bool(message['value'])

        elif message_type == 'set_image':
            # L'immagine sostituisce il video: ferma l'eventuale riproduzione.
            self._stop_video()
            self._bg_image = message['value']
            self._background_texture = self.load_texture()

        elif message_type == 'set_video':
            # value = path del file video (stringa). Il decoder vive sul suo thread.
            self._start_video(message['value'])

        elif message_type == 'clear_video':
            # Torna all'immagine di sfondo (se presente).
            self._stop_video()
            self._background_texture = self.load_texture()

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

        attack = 1.0 - math.exp(-delta_time / self._attack_tau)
        release = 1.0 - math.exp(-delta_time / self._release_tau)

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
            self._fb_width, self._fb_height = fb_width, fb_height

            # Enable blending
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glClearColor(0.0, 0.0, 0.0, 1.0)

            # RGB a 3 byte/pixel con larghezza non multipla di 4: forziamo una volta
            # per tutte l'allineamento di unpack a 1 byte (lo usa l'upload video via
            # PBO; sicuro anche per le texture RGBA, sempre 4-allineate).
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

            # Abilita l'MSAA (no-op se il driver non ha concesso il framebuffer multisample)
            glEnable(GL_MULTISAMPLE)
            if GENERAL_LOG:
                print(f"[GL] MSAA samples concessi: {glGetIntegerv(GL_SAMPLES)}")

            # Disable auto iconify
            glfw.set_window_attrib(window, glfw.AUTO_ICONIFY, glfw.FALSE)
            glfw.set_key_callback(window, self.key_callback)

            # Compila shader e crea VAO/VBO, poi carica la texture di sfondo (a contesto attivo)
            self.init_gl_objects()
            self._background_texture = self.load_texture()

            # Se è stato richiesto un video di sfondo all'avvio, avvialo subito.
            if self._bg_video:
                self._start_video(self._bg_video)

            # Timing preciso
            last_frame_time = glfw.get_time()

            # Statistiche aggregate del frame time (solo sotto PERF_LOGS): budget
            # = 1/refresh del monitor → "oltre-budget" conta i frame che mancano il vblank.
            frame_stats = FrameTimeStats(report_every=120, budget_s=1.0 / max(self.frame_rate, 1)) if PERF_LOGS else None

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
                    if FFT_LOGS:
                        print(f"FFT LOG: FFT took {(time.perf_counter() - t0) * 1000:.3f} ms")

                # Waveform mono per l'oscilloscopio (solo se serve): media dei canali.
                waveform = None
                if self._mode == MODE_OSCILLOSCOPE and fft_window.size > 0:
                    waveform = fft_window.mean(axis=1).astype(np.float32)

                # Beat detection + inviluppo di pulsazione (Blocco 2)
                if self._beat_enabled:
                    if self._detect_beat(self._bass_energy, frame_delta):
                        self._beat_pulse = 1.0
                    # decadimento esponenziale frame-rate-independent
                    self._beat_pulse *= math.exp(-frame_delta / BEAT_DECAY_TAU)
                else:
                    self._beat_pulse = 0.0

                # 3) Ballistica attacco/rilascio
                smoothed_amplitudes = self.smooth_amplitudes(self.target_amplitudes, frame_delta)

                # 3.5) Avanza il video di sfondo (se attivo) e aggiorna la sua texture.
                #      Pacing disaccoppiato dal refresh: vedi _advance_video.
                if self._video_active:
                    self._advance_video(frame_delta)

                # 4) Disegna il frame (modalità corrente) con i valori smoothed
                self._render(smoothed_amplitudes, waveform, frame_delta)

                glfw.swap_buffers(window)
                glfw.poll_events()

                if frame_stats is not None and frame_delta > 0:
                    report = frame_stats.add(frame_delta)
                    if report:
                        print(report)
        except Exception as exc:
            # Errore non gestito: notifica la GUI affinché ripristini lo stato
            # (equalizer_opengl_thread = None) invece di lasciare un thread morto.
            print(f"OpenGL thread error: {exc}")
            if self.window_close_callback:
                self.window_close_callback()
        finally:
            # Ferma il decoder video e il subprocess ffmpeg prima di liberare il GL.
            self._stop_video()
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

    # ------------------------------------------------------------------
    # Video di sfondo (solo OpenGL). La DECODIFICA (CPU) gira su un thread
    # dedicato (VideoDecodeThread); QUI, sul thread OpenGL, avviene solo
    # l'UPLOAD GPU della texture, dove il contesto GL è valido.
    # ------------------------------------------------------------------
    def _start_video(self, path: str) -> None:
        """
        Avvia (o riavvia) la riproduzione del video di sfondo: ferma un eventuale
        video in corso, crea la coda dei frame e il thread decoder, e attiva la
        modalità video. La texture video è allocata pigramente al primo frame
        (in _upload_video_frame). L'immagine corrente resta visibile finché il
        primo frame non è pronto. Da chiamare sul thread OpenGL.
        """
        self._stop_video()
        if not path:
            return
        self._video_queue = queue.Queue(maxsize=2)
        self._video_thread = VideoDecodeThread(path, self._video_queue)
        self._video_thread.start()
        self._video_tex_size = None
        self._video_time_acc = 0.0
        self._video_frame_interval = 1.0 / 30.0
        self._video_active = True

    def _stop_video(self) -> None:
        """
        Ferma il thread decoder (se attivo), svuota la coda dei frame e libera la
        texture video. Idempotente e centralizzato: usato da set_video (cambio
        video), clear_video, set_image e dal finally di run(). Richiede un contesto
        GL valido per glDeleteTextures (tutti i chiamanti girano sul thread GL).
        """
        if self._video_thread is not None:
            self._video_thread.stop()
            self._video_thread.join(timeout=1.0)
            self._video_thread = None
        self._video_queue = None
        self._video_active = False
        self._video_tex_size = None
        self._video_time_acc = 0.0
        if self._video_texture:
            try:
                glDeleteTextures(1, [self._video_texture])
            except Exception:
                pass
            self._video_texture = None

    def _advance_video(self, frame_delta: float) -> None:
        """
        Fa avanzare il video in modo indipendente dal refresh del monitor: accumula
        il tempo trascorso e consuma dalla coda ESATTAMENTE il numero di frame che il
        tempo concede (uno per intervallo-frame del video), mostrando l'ultimo dei
        frame consumati. Così il decoder — che si blocca sul put a coda piena — va in
        back-pressure alla velocità REALE del video. NON si drena tutta la coda (lo
        facevamo come per l'audio, ma il decoder produce più veloce del realtime →
        scartando i frame pronti il video accelerava). Se la coda è vuota mantiene il
        frame già visualizzato. Il pacing è disaccoppiato dall'FFT/dalle barre.

        Args:
            frame_delta (float): secondi trascorsi dall'ultimo frame di rendering.
        """
        if not self._video_active or self._video_queue is None:
            return

        # fps letto live dal thread decoder (default finché i metadati non sono pronti).
        if self._video_thread is not None and self._video_thread.fps > 0:
            self._video_frame_interval = 1.0 / self._video_thread.fps

        self._video_time_acc += frame_delta
        if self._video_time_acc < self._video_frame_interval:
            return  # non è ancora ora del prossimo frame: riusa la texture corrente

        # Numero di frame "dovuti" dato il tempo accumulato; clamp anti-fast-forward
        # dopo uno stallo abnorme (resize/breakpoint), altrimenti sottrai con precisione.
        steps = int(self._video_time_acc / self._video_frame_interval)
        if steps > MAX_CATCHUP_FRAMES:
            steps = MAX_CATCHUP_FRAMES
            self._video_time_acc = 0.0
        else:
            self._video_time_acc -= steps * self._video_frame_interval

        # Consuma ESATTAMENTE `steps` frame e mostra l'ultimo: a refresh > fps si avanza
        # di 1 frame per intervallo (gli altri tick riusano la texture); a refresh < fps
        # si saltano i frame in eccesso. NON drenare tutta la coda: il decoder corre più
        # veloce del realtime e scartarne i frame pronti accelererebbe il video.
        frame = None
        for _ in range(steps):
            try:
                frame = self._video_queue.get_nowait()
            except queue.Empty:
                break

        if frame is not None:
            self._upload_video_frame(frame)

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

    # Il rendering è ora interamente in _render() (shader + instancing, core 3.3):
    # i vecchi draw_background/draw_bars (pipeline fixed-function) sono stati rimossi.
