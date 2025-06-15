import threading
import math
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Callable, List, Tuple
from collections import deque

import glfw
from OpenGL.GL import *
import numpy as np
import scipy.fftpack as fft_pack
from PIL import Image
from PIL.Image import Transpose

from thread.AudioBufferAccumulator import AudioBufferAccumulator

RATE = 44100
N_FFT = 2048
MIN_FREQ = 20
MAX_FREQ = 20000

GENERAL_LOG = True
FFT_LOGS = True
PERF_LOGS = True


def compute_fft(channel_samples: np.ndarray) -> np.ndarray:
    """
    Compute the FFT magnitudes for one channel of audio.

    1) Applies a Hanning window to reduce spectral leakage.
    2) Zero-pads only if len(channel_samples) < N_FFT.
    3) Returns the absolute value of the FFT (magnitudes).

    Args:
        channel_samples (np.ndarray): The 1D array of audio samples for a single channel.

    Returns:
        np.ndarray: The magnitude of the FFT for these samples.
    """
    # Apply a Blackman window (come in Tkinter)
    window = np.blackman(len(channel_samples))
    windowed_samples = channel_samples * window

    # If needed, zero-pad up to N_FFT
    if len(channel_samples) < N_FFT:
        padded = np.zeros(N_FFT, dtype=windowed_samples.dtype)
        padded[:len(windowed_samples)] = windowed_samples
        fft_values = np.fft.fft(padded)
    else:
        # Otherwise, no need to pad
        fft_values = np.fft.fft(windowed_samples)

    # Return magnitude
    return np.abs(fft_values)


def process_frequency_bands(args):
    """
    process_frequency_bands handles the per-band amplitude calculation
    using np.mean for each band filter (allineato con Tkinter).
    """
    fft_values, freqs, bands = args
    band_amplitudes = []

    for low_freq, high_freq in bands:
        band_mask = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        if np.any(band_mask):
            # Usa solo mean come in Tkinter
            band_amplitude = np.mean(fft_values[band_mask]) * np.sum(band_mask)
        else:
            band_amplitude = 0.0

        band_amplitudes.append(band_amplitude)

    # LOG
    if FFT_LOGS:
        print('LOGGING RAW AMPLITUDES')
        for i, amp in enumerate(band_amplitudes):
            print(f"Raw band {i} amplitude = {amp:.4f}")

    return band_amplitudes


def compute_channel_fft(args):
    """
    compute_channel_fft is used in parallel or single-worker mode
    to compute the FFT for each channel, then process frequency bands.

    Steps:
        1) For each channel in audio_samples, call compute_fft (using the Blackman window).
        2) Filter to the positive half of freqs if desired (shown below).
        3) Use process_frequency_bands(...) to get band amplitudes for each channel.

    Returns:
        list of lists: band_amplitude_chunks, where each sub-list corresponds
                      to one channel's band amplitudes.
    """
    audio_samples, frequency_band_chunks = args

    # Compute FFT for all channels
    fft_magnitudes_list = []
    for channel_samples in audio_samples:
        fft_magnitudes = compute_fft(channel_samples)
        fft_magnitudes_list.append(fft_magnitudes)

    # Frequencies array (same length as the final FFT array)
    freqs_full = fft_pack.fftfreq(len(fft_magnitudes_list[0]), 1.0 / RATE)

    # Filter only the positive half
    pos_mask = (freqs_full >= 0)
    freqs_pos = freqs_full[pos_mask]

    # For each channel, keep only positive frequencies
    # (This ensures we don't double-count negative freq bins.)
    channel_fft_list = [
        fft_mags[pos_mask] for fft_mags in fft_magnitudes_list
    ]

    # Process frequency bands for each channel
    band_amplitude_chunks = []
    for fft_values in channel_fft_list:
        band_amplitude_chunks.append(
            process_frequency_bands(
                (fft_values, freqs_pos, frequency_band_chunks)
            )
        )

    return band_amplitude_chunks


class EqualizerOpenGLThread(threading.Thread):
    """
    Thread per la visualizzazione OpenGL dell'equalizzatore audio.

    Attributes:
        audio_queue: Coda per ricevere i dati audio
        control_queue: Coda per i messaggi di controllo
        noise_threshold: Soglia di rumore (0-1)
        channels: Numero di canali audio (default 2 per stereo)
        n_bands: Numero di bande di frequenza
        monitor: Monitor per fullscreen
        bg_image: Immagine di sfondo
        bg_alpha: Trasparenza dello sfondo
        window_close_callback: Callback alla chiusura della finestra
        workers: Numero di worker per il calcolo parallelo
        auto_workers: Se True, calcola automaticamente il numero di worker
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
        self._auto_workers = auto_workers

        # List of temporary amplitudes to create an animation effect
        self.frequency_bands = self.generate_frequency_bands(self._n_bands)
        self.previous_amplitudes = [0.0] * len(self.frequency_bands)

        # We create one VBO per bar
        self.vbos = []

        self.stop_event = threading.Event()
        self._background_texture = None

        self.window_width = self.window_height = None
        self.frame_rate = None

        self._workers = self._calculate_workers(n_bands) if auto_workers else workers

        self.executor = ProcessPoolExecutor(max_workers=self._workers)

        # AudioBufferAccumulator per overlap (come Tkinter)
        self.audio_accumulator = AudioBufferAccumulator(window_size=N_FFT,
                                                        overlap_size=N_FFT // 2,
                                                        channels=self._channels)

        # --- MIGLIORAMENTI PER FLUIDITÀ ---
        # 1. Buffer circolare per interpolazione temporale
        self.amplitude_history_size = 5  # Numero di campioni storici da mantenere
        self.amplitude_history = deque(maxlen=self.amplitude_history_size)

        # 2. Timing più preciso - SARÀ IMPOSTATO DINAMICAMENTE
        self.fft_interval = None  # Verrà calcolato in base al refresh rate
        self.last_fft_time = 0.0

        # 3. Target amplitudes per interpolazione più fluida
        self.target_amplitudes = [0.0] * self._n_bands
        self.current_amplitudes = [0.0] * self._n_bands

        # 4. Fattore di smoothing adattivo
        self.base_smooth_factor = 0.15  # Valore base per 60Hz
        self.smooth_acceleration = 2.0  # Accelerazione per cambiamenti rapidi

        # Questi verranno aggiustati in base al refresh rate nel metodo run()

    def _calculate_workers(self, bands: int) -> int:
        """
        Calcola il numero di worker basato sul numero di bande.

        Args:
            bands (int): Numero di bande

        Returns:
            int: Il numero di worker
        """
        if bands <= 10:
            return 1
        elif bands <= 30:
            return 2
        elif bands <= 50:
            return 3
        elif bands <= 70:
            return 4
        else:
            return 5

    def generate_frequency_bands(self, num_bands: int) -> List[Tuple[float, float]]:
        """
        Genera le bande di frequenza in scala logaritmica (come in Tkinter).

        Args:
            num_bands (int): Numero di bande desiderate

        Returns:
            list[tuple[float, float]]: Lista di tuple (freq_min, freq_max)
        """
        min_log_freq = np.log10(MIN_FREQ)
        max_log_freq = np.log10(MAX_FREQ)

        log_freqs = np.logspace(min_log_freq, max_log_freq, num=num_bands + 1)
        bands = [(log_freqs[i], log_freqs[i + 1]) for i in range(num_bands)]

        return bands

    def create_equalizer(self,
                         audio_data: np.ndarray,
                         frequency_bands: list[tuple[float, float]],
                         noise_threshold: float = 0.1,
                         channels: int = 2) -> list[float]:
        """
        Calcola le ampiezze per-banda per un blocco di dati audio multi-canale.
        Allineato con l'implementazione Tkinter.

        Args:
            audio_data (np.ndarray): shape (N_FFT, channels).
            frequency_bands (list): lista di tuple (low_freq, high_freq).
            noise_threshold (float): soglia di rumore lineare.
            channels (int): numero di canali (2 per stereo).

        Returns:
            list[float]: ampiezza normalizzata per ogni banda in [0..1].
        """
        # LOG dimensioni
        if FFT_LOGS:
            print(f"LOG: create_equalizer -> audio_data.shape={audio_data.shape}, freq_bands={len(frequency_bands)}")

        # Transpose so shape = (channels, N_FFT)
        audio_samples: np.ndarray = audio_data.T
        assert audio_samples.shape[0] == channels, (
            f"Input data has {audio_samples.shape[0]} channels, expected {channels}"
        )

        # Decide whether to use parallel approach or single-thread fallback
        if self._workers > 1:
            # PARALLEL BRANCH
            bands_per_worker = max(1, len(frequency_bands) // self._workers)

            frequency_band_chunks = []
            start_index = 0
            while start_index < len(frequency_bands):
                end_index = min(start_index + bands_per_worker, len(frequency_bands))
                sub_bands = frequency_bands[start_index:end_index]
                frequency_band_chunks.append(sub_bands)
                start_index = end_index

            # Construct the arguments for each chunk
            parallel_args = []
            for chunk in frequency_band_chunks:
                parallel_args.append((audio_samples, chunk))

            # Use executor to parallelize
            band_amplitude_chunks = list(
                self.executor.map(
                    compute_channel_fft,
                    parallel_args
                )
            )

            # Combine results per channel
            equalizer_data = [[] for _ in range(channels)]
            for chunk_result in band_amplitude_chunks:
                for ch_index, ch_band_amps in enumerate(chunk_result):
                    equalizer_data[ch_index].extend(ch_band_amps)

        else:
            # SINGLE-THREAD FALLBACK (come Tkinter)
            equalizer_data = []
            for channel_samples in audio_samples:
                # Apply Blackman window
                window = np.blackman(len(channel_samples))
                windowed_samples = channel_samples * window

                # Zero-padding
                padded_samples = np.pad(windowed_samples, (0, N_FFT - len(windowed_samples)), 'constant')
                fft_values = np.abs(fft_pack.fft(padded_samples))
                freqs = fft_pack.fftfreq(len(fft_values), 1.0 / RATE)

                band_amplitudes = []
                for low_freq, high_freq in frequency_bands:
                    band_filter = np.logical_and(freqs >= low_freq, freqs <= high_freq)

                    if np.any(band_filter):
                        band_amplitude = np.mean(fft_values[band_filter]) * np.sum(band_filter)
                    else:
                        band_amplitude = 0

                    band_amplitudes.append(band_amplitude)

                equalizer_data.append(band_amplitudes)

        # Merge channels (average)
        avg_band_amplitudes = np.mean(equalizer_data, axis=0)

        # Apply noise threshold
        filtered_amplitudes = [
            amp if amp >= noise_threshold else 0
            for amp in avg_band_amplitudes
        ]

        # Normalize directly without dB conversion (come Tkinter)
        max_amplitude = max(filtered_amplitudes)
        epsilon = 1e-12
        normalized_amplitudes = [
            (amplitude / (max_amplitude + epsilon)) if max_amplitude > 0 else 0
            for amplitude in filtered_amplitudes
        ]

        return normalized_amplitudes

    def init_vbos(self):
        """
        Crea esattamente len(self.frequency_bands) VBO.
        Ogni barra verrà disegnata con self.vbos[i].
        """
        # Delete old VBOs if they exist (to avoid resource leaks)
        if self.vbos:
            glDeleteBuffers(len(self.vbos), self.vbos)

        self.vbos = []
        count = len(self.frequency_bands)
        for _ in range(count):
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            # Allocate space for 6 vertices (x,y) for a rectangle
            nbytes = 6 * 2 * 4  # 6 vertices * 2 floats * 4 bytes
            glBufferData(GL_ARRAY_BUFFER, nbytes, None, GL_DYNAMIC_DRAW)
            self.vbos.append(vbo)

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
            self.frequency_bands = self.generate_frequency_bands(message["value"])
            self.previous_amplitudes = [0] * len(self.frequency_bands)
            self.target_amplitudes = [0.0] * self._n_bands
            self.current_amplitudes = [0.0] * self._n_bands
            self.amplitude_history.clear()
            # Clear and reinitialize VBOs
            if self.vbos:
                glDeleteBuffers(len(self.vbos), self.vbos)
            self.init_vbos()

            if self._auto_workers:
                new_workers_number = self._calculate_workers(message["value"])
                if new_workers_number != self._workers:
                    self._workers = new_workers_number
                    if self._workers > 1:
                        self.executor.shutdown(wait=True)
                        self.executor = None
                        self.executor = ProcessPoolExecutor(max_workers=self._workers)

        elif message_type == 'set_alpha':
            self._bg_alpha = message['value']
            self.update_texture_alpha()

        elif message_type == 'set_image':
            self._bg_image = message['value']
            self._background_texture = self.load_texture()

        elif message_type == 'set_workers':
            self._workers = message["value"]
            self.executor.shutdown(wait=True)
            self.executor = None
            self.executor = ProcessPoolExecutor(max_workers=message["value"])

        elif message_type == 'set_auto_workers':
            self._auto_workers = message["value"]
            if self._auto_workers:
                self._workers = self._calculate_workers(self._n_bands)
                if self._workers > 1:
                    self.executor.shutdown(wait=True)
                    self.executor = None
                    self.executor = ProcessPoolExecutor(max_workers=self._workers)

    # def smooth_amplitudes(self, new_amplitudes: list[float], delta_time: float) -> list[float]:
    #     """
    #     Applica uno smoothing adattivo alle ampiezze per maggiore fluidità.
    #
    #     Args:
    #         new_amplitudes: Le nuove ampiezze calcolate dall'FFT
    #         delta_time: Tempo trascorso dall'ultimo frame
    #
    #     Returns:
    #         list[float]: Ampiezze interpolate in modo fluido
    #     """
    #     # Aggiungi le nuove ampiezze alla storia
    #     self.amplitude_history.append(new_amplitudes)
    #
    #     # Se non abbiamo abbastanza storia, usa solo l'ultimo valore
    #     if len(self.amplitude_history) < 2:
    #         return new_amplitudes
    #
    #     smoothed = []
    #     for i in range(len(new_amplitudes)):
    #         # Calcola la media mobile delle ultime ampiezze
    #         historical_values = [hist[i] for hist in self.amplitude_history]
    #         avg_amplitude = np.mean(historical_values)
    #
    #         # Calcola la velocità di cambiamento
    #         if len(self.amplitude_history) >= 2:
    #             velocity = abs(self.amplitude_history[-1][i] - self.amplitude_history[-2][i])
    #         else:
    #             velocity = 0
    #
    #         # Adatta il fattore di smoothing basato sulla velocità
    #         # Più veloce il cambiamento, più reattivo dovrebbe essere
    #         adaptive_smooth = self.base_smooth_factor + (velocity * self.smooth_acceleration)
    #         adaptive_smooth = min(adaptive_smooth, 0.8)  # Limita il massimo
    #
    #         # Interpolazione esponenziale con fattore adattivo
    #         # Normalizza per il frame rate - monitor più veloci = smoothing più veloce
    #         frame_rate_factor = self.frame_rate / 60.0  # Normalizza rispetto a 60Hz
    #         time_factor = 1 - math.exp(-delta_time * adaptive_smooth * 60 * frame_rate_factor)
    #
    #         # Interpola tra il valore corrente e il target
    #         self.current_amplitudes[i] += (new_amplitudes[i] - self.current_amplitudes[i]) * time_factor
    #
    #         # Aggiungi una piccola componente della media mobile per stabilità
    #         smoothed_value = self.current_amplitudes[i] * 0.9 + avg_amplitude * 0.1
    #         smoothed.append(smoothed_value)
    #
    #     return smoothed

    def smooth_amplitudes(self, new_amplitudes: list[float], delta_time: float) -> list[float]:
        # Ensure delta_time is positive and reasonable
        delta_time = max(delta_time, 1e-6)  # Prevent negative or zero delta_time

        # Aggiungi le nuove ampiezze alla storia
        self.amplitude_history.append(new_amplitudes)

        if len(self.amplitude_history) < 2:
            return new_amplitudes

        smoothed = []
        for i in range(len(new_amplitudes)):
            historical_values = [hist[i] for hist in self.amplitude_history]
            avg_amplitude = np.mean(historical_values)

            if len(self.amplitude_history) >= 2:
                velocity = abs(self.amplitude_history[-1][i] - self.amplitude_history[-2][i])
            else:
                velocity = 0

            adaptive_smooth = self.base_smooth_factor + (velocity * self.smooth_acceleration)
            adaptive_smooth = min(adaptive_smooth, 0.8)

            frame_rate_factor = self.frame_rate / 60.0
            time_factor = 1 - math.exp(-delta_time * adaptive_smooth * 60 * frame_rate_factor)

            self.current_amplitudes[i] += (new_amplitudes[i] - self.current_amplitudes[i]) * time_factor
            smoothed_value = self.current_amplitudes[i] * 0.9 + avg_amplitude * 0.1
            smoothed.append(smoothed_value)

        return smoothed

    def run(self):
        """
        Main loop del thread OpenGL.
        """
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
            glfw.terminate()
            raise Exception('Not actual monitor instance has been found')

        # Get the primary monitor's video mode
        video_mode = glfw.get_video_mode(selected_monitor)
        self.window_width = video_mode.size.width
        self.window_height = video_mode.size.height
        self.frame_rate = video_mode.refresh_rate

        # Calcola dinamicamente l'intervallo FFT basato sul refresh rate del monitor
        # Per monitor ad alto refresh rate (>100Hz), aggiorna FFT ogni 2 frame
        # Per monitor standard (60-100Hz), aggiorna FFT ogni frame
        if self.frame_rate > 100:
            self.fft_interval = 2.0 / self.frame_rate  # Ogni 2 frame
        else:
            self.fft_interval = 1.0 / self.frame_rate  # Ogni frame

        # Adatta il fattore di smoothing base al refresh rate
        # Monitor più veloci necessitano di smoothing più aggressivo per mantenere fluidità
        if self.frame_rate >= 144:
            self.base_smooth_factor = 0.25  # Più reattivo per high refresh
        elif self.frame_rate >= 120:
            self.base_smooth_factor = 0.20
        else:
            self.base_smooth_factor = 0.15  # Default per 60Hz

        print(
            f"Monitor refresh rate: {self.frame_rate}Hz, FFT interval: {self.fft_interval * 1000:.2f}ms, Smooth factor: {self.base_smooth_factor}")

        window = glfw.create_window(self.window_width, self.window_height, "Audio Visualizer", selected_monitor, None)
        if not window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        # check if the stop event is set
        if self.stop_event.is_set():
            self.stop_event.clear()

        glfw.make_context_current(window)

        # Enable VSync per sincronizzazione con il refresh del monitor
        glfw.swap_interval(1)

        # Enable blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Load the background texture after creating the OpenGL context
        self._background_texture = self.load_texture()

        # Disable auto iconify
        glfw.set_window_attrib(window, glfw.AUTO_ICONIFY, glfw.FALSE)

        glfw.set_key_callback(window, self.key_callback)

        # Initialize VBOs
        self.init_vbos()

        frame_duration = 1 / self.frame_rate
        previous_time = time.time()

        # Timing più preciso
        last_frame_time = time.perf_counter()

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

            # 1) Pull audio data in modo più controllato
            # Per monitor ad alto refresh rate, processa più chunk per frame
            audio_chunks_processed = 0
            max_chunks_per_frame = max(3, int(self.frame_rate / 60))  # Scala con il refresh rate

            while not self._audio_queue.empty() and audio_chunks_processed < max_chunks_per_frame:
                try:
                    audio_buffer = self._audio_queue.get_nowait()
                    self.audio_accumulator.add_samples(audio_buffer)
                    audio_chunks_processed += 1
                except:
                    break

            # 2) Calcola FFT a intervalli regolari
            current_time = time.perf_counter()
            if (current_time - self.last_fft_time) >= self.fft_interval:
                fft_window = self.audio_accumulator.get_window_for_fft()
                if fft_window.size > 0:
                    if GENERAL_LOG:
                        print('LOG: Start FFT computation')

                    t0 = time.perf_counter()
                    raw_amplitudes = self.create_equalizer(
                        audio_data=fft_window,
                        frequency_bands=self.frequency_bands,
                        noise_threshold=self._noise_threshold,
                        channels=self._channels
                    )
                    t1 = time.perf_counter()

                    if PERF_LOGS:
                        print(f"PERF LOG: FFT took {(t1 - t0) * 1000:.3f} ms")

                    # Aggiorna i target con smoothing
                    self.target_amplitudes = raw_amplitudes

                self.last_fft_time = current_time

            # 3) Applica smoothing avanzato
            smoothed_amplitudes = self.smooth_amplitudes(self.target_amplitudes, frame_delta)

            # 4) Draw bars con valori smoothed
            self.draw_bars(smoothed_amplitudes, frame_delta)

            glfw.swap_buffers(window)
            glfw.poll_events()

            if PERF_LOGS and frame_delta > 0:
                fps = 1.0 / frame_delta
                print(f"PERF LOG: FrameTime = {frame_delta * 1000:.3f}ms (FPS ~ {fps:.1f})")

        # Cleanup
        glfw.destroy_window(window)
        glfw.terminate()

    def stop(self):
        """
        Ferma il thread impostando l'evento di stop.
        """
        self.stop_event.set()
        self.executor.shutdown(wait=True)

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

    def update_texture_alpha(self):
        """
        Aggiorna l'alpha della texture di sfondo.
        """
        if self._bg_image:
            self._bg_image.putalpha(int(255 * self._bg_alpha))
            self._background_texture = self.load_texture()

    def load_texture(self):
        """
        Carica la texture di sfondo.

        Returns:
            int: ID della texture OpenGL
        """
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

    def draw_background(self):
        """
        Disegna lo sfondo con la texture.
        """
        glBindTexture(GL_TEXTURE_2D, self._background_texture)
        glEnable(GL_TEXTURE_2D)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0);
        glVertex2f(0, 0)
        glTexCoord2f(1, 0);
        glVertex2f(self.window_width, 0)
        glTexCoord2f(1, 1);
        glVertex2f(self.window_width, self.window_height)
        glTexCoord2f(0, 1);
        glVertex2f(0, self.window_height)
        glEnd()
        glDisable(GL_TEXTURE_2D)

    def draw_bars(self, amplitudes: list[float], delta_time: float):
        """
        Disegna le barre dell'equalizzatore con interpolazione più fluida.

        Args:
            amplitudes (list): Lista delle ampiezze già smoothed
            delta_time (float): Tempo trascorso dall'ultimo frame
        """
        # Clear
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT)

        num_bars = len(amplitudes)
        if num_bars < 1:
            return

        total_bar_width = self.window_width / float(num_bars)
        bar_width = total_bar_width * 0.8
        bar_spacing = total_bar_width * 0.2

        # Setup orthographic projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.window_width, 0, self.window_height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.draw_background()

        # Check array lengths
        if len(amplitudes) != len(self.previous_amplitudes) or len(amplitudes) != len(self.vbos):
            return

        for i, amplitude in enumerate(amplitudes):
            # Usa direttamente le ampiezze già smoothed
            interpolated_amplitude = amplitude
            self.previous_amplitudes[i] = interpolated_amplitude

            x = i * total_bar_width + bar_spacing / 2
            y = 0

            total_height = interpolated_amplitude * self.window_height

            # Colori come in Tkinter
            green_height = min(total_height, 0.5 * self.window_height)
            yellow_height = min(max(total_height - green_height, 0), 0.3 * self.window_height)
            red_height = max(total_height - green_height - yellow_height, 0)

            def rect_to_triangles(x1, y1, x2, y2):
                """
                Restituisce 6 vertici (due triangoli) per il rettangolo [x1,y1] - [x2,y2].

                Args:
                    x1, y1: Coordinate bottom-left
                    x2, y2: Coordinate top-right

                Returns:
                    np.ndarray: Array di vertici shape (6, 2)
                """
                return np.array([
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y1],
                    [x2, y2],
                    [x1, y2],
                ], dtype=np.float32)

            # 1) GREEN part
            if green_height > 0:
                x1, y1 = x, y
                x2, y2 = x + bar_width, y + green_height
                vertices = rect_to_triangles(x1, y1, x2, y2)

                glBindBuffer(GL_ARRAY_BUFFER, self.vbos[i])
                glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

                glColor3f(0, 1, 0)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(2, GL_FLOAT, 0, ctypes.c_void_p(0))
                glDrawArrays(GL_TRIANGLES, 0, 6)
                glDisableClientState(GL_VERTEX_ARRAY)

            # 2) YELLOW part
            if yellow_height > 0:
                y_start = y + green_height
                x1, y1 = x, y_start
                x2, y2 = x + bar_width, y_start + yellow_height
                vertices = rect_to_triangles(x1, y1, x2, y2)

                glBindBuffer(GL_ARRAY_BUFFER, self.vbos[i])
                glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

                glColor3f(1, 1, 0)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(2, GL_FLOAT, 0, ctypes.c_void_p(0))
                glDrawArrays(GL_TRIANGLES, 0, 6)
                glDisableClientState(GL_VERTEX_ARRAY)

            # 3) RED part
            if red_height > 0:
                y_start = y + green_height + yellow_height
                x1, y1 = x, y_start
                x2, y2 = x + bar_width, y_start + red_height
                vertices = rect_to_triangles(x1, y1, x2, y2)

                glBindBuffer(GL_ARRAY_BUFFER, self.vbos[i])
                glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

                glColor3f(1, 0, 0)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(2, GL_FLOAT, 0, ctypes.c_void_p(0))
                glDrawArrays(GL_TRIANGLES, 0, 6)
                glDisableClientState(GL_VERTEX_ARRAY)
