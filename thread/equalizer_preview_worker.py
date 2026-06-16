import math
import queue
import threading
import time

import numpy as np
import scipy.fftpack as fft_pack
from PySide6.QtCore import QObject, Signal, Slot

RATE = 44100
FRAME_RATE = 120
N_FFT = 8192
MIN_FREQ = 20
MAX_FREQ = 20000


class EqualizerPreviewWorker(QObject):
    """
    Worker DSP dell'anteprima in-window.

    Gira su un QThread: legge l'audio dalla coda, calcola le ampiezze per banda
    (stessa pipeline del vecchio EqualizerTkinterThread: FFT 8192 punti, finestra
    di Blackman, somma per banda logaritmica, normalizzazione lineare per-frame),
    applica lo smoothing esponenziale ed emette le ampiezze normalizzate sul
    segnale ``bands_ready``. Non tocca alcun widget: il disegno avviene sul thread
    GUI nel paintEvent di EqualizerPreviewWidget (connessione queued cross-thread).
    """

    bands_ready = Signal(object)  # np.ndarray di float in [0, 1], una per banda

    def __init__(self, audio_queue, control_queue, noise_threshold=0.1, n_bands=9, channels=2):
        super().__init__()
        self.audio_queue = audio_queue
        self.control_queue = control_queue
        self.noise_threshold = noise_threshold
        self.channels = channels
        self.n_bands = n_bands

        self.frequency_bands = self.generate_frequency_bands(self.n_bands)
        self.previous_amplitudes = [0.0] * len(self.frequency_bands)
        self._stop = threading.Event()

    @Slot()
    def run(self):
        frame_duration = 1.0 / FRAME_RATE
        previous_time = time.monotonic()
        audio_buffer = None

        while not self._stop.is_set():
            # Messaggi di controllo (soglia rumore / numero bande)
            while not self.control_queue.empty():
                try:
                    message = self.control_queue.get_nowait()
                except queue.Empty:
                    break
                self.process_control_message(message)

            # Consuma la coda audio tenendo solo il buffer più recente
            while not self.audio_queue.empty():
                try:
                    audio_buffer = self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            now = time.monotonic()
            elapsed = now - previous_time

            if elapsed >= frame_duration:
                if audio_buffer is not None:
                    amplitudes = self.create_equalizer(
                        audio_buffer, self.frequency_bands, self.noise_threshold, self.channels
                    )
                    # Smoothing esponenziale (LERP) verso le nuove ampiezze
                    smooth_factor = 1 - math.exp(-elapsed / (frame_duration * 5))
                    interpolated = [
                        self.previous_amplitudes[i]
                        + smooth_factor * (amplitudes[i] - self.previous_amplitudes[i])
                        for i in range(len(amplitudes))
                    ]
                    self.previous_amplitudes = interpolated
                    # Array fresco a ogni frame: nessun aliasing col thread GUI.
                    self.bands_ready.emit(np.asarray(interpolated, dtype=float))
                previous_time = now
            else:
                # Ritmo il loop sul frame rate invece di fare busy-wait.
                time.sleep(min(frame_duration - elapsed, 0.002))

    def generate_frequency_bands(self, num_bands):
        min_log_freq = np.log10(MIN_FREQ)
        max_log_freq = np.log10(MAX_FREQ)
        log_freqs = np.logspace(min_log_freq, max_log_freq, num=num_bands + 1)
        return [(log_freqs[i], log_freqs[i + 1]) for i in range(num_bands)]

    def create_equalizer(self, audio_data, frequency_bands, noise_threshold=0.1, channels=2):
        audio_samples = audio_data.T

        assert audio_samples.shape[0] == channels, (
            f"Input data has {audio_samples.shape[0]} channels, expected {channels}"
        )

        equalizer_data = []
        for channel_samples in audio_samples:
            window = np.blackman(len(channel_samples))
            windowed_samples = channel_samples * window
            padded_samples = np.pad(windowed_samples, (0, N_FFT - len(windowed_samples)), "constant")

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

        avg_band_amplitudes = np.mean(equalizer_data, axis=0)
        filtered_amplitudes = [amp if amp >= noise_threshold else 0 for amp in avg_band_amplitudes]
        max_amplitude = max(filtered_amplitudes)
        epsilon = 1e-12
        normalized_amplitudes = [
            (amplitude / (max_amplitude + epsilon)) if max_amplitude > 0 else 0
            for amplitude in filtered_amplitudes
        ]
        return normalized_amplitudes

    def process_control_message(self, message):
        if message["type"] == "set_noise_threshold":
            self.noise_threshold = message["value"]
        elif message["type"] == "set_frequency_bands":
            self.frequency_bands = self.generate_frequency_bands(message["value"])
            self.previous_amplitudes = [0.0] * len(self.frequency_bands)

    def stop(self):
        self._stop.set()
