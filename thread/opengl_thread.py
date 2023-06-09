import threading
import math
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Callable

import glfw
from OpenGL.GL import *
import numpy as np
import scipy.fftpack as fft_pack
from PIL import Image

RATE = 44100
N_FFT= 32768
MIN_FREQ = 20
MAX_FREQ = 20000


def compute_fft(channel_samples, padded_samples):
    # Apply a window function to reduce leakage effect
    window = np.blackman(len(channel_samples))
    windowed_samples = channel_samples * window

    # Zero-padding
    padded_samples[:len(windowed_samples)] = windowed_samples
    padded_samples[len(windowed_samples):] = 0

    # Calculate FFT using NumPy
    fft_values = np.fft.fft(padded_samples)

    # Convert FFT values to magnitudes
    fft_magnitudes = np.abs(fft_values)

    return fft_magnitudes

def compute_channel_fft(args):
    audio_samples, padded_samples, volume, frequency_band_chunks = args

    # Compute FFT for all channels
    fft_magnitudes = [compute_fft(channel_samples, padded_samples.copy()) for channel_samples in audio_samples]
    freqs = fft_pack.fftfreq(len(fft_magnitudes[0]), 1.0 / RATE)

    # Process frequency bands for all channels
    band_amplitude_chunks = [process_frequency_bands((fft_values, freqs, frequency_band_chunks, volume)) for fft_values in fft_magnitudes]

    return band_amplitude_chunks

def process_frequency_bands(args):
    fft_values, freqs, bands, volume = args
    band_amplitudes = []

    for low_freq, high_freq in bands:
        band_filter = np.logical_and(freqs >= low_freq, freqs <= high_freq)

        if np.any(band_filter):
            band_amplitude = np.mean(fft_values[band_filter]) * np.sum(band_filter)
        else:
            band_amplitude = 0

        band_amplitude *= volume
        band_amplitudes.append(band_amplitude)

    return band_amplitudes


class EqualizerOpenGLThread(threading.Thread):
    def __init__(self,
                 audio_queue,
                 noise_threshold=0.1,
                 channels=2,
                 n_bands=9,
                 control_queue=None,
                 workers: int = 5,
                 auto_workers: bool = False,
                 monitor= None,
                 bg_image: Optional[Image.Image]= None,
                 bg_alpha: Optional[float]=None,
                 window_close_callback: Callable= None):
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

        self.previous_amplitudes = [0] * len(self.frequency_bands)
        self.vbos = []
        self.bar_vertices = np.array([
            [0, 0], [1, 0], [1, 1],
            [0, 0], [1, 1], [0, 1]
        ], dtype=np.float32)
        
        self.stop_event = threading.Event()
        self._background_texture = None

        self.window_width = self.window_height = None
        self.frame_rate = None

        self._workers = self._calculate_workers(n_bands) if auto_workers else workers

        self.executor = ProcessPoolExecutor(max_workers=self._workers)

    def _calculate_workers(self, bands:int) -> int:
        """
        Calculate the number fo the workers based on the bands number

        Args:
            bands (int): Number of bands

        Returns:
            int: The number of workers
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

    def generate_frequency_bands(self, num_bands):
        min_log_freq = np.log10(MIN_FREQ)
        max_log_freq = np.log10(MAX_FREQ)

        log_freqs = np.logspace(min_log_freq, max_log_freq, num=num_bands + 1)

        bands = [(log_freqs[i], log_freqs[i + 1]) for i in range(num_bands)]

        return bands
    
    def create_equalizer(self, audio_data, frequency_bands, noise_threshold=0.1, channels=2):
        audio_samples = audio_data.T

        # Check that the number of channels in the input data matches the expected number of channels
        assert audio_samples.shape[0] == channels, f"Input data has {audio_samples.shape[0]} channels, expected {channels}"

        # Calculate the volume of the input audio data
        volume = np.sqrt(np.mean(audio_data ** 2))

        if self._workers == 1:
            equalizer_data = []

            for channel_samples in audio_samples:
                # Apply a window function to reduce leakage effect
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

                    # Scale the amplitude based on the actual volume
                    band_amplitude *= volume
                    band_amplitudes.append(band_amplitude)

                equalizer_data.append(band_amplitudes)
        else:
            bands_per_worker = len(frequency_bands) // self._workers

            # Split frequency bands into smaller chunks for each worker
            frequency_band_chunks = [frequency_bands[i:i + bands_per_worker] for i in range(0, len(frequency_bands), bands_per_worker)]

            padded_samples = np.zeros(N_FFT)

            # Compute channel FFT and process frequency bands in parallel
            band_amplitude_chunks = list(self.executor.map(compute_channel_fft, [(audio_samples, padded_samples, volume, chunk) for chunk in frequency_band_chunks]))

            equalizer_data = [[] for _ in range(channels)]

            # Flatten the list of band_amplitude_chunks and group by channel
            for chunk in band_amplitude_chunks:
                for channel_index, channel_data in enumerate(chunk):
                    equalizer_data[channel_index].extend(channel_data)

        avg_band_amplitudes = np.mean(equalizer_data, axis=0)
        filtered_amplitudes = [amp if amp >= noise_threshold else 0 for amp in avg_band_amplitudes]
        max_amplitude = max(filtered_amplitudes)
        epsilon = 1e-12  # Add a small value to prevent division by very small numbers
        normalized_amplitudes = [(amplitude / (max_amplitude + epsilon)) if max_amplitude > 0 else 0 for amplitude in filtered_amplitudes]

        return normalized_amplitudes
    
    def init_vbos(self):
        self.vbos = []
        for _ in range(len(self.frequency_bands)):
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, self.bar_vertices.nbytes, self.bar_vertices, GL_DYNAMIC_DRAW)
            self.vbos.append(vbo)

    def process_control_message(self, message):
        message_type = message["type"]

        if message_type == "set_noise_threshold":
            self._noise_threshold = message["value"]

        elif message_type == 'set_frequency_bands':
            self._n_bands = message["value"]
            self.frequency_bands = self.generate_frequency_bands(message["value"])
            self.previous_amplitudes = [0] * len(self.frequency_bands)
            # Clear and reinitialize VBOs
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
            if self._auto_workers:
                self._auto_workers = False
            else:
                self._auto_workers = True
                self._workers = self._calculate_workers(self._n_bands)
                if self._workers > 1:
                    self.executor.shutdown(wait=True)
                    self.executor = None
                    self.executor = ProcessPoolExecutor(max_workers=self._workers)

    def run(self):
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

        window = glfw.create_window(self.window_width, self.window_height, "Audio Visualizer", selected_monitor, None)
        if not window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")
        
        # check if the stop event is set
        if self.stop_event.is_set():
            self.stop_event.clear()

        glfw.make_context_current(window)

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

        # initialize audio buffer
        audio_buffer = None

        frame_duration = 1 / self.frame_rate
        previous_time = time.time()

        while not glfw.window_should_close(window) and not self.stop_event.is_set():
            # Process messages from the control queue
            while not self._control_queue.empty():
                message = self._control_queue.get(block=False)
                self.process_control_message(message)

            current_time = time.time()
            elapsed_time = current_time - previous_time

            # Consume data from the queue and update the audio buffer
            while not self._audio_queue.empty():
                audio_buffer = self._audio_queue.get(block=False)

            if audio_buffer is not None:
                amplitudes = self.create_equalizer(audio_buffer, self.frequency_bands, self._noise_threshold, self._channels)

                # exponential smoothing
                smooth_factor = 1 - math.exp(-elapsed_time / (frame_duration * 5))
                # Render the bars using OpenGL
                self.draw_bars(amplitudes, smooth_factor)

            previous_time = current_time
            glfw.swap_buffers(window)
            glfw.poll_events()

        # Clean up resources before exiting
        glfw.destroy_window(window)
        glfw.terminate()

    def stop(self):
        # Stop the thread by breaking the main loop
        self.stop_event.set()
        self.executor.shutdown(wait=True)

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            self.stop()
            self.window_close_callback()

    def update_texture_alpha(self):
        self._bg_image.putalpha(int(255 * self._bg_alpha))
        self._background_texture = self.load_texture()

    def load_texture(self):
        im = self._bg_image.transpose(Image.FLIP_TOP_BOTTOM)
        im_data = im.convert('RGBA').tobytes()

        width, height = im.size

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, im_data)

        return texture
    
    def draw_background(self):
        glBindTexture(GL_TEXTURE_2D, self._background_texture)

        glEnable(GL_TEXTURE_2D)
        glColor3f(1.0, 1.0, 1.0)  # Set the color to white so the texture is not tinted
        glBegin(GL_QUADS)

        glTexCoord2f(0, 0)
        glVertex2f(0, 0)

        glTexCoord2f(1, 0)
        glVertex2f(self.window_width, 0)

        glTexCoord2f(1, 1)
        glVertex2f(self.window_width, self.window_height)

        glTexCoord2f(0, 1)
        glVertex2f(0, self.window_height)

        glEnd()
        glDisable(GL_TEXTURE_2D)

    def draw_bars(self, amplitudes, smooth_factor=0.1):
        
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT)

        num_bars = len(amplitudes)
        total_bar_width = self.window_width / num_bars
        bar_width = total_bar_width * 0.8
        bar_spacing = total_bar_width * 0.2

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.window_width, 0, self.window_height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.draw_background()

        for i, amplitude in enumerate(amplitudes):
            # Bind the VBO
            vbo = self.vbos[i]
            glBindBuffer(GL_ARRAY_BUFFER, vbo)

            # LERP between the previous amplitude and the new amplitude
            smooth_factor = 0.1
            interpolated_amplitude = self.previous_amplitudes[i] + smooth_factor * (amplitude - self.previous_amplitudes[i])
            self.previous_amplitudes[i] = interpolated_amplitude

            x = i * total_bar_width + bar_spacing / 2
            y = 0

            # Calculate the total height of the bar based on the amplitude
            total_height = interpolated_amplitude * self.window_height

            # Calculate the heights of the three color sections based on the total height
            green_height = min(total_height, 0.5 * self.window_height)
            yellow_height = min(max(total_height - green_height, 0), 0.3 * self.window_height)
            red_height = max(total_height - green_height - yellow_height, 0)

            # Draw the green rectangle
            glColor3f(0, 1, 0)
            glBegin(GL_QUADS)
            glVertex2f(x, y)
            glVertex2f(x + bar_width, y)
            glVertex2f(x + bar_width, y + green_height)
            glVertex2f(x, y + green_height)
            glEnd()

            # Draw the yellow rectangle
            glColor3f(1, 1, 0)
            glBegin(GL_QUADS)
            glVertex2f(x, y + green_height)
            glVertex2f(x + bar_width, y + green_height)
            glVertex2f(x + bar_width, y + green_height + yellow_height)
            glVertex2f(x, y + green_height + yellow_height)
            glEnd()

            # Draw the red rectangle
            glColor3f(1, 0, 0)
            glBegin(GL_QUADS)
            glVertex2f(x, y + green_height + yellow_height)
            glVertex2f(x + bar_width, y + green_height + yellow_height)
            glVertex2f(x + bar_width, y + green_height + yellow_height + red_height)
            glVertex2f(x, y + green_height + yellow_height + red_height)
            glEnd()
