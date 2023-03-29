import threading
import math
import time

import glfw
from OpenGL.GL import *
import numpy as np
import scipy.fftpack as fft_pack

RATE = 44100
FRAME_RATE = 144
N_FFT= 8192
MIN_FREQ = 20
MAX_FREQ = 20000


class EqualizerOpenGLThread(threading.Thread):
    def __init__(self, audio_queue, noise_threshold=0.1, channels=2, n_bands=9, control_queue=None):
        super().__init__()
        self.audio_queue = audio_queue
        self.control_queue = control_queue
        self.noise_threshold = noise_threshold
        self.channels = channels
        self.n_bands = n_bands

        # List of temporary amplitudes to create an animation effect
        self.frequency_bands = self.generate_frequency_bands(self.n_bands)

        self.previous_amplitudes = [0] * len(self.frequency_bands)
        self.vbos = []
        self.bar_vertices = np.array([
            [0, 0], [1, 0], [1, 1],
            [0, 0], [1, 1], [0, 1]
        ], dtype=np.float32)
        
        self.stop_event = threading.Event()

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

                band_amplitudes.append(band_amplitude)

            equalizer_data.append(band_amplitudes)

        avg_band_amplitudes = np.mean(equalizer_data, axis=0)
        filtered_amplitudes = [amp if amp >= noise_threshold else 0 for amp in avg_band_amplitudes]
        max_amplitude = max(filtered_amplitudes)
        epsilon = 1e-12  # Add a small value to prevent division by very small numbers
        normalized_amplitudes = [(amplitude / (max_amplitude + epsilon)) if max_amplitude > 0 else 0
                                for amplitude in filtered_amplitudes]

        return normalized_amplitudes
    
    def init_vbos(self):
        self.vbos = []
        for _ in range(len(self.frequency_bands)):
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, self.bar_vertices.nbytes, self.bar_vertices, GL_DYNAMIC_DRAW)
            self.vbos.append(vbo)

    def process_control_message(self, message):
        if message["type"] == "set_noise_threshold":
            self.noise_threshold = message["value"]
        elif message['type'] == 'set_frequency_bands':
            self.frequency_bands = self.generate_frequency_bands(message["value"])
            self.previous_amplitudes = [0] * len(self.frequency_bands)

            # Clear and reinitialize VBOs
            glDeleteBuffers(len(self.vbos), self.vbos)
            self.init_vbos()

    def run(self):
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        primary_monitor = glfw.get_primary_monitor()
        window = glfw.create_window(1920, 1080, "Audio Visualizer", primary_monitor, None)
        if not window:
            glfw.terminate()
            raise Exception("GLFW window creation failed")

        glfw.make_context_current(window)

        # Disable auto iconify
        glfw.set_window_attrib(window, glfw.AUTO_ICONIFY, glfw.FALSE)

        glfw.set_key_callback(window, self.key_callback)

        # Initialize VBOs
        self.init_vbos()

        # initialize audio buffer
        audio_buffer = None

        frame_duration = 1 / FRAME_RATE
        previous_time = time.time()

        while not glfw.window_should_close(window) and not self.stop_event.is_set():
            # Process messages from the control queue
            while not self.control_queue.empty():
                message = self.control_queue.get(block=False)
                self.process_control_message(message)

            current_time = time.time()
            elapsed_time = current_time - previous_time

            # Consume data from the queue and update the audio buffer
            while not self.audio_queue.empty():
                audio_buffer = self.audio_queue.get(block=False)

            if audio_buffer is not None:
                amplitudes = self.create_equalizer(audio_buffer, self.frequency_bands, self.noise_threshold, self.channels)

                # exponential smoothing
                smooth_factor = 1 - math.exp(-elapsed_time / (frame_duration * 5))
                # Render the bars using OpenGL
                self.draw_bars(amplitudes, window, smooth_factor)

            previous_time = current_time
            glfw.swap_buffers(window)
            glfw.poll_events()

        # Clean up resources before exiting
        glfw.destroy_window(window)
        glfw.terminate()
        self.stop_event.set()

    def stop(self):
        # Stop the thread by breaking the main loop
        self.stop_event.set()

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            self.stop()

    def draw_bars(self, amplitudes, window, smooth_factor=0.1):
        window_width, window_height = glfw.get_framebuffer_size(window)

        # Return early if the window dimensions are invalid
        if window_width == 0 or window_height == 0:
            return
        
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT)

        num_bars = len(amplitudes)
        window_width, window_height = glfw.get_framebuffer_size(window)
        total_bar_width = window_width / num_bars
        bar_width = total_bar_width * 0.8
        bar_spacing = total_bar_width * 0.2

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, window_width, 0, window_height, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

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
            total_height = interpolated_amplitude * window_height

            # Calculate the heights of the three color sections based on the total height
            green_height = min(total_height, 0.5 * window_height)
            yellow_height = min(max(total_height - green_height, 0), 0.3 * window_height)
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