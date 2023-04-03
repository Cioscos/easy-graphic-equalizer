import threading
import math
import time
from typing import Optional

import glfw
from OpenGL.GL import *
import numpy as np
import scipy.fftpack as fft_pack
from PIL import Image

RATE = 44100
N_FFT= 8192
MIN_FREQ = 20
MAX_FREQ = 20000


class EqualizerOpenGLThread(threading.Thread):
    def __init__(self,
                 audio_queue,
                 noise_threshold=0.1,
                 channels=2,
                 n_bands=9,
                 control_queue=None,
                 bg_image: Optional[Image.Image]= None,
                 bg_alpha: Optional[float]=None):
        super().__init__()
        self._audio_queue = audio_queue
        self._control_queue = control_queue
        self._noise_threshold = noise_threshold
        self._channels = channels
        self._n_bands = n_bands
        self._bg_image = bg_image
        self._bg_alpha = bg_alpha

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
        message_type = message["type"]

        if message_type == "set_noise_threshold":

            self._noise_threshold = message["value"]
        elif message_type == 'set_frequency_bands':

            self.frequency_bands = self.generate_frequency_bands(message["value"])
            self.previous_amplitudes = [0] * len(self.frequency_bands)
            # Clear and reinitialize VBOs
            glDeleteBuffers(len(self.vbos), self.vbos)
            self.init_vbos()
        elif message_type == 'set_alpha':

            self._bg_alpha = message['value']
            self.update_texture_alpha()
        elif message_type == 'set_image':

            self._bg_image = message['value']
            self._background_texture = self.load_texture()

    def run(self):
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        primary_monitor = glfw.get_primary_monitor()
        
            # Get the primary monitor's video mode
        video_mode = glfw.get_video_mode(primary_monitor)
        self.window_width = video_mode.size.width
        self.window_height = video_mode.size.height
        self.frame_rate = video_mode.refresh_rate

        window = glfw.create_window(self.window_width, self.window_height, "Audio Visualizer", primary_monitor, None)
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
        self.stop_event.set()

    def stop(self):
        # Stop the thread by breaking the main loop
        self.stop_event.set()

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)
            self.stop()

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
