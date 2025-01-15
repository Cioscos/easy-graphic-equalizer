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

from thread.AudioBufferAccumulator import AudioBufferAccumulator

RATE = 44100
# N_FFT= 32768
N_FFT = 2048
MIN_FREQ = 20
MAX_FREQ = 20000


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
    # Apply a Hanning window
    window = np.hanning(len(channel_samples))
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
    using np.mean for each band filter, then applies 'volume' scaling.
    """
    fft_values, freqs, bands, volume = args
    band_amplitudes = []

    for low_freq, high_freq in bands:
        band_mask = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        if np.any(band_mask):
            # Use mean for a smoother, averaged amplitude
            band_amplitude = 0.7 * np.mean(fft_values[band_mask]) + 0.3 * np.max(fft_values[band_mask])
        else:
            band_amplitude = 0.0

        # Scale by volume (RMS of the input)
        band_amplitude *= volume
        band_amplitudes.append(band_amplitude)

    return band_amplitudes

def compute_channel_fft(args):
    """
    compute_channel_fft is used in parallel or single-worker mode
    to compute the FFT for each channel, then process frequency bands.

    Steps:
        1) For each channel in audio_samples, call compute_fft (using the Hanning window).
        2) Filter to the positive half of freqs if desired (shown below).
        3) Use process_frequency_bands(...) to get band amplitudes for each channel.

    Returns:
        list of lists: band_amplitude_chunks, where each sub-list corresponds
                      to one channel's band amplitudes.
    """
    audio_samples, dummy_padded_samples, volume, frequency_band_chunks = args

    # Compute FFT for all channels
    fft_magnitudes_list = []
    for channel_samples in audio_samples:
        fft_magnitudes = compute_fft(channel_samples)
        fft_magnitudes_list.append(fft_magnitudes)

    # Frequencies array (same length as the final FFT array)
    freqs_full = fft_pack.fftfreq(len(fft_magnitudes_list[0]), 1.0 / RATE)

    # Filter only the positive half if you prefer
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
                (fft_values, freqs_pos, frequency_band_chunks, volume)
            )
        )

    return band_amplitude_chunks

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
        self.frequency_bands = self.generate_bin_based_bands(self._n_bands)

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

        # ---------------------------------------------------
        # Instantiate the AudioBufferAccumulator for overlap
        # window_size = N_FFT, overlap_size = N_FFT//2 (50%).
        # ---------------------------------------------------
        self.audio_accumulator = AudioBufferAccumulator(window_size=N_FFT,
                                                        overlap_size=N_FFT // 2,
                                                        channels=self._channels)

        # ----------------------------------------------------------
        # NEW: Throttle parameters
        # ----------------------------------------------------------
        self.fft_interval = 0.02  # e.g., 40 ms between FFT computations (~25 FPS in the audio domain)
        self.last_fft_time = 0.0

        # Keep track of the latest amplitudes we computed
        # We will draw these at every render frame, but they only update after each FFT.
        self.latest_amplitudes = [0.0] * self._n_bands

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

    def generate_bin_based_bands(self, num_bands: int) -> list[tuple[float, float]]:
        """
        Generate frequency bands by splitting the actual positive FFT bin frequencies
        (in the range [MIN_FREQ, MAX_FREQ]) into 'num_bands' groups. We do this in
        approximate log space, ensuring each group has at least one bin.

        Returns:
            list of (float, float): A list of (low_freq, high_freq) tuples, where each
                                    tuple covers a contiguous slice of the FFT bins.
        """

        # 1. Compute all FFT bin frequencies for N_FFT.
        #    'fftpack.fftfreq' returns negative freqs as well; we only keep the positive half.
        freqs = fft_pack.fftfreq(N_FFT, d=1.0 / RATE)
        half_n = N_FFT // 2  # Index for Nyquist
        positive_freqs = freqs[:half_n + 1]  # shape: (N_FFT/2 + 1,)

        # 2. Exclude freq=0 (DC) if desired, or treat it as its own band.
        #    We'll skip it here for a purely log-based distribution.
        #    So we'll start from index 1 to half_n (inclusive).
        #    Then also mask out anything above MAX_FREQ or below MIN_FREQ.
        valid_mask = (positive_freqs >= MIN_FREQ) & (positive_freqs <= MAX_FREQ)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            # If no bins in [MIN_FREQ, MAX_FREQ], return an empty list
            return []

        # 3. Extract the valid frequencies (>= MIN_FREQ and <= MAX_FREQ)
        valid_freqs = positive_freqs[valid_indices]  # shape: (some_number_of_bins,)
        # We'll convert them to log space to do an approximate log-split.
        log_valid_freqs = np.log10(valid_freqs)

        # 4. Create 'num_bands' intervals in log space between min(valid) and max(valid).
        #    We do num_bands slices => num_bands + 1 edges.
        min_log = log_valid_freqs[0]  # log10 of the lowest bin >= MIN_FREQ
        max_log = log_valid_freqs[-1]  # log10 of the highest bin <= MAX_FREQ
        if num_bands > (len(valid_freqs)):
            # If you request more bands than bins, you'll end up with some 1-bin or 0-bin groups.
            # We'll still do it, but you'll see small or empty groups.
            pass

        log_edges = np.linspace(min_log, max_log, num_bands + 1)  # shape: (num_bands+1,)

        # 5. Bin-based grouping: for each interval [log_edges[i], log_edges[i+1]),
        #    gather the bin indices that fall in that log range.
        bands = []
        idx_start = 0  # We'll walk through valid_freqs in ascending order
        for i in range(num_bands):
            low_edge = log_edges[i]
            high_edge = log_edges[i + 1]

            # Identify the subset of valid_freqs in that log range
            # We do ">= low_edge" and "< high_edge" so that the last band might
            # absorb the top edge if inclusive is desired. Adjust as needed.
            band_mask = (log_valid_freqs >= low_edge) & (log_valid_freqs < high_edge)
            band_indices = np.where(band_mask)[0]

            if len(band_indices) == 0:
                # No bins in this interval => we could skip or attempt a merge.
                # We'll simply skip here, or you could store a placeholder band.
                continue

            # The actual frequencies in that group
            group_freqs = valid_freqs[band_indices]

            # The band spans from the min to max freq in that group
            band_low_freq = group_freqs[0]
            band_high_freq = group_freqs[-1]

            bands.append((float(band_low_freq), float(band_high_freq)))

        return bands

    def create_equalizer(self,
                         audio_data: np.ndarray,
                         frequency_bands: list[tuple[float, float]],
                         noise_threshold: float = 0.1,
                         channels: int = 2) -> list[float]:
        """
        Compute per-band amplitudes for a block of multi-channel audio data.

        1) Prepare 'audio_samples' (shape = (channels, N_FFT)) from 'audio_data' (shape=(N_FFT, channels)).
        2) If self._workers > 1, split the bands and compute channel FFT in parallel via 'compute_channel_fft'.
           Otherwise, do a single-thread fallback with an internal FFT approach.
        3) Merge results across channels, apply noise threshold.
        4) Convert to dB scale, then normalize to [0..1].

        Args:
            audio_data (np.ndarray): shape (N_FFT, channels).
            frequency_bands (list): list of (low_freq, high_freq) tuples.
            noise_threshold (float): linear noise threshold (applied before dB scaling).
            channels (int): number of channels (2 for stereo).

        Returns:
            list[float]: normalized amplitude for each band in [0..1].
        """
        # Transpose so shape = (channels, N_FFT)
        audio_samples = audio_data.T
        assert audio_samples.shape[0] == channels, (
            f"Input data has {audio_samples.shape[0]} channels, expected {channels}"
        )

        # Overall volume of the input (RMS)
        volume = np.sqrt(np.mean(audio_data ** 2))

        # Decide whether to use parallel approach or single-thread fallback
        if self._workers > 1:
            # -------------------------------------------------------------
            # PARALLEL BRANCH
            # -------------------------------------------------------------
            # 1) Split frequency_bands into chunks (roughly one per worker, or adapt logic)
            bands_per_worker = len(frequency_bands) // self._workers
            # Edge case: if bands_per_worker is 0, every chunk will be empty except the first.
            # We'll handle that by ensuring we do at least 1 band per chunk.
            if bands_per_worker < 1:
                bands_per_worker = 1

            frequency_band_chunks = []
            start_index = 0
            while start_index < len(frequency_bands):
                end_index = start_index + bands_per_worker
                # slice the bands
                sub_bands = frequency_bands[start_index:end_index]
                frequency_band_chunks.append(sub_bands)
                start_index = end_index

            # 2) We'll pass 'audio_samples' + a dummy array for padded_samples
            #    to maintain the 'compute_channel_fft' signature.
            #    Since we're applying the Hanning and zero-padding inside 'compute_fft',
            #    we can pass a dummy np.zeros(N_FFT) or something similar.
            padded_samples = np.zeros(N_FFT, dtype=np.float32)

            # 3) Construct the arguments for each chunk
            parallel_args = []
            for chunk in frequency_band_chunks:
                # (audio_samples, padded_samples, volume, chunk)
                parallel_args.append((audio_samples, padded_samples, volume, chunk))

            # 4) Use our executor to parallelize
            band_amplitude_chunks = list(
                self.executor.map(
                    compute_channel_fft,
                    parallel_args
                )
            )

            # 'band_amplitude_chunks' is a list of results,
            # each result is: list of [ (band_amplitudes_per_channel) ... ]
            #
            # So if we had M chunks, we get M results,
            # where each result = [ [band_amps for chunk], [band_amps for chunk], ... ] for each channel

            # We'll combine them per channel
            # equalizer_data = list of channels, each being a list of band amplitudes
            equalizer_data = [[] for _ in range(channels)]

            # Flatten each chunk
            for chunk_result in band_amplitude_chunks:
                # chunk_result is something like:
                #   [ [band_amps_for_channel0], [band_amps_for_channel1], ... ]
                # We need to extend them in 'equalizer_data'
                for ch_index, ch_band_amps in enumerate(chunk_result):
                    # 'ch_band_amps' is a list of amplitude values for each band in that chunk
                    # Extend our final list
                    equalizer_data[ch_index].extend(ch_band_amps)

        else:
            # -------------------------------------------------------------
            # SINGLE-THREAD FALLBACK
            # -------------------------------------------------------------
            equalizer_data = []

            for channel_samples in audio_samples:
                # We apply the Hanning window
                window = np.hanning(len(channel_samples))
                windowed_samples = channel_samples * window

                # Zero-pad if needed
                if len(channel_samples) < N_FFT:
                    padded = np.zeros(N_FFT, dtype=windowed_samples.dtype)
                    padded[:len(channel_samples)] = windowed_samples
                    fft_values = np.abs(np.fft.fft(padded))
                else:
                    fft_values = np.abs(np.fft.fft(windowed_samples))

                # Keep only positive freq half
                freqs_full = fft_pack.fftfreq(len(fft_values), 1.0 / RATE)
                pos_mask = (freqs_full >= 0)
                fft_values = fft_values[pos_mask]
                freqs = freqs_full[pos_mask]

                # For each band, compute MEAN amplitude of the bins, scaled by volume
                band_amplitudes = []
                for (low_freq, high_freq) in frequency_bands:
                    band_filter = (freqs >= low_freq) & (freqs <= high_freq)
                    amp = np.mean(fft_values[band_filter]) if np.any(band_filter) else 0
                    amp *= volume
                    band_amplitudes.append(amp)

                equalizer_data.append(band_amplitudes)

        # -------------------------------------------------------------
        # Merge channels => average band amplitudes across channels
        # -------------------------------------------------------------
        # 'equalizer_data' is shape (channels, total_num_bands).
        avg_band_amplitudes = np.mean(equalizer_data, axis=0)

        # 1) Linear noise threshold
        filtered_amplitudes = [
            amp if amp >= noise_threshold else 0
            for amp in avg_band_amplitudes
        ]

        # 2) Convert to dB scale
        epsilon = 1e-12
        dB_values = [
            20.0 * np.log10(amp + epsilon) for amp in filtered_amplitudes
        ]

        # 3) Find min and max
        min_db = min(dB_values)
        max_db = max(dB_values)
        db_range = max_db - min_db

        # 4) Normalize dB => [0..1]
        if db_range < 1e-9:
            normalized_amplitudes = [0.0 for _ in dB_values]
        else:
            normalized_amplitudes = [
                (db - min_db) / db_range
                for db in dB_values
            ]

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
            self.frequency_bands = self.generate_bin_based_bands(message["value"])
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

        # -------------------------
        # MAIN LOOP
        # -------------------------
        while not glfw.window_should_close(window) and not self.stop_event.is_set():
            # Process messages from the control queue
            while not self._control_queue.empty():
                message = self._control_queue.get(block=False)
                self.process_control_message(message)

            current_time = time.time()
            elapsed_time = current_time - previous_time

            # 1) Pull all available chunks from the queue
            while not self._audio_queue.empty():
                audio_buffer = self._audio_queue.get(block=False)
                # audio_buffer shape: (CHUNK, channels) -> e.g., (1024, 2) if stereo
                self.audio_accumulator.add_samples(audio_buffer)

            # 2) Throttle the FFT
            #    Only compute a new FFT if 'fft_interval' time has passed.
            if (current_time - self.last_fft_time) >= self.fft_interval:
                # Attempt to retrieve enough samples for a new FFT
                fft_window = self.audio_accumulator.get_window_for_fft()
                if fft_window.size > 0:
                    # shape: (N_FFT, channels)
                    amplitudes = self.create_equalizer(
                        audio_data=fft_window,
                        frequency_bands=self.frequency_bands,
                        noise_threshold=self._noise_threshold,
                        channels=self._channels
                    )
                    self.latest_amplitudes = amplitudes
                self.last_fft_time = current_time

            # 3) Exponential smoothing (optional)
            #    You might still do some smoothing each frame if you want
            smooth_factor = 1 - math.exp(-elapsed_time / (frame_duration * 2))

            # 4) Draw bars, always using self.latest_amplitudes
            self.draw_bars(self.latest_amplitudes, smooth_factor)

            previous_time = current_time
            glfw.swap_buffers(window)
            glfw.poll_events()

        # Cleanup
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
