import threading
import math
import time
import numpy as np
import scipy.fftpack as fft_pack

FREQUENCY_BANDS = [
    (31.25, 62.5),
    (62.5, 125),
    (125, 250),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
    (8000, 16000)
]
RATE = 44100
FRAME_RATE = 60

class EqualizerTkinterThread(threading.Thread):
    def __init__(self, audio_queue, noise_threshold=0.1, channels=2, canvas=None):
        super().__init__()
        self.audio_queue = audio_queue
        self.noise_threshold = noise_threshold
        self.channels = channels
        self.canvas = canvas
        self.stop_event = threading.Event()

    def run(self):
        frame_duration = 1 / FRAME_RATE
        previous_time = time.time()

        # List of temporary amplitudes to create an animation effect
        previous_amplitudes = [0] * len(FREQUENCY_BANDS)

        while not self.stop_event.is_set():
            current_time = time.time()
            elapsed_time = current_time - previous_time

            # Consume data from the queue and update the audio buffer
            while not self.audio_queue.empty():
                audio_buffer = self.audio_queue.get(block=False)

            if elapsed_time >= frame_duration:
                self.canvas.delete("all")  # Clear the canvas

                if audio_buffer is not None:
                    amplitudes = self.create_equalizer(audio_buffer, FREQUENCY_BANDS, self.noise_threshold, self.channels)

                    # exponential smoothing
                    smooth_factor = 1 - math.exp(-elapsed_time / (frame_duration * 2))
                    # Render the bars
                    self.draw_bars(amplitudes, self.canvas, previous_amplitudes, smooth_factor=smooth_factor)

                previous_time = current_time

            # Sleep for a short duration to reduce CPU usage
            time.sleep(0.001)

    def create_equalizer(self, audio_data, frequency_bands, noise_threshold=0.1, channels=2):
        audio_samples = audio_data.T

        # Check that the number of channels in the input data matches the expected number of channels
        assert audio_samples.shape[0] == channels, f"Input data has {audio_samples.shape[0]} channels, expected {channels}"

        equalizer_data = []

        for channel_samples in audio_samples:
            # apply a window function to reduce leakage effect
            window = np.hanning(len(channel_samples))
            windowed_samples = channel_samples * window
            fft_values = np.abs(fft_pack.fft(windowed_samples))
            freqs = fft_pack.fftfreq(len(fft_values), 1.0 / RATE)

            band_amplitudes = []
            for low_freq, high_freq in frequency_bands:
                band_filter = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                # band_amplitude = np.mean(fft_values[band_filter])
                # band_amplitude = np.max(fft_values[band_filter])  # Change this line to use np.max instead of np.mean
                band_amplitude = np.mean(fft_values[band_filter]) * np.sum(band_filter)  # Multiply mean by the number of bins in the band
                band_amplitudes.append(band_amplitude)

            equalizer_data.append(band_amplitudes)

        avg_band_amplitudes = np.mean(equalizer_data, axis=0)
        filtered_amplitudes = [amp if amp >= noise_threshold else 0 for amp in avg_band_amplitudes]
        max_amplitude = max(filtered_amplitudes)
        epsilon = 1e-12  # Add a small value to prevent division by very small numbers
        normalized_amplitudes = [(amplitude / (max_amplitude + epsilon)) if max_amplitude > 0 else 0
                                for amplitude in filtered_amplitudes]

        return normalized_amplitudes
    
    def draw_bars(self, amplitudes, canvas, previous_amplitudes, smooth_factor=0.1):
        window_width = canvas.winfo_width()
        window_height = canvas.winfo_height()
        num_bars = len(amplitudes)
        total_bar_width = window_width / num_bars
        bar_width = total_bar_width * 0.8  # Each bar takes up 80% of its total width
        bar_spacing = total_bar_width * 0.2  # The remaining 20% is used for spacing between bars

        for i, amplitude in enumerate(amplitudes):
            # LERP between the previous amplitude and the new amplitude
            interpolated_amplitude = previous_amplitudes[i] + smooth_factor * (amplitude - previous_amplitudes[i])
            previous_amplitudes[i] = interpolated_amplitude  # Update the previous_amplitude

            x = i * total_bar_width + bar_spacing / 2
            y = window_height

            # Calculate the total height of the bar based on the amplitude
            total_height = interpolated_amplitude * window_height

            # Calculate the heights of the three color sections based on the total height
            green_height = min(total_height, 0.5 * window_height)
            yellow_height = min(max(total_height - green_height, 0), 0.3 * window_height)

            # Draw green section
            canvas.create_rectangle(x, y - green_height, x + bar_width, y, fill='green')

            # Draw yellow section
            canvas.create_rectangle(x, y - green_height - yellow_height, x + bar_width, y - green_height, fill='yellow')

            # Draw red section
            canvas.create_rectangle(x, y - total_height, x + bar_width, y - green_height - yellow_height, fill='red')

    def calculate_bar_sizes(self, window_sizes, num_bars):
        window_width, _ = window_sizes
        bar_total_width = window_width / num_bars
        bar_spacing = bar_total_width * 0.2
        bar_width = bar_total_width - bar_spacing
        return bar_width, bar_spacing
    
    def stop(self):
        # Stop the thread by breaking the main loop
        self.stop_event.set()