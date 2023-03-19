import queue
import numpy as np
import scipy.fftpack as fft_pack
import pyqtgraph as pg
import soundcard as sc
import glfw
from OpenGL.GL import *

from audioCaptureThread import AudioCaptureThread

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
MAX_QUEUE_SIZE = 200
FRAME_RATE = 60


def create_equalizer(audio_data, frequency_bands, noise_threshold=0.1, channels=2):
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

def draw_bars(amplitudes, bar_width, bar_spacing, window_width):
    num_bars = len(amplitudes)
    bar_total_width = bar_width + bar_spacing
    total_bars_width = num_bars * bar_total_width - bar_spacing
    offset_x = (window_width - total_bars_width) / 2

    for i, amplitude in enumerate(amplitudes):
        x = i * bar_total_width + bar_spacing / 2 + offset_x
        y = 0

        # Set color based on amplitude
        if amplitude <= 0.5:
            color = (0, 1, 0)  # Green
        elif amplitude <= 0.8:
            color = (1, 1, 0)  # Yellow
        else:
            color = (1, 0, 0)  # Red

        glColor3f(*color)
        glRectf(x, y, x + bar_width, y + amplitude * (600 / 2))


def show_equalizer(audio_queue, audio_thread, noise_threshold=0.1, channels=2):
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "Equalizer", None, None)
    if not window:
        glfw.terminate()
        return
    
    audio_buffer = None

    glfw.make_context_current(window)

    #glClearColor(0.8, 0.8, 0.8, 1)  # Light gray background

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 800, 0, 600, -1, 1)
    glMatrixMode(GL_MODELVIEW)

    frame_duration = 1 / FRAME_RATE
    previous_time = glfw.get_time()

    while not glfw.window_should_close(window):
        current_time = glfw.get_time()
        elapsed_time = current_time - previous_time

        # Consume data from the queue and update the audio buffer
        while not audio_queue.empty():
            audio_buffer = audio_queue.get(block=False)

        if elapsed_time >= frame_duration:
            glClear(GL_COLOR_BUFFER_BIT)

            if audio_buffer is not None:
                amplitudes = create_equalizer(audio_buffer, FREQUENCY_BANDS, noise_threshold, channels)

                # Render the bars
                draw_bars(amplitudes, window_width=800, bar_width=20, bar_spacing=10)

            glfw.swap_buffers(window)
            previous_time = current_time

        glfw.poll_events()

    # Close the window and stop the audio thread
    glfw.terminate()
    audio_thread.stop()


def main():
    devices = sc.all_microphones(include_loopback=True)
    for index, device in enumerate(devices):
        print(f"Device index: {index}, Device name: {device.name}")

    device_index = int(input('Choose one device typing the number: '))

    try:
        audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        audio_thread = AudioCaptureThread(audio_queue, device=devices[device_index])
        audio_thread.start()

        show_equalizer(audio_queue, audio_thread)
    except KeyboardInterrupt:  # Stop the program gracefully using Ctrl+C
        pass

if __name__ == '__main__':
    main()
