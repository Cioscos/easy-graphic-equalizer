import queue
import signal
import sys
import numpy as np
import scipy.fftpack as fft_pack
import pyqtgraph as pg
import soundcard as sc
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QSurfaceFormat

#format = QSurfaceFormat()
#format.setSwapInterval(1)
#QSurfaceFormat.setDefaultFormat(format)

from audioCaptureThread import AudioCaptureThread
from equalizer_window import CustomWindow
from custom_pyqt_item import ColoredBarGraphItem

FREQUENCY_BANDS = [
    (31.25, 62.5),
    (62.5, 125),
    (125, 250),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
    (8000, 16000),
    (16000, 20000)
]
RATE = 44100
MAX_QUEUE_SIZE = 200


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

def show_equalizer(audio_queue, audio_thread, noise_threshold=0.1, channels=2, animation_speed=0.05):
    app = QApplication([])
    win = CustomWindow(audio_thread)
    win.setWindowTitle('Real-time equalizer')
    num_bands = len(FREQUENCY_BANDS)
    
    plot = win.addPlot(title="Equalizer")
    plot.setLabels(left='Amplitude', bottom='Frequency Band (Hz)')
    plot.setLimits(yMin=0, yMax=1)
    plot.setYRange(0, 1)
    plot.setAspectLocked(False)
    plot.setXRange(-1, num_bands + 1)
    plot.hideAxis('left')
    #bars = pg.BarGraphItem(height=[0]*num_bands, x=range(num_bands), width=0.6)
    bars = ColoredBarGraphItem(height=[0]*num_bands, x=range(num_bands), width=0.9)
    plot.addItem(bars)
    tick_labels = {i: f"{int(start)}-{int(end)}" for i, (start, end) in enumerate(FREQUENCY_BANDS)}
    plot.getAxis('bottom').setTicks([tick_labels.items()])

    current_amplitudes = [0] * num_bands

    def update():
        nonlocal current_amplitudes
        audio_data = audio_queue.get()
        amplitudes = create_equalizer(audio_data, FREQUENCY_BANDS, noise_threshold=noise_threshold, channels=channels)
        current_amplitudes = [current_amplitude + animation_speed * (target_amplitude - current_amplitude)
                            for current_amplitude, target_amplitude in zip(current_amplitudes, amplitudes)]
        
        #print('amplitudes: ', amplitudes, 'current_amplitudes: ', current_amplitudes)
        bars.setOpts(height=current_amplitudes)
        plot.update()

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    # Update every 15 milliseconds
    timer.start(15)

    def signal_handler(signal, frame):
        print("Stopping audio capture thread...")
        audio_thread.stop()
        audio_thread.join()
        print("Exiting program...")
        app.quit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    app.exec_()
    sys.exit(0)


def main():
    devices = sc.all_microphones(include_loopback=True)
    for index, device in enumerate(devices):
        print(f"Device index: {index}, Device name: {device.name}")

    device_index = int(input('Choose one device typing the number: '))

    try:
        audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        audio_thread = AudioCaptureThread(audio_queue, device=devices[device_index])
        audio_thread.start()
        show_equalizer(audio_queue, audio_thread, noise_threshold=0.2, channels=2, animation_speed=0.50)
    except KeyboardInterrupt:  # Stop the program gracefully using Ctrl+C
        pass

if __name__ == '__main__':
    main()
