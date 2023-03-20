import threading
import queue
from time import sleep
import pyaudiowpatch as pyaudio

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2 # stereo
RATE = 44100

# Define a thread for capturing audio
class AudioCaptureThread(threading.Thread):
    def __init__(self, audio_queue, device, debug=False, defaultSampleRate=RATE, channels=2):
        threading.Thread.__init__(self)
        self._running = True
        #self.p = pyaudio.PyAudio()
        self.device = device
        self.audio_queue = audio_queue
        self._debug = debug
        self.channels = channels
        self._defaultSampleRate = defaultSampleRate
        if self._debug:
            print("Audio capture thread created")

    def run(self):
        if self._debug:
            print("Audio capture thread run() called")
        
        with self.device.recorder(samplerate=RATE, channels=self.channels) as recorder:
            while self._running:
                try:
                    if self._debug:
                        print("Capturing audio data...")
                    data = recorder.record(numframes=CHUNK)
                    if self._debug:
                        print(f"Captured {len(data)} samples of audio data")
                    self.audio_queue.put(data, block=True, timeout=1)
                except queue.Full:
                    pass
                except Exception as exception:
                    print(f"Error capturing audio: {exception}")
                    break

        if self._debug:
            print("Loop exited")

    def stop(self):
        self._running = False
