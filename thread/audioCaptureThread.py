import threading
import queue
import warnings

# Audio parameters
CHUNK = 1024
CHANNELS = 2 # stereo
RATE = 44100

# Define a thread for capturing audio
class AudioCaptureThread(threading.Thread):
    def __init__(self, audio_queue, device, debug=False, channels=2):
        threading.Thread.__init__(self)
        self._running = True
        self.device = device
        self.audio_queue = audio_queue
        self._debug = debug
        self.channels = channels
        if self._debug:
            print("Audio capture thread created")

    def run(self):
        if self._debug:
            print("Audio capture thread run() called")

        # Le catture in loopback su WASAPI marcano discontinuita in modo
        # fisiologico (riprese dopo il silenzio, glitch del flusso di rendering):
        # soundcard le emette come warning a ogni occorrenza tramite
        # simplefilter('always'). Sono benigni — i dati restano validi — quindi
        # silenziamo solo questo messaggio. Va registrato qui (a thread avviato,
        # dopo l'import di soundcard) perche' i filtri inseriti per ultimi hanno
        # precedenza: cosi' il nostro 'ignore' vince sull''always' di soundcard.
        # Filtro per messaggio (non per categoria) cosi' altri SoundcardRuntimeWarning
        # restano visibili, ed e' cross-platform (niente import della classe).
        warnings.filterwarnings("ignore", message="data discontinuity in recording")

        with self.device.recorder(samplerate=RATE, channels=self.channels) as recorder:
            while self._running:
                try:
                    if self._debug:
                        print("Capturing audio data...")
                    data = recorder.record(numframes=CHUNK)
                    if self._debug:
                        print(f"Captured {len(data)} samples of audio data")
                    # Put non bloccante: se la coda e' piena (consumer momentaneamente
                    # fermo, es. init OpenGL) scartiamo il frame piu' vecchio e inseriamo
                    # il piu' recente, invece di bloccare record() — un blocco starverebbe
                    # il buffer WASAPI e fabbricherebbe discontinuita' reali. Coerente con
                    # i consumer che tengono comunque solo l'ultimo frame.
                    try:
                        self.audio_queue.put_nowait(data)
                    except queue.Full:
                        try:
                            self.audio_queue.get_nowait()   # scarta il frame stantio
                        except queue.Empty:
                            pass
                        try:
                            self.audio_queue.put_nowait(data)
                        except queue.Full:
                            pass
                except Exception as exception:
                    print(f"Error capturing audio: {exception}")
                    break

        if self._debug:
            print("Loop exited")

    def stop(self):
        self._running = False
