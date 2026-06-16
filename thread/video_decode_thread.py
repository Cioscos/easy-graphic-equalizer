import threading
import queue
from typing import Optional

import numpy as np
import imageio.v2 as imageio


class VideoDecodeThread(threading.Thread):
    """
    Thread produttore che decodifica i frame di un video in loop e li pubblica in
    una coda, modellato sul pattern produttore/consumer di AudioCaptureThread.

    La decodifica (CPU, via backend ffmpeg di imageio) gira QUI; l'upload della
    texture GPU resta sul thread OpenGL, dove il contesto GL è valido. La coda è
    volutamente corta (maxsize piccolo): il consumer drena e tiene solo il frame
    più fresco, quindi non serve accumulare backlog (stessa filosofia dell'audio).

    Ogni frame pubblicato è un np.ndarray di shape (H, W, 3), dtype uint8, in
    ordine RGB con la riga 0 in alto (il flip verticale per OpenGL è gestito dallo
    shader di sfondo via uniform uFlipV, non qui).

    Attributes:
        fps: frame rate del video letto dai metadati (default 30.0 finché ignoto).
             Letto live dal thread OpenGL per temporizzare l'avanzamento.
    """

    def __init__(self, path: str, frame_queue: queue.Queue, debug: bool = False):
        super().__init__(daemon=True)
        self._path = path
        self._frame_queue = frame_queue
        self._debug = debug
        self.stop_event = threading.Event()
        # Default prudente finché non leggiamo i metadati reali del video.
        self.fps = 30.0
        if self._debug:
            print(f"Video decode thread created for {path}")

    def run(self):
        """
        Apre il video e ne pubblica i frame in loop infinito (riavvolgendo a fine
        file) finché stop_event non è impostato. Il reader viene SEMPRE chiuso nel
        finally per rilasciare il subprocess ffmpeg (niente processi orfani).
        """
        if self._debug:
            print("Video decode thread run() called")
        try:
            while not self.stop_event.is_set():
                reader = imageio.get_reader(self._path)
                try:
                    meta = reader.get_meta_data()
                    fps = meta.get('fps')
                    if fps and fps > 0:
                        self.fps = float(fps)
                    if self._debug:
                        print(f"Video fps={self.fps}, meta={meta}")

                    for frame in reader:
                        if self.stop_event.is_set():
                            break
                        # imageio/ffmpeg restituisce già RGB uint8; garantiamo
                        # contiguità per un upload GL diretto (glTexImage2D).
                        if frame.dtype != np.uint8 or not frame.flags["C_CONTIGUOUS"]:
                            frame = np.ascontiguousarray(frame, dtype=np.uint8)
                        self._put_frame(frame)
                finally:
                    try:
                        reader.close()
                    except Exception:
                        pass
                # Fine stream: il while esterno ricrea il reader e riparte (loop).
        except Exception as exc:
            # Path/codec non leggibile o errore di decodifica: termina il thread
            # senza far cadere l'app (il renderer continua senza sfondo video).
            print(f"Video decode thread error: {exc}")

        if self._debug:
            print("Video decode loop exited")

    def _put_frame(self, frame: np.ndarray) -> None:
        """
        Inserisce un frame nella coda bloccando se piena, ma restando reattivo allo
        stop: il pacing temporale (a quale velocità consumare) è interamente sul
        lato OpenGL, qui ci limitiamo a riempire quando c'è posto.
        """
        while not self.stop_event.is_set():
            try:
                self._frame_queue.put(frame, timeout=0.1)
                return
            except queue.Full:
                continue

    def stop(self):
        """Ferma il thread impostando l'evento di stop."""
        self.stop_event.set()
