import numpy as np


class AudioBufferAccumulator:
    """
    Ring buffer numpy che mantiene sempre gli ultimi `window_size` frame
    multi-canale (es. 2 per stereo) ricevuti dalla cattura audio.

    A differenza della versione precedente basata su deque + overlap, qui non
    c'è alcun consumo: ogni chiamata a `get_window_for_fft` restituisce uno
    *snapshot* della finestra più recente senza scartare nulla. Questo permette
    al renderer OpenGL di ricalcolare l'FFT a ogni frame sull'audio più fresco
    possibile (latenza minima) e usa operazioni vettoriali invece di iterare in
    Python frame-per-frame (costo CPU ridotto).

    Attributes:
        window_size (int): Numero di frame di una finestra FFT (es. N_FFT).
        channels (int): Numero di canali audio (2 per stereo).
    """

    def __init__(self, window_size: int = 2048, channels: int = 2):
        """
        Inizializza il ring buffer.

        Args:
            window_size (int): Numero di frame di una singola finestra FFT.
            channels (int): Numero di canali audio (2 per stereo).
        """
        self.window_size = window_size
        self.channels = channels

        # Buffer circolare preallocato: (window_size, channels)
        self.buffer = np.zeros((window_size, channels), dtype=np.float32)
        # Scratch preallocato per lo snapshot riallineato (evita allocazioni per-frame)
        self._snapshot = np.zeros((window_size, channels), dtype=np.float32)
        self.write_pos = 0          # prossima posizione di scrittura
        self.total_written = 0      # frame totali scritti (per sapere se è "pieno")

    def add_samples(self, samples: np.ndarray) -> None:
        """
        Aggiunge nuovi frame multi-canale al ring buffer, gestendo il wrap.

        Args:
            samples (np.ndarray): Array di forma (chunk_size, channels).
                                  Per stereo, (chunk_size, 2).

        Raises:
            ValueError: Se 'samples' non ha il numero di canali atteso.
        """
        if samples.ndim != 2 or samples.shape[1] != self.channels:
            raise ValueError(
                f"Expected samples with shape (chunk_size, {self.channels}), "
                f"but got {samples.shape}."
            )

        n = samples.shape[0]

        # Se il chunk è grande quanto/più del buffer, basta tenerne la coda
        if n >= self.window_size:
            self.buffer[:] = samples[-self.window_size:]
            self.write_pos = 0
        else:
            end = self.write_pos + n
            if end <= self.window_size:
                self.buffer[self.write_pos:end] = samples
            else:
                # Wrap: spezza la scrittura in due tratti
                first = self.window_size - self.write_pos
                self.buffer[self.write_pos:] = samples[:first]
                self.buffer[:n - first] = samples[first:]
            self.write_pos = end % self.window_size

        self.total_written += n

    def get_window_for_fft(self) -> np.ndarray:
        """
        Restituisce uno snapshot degli ultimi `window_size` frame, in ordine
        cronologico, di forma (window_size, channels). Non scarta nulla.

        Returns:
            np.ndarray: Array (window_size, channels), oppure un array vuoto se
                        non sono ancora arrivati abbastanza frame.

        NB (contratto): prima che il buffer sia pieno ritorna np.array([])
        (shape (0,), float64), NON una matrice (0, channels) float32; e quando
        write_pos == 0 ritorna il buffer VIVO aliasato, valido solo fino alla
        prossima add_samples. Sicuro nell'uso attuale (lettura e scrittura sul
        solo thread GL, snapshot consumato subito); qualunque nuovo chiamante
        deve rispettare entrambe le cose.
        """
        if self.total_written < self.window_size:
            # Non ancora abbastanza frame per una finestra completa
            return np.array([])

        if self.write_pos == 0:
            # Il frame più vecchio è già all'indice 0: buffer già in ordine cronologico.
            # Restituiamo direttamente il buffer (nessuna copia): lettura e scrittura
            # avvengono nello stesso thread e lo snapshot è consumato dal renderer prima
            # del successivo add_samples.
            return self.buffer

        # Riallinea in ordine cronologico dentro uno scratch preallocato (nessuna
        # allocazione per-frame): da write_pos (più vecchio) fino alla fine, poi
        # dall'inizio fino a write_pos (più recente).
        tail = self.window_size - self.write_pos
        self._snapshot[:tail] = self.buffer[self.write_pos:]
        self._snapshot[tail:] = self.buffer[:self.write_pos]
        return self._snapshot
