import numpy as np
from collections import deque

class AudioBufferAccumulator:
    """
    Accumulates multi-channel audio (e.g., 2 for stereo) from incoming chunks
    and provides a method to retrieve a 2D window of samples (window_size x channels)
    for FFT analysis, with overlap.

    Attributes:
        window_size (int): Number of frames for one FFT window (e.g., 2048).
        overlap_size (int): Number of frames that will overlap between consecutive windows.
        channels (int): Number of audio channels (e.g., 2 for stereo).
    """

    def __init__(self, window_size: int = 2048, overlap_size: int = 1024, channels: int = 2):
        """
        Initializes the accumulator.

        Args:
            window_size (int): The number of frames for a single FFT window (e.g., 2048).
            overlap_size (int): The number of frames to keep in the buffer for overlap (e.g., 1024).
            channels (int): The number of audio channels (2 for stereo).
        """
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.channels = channels

        # We'll store up to 'window_size' rows, each row shape = (channels,)
        self.buffer = deque(maxlen=window_size)

    def add_samples(self, samples: np.ndarray) -> None:
        """
        Add new multi-channel samples to the accumulator buffer.

        Args:
            samples (np.ndarray): A NumPy array of shape (chunk_size, channels).
                                  For stereo, (chunk_size, 2).

        Raises:
            ValueError: If 'samples' doesn't have the expected number of channels.
        """
        if samples.ndim != 2 or samples.shape[1] != self.channels:
            raise ValueError(
                f"Expected samples with shape (chunk_size, {self.channels}), "
                f"but got {samples.shape}."
            )

        # Append each frame (row) into the buffer
        for frame in samples:
            self.buffer.append(frame)

    def get_window_for_fft(self) -> np.ndarray:
        """
        Retrieves a 2D window of shape (window_size, channels) for FFT analysis,
        if enough frames are available. Also enforces overlap by discarding only
        'window_size - overlap_size' frames after extraction.

        Returns:
            np.ndarray: An array of shape (window_size, channels).
                        Returns an empty array if there are not enough frames.
        """
        if len(self.buffer) < self.window_size:
            # Not enough frames yet
            return np.array([])

        # Convert the deque to a NumPy array
        # Shape: (window_size, channels)
        window_data = np.array(self.buffer, dtype=np.float32)

        # Discard only the oldest frames that exceed the overlap
        num_to_discard = self.window_size - self.overlap_size
        for _ in range(num_to_discard):
            if self.buffer:
                self.buffer.popleft()

        return window_data
