import pyqtgraph as pg


class CustomWindow(pg.GraphicsLayoutWidget):
    def __init__(self, audio_thread, parent=None):
        super().__init__(parent, show=True)
        self.audio_thread = audio_thread

    def closeEvent(self, event):
        print("Stopping audio capture thread...")
        self.audio_thread.stop()
        self.audio_thread.join()
        print("Exiting program...")
        event.accept()
