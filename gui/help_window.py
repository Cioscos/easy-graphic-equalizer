from PySide6.QtWidgets import QDialog, QLabel, QTextEdit, QVBoxLayout

from gui.theme import font_title

_HELP_TEXT = (
    "Hi! Welcome to my application!\n"
    "The first thing to do is to select a device from the device list at the top.\n"
    "A single click selects it. The 'Avvia a schermo intero' button then turns green: press it to open the "
    "fullscreen visualizer. Press it again (or Esc in the fullscreen window) to stop it.\n\n"
    "Settings\n\n"
    "Noise threshold: Slider to set how much the amplitude of the signal must be high to be drawn.\n"
    "Theme: choose Light / Dark / System from the 'Tema' menu at the top.\n"
    "Background image path: Use it to change the default background image of the equalizator. You can both manually insert "
    "the path to the file or click the botton on its right to open a file picker window. Remember to press Apply button.\n"
    "Alpha amount: Use it to change the alpha of the background image. The lower the darker.\n"
    "Frequency bands: Use it to dinamically change the number of the bars showed. Take care that after some number of bars, "
    "according to your hardware it could become pretty laggy. For high number of bars is highly suggested the full screen visulization\n"
    "Avvia a schermo intero: opens the fullscreen OpenGL window where the visualization is drawn. It is the only "
    "renderer, GPU-accelerated, and handles high band counts well.\n\n"
    "Advanced (Fullscreen only)\n\n"
    "These settings tune the OpenGL fullscreen renderer. "
    "You can change them live while fullscreen is running.\n"
    "dB floor / dB ceiling: The absolute loudness range mapped to bar height. Raise the floor (less negative) "
    "or lower the ceiling to make the bars fill up more easily; widen the range for a less sensitive display.\n"
    "Attack (ms): How quickly bars rise. Lower values snap to transients (kick/snare) almost instantly.\n"
    "Release (ms): How slowly bars fall. Higher values give a smoother, more readable peak-meter falloff.\n"
    "Tilt (dB/octave): Spectral emphasis. Negative values boost the lows, positive values boost the highs "
    "(e.g. +3 roughly compensates for pink-noise spectra). 0 leaves the response flat.\n\n"
    "Bars (Fullscreen only)\n\n"
    "Color mode: Classic (green/yellow/red with adjustable yellow/red thresholds), Solid colour, "
    "vertical Gradient (base to tip), or Spectrum (hue follows each bar's frequency).\n"
    "Bar width: From thin, airy bars to thick, packed ones.\n"
    "Rounded caps: Smooth the top of each bar.\n"
    "Anchor: Grow bars from the bottom, or mirrored from a centre line.\n"
    "Band order: Standard, or Symmetric with the bass in the centre mirrored out to the edges."
)


class HelpWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Help")
        self.resize(500, 500)

        layout = QVBoxLayout(self)

        label = QLabel("How to use")
        label.setFont(font_title())
        layout.addWidget(label)

        self.explaination = QTextEdit()
        self.explaination.setReadOnly(True)
        self.explaination.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.explaination.setPlainText(_HELP_TEXT)
        layout.addWidget(self.explaination)
