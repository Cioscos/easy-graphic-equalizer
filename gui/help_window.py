from PySide6.QtWidgets import QDialog, QLabel, QTextEdit, QVBoxLayout

from gui.theme import font_title

_HELP_TEXT = (
    "Hi! Welcome to my application!\n"
    "The first thing to do is to select a device from the device list on the left.\n"
    "In order to select it you have to douple click on one of them. After you'll see the green button activate, it means "
    "that now you can press it to start the graphic equalizer! Press it again to stop it.\n\n"
    "Settings\n\n"
    "Noise threshold: Slider to set how much the amplitude of the signal must be high to be drawn.\n"
    "Appearance mode: Select the theme.\n"
    "Background image path: Use it to change the default background image of the equalizator. You can both manually insert "
    "the path to the file or click the botton on its right to open a file picker window. Remember to press Apply button.\n"
    "Alpha amount: Use it to change the alpha of the background image. The lower the darker.\n"
    "Frequency bands: Use it to dinamically change the number of the bars showed. Take care that after some number of bars, "
    "according to your hardware it could become pretty laggy. For high number of bars is highly suggested the full screen visulization\n"
    "Fullscreen button: Open a fullscreen window in which the bars are drawn. This window has high performance so it's "
    "suggested when you use high number of bands.\n\n"
    "Advanced (Fullscreen only)\n\n"
    "These settings tune the OpenGL fullscreen renderer and have no effect on the in-window preview. "
    "You can change them live while fullscreen is running.\n"
    "dB floor / dB ceiling: The absolute loudness range mapped to bar height. Raise the floor (less negative) "
    "or lower the ceiling to make the bars fill up more easily; widen the range for a less sensitive display.\n"
    "Attack (ms): How quickly bars rise. Lower values snap to transients (kick/snare) almost instantly.\n"
    "Release (ms): How slowly bars fall. Higher values give a smoother, more readable peak-meter falloff.\n"
    "Tilt (dB/octave): Spectral emphasis. Negative values boost the lows, positive values boost the highs "
    "(e.g. +3 roughly compensates for pink-noise spectra). 0 leaves the response flat."
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
