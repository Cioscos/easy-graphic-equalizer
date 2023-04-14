import sys
import multiprocessing

from gui.main_window_gui import AudioCaptureGUI


if __name__ == '__main__':
    if sys.platform.startswith('win'):
        # On Windows calling this function is necessary.
        multiprocessing.freeze_support()

    gui = AudioCaptureGUI()
    gui.run()
