import sys
import multiprocessing

from PySide6.QtWidgets import QApplication
import qdarktheme


def main() -> int:
    # IMPORTANT (Windows COM apartment ordering):
    # QApplication() calls OleInitialize(), which requires the GUI thread to be a
    # single-threaded apartment (STA). `soundcard` (imported transitively by the
    # GUI) initializes COM in *multi-threaded* apartment (MTA) mode on its import
    # thread. Whoever runs first wins the apartment, and Qt does NOT tolerate
    # finding MTA — it warns "OleInitialize() failed: RPC_E_CHANGED_MODE" and
    # native dialogs (QFileDialog) silently fall back to the non-native Qt ones.
    # soundcard, by contrast, *tolerates* COM already being STA and still works
    # (it calls WASAPI through direct cffi vtable calls, not marshaled proxies).
    # So: create the QApplication BEFORE importing the GUI/soundcard.
    app = QApplication(sys.argv)

    # Tema agganciato a quello di sistema (equivalente al 'System' di CustomTkinter).
    qdarktheme.setup_theme("auto")

    from gui.main_window_gui import AudioCaptureGUI
    gui = AudioCaptureGUI()
    gui.show()

    return app.exec()


if __name__ == '__main__':
    if sys.platform.startswith('win'):
        # On Windows calling this function is necessary.
        multiprocessing.freeze_support()

    sys.exit(main())
