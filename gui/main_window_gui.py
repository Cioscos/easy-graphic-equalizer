import os
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor

import soundcard as sc
from PIL import Image
import glfw
import qdarktheme

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from thread.audioCaptureThread import AudioCaptureThread
from thread.equalizer_preview_worker import EqualizerPreviewWorker
from thread.opengl_thread import (
    EqualizerOpenGLThread,
    DB_FLOOR, DB_CEILING, ATTACK_TAU, RELEASE_TAU, TILT_DB_PER_OCT,
)
from gui.slider_frame import SliderCustomFrame
from gui.optionmenu_frame import OptionMenuCustomFrame
from gui.background_filepicker_frame import BackgroundFilepickerFrame
from gui.help_window import HelpWindow
from gui.animated_button import AnimatedButton
from gui.equalizer_preview_widget import EqualizerPreviewWidget
from gui.theme import (
    font_title, font_button, font_label, font_body,
    COLOR_NEUTRAL, COLOR_NEUTRAL_HOVER,
    COLOR_FULLSCREEN, COLOR_FULLSCREEN_HOVER,
    COLOR_START, COLOR_START_HOVER,
    COLOR_STOP, COLOR_STOP_HOVER,
)
from resource_manager import ResourceManager

# Coda audio volutamente piccola: il renderer svuota tutta la coda a ogni frame
# tenendo solo l'audio più recente, quindi una coda corta evita backlog/latenza.
MAX_QUEUE_SIZE = 8
INITIAL_FREQUENCIES_BANDS = 9
# Estensioni trattate come video (sfondo animato, solo renderer OpenGL).
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
DEFAULT_BG_IMAGE = 'bg (19).png'

# Mappa il tema esposto nell'interfaccia alla modalità di qdarktheme.
_THEME_MODE = {"Light": "light", "Dark": "dark", "System": "auto"}


class AudioCaptureGUI(QMainWindow):
    """
    Finestra principale (PySide6).

    Possiede e coordina i thread/le code della pipeline produttore/consumatore.
    L'anteprima in-window è disegnata da EqualizerPreviewWidget, alimentato da
    EqualizerPreviewWorker su un QThread tramite il segnale bands_ready.
    """

    # Emesso quando la finestra OpenGL si chiude: il callback parte sul thread
    # GLFW, il segnale lo marshalla sul thread GUI (connessione queued).
    _opengl_closed = Signal()

    def __init__(self):
        super().__init__()

        self.resource_manager = ResourceManager()
        self.setWindowTitle("Sound wave")
        self.setMinimumSize(800, 600)

        # Stato condiviso inizializzato prima della costruzione dei widget
        self.devices = []
        self.bg_alpha = 0.5
        self.bars_alpha = 1.0  # opacità delle barre nel renderer OpenGL (1.0 = piene)
        self.bg_video_path = None  # path del video di sfondo selezionato (None = immagine)
        self.selected_monitor = None
        self.available_monitors = self.get_available_monitors()

        # Stato runtime (code, thread, flag)
        self.audio_queue = None
        self.equalizer_control_queue = queue.Queue()
        self.audio_thread = None
        self.preview_thread = None      # QThread che ospita il worker DSP
        self.preview_worker = None      # EqualizerPreviewWorker
        self.equalizer_opengl_thread = None
        self.last_device_selected = None
        self.help_window = None
        self.is_capturing = False

        # Immagine di sfondo di default (PIL): usata dal renderer OpenGL.
        self._default_bg_path = str(self.resource_manager.get_image_path(DEFAULT_BG_IMAGE, 'bg'))
        self.bg_img = Image.open(self._default_bg_path)

        # Costruzione UI
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.addWidget(self._build_left_panel(), 0)
        root_layout.addWidget(self._build_right_panel(), 1)

        # Sfondo di default sull'anteprima
        self.equalizer_preview.set_background_image(self._default_bg_path)
        self.equalizer_preview.set_background_alpha(self.bg_alpha)

        self._opengl_closed.connect(self._on_opengl_closed)

    # --- Costruzione pannelli ------------------------------------------------
    def _build_left_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)

        # Selezione device
        layout.addWidget(self._make_label("Select device:", font_label()))
        self.device_listbox = QListWidget()
        self.device_listbox.setFont(font_body())
        self.device_listbox.setMinimumWidth(260)
        layout.addWidget(self.device_listbox)
        asyncio.run(self.load_devices())
        self.device_listbox.itemDoubleClicked.connect(self.on_device_selected)

        # Impostazioni
        layout.addWidget(self._make_label("Impostazioni", font_title(), Qt.AlignmentFlag.AlignCenter))
        self._build_settings_tabs(layout)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)

        self.equalizer_preview = EqualizerPreviewWidget()
        layout.addWidget(self.equalizer_preview, 1)

        layout.addWidget(self._build_action_buttons())
        return panel

    def _make_label(self, text, font, alignment=None) -> QLabel:
        label = QLabel(text)
        label.setFont(font)
        if alignment is not None:
            label.setAlignment(alignment)
        return label

    def _build_action_buttons(self) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)

        self.start_stop_button = AnimatedButton(
            text="Start",
            command=self.start_stop_capture,
            font=font_button(),
            fg_color=COLOR_NEUTRAL,  # grigio per stato disabilitato
            hover_color=COLOR_NEUTRAL_HOVER,
            enabled=False,
            corner_radius=10,
            height=45,
        )
        row.addWidget(self.start_stop_button, 1)

        self.fullscreen_btn = AnimatedButton(
            text="Fullscreen",
            command=self.fullscreen_command,
            icon=QIcon(str(self.resource_manager.get_image_path('maximize.png', 'icons'))),
            icon_size=20,
            font=font_button(),
            fg_color=COLOR_FULLSCREEN,
            hover_color=COLOR_FULLSCREEN_HOVER,
            corner_radius=10,
            height=45,
        )
        row.addWidget(self.fullscreen_btn, 1)

        help_button = AnimatedButton(
            text="Help",
            command=self.open_help_window,
            icon=QIcon(str(self.resource_manager.get_image_path('help.png', 'icons'))),
            icon_size=20,
            font=font_button(),
            fg_color=COLOR_NEUTRAL,
            hover_color=COLOR_NEUTRAL_HOVER,
            corner_radius=10,
            height=45,
        )
        row.addWidget(help_button, 1)
        return container

    def _build_settings_tabs(self, parent_layout: QVBoxLayout):
        self.tabview = QTabWidget()
        self.tabview.addTab(self._build_visualization_tab(), "🎨 Visualizzazione")
        self.tabview.addTab(self._build_audio_tab(), "🎵 Audio")
        self.tabview.addTab(self._build_advanced_tab(), "⚙️ Avanzate")
        parent_layout.addWidget(self.tabview, 1)

    def _build_visualization_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        self.appearance_mode = OptionMenuCustomFrame(
            header_name='Tema:',
            values=["Light", "Dark", "System"],
            initial_value="System",
            command=self.change_appearance_mode_event,
        )
        v.addWidget(self.appearance_mode)

        self.file_picker = BackgroundFilepickerFrame(
            header_name='Sfondo (immagine o video):',
            placeholder_text='Percorso file...',
            apply_command=self.apply_bg_command,
        )
        v.addWidget(self.file_picker)

        self.alpha_slider = SliderCustomFrame(
            header_name='Trasparenza sfondo:',
            command=self.update_alpha,
            initial_value=self.bg_alpha,
        )
        v.addWidget(self.alpha_slider)

        self.bars_alpha_slider = SliderCustomFrame(
            header_name='Opacità barre:',
            command=self.update_bars_alpha,
            initial_value=self.bars_alpha,
        )
        v.addWidget(self.bars_alpha_slider)

        if len(self.available_monitors) > 1:
            self.monitor_option = OptionMenuCustomFrame(
                header_name='Monitor Fullscreen:',
                values=self.available_monitors,
                initial_value=self.available_monitors[0],
                command=self.change_selected_monitor,
            )
            v.addWidget(self.monitor_option)

        v.addStretch(1)
        return tab

    def _build_audio_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        self.noise_slider = SliderCustomFrame(
            header_name='Soglia Rumore:',
            command=self.update_noise_threshold,
        )
        v.addWidget(self.noise_slider)

        self.frequency_slider = SliderCustomFrame(
            header_name='Bande di Frequenza:',
            command=self.update_frequency_bands,
            from_=1,
            to=100,
            steps=32,
            initial_value=INITIAL_FREQUENCIES_BANDS,
            warning_trigger_value=15,
            warning_text='Numero elevato di bande!\nConsidera la modalità fullscreen',
            value_type=SliderCustomFrame.ValueType.INT,
        )
        v.addWidget(self.frequency_slider)

        v.addStretch(1)
        return tab

    def _build_advanced_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        # Pavimento dB: sotto questo livello la barra è a 0 (più basso = più sensibile)
        self.adv_db_floor_slider = SliderCustomFrame(
            header_name='Pavimento dB:',
            command=self.update_db_floor,
            from_=-90, to=-25,
            initial_value=int(DB_FLOOR),
            value_type=SliderCustomFrame.ValueType.INT,
        )
        v.addWidget(self.adv_db_floor_slider)

        # Tetto dB: fondo scala, barra piena (più basso = riempie prima)
        self.adv_db_ceiling_slider = SliderCustomFrame(
            header_name='Tetto dB:',
            command=self.update_db_ceiling,
            from_=-20, to=6,
            initial_value=int(DB_CEILING),
            value_type=SliderCustomFrame.ValueType.INT,
        )
        v.addWidget(self.adv_db_ceiling_slider)

        # Attacco (ms): velocità di salita delle barre
        self.adv_attack_slider = SliderCustomFrame(
            header_name='Attacco (ms):',
            command=self.update_attack_tau,
            from_=1, to=100,
            initial_value=int(ATTACK_TAU * 1000),
            value_type=SliderCustomFrame.ValueType.INT,
        )
        v.addWidget(self.adv_attack_slider)

        # Rilascio (ms): velocità di discesa delle barre
        self.adv_release_slider = SliderCustomFrame(
            header_name='Rilascio (ms):',
            command=self.update_release_tau,
            from_=20, to=1000,
            initial_value=int(RELEASE_TAU * 1000),
            value_type=SliderCustomFrame.ValueType.INT,
        )
        v.addWidget(self.adv_release_slider)

        # Tilt spettrale (dB/ottava): enfatizza bassi (<0) o alti (>0)
        self.adv_tilt_slider = SliderCustomFrame(
            header_name='Tilt (dB/ottava):',
            command=self.update_tilt,
            from_=-6, to=6, steps=24,
            initial_value=TILT_DB_PER_OCT,
            value_type=SliderCustomFrame.ValueType.DOUBLE,
        )
        v.addWidget(self.adv_tilt_slider)

        v.addStretch(1)
        return tab

    # --- Device --------------------------------------------------------------
    async def get_devices(self):
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            devices = await loop.run_in_executor(executor, sc.all_microphones, True)
        return devices

    async def load_devices(self):
        self.devices = await self.get_devices()
        for device in self.devices:
            name = str(device).replace('<', '').replace('>', '')
            self.device_listbox.addItem(name)

    def get_available_monitors(self) -> list[str]:
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        monitors = glfw.get_monitors()
        monitors_name = [
            f"{idx + 1}. {glfw.get_monitor_name(monitor).decode('utf-8')}"
            for idx, monitor in enumerate(monitors)
        ]
        glfw.terminate()
        return monitors_name

    def on_device_selected(self, item):
        """Quando un device è selezionato, abilita il pulsante start/stop."""
        row = self.device_listbox.row(item)
        if row < 0:
            return
        device = self.devices[row]

        # Se stiamo già catturando, ferma prima
        if self.is_capturing:
            self.stop_capture()
            self.is_capturing = False

        if self.last_device_selected != device:
            self.audio_queue = None
            self.stop_audio_thread()

            self.audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
            self.audio_thread = AudioCaptureThread(self.audio_queue, device=device)
            self.audio_thread.start()
            self.last_device_selected = device

            # Abilita il pulsante e imposta il colore verde
            self.start_stop_button.setEnabled(True)
            self.start_stop_button.update_base_colors(fg_color=COLOR_START, hover_color=COLOR_START_HOVER)
            self.start_stop_button.setText("Start")

    def stop_audio_thread(self):
        if self.audio_thread:
            self.audio_thread.stop()
            while self.audio_thread.is_alive():
                self.audio_thread.join(timeout=0.1)

    # --- Sfondo --------------------------------------------------------------
    def apply_bg_command(self):
        """
        Applica lo sfondo selezionato (immagine o video). Discrimina per
        estensione: i video sono gestiti SOLO dal renderer OpenGL (fullscreen);
        l'anteprima in-window mostra un segnaposto testuale.
        """
        filename = self.file_picker.get_filename()
        if not filename:
            QMessageBox.warning(self, 'Errore', 'Nessun file selezionato!')
            return

        ext = os.path.splitext(filename)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            self._apply_video_background(filename)
        else:
            self._apply_image_background(filename)

    def _apply_image_background(self, filename: str):
        """Carica un'immagine come sfondo: aggiorna l'anteprima e il renderer OpenGL."""
        try:
            self.bg_video_path = None  # torna alla modalità immagine
            self.bg_img = Image.open(filename)
            self.equalizer_preview.set_background_image(filename)
            self.update_bg_opengl(self.bg_img)
        except FileNotFoundError:
            QMessageBox.critical(self, 'Errore', 'File non trovato!')
        except Exception as e:
            QMessageBox.critical(self, 'Errore', f"Errore nel caricamento dell'immagine: {e}")

    def _apply_video_background(self, filename: str):
        """
        Seleziona un video come sfondo (feature solo-OpenGL). L'anteprima mostra
        un segnaposto; se il fullscreen è già aperto il video parte subito.
        """
        if not os.path.isfile(filename):
            QMessageBox.critical(self, 'Errore', 'File non trovato!')
            return
        self.bg_video_path = filename
        self.equalizer_preview.show_video_placeholder()
        self.update_video_opengl(filename)

    # --- Messaggi di controllo verso i renderer ------------------------------
    def update_noise_threshold(self, value):
        if self.preview_worker or self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_noise_threshold", "value": float(value)})

    def update_frequency_bands(self, value):
        if self.preview_worker or self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_frequency_bands", "value": int(value)})

    def update_alpha_opengl(self, value):
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_alpha", "value": float(value)})

    def update_bars_alpha(self, value):
        # L'opacità delle barre è gestita solo dal renderer OpenGL: l'anteprima
        # disegna barre piene e non ha un concetto di alpha barre.
        self.bars_alpha = value
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_bars_alpha", "value": float(value)})

    def update_db_floor(self, value):
        # Parametri DSP del tab Avanzate: solo renderer OpenGL (come bars_alpha).
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_db_floor", "value": float(value)})

    def update_db_ceiling(self, value):
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_db_ceiling", "value": float(value)})

    def update_attack_tau(self, value):
        # Lo slider è in millisecondi; la pipeline DSP lavora in secondi.
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_attack_tau", "value": float(value) / 1000.0})

    def update_release_tau(self, value):
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_release_tau", "value": float(value) / 1000.0})

    def update_tilt(self, value):
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_tilt_db_per_oct", "value": float(value)})

    def update_bg_opengl(self, image: Image.Image):
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_image", "value": image})

    def update_video_opengl(self, path: str):
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_video", "value": path})

    def update_alpha(self, value):
        self.bg_alpha = value
        # In modalità video l'anteprima mostra il segnaposto, non l'immagine.
        # L'alpha vale comunque per il video, inoltrato al renderer OpenGL.
        if self.bg_video_path is None:
            self.equalizer_preview.set_background_alpha(value)
        self.update_alpha_opengl(value)

    # --- Tema / monitor ------------------------------------------------------
    def change_appearance_mode_event(self, new_appearance_mode: str):
        """Cambia il tema dell'applicazione tramite qdarktheme."""
        qdarktheme.setup_theme(_THEME_MODE.get(new_appearance_mode, "auto"))

    def change_selected_monitor(self, selected_monitor_name: str) -> None:
        self.selected_monitor = selected_monitor_name

    # --- Cattura / anteprima -------------------------------------------------
    def start_capture(self):
        """Avvia il worker DSP dell'anteprima quando si preme Start."""
        if self.preview_worker is not None:
            return
        if not (self.audio_thread and self.audio_thread.is_alive()):
            self.show_no_audio_thread_warning()
            return
        try:
            self.preview_worker = EqualizerPreviewWorker(
                self.audio_queue,
                self.equalizer_control_queue,
                noise_threshold=self.noise_slider.get_value(),
                n_bands=int(self.frequency_slider.get_value()),
            )
            self.preview_thread = QThread()
            self.preview_worker.moveToThread(self.preview_thread)
            self.preview_thread.started.connect(self.preview_worker.run)
            self.preview_worker.bands_ready.connect(self.equalizer_preview.on_bands)
            self.preview_thread.start()
        except Exception:
            self.preview_worker = None
            self.preview_thread = None

    def stop_capture(self):
        """Ferma il worker DSP dell'anteprima e attende la fine del thread."""
        if self.preview_worker is not None:
            self.preview_worker.stop()
        if self.preview_thread is not None:
            self.preview_thread.quit()
            self.preview_thread.wait()
        self.preview_worker = None
        self.preview_thread = None
        self.equalizer_preview.clear_bars()

    def start_stop_capture(self):
        """Avvia o ferma la cattura audio."""
        if not self.audio_thread or not self.audio_thread.is_alive():
            self.show_no_audio_thread_warning()
            return
        if not self.audio_queue:
            return

        if not self.is_capturing:
            # Svuota la coda prima di iniziare
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            self.start_capture()
            QTimer.singleShot(100, self._check_capture_started)
        else:
            self.stop_capture()
            self.is_capturing = False
            self.start_stop_button.update_base_colors(fg_color=COLOR_START, hover_color=COLOR_START_HOVER)
            self.start_stop_button.setText("Start")

    def _check_capture_started(self):
        """Verifica che la cattura sia partita e aggiorna l'UI."""
        if self.preview_thread is not None and self.preview_thread.isRunning():
            self.is_capturing = True
            self.start_stop_button.update_base_colors(fg_color=COLOR_STOP, hover_color=COLOR_STOP_HOVER)
            self.start_stop_button.setText("Stop")
        else:
            self.is_capturing = False

    # --- Fullscreen OpenGL ---------------------------------------------------
    def fullscreen_command(self):
        """Apre (o chiude) la visualizzazione fullscreen OpenGL."""
        # Se stiamo catturando in anteprima, ferma prima.
        if self.is_capturing:
            self.stop_capture()
            self.is_capturing = False
            self.start_stop_button.update_base_colors(fg_color=COLOR_START, hover_color=COLOR_START_HOVER)
            self.start_stop_button.setText("Start")

        if not self.audio_thread:
            self.show_no_audio_thread_warning()
            return

        if self.equalizer_opengl_thread:
            self.equalizer_opengl_thread.stop()
            while self.equalizer_opengl_thread.is_alive():
                self.equalizer_opengl_thread.join(timeout=0.1)
            self.equalizer_opengl_thread = None
        else:
            self.equalizer_opengl_thread = EqualizerOpenGLThread(
                self.audio_queue,
                noise_threshold=self.noise_slider.get_value(),
                n_bands=int(self.frequency_slider.get_value()),
                control_queue=self.equalizer_control_queue,
                bg_image=self.bg_img,
                bg_alpha=self.bg_alpha,
                bars_alpha=self.bars_alpha,
                bg_video=self.bg_video_path,
                db_floor=self.adv_db_floor_slider.get_value(),
                db_ceiling=self.adv_db_ceiling_slider.get_value(),
                attack_tau=self.adv_attack_slider.get_value() / 1000.0,
                release_tau=self.adv_release_slider.get_value() / 1000.0,
                tilt_db_per_oct=self.adv_tilt_slider.get_value(),
                monitor=self.selected_monitor,
                window_close_callback=self.opengl_window_closed,
            )
            self.equalizer_opengl_thread.start()

    def opengl_window_closed(self):
        # Chiamato sul thread GLFW: marshalla sul thread GUI via segnale.
        self._opengl_closed.emit()

    def _on_opengl_closed(self):
        self.equalizer_opengl_thread = None

    # --- Help / avvisi -------------------------------------------------------
    def open_help_window(self):
        if self.help_window is None:
            self.help_window = HelpWindow(self)
        self.help_window.show()
        self.help_window.raise_()
        self.help_window.activateWindow()

    def show_no_audio_thread_warning(self):
        QMessageBox.information(self, 'Info', 'Select a device first!')

    # --- Ciclo di vita -------------------------------------------------------
    def closeEvent(self, event):
        """
        Alla chiusura della finestra ferma con grazia tutti i thread avviati.
        """
        self.stop_capture()

        if self.audio_thread:
            self.audio_thread.stop()
            while self.audio_thread.is_alive():
                self.audio_thread.join(timeout=0.1)

        if self.equalizer_opengl_thread:
            self.equalizer_opengl_thread.stop()
            while self.equalizer_opengl_thread.is_alive():
                self.equalizer_opengl_thread.join(timeout=0.1)

        event.accept()
