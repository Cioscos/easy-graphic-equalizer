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
from gui.color_picker_frame import ColorPickerFrame
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

# Etichette del menu "Modalità colore" -> intero uColorMode del renderer.
_COLOR_MODE_TO_INT = {"Classico": 0, "Tinta unica": 1, "Gradiente": 2, "Spettro": 3}

# Etichette del menu "Modalità" -> intero self._mode del renderer.
_VIZ_MODE_TO_INT = {"Barre": 0, "Radiale": 1, "Oscilloscopio": 2, "Linea/Area": 3}


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

        # Stato GUI dell'aspetto barre (Blocco 1). Default = look attuale.
        self.bars_color_mode = 0
        self.bars_color_a = (0.231, 0.420, 1.0)
        self.bars_color_b = (1.0, 0.231, 0.816)
        self.bars_green_split = 0.5
        self.bars_yellow_split = 0.8
        self.bars_width = 0.8
        self.bars_rounded = False
        self.bars_anchor_center = False
        self.bars_band_symmetric = False

        # Stato GUI effetti reattivi (Blocco 2). Default = OFF.
        self.fx_peakcap_enabled = False
        self.fx_peakcap_color = (1.0, 1.0, 1.0)
        self.fx_peakcap_hold = 0.5
        self.fx_peakcap_fall = 0.8
        self.fx_bloom_enabled = False
        self.fx_bloom_intensity = 1.0
        self.fx_beat_enabled = False
        self.fx_beat_intensity = 0.08
        self.fx_beat_sensitivity = 1.5

        # Stato GUI modalità di visualizzazione (Blocco 3). Default = Barre.
        self.viz_mode = 0
        self.viz_inner_radius = 0.18
        self.viz_line_thickness = 0.012
        self.viz_fill = True
        self.viz_osc_mirror = False
        self.viz_osc_smoothing = 0.5
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
        self.tabview.addTab(self._build_shape_tab(), "🎛️ Forma")
        self.tabview.addTab(self._build_effects_tab(), "✨ Effetti")
        parent_layout.addWidget(self.tabview, 1)

        # Visibilità condizionale iniziale dei controlli di forma/effetti (modalità Barre).
        self.on_viz_mode_changed("Barre")

    def _build_visualization_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        self.viz_mode_option = OptionMenuCustomFrame(
            header_name='Modalità:',
            values=["Barre", "Radiale", "Oscilloscopio", "Linea/Area"],
            initial_value="Barre",
            command=self.on_viz_mode_changed,
        )
        v.addWidget(self.viz_mode_option)

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

    def _build_shape_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        v.addWidget(self._make_label("Colore", font_label()))

        self.bars_mode_option = OptionMenuCustomFrame(
            header_name='Modalità colore:',
            values=["Classico", "Tinta unica", "Gradiente", "Spettro"],
            initial_value="Classico",
            command=self.on_color_mode_changed,
        )
        v.addWidget(self.bars_mode_option)

        # Classico: due soglie regolabili
        self.bars_yellow_thr = SliderCustomFrame(
            header_name='Soglia giallo:',
            command=self.update_green_split,
            initial_value=self.bars_green_split,
        )
        v.addWidget(self.bars_yellow_thr)

        self.bars_red_thr = SliderCustomFrame(
            header_name='Soglia rosso:',
            command=self.update_yellow_split,
            initial_value=self.bars_yellow_split,
        )
        v.addWidget(self.bars_red_thr)

        # Tinta unica: un colore
        self.bars_solid_color = ColorPickerFrame(
            header_name='Colore barre:',
            initial_color='#22d36a',
            command=self.update_color_a,
        )
        v.addWidget(self.bars_solid_color)

        # Gradiente: due colori
        self.bars_grad_base = ColorPickerFrame(
            header_name='Colore base:',
            initial_color='#3b6bff',
            command=self.update_color_a,
        )
        v.addWidget(self.bars_grad_base)

        self.bars_grad_top = ColorPickerFrame(
            header_name='Colore cima:',
            initial_color='#ff3bd0',
            command=self.update_color_b,
        )
        v.addWidget(self.bars_grad_top)

        v.addWidget(self._make_label("Forma", font_label()))

        self.bars_width_slider = SliderCustomFrame(
            header_name='Larghezza barre:',
            command=self.update_bar_width,
            from_=0.1, to=1.0,
            initial_value=self.bars_width,
        )
        v.addWidget(self.bars_width_slider)

        self.bars_rounded_option = OptionMenuCustomFrame(
            header_name='Cime arrotondate:',
            values=["No", "Sì"],
            initial_value="No",
            command=self.update_rounded,
        )
        v.addWidget(self.bars_rounded_option)

        self.bars_anchor_option = OptionMenuCustomFrame(
            header_name='Ancoraggio:',
            values=["Dal basso", "Centro"],
            initial_value="Dal basso",
            command=self.update_anchor,
        )
        v.addWidget(self.bars_anchor_option)

        self.bars_order_option = OptionMenuCustomFrame(
            header_name='Ordine bande:',
            values=["Standard", "Simmetrico"],
            initial_value="Standard",
            command=self.update_band_order,
        )
        v.addWidget(self.bars_order_option)

        # Radiale
        self.viz_inner_radius_slider = SliderCustomFrame(
            header_name='Raggio foro:',
            command=self.update_inner_radius,
            from_=0.0, to=0.8,
            initial_value=self.viz_inner_radius,
        )
        v.addWidget(self.viz_inner_radius_slider)

        # Oscilloscopio / Linea-Area
        self.viz_thickness_slider = SliderCustomFrame(
            header_name='Spessore linea:',
            command=self.update_line_thickness,
            from_=0.002, to=0.06,
            initial_value=self.viz_line_thickness,
        )
        v.addWidget(self.viz_thickness_slider)

        # Linea/Area
        self.viz_fill_option = OptionMenuCustomFrame(
            header_name='Riempimento:', values=["Linea", "Area"], initial_value="Area",
            command=self.update_fill)
        v.addWidget(self.viz_fill_option)

        # Oscilloscopio
        self.viz_osc_mirror_option = OptionMenuCustomFrame(
            header_name='Specchiato:', values=["No", "Sì"], initial_value="No",
            command=self.update_osc_mirror)
        v.addWidget(self.viz_osc_mirror_option)

        self.viz_osc_smooth_slider = SliderCustomFrame(
            header_name='Fluidità (oscill.):',
            command=self.update_osc_smoothing,
            from_=0.0, to=1.0,
            initial_value=self.viz_osc_smoothing,
        )
        v.addWidget(self.viz_osc_smooth_slider)

        v.addStretch(1)

        # Imposta la visibilità condizionale iniziale (modalità = Classico)
        self.on_color_mode_changed("Classico")
        return tab

    def _build_effects_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)

        self.fx_peakcap_label = self._make_label("Peak-cap", font_label())
        v.addWidget(self.fx_peakcap_label)
        self.fx_peakcap_option = OptionMenuCustomFrame(
            header_name='Peak-cap:', values=["Off", "On"], initial_value="Off",
            command=self.update_peakcap_enabled)
        v.addWidget(self.fx_peakcap_option)
        self.fx_peakcap_color_picker = ColorPickerFrame(
            header_name='Colore cap:', initial_color='#ffffff', command=self.update_peakcap_color)
        v.addWidget(self.fx_peakcap_color_picker)
        self.fx_peakcap_hold_slider = SliderCustomFrame(
            header_name='Hold (ms):', command=self.update_peakcap_hold,
            from_=0, to=2000, initial_value=int(self.fx_peakcap_hold * 1000),
            value_type=SliderCustomFrame.ValueType.INT)
        v.addWidget(self.fx_peakcap_hold_slider)
        self.fx_peakcap_fall_slider = SliderCustomFrame(
            header_name='Caduta (/s):', command=self.update_peakcap_fall,
            from_=0.1, to=3.0, initial_value=self.fx_peakcap_fall,
            value_type=SliderCustomFrame.ValueType.DOUBLE)
        v.addWidget(self.fx_peakcap_fall_slider)

        v.addWidget(self._make_label("Glow (bloom)", font_label()))
        self.fx_bloom_option = OptionMenuCustomFrame(
            header_name='Glow:', values=["Off", "On"], initial_value="Off",
            command=self.update_bloom_enabled)
        v.addWidget(self.fx_bloom_option)
        self.fx_bloom_slider = SliderCustomFrame(
            header_name='Intensità:', command=self.update_bloom_intensity,
            from_=0.0, to=3.0, initial_value=self.fx_bloom_intensity,
            value_type=SliderCustomFrame.ValueType.DOUBLE)
        v.addWidget(self.fx_bloom_slider)

        v.addWidget(self._make_label("Beat pulse", font_label()))
        self.fx_beat_option = OptionMenuCustomFrame(
            header_name='Beat pulse:', values=["Off", "On"], initial_value="Off",
            command=self.update_beat_enabled)
        v.addWidget(self.fx_beat_option)
        self.fx_beat_intensity_slider = SliderCustomFrame(
            header_name='Intensità zoom:', command=self.update_beat_intensity,
            from_=0.0, to=0.3, initial_value=self.fx_beat_intensity,
            value_type=SliderCustomFrame.ValueType.DOUBLE)
        v.addWidget(self.fx_beat_intensity_slider)
        self.fx_beat_sens_slider = SliderCustomFrame(
            header_name='Sensibilità:', command=self.update_beat_sensitivity,
            from_=1.05, to=3.0, initial_value=self.fx_beat_sensitivity,
            value_type=SliderCustomFrame.ValueType.DOUBLE)
        v.addWidget(self.fx_beat_sens_slider)

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

    def on_color_mode_changed(self, label: str):
        """Mostra solo i controlli colore pertinenti e inoltra la modalità."""
        classico = label == "Classico"
        tinta = label == "Tinta unica"
        grad = label == "Gradiente"
        self.bars_yellow_thr.setVisible(classico)
        self.bars_red_thr.setVisible(classico)
        self.bars_solid_color.setVisible(tinta)
        self.bars_grad_base.setVisible(grad)
        self.bars_grad_top.setVisible(grad)
        self.bars_color_mode = _COLOR_MODE_TO_INT.get(label, 0)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_color_mode", "value": self.bars_color_mode})

    def on_viz_mode_changed(self, label: str):
        """Mostra solo i controlli di forma pertinenti alla modalità e inoltra la scelta."""
        bars = label == "Barre"
        radial = label == "Radiale"
        osc = label == "Oscilloscopio"
        line = label == "Linea/Area"
        # Forma (tab "🎛️ Forma")
        self.bars_width_slider.setVisible(bars or radial)
        self.bars_rounded_option.setVisible(bars)
        self.bars_anchor_option.setVisible(bars)
        self.bars_order_option.setVisible(bars or radial or line)
        self.viz_inner_radius_slider.setVisible(radial)
        self.viz_thickness_slider.setVisible(osc or line)
        self.viz_fill_option.setVisible(line)
        self.viz_osc_mirror_option.setVisible(osc)
        self.viz_osc_smooth_slider.setVisible(osc)
        # Peak-cap: solo in modalità Barre (tab "✨ Effetti")
        self.fx_peakcap_label.setVisible(bars)
        self.fx_peakcap_option.setVisible(bars)
        self.fx_peakcap_color_picker.setVisible(bars)
        self.fx_peakcap_hold_slider.setVisible(bars)
        self.fx_peakcap_fall_slider.setVisible(bars)
        self.viz_mode = _VIZ_MODE_TO_INT.get(label, 0)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_viz_mode", "value": self.viz_mode})

    def update_inner_radius(self, value):
        self.viz_inner_radius = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_inner_radius", "value": float(value)})

    def update_line_thickness(self, value):
        self.viz_line_thickness = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_line_thickness", "value": float(value)})

    def update_fill(self, label):
        self.viz_fill = (label == "Area")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_fill", "value": self.viz_fill})

    def update_osc_mirror(self, label):
        self.viz_osc_mirror = (label == "Sì")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_osc_mirror", "value": self.viz_osc_mirror})

    def update_osc_smoothing(self, value):
        self.viz_osc_smoothing = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_osc_smoothing", "value": float(value)})

    def update_color_a(self, rgb):
        self.bars_color_a = rgb
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_color_a", "value": rgb})

    def update_color_b(self, rgb):
        self.bars_color_b = rgb
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_color_b", "value": rgb})

    def update_green_split(self, value):
        self.bars_green_split = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_green_split", "value": float(value)})

    def update_yellow_split(self, value):
        self.bars_yellow_split = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_yellow_split", "value": float(value)})

    def update_bar_width(self, value):
        self.bars_width = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_bar_width", "value": float(value)})

    def update_rounded(self, label):
        self.bars_rounded = (label == "Sì")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_rounded", "value": self.bars_rounded})

    def update_anchor(self, label):
        self.bars_anchor_center = (label == "Centro")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_bar_anchor", "value": self.bars_anchor_center})

    def update_band_order(self, label):
        self.bars_band_symmetric = (label == "Simmetrico")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_band_order", "value": self.bars_band_symmetric})

    def update_peakcap_enabled(self, label):
        self.fx_peakcap_enabled = (label == "On")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_peakcap_enabled", "value": self.fx_peakcap_enabled})

    def update_peakcap_color(self, rgb):
        self.fx_peakcap_color = rgb
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_peakcap_color", "value": rgb})

    def update_peakcap_hold(self, value):
        self.fx_peakcap_hold = float(value) / 1000.0
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_peakcap_hold", "value": float(value) / 1000.0})

    def update_peakcap_fall(self, value):
        self.fx_peakcap_fall = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_peakcap_fall", "value": float(value)})

    def update_bloom_enabled(self, label):
        self.fx_bloom_enabled = (label == "On")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_bloom_enabled", "value": self.fx_bloom_enabled})

    def update_bloom_intensity(self, value):
        self.fx_bloom_intensity = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_bloom_intensity", "value": float(value)})

    def update_beat_enabled(self, label):
        self.fx_beat_enabled = (label == "On")
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_beat_enabled", "value": self.fx_beat_enabled})

    def update_beat_intensity(self, value):
        self.fx_beat_intensity = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_beat_intensity", "value": float(value)})

    def update_beat_sensitivity(self, value):
        self.fx_beat_sensitivity = float(value)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_beat_sensitivity", "value": float(value)})

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
                color_mode=self.bars_color_mode,
                color_a=self.bars_color_a,
                color_b=self.bars_color_b,
                green_split=self.bars_green_split,
                yellow_split=self.bars_yellow_split,
                bar_width=self.bars_width,
                rounded=self.bars_rounded,
                mirror=self.bars_anchor_center,
                band_order_symmetric=self.bars_band_symmetric,
                peakcap_enabled=self.fx_peakcap_enabled,
                peakcap_color=self.fx_peakcap_color,
                peakcap_hold=self.fx_peakcap_hold,
                peakcap_fall=self.fx_peakcap_fall,
                beat_enabled=self.fx_beat_enabled,
                beat_intensity=self.fx_beat_intensity,
                beat_sensitivity=self.fx_beat_sensitivity,
                bloom_enabled=self.fx_bloom_enabled,
                bloom_intensity=self.fx_bloom_intensity,
                mode=self.viz_mode,
                inner_radius=self.viz_inner_radius,
                line_thickness=self.viz_line_thickness,
                fill=self.viz_fill,
                osc_mirror=self.viz_osc_mirror,
                osc_smoothing=self.viz_osc_smoothing,
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
