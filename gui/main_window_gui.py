import os
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor

import soundcard as sc
from PIL import Image
import glfw
import qdarktheme

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QAction, QActionGroup
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from thread.audioCaptureThread import AudioCaptureThread
from thread.menu_keys import TOGGLE_KEY_NAMES
from thread.opengl_thread import (
    EqualizerOpenGLThread,
    DB_FLOOR, DB_CEILING, ATTACK_TAU, RELEASE_TAU, TILT_DB_PER_OCT,
)
from gui.slider_frame import SliderCustomFrame
from gui.optionmenu_frame import OptionMenuCustomFrame
from gui.background_filepicker_frame import BackgroundFilepickerFrame
from gui.help_window import HelpWindow
from gui.animated_button import AnimatedButton
from gui.color_picker_frame import ColorPickerFrame
from gui.theme import (
    font_button, font_label, font_section, font_body,
    theme_kwargs,
    SPACE_SM, SPACE_MD,
    COLOR_NEUTRAL, COLOR_NEUTRAL_HOVER,
    COLOR_START, COLOR_START_HOVER,
    COLOR_STOP, COLOR_STOP_HOVER,
)
from resource_manager import ResourceManager
import profiles
from profiles import ProfileStore
from gui.profile_bar_frame import ProfileBarFrame

# Coda audio volutamente piccola: il renderer svuota tutta la coda a ogni frame
# tenendo solo l'audio più recente, quindi una coda corta evita backlog/latenza.
MAX_QUEUE_SIZE = 8
INITIAL_FREQUENCIES_BANDS = 9
# Estensioni trattate come video (sfondo animato, solo renderer OpenGL).
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')

# Mappa il tema esposto nell'interfaccia alla modalità di qdarktheme.
_THEME_MODE = {"Light": "light", "Dark": "dark", "System": "auto"}

# Etichette del menu "Modalità colore" -> intero uColorMode del renderer.
_COLOR_MODE_TO_INT = {"Classico": 0, "Tinta unica": 1, "Gradiente": 2, "Spettro": 3}

# Etichette del menu "Modalità" -> intero self._mode del renderer.
_VIZ_MODE_TO_INT = {"Barre": 0, "Radiale": 1, "Oscilloscopio": 2, "Linea/Area": 3}

# Mappe inverse (intero renderer -> etichetta GUI), usate dal caricamento profili.
_INT_TO_COLOR_MODE = {v: k for k, v in _COLOR_MODE_TO_INT.items()}
_INT_TO_VIZ_MODE = {v: k for k, v in _VIZ_MODE_TO_INT.items()}


class AudioCaptureGUI(QMainWindow):
    """
    Finestra principale (PySide6).

    Possiede e coordina i thread/le code della pipeline produttore/consumatore.
    È un pannello di controllo: configura i parametri e lancia il renderer
    fullscreen OpenGL (EqualizerOpenGLThread), unico consumer dell'audio.
    """

    # Emesso quando la finestra OpenGL si chiude: il callback parte sul thread
    # GLFW, il segnale lo marshalla sul thread GUI (connessione queued).
    _opengl_closed = Signal()
    _setting_changed = Signal(dict)
    # Emesso quando la cattura audio muore per errore: il callback parte sul
    # thread di cattura, il segnale lo marshalla sul thread GUI.
    _audio_error = Signal(str)

    def __init__(self):
        super().__init__()

        self.resource_manager = ResourceManager()
        self.setWindowTitle("Sound wave")
        self.setMinimumSize(900, 620)

        # Stato condiviso inizializzato prima della costruzione dei widget
        self.devices = []
        self.bg_alpha = 0.5
        self.bars_alpha = 1.0  # opacità delle barre nel renderer OpenGL (1.0 = piene)

        # Stato GUI dell'aspetto barre (Blocco 1). Default = look attuale.
        self.menu_toggle_key = "F1"   # tasto che apre/chiude il menù in-window
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
        self.equalizer_opengl_thread = None
        self.last_device_selected = None
        self.help_window = None

        # Store dei profili (gestore con nomi + import/export).
        self._profile_store = ProfileStore()

        # Immagine di sfondo di default (PIL): usata dal renderer OpenGL.
        self._default_bg_path = str(self.resource_manager.get_image_path(profiles.DEFAULT_BG_NAME, 'bg'))
        self.bg_image_path = self._default_bg_path  # path immagine attiva (per i profili)
        self.bg_img = Image.open(self._default_bg_path)

        # Costruzione UI: pannello di controllo (nessuna anteprima)
        self._build_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        root_layout.setSpacing(SPACE_MD)

        # Selezione dispositivo (in alto, a tutta larghezza)
        root_layout.addWidget(self._build_device_selector())

        # Barra profili (gestore con nomi): sotto il device selector
        root_layout.addWidget(self._build_profile_bar())

        # Corpo: nav a sinistra | pagina impostazioni a destra
        nav_widget = self._build_nav()
        pages_widget = self._build_pages()
        self.nav.currentRowChanged.connect(self.settings_stack.setCurrentIndex)

        nav_pages = QSplitter(Qt.Orientation.Horizontal)
        nav_pages.setChildrenCollapsible(False)
        nav_pages.addWidget(nav_widget)
        nav_pages.addWidget(pages_widget)
        nav_pages.setStretchFactor(0, 0)
        nav_pages.setStretchFactor(1, 1)
        nav_pages.setSizes([220, 540])
        root_layout.addWidget(nav_pages, 1)

        # Footer: azione primaria (avvia/ferma il fullscreen)
        root_layout.addWidget(self._build_footer())

        self.nav.setCurrentRow(0)
        self._opengl_closed.connect(self._on_opengl_closed)
        self._setting_changed.connect(self._on_setting_changed)
        self._audio_error.connect(self._on_audio_error)

        # Auto-load dell'ultimo profilo usato (se ancora presente). Protetto:
        # un profilo corrotto su disco non deve impedire l'avvio — l'app parte
        # coi default e segnala il problema senza cancellare il file.
        last = None
        try:
            last = self._profile_store.get_last()
            if last and last in self._profile_store.list_names():
                self.profile_bar.set_profiles(self._profile_store.list_names(), last)
                self._apply_profile(self._profile_store.load(last))
        except Exception as e:
            msg = (f"Impossibile caricare l'ultimo profilo «{last}»: {e}\n"
                   "Avvio con le impostazioni predefinite.")
            QTimer.singleShot(0, lambda: QMessageBox.warning(self, "Profilo", msg))

    # --- Costruzione pannelli ------------------------------------------------
    def _build_menu_bar(self):
        """Barra dei menu: 'Tema' (Light/Dark/System esclusivi) e 'Aiuto'."""
        menu_bar = self.menuBar()

        theme_menu = menu_bar.addMenu("Tema")
        self._theme_action_group = QActionGroup(self)
        self._theme_action_group.setExclusive(True)
        for label in ("Light", "Dark", "System"):
            action = QAction(label, self, checkable=True)
            action.triggered.connect(
                lambda _checked, name=label: self.change_appearance_mode_event(name)
            )
            self._theme_action_group.addAction(action)
            theme_menu.addAction(action)
            if label == "System":  # default come prima (initial_value="System")
                action.setChecked(True)

        help_menu = menu_bar.addMenu("Aiuto")
        help_menu.addAction("Apri guida", self.open_help_window)

    def _build_device_selector(self) -> QWidget:
        """Riga in alto: selezione dispositivo (precondizione per l'avvio)."""
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(SPACE_SM)
        v.addWidget(self._make_label("Seleziona dispositivo:", font_label()))
        self.device_listbox = QListWidget()
        self.device_listbox.setFont(font_body())
        self.device_listbox.setMaximumHeight(120)
        v.addWidget(self.device_listbox)
        asyncio.run(self.load_devices())
        # Singolo clic per selezionare (più scopribile); il doppio clic resta valido.
        self.device_listbox.itemClicked.connect(self.on_device_selected)
        self.device_listbox.itemDoubleClicked.connect(self.on_device_selected)
        return container

    def _build_nav(self) -> QWidget:
        """Lista di navigazione (sinistra): seleziona la pagina di impostazioni."""
        self.nav = QListWidget()
        self.nav.setObjectName("navList")
        self.nav.setFont(font_label())
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav.setMinimumWidth(170)
        self.nav.setMaximumWidth(240)
        for icon, text in (
            ("🖥️", "Visualizzazione"),
            ("🎛️", "Forma"),
            ("🎨", "Colore"),
            ("🎵", "Audio"),
            ("✨", "Effetti"),
        ):
            self.nav.addItem(f"{icon}  {text}")
        return self.nav

    def _build_pages(self) -> QWidget:
        """Pile delle pagine di impostazioni (destra)."""
        self.settings_stack = QStackedWidget()
        self._build_settings_pages()
        return self.settings_stack

    def _build_profile_bar(self) -> QWidget:
        """Barra profili: selettore + Salva/Salva come…/Elimina/Importa…/Esporta…."""
        self.profile_bar = ProfileBarFrame(
            on_select=self._on_profile_selected,
            on_save=self._on_profile_save,
            on_save_as=self._on_profile_save_as,
            on_delete=self._on_profile_delete,
            on_import=self._on_profile_import,
            on_export=self._on_profile_export,
        )
        self.profile_bar.set_profiles(self._profile_store.list_names(), None)
        return self.profile_bar

    def _refresh_profile_bar(self, selected=None) -> None:
        self.profile_bar.set_profiles(self._profile_store.list_names(), selected)

    def _build_footer(self) -> QWidget:
        """Azione primaria a tutta larghezza: avvia/ferma il fullscreen OpenGL."""
        self.fullscreen_btn = AnimatedButton(
            text="Avvia a schermo intero",
            command=self.fullscreen_command,
            icon=QIcon(str(self.resource_manager.get_image_path('maximize.png', 'icons'))),
            icon_size=20,
            font=font_button(),
            fg_color=COLOR_NEUTRAL,  # grigio finché non si seleziona un dispositivo
            hover_color=COLOR_NEUTRAL_HOVER,
            enabled=False,
            corner_radius=10,
            height=48,
        )
        return self.fullscreen_btn

    def _make_label(self, text, font, alignment=None) -> QLabel:
        label = QLabel(text)
        label.setFont(font)
        if alignment is not None:
            label.setAlignment(alignment)
        return label

    def _build_settings_pages(self):
        """Costruisce le pagine impostazioni e le aggiunge al QStackedWidget.

        L'ordine deve combaciare con le voci di nav in _build_settings_column:
        Visualizzazione, Forma, Colore, Audio, Effetti.
        """
        self._setting_groups = []  # QGroupBox da nascondere se tutti i controlli sono nascosti
        self.settings_stack.addWidget(self._build_visualization_page())
        self.settings_stack.addWidget(self._build_shape_page())
        self.settings_stack.addWidget(self._build_color_page())
        self.settings_stack.addWidget(self._build_audio_page())
        self.settings_stack.addWidget(self._build_effects_page())

        # Visibilità condizionale iniziale: tutte le pagine esistono già, quindi le
        # callback possono toccare widget di pagine diverse senza errori.
        self.on_color_mode_changed("Classico")
        self.on_viz_mode_changed("Barre")

    def _page(self, *groups) -> QWidget:
        """Pagina scrollabile: gruppi impilati dentro una QScrollArea (no clipping)."""
        content = QWidget()
        v = QVBoxLayout(content)
        v.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        v.setSpacing(SPACE_MD)
        for group in groups:
            v.addWidget(group)
        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    def _group(self, title: str, *widgets) -> QGroupBox:
        """Sezione titolata che raccoglie alcuni frame-controllo."""
        box = QGroupBox(title)
        box.setFont(font_section())
        v = QVBoxLayout(box)
        v.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        v.setSpacing(SPACE_SM)
        for w in widgets:
            v.addWidget(w)
        box._controls = list(widgets)
        self._setting_groups.append(box)
        return box

    def _refresh_group_visibility(self):
        """Nasconde i gruppi i cui controlli sono tutti nascosti (niente box vuoti)."""
        for box in getattr(self, "_setting_groups", []):
            box.setVisible(any(w.isVisibleTo(box) for w in box._controls))

    def _build_visualization_page(self) -> QWidget:
        self.viz_mode_option = OptionMenuCustomFrame(
            header_name='Modalità:',
            values=["Barre", "Radiale", "Oscilloscopio", "Linea/Area"],
            initial_value="Barre",
            command=self.on_viz_mode_changed,
        )
        self.file_picker = BackgroundFilepickerFrame(
            header_name='Sfondo (immagine o video):',
            placeholder_text='Percorso file...',
            apply_command=self.apply_bg_command,
        )
        self.alpha_slider = SliderCustomFrame(
            header_name='Trasparenza sfondo:',
            command=self.update_alpha,
            initial_value=self.bg_alpha,
        )
        self.bars_alpha_slider = SliderCustomFrame(
            header_name='Opacità barre:',
            command=self.update_bars_alpha,
            initial_value=self.bars_alpha,
        )
        self.menu_toggle_option = OptionMenuCustomFrame(
            header_name='Tasto menù in-window:',
            values=TOGGLE_KEY_NAMES,
            initial_value=self.menu_toggle_key,
            command=self.update_menu_toggle_key,
        )

        resa_widgets = [self.bars_alpha_slider]
        if len(self.available_monitors) > 1:
            self.monitor_option = OptionMenuCustomFrame(
                header_name='Monitor Fullscreen:',
                values=self.available_monitors,
                initial_value=self.available_monitors[0],
                command=self.change_selected_monitor,
            )
            resa_widgets.append(self.monitor_option)

        return self._page(
            self._group("Modalità", self.viz_mode_option),
            self._group("Sfondo", self.file_picker, self.alpha_slider),
            self._group("Resa", *resa_widgets),
            self._group("Menù in-window", self.menu_toggle_option),
        )

    def _build_audio_page(self) -> QWidget:
        self.noise_slider = SliderCustomFrame(
            header_name='Soglia Rumore:',
            command=self.update_noise_threshold,
        )
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
        return self._page(
            self._group("Sorgente audio", self.noise_slider, self.frequency_slider),
            self._build_advanced_group(),
        )

    def _build_advanced_group(self) -> QGroupBox:
        # Pavimento dB: sotto questo livello la barra è a 0 (più basso = più sensibile)
        self.adv_db_floor_slider = SliderCustomFrame(
            header_name='Pavimento dB:',
            command=self.update_db_floor,
            from_=-90, to=-25,
            initial_value=int(DB_FLOOR),
            value_type=SliderCustomFrame.ValueType.INT,
        )
        # Tetto dB: fondo scala, barra piena (più basso = riempie prima)
        self.adv_db_ceiling_slider = SliderCustomFrame(
            header_name='Tetto dB:',
            command=self.update_db_ceiling,
            from_=-20, to=6,
            initial_value=int(DB_CEILING),
            value_type=SliderCustomFrame.ValueType.INT,
        )
        # Attacco (ms): velocità di salita delle barre
        self.adv_attack_slider = SliderCustomFrame(
            header_name='Attacco (ms):',
            command=self.update_attack_tau,
            from_=1, to=100,
            initial_value=int(ATTACK_TAU * 1000),
            value_type=SliderCustomFrame.ValueType.INT,
        )
        # Rilascio (ms): velocità di discesa delle barre
        self.adv_release_slider = SliderCustomFrame(
            header_name='Rilascio (ms):',
            command=self.update_release_tau,
            from_=20, to=1000,
            initial_value=int(RELEASE_TAU * 1000),
            value_type=SliderCustomFrame.ValueType.INT,
        )
        # Tilt spettrale (dB/ottava): enfatizza bassi (<0) o alti (>0)
        self.adv_tilt_slider = SliderCustomFrame(
            header_name='Tilt (dB/ottava):',
            command=self.update_tilt,
            from_=-6, to=6, steps=24,
            initial_value=TILT_DB_PER_OCT,
            value_type=SliderCustomFrame.ValueType.DOUBLE,
        )
        return self._group(
            "Avanzate (DSP)",
            self.adv_db_floor_slider, self.adv_db_ceiling_slider,
            self.adv_attack_slider, self.adv_release_slider, self.adv_tilt_slider,
        )

    def _build_color_page(self) -> QWidget:
        self.bars_mode_option = OptionMenuCustomFrame(
            header_name='Modalità colore:',
            values=["Classico", "Tinta unica", "Gradiente", "Spettro"],
            initial_value="Classico",
            command=self.on_color_mode_changed,
        )
        # Classico: due soglie regolabili
        self.bars_yellow_thr = SliderCustomFrame(
            header_name='Soglia giallo:',
            command=self.update_green_split,
            initial_value=self.bars_green_split,
        )
        self.bars_red_thr = SliderCustomFrame(
            header_name='Soglia rosso:',
            command=self.update_yellow_split,
            initial_value=self.bars_yellow_split,
        )
        # Tinta unica: un colore
        self.bars_solid_color = ColorPickerFrame(
            header_name='Colore barre:',
            initial_color='#22d36a',
            command=self.update_color_a,
        )
        # Gradiente: due colori
        self.bars_grad_base = ColorPickerFrame(
            header_name='Colore base:',
            initial_color='#3b6bff',
            command=self.update_color_a,
        )
        self.bars_grad_top = ColorPickerFrame(
            header_name='Colore cima:',
            initial_color='#ff3bd0',
            command=self.update_color_b,
        )
        return self._page(
            self._group("Modalità colore", self.bars_mode_option),
            self._group("Classico", self.bars_yellow_thr, self.bars_red_thr),
            self._group("Tinta unica", self.bars_solid_color),
            self._group("Gradiente", self.bars_grad_base, self.bars_grad_top),
        )

    def _build_shape_page(self) -> QWidget:
        self.bars_width_slider = SliderCustomFrame(
            header_name='Larghezza barre:',
            command=self.update_bar_width,
            from_=0.1, to=1.0,
            initial_value=self.bars_width,
        )
        self.bars_rounded_option = OptionMenuCustomFrame(
            header_name='Cime arrotondate:',
            values=["No", "Sì"],
            initial_value="No",
            command=self.update_rounded,
        )
        self.bars_anchor_option = OptionMenuCustomFrame(
            header_name='Ancoraggio:',
            values=["Dal basso", "Centro"],
            initial_value="Dal basso",
            command=self.update_anchor,
        )
        self.bars_order_option = OptionMenuCustomFrame(
            header_name='Ordine bande:',
            values=["Standard", "Simmetrico"],
            initial_value="Standard",
            command=self.update_band_order,
        )
        # Radiale
        self.viz_inner_radius_slider = SliderCustomFrame(
            header_name='Raggio foro:',
            command=self.update_inner_radius,
            from_=0.0, to=0.8,
            initial_value=self.viz_inner_radius,
        )
        # Oscilloscopio / Linea-Area
        self.viz_thickness_slider = SliderCustomFrame(
            header_name='Spessore linea:',
            command=self.update_line_thickness,
            from_=0.002, to=0.06,
            initial_value=self.viz_line_thickness,
        )
        # Linea/Area
        self.viz_fill_option = OptionMenuCustomFrame(
            header_name='Riempimento:', values=["Linea", "Area"], initial_value="Area",
            command=self.update_fill)
        # Oscilloscopio
        self.viz_osc_mirror_option = OptionMenuCustomFrame(
            header_name='Specchiato:', values=["No", "Sì"], initial_value="No",
            command=self.update_osc_mirror)
        self.viz_osc_smooth_slider = SliderCustomFrame(
            header_name='Fluidità (oscill.):',
            command=self.update_osc_smoothing,
            from_=0.0, to=1.0,
            initial_value=self.viz_osc_smoothing,
        )
        return self._page(
            self._group(
                "Forma barre",
                self.bars_width_slider, self.bars_rounded_option,
                self.bars_anchor_option, self.bars_order_option,
            ),
            self._group("Radiale", self.viz_inner_radius_slider),
            self._group(
                "Linea / Oscilloscopio",
                self.viz_thickness_slider, self.viz_fill_option,
                self.viz_osc_mirror_option, self.viz_osc_smooth_slider,
            ),
        )

    def _build_effects_page(self) -> QWidget:
        # Peak-cap (solo modalità Barre: il gruppo viene nascosto altrove)
        self.fx_peakcap_option = OptionMenuCustomFrame(
            header_name='Peak-cap:', values=["Off", "On"], initial_value="Off",
            command=self.update_peakcap_enabled)
        self.fx_peakcap_color_picker = ColorPickerFrame(
            header_name='Colore cap:', initial_color='#ffffff', command=self.update_peakcap_color)
        self.fx_peakcap_hold_slider = SliderCustomFrame(
            header_name='Hold (ms):', command=self.update_peakcap_hold,
            from_=0, to=2000, initial_value=int(self.fx_peakcap_hold * 1000),
            value_type=SliderCustomFrame.ValueType.INT)
        self.fx_peakcap_fall_slider = SliderCustomFrame(
            header_name='Caduta (/s):', command=self.update_peakcap_fall,
            from_=0.1, to=3.0, initial_value=self.fx_peakcap_fall,
            value_type=SliderCustomFrame.ValueType.DOUBLE)

        self.fx_bloom_option = OptionMenuCustomFrame(
            header_name='Glow:', values=["Off", "On"], initial_value="Off",
            command=self.update_bloom_enabled)
        self.fx_bloom_slider = SliderCustomFrame(
            header_name='Intensità:', command=self.update_bloom_intensity,
            from_=0.0, to=3.0, initial_value=self.fx_bloom_intensity,
            value_type=SliderCustomFrame.ValueType.DOUBLE)

        self.fx_beat_option = OptionMenuCustomFrame(
            header_name='Beat pulse:', values=["Off", "On"], initial_value="Off",
            command=self.update_beat_enabled)
        self.fx_beat_intensity_slider = SliderCustomFrame(
            header_name='Intensità zoom:', command=self.update_beat_intensity,
            from_=0.0, to=0.3, initial_value=self.fx_beat_intensity,
            value_type=SliderCustomFrame.ValueType.DOUBLE)
        self.fx_beat_sens_slider = SliderCustomFrame(
            header_name='Sensibilità:', command=self.update_beat_sensitivity,
            from_=1.05, to=3.0, initial_value=self.fx_beat_sensitivity,
            value_type=SliderCustomFrame.ValueType.DOUBLE)

        return self._page(
            self._group(
                "Peak-cap",
                self.fx_peakcap_option, self.fx_peakcap_color_picker,
                self.fx_peakcap_hold_slider, self.fx_peakcap_fall_slider,
            ),
            self._group("Glow (bloom)", self.fx_bloom_option, self.fx_bloom_slider),
            self._group(
                "Beat pulse",
                self.fx_beat_option, self.fx_beat_intensity_slider, self.fx_beat_sens_slider,
            ),
        )

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
        """Alla selezione di un dispositivo avvia la cattura e abilita l'avvio del fullscreen."""
        row = self.device_listbox.row(item)
        if row < 0:
            return
        device = self.devices[row]

        if self.last_device_selected != device:
            self.stop_audio_thread()

            # Riusa la STESSA coda: il renderer in esecuzione ne tiene il
            # riferimento ricevuto alla costruzione — ricrearla lo lascerebbe
            # per sempre su una coda morta (barre congelate). Drena il residuo
            # del vecchio device.
            if self.audio_queue is None:
                self.audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
            else:
                while True:
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        break
            self.audio_thread = AudioCaptureThread(
                self.audio_queue, device=device,
                on_error=lambda msg: self._audio_error.emit(msg),
            )
            self.audio_thread.start()
            self.last_device_selected = device

            # Abilita l'avvio del fullscreen (verde), se non è già in esecuzione
            self.fullscreen_btn.setEnabled(True)
            if not self.equalizer_opengl_thread:
                self.fullscreen_btn.update_base_colors(fg_color=COLOR_START, hover_color=COLOR_START_HOVER)

    def stop_audio_thread(self):
        if self.audio_thread:
            self.audio_thread.stop()
            # Join limitato: un record() bloccato (Linux, silenzio di sistema)
            # non deve congelare la GUI. Il thread è daemon: se non esce qui,
            # muore comunque col processo.
            self.audio_thread.join(timeout=2.0)
            self.audio_thread = None

    # --- Sfondo --------------------------------------------------------------
    def apply_bg_command(self):
        """
        Applica lo sfondo selezionato (immagine o video). Discrimina per
        estensione: lo sfondo configura il renderer OpenGL (fullscreen).
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
        """Carica un'immagine come sfondo del renderer OpenGL."""
        try:
            self.bg_video_path = None  # torna alla modalità immagine
            self.bg_img = Image.open(filename)
            self.bg_image_path = filename
            self.update_bg_opengl(self.bg_img)
        except FileNotFoundError:
            QMessageBox.critical(self, 'Errore', 'File non trovato!')
        except Exception as e:
            QMessageBox.critical(self, 'Errore', f"Errore nel caricamento dell'immagine: {e}")

    def _apply_video_background(self, filename: str):
        """Seleziona un video come sfondo: se il fullscreen è aperto parte subito."""
        if not os.path.isfile(filename):
            QMessageBox.critical(self, 'Errore', 'File non trovato!')
            return
        self.bg_video_path = filename
        self.update_video_opengl(filename)

    # --- Messaggi di controllo verso il renderer -----------------------------
    def update_noise_threshold(self, value):
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_noise_threshold", "value": float(value)})

    def update_frequency_bands(self, value):
        if self.equalizer_opengl_thread:
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
        self._refresh_group_visibility()
        self.bars_color_mode = _COLOR_MODE_TO_INT.get(label, 0)
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_color_mode", "value": self.bars_color_mode})

    def on_viz_mode_changed(self, label: str):
        """Mostra solo i controlli di forma pertinenti alla modalità e inoltra la scelta."""
        bars = label == "Barre"
        radial = label == "Radiale"
        osc = label == "Oscilloscopio"
        line = label == "Linea/Area"
        # Forma (pagina "Forma")
        self.bars_width_slider.setVisible(bars or radial)
        self.bars_rounded_option.setVisible(bars)
        self.bars_anchor_option.setVisible(bars)
        self.bars_order_option.setVisible(bars or radial or line)
        self.viz_inner_radius_slider.setVisible(radial)
        self.viz_thickness_slider.setVisible(osc or line)
        self.viz_fill_option.setVisible(line)
        self.viz_osc_mirror_option.setVisible(osc)
        self.viz_osc_smooth_slider.setVisible(osc)
        # Peak-cap: solo in modalità Barre (pagina "Effetti")
        self.fx_peakcap_option.setVisible(bars)
        self.fx_peakcap_color_picker.setVisible(bars)
        self.fx_peakcap_hold_slider.setVisible(bars)
        self.fx_peakcap_fall_slider.setVisible(bars)
        self._refresh_group_visibility()
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

    def update_menu_toggle_key(self, label: str):
        self.menu_toggle_key = label
        if self.equalizer_opengl_thread:
            self.equalizer_control_queue.put({"type": "set_menu_toggle_key", "value": label})

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
        self.update_alpha_opengl(value)

    # --- Tema / monitor ------------------------------------------------------
    def change_appearance_mode_event(self, new_appearance_mode: str):
        """Cambia il tema dell'applicazione tramite qdarktheme.

        Ri-passa accento di brand + QSS: setup_theme rigenera l'intero stylesheet,
        quindi senza theme_kwargs() lo strato personalizzato andrebbe perso.
        """
        qdarktheme.setup_theme(_THEME_MODE.get(new_appearance_mode, "auto"), **theme_kwargs())

    def change_selected_monitor(self, selected_monitor_name: str) -> None:
        self.selected_monitor = selected_monitor_name

    # --- Fullscreen OpenGL ---------------------------------------------------
    def _set_fullscreen_button_running(self, running: bool):
        """Riflette lo stato del fullscreen sul pulsante (Ferma/rosso vs Avvia/verde)."""
        if running:
            self.fullscreen_btn.update_base_colors(fg_color=COLOR_STOP, hover_color=COLOR_STOP_HOVER)
            self.fullscreen_btn.setText("Ferma")
        else:
            self.fullscreen_btn.update_base_colors(fg_color=COLOR_START, hover_color=COLOR_START_HOVER)
            self.fullscreen_btn.setText("Avvia a schermo intero")

    def fullscreen_command(self):
        """Avvia o ferma la visualizzazione fullscreen OpenGL."""
        if not self.audio_thread:
            self.show_no_audio_thread_warning()
            return

        if self.equalizer_opengl_thread:
            self.equalizer_opengl_thread.stop()
            # Join limitato: con lo stop non più inghiottibile (fix C1) il
            # teardown è rapido; il limite protegge comunque la GUI.
            self.equalizer_opengl_thread.join(timeout=5.0)
            self.equalizer_opengl_thread = None
            self._set_fullscreen_button_running(False)
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
                menu_toggle_key=self.menu_toggle_key,
                on_setting_changed=self.inwindow_setting_changed,
                window_close_callback=self.opengl_window_closed,
            )
            self.equalizer_opengl_thread.start()
            self._set_fullscreen_button_running(True)

    def opengl_window_closed(self):
        # Chiamato sul thread GLFW: marshalla sul thread GUI via segnale.
        self._opengl_closed.emit()

    def inwindow_setting_changed(self, message: dict):
        # Chiamato sul thread GL dall'overlay ImGui: marshalla sul thread GUI via
        # segnale (consegnato in coda), come opengl_window_closed → _opengl_closed.
        self._setting_changed.emit(message)

    def _on_opengl_closed(self):
        # Il callback parte a teardown GLFW completato, quindi il join è quasi
        # istantaneo. Idempotente: se lo stop è partito dalla GUI (fullscreen_
        # command/closeEvent) il riferimento è già None e resta solo il bottone.
        thread = self.equalizer_opengl_thread
        if thread is not None:
            thread.join(timeout=2.0)
            self.equalizer_opengl_thread = None
        self._set_fullscreen_button_running(False)

    def _on_audio_error(self, message: str) -> None:
        """Slot (thread GUI): la cattura audio è morta. Azzera lo stato così
        riselezionare lo STESSO device fa ripartire la cattura (senza questo
        reset la guardia last_device_selected renderebbe il click un no-op)."""
        self.audio_thread = None
        self.last_device_selected = None
        QMessageBox.warning(
            self, "Audio",
            f"La cattura audio si è interrotta: {message}\n"
            "Riseleziona il dispositivo per riprovare.",
        )

    def _on_setting_changed(self, message: dict) -> None:
        """
        Slot (thread GUI): una modifica fatta dal menù in-window torna qui e aggiorna
        il widget + lo stato self.* corrispondente, SENZA ri-mettere il messaggio sul
        control queue (il renderer ha già applicato il valore → nessun loop). I setter
        dei widget (set_value/set_color) bloccano i segnali, quindi non riscattano le
        callback update_*. Per i due combo con visibilità (viz/color) richiamiamo le
        on_*_changed: rimettono una sola volta lo stesso messaggio in coda (echo innocuo,
        il renderer non genera on_change → niente loop).
        """
        t = message["type"]
        v = message["value"]

        # --- Audio (solo widget: il costruttore li rilegge dai widget) ---
        if t == "set_noise_threshold":
            self.noise_slider.set_value(v)
        elif t == "set_frequency_bands":
            self.frequency_slider.set_value(v)
        elif t == "set_db_floor":
            self.adv_db_floor_slider.set_value(v)
        elif t == "set_db_ceiling":
            self.adv_db_ceiling_slider.set_value(v)
        elif t == "set_attack_tau":
            self.adv_attack_slider.set_value(v * 1000.0)   # s → ms
        elif t == "set_release_tau":
            self.adv_release_slider.set_value(v * 1000.0)  # s → ms
        elif t == "set_tilt_db_per_oct":
            self.adv_tilt_slider.set_value(v)

        # --- Visualizzazione ---
        elif t == "set_alpha":
            self.bg_alpha = float(v); self.alpha_slider.set_value(self.bg_alpha)
        elif t == "set_bars_alpha":
            self.bars_alpha = float(v); self.bars_alpha_slider.set_value(self.bars_alpha)
        elif t == "set_viz_mode":
            label = _INT_TO_VIZ_MODE.get(int(v), "Barre")
            self.viz_mode_option.set_value(label)   # silenzioso (blockSignals)
            self.on_viz_mode_changed(label)          # visibilità + self.viz_mode (+ echo)

        # --- Colore ---
        elif t == "set_color_mode":
            label = _INT_TO_COLOR_MODE.get(int(v), "Classico")
            self.bars_mode_option.set_value(label)
            self.on_color_mode_changed(label)        # visibilità + self.bars_color_mode (+ echo)
        elif t == "set_color_a":
            self.bars_color_a = tuple(v)
            self.bars_solid_color.set_color(self.bars_color_a)
            self.bars_grad_base.set_color(self.bars_color_a)
        elif t == "set_color_b":
            self.bars_color_b = tuple(v)
            self.bars_grad_top.set_color(self.bars_color_b)
        elif t == "set_green_split":
            self.bars_green_split = float(v); self.bars_yellow_thr.set_value(self.bars_green_split)
        elif t == "set_yellow_split":
            self.bars_yellow_split = float(v); self.bars_red_thr.set_value(self.bars_yellow_split)

        # --- Forma ---
        elif t == "set_bar_width":
            self.bars_width = float(v); self.bars_width_slider.set_value(self.bars_width)
        elif t == "set_rounded":
            self.bars_rounded = bool(v); self.bars_rounded_option.set_value("Sì" if self.bars_rounded else "No")
        elif t == "set_bar_anchor":
            self.bars_anchor_center = bool(v)
            self.bars_anchor_option.set_value("Centro" if self.bars_anchor_center else "Dal basso")
        elif t == "set_band_order":
            self.bars_band_symmetric = bool(v)
            self.bars_order_option.set_value("Simmetrico" if self.bars_band_symmetric else "Standard")
        elif t == "set_inner_radius":
            self.viz_inner_radius = float(v); self.viz_inner_radius_slider.set_value(self.viz_inner_radius)
        elif t == "set_line_thickness":
            self.viz_line_thickness = float(v); self.viz_thickness_slider.set_value(self.viz_line_thickness)
        elif t == "set_fill":
            self.viz_fill = bool(v); self.viz_fill_option.set_value("Area" if self.viz_fill else "Linea")
        elif t == "set_osc_mirror":
            self.viz_osc_mirror = bool(v); self.viz_osc_mirror_option.set_value("Sì" if self.viz_osc_mirror else "No")
        elif t == "set_osc_smoothing":
            self.viz_osc_smoothing = float(v); self.viz_osc_smooth_slider.set_value(self.viz_osc_smoothing)

        # --- Effetti ---
        elif t == "set_peakcap_enabled":
            self.fx_peakcap_enabled = bool(v); self.fx_peakcap_option.set_value("On" if self.fx_peakcap_enabled else "Off")
        elif t == "set_peakcap_color":
            self.fx_peakcap_color = tuple(v); self.fx_peakcap_color_picker.set_color(self.fx_peakcap_color)
        elif t == "set_peakcap_hold":
            self.fx_peakcap_hold = float(v); self.fx_peakcap_hold_slider.set_value(self.fx_peakcap_hold * 1000.0)  # s → ms
        elif t == "set_peakcap_fall":
            self.fx_peakcap_fall = float(v); self.fx_peakcap_fall_slider.set_value(self.fx_peakcap_fall)
        elif t == "set_bloom_enabled":
            self.fx_bloom_enabled = bool(v); self.fx_bloom_option.set_value("On" if self.fx_bloom_enabled else "Off")
        elif t == "set_bloom_intensity":
            self.fx_bloom_intensity = float(v); self.fx_bloom_slider.set_value(self.fx_bloom_intensity)
        elif t == "set_beat_enabled":
            self.fx_beat_enabled = bool(v); self.fx_beat_option.set_value("On" if self.fx_beat_enabled else "Off")
        elif t == "set_beat_intensity":
            self.fx_beat_intensity = float(v); self.fx_beat_intensity_slider.set_value(self.fx_beat_intensity)
        elif t == "set_beat_sensitivity":
            self.fx_beat_sensitivity = float(v); self.fx_beat_sens_slider.set_value(self.fx_beat_sensitivity)
        elif t == "set_menu_toggle_key":
            self.menu_toggle_key = str(v); self.menu_toggle_option.set_value(self.menu_toggle_key)

    # --- Help / avvisi -------------------------------------------------------
    def open_help_window(self):
        if self.help_window is None:
            self.help_window = HelpWindow(self)
        self.help_window.show()
        self.help_window.raise_()
        self.help_window.activateWindow()

    def show_no_audio_thread_warning(self):
        QMessageBox.information(self, 'Info', 'Select a device first!')

    # --- Profili -------------------------------------------------------------
    def _collect_profile(self) -> dict:
        """Raccoglie lo stato corrente in un dict-profilo (unità native del renderer)."""
        if self.bg_video_path:
            bg = {"type": "video", "ref": profiles.encode_bg_ref(self.bg_video_path, self.resource_manager)}
        else:
            bg = {"type": "image", "ref": profiles.encode_bg_ref(self.bg_image_path, self.resource_manager)}
        return {
            "noise_threshold": float(self.noise_slider.get_value()),
            "n_bands": int(self.frequency_slider.get_value()),
            "db_floor": int(self.adv_db_floor_slider.get_value()),
            "db_ceiling": int(self.adv_db_ceiling_slider.get_value()),
            "attack_tau": self.adv_attack_slider.get_value() / 1000.0,
            "release_tau": self.adv_release_slider.get_value() / 1000.0,
            "tilt_db_per_oct": float(self.adv_tilt_slider.get_value()),
            "viz_mode": int(self.viz_mode),
            "bg_alpha": float(self.bg_alpha),
            "bars_alpha": float(self.bars_alpha),
            "background": bg,
            "color_mode": int(self.bars_color_mode),
            "color_a": list(self.bars_color_a),
            "color_b": list(self.bars_color_b),
            "green_split": float(self.bars_green_split),
            "yellow_split": float(self.bars_yellow_split),
            "bar_width": float(self.bars_width),
            "rounded": bool(self.bars_rounded),
            "anchor_center": bool(self.bars_anchor_center),
            "band_symmetric": bool(self.bars_band_symmetric),
            "inner_radius": float(self.viz_inner_radius),
            "line_thickness": float(self.viz_line_thickness),
            "fill": bool(self.viz_fill),
            "osc_mirror": bool(self.viz_osc_mirror),
            "osc_smoothing": float(self.viz_osc_smoothing),
            "menu_toggle_key": self.menu_toggle_key,
            "peakcap_enabled": bool(self.fx_peakcap_enabled),
            "peakcap_color": list(self.fx_peakcap_color),
            "peakcap_hold": float(self.fx_peakcap_hold),
            "peakcap_fall": float(self.fx_peakcap_fall),
            "bloom_enabled": bool(self.fx_bloom_enabled),
            "bloom_intensity": float(self.fx_bloom_intensity),
            "beat_enabled": bool(self.fx_beat_enabled),
            "beat_intensity": float(self.fx_beat_intensity),
            "beat_sensitivity": float(self.fx_beat_sensitivity),
        }

    def _apply_profile(self, s: dict) -> None:
        """Applica un dict-profilo: widget + stato self.* + (se il renderer gira) control message."""
        push = self.equalizer_opengl_thread is not None
        q = self.equalizer_control_queue
        viz_label = _INT_TO_VIZ_MODE.get(int(s["viz_mode"]), "Barre")
        color_label = _INT_TO_COLOR_MODE.get(int(s["color_mode"]), "Classico")

        # --- Audio ---
        self.noise_slider.set_value(s["noise_threshold"])
        self.frequency_slider.set_value(s["n_bands"])
        if push:
            q.put({"type": "set_noise_threshold", "value": float(s["noise_threshold"])})
            q.put({"type": "set_frequency_bands", "value": int(s["n_bands"])})

        # --- DSP avanzato (slider in ms, profilo in s) ---
        self.adv_db_floor_slider.set_value(s["db_floor"])
        self.adv_db_ceiling_slider.set_value(s["db_ceiling"])
        self.adv_attack_slider.set_value(s["attack_tau"] * 1000.0)
        self.adv_release_slider.set_value(s["release_tau"] * 1000.0)
        self.adv_tilt_slider.set_value(s["tilt_db_per_oct"])
        if push:
            q.put({"type": "set_db_floor", "value": float(s["db_floor"])})
            q.put({"type": "set_db_ceiling", "value": float(s["db_ceiling"])})
            q.put({"type": "set_attack_tau", "value": float(s["attack_tau"])})
            q.put({"type": "set_release_tau", "value": float(s["release_tau"])})
            q.put({"type": "set_tilt_db_per_oct", "value": float(s["tilt_db_per_oct"])})

        # --- Visualizzazione (viz_mode lo finalizza on_viz_mode_changed in fondo) ---
        self.bg_alpha = float(s["bg_alpha"]); self.alpha_slider.set_value(self.bg_alpha)
        self.bars_alpha = float(s["bars_alpha"]); self.bars_alpha_slider.set_value(self.bars_alpha)
        self.viz_mode_option.set_value(viz_label)
        self.menu_toggle_key = str(s.get("menu_toggle_key", "F1"))
        self.menu_toggle_option.set_value(self.menu_toggle_key)
        if push:
            q.put({"type": "set_alpha", "value": self.bg_alpha})
            q.put({"type": "set_bars_alpha", "value": self.bars_alpha})
            q.put({"type": "set_menu_toggle_key", "value": self.menu_toggle_key})

        # --- Colore (color_mode lo finalizza on_color_mode_changed in fondo) ---
        self.bars_mode_option.set_value(color_label)
        self.bars_color_a = tuple(s["color_a"]); self.bars_color_b = tuple(s["color_b"])
        self.bars_solid_color.set_color(self.bars_color_a)
        self.bars_grad_base.set_color(self.bars_color_a)
        self.bars_grad_top.set_color(self.bars_color_b)
        self.bars_green_split = float(s["green_split"]); self.bars_yellow_thr.set_value(self.bars_green_split)
        self.bars_yellow_split = float(s["yellow_split"]); self.bars_red_thr.set_value(self.bars_yellow_split)
        if push:
            q.put({"type": "set_color_a", "value": self.bars_color_a})
            q.put({"type": "set_color_b", "value": self.bars_color_b})
            q.put({"type": "set_green_split", "value": self.bars_green_split})
            q.put({"type": "set_yellow_split", "value": self.bars_yellow_split})

        # --- Forma ---
        self.bars_width = float(s["bar_width"]); self.bars_width_slider.set_value(self.bars_width)
        self.bars_rounded = bool(s["rounded"]); self.bars_rounded_option.set_value("Sì" if self.bars_rounded else "No")
        self.bars_anchor_center = bool(s["anchor_center"]); self.bars_anchor_option.set_value("Centro" if self.bars_anchor_center else "Dal basso")
        self.bars_band_symmetric = bool(s["band_symmetric"]); self.bars_order_option.set_value("Simmetrico" if self.bars_band_symmetric else "Standard")
        self.viz_inner_radius = float(s["inner_radius"]); self.viz_inner_radius_slider.set_value(self.viz_inner_radius)
        self.viz_line_thickness = float(s["line_thickness"]); self.viz_thickness_slider.set_value(self.viz_line_thickness)
        self.viz_fill = bool(s["fill"]); self.viz_fill_option.set_value("Area" if self.viz_fill else "Linea")
        self.viz_osc_mirror = bool(s["osc_mirror"]); self.viz_osc_mirror_option.set_value("Sì" if self.viz_osc_mirror else "No")
        self.viz_osc_smoothing = float(s["osc_smoothing"]); self.viz_osc_smooth_slider.set_value(self.viz_osc_smoothing)
        if push:
            q.put({"type": "set_bar_width", "value": self.bars_width})
            q.put({"type": "set_rounded", "value": self.bars_rounded})
            q.put({"type": "set_bar_anchor", "value": self.bars_anchor_center})
            q.put({"type": "set_band_order", "value": self.bars_band_symmetric})
            q.put({"type": "set_inner_radius", "value": self.viz_inner_radius})
            q.put({"type": "set_line_thickness", "value": self.viz_line_thickness})
            q.put({"type": "set_fill", "value": self.viz_fill})
            q.put({"type": "set_osc_mirror", "value": self.viz_osc_mirror})
            q.put({"type": "set_osc_smoothing", "value": self.viz_osc_smoothing})

        # --- Effetti (peakcap_hold: slider ms, profilo s) ---
        self.fx_peakcap_enabled = bool(s["peakcap_enabled"]); self.fx_peakcap_option.set_value("On" if self.fx_peakcap_enabled else "Off")
        self.fx_peakcap_color = tuple(s["peakcap_color"]); self.fx_peakcap_color_picker.set_color(self.fx_peakcap_color)
        self.fx_peakcap_hold = float(s["peakcap_hold"]); self.fx_peakcap_hold_slider.set_value(self.fx_peakcap_hold * 1000.0)
        self.fx_peakcap_fall = float(s["peakcap_fall"]); self.fx_peakcap_fall_slider.set_value(self.fx_peakcap_fall)
        self.fx_bloom_enabled = bool(s["bloom_enabled"]); self.fx_bloom_option.set_value("On" if self.fx_bloom_enabled else "Off")
        self.fx_bloom_intensity = float(s["bloom_intensity"]); self.fx_bloom_slider.set_value(self.fx_bloom_intensity)
        self.fx_beat_enabled = bool(s["beat_enabled"]); self.fx_beat_option.set_value("On" if self.fx_beat_enabled else "Off")
        self.fx_beat_intensity = float(s["beat_intensity"]); self.fx_beat_intensity_slider.set_value(self.fx_beat_intensity)
        self.fx_beat_sensitivity = float(s["beat_sensitivity"]); self.fx_beat_sens_slider.set_value(self.fx_beat_sensitivity)
        if push:
            q.put({"type": "set_peakcap_enabled", "value": self.fx_peakcap_enabled})
            q.put({"type": "set_peakcap_color", "value": self.fx_peakcap_color})
            q.put({"type": "set_peakcap_hold", "value": self.fx_peakcap_hold})
            q.put({"type": "set_peakcap_fall", "value": self.fx_peakcap_fall})
            q.put({"type": "set_bloom_enabled", "value": self.fx_bloom_enabled})
            q.put({"type": "set_bloom_intensity", "value": self.fx_bloom_intensity})
            q.put({"type": "set_beat_enabled", "value": self.fx_beat_enabled})
            q.put({"type": "set_beat_intensity", "value": self.fx_beat_intensity})
            q.put({"type": "set_beat_sensitivity", "value": self.fx_beat_sensitivity})

        # --- Sfondo (safe fallback) ---
        self._apply_background_from_profile(s["background"])

        # --- Visibilità condizionale + finalizzazione viz/color mode ---
        # (le callback non scattano sul set programmatico dei combo; le invochiamo
        #  ora: aggiornano visibilità, self.viz_mode/self.bars_color_mode e, se il
        #  renderer gira, inviano set_viz_mode/set_color_mode)
        self.on_color_mode_changed(color_label)
        self.on_viz_mode_changed(viz_label)

    def _apply_background_from_profile(self, bg: dict) -> None:
        """Risolve e applica lo sfondo del profilo, con safe fallback all'immagine bundle."""
        ref = bg.get("ref", {}) if isinstance(bg, dict) else {}
        bg_type = bg.get("type", "image") if isinstance(bg, dict) else "image"
        path, ok = profiles.resolve_bg_ref(ref, self.resource_manager)

        if bg_type == "video" and ok:
            self.bg_video_path = path
            self.file_picker.set_filename(path)
            if self.equalizer_opengl_thread:
                self.equalizer_control_queue.put({"type": "set_video", "value": path})
        else:
            # immagine, oppure video mancante → fallback immagine
            self.bg_video_path = None
            self.bg_image_path = path
            try:
                self.bg_img = Image.open(path)
            except Exception:
                ok = False
            self.file_picker.set_filename(path)
            if self.equalizer_opengl_thread:
                self.equalizer_control_queue.put({"type": "set_image", "value": self.bg_img})

        if not ok:
            ref_desc = ref.get("value") or ref.get("name") or str(ref)
            QMessageBox.warning(
                self, "Sfondo non trovato",
                f"Sfondo non trovato: {ref_desc}. Uso lo sfondo predefinito.",
            )

    def _on_profile_selected(self, name) -> None:
        if not name:
            return
        try:
            settings = self._profile_store.load(name)
            self._apply_profile(settings)
        except Exception as e:
            QMessageBox.warning(self, "Profilo", f"Impossibile caricare il profilo: {e}")
            return
        self._profile_store.set_last(name)

    def _on_profile_save(self) -> None:
        name = self.profile_bar.current_name()
        if not name:
            self._on_profile_save_as()
            return
        self._profile_store.save(name, self._collect_profile())
        self._profile_store.set_last(name)

    def _on_profile_save_as(self) -> None:
        name, accepted = QInputDialog.getText(self, "Salva profilo", "Nome del profilo:")
        if not accepted or not name.strip():
            return
        saved = profiles.sanitize_name(name)
        # Il confronto va fatto sul nome SANITIZZATO: nomi diversi possono
        # collidere sullo stesso file (es. "a/b" → "a_b").
        if saved in self._profile_store.list_names():
            confirm = QMessageBox.question(
                self, "Salva profilo",
                f"Il profilo «{saved}» esiste già. Sovrascriverlo?",
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
        self._profile_store.save(name, self._collect_profile())
        self._refresh_profile_bar(selected=saved)
        self._profile_store.set_last(saved)

    def _on_profile_delete(self) -> None:
        name = self.profile_bar.current_name()
        if not name:
            return
        confirm = QMessageBox.question(
            self, "Elimina profilo", f"Eliminare il profilo «{name}»?"
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._profile_store.delete(name)
        self._refresh_profile_bar()

    def _on_profile_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Importa profilo", "", "Profili Sound Wave (*.json);;Tutti i file (*)"
        )
        if not path:
            return
        try:
            name = self._profile_store.import_file(path)
            self._refresh_profile_bar(selected=name)
            self._apply_profile(self._profile_store.load(name))
        except Exception as e:
            QMessageBox.warning(self, "Importa profilo", f"File non valido: {e}")
            return
        self._profile_store.set_last(name)

    def _on_profile_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta profilo", "profilo.json", "Profili Sound Wave (*.json)"
        )
        if not path:
            return
        name = self.profile_bar.current_name()
        try:
            if name:
                self._profile_store.export(name, path)
            else:
                self._profile_store.write_envelope(self._collect_profile(), path)
        except Exception as e:
            QMessageBox.warning(self, "Esporta profilo", f"Impossibile esportare: {e}")

    # --- Ciclo di vita -------------------------------------------------------
    def closeEvent(self, event):
        """
        Alla chiusura della finestra ferma con grazia tutti i thread avviati.
        """
        if self.audio_thread:
            self.audio_thread.stop()
            self.audio_thread.join(timeout=2.0)

        if self.equalizer_opengl_thread:
            self.equalizer_opengl_thread.stop()
            self.equalizer_opengl_thread.join(timeout=5.0)

        event.accept()
