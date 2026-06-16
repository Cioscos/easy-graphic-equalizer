import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor

import customtkinter as ctk
import tkinter as tk
from CTkMessagebox import CTkMessagebox
import soundcard as sc
from PIL import Image, ImageTk
import glfw

from thread.audioCaptureThread import AudioCaptureThread
from thread.equalizer_tkinter_thread import EqualizerTkinterThread
from thread.opengl_thread import EqualizerOpenGLThread
from gui.slider_frame import SliderCustomFrame
from gui.optionmenu_frame import OptionMenuCustomFrame
from gui.background_filepicker_frame import BackgroundFilepickerFrame
from gui.help_window import HelpWindow
from gui.animated_button import AnimatedButton
from gui.theme import (
    FONT_TITLE, FONT_BUTTON, FONT_LABEL, FONT_BODY,
    COLOR_NEUTRAL, COLOR_NEUTRAL_HOVER,
    COLOR_FULLSCREEN, COLOR_FULLSCREEN_HOVER,
    COLOR_START, COLOR_START_HOVER,
    COLOR_STOP, COLOR_STOP_HOVER,
    COLOR_LISTBOX_DARK, COLOR_LISTBOX_LIGHT,
)
from resource_manager import ResourceManager

# Coda audio volutamente piccola: il renderer svuota tutta la coda a ogni frame
# tenendo solo l'audio più recente, quindi una coda corta evita backlog/latenza.
MAX_QUEUE_SIZE = 8
INITIAL_FREQUENCIES_BANDS = 9
RESIZE_DEBOUNCE_MS = 500
DOUBLE_LEFT_MOUSE_BUTTON_CLICK = '<Double-Button-1>'
CLOSE_WINDOW_EVENT = 'WM_DELETE_WINDOW'
CONFIGURE_EVENT = '<Configure>'


class AudioCaptureGUI(ctk.CTk):
    """
    The main window which directly extends Ctk class
    """
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode('System')
        ctk.set_default_color_theme('blue')

        self.resource_manager = ResourceManager()
        self.title("Sound wave")
        self.minsize(800, 600)
        self.protocol(CLOSE_WINDOW_EVENT, self.on_close)

        # Stato condiviso inizializzato prima della costruzione dei widget
        self.devices = []
        self.canvas_width = self.canvas_height = None
        self.bg_alpha = 0.5
        self.selected_monitor = None
        self.available_monitors = self.get_available_monitors()

        # Pannello sinistro: selezione device + impostazioni
        left_frame = ctk.CTkFrame(self, border_width=2)
        left_frame.pack(side=ctk.LEFT, padx=10, pady=10, fill=ctk.BOTH)
        self._build_device_panel(left_frame)

        settings_label = ctk.CTkLabel(left_frame, text="Impostazioni", font=FONT_TITLE)
        settings_label.pack(side=tk.TOP, padx=5, pady=5)
        self._build_settings_tabs(left_frame)

        # Pannello destro: canvas dell'equalizzatore + pulsanti
        right_frame = ctk.CTkFrame(self, border_width=2)
        right_frame.pack(side=ctk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        equalizer_frame = ctk.CTkFrame(right_frame, border_width=2)
        equalizer_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._build_canvas(equalizer_frame)

        buttons_frame = ctk.CTkFrame(right_frame)
        buttons_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self._build_action_buttons(buttons_frame)

        # Ricalcola lo sfondo quando la finestra viene ridimensionata
        on_resize_widget = {'equalizer_canvas': self.equalizer_canvas}
        self.bind(CONFIGURE_EVENT, lambda event: self.on_resize(event, on_resize_widget))

        self._init_runtime_state()

    def _build_device_panel(self, parent):
        """
        Costruisce il pannello di selezione del device audio (label + listbox).
        """
        device_frame = ctk.CTkFrame(parent)
        device_frame.pack(padx=5, pady=5, anchor='n', fill=ctk.X)
        ctk.CTkLabel(device_frame, text="Select device:", font=FONT_LABEL).pack(pady=5)

        device_listbox_frame = ctk.CTkFrame(device_frame)
        device_listbox_frame.pack(fill=ctk.BOTH, expand=True)

        self.device_listbox = tk.Listbox(device_listbox_frame, width=40, font=FONT_BODY, bg=COLOR_LISTBOX_DARK)
        self.device_listbox.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)
        scrollbar = ctk.CTkScrollbar(device_listbox_frame, orientation="vertical", command=self.device_listbox.yview)
        scrollbar.pack(side=ctk.LEFT)
        self.device_listbox.config(yscrollcommand=scrollbar.set)

        # Popola la lista dei device
        asyncio.run(self.load_devices())

        self.device_listbox.bind(DOUBLE_LEFT_MOUSE_BUTTON_CLICK, self.on_device_selected)

    def _build_canvas(self, parent):
        """
        Costruisce il canvas dell'equalizzatore e carica l'immagine di sfondo.
        """
        self.equalizer_canvas = ctk.CTkCanvas(parent, bg='#000')

        # Carica l'immagine di sfondo del canvas
        self.bg_img = Image.open(self.resource_manager.get_image_path('bg (19).png', 'bg'))
        self.bg_img_used = self.bg_img.copy()
        self.bg_img_used.putalpha(int(255 * self.bg_alpha))

        # Ridimensiona l'immagine se il canvas è più grande dell'immagine
        self.canvas_width, self.canvas_height = self.get_canvas_size()
        if self.canvas_height > self.bg_img_used.height or self.canvas_width > self.bg_img_used.width:
            self.bg_img_used = self.bg_img_used.resize((self.canvas_width, self.canvas_height), reducing_gap=2.0)

        self.canvas_image = ImageTk.PhotoImage(self.bg_img_used)
        self.equalizer_canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image, tags='background')
        self.equalizer_canvas.tag_lower('background')
        self.equalizer_canvas.pack(fill=tk.BOTH, expand=True)

    def _build_action_buttons(self, parent):
        """
        Costruisce i pulsanti animati Start/Stop, Fullscreen e Help.
        """
        self.start_stop_button = AnimatedButton(
            parent,
            text="Start",
            command=self.start_stop_capture,
            font=FONT_BUTTON,
            fg_color=COLOR_NEUTRAL,  # Grigio per stato disabilitato
            hover_color=COLOR_NEUTRAL_HOVER,
            state='disabled',
            corner_radius=10,
            height=45,
            animation_duration=200,
            hover_scale=1.05,
            click_scale=0.95
        )
        self.start_stop_button.pack(padx=5, pady=5, fill=ctk.BOTH, side=tk.LEFT, expand=True)

        self.fullscreen_icn = ctk.CTkImage(
            dark_image=Image.open(self.resource_manager.get_image_path('maximize.png', 'icons')),
            size=(20, 20)
        )
        self.fullscreen_btn = AnimatedButton(
            parent,
            command=self.fullscreen_command,
            image=self.fullscreen_icn,
            text='Fullscreen',
            font=FONT_BUTTON,
            fg_color=COLOR_FULLSCREEN,
            hover_color=COLOR_FULLSCREEN_HOVER,
            corner_radius=10,
            height=45,
            animation_duration=200,
            hover_scale=1.05,
            click_scale=0.95
        )
        self.fullscreen_btn.pack(padx=5, pady=5, fill=ctk.BOTH, side=tk.LEFT, expand=True)

        help_icn = ctk.CTkImage(
            dark_image=Image.open(self.resource_manager.get_image_path('help.png', 'icons')),
            size=(20, 20)
        )
        help_button = AnimatedButton(
            parent,
            command=self.open_help_window,
            image=help_icn,
            text='Help',
            font=FONT_BUTTON,
            fg_color=COLOR_NEUTRAL,
            hover_color=COLOR_NEUTRAL_HOVER,
            corner_radius=10,
            height=45,
            animation_duration=200,
            hover_scale=1.05,
            click_scale=0.95
        )
        help_button.pack(padx=5, pady=5, fill=ctk.BOTH, side=tk.LEFT, expand=True)

    def _init_runtime_state(self):
        """
        Inizializza code, thread e flag usati a runtime.
        """
        self.audio_queue = None
        self.equalizer_control_queue = queue.Queue()
        self.audio_thread = None
        self.canvas_thread = None
        self.equalizer_opengl_thread = None
        self.last_device_selected = None
        self.resize_scheduled = False
        self.help_window = None
        self.is_capturing = False

    def get_canvas_size(self) -> tuple[int, int]:
        """
        Return the size of the equalizer_canvas
        Returns:
            tuple: width and height of the canvas
        """
        return (self.equalizer_canvas.winfo_width(), self.equalizer_canvas.winfo_height())

    def apply_bg_command(self):
        """
        Applica l'immagine di sfondo selezionata all'equalizzatore.

        Raises:
            FileNotFoundError: Se il file non esiste
            PIL.UnidentifiedImageError: Se il file non è un'immagine valida
        """
        filename = self.file_picker.get_filename()
        if not filename:
            CTkMessagebox(
                title='Errore',
                message='Nessun file selezionato!',
                icon="warning"
            )
            return

        try:
            self.bg_img = Image.open(filename)
            self.bg_img_used = self.bg_img.copy()
            self.bg_img_used.putalpha(int(255 * self.bg_alpha))
            self.apply_background()
            self.update_bg_opengl(self.bg_img)
        except FileNotFoundError:
            CTkMessagebox(
                title='Errore',
                message='File non trovato!',
                icon="error"
            )
        except Exception as e:
            CTkMessagebox(
                title='Errore',
                message=f'Errore nel caricamento dell\'immagine: {str(e)}',
                icon="error"
            )

    def _build_settings_tabs(self, parent):
        """
        Crea il sistema a tab per organizzare le impostazioni.

        Args:
            parent: Il widget padre
        """
        tabview = ctk.CTkTabview(parent, border_width=2)
        tabview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._build_visualization_tab(tabview)
        self._build_audio_tab(tabview)

    def _build_visualization_tab(self, tabview):
        """
        Tab Visualizzazione: tema, immagine di sfondo, trasparenza e
        selettore del monitor per il fullscreen.
        """
        tabview.add("🎨 Visualizzazione")
        vis_tab = tabview.tab("🎨 Visualizzazione")

        # Tema
        self.appearance_mode = OptionMenuCustomFrame(
            vis_tab,
            header_name='Tema:',
            values=["Light", "Dark", "System"],
            initial_value="System",
            command=self.change_appearance_mode_event
        )
        self.appearance_mode.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        # Immagine di sfondo
        self.file_picker = BackgroundFilepickerFrame(
            vis_tab,
            header_name='Immagine di Sfondo:',
            placeholder_text='Percorso immagine...',
            apply_command=self.apply_bg_command
        )
        self.file_picker.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        # Trasparenza
        self.alpha_slider = SliderCustomFrame(
            vis_tab,
            header_name='Trasparenza:',
            command=self.update_alpha,
            initial_value=self.bg_alpha
        )
        self.alpha_slider.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        # Selettore monitor per il fullscreen (solo con più monitor disponibili)
        if len(self.available_monitors) > 1:
            self.monitor_option = OptionMenuCustomFrame(
                vis_tab,
                header_name='Monitor Fullscreen:',
                values=self.available_monitors,
                initial_value=self.available_monitors[0],
                command=self.change_selected_monitor
            )
            self.monitor_option.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

    def _build_audio_tab(self, tabview):
        """
        Tab Audio: soglia di rumore e numero di bande di frequenza.
        """
        tabview.add("🎵 Audio")
        audio_tab = tabview.tab("🎵 Audio")

        # Soglia rumore
        self.noise_slider = SliderCustomFrame(
            audio_tab,
            header_name='Soglia Rumore:',
            command=self.update_noise_threshold
        )
        self.noise_slider.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        # Bande di frequenza
        self.frequency_slider = SliderCustomFrame(
            audio_tab,
            header_name='Bande di Frequenza:',
            command=self.update_frequency_bands,
            from_=1,
            to=100,
            steps=32,
            initial_value=INITIAL_FREQUENCIES_BANDS,
            warning_trigger_value=15,
            warning_text='Numero elevato di bande!\nConsidera la modalità fullscreen',
            value_type=SliderCustomFrame.ValueType.INT
        )
        self.frequency_slider.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

    def apply_background(self):
        self.equalizer_canvas.update_idletasks()
        self.canvas_width, self.canvas_height = self.equalizer_canvas.winfo_width(), self.equalizer_canvas.winfo_height()
        if self.canvas_height != self.bg_img_used.height or self.canvas_width != self.bg_img_used.width:
            # Resize the image using the resize() method
            self.bg_img_used = self.bg_img_used.resize((self.canvas_width, self.canvas_height), reducing_gap=2.0)

        self.canvas_image = ImageTk.PhotoImage(self.bg_img_used)
        self.equalizer_canvas.create_image(0, 0, anchor=ctk.NW, image=self.canvas_image, tags='background')
        self.equalizer_canvas.tag_lower('background')

    async def get_devices(self):
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            devices = await loop.run_in_executor(executor, sc.all_microphones, True)
        return devices

    async def load_devices(self):
        self.devices = await self.get_devices()
        for device in self.devices:
            device = str(device).replace('<', '').replace('>', '')
            self.device_listbox.insert(tk.END, f"{device}")

    def get_available_monitors(self) -> list[str]:
        if not glfw.init():
            raise Exception("GLFW initialization failed")

        monitors = glfw.get_monitors()
        monitors_name = [f"{idx + 1}. {glfw.get_monitor_name(monitor).decode('utf-8')}" for idx, monitor in enumerate(monitors)]

        glfw.terminate()

        return monitors_name

    def on_device_selected(self, event):
        """
        Quando un device è selezionato, abilita il pulsante start/stop.
        """
        selection = self.device_listbox.curselection()
        if selection:
            device_index = selection[0]
            device = self.devices[device_index]

            # Se stiamo già catturando, ferma prima
            if self.is_capturing:
                self.stop_capture()
                self.is_capturing = False

            if self.last_device_selected != device:
                if self.audio_queue:
                    self.audio_queue = None

                self.stop_audio_thread()

                self.audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
                self.audio_thread = AudioCaptureThread(
                    self.audio_queue, device=device)
                self.audio_thread.start()
                self.last_device_selected = device

                # Abilita il pulsante e imposta il colore verde
                self.start_stop_button.configure(state='normal')
                self.start_stop_button.update_base_colors(
                    fg_color=COLOR_START,
                    hover_color=COLOR_START_HOVER
                )
                self.start_stop_button.configure(text="Start")

    def stop_audio_thread(self):
        if self.audio_thread:
            self.audio_thread.stop()
            while self.audio_thread.is_alive():
                self.audio_thread.join(timeout=0.1)

    def update_noise_threshold(self, value):
        """
        Update the noise threshold in the equalizer sending
        a message to the EqualizerTkinterThread
        Args:
            value (ctk.DoubleVar):A DoubleVar object to track the threshold value
        """
        if self.canvas_thread or self.equalizer_opengl_thread:
            message = {
                "type": "set_noise_threshold",
                "value": float(value)
            }
            self.equalizer_control_queue.put(message)

    def update_frequency_bands(self, value):
        if self.canvas_thread or self.equalizer_opengl_thread:
            message = {
                "type": "set_frequency_bands",
                "value": int(value)
            }
            self.equalizer_control_queue.put(message)

    def update_alpha_opengl(self, value):
        if self.equalizer_opengl_thread:
            message = {
                "type": "set_alpha",
                "value": float(value)
            }
            self.equalizer_control_queue.put(message)

    def update_bg_opengl(self, image: Image.Image):
        if self.equalizer_opengl_thread:
            message = {
                "type": "set_image",
                "value": image
            }
            self.equalizer_control_queue.put(message)

    def update_alpha(self, value):
        self.bg_alpha = value
        self.bg_img_used.putalpha(int(255 * self.bg_alpha))
        self.canvas_image = ImageTk.PhotoImage(self.bg_img_used)
        self.equalizer_canvas.create_image(0, 0, anchor=ctk.NW, image=self.canvas_image, tags='background')
        self.equalizer_canvas.tag_lower('background')

        # send msg to opengl window if active
        self.update_alpha_opengl(value)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        """
        Change the tkinter theme
        """
        ctk.set_appearance_mode(new_appearance_mode)

        if ctk.get_appearance_mode().lower() == 'dark':
            self.device_listbox.configure(bg=COLOR_LISTBOX_DARK)
        else:
            self.device_listbox.configure(bg=COLOR_LISTBOX_LIGHT)

    def change_selected_monitor(self, selected_monitor_name: str) -> None:
        self.selected_monitor = selected_monitor_name

    def start_capture(self):
        """
        Start the equalizer thread when the start button is pressed
        """
        if not self.canvas_thread:
            if self.audio_thread and self.audio_thread.is_alive():
                try:
                    # Start the equalizer thread
                    self.canvas_thread = EqualizerTkinterThread(
                        self.audio_queue,
                        noise_threshold=self.noise_slider.get_value(),
                        n_bands=int(self.frequency_slider.get_value()),
                        canvas=self.equalizer_canvas,
                        control_queue=self.equalizer_control_queue)
                    self.canvas_thread.start()
                except Exception:
                    self.canvas_thread = None
            else:
                self.show_no_audio_thread_warning()

    def stop_capture(self):
        """
        Check if the equalizer thread is running and if it is
        gracefully stop it
        """
        if self.canvas_thread and self.canvas_thread.is_alive():
            self.canvas_thread.stop()
            # Aspetta che il thread termini completamente
            while self.canvas_thread.is_alive():
                self.canvas_thread.join(timeout=0.1)
                self.update()  # Mantieni la GUI responsiva

        # Pulisci il canvas dopo aver fermato il thread
        if self.equalizer_canvas:
            self.equalizer_canvas.delete("bar")  # Rimuovi tutte le barre

        self.canvas_thread = None

    def start_stop_capture(self):
        """
        Avvia o ferma la cattura audio.
        """
        if not self.audio_thread or not self.audio_thread.is_alive():
            self.show_no_audio_thread_warning()
            return

        # Verifica che la coda audio sia valida
        if not self.audio_queue:
            return

        if not self.is_capturing:
            # Svuota la coda prima di iniziare
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Avvia la cattura
            self.start_capture()
            # Verifica che sia partito correttamente prima di cambiare stato
            self.after(100, self._check_capture_started)
        else:
            # Ferma la cattura
            self.stop_capture()
            self.is_capturing = False
            # Aggiorna pulsante a verde
            self.start_stop_button.update_base_colors(
                fg_color=COLOR_START,
                hover_color=COLOR_START_HOVER
            )
            self.start_stop_button.configure(text="Start")

    def _check_capture_started(self):
        """
        Verifica che la cattura sia effettivamente partita e aggiorna l'UI.
        """
        if self.canvas_thread and self.canvas_thread.is_alive():
            self.is_capturing = True
            # Aggiorna pulsante a rosso
            self.start_stop_button.update_base_colors(
                fg_color=COLOR_STOP,
                hover_color=COLOR_STOP_HOVER
            )
            self.start_stop_button.configure(text="Stop")
        else:
            self.is_capturing = False

    def fullscreen_command(self):
        """
        Apre la visualizzazione fullscreen.
        """
        # Se stiamo catturando, ferma prima
        if self.is_capturing:
            self.stop_capture()
            self.is_capturing = False
            self.start_stop_button.update_base_colors(
                fg_color=COLOR_START,
                hover_color=COLOR_START_HOVER
            )
            self.start_stop_button.configure(text="Start")

        if self.audio_thread:
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
                    monitor=self.selected_monitor,
                    window_close_callback=self.opengl_window_closed
                )
                self.equalizer_opengl_thread.start()
        else:
            self.show_no_audio_thread_warning()

    def opengl_window_closed(self):
        self.equalizer_opengl_thread = None

    def open_help_window(self):
        if self.help_window is None or not self.help_window.winfo_exists():
            self.help_window = HelpWindow(self)
            self.help_window.focus()
            self.help_window.after(20, self.help_window.lift)
        else:
            self.help_window.focus()

    def show_no_audio_thread_warning(self):
        CTkMessagebox(title='Info', message='Select a device first!')

    def on_resize(self, _, widgets: dict[str, ctk.CTkBaseClass]):
        for widget_name, widget in widgets.items():
            if widget_name == 'equalizer_canvas':
                if not self.resize_scheduled:
                    self.resize_scheduled = True
                    self.after(RESIZE_DEBOUNCE_MS, self.resize_background, widget)

    def resize_background(self, canvas: tk.Canvas):
        self.canvas_width, self.canvas_height = canvas.winfo_width(), canvas.winfo_height()
        # Resize the image using the resize() method starting from the original image
        self.bg_img_used = self.bg_img.resize((self.canvas_width, self.canvas_height), reducing_gap=2.0).copy()
        self.bg_img_used.putalpha(int(255 * self.bg_alpha))
        self.canvas_image = ImageTk.PhotoImage(self.bg_img_used)
        self.equalizer_canvas.create_image(0, 0, anchor=ctk.NW, image=self.canvas_image, tags='background')
        self.equalizer_canvas.tag_lower('background')
        self.resize_scheduled = False

    def run(self):
        """
        Starts the tkinter main loop
        """
        self.mainloop()

    def on_close(self):
        """
        Starts when the tkinter window is closed.
        It gracefully stop all the started threads.
        """
        if self.canvas_thread:
            self.canvas_thread.stop()
            while self.canvas_thread.is_alive():
                self.canvas_thread.join(timeout=0.1)
                # Update the GUI main loop to process events
                # it avoid deadlock between the threads
                self.update()

        # Stop the audio thread and close the window
        if self.audio_thread:
            self.audio_thread.stop()
            while self.audio_thread.is_alive():
                self.audio_thread.join(timeout=0.1)
                self.update()

        if self.equalizer_opengl_thread:
            self.equalizer_opengl_thread.stop()
            while self.equalizer_opengl_thread.is_alive():
                self.equalizer_opengl_thread.join(timeout=0.1)
                self.update()

        self.destroy()
