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
from gui.processes_number_frame import ProcessesNumberFrame
from resource_manager import ResourceManager

MAX_QUEUE_SIZE = 200
INITIAL_FREQUENCIES_BANDS = 9
SINGLE_LEFT_MOUSE_BOTTON_CLICK = '<Button-1>'
DOUBLE_LEFT_MOUSE_BOTTON_CLICK = '<Double-Button-1>'
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

        self.devices = []

        #Initialize the canvas size to none
        self.canvas_width = self.canvas_height = None
        self.bg_alpha = 0.5

        # Create the resource manager
        self.resource_manager = ResourceManager()

        # Create main window title
        self.title("Sound wave")
        self.minsize(800, 600)

        self.protocol(CLOSE_WINDOW_EVENT, self.on_close)

        # create left frame
        left_frame = ctk.CTkFrame(self, border_width=2)
        left_frame.pack(side=ctk.LEFT, padx=10, pady=10, fill=ctk.BOTH)

        # Create device frame
        device_frame = ctk.CTkFrame(left_frame)
        device_frame.pack(padx=5, pady=5, anchor='n', fill=ctk.X)
        ctk.CTkLabel(device_frame, text="Select device:", font=("Roboto", 14)).pack(pady=5)

        # Create the listbox frame
        device_listbox_frame = ctk.CTkFrame(device_frame)
        device_listbox_frame.pack(fill=ctk.BOTH, expand=True)

        self.device_listbox = tk.Listbox(device_listbox_frame, width=40, font=("Roboto", 12), bg="#5a5a5a")
        self.device_listbox.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)
        scrollbar = ctk.CTkScrollbar(device_listbox_frame, orientation="vertical", command=self.device_listbox.yview)
        scrollbar.pack(side=ctk.LEFT)
        self.device_listbox.config(yscrollcommand=scrollbar.set)
        
        # Start the coroutine to retrieve the devices list
        asyncio.run(self.load_devices())

        self.device_listbox.bind(DOUBLE_LEFT_MOUSE_BOTTON_CLICK, self.on_device_selected)

        # Create settings frame
        # Inizializza le variabili necessarie prima di creare i tabs
        self.bg_alpha = 0.5
        self.bg_filename = None
        self.selected_monitor = None
        self.available_monitors = self.get_available_monitors()

        # Create settings tabs invece del settings frame
        settings_label = ctk.CTkLabel(left_frame, text="Impostazioni", font=("Roboto", 18, "bold"))
        settings_label.pack(side=tk.TOP, padx=5, pady=5)

        # Crea il sistema a tabs
        self.settings_tabs = self.create_settings_tabs(left_frame)

        # Create right frame
        right_frame = ctk.CTkFrame(self, border_width=2)
        right_frame.pack(side=ctk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Create equalizer frame and put it into the right frame
        equalizer_frame = ctk.CTkFrame(right_frame, border_width=2)
        equalizer_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create equalizer canvas and put into equalizer_frame
        self.equalizer_canvas = ctk.CTkCanvas(equalizer_frame, bg='#000')

        # load canvas bg image
        self.bg_img = Image.open(self.resource_manager.get_image_path('bg (19).png', 'bg'))
        self.bg_img_used = self.bg_img.copy()
        self.bg_img_used.putalpha(int(255 * self.bg_alpha))

        # Chech image size to understand if it must be resized
        self.canvas_width, self.canvas_height = self.get_canvas_size()

        if self.canvas_height > self.bg_img_used.height or self.canvas_width > self.bg_img_used.width:
            # Resize the image using the resize() method
            self.bg_img_used = self.bg_img_used.resize((self.canvas_width, self.canvas_height), reducing_gap=2.0)

        self.canvas_image = ImageTk.PhotoImage(self.bg_img_used)

        self.equalizer_canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image, tags='background')
        self.equalizer_canvas.tag_lower('background')
        self.equalizer_canvas.pack(fill=tk.BOTH, expand=True)

        # Create buttons frame and put into right frame
        buttons_frame = ctk.CTkFrame(right_frame)
        buttons_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # controls the state of the start/stop button.
        self.is_on_start = True

        # Create start/stop button
        # Create start/stop button animato
        self.start_stop_button = AnimatedButton(
            buttons_frame,
            text="Start",  # Testo iniziale
            command=self.start_stop_capture,
            font=("Roboto", 16),
            fg_color='#6c757d',  # Grigio per stato disabilitato
            hover_color='#5a6268',
            state='disabled',
            corner_radius=10,
            height=45,
            animation_duration=200,
            hover_scale=1.05,
            click_scale=0.95
        )
        self.start_stop_button.pack(padx=5, pady=5, fill=ctk.BOTH, side=tk.LEFT, expand=True)

        # Fullscreen button animato
        self.fullscreen_icn = ctk.CTkImage(
            dark_image=Image.open(self.resource_manager.get_image_path('maximize.png', 'icons')),
            size=(20, 20)
        )
        self.fullscreen_btn = AnimatedButton(
            buttons_frame,
            command=self.fullscreen_command,
            image=self.fullscreen_icn,
            text='Fullscreen',
            font=("Roboto", 16),
            fg_color='#17a2b8',  # Azzurro
            hover_color='#138496',
            corner_radius=10,
            height=45,
            animation_duration=200,
            hover_scale=1.05,
            click_scale=0.95
        )
        self.fullscreen_btn.pack(padx=5, pady=5, fill=ctk.BOTH, side=tk.LEFT, expand=True)

        # Help button animato
        help_icn = ctk.CTkImage(
            dark_image=Image.open(self.resource_manager.get_image_path('help.png', 'icons')),
            size=(20, 20)
        )
        help_button = AnimatedButton(
            buttons_frame,
            command=self.open_help_window,
            image=help_icn,
            text='Help',
            font=("Roboto", 16),
            fg_color='#6c757d',  # Grigio
            hover_color='#5a6268',
            corner_radius=10,
            height=45,
            animation_duration=200,
            hover_scale=1.05,
            click_scale=0.95
        )
        help_button.pack(padx=5, pady=5, fill=ctk.BOTH, side=tk.LEFT, expand=True)

        # BIND THE EVENT
        # Create a dictionary with all the widget to check in the on_resize callback
        on_resize_widget = {'equalizer_canvas': self.equalizer_canvas}

        self.bind(CONFIGURE_EVENT, lambda event: self.on_resize(event, on_resize_widget))

        self.audio_queue = None
        self.equalizer_control_queue = queue.Queue()
        self.audio_thread = None
        self.canvas_thread = None
        self.equalizer_opengl_thread = None
        self.last_device_selected = None
        self.resize_scheduled = False
        self.help_window = None
        self.is_capturing = False

    def debug_state(self):
        """
        Stampa lo stato corrente per debug.
        """
        print(f"""
        === DEBUG STATE ===
        is_capturing: {self.is_capturing}
        audio_thread attivo: {self.audio_thread is not None and self.audio_thread.is_alive()}
        canvas_thread attivo: {self.canvas_thread is not None and self.canvas_thread.is_alive()}
        pulsante abilitato: {self.start_stop_button.cget('state') == 'normal'}
        testo pulsante: {self.start_stop_button.cget('text')}
        ==================
        """)

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

    # gui/main_window_gui.py - aggiungi questo metodo alla classe AudioCaptureGUI

    def create_settings_tabs(self, parent):
        """
        Crea un sistema a tab per organizzare meglio le impostazioni.

        Args:
            parent: Il widget padre

        Returns:
            CTkTabview: Il widget tabview creato
        """
        tabview = ctk.CTkTabview(parent, border_width=2)
        tabview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab Visualizzazione
        tabview.add("🎨 Visualizzazione")
        vis_tab = tabview.tab("🎨 Visualizzazione")

        # Appearance mode
        self.appearance_mode = OptionMenuCustomFrame(
            vis_tab,
            header_name='Tema:',
            values=["Light", "Dark", "System"],
            initial_value="System",
            command=self.change_appearance_mode_event
        )
        self.appearance_mode.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        # Background settings
        self.file_picker = BackgroundFilepickerFrame(
            vis_tab,
            header_name='Immagine di Sfondo:',
            placeholder_text='Percorso immagine...',
            apply_command=self.apply_bg_command
        )
        self.file_picker.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        # Alpha slider
        self.alpha_slider = SliderCustomFrame(
            vis_tab,
            header_name='Trasparenza:',
            command=self.update_alpha,
            initial_value=self.bg_alpha
        )
        self.alpha_slider.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        # Tab Audio
        tabview.add("🎵 Audio")
        audio_tab = tabview.tab("🎵 Audio")

        # Noise threshold
        self.noise_slider = SliderCustomFrame(
            audio_tab,
            header_name='Soglia Rumore:',
            command=self.update_noise_threshold
        )
        self.noise_slider.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        # Frequency bands
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

        # Tab Performance
        tabview.add("⚡ Performance")
        perf_tab = tabview.tab("⚡ Performance")

        # Processors
        self.processors_number_frame = ProcessesNumberFrame(
            perf_tab,
            max_value=5,
            initial_value=1,
            command=self.update_workers_number,
            auto_callback=self.update_auto_workers
        )
        self.processors_number_frame.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        # Monitor selection (solo se ci sono più monitor)
        if len(self.available_monitors) > 1:
            self.monitor_option = OptionMenuCustomFrame(
                perf_tab,
                header_name='Monitor Fullscreen:',
                values=self.available_monitors,
                initial_value=self.available_monitors[0],
                command=self.change_selected_monitor
            )
            self.monitor_option.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

        return tabview

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

    def on_device_selected(self, event):  # Nota: event invece di _
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
                    fg_color='#28a745',  # Verde
                    hover_color='#218838'
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

    def update_workers_number(self, value):
        if self.equalizer_opengl_thread:
            message = {
                "type": "set_workers",
                "value": value
            }
            self.equalizer_control_queue.put(message)

    def update_auto_workers(self, value):
        if self.equalizer_opengl_thread:
            message = {
                "type": "set_auto_workers",
                "value": value
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
            self.device_listbox.configure(bg="#5a5a5a")
        else:
            self.device_listbox.configure(bg='#fff')

    def change_selected_monitor(self, selected_monitor_name: str) -> None:
        self.selected_monitor = selected_monitor_name

    def start_capture(self):
        """
        Start the equalizer thread when the start button is pressed
        """
        print(f"start_capture: canvas_thread = {self.canvas_thread}")
        print(f"start_capture: audio_thread attivo = {self.audio_thread and self.audio_thread.is_alive()}")

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
                    print("Thread equalizer creato e avviato")
                except Exception as e:
                    print(f"Errore nella creazione del thread: {e}")
                    self.canvas_thread = None
            else:
                print("Audio thread non attivo!")
                self.show_no_audio_thread_warning()
        else:
            print("Canvas thread già esistente!")

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
            print("ERRORE: audio_queue non valida!")
            return

        if not self.is_capturing:
            # Svuota la coda prima di iniziare
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except:
                    break

            # Avvia la cattura
            print(f"Avvio cattura... Dimensione coda: {self.audio_queue.qsize()}")
            self.start_capture()
            # Verifica che sia partito correttamente prima di cambiare stato
            self.after(100, self._check_capture_started)
        else:
            # Ferma la cattura
            print("Arresto cattura...")
            self.stop_capture()
            self.is_capturing = False
            # Aggiorna pulsante a verde
            self.start_stop_button.update_base_colors(
                fg_color='#28a745',
                hover_color='#218838'
            )
            self.start_stop_button.configure(text="Start")
            print("Cattura arrestata")

    def _check_capture_started(self):
        """
        Verifica che la cattura sia effettivamente partita e aggiorna l'UI.
        """
        if self.canvas_thread and self.canvas_thread.is_alive():
            self.is_capturing = True
            # Aggiorna pulsante a rosso
            self.start_stop_button.update_base_colors(
                fg_color='#dc3545',
                hover_color='#c82333'
            )
            self.start_stop_button.configure(text="Stop")
            print("Cattura avviata con successo")
        else:
            print("Errore: thread non avviato")
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
                fg_color='#28a745',
                hover_color='#218838'
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
                    window_close_callback=self.opengl_window_closed,
                    workers=self.processors_number_frame.get_value(),
                    auto_workers=self.processors_number_frame.is_auto()
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
                    self.after(500, self.resize_background, widget)

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
