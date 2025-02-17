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
from gui.processes_number_frame import ProcessesNumberFrame
from resource_manager import ResourceManager

MAX_QUEUE_SIZE = 200
INITIAL_FREQUENCIES_BANDS = 9
SINGLE_LEFT_MOUSE_BOTTON_CLICK = '<Button-1>'
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

        self.device_listbox.bind(SINGLE_LEFT_MOUSE_BOTTON_CLICK, self.on_device_selected)

        # Create settings frame
        settings_frame = ctk.CTkScrollableFrame(left_frame, border_width=2)
        settings_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        ctk.CTkLabel(settings_frame, text="Settings", font=("Roboto", 18, "bold")).pack(side=tk.TOP, padx=5, pady=5)
        
        # Create theme Option
        self.appearance_mode = OptionMenuCustomFrame(settings_frame,
                                                     header_name='Appearance Mode:',
                                                     values=["Light", "Dark", "System"],
                                                     initial_value="System",
                                                     command=self.change_appearance_mode_event)
        self.appearance_mode.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X, expand=False)

        # Background file picker
        self.file_picker = BackgroundFilepickerFrame(settings_frame,
                                                     header_name='Background Image Path:',
                                                     placeholder_text= 'Insert background image path',
                                                     apply_command=self.apply_bg_comand)
        self.file_picker.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X, expand=False)
        self.bg_filename = None

        # fullscreen monitor option
        self.selected_monitor = None
        self.available_monitors = self.get_available_monitors()
        if len(self.available_monitors) != 1:
            self.monitor_option = OptionMenuCustomFrame(settings_frame,
                                                        header_name='Fullscreen monitor',
                                                        values=self.available_monitors,
                                                        initial_value=self.available_monitors[0],
                                                        command=self.change_selected_monitor)
            self.monitor_option.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X, expand=False)

        # alpha slider
        self.alpha_slider = SliderCustomFrame(settings_frame,
                                              header_name='Alpha amount:',
                                              command=self.update_alpha,
                                              initial_value=self.bg_alpha)
        self.alpha_slider.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X, expand=False)

        # noise threshold slider 
        self.noise_slider = SliderCustomFrame(settings_frame,
                                              header_name='Noise threshold:',
                                              command=self.update_noise_threshold)
        self.noise_slider.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X, expand=False)

        # Bands number
        self.frequency_slider = SliderCustomFrame(settings_frame,
                                              header_name='Frequency bands:',
                                              command=self.update_frequency_bands,
                                              from_=1,
                                              to=100,
                                              steps=32,
                                              initial_value=INITIAL_FREQUENCIES_BANDS,
                                              warning_trigger_value=15,
                                              warning_text='The number of the bard could be too high.\nConsider to use the fullscreen view',
                                              value_type=SliderCustomFrame.ValueType.INT)
        self.frequency_slider.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X, expand=False)

        # Processors number
        self.processors_number_frame = ProcessesNumberFrame(
            settings_frame,
            max_value=5,
            initial_value=1,
            command=self.update_workers_number,
            auto_callback=self.update_auto_workers
        )
        self.processors_number_frame.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X, expand=False)

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
        self.start_stop_button = ctk.CTkButton(buttons_frame,
                  text="Start/Pause",
                  command=self.start_stop_capture,
                  font=("Roboto", 16),
                  fg_color='green',
                  state='disabled')
        self.start_stop_button.pack(padx=5, pady=5, fill=ctk.BOTH, side=tk.LEFT, expand=True)

        self.fullscreen_icn = ctk.CTkImage(dark_image=Image.open(self.resource_manager.get_image_path('maximize.png', 'icons')))
        self.fullscreen_btn = ctk.CTkButton(buttons_frame,
                                            command=self.fullscreen_command,
                                            image=self.fullscreen_icn,
                                            text='Fullscreen',
                                            font=("Roboto", 16))
        self.fullscreen_btn.pack(padx=5, pady=5, fill=ctk.BOTH, side=tk.LEFT, expand=True)

        help_icn = ctk.CTkImage(dark_image=Image.open(self.resource_manager.get_image_path('help.png', 'icons')))
        help_button = ctk.CTkButton(buttons_frame,
                                    command=self.open_help_window,
                                    image=help_icn,
                                    text='Help',
                                    font=("Roboto", 16))
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

    def get_canvas_size(self) -> tuple[int, int]:
        """
        Return the size of the equalizer_canvas
        Returns:
            tuple: width and height of the canvas
        """
        return (self.equalizer_canvas.winfo_width(), self.equalizer_canvas.winfo_height())
    
    def apply_bg_comand(self):
        # load canvas bg image
        self.bg_img = Image.open(self.file_picker.get_filename())
        self.bg_img_used = self.bg_img.copy()
        self.bg_img_used.putalpha(int(255 * self.bg_alpha))
        self.apply_background()
        self.update_bg_opengl(self.bg_img)

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

    def on_device_selected(self, _):
        """
        When a device from the listbox is choosen, it starts the audio_thread
        and it says which devide the thread has to listen.
        """
        # Start audio capture when a device is selected
        selection = self.device_listbox.curselection()
        if selection:
            device_index = selection[0]
            device = self.devices[device_index]

            # Check if the device was already selected
            if (self.last_device_selected != device):
                if self.audio_queue:
                    self.audio_queue = None

                self.stop_audio_thread()

                self.audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
                self.audio_thread = AudioCaptureThread(
                    self.audio_queue, device=device)
                self.audio_thread.start()
                self.last_device_selected = device
                self.start_stop_button.configure(state='normal')

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
        if not self.canvas_thread:
            if self.audio_thread:
                # Start the OpenGL window in a new thread
                self.canvas_thread = EqualizerTkinterThread(
                    self.audio_queue,
                    noise_threshold=self.noise_slider.get_value(),
                    n_bands=int(self.frequency_slider.get_value()),
                    canvas=self.equalizer_canvas,
                    control_queue=self.equalizer_control_queue)
                self.canvas_thread.start()

            else:
                self.show_no_audio_thread_warning()

    def stop_capture(self):
        """
        Check if the equalizer thread is running and if it is
        grecefully stop it
        """
        if self.canvas_thread and self.canvas_thread.is_alive():
            self.canvas_thread.stop()
            self.canvas_thread = None

    def start_stop_capture(self):
        if self.is_on_start:
            self.start_stop_button.configure(fg_color='red')
            self.start_capture()
            self.is_on_start = False
        else:
            self.start_stop_button.configure(fg_color='green')
            self.stop_capture()
            self.is_on_start = True

    def fullscreen_command(self):
        self.stop_capture()
        self.is_on_start = True
        self.start_stop_button.configure(fg_color='green')

        if self.audio_thread:
            if self.equalizer_opengl_thread:
                self.equalizer_opengl_thread.stop()
                while self.equalizer_opengl_thread.is_alive():
                    self.equalizer_opengl_thread.join(timeout=0.1)
                self.equalizer_opengl_thread = None
            else:
                self.equalizer_opengl_thread = EqualizerOpenGLThread(self.audio_queue,
                                                                    noise_threshold=self.noise_slider.get_value(),
                                                                    n_bands=int(self.frequency_slider.get_value()),
                                                                    control_queue=self.equalizer_control_queue,
                                                                    bg_image=self.bg_img,
                                                                    monitor=self.selected_monitor,
                                                                    window_close_callback=self.opengl_window_closed,
                                                                    workers=self.processors_number_frame.get_value(),
                                                                    auto_workers=self.processors_number_frame.is_auto())

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
