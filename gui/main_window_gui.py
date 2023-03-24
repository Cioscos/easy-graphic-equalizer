import customtkinter as ctk
import tkinter as tk
import soundcard as sc
import queue

from thread.audioCaptureThread import AudioCaptureThread
from thread.equalizer_tkinter_thread import EqualizerTkinterThread

MAX_QUEUE_SIZE = 200


class AudioCaptureGUI(ctk.CTk):
    """
    The main window which directly extends Ctk class
    """
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode('System')
        ctk.set_default_color_theme('blue')

        self.devices = sc.all_microphones(include_loopback=True)

        # Create main window title
        self.title("Audio Capture")
        self.minsize(800, 600)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # create left frame
        left_frame = ctk.CTkFrame(self, border_width=2)
        left_frame.pack(side=ctk.LEFT, padx=10, pady=10, fill=ctk.Y)

        # Create device frame
        device_frame = ctk.CTkFrame(left_frame)
        device_frame.pack(padx=10, pady=10, anchor='n')
        ctk.CTkLabel(device_frame, text="Select device:", font=("Roboto", 14)).pack(pady=5)

        # Create the listbox frame
        device_listbox_frame = ctk.CTkFrame(device_frame)
        device_listbox_frame.pack(fill=ctk.BOTH, expand=True)

        self.device_listbox = tk.Listbox(device_listbox_frame, width=40, font=("Roboto", 12), bg="#fff", fg="#333")
        self.device_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ctk.CTkScrollbar(device_listbox_frame, orientation="vertical", command=self.device_listbox.yview)
        scrollbar.pack(side=ctk.RIGHT, fill=ctk.Y)
        self.device_listbox.config(yscrollcommand=scrollbar.set)
        for i, device in enumerate(self.devices):
            self.device_listbox.insert(tk.END, f"{i}: {device}")
        self.device_listbox.bind("<Double-Button-1>", self.on_device_selected)

        # Create settings frame
        settings_frame = ctk.CTkFrame(left_frame, border_width=2)
        settings_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=10)
        ctk.CTkLabel(settings_frame, text="Settings", font=("Roboto", 18, "bold")).pack(side=tk.TOP)
        
        # Create scale widget
        self.noise_threshold = ctk.DoubleVar(value=0.1)
        scale_frame = ctk.CTkFrame(settings_frame)
        scale_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X, expand=False)
        ctk.CTkLabel(scale_frame, text="Noise threshold:").pack(side=ctk.LEFT, padx=10)
        ctk.CTkSlider(scale_frame,
                 from_=0,
                 to=1,
                 orientation=ctk.HORIZONTAL,
                 variable=self.noise_threshold,
                 command=self.update_noise_threshold
                 ).pack(side=ctk.LEFT, fill=ctk.X, expand=True)
        
        # Create right frame
        right_frame = ctk.CTkFrame(self, border_width=2)
        right_frame.pack(side=ctk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Create equalizer frame and put it into the right frame
        equalizer_frame = ctk.CTkFrame(right_frame, border_width=2)
        equalizer_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create equalizer canvas and put into equalizer_frame
        self.equalizer_canvas = ctk.CTkCanvas(equalizer_frame, bg="#333")
        self.equalizer_canvas.pack(fill=tk.BOTH, expand=True)

        # Create buttons frame and put into right frame
        buttons_frame = ctk.CTkFrame(right_frame)
        buttons_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Create start button
        self.start_button = ctk.CTkButton(buttons_frame,
                  text="Start",
                  command=self.start_capture,
                  font=("Roboto", 16),
                  fg_color='green')
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)

        # Create stop button
        self.stop_button = ctk.CTkButton(buttons_frame,
                                         text='Stop',
                                         font=("Roboto", 16),
                                         fg_color='red',
                                         command=self.stop_capture)
        self.stop_button.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.X, expand=True)

        self.audio_queue = None
        self.equalizer_control_queue = queue.Queue()
        self.audio_thread = None
        self.opengl_thread = None
        self.last_device_selected = None

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
                self.audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
                self.audio_thread = AudioCaptureThread(
                    self.audio_queue, device=device)
                self.audio_thread.start()
                self.last_device_selected = device

    def update_noise_threshold(self, value):
        """
        Update the noise threshold in the equalizer sending
        a message to the EqualizerTkinterThread

        Args:
            value (ctk.DoubleVar):A DoubleVar object to trak the threshold value
        """
        if self.opengl_thread:
            message = {
                "type": "set_noise_threshold",
                "value": float(value)
            }
            self.equalizer_control_queue.put(message)

    def start_capture(self):
        """
        Start the equalizer thread when the start button is pressed
        """
        if self.audio_thread and not self.opengl_thread:
            # Start the OpenGL window in a new thread
            self.opengl_thread = EqualizerTkinterThread(
                self.audio_queue,
                noise_threshold=self.noise_threshold.get(),
                canvas=self.equalizer_canvas,
                control_queue=self.equalizer_control_queue)

            self.opengl_thread.start()

    def stop_capture(self):
        """
        Check if the equalizer thread is running and if it is
        grecefully stop it
        """
        if self.opengl_thread and self.opengl_thread.is_alive():
            self.opengl_thread.stop()
            self.opengl_thread = None


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
        if self.opengl_thread:
            self.opengl_thread.stop()
            while self.opengl_thread.is_alive():
                self.opengl_thread.join(timeout=0.1)
                # Update the GUI main loop to process events
                # it avoid deadlock between the threads
                self.update()

        # Stop the audio thread and close the window
        if self.audio_thread:
            self.audio_thread.stop()
            while self.audio_thread.is_alive():
                self.audio_thread.join(timeout=0.1)
                self.update()

        self.destroy()
