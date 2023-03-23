import tkinter as tk
from tkinter import ttk
import soundcard as sc
import queue

from thread.audioCaptureThread import AudioCaptureThread
from thread.equalizer_tkinter_thread import EqualizerTkinterThread

MAX_QUEUE_SIZE = 200


class AudioCaptureGUI:
    def __init__(self):
        self.devices = sc.all_microphones(include_loopback=True)

        # Create main window
        self.root = tk.Tk()
        self.root.title("Audio Capture")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # create left frame
        left_frame = tk.Frame(self.root, borderwidth=2, relief=tk.SUNKEN, bg="#f5f5f5")
        left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
        # Create device listbox
        device_frame = tk.Frame(left_frame, borderwidth=2, relief=tk.SUNKEN, bg="#f5f5f5")
        #device_frame.place(relx=0, rely=0, relwidth=1, relheight=0.5)
        device_frame.pack(padx=10, pady=10, anchor='n')
        tk.Label(device_frame, text="Select device:", font=("Roboto", 14), bg="#f5f5f5", fg="#333").pack(pady=5)
        device_listbox_frame = tk.Frame(device_frame, bg="#f5f5f5")
        device_listbox_frame.pack(fill=tk.BOTH, expand=True)
        self.device_listbox = tk.Listbox(device_listbox_frame, width=40, font=("Roboto", 12), bg="#fff", fg="#333")
        self.device_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(device_listbox_frame, orient="vertical", command=self.device_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.device_listbox.config(yscrollcommand=scrollbar.set)
        for i, device in enumerate(self.devices):
            self.device_listbox.insert(tk.END, f"{i}: {device}")
        self.device_listbox.bind("<Double-Button-1>", self.on_device_selected)

        # Create settings frame
        settings_frame = tk.Frame(left_frame, borderwidth=2, relief=tk.SUNKEN, bg="#f5f5f5")
        settings_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=10)
        tk.Label(settings_frame, text="Settings", font=("Roboto", 18, "bold"), bg="#f5f5f5", fg="#333").pack(side=tk.TOP)
        self.noise_threshold = tk.DoubleVar(value=0.1)
        
        # Create scale widget
        scale_frame = tk.Frame(settings_frame)
        scale_frame.pack(side=tk.TOP, padx=10, pady=10, fill=tk.X, expand=False)
        tk.Label(scale_frame, text="Noise threshold:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Style().configure("Custom.TScale", troughrelief=tk.RAISED, sliderrelief=tk.RAISED, borderwidth=0)
        tk.Scale(scale_frame,
                 from_=0,
                 to=1,
                 resolution=0.01,
                 orient=tk.HORIZONTAL,
                 variable=self.noise_threshold,
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Create right frame
        right_frame = tk.Frame(self.root, borderwidth=2, relief=tk.SUNKEN, bg="#f5f5f5")
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Create equalizer frame
        equalizer_frame = tk.Frame(right_frame, borderwidth=2, relief=tk.SUNKEN, bg="#f5f5f5")
        equalizer_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create equalizer canvas
        self.equalizer_canvas = tk.Canvas(
            equalizer_frame, bg="#333")
        self.equalizer_canvas.pack(fill=tk.BOTH, expand=True)

        # Create buttons frame
        buttons_frame = tk.Frame(right_frame, bg="#f5f5f5")
        buttons_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Create start button
        tk.Button(buttons_frame, 
                  text="Start", 
                  command=self.start_capture, 
                  font=("Roboto", 16),
                  bg="#4CAF50", 
                  fg="#fff", 
                  relief=tk.FLAT, 
                  padx=20, 
                  pady=10).pack(side=tk.BOTTOM, padx=10, pady=10, fill=tk.X)

        self.audio_queue = None
        self.equalizer_control_queue = queue.Queue()
        self.audio_thread = None
        self.opengl_thread = None
        self.last_device_selected = None

    def on_device_selected(self, event):
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
        if self.opengl_thread:
            message = {
                "type": "set_noise_threshold",
                "value": float(value)
            }
            self.equalizer_control_queue.put(message)

    def start_capture(self):
        if self.audio_thread and not self.opengl_thread:
            # Start the OpenGL window in a new thread
            self.opengl_thread = EqualizerTkinterThread(
                self.audio_queue,
                noise_threshold=self.noise_threshold.get(),
                canvas=self.equalizer_canvas,
                control_queue=self.equalizer_control_queue)

            self.opengl_thread.start()

    def run(self):
        self.root.mainloop()

    def on_close(self):
        if self.opengl_thread:
            self.opengl_thread.stop()
            while self.opengl_thread.is_alive():
                self.opengl_thread.join(timeout=0.1)
                # Update the GUI main loop to process events
                # it avoid deadlock between the threads
                self.root.update()

        # Stop the audio thread and close the window
        if self.audio_thread:
            self.audio_thread.stop()
            while self.audio_thread.is_alive():
                self.audio_thread.join(timeout=0.1)
                self.root.update()

        self.root.destroy()
