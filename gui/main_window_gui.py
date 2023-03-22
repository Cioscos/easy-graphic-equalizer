import tkinter as tk
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
        
        # Create device listbox
        device_frame = tk.Frame(self.root)
        device_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)
        tk.Label(device_frame, text="Select device:").pack()
        self.device_listbox = tk.Listbox(device_frame, width=40)
        self.device_listbox.pack(fill=tk.Y, expand=True)
        for i, device in enumerate(self.devices):
            self.device_listbox.insert(tk.END, f"{i}: {device.name}")
        self.device_listbox.bind("<Double-Button-1>", self.on_device_selected)
        
        # Create settings frame
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        tk.Label(settings_frame, text="Settings").pack()
        self.noise_threshold = tk.DoubleVar(value=0.1)
        tk.Scale(settings_frame, 
                 from_=0, 
                 to=1, 
                 resolution=0.01, 
                 orient=tk.HORIZONTAL, 
                 label="Noise threshold",
                   variable=self.noise_threshold, 
                   command=self.update_noise_threshold
                ).pack()

        # Create equalizer canvas
        self.equalizer_canvas = tk.Canvas(settings_frame, width=800, height=600)
        self.equalizer_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create start button
        tk.Button(self.root, text="Start", command=self.start_capture).pack(side=tk.BOTTOM, padx=10, pady=10)
        
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
            if(self.last_device_selected != device):
                self.audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
                self.audio_thread = AudioCaptureThread(self.audio_queue, device=device)
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
