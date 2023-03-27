from typing import Callable, Optional

import tkinter as tk
import customtkinter as ctk


class BackgroundFilepickerFrame(ctk.CTkFrame):
    """
    Custom class to wrap a file picker inside a frame
    """
    def __init__(self,
                 *args,
                 header_name: str = 'BackgroundFilepickerFrame',
                 placeholder_text : Optional[str] = None,
                 apply_command: Optional[Callable] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._filename = None

        self.columnconfigure(0, weight=2)

        # Create the label in the first row
        self.label = ctk.CTkLabel(self, text=header_name)
        self.label.grid(row=0, column=0, columnspan=2, sticky='N')

        # Create the entry in the second row
        self.image_path_entry = ctk.CTkEntry(self, placeholder_text=placeholder_text if placeholder_text else '')
        self.image_path_entry.grid(row=1, column=0, sticky="EW")

        # Create the button next to the entry in the second row
        self.file_picker_button = ctk.CTkButton(self, text='...', width=10, command=self.open_file_dialog)
        self.file_picker_button.grid(row=1, column=1, sticky="E")

        # Create the apply button in the third row
        self.apply_button = ctk.CTkButton(self, text='Apply', command=apply_command)
        self.apply_button.grid(row=2, column=0, columnspan=2, sticky="EW")

    def open_file_dialog(self) -> None:
        self._filename = tk.filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes = (("Image files", "*.png;*.jpg;*.jpeg;*.gif"), ("all files", "*.*")))
        if self._filename:
            self.image_path_entry.delete(0, ctk.END)
            self.image_path_entry.insert(0, self._filename)

    def get_filename(self) -> str:
        return self._filename
