from typing import Callable, Optional
import customtkinter as tk

FONT = ("Roboto", 12)

class OptionMenuCustomFrame(tk.CTkFrame):
    """
    Custom class to wrap an optionmenu inside a frame
    """
    def __init__(self,
                 *args,
                 header_name: str = 'OptionMenuCustomFrame',
                 values: Optional[list] = None,
                 initial_value : Optional[str] = None,
                 command: Optional[Callable] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        header = tk.CTkLabel(self, text=header_name)
        header.pack(side=tk.LEFT, padx=10)


        if not values:
            values = ['--Select--']

        self._optionmenu_var = tk.StringVar(value=initial_value if initial_value else values[0])
        self._option_menu = tk.CTkOptionMenu(self,
                                            values=values,
                                            command=command,
                                            variable=self._optionmenu_var,
                                            font=FONT,
                                            dropdown_font=FONT)
        self._option_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def get_value(self) -> str:
        """
        Returns the value of the option menu
        """
        return self._optionmenu_var.get()
    
    def set_value(self, value: str) -> None:
        """
        Set the value of the option menu
        """
        self._optionmenu_var.set(value)
