from typing import Callable, Optional, Union
from enum import Enum

import customtkinter as tk


class SliderCustomFrame(tk.CTkFrame):
    """
    Custom class to wrap an Slider inside a frame
    """
    class ValueType(Enum):
        INT = 'int'
        DOUBLE = 'double'

    def __init__(self,
                 *args,
                 header_name: str = 'SliderCustomFrame',
                 initial_value: Union[float, int] = 0.1,
                 from_: int = 0,
                 to: int = 1,
                 steps: Optional[int] = None,
                 command: Optional[Callable] = None,
                 warning_text: Optional[str] = None,
                 warning_trigger_value: Optional[int] = None,
                 value_type: ValueType = ValueType.DOUBLE,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if warning_trigger_value and not warning_text:
            raise Exception('No warning text with when warning_trigger_value set')
        
        self._warning_trigger_value = warning_trigger_value

        if value_type == self.ValueType.INT:
            self.slider_value = tk.IntVar()
        elif value_type == self.ValueType.DOUBLE:
            self.slider_value = tk.DoubleVar()
        else:
            raise Exception(f'No valid value passed to value_type. Use one of these values: {[type.value for type in self.ValueType]}')
        
        self.slider_value.set(value=initial_value)

        slider_frame = tk.CTkFrame(self)
        slider_frame.pack(expand=True, fill=tk.X, pady=2)

        header = tk.CTkLabel(slider_frame, text=header_name)
        header.pack(side=tk.LEFT, padx=10)

        slider = tk.CTkSlider(slider_frame,
                 from_=from_,
                 to=to,
                 orientation=tk.HORIZONTAL,
                 variable=self.slider_value,
                 number_of_steps=steps,
                 command=self._combine_funcs(command, self._update_value_text)
                 )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._value_text = tk.CTkLabel(slider_frame, text=str(self.slider_value.get()))
        self._value_text.pack(side=tk.LEFT, expand=False, padx=5)

        # Warning message
        self.warning_label = tk.CTkLabel(self,
                                text=warning_text,
                                font=("Roboto", 12),
                                text_color='yellow',
                                justify='center')

        # 'The number of the bard could be too high. Consider to use the fullscreen view'
        if warning_trigger_value and self.slider_value.get() >= warning_trigger_value:
            self.warning_label.pack()

    def _combine_funcs(self, *funcs):
        """
        this function will call the passed functions
        with the arguments that are passed to the functions
        """
        def inner_combined_func(*args, **kwargs):
            for f in funcs:
                # Calling functions with arguments, if any
                f(*args, **kwargs)
    
        # returning the reference of inner_combined_func
        # this reference will have the called result of all
        # the functions that are passed to the combined_funcs
        return inner_combined_func
    

    def _update_value_text(self, _):
        """
        Update the text box every update of the slider and eventually shows warning
        """
        self._value_text.configure(text=str(round(self.get_value(), 2)))

        if self._warning_trigger_value:
            if self._warning_trigger_value <= self.slider_value.get():
                self.warning_label.pack()
            else:
                self.warning_label.pack_forget()

    def get_value(self) -> float:
        """
        return the slider value as float number
        """
        return self.slider_value.get()
