from typing import Callable, Optional
import customtkinter as tk


class SliderCustomFrame(tk.CTkFrame):
    """
    Custom class to wrap an Slider inside a frame
    """
    def __init__(self,
                 *args,
                 header_name: str = 'SliderCustomFrame',
                 initial_value: float = 0.1,
                 from_: int = 0,
                 to: int = 1,
                 steps: Optional[int] = None,
                 command: Optional[Callable] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        header = tk.CTkLabel(self, text=header_name)
        header.pack(side=tk.LEFT, padx=10)

        self.slider_value = tk.DoubleVar(value=initial_value)

        slider = tk.CTkSlider(self,
                 from_=from_,
                 to=to,
                 orientation=tk.HORIZONTAL,
                 variable=self.slider_value,
                 number_of_steps=steps,
                 command=self._combine_funcs(command, self._update_value_text)
                 )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._value_text = tk.CTkLabel(self, text=str(self.slider_value.get()))
        self._value_text.pack(side=tk.LEFT, expand=False)

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
        Update the text box every update of the slider
        """
        self._value_text.configure(text=str(round(self.get_value(), 2)))


    def get_value(self) -> float:
        """
        return the slider value as float number
        """
        return self.slider_value.get()
