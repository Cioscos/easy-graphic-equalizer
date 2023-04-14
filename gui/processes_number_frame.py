import customtkinter as ctk
from typing import Callable, Optional

class ProcessesNumberFrame(ctk.CTkFrame):

    RETURN_BUTTON = '<Return>'

    def __init__(self,
                 *args,
                 max_value: int = 100,
                 initial_value : Optional[int] = None,
                 command: Callable,
                 auto_callback: Callable,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._max_value = max_value
        self._initial_value = initial_value
        self._command = command
        self._auto_callback = auto_callback
        self._create_widgets()

    def _create_widgets(self):
        self.columnconfigure(0, weight=2)
        
        header = ctk.CTkLabel(self, text='Number of processors')
        header.pack(pady=5)

        whole_frame = ctk.CTkFrame(self)

        self._var_checkbox = ctk.BooleanVar()
        self._checkbox = ctk.CTkCheckBox(
            whole_frame,
            text='Auto',
            variable=self._var_checkbox,
            command=self._combine_funcs(self._toggle_state, self._return_auto_callback)
        )
        whole_frame.columnconfigure(0, weight=2)

        self._checkbox.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky=ctk.W)

        if not self._initial_value:
            self._initial_value = 1
        
        self._var_inputbox = ctk.IntVar(value=self._initial_value)

        input_selection_frame = ctk.CTkFrame(whole_frame)

        self._inputbox = ctk.CTkEntry(input_selection_frame,
                                      textvariable=self._var_inputbox,
                                      width=50)
        self._inputbox.pack(side=ctk.LEFT, padx=5, pady=5)
        self._inputbox.bind(self.RETURN_BUTTON, command=self._command)

        arrows_frame = ctk.CTkFrame(input_selection_frame)

        self._arrow_up_button = ctk.CTkButton(arrows_frame,
                                              text="↑",
                                              width=5,
                                              height=5,
                                              font=("Roboto", 10),
                                              command=self._combine_funcs(self._increase_value, self._return_value_callback))
        self._arrow_up_button.pack()

        self._arrow_down_button = ctk.CTkButton(arrows_frame,
                                                text="↓",
                                                width=5,
                                                height=5,
                                                font=("Roboto", 10),
                                                command=self._combine_funcs(self._decrease_value, self._return_value_callback))
        self._arrow_down_button.pack()

        arrows_frame.pack(side=ctk.LEFT, padx=2)

        input_selection_frame.grid(row=0, column=2, padx= 5, pady=5)

        whole_frame.pack(fill=ctk.X, padx=5, pady=5)

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
    
    def _return_value_callback(self):
        self._command(self.get_value())

    def _return_auto_callback(self):
        self._auto_callback(self.is_auto())

    def _toggle_state(self):
        if self._var_checkbox.get():
            self._inputbox.configure(state=ctk.DISABLED)
            self._arrow_up_button.configure(state=ctk.DISABLED)
            self._arrow_down_button.configure(state=ctk.DISABLED)
        else:
            self._inputbox.configure(state=ctk.NORMAL)
            self._arrow_up_button.configure(state=ctk.NORMAL)
            self._arrow_down_button.configure(state=ctk.NORMAL)

    def _increase_value(self):
        new_value = self._var_inputbox.get() + 1
        if new_value <= self._max_value:
            self._var_inputbox.set(new_value)

    def _decrease_value(self):
        new_value = self._var_inputbox.get() - 1
        if new_value >= 0:
            self._var_inputbox.set(new_value)

    def get_value(self):
        return self._var_inputbox.get()
    
    def is_auto(self):
        return self._var_checkbox.get()
