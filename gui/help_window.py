import customtkinter as ctk


class HelpWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("400x300")
        self.title('Help')

        self.label = ctk.CTkLabel(self, text="How to use", font=("Roboto", 18, "bold"))
        self.label.pack(pady=10)

        self.explaination = ctk.CTkTextbox(self, wrap='word')
        self.explaination.pack(expand=True, fill=ctk.BOTH)
        self.explaination.insert('0.0', 'Hi! Welcome to my application!\nThe first thing to do is to select a device from the device list on the left.\n' \
                                 'In order to select it you have to douple click on one of them. After you\'ll see the green button activate, it means ' \
                                 'that now you can press it to start the graphic equalizer! Press it again to stop it.\n\n' \
                                 'Settings\n\n' \
                                 'Noise threshold: Slider to set how much the amplitude of the signal must be high to be drawn.\n' \
                                 'Appearance mode: Select the theme.\n' \
                                 'Background image path: Use it to change the default background image of the equalizator. You can both manually insert '\
                                 'the path to the file or click the botton on its right to open a file picker window. Remember to press Apply button.\n' \
                                 'Alpha amount: Use it to change the alpha of the background image. The lower the darker.\n' \
                                 'Frequency bands: Use it to dinamically change the number of the bars showed. Take care that after some number of bars, ' \
                                 'according to your hardware it could become pretty laggy. For high number of bars is highly suggested the full screen visulization\n' \
                                 'Fullscreen button: Open a fullscreen window in which the bars are drawn. This window has high performance so it\'s ' \
                                 'suggested when you use high number of bands.')
        self.explaination.configure(state='disabled')
