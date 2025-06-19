# gui/animated_button.py
"""
Pulsante animato con effetti hover e click per CustomTkinter.
"""
import customtkinter as ctk
from typing import Optional, Callable


class AnimatedButton(ctk.CTkButton):
    """
    Pulsante con animazioni fluide per hover e click.
    Supporta animazioni di scala e colore.
    """

    def __init__(self,
                 master,
                 *args,
                 animation_duration: int = 200,
                 hover_scale: float = 1.05,
                 click_scale: float = 0.95,
                 **kwargs):
        """
        Inizializza il pulsante animato.

        Args:
            master: Il widget padre
            animation_duration: Durata dell'animazione in millisecondi
            hover_scale: Fattore di scala quando il mouse è sopra il pulsante
            click_scale: Fattore di scala quando il pulsante è cliccato
            *args, **kwargs: Argomenti passati a CTkButton
        """
        # Salva i parametri originali
        self._fg_color_original = kwargs.get('fg_color', '#1f538d')
        self._hover_color_original = kwargs.get('hover_color', None)

        super().__init__(master, *args, **kwargs)

        # Parametri di animazione
        self.animation_duration = animation_duration
        self.hover_scale = hover_scale
        self.click_scale = click_scale

        # Stato dell'animazione
        self._animation_running = False
        self._current_scale = 1.0
        self._target_scale = 1.0
        self._animation_step = 0
        self._animation_start_scale = 1.0
        self._is_pressed = False

        # Salva le dimensioni originali dopo che il widget è stato creato
        self.after(100, self._save_original_dimensions)

        # Override dei metodi di binding per CustomTkinter
        self._setup_bindings()

    def _save_original_dimensions(self):
        """
        Salva le dimensioni originali del pulsante dopo l'inizializzazione.
        """
        self._original_width = self._current_width if hasattr(self, '_current_width') else self.winfo_width()
        self._original_height = self._current_height if hasattr(self, '_current_height') else self.winfo_height()

    def _setup_bindings(self):
        """
        Configura i binding degli eventi compatibili con CustomTkinter.
        """
        # Usa il canvas interno di CustomTkinter per i binding
        if hasattr(self, '_canvas'):
            self._canvas.bind("<Enter>", self._on_enter)
            self._canvas.bind("<Leave>", self._on_leave)
            self._canvas.bind("<ButtonPress-1>", self._on_press)
            self._canvas.bind("<ButtonRelease-1>", self._on_release)
        else:
            # Fallback per compatibilità
            self.bind("<Enter>", self._on_enter)
            self.bind("<Leave>", self._on_leave)
            self.bind("<ButtonPress-1>", self._on_press)
            self.bind("<ButtonRelease-1>", self._on_release)

    def _on_enter(self, event=None):
        """
        Gestisce l'evento di mouse enter.

        Args:
            event: L'evento del mouse (opzionale)
        """
        if not self._is_pressed and self.cget("state") != "disabled":
            self._animate_to_scale(self.hover_scale)
            # Effetto di luminosità aumentata
            self._animate_color_brightness(1.1)

    def _on_leave(self, event=None):
        """
        Gestisce l'evento di mouse leave.

        Args:
            event: L'evento del mouse (opzionale)
        """
        if not self._is_pressed and self.cget("state") != "disabled":
            self._animate_to_scale(1.0)
            # Ripristina il colore originale
            self._animate_color_brightness(1.0)

    def _on_press(self, event=None):
        """
        Gestisce l'evento di click del mouse.

        Args:
            event: L'evento del mouse (opzionale)
        """
        if self.cget("state") != "disabled":
            self._is_pressed = True
            self._animate_to_scale(self.click_scale)

    def _on_release(self, event=None):
        """
        Gestisce il rilascio del click del mouse.

        Args:
            event: L'evento del mouse (opzionale)
        """
        self._is_pressed = False
        if self.cget("state") != "disabled":
            # Controlla se il mouse è ancora sopra il pulsante
            try:
                x = self.winfo_pointerx()
                y = self.winfo_pointery()
                widget_x = self.winfo_rootx()
                widget_y = self.winfo_rooty()
                widget_width = self.winfo_width()
                widget_height = self.winfo_height()

                if (widget_x <= x <= widget_x + widget_width and
                    widget_y <= y <= widget_y + widget_height):
                    self._animate_to_scale(self.hover_scale)
                else:
                    self._animate_to_scale(1.0)
            except:
                self._animate_to_scale(1.0)

    def _animate_to_scale(self, target_scale: float):
        """
        Avvia l'animazione verso la scala target.

        Args:
            target_scale: La scala obiettivo
        """
        self._target_scale = target_scale
        if not self._animation_running:
            self._animation_running = True
            self._animation_start_scale = self._current_scale
            self._animation_step = 0
            self._animate_scale()

    def _animate_scale(self):
        """
        Esegue un passo dell'animazione della scala.
        """
        if not self._animation_running:
            return

        # Calcola il progresso dell'animazione (0.0 a 1.0)
        self._animation_step += 20  # 20ms per frame
        progress = min(self._animation_step / self.animation_duration, 1.0)

        # Usa una funzione di easing (ease-out cubic)
        eased_progress = 1 - pow(1 - progress, 3)

        # Calcola la scala corrente
        scale_diff = self._target_scale - self._animation_start_scale
        self._current_scale = self._animation_start_scale + (scale_diff * eased_progress)

        # Applica la scala solo se abbiamo le dimensioni originali
        if hasattr(self, '_original_width') and hasattr(self, '_original_height'):
            new_width = int(self._original_width * self._current_scale)
            new_height = int(self._original_height * self._current_scale)

            # Configura le nuove dimensioni
            try:
                self.configure(width=new_width, height=new_height)
            except:
                pass  # Ignora errori durante l'animazione

        # Continua l'animazione se non è completa
        if progress < 1.0:
            self.after(20, self._animate_scale)
        else:
            self._animation_running = False

    def _animate_color_brightness(self, factor: float):
        """
        Anima la luminosità del colore del pulsante.

        Args:
            factor: Fattore di luminosità (1.0 = normale, >1.0 = più chiaro)
        """
        try:
            if isinstance(self._fg_color_original, str) and self._fg_color_original.startswith('#'):
                # Converti il colore hex in RGB
                hex_color = self._fg_color_original.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

                # Applica il fattore di luminosità
                r = min(int(r * factor), 255)
                g = min(int(g * factor), 255)
                b = min(int(b * factor), 255)

                # Converti di nuovo in hex
                new_color = f'#{r:02x}{g:02x}{b:02x}'

                # Applica il nuovo colore
                self.configure(fg_color=new_color)
        except:
            pass  # Ignora errori nella manipolazione del colore

    def set_animation_params(self, duration: int = None,
                           hover_scale: float = None,
                           click_scale: float = None):
        """
        Modifica i parametri dell'animazione dinamicamente.

        Args:
            duration: Nuova durata dell'animazione in ms
            hover_scale: Nuovo fattore di scala per hover
            click_scale: Nuovo fattore di scala per click
        """
        if duration is not None:
            self.animation_duration = duration
        if hover_scale is not None:
            self.hover_scale = hover_scale
        if click_scale is not None:
            self.click_scale = click_scale

    def update_base_colors(self, fg_color: str, hover_color: str = None):
        """
        Aggiorna i colori base del pulsante mantenendo le animazioni.

        Args:
            fg_color: Nuovo colore di base
            hover_color: Nuovo colore hover (opzionale)
        """
        self._fg_color_original = fg_color
        self._hover_color_original = hover_color

        # Aggiorna immediatamente il colore se non siamo in hover
        if not self._is_pressed and self._current_scale == 1.0:
            self.configure(fg_color=fg_color)
            if hover_color:
                self.configure(hover_color=hover_color)

    def _on_leave(self, event=None):
        """
        Gestisce l'evento di mouse leave.

        Args:
            event: L'evento del mouse (opzionale)
        """
        if not self._is_pressed and self.cget("state") != "disabled":
            self._animate_to_scale(1.0)
            # Ripristina il colore originale ATTUALE
            self.configure(fg_color=self._fg_color_original)

    def _animate_color_brightness(self, factor: float):
        """
        Anima la luminosità del colore del pulsante.

        Args:
            factor: Fattore di luminosità (1.0 = normale, >1.0 = più chiaro)
        """
        try:
            current_fg_color = self._fg_color_original  # Usa sempre il colore originale attuale
            if isinstance(current_fg_color, str) and current_fg_color.startswith('#'):
                # Converti il colore hex in RGB
                hex_color = current_fg_color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

                # Applica il fattore di luminosità
                r = min(int(r * factor), 255)
                g = min(int(g * factor), 255)
                b = min(int(b * factor), 255)

                # Converti di nuovo in hex
                new_color = f'#{r:02x}{g:02x}{b:02x}'

                # Applica il nuovo colore
                self.configure(fg_color=new_color)
        except:
            pass  # Ignora errori nella manipolazione del colore
