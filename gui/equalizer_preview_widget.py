from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QRectF, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

# Split colore identico al renderer Tkinter originale: verde 0-50%, giallo
# 50-80%, rosso 80-100% dell'altezza.
_GREEN_SPLIT = 0.5
_YELLOW_SPLIT = 0.3  # frazione aggiuntiva (50% -> 80%)

_GREEN = QColor(0, 255, 0)
_YELLOW = QColor(255, 255, 0)
_RED = QColor(255, 0, 0)


class EqualizerPreviewWidget(QWidget):
    """
    Anteprima in-window dell'equalizzatore.

    Riceve le ampiezze per banda (già normalizzate e smussate) dal worker DSP
    tramite lo slot ``on_bands`` e ridisegna le barre nel paintEvent, sul thread
    GUI. Disegna anche lo sfondo (immagine scalata con opacità) o, in modalità
    video, un segnaposto testuale (il video gira solo nel renderer OpenGL).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._amplitudes = np.zeros(0, dtype=float)
        self._bg_pixmap: Optional[QPixmap] = None
        self._bg_alpha = 0.5
        self._video_placeholder = False
        self.setMinimumSize(200, 150)

    # --- Slot di aggiornamento dati -----------------------------------------
    @Slot(object)
    def on_bands(self, amplitudes) -> None:
        self._amplitudes = np.asarray(amplitudes, dtype=float)
        self.update()

    def clear_bars(self) -> None:
        self._amplitudes = np.zeros(0, dtype=float)
        self.update()

    # --- Sfondo --------------------------------------------------------------
    def set_background_image(self, path: str) -> None:
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self._bg_pixmap = pixmap
        self._video_placeholder = False
        self.update()

    def set_background_alpha(self, alpha: float) -> None:
        self._bg_alpha = float(alpha)
        self.update()

    def show_video_placeholder(self) -> None:
        self._video_placeholder = True
        self.update()

    # --- Rendering -----------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()

        # Sfondo nero (come il canvas '#000' originale).
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        if self._video_placeholder:
            painter.setPen(QColor(255, 255, 255))
            font = QFont("Roboto", 16, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Video di sfondo attivo\n(visibile in modalità Fullscreen)",
            )
        elif self._bg_pixmap is not None:
            painter.setOpacity(max(0.0, min(1.0, self._bg_alpha)))
            painter.drawPixmap(self.rect(), self._bg_pixmap)
            painter.setOpacity(1.0)

        self._draw_bars(painter, w, h)
        painter.end()

    def _draw_bars(self, painter: QPainter, w: int, h: int) -> None:
        num_bars = len(self._amplitudes)
        if num_bars == 0 or w <= 0 or h <= 0:
            return

        total_bar_width = w / num_bars
        bar_width = total_bar_width * 0.8
        bar_spacing = total_bar_width * 0.2

        green_cap = _GREEN_SPLIT * h
        yellow_cap = _YELLOW_SPLIT * h

        for i, amplitude in enumerate(self._amplitudes):
            x = i * total_bar_width + bar_spacing / 2
            total_height = float(amplitude) * h

            green_height = min(total_height, green_cap)
            yellow_height = min(max(total_height - green_height, 0.0), yellow_cap)
            red_height = max(total_height - green_height - yellow_height, 0.0)

            if green_height > 0:
                painter.fillRect(
                    QRectF(x, h - green_height, bar_width, green_height), _GREEN
                )
            if yellow_height > 0:
                painter.fillRect(
                    QRectF(x, h - green_height - yellow_height, bar_width, yellow_height),
                    _YELLOW,
                )
            if red_height > 0:
                painter.fillRect(
                    QRectF(
                        x,
                        h - green_height - yellow_height - red_height,
                        bar_width,
                        red_height,
                    ),
                    _RED,
                )
