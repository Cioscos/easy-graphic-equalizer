from typing import Callable, Optional, Sequence, Tuple

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gui.theme import font_body

Rgb01 = Tuple[float, float, float]


def hex_to_rgb01(value: str) -> Rgb01:
    """'#rrggbb' (o 'rrggbb') -> (r, g, b) in [0,1]."""
    h = value.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)


def rgb01_to_hex(rgb: Rgb01) -> str:
    """(r, g, b) in [0,1] -> '#rrggbb'."""
    r, g, b = (max(0, min(255, int(round(c * 255.0)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


# Preset rapidi mostrati come piccoli swatch sotto il bottone principale.
_PRESETS = ["#22d36a", "#3bffd0", "#3b6bff", "#ff3bd0", "#ffae3b"]


class ColorPickerFrame(QWidget):
    """
    Etichetta + bottone-swatch che apre QColorDialog, più una riga di preset
    rapidi. Espone get_color()/set_color() in RGB normalizzato e invoca
    command(rgb) a ogni cambio (modello: BackgroundFilepickerFrame).
    """

    def __init__(
        self,
        parent=None,
        *,
        header_name: str = "ColorPickerFrame",
        initial_color: str = "#3b6bff",
        command: Optional[Callable[[Rgb01], None]] = None,
        presets: Optional[Sequence[str]] = None,
    ):
        super().__init__(parent)
        self._command = command
        self._rgb: Rgb01 = hex_to_rgb01(initial_color)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(3)

        header = QLabel(header_name)
        header.setFont(font_body())
        layout.addWidget(header)

        self._swatch = QPushButton()
        self._swatch.setFixedHeight(26)
        self._swatch.clicked.connect(self._open_dialog)
        layout.addWidget(self._swatch)

        presets_row = QHBoxLayout()
        presets_row.setSpacing(4)
        for hex_value in (presets or _PRESETS):
            btn = QPushButton()
            btn.setFixedSize(22, 18)
            btn.setStyleSheet(f"background-color: {hex_value}; border: 1px solid #555; border-radius: 4px;")
            btn.clicked.connect(lambda _checked=False, h=hex_value: self._apply(hex_to_rgb01(h)))
            presets_row.addWidget(btn)
        presets_row.addStretch(1)
        layout.addLayout(presets_row)

        self._refresh_swatch()

    def _open_dialog(self) -> None:
        r, g, b = self._rgb
        initial = QColor.fromRgbF(r, g, b)
        chosen = QColorDialog.getColor(initial, self, "Scegli un colore")
        if chosen.isValid():
            self._apply((chosen.redF(), chosen.greenF(), chosen.blueF()))

    def _apply(self, rgb: Rgb01) -> None:
        self._rgb = rgb
        self._refresh_swatch()
        if self._command is not None:
            self._command(self._rgb)

    def _refresh_swatch(self) -> None:
        self._swatch.setStyleSheet(
            f"background-color: {rgb01_to_hex(self._rgb)}; border: 1px solid #555; border-radius: 4px;"
        )

    def get_color(self) -> Rgb01:
        """Colore corrente in RGB normalizzato [0,1]."""
        return self._rgb

    def set_color(self, rgb: Rgb01) -> None:
        """Imposta il colore senza invocare la callback."""
        self._rgb = rgb
        self._refresh_swatch()
