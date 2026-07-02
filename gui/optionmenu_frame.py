from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from gui.theme import font_body


class OptionMenuCustomFrame(QWidget):
    """Wrappa un menu a tendina (QComboBox) con un'etichetta di intestazione."""

    def __init__(
        self,
        parent=None,
        *,
        header_name: str = "OptionMenuCustomFrame",
        values: Optional[list] = None,
        initial_value: Optional[str] = None,
        command: Optional[Callable] = None,
    ):
        super().__init__(parent)

        if not values:
            values = ["--Select--"]

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        header = QLabel(header_name)
        header.setFont(font_body())

        self._combo = QComboBox()
        self._combo.setFont(font_body())
        self._combo.addItems([str(v) for v in values])

        initial = str(initial_value) if initial_value is not None else str(values[0])
        # Imposta il valore iniziale prima di collegare il segnale: evita che la
        # callback venga invocata durante la costruzione.
        idx = self._combo.findText(initial)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)

        layout.addWidget(header)
        layout.addWidget(self._combo, 1)

        self._command = command
        if command is not None:
            self._combo.currentTextChanged.connect(command)

    def get_value(self) -> str:
        """Restituisce il valore corrente del menu."""
        return self._combo.currentText()

    def set_value(self, value: str) -> None:
        """Imposta il valore del menu senza scatenare la callback. Un valore
        sconosciuto lascia la selezione invariata (con log): dopo la
        validazione dei profili non dovrebbe più accadere — se accade è un
        bug a monte da rendere visibile, non da nascondere."""
        idx = self._combo.findText(str(value))
        if idx < 0:
            print(f"OptionMenu: valore sconosciuto {value!r}, selezione invariata")
            return
        self._combo.blockSignals(True)
        self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)
