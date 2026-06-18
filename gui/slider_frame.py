from enum import Enum
from typing import Callable, Optional, Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from gui.theme import font_body

# Risoluzione interna per gli slider a virgola mobile senza un numero di step
# esplicito: QSlider lavora solo su interi, quindi mappiamo [from_, to] su
# [0, _DOUBLE_RESOLUTION].
_DOUBLE_RESOLUTION = 1000


class SliderCustomFrame(QWidget):
    """Wrappa uno slider con etichetta, valore corrente ed eventuale warning."""

    class ValueType(Enum):
        INT = "int"
        DOUBLE = "double"

    def __init__(
        self,
        parent=None,
        *,
        header_name: str = "SliderCustomFrame",
        initial_value: Union[float, int] = 0.1,
        from_: float = 0,
        to: float = 1,
        steps: Optional[int] = None,
        command: Optional[Callable] = None,
        warning_text: Optional[str] = None,
        warning_trigger_value: Optional[float] = None,
        value_type: "SliderCustomFrame.ValueType" = ValueType.DOUBLE,
    ):
        super().__init__(parent)

        if warning_trigger_value and not warning_text:
            raise ValueError("No warning text when warning_trigger_value set")

        self._command = command
        self._from = from_
        self._to = to
        self._warning_trigger_value = warning_trigger_value
        self._is_int = value_type == self.ValueType.INT
        self._ticks = None if self._is_int else int(steps) if steps else _DOUBLE_RESOLUTION

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        row = QHBoxLayout()
        header = QLabel(header_name)
        header.setFont(font_body())

        self._slider = QSlider(Qt.Orientation.Horizontal)
        if self._is_int:
            self._slider.setMinimum(int(from_))
            self._slider.setMaximum(int(to))
            self._slider.setSingleStep(1)
            self._slider.setValue(int(round(initial_value)))
        else:
            self._slider.setMinimum(0)
            self._slider.setMaximum(self._ticks)
            self._slider.setValue(self._to_ticks(initial_value))

        self._value_label = QLabel()
        self._value_label.setFont(font_body())
        self._value_label.setMinimumWidth(36)
        self._value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        row.addWidget(header)
        row.addWidget(self._slider, 1)
        row.addWidget(self._value_label)
        layout.addLayout(row)

        self.warning_label = QLabel(warning_text or "")
        self.warning_label.setFont(font_body())
        self.warning_label.setStyleSheet("color: yellow;")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.warning_label)
        self.warning_label.setVisible(
            bool(warning_trigger_value and self.get_value() >= warning_trigger_value)
        )

        # Aggiorna l'etichetta iniziale *prima* di collegare il segnale, così la
        # callback utente non viene invocata durante la costruzione.
        self._refresh_value_label()
        self._slider.valueChanged.connect(self._on_changed)

    def _to_ticks(self, value: float) -> int:
        span = self._to - self._from
        if span == 0:
            return 0
        return int(round((value - self._from) / span * self._ticks))

    def _from_ticks(self, ticks: int) -> float:
        span = self._to - self._from
        return self._from + (ticks / self._ticks) * span

    def _refresh_value_label(self) -> None:
        value = self.get_value()
        self._value_label.setText(str(int(value) if self._is_int else round(value, 2)))

    def _on_changed(self, _) -> None:
        self._refresh_value_label()
        if self._warning_trigger_value:
            self.warning_label.setVisible(self._warning_trigger_value <= self.get_value())
        if self._command:
            self._command(self.get_value())

    def get_value(self) -> Union[float, int]:
        """Restituisce il valore corrente dello slider (int o float)."""
        if self._is_int:
            return int(self._slider.value())
        return self._from_ticks(self._slider.value())

    def set_value(self, value: Union[float, int]) -> None:
        """Imposta il valore mostrato senza invocare la command (caricamento profili)."""
        self._slider.blockSignals(True)
        if self._is_int:
            self._slider.setValue(int(round(value)))
        else:
            self._slider.setValue(self._to_ticks(value))
        self._slider.blockSignals(False)
        self._refresh_value_label()
        if self._warning_trigger_value:
            self.warning_label.setVisible(self._warning_trigger_value <= self.get_value())
