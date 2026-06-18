from typing import Callable, List, Optional

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget

from gui.theme import font_body, font_label, SPACE_SM

# Voce mostrata quando non esiste alcun profilo salvato.
PLACEHOLDER = "— Nessun profilo —"


class ProfileBarFrame(QWidget):
    """Barra orizzontale: combo dei profili + Salva / Salva come… / Elimina / Importa… / Esporta…."""

    def __init__(
        self,
        parent=None,
        *,
        on_select: Callable[[Optional[str]], None],
        on_save: Callable[[], None],
        on_save_as: Callable[[], None],
        on_delete: Callable[[], None],
        on_import: Callable[[], None],
        on_export: Callable[[], None],
    ):
        super().__init__(parent)
        self._on_select = on_select

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(SPACE_SM)

        label = QLabel("Profilo:")
        label.setFont(font_label())
        layout.addWidget(label)

        self._combo = QComboBox()
        self._combo.setFont(font_body())
        self._combo.setMinimumWidth(160)
        self._combo.addItem(PLACEHOLDER)
        self._combo.currentTextChanged.connect(self._emit_select)
        layout.addWidget(self._combo, 1)

        for text, callback in (
            ("Salva", on_save),
            ("Salva come…", on_save_as),
            ("Elimina", on_delete),
            ("Importa…", on_import),
            ("Esporta…", on_export),
        ):
            btn = QPushButton(text)
            btn.setFont(font_body())
            btn.clicked.connect(lambda _checked=False, c=callback: c())
            layout.addWidget(btn)

    def _emit_select(self, text: str) -> None:
        self._on_select(None if text == PLACEHOLDER else (text or None))

    def set_profiles(self, names: List[str], selected: Optional[str]) -> None:
        """Ripopola la combo. Blocca i segnali: NON scatena on_select."""
        self._combo.blockSignals(True)
        self._combo.clear()
        if names:
            self._combo.addItems(names)
        else:
            self._combo.addItem(PLACEHOLDER)
        if selected and selected in names:
            self._combo.setCurrentText(selected)
        self._combo.blockSignals(False)

    def current_name(self) -> Optional[str]:
        text = self._combo.currentText()
        return None if text == PLACEHOLDER else (text or None)
