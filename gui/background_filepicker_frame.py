import os
from typing import Callable, Optional

from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)


class BackgroundFilepickerFrame(QWidget):
    """Wrappa un selettore di file (campo + sfoglia + applica) in un riquadro."""

    def __init__(
        self,
        parent=None,
        *,
        header_name: str = "BackgroundFilepickerFrame",
        placeholder_text: Optional[str] = None,
        apply_command: Optional[Callable] = None,
        start_dir: Optional[str] = None,
    ):
        super().__init__(parent)
        self._start_dir = start_dir

        layout = QGridLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setColumnStretch(0, 2)

        self.label = QLabel(header_name)
        layout.addWidget(self.label, 0, 0, 1, 2)

        self.image_path_entry = QLineEdit()
        if placeholder_text:
            self.image_path_entry.setPlaceholderText(placeholder_text)
        layout.addWidget(self.image_path_entry, 1, 0)

        self.file_picker_button = QPushButton("...")
        self.file_picker_button.setFixedWidth(40)
        self.file_picker_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.file_picker_button, 1, 1)

        self.apply_button = QPushButton("Apply")
        if apply_command is not None:
            self.apply_button.clicked.connect(lambda: apply_command())
        layout.addWidget(self.apply_button, 2, 0, 1, 2)

    def open_file_dialog(self) -> None:
        # Dir iniettata dal chiamante (ResourceManager): il path relativo alla
        # CWD era sbagliato nel frozen build e con CWD ≠ radice del repo.
        start_dir = self._start_dir if self._start_dir and os.path.isdir(self._start_dir) else ""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select a File",
            start_dir,
            "Image files (*.png *.jpg *.jpeg *.gif);;"
            "Video files (*.mp4 *.avi *.mov *.mkv);;"
            "All files (*)",
        )
        if filename:
            self.image_path_entry.setText(filename)

    def get_filename(self) -> Optional[str]:
        """Restituisce il path corrente (digitato a mano o scelto dal dialog)."""
        text = self.image_path_entry.text().strip()
        return text or None

    def set_filename(self, path: str) -> None:
        """Mostra `path` nel campo (senza applicarlo): usato dal caricamento profili."""
        self.image_path_entry.setText(path or "")
