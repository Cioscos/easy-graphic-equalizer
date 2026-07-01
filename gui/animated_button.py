# gui/animated_button.py
"""
Pulsante animato con effetti hover e click per PySide6.

Riproduce il comportamento del vecchio AnimatedButton CustomTkinter:
- variazione di luminosità del colore di sfondo su hover/press (animata);
- leggero effetto di scala ("pop") su hover/click.

La luminosità è animata con una QPropertyAnimation su una proprietà QColor
(`bg`); la scala con una QPropertyAnimation su una proprietà float (`scale`)
consumata nel paintEvent tramite una trasformazione del QStylePainter.
"""

from typing import Callable, Optional

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter
from PySide6.QtWidgets import (
    QPushButton,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
)

from gui.theme import COLOR_NEUTRAL


class AnimatedButton(QPushButton):
    """Pulsante con animazioni fluide per hover e click (scala + colore)."""

    def __init__(
        self,
        parent=None,
        *,
        text: str = "",
        command: Optional[Callable] = None,
        fg_color: str = "#1f538d",
        hover_color: Optional[str] = None,
        font=None,
        corner_radius: int = 10,
        height: int = 45,
        enabled: bool = True,
        icon: Optional[QIcon] = None,
        icon_size: int = 20,
        animation_duration: int = 200,
        hover_scale: float = 1.05,
        click_scale: float = 0.95,
    ):
        super().__init__(text, parent)

        # Parametri di animazione
        self.animation_duration = animation_duration
        self.hover_scale = hover_scale
        self.click_scale = click_scale

        # Colore base (l'hover/press sono derivati schiarendo/scurendo)
        self._base_color = QColor(fg_color)
        self._hover_color_override = QColor(hover_color) if hover_color else None
        self._corner_radius = corner_radius

        # Stato animato
        self._scale_val = 1.0
        self._bg_val = QColor(self._base_color)

        if font is not None:
            self.setFont(font)
        if height:
            self.setFixedHeight(height)
        if icon is not None:
            self.setIcon(icon)
            self.setIconSize(QSize(icon_size, icon_size))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setEnabled(enabled)

        if command is not None:
            self.clicked.connect(lambda: command())

        # Animazioni
        self._scale_anim = QPropertyAnimation(self, b"scale", self)
        self._scale_anim.setDuration(animation_duration)
        self._scale_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._color_anim = QPropertyAnimation(self, b"bg", self)
        self._color_anim.setDuration(animation_duration)
        self._color_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._apply_stylesheet()

    # --- Proprietà animabili -------------------------------------------------
    def _get_scale(self) -> float:
        return self._scale_val

    def _set_scale(self, value: float) -> None:
        self._scale_val = value
        self.update()

    scale = Property(float, _get_scale, _set_scale)

    def _get_bg(self) -> QColor:
        return self._bg_val

    def _set_bg(self, color: QColor) -> None:
        self._bg_val = color
        self._apply_stylesheet()

    bg = Property(QColor, _get_bg, _set_bg)

    # --- Colori derivati -----------------------------------------------------
    def _hover_brush(self) -> QColor:
        if self._hover_color_override is not None:
            return QColor(self._hover_color_override)
        return self._base_color.lighter(115)

    def _press_brush(self) -> QColor:
        return self._base_color.darker(110)

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {self._bg_val.name()};
                color: white;
                border: none;
                border-radius: {self._corner_radius}px;
                padding: 6px 12px;
            }}
            QPushButton:disabled {{
                background-color: {COLOR_NEUTRAL};
                color: #cfcfcf;
            }}
            """
        )

    # --- Animazioni ----------------------------------------------------------
    def _animate_scale(self, target: float) -> None:
        self._scale_anim.stop()
        self._scale_anim.setStartValue(self._scale_val)
        self._scale_anim.setEndValue(target)
        self._scale_anim.start()

    def _animate_bg(self, target: QColor) -> None:
        self._color_anim.stop()
        self._color_anim.setStartValue(QColor(self._bg_val))
        self._color_anim.setEndValue(target)
        self._color_anim.start()

    # --- Eventi --------------------------------------------------------------
    def enterEvent(self, event):
        if self.isEnabled():
            self._animate_scale(self.hover_scale)
            self._animate_bg(self._hover_brush())
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self.isEnabled():
            self._animate_scale(1.0)
            self._animate_bg(QColor(self._base_color))
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self._animate_scale(self.click_scale)
            self._animate_bg(self._press_brush())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.isEnabled():
            if self.rect().contains(event.position().toPoint()):
                self._animate_scale(self.hover_scale)
                self._animate_bg(self._hover_brush())
            else:
                self._animate_scale(1.0)
                self._animate_bg(QColor(self._base_color))

    def paintEvent(self, event):
        # Senza scala disegna normalmente.
        if abs(self._scale_val - 1.0) < 1e-3:
            super().paintEvent(event)
            return

        option = QStyleOptionButton()
        self.initStyleOption(option)

        painter = QStylePainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        painter.translate(cx, cy)
        painter.scale(self._scale_val, self._scale_val)
        painter.translate(-cx, -cy)
        painter.drawControl(QStyle.ControlElement.CE_PushButton, option)

    # --- API pubblica --------------------------------------------------------
    def set_animation_params(
        self,
        duration: Optional[int] = None,
        hover_scale: Optional[float] = None,
        click_scale: Optional[float] = None,
    ) -> None:
        """Modifica i parametri dell'animazione dinamicamente."""
        if duration is not None:
            self.animation_duration = duration
            self._scale_anim.setDuration(duration)
            self._color_anim.setDuration(duration)
        if hover_scale is not None:
            self.hover_scale = hover_scale
        if click_scale is not None:
            self.click_scale = click_scale

    def update_base_colors(self, fg_color: str, hover_color: Optional[str] = None) -> None:
        """Aggiorna i colori base del pulsante mantenendo le animazioni."""
        self._base_color = QColor(fg_color)
        self._hover_color_override = QColor(hover_color) if hover_color else None
        # Applica SUBITO il nuovo colore: verso l'hover se il mouse è sopra
        # (caso tipico: il click sul footer che alterna Avvia/Ferma avviene
        # col mouse sul bottone), altrimenti verso il base.
        self._color_anim.stop()
        self._set_bg(QColor(self._hover_brush() if self.underMouse() else self._base_color))
