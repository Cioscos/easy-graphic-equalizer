# gui/theme.py
"""
Token di stile condivisi dell'interfaccia (colori e font).

Centralizza i valori ricorrenti usati dai widget PySide6, per evitare
stringhe esadecimali e definizioni di font duplicate nel codice della GUI.

I font sono esposti come *factory* (funzioni che restituiscono un ``QFont``)
e non come istanze a livello di modulo: ``QFont`` va costruito dopo che esiste
una ``QApplication``, mentre questo modulo viene importato prima.
"""

from PySide6.QtGui import QFont

# --- Famiglia e dimensioni dei font ---
_FONT_FAMILY = "Roboto"  # se assente, Qt sostituisce con il font di sistema


def font_title() -> QFont:
    return QFont(_FONT_FAMILY, 18, QFont.Weight.Bold)


def font_button() -> QFont:
    return QFont(_FONT_FAMILY, 16)


def font_label() -> QFont:
    return QFont(_FONT_FAMILY, 14)


def font_body() -> QFont:
    return QFont(_FONT_FAMILY, 12)


# --- Colori dei pulsanti (coppie colore base / hover) ---
COLOR_NEUTRAL = "#6c757d"          # pulsante disabilitato / secondario (es. Help)
COLOR_NEUTRAL_HOVER = "#5a6268"
COLOR_FULLSCREEN = "#17a2b8"
COLOR_FULLSCREEN_HOVER = "#138496"
COLOR_START = "#28a745"
COLOR_START_HOVER = "#218838"
COLOR_STOP = "#dc3545"
COLOR_STOP_HOVER = "#c82333"
