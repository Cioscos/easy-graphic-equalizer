# gui/theme.py
"""
Token di stile condivisi dell'interfaccia (colori e font).

Centralizza i valori ricorrenti usati dai widget CustomTkinter, per evitare
stringhe esadecimali e tuple di font duplicate nel codice della GUI.
"""

# --- Font ---
FONT_TITLE = ("Roboto", 18, "bold")
FONT_BUTTON = ("Roboto", 16)
FONT_LABEL = ("Roboto", 14)
FONT_BODY = ("Roboto", 12)

# --- Colori dei pulsanti (coppie colore base / hover) ---
COLOR_NEUTRAL = "#6c757d"          # pulsante disabilitato / secondario (es. Help)
COLOR_NEUTRAL_HOVER = "#5a6268"
COLOR_FULLSCREEN = "#17a2b8"
COLOR_FULLSCREEN_HOVER = "#138496"
COLOR_START = "#28a745"
COLOR_START_HOVER = "#218838"
COLOR_STOP = "#dc3545"
COLOR_STOP_HOVER = "#c82333"

# --- Colori della listbox dei device (sfondo per tema scuro / chiaro) ---
COLOR_LISTBOX_DARK = "#5a5a5a"
COLOR_LISTBOX_LIGHT = "#fff"
