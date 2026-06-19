"""
Mappa nome-tasto → codice GLFW per il tasto-toggle del menù in-window.
Modulo minimale (importa solo glfw) così è testabile headless senza imgui-bundle.
"""
import glfw

# Nomi selezionabili dalla GUI (combo "Tasto menù"). Tab è escluso di proposito:
# ImGui lo usa per la navigazione tra i campi.
TOGGLE_KEY_NAMES = [
    "F1", "F2", "F3", "F4", "F5", "F6",
    "F7", "F8", "F9", "F10", "F11", "F12",
    "M", "N", "P",
]

_NAME_TO_GLFW = {
    "F1": glfw.KEY_F1, "F2": glfw.KEY_F2, "F3": glfw.KEY_F3, "F4": glfw.KEY_F4,
    "F5": glfw.KEY_F5, "F6": glfw.KEY_F6, "F7": glfw.KEY_F7, "F8": glfw.KEY_F8,
    "F9": glfw.KEY_F9, "F10": glfw.KEY_F10, "F11": glfw.KEY_F11, "F12": glfw.KEY_F12,
    "M": glfw.KEY_M, "N": glfw.KEY_N, "P": glfw.KEY_P,
}


def toggle_key_to_glfw(name: str) -> int:
    """Codice GLFW per il nome dato; F1 come fallback per nomi sconosciuti."""
    return _NAME_TO_GLFW.get(str(name), glfw.KEY_F1)
