"""
Menù impostazioni in-window: overlay Dear ImGui sulla finestra GLFW/OpenGL del
renderer. Vive interamente sul thread OpenGL (lo stesso che possiede il contesto
GL e gli attributi del renderer), quindi legge lo stato live del renderer
direttamente e scrive i messaggi di controllo sulla stessa coda della GUI.
"""
# NB: importare il glfw di sistema PRIMA di imgui_bundle (così si usa quello standard).
import glfw
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

from thread.menu_keys import TOGGLE_KEY_NAMES


class ImGuiOverlay:
    def __init__(self, window, renderer, control_queue=None, on_change=None):
        self._window = window
        self._renderer = renderer
        self._control_queue = control_queue
        self._on_change = on_change
        self._open = False
        self._first_ui = True   # dimensiona la finestra solo alla prima apparizione

        # Il contesto ImGui va creato PRIMA del backend (altrimenti RuntimeError).
        imgui.create_context()
        # attach_callbacks=False: registriamo noi le callback GLFW in run() (il backend
        # le sovrascriverebbe senza concatenarle alla nostra key_callback).
        self.impl = GlfwRenderer(window, attach_callbacks=False)

    # --- stato apertura ---
    def is_open(self) -> bool:
        return self._open

    def set_open(self, value: bool) -> None:
        self._open = bool(value)

    def toggle(self) -> None:
        self._open = not self._open

    # --- ciclo per-frame ---
    def render_frame(self) -> None:
        """Da chiamare ogni frame, dopo il render della visualizzazione e prima dello
        swap. Quando il menù è chiuso non costruisce widget (costo trascurabile)."""
        self.impl.process_inputs()
        imgui.new_frame()
        if self._open:
            self._build_ui()
        imgui.render()
        self.impl.render(imgui.get_draw_data())

    def shutdown(self) -> None:
        """Libera le risorse GL del backend e distrugge il contesto ImGui. Richiede
        contesto GL ancora attivo (chiamare prima di destroy_window/terminate)."""
        try:
            self.impl.shutdown()
        except Exception:
            pass
        try:
            imgui.destroy_context()
        except Exception:
            pass

    # --- invio modifiche ---
    def _emit(self, msg_type: str, value) -> None:
        """Applica la modifica al renderer (control queue) e notifica la GUI (on_change),
        esattamente come uno slider del pannello PySide6."""
        msg = {"type": msg_type, "value": value}
        if self._control_queue is not None:
            self._control_queue.put(msg)
        if self._on_change is not None:
            self._on_change(msg)

    # --- UI (riempita in Task 5) ---
    def _build_ui(self) -> None:
        if self._first_ui:
            imgui.set_next_window_size((440, 600))
            self._first_ui = False
        imgui.begin("Impostazioni")
        imgui.text("Menu in costruzione")
        imgui.end()
