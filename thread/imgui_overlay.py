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

    # --- UI: 5 schede che rispecchiano il pannello PySide6 ---
    def _build_ui(self) -> None:
        r = self._renderer
        mode = int(r._mode)         # 0 Barre, 1 Radiale, 2 Oscilloscopio, 3 Linea/Area
        cmode = int(r._color_mode)  # 0 Classico, 1 Tinta, 2 Gradiente, 3 Spettro

        if self._first_ui:
            imgui.set_next_window_size((440, 600))
            self._first_ui = False
        imgui.begin("Impostazioni")

        if imgui.begin_tab_bar("sw_tabs"):
            # ---------------- Visualizzazione ----------------
            sel, _ = imgui.begin_tab_item("Visualizzazione")
            if sel:
                changed, idx = imgui.combo("Modalità", mode,
                                           ["Barre", "Radiale", "Oscilloscopio", "Linea/Area"])
                if changed:
                    self._emit("set_viz_mode", int(idx))
                changed, val = imgui.slider_float("Trasparenza sfondo", float(r._bg_alpha or 0.0), 0.0, 1.0)
                if changed:
                    self._emit("set_alpha", float(val))
                changed, val = imgui.slider_float("Opacità barre", float(r._bars_alpha), 0.0, 1.0)
                if changed:
                    self._emit("set_bars_alpha", float(val))
                imgui.end_tab_item()

            # ---------------- Forma ----------------
            sel, _ = imgui.begin_tab_item("Forma")
            if sel:
                if mode in (0, 1):
                    changed, val = imgui.slider_float("Larghezza barre", float(r._bar_width_frac), 0.1, 1.0)
                    if changed:
                        self._emit("set_bar_width", float(val))
                if mode == 0:
                    changed, val = imgui.checkbox("Cime arrotondate", bool(r._rounded))
                    if changed:
                        self._emit("set_rounded", bool(val))
                    changed, val = imgui.checkbox("Ancoraggio dal centro", bool(r._mirror))
                    if changed:
                        self._emit("set_bar_anchor", bool(val))
                if mode in (0, 1, 3):
                    changed, val = imgui.checkbox("Ordine simmetrico (bassi al centro)", bool(r._band_order_symmetric))
                    if changed:
                        self._emit("set_band_order", bool(val))
                if mode == 1:
                    changed, val = imgui.slider_float("Raggio foro", float(r._inner_radius), 0.0, 0.8)
                    if changed:
                        self._emit("set_inner_radius", float(val))
                if mode in (2, 3):
                    changed, val = imgui.slider_float("Spessore linea", float(r._line_thickness), 0.002, 0.06)
                    if changed:
                        self._emit("set_line_thickness", float(val))
                if mode == 3:
                    changed, val = imgui.checkbox("Riempi area", bool(r._fill))
                    if changed:
                        self._emit("set_fill", bool(val))
                if mode == 2:
                    changed, val = imgui.checkbox("Specchiato", bool(r._osc_mirror))
                    if changed:
                        self._emit("set_osc_mirror", bool(val))
                    changed, val = imgui.slider_float("Fluidità", float(r._osc_smoothing), 0.0, 1.0)
                    if changed:
                        self._emit("set_osc_smoothing", float(val))
                imgui.end_tab_item()

            # ---------------- Colore ----------------
            sel, _ = imgui.begin_tab_item("Colore")
            if sel:
                changed, idx = imgui.combo("Modalità colore", cmode,
                                           ["Classico", "Tinta unica", "Gradiente", "Spettro"])
                if changed:
                    self._emit("set_color_mode", int(idx))
                if cmode == 0:
                    changed, val = imgui.slider_float("Soglia giallo", float(r._green_split), 0.0, 1.0)
                    if changed:
                        self._emit("set_green_split", float(val))
                    changed, val = imgui.slider_float("Soglia rosso", float(r._yellow_split), 0.0, 1.0)
                    if changed:
                        self._emit("set_yellow_split", float(val))
                if cmode in (1, 2):
                    label_a = "Colore barre" if cmode == 1 else "Colore base"
                    changed, col = imgui.color_edit3(label_a, [float(c) for c in r._color_a])
                    if changed:
                        self._emit("set_color_a", (float(col[0]), float(col[1]), float(col[2])))
                if cmode == 2:
                    changed, col = imgui.color_edit3("Colore cima", [float(c) for c in r._color_b])
                    if changed:
                        self._emit("set_color_b", (float(col[0]), float(col[1]), float(col[2])))
                imgui.end_tab_item()

            # ---------------- Audio ----------------
            sel, _ = imgui.begin_tab_item("Audio")
            if sel:
                changed, val = imgui.slider_float("Soglia rumore", float(r._noise_threshold), 0.0, 1.0)
                if changed:
                    self._emit("set_noise_threshold", float(val))
                changed, val = imgui.slider_int("Bande di frequenza", int(r._n_bands), 1, 100)
                if changed:
                    self._emit("set_frequency_bands", int(val))
                changed, val = imgui.slider_int("Pavimento dB", int(round(r._db_floor)), -90, -25)
                if changed:
                    self._emit("set_db_floor", float(val))
                changed, val = imgui.slider_int("Tetto dB", int(round(r._db_ceiling)), -20, 6)
                if changed:
                    self._emit("set_db_ceiling", float(val))
                changed, val = imgui.slider_int("Attacco (ms)", int(round(r._attack_tau * 1000)), 1, 100)
                if changed:
                    self._emit("set_attack_tau", val / 1000.0)
                changed, val = imgui.slider_int("Rilascio (ms)", int(round(r._release_tau * 1000)), 20, 1000)
                if changed:
                    self._emit("set_release_tau", val / 1000.0)
                changed, val = imgui.slider_float("Tilt (dB/ottava)", float(r._tilt_db_per_oct), -6.0, 6.0)
                if changed:
                    self._emit("set_tilt_db_per_oct", float(val))
                imgui.end_tab_item()

            # ---------------- Effetti ----------------
            sel, _ = imgui.begin_tab_item("Effetti")
            if sel:
                if mode == 0:  # peak-cap solo in modalità Barre
                    changed, val = imgui.checkbox("Peak-cap", bool(r._peakcap_enabled))
                    if changed:
                        self._emit("set_peakcap_enabled", bool(val))
                    changed, col = imgui.color_edit3("Colore cap", [float(c) for c in r._peakcap_color])
                    if changed:
                        self._emit("set_peakcap_color", (float(col[0]), float(col[1]), float(col[2])))
                    changed, val = imgui.slider_int("Hold (ms)", int(round(r._peakcap_hold * 1000)), 0, 2000)
                    if changed:
                        self._emit("set_peakcap_hold", val / 1000.0)
                    changed, val = imgui.slider_float("Caduta (/s)", float(r._peakcap_fall), 0.1, 3.0)
                    if changed:
                        self._emit("set_peakcap_fall", float(val))
                changed, val = imgui.checkbox("Glow", bool(r._bloom_enabled))
                if changed:
                    self._emit("set_bloom_enabled", bool(val))
                changed, val = imgui.slider_float("Intensità glow", float(r._bloom_intensity), 0.0, 3.0)
                if changed:
                    self._emit("set_bloom_intensity", float(val))
                changed, val = imgui.checkbox("Beat pulse", bool(r._beat_enabled))
                if changed:
                    self._emit("set_beat_enabled", bool(val))
                changed, val = imgui.slider_float("Intensità zoom", float(r._beat_intensity), 0.0, 0.3)
                if changed:
                    self._emit("set_beat_intensity", float(val))
                changed, val = imgui.slider_float("Sensibilità", float(r._beat_sensitivity), 1.05, 3.0)
                if changed:
                    self._emit("set_beat_sensitivity", float(val))
                imgui.end_tab_item()

            imgui.end_tab_bar()

        imgui.end()
