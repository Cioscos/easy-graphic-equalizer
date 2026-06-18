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


def font_section() -> QFont:
    """Titoli di sezione/gruppo (livello intermedio della scala tipografica)."""
    return QFont(_FONT_FAMILY, 13, QFont.Weight.DemiBold)


def font_body() -> QFont:
    return QFont(_FONT_FAMILY, 12)


# --- Token di spaziatura (ritmo verticale coerente nei layout) ---
SPACE_XS = 2
SPACE_SM = 6
SPACE_MD = 10
SPACE_LG = 16
SPACE_XL = 24

# --- Colore d'accento del brand ---
# Verde vivido (già preset di ColorPickerFrame): leggibile sia in tema chiaro
# che scuro e coerente con la rampa verde→giallo→rosso dell'anteprima. Usato
# come 'primary' di qdarktheme (selezioni/focus/slider) e nella nav laterale.
ACCENT = "#22d36a"
ACCENT_HOVER = "#1bb259"

# --- Colori dei pulsanti (coppie colore base / hover) ---
COLOR_NEUTRAL = "#6c757d"          # pulsante disabilitato / secondario (es. Help)
COLOR_NEUTRAL_HOVER = "#5a6268"
COLOR_FULLSCREEN = "#17a2b8"
COLOR_FULLSCREEN_HOVER = "#138496"
COLOR_START = "#28a745"
COLOR_START_HOVER = "#218838"
COLOR_STOP = "#dc3545"
COLOR_STOP_HOVER = "#c82333"


def additional_qss() -> str:
    """
    Sottile strato QSS *strutturale* applicato sopra qdarktheme.

    Stilizza solo geometria/spaziatura e l'accento; i colori di base sono
    delegati alla palette di qdarktheme (``palette(...)``) così il foglio
    funziona identico in tema chiaro e scuro. Niente regole ``QPushButton``:
    AnimatedButton gestisce il proprio stile per-widget.
    """
    return f"""
    QListWidget#navList {{
        border: none;
        background: transparent;
        outline: none;
    }}
    QListWidget#navList::item {{
        padding: 8px 12px;
        margin: 1px 0;
        border-radius: 6px;
        min-height: 22px;
    }}
    QListWidget#navList::item:hover {{
        background: palette(midlight);
    }}
    QListWidget#navList::item:selected {{
        background: {ACCENT};
        color: #ffffff;
    }}
    QGroupBox {{
        margin-top: {SPACE_LG}px;
        border: 1px solid palette(mid);
        border-radius: 8px;
        padding: {SPACE_MD}px {SPACE_SM}px {SPACE_SM}px {SPACE_SM}px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 4px;
    }}
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    """


def theme_kwargs() -> dict:
    """
    Argomenti comuni per ``qdarktheme.setup_theme``: accento di brand + QSS.

    Unica sorgente di verità per i due call-site (``main.py`` all'avvio e il
    gestore del cambio tema): ``setup_theme`` rigenera l'intero stylesheet, quindi
    accento e QSS vanno ri-passati a ogni cambio, altrimenti vengono persi.
    """
    return {"custom_colors": {"primary": ACCENT}, "additional_qss": additional_qss()}
