"""Helper puri per i parametri di rendering (testabili senza contesto GL)."""
import math


def sanitize_fps(value) -> float | None:
    """fps dai metadati video: accetta solo numeri finiti e positivi.

    ffmpeg può riportare inf/nan/0 su file con metadati rotti; un fps non
    finito produrrebbe un frame-interval pari a 0 e una divisione per zero
    nel pacing video del thread GL."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) \
            and math.isfinite(value) and value > 0:
        return float(value)
    return None


def effective_splits(green: float, yellow: float) -> tuple[float, float]:
    """Soglie effettive (verde→giallo, giallo→rosso) del color-mode classico.

    I due valori arrivano da messaggi di controllo indipendenti: il vincolo
    green <= yellow va applicato all'USO, non alla ricezione, altrimenti il
    risultato dipende dall'ordine dei messaggi (caricare un profilo poteva
    lasciare il renderer con una soglia diversa da quella mostrata in GUI).
    Clampa entrambi in [0, 1] e li ritorna ordinati.
    """
    g = min(max(float(green), 0.0), 1.0)
    y = min(max(float(yellow), 0.0), 1.0)
    return (g, y) if g <= y else (y, g)
