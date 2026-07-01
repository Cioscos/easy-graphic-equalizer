"""Helper puri per i parametri di rendering (testabili senza contesto GL)."""


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
