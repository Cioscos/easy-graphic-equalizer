"""
Statistiche aggregate sul frame time per la diagnosi di performance del renderer
OpenGL. Puro Python, nessuna dipendenza GL → importabile e verificabile headless.

Accumula i delta-tempo dei frame e, ogni `report_every` frame, produce una riga
di riepilogo usata sotto il flag PERF_LOGS per confrontare le performance. Riporta
la MEDIANA (robusta agli outlier, a differenza della media — un singolo hitch non
falsa il numero), il massimo (cattura gli hitch) e il numero di "lag spike": frame
il cui tempo supera `lag_factor`× il budget di refresh, così la normale jitter del
VSync attorno all'intervallo di refresh non viene contata come problema.
"""
from __future__ import annotations


class FrameTimeStats:
    def __init__(self, report_every: int = 120, budget_s: float = 1.0 / 60.0,
                 lag_factor: float = 1.5) -> None:
        """
        Args:
            report_every: numero di frame per ogni riga di riepilogo (>= 1).
            budget_s: budget di frame time in secondi (tipicamente 1/refresh_rate).
            lag_factor: un frame conta come "lag spike" se supera lag_factor*budget_s
                (default 1.5×: ignora la jitter del VSync, segna i veri hitch).
        """
        if report_every < 1:
            raise ValueError("report_every must be >= 1")
        self._report_every = int(report_every)
        self._budget_s = float(budget_s)
        self._lag_threshold = float(budget_s) * float(lag_factor)
        self._samples: list[float] = []

    def add(self, dt: float) -> str | None:
        """
        Accumula un delta-tempo di frame (secondi). Ritorna una stringa di
        riepilogo ogni `report_every` frame (e poi azzera la finestra), altrimenti
        None.
        """
        self._samples.append(dt)
        if len(self._samples) < self._report_every:
            return None
        report = self._format()
        self._samples = []
        return report

    def _format(self) -> str:
        s = sorted(self._samples)
        n = len(s)
        median = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
        avg = sum(s) / n
        mx = s[-1]
        lag = sum(1 for x in s if x > self._lag_threshold)
        median_fps = 1.0 / median if median > 0 else 0.0
        return (
            f"[PERF] {n} frame: "
            f"median={median * 1e3:.2f}ms avg={avg * 1e3:.2f}ms max={mx * 1e3:.2f}ms "
            f"lag-spikes(>{self._lag_threshold * 1e3:.1f}ms)={lag} "
            f"(~{median_fps:.0f}fps mediani)"
        )
