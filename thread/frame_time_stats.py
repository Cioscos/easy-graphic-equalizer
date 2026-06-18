"""
Statistiche aggregate sul frame time per la diagnosi di performance del renderer
OpenGL. Puro Python, nessuna dipendenza GL → importabile e verificabile headless.

Accumula i delta-tempo dei frame e, ogni `report_every` frame, produce una riga
di riepilogo (avg/min/max + percentuale di frame oltre il budget) usata sotto il
flag PERF_LOGS per confrontare le performance prima/dopo un intervento.
"""
from __future__ import annotations


class FrameTimeStats:
    def __init__(self, report_every: int = 120, budget_s: float = 1.0 / 60.0) -> None:
        """
        Args:
            report_every: numero di frame per ogni riga di riepilogo (>= 1).
            budget_s: budget di frame time in secondi; i frame più lenti di questo
                contano come "oltre budget" (tipicamente 1/refresh_rate).
        """
        if report_every < 1:
            raise ValueError("report_every must be >= 1")
        self._report_every = int(report_every)
        self._budget_s = float(budget_s)
        self._reset()

    def _reset(self) -> None:
        self._count = 0
        self._sum = 0.0
        self._min = float("inf")
        self._max = 0.0
        self._over_budget = 0

    def add(self, dt: float) -> str | None:
        """
        Accumula un delta-tempo di frame (secondi). Ritorna una stringa di
        riepilogo ogni `report_every` frame (e poi azzera la finestra), altrimenti
        None.
        """
        self._count += 1
        self._sum += dt
        if dt < self._min:
            self._min = dt
        if dt > self._max:
            self._max = dt
        if dt > self._budget_s:
            self._over_budget += 1

        if self._count < self._report_every:
            return None

        report = self._format()
        self._reset()
        return report

    def _format(self) -> str:
        avg = self._sum / self._count
        over_pct = 100.0 * self._over_budget / self._count
        avg_fps = 1.0 / avg if avg > 0 else 0.0
        return (
            f"[PERF] {self._count} frame: "
            f"avg={avg * 1e3:.2f}ms min={self._min * 1e3:.2f}ms max={self._max * 1e3:.2f}ms "
            f"oltre-budget={over_pct:.0f}% "
            f"(budget={self._budget_s * 1e3:.2f}ms, ~{avg_fps:.0f}fps medi)"
        )
