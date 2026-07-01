"""
Verifica headless (niente pytest nel repo): asserzioni pure su render_params.
Eseguire dalla radice del repo con:  uv run python tests/test_render_params.py
Uscita 0 + "OK" = passato.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thread.render_params import effective_splits, sanitize_fps


def test_order_independent():
    assert effective_splits(0.5, 0.8) == (0.5, 0.8)
    # green oltre yellow: il vincolo si applica all'uso, ordinando — non
    # azzoppando uno dei due in base all'ordine di arrivo dei messaggi
    assert effective_splits(0.85, 0.3) == (0.3, 0.85)


def test_clamped():
    assert effective_splits(-1.0, 2.0) == (0.0, 1.0)
    assert effective_splits(1.5, 0.2) == (0.2, 1.0)


def test_sanitize_fps():
    assert sanitize_fps(30) == 30.0
    assert sanitize_fps(29.97) == 29.97
    # inf/nan/zero/negativi/non-numeri → None: con interval = 1/inf = 0.0
    # il pacing video sul thread GL dividerebbe per zero (ZeroDivisionError
    # → renderer chiuso per colpa di un metadato).
    assert sanitize_fps(float("inf")) is None
    assert sanitize_fps(float("nan")) is None
    assert sanitize_fps(0) is None
    assert sanitize_fps(-25) is None
    assert sanitize_fps("30") is None
    assert sanitize_fps(None) is None
    assert sanitize_fps(True) is None


def main():
    test_order_independent()
    test_clamped()
    test_sanitize_fps()
    print("OK")


if __name__ == "__main__":
    main()
