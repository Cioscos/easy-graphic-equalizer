"""
Verifica headless (niente pytest nel repo): asserzioni pure su render_params.
Eseguire dalla radice del repo con:  uv run python tests/test_render_params.py
Uscita 0 + "OK" = passato.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thread.render_params import effective_splits


def test_order_independent():
    assert effective_splits(0.5, 0.8) == (0.5, 0.8)
    # green oltre yellow: il vincolo si applica all'uso, ordinando — non
    # azzoppando uno dei due in base all'ordine di arrivo dei messaggi
    assert effective_splits(0.85, 0.3) == (0.3, 0.85)


def test_clamped():
    assert effective_splits(-1.0, 2.0) == (0.0, 1.0)
    assert effective_splits(1.5, 0.2) == (0.2, 1.0)


def main():
    test_order_independent()
    test_clamped()
    print("OK")


if __name__ == "__main__":
    main()
