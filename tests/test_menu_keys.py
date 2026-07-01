"""
Verifica headless (niente pytest nel repo): asserzioni pure su menu_keys.
Eseguire dalla radice del repo con:  uv run python tests/test_menu_keys.py
Uscita 0 + "OK" = passato.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glfw
from thread.menu_keys import TOGGLE_KEY_NAMES, _NAME_TO_GLFW, toggle_key_to_glfw


def test_known_keys_map_to_glfw_codes():
    assert toggle_key_to_glfw("F1") == glfw.KEY_F1
    assert toggle_key_to_glfw("F12") == glfw.KEY_F12
    assert toggle_key_to_glfw("M") == glfw.KEY_M


def test_unknown_key_defaults_to_f1():
    assert toggle_key_to_glfw("not-a-key") == glfw.KEY_F1
    assert toggle_key_to_glfw("") == glfw.KEY_F1


def test_names_list_is_nonempty_and_default_present():
    assert "F1" in TOGGLE_KEY_NAMES
    assert all(isinstance(n, str) for n in TOGGLE_KEY_NAMES)
    # Ogni nome esposto deve avere una voce ESPLICITA nella mappa: il vecchio
    # check `>= 0` passava anche per nomi non mappati, perché il fallback F1
    # di toggle_key_to_glfw è a sua volta un codice valido.
    for name in TOGGLE_KEY_NAMES:
        assert name in _NAME_TO_GLFW, name
        assert toggle_key_to_glfw(name) == _NAME_TO_GLFW[name]


def main():
    test_known_keys_map_to_glfw_codes()
    test_unknown_key_defaults_to_f1()
    test_names_list_is_nonempty_and_default_present()
    print("OK")


if __name__ == "__main__":
    main()
