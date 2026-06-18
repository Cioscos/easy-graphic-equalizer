"""
Verifica headless (niente pytest nel repo): asserzioni pure su profiles.py.
Eseguire dalla radice del repo con:  uv run python tests/test_profiles.py
Uscita 0 + "OK" = passato.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import profiles
from resource_manager import ResourceManager

RM = ResourceManager()


def test_serialize_roundtrip_preserves_values():
    s = dict(profiles.DEFAULTS)
    s["n_bands"] = 21
    s["color_a"] = (0.1, 0.2, 0.3)  # tuple → list
    out = profiles.deserialize(profiles.serialize(s))
    assert out["n_bands"] == 21, out["n_bands"]
    assert out["color_a"] == [0.1, 0.2, 0.3], out["color_a"]
    assert out["viz_mode"] == profiles.DEFAULTS["viz_mode"]


def test_deserialize_fills_missing_and_ignores_unknown():
    out = profiles.deserialize({"n_bands": 5, "chiave_ignota": 123})
    assert out["n_bands"] == 5
    assert "chiave_ignota" not in out
    assert out["bars_alpha"] == profiles.DEFAULTS["bars_alpha"]  # default riempito
    assert set(out.keys()) == set(profiles.DEFAULTS.keys())


def test_deserialize_rejects_non_dict():
    try:
        profiles.deserialize(["non", "un", "dict"])
    except ValueError:
        return
    raise AssertionError("deserialize doveva sollevare ValueError")


def test_sanitize_name():
    assert profiles.sanitize_name("  Il mio/preset  ") == "Il mio_preset"
    assert profiles.sanitize_name("") == "profilo"
    assert profiles.sanitize_name("...") == "profilo"
    # le lettere accentate sono preservate (UNICODE \w)
    assert profiles.sanitize_name("perché") == "perché"


def test_encode_bg_ref_resource_vs_path():
    inside = str(RM.base_path / "bg" / "bg (1).png")
    ref = profiles.encode_bg_ref(inside, RM)
    assert ref == {"kind": "resource", "name": "bg (1).png"}, ref

    outside = profiles.encode_bg_ref("/tmp/qualcosa/video.mp4", RM)
    assert outside["kind"] == "path"
    assert outside["value"].endswith("video.mp4")


def test_resolve_bg_ref_ok_and_fallback():
    # risorsa esistente
    path, ok = profiles.resolve_bg_ref({"kind": "resource", "name": profiles.DEFAULT_BG_NAME}, RM)
    assert ok is True and path.endswith(profiles.DEFAULT_BG_NAME)

    # risorsa mancante → fallback al default, ok False
    path, ok = profiles.resolve_bg_ref({"kind": "resource", "name": "non-esiste.png"}, RM)
    assert ok is False and path.endswith(profiles.DEFAULT_BG_NAME)

    # path assoluto inesistente → fallback al default, ok False
    path, ok = profiles.resolve_bg_ref({"kind": "path", "value": "/non/esiste.mp4"}, RM)
    assert ok is False and path.endswith(profiles.DEFAULT_BG_NAME)


def main():
    test_serialize_roundtrip_preserves_values()
    test_deserialize_fills_missing_and_ignores_unknown()
    test_deserialize_rejects_non_dict()
    test_sanitize_name()
    test_encode_bg_ref_resource_vs_path()
    test_resolve_bg_ref_ok_and_fallback()
    print("OK")


if __name__ == "__main__":
    main()
