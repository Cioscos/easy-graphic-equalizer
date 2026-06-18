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


def _fresh_store():
    tmp = tempfile.mkdtemp(prefix="swprof_")
    return profiles.ProfileStore(config_dir=tmp), tmp


def test_store_save_list_load_delete():
    store, _ = _fresh_store()
    assert store.list_names() == []
    s = dict(profiles.DEFAULTS); s["n_bands"] = 33
    store.save("Il mio preset", s)
    assert store.list_names() == ["Il mio preset"]
    loaded = store.load("Il mio preset")
    assert loaded["n_bands"] == 33
    store.delete("Il mio preset")
    assert store.list_names() == []


def test_store_last_profile():
    store, _ = _fresh_store()
    assert store.get_last() is None
    store.set_last("preset-x")
    assert store.get_last() == "preset-x"


def test_store_export_import_roundtrip_and_collision():
    store, _ = _fresh_store()
    s = dict(profiles.DEFAULTS); s["bars_alpha"] = 0.42
    store.save("alfa", s)
    dest = Path(tempfile.mkdtemp(prefix="swexp_")) / "alfa.json"
    store.export("alfa", dest)
    assert dest.is_file()

    # importare due volte lo stesso file dà nomi distinti (anti-collisione)
    n1 = store.import_file(dest)
    n2 = store.import_file(dest)
    assert n1 != n2, (n1, n2)
    assert store.load(n1)["bars_alpha"] == 0.42


def test_store_import_invalid_raises():
    store, tmp = _fresh_store()
    bad = Path(tmp) / "bad.json"
    bad.write_text("non è json", encoding="utf-8")
    try:
        store.import_file(bad)
    except (ValueError, json.JSONDecodeError):
        return
    raise AssertionError("import_file doveva fallire su JSON non valido")


def main():
    test_serialize_roundtrip_preserves_values()
    test_deserialize_fills_missing_and_ignores_unknown()
    test_deserialize_rejects_non_dict()
    test_sanitize_name()
    test_encode_bg_ref_resource_vs_path()
    test_resolve_bg_ref_ok_and_fallback()
    test_store_save_list_load_delete()
    test_store_last_profile()
    test_store_export_import_roundtrip_and_collision()
    test_store_import_invalid_raises()
    print("OK")


if __name__ == "__main__":
    main()
