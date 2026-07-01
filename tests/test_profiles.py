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


_TMPDIRS: list = []


def _fresh_store():
    tmp = tempfile.TemporaryDirectory(prefix="swprof_")
    _TMPDIRS.append(tmp)
    return profiles.ProfileStore(config_dir=tmp.name), tmp.name


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
    exp = tempfile.TemporaryDirectory(prefix="swexp_")
    _TMPDIRS.append(exp)
    dest = Path(exp.name) / "alfa.json"
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


def test_deserialize_wrong_types_fall_back_to_defaults():
    out = profiles.deserialize({
        "viz_mode": None,          # int atteso
        "n_bands": "x",            # int atteso
        "bg_alpha": "0.5",         # float atteso (stringa non accettata)
        "rounded": "sì",           # bool atteso
        "color_a": [1, 2],         # rgb: servono 3 componenti
        "peakcap_color": "bianco", # rgb atteso
        "background": "non-dict",  # dict atteso
        "attack_tau": float("nan"),  # float non finito
    })
    for key in ("viz_mode", "n_bands", "bg_alpha", "rounded", "color_a",
                "peakcap_color", "background", "attack_tau"):
        assert out[key] == profiles.DEFAULTS[key], (key, out[key])


def test_deserialize_out_of_range_clamped():
    out = profiles.deserialize({
        "bar_width": 5.0,           # range [0.1, 1.0]
        "n_bands": 1000,            # range [1, 100]
        "db_floor": -200,           # range [-90, -25]
        "color_b": [2.0, -1.0, 0.5],
        "beat_sensitivity": 0.0,    # range [1.05, 3.0]
    })
    assert out["bar_width"] == 1.0
    assert out["n_bands"] == 100
    assert out["db_floor"] == -90.0
    assert out["color_b"] == [1.0, 0.0, 0.5]
    assert out["beat_sensitivity"] == 1.05


def test_write_envelope_atomic_no_tmp_leftovers():
    store, tmp = _fresh_store()
    store.save("alfa", dict(profiles.DEFAULTS))
    store.set_last("alfa")
    leftovers = list(Path(tmp).rglob("*.tmp"))
    assert leftovers == [], leftovers
    assert store.load("alfa")["n_bands"] == profiles.DEFAULTS["n_bands"]
    assert store.get_last() == "alfa"


def test_load_truncated_profile_raises():
    store, _ = _fresh_store()
    store.save("alfa", dict(profiles.DEFAULTS))
    path = store._path_for("alfa")
    path.write_text(path.read_text(encoding="utf-8")[:20], encoding="utf-8")
    try:
        store.load("alfa")
    except (ValueError, json.JSONDecodeError):
        return
    raise AssertionError("load doveva fallire su JSON troncato")


def main():
    try:
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
        test_write_envelope_atomic_no_tmp_leftovers()
        test_load_truncated_profile_raises()
        test_deserialize_wrong_types_fall_back_to_defaults()
        test_deserialize_out_of_range_clamped()
        print("OK")
    finally:
        for t in _TMPDIRS:
            t.cleanup()


if __name__ == "__main__":
    main()
