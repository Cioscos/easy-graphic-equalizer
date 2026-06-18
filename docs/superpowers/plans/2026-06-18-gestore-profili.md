# Gestore profili (import/export configurazioni) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Salvare/caricare configurazioni come profili con nome (persistiti + import/export su file), con safe fallback per sfondi mancanti; più tre pulizie off-topic.

**Architecture:** Un modulo puro `profiles.py` (niente Qt/PIL) gestisce cartella di config cross-platform, un file JSON per profilo, (de)serializzazione tollerante e riferimenti-sfondo portabili con fallback. La GUI (`AudioCaptureGUI`) ottiene `_collect_profile()`/`_apply_profile()` espliciti più una barra profili dedicata in alto (`gui/profile_bar_frame.py`). Le callback per-controllo esistenti restano intatte.

**Tech Stack:** Python 3.12+, PySide6 (Qt), Pillow, uv. Test = script Python puri (niente pytest).

## Global Constraints

- **Python `>=3.12`**, dipendenze via **uv** (`pyproject.toml` + `uv.lock`); **non** esiste `requirements.txt`. Eseguire i comandi con `uv run …`.
- **Niente pytest:** i test sono script Python puri eseguiti con `uv run python tests/test_x.py`, che stampano `OK` ed escono 0 (modello: `tests/test_frame_time_stats.py`).
- **Commit come utente `Cioscos`** (già configurato). **MAI** aggiungere righe `Co-Authored-By` o altri trailer di attribuzione.
- **UI e commenti in italiano**, con accenti corretti (è, à, ò, …).
- **Non committare `CLAUDE.md`** (è gitignored in questo repo).
- Lavorare sul branch `feature/gestore-profili` (già creato).

---

### Task 1: Pulizie off-topic (PERF_LOGS, ResourceManager frozen, file morto)

Tre modifiche indipendenti e veloci, raggruppate perché nessuna ha un test automatico e tutte si verificano con un import/compile.

**Files:**
- Modify: `thread/opengl_thread.py:26`
- Modify: `resource_manager.py:1-10`
- Delete: `py-to-exe-settings.json`

**Interfaces:**
- Consumes: niente.
- Produces: `ResourceManager.base_path` ora corretto anche nel frozen build (usato da Task 2 per `resolve_bg_ref`).

- [ ] **Step 1: Rimetti `PERF_LOGS` a `False`**

In `thread/opengl_thread.py` riga 26:

```python
PERF_LOGS = False
```

- [ ] **Step 2: Rendi `ResourceManager` `sys._MEIPASS`-aware**

Sostituisci l'inizio di `resource_manager.py` (righe 1-10) con:

```python
import sys
from pathlib import Path


class ResourceManager:
    """
    Utility class to get acces to the resources in the application
    """
    def __init__(self) -> None:
        # In un bundle PyInstaller le risorse vivono sotto sys._MEIPASS; in
        # sviluppo accanto a questo file.
        if getattr(sys, "frozen", False):
            self.base_path = Path(sys._MEIPASS) / "resources"
        else:
            self.base_path = Path(__file__).parent / "resources"
```

(Il resto del file — `get_image_path` — resta invariato.)

- [ ] **Step 3: Cancella il file di build stale**

```bash
git rm py-to-exe-settings.json
```

- [ ] **Step 4: Verifica che tutto importi/compili**

Run:
```bash
uv run python -c "import resource_manager, sys; rm=resource_manager.ResourceManager(); print(rm.base_path); print('ok')"
uv run python -m py_compile thread/opengl_thread.py
```
Expected: stampa il path `.../resources` e `ok`; nessun errore di compilazione.

- [ ] **Step 5: Commit**

```bash
git add thread/opengl_thread.py resource_manager.py
git commit -m "chore: PERF_LOGS off, ResourceManager frozen-aware, drop stale build config"
```

---

### Task 2: `profiles.py` — serializzazione, default e riferimenti-sfondo

Il cuore puro e testabile: nessun import Qt/PIL.

**Files:**
- Create: `profiles.py`
- Test: `tests/test_profiles.py`

**Interfaces:**
- Consumes: `ResourceManager` (da `resource_manager.py`) per risolvere i path sfondo.
- Produces (usati da Task 3/6/7):
  - `CURRENT_VERSION: int`, `APP_TAG: str`, `DEFAULT_BG_NAME: str`, `DEFAULTS: dict`
  - `serialize(settings: dict) -> dict`
  - `deserialize(raw: dict) -> dict`  (riempie i default, ignora chiavi sconosciute, `ValueError` se non è un dict)
  - `sanitize_name(name: str) -> str`
  - `encode_bg_ref(path: str, rm: ResourceManager) -> dict`
  - `resolve_bg_ref(ref: dict, rm: ResourceManager) -> tuple[str, bool]`  (`(path, ok)`; fallback al default bundle se mancante)

- [ ] **Step 1: Scrivi i test (falliranno: modulo assente)**

Crea `tests/test_profiles.py`:

```python
"""
Verifica headless (niente pytest nel repo): asserzioni pure su profiles.py.
Eseguire dalla radice del repo con:  uv run python tests/test_profiles.py
Uscita 0 + "OK" = passato.
"""
import sys
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
```

- [ ] **Step 2: Esegui i test e verifica che falliscano**

Run: `uv run python tests/test_profiles.py`
Expected: FAIL con `ModuleNotFoundError: No module named 'profiles'`.

- [ ] **Step 3: Implementa `profiles.py` (parte serializzazione + sfondo)**

Crea `profiles.py`:

```python
"""
Persistenza e (de)serializzazione dei profili di configurazione di "Sound Wave".

Modulo puro (niente Qt/PIL): definisce lo schema dei profili, la (de)serializzazione
tollerante fra versioni, i riferimenti-sfondo portabili con safe fallback e lo
store su disco (ProfileStore).
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from resource_manager import ResourceManager

APP_TAG = "sound-wave"
CURRENT_VERSION = 1
DEFAULT_BG_NAME = "bg (19).png"

# Valori di default di un profilo: usati per riempire le chiavi mancanti in
# deserialize() (tolleranza fra versioni). Combaciano con i default GUI/renderer.
DEFAULTS = {
    "noise_threshold": 0.1,
    "n_bands": 9,
    "db_floor": -60,
    "db_ceiling": 0,
    "attack_tau": 0.012,     # s
    "release_tau": 0.220,    # s
    "tilt_db_per_oct": 0.0,
    "viz_mode": 0,
    "bg_alpha": 0.5,
    "bars_alpha": 1.0,
    "background": {"type": "image", "ref": {"kind": "resource", "name": DEFAULT_BG_NAME}},
    "color_mode": 0,
    "color_a": [0.231, 0.420, 1.0],
    "color_b": [1.0, 0.231, 0.816],
    "green_split": 0.5,
    "yellow_split": 0.8,
    "bar_width": 0.8,
    "rounded": False,
    "anchor_center": False,
    "band_symmetric": False,
    "inner_radius": 0.18,
    "line_thickness": 0.012,
    "fill": True,
    "osc_mirror": False,
    "osc_smoothing": 0.5,
    "peakcap_enabled": False,
    "peakcap_color": [1.0, 1.0, 1.0],
    "peakcap_hold": 0.5,     # s
    "peakcap_fall": 0.8,
    "bloom_enabled": False,
    "bloom_intensity": 1.0,
    "beat_enabled": False,
    "beat_intensity": 0.08,
    "beat_sensitivity": 1.5,
}

_INVALID_NAME = re.compile(r"[^\w\-. ]", re.UNICODE)


def serialize(settings: dict) -> dict:
    """Rende il dict JSON-safe (tuple→list) sull'insieme di chiavi noto. No validazione range."""
    out = {}
    for key, default in DEFAULTS.items():
        value = settings.get(key, default)
        if isinstance(value, tuple):
            value = list(value)
        out[key] = value
    return out


def deserialize(raw: dict) -> dict:
    """Riempie le chiavi mancanti con i default e ignora le sconosciute.

    Tollerante fra versioni di profilo. Solleva ValueError se `raw` non è un dict.
    """
    if not isinstance(raw, dict):
        raise ValueError("settings non è un oggetto JSON")
    return {key: raw.get(key, default) for key, default in DEFAULTS.items()}


def sanitize_name(name: str) -> str:
    """Nome profilo → nome-file sicuro (preserva lettere accentate, rimuove separatori)."""
    cleaned = _INVALID_NAME.sub("_", (name or "").strip()).strip(". ")
    return cleaned or "profilo"


def _default_bg_path(rm: ResourceManager) -> str:
    return str(rm.get_image_path(DEFAULT_BG_NAME, "bg"))


def encode_bg_ref(path: str, rm: ResourceManager) -> dict:
    """Codifica un path-sfondo in riferimento portabile.

    Se il file è dentro resources/bg → riferimento-risorsa (portabile fra macchine
    e nel frozen build); altrimenti path assoluto.
    """
    bg_dir = (rm.base_path / "bg").resolve()
    try:
        p = Path(path).resolve()
    except OSError:
        p = Path(path)
    if p.is_relative_to(bg_dir):
        return {"kind": "resource", "name": p.name}
    return {"kind": "path", "value": str(p)}


def resolve_bg_ref(ref: dict, rm: ResourceManager) -> Tuple[str, bool]:
    """Risolve un riferimento-sfondo in un path assoluto.

    Ritorna (path, ok). Se il file non esiste → (path predefinito bundle, False).
    """
    default = _default_bg_path(rm)
    if not isinstance(ref, dict):
        return default, False
    kind = ref.get("kind")
    if kind == "resource":
        name = ref.get("name") or DEFAULT_BG_NAME
        try:
            return str(rm.get_image_path(name, "bg")), True
        except FileNotFoundError:
            return default, False
    if kind == "path":
        value = ref.get("value") or ""
        if value and os.path.isfile(value):
            return value, True
        return default, False
    return default, False
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `uv run python tests/test_profiles.py`
Expected: stampa `OK`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add profiles.py tests/test_profiles.py
git commit -m "feat: profile serialization + portable background refs"
```

---

### Task 3: `ProfileStore` — persistenza su disco

Aggiunge lo store (cartella di config, un file JSON per profilo, last_profile) a `profiles.py`, con test su directory temporanea.

**Files:**
- Modify: `profiles.py` (append `ProfileStore` + `_default_config_dir`)
- Test: `tests/test_profiles.py` (append test dello store)

**Interfaces:**
- Consumes: `serialize`/`deserialize`/`sanitize_name` (Task 2).
- Produces (usati da Task 6/7):
  - `ProfileStore(config_dir: Optional[Path] = None)`
  - `.list_names() -> List[str]`
  - `.load(name) -> dict`  (settings deserializzati)
  - `.save(name, settings) -> None`
  - `.delete(name) -> None`
  - `.export(name, dest_path) -> None`
  - `.write_envelope(settings, dest_path) -> None`
  - `.import_file(src_path) -> str`  (ritorna il nome assegnato)
  - `.get_last() -> Optional[str]` / `.set_last(name) -> None`

- [ ] **Step 1: Aggiungi i test dello store (falliranno)**

In `tests/test_profiles.py`, aggiungi prima di `def main():`:

```python
import tempfile


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
```

E aggiorna `main()` per chiamarli:

```python
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
```

(`import json` è già presente in cima al test? No — aggiungi `import json` tra gli import in testa al file se manca.)

- [ ] **Step 2: Esegui i test e verifica che falliscano**

Run: `uv run python tests/test_profiles.py`
Expected: FAIL con `AttributeError: module 'profiles' has no attribute 'ProfileStore'`.

- [ ] **Step 3: Implementa `ProfileStore`**

Appendi in fondo a `profiles.py`:

```python
def _default_config_dir() -> Path:
    """Cartella di config per-utente, cross-platform, senza dipendenze esterne."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "sound-wave"


class ProfileStore:
    """Gestisce i profili su disco: un file JSON per profilo + un puntatore last_profile.

    L'identità di un profilo è il nome-file sanitizzato (sanitize_name); il nome
    mostrato nella combo coincide con esso.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir) if config_dir else _default_config_dir()
        self.profiles_dir = self.config_dir / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._settings_file = self.config_dir / "settings.json"

    def _path_for(self, name: str) -> Path:
        return self.profiles_dir / f"{sanitize_name(name)}.json"

    def list_names(self) -> List[str]:
        return sorted(p.stem for p in self.profiles_dir.glob("*.json"))

    def load(self, name: str) -> dict:
        raw = json.loads(self._path_for(name).read_text(encoding="utf-8"))
        return deserialize(raw.get("settings", {}))

    def save(self, name: str, settings: dict) -> None:
        self.write_envelope(settings, self._path_for(name))

    def delete(self, name: str) -> None:
        self._path_for(name).unlink(missing_ok=True)

    def export(self, name: str, dest_path) -> None:
        shutil.copyfile(self._path_for(name), dest_path)

    def write_envelope(self, settings: dict, dest_path) -> None:
        envelope = {"app": APP_TAG, "version": CURRENT_VERSION, "settings": serialize(settings)}
        Path(dest_path).write_text(
            json.dumps(envelope, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def import_file(self, src_path) -> str:
        raw = json.loads(Path(src_path).read_text(encoding="utf-8"))
        settings = deserialize(raw.get("settings", {}))
        base = sanitize_name(Path(src_path).stem)
        existing = set(self.list_names())
        name, i = base, 2
        while name in existing:
            name = f"{base}-{i}"
            i += 1
        self.save(name, settings)
        return name

    def get_last(self) -> Optional[str]:
        try:
            data = json.loads(self._settings_file.read_text(encoding="utf-8"))
            return data.get("last_profile")
        except (json.JSONDecodeError, OSError):
            return None

    def set_last(self, name: str) -> None:
        self._settings_file.write_text(
            json.dumps({"last_profile": name}, ensure_ascii=False), encoding="utf-8"
        )
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `uv run python tests/test_profiles.py`
Expected: stampa `OK`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add profiles.py tests/test_profiles.py
git commit -m "feat: ProfileStore (disk persistence + import/export)"
```

---

### Task 4: Setter sui widget (`SliderCustomFrame`, `BackgroundFilepickerFrame`)

Aggiunte minime che servono a `_apply_profile` per impostare i widget senza scatenare le loro callback. Verificate per import/compile (nessun test Qt nel repo).

**Files:**
- Modify: `gui/slider_frame.py` (aggiungi `set_value`)
- Modify: `gui/background_filepicker_frame.py` (aggiungi `set_filename`)

**Interfaces:**
- Produces (usati da Task 6):
  - `SliderCustomFrame.set_value(value: Union[float, int]) -> None`
  - `BackgroundFilepickerFrame.set_filename(path: str) -> None`

- [ ] **Step 1: Aggiungi `set_value` a `SliderCustomFrame`**

In `gui/slider_frame.py`, dopo `get_value` (fine classe), aggiungi:

```python
    def set_value(self, value: Union[float, int]) -> None:
        """Imposta il valore mostrato senza invocare la command (caricamento profili)."""
        self._slider.blockSignals(True)
        if self._is_int:
            self._slider.setValue(int(round(value)))
        else:
            self._slider.setValue(self._to_ticks(value))
        self._slider.blockSignals(False)
        self._refresh_value_label()
        if self._warning_trigger_value:
            self.warning_label.setVisible(self._warning_trigger_value <= self.get_value())
```

- [ ] **Step 2: Aggiungi `set_filename` a `BackgroundFilepickerFrame`**

In `gui/background_filepicker_frame.py`, dopo `get_filename`, aggiungi:

```python
    def set_filename(self, path: str) -> None:
        """Mostra `path` nel campo (senza applicarlo): usato dal caricamento profili."""
        self.image_path_entry.setText(path or "")
```

- [ ] **Step 3: Verifica compile**

Run: `uv run python -m py_compile gui/slider_frame.py gui/background_filepicker_frame.py`
Expected: nessun output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add gui/slider_frame.py gui/background_filepicker_frame.py
git commit -m "feat: programmatic setters on slider/filepicker widgets"
```

---

### Task 5: `gui/profile_bar_frame.py` — la barra profili

Widget composito (combo + 5 pulsanti) con callback iniettate. Verificato per compile.

**Files:**
- Create: `gui/profile_bar_frame.py`

**Interfaces:**
- Consumes: `gui.theme.font_label`, `font_body`, `SPACE_SM`.
- Produces (usati da Task 7):
  - `PLACEHOLDER: str`
  - `ProfileBarFrame(parent=None, *, on_select, on_save, on_save_as, on_delete, on_import, on_export)`
  - `.set_profiles(names: list[str], selected: Optional[str]) -> None`  (blocca i segnali: non scatena `on_select`)
  - `.current_name() -> Optional[str]`  (None se è mostrato il placeholder)

- [ ] **Step 1: Crea il file**

```python
from typing import Callable, List, Optional

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget

from gui.theme import font_body, font_label, SPACE_SM

# Voce mostrata quando non esiste alcun profilo salvato.
PLACEHOLDER = "— Nessun profilo —"


class ProfileBarFrame(QWidget):
    """Barra orizzontale: combo dei profili + Salva / Salva come… / Elimina / Importa… / Esporta…."""

    def __init__(
        self,
        parent=None,
        *,
        on_select: Callable[[Optional[str]], None],
        on_save: Callable[[], None],
        on_save_as: Callable[[], None],
        on_delete: Callable[[], None],
        on_import: Callable[[], None],
        on_export: Callable[[], None],
    ):
        super().__init__(parent)
        self._on_select = on_select

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(SPACE_SM)

        label = QLabel("Profilo:")
        label.setFont(font_label())
        layout.addWidget(label)

        self._combo = QComboBox()
        self._combo.setFont(font_body())
        self._combo.setMinimumWidth(160)
        self._combo.addItem(PLACEHOLDER)
        self._combo.currentTextChanged.connect(self._emit_select)
        layout.addWidget(self._combo, 1)

        for text, callback in (
            ("Salva", on_save),
            ("Salva come…", on_save_as),
            ("Elimina", on_delete),
            ("Importa…", on_import),
            ("Esporta…", on_export),
        ):
            btn = QPushButton(text)
            btn.setFont(font_body())
            btn.clicked.connect(lambda _checked=False, c=callback: c())
            layout.addWidget(btn)

    def _emit_select(self, text: str) -> None:
        self._on_select(None if text == PLACEHOLDER else (text or None))

    def set_profiles(self, names: List[str], selected: Optional[str]) -> None:
        """Ripopola la combo. Blocca i segnali: NON scatena on_select."""
        self._combo.blockSignals(True)
        self._combo.clear()
        if names:
            self._combo.addItems(names)
        else:
            self._combo.addItem(PLACEHOLDER)
        if selected and selected in names:
            self._combo.setCurrentText(selected)
        self._combo.blockSignals(False)

    def current_name(self) -> Optional[str]:
        text = self._combo.currentText()
        return None if text == PLACEHOLDER else (text or None)
```

- [ ] **Step 2: Verifica compile**

Run: `uv run python -m py_compile gui/profile_bar_frame.py`
Expected: nessun output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add gui/profile_bar_frame.py
git commit -m "feat: ProfileBarFrame widget (profile selector + actions)"
```

---

### Task 6: GUI — stato, `_collect_profile`, `_apply_profile`

Aggiunge a `AudioCaptureGUI` lo store, il tracking del path immagine, e i due metodi collect/apply (incluso il fallback sfondo). Nessun pulsante ancora collegato (Task 7).

**Files:**
- Modify: `gui/main_window_gui.py` (import; costanti reverse-map; `self._profile_store`, `self.bg_image_path`; `_apply_image_background`; metodi `_collect_profile`/`_apply_profile`/`_apply_background_from_profile`)

**Interfaces:**
- Consumes: `profiles` (Task 2/3), setter widget (Task 4).
- Produces (usati da Task 7): `self._profile_store`, `self._collect_profile()`, `self._apply_profile(settings)`.

- [ ] **Step 1: Aggiorna gli import e le costanti modulo**

In `gui/main_window_gui.py`, nel blocco import di `PySide6.QtWidgets` (righe 13-25) aggiungi `QFileDialog` e `QInputDialog`:

```python
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
```

Dopo l'import di `ResourceManager` (riga 46) aggiungi:

```python
import profiles
from profiles import ProfileStore
from gui.profile_bar_frame import ProfileBarFrame
```

Sostituisci la costante `DEFAULT_BG_IMAGE = 'bg (19).png'` (riga 54) con l'uso del valore condiviso: rimuovi quella riga e, più sotto, le mappe reverse. Dopo la riga `_VIZ_MODE_TO_INT = {...}` (riga 63) aggiungi:

```python
_INT_TO_COLOR_MODE = {v: k for k, v in _COLOR_MODE_TO_INT.items()}
_INT_TO_VIZ_MODE = {v: k for k, v in _VIZ_MODE_TO_INT.items()}
```

- [ ] **Step 2: Inizializza store e path immagine**

Sostituisci le righe 132-134 (blocco immagine di default):

```python
        # Immagine di sfondo di default (PIL): usata dal renderer OpenGL.
        self._default_bg_path = str(self.resource_manager.get_image_path(DEFAULT_BG_IMAGE, 'bg'))
        self.bg_img = Image.open(self._default_bg_path)
```

con (nota: `profiles.DEFAULT_BG_NAME` al posto della costante locale rimossa, e nuovo `self.bg_image_path`):

```python
        # Store dei profili (gestore con nomi + import/export).
        self._profile_store = ProfileStore()

        # Immagine di sfondo di default (PIL): usata dal renderer OpenGL.
        self._default_bg_path = str(self.resource_manager.get_image_path(profiles.DEFAULT_BG_NAME, 'bg'))
        self.bg_image_path = self._default_bg_path  # path immagine attiva (per i profili)
        self.bg_img = Image.open(self._default_bg_path)
```

- [ ] **Step 3: Traccia il path nell'apply immagine**

In `_apply_image_background` (righe 639-648), aggiungi l'assegnazione di `self.bg_image_path`:

```python
    def _apply_image_background(self, filename: str):
        """Carica un'immagine come sfondo del renderer OpenGL."""
        try:
            self.bg_video_path = None  # torna alla modalità immagine
            self.bg_img = Image.open(filename)
            self.bg_image_path = filename
            self.update_bg_opengl(self.bg_img)
        except FileNotFoundError:
            QMessageBox.critical(self, 'Errore', 'File non trovato!')
        except Exception as e:
            QMessageBox.critical(self, 'Errore', f"Errore nel caricamento dell'immagine: {e}")
```

- [ ] **Step 4: Aggiungi `_collect_profile`**

Aggiungi come metodo di `AudioCaptureGUI` (es. subito prima di `# --- Ciclo di vita ---`, riga 961):

```python
    # --- Profili -------------------------------------------------------------
    def _collect_profile(self) -> dict:
        """Raccoglie lo stato corrente in un dict-profilo (unità native del renderer)."""
        if self.bg_video_path:
            bg = {"type": "video", "ref": profiles.encode_bg_ref(self.bg_video_path, self.resource_manager)}
        else:
            bg = {"type": "image", "ref": profiles.encode_bg_ref(self.bg_image_path, self.resource_manager)}
        return {
            "noise_threshold": float(self.noise_slider.get_value()),
            "n_bands": int(self.frequency_slider.get_value()),
            "db_floor": int(self.adv_db_floor_slider.get_value()),
            "db_ceiling": int(self.adv_db_ceiling_slider.get_value()),
            "attack_tau": self.adv_attack_slider.get_value() / 1000.0,
            "release_tau": self.adv_release_slider.get_value() / 1000.0,
            "tilt_db_per_oct": float(self.adv_tilt_slider.get_value()),
            "viz_mode": int(self.viz_mode),
            "bg_alpha": float(self.bg_alpha),
            "bars_alpha": float(self.bars_alpha),
            "background": bg,
            "color_mode": int(self.bars_color_mode),
            "color_a": list(self.bars_color_a),
            "color_b": list(self.bars_color_b),
            "green_split": float(self.bars_green_split),
            "yellow_split": float(self.bars_yellow_split),
            "bar_width": float(self.bars_width),
            "rounded": bool(self.bars_rounded),
            "anchor_center": bool(self.bars_anchor_center),
            "band_symmetric": bool(self.bars_band_symmetric),
            "inner_radius": float(self.viz_inner_radius),
            "line_thickness": float(self.viz_line_thickness),
            "fill": bool(self.viz_fill),
            "osc_mirror": bool(self.viz_osc_mirror),
            "osc_smoothing": float(self.viz_osc_smoothing),
            "peakcap_enabled": bool(self.fx_peakcap_enabled),
            "peakcap_color": list(self.fx_peakcap_color),
            "peakcap_hold": float(self.fx_peakcap_hold),
            "peakcap_fall": float(self.fx_peakcap_fall),
            "bloom_enabled": bool(self.fx_bloom_enabled),
            "bloom_intensity": float(self.fx_bloom_intensity),
            "beat_enabled": bool(self.fx_beat_enabled),
            "beat_intensity": float(self.fx_beat_intensity),
            "beat_sensitivity": float(self.fx_beat_sensitivity),
        }
```

- [ ] **Step 5: Aggiungi `_apply_profile` e `_apply_background_from_profile`**

Subito dopo `_collect_profile`:

```python
    def _apply_profile(self, s: dict) -> None:
        """Applica un dict-profilo: widget + stato self.* + (se il renderer gira) control message."""
        push = self.equalizer_opengl_thread is not None
        q = self.equalizer_control_queue
        viz_label = _INT_TO_VIZ_MODE.get(int(s["viz_mode"]), "Barre")
        color_label = _INT_TO_COLOR_MODE.get(int(s["color_mode"]), "Classico")

        # --- Audio ---
        self.noise_slider.set_value(s["noise_threshold"])
        self.frequency_slider.set_value(s["n_bands"])
        if push:
            q.put({"type": "set_noise_threshold", "value": float(s["noise_threshold"])})
            q.put({"type": "set_frequency_bands", "value": int(s["n_bands"])})

        # --- DSP avanzato (slider in ms, profilo in s) ---
        self.adv_db_floor_slider.set_value(s["db_floor"])
        self.adv_db_ceiling_slider.set_value(s["db_ceiling"])
        self.adv_attack_slider.set_value(s["attack_tau"] * 1000.0)
        self.adv_release_slider.set_value(s["release_tau"] * 1000.0)
        self.adv_tilt_slider.set_value(s["tilt_db_per_oct"])
        if push:
            q.put({"type": "set_db_floor", "value": float(s["db_floor"])})
            q.put({"type": "set_db_ceiling", "value": float(s["db_ceiling"])})
            q.put({"type": "set_attack_tau", "value": float(s["attack_tau"])})
            q.put({"type": "set_release_tau", "value": float(s["release_tau"])})
            q.put({"type": "set_tilt_db_per_oct", "value": float(s["tilt_db_per_oct"])})

        # --- Visualizzazione (viz_mode lo finalizza on_viz_mode_changed in fondo) ---
        self.bg_alpha = float(s["bg_alpha"]); self.alpha_slider.set_value(self.bg_alpha)
        self.bars_alpha = float(s["bars_alpha"]); self.bars_alpha_slider.set_value(self.bars_alpha)
        self.viz_mode_option.set_value(viz_label)
        if push:
            q.put({"type": "set_alpha", "value": self.bg_alpha})
            q.put({"type": "set_bars_alpha", "value": self.bars_alpha})

        # --- Colore (color_mode lo finalizza on_color_mode_changed in fondo) ---
        self.bars_mode_option.set_value(color_label)
        self.bars_color_a = tuple(s["color_a"]); self.bars_color_b = tuple(s["color_b"])
        self.bars_solid_color.set_color(self.bars_color_a)
        self.bars_grad_base.set_color(self.bars_color_a)
        self.bars_grad_top.set_color(self.bars_color_b)
        self.bars_green_split = float(s["green_split"]); self.bars_yellow_thr.set_value(self.bars_green_split)
        self.bars_yellow_split = float(s["yellow_split"]); self.bars_red_thr.set_value(self.bars_yellow_split)
        if push:
            q.put({"type": "set_color_a", "value": self.bars_color_a})
            q.put({"type": "set_color_b", "value": self.bars_color_b})
            q.put({"type": "set_green_split", "value": self.bars_green_split})
            q.put({"type": "set_yellow_split", "value": self.bars_yellow_split})

        # --- Forma ---
        self.bars_width = float(s["bar_width"]); self.bars_width_slider.set_value(self.bars_width)
        self.bars_rounded = bool(s["rounded"]); self.bars_rounded_option.set_value("Sì" if self.bars_rounded else "No")
        self.bars_anchor_center = bool(s["anchor_center"]); self.bars_anchor_option.set_value("Centro" if self.bars_anchor_center else "Dal basso")
        self.bars_band_symmetric = bool(s["band_symmetric"]); self.bars_order_option.set_value("Simmetrico" if self.bars_band_symmetric else "Standard")
        self.viz_inner_radius = float(s["inner_radius"]); self.viz_inner_radius_slider.set_value(self.viz_inner_radius)
        self.viz_line_thickness = float(s["line_thickness"]); self.viz_thickness_slider.set_value(self.viz_line_thickness)
        self.viz_fill = bool(s["fill"]); self.viz_fill_option.set_value("Area" if self.viz_fill else "Linea")
        self.viz_osc_mirror = bool(s["osc_mirror"]); self.viz_osc_mirror_option.set_value("Sì" if self.viz_osc_mirror else "No")
        self.viz_osc_smoothing = float(s["osc_smoothing"]); self.viz_osc_smooth_slider.set_value(self.viz_osc_smoothing)
        if push:
            q.put({"type": "set_bar_width", "value": self.bars_width})
            q.put({"type": "set_rounded", "value": self.bars_rounded})
            q.put({"type": "set_bar_anchor", "value": self.bars_anchor_center})
            q.put({"type": "set_band_order", "value": self.bars_band_symmetric})
            q.put({"type": "set_inner_radius", "value": self.viz_inner_radius})
            q.put({"type": "set_line_thickness", "value": self.viz_line_thickness})
            q.put({"type": "set_fill", "value": self.viz_fill})
            q.put({"type": "set_osc_mirror", "value": self.viz_osc_mirror})
            q.put({"type": "set_osc_smoothing", "value": self.viz_osc_smoothing})

        # --- Effetti (peakcap_hold: slider ms, profilo s) ---
        self.fx_peakcap_enabled = bool(s["peakcap_enabled"]); self.fx_peakcap_option.set_value("On" if self.fx_peakcap_enabled else "Off")
        self.fx_peakcap_color = tuple(s["peakcap_color"]); self.fx_peakcap_color_picker.set_color(self.fx_peakcap_color)
        self.fx_peakcap_hold = float(s["peakcap_hold"]); self.fx_peakcap_hold_slider.set_value(self.fx_peakcap_hold * 1000.0)
        self.fx_peakcap_fall = float(s["peakcap_fall"]); self.fx_peakcap_fall_slider.set_value(self.fx_peakcap_fall)
        self.fx_bloom_enabled = bool(s["bloom_enabled"]); self.fx_bloom_option.set_value("On" if self.fx_bloom_enabled else "Off")
        self.fx_bloom_intensity = float(s["bloom_intensity"]); self.fx_bloom_slider.set_value(self.fx_bloom_intensity)
        self.fx_beat_enabled = bool(s["beat_enabled"]); self.fx_beat_option.set_value("On" if self.fx_beat_enabled else "Off")
        self.fx_beat_intensity = float(s["beat_intensity"]); self.fx_beat_intensity_slider.set_value(self.fx_beat_intensity)
        self.fx_beat_sensitivity = float(s["beat_sensitivity"]); self.fx_beat_sens_slider.set_value(self.fx_beat_sensitivity)
        if push:
            q.put({"type": "set_peakcap_enabled", "value": self.fx_peakcap_enabled})
            q.put({"type": "set_peakcap_color", "value": self.fx_peakcap_color})
            q.put({"type": "set_peakcap_hold", "value": self.fx_peakcap_hold})
            q.put({"type": "set_peakcap_fall", "value": self.fx_peakcap_fall})
            q.put({"type": "set_bloom_enabled", "value": self.fx_bloom_enabled})
            q.put({"type": "set_bloom_intensity", "value": self.fx_bloom_intensity})
            q.put({"type": "set_beat_enabled", "value": self.fx_beat_enabled})
            q.put({"type": "set_beat_intensity", "value": self.fx_beat_intensity})
            q.put({"type": "set_beat_sensitivity", "value": self.fx_beat_sensitivity})

        # --- Sfondo (safe fallback) ---
        self._apply_background_from_profile(s["background"])

        # --- Visibilità condizionale + finalizzazione viz/color mode ---
        # (le callback non scattano sul set programmatico dei combo; le invochiamo
        #  ora: aggiornano visibilità, self.viz_mode/self.bars_color_mode e, se il
        #  renderer gira, inviano set_viz_mode/set_color_mode)
        self.on_color_mode_changed(color_label)
        self.on_viz_mode_changed(viz_label)

    def _apply_background_from_profile(self, bg: dict) -> None:
        """Risolve e applica lo sfondo del profilo, con safe fallback all'immagine bundle."""
        ref = bg.get("ref", {}) if isinstance(bg, dict) else {}
        bg_type = bg.get("type", "image") if isinstance(bg, dict) else "image"
        path, ok = profiles.resolve_bg_ref(ref, self.resource_manager)

        if bg_type == "video" and ok:
            self.bg_video_path = path
            self.file_picker.set_filename(path)
            if self.equalizer_opengl_thread:
                self.equalizer_control_queue.put({"type": "set_video", "value": path})
        else:
            # immagine, oppure video mancante → fallback immagine
            self.bg_video_path = None
            self.bg_image_path = path
            try:
                self.bg_img = Image.open(path)
            except Exception:
                ok = False
            self.file_picker.set_filename(path)
            if self.equalizer_opengl_thread:
                self.equalizer_control_queue.put({"type": "set_image", "value": self.bg_img})

        if not ok:
            ref_desc = ref.get("value") or ref.get("name") or str(ref)
            QMessageBox.warning(
                self, "Sfondo non trovato",
                f"Sfondo non trovato: {ref_desc}. Uso lo sfondo predefinito.",
            )
```

- [ ] **Step 6: Verifica compile**

Run: `uv run python -m py_compile gui/main_window_gui.py`
Expected: nessun output, exit 0.

- [ ] **Step 7: Commit**

```bash
git add gui/main_window_gui.py
git commit -m "feat: collect/apply profile state in main window"
```

---

### Task 7: GUI — barra profili, handler dei pulsanti, auto-load all'avvio

Collega tutto: inserisce la barra nel layout, implementa gli handler e l'auto-load. Deliverable verificato manualmente eseguendo l'app.

**Files:**
- Modify: `gui/main_window_gui.py` (layout `__init__`; `_build_profile_bar`; `_refresh_profile_bar`; handler `_on_profile_*`; auto-load)

**Interfaces:**
- Consumes: `ProfileBarFrame` (Task 5), `self._profile_store`, `_collect_profile`/`_apply_profile` (Task 6).
- Produces: feature completa.

- [ ] **Step 1: Inserisci la barra nel layout di `__init__`**

In `__init__`, dopo `root_layout.addWidget(self._build_device_selector())` (riga 146) e prima della costruzione di `nav_widget`, aggiungi:

```python
        # Barra profili (gestore con nomi): sotto il device selector
        root_layout.addWidget(self._build_profile_bar())
```

- [ ] **Step 2: Aggiungi auto-load in coda a `__init__`**

In fondo a `__init__`, dopo `self._opengl_closed.connect(self._on_opengl_closed)` (riga 166), aggiungi:

```python
        # Auto-load dell'ultimo profilo usato (se ancora presente)
        last = self._profile_store.get_last()
        if last and last in self._profile_store.list_names():
            self.profile_bar.set_profiles(self._profile_store.list_names(), last)
            self._apply_profile(self._profile_store.load(last))
```

- [ ] **Step 3: Aggiungi `_build_profile_bar` e `_refresh_profile_bar`**

Nella sezione "Costruzione pannelli" (es. dopo `_build_footer`, riga 244):

```python
    def _build_profile_bar(self) -> QWidget:
        """Barra profili: selettore + Salva/Salva come…/Elimina/Importa…/Esporta…."""
        self.profile_bar = ProfileBarFrame(
            on_select=self._on_profile_selected,
            on_save=self._on_profile_save,
            on_save_as=self._on_profile_save_as,
            on_delete=self._on_profile_delete,
            on_import=self._on_profile_import,
            on_export=self._on_profile_export,
        )
        self.profile_bar.set_profiles(self._profile_store.list_names(), None)
        return self.profile_bar

    def _refresh_profile_bar(self, selected=None) -> None:
        self.profile_bar.set_profiles(self._profile_store.list_names(), selected)
```

- [ ] **Step 4: Aggiungi gli handler dei profili**

Subito dopo `_apply_background_from_profile` (Task 6):

```python
    def _on_profile_selected(self, name) -> None:
        if not name:
            return
        try:
            settings = self._profile_store.load(name)
        except Exception as e:
            QMessageBox.warning(self, "Profilo", f"Impossibile caricare il profilo: {e}")
            return
        self._apply_profile(settings)
        self._profile_store.set_last(name)

    def _on_profile_save(self) -> None:
        name = self.profile_bar.current_name()
        if not name:
            self._on_profile_save_as()
            return
        self._profile_store.save(name, self._collect_profile())
        self._profile_store.set_last(name)

    def _on_profile_save_as(self) -> None:
        name, accepted = QInputDialog.getText(self, "Salva profilo", "Nome del profilo:")
        if not accepted or not name.strip():
            return
        self._profile_store.save(name, self._collect_profile())
        saved = profiles.sanitize_name(name)
        self._refresh_profile_bar(selected=saved)
        self._profile_store.set_last(saved)

    def _on_profile_delete(self) -> None:
        name = self.profile_bar.current_name()
        if not name:
            return
        confirm = QMessageBox.question(
            self, "Elimina profilo", f"Eliminare il profilo «{name}»?"
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._profile_store.delete(name)
        self._refresh_profile_bar()

    def _on_profile_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Importa profilo", "", "Profili Sound Wave (*.json);;Tutti i file (*)"
        )
        if not path:
            return
        try:
            name = self._profile_store.import_file(path)
        except Exception as e:
            QMessageBox.warning(self, "Importa profilo", f"File non valido: {e}")
            return
        self._refresh_profile_bar(selected=name)
        self._apply_profile(self._profile_store.load(name))
        self._profile_store.set_last(name)

    def _on_profile_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta profilo", "profilo.json", "Profili Sound Wave (*.json)"
        )
        if not path:
            return
        name = self.profile_bar.current_name()
        try:
            if name:
                self._profile_store.export(name, path)
            else:
                self._profile_store.write_envelope(self._collect_profile(), path)
        except Exception as e:
            QMessageBox.warning(self, "Esporta profilo", f"Impossibile esportare: {e}")
```

- [ ] **Step 5: Verifica compile**

Run: `uv run python -m py_compile gui/main_window_gui.py`
Expected: nessun output, exit 0.

- [ ] **Step 6: Verifica manuale (eseguendo l'app)**

Run: `uv run python main.py`

Checklist (su una macchina con audio + display; in WSL2 servono PulseAudio/PipeWire + display GL):
1. La barra "Profilo:" compare sotto il selettore dispositivo con `— Nessun profilo —`.
2. Modifica alcune impostazioni (colore, modalità, sfondo) → **Salva come…** → dai un nome → compare nella combo.
3. Cambia altre impostazioni → **Salva** → sovrascrive; cambia di nuovo e ri-seleziona il profilo dalla combo → i widget tornano ai valori salvati.
4. Avvia il fullscreen, poi seleziona un profilo diverso → la visualizzazione si aggiorna **a caldo**.
5. **Esporta…** un profilo su file; **Importa…** lo stesso file → compare con nome anti-collisione e viene applicato.
6. Crea un profilo con uno sfondo immagine esterno, poi rinomina/sposta quel file e ricarica il profilo → avviso "Sfondo non trovato…" + fallback all'immagine predefinita (nessun crash).
7. **Elimina** un profilo → sparisce dalla combo.
8. Chiudi e riapri l'app → l'ultimo profilo selezionato viene ricaricato automaticamente.

Expected: tutti i punti superati.

- [ ] **Step 7: Esegui di nuovo i test puri (non-regressione)**

Run: `uv run python tests/test_profiles.py && uv run python tests/test_frame_time_stats.py`
Expected: due `OK`.

- [ ] **Step 8: Commit**

```bash
git add gui/main_window_gui.py
git commit -m "feat: wire profile bar, handlers and startup auto-load"
```

---

## Self-Review

**1. Spec coverage** (ogni sezione dello spec → task):
- Gestore profili con nomi + persistenza → Task 3 (store) + Task 7 (UI/handler/auto-load). ✓
- Portata profilo (visivo+audio/DSP+sfondo, no device/monitor/tema) → `DEFAULTS`/`_collect_profile` (Task 2/6); nessuna chiave per device/monitor/tema. ✓
- UI barra dedicata in alto → Task 5 + Task 7 Step 1. ✓
- `profiles.py` puro (store, serialize tollerante, versioning, bg refs) → Task 2/3. ✓
- `profile_bar_frame.py` → Task 5. ✓
- `_collect_profile`/`_apply_profile` + `set_value` slider → Task 4/6. ✓
- Safe fallback sfondo (immagine/video mancante → default + avviso) → `resolve_bg_ref` (Task 2) + `_apply_background_from_profile` (Task 6); video mancante coperto. ✓
- Conversioni ms↔s (attack/release/peakcap_hold) → collect `/1000`, apply `*1000` (Task 6). ✓
- Off-topic 1 PERF_LOGS / 2 ResourceManager frozen / 3 delete py-to-exe-settings.json → Task 1. ✓
- Test puri per profiles.py → Task 2/3; verifica manuale → Task 7. ✓
- File nuovi/modificati/cancellati combaciano con lo spec. ✓

**2. Placeholder scan:** nessun TBD/TODO; tutti gli step di codice contengono il codice completo. ✓

**3. Type consistency:** `ProfileStore`, `_collect_profile`/`_apply_profile`, `set_value`/`set_color`/`set_filename`, `set_profiles`/`current_name`, `PLACEHOLDER`, `_INT_TO_*` usati con gli stessi nomi/firme fra i task. I tipi dei control message combaciano con `process_control_message` (`opengl_thread.py`). Default in `DEFAULTS` allineati alle costanti reali (`DB_FLOOR=-60`, `DB_CEILING=0`, `ATTACK_TAU=0.012`, `RELEASE_TAU=0.220`). ✓

## Execution Handoff

(vedi messaggio al termine: scelta tra esecuzione subagent-driven o inline)
