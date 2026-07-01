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


def _default_config_dir() -> Path:
    """Cartella di config per-utente, cross-platform, senza dipendenze esterne."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "sound-wave"


def _atomic_write_text(dest: Path, text: str) -> None:
    """Scrittura atomica: file temporaneo nella stessa directory + os.replace.

    Un kill/blackout a metà scrittura non può lasciare il file di destinazione
    troncato (os.replace è atomico sullo stesso filesystem)."""
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, dest)


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
        _atomic_write_text(
            Path(dest_path), json.dumps(envelope, indent=2, ensure_ascii=False)
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
        _atomic_write_text(
            self._settings_file, json.dumps({"last_profile": name}, ensure_ascii=False)
        )
