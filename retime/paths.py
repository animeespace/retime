"""Emplacements des modèles, du cache et des binaires externes."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)


def _app_dir() -> Path:
    if FROZEN:
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


APP_DIR = _app_dir()
# Les modèles et le cache vivent hors du dossier de l'exe : ils survivent aux mises à jour.
DATA_DIR = Path(os.environ.get("RETIME_DATA") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Retime")
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"
SETTINGS_FILE = DATA_DIR / "settings.json"
LOG_FILE = DATA_DIR / "retime.log"

for _d in (MODELS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Pas de fenêtre console qui clignote quand on lance ffmpeg depuis la GUI.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def find_tool(name: str) -> str:
    candidates = [APP_DIR / "bin" / f"{name}.exe", APP_DIR / f"{name}.exe"]
    for c in candidates:
        if c.exists():
            return str(c)
    found = shutil.which(name)
    if found:
        return found
    for c in (Path(r"C:\Program Files\FFmpeg\bin") / f"{name}.exe", Path(r"C:\ffmpeg\bin") / f"{name}.exe"):
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"{name} introuvable (ni dans {APP_DIR / 'bin'} ni dans le PATH)")
