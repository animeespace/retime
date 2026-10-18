# -*- mode: python ; coding: utf-8 -*-
# Build : .venv\Scripts\pyinstaller.exe Retime.spec --noconfirm   (voir build.ps1)
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

datas, binaries, hiddenimports = [], [], []

# Paquets à charger en entier (DLL natives, fichiers de données, imports dynamiques)
for pkg in ("faster_whisper", "ctranslate2", "demucs", "sentence_transformers", "pyqtgraph", "onnxruntime",
            "sounddevice", "_sounddevice_data", "tokenizers", "sentencepiece"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# transformers charge ses modèles à la demande : on n'embarque que ceux utilisés
for sub in ("transformers.models.auto", "transformers.models.m2m_100", "transformers.models.nllb",
            "transformers.models.bert", "transformers.generation"):
    hiddenimports += collect_submodules(sub)

# Métadonnées lues via importlib.metadata (vérifications de versions)
for pkg in ("transformers", "tokenizers", "huggingface_hub", "safetensors", "torch", "tqdm", "regex", "requests",
            "packaging", "filelock", "numpy", "pyyaml", "sentence_transformers", "faster_whisper", "ctranslate2",
            "demucs", "scipy", "sentencepiece", "onnxruntime"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

a = Analysis(
    ["run_retime.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["retime.gui.app", "retime.gui.batch", "scipy.signal", "scipy.ndimage"],
    excludes=["matplotlib", "tkinter", "IPython", "jupyter", "notebook", "pytest", "pyflakes",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
              "PySide6.Qt3DCore", "PySide6.QtQuick3D", "PySide6.QtMultimedia", "PySide6.QtPdf",
              "torch.utils.tensorboard", "tensorboard", "torchvision"],
    module_collection_mode={"torch": "pyz+py", "transformers": "pyz+py", "demucs": "pyz+py",
                            "sentence_transformers": "pyz+py"},
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Retime",
    console=False,
    icon="assets/retime.ico",
    upx=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Retime", upx=False)
