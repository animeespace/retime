"""Génère assets/retime.ico à partir de l'icône dessinée par l'application."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtGui, QtWidgets
from retime.gui.app import make_icon
app = QtWidgets.QApplication(sys.argv)
icon = make_icon()
out = Path(__file__).resolve().parent.parent / "assets" / "retime.ico"
img = icon.pixmap(256, 256).toImage()
ok = img.save(str(out), "ICO")
icon.pixmap(256, 256).save(str(out.with_suffix(".png")), "PNG")
print("ico", ok, out)
