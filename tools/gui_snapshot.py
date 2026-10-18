"""Capture de l'interface après un traitement complet (rendu hors écran)."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtCore, QtWidgets
from retime.gui.app import MainWindow, apply_dark_theme
VID, ASS, OUT = sys.argv[1:4]
app = QtWidgets.QApplication(sys.argv)
apply_dark_theme(app)
w = MainWindow(); w.resize(1700, 1000); w.show()
w.set_media(VID); w.set_ass(ASS)
loop = QtCore.QEventLoop()
w.run()
w.worker.finished.connect(loop.quit)
loop.exec()
for _ in range(20): app.processEvents()
w.grab().save(OUT.replace(".png", "_overview.png"))
w.select_event(48)
for _ in range(20): app.processEvents()
w.grab().save(OUT)
print("lignes:", w.model.rowCount(), "|", w.stats_label.text(), "|", w.stage_label.text())
