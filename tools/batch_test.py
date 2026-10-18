import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtCore, QtWidgets
from retime.gui.app import apply_dark_theme
from retime.gui.batch import BatchDialog
from retime.pipeline import PipelineSettings
OUTDIR = sys.argv[1]
app = QtWidgets.QApplication(sys.argv); apply_dark_theme(app)
store = {}
dlg = BatchDialog(PipelineSettings(), store)
V = r"D:\2 Beyblade Metal\Anime\Vostfr\1 - Beyblade Metal Fight"
dlg.videos = [os.path.join(V, f) for f in sorted(os.listdir(V)) if f.endswith(".mkv")][:20]
dlg.asses = [os.path.join("Sub noretime", f) for f in ("MFB 11 v2.ass", "MFB 13 v2.ass")]
dlg.out_dir.setText(OUTDIR); dlg.out_mode.setCurrentIndex(1)
dlg._rebuild()
for j in dlg.jobs: print("paire :", Path(j["ass"]).name, "->", Path(j["media"]).name if j["media"] else None, "piste", j["stream"], "->", j["out"])
loop = QtCore.QEventLoop()
dlg.start(); dlg.worker.finished_all.connect(loop.quit); loop.exec()
for r in range(dlg.table.rowCount()): print("état :", dlg.table.item(r, 4).text())
print("fichiers :", os.listdir(OUTDIR))
