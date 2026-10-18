"""Traitement par lot : appariement automatique vidéo <-> .ass par numéro d'épisode."""
from __future__ import annotations

import os
import re
from pathlib import Path

from PySide6 import QtWidgets

from .workers import BatchWorker

VIDEO_EXT = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m2ts", ".ts", ".m4v", ".wav", ".flac", ".mka", ".mp3",
             ".m4a", ".opus"}


def episode_number(name: str) -> int | None:
    s = Path(name).stem
    m = re.search(r"(?i)\bs\d{1,2}\s*e(\d{1,4})\b", s)
    if m:
        return int(m.group(1))
    s = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", s)
    s = re.sub(r"(?i)\b(\d{3,4}p|[xh]\.?26[45]|1[02]bit|8bit|season\s*\d+|saison\s*\d+|part\s*\d+|v\d)\b", " ", s)
    m = re.search(r"(?i)(?:\bep|\bepisode|\bépisode|#)\s*\.?\s*(\d{1,4})\b", s)
    if m:
        return int(m.group(1))
    nums = re.findall(r"(?<![\d.])(\d{1,4})(?![\d.])", s)
    return int(nums[0]) if nums else None


COLS = ["Vidéo", "Piste", ".ass", "Sortie", "État"]


class BatchDialog(QtWidgets.QDialog):
    def __init__(self, settings, store: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Traitement par lot")
        self.resize(1250, 620)
        self.settings = settings
        self.store = store
        self.videos: list[str] = []
        self.asses: list[str] = []
        self.jobs: list[dict] = []
        self.worker: BatchWorker | None = None

        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("Ajouter des vidéos…")
        b1.clicked.connect(self.add_videos)
        b2 = QtWidgets.QPushButton("Ajouter des .ass…")
        b2.clicked.connect(self.add_asses)
        b3 = QtWidgets.QPushButton("Vider")
        b3.clicked.connect(self.clear)
        top.addWidget(b1)
        top.addWidget(b2)
        top.addWidget(b3)
        top.addStretch(1)
        top.addWidget(QtWidgets.QLabel("Sortie :"))
        self.out_mode = QtWidgets.QComboBox()
        self.out_mode.addItems(["À côté du .ass (« … retime.ass »)", "Dans un dossier choisi"])
        self.out_mode.currentIndexChanged.connect(self._out_mode_changed)
        self.out_dir = QtWidgets.QLineEdit(store.get("batch_out_dir", ""))
        self.out_dir.setPlaceholderText("Dossier de sortie")
        self.out_dir.setEnabled(False)
        bo = QtWidgets.QToolButton()
        bo.setText("…")
        bo.clicked.connect(self._browse_out)
        top.addWidget(self.out_mode)
        top.addWidget(self.out_dir, 1)
        top.addWidget(bo)
        lay.addLayout(top)

        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().hide()
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setColumnWidth(1, 150)
        lay.addWidget(self.table, 1)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        lay.addWidget(self.log_view)

        bottom = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1000)
        self.status = QtWidgets.QLabel("Ajoute des vidéos et des .ass : ils sont appariés par numéro d'épisode.")
        self.start_btn = QtWidgets.QPushButton("▶  Lancer le lot")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QtWidgets.QPushButton("Annuler")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        close = QtWidgets.QPushButton("Fermer")
        close.clicked.connect(self.close)
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.progress)
        bottom.addWidget(self.start_btn)
        bottom.addWidget(self.stop_btn)
        bottom.addWidget(close)
        lay.addLayout(bottom)

    # ------------------------------------------------------------------ fichiers
    def _browse_out(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Dossier de sortie", self.out_dir.text())
        if d:
            self.out_dir.setText(d)
            self.out_mode.setCurrentIndex(1)

    def _out_mode_changed(self, i: int) -> None:
        self.out_dir.setEnabled(i == 1)
        self._rebuild()

    def add_videos(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Vidéos", self.store.get("batch_video_dir", ""),
            "Vidéo / audio (" + " ".join("*" + e for e in sorted(VIDEO_EXT)) + ")")
        if files:
            self.store["batch_video_dir"] = str(Path(files[0]).parent)
            self.videos += [f for f in files if f not in self.videos]
            self._rebuild()

    def add_asses(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Sous-titres", self.store.get("batch_ass_dir", ""),
                                                          "ASS (*.ass *.ssa)")
        if files:
            self.store["batch_ass_dir"] = str(Path(files[0]).parent)
            self.asses += [f for f in files if f not in self.asses]
            self._rebuild()

    def clear(self) -> None:
        self.videos, self.asses = [], []
        self._rebuild()

    def _out_path(self, ass: str) -> str:
        p = Path(ass)
        name = f"{p.stem} retime{p.suffix}"
        if self.out_mode.currentIndex() == 1 and self.out_dir.text():
            return str(Path(self.out_dir.text()) / name)
        return str(p.with_name(name))

    def _rebuild(self) -> None:
        by_ep_v: dict[int, str] = {}
        for v in self.videos:
            n = episode_number(v)
            if n is not None and n not in by_ep_v:
                by_ep_v[n] = v
        self.jobs = []
        for a in sorted(self.asses, key=lambda x: (episode_number(x) or 0, x)):
            n = episode_number(a)
            v = by_ep_v.get(n) if n is not None else None
            self.jobs.append({"ass": a, "media": v, "stream": None, "out": self._out_path(a), "ep": n})
        self.table.setRowCount(len(self.jobs))
        for r, job in enumerate(self.jobs):
            combo_v = QtWidgets.QComboBox()
            combo_v.addItem("— aucune vidéo —", None)
            for v in self.videos:
                combo_v.addItem(Path(v).name, v)
            if job["media"]:
                combo_v.setCurrentIndex(self.videos.index(job["media"]) + 1)
            combo_v.currentIndexChanged.connect(lambda _i, r=r, c=combo_v: self._set_video(r, c.currentData()))
            self.table.setCellWidget(r, 0, combo_v)
            self.table.setCellWidget(r, 1, QtWidgets.QComboBox())
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(Path(job["ass"]).name))
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(job["out"]))
            self.table.setItem(r, 4, QtWidgets.QTableWidgetItem("prêt" if job["media"] else "vidéo manquante"))
            self._fill_tracks(r)
        self.status.setText(f"{sum(1 for j in self.jobs if j['media'])} / {len(self.jobs)} paires prêtes.")

    def _set_video(self, r: int, v: str | None) -> None:
        self.jobs[r]["media"] = v
        self.table.item(r, 4).setText("prêt" if v else "vidéo manquante")
        self._fill_tracks(r)

    def _fill_tracks(self, r: int) -> None:
        from .. import media

        combo = self.table.cellWidget(r, 1)
        combo.clear()
        v = self.jobs[r]["media"]
        if not v:
            return
        try:
            info = media.probe(v)
        except Exception:  # noqa: BLE001
            return
        for s in info.audio_streams:
            combo.addItem(s.label(), s.index)
        d = info.default_audio()
        if d:
            combo.setCurrentIndex(info.audio_streams.index(d))
            self.jobs[r]["stream"] = d.index
        combo.currentIndexChanged.connect(lambda _i, r=r, c=combo: self.jobs[r].__setitem__("stream", c.currentData()))

    # ------------------------------------------------------------------ exécution
    def start(self) -> None:
        jobs = [j for j in self.jobs if j["media"] and j["stream"] is not None]
        if not jobs:
            return
        for j in jobs:
            j["out"] = self._out_path(j["ass"])
            os.makedirs(Path(j["out"]).parent, exist_ok=True)
        self.store["batch_out_dir"] = self.out_dir.text()
        self._run_jobs = jobs
        self.worker = BatchWorker(jobs, self.settings, self)
        self.worker.progress.connect(self._progress)
        self.worker.log.connect(self.log_view.appendPlainText)
        self.worker.item_done.connect(self._item_done)
        self.worker.item_failed.connect(self._item_failed)
        self.worker.finished_all.connect(self._all_done)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start()

    def stop(self) -> None:
        if self.worker:
            self.worker.cancel()

    def _row(self, k: int) -> int:
        return self.jobs.index(self._run_jobs[k])

    def _progress(self, k: int, stage: str, frac: float) -> None:
        n = len(self._run_jobs)
        self.progress.setValue(int((k + frac) / n * 1000))
        self.table.item(self._row(k), 4).setText(f"{stage} {frac * 100:.0f}%")
        self.status.setText(f"Épisode {k + 1}/{n} : {stage}")

    def _item_done(self, k: int, out: str, st: dict) -> None:
        self.table.item(self._row(k), 4).setText(
            f"✔ {st['aligned']}/{st['units']} calées, {st['interpolated']} interp. ({st['seconds']:.0f} s)")
        self.log_view.appendPlainText(f"✔ {out}")

    def _item_failed(self, k: int, msg: str) -> None:
        self.table.item(self._row(k), 4).setText(f"✖ {msg[:80]}")

    def _all_done(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(1000)
        self.status.setText("Lot terminé.")

    def closeEvent(self, ev) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        super().closeEvent(ev)
