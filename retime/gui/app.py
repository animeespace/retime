"""Fenêtre principale de Retime."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__, align
from ..assfile import AssDocument
from ..paths import CACHE_DIR, DATA_DIR, LOG_FILE, SETTINGS_FILE
from ..pipeline import PipelineSettings
from .table import C_APPLY, C_JA, C_NUM, C_STATUS, C_TEXT, FILTERS, LinesFilter, LinesModel, needs_review
from .timeline import Timeline, fmt_time
from .workers import Player, RetimeWorker

VIDEO_EXT = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".m2ts", ".ts", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg"}
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".opus", ".ogg", ".mka", ".ac3", ".eac3", ".dts", ".wma"}
SUB_EXT = {".ass", ".ssa"}
MEDIA_FILTER = "Vidéo / audio (" + " ".join(f"*{e}" for e in sorted(VIDEO_EXT | AUDIO_EXT)) + ");;Tous (*.*)"


def load_store() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_store(d: dict) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def make_icon() -> QtGui.QIcon:
    icon = QtGui.QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pm = QtGui.QPixmap(size, size)
        pm.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(pm)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        g = QtGui.QLinearGradient(0, 0, size, size)
        g.setColorAt(0, QtGui.QColor(70, 120, 235))
        g.setColorAt(1, QtGui.QColor(40, 190, 140))
        p.setBrush(g)
        p.setPen(QtCore.Qt.NoPen)
        p.drawRoundedRect(QtCore.QRectF(0, 0, size, size), size * 0.22, size * 0.22)
        p.setBrush(QtGui.QColor(255, 255, 255, 235))
        heights = [0.25, 0.5, 0.8, 0.45, 0.65, 0.3]
        bw = size / 11
        for k, h in enumerate(heights):
            x = size * 0.16 + k * bw * 1.25
            hh = size * 0.62 * h
            p.drawRoundedRect(QtCore.QRectF(x, size * 0.5 - hh / 2, bw * 0.8, hh), bw * 0.3, bw * 0.3)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 220, 90), max(1.0, size / 22)))
        p.drawLine(QtCore.QPointF(size * 0.84, size * 0.16), QtCore.QPointF(size * 0.84, size * 0.84))
        p.end()
        icon.addPixmap(pm)
    return icon


def apply_dark_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    base, alt, text = QtGui.QColor(30, 32, 38), QtGui.QColor(37, 40, 47), QtGui.QColor(225, 228, 235)
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(26, 28, 33))
    pal.setColor(QtGui.QPalette.WindowText, text)
    pal.setColor(QtGui.QPalette.Base, base)
    pal.setColor(QtGui.QPalette.AlternateBase, alt)
    pal.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(45, 48, 56))
    pal.setColor(QtGui.QPalette.ToolTipText, text)
    pal.setColor(QtGui.QPalette.Text, text)
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor(42, 45, 53))
    pal.setColor(QtGui.QPalette.ButtonText, text)
    pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(60, 110, 210))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
    pal.setColor(QtGui.QPalette.Link, QtGui.QColor(100, 160, 255))
    pal.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(130, 134, 145))
    for role in (QtGui.QPalette.WindowText, QtGui.QPalette.Text, QtGui.QPalette.ButtonText):
        pal.setColor(QtGui.QPalette.Disabled, role, QtGui.QColor(110, 112, 120))
    app.setPalette(pal)
    app.setFont(QtGui.QFont("Segoe UI", 9))
    app.setStyleSheet("""
        QGroupBox { border: 1px solid #343843; border-radius: 6px; margin-top: 10px; padding-top: 6px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #8fb4ff; font-weight: 600; }
        QPushButton { padding: 5px 10px; border-radius: 4px; border: 1px solid #3b404c; background: #2c3039; }
        QPushButton:hover { background: #353a45; }
        QPushButton#primary { background: #2f6fe0; border: 1px solid #3f7ff0; color: white; font-weight: 600; }
        QPushButton#primary:hover { background: #3b7cf0; }
        QPushButton#primary:disabled { background: #2b3446; color: #8790a0; }
        QPushButton#export { background: #23955d; border: 1px solid #2aa86a; color: white; font-weight: 600; }
        QPushButton#export:hover { background: #29a868; }
        QPushButton#export:disabled { background: #273a31; color: #7d8a82; }
        QTableView { gridline-color: #2c3038; selection-background-color: #2f5fb8; }
        QHeaderView::section { background: #262930; padding: 4px; border: 0; border-right: 1px solid #333; }
        QProgressBar { border: 1px solid #3a3f4b; border-radius: 4px; text-align: center; height: 16px; }
        QProgressBar::chunk { background: #2f6fe0; border-radius: 3px; }
        QToolTip { border: 1px solid #444; }
        QTabBar::tab { padding: 5px 14px; }
    """)


class PathEdit(QtWidgets.QWidget):
    changed = QtCore.Signal(str)

    def __init__(self, placeholder: str, browse, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.btn = QtWidgets.QToolButton()
        self.btn.setText("…")
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.btn)
        self.btn.clicked.connect(browse)
        self.edit.editingFinished.connect(lambda: self.changed.emit(self.edit.text()))

    def text(self) -> str:
        return self.edit.text().strip().strip('"')

    def setText(self, t: str) -> None:
        self.edit.setText(t)
        self.edit.setCursorPosition(0)
        self.edit.setToolTip(t)


def ms_spin(lo: int, hi: int, step: int = 10, suffix: str = " ms") -> QtWidgets.QSpinBox:
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setSuffix(suffix)
    return s


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Retime {__version__} — recalage de sous-titres sur l'audio japonais")
        self.setWindowIcon(make_icon())
        self.setAcceptDrops(True)
        self.store = load_store()
        self.ps = PipelineSettings.from_dict(self.store.get("pipeline", {}))
        self.media_info = None
        self.ass_doc: AssDocument | None = None
        self.analysis = None
        self.analysis_key = None
        self.output = None
        self.worker: RetimeWorker | None = None
        self.player = Player(self)
        self._build_ui()
        self._load_settings_to_ui()
        self.player.position.connect(self.timeline.set_playhead)
        geo = self.store.get("geometry")
        if geo:
            self.restoreGeometry(QtCore.QByteArray.fromBase64(geo.encode()))
        else:
            self.resize(1600, 950)

    # ================================================================== UI
    def _build_ui(self) -> None:
        # ---------- panneau gauche
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(8, 8, 8, 8)

        g_files = QtWidgets.QGroupBox("Fichiers")
        fl = QtWidgets.QFormLayout(g_files)
        self.media_edit = PathEdit("Vidéo ou audio (.mkv, .mp4, .wav…)", self.browse_media)
        self.media_edit.changed.connect(self.set_media)
        self.track_combo = QtWidgets.QComboBox()
        self.track_combo.currentIndexChanged.connect(self._track_changed)
        self.ass_edit = PathEdit("Sous-titres .ass à recaler", self.browse_ass)
        self.ass_edit.changed.connect(self.set_ass)
        self.out_edit = PathEdit("Fichier .ass de sortie", self.browse_out)
        self.media_label = QtWidgets.QLabel("")
        self.media_label.setStyleSheet("color:#8a90a0")
        fl.addRow("Source", self.media_edit)
        fl.addRow("Piste audio", self.track_combo)
        fl.addRow("", self.media_label)
        fl.addRow("Sous-titres", self.ass_edit)
        fl.addRow("Sortie", self.out_edit)
        lv.addWidget(g_files)

        g_an = QtWidgets.QGroupBox("Analyse audio (GPU)")
        al = QtWidgets.QFormLayout(g_an)
        from ..asr import WHISPER_MODELS
        from ..semantic import TARGET_LANGS, TRANSLATION_MODELS
        self.whisper_combo = QtWidgets.QComboBox()
        self.whisper_combo.addItems(list(WHISPER_MODELS))
        self.sep_check = QtWidgets.QCheckBox("Isoler la voix (Demucs) — recommandé")
        self.sep_check.setToolTip("Supprime musique et bruitages avant de tracer la courbe d'énergie :\n"
                                  "attaques et fins de voix beaucoup plus nettes.")
        self.asr_voc_check = QtWidgets.QCheckBox("Whisper sur la voix isolée")
        self.translate_check = QtWidgets.QCheckBox("Traduire le japonais pour comparer")
        self.lang_combo = QtWidgets.QComboBox()
        self.lang_combo.addItems(list(TARGET_LANGS))
        self.trmodel_combo = QtWidgets.QComboBox()
        self.trmodel_combo.addItems(list(TRANSLATION_MODELS))
        self.sens_spin = QtWidgets.QDoubleSpinBox()
        self.sens_spin.setRange(-10, 10)
        self.sens_spin.setSingleStep(0.5)
        self.sens_spin.setSuffix(" dB")
        self.sens_spin.setToolTip("Positif = détecte des voix plus faibles (attaques plus tôt, fins plus tard).")
        al.addRow("Whisper", self.whisper_combo)
        al.addRow(self.sep_check)
        al.addRow(self.asr_voc_check)
        al.addRow(self.translate_check)
        al.addRow("Langue du .ass", self.lang_combo)
        al.addRow("Traducteur", self.trmodel_combo)
        al.addRow("Sensibilité voix", self.sens_spin)
        lv.addWidget(g_an)

        g_tm = QtWidgets.QGroupBox("Règles de timing")
        tl = QtWidgets.QFormLayout(g_tm)
        self.leadin_spin = ms_spin(0, 1000)
        self.leadin_spin.setToolTip("Apparition du sous-titre avant le début de la voix.")
        self.leadout_spin = ms_spin(0, 2000)
        self.leadout_spin.setToolTip("Maintien du sous-titre après la fin de la voix.")
        self.link_spin = ms_spin(0, 2000)
        self.link_spin.setToolTip("Si deux lignes sont séparées de moins que ça, la première dure jusqu'à la suivante.")
        self.mindur_spin = ms_spin(0, 3000, 50)
        self.tol_spin = ms_spin(0, 500)
        self.tol_spin.setToolTip("Si le nouveau timing est à moins de cet écart de l'original, l'original est gardé\n"
                                 "(comme un retimeur qui ne touche pas ce qui est déjà bon). 0 = tout recaler.")
        self.frames_check = QtWidgets.QCheckBox("Caler sur les frames de la vidéo")
        self.kf_check = QtWidgets.QCheckBox("Accrocher aux keyframes (changements de plan)")
        self.kf_start_spin = ms_spin(0, 1000)
        self.kf_end_spin = ms_spin(0, 1500)
        self.maxoff_spin = QtWidgets.QSpinBox()
        self.maxoff_spin.setRange(1, 600)
        self.maxoff_spin.setSuffix(" s")
        self.maxoff_spin.setToolTip("Décalage global maximal recherché entre le .ass et l'audio.")
        tl.addRow("Lead-in", self.leadin_spin)
        tl.addRow("Lead-out", self.leadout_spin)
        tl.addRow("Lier si écart <", self.link_spin)
        tl.addRow("Durée minimale", self.mindur_spin)
        tl.addRow("Garder si écart <", self.tol_spin)
        tl.addRow(self.frames_check)
        tl.addRow(self.kf_check)
        tl.addRow("Keyframe début ±", self.kf_start_spin)
        tl.addRow("Keyframe fin ±", self.kf_end_spin)
        tl.addRow("Décalage max", self.maxoff_spin)
        lv.addWidget(g_tm)

        g_st = QtWidgets.QGroupBox("Styles du .ass")
        sl = QtWidgets.QVBoxLayout(g_st)
        self.style_table = QtWidgets.QTableWidget(0, 3)
        self.style_table.setHorizontalHeaderLabels(["Style", "Lignes", "Traitement"])
        self.style_table.verticalHeader().hide()
        self.style_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.style_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.style_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.style_table.setMinimumHeight(110)
        self.style_table.setToolTip("Recaler = dialogue calé sur la voix. Décaler = panneaux/typeset : suivent\n"
                                    "seulement le décalage global. Les lignes avec \\pos/\\move sont toujours décalées.")
        sl.addWidget(self.style_table)
        lv.addWidget(g_st)

        self.run_btn = QtWidgets.QPushButton("▶  Analyser et recaler")
        self.run_btn.setObjectName("primary")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self.run)
        self.cancel_btn = QtWidgets.QPushButton("Annuler")
        self.cancel_btn.clicked.connect(self.cancel)
        self.cancel_btn.hide()
        self.export_btn = QtWidgets.QPushButton("💾  Exporter le .ass recalé")
        self.export_btn.setObjectName("export")
        self.export_btn.setMinimumHeight(34)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export)
        lv.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        # boutons d'action épinglés sous les réglages (toujours visibles)
        actions = QtWidgets.QWidget()
        av = QtWidgets.QVBoxLayout(actions)
        av.setContentsMargins(8, 4, 8, 8)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.run_btn, 1)
        row.addWidget(self.cancel_btn)
        av.addLayout(row)
        av.addWidget(self.export_btn)
        misc = QtWidgets.QHBoxLayout()
        self.jp_btn = QtWidgets.QPushButton("Exporter subs JP")
        self.jp_btn.setToolTip("Enregistre les sous-titres japonais générés (Whisper + courbe) en .ass")
        self.jp_btn.setEnabled(False)
        self.jp_btn.clicked.connect(self.export_jp)
        self.batch_btn = QtWidgets.QPushButton("Traitement par lot…")
        self.batch_btn.clicked.connect(self.open_batch)
        self.tools_btn = QtWidgets.QToolButton()
        self.tools_btn.setText("⋯")
        self.tools_btn.setToolTip("Outils")
        self.tools_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(self.tools_btn)
        menu.addAction("Ouvrir le dossier des modèles et du cache", self.open_data_dir)
        menu.addAction("Vider le cache d'analyse…", self.clear_cache)
        menu.addAction("Libérer la mémoire GPU", self.free_gpu)
        self.tools_btn.setMenu(menu)
        misc.addWidget(self.jp_btn)
        misc.addWidget(self.batch_btn)
        misc.addWidget(self.tools_btn)
        av.addLayout(misc)
        side = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(0)
        sv.addWidget(scroll, 1)
        sv.addWidget(actions)
        side.setMinimumWidth(380)

        # ---------- zone centrale
        center = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(center)
        cv.setContentsMargins(4, 6, 6, 4)
        tb = QtWidgets.QHBoxLayout()
        self.play_new_btn = QtWidgets.QPushButton("▶ Nouveau (Espace)")
        self.play_new_btn.clicked.connect(lambda: self.play_selected(new=True))
        self.play_old_btn = QtWidgets.QPushButton("▶ Original (Maj+Espace)")
        self.play_old_btn.clicked.connect(lambda: self.play_selected(new=False))
        self.stop_btn = QtWidgets.QPushButton("■")
        self.stop_btn.setFixedWidth(34)
        self.stop_btn.clicked.connect(self.player.stop)
        self.listen_combo = QtWidgets.QComboBox()
        self.listen_combo.addItems(["Écouter : mixage", "Écouter : voix isolée"])
        self.all_btn = QtWidgets.QPushButton("Vue d'ensemble")
        self.all_btn.clicked.connect(lambda: self.timeline.zoom_to(0, self.timeline.duration))
        self.next_btn = QtWidgets.QPushButton("Prochaine à vérifier (N)")
        self.next_btn.clicked.connect(self.next_review)
        for w in (self.play_new_btn, self.play_old_btn, self.stop_btn, self.listen_combo):
            tb.addWidget(w)
        tb.addStretch(1)
        tb.addWidget(self.next_btn)
        tb.addWidget(self.all_btn)
        cv.addLayout(tb)

        self.timeline = Timeline()
        self.timeline.lineClicked.connect(self.select_event)
        self.timeline.lineDoubleClicked.connect(lambda e: (self.select_event(e), self.play_selected(True)))
        self.timeline.phraseClicked.connect(self._phrase_clicked)
        self.timeline.regionEdited.connect(self._region_edited)
        self.timeline.seekRequested.connect(self._seek)

        self.model = LinesModel(self)
        self.proxy = LinesFilter(self)
        self.proxy.setSourceModel(self.model)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setWordWrap(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)
        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(True)
        for c, w in ((C_APPLY, 26), (C_NUM, 42), (C_STATUS, 72), (C_TEXT, 380), (C_JA, 260)):
            self.table.setColumnWidth(c, w)
        for c in range(C_STATUS + 1, C_TEXT):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        self.table.selectionModel().currentRowChanged.connect(self._row_changed)
        self.model.lineEdited.connect(self._line_edited)

        tab_lines = QtWidgets.QWidget()
        tlv = QtWidgets.QVBoxLayout(tab_lines)
        tlv.setContentsMargins(0, 4, 0, 0)
        fr = QtWidgets.QHBoxLayout()
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(FILTERS)
        self.filter_combo.currentIndexChanged.connect(self.proxy.set_mode)
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Rechercher dans le texte…")
        self.search_edit.textChanged.connect(self.proxy.set_search)
        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setStyleSheet("color:#9aa3b5")
        self.check_all_btn = QtWidgets.QPushButton("Tout cocher")
        self.check_all_btn.clicked.connect(lambda: self._check_all(True))
        self.uncheck_low_btn = QtWidgets.QPushButton("Décocher les douteuses")
        self.uncheck_low_btn.clicked.connect(self._uncheck_doubtful)
        fr.addWidget(self.filter_combo)
        fr.addWidget(self.search_edit, 1)
        fr.addWidget(self.check_all_btn)
        fr.addWidget(self.uncheck_low_btn)
        fr.addWidget(self.stats_label)
        tlv.addLayout(fr)
        tlv.addWidget(self.table)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QtGui.QFont("Consolas", 9))
        self.jp_view = QtWidgets.QPlainTextEdit()
        self.jp_view.setReadOnly(True)
        self.jp_view.setFont(QtGui.QFont("Yu Gothic UI", 10))
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(tab_lines, "Lignes")
        self.tabs.addTab(self.jp_view, "Transcription japonaise")
        self.tabs.addTab(self.log_view, "Journal")

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(self.timeline)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 4)
        cv.addWidget(split, 1)

        main_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_split.addWidget(side)
        main_split.addWidget(center)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([380, 1200])
        self.setCentralWidget(main_split)

        sb = self.statusBar()
        self.stage_label = QtWidgets.QLabel("Prêt — glisse une vidéo et un .ass dans la fenêtre.")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(380)
        self.progress.setRange(0, 1000)
        self.progress.hide()
        sb.addWidget(self.stage_label, 1)
        sb.addPermanentWidget(self.progress)

        for seq, fn in (("Space", lambda: self.play_selected(True)), ("Shift+Space", lambda: self.play_selected(False)),
                        ("Escape", self.player.stop), ("Ctrl+S", self.export), ("F5", self.run),
                        ("Ctrl+O", self.browse_media), ("N", self.next_review)):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            sc.setContext(QtCore.Qt.WindowShortcut)
            sc.activated.connect(fn)

    # ================================================================== réglages
    def _load_settings_to_ui(self) -> None:
        ps, rt = self.ps, self.ps.retime
        self.whisper_combo.setCurrentText(ps.whisper_model)
        self.sep_check.setChecked(ps.vocal_separation)
        self.asr_voc_check.setChecked(ps.asr_on_vocals)
        self.translate_check.setChecked(ps.translate)
        self.lang_combo.setCurrentText(ps.target_lang)
        self.trmodel_combo.setCurrentText(ps.translation_model)
        self.sens_spin.setValue(ps.sensitivity_db)
        self.leadin_spin.setValue(int(round(rt.lead_in * 1000)))
        self.leadout_spin.setValue(int(round(rt.lead_out * 1000)))
        self.link_spin.setValue(int(round(rt.link_gap * 1000)))
        self.mindur_spin.setValue(int(round(rt.min_duration * 1000)))
        self.tol_spin.setValue(int(round(rt.keep_tolerance * 1000)))
        self.frames_check.setChecked(rt.snap_frames)
        self.kf_check.setChecked(rt.snap_keyframes)
        self.kf_start_spin.setValue(int(round(rt.kf_start_before * 1000)))
        self.kf_end_spin.setValue(int(round(rt.kf_end_after * 1000)))
        self.maxoff_spin.setValue(int(rt.max_offset))
        for key, edit in (("media", self.media_edit), ("ass", self.ass_edit)):
            p = self.store.get("last_" + key)
            if p and os.path.exists(p):
                (self.set_media if key == "media" else self.set_ass)(p)

    def _ui_to_settings(self) -> PipelineSettings:
        ps = self.ps
        ps.whisper_model = self.whisper_combo.currentText()
        ps.vocal_separation = self.sep_check.isChecked()
        ps.asr_on_vocals = self.asr_voc_check.isChecked()
        ps.translate = self.translate_check.isChecked()
        ps.target_lang = self.lang_combo.currentText()
        ps.translation_model = self.trmodel_combo.currentText()
        ps.sensitivity_db = self.sens_spin.value()
        rt = ps.retime
        rt.lead_in = self.leadin_spin.value() / 1000
        rt.lead_out = self.leadout_spin.value() / 1000
        rt.link_gap = self.link_spin.value() / 1000
        rt.min_duration = self.mindur_spin.value() / 1000
        rt.keep_tolerance = self.tol_spin.value() / 1000
        rt.snap_frames = self.frames_check.isChecked()
        rt.snap_keyframes = self.kf_check.isChecked()
        rt.kf_start_before = rt.kf_end_before = self.kf_start_spin.value() / 1000
        rt.kf_start_after = min(0.1, rt.kf_start_before)
        rt.kf_end_after = self.kf_end_spin.value() / 1000
        rt.max_offset = float(self.maxoff_spin.value())
        pol = dict(rt.style_policies)
        for r in range(self.style_table.rowCount()):
            name = self.style_table.item(r, 0).text()
            pol[name] = self.style_table.cellWidget(r, 2).currentData()
        rt.style_policies = pol
        self.store["pipeline"] = ps.to_dict()
        save_store(self.store)
        return ps

    def closeEvent(self, ev) -> None:
        self._ui_to_settings()
        self.store["geometry"] = bytes(self.saveGeometry().toBase64()).decode()
        save_store(self.store)
        self.player.stop()
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        super().closeEvent(ev)

    # ================================================================== fichiers
    def _start_dir(self, key: str) -> str:
        p = self.store.get("last_" + key, "")
        return str(Path(p).parent) if p else str(Path.home())

    def browse_media(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Vidéo ou audio", self._start_dir("media"), MEDIA_FILTER)
        if p:
            self.set_media(p)

    def browse_ass(self) -> None:
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Sous-titres", self._start_dir("ass"),
                                                     "Sous-titres ASS (*.ass *.ssa)")
        if p:
            self.set_ass(p)

    def browse_out(self) -> None:
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Enregistrer sous", self.out_edit.text() or
                                                     self._start_dir("ass"), "Sous-titres ASS (*.ass)")
        if p:
            self.out_edit.setText(p)

    def set_media(self, path: str) -> None:
        path = os.path.normpath(path.strip().strip('"'))
        if not path or not os.path.exists(path):
            return
        from .. import media
        try:
            info = media.probe(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Média illisible", str(e))
            return
        if not info.audio_streams:
            QtWidgets.QMessageBox.warning(self, "Aucune piste audio", "Ce fichier ne contient pas de piste audio.")
            return
        self.media_info = info
        self.media_edit.setText(path)
        self.store["last_media"] = path
        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        for s in info.audio_streams:
            self.track_combo.addItem(s.label() + ("  ← japonais" if s.is_japanese else ""), s.index)
        d = info.default_audio()
        self.track_combo.setCurrentIndex(info.audio_streams.index(d))
        self.track_combo.blockSignals(False)
        fps = f"{float(info.fps):.3f} fps" if info.fps else "audio seul"
        self.media_label.setText(f"{fmt_time(info.duration, False)} · {fps} · {len(info.audio_streams)} piste(s) audio")
        self._invalidate_analysis()
        if not self.ass_edit.text():
            self._guess_ass(path)

    def _guess_ass(self, media_path: str) -> None:
        for ext in (".ass", ".ssa"):
            c = Path(media_path).with_suffix(ext)
            if c.exists():
                self.set_ass(str(c))
                return

    def set_ass(self, path: str) -> None:
        path = os.path.normpath(path.strip().strip('"'))
        if not path or not os.path.exists(path):
            return
        try:
            doc = AssDocument(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, ".ass illisible", str(e))
            return
        self.ass_doc = doc
        self.ass_edit.setText(path)
        self.store["last_ass"] = path
        p = Path(path)
        self.out_edit.setText(str(p.with_name(f"{p.stem} retime{p.suffix}")))
        self._fill_styles(doc)
        n = sum(1 for e in doc.events if not e.is_comment)
        self.stage_label.setText(f"{p.name} : {n} lignes, {len(doc.styles)} styles.")

    def _fill_styles(self, doc: AssDocument) -> None:
        counts: dict[str, int] = {}
        for e in doc.events:
            if not e.is_comment:
                counts[e.style] = counts.get(e.style, 0) + 1
        names = [s for s in doc.styles if s in counts] + [s for s in counts if s not in doc.styles]
        saved = self.ps.retime.style_policies
        self.style_table.setRowCount(len(names))
        for r, name in enumerate(names):
            it = QtWidgets.QTableWidgetItem(name)
            it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.style_table.setItem(r, 0, it)
            c = QtWidgets.QTableWidgetItem(str(counts[name]))
            c.setFlags(c.flags() & ~QtCore.Qt.ItemIsEditable)
            self.style_table.setItem(r, 1, c)
            combo = QtWidgets.QComboBox()
            for key, label in align.POLICY_LABELS.items():
                combo.addItem(label, key)
            pol = saved.get(name) or align.default_policy(name, doc.events)
            combo.setCurrentIndex(list(align.POLICY_LABELS).index(pol))
            self.style_table.setCellWidget(r, 2, combo)

    def _track_changed(self) -> None:
        self._invalidate_analysis()

    def _invalidate_analysis(self) -> None:
        self.analysis = None
        self.analysis_key = None

    def dragEnterEvent(self, ev) -> None:
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev) -> None:
        for url in ev.mimeData().urls():
            p = url.toLocalFile()
            ext = Path(p).suffix.lower()
            if ext in SUB_EXT:
                self.set_ass(p)
            elif ext in VIDEO_EXT | AUDIO_EXT:
                self.set_media(p)

    # ================================================================== traitement
    def _analysis_key(self, ps: PipelineSettings):
        return (self.media_edit.text(), self.track_combo.currentData(), ps.vocal_separation, ps.whisper_model,
                ps.asr_on_vocals, ps.sensitivity_db)

    def run(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        media_path, ass_path = self.media_edit.text(), self.ass_edit.text()
        if not (media_path and os.path.exists(media_path)):
            QtWidgets.QMessageBox.information(self, "Retime", "Choisis d'abord une vidéo ou un fichier audio.")
            return
        if not (ass_path and os.path.exists(ass_path)):
            QtWidgets.QMessageBox.information(self, "Retime", "Choisis d'abord le fichier .ass à recaler.")
            return
        ps = self._ui_to_settings()
        key = self._analysis_key(ps)
        reuse = self.analysis if key == self.analysis_key else None
        self.player.stop()
        self.worker = RetimeWorker(media_path, self.track_combo.currentData(), ass_path, ps, reuse, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self.log)
        self.worker.done.connect(lambda ma, out: self._on_done(ma, out, key))
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self.run_btn.setEnabled(False)
        self.cancel_btn.show()
        self.progress.setValue(0)
        self.progress.show()
        self.log(f"=== {Path(media_path).name}  +  {Path(ass_path).name}")
        self.worker.start()

    def cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.stage_label.setText("Annulation…")

    def _on_progress(self, stage: str, frac: float, total: float) -> None:
        self.progress.setValue(int(total * 1000))
        self.stage_label.setText(f"{stage}… {frac * 100:.0f}%")

    def _on_finished(self) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.hide()
        self.progress.hide()

    def _on_failed(self, msg: str) -> None:
        self.log(msg)
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass
        self.stage_label.setText("Erreur — voir l'onglet Journal." if msg != "Annulé." else "Annulé.")
        if msg != "Annulé.":
            self.tabs.setCurrentWidget(self.log_view)
            QtWidgets.QMessageBox.critical(self, "Erreur", msg.strip().splitlines()[-1][:500])

    def _on_done(self, ma, out, key) -> None:
        new_media = self.analysis is not ma
        self.analysis, self.analysis_key, self.output = ma, key, out
        if new_media:
            self.timeline.set_analysis(ma)
        self.timeline.set_result(out)
        self.model.set_result(out)
        self.jp_view.setPlainText("\n".join(
            f"{fmt_time(p.start)} → {fmt_time(p.end)}   {p.text}" + (f"\n{'':25}{p.translation}" if p.translation else "")
            for p in out.phrases))
        st = out.result.stats
        n_rev = sum(1 for lr in out.result.lines if needs_review(lr) and lr.apply)
        self.stats_label.setText(f"{st['aligned']}/{st['units']} calées · {st['interpolated']} interpolées · "
                                 f"{n_rev} à vérifier · décalage {st['median_offset'] * 1000:+.0f} ms")
        self.stage_label.setText(f"Terminé : {st['changed_lines']} lignes modifiées. Vérifie puis exporte (Ctrl+S).")
        self.export_btn.setEnabled(True)
        self.jp_btn.setEnabled(True)
        self.tabs.setCurrentIndex(0)
        if self.proxy.rowCount():
            self.table.selectRow(0)

    def log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    # ================================================================== sélection / édition
    def _current_line(self):
        idx = self.table.currentIndex()
        if not idx.isValid():
            return None
        return self.model.lines[self.proxy.mapToSource(idx).row()]

    def _row_changed(self, cur, _prev) -> None:
        if not cur.isValid():
            return
        lr = self.model.lines[self.proxy.mapToSource(cur).row()]
        self.timeline.select_line(lr)

    def select_event(self, event_idx: int) -> None:
        row = self.model.row_of_event(event_idx)
        if row < 0:
            return
        pidx = self.proxy.mapFromSource(self.model.index(row, 0))
        if not pidx.isValid():
            self.filter_combo.setCurrentIndex(0)
            pidx = self.proxy.mapFromSource(self.model.index(row, 0))
        self.table.selectRow(pidx.row())
        self.table.scrollTo(pidx, QtWidgets.QAbstractItemView.PositionAtCenter)

    def _phrase_clicked(self, pid: int) -> None:
        if not self.output:
            return
        p = self.output.phrases[pid]
        best, best_ov = None, 0.0
        for lr in self.output.result.lines:
            if lr.voice:
                ov = min(p.end, lr.voice[1]) - max(p.start, lr.voice[0])
                if ov > best_ov:
                    best, best_ov = lr, ov
        if best is not None:
            self.select_event(best.event_idx)
        else:
            self.stage_label.setText(f"Phrase sans ligne correspondante : {p.text}")

    def _region_edited(self, event_idx: int, a: float, b: float) -> None:
        row = self.model.row_of_event(event_idx)
        if row < 0:
            return
        lr = self.model.lines[row]
        fps = self.analysis.media.fps if self.analysis else None
        if fps and self.ps.retime.snap_frames:
            a, b = align.frame_time(a, fps), align.frame_time(b, fps)
        lr.new_start, lr.new_end = round(a, 2), round(max(b, a + 0.01), 2)
        lr.apply = True
        self.model.refresh_row(row)
        self.timeline.update_line(lr)
        self.timeline.select_line(lr, zoom=False)

    def _line_edited(self, row: int) -> None:
        lr = self.model.lines[row]
        self.timeline.update_line(lr)
        cur = self._current_line()
        if cur is lr:
            self.timeline.select_line(lr, zoom=False)

    def _check_all(self, value: bool) -> None:
        for lr in self.model.lines:
            if lr.changed:
                lr.apply = value
        self.model.layoutChanged.emit()
        if self.output:
            self.timeline.set_result(self.output)

    def _uncheck_doubtful(self) -> None:
        for lr in self.model.lines:
            if needs_review(lr):
                lr.apply = False
        self.model.layoutChanged.emit()
        if self.output:
            self.timeline.set_result(self.output)

    def next_review(self) -> None:
        if not self.model.lines:
            return
        cur = self.table.currentIndex()
        start = self.proxy.mapToSource(cur).row() + 1 if cur.isValid() else 0
        n = len(self.model.lines)
        for k in range(n):
            row = (start + k) % n
            if needs_review(self.model.lines[row]):
                self.select_event(self.model.lines[row].event_idx)
                return

    # ================================================================== lecture
    def _listen_signal(self):
        if not self.analysis:
            return None
        if self.listen_combo.currentIndex() == 1 and self.analysis.vocals is not None:
            return self.analysis.vocals
        return self.analysis.audio

    def play_selected(self, new: bool = True) -> None:
        if isinstance(QtWidgets.QApplication.focusWidget(), (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox)):
            return
        lr = self._current_line()
        sig = self._listen_signal()
        if lr is None or sig is None:
            return
        a, b = (lr.new_start, lr.new_end) if new else (lr.old_start, lr.old_end)
        self.player.play(sig, 16000, a, b)

    def _seek(self, t: float) -> None:
        sig = self._listen_signal()
        if sig is not None:
            self.player.play(sig, 16000, t, t + 4.0)

    # ================================================================== export
    def export(self) -> None:
        if not self.output:
            return
        out_path = self.out_edit.text()
        if not out_path:
            self.browse_out()
            out_path = self.out_edit.text()
            if not out_path:
                return
        if os.path.normcase(os.path.abspath(out_path)) == os.path.normcase(os.path.abspath(self.ass_edit.text())):
            r = QtWidgets.QMessageBox.question(self, "Écraser l'original ?",
                                               "La sortie est le fichier d'origine. L'écraser ?")
            if r != QtWidgets.QMessageBox.Yes:
                return
        elif os.path.exists(out_path):
            r = QtWidgets.QMessageBox.question(self, "Fichier existant", f"Remplacer {Path(out_path).name} ?")
            if r != QtWidgets.QMessageBox.Yes:
                return
        from ..pipeline import apply_result
        try:
            doc = apply_result(self.output)
            doc.save(out_path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Export impossible", str(e))
            return
        n = sum(1 for lr in self.output.result.lines if lr.apply and lr.changed)
        self.stage_label.setText(f"Exporté : {out_path}  ({n} lignes recalées)")
        self.log(f"Exporté → {out_path} ({n} lignes recalées)")

    def export_jp(self) -> None:
        if not self.output:
            return
        default = str(Path(self.media_edit.text()).with_suffix("")) + " [JP auto].ass"
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Sous-titres japonais", default, "ASS (*.ass)")
        if p:
            from ..pipeline import export_japanese
            export_japanese(self.output, self.analysis, self.ps.retime, p)
            self.log(f"Sous-titres japonais exportés → {p}")

    def open_data_dir(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(DATA_DIR)))

    def clear_cache(self) -> None:
        import shutil
        dirs = [d for d in CACHE_DIR.iterdir() if d.is_dir()]
        size = sum(f.stat().st_size for d in dirs for f in d.rglob("*") if f.is_file()) / 1e6
        r = QtWidgets.QMessageBox.question(
            self, "Vider le cache", f"Supprimer l'analyse en cache de {len(dirs)} média(s) ({size:.0f} Mo) ?\n"
            "Les modèles ne sont pas touchés ; les médias seront simplement réanalysés.")
        if r == QtWidgets.QMessageBox.Yes:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
            self._invalidate_analysis()
            self.log(f"Cache vidé ({size:.0f} Mo).")

    def free_gpu(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        from ..pipeline import release_models
        release_models()
        self.log("Modèles déchargés de la VRAM.")

    def open_batch(self) -> None:
        from .batch import BatchDialog
        dlg = BatchDialog(self._ui_to_settings(), self.store, self)
        dlg.exec()
        save_store(self.store)


def main() -> None:
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Retime")
    apply_dark_theme(app)
    app.setWindowIcon(make_icon())

    def excepthook(t, v, tb):
        msg = "".join(traceback.format_exception(t, v, tb))
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass
        QtWidgets.QMessageBox.critical(None, "Erreur inattendue", msg[-1500:])

    sys.excepthook = excepthook
    w = MainWindow()
    w.show()
    args = [a for a in sys.argv[1:] if os.path.exists(a)]
    for a in args:
        ext = Path(a).suffix.lower()
        if ext in SUB_EXT:
            w.set_ass(a)
        else:
            w.set_media(a)
    sys.exit(app.exec())
