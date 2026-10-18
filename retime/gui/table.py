"""Tableau des lignes : ancien / nouveau timing, statut, texte .ass, japonais entendu, traduction."""
from __future__ import annotations

from PySide6 import QtCore, QtGui

from ..assfile import format_time, parse_time

COLS = ["", "#", "Statut", "Conf.", "Sens", "Début", "Fin", "Nouv. début", "Nouv. fin", "Δ début", "Δ fin", "Style",
        "Texte (.ass)", "Japonais entendu", "Traduction auto"]
(C_APPLY, C_NUM, C_STATUS, C_CONF, C_SIM, C_OS, C_OE, C_NS, C_NE, C_DS, C_DE, C_STYLE, C_TEXT, C_JA,
 C_TR) = range(len(COLS))

STATUS_COLORS = {
    "calé": QtGui.QColor(70, 170, 100),
    "calé (début)": QtGui.QColor(150, 170, 80),
    "calé (fin)": QtGui.QColor(150, 170, 80),
    "interpolé": QtGui.QColor(210, 110, 80),
    "décalé": QtGui.QColor(100, 140, 220),
    "inchangé": QtGui.QColor(120, 120, 130),
}

FILTERS = ["Toutes les lignes", "À vérifier", "Modifiées", "Interpolées", "Inchangées"]


RETIMED = ("calé", "calé (début)", "calé (fin)", "interpolé")


def needs_review(lr) -> bool:
    if lr.status == "interpolé":
        return True
    if lr.status in RETIMED:
        return lr.confidence < 0.45 or abs(lr.new_start - lr.old_start) > 1.0 or abs(lr.new_end - lr.old_end) > 1.5
    return False


class LinesModel(QtCore.QAbstractTableModel):
    lineEdited = QtCore.Signal(int)     # ligne du modèle

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lines = []
        self.phrases = []

    def set_result(self, out) -> None:
        self.beginResetModel()
        self.lines = list(out.result.lines) if out else []
        self.phrases = list(out.phrases) if out else []
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.lines)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(COLS)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return COLS[section]
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.ToolTipRole and section == C_APPLY:
            return "Appliquer le nouveau timing à cette ligne"
        return None

    def flags(self, index):
        f = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if index.column() == C_APPLY:
            f |= QtCore.Qt.ItemIsUserCheckable
        if index.column() in (C_NS, C_NE):
            f |= QtCore.Qt.ItemIsEditable
        return f

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        lr = self.lines[index.row()]
        c = index.column()
        if role == QtCore.Qt.CheckStateRole and c == C_APPLY:
            return QtCore.Qt.Checked if lr.apply else QtCore.Qt.Unchecked
        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            if c == C_NUM:
                return lr.event_idx + 1
            if c == C_STATUS:
                return lr.status
            if c == C_CONF:
                return f"{lr.confidence * 100:.0f}%" if lr.status in RETIMED else ""
            if c == C_SIM:
                return f"{lr.similarity * 100:.0f}%" if lr.similarity >= 0 and lr.status in RETIMED else ""
            if c == C_OS:
                return format_time(lr.old_start)
            if c == C_OE:
                return format_time(lr.old_end)
            if c == C_NS:
                return format_time(lr.new_start)
            if c == C_NE:
                return format_time(lr.new_end)
            if c == C_DS:
                return f"{(lr.new_start - lr.old_start) * 1000:+.0f}" if lr.changed else ""
            if c == C_DE:
                return f"{(lr.new_end - lr.old_end) * 1000:+.0f}" if lr.changed else ""
            if c == C_STYLE:
                return lr.style
            if c == C_TEXT:
                return lr.text
            if c == C_JA:
                return lr.heard
            if c == C_TR:
                return lr.translation
        if role == QtCore.Qt.ForegroundRole:
            if c == C_STATUS:
                return STATUS_COLORS.get(lr.status)
            if c == C_CONF and lr.status in RETIMED:
                return QtGui.QColor(90, 200, 120) if lr.confidence >= 0.6 else (
                    QtGui.QColor(235, 190, 70) if lr.confidence >= 0.45 else QtGui.QColor(235, 100, 80))
            if c == C_SIM and lr.similarity >= 0 and lr.status in RETIMED:
                return QtGui.QColor(90, 200, 120) if lr.similarity >= 0.5 else (
                    QtGui.QColor(235, 190, 70) if lr.similarity >= 0.3 else QtGui.QColor(235, 100, 80))
            if c in (C_DS, C_DE) and lr.changed:
                d = (lr.new_start - lr.old_start) if c == C_DS else (lr.new_end - lr.old_end)
                return QtGui.QColor(235, 100, 80) if abs(d) > 1.0 else None
            if not lr.apply:
                return QtGui.QColor(115, 115, 125)
        if role == QtCore.Qt.BackgroundRole and needs_review(lr) and lr.apply:
            return QtGui.QColor(70, 52, 30)
        if role == QtCore.Qt.TextAlignmentRole and c in (C_NUM, C_CONF, C_SIM, C_DS, C_DE, C_OS, C_OE, C_NS, C_NE):
            return int(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        if role == QtCore.Qt.ToolTipRole and c in (C_TEXT, C_JA, C_TR):
            return f"{lr.text}\n\nJA : {lr.heard}\nTrad. : {lr.translation}"
        return None

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if not index.isValid():
            return False
        lr = self.lines[index.row()]
        c = index.column()
        if role == QtCore.Qt.CheckStateRole and c == C_APPLY:
            lr.apply = (QtCore.Qt.CheckState(value) == QtCore.Qt.Checked)
        elif role == QtCore.Qt.EditRole and c in (C_NS, C_NE):
            try:
                t = parse_time(str(value))
            except (ValueError, IndexError):
                return False
            if c == C_NS:
                lr.new_start = min(t, lr.new_end - 0.01)
            else:
                lr.new_end = max(t, lr.new_start + 0.01)
            lr.apply = True
        else:
            return False
        self.refresh_row(index.row())
        self.lineEdited.emit(index.row())
        return True

    def refresh_row(self, row: int) -> None:
        self.dataChanged.emit(self.index(row, 0), self.index(row, len(COLS) - 1))

    def row_of_event(self, event_idx: int) -> int:
        for i, lr in enumerate(self.lines):
            if lr.event_idx == event_idx:
                return i
        return -1


class LinesFilter(QtCore.QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = 0
        self.search = ""

    def set_mode(self, mode: int) -> None:
        self.mode = mode
        self.invalidateFilter()

    def set_search(self, text: str) -> None:
        self.search = text.lower().strip()
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        lr = self.sourceModel().lines[row]
        if self.mode == 1 and not needs_review(lr):
            return False
        if self.mode == 2 and not lr.changed:
            return False
        if self.mode == 3 and lr.status != "interpolé":
            return False
        if self.mode == 4 and lr.changed:
            return False
        if self.search and self.search not in lr.text.lower():
            return False
        return True
