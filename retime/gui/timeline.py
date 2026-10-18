"""Timeline audio : spectrogramme, forme d'onde, courbe d'énergie et pistes de sous-titres."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import analysis as an

pg.setConfigOptions(antialias=False, useOpenGL=False, background="#15171c", foreground="#c8ccd4")

# Coordonnées verticales du graphe principal
WAVE_C, WAVE_A = 3.05, 0.9
EN_LO, EN_HI = 0.15, 1.95
LANES = {"orig": (-0.70, -0.12), "asr": (-1.45, -0.87), "new": (-2.20, -1.62)}
LANE_LABELS = [(WAVE_C, "Onde"), (1.05, "Énergie"), (-0.41, "Original"), (-1.16, "Whisper JP"), (-1.91, "Nouveau")]
Y_MIN, Y_MAX = -2.35, 4.05

COL_ORIG = QtGui.QColor(232, 146, 60)
COL_ASR = QtGui.QColor(86, 156, 245)
COL_NEW = QtGui.QColor(80, 200, 120)
COL_WARN = QtGui.QColor(240, 196, 70)
COL_INTERP = QtGui.QColor(230, 110, 90)


def fmt_time(v: float, precise: bool = True) -> str:
    if v < 0:
        return ""
    m, s = divmod(v, 60.0)
    h, m = divmod(int(m), 60)
    if h:
        return f"{h}:{m:02d}:{s:05.2f}" if precise else f"{h}:{m:02d}:{int(s):02d}"
    return f"{m:02d}:{s:05.2f}" if precise else f"{m:02d}:{int(s):02d}"


class TimeAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [fmt_time(v, precise=spacing < 1.0) for v in values]


class LaneAxis(pg.AxisItem):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setTicks([[(y, t) for y, t in LANE_LABELS], []])
        self.setWidth(78)


class Timeline(QtWidgets.QWidget):
    lineClicked = QtCore.Signal(int)            # index d'événement .ass
    lineDoubleClicked = QtCore.Signal(int)
    phraseClicked = QtCore.Signal(int)
    regionEdited = QtCore.Signal(int, float, float)
    seekRequested = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.glw)
        self.spec = self.glw.addPlot(row=0, col=0, axisItems={"bottom": TimeAxis("bottom")})
        self.main = self.glw.addPlot(row=1, col=0, axisItems={"bottom": TimeAxis("bottom"),
                                                              "left": LaneAxis("left")})
        self.glw.ci.layout.setRowStretchFactor(0, 2)
        self.glw.ci.layout.setRowStretchFactor(1, 5)
        self.spec.setXLink(self.main)
        for p in (self.spec, self.main):
            p.setMouseEnabled(x=True, y=False)
            p.setMenuEnabled(False)
            p.hideButtons()
            p.showGrid(x=True, y=False, alpha=0.12)
        self.spec.getAxis("bottom").setStyle(showValues=False)
        self.spec.getAxis("left").setWidth(78)
        self.spec.getAxis("left").setTicks([[(0, "Spectro")], []])
        self.main.setYRange(Y_MIN, Y_MAX, padding=0)

        self.spec_img = pg.ImageItem(axisOrder="row-major")
        self.spec_img.setLookupTable(pg.colormap.get("inferno").getLookupTable(nPts=256))
        self.spec.addItem(self.spec_img)

        pen_wave = pg.mkPen((120, 170, 230), width=1)
        self.wave_hi = pg.PlotDataItem(pen=pen_wave, fillLevel=WAVE_C, brush=(80, 130, 200, 120))
        self.wave_lo = pg.PlotDataItem(pen=pen_wave, fillLevel=WAVE_C, brush=(80, 130, 200, 120))
        self.active_fill = pg.PlotDataItem(pen=None, fillLevel=EN_LO, brush=(80, 200, 120, 45))
        self.energy = pg.PlotDataItem(pen=pg.mkPen((235, 235, 240), width=1))
        self.thr = pg.PlotDataItem(pen=pg.mkPen((240, 120, 90, 170), width=1, style=QtCore.Qt.DashLine))
        self.kf = pg.PlotDataItem(pen=pg.mkPen((190, 120, 255, 70), width=1), connect="pairs")
        for it in (self.kf, self.active_fill, self.wave_hi, self.wave_lo, self.energy, self.thr):
            it.setDownsampling(auto=True, method="peak")
            it.setClipToView(True)
            self.main.addItem(it)

        self.lane_items: dict[str, pg.BarGraphItem] = {}
        self.lane_data: dict[str, np.ndarray] = {}     # (n, 3) : début, fin, identifiant
        self.lane_text: dict[str, list[str]] = {}
        self._colors: dict[str, list[QtGui.QColor]] = {}
        self.text_pool: list[pg.TextItem] = []

        self.reg_orig = pg.LinearRegionItem(movable=False, brush=(232, 146, 60, 35),
                                            pen=pg.mkPen((232, 146, 60, 160)))
        self.reg_new = pg.LinearRegionItem(movable=True, brush=(80, 200, 120, 45), pen=pg.mkPen((80, 220, 130), width=2))
        self.reg_new.setZValue(20)
        for r in (self.reg_orig, self.reg_new):
            r.hide()
            self.main.addItem(r)
        self.reg_new.sigRegionChangeFinished.connect(self._region_done)
        self._sel_event: int | None = None
        self._suspend_region = False

        self.playhead = pg.InfiniteLine(angle=90, pen=pg.mkPen((255, 255, 255), width=1.5))
        self.playhead.setZValue(30)
        self.playhead.hide()
        self.main.addItem(self.playhead)

        self.main.scene().sigMouseClicked.connect(self._clicked)
        self.main.sigXRangeChanged.connect(self._update_labels)
        self.duration = 0.0

    # ------------------------------------------------------------------ données
    def clear(self) -> None:
        for it in (self.wave_hi, self.wave_lo, self.energy, self.thr, self.active_fill, self.kf):
            it.setData([], [])
        self.spec_img.clear()
        for it in self.lane_items.values():
            self.main.removeItem(it)
        self.lane_items.clear()
        self.lane_data.clear()
        self.lane_text.clear()
        self._hide_texts()
        self.reg_new.hide()
        self.reg_orig.hide()

    def set_analysis(self, ma) -> None:
        self.clear()
        sig = ma.audio
        self.duration = len(sig) / an.SR
        lo, hi = an.waveform_envelope(sig)
        peak = max(1e-4, float(np.percentile(np.maximum(-lo, hi), 99.9)))
        t = (np.arange(len(lo)) + 0.5) * (80 / an.SR)
        self.wave_hi.setData(t, WAVE_C + np.clip(hi / peak, -1, 1) * WAVE_A)
        self.wave_lo.setData(t, WAVE_C + np.clip(lo / peak, -1, 1) * WAVE_A)
        f = ma.features
        te = (np.arange(f.n) + 0.5) * an.HOP
        e_lo = float(np.percentile(f.energy_db, 2))
        e_hi = float(np.percentile(f.energy_db, 99.8))

        def ymap(db):
            return EN_LO + np.clip((db - e_lo) / max(1e-3, e_hi - e_lo), 0, 1) * (EN_HI - EN_LO)

        self.energy.setData(te, ymap(f.energy_db))
        self.thr.setData(te, ymap(f.thr_db))
        self.active_fill.setData(te, np.where(f.active, EN_HI, EN_LO))
        if ma.spectrogram is not None:
            self.spec_img.setImage(ma.spectrogram.T, autoLevels=False, levels=(0, 255))
            self.spec_img.setRect(QtCore.QRectF(0, 0, ma.spectrogram.shape[0] * an.HOP, ma.spectrogram.shape[1]))
            self.spec.setYRange(0, ma.spectrogram.shape[1], padding=0)
        if len(ma.keyframes):
            k = np.repeat(ma.keyframes, 2)
            y = np.tile([Y_MIN, Y_MAX], len(ma.keyframes))
            self.kf.setData(k, y)
        self.main.setLimits(xMin=-2, xMax=self.duration + 2, minXRange=0.3)
        self.main.setXRange(0, min(self.duration, 60.0), padding=0)

    def set_result(self, out) -> None:
        res = out.result
        ok_status = ("calé", "calé (début)", "calé (fin)")
        orig, new, ocol, ncol, otext = [], [], [], [], []
        for lr in res.lines:
            if lr.policy == "keep" and not lr.changed:
                continue
            orig.append((lr.old_start, lr.old_end, lr.event_idx))
            otext.append(lr.text)
            ocol.append(COL_ORIG)
            new.append((lr.new_start, lr.new_end, lr.event_idx))
            if not lr.apply:
                ncol.append(QtGui.QColor(110, 110, 120))
            elif lr.status in ok_status:
                ncol.append(COL_NEW if lr.confidence >= 0.45 else COL_WARN)
            elif lr.status == "interpolé":
                ncol.append(COL_INTERP)
            else:
                ncol.append(QtGui.QColor(120, 160, 230))
        asr_rows = [(p.start, p.end, p.idx) for p in out.phrases]
        asr_text = [p.text for p in out.phrases]
        self._set_lane("orig", orig, ocol, otext)
        self._set_lane("asr", asr_rows, [COL_ASR] * len(asr_rows), asr_text)
        self._set_lane("new", new, ncol, [""] * len(new))
        self._update_labels()

    def update_line(self, lr) -> None:
        d = self.lane_data.get("new")
        if d is None:
            return
        k = np.where(d[:, 2] == lr.event_idx)[0]
        if len(k):
            d[k[0], 0], d[k[0], 1] = lr.new_start, lr.new_end
            self._redraw_lane("new")

    def _set_lane(self, name, rows, colors, texts) -> None:
        if name in self.lane_items:
            self.main.removeItem(self.lane_items.pop(name))
        arr = np.array(rows, dtype=float).reshape(-1, 3)
        self.lane_data[name] = arr
        self.lane_text[name] = texts
        self._colors[name] = colors
        self._redraw_lane(name)

    def _redraw_lane(self, name) -> None:
        if name in self.lane_items:
            self.main.removeItem(self.lane_items.pop(name))
        arr = self.lane_data[name]
        if not len(arr):
            return
        y0, y1 = LANES[name]
        brushes = [pg.mkBrush(c.red(), c.green(), c.blue(), 170) for c in self._colors[name]]
        pens = [pg.mkPen(c.lighter(140), width=1) for c in self._colors[name]]
        item = pg.BarGraphItem(x0=arr[:, 0], x1=np.maximum(arr[:, 1], arr[:, 0] + 0.02), y0=y0,
                               height=y1 - y0, brushes=brushes, pens=pens)
        item.setZValue(5)
        self.main.addItem(item)
        self.lane_items[name] = item

    # ------------------------------------------------------------------ étiquettes
    def _hide_texts(self) -> None:
        for t in self.text_pool:
            t.hide()

    def _update_labels(self) -> None:
        self._hide_texts()
        (x0, x1), _ = self.main.viewRange()
        if x1 - x0 > 45 or not self.lane_data:
            return
        k = 0
        for name in ("orig", "asr"):
            arr = self.lane_data.get(name)
            if arr is None or not len(arr):
                continue
            vis = np.where((arr[:, 1] >= x0) & (arr[:, 0] <= x1))[0][:80]
            y1 = LANES[name][1]
            for i in vis:
                if k >= len(self.text_pool):
                    ti = pg.TextItem(color=(240, 240, 245), anchor=(0, 0))
                    ti.setFont(QtGui.QFont("Segoe UI", 8))
                    ti.setZValue(10)
                    self.main.addItem(ti)
                    self.text_pool.append(ti)
                ti = self.text_pool[k]
                ti.setText(self.lane_text[name][i][:60])
                ti.setPos(max(arr[i, 0], x0), y1)
                ti.show()
                k += 1

    # ------------------------------------------------------------------ interaction
    def _hit(self, lane: str, x: float) -> int | None:
        arr = self.lane_data.get(lane)
        if arr is None or not len(arr):
            return None
        inside = np.where((arr[:, 0] <= x) & (arr[:, 1] >= x))[0]
        if len(inside):
            return int(arr[inside[np.argmin(arr[inside, 1] - arr[inside, 0])], 2])
        return None

    def _clicked(self, ev) -> None:
        if ev.button() != QtCore.Qt.LeftButton:
            return
        vb = self.main.vb
        if not self.main.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = vb.mapSceneToView(ev.scenePos())
        x, y = p.x(), p.y()
        for lane, (y0, y1) in LANES.items():
            if y0 <= y <= y1:
                hit = self._hit(lane, x)
                if hit is not None:
                    if lane == "asr":
                        self.phraseClicked.emit(hit)
                    elif ev.double():
                        self.lineDoubleClicked.emit(hit)
                    else:
                        self.lineClicked.emit(hit)
                return
        if y > LANES["orig"][1]:
            self.seekRequested.emit(max(0.0, x))

    def select_line(self, lr, zoom: bool = True) -> None:
        self._suspend_region = True
        self._sel_event = lr.event_idx
        self.reg_orig.setRegion((lr.old_start, lr.old_end))
        self.reg_new.setRegion((lr.new_start, lr.new_end))
        self.reg_orig.show()
        self.reg_new.show()
        self._suspend_region = False
        if zoom:
            (x0, x1), _ = self.main.viewRange()
            a, b = min(lr.old_start, lr.new_start), max(lr.old_end, lr.new_end)
            width = min(max(x1 - x0, 6.0), 30.0) if x1 - x0 < 60 else 12.0
            if not (x0 <= a and b <= x1) or x1 - x0 > 60:
                c = (a + b) / 2
                self.main.setXRange(c - width / 2, c + width / 2, padding=0)

    def _region_done(self) -> None:
        if self._suspend_region or self._sel_event is None:
            return
        a, b = self.reg_new.getRegion()
        self.regionEdited.emit(self._sel_event, float(a), float(b))

    def set_playhead(self, t: float) -> None:
        if t < 0:
            self.playhead.hide()
        else:
            self.playhead.setPos(t)
            self.playhead.show()

    def zoom_to(self, a: float, b: float) -> None:
        self.main.setXRange(a, b, padding=0)
