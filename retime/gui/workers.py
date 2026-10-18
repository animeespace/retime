"""Traitements longs exécutés hors du thread de l'interface, et lecture audio."""
from __future__ import annotations

import time
import traceback

import numpy as np
from PySide6 import QtCore


class RetimeWorker(QtCore.QThread):
    progress = QtCore.Signal(str, float, float)       # étape, avancement étape, avancement global
    log = QtCore.Signal(str)
    done = QtCore.Signal(object, object)              # MediaAnalysis, RetimeOutput
    failed = QtCore.Signal(str)

    def __init__(self, media_path, stream_index, ass_path, settings, analysis=None, parent=None):
        super().__init__(parent)
        self.media_path, self.stream_index, self.ass_path = media_path, stream_index, ass_path
        self.settings = settings
        self.analysis = analysis
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        from .. import pipeline

        try:
            cancel = lambda: self._cancel  # noqa: E731
            ma = self.analysis
            if ma is None:
                ma = pipeline.analyze_media(self.media_path, self.stream_index, self.settings,
                                            progress=lambda s, f, t: self.progress.emit(s, f, t * 0.85),
                                            cancel=cancel, log=self.log.emit)
            out = pipeline.run_retime(ma, self.ass_path, self.settings,
                                      progress=lambda s, f, t: self.progress.emit(s, f, 0.85 + 0.15 * t),
                                      cancel=cancel, log=self.log.emit)
            self.done.emit(ma, out)
        except InterruptedError:
            self.failed.emit("Annulé.")
        except Exception:
            self.failed.emit(traceback.format_exc())


class BatchWorker(QtCore.QThread):
    progress = QtCore.Signal(int, str, float)         # n° de tâche, étape, avancement global de la tâche
    log = QtCore.Signal(str)
    item_done = QtCore.Signal(int, str, dict)         # n° de tâche, chemin de sortie, stats
    item_failed = QtCore.Signal(int, str)
    finished_all = QtCore.Signal()

    def __init__(self, jobs, settings, parent=None):
        super().__init__(parent)
        self.jobs = jobs          # liste de dicts : media, stream, ass, out
        self.settings = settings
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        from .. import pipeline

        cancel = lambda: self._cancel  # noqa: E731
        for k, job in enumerate(self.jobs):
            if self._cancel:
                break
            try:
                t0 = time.time()
                self.log.emit(f"— {job['ass']}")
                ma = pipeline.analyze_media(job["media"], job["stream"], self.settings,
                                            progress=lambda s, f, t, k=k: self.progress.emit(k, s, t * 0.85),
                                            cancel=cancel, log=self.log.emit)
                out = pipeline.run_retime(ma, job["ass"], self.settings,
                                          progress=lambda s, f, t, k=k: self.progress.emit(k, s, 0.85 + 0.15 * t),
                                          cancel=cancel, log=self.log.emit)
                doc = pipeline.apply_result(out)
                doc.save(job["out"])
                st = dict(out.result.stats)
                st["seconds"] = time.time() - t0
                self.item_done.emit(k, job["out"], st)
            except InterruptedError:
                self.item_failed.emit(k, "Annulé")
                break
            except Exception as e:  # noqa: BLE001
                self.log.emit(traceback.format_exc())
                self.item_failed.emit(k, str(e))
        self.finished_all.emit()


class Player(QtCore.QObject):
    """Lecture d'un extrait (signal 16 kHz en mémoire : démarrage instantané)."""
    position = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(25)
        self.timer.timeout.connect(self._tick)
        self.t0 = 0.0
        self.start = 0.0
        self.end = 0.0

    def play(self, audio: np.ndarray, sr: int, start: float, end: float) -> None:
        import sounddevice as sd

        sd.stop()
        a, b = max(0, int(start * sr)), min(len(audio), int(end * sr))
        if b <= a:
            return
        clip = audio[a:b].astype(np.float32)
        peak = float(np.abs(clip).max()) or 1.0
        if peak > 0.98:
            clip = clip * (0.98 / peak)
        fade = min(len(clip) // 2, int(0.004 * sr))
        if fade > 1:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]
        sd.play(clip, sr)
        self.start, self.end = a / sr, b / sr
        self.t0 = time.perf_counter()
        self.timer.start()

    def stop(self) -> None:
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:  # noqa: BLE001
            pass
        self.timer.stop()
        self.position.emit(-1.0)

    def _tick(self) -> None:
        t = self.start + (time.perf_counter() - self.t0)
        if t >= self.end:
            self.timer.stop()
            self.position.emit(-1.0)
        else:
            self.position.emit(t)
