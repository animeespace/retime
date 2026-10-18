"""Enchaînement complet : média -> analyse (mise en cache) -> recalage du .ass."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

from . import align, analysis as an, asr, media, semantic
from .assfile import AssDocument, write_simple_ass
from .paths import CACHE_DIR


@dataclass
class PipelineSettings:
    whisper_model: str = "large-v3 (précis)"
    vocal_separation: bool = True
    asr_on_vocals: bool = False
    translate: bool = True
    translation_model: str = "NLLB-200 600M (rapide)"
    target_lang: str = "Français"
    sensitivity_db: float = 0.0
    retime: align.RetimeSettings = field(default_factory=align.RetimeSettings)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineSettings":
        d = dict(d or {})
        rt = align.RetimeSettings(**{k: v for k, v in (d.pop("retime", {}) or {}).items()
                                     if k in align.RetimeSettings.__dataclass_fields__})
        ps = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "retime"})
        ps.retime = rt
        return ps


@dataclass
class MediaAnalysis:
    media: media.MediaInfo
    stream: media.AudioStream
    audio: np.ndarray                 # mixage 16 kHz mono
    vocals: np.ndarray | None         # voix isolée 16 kHz mono
    features: an.AudioFeatures
    segments: list[dict]
    keyframes: np.ndarray
    spectrogram: np.ndarray | None = None
    timings: dict = field(default_factory=dict)

    @property
    def signal(self) -> np.ndarray:
        return self.vocals if self.vocals is not None else self.audio


class Progress:
    """Convertit l'avancement d'étapes pondérées en un pourcentage global."""

    def __init__(self, callback, stages: list[tuple[str, float]]):
        self.cb = callback
        self.stages = dict(stages)
        self.order = [s for s, _ in stages]
        self.total = sum(w for _, w in stages) or 1.0
        self.done = 0.0

    def stage(self, name: str):
        base = sum(self.stages[s] for s in self.order[: self.order.index(name)])
        w = self.stages[name]

        def report(frac: float):
            if self.cb:
                self.cb(name, frac, (base + w * frac) / self.total)

        report(0.0)
        return report


def _cache_dir(path: str, stream_index: int) -> str:
    st = os.stat(path)
    key = f"{os.path.abspath(path).lower()}|{st.st_size}|{st.st_mtime_ns}|{stream_index}"
    d = CACHE_DIR / hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    d.mkdir(parents=True, exist_ok=True)
    meta = d / "meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"path": path, "stream": stream_index}, ensure_ascii=False), encoding="utf-8")
    return str(d)


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()


def analyze_media(path: str, stream_index: int, ps: PipelineSettings, progress=None, cancel=None,
                  log=print) -> MediaAnalysis:
    info = media.probe(path)
    stream = next((s for s in info.audio_streams if s.index == stream_index), None)
    if stream is None:
        raise ValueError("Piste audio introuvable")
    cdir = _cache_dir(path, stream_index)
    voc_file = os.path.join(cdir, "vocals16k.npy")
    src = "voc" if (ps.asr_on_vocals and ps.vocal_separation) else "mix"
    seg_file = os.path.join(cdir, f"whisper_{_slug(ps.whisper_model)}_{src}.json")
    run_demucs = ps.vocal_separation and not os.path.exists(voc_file)
    run_whisper = not os.path.exists(seg_file)
    # Whisper (CTranslate2) et Demucs (torch) tournent en même temps sur le GPU quand Whisper lit le mixage
    parallel = run_demucs and run_whisper and src == "mix"
    stages = [("Extraction audio", 3)]
    if parallel:
        stages.append(("Isolation vocale + transcription (GPU)", 70))
    else:
        stages += [("Isolation vocale (Demucs)", 22 if run_demucs else 1),
                   ("Transcription Whisper", 50 if run_whisper else 1)]
    stages += [("Courbes d'énergie", 3), ("Spectrogramme", 3)]
    prog = Progress(progress, stages)
    timings = {}

    pool = ThreadPoolExecutor(max_workers=3)
    kf_future = pool.submit(_cached_keyframes, path, cdir) if info.has_video else None

    t0 = time.time()
    rep = prog.stage("Extraction audio")
    stereo_future = pool.submit(media.extract_audio, path, stream_index, 44100, 2) if run_demucs else None
    audio = media.extract_audio(path, stream_index, an.SR, 1)
    rep(1.0)
    timings["extraction"] = time.time() - t0
    log(f"Audio extrait : {len(audio) / an.SR:.1f} s ({stream.label()})")

    def do_whisper(signal, report):
        t = time.time()
        segs = asr.transcribe(signal, ps.whisper_model, progress=report, cancel=cancel)
        with open(seg_file, "w", encoding="utf-8") as fh:
            json.dump(segs, fh, ensure_ascii=False)
        log(f"Transcription Whisper : {len(segs)} segments en {time.time() - t:.1f} s")
        return segs

    def do_demucs(report):
        t = time.time()
        stereo = stereo_future.result()
        voc = an.separate_vocals(stereo, progress=report, cancel=cancel)
        del stereo
        np.save(voc_file, voc.astype(np.float16))
        log(f"Voix isolée (Demucs) en {time.time() - t:.1f} s")
        return voc

    t0 = time.time()
    vocals, segments = None, None
    if parallel:
        rep = prog.stage("Isolation vocale + transcription (GPU)")
        fr = {"d": 0.0, "w": 0.0}

        def report(key):
            def _r(x):
                fr[key] = x
                rep((22 * fr["d"] + 50 * fr["w"]) / 72)
            return _r

        w_future = pool.submit(do_whisper, audio, report("w"))
        vocals = do_demucs(report("d"))
        segments = w_future.result()
        rep(1.0)
    else:
        rep = prog.stage("Isolation vocale (Demucs)")
        if ps.vocal_separation:
            if run_demucs:
                vocals = do_demucs(rep)
            else:
                vocals = np.load(voc_file).astype(np.float32)
                log("Voix isolée : chargée depuis le cache")
        rep(1.0)
        if cancel and cancel():
            raise InterruptedError
        rep = prog.stage("Transcription Whisper")
        if run_whisper:
            segments = do_whisper(vocals if (src == "voc" and vocals is not None) else audio, rep)
        else:
            segments = json.loads(open(seg_file, encoding="utf-8").read())
            log("Transcription : chargée depuis le cache")
        rep(1.0)
    timings["gpu"] = time.time() - t0
    if vocals is not None:
        vocals = vocals[:len(audio)].astype(np.float32)
    if cancel and cancel():
        raise InterruptedError

    t0 = time.time()
    rep = prog.stage("Courbes d'énergie")
    sig = vocals if vocals is not None else audio
    feats = an.compute_features(sig, ps.sensitivity_db)
    rep(1.0)
    timings["features"] = time.time() - t0

    rep = prog.stage("Spectrogramme")
    spec = an.mel_spectrogram_u8(sig)
    rep(1.0)
    kfs = kf_future.result() if kf_future else np.zeros(0)
    pool.shutdown(wait=False)
    return MediaAnalysis(info, stream, audio, vocals, feats, segments, kfs, spec, timings)


def _cached_keyframes(path: str, cdir: str) -> np.ndarray:
    f = os.path.join(cdir, "keyframes.npy")
    if os.path.exists(f):
        return np.load(f)
    k = media.keyframes(path)
    np.save(f, k)
    return k


@dataclass
class RetimeOutput:
    doc: AssDocument
    result: align.RetimeResult
    phrases: list[asr.Phrase]


def run_retime(ma: MediaAnalysis, ass_path: str, ps: PipelineSettings, progress=None, cancel=None,
               log=print) -> RetimeOutput:
    stages = [("Modèle de similarité (LaBSE)", 4), ("Alignement", 4)]
    if ps.translate:
        stages.append(("Traduction", 12))
    stages.append(("Vérification du sens", 3))
    prog = Progress(progress, stages)

    rep = prog.stage("Modèle de similarité (LaBSE)")
    sentences = asr.build_sentences(ma.segments, ma.features)
    chars = asr.all_chars(ma.segments)
    emb = semantic.get_embedder()
    rep(1.0)
    log(f"{len(sentences)} phrases japonaises, {len(chars)} caractères horodatés")

    rep = prog.stage("Alignement")
    doc = AssDocument(ass_path)
    res = align.retime(doc, sentences, chars, ma.features, emb, ps.retime, ma.media.fps, ma.keyframes)
    rep(1.0)
    st = res.stats
    log(f"Alignement : {st['aligned']}/{st['units']} lignes calées sur la voix, {st['interpolated']} interpolées, "
        f"décalage global {st['global_offset'] * 1000:+.0f} ms, local médian {st['median_offset'] * 1000:+.0f} ms "
        f"({st['anchors']} ancres sémantiques)")
    if cancel and cancel():
        raise InterruptedError

    translations = None
    if ps.translate:
        t0 = time.time()
        rep = prog.stage("Traduction")
        tgt = semantic.TARGET_LANGS.get(ps.target_lang, ps.target_lang)
        heard = sorted({lr.heard for lr in res.lines if lr.heard})
        trs = semantic.translate(heard, ps.translation_model, tgt, progress=rep, cancel=cancel)
        translations = dict(zip(heard, trs))
        log(f"Traduction JA→{ps.target_lang} de {len(heard)} répliques en {time.time() - t0:.1f} s")
    rep = prog.stage("Vérification du sens")
    align.score_text(res, emb, translations)
    rep(1.0)
    return RetimeOutput(doc, res, sentences)


def apply_result(out: RetimeOutput) -> AssDocument:
    """Document neuf (relu depuis le disque) avec les nouveaux temps des lignes cochées."""
    doc = AssDocument(out.doc.path)
    for lr in out.result.lines:
        if lr.apply and lr.changed:
            doc.set_times(doc.events[lr.event_idx], lr.new_start, lr.new_end)
    return doc


def japanese_subs(out: RetimeOutput, ma: MediaAnalysis, rt: align.RetimeSettings) -> list[tuple[float, float, str]]:
    """Sous-titres japonais générés : phrases Whisper accrochées sur la courbe d'énergie."""
    f = ma.features
    items = []
    for p in out.phrases:
        s, e = p.start, p.end
        c = f.on_t[(f.on_t >= s - 0.35) & (f.on_t <= s + 0.25)]
        if len(c):
            s = float(c[np.argmin(np.abs(c - s))])
        c = f.off_t[(f.off_t >= e - 0.25) & (f.off_t <= e + 0.4)]
        if len(c):
            e = float(c[np.argmin(np.abs(c - e))])
        s, e = s - rt.lead_in, max(e + rt.lead_out, s + 0.3)
        if ma.media.fps:
            s, e = align.frame_time(s, ma.media.fps), align.frame_time(e, ma.media.fps)
        items.append((s, e, p.text))
    for k in range(len(items) - 1):
        if items[k][1] > items[k + 1][0]:
            items[k] = (items[k][0], items[k + 1][0], items[k][2])
    return items


def export_japanese(out: RetimeOutput, ma: MediaAnalysis, rt: align.RetimeSettings, path: str) -> None:
    write_simple_ass(path, japanese_subs(out, ma, rt))


def release_models() -> None:
    asr.release()
    semantic.release()
