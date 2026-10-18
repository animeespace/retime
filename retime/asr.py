"""Transcription japonaise (faster-whisper sur GPU) et découpage en phrases guidé par l'audio."""
from __future__ import annotations

import gc
from dataclasses import dataclass, field

import numpy as np

from . import analysis as an
from .paths import MODELS_DIR

WHISPER_MODELS = {
    "large-v3 (précis)": "large-v3",
    "large-v3-turbo (rapide)": "large-v3-turbo",
    "kotoba-whisper v2 (japonais, rapide)": "kotoba-tech/kotoba-whisper-v2.0-faster",
}

# Hallucinations classiques de Whisper en japonais sur du silence / de la musique.
HALLUCINATIONS = ("ご視聴ありがとうございました", "チャンネル登録", "お疲れ様でした", "字幕", "最後までご視聴",
                  "おやすみなさい", "ご清聴ありがとうございました")


@dataclass
class Phrase:
    idx: int
    start: float                 # début de voix (attaque détectée sur la courbe)
    end: float                   # fin de voix
    text: str                    # japonais
    chars: list[tuple[float, float, str]] = field(default_factory=list)   # (début, fin, texte) par token
    translation: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


_loaded: dict = {}


def _get_model(model_id: str):
    """Le modèle reste en VRAM entre deux épisodes (évite ~10 s de rechargement)."""
    import torch  # noqa: F401  charge cuBLAS/cuDNN (CUDA 12) utilisés par CTranslate2
    from faster_whisper import WhisperModel

    if _loaded.get("id") != model_id:
        release()
        kw = dict(device="cuda", compute_type="float16", download_root=str(MODELS_DIR / "whisper"))
        try:      # cache local d'abord : aucun accès réseau, démarrage immédiat
            _loaded["model"] = WhisperModel(model_id, local_files_only=True, **kw)
        except Exception:  # noqa: BLE001  premier lancement : téléchargement
            _loaded["model"] = WhisperModel(model_id, **kw)
        _loaded["id"] = model_id
    return _loaded["model"]


def release() -> None:
    _loaded.clear()
    gc.collect()


def transcribe(audio16k: np.ndarray, model_key: str, beam_size: int = 5, batch_size: int = 16,
               progress=None, cancel=None) -> list[dict]:
    from faster_whisper import BatchedInferencePipeline

    model = _get_model(WHISPER_MODELS.get(model_key, model_key))
    pipe = BatchedInferencePipeline(model)
    duration = len(audio16k) / an.SR
    segments, _ = pipe.transcribe(
        audio16k, language="ja", task="transcribe", beam_size=beam_size, batch_size=batch_size,
        word_timestamps=True, vad_filter=True, condition_on_previous_text=False, temperature=0.0,
    )
    out = []
    for s in segments:
        if cancel and cancel():
            raise InterruptedError
        out.append({
            "start": float(s.start), "end": float(s.end), "text": s.text.strip(),
            "avg_logprob": float(s.avg_logprob), "no_speech_prob": float(s.no_speech_prob),
            "compression_ratio": float(s.compression_ratio),
            "words": [[float(w.start), float(w.end), w.word, float(w.probability)] for w in (s.words or [])],
        })
        if progress and duration:
            progress(min(1.0, s.end / duration))
    return out


def _keep_segment(seg: dict) -> bool:
    text = seg["text"]
    if not text or not seg["words"]:
        return False
    if seg["compression_ratio"] > 2.6:
        return False
    if seg["no_speech_prob"] > 0.75 and seg["avg_logprob"] < -0.8:
        return False
    if any(h in text for h in HALLUCINATIONS) and seg["avg_logprob"] < -0.4:
        return False
    return True


SENTENCE_END = ("。", "！", "？", "!", "?", "…", "♪")


def all_chars(segments: list[dict]) -> list[tuple[float, float, str]]:
    out = []
    for seg in segments:
        if _keep_segment(seg):
            out += [(ws, max(we, ws + 0.02), w.strip()) for ws, we, w, _p in seg["words"] if w.strip()]
    out.sort()
    return out


def build_sentences(segments: list[dict], f: an.AudioFeatures, max_len: float = 5.0) -> list[Phrase]:
    """Découpe la transcription en phrases : ponctuation de Whisper, puis silences de la courbe
    pour les passages longs sans ponctuation."""
    groups: list[list[tuple[float, float, str]]] = []
    for seg in segments:
        if not _keep_segment(seg):
            continue
        cur: list[tuple[float, float, str]] = []
        for ws, we, w, _p in seg["words"]:
            w = w.strip()
            if not w:
                continue
            if cur and ws - cur[-1][1] > 0.6:
                groups.append(cur)
                cur = []
            cur.append((ws, max(we, ws + 0.02), w))
            if w.endswith(SENTENCE_END):
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
    # passages longs : coupe aux silences >= 300 ms de la courbe
    final: list[list[tuple[float, float, str]]] = []
    for g in groups:
        if g[-1][1] - g[0][0] <= max_len:
            final.append(g)
            continue
        i0, i1 = f.t2i(g[0][0]), f.t2i(g[-1][1])
        act = an._clean_mask(f.active[i0:i1], fill_gap=29, min_run=4)
        d = np.diff(np.concatenate([[1], act.astype(np.int8), [1]]))
        cuts = [(i0 + (a + b) / 2) * an.HOP for a, b in zip(np.where(d == -1)[0], np.where(d == 1)[0])
                if 0 < a and b < len(act)]
        cur = []
        for c in g:
            if cur and cuts and (c[0] + c[1]) / 2 > cuts[0]:
                while cuts and (c[0] + c[1]) / 2 > cuts[0]:
                    cuts.pop(0)
                final.append(cur)
                cur = []
            cur.append(c)
        if cur:
            final.append(cur)
    out = []
    for g in final:
        text = "".join(c[2] for c in g).strip()
        if text:
            out.append(Phrase(idx=len(out), start=g[0][0], end=g[-1][1], text=text, chars=g))
    return out


def assign_chars(chars: list[tuple[float, float, str]], intervals: list[tuple[float, float]],
                 max_dist: float = 0.6) -> list[str]:
    """Répartit chaque caractère Whisper vers l'intervalle (ligne) le plus proche de son milieu :
    les temps par caractère de Whisper débordent souvent de ~0,2 s, une simple fenêtre en perd."""
    out = [[] for _ in intervals]
    if not intervals:
        return []
    order = np.argsort([a for a, _ in intervals])
    st = np.array([intervals[k][0] for k in order])
    en = np.array([intervals[k][1] for k in order])
    for c in chars:
        m = (c[0] + c[1]) / 2
        i = int(np.searchsorted(st, m))
        best, best_d = -1, max_dist
        for j in range(max(0, i - 3), min(len(st), i + 2)):
            d = max(st[j] - m, m - en[j], 0.0)
            if d < best_d or (d == best_d == 0.0):
                best, best_d = j, d
        if best >= 0:
            out[order[best]].append(c[2])
    return ["".join(x) for x in out]
