"""Analyse du signal : isolation vocale, courbe d'énergie, spectrogramme, points d'accroche (attaques/fins)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import butter, resample_poly, sosfiltfilt

SR = 16000
HOP = 0.01                      # une trame = 10 ms
HOP_SAMPLES = int(SR * HOP)


@dataclass
class AudioFeatures:
    energy_db: np.ndarray       # énergie bande voix (dB) par trame de 10 ms
    smooth_db: np.ndarray       # idem lissée (30 ms)
    floor_db: np.ndarray        # bruit de fond local
    thr_db: np.ndarray          # seuil de parole local
    active: np.ndarray          # décision parole / silence (bool)
    onsets: np.ndarray          # trames du franchissement montant
    offsets: np.ndarray         # trames du franchissement descendant (première trame silencieuse)
    on_t: np.ndarray            # début de voix (franchissement montant), s
    on_w: np.ndarray            # netteté 0..1
    on_gap: np.ndarray          # durée du silence qui précède, s
    off_t: np.ndarray           # fin de voix (première trame sous le seuil), s
    off_w: np.ndarray
    off_gap: np.ndarray         # durée du silence qui suit, s
    dip_t: np.ndarray           # creux marqués à l'intérieur de la parole (coupes possibles sans silence)
    dip_w: np.ndarray

    @property
    def n(self) -> int:
        return len(self.energy_db)

    def t2i(self, t: float) -> int:
        return int(np.clip(round(t / HOP), 0, self.n - 1))


# ----------------------------------------------------------------------------------------------
# Isolation vocale (Demucs htdemucs, GPU)
# ----------------------------------------------------------------------------------------------
def separate_vocals(stereo44k: np.ndarray, progress=None, cancel=None, device: str = "cuda") -> np.ndarray:
    """Renvoie la voix isolée en mono 16 kHz. stereo44k : (n, 2) float32 à 44,1 kHz."""
    import torch
    # Poids officiels Demucs (pickles de config) : torch>=2.6 refuse par défaut weights_only=False.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model("htdemucs")
    model.to(device).eval()
    vocals_idx = model.sources.index("vocals")
    sr = model.samplerate
    wav = torch.from_numpy(np.ascontiguousarray(stereo44k.T))           # (2, n)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    n = wav.shape[1]
    chunk = 180 * sr
    pad = 3 * sr
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    # fp16 : courbes identiques à fp32 (écart < 0,1 dB) et plus rapide
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        while pos < n:
            if cancel and cancel():
                raise InterruptedError
            a, b = max(0, pos - pad), min(n, pos + chunk + pad)
            seg = wav[:, a:b][None].to(device)
            src = apply_model(model, seg, device=device, shifts=0, split=True, overlap=0.25, progress=False)
            voc = src[0, vocals_idx].mean(0).float().cpu().numpy()
            keep_a, keep_b = pos - a, pos - a + min(chunk, n - pos)
            out[pos:pos + (keep_b - keep_a)] = voc[keep_a:keep_b]
            pos += chunk
            if progress:
                progress(min(1.0, pos / n))
    del model
    torch.cuda.empty_cache()
    out = out * float(std) + float(mean)
    # 44100 -> 16000 = *160/441
    return resample_poly(out, 160, 441).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# Courbes
# ----------------------------------------------------------------------------------------------
def band_energy_db(x: np.ndarray, lo: float = 150.0, hi: float = 4000.0) -> np.ndarray:
    """Énergie dans la bande vocale, filtre à phase nulle (aucun retard => timings exacts)."""
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    y = sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)
    n = len(y) // HOP_SAMPLES
    e = (y[: n * HOP_SAMPLES].reshape(n, HOP_SAMPLES) ** 2).mean(axis=1)
    e = uniform_filter1d(e, 2)
    return (10.0 * np.log10(e + 1e-10)).astype(np.float32)


def _rolling_percentile(x: np.ndarray, win: int, q: float, step: int = 10) -> np.ndarray:
    """Percentile glissant approché (calculé tous les `step` points puis interpolé)."""
    n = len(x)
    centers = np.arange(0, n, step)
    half = win // 2
    vals = np.empty(len(centers), dtype=np.float32)
    for k, c in enumerate(centers):
        vals[k] = np.percentile(x[max(0, c - half): min(n, c + half + 1)], q)
    return np.interp(np.arange(n), centers, vals).astype(np.float32)


def _clean_mask(mask: np.ndarray, fill_gap: int, min_run: int) -> np.ndarray:
    m = mask.copy()
    if fill_gap > 0:      # comble les trous courts (occlusives, respirations)
        m = minimum_filter1d(maximum_filter1d(m.astype(np.uint8), fill_gap + 1), fill_gap + 1).astype(bool) | m
    if min_run > 1:       # supprime les blips trop courts
        d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if e - s < min_run:
                m[s:e] = False
    return m


def compute_features(x16k: np.ndarray, sensitivity_db: float = 0.0) -> AudioFeatures:
    energy = band_energy_db(x16k)
    es = uniform_filter1d(energy, 3)
    floor = _rolling_percentile(energy, win=400, q=8.0, step=25)
    # niveau typique de la voix (local, 4 s) : le seuil suit les chuchotements comme les cris
    lvl = uniform_filter1d(maximum_filter1d(uniform_filter1d(energy, 15), 400), 200)
    thr = np.maximum(floor + 7.0, lvl - 28.0) - sensitivity_db
    active = _clean_mask(energy > thr, fill_gap=8, min_run=4)
    d = np.diff(np.concatenate([[0], active.astype(np.int8), [0]]))
    onsets, offsets = np.where(d == 1)[0], np.where(d == -1)[0]
    n = len(energy)
    # netteté du front : saut d'énergie entre le creux voisin (100 ms) et le pic voisin (120 ms)
    on_w = np.array([np.clip((es[c:c + 12].max() - es[max(0, c - 10):c + 1].min()) / 30.0, 0.2, 1.0)
                     for c in onsets])
    off_w = np.array([np.clip((es[max(0, c - 12):c + 1].max() - es[c:c + 10].min()) / 30.0, 0.2, 1.0)
                      if c < n else 0.2 for c in offsets])
    gaps = (onsets[1:] - offsets[:-1]) * HOP
    on_gap = np.concatenate([[onsets[0] * HOP if len(onsets) else 0.0], gaps]) if len(onsets) else np.zeros(0)
    off_gap = np.concatenate([gaps, [(n - offsets[-1]) * HOP if len(offsets) else 0.0]]) if len(offsets) \
        else np.zeros(0)
    # creux marqués au milieu de la parole : enchaînements sans silence
    local_max = maximum_filter1d(es, 31)
    is_min = (es == minimum_filter1d(es, 13)) & active & (local_max - es >= 10.0)
    dips = np.where(is_min)[0]
    dip_w = np.clip((local_max[dips] - es[dips]) / 30.0, 0.15, 0.6)
    return AudioFeatures(energy, es, floor, thr, active, onsets, offsets,
                         onsets * HOP, on_w, on_gap, offsets * HOP, off_w, off_gap,
                         dips * HOP + HOP / 2, dip_w)


def mel_spectrogram_u8(x16k: np.ndarray, n_mels: int = 96, fmax: float = 8000.0) -> np.ndarray:
    """Spectrogramme log-mel (trames de 10 ms) quantifié en uint8 pour l'affichage."""
    n_fft = 512
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = len(x16k) // HOP_SAMPLES
    xp = np.pad(x16k, (n_fft // 2, n_fft // 2 + HOP_SAMPLES))

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    mpts = mel2hz(np.linspace(hz2mel(40.0), hz2mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = mpts[i], mpts[i + 1], mpts[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (ce - lo), (hi - freqs) / (hi - ce)), 0, None)
    out = np.empty((n_frames, n_mels), dtype=np.float32)
    step = 20000
    for a in range(0, n_frames, step):
        b = min(n_frames, a + step)
        idx = (np.arange(a, b) * HOP_SAMPLES)[:, None] + np.arange(n_fft)[None, :]
        spec = np.abs(np.fft.rfft(xp[idx] * win, axis=1)) ** 2
        out[a:b] = 10 * np.log10(spec @ fb.T + 1e-10)
    lo, hi = np.percentile(out, 5), np.percentile(out, 99.7)
    return np.clip((out - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)


def waveform_envelope(x16k: np.ndarray, hop: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Min/max par bloc de 5 ms pour l'affichage de la forme d'onde."""
    n = len(x16k) // hop
    blk = x16k[: n * hop].reshape(n, hop)
    return blk.min(axis=1), blk.max(axis=1)


def speech_ratio(f: AudioFeatures, a: float, b: float) -> float:
    i, j = f.t2i(a), f.t2i(b)
    if j <= i:
        return 0.0
    return float(f.active[i:j].mean())
