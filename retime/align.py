"""Mise en correspondance des lignes .ass avec la voix et calcul des nouveaux timings.

Étapes :
  1. décalage global par corrélation (activité vocale vs couverture des sous-titres) ;
  2. ancres sémantiques : phrases japonaises de Whisper <-> lignes du .ass (LaBSE multilingue),
     alignement monotone par programmation dynamique -> courbe de décalage local ;
  3. accroche de chaque ligne sur la courbe d'énergie : attaque / fin de voix les plus probables
     autour du timing d'origine recalé (proximité, netteté du front, silence avant/après) ;
  4. affinage du décalage local à partir des accroches fiables, puis second passage ;
  5. post-traitement : lead-in/out, chevauchements, liaison, durée mini, keyframes, frames, tolérance.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from . import analysis as an
from .asr import Phrase, assign_chars
from .assfile import AssDocument, AssEvent

RETIME, SHIFT, KEEP = "retime", "shift", "keep"
POLICY_LABELS = {RETIME: "Recaler sur la voix", SHIFT: "Décaler seulement", KEEP: "Ne pas toucher"}
OK_STATUS = ("calé", "calé (début)", "calé (fin)")

_SIGN_WORDS = ("sign", "signs", "ts", "typeset", "titre", "title", "card", "note", "notes", "op", "ed",
               "kara", "karaoke", "song", "chanson", "insert", "logo", "panneau", "next", "credit", "credits",
               "staff", "eyecatch", "screen", "ecran", "écran", "lyrics", "paroles")


def default_policy(style: str, events: list[AssEvent]) -> str:
    tokens = [t for t in re.split(r"[^0-9a-zàâäçéèêëîïôöûùüÿœ]+", style.lower()) if t]
    if any(t in _SIGN_WORDS for t in tokens):
        return SHIFT
    evs = [e for e in events if e.style == style and not e.is_comment]
    if evs and sum(e.is_positioned or e.is_karaoke or e.is_drawing for e in evs) / len(evs) > 0.5:
        return SHIFT
    return RETIME


@dataclass
class RetimeSettings:
    lead_in: float = 0.03            # le sous-titre apparaît un peu avant l'attaque de la voix
    lead_out: float = 0.08           # et reste un peu après la retombée
    link_gap: float = 0.0
    min_duration: float = 0.30
    keep_tolerance: float = 0.0      # écart en deçà duquel le timing d'origine est conservé
    snap_frames: bool = True
    snap_keyframes: bool = False
    kf_start_before: float = 0.15
    kf_start_after: float = 0.10
    kf_end_before: float = 0.10
    kf_end_after: float = 0.35
    max_offset: float = 30.0
    search_window: float = 0.8       # rayon de recherche des points d'accroche autour du timing recalé
    min_shift: float = 0.20          # en deçà, les lignes "décaler seulement" ne bougent pas
    style_policies: dict = field(default_factory=dict)


@dataclass
class Unit:
    idx: int
    events: list[int]
    start: float
    end: float
    text: str


@dataclass
class Match:
    units: tuple[int, int]
    phrases: tuple[int, int]
    sem: float


@dataclass
class LineResult:
    event_idx: int
    style: str
    text: str
    old_start: float
    old_end: float
    new_start: float
    new_end: float
    status: str
    confidence: float = 0.0
    similarity: float = -1.0
    heard: str = ""
    translation: str = ""
    unit: int | None = None
    apply: bool = True
    policy: str = RETIME
    voice: tuple[float, float] | None = None

    @property
    def changed(self) -> bool:
        return abs(self.new_start - self.old_start) >= 0.005 or abs(self.new_end - self.old_end) >= 0.005


@dataclass
class RetimeResult:
    lines: list[LineResult]
    units: list[Unit]
    phrases: list[Phrase]
    matches: list[Match]
    offset_t: np.ndarray
    offset_v: np.ndarray
    global_offset: float
    stats: dict

    def offset_at(self, t: float) -> float:
        return float(np.interp(t, self.offset_t, self.offset_v))


# ----------------------------------------------------------------------------------------------
# Unités (groupes de lignes au timing identique)
# ----------------------------------------------------------------------------------------------
def event_policy(ev: AssEvent, policies: dict) -> str:
    if ev.is_comment:
        return KEEP
    pol = policies.get(ev.style, RETIME)
    if pol == RETIME and (ev.is_positioned or ev.is_karaoke or ev.is_drawing):
        return SHIFT
    if pol == RETIME and not ev.plain:
        return KEEP
    return pol


def build_units(doc: AssDocument, policies: dict) -> list[Unit]:
    evs = [e for e in doc.events if event_policy(e, policies) == RETIME]
    evs.sort(key=lambda e: (e.start, e.end, e.idx))
    units: list[Unit] = []
    for e in evs:
        if units and abs(units[-1].start - e.start) < 0.005 and abs(units[-1].end - e.end) < 0.005:
            u = units[-1]
            u.events.append(e.idx)
            if e.plain not in u.text:
                u.text = f"{u.text} {e.plain}"
        else:
            units.append(Unit(len(units), [e.idx], e.start, e.end, e.plain))
    return units


# ----------------------------------------------------------------------------------------------
# Décalage global
# ----------------------------------------------------------------------------------------------
_F23, _F29 = 24000 / 1001, 30000 / 1001
FPS_RATIOS = (1.0, 25 / _F23, _F23 / 25, 24 / _F23, _F23 / 24, 25 / 24, 24 / 25, 30 / _F29, _F29 / 30)


def _xcorr(units: list[Unit], f: an.AudioFeatures, max_offset: float, scale: float) -> tuple[float, float]:
    """Décalage maximisant la corrélation activité vocale / couverture des sous-titres (temps x scale)."""
    n = f.n
    sub = np.zeros(n, np.float32)
    for u in units:
        sub[f.t2i(u.start * scale): f.t2i(u.end * scale)] = 1.0
    act = f.active.astype(np.float32)
    a, b = act - act.mean(), sub - sub.mean()
    size = 1 << int(math.ceil(math.log2(2 * n)))
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    k = min(int(max_offset / an.HOP), n - 1)
    lags = np.concatenate([np.arange(0, k + 1), np.arange(-k, 0)])
    vals = np.concatenate([corr[: k + 1], corr[size - k:]])
    best = int(np.argmax(vals))
    return float(lags[best] * an.HOP), float(vals[best] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def estimate_scale_offset(units: list[Unit], f: an.AudioFeatures, max_offset: float) -> tuple[float, float]:
    """Décalage global, et rapport de framerate s'il y a dérive (ex. sous-titres calés sur du 25 fps)."""
    if not units or f.n < 10:
        return 1.0, 0.0
    off1, peak1 = _xcorr(units, f, max_offset, 1.0)
    best = (1.0, off1, peak1)
    for r in FPS_RATIOS[1:]:
        off, peak = _xcorr(units, f, max_offset, r)
        if peak > best[2]:
            best = (r, off, peak)
    if best[0] != 1.0 and best[2] < peak1 * 1.25:     # la dérive doit être nettement plus probable
        best = (1.0, off1, peak1)
    return best[0], best[1]


# ----------------------------------------------------------------------------------------------
# Ancres sémantiques (phrases japonaises <-> lignes)
# ----------------------------------------------------------------------------------------------
K_SPAN = 3


def _spans(texts: list[str], k: int, sep: str) -> list[str]:
    return [sep.join(texts[i:i + k]) for i in range(len(texts))]


def similarity_tables(units: list[Unit], phrases: list[Phrase], embedder) -> tuple[dict, dict]:
    """A[k][i, j] = sim(ligne i, phrases j..j+k-1) ; B[r][i, j] = sim(lignes i..i+r-1, phrase j)."""
    ut = [u.text for u in units]
    pt = [p.text for p in phrases]
    U = {r: embedder.encode(_spans(ut, r, " ")) for r in range(1, K_SPAN + 1)}
    P = {k: embedder.encode(_spans(pt, k, "")) for k in range(1, K_SPAN + 1)}
    A = {k: U[1] @ P[k].T for k in range(1, K_SPAN + 1)}
    B = {r: U[r] @ P[1].T for r in range(2, K_SPAN + 1)}
    return A, B


W_SEM, SEM0 = 2.5, 0.33
W_TIME, TIME0 = 1.0, 0.30
SPAN_PEN = 0.12
SKIP_U, SKIP_P, M1, MR = 1, 2, 10, 20


def _score(sem: float, ds: float, de: float, sigma: float) -> float:
    t = 0.5 * (math.exp(-0.5 * (ds / sigma) ** 2) + math.exp(-0.5 * (de / sigma) ** 2))
    return W_SEM * (sem - SEM0) + W_TIME * (t - TIME0)


def dp_align(units: list[Unit], phrases: list[Phrase], A: dict, B: dict, offs: np.ndarray,
             sigma: float, band: float) -> list[Match]:
    """Alignement monotone lignes <-> phrases (1:k et r:1), lignes ou phrases sautables."""
    n, m = len(units), len(phrases)
    if n == 0 or m == 0:
        return []
    us = np.array([u.start for u in units])
    ue = np.array([u.end for u in units])
    ps = np.array([p.start for p in phrases])
    pe = np.array([p.end for p in phrases])
    D = np.zeros((n + 1, m + 1))
    BP = np.zeros((n + 1, m + 1), np.int8)
    BP[0, 1:] = SKIP_P
    SEM = np.zeros((n + 1, m + 1), np.float32)
    for i in range(1, n + 1):
        u = i - 1
        row = D[i - 1].copy()
        bp = np.full(m + 1, SKIP_U, np.int8)
        sem_row = np.zeros(m + 1, np.float32)
        o = offs[u]
        jlo = int(np.searchsorted(pe, us[u] + o - band))
        jhi = int(np.searchsorted(ps, ue[u] + o + band))
        for j in range(jlo + 1, jhi + 1):
            last = j - 1
            for k in range(1, K_SPAN + 1):
                first = j - k
                if first < jlo:
                    break
                sem = float(A[k][u, first])
                sc = D[i - 1, first] + _score(sem, ps[first] - us[u] - o, pe[last] - ue[u] - o, sigma) \
                    - SPAN_PEN * (k - 1)
                if sc > row[j]:
                    row[j], bp[j], sem_row[j] = sc, M1 + k, sem
            for r in range(2, K_SPAN + 1):
                a = u - r + 1
                if a < 0:
                    break
                sem = float(B[r][a, last])
                sc = D[i - r, last] + _score(sem, ps[last] - us[a] - offs[a], pe[last] - ue[u] - o, sigma) \
                    - SPAN_PEN * (r - 1)
                if sc > row[j]:
                    row[j], bp[j], sem_row[j] = sc, MR + r, sem
        acc = np.maximum.accumulate(row)
        bp[acc > row] = SKIP_P
        D[i], BP[i], SEM[i] = acc, bp, sem_row
    matches: list[Match] = []
    i, j = n, m
    while i > 0 and j > 0:
        code = int(BP[i, j])
        if code == SKIP_U:
            i -= 1
        elif code == SKIP_P:
            j -= 1
        elif M1 < code <= M1 + K_SPAN:
            k = code - M1
            matches.append(Match((i - 1, i), (j - k, j), float(SEM[i, j])))
            i, j = i - 1, j - k
        else:
            r = code - MR
            matches.append(Match((i - r, i), (j - 1, j), float(SEM[i, j])))
            i, j = i - r, j - 1
    matches.reverse()
    return matches


def semantic_anchors(matches: list[Match], units: list[Unit], phrases: list[Phrase]) -> list[tuple[int, float, float]]:
    """(ligne, décalage, similarité) pour les correspondances fiables ligne <-> phrase japonaise."""
    out = []
    for mt in matches:
        if mt.sem < 0.45 or mt.units[1] - mt.units[0] != 1:
            continue
        u, p = units[mt.units[0]], phrases[mt.phrases[0]]
        out.append((mt.units[0], p.start - u.start, mt.sem))
    return out


def track_offsets(units: list[Unit], f: an.AudioFeatures, g: float, anchors: list[tuple[int, float, float]],
                  radius: float, w_sem: float = 0.8, lam: float = 4.0, jump: float = 5.0) -> np.ndarray:
    """Suivi du décalage ligne par ligne (Viterbi) sur une grille g ± radius.

    Émission : la ligne décalée tombe-t-elle sur de la voix (corrélation) + ancres sémantiques proches.
    Transition : varier coûte lam/s, un saut franc (coupure, pub retirée) coûte `jump`."""
    n = len(units)
    if n == 0:
        return np.zeros(0)
    step = max(0.05, 2 * radius / 1200)
    deltas = g + np.arange(-radius, radius + 1e-9, step)
    S = len(deltas)
    pref = np.concatenate([[0.0], np.cumsum(2.0 * f.active.astype(np.float32) - 1.0)])
    us = np.array([u.start for u in units])
    ue = np.array([u.end for u in units])
    si = np.clip(np.round((us[:, None] + deltas[None, :]) / an.HOP), 0, f.n).astype(int)
    ei = np.clip(np.round((ue[:, None] + deltas[None, :]) / an.HOP), 0, f.n).astype(int)
    M = (pref[ei] - pref[si]) / np.maximum(1, ei - si)
    for j, a, sem in anchors:
        M[j] += w_sem * np.clip((sem - 0.3) / 0.4, 0, 1.5) * np.exp(-0.5 * ((deltas - a) / 0.35) ** 2)
    c = lam * step
    idx = np.arange(S)
    V = M[0] - 0.02 * np.abs(deltas - g)
    hist = [V]
    for j in range(1, n):
        fwd = np.maximum.accumulate(V + c * idx) - c * idx
        bwd = (np.maximum.accumulate((V - c * idx)[::-1]) + c * idx[::-1])[::-1]
        V = np.maximum(np.maximum(fwd, bwd), V.max() - jump) + M[j]
        hist.append(V)
    k = int(np.argmax(hist[-1]))
    path = [k]
    for j in range(n - 2, -1, -1):
        prev = hist[j] - np.minimum(c * np.abs(idx - k), jump)
        k = int(np.argmax(prev))
        path.append(k)
    return deltas[np.array(path[::-1])]


# ----------------------------------------------------------------------------------------------
# Accroche sur la courbe
# ----------------------------------------------------------------------------------------------
@dataclass
class _Snap:
    vs: float
    ve: float
    s_score: float
    e_score: float


def local_offsets(units: list[Unit], f: an.AudioFeatures, base: np.ndarray, radius: float = 1.0,
                  step: float = 0.02, smooth: float = 12.0) -> np.ndarray:
    """Décalage local par corrélation glissante entre l'activité vocale et la disposition des sous-titres
    (±radius autour de `base`, lissé sur ~smooth secondes). Indépendant des temps de Whisper."""
    if not units:
        return base
    pref = np.concatenate([[0.0], np.cumsum(2.0 * f.active.astype(np.float32) - 1.0)])
    deltas = np.arange(-radius, radius + 1e-9, step)
    us = np.array([u.start for u in units])
    ue = np.array([u.end for u in units])
    s_idx = np.clip(np.round((us[:, None] + base[:, None] + deltas[None, :]) / an.HOP), 0, f.n).astype(int)
    e_idx = np.clip(np.round((ue[:, None] + base[:, None] + deltas[None, :]) / an.HOP), 0, f.n).astype(int)
    M = pref[e_idx] - pref[s_idx]
    W = np.exp(-0.5 * ((us[:, None] - us[None, :]) / smooth) ** 2)
    Ms = W @ M
    Ms -= 0.08 * np.abs(deltas)[None, :] / radius * np.abs(Ms).max(axis=1, keepdims=True)
    return base + deltas[np.argmax(Ms, axis=1)]


FINE_RADIUS, FINE_SMOOTH = 0.6, 8.0
MIN_SNAP = 0.18
TAU_NEAR, TAU_FAR = 0.13, 0.20
DIP_FACTOR = 0.45


def _pick(t, w, gap, is_dip, target, tau_before, tau_after, lo, hi):
    """Meilleur candidat dans [lo, hi] : proximité (asymétrique) x netteté x silence adjacent."""
    sel = (t >= lo) & (t <= hi)
    if not sel.any():
        return None, 0.0
    tt, ww, gg, dd = t[sel], w[sel], gap[sel], is_dip[sel]
    d = tt - target
    prox = np.where(d < 0, np.exp(d / tau_before), np.exp(-d / tau_after))
    sc = prox * (0.5 + 0.25 * ww + 0.25 * np.minimum(1.0, gg / 0.3)) * np.where(dd, DIP_FACTOR, 1.0)
    k = int(np.argmax(sc))
    if sc[k] < MIN_SNAP:
        return None, 0.0
    return float(tt[k]), float(sc[k])


def snap_units(units: list[Unit], f: an.AudioFeatures, offs: np.ndarray, win: float) -> list[_Snap]:
    """Accroche début/fin de chaque ligne. Comme un humain devant la courbe : un début posé dans un
    silence cherche l'attaque suivante, posé en pleine voix il remonte à l'attaque précédente ; une fin
    posée dans un silence revient à la retombée précédente, en pleine voix elle va à la suivante."""
    nd = len(f.dip_t)
    st_t = np.concatenate([f.on_t, f.dip_t])
    st_w = np.concatenate([f.on_w, f.dip_w])
    st_g = np.concatenate([f.on_gap, np.zeros(nd)])
    st_d = np.concatenate([np.zeros(len(f.on_t), bool), np.ones(nd, bool)])
    en_t = np.concatenate([f.off_t, f.dip_t])
    en_w = np.concatenate([f.off_w, f.dip_w])
    en_g = np.concatenate([f.off_gap, np.zeros(nd)])
    en_d = np.concatenate([np.zeros(len(f.off_t), bool), np.ones(nd, bool)])
    out: list[_Snap] = []
    prev_end, prev_orig_end = -1e9, -1e9
    for k, u in enumerate(units):
        S, E = u.start + offs[k], u.end + offs[k]
        short = E - S < 0.25                    # ligne de durée nulle/aberrante : fin inconnue
        overlap_orig = prev_orig_end > u.start + 0.1
        lo = S - win if overlap_orig else max(S - win, prev_end - 0.02)
        s_in_voice = bool(f.active[f.t2i(S)])
        tb, ta = (TAU_FAR, TAU_NEAR) if s_in_voice else (TAU_NEAR, TAU_FAR)
        vs, ss = _pick(st_t, st_w, st_g, st_d, S, tb, ta, lo, S + win if short else min(S + win, E - 0.1))
        if vs is None:
            vs, ss = max(S, lo), 0.0
        nxt = units[k + 1].start + offs[k + 1] if k + 1 < len(units) else 1e9
        if short:
            E = vs + 0.6
            hi = min(vs + 3.0, nxt + 0.15)
            tb, ta = TAU_NEAR, 1.5
        else:
            hi = max(E + 0.15, min(E + win, nxt + 0.15))
            e_in_voice = bool(f.active[f.t2i(E - 0.02)])
            tb, ta = (TAU_NEAR, TAU_FAR) if e_in_voice else (TAU_FAR, TAU_NEAR)
        ve, se = _pick(en_t, en_w, en_g, en_d, E, tb, ta, max(vs + 0.15, E - win), hi)
        if ve is None:
            ve, se = max(E, vs + 0.2), 0.0
        out.append(_Snap(vs, ve, ss, se))
        prev_end, prev_orig_end = ve, max(prev_orig_end, u.end)
    return out


# ----------------------------------------------------------------------------------------------
# Post-traitement
# ----------------------------------------------------------------------------------------------
def frame_time(t: float, fps: Fraction) -> float:
    """Calage sur la frame la plus proche, écrit au centième inférieur (convention Aegisub)."""
    n = round(t * fps)
    cs = (n * 100 * fps.denominator) // fps.numerator
    return cs / 100.0


@dataclass
class _UT:
    start: float
    end: float
    voice_start: float
    voice_end: float
    orig_start: float
    orig_end: float
    status: str
    conf: float


def postprocess(ut: list[_UT], s: RetimeSettings, fps: Fraction | None, kfs: np.ndarray | None) -> None:
    order = sorted(range(len(ut)), key=lambda k: (ut[k].voice_start, ut[k].voice_end))
    for x in ut:
        x.start = x.voice_start - s.lead_in
        x.end = x.voice_end + s.lead_out

    def fix_overlaps():
        for a, b in zip(order, order[1:]):
            A, B = ut[a], ut[b]
            if A.orig_end > B.orig_start + 0.1:      # chevauchement voulu (deux personnages)
                continue
            if A.end > B.start:
                if A.voice_end <= B.start:
                    A.end = B.start
                else:
                    B.start = max(B.start, min(A.voice_end, B.voice_start))
                    A.end = B.start
            elif s.link_gap > 0 and B.start - A.end < s.link_gap:
                A.end = B.start

    fix_overlaps()
    for a, b in zip(order, order[1:] + [None]):
        A = ut[a]
        if A.end - A.start < s.min_duration:
            limit = ut[b].start if b is not None else A.start + s.min_duration
            A.end = max(A.end, min(A.start + s.min_duration, limit))
    if s.snap_keyframes and kfs is not None and len(kfs):
        for x in ut:
            lo, hi = x.start - s.kf_start_before, min(x.start + s.kf_start_after, x.voice_start + 0.045)
            k = kfs[(kfs >= lo) & (kfs <= hi)]
            if len(k):
                x.start = float(k[np.argmin(np.abs(k - x.start))])
            lo, hi = max(x.end - s.kf_end_before, x.voice_end - 0.045), x.end + s.kf_end_after
            k = kfs[(kfs >= lo) & (kfs <= hi)]
            if len(k):
                x.end = float(k[np.argmin(np.abs(k - x.end))])
        fix_overlaps()
    for x in ut:
        if fps:
            x.start, x.end = frame_time(x.start, fps), frame_time(x.end, fps)
        else:
            x.start, x.end = round(x.start, 2), round(x.end, 2)
        if s.keep_tolerance > 0:
            if abs(x.start - x.orig_start) <= s.keep_tolerance:
                x.start = x.orig_start
            if abs(x.end - x.orig_end) <= s.keep_tolerance:
                x.end = x.orig_end
        if x.end <= x.start:
            x.end = x.start + 0.05
            if fps:
                x.end = frame_time(x.end + 0.03, fps)


# ----------------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------------
def retime(doc: AssDocument, sentences: list[Phrase], chars: list[tuple[float, float, str]],
           f: an.AudioFeatures, embedder, s: RetimeSettings, fps: Fraction | None,
           kfs: np.ndarray | None) -> RetimeResult:
    policies = {st: s.style_policies.get(st) or default_policy(st, doc.events)
                for st in {e.style for e in doc.events} | set(doc.styles)}
    units = build_units(doc, policies)
    orig_times = [(u.start, u.end) for u in units]
    scale, g = estimate_scale_offset(units, f, s.max_offset)
    if scale != 1.0:                      # dérive de framerate : on travaille en temps corrigés
        for u in units:
            u.start, u.end = u.start * scale, u.end * scale
    us = np.array([u.start for u in units])

    # 1) ancres sémantiques (phrases japonaises <-> lignes) + suivi du décalage (coupures, scènes retirées)
    matches: list[Match] = []
    anchors: list[tuple[int, float, float]] = []
    if units and sentences and embedder is not None:
        A, B = similarity_tables(units, sentences, embedder)
        matches = dp_align(units, sentences, A, B, np.full(len(units), g), sigma=3.0, band=min(s.max_offset, 20.0))
        anchors = semantic_anchors(matches, units, sentences)
    n_anchor = len(anchors)
    if anchors:   # Whisper place les débuts de phrase avec un biais propre : on le retire
        near = [a for a in anchors if abs(a[1] - g) < 1.0]
        bias = float(np.median([a[1] for a in near])) - g if len(near) >= 5 else 0.0
        anchors = [(j, a - bias, sm) for j, a, sm in anchors]
    track = track_offsets(units, f, g, anchors, radius=min(s.max_offset, 20.0))
    # seuls les vrais écarts (coupures, scènes décalées) sont gardés ; le fin est fait par l'audio
    base = np.where(np.abs(track - g) > 0.5, track, g)

    # 2) affinage fin par corrélation voix <-> sous-titres, puis accroche sur la courbe
    offs = local_offsets(units, f, base, radius=FINE_RADIUS, smooth=FINE_SMOOTH)
    snaps = snap_units(units, f, offs, win=s.search_window)
    ot, ov = (us, offs) if len(units) else (np.array([0.0, 1.0]), np.array([g, g]))

    ut: list[_UT] = []
    for k, (u, sn) in enumerate(zip(units, snaps)):
        if sn.s_score > 0 and sn.e_score > 0:
            status = "calé"
        elif sn.s_score > 0:
            status = "calé (début)"
        elif sn.e_score > 0:
            status = "calé (fin)"
        else:
            status = "interpolé"
        conf = 0.5 * (sn.s_score + sn.e_score) / 0.75
        os_, oe = orig_times[k]
        ut.append(_UT(sn.vs, sn.ve, sn.vs, sn.ve, os_, oe, status, float(np.clip(conf, 0, 1))))
    postprocess(ut, s, fps, kfs)

    heard = assign_chars(chars, [(x.voice_start, x.voice_end) for x in ut])
    lines: list[LineResult] = []
    unit_of = {e: k for k, u in enumerate(units) for e in u.events}
    for ev in doc.events:
        pol = event_policy(ev, policies)
        lr = LineResult(ev.idx, ev.style, ev.plain, ev.start, ev.end, ev.start, ev.end, "inchangé", policy=pol)
        if pol == RETIME and ev.idx in unit_of:
            k = unit_of[ev.idx]
            x = ut[k]
            lr.new_start, lr.new_end = x.start, x.end
            lr.status, lr.confidence, lr.unit = x.status, x.conf, k
            lr.voice = (x.voice_start, x.voice_end)
            lr.heard = heard[k]
        elif pol == SHIFT:
            o = float(np.interp(ev.start * scale, ot, ov)) + ev.start * (scale - 1.0)
            if abs(o) >= s.min_shift:
                ns, ne = ev.start + o, ev.start + o + (ev.end - ev.start) * scale
                if fps:
                    ns, ne = frame_time(ns, fps), frame_time(ne, fps)
                lr.new_start, lr.new_end, lr.status, lr.confidence = ns, ne, "décalé", 0.5
        lr.apply = lr.changed
        lines.append(lr)

    stats = {
        "units": len(units),
        "phrases": len(sentences),
        "anchors": n_anchor,
        "aligned": sum(1 for x in ut if x.status in OK_STATUS),
        "interpolated": sum(1 for x in ut if x.status == "interpolé"),
        "global_offset": g,
        "scale": scale,
        "median_offset": float(np.median(offs)) if len(offs) else g,
        "changed_lines": sum(1 for x in lines if x.changed),
    }
    return RetimeResult(lines, units, sentences, matches, ot, ov, g, stats)


def score_text(res: RetimeResult, embedder, translations: dict[str, str] | None) -> None:
    """Similarité texte .ass <-> japonais entendu (et sa traduction) : alimente la confiance."""
    rows = [lr for lr in res.lines if lr.unit is not None]
    if not rows or embedder is None:
        return
    ass = embedder.encode([lr.text for lr in rows])
    ja = embedder.encode([lr.heard or "…" for lr in rows])
    sims = (ass * ja).sum(1)
    if translations:
        for lr in rows:
            lr.translation = translations.get(lr.heard, "")
        tr = embedder.encode([lr.translation or "…" for lr in rows])
        sims = 0.4 * sims + 0.6 * (ass * tr).sum(1)
    for lr, sim in zip(rows, sims):
        lr.similarity = float(sim) if lr.heard else 0.0
        text_conf = float(np.clip((lr.similarity - 0.15) / 0.45, 0, 1))
        lr.confidence = float(np.clip(0.55 * lr.confidence + 0.45 * text_conf, 0, 1))
