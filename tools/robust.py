"""Robustesse : .ass décalés artificiellement (global, coupure, dérive de framerate) -> recalage vs vérité terrain."""
import sys, difflib, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from retime import media, pipeline, analysis as an, align, asr, semantic
from retime.assfile import AssDocument
VID, ASS, GT, TMP = sys.argv[1:5]
ps = pipeline.PipelineSettings()
info = media.probe(VID); st = info.default_audio()
ma = pipeline.analyze_media(VID, st.index, ps, log=lambda *a: None)
sent = asr.build_sentences(ma.segments, ma.features); chars = asr.all_chars(ma.segments)
emb = semantic.get_embedder()
gt = AssDocument(GT)
def variant(name, fn):
    d = AssDocument(ASS)
    for e in d.events:
        s, t = fn(e.start), fn(e.end)
        d.set_times(e, max(0, s), max(0, t))
    p = os.path.join(TMP, f"var_{name}.ass"); d.save(p); return p
variants = {
    "original": lambda t: t,
    "+2.5s": lambda t: t + 2.5,
    "-7s": lambda t: t - 7.0,
    "coupure +4s à 12min": lambda t: t + (4.0 if t > 720 else 0.0),
    "dérive 25→23.976": lambda t: t * 25 / (24000 / 1001),
    "bruit ±0.3s": None,
}
rng = np.random.default_rng(0)
noise = {}
for name, fn in variants.items():
    if fn is None:
        fn = lambda t: t + noise.setdefault(round(t, 2), rng.uniform(-0.3, 0.3))
    p = variant(name, fn)
    orig = AssDocument(p)
    res = align.retime(orig, sent, chars, ma.features, emb, align.RetimeSettings(), info.fps, ma.keyframes)
    sm = difflib.SequenceMatcher(a=[e.plain for e in orig.events], b=[e.plain for e in gt.events], autojunk=False)
    pairs = [(b.a + k, b.b + k) for b in sm.get_matching_blocks() for k in range(b.size)]
    L = {lr.event_idx: lr for lr in res.lines}
    ds = np.array([L[i].new_start - gt.events[j].start for i, j in pairs if L[i].policy == align.RETIME])
    de = np.array([L[i].new_end - gt.events[j].end for i, j in pairs if L[i].policy == align.RETIME])
    os_ = np.array([orig.events[i].start - gt.events[j].start for i, j in pairs if L[i].policy == align.RETIME])
    a, b = np.abs(ds), np.abs(de)
    print(f"{name:22s} avant: MAE début {np.abs(os_).mean()*1000:7.0f} ms | après: début MAE {a.mean()*1000:5.0f} ≤2f {np.mean(a<=0.084)*100:4.1f}% >0.5s {np.mean(a>0.5)*100:4.1f}% | fin MAE {b.mean()*1000:5.0f} ≤2f {np.mean(b<=0.084)*100:4.1f}% >0.5s {np.mean(b>0.5)*100:4.1f}% | glob {res.global_offset:+.2f} ancres {res.stats['anchors']}")
