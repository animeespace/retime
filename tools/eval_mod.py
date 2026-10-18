import sys, difflib, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from retime import media, pipeline, analysis as an, align, asr, semantic
from retime.assfile import AssDocument
VID, ASS, GT = sys.argv[1:4]
ps = pipeline.PipelineSettings()
info = media.probe(VID); st = info.default_audio()
ma = pipeline.analyze_media(VID, st.index, ps, log=lambda *a: None)
sent = asr.build_sentences(ma.segments, ma.features); chars = asr.all_chars(ma.segments)
emb = semantic.get_embedder()
orig, gt = AssDocument(ASS), AssDocument(GT)
sm = difflib.SequenceMatcher(a=[e.plain for e in orig.events], b=[e.plain for e in gt.events], autojunk=False)
pairs = [(b.a + k, b.b + k) for b in sm.get_matching_blocks() for k in range(b.size)]
def fmt(a):
    a = np.abs(a); return f"n={len(a):3d} MAE {a.mean()*1000:5.1f} méd {np.median(a)*1000:4.0f} ≤1f {np.mean(a<=0.042)*100:4.1f}% ≤2f {np.mean(a<=0.084)*100:4.1f}% >150 {np.mean(a>0.15)*100:4.1f}%"
def run(**kw):
    rt = align.RetimeSettings(**kw)
    res = align.retime(AssDocument(ASS), sent, chars, ma.features, emb, rt, info.fps, ma.keyframes)
    L = {lr.event_idx: lr for lr in res.lines}
    out = {"ms": [], "me": [], "us": [], "ue": []}
    for i, j in pairs:
        lr = L[i]
        if lr.policy != align.RETIME: continue
        o, g = orig.events[i], gt.events[j]
        out["ms" if abs(o.start - g.start) > 0.005 else "us"].append(lr.new_start - g.start)
        out["me" if abs(o.end - g.end) > 0.005 else "ue"].append(lr.new_end - g.end)
    return out
for kw in (dict(), dict(lead_out=0.10), dict(lead_in=0.02, lead_out=0.10), dict(keep_tolerance=0.045), dict(lead_out=0.10, keep_tolerance=0.045)):
    o = run(**kw)
    print(kw)
    print("  débuts corrigés par toi  ", fmt(o["ms"]))
    print("  fins corrigées par toi   ", fmt(o["me"]))
    print("  débuts gardés par toi    ", fmt(o["us"]))
    print("  fins gardées par toi     ", fmt(o["ue"]))
print("---- signé (auto - GT), défaut")
o = run()
for k, lbl in (("ms", "débuts corrigés"), ("us", "débuts gardés"), ("me", "fins corrigées"), ("ue", "fins gardées")):
    a = np.array(o[k])
    print(f"{lbl:16s} méd {np.median(a)*1000:+5.0f}  hist", np.histogram(a * 1000, bins=[-1000, -150, -100, -60, -40, -20, 0, 20, 40, 60, 100, 150, 1000])[0])
