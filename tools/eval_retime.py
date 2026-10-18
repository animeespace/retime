"""Évalue le recalage automatique contre un retime manuel (vérité terrain).

usage : python tools/eval_retime.py VIDEO ASS_ORIGINAL ASS_RETIME_MANUEL [SORTIE.ass] [--set clé=valeur ...]
"""
from __future__ import annotations

import difflib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from retime import align, media, pipeline  # noqa: E402
from retime.assfile import AssDocument  # noqa: E402


def text_pairs(a: AssDocument, b: AssDocument) -> list[tuple[int, int]]:
    sm = difflib.SequenceMatcher(a=[e.plain for e in a.events], b=[e.plain for e in b.events], autojunk=False)
    out = []
    for blk in sm.get_matching_blocks():
        out += [(blk.a + k, blk.b + k) for k in range(blk.size)]
    return out


def report(name: str, ds: np.ndarray, de: np.ndarray) -> None:
    def q(x):
        ax = np.abs(x)
        return (f"MAE {ax.mean() * 1000:6.1f} ms | méd {np.median(ax) * 1000:5.1f} | biais {x.mean() * 1000:+6.1f} | "
                f"≤1f {np.mean(ax <= 0.042) * 100:5.1f}% | ≤2f {np.mean(ax <= 0.084) * 100:5.1f}% | "
                f">250ms {np.mean(ax > 0.25) * 100:4.1f}%")
    print(f"{name:10s} début : {q(ds)}")
    print(f"{'':10s} fin   : {q(de)}")


def main() -> None:
    args = [a for a in sys.argv[1:]]
    sets = {}
    while "--set" in args:
        i = args.index("--set")
        k, v = args[i + 1].split("=", 1)
        sets[k] = json.loads(v)
        del args[i:i + 2]
    video, ass_in, ass_gt = args[:3]
    out_path = args[3] if len(args) > 3 else None
    ps = pipeline.PipelineSettings()
    for k, v in sets.items():
        tgt = ps.retime if hasattr(ps.retime, k) else ps
        setattr(tgt, k, v)
    info = media.probe(video)
    st = info.default_audio()
    t0 = time.time()

    def prog(stage, frac, total):
        if frac in (0.0, 1.0):
            print(f"  [{total * 100:5.1f}%] {stage} {'...' if frac == 0 else 'ok'}", flush=True)

    ma = pipeline.analyze_media(video, st.index, ps, progress=prog)
    out = pipeline.run_retime(ma, ass_in, ps, progress=prog)
    print(f"Temps total : {time.time() - t0:.1f} s  {ma.timings}")
    print("stats", out.result.stats)
    doc = pipeline.apply_result(out)
    if out_path:
        doc.save(out_path)

    orig = AssDocument(ass_in)
    gt = AssDocument(ass_gt)
    pairs = text_pairs(orig, gt)
    lines = {lr.event_idx: lr for lr in out.result.lines}
    rows = []
    for i, j in pairs:
        lr = lines[i]
        if lr.policy != align.RETIME:
            continue
        g = gt.events[j]
        rows.append((i, orig.events[i].start - g.start, orig.events[i].end - g.end,
                     lr.new_start - g.start, lr.new_end - g.end, lr.status, lr.confidence, g, lr))
    o_s = np.array([r[1] for r in rows]); o_e = np.array([r[2] for r in rows])
    n_s = np.array([r[3] for r in rows]); n_e = np.array([r[4] for r in rows])
    print(f"\n{len(rows)} lignes de dialogue comparées au retime manuel")
    report("ORIGINAL", o_s, o_e)
    report("AUTO", n_s, n_e)
    by = {}
    for r in rows:
        by.setdefault(r[5], []).append(r)
    for k, v in by.items():
        print(f"  {k:10s} n={len(v):3d}  MAE début {np.mean([abs(x[3]) for x in v]) * 1000:6.1f}  "
              f"fin {np.mean([abs(x[4]) for x in v]) * 1000:6.1f}")
    print("\nPires lignes :")
    rows.sort(key=lambda r: -(abs(r[3]) + abs(r[4])))
    for r in rows[:30]:
        g, lr = r[7], r[8]
        print(f"#{r[0]:3d} {lr.status:12s} c={r[6]:.2f} s={lr.similarity:.2f} GT {g.start:8.2f}-{g.end:8.2f} AUTO "
              f"{lr.new_start:8.2f}-{lr.new_end:8.2f} ORIG {lr.old_start:8.2f}-{lr.old_end:8.2f}  "
              f"Δ {r[3] * 1000:+5.0f}/{r[4] * 1000:+5.0f}  {g.plain[:34]}  <{lr.heard[:24]}> {lr.translation[:30]}")


if __name__ == "__main__":
    main()
