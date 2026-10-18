"""Recalage en ligne de commande : python tools/run_cli.py VIDEO ASS SORTIE.ass"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retime import media, pipeline
VID, ASS, OUT = sys.argv[1:4]
ps = pipeline.PipelineSettings()
info = media.probe(VID); st = info.default_audio()
t0 = time.time()
def prog(stage, frac, total):
    if frac in (0.0, 1.0): print(f"  [{total*100:5.1f}%] {stage} {'...' if frac == 0 else 'ok'} ({time.time()-t0:.1f}s)", flush=True)
ma = pipeline.analyze_media(VID, st.index, ps, progress=prog)
out = pipeline.run_retime(ma, ASS, ps, progress=prog)
pipeline.apply_result(out).save(OUT)
print(f"total {time.time()-t0:.1f}s", ma.timings, out.result.stats)
