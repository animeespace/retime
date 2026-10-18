"""Mode ligne de commande : Retime.exe --cli VIDEO ASS SORTIE.ass [--log fichier.txt] [--track N]

Sans console (exe fenêtré), tout est écrit dans le journal (par défaut %LOCALAPPDATA%\\Retime\\retime.log)."""
from __future__ import annotations

import time
import traceback

from .paths import LOG_FILE


def main(args: list[str]) -> int:
    log_path = str(LOG_FILE)
    if "--log" in args:
        i = args.index("--log")
        log_path = args[i + 1]
        del args[i:i + 2]
    track = None
    if "--track" in args:
        i = args.index("--track")
        track = int(args[i + 1])
        del args[i:i + 2]
    fh = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        fh.write(msg + "\n")
        fh.flush()

    try:
        from . import media, pipeline

        video, ass, out = args[:3]
        t0 = time.time()
        log(f"=== Retime CLI : {video} + {ass}")
        info = media.probe(video)
        stream = info.audio_streams[track] if track is not None else info.default_audio()
        ps = pipeline.PipelineSettings()

        def prog(stage, frac, total):
            if frac in (0.0, 1.0):
                log(f"  [{total * 100:5.1f}%] {stage} {'...' if frac == 0 else 'ok'} ({time.time() - t0:.1f}s)")

        ma = pipeline.analyze_media(video, stream.index, ps, progress=prog, log=log)
        res = pipeline.run_retime(ma, ass, ps, progress=prog, log=log)
        pipeline.apply_result(res).save(out)
        log(f"OK en {time.time() - t0:.1f}s -> {out}  {res.result.stats}")
        return 0
    except Exception:  # noqa: BLE001
        log(traceback.format_exc())
        return 1
    finally:
        fh.close()
