"""Lecture des médias via ffmpeg/ffprobe : pistes audio, extraction PCM, keyframes."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .paths import NO_WINDOW, find_tool

JAPANESE_TAGS = ("jpn", "ja", "jap", "japanese")


@dataclass
class AudioStream:
    index: int          # index absolu du flux dans le conteneur
    order: int          # index parmi les pistes audio (a:N)
    codec: str
    channels: int
    sample_rate: int
    language: str
    title: str
    default: bool

    def label(self) -> str:
        parts = [f"#{self.order + 1}"]
        if self.language:
            parts.append(f"[{self.language}]")
        if self.title:
            parts.append(self.title)
        parts.append(f"{self.codec} {self.channels}ch {self.sample_rate // 1000}kHz")
        return " ".join(parts)

    @property
    def is_japanese(self) -> bool:
        lang = self.language.lower()
        return lang in JAPANESE_TAGS or "japan" in self.title.lower() or "日本" in self.title


@dataclass
class MediaInfo:
    path: str
    duration: float
    audio_streams: list[AudioStream] = field(default_factory=list)
    fps: Fraction | None = None
    has_video: bool = False

    def default_audio(self) -> AudioStream | None:
        if not self.audio_streams:
            return None
        for s in self.audio_streams:
            if s.is_japanese:
                return s
        for s in self.audio_streams:
            if s.default:
                return s
        return self.audio_streams[0]


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW, **kw)


def probe(path: str) -> MediaInfo:
    cmd = [find_tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe a échoué : {proc.stderr.decode(errors='replace')[:500]}")
    data = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    fmt = data.get("format", {})
    info = MediaInfo(path=path, duration=float(fmt.get("duration") or 0.0))
    order = 0
    for st in data.get("streams", []):
        ctype = st.get("codec_type")
        tags = {k.lower(): v for k, v in (st.get("tags") or {}).items()}
        if ctype == "audio":
            info.audio_streams.append(AudioStream(
                index=int(st["index"]),
                order=order,
                codec=st.get("codec_name", "?"),
                channels=int(st.get("channels") or 0),
                sample_rate=int(st.get("sample_rate") or 0),
                language=tags.get("language", ""),
                title=tags.get("title", ""),
                default=bool((st.get("disposition") or {}).get("default")),
            ))
            order += 1
            if not info.duration and st.get("duration"):
                info.duration = float(st["duration"])
        elif ctype == "video" and not (st.get("disposition") or {}).get("attached_pic"):
            if not info.has_video:
                info.has_video = True
                rate = st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/1"
                try:
                    fr = Fraction(rate)
                    if fr <= 0 or fr > 300:
                        fr = Fraction(st.get("r_frame_rate") or "0/1")
                    info.fps = fr if 0 < fr <= 300 else None
                except (ValueError, ZeroDivisionError):
                    info.fps = None
    return info


def extract_audio(path: str, stream_index: int, sr: int = 16000, channels: int = 1,
                  start: float | None = None, duration: float | None = None) -> np.ndarray:
    """Décode une piste en float32 (n,) si mono, (n, channels) sinon."""
    cmd = [find_tool("ffmpeg"), "-nostdin", "-hide_banner", "-v", "error", "-threads", "0"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, start):.3f}"]
    cmd += ["-i", path]
    if duration is not None:
        cmd += ["-t", f"{max(0.01, duration):.3f}"]
    cmd += ["-map", f"0:{stream_index}", "-vn", "-sn", "-dn", "-ac", str(channels), "-ar", str(sr),
            "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg a échoué : {proc.stderr.decode(errors='replace')[:500]}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if channels > 1:
        audio = audio[: len(audio) // channels * channels].reshape(-1, channels)
    return audio


def keyframes(path: str) -> np.ndarray:
    """Temps (s) des images clés du flux vidéo, lus dans les paquets (aucun décodage : très rapide)."""
    cmd = [find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0",
           "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", path]
    proc = _run(cmd)
    if proc.returncode != 0:
        return np.zeros(0)
    times = []
    for line in proc.stdout.decode(errors="replace").splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2 and "K" in parts[1]:
            try:
                times.append(float(parts[0]))
            except ValueError:
                pass
    return np.unique(np.asarray(times, dtype=np.float64))
