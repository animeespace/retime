"""Lecture/écriture .ass sans perte : seules les colonnes Start/End des événements modifiés changent."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TAG_RE = re.compile(r"\{[^}]*\}")
POS_RE = re.compile(r"\\(pos|move|org|clip|iclip)\s*\(")
KARAOKE_RE = re.compile(r"\\[kK][fo]?\d")
DRAWING_RE = re.compile(r"\\p[1-9]")


def parse_time(s: str) -> float:
    h, m, rest = s.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def format_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def plain_text(text: str) -> str:
    t = TAG_RE.sub("", text)
    t = t.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class AssEvent:
    idx: int            # position parmi les événements
    line_no: int        # numéro de ligne dans le fichier
    kind: str           # Dialogue / Comment
    fields: list[str]
    start: float
    end: float
    style: str
    name: str
    text: str

    @property
    def plain(self) -> str:
        return plain_text(self.text)

    @property
    def is_comment(self) -> bool:
        return self.kind.lower() == "comment"

    @property
    def is_positioned(self) -> bool:
        return bool(POS_RE.search(self.text))

    @property
    def is_karaoke(self) -> bool:
        return bool(KARAOKE_RE.search(self.text))

    @property
    def is_drawing(self) -> bool:
        return bool(DRAWING_RE.search(self.text))


class AssDocument:
    def __init__(self, path: str):
        self.path = path
        raw = Path(path).read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            self.encoding = "utf-8-sig"
        elif raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            self.encoding = "utf-16"
        else:
            try:
                raw.decode("utf-8")
                self.encoding = "utf-8"
            except UnicodeDecodeError:
                self.encoding = "cp1252"
        text = raw.decode(self.encoding)
        self.newline = "\r\n" if "\r\n" in text else "\n"
        self.trailing_newline = text.endswith(("\n", "\r"))
        self.lines: list[str] = text.splitlines()
        self.styles: list[str] = []
        self.events: list[AssEvent] = []
        self.event_format: list[str] = ["Layer", "Start", "End", "Style", "Name", "MarginL", "MarginR",
                                         "MarginV", "Effect", "Text"]
        self.script_info: dict[str, str] = {}
        self._parse()

    def _parse(self) -> None:
        section = ""
        for i, line in enumerate(self.lines):
            s = line.strip()
            if s.startswith("[") and s.endswith("]"):
                section = s.lower()
                continue
            if section == "[script info]" and ":" in s and not s.startswith(";"):
                k, v = s.split(":", 1)
                self.script_info[k.strip()] = v.strip()
            elif section in ("[v4+ styles]", "[v4 styles]", "[v4+ styles ]") and s.lower().startswith("style:"):
                self.styles.append(s.split(":", 1)[1].split(",", 1)[0].strip())
            elif section == "[events]":
                if s.lower().startswith("format:"):
                    self.event_format = [f.strip() for f in s.split(":", 1)[1].split(",")]
                    continue
                m = re.match(r"^(Dialogue|Comment)\s*:\s?(.*)$", line, re.IGNORECASE)
                if not m:
                    continue
                kind, body = m.group(1), m.group(2)
                fields = body.split(",", len(self.event_format) - 1)
                if len(fields) < len(self.event_format):
                    continue
                f = dict(zip([x.lower() for x in self.event_format], fields))
                try:
                    start, end = parse_time(f["start"]), parse_time(f["end"])
                except (KeyError, ValueError):
                    continue
                self.events.append(AssEvent(
                    idx=len(self.events), line_no=i, kind=kind, fields=fields, start=start, end=end,
                    style=f.get("style", "Default").strip(), name=f.get("name", "").strip(),
                    text=f.get("text", ""),
                ))

    def set_times(self, ev: AssEvent, start: float, end: float) -> None:
        names = [x.lower() for x in self.event_format]
        fields = list(ev.fields)
        fields[names.index("start")] = format_time(start)
        fields[names.index("end")] = format_time(end)
        ev.fields = fields
        ev.start, ev.end = round(start, 2), round(end, 2)
        self.lines[ev.line_no] = f"{ev.kind}: " + ",".join(fields)

    def save(self, path: str) -> None:
        text = self.newline.join(self.lines)
        if self.trailing_newline:
            text += self.newline
        Path(path).write_bytes(text.encode(self.encoding))


def write_simple_ass(path: str, items: list[tuple[float, float, str]], title: str = "Retime JP") -> None:
    """Écrit un .ass minimal (sous-titres japonais générés)."""
    head = [
        "[Script Info]", f"Title: {title}", "ScriptType: v4.00+", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "PlayResX: 1920", "PlayResY: 1080", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Yu Gothic UI,64,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,1,"
        "2,60,60,60,128",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    body = [f"Dialogue: 0,{format_time(s)},{format_time(e)},Default,,0,0,0,,{t}" for s, e, t in items]
    Path(path).write_text("\r\n".join(head + body) + "\r\n", encoding="utf-8-sig")
