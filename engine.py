"""
engine.py - verse -> interpretation audio builder (LangGraph + Gemini)

Per (surah, language) the graph is:

    prepare -> listen -> check --ok--------------------> assemble -> END
                 ^          |
                 |          +--bad map, retries left--> (back to listen, with the exact complaint)
                 |          +--bad map, no retries ----> END   (reported as "needs review", never silently skipped)

Gemini LISTENS to the interpretation audio and returns, for every verse, where its
explanation starts and ends.  No docx colours, no line counting, no "drop the first N".
"""
import base64
import io
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, TypedDict

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pydub import AudioSegment

# --------------------------------------------------------------------------- #
# Surah data                                                                   #
# --------------------------------------------------------------------------- #
_NAMES = [
    "الفاتحة", "البقرة", "آل عمران", "النساء", "المائدة", "الأنعام", "الأعراف", "الأنفال",
    "التوبة", "يونس", "هود", "يوسف", "الرعد", "إبراهيم", "الحجر", "النحل", "الإسراء",
    "الكهف", "مريم", "طه", "الأنبياء", "الحج", "المؤمنون", "النور", "الفرقان", "الشعراء",
    "النمل", "القصص", "العنكبوت", "الروم", "لقمان", "السجدة", "الأحزاب", "سبأ", "فاطر",
    "يس", "الصافات", "ص", "الزمر", "غافر", "فصلت", "الشورى", "الزخرف", "الدخان",
    "الجاثية", "الأحقاف", "محمد", "الفتح", "الحجرات", "ق", "الذاريات", "الطور", "النجم",
    "القمر", "الرحمن", "الواقعة", "الحديد", "المجادلة", "الحشر", "الممتحنة", "الصف",
    "الجمعة", "المنافقون", "التغابن", "الطلاق", "التحريم", "الملك", "القلم", "الحاقة",
    "المعارج", "نوح", "الجن", "المزمل", "المدثر", "القيامة", "الإنسان", "المرسلات",
    "النبأ", "النازعات", "عبس", "التكوير", "الانفطار", "المطففين", "الانشقاق", "البروج",
    "الطارق", "الأعلى", "الغاشية", "الفجر", "البلد", "الشمس", "الليل", "الضحى", "الشرح",
    "التين", "العلق", "القدر", "البينة", "الزلزلة", "العاديات", "القارعة", "التكاثر",
    "العصر", "الهمزة", "الفيل", "قريش", "الماعون", "الكوثر", "الكافرون", "النصر", "المسد",
    "الإخلاص", "الفلق", "الناس",
]
_VERSES = [
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99, 128, 111, 110, 98,
    135, 112, 78, 118, 64, 77, 227, 93, 88, 69, 60, 34, 30, 73, 54, 45, 83, 182, 88, 75, 85,
    54, 53, 89, 59, 37, 35, 38, 29, 18, 45, 60, 49, 62, 55, 78, 96, 29, 22, 24, 13, 14, 11,
    11, 18, 12, 12, 30, 52, 52, 44, 28, 28, 20, 56, 40, 31, 50, 40, 46, 42, 29, 19, 36, 25,
    22, 17, 19, 26, 30, 20, 15, 21, 11, 8, 8, 19, 5, 8, 8, 11, 11, 8, 3, 9, 5, 4, 7, 3, 6,
    3, 5, 4, 5, 6,
]
assert len(_NAMES) == 114 and sum(_VERSES) == 6236
SURAHS = list(zip(_NAMES, _VERSES))  # index 0 == surah 1

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".wma"}
TEXT_EXTS = {".txt", ".md", ".srt", ".vtt", ".docx"}
RECITATION_HINTS = ("تلاوه", "قران", "recit", "quran", "tilawa")


# --------------------------------------------------------------------------- #
# Settings (filled by the app)                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    api_key: str = ""
    model: str = "gemini-2.5-flash"
    output_dir: Path = Path("output")        # one sub-folder per surah
    recitations_dir: Path = None             # full-surah recitation files
    final_dir: Path = Path("final")
    playlist_url: str = ""                   # YouTube playlist with the recitations (optional)
    download_only: bool = False
    docx_dir: Path = None                    # optional: .docx tafsir used as reference text
    languages: tuple = ()                    # empty = every language found in the file names
    surah_from: int = 1                      # range of surahs to process (inclusive)
    surah_to: int = 114
    basmala: str = "keep"                    # keep / drop / none
    fmt: str = "mp3"
    bitrate: str = "192k"
    gap_after_verse: int = 400
    gap_after_tafsir: int = 900
    pad: int = 150                           # ms of natural silence kept around clips
    edge_pad: int = 250                      # ms kept before verse 1 / after the last verse
    snap_window: int = 1200                  # ms a cut may move to land in a real pause
    min_silence: int = 900                   # recitation pause length (ms)
    max_intro: float = 25.0                  # s: longer than this = verse 1 was swallowed
    retries: int = 3                         # extra Gemini attempts per file
    overwrite: bool = False                  # rebuild finished files
    relisten: bool = False                   # ignore saved timing maps
    dry_run: bool = False
    log: Callable = print
    progress: Callable = lambda done, total: None
    stop: threading.Event = field(default_factory=threading.Event)


# --------------------------------------------------------------------------- #
# Name matching                                                                #
# --------------------------------------------------------------------------- #
_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه",
                          "ؤ": "و", "ئ": "ي", "ء": ""})


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _TASHKEEL.sub("", text).translate(_LETTERS).lower()
    return re.sub(r"[^0-9a-z\u0600-\u06ff]+", " ", text).strip()


def _key(text: str) -> str:
    t = fold(text).replace(" ", "")
    return t[2:] if t.startswith("ال") and len(t) > 3 else t


_NAME_KEYS = {_key(n): i + 1 for i, (n, _) in enumerate(SURAHS)}
_NOISE = {"سوره", "سورت", "surah", "sura", "surat"}


def surah_of(name: str):
    """Surah number for a folder/file name ('سورة النبأ', '078', 'تلاوة النبأ'...), or None."""
    f = fold(name)
    words = [w for w in re.sub(r"\d+", " ", f).split() if w not in _NOISE]
    nums = [int(x) for x in re.findall(r"\d+", f)]
    if words:
        exact = _NAME_KEYS.get(_key("".join(words)))
        if exact:
            return exact
        named = {_NAME_KEYS.get(_key("".join(words[i:j])))
                 for i in range(len(words)) for j in (i + 1, i + 2)} - {None}
        return next(iter(named)) if len(named) == 1 else None
    return nums[0] if nums and 1 <= nums[0] <= 114 else None


_LANG_RE = re.compile(r"^\s*([^\W\d_][^\d_]*?)\s*_\s*\d{1,3}\b")      # Spanish_114 سورة الناس_Tafseer
_LANG_RE2 = re.compile(r"^\s*([^\W\d_][^\d_]*?)\s*_.*taf[sz]?ee?r", re.I)   # Spanish_Tafseer


def detect_language(stem: str):
    """Language / dialect = the word before the first '_' in '<Language>_<no> سورة <name>_Tafseer'.
    Any name works (Spanish, Saudi, Persian, Moroccan...), so nothing is hard-coded."""
    if any(h in fold(stem) for h in RECITATION_HINTS):
        return None
    m = _LANG_RE.match(stem) or _LANG_RE2.match(stem)
    return m.group(1).strip().title() if m else None


def scan_languages(output_dir, lo=1, hi=114):
    """{language: number of surahs it appears in} across all surah folders."""
    found = {}
    for d in Path(output_dir).iterdir():
        if not d.is_dir() or not (n := surah_of(d.name)) or not lo <= n <= hi:
            continue
        for lang in {detect_language(p.stem) for p in _files(d, AUDIO_EXTS)} - {None}:
            found[lang] = found.get(lang, 0) + 1
    return dict(sorted(found.items()))


def _files(root: Path, exts):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix.lower() in exts)


def fmt_ms(ms) -> str:
    s, ms = divmod(int(ms), 1000)
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}.{ms:03d}"


# --------------------------------------------------------------------------- #
# Audio analysis (only used for the Arabic recitation and to snap cuts)        #
# --------------------------------------------------------------------------- #
class SplitError(Exception):
    pass


def _frame_db(audio) -> np.ndarray:
    mono = audio.set_channels(1).set_frame_rate(16000)
    x = np.asarray(mono.get_array_of_samples(), dtype=np.float32)
    n = len(x) // 160                                   # 10 ms frames
    if n == 0:
        return np.zeros(0)
    rms = np.sqrt(np.mean(x[: n * 160].reshape(n, 160) ** 2, axis=1)) + 1e-9
    return 20 * np.log10(rms / np.percentile(rms, 99))


def _gaps(db, min_ms, rel_db):
    """Internal silent runs [(start_ms, end_ms)] at least min_ms long."""
    if len(db) == 0:
        return []
    d = np.diff(np.concatenate(([0], (db < -rel_db).astype(np.int8), [0])))
    return [(int(s) * 10, int(e) * 10) for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])
            if s != 0 and e != len(db) and (e - s) * 10 >= min_ms]


def silence_mids(audio, min_ms=200):
    return [(s + e) // 2 for s, e in _gaps(_frame_db(audio), min_ms, 36)]


def _find_gaps(db, need, start_ms):
    for min_ms in [start_ms] + [m for m in (700, 500, 350, 250, 180) if m < start_ms]:
        for rel in (42, 36, 30, 26):
            g = _gaps(db, min_ms, rel)
            if len(g) >= need:
                return g
    raise SplitError(f"recitation: could not find {need} pauses - is this the right file for this surah?")


def trim_edges(seg, keep_ms, rel_db=40):
    loud = np.where(_frame_db(seg) > -rel_db)[0]
    if len(loud) == 0:
        return seg
    a = max(0, int(loud[0]) * 10 - keep_ms)
    b = min(len(seg), (int(loud[-1]) + 1) * 10 + keep_ms)
    return seg[a:b].fade_in(8).fade_out(8)


def split_recitation(audio, verses, number, cfg):
    """The recitation has clear pauses: cut at the (verses-1) longest ones.
    The first pause (after the basmala) is never a verse end (surahs 1 and 9 are exempt)."""
    skip = cfg.basmala != "none" and number not in (1, 9)
    gaps = _find_gaps(_frame_db(audio), verses if skip else verses - 1, cfg.min_silence)
    head = 0
    if skip:
        head = (gaps[0][0] + gaps[0][1]) // 2
        gaps = gaps[1:]
    top = sorted(sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)[: verses - 1])
    cuts = [(s + e) // 2 for s, e in top]
    edges = ([0] if cfg.basmala == "keep" else [head]) + cuts + [len(audio)]
    return [trim_edges(audio[a:b], cfg.pad) for a, b in zip(edges, edges[1:])]


def concat(parts, tpl):
    raws = [p.set_frame_rate(tpl.frame_rate).set_channels(tpl.channels)
             .set_sample_width(tpl.sample_width).raw_data for p in parts]
    return tpl._spawn(b"".join(raws))


def snap(t, mids, window, direction=0):
    """Move t onto the nearest real pause. direction -1: only earlier, +1: only later."""
    c = [m for m in mids if abs(m - t) <= window and (direction == 0 or (m - t) * direction >= 0)]
    return min(c, key=lambda m: abs(m - t)) if c else t


# --------------------------------------------------------------------------- #
# Gemini "ears"                                                                #
# --------------------------------------------------------------------------- #
class VerseSpan(BaseModel):
    verse: int = Field(description="verse number, starting at 1")
    start: str = Field(description="MM:SS.mmm - when the explanation of this verse begins")
    end: str = Field(description="MM:SS.mmm - when the explanation of this verse ends")
    first_words: str = Field(description="the first ~6 words spoken, exactly as heard")


class TafsirMap(BaseModel):
    intro_heard: str = Field(description="what the opening announcement says (e.g. 'begin, Surah X'), or empty")
    verses: list[VerseSpan]
    outro_heard: str = Field(description="what the closing line says (e.g. 'end'), or empty")


SYSTEM = """You help an audio editor. You listen to ONE recording of a spoken interpretation (tafsir) of a Quran surah and mark where the explanation of each verse starts and ends.

The recording is structured like this:
 1. a short opening announcement (for example the word 'begin' and the surah's title). It is NOT part of any verse.
 2. the explanation of verse 1, then verse 2, ... up to verse {n}, in order.
 3. a short closing line (for example 'end'). It is NOT part of any verse.

Rules:
- Return EXACTLY {n} entries, numbered 1..{n}, in order, without overlaps.
- The explanation of verse 1 begins right after the opening announcement. Never count it as part of the announcement and never skip it, even if the two are close together.
- One verse's explanation continues through pauses, examples, stories and repeated quotations of the ayah. Start a new entry only when the speaker moves on to the NEXT verse. Never split one verse in two and never merge two verses.
- Times are MM:SS.mmm counted from the very beginning of the audio. start = first spoken word of that explanation, end = its last spoken word."""


@lru_cache(maxsize=4)
def _llm(key, model):
    return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0) \
        .with_structured_output(TafsirMap)


def _audio_b64(audio) -> str:
    buf = io.BytesIO()
    audio.set_channels(1).set_frame_rate(16000).export(buf, format="mp3", bitrate="32k")
    data = buf.getvalue()
    if len(data) > 18 * 1024 * 1024:
        raise SplitError("interpretation audio is too long to send in one request (>18 MB)")
    return base64.b64encode(data).decode()


def _seconds(v) -> float:
    t = 0.0
    for part in str(v).strip().replace(",", ".").split(":"):
        t = t * 60 + float(part)
    return t


def read_reference(path: Path, limit=40000) -> str:
    """Written text of the interpretation. Only a HINT for Gemini - never used to count anything."""
    if path is None:
        return ""
    if path.suffix.lower() == ".docx":
        try:
            import docx
        except ImportError:
            return ""
        text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs if p.text.strip())
    else:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        text = re.sub(r"^.*-->.*$|^\d+$|^WEBVTT$", "", text, flags=re.M)
    return re.sub(r"\n\s*\n+", "\n", text).strip()[:limit]


# --------------------------------------------------------------------------- #
# The graph                                                                    #
# --------------------------------------------------------------------------- #
class State(TypedDict, total=False):
    cfg: Settings
    number: int
    name: str
    verses: int
    lang: str
    rec_path: Path
    audio_path: Path
    reference: str
    map_path: Path
    out_path: Path
    rec: Any
    rec_clips: list
    tafsir: Any
    tmap: Any
    spans: list
    attempt: int
    error: str
    output: str


def say(s, msg):
    s["cfg"].log(msg)


def prepare(s: State):
    cfg = s["cfg"]
    rec = AudioSegment.from_file(str(s["rec_path"]))
    clips = split_recitation(rec, s["verses"], s["number"], cfg)
    say(s, f"   recitation: {len(clips)} verse clips")
    return {"rec": rec, "rec_clips": clips, "attempt": 0, "error": "",
            "tafsir": AudioSegment.from_file(str(s["audio_path"]))}


def listen(s: State):
    cfg, mp, attempt = s["cfg"], s["map_path"], s["attempt"]
    if attempt == 0 and mp.exists() and not cfg.relisten:
        try:
            tm = TafsirMap.model_validate_json(mp.read_text("utf-8"))
            say(s, f"   [{s['lang']}] using saved timing map ({mp.name})")
            return {"tmap": tm, "attempt": 1}
        except Exception:
            pass
    if not cfg.api_key:
        raise SplitError("no Gemini API key")
    say(s, f"   [{s['lang']}] Gemini is listening (try {attempt + 1})...")
    text = f"Surah {s['name']}, {s['verses']} verses. Language or dialect of the recording: {s['lang']}."
    if s.get("reference"):
        text += ("\n\nReference text (the written version of what is spoken; it may also contain the "
                 "Arabic verses, titles or numbers). Use it ONLY to recognise where each verse's "
                 "explanation begins:\n" + s["reference"])
    if s.get("error"):
        text += f"\n\nYour previous answer was rejected: {s['error']}\nListen again and fix exactly that."
    tm = _llm(cfg.api_key, cfg.model).invoke([
        SystemMessage(SYSTEM.format(n=s["verses"])),
        HumanMessage(content=[{"type": "media", "mime_type": "audio/mpeg",
                               "data": _audio_b64(s["tafsir"])},
                              {"type": "text", "text": text}])])
    if tm is not None:
        mp.write_text(tm.model_dump_json(indent=2), "utf-8")   # hand-editable, reused next run
    return {"tmap": tm, "attempt": attempt + 1}


def check(s: State):
    """The guard that replaces the old 'verse count exceeds' crash: the map is validated, and if
    it is wrong Gemini gets the exact complaint and tries again."""
    cfg, tm, n = s["cfg"], s.get("tmap"), s["verses"]
    total = len(s["tafsir"]) / 1000
    if tm is None:
        return {"error": "the answer was not valid JSON for the requested schema."}
    v = sorted(tm.verses, key=lambda x: x.verse)
    if [x.verse for x in v] != list(range(1, n + 1)):
        return {"error": f"you returned {len(v)} entries; I need exactly {n}, numbered 1..{n}, "
                         f"one per verse (do not add entries for the announcement or the closing line)."}
    try:
        spans = [(_seconds(x.start), _seconds(x.end)) for x in v]
    except ValueError:
        return {"error": "a time was not in MM:SS.mmm format."}
    for i, (a, b) in enumerate(spans, 1):
        if b - a < 0.8:
            return {"error": f"verse {i} lasts only {b - a:.1f}s - too short to be a real explanation."}
        if i > 1 and a < spans[i - 2][1] - 0.3:
            return {"error": f"verse {i} starts before verse {i - 1} ends."}
    if spans[0][0] > cfg.max_intro:
        return {"error": f"verse 1 starts at {spans[0][0]:.0f}s, but the opening announcement is only "
                         f"a few seconds long. Verse 1's explanation was probably swallowed into the intro."}
    if spans[-1][1] > total + 1:
        return {"error": f"verse {n} ends at {spans[-1][1]:.0f}s but the audio is only {total:.0f}s long."}
    return {"error": "", "spans": spans}


def route(s: State):
    if not s["error"]:
        return "assemble"
    say(s, f"   ! [{s['lang']}] map rejected: {s['error']}")
    return "listen" if s["attempt"] <= s["cfg"].retries else "fail"


def assemble(s: State):
    cfg, a, spans, tm = s["cfg"], s["tafsir"], s["spans"], s["tmap"]
    total, lang = len(a), s["lang"]
    ms = [(int(x * 1000), int(y * 1000)) for x, y in spans]
    mids = silence_mids(a) if cfg.snap_window > 0 else []
    # start may only move EARLIER and end only LATER, so a word of verse 1 can never be cut off
    b0 = max(0, snap(ms[0][0] - cfg.edge_pad, mids, cfg.snap_window, -1))
    b1 = min(total, snap(ms[-1][1] + cfg.edge_pad, mids, cfg.snap_window, +1))
    cuts = []
    for (_, e1), (s2, _) in zip(ms[:-1], ms[1:]):
        c = snap((e1 + s2) // 2, mids, cfg.snap_window)
        cuts.append(max((cuts[-1] if cuts else b0) + 300, min(c, b1 - 300)))

    say(s, f"   [{lang}] removed intro ({b0 / 1000:.1f}s): \"{tm.intro_heard}\"")
    say(s, f"   [{lang}] removed outro ({(total - b1) / 1000:.1f}s): \"{tm.outro_heard}\"")
    for v in sorted(tm.verses, key=lambda x: x.verse)[:2] + [max(tm.verses, key=lambda x: x.verse)]:
        say(s, f"   [{lang}] verse {v.verse:>3} starts: \"{v.first_words}\"")
    if cfg.dry_run:
        say(s, f"   [{lang}] dry run: {len(ms)} verses, {fmt_ms(b0)} -> {fmt_ms(b1)}")
        return {"output": "dry run"}

    edges = [b0] + cuts + [b1]
    clips = [trim_edges(a[x:y], cfg.pad) for x, y in zip(edges, edges[1:])]
    rate = s["rec"].frame_rate
    pv = AudioSegment.silent(cfg.gap_after_verse, frame_rate=rate)
    pt = AudioSegment.silent(cfg.gap_after_tafsir, frame_rate=rate)
    parts = []
    for verse, tafsir in zip(s["rec_clips"], clips):
        parts += [verse, pv, tafsir, pt]
    out = concat(parts, s["rec"])
    kw = {"bitrate": cfg.bitrate} if cfg.fmt == "mp3" else {}
    out.export(str(s["out_path"]), format=cfg.fmt, **kw)
    say(s, f"   saved -> {s['out_path'].name}  ({fmt_ms(len(out))})")
    return {"output": str(s["out_path"])}


def build_graph():
    g = StateGraph(State)
    for name, fn in (("prepare", prepare), ("listen", listen), ("check", check), ("assemble", assemble)):
        g.add_node(name, fn)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "listen")
    g.add_edge("listen", "check")
    g.add_conditional_edges("check", route, {"listen": "listen", "assemble": "assemble", "fail": END})
    g.add_edge("assemble", END)
    return g.compile()


# --------------------------------------------------------------------------- #
# YouTube playlist -> recitation files                                         #
# --------------------------------------------------------------------------- #
class _YtLog:
    def __init__(self, log):
        self.log = log

    def debug(self, m):
        pass

    info = debug

    def warning(self, m):
        self.log(f"   ! {m}")

    def error(self, m):
        self.log(f"   ✗ {m}")


def download_recitations(cfg):
    """Download every video of the playlist as mp3 into cfg.recitations_dir (already-downloaded
    videos are skipped through .downloaded.txt in that folder), then report which files could
    not be matched to a surah."""
    try:
        import yt_dlp
    except ImportError:
        raise SplitError("downloading needs yt-dlp:  pip install -U yt-dlp")
    dest = Path(cfg.recitations_dir)
    dest.mkdir(parents=True, exist_ok=True)
    cfg.log(f"Downloading recitations into {dest} ...")

    def hook(d):
        if d.get("status") == "finished":
            cfg.log(f"   downloaded: {Path(d['filename']).name}")

    opts = {"format": "bestaudio/best", "quiet": True, "no_warnings": True, "ignoreerrors": True,
            "outtmpl": str(dest / "%(playlist_index)s - %(title)s.%(ext)s"),
            "windowsfilenames": True, "download_archive": str(dest / ".downloaded.txt"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3",
                                "preferredquality": "192"}],
            "logger": _YtLog(cfg.log), "progress_hooks": [hook]}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([cfg.playlist_url.strip()])
    files = _files(dest, AUDIO_EXTS)
    unknown = [p.name for p in files if surah_of(p.stem) is None]
    cfg.log(f"Recitations folder now has {len(files)} audio file(s).")
    for n in unknown:
        cfg.log(f"   ! cannot tell which surah this is (no surah name in the title): {n}")


# --------------------------------------------------------------------------- #
# Finding files + running everything                                           #
# --------------------------------------------------------------------------- #
def find_recitation(cfg, number, folder):
    if cfg.recitations_dir:
        hits = [p for p in _files(cfg.recitations_dir, AUDIO_EXTS)
                if surah_of(p.stem) == number or (surah_of(p.stem) is None and surah_of(p.parent.name) == number)]
        if hits:
            return hits[0]
    hint = [p for p in _files(folder, AUDIO_EXTS) if any(h in fold(p.stem) for h in RECITATION_HINTS)]
    return hint[0] if hint else None


def find_reference(cfg, number, folder, lang):
    texts = _files(folder, TEXT_EXTS)
    for p in texts:
        if detect_language(p.stem) == lang:
            return p
    if cfg.docx_dir and lang.lower() == "egyptian" and Path(cfg.docx_dir).is_dir():
        for p in _files(cfg.docx_dir, {".docx"}):
            if surah_of(p.stem) == number:
                return p
    return None


def discover(cfg):
    jobs = []
    for d in sorted(p for p in Path(cfg.output_dir).iterdir() if p.is_dir()):
        num = surah_of(d.name)
        if not num or not cfg.surah_from <= num <= cfg.surah_to or d.resolve() == Path(cfg.final_dir).resolve():
            continue
        rec = find_recitation(cfg, num, d)
        by_lang = {}
        for p in _files(d, AUDIO_EXTS):
            lang = detect_language(p.stem)
            if lang and p != rec and (not cfg.languages or lang in cfg.languages):
                by_lang.setdefault(lang, []).append(p)
        for lang, auds in by_lang.items():
            if len(auds) > 1:
                cfg.log(f"! [{num:03d}] {lang}: {len(auds)} audio files, using {auds[0].name}")
            jobs.append((num, lang, d, rec, auds[0]))
    return sorted(jobs, key=lambda j: (j[0], j[1]))


def run_batch(cfg: Settings):
    cfg = cfg
    if cfg.playlist_url.strip():
        try:
            download_recitations(cfg)
        except Exception as e:
            cfg.log(f"✗ download failed: {e}")
            if cfg.download_only:
                return []
    if cfg.download_only:
        return []
    Path(cfg.final_dir).mkdir(parents=True, exist_ok=True)
    graph, jobs, results = build_graph(), discover(cfg), []
    cfg.log(f"{len(jobs)} job(s) found")
    for i, (num, lang, folder, rec, audio) in enumerate(jobs, 1):
        if cfg.stop.is_set():
            cfg.log("stopped")
            break
        name, verses = SURAHS[num - 1]
        label = f"[{num:03d}] {name} / {lang}"
        base = Path(cfg.final_dir) / f"{num:03d}_{name}_{lang}"
        out = base.with_suffix("." + cfg.fmt)
        cfg.log(f"\n=== {label}  ({verses} verses)")
        try:
            if out.exists() and not (cfg.overwrite or cfg.dry_run):
                status = "skipped (already built - tick 'Overwrite' to rebuild)"
            elif rec is None:
                status = "no recitation file found"
            else:
                st = graph.invoke({
                    "cfg": cfg, "number": num, "name": name, "verses": verses, "lang": lang,
                    "rec_path": rec, "audio_path": audio, "out_path": out,
                    "map_path": Path(str(base) + ".map.json"),
                    "reference": read_reference(find_reference(cfg, num, folder, lang))})
                status = "ok" if st.get("output") else f"NEEDS REVIEW: {st.get('error')}"
        except Exception as e:      # one bad file never stops the batch
            status = f"error: {e}"
        cfg.log(f"   -> {status}")
        results.append((label, status))
        cfg.progress(i, len(jobs))
    cfg.log("\n=========== SUMMARY ===========")
    for label, status in results:
        cfg.log(f"{label:<32} {status}")
    return results
