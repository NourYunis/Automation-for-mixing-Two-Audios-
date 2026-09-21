"""
engine.py - verse -> interpretation audio builder (LangGraph + Gemini)

Per (surah, language) the graph is:

    prepare -> listen -> check --ok--> assemble -> verify --ok----------------> END (saved)
                 ^          |                         |
                 |          +--bad map, retries left-->|-- listen, exact complaint
                 |          +--bad map, no retries ----+   (never silently skipped)
                                                        |
                                    problem found, repair attempts left --> repair -> prepare (re-cut, re-check)
                                       repair first works out WHERE it went wrong: a recitation clip that is not its
                                       verse -> re-cut the recitation; otherwise re-listen only from the verse before
                                       the first mismatch. The next final check only re-listens to what changed.
                                    problem found, no attempts left     --> END (reported as "needs review")

Gemini LISTENS to the interpretation audio and returns, for every verse, where its
explanation starts and ends. It is then asked a SECOND time, per suspiciously long/short
recitation clip, to confirm the clip really is one complete verse rather than a bad cut. Once
the file is assembled, Gemini is asked a THIRD time - listening to the actual finished file -
to confirm every verse's recitation is immediately followed by its own matching interpretation
with no long dead air and nothing cut off; if not, the file is rebuilt with adjusted padding/
gaps and re-checked, up to `verify_retries` times, before being reported as needing review.
No docx colours, no line counting, no "drop the first N".

Timing source (Settings.timing_source): by default the timing no longer comes from a chat model that
listens and guesses timestamps. gemini-3.5-transcribe returns every word with its real start/end
offset (in <= 25-minute pieces, its word-timestamp limit is 30), and the chat model then only reads
the numbered TRANSCRIPT and says which word each verse starts at - text tokens instead of 32 tokens
per second of audio, and every time in the map is the transcriber's own. 'chat' restores the old
listening; 'auto' falls back to it if the transcription model cannot be used.

What costs what
---------------
Gemini bills audio by DURATION - 32 tokens per second, 1,920 per minute - not by file size, so
the 16 kHz/32 kbps downsampling below saves upload time and nothing else. Three things follow:

  * `listen` (the interpretation, once per language) is the irreducible cost. Everything it
    learns has to come from hearing the recording.
  * The final check used to re-send the WHOLE assembled file - recitation AND interpretation,
    per language, per repair pass - which made it the single most expensive step in the system,
    larger than `listen` itself. It asked three questions; two of them ('cutoff', 'gap') are
    about silences this program inserted itself, so they are now MEASURED (see local_defects)
    instead of being asked about. The third ('mismatch') needs ears, and gets only the two
    joins of every verse (see build_digest) rather than the whole file: about 26 s per verse, so
    the cost follows the number of verses, not the length of the commentary. Both joins matter: a
    boundary error shows up on BOTH sides of a cut, and an earlier digest that heard only one side
    could not see most of them. Settings.verify_mode = "full" restores the original behaviour.
  * A word of the neighbouring verse at a clip edge means a CUT is in the wrong place. Wider
    padding cannot fix that (it only keeps more silence inside a clip's window), so repair moves
    the cut itself, by whole pauses, in the direction the edge and kind imply (edge_move).
  * A rejected answer used to re-send the whole recording. A complaint about verse i is local,
    so it now re-listens from verse i-1 to the end through the same tail machinery that repair
    already used, and asks slightly warmer so the answer can actually differ.

Settings.usage records what every call cost; run_batch prints it. Nothing about spending was
observable before, which made every other decision here guesswork.

What is NOT done, and is the larger remaining lever
---------------------------------------------------
  * Batch API: half price on input and output, 24-hour turnaround. This whole pipeline is a
    batch job (114 surahs x N languages, nobody waiting), so the first `listen` pass over every
    job belongs there, with only the failures falling back to interactive calls.
  * Files API + explicit context caching: cache reads bill at 10% of the input rate, and the
    retry loop re-pays full price for identical audio today. It would also lift the 18 MB
    inline limit in _audio_b64, which is what makes a recitation over ~75 minutes fail outright.
    Note the message order below - media part first, text second - is what lets Gemini's
    implicit cache match a retry's prefix. Do not reorder it.

Speed
-----
The loudness analysis of a recording is computed ONCE (class Analysis) and sliced, instead of
being recomputed for every clip, every repair pass and every language: 2-4x on a single job and
8-12x for the second and later languages of the same surah. The pause search makes 4 passes over
the array where it used to make 24. Byte-identical results - see test_equiv.py.

Saving (measured, see README): the encode is single-threaded LAME at ~115x real time and is the
floor. concat() is ~0.2 s for an Al-Baqarah-sized file. stream_encode() pipes raw PCM into ffmpeg
instead of pydub's export: same speed, byte-identical mp3, but no 4 GiB WAV-header ceiling, no
temp files, progress + Stop, and an existing file survives a failed save. The interpretation is
converted to the recitation's format once and reused across repair passes.
"""
import base64
import io
import json
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field, replace
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
    cookies_browser: str = ""               # chrome / edge / firefox / brave, only if YouTube blocks
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
    verify_retries: int = 3                  # extra rebuild+recheck attempts if the final file review finds a problem
    boundary_tol: int = 400                  # ms either side of Gemini's end/start times inside which a verse cut may be placed
    repair_pad_step: int = 100               # ms added to clip padding per repair attempt (fixes cut-off words)
    repair_gap_shrink: float = 0.7           # gap_after_verse/gap_after_tafsir multiplier per repair attempt (fixes dead air)
    verify_clips: bool = True                # ask Gemini about outlier recitation clips (repair turns this off after the first pass so clips are not re-judged for nothing)
    suspect_margin: float = 1.0              # an unused pause counts as evidence at this multiple of the weakest pause that WAS accepted as a verse end; raise it to ask about fewer clips
    suspect_max: int = 10                    # clips worth asking Gemini about; more than this means the pauses do not describe the verses at all - re-cut instead of auditing
    map_verses: int = 40                     # verses marked per request when reading a transcript; a long surah is marked in windows rather than in one 286-index answer
    chat_listen_max_min: float = 45.0        # longest recording the CHAT model is sent in one request; the transcription path splits long audio by itself, the chat path cannot
    opening_check: bool = True               # hear what the recitation opens with (isti'adha / basmala / title) instead of assuming it from the surah number
    opening_max_ms: int = 20000              # most of one leading stretch that is worth sending; the wording is decided in its first seconds
    spacing_factor: float = 0.4              # recitation cuts may not sit closer than this fraction of the average verse length
    rec_align: str = "auto"                   # recitation cutting: auto = silences first, Gemini marks the verses when they fail | gemini = always Gemini | silence = never Gemini
    api_retries: int = 4                     # waits + re-asks for transient Gemini errors (429/500/503/timeouts); these do NOT use up `retries`
    check_chunk_min: float = 6.0             # minutes of finished audio per final-check request (smaller = cheaper re-checks)
    # --- final check: how much audio it costs ------------------------------------------------ #
    stream_save: bool = True                 # pipe raw PCM straight into ffmpeg instead of pydub's export (see stream_encode)
    mp3_quality: Any = None                  # LAME algorithm quality, 0 (best, slowest) .. 9 (fastest). None = ffmpeg's default
                                             # (5), i.e. output unchanged. 7 encodes ~1.3-1.5x faster for a slightly
                                             # lower-quality encode. NEVER 0: that is the slowest setting, ~4-5x slower.
    verify_mode: str = "digest"              # digest = listen to the verse->interpretation junctions only (default)
                                             # full   = listen to the whole finished file (the original behaviour)
                                             # local  = no audio at all, deterministic checks only
    digest_rec_ms: int = 4000                # ms of the END of each verse's recitation put in the digest
    digest_taf_ms: int = 12000               # ms of the START of each interpretation put in the digest
    digest_taf_tail_ms: int = 6000           # ms of the END of each interpretation put in the digest
    digest_rec_head_ms: int = 3000           # ms of the START of each recitation put in the digest
    digest_verses: int = 6                   # verses per digest request (two excerpts each)
    local_cutoff: bool = False               # act on measured mid-word cuts, not just log them
    retry_temperature: float = 0.3           # a rejected answer is re-asked slightly warmer, so the
                                             # model can actually produce a DIFFERENT answer
    # --- timing source: which model marks the verses ------------------------------------------- #
    timing_source: str = "auto"              # auto       = gemini-3.5-transcribe gives the words + times, the chat model marks the
                                             #              verses in the TEXT; falls back to listening if transcription is unusable
                                             # transcribe = the same, never falls back
                                             # chat       = the chat model listens to the audio and guesses the times (before)
    transcribe_model: str = "gemini-3.5-transcribe"
    transcribe_chunk_min: float = 25.0       # word timestamps are limited to 30 minutes of audio per request
    retranscribe: bool = False               # ignore saved transcripts (*.words.json) and transcribe again
    rec_language_code: str = ""              # BCP-47 hint for the recitation ("" = automatic detection)
    overwrite: bool = False                  # rebuild finished files
    relisten: bool = False                   # ignore saved timing maps
    dry_run: bool = False
    log: Callable = print
    progress: Callable = lambda done, total: None
    status: Callable = lambda msg: None      # one-line 'what is the app doing right now'
    stop: threading.Event = field(default_factory=threading.Event)
    usage: dict = field(default_factory=lambda: {"calls": 0, "input": 0, "output": 0,
                                                 "thinking": 0, "cached": 0, "by_step": {}})


# --------------------------------------------------------------------------- #
# Name matching                                                                #
# --------------------------------------------------------------------------- #
_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه",
                          "ؤ": "و", "ئ": "ي", "ء": ""})


@lru_cache(maxsize=8192)
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


@lru_cache(maxsize=8192)
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


@lru_cache(maxsize=8192)
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


@lru_cache(maxsize=256)
def _walk(root: str):
    """Every file under `root`, walked ONCE. find_recitation() used to re-walk the whole
    recitations folder for each of the 114 surahs, and find_reference() again for each job."""
    try:
        return tuple(sorted(p for p in Path(root).rglob("*") if p.is_file()))
    except OSError:
        return ()


def _files(root: Path, exts):
    return [p for p in _walk(str(root)) if p.suffix.lower() in exts]


def forget_files():
    """Drop the cached listings (after a download wrote new files, or before a fresh run)."""
    _walk.cache_clear()


def fmt_ms(ms) -> str:
    s, ms = divmod(int(ms), 1000)
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}.{ms:03d}"


# --------------------------------------------------------------------------- #
# Audio analysis (only used for the Arabic recitation and to snap cuts)        #
# --------------------------------------------------------------------------- #
class SplitError(Exception):
    pass


class Stopped(Exception):
    """The user pressed Stop."""


def check_stop(cfg):
    if cfg.stop.is_set():
        raise Stopped()


class Analysis:
    """The 10 ms loudness frames of ONE recording, computed once and then sliced.

    The old code called _frame_db() again for every clip (trim_edges), for every repair pass
    (silence_mids) and for every language of the same surah - each call re-converted the whole
    recording to 16 kHz mono and re-ran the RMS over it. For a 286-verse surah that is hundreds
    of full passes over the audio. Here the frame RMS is computed once; db(a, b) slices it and
    normalises to the 99th percentile OF THAT SLICE, which is exactly what _frame_db(audio[a:b])
    used to return (identical to within one 10 ms frame at the edges)."""
    __slots__ = ("rms", "total_ms", "ref", "_runs", "_gc")

    def __init__(self, audio):
        mono = audio.set_channels(1).set_frame_rate(16000)
        x = np.asarray(mono.get_array_of_samples(), dtype=np.float32)
        n = len(x) // 160                                   # 10 ms frames
        self.rms = (np.sqrt(np.mean(x[: n * 160].reshape(n, 160) ** 2, axis=1)) + 1e-9
                    if n else np.zeros(0))
        self.total_ms = len(audio)
        self.ref = float(np.percentile(self.rms, 99)) if len(self.rms) else 0.0
        self._runs = {}                                     # rel_db -> silent runs, computed once
        self._gc = {}                                       # (min_ms, rel_db) -> those runs, filtered by length

    def db(self, a_ms=0, b_ms=None) -> np.ndarray:
        a = max(0, int(a_ms) // 10)
        b = len(self.rms) if b_ms is None else min(len(self.rms), -(-int(b_ms) // 10))
        r = self.rms[a:b]
        if len(r) == 0:
            return np.zeros(0)
        return 20 * np.log10(r / np.percentile(r, 99))

    def level(self, a_ms=0, b_ms=None) -> float:
        """The loudest 10 ms of [a, b) in dB against the WHOLE recording's 99th percentile.
        db() normalises each slice against itself, which is right for finding pauses inside a
        clip but blind to a clip that holds no speech at all - room tone normalised against
        room tone looks like speech. This is the absolute yardstick that can tell them apart."""
        a = max(0, int(a_ms) // 10)
        b = len(self.rms) if b_ms is None else min(len(self.rms), -(-int(b_ms) // 10))
        r = self.rms[a:b]
        if len(r) == 0 or self.ref <= 0:
            return -120.0
        return float(20 * np.log10(float(r.max()) / self.ref))

    def runs(self, rel_db):
        """Internal silent runs [(start_ms, end_ms)] over the WHOLE recording, cached per
        threshold. _find_gaps used to recompute these for every (min_ms, rel_db) pair - 24
        passes over the array where 4 suffice, since min_ms only filters by length."""
        if rel_db not in self._runs:
            db = self.db()
            if len(db) == 0:
                self._runs[rel_db] = []
            else:
                d = np.diff(np.concatenate(([0], (db < -rel_db).astype(np.int8), [0])))
                self._runs[rel_db] = [
                    (int(s) * 10, int(e) * 10)
                    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])
                    if s != 0 and e != len(db)]
        return self._runs[rel_db]

    def gaps(self, min_ms, rel_db):
        """Silent runs at least min_ms long. The returned list is shared - do not modify it."""
        k = (min_ms, rel_db)
        if k not in self._gc:
            self._gc[k] = [g for g in self.runs(rel_db) if g[1] - g[0] >= min_ms]
        return self._gc[k]


def analyse(audio, cache=None, key=None) -> Analysis:
    """Analysis(audio), reusing a cached one when the caller has somewhere to keep it."""
    if cache is None or key is None:
        return Analysis(audio)
    if key not in cache:
        cache[key] = Analysis(audio)
    return cache[key]


def silence_mids(an: Analysis, min_ms=200):
    return [(s + e) // 2 for s, e in an.gaps(min_ms, 36)]


def _find_gaps(an: Analysis, need, start_ms):
    for min_ms in [start_ms] + [m for m in (700, 500, 350, 250, 180) if m < start_ms]:
        for rel in (42, 36, 30, 26):
            g = an.gaps(min_ms, rel)
            if len(g) >= need:
                return g
    raise SplitError(f"recitation: could not find {need} pauses - is this the right file for this surah?")


EDGE_TOUCH_MS = 20      # speech this close to a cut means the cut landed inside a word
SILENT_DB = -35.0       # a clip this far below the recording's own level holds no speech


def trim_window(an: Analysis, a, b, keep_ms, rel_db=40):
    """Where trim_edges would cut inside the window [a, b) of the parent recording, and how
    much silence it found at each end. lead/trail near zero means speech runs right up to the
    cut - i.e. a word was sliced in half. This is the 'cutoff' that the final check used to
    spend audio tokens asking Gemini about; here it is simply measured."""
    db = an.db(a, b)
    loud = np.where(db > -rel_db)[0]
    if len(loud) == 0:
        return a, b, None, None
    lead = int(loud[0]) * 10
    trail = (len(db) - 1 - int(loud[-1])) * 10
    x = max(a, a + lead - keep_ms)
    y = min(b, a + (int(loud[-1]) + 1) * 10 + keep_ms)
    return x, y, lead, trail


def trim_edges(audio, an: Analysis, a, b, keep_ms, rel_db=40):
    """The trimmed clip for window [a, b), plus the silence the clip actually KEEPS at each end
    (the trim never keeps more than keep_ms, so the reported value is clamped to it). Both uses
    stay correct: the junction silence is the sum of what was kept, and a value below
    EDGE_TOUCH_MS still means speech runs right up to the cut."""
    x, y, lead, trail = trim_window(an, a, b, keep_ms, rel_db)
    if lead is None:                                  # nothing audible - leave the window alone
        return audio[a:b], (None, None)
    return audio[x:y].fade_in(8).fade_out(8), (min(lead, keep_ms), min(trail, keep_ms))


def _spread_cuts(gaps, need, min_spacing, report=None):
    """Pick `need` cut points from `gaps`, preferring the longest, but never place two cuts
    closer together than `min_spacing`. Without this, a couple of extra breathing-pauses
    inside one verse (often longer than the quick, unpaused transition between two OTHER
    verses) can out-rank a real verse boundary: several cuts then bunch up inside one verse,
    an earlier boundary is never cut at all, several verses collapse into one clip, and every
    verse after that point is off by one against its interpretation."""
    candidates = sorted(gaps, key=lambda g: g[1] - g[0], reverse=True)
    chosen = []
    for s, e in candidates:
        mid = (s + e) // 2
        if all(abs(mid - c) >= min_spacing for c in chosen):
            chosen.append(mid)
        if len(chosen) == need:
            break
    if len(chosen) < need:            # not enough well-spaced gaps - fall back rather than crash
        # The fallback takes the longest pauses wherever they are, so the cuts can bunch up inside
        # one long verse and leave real boundaries uncut. That is a guess, not a reading: whoever
        # asked for a report is told, so the caller can hand the surah to Gemini instead.
        if report is not None:
            report["crowded"] = need - len(chosen)
        chosen = [(s + e) // 2 for s, e in candidates[:need]]
    return sorted(chosen)


class Clips:
    """The cut clips plus, per clip, how much silence sat before the first word and after the
    last one. A near-zero value means the cut landed inside a word."""
    __slots__ = ("segs", "edges", "lead", "trail", "level")

    def __init__(self, segs, edges, lead, trail, level):
        self.segs, self.edges, self.lead, self.trail = segs, edges, lead, trail
        self.level = level

    def __len__(self):
        return len(self.segs)

    def __iter__(self):
        return iter(self.segs)

    def __getitem__(self, i):
        return self.segs[i]


def cut_all(audio, an: Analysis, edges, pad):
    segs, lead, trail, level = [], [], [], []
    for a, b in zip(edges, edges[1:]):
        seg, (le, tr) = trim_edges(audio, an, a, b, pad)
        segs.append(seg)
        lead.append(le)
        trail.append(tr)
        level.append(an.level(a, b))
    return Clips(segs, list(edges), lead, trail, level)


def assumed_lead(number, cfg):
    """How many spoken formulas the SURAH implies before verse 1, knowing nothing about the
    recording: the basmala in every surah except Al-Fatiha (where the basmala IS verse 1) and
    At-Tawbah (which has none). An isti'adha is the reciter's own addition, not the surah's, so it
    is never assumed here - check_opening() is what hears whether this reciter said one."""
    return 1 if (cfg.basmala != "none" and number not in (1, 9)) else 0


def recitation_edges(an: Analysis, verses, number, cfg, lead=None):
    """The verse boundaries of a silence-cut recitation. Kept separate from the cutting because
    they do not depend on cfg.pad: a 'cutoff' repair only widens the padding, so it can re-cut
    from these instead of re-running the whole silence search.

    `lead` is how many stretches of speech come before verse 1 in THIS recording - an isti'adha, a
    basmala, a spoken title, or several of them. Each one ends in a pause, and a pause spent as a
    verse boundary shifts EVERY verse after it by one clip. That is exactly what an isti'adha
    before Al-Fatiha used to do: surahs 1 and 9 were hard-coded here as 'nothing precedes verse 1',
    so the pause after the isti'adha was spent as verse 1's end, verse 1's explanation was paired
    with the isti'adha, and the whole surah ran one clip late. None = fall back to what the surah
    implies (assumed_lead), which is the behaviour before anything has been heard."""
    lead = assumed_lead(number, cfg) if lead is None else max(0, int(lead))
    total = an.total_ms
    gaps = _find_gaps(an, verses - 1 + lead, cfg.min_silence)
    head = 0
    if lead:
        head = (gaps[lead - 1][0] + gaps[lead - 1][1]) // 2
        gaps = gaps[lead:]
    # expect verses to be roughly evenly paced; forbid two cuts closer than ~40% of that average,
    # so multiple genuine short verses in a row still get their own cuts.
    # The spacing rule assumes every verse is roughly one AVERAGE verse long. A surah that mixes a
    # one-word verse (الم, طه, حم) with a three-minute one (Al-Baqarah 282) breaks that assumption
    # in both directions at once: the floor vetoes the genuine pause after the one-word verse, and
    # the long verse's own breathing pauses are long enough to be mistaken for verse ends. The old
    # code answered a veto by taking "the longest pauses, wherever they are" - not a worse cut, an
    # arbitrary one. Relax the floor instead, one step at a time, and keep as much anti-clustering
    # as the recitation actually allows.
    min_spacing = max(1000, (total / verses) * cfg.spacing_factor)
    report, relaxed = {}, 0
    cuts = _spread_cuts(gaps, verses - 1, min_spacing, report)
    while report.get("crowded") and min_spacing > 1000:
        min_spacing, relaxed = max(1000, min_spacing * 0.6), relaxed + 1
        report.clear()
        cuts = _spread_cuts(gaps, verses - 1, min_spacing, report)
    if report.get("crowded"):
        raise SplitError(f"recitation: {report['crowded']} verse boundary/ies could not be placed even "
                         f"with the smallest spacing - the pauses of this recitation do not separate "
                         f"its verses")
    if relaxed:
        getattr(cfg, "log", lambda *_: None)(
            f"   recitation: verse lengths here are very uneven - the minimum spacing between cuts was "
            f"relaxed to {min_spacing / 1000:.1f}s so short verses keep their own boundary")
    return ([0] if cfg.basmala == "keep" else [head]) + cuts + [total]


def split_recitation(audio, an: Analysis, verses, number, cfg, edges=None):
    """The recitation has clear pauses: cut at the (verses-1) longest ones, spread across the
    whole recording so a burst of pauses inside one verse can't swallow a neighbouring boundary.
    The first pause (after the basmala) is never a verse end (surahs 1 and 9 are exempt)."""
    if edges is None:
        edges = recitation_edges(an, verses, number, cfg)
    clips = cut_all(audio, an, edges, cfg.pad)
    if cfg.verify_clips:
        _check_clip_lengths(clips, an, cfg, SURAHS[number - 1][0], verses)
    return clips


def suspect_clips(an: Analysis, edges, cfg):
    """Which clips of a silence cut are worth a second opinion, ranked worst first.

    Duration on its own is no evidence: a one-word verse next to a three-minute one is normal in
    the Quran, so `4x the median` flags almost nothing real and misses both failures that matter.
    The signal that IS evidence is the cutter's own behaviour. It accepted some pauses as verse
    ends; a clip that CONTAINS an unused pause at least as convincing as the weakest one it
    accepted means the boundary was rejected by the spacing rule, not by the audio - the signature
    of two verses merged into one clip.

    That single test covers the opposite failure too, because the cut count is fixed: every extra
    cut taken inside one long verse is a real boundary left uncut somewhere else, and that
    somewhere else is a merged clip with an unused pause in it. So a verse split into pieces is
    caught by the merge it forces elsewhere.

    Suspicion is not a verdict - only Gemini, which knows the text, can say whether a long clip is
    one long verse or two short ones. This just decides who to ask about."""
    gaps = an.gaps(cfg.min_silence, 36)
    mids = [((g[0] + g[1]) // 2, g[1] - g[0]) for g in gaps]

    def pause_at(t):
        return max((ln for m, ln in mids if abs(m - t) <= 80), default=0)

    used = [pause_at(t) for t in edges[1:-1]]
    floor = min([p for p in used if p] or [cfg.min_silence])
    durs = sorted(b - a for a, b in zip(edges, edges[1:]))
    med = durs[len(durs) // 2] or 1
    out = []
    for i, (a, b) in enumerate(zip(edges, edges[1:]), 1):
        # Length is deliberately NOT a condition. When cuts bunch inside one long verse, the clips
        # that swallowed two short verses are SHORTER than the median, not longer - the old
        # `4x the median` rule could not see them. What is always true is this: _spread_cuts takes
        # the longest pauses first, so any unused pause at least as long as the weakest one it
        # accepted was rejected by the spacing rule, not by the audio.
        best = max((ln for m, ln in mids if a + 500 <= m <= b - 500), default=0)
        if best >= floor * cfg.suspect_margin:
            out.append((best / max(floor, 1), i,
                        f"it runs {(b - a) / 1000:.1f}s (typical here: {med / 1000:.1f}s) and holds an "
                        f"unused {best / 1000:.1f}s pause, though boundaries with pauses as short as "
                        f"{floor / 1000:.1f}s were accepted elsewhere"))
    out.sort(reverse=True)
    return [(i, why) for _, i, why in out]


def _check_clip_lengths(clips, an: Analysis, cfg, surah_name, total_verses):
    """A suspect clip is never proof of a mistake, so Gemini - which knows the Quran text, unlike a
    silence detector - makes the actual call by listening to it. Only a clip Gemini itself does not
    confirm as one complete verse raises, and in `auto` that hands the whole surah to the
    verse-by-verse alignment instead of saving a misaligned file silently."""
    suspects = suspect_clips(an, clips.edges, cfg)
    if not suspects:
        return
    if len(suspects) > cfg.suspect_max:
        # Auditing clip by clip is only worth it when the cut is mostly right. This many means the
        # pauses of this recitation do not describe its verses at all - don't pay to confirm that.
        raise SplitError(f"recitation: {len(suspects)} of {total_verses} clips look like more than one "
                         f"verse - the pauses of this recitation do not line up with its verses")
    for i, why in suspects[:cfg.suspect_max]:
        verdict = _verify_clip(cfg, clips[i - 1], surah_name, i, total_verses)
        if verdict is not None and verdict.is_single_complete_verse:
            cfg.log(f"   [{surah_name}] verse {i}'s clip was flagged ({why}) - Gemini confirms it is "
                    f"genuinely one complete verse: {verdict.explanation}")
            continue
        reason = (f"Gemini heard: {verdict.explanation}" if verdict is not None
                  else "it could not be verified (no Gemini API key set)")
        raise SplitError(f"recitation: verse {i}'s clip is not one verse - {why}, and {reason}")


def to_format(seg, tpl):
    """`seg` in the sample rate, channel count and sample width of `tpl`. pydub returns the very
    same object when a setting already matches, so for audio that is already in that format
    this costs three attribute comparisons and copies nothing."""
    return seg.set_frame_rate(tpl.frame_rate).set_channels(tpl.channels) \
              .set_sample_width(tpl.sample_width)


def concat(parts, tpl):
    """Join segments into one in tpl's format.

    Two things this deliberately does NOT do, both measured:
      * append to a bytearray and copy it out again. b"".join sizes the result once and copies
        once; the bytearray route copies twice - 1,144 parts / 313 MB took 0.2 s and +303 MB of
        peak memory with join, 0.4 s and +607 MB with bytearray (test_save.py --bench).
      * skip the conversion. The parts do not all come from one file: the interpretation can be
        a different rate/width/channel count from the recitation, and pydub's silence is always
        mono. Raw bytes appended at the wrong format play at the wrong speed or as noise. Nothing
        is converted here when the format already matches - assemble() normalises once, up
        front, so these calls return the segment itself."""
    return tpl._spawn(b"".join(to_format(p, tpl).raw_data for p in parts))


def snap(t, mids, window, direction=0):
    """Move t onto the nearest real pause. direction -1: only earlier, +1: only later."""
    c = [m for m in mids if abs(m - t) <= window and (direction == 0 or (m - t) * direction >= 0)]
    return min(c, key=lambda m: abs(m - t)) if c else t


_EDGES = ("recitation_start", "recitation_end", "interpretation_start", "interpretation_end")


def place_cut(an: Analysis, e1, s2, tol, lo, hi):
    """Where to cut between an explanation that ends at e1 and the next one that starts at s2 (ms).

    Gemini's times are the only thing that says WHICH pause is the boundary, and they are only
    approximate - so the search is confined to the interval Gemini gave, widened by `tol` each side,
    and never leaves [lo, hi]. It used to snap to the nearest pause of 200 ms or more within 1.2 s
    of the midpoint, in either direction: when the real gap between two explanations was shorter
    than that, the nearest qualifying pause was often INSIDE the next explanation, and its first
    words rode along in the previous clip.

    Returns (cut_ms, pause_ms). pause_ms is the length of the real pause the cut sits in; 0 means
    none was found and the quietest instant of the interval was used instead (a weak boundary)."""
    a, b = max(lo, min(e1, s2) - tol), min(hi, max(e1, s2) + tol)
    if b <= a:
        return max(lo, min(hi, (e1 + s2) // 2)), 0
    g0, g1 = min(e1, s2), max(e1, s2)
    for rel in (36, 30):
        best = None
        for s0, e0 in an.gaps(120, rel):
            if e0 <= a:
                continue
            if s0 >= b:
                break
            x, y = max(s0, a), min(e0, b)
            if y - x < 80:
                continue
            # prefer the pause that covers the most of Gemini's OWN gap, then the most of the window.
            # (Ranking by the pause's full length let a long pause that merely brushed the window
            # edge beat the short real one sitting right where Gemini said the boundary was.)
            key = (max(0, min(e0, g1) - max(s0, g0)), y - x)
            if best is None or key > best[0]:
                best = (key, (x + y) // 2, e0 - s0)
        if best:
            return best[1], best[2]
    db = an.db(a, b)
    if len(db) == 0:
        return (a + b) // 2, 0
    return a + int(np.argmin(np.convolve(db, np.ones(3) / 3, mode="same"))) * 10 + 5, 0


def move_by_pauses(an: Analysis, cut, steps, lo, hi):
    """`cut` moved `steps` real pauses (negative = earlier, positive = later), staying inside
    [lo, hi]. The pause the cut currently sits in does not count as a step. Unchanged if there are
    not that many pauses in range."""
    runs = an.gaps(120, 36)
    cur = next((r for r in runs if r[0] - 20 <= cut <= r[1] + 20), None)
    if steps < 0:
        ref = cur[0] if cur else cut
        cand = [r for r in runs if r[1] < ref - 20 and lo <= (r[0] + r[1]) // 2 <= hi]
        pick = cand[steps] if len(cand) >= -steps else None
    else:
        ref = cur[1] if cur else cut
        cand = [r for r in runs if r[0] > ref + 20 and lo <= (r[0] + r[1]) // 2 <= hi]
        pick = cand[steps - 1] if len(cand) >= steps else None
    return (pick[0] + pick[1]) // 2 if pick else cut


def edge_move(verse, edge, kind, n):
    """A problem heard at one edge of one clip says which cut is wrong and which way it should go.
    -> (part, boundary, sign) - boundary b is the cut between verses b and b+1, sign -1 = earlier,
    +1 = later - or None when the edge is not a verse-to-verse boundary (the intro/outro edges are
    governed by edge_pad instead).
        end of a clip + words of the NEXT verse in it  -> the cut is too late  -> earlier
        end of a clip + its own last word missing      -> the cut is too early -> later
        start of a clip + words of the PREVIOUS verse  -> the cut is too early -> later
        start of a clip + its own first word missing   -> the cut is too late  -> earlier"""
    if edge not in _EDGES or kind not in ("spill", "cutoff"):
        return None
    end = edge.endswith("_end")
    b = verse if end else verse - 1
    if not 1 <= b < n:
        return None
    return ("taf" if edge.startswith("interpretation") else "rec", b, -1 if end == (kind == "spill") else 1)


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


def _thinking_kwargs(budget):
    """Thinking tokens are billed at the OUTPUT rate - the most expensive stream there is - and
    the default is 'on'. The clip/identity/mismatch questions are near-classification work that
    does not need a scratchpad, so they get an explicit budget of 0."""
    if budget is None:
        return {}
    return {"thinking_budget": budget}


@lru_cache(maxsize=32)
def _chat(key, model, schema, temperature=0.0, thinking=None):
    """One factory for every call. include_raw=True keeps the AIMessage alongside the parsed
    object, which is the only way to see usage_metadata - without it the pipeline could not
    report what it spends."""
    try:
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=temperature,
                                     **_thinking_kwargs(thinking))
    except (TypeError, ValueError):       # older langchain-google-genai: no thinking_budget
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=temperature)
    return llm.with_structured_output(schema, include_raw=True)


def _llm(key, model, temperature=0.0):
    return _chat(key, model, TafsirMap, temperature, None)


def _temp(cfg, attempt):
    """At temperature 0 a re-ask tends to reproduce the answer that was just rejected, burning a
    full-price call to be told the same thing. Later attempts are asked slightly warmer."""
    return 0.0 if attempt == 0 else float(cfg.retry_temperature)


class ClipVerseCheck(BaseModel):
    is_single_complete_verse: bool = Field(description="true ONLY if this clip is the complete "
        "recitation of exactly one Quranic verse, start to finish - nothing missing at either "
        "end and nothing extra from a neighbouring verse")
    explanation: str = Field(description="briefly, what you actually heard - e.g. two verses run "
        "together, only part of a verse, a verse plus a fragment of the next one, or confirm it's "
        "one complete verse and (if relevant) why it's unusually long or short")


CLIP_CHECK_SYSTEM = """You are checking one short audio clip a program cut out of a full Quran
recitation, on the claim that it is the complete recitation of verse {verse} of Surah {surah}
({total} verses total). You know the Quran text well.

Judge only whether the clip is the recitation of exactly ONE verse, complete from its first word
to its last. Quran verses vary hugely in length - some are a single word, some are long sentences
- so judge by whether the wording is one complete, self-contained verse, NEVER by how long or
short the clip sounds. A clip may be flagged simply for being unusually long or short; that alone
is not evidence of a mistake."""


def _llm_clip_check(key, model):
    return _chat(key, model, ClipVerseCheck, 0.0, 0)


def _verify_clip(cfg, clip, surah_name, verse_number, total_verses):
    """Ask Gemini - which knows the Quran text, unlike a silence detector - whether a flagged
    clip really is one complete verse. Returns None (caller then fails closed) if there is no
    API key to ask with."""
    if not cfg.api_key:
        return None
    data = _audio_b64(clip)
    return ask(cfg, _llm_clip_check(cfg.api_key, cfg.model), [
        SystemMessage(CLIP_CHECK_SYSTEM.format(verse=verse_number, surah=surah_name, total=total_verses)),
        HumanMessage(content=[{"type": "media", "mime_type": "audio/mpeg", "data": data},
                              {"type": "text", "text": "Judge this clip."}])],
        f"checking recitation clip {verse_number}")


class ClipIdentity(BaseModel):
    verse_heard: int = Field(description="the number of the verse of this surah that the clip recites; "
        "0 if you cannot tell or the clip mixes several verses")
    complete: bool = Field(description="true only if the clip holds that verse from its first word to its "
        "last, with nothing missing and nothing from a neighbouring verse")
    explanation: str = Field(description="briefly, what you actually heard")


CLIP_ID_SYSTEM = """You know the Quran text well. The audio is one clip that a program cut out of a full recitation of Surah {surah} ({total} verses). The program believes it is verse {verse}. Say which verse of Surah {surah} it really is.

The clip may begin with the basmala; that is not a verse (except in Surah Al-Fatiha, where it is verse 1), so ignore it. Judge by the wording of what is recited, NEVER by how long or short the clip sounds."""


def _llm_clip_identity(key, model):
    return _chat(key, model, ClipIdentity, 0.0, 0)


# --------------------------------------------------------------------------- #
# What the recitation opens with                                               #
# --------------------------------------------------------------------------- #
class SegmentCheck(BaseModel):
    kind: str = Field(description="exactly one of: 'istiadha' - the clip is the isti'adha "
        "(a'udhu billahi min ash-shaytan ir-rajim, in any of its forms); 'basmala' - the clip is "
        "the basmala (bismillah ir-rahman ir-rahim); 'verse' - the clip holds the text of the "
        "surah itself; 'other' - anything else, such as a spoken title, an announcement or noise")
    heard: str = Field(description="briefly, what you actually heard")


SEGMENT_SYSTEM = """You know the Quran text well. The audio is one short stretch of speech that a program cut out of the very beginning of a recording of Surah {surah} ({total} verses), at the reciter's own pauses. It is stretch number {pos} of that recording.

Reciters often begin with the isti'adha, the basmala, or a spoken title before the surah's own text starts. Say which of those this stretch is, judging ONLY by the wording you hear, never by how long it sounds. If the stretch holds the surah's own text - even if a formula is recited first and runs straight into it without a pause - answer 'verse'."""


def _llm_segment(key, model):
    return _chat(key, model, SegmentCheck, 0.0, 0)


def leading_segments(an: Analysis, cfg, limit=3):
    """The first few stretches of speech, as (start_ms, end_ms) windows, split at the same pauses
    the verse cut will use. Only the first `limit` are ever needed: nothing puts more than an
    isti'adha, a basmala and a title in front of verse 1."""
    g = []
    for rel in (42, 36, 30, 26):
        g = an.gaps(cfg.min_silence, rel)
        if len(g) >= limit:
            break
    out, cur = [], 0
    for s0, e0 in g[:limit]:
        if s0 - cur >= 300:                       # ignore a sliver that holds no words
            out.append((cur, s0))
        cur = e0
    return out


def is_lead(kind, number):
    """Whether a stretch of speech at the head of the recording comes BEFORE verse 1.

    Two surahs differ from the rest: in Al-Fatiha the basmala IS verse 1, so hearing it means the
    surah has started, and At-Tawbah has no basmala of its own, so one recited before it is a
    formula like any other. An isti'adha precedes nothing in the text and can be added to ANY
    surah - which is the case surahs 1 and 9 had no way to express before."""
    k = (kind or "").strip().lower()
    if k == "basmala":
        return number != 1
    return k in ("istiadha", "isti'adha", "other")


def check_opening(s, cfg, rec, an):
    """How many stretches of speech come before verse 1, and what they were. This is the one
    question a silence detector cannot answer: it can hear THAT the reciter paused, never WHAT was
    recited before the pause, and a pause spent as a verse boundary misaligns the entire surah.

    Each leading stretch is judged on its own - a few seconds of audio, one unambiguous question -
    and the walk stops at the first stretch that is the surah's own text. That costs one to three
    small requests per surah, shared by every language of it, instead of the whole-recitation
    alignment that used to be the only way to know. Without an API key it falls back to what the
    surah implies, which is what the program assumed before."""
    number = s["number"]
    base = assumed_lead(number, cfg)
    if not cfg.api_key or not getattr(cfg, "opening_check", True):
        return base, ""
    lead, heard = 0, []
    for pos, (a, b) in enumerate(leading_segments(an, cfg), 1):
        check_stop(cfg)
        try:
            r = ask(cfg, _llm_segment(cfg.api_key, cfg.model), [
                SystemMessage(SEGMENT_SYSTEM.format(surah=s["name"], total=s["verses"], pos=pos)),
                HumanMessage(content=[_media(rec[a:min(b, a + cfg.opening_max_ms)]),
                                      {"type": "text", "text": "What is this stretch?"}])],
                f"hearing opening stretch {pos} of the recitation")
        except Stopped:
            raise
        except Exception as e:
            say(s, f"   ! could not hear what the recitation opens with ({e}) - assuming the usual "
                   f"{base} formula(s) before verse 1")
            return base, ""
        if r is None or not is_lead(r.kind, number):
            break
        lead += 1
        heard.append(r.heard or r.kind)
    return lead, "; ".join(heard)


PARTIAL_SYSTEM = """You help an audio editor. The audio you get is the SECOND PART of a recording of a spoken interpretation (tafsir) of Surah {surah} ({n} verses). An earlier attempt to mark where each verse's explanation starts and ends went wrong somewhere in this part, so it has to be marked again.

The excerpt begins at (or just before) the explanation of verse {first} and runs through the explanation of verse {n}, in order, followed by a short closing line (for example 'end') that is NOT part of any verse.

Rules:
- Return EXACTLY {count} entries, numbered {first}..{n}, in order, without overlaps. The first entry is verse {first}.
- One verse's explanation continues through pauses, examples, stories and repeated quotations of the ayah. Start a new entry only when the speaker moves on to the NEXT verse. Never split one verse in two and never merge two verses.
- Times are MM:SS.mmm counted from the very beginning of THIS EXCERPT (not of the whole recording). start = first spoken word of that explanation, end = its last spoken word.
- intro_heard: leave it empty. outro_heard: what the closing line says."""


_FATAL_HINTS = ("api key", "api_key", "permission", "unauthenticated", "invalid_argument", "invalid argument",
                "blocked", "safety", "limit: 0", "billing", "not found")
_TRANSIENT_HINTS = ("429", "500", "502", "503", "504", "resource_exhausted", "unavailable", "overloaded",
                    "deadline", "timeout", "timed out", "temporarily", "connection", "internal error",
                    "try again")


def gemini_error(e) -> str:
    """One readable line out of whatever the Gemini client raised, so the user sees Gemini's own
    complaint (with a plain-words hint when it is a familiar one) instead of a bare traceback."""
    msg = re.sub(r"\s+", " ", str(e)).strip() or type(e).__name__
    low = msg.lower()
    if "api key" in low or "api_key" in low or "unauthenticated" in low:
        hint = "the API key was rejected - check it in the app"
    elif "429" in low or "resource_exhausted" in low or "quota" in low:
        hint = "rate limit / quota reached"
    elif "503" in low or "unavailable" in low or "overloaded" in low:
        hint = "Gemini is overloaded right now"
    elif "safety" in low or "blocked" in low:
        hint = "the request was blocked by Gemini's safety filter"
    else:
        hint = ""
    return msg[:280] + ("..." if len(msg) > 280 else "") + (f"  [{hint}]" if hint else "")


def _is_transient(e) -> bool:
    low = str(e).lower()
    if any(h in low for h in _FATAL_HINTS):
        return False
    return isinstance(e, (TimeoutError, ConnectionError)) or any(h in low for h in _TRANSIENT_HINTS)


def _record(cfg, raw, what):
    """Keep what each call cost. Without this nothing about spending is observable, and there is
    no way to tell whether a change to the pipeline actually saved anything."""
    u = getattr(raw, "usage_metadata", None) or {}
    if not u:
        return
    det = u.get("output_token_details") or {}
    cfg.usage["calls"] += 1
    cfg.usage["input"] += int(u.get("input_tokens") or 0)
    cfg.usage["output"] += int(u.get("output_tokens") or 0)
    cfg.usage["thinking"] += int(det.get("reasoning") or 0)
    cfg.usage["cached"] += int((u.get("input_token_details") or {}).get("cache_read") or 0)
    cfg.usage["by_step"][what.split(" ")[0]] = cfg.usage["by_step"].get(what.split(" ")[0], 0) \
        + int(u.get("input_tokens") or 0) + int(u.get("output_tokens") or 0)


def _unwrap(cfg, out, what):
    """with_structured_output(include_raw=True) returns {'raw', 'parsed', 'parsing_error'}."""
    if isinstance(out, dict) and ("parsed" in out or "raw" in out):
        _record(cfg, out.get("raw"), what)
        return out.get("parsed")
    return out


def ask(cfg, chain, messages, what):
    """chain.invoke() with patience for transient API trouble (rate limit, overload, timeout): show
    Gemini's message, wait, ask again. These waits never use up `retries` / `verify_retries`, which
    are for Gemini's ANSWERS being wrong. Anything not transient becomes a SplitError that carries
    Gemini's own message."""
    waits = [5, 15, 30, 60][:max(0, cfg.api_retries)]
    for i in range(len(waits) + 1):
        check_stop(cfg)
        try:
            return _unwrap(cfg, chain.invoke(messages), what)
        except Stopped:
            raise
        except Exception as e:
            if i >= len(waits) or not _is_transient(e):
                raise SplitError(f"Gemini error while {what}: {gemini_error(e)}") from e
            cfg.log(f"   ! Gemini error while {what}: {gemini_error(e)}")
            cfg.log(f"     temporary - waiting {waits[i]}s, then asking again ({i + 1}/{len(waits)}); "
                    f"this does not use up a retry")
            cfg.status(f"Gemini is busy - waiting {waits[i]}s before asking again ({i + 1}/{len(waits)})...")
            if cfg.stop.wait(waits[i]):
                raise Stopped()


def _audio_b64(audio) -> str:
    buf = io.BytesIO()
    audio.set_channels(1).set_frame_rate(16000).export(buf, format="mp3", bitrate="32k")
    data = buf.getvalue()
    if len(data) > 18 * 1024 * 1024:
        raise SplitError(
            f"audio is too long to send to Gemini inline (>18 MB, about {len(audio) / 60000:.0f} "
            f"minutes). Split the recording, or set Recitation cutting to 'Silences only' if it is "
            f"the recitation. Lifting this needs the Files API (see the note at the top).")
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
# Timing from a transcription model (gemini-3.5-transcribe)                    #
# --------------------------------------------------------------------------- #
# A chat model that "listens" has to guess timestamps, and it drifts on long recordings. A
# transcription model returns the words with their real start/end offsets. The verses are then
# marked in TEXT: the chat model reads the numbered transcript and only answers "which word does
# each verse start at" - integers, never times. Every time in the map therefore comes from the
# transcriber, and the mapping call costs text tokens instead of 32 tokens per second of audio.
#
# Limits that shape this code (Gemini API docs): word timestamps work on at most 30 minutes of
# audio per request (-> chunk_edges), they may lower accuracy slightly, and the model is in
# public preview.
@dataclass
class Word:
    start: int          # ms, from the beginning of the WHOLE recording
    end: int
    text: str


class TranscribeError(SplitError):
    """Transcription itself failed (SDK too old, model unavailable, no words back). The mapping step
    raises plain SplitError, so 'auto' mode can fall back to the chat model for THIS problem only."""


def _g(o, k, default=None):
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


def _offset_ms(v):
    """'0.450s' / 0.45 / '1.2' -> 450 / 450 / 1200 ms."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return int(float(v) * 1000)
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*s?\s*$", str(v))
    return int(float(m.group(1)) * 1000) if m else None


def extract_words(interaction):
    """The word_info annotations of an Interactions API response, as [Word] (times of THIS request)."""
    words = []
    for step in _g(interaction, "steps", None) or []:
        for content in _g(step, "content", None) or []:
            for a in _g(content, "annotations", None) or []:
                if _g(a, "type") != "word_info":
                    continue
                st, en = _offset_ms(_g(a, "start_offset")), _offset_ms(_g(a, "end_offset"))
                text = str(_g(a, "text") or "").strip()
                if text and st is not None and en is not None:
                    words.append(Word(st, max(en, st + 1), text))
    words.sort(key=lambda w: w.start)
    return words


def chunk_edges(an: Analysis, total_ms, max_ms):
    """Cut points so that no piece is longer than max_ms, each one in the LONGEST pause of its window
    (a chunk boundary must never fall inside a word). Word timestamps are limited to 30 min of audio."""
    if total_ms <= max_ms:
        return [0, total_ms]
    edges = [0]
    while total_ms - edges[-1] > max_ms:
        lo, hi = edges[-1] + int(max_ms * 0.6), edges[-1] + max_ms
        cand = [g for g in an.gaps(150, 36) if lo <= (g[0] + g[1]) // 2 <= hi]
        if cand:
            g = max(cand, key=lambda x: x[1] - x[0])
            cut = (g[0] + g[1]) // 2
        else:                                          # no pause at all: the quietest instant
            db = an.db(lo, hi)
            cut = lo + int(np.argmin(np.convolve(db, np.ones(3) / 3, mode="same"))) * 10 if len(db) else hi
        edges.append(int(cut))
    edges.append(total_ms)
    return edges


# BCP-47 codes the model lists. A language that is not here (Saudi, Moroccan, Urdu, ...) is left to
# the model's automatic detection, which is what the docs recommend when the code is not known.
_LANG_CODES = {
    "english": "en-US", "american": "en-US", "british": "en-GB", "spanish": "es-419", "espanol": "es-419",
    "french": "fr-FR", "francais": "fr-FR", "egyptian": "ar-EG", "masri": "ar-EG", "persian": "fa-IR",
    "farsi": "fa-IR", "german": "de-DE", "turkish": "tr-TR", "indonesian": "id-ID", "malay": "ms-MY",
    "russian": "ru-RU", "portuguese": "pt-BR", "brazilian portuguese": "pt-BR", "italian": "it-IT",
    "hindi": "hi-IN", "bengali": "bn-BD", "swahili": "sw-KE", "hausa": "ha-NG", "chinese": "cmn-Hans-CN",
    "mandarin": "cmn-Hans-CN", "japanese": "ja-JP", "korean": "ko-KR", "dutch": "nl-NL", "polish": "pl-PL",
    "ukrainian": "uk-UA", "vietnamese": "vi-VN", "thai": "th-TH", "filipino": "fil-PH", "tagalog": "fil-PH",
}


def lang_code(lang):
    return _LANG_CODES.get(fold(lang or ""))


_GENAI = {}


def _genai_client(key):
    """The google-genai client itself (not langchain): the transcription model is served by the
    Interactions API, which langchain-google-genai does not wrap."""
    if key not in _GENAI:
        try:
            from google import genai
        except ImportError:
            raise TranscribeError("the transcription model needs the google-genai package:  "
                                  "pip install -U google-genai")
        client = genai.Client(api_key=key)
        if not hasattr(client, "interactions"):
            raise TranscribeError("this google-genai version has no Interactions API, which "
                                  "gemini-3.5-transcribe needs:  pip install -U google-genai")
        _GENAI[key] = client
    return _GENAI[key]


class _Call:
    """Lets ask() give any callable the same patience for transient API errors as a chat call."""
    def __init__(self, fn):
        self.fn = fn

    def invoke(self, _messages=None):
        return self.fn()


def transcribe_segment(cfg, seg, code, what):
    """One request: <= 30 min of audio -> [Word] with times relative to the start of `seg`."""
    import os
    import tempfile
    client = _genai_client(cfg.api_key)
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    up = None
    try:
        seg.set_channels(1).set_frame_rate(16000).export(path, format="mp3", bitrate="64k")

        def call():
            nonlocal up
            if up is None:
                up = client.files.upload(file=path)
            tc = {"mode": {"type": "verbatim", "timestamp_granularities": ["word"]}}
            if code:
                tc["language_codes"] = [code]
            return client.interactions.create(
                model=cfg.transcribe_model,
                input=[{"type": "audio", "uri": up.uri, "mime_type": up.mime_type}],
                generation_config={"transcription_config": tc})

        interaction = ask(cfg, _Call(call), None, what)
    finally:
        if up is not None:
            try:
                client.files.delete(name=up.name)
            except Exception:
                pass
        try:
            os.unlink(path)
        except OSError:
            pass
    words = extract_words(interaction)
    if not words:
        raise TranscribeError(f"{what}: the model returned no word timestamps")
    cfg.usage["transcribed_s"] = cfg.usage.get("transcribed_s", 0) + len(seg) / 1000
    return words


def _audio_stamp(path):
    try:
        st = Path(path).stat()
        return f"{Path(path).name}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return ""


def get_words(s, cfg, audio, an, audio_path, cache, code, label):
    """The word-timed transcript of one recording: from `cache` if it is still the same audio,
    model and language, otherwise transcribed in <= transcribe_chunk_min pieces (timestamps of every
    piece are shifted to the whole recording's clock)."""
    stamp = f"{_audio_stamp(audio_path)}|{cfg.transcribe_model}|{code or ''}"
    if cache is not None and Path(cache).exists() and not cfg.retranscribe:
        try:
            raw = json.loads(Path(cache).read_text("utf-8"))
            if raw.get("stamp") == stamp:
                words = [Word(int(a), int(b), t) for a, b, t in raw["words"]]
                if words:
                    say(s, f"   using the saved transcript of {label} ({Path(cache).name}, {len(words)} words)")
                    return words
        except Exception:
            pass
    edges = chunk_edges(an, len(audio), int(cfg.transcribe_chunk_min * 60000))
    words = []
    for i, (a, b) in enumerate(zip(edges, edges[1:]), 1):
        check_stop(cfg)
        part = f" (part {i}/{len(edges) - 1}, {fmt_ms(a)}-{fmt_ms(b)})" if len(edges) > 2 else ""
        say(s, f"   transcribing {label}{part} with {cfg.transcribe_model}...")
        doing(s, f"transcribing {label}{part} - this can take a minute...")
        for w in transcribe_segment(cfg, audio[a:b], code, f"transcribing {label}{part}"):
            words.append(Word(w.start + a, w.end + a, w.text))
    say(s, f"   transcribed {label}: {len(words)} words")
    if cache is not None:
        try:
            Path(cache).write_text(json.dumps(
                {"stamp": stamp, "model": cfg.transcribe_model, "words": [[w.start, w.end, w.text] for w in words]},
                ensure_ascii=False), "utf-8")
        except OSError:
            pass
    return words


def _transcript(s, cfg, audio, an, audio_path, cache, code, label):
    try:
        return get_words(s, cfg, audio, an, audio_path, cache, code, label)
    except TranscribeError:
        raise
    except SplitError as e:                            # e.g. a Gemini error while transcribing
        raise TranscribeError(str(e)) from e


class WordMap(BaseModel):
    intro_heard: str = Field(description="what the opening announcement / isti'adha / basmala says, or empty")
    verse_starts: list[int] = Field(description="the index of the FIRST word of each verse, exactly one per "
                                                "verse, in order, strictly increasing")
    outro_start: int = Field(description="the index of the first word of the closing line, or -1 if there is none")
    outro_heard: str = Field(description="what the closing line says, or empty")


TAFSIR_MAP_SYSTEM = """You help an audio editor. Below is the word-for-word transcript of ONE recording of a spoken interpretation (tafsir) of Surah {surah}, which has {n} verses. Every word carries its index in square brackets, like [57]word, and each line starts with the time (@MM:SS) at which it begins. The transcript is machine-made: it can contain recognition mistakes, so judge by meaning, not by exact spelling.

The recording is structured like this:
 1. a short opening announcement (for example the word 'begin' and the surah's title). It is NOT part of any verse.
 2. the explanation of verse 1, then verse 2, ... up to verse {n}, in order.
 3. a short closing line (for example 'end'). It is NOT part of any verse.

Return:
- verse_starts: EXACTLY {n} numbers - the index of the first word of the explanation of verse 1, verse 2, ... verse {n}. Strictly increasing.
- outro_start: the index of the first word of the closing line, or -1 if the recording has none.
- intro_heard / outro_heard: what the opening / closing lines say.

Rules:
- The explanation of verse 1 begins right after the opening announcement. Never count it as part of the announcement and never skip it, even if the two are close together.
- One verse's explanation continues through examples, stories and repeated quotations of the ayah. Start a new verse only where the speaker moves on to the NEXT verse. Never split one verse in two and never merge two verses."""

REC_MAP_SYSTEM = """You help an audio editor. Below is a machine transcript of ONE complete recitation of Surah {surah} ({n} verses). Every word carries its index in square brackets, like [57]word, and each line starts with the time (@MM:SS) at which it begins. Recognition of recited Quranic Arabic contains mistakes, so use your knowledge of the Quran text to recognise where each verse begins.

Return:
- verse_starts: EXACTLY {n} numbers - the index of the first word of verse 1, verse 2, ... verse {n}. Strictly increasing.
- outro_start: the index of the first word of anything recited after the last verse, or -1 if there is none.
- intro_heard / outro_heard: what is recited before verse 1 (isti'adha, basmala) / after the last verse.

Rules:
- {basmala}
- The reciter may repeat words or whole phrases: a repetition belongs to the verse it repeats and never starts a new verse. A verse begins at the first word of its own text, however short the pause before it is - many reciters run verses together.
- Verses can be a single word or very long: judge only by the wording, never by how long a stretch sounds."""


def render_words(words, per_line=12):
    """The transcript as the mapping model reads it: an index on every word, a clock at every line."""
    lines = []
    for i in range(0, len(words), per_line):
        chunk = words[i:i + per_line]
        m, sec = divmod(chunk[0].start // 1000, 60)
        lines.append(f"@{m:02d}:{sec:02d} " + " ".join(f"[{i + j}]{w.text}" for j, w in enumerate(chunk)))
    return "\n".join(lines)


def _check_word_map(wm, n, nwords):
    """'' if the answer is usable, else the exact complaint to send back."""
    st = list(wm.verse_starts or [])
    if len(st) != n:
        return (f"you returned {len(st)} start indices; I need exactly {n}, one per verse (none for the "
                f"announcement, the isti'adha, the basmala or the closing line).")
    if any(not isinstance(x, int) or x < 0 or x >= nwords for x in st):
        return f"a start index is outside the transcript (valid indices are 0..{nwords - 1})."
    for i in range(1, n):
        if st[i] <= st[i - 1]:
            return (f"verse {i + 1} starts at word {st[i]}, which is not after the start of verse {i} "
                    f"(word {st[i - 1]}); the starts must strictly increase.")
    if wm.outro_start not in (None, -1) and wm.outro_start <= st[-1]:
        return "outro_start must come after the first word of the last verse (or be -1)."
    return ""


def map_from_words(wm, words, n):
    """The TafsirMap (exactly what a chat-model listen produced) built from the transcriber's own
    times: a verse runs from its first word to the word before the next verse's first word."""
    st = list(wm.verse_starts)
    last = wm.outro_start - 1 if wm.outro_start not in (None, -1) else len(words) - 1
    verses = []
    for i in range(n):
        a = st[i]
        b = max(a, st[i + 1] - 1 if i < n - 1 else last)
        verses.append(VerseSpan(verse=i + 1, start=fmt_ms(words[a].start), end=fmt_ms(words[b].end),
                                first_words=" ".join(w.text for w in words[a:a + 6])))
    intro = wm.intro_heard or " ".join(w.text for w in words[:min(st[0], 12)])
    outro = wm.outro_heard or (" ".join(w.text for w in words[last + 1:last + 13]) if last + 1 < len(words) else "")
    return TafsirMap(intro_heard=intro, verses=verses, outro_heard=outro)


def _window_rule(first, last, n, at_start):
    where = ("at the very beginning of the recording" if at_start else
             f"part-way through verse {first - 1}, which has already been marked")
    return (f"\n\nIMPORTANT - this transcript is only PART of the recording, and it begins {where}. "
            f"Ignore the count asked for above. Return EXACTLY {last - first + 1} start indices: the "
            f"first word of verse {first}, of verse {first + 1}, ... of verse {last}, counted from the "
            f"start of THIS transcript."
            + ("" if at_start else f" Index 0 is inside verse {first - 1}, so the first index you "
                                   f"return is never 0.")
            + ("" if last == n else f" Verse {n} is NOT in this part, so set outro_start to -1."))


def _ask_window(s, cfg, words, lo, hi, system, header, notes, first, last, n, what):
    """One window of the transcript: where every verse from `first` to `last` begins, as indices
    into the WHOLE transcript. Returns (starts, intro_heard, outro_start_or_-1)."""
    count, err = last - first + 1, ""
    sl = words[lo:hi]
    transcript = render_words(sl)
    rule = _window_rule(first, last, n, lo == 0)
    for attempt in range(cfg.retries + 1):
        check_stop(cfg)
        text = f"{header}{rule}\n\nTranscript:\n{transcript}"
        for note in notes:
            text += f"\n\n{note}"
        if err:
            text += f"\n\nYour previous answer was rejected: {err}\nRead again and fix exactly that."
        doing(s, f"Gemini is marking verses {first}-{last} of {n} (try {attempt + 1})...")
        wm = ask(cfg, _chat(cfg.api_key, cfg.model, WordMap, _temp(cfg, attempt), None),
                 [SystemMessage(system), HumanMessage(content=text)], what)
        err = ("the answer was not valid JSON for the requested schema." if wm is None
               else _check_word_map(wm, count, len(sl)))
        if not err and lo and (wm.verse_starts or [0])[0] == 0:
            err = (f"the first index is 0, but index 0 is inside verse {first - 1}; verse {first} "
                   f"starts later than that.")
        if not err:
            outro = wm.outro_start if (last == n and wm.outro_start not in (None, -1)) else -1
            return ([lo + i for i in wm.verse_starts], wm.intro_heard or "",
                    lo + outro if outro != -1 else -1)
        say(s, f"   ! verses {first}-{last} rejected: {err}")
    raise SplitError(f"could not mark verses {first}-{last} from the transcript ({err})")


def _map_words_windowed(s, cfg, words, system, header, notes, validate, what):
    """A surah with hundreds of verses cannot be marked in one answer: the model is being asked for
    286 strictly increasing indices in a single reply, and one slip anywhere rejects the whole
    thing. The transcription itself is already cut into <= transcribe_chunk_min pieces; this does
    the same for the reading of it. Each window is asked about a slice of the transcript that
    starts where the previous window's last verse started, so every window has the verse before it
    as an anchor and the windows cannot drift apart."""
    n = s["verses"]
    avg = len(words) / max(1, n)
    per = max(10, int(cfg.map_verses))
    starts, intro, outro, lo, first = [], "", -1, 0, 1
    while first <= n:
        last = min(n, first + per - 1)
        hi = (len(words) if last == n
              else min(len(words), lo + int((last - first + 2) * avg * 2.5) + 300))
        got, heard, out = _ask_window(s, cfg, words, lo, hi, system, header, notes, first, last, n, what)
        if starts and got[0] <= starts[-1]:
            raise SplitError(f"verse {first} was marked at or before verse {first - 1}")
        starts.extend(got)
        intro, outro = intro or heard, out if out != -1 else outro
        lo, first = got[-1], last + 1
    wm = WordMap(intro_heard=intro, verse_starts=starts, outro_start=outro, outro_heard="")
    err = _check_word_map(wm, n, len(words))
    if err:
        raise SplitError(f"the windows did not fit together ({err})")
    tm = map_from_words(wm, words, n)
    err = validate(tm) if validate else ""
    if err:
        raise SplitError(f"could not mark {n} verses from the transcript ({err})")
    say(s, f"   marked {n} verses in {-(-n // per)} window(s) of the transcript")
    return tm


def _map_words(s, cfg, words, system, header, notes, validate, what):
    """Ask the chat model (TEXT only) where each verse starts. Answers are checked here and Gemini is
    told exactly what was wrong; a few cheap attempts cost less than one listen to the audio."""
    n, err = s["verses"], ""
    if n > cfg.map_verses and len(words) > n:
        return _map_words_windowed(s, cfg, words, system, header, notes, validate, what)
    transcript = render_words(words)
    for attempt in range(cfg.retries + 1):
        check_stop(cfg)
        # the long, unchanging part goes first so Gemini's implicit cache can match a retry's prefix
        text = f"{header}\n\nTranscript:\n{transcript}"
        for note in notes:
            text += f"\n\n{note}"
        if err:
            text += f"\n\nYour previous answer was rejected: {err}\nRead again and fix exactly that."
        doing(s, f"Gemini is reading the transcript to mark the verses (try {attempt + 1})...")
        wm = ask(cfg, _chat(cfg.api_key, cfg.model, WordMap, _temp(cfg, attempt), None),
                 [SystemMessage(system), HumanMessage(content=text)], what)
        err = ("the answer was not valid JSON for the requested schema." if wm is None
               else _check_word_map(wm, n, len(words)))
        if not err:
            tm = map_from_words(wm, words, n)
            err = validate(tm) if validate else ""
            if not err:
                return tm
        say(s, f"   ! mapping rejected: {err}")
    raise SplitError(f"could not mark {n} verses from the transcript ({err})")


def _listen_by_transcript(s, cfg):
    """listen(), but the timing comes from the transcription model."""
    n, lang = s["verses"], s["lang"]
    cache = Path(str(s["map_path"])[:-len(".map.json")] + ".words.json")
    words = _transcript(s, cfg, s["tafsir"], s["taf_an"], s["audio_path"], cache, lang_code(lang),
                        f"the {lang} interpretation")
    header = f"Surah {s['name']}, {n} verses. Language or dialect of the recording: {lang}."
    if s.get("reference"):
        header += ("\n\nReference text (the written version of what is spoken; it may also contain the Arabic "
                   "verses, titles or numbers). Use it ONLY to recognise where each verse's explanation "
                   "begins:\n" + s["reference"])
    notes = []
    if s.get("error"):
        notes.append(f"An earlier answer was rejected: {s['error']}\nFix exactly that.")
    if s.get("relisten_note"):
        notes.append(f"A later review of the finished file found a problem with the previous marking: "
                     f"{s['relisten_note']}\nFix exactly that.")
    return _map_words(s, cfg, words, TAFSIR_MAP_SYSTEM.format(surah=s["name"], n=n), header, notes,
                      None, "marking the interpretation")


def _align_recitation_by_transcript(s, cfg, rec):
    """align_recitation(), but the timing comes from the transcription model."""
    n, name, num = s["verses"], s["name"], s["number"]
    same = _REC_CACHE.get("key") == str(s["rec_path"]) and _REC_CACHE.get("an") is not None
    an = _REC_CACHE["an"] if same else Analysis(rec)
    rp = s.get("rec_map_path")
    cache = Path(rp).with_name(Path(rp).stem + ".words.json") if rp else None
    words = _transcript(s, cfg, rec, an, s["rec_path"], cache, cfg.rec_language_code or None,
                        "the recitation")
    notes = []
    if _REC_CACHE.get("opening"):
        notes.append(f"This recording was already heard to open with: \"{_REC_CACHE['opening']}\" - that "
                     f"is recited before verse 1 and is not part of it.")
    if s.get("realign_note"):
        notes.append(f"A later check of the finished audio found: {s['realign_note']}\n"
                     f"Mark the verses correctly.")
    tm = _map_words(s, cfg, words, REC_MAP_SYSTEM.format(surah=name, n=n, basmala=_basmala_rule(num)),
                    f"Surah {name}, {n} verses.", notes,
                    lambda t: _check_rec_map(t, n, len(rec)), "marking the recitation")
    if rp:
        Path(rp).write_text(tm.model_dump_json(indent=2), "utf-8")
    return tm


# --------------------------------------------------------------------------- #
# The graph                                                                    #
# --------------------------------------------------------------------------- #
class State(TypedDict, total=False):
    cfg: Settings
    job_cfg: Settings           # cfg for this attempt - equals cfg, except repair() nudges padding/gaps in a copy
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
    rec_an: Any                 # loudness frames of the recitation, computed once per surah
    taf_an: Any                 # loudness frames of the interpretation, computed once per job
    taf_fmt: Any                # (format, audio): the interpretation converted to the recitation's format, once
    cut_shift: dict             # {("taf", b): pauses} - interpretation cuts a repair moved
    rec_clips: Any
    part_bounds: list           # per verse, where its recitation and its interpretation sit in the output
    local_problems: list        # defects measured directly, without asking Gemini
    tafsir: Any
    tmap: Any
    spans: list
    attempt: int
    error: str
    output: str
    assembled_audio: Any        # the stitched-together file, before it's written to disk
    unit_bounds: list           # [(start_ms, end_ms, verse_number), ...] of each verse+interpretation block
    verify_attempt: int
    needs_repair: bool
    problems: list
    relisten_note: str          # set by repair() when a 'mismatch' problem forces a fresh Gemini listen
    relisten_from: int          # >= 2: only verses relisten_from..N are marked again (tail re-listen); 0 = whole recording
    recheck: Any                # verses the next final check must listen to again (None = everything)
    bad_history: dict           # first mismatching verse -> how many repairs it has needed (widens the tail re-listen)
    resplit_done: bool          # the recitation was already re-cut once because a clip was not the verse it should be
    full_relisten_done: bool    # the whole recording was already re-listened to once because of a mismatch
    give_up: bool               # repair decided another round cannot help
    rec_map_path: Path          # saved Gemini alignment of this surah's recitation (shared by every language)
    realign_note: str           # set by repair(): the recitation cut was found wrong, so Gemini must mark it again


def say(s, msg):
    s["cfg"].log(msg)


def doing(s, msg):
    s["cfg"].status(f"[{s['number']:03d} {s['name']} / {s['lang']}] {msg}")


REC_SYSTEM = """You help an audio editor. You listen to ONE recording of the complete recitation of Surah {surah} ({n} verses) and mark where each verse starts and ends. You know the Quran text, so recognise the verses by their wording.

Rules:
- Return EXACTLY {n} entries, numbered 1..{n}, in order, without overlaps.
- {basmala}
- A verse ends where the reciter finishes its last word, however short the pause after it is - many reciters run verses together, so do not rely on pauses. The next verse starts at the reciter's first word of it. Verses can be a single word or very long: judge only by the wording, never by how long a stretch sounds.
- Times are MM:SS.mmm counted from the very beginning of the audio. start = the first word of that verse, end = its last word. first_words = its first ~6 words exactly as recited.
- intro_heard: what is recited before verse 1 (isti'adha, basmala), or empty. outro_heard: anything after the last verse, or empty."""


def _basmala_rule(number):
    # An isti'adha can precede ANY surah: it is the reciter's own addition, not part of the text.
    # Surahs 1 and 9 used to be told nothing about it, which is why a recitation of Al-Fatiha that
    # opened with one had no rule to follow and the isti'adha was marked as verse 1.
    isti = ("An isti'adha (a'udhu billahi min ash-shaytan ir-rajim, in any of its forms) recited "
            "before it is NOT a verse and NOT part of verse 1: it belongs in intro_heard.")
    if number == 1:
        return ("The basmala IS verse 1 of this surah: entry 1 is the basmala itself and starts at its "
                "first word. " + isti)
    if number == 9:
        return "This surah has no basmala; verse 1 starts at the first verse word. " + isti
    return ("The basmala (and any isti'adha before it) is NOT part of verse 1: it goes in intro_heard and "
            "verse 1 starts right after it. If the recording has no basmala, verse 1 starts at the first word.")


def _check_rec_map(tm, n, total_ms):
    """'' if the recitation map is usable, else the exact complaint to send back to Gemini."""
    v = sorted(tm.verses, key=lambda x: x.verse)
    if [x.verse for x in v] != list(range(1, n + 1)):
        return (f"you returned {len(v)} entries; I need exactly {n}, numbered 1..{n}, one per verse "
                f"(the basmala is not a verse, except in Surah Al-Fatiha).")
    try:
        spans = [(_seconds(x.start), _seconds(x.end)) for x in v]
    except ValueError:
        return "a time was not in MM:SS.mmm format."
    for i, (a, b) in enumerate(spans, 1):
        if b - a < 0.4:
            return f"verse {i} lasts only {b - a:.1f}s - too short to be a recited verse."
        if i > 1 and a < spans[i - 2][1] - 0.5:
            return f"verse {i} starts before verse {i - 1} ends."
    if spans[-1][1] > total_ms / 1000 + 1:
        return f"verse {n} ends at {spans[-1][1]:.0f}s but the audio is only {total_ms / 1000:.0f}s long."
    return ""


def align_recitation(s, cfg, rec):
    """Gemini listens to the whole recitation and marks every verse - the recitation counterpart of
    listen(). The silence-based cut cannot tell a verse boundary from a breath inside a long verse
    (or find a boundary at all where the reciter runs verses together); Gemini knows the text. The
    answer is checked, Gemini is told exactly what was wrong and asks again, and it is saved once
    per surah (every language reuses it)."""
    if not cfg.api_key:
        raise SplitError("no Gemini API key")
    n, name, num, total = s["verses"], s["name"], s["number"], len(rec)
    if cfg.timing_source != "chat" and not cfg.usage.get("transcribe_off"):
        try:
            return _align_recitation_by_transcript(s, cfg, rec)
        except TranscribeError as e:
            if cfg.timing_source == "transcribe":
                raise
            cfg.usage["transcribe_off"] = True
            say(s, f"   ! the transcription model cannot be used ({e}) - the chat model will listen to "
                   f"the recitation instead for the rest of this run")
    if total > cfg.chat_listen_max_min * 60000:
        # The transcription path cuts long audio into <= transcribe_chunk_min pieces and stitches the
        # word times back together; the chat path has to send the whole recording in one request, and
        # a two-hour surah is far past what that can carry.
        raise SplitError(f"recitation: {fmt_ms(total)} is too long to send to the chat model in one "
                         f"request (limit {cfg.chat_listen_max_min:.0f} min). The transcription path "
                         f"splits long recordings by itself - set 'Timing source' to Gemini 3.5 "
                         f"Transcribe for this surah.")
    complaint, err, tm = s.get("realign_note") or "", "", None
    opening = _REC_CACHE.get("opening") or ""
    for attempt in range(cfg.retries + 1):
        text = f"Surah {name}, {n} verses."
        if opening:
            text += (f"\n\nThis recording was already heard to open with: \"{opening}\" - that is "
                     f"recited before verse 1 and is not part of it.")
        if complaint:
            text += (f"\n\nA later check of the finished audio found: {complaint}\n"
                     f"Listen again and mark the verses correctly.")
        if err:
            text += f"\n\nYour previous answer was rejected: {err}\nListen again and fix exactly that."
        say(s, f"   recitation: Gemini is listening to {fmt_ms(total)} of recitation to mark every verse "
               f"(try {attempt + 1})...")
        doing(s, f"Gemini is listening to the recitation (try {attempt + 1}) - this can take a minute...")
        tm = ask(cfg, _llm(cfg.api_key, cfg.model, _temp(cfg, attempt)), [
            SystemMessage(REC_SYSTEM.format(surah=name, n=n, basmala=_basmala_rule(num))),
            HumanMessage(content=[_media(rec), {"type": "text", "text": text}])],
            "aligning the recitation")
        err = "the answer was not valid JSON for the requested schema." if tm is None else _check_rec_map(tm, n, total)
        if not err:
            if s.get("rec_map_path"):
                Path(s["rec_map_path"]).write_text(tm.model_dump_json(indent=2), "utf-8")
            return tm
        say(s, f"   ! recitation alignment rejected: {err}")
    raise SplitError(f"recitation: Gemini could not mark {n} verses ({err})")


def map_edges(an: Analysis, tm, number, cfg):
    """The verse boundaries Gemini's recitation map implies. Never trust its clock to the
    millisecond: each cut moves to the QUIETEST spot around the boundary (a real pause when there
    is one, the least-loud instant when the reciter runs verses together). Independent of
    cfg.pad, so a padding-only repair reuses them."""
    total, db = an.total_ms, an.db()
    sp = [(int(_seconds(x.start) * 1000), int(_seconds(x.end) * 1000))
          for x in sorted(tm.verses, key=lambda x: x.verse)]

    def quiet(lo, hi):
        lo, hi = max(0, lo), min(total, hi)
        a, b = lo // 10, max(lo // 10 + 1, hi // 10)
        seg = db[a:b]
        if len(seg) == 0:
            return (lo + hi) // 2
        return (a + int(np.argmin(np.convolve(seg, np.ones(3) / 3, mode="same")))) * 10 + 5

    cuts = []
    for (s1, e1), (s2, e2) in zip(sp[:-1], sp[1:]):
        lo, hi = max(s1 + 300, e1 - 500), min(e2 - 300, s2 + 500)
        c = quiet(lo, hi) if lo < hi else (e1 + s2) // 2
        cuts.append(max(c, (cuts[-1] if cuts else 0) + 300))
    # `keep` keeps whatever precedes verse 1 as intro audio at the head of verse 1's clip; `drop`
    # trims back to where verse 1 actually begins. Surahs 1 and 9 used to be forced to start at 0
    # on the grounds that nothing precedes their verse 1 - true of the TEXT, but not of a reciter
    # who opens with the isti'adha, and it threw away the one answer that knew where it ended.
    start = 0 if cfg.basmala != "drop" else quiet(sp[0][0] - 800, sp[0][0])
    return [start] + cuts + [total]


def split_by_map(audio, an: Analysis, tm, number, cfg, edges=None):
    if edges is None:
        edges = map_edges(an, tm, number, cfg)
    return cut_all(audio, an, edges, cfg.pad)


# One surah's recitation, kept between the languages of that surah. Every language used to
# re-read the file from disk, re-run the loudness analysis over the whole recording and re-find
# the pauses - identical work, L times. Cleared when the surah changes, so only one is held.
_REC_CACHE = {"key": None, "audio": None, "an": None, "edges": None, "edge_key": None, "shifts": {},
              "lead": None, "opening": ""}


def forget_recitation():
    _REC_CACHE.update(key=None, audio=None, an=None, edges=None, edge_key=None, shifts={},
                      lead=None, opening="")


def _apply_rec_shifts(an, edges):
    """The verse boundaries of this surah's recitation with every repair shift applied. The shifts
    live with the surah, not the language: a recitation cut that a repair moved is right for all of
    them. (Returns a new list; the cached edges are never modified.)"""
    shifts = _REC_CACHE["shifts"]
    if not shifts:
        return edges
    edges = list(edges)
    for (part, b), steps in sorted(shifts.items()):
        if part == "rec" and 1 <= b < len(edges) - 1:
            edges[b] = move_by_pauses(an, edges[b], steps, edges[b - 1] + 600, edges[b + 1] - 600)
    return edges


def _rec_audio(s):
    """The decoded recitation + its loudness analysis, shared by every language of this surah."""
    key = str(s["rec_path"])
    if _REC_CACHE["key"] != key:
        forget_recitation()
        audio = AudioSegment.from_file(str(s["rec_path"]))
        _REC_CACHE.update(key=key, audio=audio, an=Analysis(audio))
    return _REC_CACHE["audio"], _REC_CACHE["an"]


def _edge_key(cfg, num, mode, stamp="", lead=None):
    """What the verse boundaries depend on. cfg.pad is deliberately NOT in here: a 'cutoff'
    repair only widens the padding, so the boundaries it found last time still stand. `lead` is,
    though: a different count of formulas before verse 1 means different boundaries."""
    return (mode, num, cfg.basmala, cfg.min_silence, round(cfg.spacing_factor, 3), stamp, lead)


def _lead_of(s, cfg, rec, an):
    """How many stretches of speech precede verse 1 of this recitation, heard once and then shared
    by every language of the surah (the recording is the same file for all of them)."""
    if _REC_CACHE["lead"] is None:
        lead, heard = check_opening(s, cfg, rec, an)
        _REC_CACHE.update(lead=lead, opening=heard)
        if lead:
            say(s, f"   recitation: {lead} stretch(es) before verse 1"
                   + (f' - "{heard}"' if heard else "")
                   + (" - kept as intro at the head of verse 1's clip" if cfg.basmala == "keep"
                      else " - removed"))
        elif heard == "" and cfg.api_key and getattr(cfg, "opening_check", True):
            say(s, "   recitation: it opens on verse 1 itself - nothing to skip")
    return _REC_CACHE["lead"]


def make_recitation_clips(s, cfg, rec, an):
    """auto (default): cut at the silences as before; only if that fails - or repair() found the cut
    wrong - Gemini marks the verses instead. gemini: always Gemini. silence: never Gemini."""
    n, num = s["verses"], s["number"]
    force = bool(s.get("realign_note"))           # repair() says the current cut is wrong
    if force:
        _REC_CACHE["shifts"] = {}                 # a fresh alignment starts from unshifted boundaries

    def cached(key):
        return _REC_CACHE["edges"] if _REC_CACHE["edge_key"] == key and not force else None

    def keep(key, edges):
        _REC_CACHE.update(edges=edges, edge_key=key)
        return edges

    if cfg.rec_align == "silence":
        lead = _lead_of(s, cfg, rec, an)
        key = _edge_key(cfg, num, "silence", lead=lead)
        edges = cached(key) or keep(key, recitation_edges(an, n, num, cfg, lead))
        return split_recitation(rec, an, n, num, cfg, edges=_apply_rec_shifts(an, edges))
    rp = s.get("rec_map_path")
    tm = None
    if rp is not None and Path(rp).exists() and not force:
        try:
            tm = TafsirMap.model_validate_json(Path(rp).read_text("utf-8"))
            if _check_rec_map(tm, n, an.total_ms):
                tm = None                         # stale or wrong - ask again
            else:
                say(s, f"   recitation: using the saved alignment ({Path(rp).name})")
        except Exception:
            tm = None
    dsp_error = ""
    if tm is None and cfg.rec_align == "auto" and not force:
        try:
            lead = _lead_of(s, cfg, rec, an)
            key = _edge_key(cfg, num, "silence", lead=lead)
            edges = cached(key) or keep(key, recitation_edges(an, n, num, cfg, lead))
            return split_recitation(rec, an, n, num, cfg, edges=_apply_rec_shifts(an, edges))
        except SplitError as e:
            dsp_error = str(e)
            say(s, f"   ! {e}")
            say(s, "   ! cutting at the silences is not reliable for this recitation - asking Gemini to mark "
                   "every verse instead (one request, saved for every language of this surah)")
    if tm is None:
        try:
            tm = align_recitation(s, cfg, rec)
        except SplitError as e:
            raise SplitError(f"{dsp_error}  |  Gemini could not mark the verses either: {e}" if dsp_error else str(e))
    key = _edge_key(cfg, num, "map", tm.model_dump_json())
    edges = cached(key) or keep(key, map_edges(an, tm, num, cfg))
    return split_by_map(rec, an, tm, num, cfg, edges=_apply_rec_shifts(an, edges))


def prepare(s: State):
    cfg = s.get("job_cfg") or s["cfg"]
    check_stop(cfg)
    doing(s, "loading and cutting the recitation...")
    # a repair loops back through here: audio that is already in memory is not read from disk again
    rec, rec_an = _rec_audio(s)
    tafsir = s["tafsir"] if s.get("tafsir") is not None else AudioSegment.from_file(str(s["audio_path"]))
    taf_an = s["taf_an"] if s.get("taf_an") is not None else Analysis(tafsir)
    clips = make_recitation_clips(s, cfg, rec, rec_an)
    say(s, f"   recitation: {len(clips)} verse clips")
    return {"rec": rec, "rec_an": rec_an, "rec_clips": clips, "attempt": 0, "error": "",
            "job_cfg": cfg, "tafsir": tafsir, "taf_an": taf_an, "realign_note": ""}


def _media(audio):
    return {"type": "media", "mime_type": "audio/mpeg", "data": _audio_b64(audio)}


def _merge_tail(old, new, a, t0):
    """Keep the verses BEFORE `a` from the old map and replace verse a..N with Gemini's new answer for
    the excerpt that started at t0 ms (its times are relative to the excerpt, so t0 is added back)."""
    if new is None:
        return None
    keep = [v for v in sorted(old.verses, key=lambda x: x.verse) if v.verse < a]
    moved = []
    for i, v in enumerate(sorted(new.verses, key=lambda x: x.verse)):
        try:
            st, en = fmt_ms(_seconds(v.start) * 1000 + t0), fmt_ms(_seconds(v.end) * 1000 + t0)
        except ValueError:                      # leave unparsable times as they are - check() rejects them
            st, en = v.start, v.end
        moved.append(VerseSpan(verse=a + i, start=st, end=en, first_words=v.first_words))
    return TafsirMap(intro_heard=old.intro_heard, verses=keep + moved, outro_heard=new.outro_heard)


def listen(s: State):
    cfg, mp, attempt = s.get("job_cfg") or s["cfg"], s["map_path"], s["attempt"]
    check_stop(cfg)
    n, lang = s["verses"], s["lang"]
    a = s.get("relisten_from", 0)       # >= 2: only verses a..n are marked again, from the tail of the recording
    partial = a >= 2 and s.get("tmap") is not None and bool(s.get("spans"))
    if attempt == 0 and not partial and mp.exists() and not cfg.relisten:
        try:
            tm = TafsirMap.model_validate_json(mp.read_text("utf-8"))
            say(s, f"   [{lang}] using saved timing map ({mp.name})")
            return {"tmap": tm, "attempt": 1}
        except Exception:
            pass
    if not cfg.api_key:
        raise SplitError("no Gemini API key")
    if cfg.timing_source != "chat" and not cfg.usage.get("transcribe_off"):
        try:
            tm = _listen_by_transcript(s, cfg)
            mp.write_text(tm.model_dump_json(indent=2), "utf-8")      # hand-editable, reused next run
            return {"tmap": tm, "attempt": attempt + 1, "relisten_note": ""}
        except TranscribeError as e:
            if cfg.timing_source == "transcribe":
                raise
            cfg.usage["transcribe_off"] = True                          # do not retry it for every job
            say(s, f"   ! [{lang}] the transcription model cannot be used ({e}) - the chat model will "
                   f"listen to the audio instead for the rest of this run")
    text = f"Surah {s['name']}, {n} verses. Language or dialect of the recording: {lang}."
    if s.get("reference"):
        text += ("\n\nReference text (the written version of what is spoken; it may also contain the "
                 "Arabic verses, titles or numbers). Use it ONLY to recognise where each verse's "
                 "explanation begins:\n" + s["reference"])
    if s.get("error"):
        text += f"\n\nYour previous answer was rejected: {s['error']}\nListen again and fix exactly that."
    if s.get("relisten_note"):
        text += (f"\n\nA later review of the finished file found a problem with your previous "
                 f"timing: {s['relisten_note']}\nListen again and fix exactly that.")
    if partial:
        t0 = max(0, int(s["spans"][a - 1][0] * 1000) - 1500)      # a little before verse a begins
        tail = s["tafsir"][t0:]
        share = len(tail) / max(1, len(s["tafsir"]))
        say(s, f"   [{lang}] Gemini is listening again to verses {a}-{n} only "
               f"(from {fmt_ms(t0)}, {share:.0%} of the audio, try {attempt + 1})...")
        doing(s, f"Gemini is listening again to verses {a}-{n} (try {attempt + 1}) - this can take a minute...")
        new = ask(cfg, _llm(cfg.api_key, cfg.model, _temp(cfg, attempt)), [
            SystemMessage(PARTIAL_SYSTEM.format(surah=s["name"], n=n, first=a, count=n - a + 1)),
            HumanMessage(content=[_media(tail), {"type": "text", "text": text}])],
            f"listening again to verses {a}-{n}")
        tm = _merge_tail(s["tmap"], new, a, t0)
    else:
        say(s, f"   [{lang}] Gemini is listening (try {attempt + 1})...")
        doing(s, f"Gemini is listening to the audio (try {attempt + 1}) - this can take a minute...")
        tm = ask(cfg, _llm(cfg.api_key, cfg.model, _temp(cfg, attempt)), [
            SystemMessage(SYSTEM.format(n=n)),
            HumanMessage(content=[_media(s["tafsir"]), {"type": "text", "text": text}])],
            "listening to the interpretation")
    if tm is not None:
        mp.write_text(tm.model_dump_json(indent=2), "utf-8")   # hand-editable, reused next run
    return {"tmap": tm, "attempt": attempt + 1, "relisten_note": ""}


def check(s: State):
    """The guard that replaces the old 'verse count exceeds' crash: the map is validated, and if
    it is wrong Gemini gets the exact complaint and tries again."""
    cfg, tm, n = s.get("job_cfg") or s["cfg"], s.get("tmap"), s["verses"]
    check_stop(cfg)
    total = len(s["tafsir"]) / 1000
    if tm is None:
        return {"error": "the answer was not valid JSON for the requested schema."}
    v = sorted(tm.verses, key=lambda x: x.verse)
    if [x.verse for x in v] != list(range(1, n + 1)):
        a = s.get("relisten_from", 0)
        if a >= 2:                      # tail re-listen: say it in the excerpt's own terms
            return {"error": f"in the excerpt you returned {max(0, len(v) - (a - 1))} entries; I need exactly "
                             f"{n - a + 1}, numbered {a}..{n}, one per verse (do not add an entry for the "
                             f"closing line)."}
        return {"error": f"you returned {len(v)} entries; I need exactly {n}, numbered 1..{n}, "
                         f"one per verse (do not add entries for the announcement or the closing line).",
                "relisten_from": 0}
    try:
        spans = [(_seconds(x.start), _seconds(x.end)) for x in v]
    except ValueError:
        return {"error": "a time was not in MM:SS.mmm format."}
    # A complaint about verse i is LOCAL: the verses before it were fine, so only the tail has to
    # be marked again. `spans` is returned alongside the error purely as the anchor listen() needs
    # to find where verse `a` begins; assemble() never sees it, because route() goes to listen().
    for i, (a, b) in enumerate(spans, 1):
        if b - a < 0.8:
            return {"error": f"verse {i} lasts only {b - a:.1f}s - too short to be a real explanation.",
                    "spans": spans, "relisten_from": max(2, i - 1)}
        if i > 1 and a < spans[i - 2][1] - 0.3:
            return {"error": f"verse {i} starts before verse {i - 1} ends.",
                    "spans": spans, "relisten_from": max(2, i - 1)}
    if spans[0][0] > cfg.max_intro:
        return {"error": f"verse 1 starts at {spans[0][0]:.0f}s, but the opening announcement is only "
                         f"a few seconds long. Verse 1's explanation was probably swallowed into the intro.",
                "relisten_from": 0}
    if spans[-1][1] > total + 1:
        return {"error": f"verse {n} ends at {spans[-1][1]:.0f}s but the audio is only {total:.0f}s long."}
    return {"error": "", "spans": spans, "relisten_from": 0}


def route(s: State):
    if not s["error"]:
        return "assemble"
    say(s, f"   ! [{s['lang']}] map rejected: {s['error']}")
    cfg = s.get("job_cfg") or s["cfg"]
    return "listen" if s["attempt"] <= cfg.retries else "fail"


def assemble(s: State):
    cfg = s.get("job_cfg") or s["cfg"]
    a, spans, tm = s["tafsir"], s["spans"], s["tmap"]
    an = s["taf_an"]
    total, lang = len(a), s["lang"]
    check_stop(cfg)
    doing(s, "cutting and merging the verses...")
    ms = [(int(x * 1000), int(y * 1000)) for x, y in spans]
    # the pauses of the interpretation never change, so they survive every repair pass
    mids = silence_mids(an) if cfg.snap_window > 0 else []
    # start may only move EARLIER and end only LATER, so a word of verse 1 can never be cut off
    b0 = max(0, snap(ms[0][0] - cfg.edge_pad, mids, cfg.snap_window, -1))
    b1 = min(total, snap(ms[-1][1] + cfg.edge_pad, mids, cfg.snap_window, +1))
    cuts, weak = [], []
    shifts = s.get("cut_shift") or {}          # cuts a repair moved by whole pauses: {("taf", b): steps}
    for i, ((_, e1), (s2, _)) in enumerate(zip(ms[:-1], ms[1:]), 1):
        lo = (cuts[-1] if cuts else b0) + 300
        c, pause = place_cut(an, e1, s2, cfg.boundary_tol, lo, b1 - 300)
        if not pause:
            weak.append(i)
        if shifts.get(("taf", i)):
            c = move_by_pauses(an, c, shifts[("taf", i)], lo, b1 - 300)
        cuts.append(max(lo, min(c, b1 - 300)))
    if weak:
        say(s, f"   [{lang}] {len(weak)} of {len(cuts)} verse boundaries have no clear pause between the "
               f"explanations (cut at the quietest instant) - after verse "
               + ", ".join(map(str, weak[:15])) + (" ..." if len(weak) > 15 else ""))

    say(s, f"   [{lang}] removed intro ({b0 / 1000:.1f}s): \"{tm.intro_heard}\"")
    say(s, f"   [{lang}] removed outro ({(total - b1) / 1000:.1f}s): \"{tm.outro_heard}\"")
    for v in sorted(tm.verses, key=lambda x: x.verse)[:2] + [max(tm.verses, key=lambda x: x.verse)]:
        say(s, f"   [{lang}] verse {v.verse:>3} starts: \"{v.first_words}\"")
    if cfg.dry_run:
        say(s, f"   [{lang}] dry run: {len(ms)} verses, {fmt_ms(b0)} -> {fmt_ms(b1)}")
        return {"output": "dry run"}

    edges = [b0] + cuts + [b1]
    # Bring the interpretation into the recitation's format ONCE, not once per clip on every
    # pass: a repair re-assembles up to verify_retries+1 times, and every pass used to convert
    # all ~570 interpretation clips again. Cut positions come from `an` (built from the original
    # audio, so no decision moves); this only changes which buffer the clips are sliced from.
    rec = s["rec"]
    fmt = (rec.frame_rate, rec.channels, rec.sample_width)
    cached = s.get("taf_fmt")
    a_fmt = cached[1] if cached and cached[0] == fmt else to_format(a, rec)
    clips = cut_all(a_fmt, an, edges, cfg.pad)
    # pydub's silence is always mono/16-bit; make it the right format once, not 570 times
    pv = to_format(AudioSegment.silent(cfg.gap_after_verse, frame_rate=rec.frame_rate), rec)
    pt = to_format(AudioSegment.silent(cfg.gap_after_tafsir, frame_rate=rec.frame_rate), rec)
    # keep the exact ms span of each verse's [recitation, gap, interpretation, gap] block, so the
    # final check (below) can review it in verse-numbered chunks rather than blindly re-listening
    # to the whole file, and so a repair only has to touch the settings, not re-derive any of this.
    # part_bounds goes one level finer - where the recitation ends and the interpretation begins -
    # which is what lets the check listen to the junctions instead of the whole recording.
    rec_clips = s["rec_clips"]
    parts, unit_bounds, part_bounds, cursor = [], [], [], 0
    for verse_num, (verse, tafsir) in enumerate(zip(rec_clips, clips), 1):
        start = cursor
        marks = []
        for seg in (verse, pv, tafsir, pt):
            marks.append((cursor, cursor + len(seg)))
            parts.append(seg)
            cursor += len(seg)
        unit_bounds.append((start, cursor, verse_num))
        part_bounds.append({"verse": verse_num, "rec": marks[0], "taf": marks[2], "next_rec": None})
    for k in range(len(part_bounds) - 1):          # where the NEXT verse's recitation starts (for the hand-over check)
        part_bounds[k]["next_rec"] = part_bounds[k + 1]["rec"]
    out = concat(parts, s["rec"])
    defects = local_defects(rec_clips, clips, cfg)
    return {"assembled_audio": out, "unit_bounds": unit_bounds, "part_bounds": part_bounds,
            "local_problems": defects, "taf_fmt": (fmt, a_fmt)}


# --------------------------------------------------------------------------- #
# Final check: Gemini reviews the ACTUAL assembled file, not just its inputs   #
# --------------------------------------------------------------------------- #
class VerseProblem(BaseModel):
    verse: int = Field(description="verse number this problem is about")
    kind: str = Field(description="exactly one of: 'cutoff' - a word is missing at the start or "
        "end of the recitation or the interpretation; 'gap' - the silence before or after the "
        "interpretation is noticeably longer or shorter than a natural pause; 'mismatch' - the "
        "interpretation heard does not explain THIS verse (it's for a different verse, or an "
        "unrelated/leftover fragment)")
    issue: str = Field(description="briefly, what you actually heard")
    heard_verse: int = Field(description="only for 'mismatch': the number of the verse of this surah that the "
        "interpretation you heard is actually about, or 0 if you cannot tell. Always 0 for the other kinds.")


class FinalCheck(BaseModel):
    problems: list[VerseProblem] = Field(description="every verse where something is audibly "
        "wrong with the recitation -> interpretation pairing or the pause between them; empty "
        "if everything in this chunk is correct")


class EdgeProblem(BaseModel):
    verse: int = Field(description="the verse that OWNS the edge where the problem is heard - the verse "
        "whose recitation or interpretation starts or ends there")
    kind: str = Field(description="exactly one of: 'mismatch' - the interpretation is not the "
        "explanation of the verse just recited; 'spill' - a word or phrase of the NEIGHBOURING verse is "
        "audible at this edge; 'cutoff' - a word of THIS verse is missing at this edge")
    edge: str = Field(description="for 'spill' and 'cutoff': exactly one of recitation_start, "
        "recitation_end, interpretation_start, interpretation_end. Empty for 'mismatch'.")
    issue: str = Field(description="briefly, what you actually heard")
    heard_verse: int = Field(description="only for 'mismatch': the verse of this surah the interpretation "
        "actually explains, or 0. Always 0 for the other kinds.")


class EdgeCheck(BaseModel):
    problems: list[EdgeProblem] = Field(description="every problem heard at the joins of these excerpts; "
        "empty if everything is correct")


def local_defects(rec_clips: Clips, taf_clips: Clips, cfg):
    """'cutoff' and 'gap' are physical, not semantic: the program chose every cut and inserted
    every silence itself, so it can MEASURE them instead of paying audio tokens to ask whether
    they sound wrong. Only 'mismatch' - is this the right verse's explanation? - actually needs
    ears, so that is all the audio check is left to do.

    What is reported:
      * a clip with nothing audible in it at all  -> dead air where a verse should be ('gap')
      * a junction whose real silence exceeds what the settings can produce -> ('gap')
      * speech running right up to a cut          -> a word was sliced ('cutoff')
    The first two cannot false-fire (they are arithmetic on values the program set). The third is
    a threshold judgement, so by default it is logged but left to the audio check to confirm;
    set cfg.local_cutoff to act on it directly."""
    probs, n = [], len(rec_clips)
    # the most silence the settings can possibly produce at a junction, plus a wide margin
    ceiling = 2 * cfg.pad + max(cfg.gap_after_verse, cfg.gap_after_tafsir) + 600

    def add(verse, kind, issue):
        probs.append(VerseProblem(verse=verse, kind=kind, issue=issue, heard_verse=0))

    for i in range(n):
        v = i + 1
        for what, clips in (("recitation", rec_clips), ("interpretation", taf_clips)):
            if i >= len(clips.lead):
                continue
            if clips.lead[i] is None or clips.level[i] < SILENT_DB:
                add(v, "gap", f"the {what} clip for verse {v} holds no audible speech "
                              f"({len(clips.segs[i]) / 1000:.1f}s at {clips.level[i]:.0f} dB below "
                              f"the rest of the recording)")
        # verse -> interpretation, then interpretation -> next verse
        if i < len(taf_clips.lead):
            sil = (rec_clips.trail[i] or 0) + cfg.gap_after_verse + (taf_clips.lead[i] or 0)
            if sil > ceiling:
                add(v, "gap", f"{sil / 1000:.1f}s of silence between the recitation and the "
                              f"interpretation of verse {v}")
            if rec_clips.trail[i] is not None and rec_clips.trail[i] < EDGE_TOUCH_MS \
                    and rec_clips.edges[i + 1] < rec_clips.edges[-1]:
                add(v, "cutoff", f"the recitation of verse {v} is still being spoken where the "
                                 f"clip ends - the cut landed inside a word")
            if taf_clips.lead[i] is not None and taf_clips.lead[i] < EDGE_TOUCH_MS \
                    and taf_clips.edges[i] > taf_clips.edges[0]:
                add(v, "cutoff", f"the interpretation of verse {v} is already being spoken at the "
                                 f"very first moment of the clip - the cut landed inside a word")
    return probs


def describe_local(probs):
    return [f"       verse {p.verse:>3}  [{p.kind}]: {p.issue}" for p in probs]


FINAL_CHECK_SYSTEM = """You are doing final quality control on a finished audio file, verses
{lo}-{hi} of Surah {surah} ({total} verses total). It should sound like, for EACH verse in this
range, in order: the Quran recitation of that verse, a short natural pause, the spoken
interpretation/explanation of THAT SAME verse, another short pause, then the next verse.

Report a problem for any verse where:
- 'cutoff': the recitation or the interpretation is cut off / missing a word at its start or end
- 'gap': the pause before or after the interpretation is a noticeably long silence, much longer
  than a natural breath - the goal is verse -> short pause -> interpretation -> short pause ->
  next verse, with no dead air
- 'mismatch': the interpretation that follows is not for this verse - wrong verse's explanation,
  or a leftover fragment of a different verse's explanation. This is a content problem, not a
  timing/padding one, and needs to be reported precisely: say what the interpretation actually
  talks about so it's clear it's the wrong one. If you can tell which verse of the surah it actually
  explains, put that verse's number in heard_verse (otherwise 0).

Do not report a problem just because a verse or its interpretation is naturally long or short -
only report defects you can actually hear."""


DIGEST_SYSTEM = """You are doing quality control on a finished Quran audio file. For each verse, the recitation of that verse is followed by the spoken interpretation of THAT SAME verse, and then the next verse begins.

The audio holds {count} verses, {lo} to {hi} of Surah {surah} ({total} verses in all). For EACH verse there are two excerpts, always in this order:
  A) the END of the verse's recitation, the pause after it, and the BEGINNING of the interpretation that follows;
  B) the END of that interpretation, the pause after it, and the BEGINNING of the next verse's recitation (for the last verse of the surah, only the end of the interpretation).
Excerpts are separated from each other by one second of silence. They are cut short on purpose, so an excerpt that starts or stops abruptly at its OUTER ends is how the file was prepared for you: never report that. What matters is what you hear at the joins INSIDE an excerpt, where one part meets the next.

Report only what you can actually hear there:
- 'mismatch': the interpretation is not the explanation of the verse just recited (a different verse's explanation, or a leftover fragment of one). Put the verse it really explains in heard_verse if you can tell, else 0.
- 'spill': a word or phrase that belongs to the NEIGHBOURING verse is audible at an edge. For example: a recitation ends with the first word of the next verse; an interpretation finishes by starting to explain the next verse; a recitation begins with the last word of the previous verse; an interpretation begins with the tail of the previous explanation.
- 'cutoff': a word of THIS verse is missing at an edge (a recitation stops before its last word, an explanation is cut off mid-sentence, a clip begins without its first word).
For 'spill' and 'cutoff', set edge to exactly one of recitation_start, recitation_end, interpretation_start, interpretation_end, and set verse to the number of the verse that OWNS that edge, meaning the verse whose recitation or interpretation starts or ends there. So a problem heard at the start of the next verse's recitation, at the end of excerpt B of verse k, belongs to verse k+1. Say briefly in issue what you heard.

For a short verse the two excerpts may run together as one continuous stretch; the same joins are there, in the same order.

Count the verses as you go: the first is verse {lo}, the next {lo2}, and so on up to {hi}."""


def _llm_final_check(key, model):
    return _chat(key, model, FinalCheck, 0.0, 0)


def _llm_edge_check(key, model):
    return _chat(key, model, EdgeCheck, 0.0, 0)


def digest_windows(p, cfg, out_len):
    """The two contiguous stretches of the FINISHED file that hold every edge of one verse's clips:
      A  end of the recitation -> pause -> start of the interpretation   (is it the right pairing?)
      B  end of the interpretation -> pause -> start of the NEXT recitation   (is the hand-over clean?)
    Both are slices of the real output, so Gemini hears exactly what will be saved. The earlier
    digest sent only A's two halves, which left the start of every recitation unheard and the end
    of every interpretation unheard unless that explanation happened to be short."""
    ra, rb = p["rec"]
    ta, tb = p["taf"]
    nxt = p.get("next_rec")
    a = (max(ra, rb - cfg.digest_rec_ms), min(tb, ta + cfg.digest_taf_ms))
    end = min(nxt[1], nxt[0] + cfg.digest_rec_head_ms) if nxt else min(out_len, tb + cfg.gap_after_tafsir)
    b = (max(ta, tb - cfg.digest_taf_tail_ms), end)
    if a[1] >= b[0]:                    # a short verse: the two stretches meet - send it once, not twice
        return [(a[0], max(a[1], b[1]))]
    return [w for w in (a, b) if w[1] > w[0]]


def build_digest(out, items, cfg):
    """Both joins of every verse in `items`, one second of silence between excerpts so they cannot
    be miscounted. Cost scales with the NUMBER of verses (about 25 s each), not with how long the
    commentary runs - the minutes of explanation between the joins say nothing about whether the
    cuts are right and cost 32 tokens a second."""
    sep = AudioSegment.silent(1000, frame_rate=out.frame_rate)
    parts = []
    for p in items:
        for x, y in digest_windows(p, cfg, len(out)):
            parts += [out[x:y], sep]
    return concat(parts[:-1], out)


def _check_digest(cfg, out, items, surah_name, total_verses):
    lo, hi = items[0]["verse"], items[-1]["verse"]
    audio = build_digest(out, items, cfg)
    return ask(cfg, _llm_edge_check(cfg.api_key, cfg.model), [
        SystemMessage(DIGEST_SYSTEM.format(count=len(items), lo=lo, lo2=lo + 1, hi=hi,
                                           surah=surah_name, total=total_verses)),
        HumanMessage(content=[_media(audio),
                              {"type": "text", "text": "Report every problem you hear at the joins "
                                                       "of these excerpts, if any."}])],
        f"junction check of verses {lo}-{hi}"), len(audio)


MAX_CHECK_CHUNK_MS = 6 * 60 * 1000    # default chunk of finished audio per request (Settings.check_chunk_min overrides)


def _chunk_units(unit_bounds, max_ms=MAX_CHECK_CHUNK_MS):
    """Group consecutive verse units into runs no longer than max_ms, never splitting a unit -
    so each chunk sent to Gemini is a clean, complete run of whole verse+interpretation blocks."""
    chunks, cur, cur_start = [], [], None
    for start, end, verse in unit_bounds:
        if cur and end - cur_start > max_ms:
            chunks.append(cur)
            cur, cur_start = [], None
        if cur_start is None:
            cur_start = start
        cur.append((start, end, verse))
    if cur:
        chunks.append(cur)
    return chunks


def _check_chunk(cfg, chunk_audio, surah_name, lo, hi, total_verses):
    data = _audio_b64(chunk_audio)
    return ask(cfg, _llm_final_check(cfg.api_key, cfg.model), [
        SystemMessage(FINAL_CHECK_SYSTEM.format(lo=lo, hi=hi, surah=surah_name, total=total_verses)),
        HumanMessage(content=[{"type": "media", "mime_type": "audio/mpeg", "data": data},
                              {"type": "text", "text": "List every problem you hear, if any."}])],
        f"final check of verses {lo}-{hi}")


_PCM_FORMATS = {2: "s16le", 4: "s32le"}     # pydub sample width (bytes) -> ffmpeg raw format
_SAVE_CHUNK = 4 << 20                       # bytes handed to ffmpeg at a time


def stream_encode(cfg, audio, dst, on_progress=None):
    """Encode `audio` to mp3 at `dst` by piping its raw PCM into ffmpeg.

    pydub's export() does something different, and it matters for long files:
      1. writes the ENTIRE PCM to a temporary .wav on disk, through Python's `wave` module,
      2. has ffmpeg encode that to a second temporary file,
      3. reads that whole file back into memory and copies it to the destination,
    and it opens the destination for writing BEFORE any of this, which empties an existing file.
    The .wav route has a hard ceiling: a WAV header stores its size in 32 bits, so the export fails
    (struct.error) once the PCM passes 4 GiB - about 6.7 hours of 44.1 kHz stereo, which the
    longest surahs can reach. Raw PCM on a pipe has no header and no ceiling.

    Same bytes in, same encoder, same flags: the mp3 produced is byte-identical to export()'s
    (test_save.py checks the md5). It is written to '<name>.part' and only moved into place once
    ffmpeg has finished, so a failed or stopped save never damages a file that already exists.

    Encoding itself is single-threaded LAME and dominates the time (~116x real time on the
    machine this was measured on); streaming does not change that. What it removes is the disk
    round-trips, the size ceiling and the blind wait: it reports progress and honours Stop."""
    import subprocess
    import sys
    import tempfile
    fmt = _PCM_FORMATS.get(audio.sample_width)
    if fmt is None or sys.byteorder != "little":
        raise SplitError(f"raw streaming does not support {audio.sample_width * 8}-bit audio "
                         f"on this machine")
    dst = Path(dst)
    part = dst.with_name(dst.name + ".part")
    cmd = [getattr(AudioSegment, "converter", None) or "ffmpeg", "-y", "-loglevel", "error",
           "-f", fmt, "-ar", str(audio.frame_rate), "-ac", str(audio.channels), "-i", "pipe:0",
           "-b:a", str(cfg.bitrate)]
    if cfg.mp3_quality is not None:
        cmd += ["-compression_level", str(int(cfg.mp3_quality))]
    cmd += ["-id3v2_version", "4", "-f", "mp3", str(part)]     # the flags pydub's export uses
    data = memoryview(audio.raw_data)
    total = max(1, len(data))
    err = tempfile.TemporaryFile()               # a file, not a pipe: it can never fill and block ffmpeg
    proc, ok = None, False
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
        try:
            for i in range(0, len(data), _SAVE_CHUNK):
                check_stop(cfg)
                proc.stdin.write(data[i:i + _SAVE_CHUNK])
                if on_progress:
                    on_progress(min(1.0, (i + _SAVE_CHUNK) / total))
            proc.stdin.close()
        except BrokenPipeError:                  # ffmpeg exited early; its own message explains why
            pass
        rc = proc.wait()
        err.seek(0)
        if rc != 0:
            raise SplitError(f"ffmpeg exited with code {rc}: "
                             f"{err.read().decode(errors='ignore').strip()[-300:] or 'no message'}")
        part.replace(dst)
        ok = True
    finally:
        if proc is not None and proc.poll() is None:     # Stop, or an error while writing
            proc.kill()
            proc.wait()
        err.close()
        if not ok and part.exists():
            part.unlink()


def _save(s, out):
    cfg = s.get("job_cfg") or s["cfg"]
    doing(s, "saving the file...")
    dst = Path(s["out_path"])
    t0, streamed, last = time.perf_counter(), False, [-1]

    def progress(f):
        pct = int(f * 100)
        if pct != last[0]:                       # one status update per percent, not per chunk
            last[0] = pct
            doing(s, f"saving the file... {pct}%")

    if cfg.stream_save and cfg.fmt == "mp3":
        try:
            stream_encode(cfg, out, dst, progress)
            streamed = True
        except Stopped:
            raise
        except Exception as e:                   # never worse than before: fall back to pydub
            say(s, f"   ! streaming save failed ({e}) - using the standard export instead")
    if not streamed:
        kw = {}
        if cfg.fmt == "mp3":
            kw["bitrate"] = cfg.bitrate
            if cfg.mp3_quality is not None:
                kw["parameters"] = ["-compression_level", str(int(cfg.mp3_quality))]
        out.export(str(dst), format=cfg.fmt, **kw)
    took = max(time.perf_counter() - t0, 1e-6)
    say(s, f"   saved -> {dst.name}  ({fmt_ms(len(out))}; written in {took:.0f}s, "
           f"{len(out) / 1000 / took:.0f}x real time)")
    return {"output": str(dst)}


def _problem_line(p, starts):
    where = f" @ {fmt_ms(starts[p.verse])} in the finished file" if p.verse in starts else ""
    extra = (f" - the interpretation heard explains verse {p.heard_verse}"
             if p.kind == "mismatch" and p.heard_verse and p.heard_verse != p.verse else "")
    edge = f" {p.edge.replace('_', ' ')}" if getattr(p, "edge", "") else ""
    return f"       verse {p.verse:>3}{where}  [{p.kind}{edge}]{extra}: {p.issue}"


def _reject_map(s, problems):
    """A timing map that produced a mismatch must not be trusted again on the next run - keep it for
    inspection under another name so the next run asks Gemini afresh."""
    mp = s["map_path"]
    if any(p.kind == "mismatch" for p in problems) and mp.exists():
        dest = Path(str(mp)[:-len(".map.json")] + ".map.rejected.json")
        mp.replace(dest)
        say(s, f"   [{s['lang']}] the timing map produced a mismatch, so it was renamed {dest.name} "
               f"(the next run asks Gemini again)")


def _ask_about_mismatch(s, cfg, out, bounds, recheck):
    """Send audio for the one question that cannot be measured. In 'digest' mode only the
    verse->interpretation junctions go out (a few seconds each); in 'full' mode the whole
    finished file does, as before."""
    lang, name, n = s["lang"], s["name"], s["verses"]
    problems = []
    if cfg.verify_mode == "full" or not s.get("part_bounds"):
        chunks = _chunk_units(bounds, int(cfg.check_chunk_min * 60 * 1000))
        if recheck is not None:
            chunks = [c for c in chunks if any(u[2] in recheck for u in c)]
        if not chunks:
            return problems
        if recheck is not None:
            share = sum(c[-1][1] - c[0][0] for c in chunks) / max(1, bounds[-1][1])
            say(s, f"   [{lang}] re-checking only verses {chunks[0][0][2]}-{chunks[-1][-1][2]} "
                   f"({share:.0%} of the file) - the rest was already confirmed")
        doing(s, "final check - listening to the finished file..." if recheck is None
                 else "final check - listening again to the repaired part...")
        for chunk in chunks:
            lo, hi = chunk[0][2], chunk[-1][2]
            result = _check_chunk(cfg, out[chunk[0][0]:chunk[-1][1]], name, lo, hi, n)
            for p in (result.problems if result else []):
                p.verse = min(max(p.verse, lo), hi)        # a verse number outside the chunk is a slip
                problems.append(p)
        return problems

    items = [p for p in s["part_bounds"] if recheck is None or p["verse"] in recheck]
    if not items:
        return problems
    per = max(1, cfg.digest_verses)
    groups = [items[i:i + per] for i in range(0, len(items), per)]
    sent = sum(y - x for p in items for x, y in digest_windows(p, cfg, len(out)))
    whole = bounds[-1][1] if bounds else 1
    say(s, f"   [{lang}] junction check: both joins of {len(items)} verse(s) in {len(groups)} request(s), "
           f"{fmt_ms(sent)} of audio instead of {fmt_ms(whole)} ({sent / max(1, whole):.0%})")
    doing(s, "final check - listening to the verse junctions...")
    for g in groups:
        lo, hi = g[0]["verse"], g[-1]["verse"]
        result, _ = _check_digest(cfg, out, g, name, n)
        for p in (result.problems if result else []):
            if p.kind not in ("mismatch", "spill", "cutoff"):
                continue
            edge = getattr(p, "edge", "") or ""
            if p.kind == "mismatch":
                edge, p.verse = "", min(max(p.verse, lo), hi)
            else:                                  # the start of the NEXT recitation belongs to verse hi+1
                edge, p.verse = (edge if edge in _EDGES else ""), min(max(p.verse, lo), min(hi + 1, n))
            p.edge = edge
            problems.append(p)
    return problems


def verify(s: State):
    """Listens to the file that was ACTUALLY assembled - not just the inputs that went into it -
    and confirms every verse's recitation is immediately followed by its own interpretation with
    no long dead air and nothing cut off. A problem here routes to repair() instead of saving, so
    a misaligned or badly-paced file never reaches disk without either being fixed or reported.
    After a repair only the chunks that could have changed (s['recheck']) are listened to again."""
    if s.get("output"):        # dry run already finished in assemble() - nothing to verify/save
        return {}
    cfg = s.get("job_cfg") or s["cfg"]
    check_stop(cfg)
    lang, name, n = s["lang"], s["name"], s["verses"]
    out, bounds = s["assembled_audio"], s["unit_bounds"]
    attempt = s.get("verify_attempt", 0)
    if not cfg.api_key:
        doing(s, "final check - listening to the finished file...")
        say(s, f"   [{lang}] skipping final check (no Gemini API key) - saving unverified")
        return _save(s, out)
    starts = {v: st for st, _, v in bounds}
    recheck = s.get("recheck")
    problems = []

    # --- measured, free: dead air and mid-word cuts ---------------------------------------- #
    local = [p for p in (s.get("local_problems") or [])
             if recheck is None or p.verse in recheck]
    acted = [p for p in local if p.kind == "gap" or cfg.local_cutoff]
    if local:
        say(s, f"   [{lang}] measured {len(local)} defect(s) without asking Gemini:")
        for line in describe_local(local[:8]):
            say(s, line)
        if len(local) > 8:
            say(s, f"       +{len(local) - 8} more")
        if len(acted) < len(local):
            say(s, f"   [{lang}] {len(local) - len(acted)} of them are mid-word cuts - left for the "
                   f"audio check to confirm (set local_cutoff=True in Settings to act on them directly)")
    problems.extend(acted)

    # --- the part that genuinely needs ears: is this the right verse's explanation? --------- #
    if cfg.verify_mode == "local":
        say(s, f"   [{lang}] final check: measured only (verify_mode='local') - no audio sent")
    else:
        try:
            problems.extend(_ask_about_mismatch(s, cfg, out, bounds, recheck))
        except Stopped:
            raise
        except Exception as e:  # a check that fails to run should not block a file that's likely fine
            again = "" if attempt == 0 else " (problems found earlier may still be there)"
            say(s, f"   ! [{lang}] final check could not run ({e}) - saving unverified{again}")
            return _save(s, out)
    seen, merged = set(), []
    for p in problems:
        key = (p.verse, p.kind, getattr(p, "edge", ""))
        if key not in seen:
            seen.add(key)
            merged.append(p)
    problems = sorted(merged, key=lambda p: p.verse)
    if not problems:
        say(s, f"   [{lang}] final check: every verse -> interpretation pairing and pause sounds correct")
        return _save(s, out)
    say(s, f"   ! [{lang}] final check found {len(problems)} problem(s):")
    for p in problems[:12]:
        say(s, _problem_line(p, starts))
    if len(problems) > 12:
        say(s, f"       +{len(problems) - 12} more")
    msg = "; ".join(f"verse {p.verse} ({p.kind}{' at ' + p.edge.replace('_', ' ') if getattr(p, 'edge', '') else ''}): "
                    f"{p.issue}" for p in problems[:6])
    if len(problems) > 6:
        msg += f"; +{len(problems) - 6} more"
    if attempt >= cfg.verify_retries:
        _reject_map(s, problems)
        return {"error": f"final check still failing after {attempt} repair attempt(s): {msg}", "output": ""}
    return {"needs_repair": True, "problems": problems, "verify_attempt": attempt + 1, "error": msg}


def verify_route(s: State):
    return "repair" if s.get("needs_repair") else "end"


def repair_route(s: State):
    return "end" if s.get("give_up") else "prepare"


def _zone(lo, n):
    return set(range(max(1, lo), n + 1))


def _diagnose_recitation(s, cfg, first):
    """A 'mismatch' can start on the recitation side: if clip `first` (or the one before it) is not
    really that verse, every verse after it is shifted, and re-listening to the interpretation can
    never fix it. Two short clips are much cheaper to ask about than the whole recording.
    Returns (clip, verse_it_really_is, what_gemini_heard) for the first bad clip, else None."""
    if not cfg.api_key:
        return None
    n, name = s["verses"], s["name"]
    for v in sorted({max(1, first - 1), first}):
        try:
            r = ask(cfg, _llm_clip_identity(cfg.api_key, cfg.model), [
                SystemMessage(CLIP_ID_SYSTEM.format(surah=name, total=n, verse=v)),
                HumanMessage(content=[_media(s["rec_clips"][v - 1]),
                                      {"type": "text", "text": "Which verse is this clip?"}])],
                f"identifying recitation clip {v}")
        except Stopped:
            raise
        except Exception as e:
            say(s, f"   ! [{s['lang']}] could not identify recitation clip {v} ({e}) - assuming the "
                   f"interpretation side")
            return None
        if r is not None and (r.verse_heard != v or not r.complete):
            return v, r.verse_heard, r.explanation
    return None


def repair(s: State):
    """Works out what actually went wrong and applies only the fix that can work:
    - 'cutoff' / 'gap': timing/padding. Wider clip padding recovers a word cut off at an edge and
      shorter fixed gaps remove dead air. No new Gemini listening is needed (the saved timing map is
      reused); only the affected chunks are checked again.
    - 'mismatch': everything before the FIRST mismatching verse was confirmed correct, so the fault
      is at or just before it. First ask whether the RECITATION clip there really is that verse: if
      not, re-cut the recitation (re-listening to the tafsir could never help). Otherwise re-listen
      only from the verse before the first mismatch to the end and keep the earlier verses. If the
      same spot fails again, start one verse further back each time, finally the whole recording;
      when even that fails there is nothing more to try - stop and say where to listen.
    A mixed batch gets both fixes in the same attempt."""
    cfg = s.get("job_cfg") or s["cfg"]
    n, lang, attempt = s["verses"], s["lang"], s["verify_attempt"]
    problems = s.get("problems") or []
    mism = sorted((p for p in problems if p.kind == "mismatch"), key=lambda p: p.verse)
    others = [p for p in problems if p.kind != "mismatch"]
    tag = f"repair {attempt}/{cfg.verify_retries}"
    hist = dict(s.get("bad_history") or {})
    new_cfg = replace(cfg, relisten=False, verify_clips=False)   # clips were already judged on the first pass
    upd = {"needs_repair": False, "relisten_note": "", "relisten_from": 0}
    recheck, everything = set(), False

    # A word of the neighbouring verse at a clip's edge means a CUT is in the wrong place. Wider padding
    # can never fix that (it only keeps more silence INSIDE a clip's window), so those go to the cut
    # itself: it is moved to the next real pause, in the direction the edge and the kind imply.
    moves, plain = [], []
    for p in others:
        mv = edge_move(p.verse, getattr(p, "edge", ""), p.kind, n)
        (moves if mv else plain).append((p, mv) if mv else p)
    if moves:
        shifts = dict(s.get("cut_shift") or {})
        for p, (part, b, sign) in moves:
            tgt = shifts if part == "taf" else _REC_CACHE["shifts"]
            tgt[(part, b)] = tgt.get((part, b), 0) + sign
            recheck |= {b, b + 1}
            say(s, f"   [{lang}] {tag}: {p.kind} at the {p.edge.replace('_', ' ')} of verse {p.verse} - moving the "
                   f"{'interpretation' if part == 'taf' else 'recitation'} cut between verses {b} and {b + 1} "
                   f"one pause {'earlier' if sign < 0 else 'later'} ({p.issue})")
        upd["cut_shift"] = shifts
    others = plain
    if others:
        new_cfg = replace(new_cfg, pad=cfg.pad + cfg.repair_pad_step,
                          edge_pad=cfg.edge_pad + cfg.repair_pad_step,
                          gap_after_verse=max(150, int(cfg.gap_after_verse * cfg.repair_gap_shrink)),
                          gap_after_tafsir=max(300, int(cfg.gap_after_tafsir * cfg.repair_gap_shrink)))
        recheck |= {p.verse for p in others}
        say(s, f"   [{lang}] {tag}: timing fix for verse(s) {', '.join(str(p.verse) for p in others)} - "
               f"padding {new_cfg.pad}ms, gaps {new_cfg.gap_after_verse}/{new_cfg.gap_after_tafsir}ms "
               f"(no new listening needed)")

    if mism:
        first = mism[0].verse
        hist[first] = hist.get(first, 0) + 1
        say(s, f"   [{lang}] {tag}: the first mismatch is at verse {first}; the verses before it were "
               f"confirmed correct and are kept.")
        culprit = _diagnose_recitation(s, cfg, first)
        if culprit is not None:
            v, heard, why = culprit
            what = f"verse {heard}" if heard else "a mix of verses"
            if s.get("resplit_done"):
                say(s, f"   ! [{lang}] {tag}: recitation clip {v} is still not verse {v} after re-cutting ({why}).")
                return {**upd, "job_cfg": new_cfg, "give_up": True, "output": "",
                        "error": f"recitation clip {v} is really {what} ({why}) even after cutting it again - check "
                                 f"that this is the right recitation file for this surah, or lower "
                                 f"`min_silence` in Settings"}
            if cfg.rec_align == "silence":          # Gemini is switched off for the recitation: only nudge the spacing
                say(s, f"   [{lang}] {tag}: the cause is on the RECITATION side - clip {v} is really {what} "
                       f"({why}). Re-listening to the interpretation could not fix that, so the recitation is "
                       f"re-cut with wider spacing between cuts instead.")
                new_cfg = replace(new_cfg, spacing_factor=cfg.spacing_factor + 0.2, verify_clips=True)
            else:
                say(s, f"   [{lang}] {tag}: the cause is on the RECITATION side - clip {v} is really {what} "
                       f"({why}). Re-listening to the interpretation could not fix that, so Gemini is asked "
                       f"to mark every verse of the recitation instead (one request, saved for the other "
                       f"languages of this surah).")
                upd["realign_note"] = f"recitation clip {v} was found to be {what} ({why})"
            upd["resplit_done"] = True
            everything = True           # every recitation clip is cut again, so everything is checked again
        else:
            a = max(1, first - hist[first])          # 1st time one verse back, every repeat one further
            if a >= 2:
                say(s, f"   [{lang}] {tag}: the recitation clips are right, so the fault is in the "
                       f"interpretation timing - re-listening only from verse {a} to the end.")
                upd["relisten_from"] = a
                recheck |= _zone(a, n)
            else:
                if s.get("full_relisten_done"):
                    at = fmt_ms(int(s["spans"][first - 1][0] * 1000))
                    say(s, f"   ! [{lang}] {tag}: verse {first} still does not match after listening to the "
                           f"whole recording again.")
                    _reject_map(s, mism)
                    return {**upd, "job_cfg": new_cfg, "give_up": True, "output": "",
                            "error": f"verse {first}'s interpretation still does not match after re-listening "
                                     f"to the whole recording ({mism[0].issue}). The recording itself probably "
                                     f"lacks or misorders that verse's explanation - listen around {at} of the "
                                     f"interpretation audio"}
                say(s, f"   [{lang}] {tag}: re-listening to the whole recording.")
                new_cfg = replace(new_cfg, relisten=True)
                upd["full_relisten_done"] = True
                everything = True
            upd["relisten_note"] = "; ".join(
                f"verse {p.verse} - {p.issue}"
                + (f" (it explains verse {p.heard_verse})" if p.heard_verse and p.heard_verse != p.verse else "")
                for p in mism[:4])
    upd.update(job_cfg=new_cfg, recheck=None if everything else sorted(recheck), bad_history=hist)
    return upd


def build_graph():
    g = StateGraph(State)
    for name, fn in (("prepare", prepare), ("listen", listen), ("check", check),
                     ("assemble", assemble), ("verify", verify), ("repair", repair)):
        g.add_node(name, fn)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "listen")
    g.add_edge("listen", "check")
    g.add_conditional_edges("check", route, {"listen": "listen", "assemble": "assemble", "fail": END})
    g.add_edge("assemble", "verify")
    g.add_conditional_edges("verify", verify_route, {"repair": "repair", "end": END})
    g.add_conditional_edges("repair", repair_route, {"prepare": "prepare", "end": END})
    return g.compile()


# --------------------------------------------------------------------------- #
# YouTube playlist -> recitation files                                         #
# --------------------------------------------------------------------------- #
class _YtLog:
    """Receives yt-dlp's own messages; keeps the useful ones (which item, skipped, errors)."""

    def __init__(self, cfg, counts):
        self.cfg, self.counts = cfg, counts

    @staticmethod
    def _clean(m):
        return re.sub(r"\x1b\[[0-9;]*m", "", str(m))

    def debug(self, m):
        m = self._clean(m)
        item = re.search(r"Downloading item (\d+) of (\d+)", m)
        if item:
            self.cfg.status(f"Item {item[1]} of {item[2]}: reading video info...")
            self.cfg.log(f"\n▶ item {item[1]} of {item[2]}")
        elif "already been recorded in the archive" in m:
            self.counts["skipped"] += 1
            self.cfg.log("   ↷ skipped - already downloaded before")
        elif re.search(r"Downloading playlist|Extracting URL|Finished downloading playlist", m):
            text = re.sub(r"^\[[^\]]+\]\s*", "", m)
            self.cfg.log("   " + text)

    info = debug

    def warning(self, m):
        self.cfg.log(f"   ! {self._clean(m)}")

    def error(self, m):
        self.counts["failed"] += 1
        self.cfg.log(f"   ✗ {self._clean(m)}")


def _find_deno():
    """Deno path: PATH first, then the folder the PowerShell installer uses. Looking in the
    default folder too means it works even if this app was started before Deno was installed
    (a running program does not see a PATH changed afterwards)."""
    import os
    import shutil
    found = shutil.which("deno")
    if found:
        return found
    for base in (os.environ.get("DENO_INSTALL"), str(Path.home() / ".deno")):
        if base:
            for name in ("deno.exe", "deno"):
                cand = Path(base) / "bin" / name
                if cand.is_file():
                    return str(cand)
    return None


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
    url = cfg.playlist_url.strip()
    m = re.search(r"[?&]list=([\w-]+)", url)
    if m and "/playlist?list=" not in url:          # watch?v=..&list=.. -> the plain playlist page
        url = f"https://www.youtube.com/playlist?list={m.group(1)}"
        cfg.log(f"   using the playlist page: {url}")
    if m and m.group(1).startswith(("RD", "UL")):
        cfg.log("   ! this looks like an auto-generated Mix, not a real playlist - YouTube often blocks these")
    cfg.log(f"Downloading recitations into {dest} (yt-dlp {yt_dlp.version.__version__}) ...")

    counts = {"skipped": 0, "failed": 0, "done": 0}

    def stop_filter(info, *, incomplete=False):     # asked before every video: reject = stop the playlist
        return "stopped by user" if cfg.stop.is_set() else None

    seen = set()

    def where(d):
        info = d.get("info_dict") or {}
        idx = info.get("playlist_index")
        total = info.get("n_entries") or info.get("playlist_count")
        return idx, total, (info.get("title") or "")[:70]

    def hook(d):                                   # the download itself
        if cfg.stop.is_set():
            raise yt_dlp.utils.DownloadCancelled()
        idx, total, title = where(d)
        tag = f"{idx}/{total}" if idx and total else "?"
        if d.get("status") == "downloading":
            got, size = d.get("downloaded_bytes") or 0, d.get("total_bytes") or d.get("total_bytes_estimate")
            pct = got / size * 100 if size else 0
            if tag not in seen:
                seen.add(tag)
                cfg.log(f"   ⬇ downloading {tag}: {title}")
            cfg.status(f"⬇ Downloading {tag}: {title} - {pct:.0f}%")
            if idx and total:
                cfg.progress(idx - 1 + pct / 100, total)
        elif d.get("status") == "finished":
            cfg.log(f"   ✔ downloaded {tag}: {title}")
            cfg.status(f"🎧 Converting {tag} to mp3: {title}")

    def pp_hook(d):                                # the mp3 conversion
        if d.get("postprocessor") == "ExtractAudio" and d.get("status") == "finished":
            counts["done"] += 1
            idx, total, title = where(d)
            cfg.log(f"   ✔ ready (mp3): {title}")
            if idx and total:
                cfg.progress(idx, total)

    opts = {"format": "bestaudio/best", "quiet": True, "no_warnings": True, "ignoreerrors": True,
            "outtmpl": str(dest / "%(playlist_index)s - %(title)s.%(ext)s"),
            "windowsfilenames": True, "download_archive": str(dest / ".downloaded.txt"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3",
                                "preferredquality": "192"}],
            "logger": _YtLog(cfg, counts), "progress_hooks": [hook],
            "postprocessor_hooks": [pp_hook],
            "match_filter": stop_filter, "break_on_reject": True,
            "no_color": True, "remote_components": ["ejs:github"],
            "retries": 10, "fragment_retries": 10, "extractor_retries": 3,
            "sleep_interval": 1, "max_sleep_interval": 3}
    if cfg.cookies_browser:
        opts["cookiesfrombrowser"] = (cfg.cookies_browser,)
    deno = _find_deno()
    if deno:
        opts["js_runtimes"] = {"deno": {"path": deno}}
        cfg.log(f"   using Deno: {deno}")
    else:
        cfg.log("   ! Deno not found - install it (winget install DenoLand.Deno) and restart the app")
    # YouTube answers "403 Forbidden" for some videos/clients, often only some of the time. A failed
    # video is not recorded in the archive, so every extra pass re-tries ONLY the failed ones,
    # each time through a different YouTube client.
    passes = [None, {"player_client": ["tv"]}, {"player_client": ["web_safari"]},
              {"player_client": ["mweb"]}]
    cfg.status("Reading the playlist...")
    skipped_first = 0
    for n, clients in enumerate(passes, 1):
        counts["failed"] = 0
        seen.clear()
        o = dict(opts)
        if clients:
            o["extractor_args"] = {"youtube": clients}
        if n > 1:
            cfg.log(f"\n--- pass {n}/{len(passes)}: retrying the failed videos through the "
                    f"'{clients['player_client'][0]}' YouTube client")
        try:
            with yt_dlp.YoutubeDL(o) as ydl:
                ydl.download([url])
        except yt_dlp.utils.DownloadCancelled:
            break
        if cfg.stop.is_set():
            break
        if n == 1:
            skipped_first = counts["skipped"]
        counts["skipped"] = skipped_first
        if not counts["failed"] or n == len(passes):
            break
        cfg.log(f"   {counts['failed']} video(s) failed (YouTube 403/blocked) - waiting a few seconds, then trying again")
        cfg.status("Some videos were blocked - retrying with another YouTube client...")
        cfg.stop.wait(5)                     # a wait that Stop can interrupt
    cfg.status("Download stopped" if cfg.stop.is_set() else "Download finished")
    cfg.log(f"\nDownload summary: {counts['done']} new, {counts['skipped']} already had, "
            f"{counts['failed']} still failing")
    if counts["failed"]:
        cfg.log("   ! YouTube keeps blocking some videos. Try, in this order:  1) press the button again "
                "later (the blocking is often temporary; finished videos are never downloaded twice)  "
                "2) pip install -U --pre \"yt-dlp[default]\"  3) choose your browser in 'Browser cookies'")
    files = _files(dest, AUDIO_EXTS)
    unknown = [p.name for p in files if surah_of(p.stem) is None]
    cfg.log(f"Recitations folder now has {len(files)} audio file(s).")
    if not files:
        cfg.log("   ! nothing was downloaded. Try: 1) pip install -U --pre \"yt-dlp[default]\"  "
                "2) make sure the playlist is Public or Unlisted (open the link in a private window)  "
                "3) pick your browser in 'Browser cookies' below and try again")
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


def cost_report(cfg):
    u = cfg.usage
    if not u["calls"] and not u.get("transcribed_s"):
        return ["no Gemini calls were made"]
    lines = []
    if u.get("transcribed_s"):
        lines.append(f"Transcription: {u['transcribed_s'] / 60:,.1f} min of audio (billed by duration)")
    if u["calls"]:
        lines.append(f"Gemini: {u['calls']} call(s), {u['input']:,} input + {u['output']:,} output tokens")
    if u["thinking"]:
        lines.append(f"        {u['thinking']:,} of the output tokens were thinking tokens")
    if u["cached"]:
        lines.append(f"        {u['cached']:,} input tokens were served from cache")
    if u["by_step"]:
        worst = sorted(u["by_step"].items(), key=lambda kv: -kv[1])
        lines.append("        by step: " + ", ".join(f"{k} {v:,}" for k, v in worst))
    return lines


def run_batch(cfg: Settings):
    forget_files()          # a previous run's listing must not hide files added since
    forget_recitation()
    if cfg.playlist_url.strip():
        try:
            download_recitations(cfg)
        except Exception as e:
            cfg.log(f"✗ download failed: {e}")
            if cfg.download_only:
                return []
        forget_files()      # the download just wrote new files into the recitations folder
    if cfg.stop.is_set():
        cfg.status("Stopped")
        cfg.log("⏹ stopped by you")
        return []
    if cfg.download_only:
        return []
    Path(cfg.final_dir).mkdir(parents=True, exist_ok=True)
    graph, jobs, results = build_graph(), discover(cfg), []
    cfg.log(f"{len(jobs)} job(s) found")
    for i, (num, lang, folder, rec, audio) in enumerate(jobs, 1):
        if cfg.stop.is_set():
            cfg.log("⏹ stopped by you")
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
                    "rec_map_path": Path(cfg.final_dir) / f"{num:03d}_{name}.recitation.json",
                    "reference": read_reference(find_reference(cfg, num, folder, lang))},
                    {"recursion_limit": 100})
                status = "ok" if st.get("output") else f"NEEDS REVIEW: {st.get('error')}"
        except Stopped:
            cfg.log("   ⏹ stopped by you")
            results.append((label, "stopped"))
            break
        except Exception as e:      # one bad file never stops the batch
            status = f"error: {e}"
        cfg.log(f"   -> {status}")
        cfg.status(f"{label}: {status}")
        results.append((label, status))
        cfg.progress(i, len(jobs))
    forget_recitation()     # release the last surah's audio
    cfg.status("Finished")
    cfg.log("\n=========== SUMMARY ===========")
    for label, status in results:
        cfg.log(f"{label:<32} {status}")
    for line in cost_report(cfg):
        cfg.log(line)
    return results
