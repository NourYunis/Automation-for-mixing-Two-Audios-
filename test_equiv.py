"""Checks the refactor did not change what the numbers come out as.

pydub is not installed here, so a minimal stand-in provides just the bits engine.py touches:
a raw int16 buffer with ms slicing. The ORIGINAL _frame_db/_gaps/_find_gaps/trim_edges/
_spread_cuts are re-implemented verbatim below and compared against the new Analysis path.
"""
import sys
import types

import numpy as np

# ---- minimal pydub stand-in ------------------------------------------------------------- #
SR = 16000


class Seg:
    def __init__(self, samples):
        self.samples = np.asarray(samples, dtype=np.int16)
        self.frame_rate, self.channels, self.sample_width = SR, 1, 2

    def __len__(self):                       # duration in ms
        return int(len(self.samples) * 1000 / SR)

    def __getitem__(self, sl):
        a = int((sl.start or 0) * SR / 1000)
        b = len(self.samples) if sl.stop is None else int(sl.stop * SR / 1000)
        return Seg(self.samples[max(0, a):max(0, b)])

    def set_channels(self, _):
        return self

    def set_frame_rate(self, _):
        return self

    def get_array_of_samples(self):
        return self.samples

    def fade_in(self, _):
        return self

    def fade_out(self, _):
        return self


fake = types.ModuleType("pydub")
fake.AudioSegment = Seg
sys.modules["pydub"] = fake
for name in ("langchain_core", "langchain_core.messages", "langchain_google_genai",
             "langgraph", "langgraph.graph", "pydantic"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["langchain_core.messages"].HumanMessage = object
sys.modules["langchain_core.messages"].SystemMessage = object
sys.modules["langchain_google_genai"].ChatGoogleGenerativeAI = object
sys.modules["langgraph.graph"].END = sys.modules["langgraph.graph"].START = None
sys.modules["langgraph.graph"].StateGraph = object


class _BM:
    """Enough of pydantic.BaseModel for the tests: keyword init and attribute access."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"{type(self).__name__}({self.__dict__})"


sys.modules["pydantic"].BaseModel = _BM
sys.modules["pydantic"].Field = lambda **kw: None

import engine  # noqa: E402


# ---- the ORIGINAL implementations, copied verbatim from the uploaded engine.py ---------- #
def old_frame_db(audio):
    mono = audio.set_channels(1).set_frame_rate(16000)
    x = np.asarray(mono.get_array_of_samples(), dtype=np.float32)
    n = len(x) // 160
    if n == 0:
        return np.zeros(0)
    rms = np.sqrt(np.mean(x[: n * 160].reshape(n, 160) ** 2, axis=1)) + 1e-9
    return 20 * np.log10(rms / np.percentile(rms, 99))


def old_gaps(db, min_ms, rel_db):
    if len(db) == 0:
        return []
    d = np.diff(np.concatenate(([0], (db < -rel_db).astype(np.int8), [0])))
    return [(int(s) * 10, int(e) * 10) for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0])
            if s != 0 and e != len(db) and (e - s) * 10 >= min_ms]


def old_find_gaps(db, need, start_ms):
    for min_ms in [start_ms] + [m for m in (700, 500, 350, 250, 180) if m < start_ms]:
        for rel in (42, 36, 30, 26):
            g = old_gaps(db, min_ms, rel)
            if len(g) >= need:
                return g
    raise engine.SplitError("no")


def old_trim_edges(seg, keep_ms, rel_db=40):
    loud = np.where(old_frame_db(seg) > -rel_db)[0]
    if len(loud) == 0:
        return seg
    a = max(0, int(loud[0]) * 10 - keep_ms)
    b = min(len(seg), (int(loud[-1]) + 1) * 10 + keep_ms)
    return seg[a:b]


def old_split_recitation_edges(audio, verses, number, cfg):
    skip = cfg.basmala != "none" and number not in (1, 9)
    total = len(audio)
    gaps = old_find_gaps(old_frame_db(audio), verses if skip else verses - 1, cfg.min_silence)
    head = 0
    if skip:
        head = (gaps[0][0] + gaps[0][1]) // 2
        gaps = gaps[1:]
    min_spacing = max(1000, (total / verses) * cfg.spacing_factor)
    cuts = engine._spread_cuts(gaps, verses - 1, min_spacing)
    return ([0] if cfg.basmala == "keep" else [head]) + cuts + [total]


# ---- synthetic recitation: N spoken verses separated by real pauses --------------------- #
def make_audio(verse_ms, pause_ms=1200, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for k, d in enumerate(verse_ms):
        n = int(d * SR / 1000)
        env = 0.5 + 0.5 * np.sin(np.linspace(0, 40, n))
        parts.append((rng.normal(0, 2500, n) * env).astype(np.int16))
        if k != len(verse_ms) - 1:
            p = int(pause_ms * SR / 1000)
            parts.append(rng.normal(0, 3, p).astype(np.int16))
    return Seg(np.concatenate(parts))


fails = []


def check(name, a, b):
    ok = a == b
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        fails.append(name)
        print(f"      old={a!r}\n      new={b!r}")


def old_was_a_guess(audio, verses, number, cfg):
    """True where the ORIGINAL code could not place the cuts under its own spacing floor and fell
    back to 'the longest pauses, wherever they are'. That fallback is the one thing the new code
    deliberately does NOT reproduce: it relaxes the floor instead, so a genuinely short verse keeps
    its own boundary. Everywhere else the numbers must still be identical."""
    skip = cfg.basmala != "none" and number not in (1, 9)
    gaps = old_find_gaps(old_frame_db(audio), verses if skip else verses - 1, cfg.min_silence)
    if skip:
        gaps = gaps[1:]
    report = {}
    engine._spread_cuts(gaps, verses - 1,
                        max(1000, (len(audio) / verses) * cfg.spacing_factor), report)
    return bool(report.get("crowded"))


def compare_edges(label, audio, an, nv, num, cfg):
    """Identical to the old code, except where the old code was guessing - there, check instead
    that the new cut is a real one: the right number of clips, strictly increasing, every internal
    cut sitting in an actual pause."""
    if not old_was_a_guess(audio, nv, num, cfg):
        check(label, old_split_recitation_edges(audio, nv, num, cfg),
              engine.recitation_edges(an, nv, num, cfg))
        return
    new = engine.recitation_edges(an, nv, num, cfg)
    mids = [(g[0] + g[1]) // 2 for g in an.gaps(cfg.min_silence, 26)]
    ok = (len(new) - 1 == nv and all(b > a for a, b in zip(new, new[1:]))
          and all(any(abs(c - m) <= 20 for m in mids) for c in new[1:-1]))
    check(label + " [old fell back to a guess; new places every cut in a real pause]", True, ok)


def run_all():
  cases = [
    ("even verses", [3000] * 8, 1200, 0),
    ("uneven verses", [1200, 5200, 800, 3000, 2200, 900, 4100], 1000, 1),
    ("many short", [700] * 14, 950, 2),
    ("long tail", [2000, 2000, 9000, 1500, 1500, 1500], 1500, 3),
  ]

  for label, verses, pause, seed in cases:
      audio = make_audio(verses, pause, seed)
      an = engine.Analysis(audio)

      check(f"[{label}] db curve", old_frame_db(audio).round(6).tolist(), an.db().round(6).tolist())

      for rel in (42, 36, 30, 26):
          for min_ms in (900, 500, 250):
              check(f"[{label}] gaps rel={rel} min={min_ms}",
                    old_gaps(old_frame_db(audio), min_ms, rel), an.gaps(min_ms, rel))

      need = len(verses) - 1
      check(f"[{label}] find_gaps", old_find_gaps(old_frame_db(audio), need, 900),
            engine._find_gaps(an, need, 900))

      check(f"[{label}] silence_mids",
            [(s + e) // 2 for s, e in old_gaps(old_frame_db(audio), 200, 36)], engine.silence_mids(an))

      # trim_edges: same kept window for arbitrary sub-ranges
      for (a, b) in [(0, len(audio)), (500, 7000), (1230, 5000), (2000, 2600)]:
          old_seg = old_trim_edges(audio[a:b], 150)
          x, y, lead, trail = engine.trim_window(an, a, b, 150)
          check(f"[{label}] trim window {a}-{b} length", len(old_seg), y - x)

      class Cfg:
          basmala, min_silence, spacing_factor, pad, verify_clips = "keep", 900, 0.4, 150, False

      # the first block stands in for the basmala, so a "keep"/"drop" run has the extra pause it needs
      nv = len(verses) - 1
      for bas, num in (("keep", 5), ("drop", 5), ("none", 5), ("keep", 1), ("keep", 9)):
          Cfg.basmala = bas
          compare_edges(f"[{label}] edges basmala={bas} surah={num}", audio, an, nv, num, Cfg)
      for sf in (0.2, 0.6, 0.9):
          Cfg.basmala, Cfg.spacing_factor = "keep", sf
          compare_edges(f"[{label}] edges spacing={sf}", audio, an, nv, 5, Cfg)
      Cfg.basmala, Cfg.spacing_factor = "keep", 0.4

  print()
  print("FAILURES:", fails if fails else "none")
  return fails


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
