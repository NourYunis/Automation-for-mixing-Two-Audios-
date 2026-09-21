"""Tests for the isti'adha fix.   python test_opening.py

Reproduces the real failure first: a recitation of Al-Fatiha that opens with the isti'adha. Surahs 1
and 9 were hard-coded as "nothing precedes verse 1", so the pause after the isti'adha was spent as a
verse boundary, verse 1's clip held the isti'adha, and every verse after it ran one clip late -
which is what put verse 1's explanation in the wrong place.

Covered against the real engine code: the leading-segment walk, is_lead()'s surah rules, the edges
with and without a known lead, the 'drop' variant, map_edges no longer forcing surah 1 to start at
0, and the fact that a run WITHOUT an isti'adha is unchanged.
"""
import sys

import numpy as np

from test_equiv import SR, Seg, engine

rng = np.random.default_rng(3)
fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)
        if detail:
            print("      " + str(detail))


def speech(sec):
    n = int(sec * SR)
    env = 0.5 + 0.5 * np.sin(np.linspace(0, 40, n))
    return (rng.normal(0, 2800, n) * env).astype(np.int16)


def quiet(sec):
    return rng.normal(0, 3, int(sec * SR)).astype(np.int16)


def build(segments, pauses):
    """One recitation, plus the true (start_ms, end_ms) of every spoken stretch."""
    parts, marks, t = [], [], 0.0
    for i, d in enumerate(segments):
        parts.append(speech(d))
        marks.append((int(t * 1000), int((t + d) * 1000)))
        t += d
        if i < len(segments) - 1:
            parts.append(quiet(pauses[i]))
            t += pauses[i]
    return Seg(np.concatenate(parts)), marks


class Cfg:
    basmala, min_silence, spacing_factor = "keep", 900, 0.4
    pad, edge_pad, verify_clips = 150, 250, False
    api_key, opening_check, opening_max_ms = "", True, 20000


# Al-Fatiha as a reciter actually reads it: isti'adha, then the basmala (which IS verse 1), then
# verses 2..7. The pause before verse 7 is the shortest, so it is the one _spread_cuts drops when
# the count is wrong - which is what makes the old failure deterministic here.
FATIHA = [3.0, 4.0, 3.5, 5.0, 2.5, 4.5, 3.0, 6.0]          # isti'adha + 7 verses
PAUSES = [1.30, 1.25, 1.20, 1.20, 1.15, 1.10, 0.95]
audio, marks = build(FATIHA, PAUSES)
an = engine.Analysis(audio)
ISTI, VERSES = marks[0], marks[1:]                          # verse k is VERSES[k - 1]


def clip_of(edges, k):
    return edges[k - 1], edges[k]


def holds(edges, k, span, slack=400):
    """Clip k covers that spoken stretch (its own words are inside it, start to finish)."""
    a, b = clip_of(edges, k)
    return a <= span[0] + slack and b >= span[1] - slack


def test_the_old_failure():
    old = engine.recitation_edges(an, 7, 1, Cfg)            # lead=None -> the old assumption
    check("the old assumption still produces 7 clips (nothing crashed - that was the problem)",
          len(old) - 1 == 7, old)
    check("REPRODUCED: verse 1's clip is the isti'adha, not the basmala",
          holds(old, 1, ISTI) and not holds(old, 1, VERSES[0]), (old, ISTI, VERSES[0]))
    check("...and so every later verse sits one clip late",
          holds(old, 2, VERSES[0]), old)
    shifted = [k for k in range(1, 8) if not holds(old, k, VERSES[k - 1])]
    check("...6 of 7 verses end up on a clip that is not their own", shifted == [1, 2, 3, 4, 5, 6],
          shifted)
    a, b = clip_of(old, 7)
    check("...and the last clip swallows verses 6 AND 7, because one cut was spent on the isti'adha",
          a <= VERSES[5][0] and b >= VERSES[6][1] - 400, (a, b, VERSES[5], VERSES[6]))


def test_the_fix():
    new = engine.recitation_edges(an, 7, 1, Cfg, lead=1)
    check("with the isti'adha counted, there are still 7 clips", len(new) - 1 == 7, new)
    wrong = [k for k in range(1, 8) if not holds(new, k, VERSES[k - 1])]
    check("every verse is now on its own clip", not wrong, f"wrong: {wrong}  edges={new}")
    check("verse 1's clip is the basmala", holds(new, 1, VERSES[0]), new)
    check("the isti'adha is KEPT, prepended at the head of verse 1's clip (basmala=keep)",
          new[0] == 0 and holds(new, 1, ISTI), new)


def test_drop():
    class Drop(Cfg):
        basmala = "drop"
    d = engine.recitation_edges(an, 7, 1, Drop, lead=1)
    check("basmala=drop on Al-Fatiha removes the isti'adha and keeps verse 1",
          ISTI[1] <= d[0] <= VERSES[0][0] and holds(d, 1, VERSES[0]), (d, ISTI, VERSES[0]))
    wrong = [k for k in range(1, 8) if not holds(d, k, VERSES[k - 1])]
    check("...without shifting anything else", not wrong, f"wrong: {wrong}")


def test_no_istiadha_is_unchanged():
    """A recording that opens straight on the basmala must behave exactly as it always did."""
    plain, m = build(FATIHA[1:], PAUSES[1:])
    a2 = engine.Analysis(plain)
    before = engine.recitation_edges(a2, 7, 1, Cfg)
    after = engine.recitation_edges(a2, 7, 1, Cfg, lead=0)
    check("no isti'adha: heard lead 0 gives exactly the old edges", before == after, (before, after))
    wrong = [k for k in range(1, 8) if not holds(after, k, m[k - 1])]
    check("no isti'adha: every verse is on its own clip", not wrong, f"wrong: {wrong}")


def test_is_lead():
    check("an isti'adha precedes verse 1 in every surah",
          all(engine.is_lead("istiadha", n) for n in (1, 5, 9, 114)))
    check("the basmala is verse 1 of Al-Fatiha, so it is NOT a leading formula there",
          not engine.is_lead("basmala", 1))
    check("...but it is one in every other surah, At-Tawbah included",
          engine.is_lead("basmala", 5) and engine.is_lead("basmala", 9))
    check("the surah's own text ends the walk", not any(engine.is_lead("verse", n) for n in (1, 5, 9)))
    check("a spoken title or announcement also precedes verse 1", engine.is_lead("other", 1))
    check("an answer the model did not give is not treated as a formula", not engine.is_lead("", 1))


def test_leading_segments():
    segs = engine.leading_segments(an, Cfg)
    check("the first stretch is the isti'adha on its own",
          len(segs) >= 2 and abs(segs[0][0] - ISTI[0]) < 200 and abs(segs[0][1] - ISTI[1]) < 300,
          (segs[:2], ISTI))
    check("the second stretch is the basmala on its own",
          abs(segs[1][0] - VERSES[0][0]) < 300 and abs(segs[1][1] - VERSES[0][1]) < 300,
          (segs[:2], VERSES[0]))
    check("never more than three are looked at", len(segs) <= 3, segs)


def test_check_opening():
    """The walk itself, with a stand-in for what Gemini hears."""
    engine.SystemMessage = engine.HumanMessage = lambda *a, **k: (a[0] if a else k.get("content"))
    engine._media = lambda a: {"type": "media"}
    engine._llm_segment = lambda k, m: None
    cfg = engine.Settings()
    logs = []
    cfg.log, cfg.status, cfg.api_key = logs.append, (lambda m: None), "fake"
    s = {"cfg": cfg, "number": 1, "name": "Al-Fatiha", "verses": 7, "lang": "L"}

    def answers(*kinds):
        seen = []

        def fake(cfg_, chain, messages, what):
            seen.append(what)
            k = kinds[len(seen) - 1]
            return engine.SegmentCheck(kind=k, heard=k)
        engine.ask = fake
        return seen

    seen = answers("istiadha", "basmala")
    lead, heard = engine.check_opening(s, cfg, audio, an)
    check("Al-Fatiha opening with the isti'adha: lead 1", lead == 1, (lead, heard))
    check("...and it stopped as soon as it reached verse 1 (2 small requests, not the whole surah)",
          len(seen) == 2, seen)

    answers("basmala")
    lead, _ = engine.check_opening(s, cfg, audio, an)
    check("Al-Fatiha opening on the basmala: lead 0 (the basmala is verse 1)", lead == 0, lead)

    s5 = dict(s, number=5, name="Al-Ma'idah", verses=120)
    seen = answers("istiadha", "basmala", "verse")
    lead, _ = engine.check_opening(s5, cfg, audio, an)
    check("another surah with isti'adha AND basmala: lead 2", lead == 2, lead)
    check("...which took three requests and no more", len(seen) == 3, seen)

    s9 = dict(s, number=9, name="At-Tawbah", verses=129)
    answers("istiadha", "verse")
    lead, _ = engine.check_opening(s9, cfg, audio, an)
    check("At-Tawbah with an isti'adha: lead 1 (it has no basmala of its own)", lead == 1, lead)

    def boom(*a, **k):
        raise RuntimeError("gemini is down")
    engine.ask = boom
    lead, heard = engine.check_opening(s5, cfg, audio, an)
    check("a failed check falls back to what the surah implies, and says so",
          lead == engine.assumed_lead(5, cfg) and any("could not hear" in l for l in logs), (lead, logs))

    cfg.api_key = ""
    check("no API key: the old assumption, no request attempted",
          engine.check_opening(s, cfg, audio, an) == (0, ""))
    cfg.api_key = "fake"
    cfg.opening_check = False
    check("the check can be switched off", engine.check_opening(s, cfg, audio, an) == (0, ""))


def test_map_edges_surah_1():
    """The Gemini path threw away its own answer for surahs 1 and 9."""
    def tm_for(first_start):
        vs = []
        for i, (a, b) in enumerate(VERSES, 1):
            a = a if i > 1 else first_start
            vs.append(engine.VerseSpan(verse=i, start=engine.fmt_ms(a), end=engine.fmt_ms(b),
                                       first_words=f"w{i}"))
        return engine.TafsirMap(intro_heard="isti'adha", outro_heard="", verses=vs)

    tm = tm_for(VERSES[0][0])

    class Drop(Cfg):
        basmala = "drop"
    e = engine.map_edges(an, tm, 1, Drop)
    check("map_edges no longer forces Al-Fatiha to start at 0: the isti'adha is trimmed",
          ISTI[1] - 300 <= e[0] <= VERSES[0][0] + 100, (e[0], ISTI, VERSES[0]))
    k = engine.map_edges(an, tm, 1, Cfg)
    check("with basmala=keep it still starts at 0, so the isti'adha stays as intro", k[0] == 0, k)
    check("the verse boundaries themselves are untouched by either", e[1:] == k[1:])


if __name__ == "__main__":
    test_the_old_failure()
    test_the_fix()
    test_drop()
    test_no_istiadha_is_unchanged()
    test_is_lead()
    test_leading_segments()
    test_check_opening()
    test_map_edges_surah_1()
    print()
    print("FAILURES:", fails if fails else "none")
    sys.exit(1 if fails else 0)
