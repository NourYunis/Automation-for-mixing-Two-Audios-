"""Tests for verses the silence cut cannot judge on its own.   python test_verses.py

Three things pauses alone get wrong, all of them silent:
  * a one-word verse (الم, طه, حم) whose genuine boundary the spacing floor vetoes, because the
    floor is a fraction of the AVERAGE verse and this verse is nothing like the average,
  * a very long verse (Al-Baqarah 282) whose own breathing pauses are longer than the pauses
    between real verses, so cuts bunch inside it and real boundaries elsewhere go uncut,
  * a surah too long to mark in one answer - 286 strictly increasing indices in a single reply.

The audio is transcribed in <= transcribe_chunk_min pieces already; what is tested here is the
reading of it, the spacing relaxation, and the suspicion gate that hands the undecidable cases to
Gemini instead of shipping them.
"""
import sys

import numpy as np

from test_equiv import SR, Seg, engine

rng = np.random.default_rng(7)
fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)
        if detail:
            print("      " + str(detail))


def sp(sec):
    n = int(sec * SR)
    env = 0.5 + 0.5 * np.sin(np.linspace(0, 40, n))
    return (rng.normal(0, 2800, n) * env).astype(np.int16)


def qt(sec):
    return rng.normal(0, 3, int(sec * SR)).astype(np.int16)


def build(spec):
    """spec = [(speech_s, pause_after_s), ...]; returns the audio and every pause's midpoint."""
    parts, mids, t = [], [], 0.0
    for d, p in spec:
        parts.append(sp(d))
        t += d
        if p:
            parts.append(qt(p))
            mids.append(int((t + p / 2) * 1000))
            t += p
    return Seg(np.concatenate(parts)), mids


class Cfg:
    basmala, min_silence, spacing_factor = "none", 900, 0.4
    pad, edge_pad, verify_clips = 150, 250, False
    suspect_margin, suspect_max = 1.0, 10
    api_key, opening_check, opening_max_ms = "", True, 20000
    log = staticmethod(lambda *_: None)


def near(cut, mids, tol=25):
    return any(abs(cut - m) <= tol for m in mids)


# --------------------------------------------------------------- one-word verses ------------ #
# Six short verses (3s / 1.5s alternating, the shape of a refrain like فبأي آلاء ربكما تكذبان)
# followed by one 40s verse. The average verse is ~8.6s, so the spacing floor is ~3.4s - wider
# than a 1.5s verse plus its pause, which is how a genuine boundary gets vetoed.
SHORT = [(3, 1.1), (1.5, 1.1), (3, 1.1), (1.5, 1.1), (3, 1.1), (1.5, 1.1), (40, 0)]
short_audio, short_mids = build(SHORT)
short_an = engine.Analysis(short_audio)


def test_short_verses():
    gaps = engine._find_gaps(short_an, 6, Cfg.min_silence)
    floor = max(1000, (short_an.total_ms / 7) * Cfg.spacing_factor)
    report = {}
    engine._spread_cuts(gaps, 6, floor, report)
    check("the spacing floor really does veto the short verses' boundaries",
          report.get("crowded") == 3, (report, floor))

    logs = []

    class Loud(Cfg):
        log = staticmethod(logs.append)
    e = engine.recitation_edges(short_an, 7, 9, Loud)
    check("...and relaxing it recovers every one of them", all(near(c, short_mids) for c in e[1:-1]),
          [round(c / 1000, 2) for c in e])
    check("...giving one clip per verse", len(e) - 1 == 7, e)
    check("...and it says so in the log", any("uneven" in l and "relaxed" in l for l in logs), logs)
    check("no clip holds an unused pause afterwards", engine.suspect_clips(short_an, e, Cfg) == [],
          engine.suspect_clips(short_an, e, Cfg))


# --------------------------------------------------------------- a very long verse ---------- #
# The same six verses, then one long verse broken by breathing pauses LONGER than the pauses
# between the real verses - which is ordinary in a 3-minute ayah. The cutter prefers the longest
# pauses, so it spends cuts inside the long verse and merges real verses elsewhere to pay for it.
LONG = [(3, 1.1), (1.5, 1.1), (3, 1.1), (1.5, 1.1), (3, 1.1), (1.5, 1.1),
        (12, 1.3), (12, 1.3), (12, 1.3), (12, 0)]
long_audio, long_mids = build(LONG)
long_an = engine.Analysis(long_audio)
REAL = long_mids[:6]                      # the six genuine verse boundaries
BREATH = long_mids[6:]                    # the three breaths inside verse 7


def test_long_verse():
    e = engine.recitation_edges(long_an, 7, 9, Cfg)
    inside = [c for c in e[1:-1] if near(c, BREATH)]
    missed = [m for m in REAL if not any(abs(m - c) <= 25 for c in e[1:-1])]
    check("REPRODUCED: cuts land inside the long verse", len(inside) == 3, e)
    check("...and exactly that many real boundaries go uncut", len(missed) == 3, missed)

    durs = sorted(b - a for a, b in zip(e, e[1:]))
    med = durs[len(durs) // 2]
    old_rule = [i for i, (a, b) in enumerate(zip(e, e[1:]), 1)
                if (b - a) > max(med * 4, 8000) or (b - a) < 600]
    check("the old '4x the median' rule is blind to it - it flags nothing at all", old_rule == [],
          old_rule)

    sus = engine.suspect_clips(long_an, e, Cfg)
    check("the new gate flags every merged clip", sorted(i for i, _ in sus) == [2, 3, 4], sus)
    shorter = [i for i, _ in sus if (e[i] - e[i - 1]) < med]
    check("...including two that are SHORTER than the median, which is why length was no use",
          len(shorter) == 2, shorter)


def test_clean_cut_is_never_questioned():
    """A cut where nothing was vetoed must cost no Gemini calls at all."""
    even, mids = build([(4, 1.2)] * 7 + [(4, 0)])
    an = engine.Analysis(even)
    e = engine.recitation_edges(an, 8, 9, Cfg)
    check("an evenly paced recitation raises no suspicion", engine.suspect_clips(an, e, Cfg) == [],
          engine.suspect_clips(an, e, Cfg))
    check("...and every verse still gets its own clip",
          len(e) - 1 == 8 and all(near(c, mids) for c in e[1:-1]), e)


def test_gate_routes_to_gemini():
    asked, clips = [], engine.cut_all(long_audio, long_an, engine.recitation_edges(long_an, 7, 9, Cfg), 150)

    class Ask(Cfg):
        verify_clips = True
        api_key = "fake"
    keep = engine._verify_clip

    def verdict(single):
        def f(cfg, clip, name, i, total):
            asked.append(i)
            return engine.ClipVerseCheck(is_single_complete_verse=single,
                                         explanation="two verses run together")
        return f
    try:
        engine._verify_clip = verdict(False)
        try:
            engine._check_clip_lengths(clips, long_an, Ask, "Al-Baqarah", 7)
            check("a clip Gemini says is not one verse escalates", False)
        except engine.SplitError as ex:
            check("a clip Gemini says is not one verse escalates", "not one verse" in str(ex), ex)
        check("...after asking about the worst clip first", asked and asked[0] == 4, asked)

        asked.clear()
        engine._verify_clip = verdict(True)
        engine._check_clip_lengths(clips, long_an, Ask, "Al-Baqarah", 7)
        check("a long verse Gemini confirms is one verse passes, and is not re-cut", len(asked) == 3, asked)

        class Few(Ask):
            suspect_max = 2
        asked.clear()
        try:
            engine._check_clip_lengths(clips, long_an, Few, "Al-Baqarah", 7)
            check("too many suspect clips re-cut instead of auditing", False)
        except engine.SplitError as ex:
            check("too many suspect clips re-cut instead of auditing",
                  "do not line up" in str(ex) and not asked, (ex, asked))
    finally:
        engine._verify_clip = keep


# --------------------------------------------------------------- long surahs ---------------- #
def test_transcription_is_chunked():
    an = engine.Analysis(build([(4, 1.2)] * 20 + [(4, 0)])[0])
    e = engine.chunk_edges(an, 7_800_000, int(25.0 * 60000))     # Al-Baqarah-sized, 25 min limit
    check("a 2h10m recitation is transcribed in pieces, none over the limit",
          len(e) - 1 == 6 and max(b - a for a, b in zip(e, e[1:])) <= 25 * 60000, e)
    check("the pieces cover the whole recording with no hole",
          e[0] == 0 and e[-1] == 7_800_000 and all(b > a for a, b in zip(e, e[1:])))


def test_windowed_marking():
    """286 indices in one answer is the weak link, not the transcription. Marking runs in windows,
    each anchored on the previous window's last verse."""
    engine.SystemMessage = engine.HumanMessage = lambda *a, **k: (a[0] if a else k.get("content"))
    engine._chat = lambda *a, **k: None
    n, per_verse = 90, 10
    words = [engine.Word(i * 1000, i * 1000 + 800, f"w{i}") for i in range(n * per_verse)]
    truth = [v * per_verse for v in range(n)]                    # verse v+1 starts at word v*10
    cfg = engine.Settings()
    logs = []
    cfg.log, cfg.status, cfg.api_key, cfg.map_verses = logs.append, (lambda m: None), "fake", 40
    s = {"cfg": cfg, "number": 2, "name": "Al-Baqarah", "verses": n, "lang": "L"}

    seen = []

    def fake(cfg_, chain, messages, what):
        text = messages[1]
        first = int(text.split("first word of verse ")[1].split(",")[0])
        last = int(text.split("... of verse ")[1].split(",")[0])
        lo = truth[first - 2] if first > 1 else 0               # the slice starts at the anchor verse
        seen.append((first, last))
        return engine.WordMap(intro_heard="begin" if first == 1 else "",
                              verse_starts=[truth[v - 1] - lo for v in range(first, last + 1)],
                              outro_start=-1, outro_heard="")
    engine.ask = fake

    tm = engine._map_words(s, cfg, words, "SYS", "Surah Al-Baqarah, 90 verses.", [], None, "marking")
    check("90 verses are marked in 3 windows of 40", seen == [(1, 40), (41, 80), (81, 90)], seen)
    check("every verse comes back, in order, exactly once", len(tm.verses) == n
          and [v.verse for v in tm.verses] == list(range(1, n + 1)))
    got = [engine._seconds(v.start) for v in tm.verses]
    check("...and every start is the right word's own time",
          got == [words[t].start / 1000 for t in truth], got[:5])
    check("the windows are anchored: each one is told it starts inside the verse before it",
          seen[1][0] == seen[0][1] + 1 and seen[2][0] == seen[1][1] + 1, seen)
    check("it says how many windows it took", any("window(s)" in l for l in logs), logs)

    s3 = dict(s, verses=3)
    seen.clear()
    small = [engine.Word(i * 1000, i * 1000 + 800, f"w{i}") for i in range(30)]

    def one(cfg_, chain, messages, what):
        seen.append(1)
        return engine.WordMap(intro_heard="b", verse_starts=[0, 10, 20], outro_start=-1, outro_heard="")
    engine.ask = one
    engine._map_words(s3, cfg, small, "SYS", "h", [], None, "marking")
    check("a short surah is still marked in a single request, as before", len(seen) == 1, seen)


def test_chat_path_guard():
    """The chat model has to carry the whole recording in one request; the transcription path does
    not. A surah past that size must say so instead of failing inside the API."""
    cfg = engine.Settings()
    cfg.log = cfg.status = lambda *a: None
    cfg.api_key, cfg.timing_source, cfg.chat_listen_max_min = "fake", "chat", 0.001
    s = {"cfg": cfg, "number": 2, "name": "Al-Baqarah", "verses": 286, "lang": "L"}
    try:
        engine.align_recitation(s, cfg, build([(4, 0)])[0])
        check("an over-long recording is refused by the chat path with a usable message", False)
    except engine.SplitError as ex:
        check("an over-long recording is refused by the chat path with a usable message",
              "too long" in str(ex) and "Transcribe" in str(ex), ex)


if __name__ == "__main__":
    test_short_verses()
    test_long_verse()
    test_clean_cut_is_never_questioned()
    test_gate_routes_to_gemini()
    test_transcription_is_chunked()
    test_windowed_marking()
    test_chat_path_guard()
    print()
    print("FAILURES:", fails if fails else "none")
    sys.exit(1 if fails else 0)
