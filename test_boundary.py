"""Tests for the verse-boundary fixes.   python test_boundary.py

Covers, against the real engine code:
  * place_cut never crosses into the neighbouring explanation (and matches the old result when the
    timestamps are good),
  * edge_move / move_by_pauses turn "a word of verse 100 at the end of verse 99" into the right cut moving
    the right way,
  * the digest now hears both joins of every verse,
  * the whole loop - assemble -> verify -> repair -> assemble -> verify - with a stand-in for Gemini's answers,
    on audio where Gemini's times are wrong enough to pick the wrong pause.
Gemini's own listening cannot be tested offline; what is tested is everything around it.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from test_save import PCM, audioop, engine

RATE = 16000
rng = np.random.default_rng(11)
fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)
        if detail:
            print("      " + str(detail))


def flat(sec):                           # constant-level "speech"
    n = int(sec * RATE)
    x = rng.normal(0, 3000, n)
    x[:80] *= np.linspace(0, 1, 80)
    x[-80:] *= np.linspace(1, 0, 80)
    return x


def quiet(sec):
    return rng.normal(0, 2, int(sec * RATE))


def audio(*chunks):
    return PCM(np.concatenate(chunks).astype(np.int16).tobytes(), RATE, 1, 2)


def old_cut(an, e1, s2):                 # the placement logic assemble() used before, verbatim
    return engine.snap((e1 + s2) // 2, engine.silence_mids(an), 1200)


def test_place_cut():
    # 1. the reproduced failure: a 100 ms breath between explanations, a real pause 0.8 s into the next
    taf = audio(flat(1.0), flat(3.0), quiet(0.10), flat(0.6), quiet(0.30), flat(3.0), flat(1.0))
    an = engine.Analysis(taf)
    e1, s2 = 4000, 4100
    old = old_cut(an, e1, s2)
    new, pause = engine.place_cut(an, e1, s2, 400, 0, an.total_ms)
    check("old logic cut INTO the next explanation (the bug)", old > s2 + 300, f"old={old}")
    check("new cut stays inside Gemini's interval +/- tolerance", e1 - 400 <= new <= s2 + 400, f"new={new}")
    check("...and lands in the real 100 ms breath", 4000 <= new <= 4100, f"new={new}")
    check("...reported as a weak boundary (no pause >= 120 ms)", pause == 0)

    # 2. good timestamps: the new placement agrees with the old one (no regression on the normal case)
    worst, inside = 0, True
    for trial in range(40):
        p = int(rng.integers(300, 900))
        a = audio(flat(1.5), quiet(p / 1000), flat(1.5), quiet(0.5), flat(1.0))
        an2 = engine.Analysis(a)
        t_end, t_start = 1500, 1500 + p
        e1_, s2_ = t_end + int(rng.integers(-80, 80)), t_start + int(rng.integers(-80, 80))
        o = old_cut(an2, e1_, s2_)
        n_, pz = engine.place_cut(an2, e1_, s2_, 400, 0, an2.total_ms)
        worst = max(worst, abs(o - n_))
        inside &= (t_end - 30 <= n_ <= t_start + 30) and pz >= 120
    check("40 random clean boundaries: new cut within 20 ms of the old one, inside the pause",
          worst <= 20 and inside, f"worst difference {worst} ms")

    # 3. a pause that exists only OUTSIDE the allowed interval is not used
    a = audio(flat(2.0), quiet(0.05), flat(2.0), quiet(0.6), flat(2.0))       # real pause 4.05-4.65 s
    an3 = engine.Analysis(a)
    c, pz = engine.place_cut(an3, 2000, 2050, 400, 0, an3.total_ms)           # Gemini says the boundary is at 2.0 s
    check("a real pause 2 s away is NOT used to 'rescue' the cut", 1600 <= c <= 2450, f"cut={c}")

    # 4. a long pause that only brushes the window must not beat the short real one at Gemini's gap
    a = audio(flat(2.0), quiet(0.9), flat(0.3), quiet(0.5), flat(2.0))        # long pause 2.0-2.9, real 3.2-3.7
    an4 = engine.Analysis(a)
    c, pz = engine.place_cut(an4, 3200, 3700, 400, 0, an4.total_ms)           # window 2800-4100 brushes the long one
    check("the pause at Gemini's own gap wins over a longer pause brushing the window",
          3200 <= c <= 3700 and pz < 900, f"cut={c} pause={pz}")


def test_moves():
    check("end + spill -> the cut is too late -> earlier",
          engine.edge_move(99, "interpretation_end", "spill", 111) == ("taf", 99, -1))
    check("end + cutoff -> the cut is too early -> later",
          engine.edge_move(99, "recitation_end", "cutoff", 111) == ("rec", 99, 1))
    check("start + spill -> the cut is too early -> later (boundary is the one BEFORE the verse)",
          engine.edge_move(100, "interpretation_start", "spill", 111) == ("taf", 99, 1))
    check("start + cutoff -> the cut is too late -> earlier",
          engine.edge_move(100, "recitation_start", "cutoff", 111) == ("rec", 99, -1))
    check("intro/outro edges are not verse boundaries (left to edge_pad)",
          engine.edge_move(1, "interpretation_start", "spill", 111) is None
          and engine.edge_move(111, "interpretation_end", "spill", 111) is None)
    check("a mismatch or a missing edge is never turned into a cut move",
          engine.edge_move(50, "", "spill", 111) is None and engine.edge_move(50, "recitation_end", "mismatch", 111) is None)

    # pauses at 2.0-2.5, 4.0-4.5, 6.0-6.5, 8.0-8.5 s
    a = audio(flat(2.0), quiet(0.5), flat(1.5), quiet(0.5), flat(1.5), quiet(0.5), flat(1.5), quiet(0.5), flat(1.0))
    an = engine.Analysis(a)
    mids = [2250, 4250, 6250, 8250]
    cut = mids[1]
    check("one pause earlier", abs(engine.move_by_pauses(an, cut, -1, 0, an.total_ms) - mids[0]) <= 20)
    check("one pause later", abs(engine.move_by_pauses(an, cut, +1, 0, an.total_ms) - mids[2]) <= 20)
    check("two pauses later", abs(engine.move_by_pauses(an, cut, +2, 0, an.total_ms) - mids[3]) <= 20)
    check("not enough pauses in range -> unchanged", engine.move_by_pauses(an, cut, -3, 0, an.total_ms) == cut)
    check("never moves outside [lo, hi]", engine.move_by_pauses(an, cut, +1, 0, mids[2] - 500) == cut)
    mid_word = 3000                                        # a cut sitting inside speech (no pause under it)
    check("from inside speech, 'one earlier' is the nearest pause before it",
          abs(engine.move_by_pauses(an, mid_word, -1, 0, an.total_ms) - mids[0]) <= 20)

    # recitation shifts apply to a copy, per surah, and re-clamp between the neighbours
    engine._REC_CACHE["shifts"] = {("rec", 2): 1}
    edges = [0, 2250, 4250, 8250, an.total_ms]
    moved = engine._apply_rec_shifts(an, edges)
    check("recitation shift moves boundary 2 one pause later", abs(moved[2] - 6250) <= 20 and edges[2] == 4250, moved)
    engine._REC_CACHE["shifts"] = {}


def test_digest():
    cfg = engine.Settings()

    def geometry(rec_s, taf_s, verses):
        parts, cur = [], 0
        for v in range(1, verses + 1):
            ra, rb = cur, cur + rec_s * 1000
            ta, tb = rb + 400, rb + 400 + taf_s * 1000
            parts.append({"verse": v, "rec": (ra, rb), "taf": (ta, tb), "next_rec": None})
            cur = tb + 900
        for k in range(len(parts) - 1):
            parts[k]["next_rec"] = parts[k + 1]["rec"]
        return parts, cur

    parts, total = geometry(10, 45, 3)
    a, b = engine.digest_windows(parts[0], cfg, total)
    p = parts[0]
    check("A holds the end of the recitation, the pause and the start of the interpretation",
          a == (p["rec"][1] - cfg.digest_rec_ms, p["taf"][0] + cfg.digest_taf_ms), a)
    check("B holds the END of a 45 s interpretation (the old digest never heard it)",
          b[0] == p["taf"][1] - cfg.digest_taf_tail_ms and b[0] > p["taf"][0] + cfg.digest_taf_ms, b)
    check("B reaches into the START of the next verse's recitation (never heard before)",
          b[1] == parts[1]["rec"][0] + cfg.digest_rec_head_ms, b)
    last = engine.digest_windows(parts[-1], cfg, total)
    check("the last verse has no next recitation: B ends at the end of the file", last[1][1] <= total)

    sp, tot = geometry(6, 5, 2)
    w = engine.digest_windows(sp[0], cfg, tot)
    whole = sp[0]["taf"][1] + cfg.gap_after_tafsir - sp[0]["rec"][0]
    check("a short verse is sent once, not as two overlapping copies",
          len(w) == 1 and (w[0][1] - w[0][0]) <= whole + cfg.digest_rec_head_ms, (w, whole))

    old_heard_end = lambda taf_s: taf_s <= 15
    check("for a 45 s explanation the old digest could not hear its end; the new one can",
          not old_heard_end(45) and b[0] < p["taf"][1])

    print("\n    seconds of audio per verse (recitation 10 s):")
    print(f"    {'explanation':>12} {'old digest':>11} {'new digest':>11} {'whole file':>11}")
    for L in (8, 20, 45, 90):
        pp, tot = geometry(10, L, 2)
        new = sum(y - x for x, y in engine.digest_windows(pp[0], cfg, tot)) / 1000
        old = (min(4, 10) + 0.4 + min(L, 15))
        print(f"    {L:>10} s {old:>10.1f}s {new:>10.1f}s {10 + 0.4 + L + 0.9:>10.1f}s")
    print()


def build_state(n_speech_verses=3):
    """Recitation + interpretation for 3 verses. Between the end of explanation 2 and explanation 3 there is a
    LONG pause inside explanation 2 (8.2-9.1 s), then its last words (9.1-9.4 s), then the real boundary pause
    (9.4-9.9 s)."""
    rec = audio(flat(2.0), quiet(1.0), flat(2.0), quiet(1.0), flat(2.0), quiet(1.0))
    taf = audio(flat(1.5), quiet(0.6),                                   # intro 0-2.1
                flat(3.0), quiet(0.8),                                   # verse 1: 2.1-5.1, pause to 5.9
                flat(2.3), quiet(0.9), flat(0.3), quiet(0.5),            # verse 2: 5.9-8.2, [pause], tail 9.1-9.4, boundary 9.4-9.9
                flat(3.0), quiet(0.6), flat(1.0))                        # verse 3: 9.9-12.9, outro 13.5-14.5
    cfg = engine.Settings()
    logs = []
    cfg.log, cfg.status, cfg.api_key = logs.append, (lambda m: None), "fake"
    an_rec, an_taf = engine.Analysis(rec), engine.Analysis(taf)
    edges = [0, 2500, 5500, an_rec.total_ms]
    # Gemini's times for explanation 2 / 3 are ~0.8 s early: it says verse 2 ends at 8.6 s and verse 3 starts at 9.0 s
    spans = [(2.1, 5.1), (5.9, 8.6), (9.0, 12.9)]
    tm = engine.TafsirMap(intro_heard="begin", outro_heard="end", verses=[
        engine.VerseSpan(verse=i + 1, start="", end="", first_words=f"w{i + 1}") for i in range(3)])
    st = {"cfg": cfg, "job_cfg": cfg, "tafsir": taf, "taf_an": an_taf, "spans": spans, "tmap": tm,
          "rec": rec, "rec_clips": engine.cut_all(rec, an_rec, edges, cfg.pad),
          "lang": "L", "number": 12, "name": "yusuf", "verses": 3, "verify_attempt": 0,
          "out_path": Path(tempfile.mkdtemp()) / "final.mp3"}
    return st, logs


def test_loop():
    keep_api, engine.AudioSegment = engine.AudioSegment, PCM
    try:
        _loop()
    finally:
        engine.AudioSegment = keep_api


def _loop():
    st, logs = build_state()
    engine.SystemMessage = engine.HumanMessage = lambda *a, **k: None
    engine._media = lambda a: {"type": "media"}
    engine._llm_edge_check = lambda k, m: None
    calls = []

    def fake_ask(cfg, chain, messages, what):
        calls.append(what)
        if len(calls) == 1:              # the first listen: the tail of verse 2's explanation opens verse 3's clip
            return engine.EdgeCheck(problems=[engine.EdgeProblem(
                verse=3, kind="spill", edge="interpretation_start", heard_verse=0,
                issue="the interpretation of verse 3 starts with the last words of the previous explanation")])
        return engine.EdgeCheck(problems=[])
    engine.ask = fake_ask

    r1 = engine.assemble(st)
    st.update(r1)
    pb = st["part_bounds"]
    len2_before = pb[1]["taf"][1] - pb[1]["taf"][0]
    check("before repair: the wrong pause was chosen (explanation 2 is cut short, ~2.6 s)", len2_before < 3200,
          f"{len2_before} ms")

    v1 = engine.verify(st)
    check("verify() hears the spill and asks for a repair", v1.get("needs_repair") is True
          and v1["problems"][0].edge == "interpretation_start", v1)
    check("the problem line names the edge", "interpretation start" in engine._problem_line(v1["problems"][0], {}))
    check("digest was asked once, with both joins (the request carries the verse range)",
          len(calls) == 1 and "junction check" in calls[0], calls)
    st.update(v1)

    up = engine.repair(st)
    check("repair() moves the interpretation cut between verses 2 and 3 one pause LATER",
          up.get("cut_shift") == {("taf", 2): 1}, up.get("cut_shift"))
    check("...and re-checks only verses 2 and 3", up.get("recheck") == [2, 3], up.get("recheck"))
    check("...instead of the old padding change", up["job_cfg"].pad == st["job_cfg"].pad)
    st.update(up)

    r2 = engine.assemble(st)
    st.update(r2)
    pb2 = st["part_bounds"]
    len2_after = pb2[1]["taf"][1] - pb2[1]["taf"][0]
    check("after repair: explanation 2 now includes its last words (~3.8 s)", len2_after > 3400,
          f"{len2_after} ms (was {len2_before} ms)")
    check("...and explanation 3 no longer starts with them",
          (pb2[2]["taf"][1] - pb2[2]["taf"][0]) < (pb[2]["taf"][1] - pb[2]["taf"][0]) - 200)

    v2 = engine.verify(st)
    check("the recheck listens only to verses 2 and 3 and finds nothing: the file is saved",
          v2.get("output", "").endswith("final.mp3") and st["out_path"].exists(), v2)
    check("the loop cost 2 junction requests in total", len(calls) == 2, calls)


def test_padding_still_cannot_help():
    """The old repair for a 'cutoff' only ever changed padding; keep the proof that this is a no-op for a cut."""
    a = audio(flat(4.0), flat(1.0))
    an = engine.Analysis(a)
    ends = {engine.trim_window(an, 3000, 4400, pad)[1] for pad in (150, 250, 600, 5000)}
    check("padding never moves a clip's edge past its window", ends == {4400}, ends)


if __name__ == "__main__":
    if audioop is None:
        print("SKIP  needs audioop")
        sys.exit(0)
    test_place_cut()
    test_moves()
    test_digest()
    test_loop()
    test_padding_still_cannot_help()
    print()
    print("FAILURES:", fails if fails else "none")
    sys.exit(1 if fails else 0)
