"""Speed of the refactor, plus behaviour of the new deterministic gate."""
import time

import numpy as np
from test_equiv import Seg, SR, engine, make_audio, old_frame_db, old_trim_edges


class Cfg:
    basmala, min_silence, spacing_factor = "keep", 900, 0.4
    pad, edge_pad, verify_clips = 150, 250, False
    gap_after_verse, gap_after_tafsir = 400, 900
    local_cutoff = True


# ---------------------------------------------------------------- speed --------------- #
print("=== cost of cutting one surah into verse clips ===")
for nverses in (30, 120, 286):
    audio = make_audio([2500] * (nverses + 1), 1100, seed=7)
    mins = len(audio) / 60000
    t = time.perf_counter()
    edges_old = None
    db = old_frame_db(audio)                       # once, as the old split_recitation did
    gaps = [g for g in engine.Analysis(audio).gaps(900, 42)]
    edges = [0] + [(s + e) // 2 for s, e in gaps[1:nverses]] + [len(audio)]
    old_clips = [old_trim_edges(audio[a:b], 150) for a, b in zip(edges, edges[1:])]
    old_t = time.perf_counter() - t

    t = time.perf_counter()
    an = engine.Analysis(audio)
    new_clips = engine.cut_all(audio, an, edges, 150)
    new_t = time.perf_counter() - t

    # and again, as a second language of the same surah would (analysis already cached)
    t = time.perf_counter()
    engine.cut_all(audio, an, edges, 150)
    reuse_t = time.perf_counter() - t

    assert [len(c) for c in old_clips] == [len(c) for c in new_clips], "clip lengths differ!"
    print(f"{nverses:>4} verses ({mins:4.1f} min audio):  old {old_t * 1000:7.1f} ms   "
          f"new {new_t * 1000:7.1f} ms ({old_t / new_t:4.1f}x)   "
          f"next language {reuse_t * 1000:6.1f} ms ({old_t / reuse_t:5.1f}x)")

print()
print("=== cost of the pause search (_find_gaps) ===")
audio = make_audio([2500] * 120, 1100, seed=3)


def old_find_gaps_cost(audio, need, start_ms):
    db = old_frame_db(audio)
    passes = 0
    for min_ms in [start_ms] + [m for m in (700, 500, 350, 250, 180) if m < start_ms]:
        for rel in (42, 36, 30, 26):
            passes += 1
            d = np.diff(np.concatenate(([0], (db < -rel).astype(np.int8), [0])))
            g = [(int(s) * 10, int(e) * 10) for s, e in
                 zip(np.where(d == 1)[0], np.where(d == -1)[0])
                 if s != 0 and e != len(db) and (e - s) * 10 >= min_ms]
            if len(g) >= need:
                return passes
    return passes


need = 400                                     # deliberately unreachable: worst case
t = time.perf_counter()
p_old = old_find_gaps_cost(audio, need, 900)
old_t = time.perf_counter() - t
an = engine.Analysis(audio)
t = time.perf_counter()
try:
    engine._find_gaps(an, need, 900)
except engine.SplitError:
    pass
new_t = time.perf_counter() - t
print(f"worst case: old {p_old} array passes / {old_t * 1000:.1f} ms   ->   "
      f"new {len(an._runs)} passes / {new_t * 1000:.1f} ms  ({old_t / new_t:.1f}x)")

# ------------------------------------------------------- deterministic gate ----------- #
print()
print("=== the measured gate ===")


def clips_from(audio, edges, pad=150):
    return engine.cut_all(audio, engine.Analysis(audio), edges, pad)


good = make_audio([2500] * 6, 1200, seed=11)
an = engine.Analysis(good)
edges = engine.recitation_edges(an, 5, 5, Cfg)
rec = engine.cut_all(good, an, edges, 150)
taf = engine.cut_all(good, an, edges, 150)
probs = engine.local_defects(rec, taf, Cfg)
print(f"clean cut at the pauses            -> {len(probs)} defect(s)   {'OK' if not probs else probs}")

# a cut deliberately placed in the middle of speech
bad_edges = [0, 2000, 6000, 10000, 14000, len(good)]
rec_bad = engine.cut_all(good, an, bad_edges, 150)
probs = engine.local_defects(rec_bad, rec_bad, Cfg)
kinds = sorted({p.kind for p in probs})
print(f"cuts placed mid-word               -> {len(probs)} defect(s), kinds={kinds}")
assert any(p.kind == "cutoff" for p in probs), "should have caught the mid-word cuts"

# a clip that is pure silence
silent = Seg(np.concatenate([np.random.default_rng(1).normal(0, 3, SR * 3).astype(np.int16),
                             (np.random.default_rng(2).normal(0, 2500, SR * 3)).astype(np.int16)]))
an2 = engine.Analysis(silent)
sc = engine.cut_all(silent, an2, [0, 3000, 6000], 150)
probs = engine.local_defects(sc, sc, Cfg)
print(f"a clip holding no speech at all    -> {len(probs)} defect(s), "
      f"kinds={sorted({p.kind for p in probs})}")
assert any(p.kind == "gap" for p in probs), "should have caught the silent clip"

# ------------------------------------------------------------- digest ---------------- #
print()
print("=== how much audio the mismatch check sends ===")


real = engine.Settings()
for label, nverses, rec_s, taf_s in [("short surah", 6, 8, 45),
                                     ("Yusuf-sized", 111, 10, 45),
                                     ("medium surah", 30, 10, 60),
                                     ("Al-Baqarah-ish", 286, 12, 75)]:
    part_bounds, cursor = [], 0
    for v in range(1, nverses + 1):
        ra, rb = cursor, cursor + rec_s * 1000
        ta, tb = rb + 400, rb + 400 + taf_s * 1000
        part_bounds.append({"verse": v, "rec": (ra, rb), "taf": (ta, tb), "next_rec": None})
        cursor = tb + 900
    for k in range(len(part_bounds) - 1):
        part_bounds[k]["next_rec"] = part_bounds[k + 1]["rec"]
    whole = cursor
    sent = sum(y - x + 1000 for p in part_bounds for x, y in engine.digest_windows(p, real, whole))
    reqs = -(-nverses // real.digest_verses)
    print(f"{label:>16}: whole file {whole / 60000:6.1f} min ({whole / 1000 * 32:>9,.0f} tok)  ->  "
          f"both joins {sent / 60000:5.1f} min ({sent / 1000 * 32:>8,.0f} tok) in {reqs:>3} request(s)  "
          f"= {sent / whole:5.1%}")
print()
print("all new-behaviour checks passed")
