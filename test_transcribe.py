"""Tests for the transcription timing source.   python test_transcribe.py

Everything AROUND the transcription call is tested against the real engine code: parsing the response
(object- and dict-shaped), chunking at pauses, the numbered transcript, the answer checks, the
transcript -> map conversion, the retry loop with its complaint, and the auto/transcribe/chat fallback.
The transcription call and the mapping model themselves cannot be tested offline.
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from test_equiv import SR, Seg, engine

REAL_GET_WORDS = engine.get_words          # test_listen() replaces engine.get_words with fakes
fails = []
rng = np.random.default_rng(5)


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)
        if detail:
            print("      " + str(detail))


def speech(sec):
    return rng.normal(0, 3000, int(sec * SR))


def quiet(sec):
    return rng.normal(0, 2, int(sec * SR))


def W(t, a, b):
    return engine.Word(int(a * 1000), int(b * 1000), t)


# ---------------------------------------------------------------- response parsing ---------- #
def test_parse():
    ann = [{"type": "word_info", "text": "Hello", "start_offset": "0.100s", "end_offset": "0.450s"},
           {"type": "word_info", "text": "world", "start_offset": "0.500s", "end_offset": "0.850s"},
           {"type": "other", "text": "x"}]
    as_dict = {"steps": [{"content": [{"annotations": ann}]}]}
    as_obj = NS(steps=[NS(content=[NS(annotations=[NS(**a) for a in ann])])])
    for label, resp in (("dict", as_dict), ("object", as_obj)):
        w = engine.extract_words(resp)
        check(f"[{label}] two words, times in ms", [(x.text, x.start, x.end) for x in w]
              == [("Hello", 100, 450), ("world", 500, 850)], w)
    check("no steps -> no words", engine.extract_words(NS(steps=None)) == [])
    check("offset formats", (engine._offset_ms("1.5s"), engine._offset_ms(0.25), engine._offset_ms("2"),
                             engine._offset_ms("abc")) == (1500, 250, 2000, None))


# ---------------------------------------------------------------- language codes ------------ #
def test_lang():
    check("known languages map to the model's codes",
          [engine.lang_code(x) for x in ("Spanish", "Egyptian", "Persian", "French", "Brazilian Portuguese")]
          == ["es-419", "ar-EG", "fa-IR", "fr-FR", "pt-BR"])
    check("an unlisted dialect is left to automatic detection",
          engine.lang_code("Moroccan") is None and engine.lang_code("Saudi") is None)


# ---------------------------------------------------------------- chunking ------------------ #
def test_chunks():
    # 12 blocks of 3 s speech separated by pauses of different lengths; limit 10 s per chunk
    pauses = [0.4, 0.6, 0.5, 1.2, 0.5, 0.4, 0.9, 0.5, 0.4, 0.7, 0.5]
    parts = []
    for i in range(12):
        parts.append(speech(3.0))
        if i < 11:
            parts.append(quiet(pauses[i]))
    audio = Seg(np.concatenate(parts).astype(np.int16))
    an = engine.Analysis(audio)
    edges = engine.chunk_edges(an, len(audio), 10000)
    sizes = [b - a for a, b in zip(edges, edges[1:])]
    check("no chunk is longer than the limit", max(sizes) <= 10000, sizes)
    check("edges cover the whole recording", edges[0] == 0 and edges[-1] == len(audio))
    inside = all(any(g[0] - 30 <= e <= g[1] + 30 for g in an.gaps(150, 36)) for e in edges[1:-1])
    check("every cut sits inside a real pause (never inside a word)", inside, edges)
    check("a short recording stays in one piece", engine.chunk_edges(an, 5000, 10000) == [0, 5000])

    # no pause anywhere: still terminates and respects the limit
    solid = Seg(speech(30).astype(np.int16))
    e2 = engine.chunk_edges(engine.Analysis(solid), len(solid), 10000)
    check("continuous speech is still split, within the limit",
          max(b - a for a, b in zip(e2, e2[1:])) <= 10000 and e2[-1] == len(solid), e2)


# ---------------------------------------------------------------- transcript + answer checks - #
def test_map():
    words = [W(f"w{i}", i, i + 0.8) for i in range(30)]
    txt = engine.render_words(words, per_line=10)
    check("every word carries its index", "[0]w0" in txt and "[29]w29" in txt)
    check("every line starts with its clock", txt.splitlines()[1].startswith("@00:10 [10]w10"), txt.splitlines()[1])

    def wm(starts, outro=-1):
        return engine.WordMap(intro_heard="begin", verse_starts=starts, outro_start=outro, outro_heard="end")

    check("a correct answer is accepted", engine._check_word_map(wm([4, 12, 20], 28), 3, 30) == "")
    check("wrong count is named", "2 start indices" in engine._check_word_map(wm([4, 12]), 3, 30))
    check("an index outside the transcript is rejected", "outside" in engine._check_word_map(wm([4, 12, 99]), 3, 30))
    check("starts must increase", "strictly increase" in engine._check_word_map(wm([4, 20, 12]), 3, 30))
    check("the closing line must come after the last verse start",
          "outro_start" in engine._check_word_map(wm([4, 12, 20], 10), 3, 30))

    tm = engine.map_from_words(wm([4, 12, 20], 28), words, 3)
    v = tm.verses
    check("verse 1 starts at ITS first word's own time (no guessing)", v[0].start == "00:04.000", v[0].start)
    check("verse 1 ends at the last word BEFORE verse 2 starts", v[0].end == "00:11.800", v[0].end)
    check("the last verse ends where the closing line begins", v[2].end == "00:27.800", v[2].end)
    check("first_words comes from the transcript", v[1].first_words == "w12 w13 w14 w15 w16 w17", v[1].first_words)
    check("no closing line: the last verse runs to the end",
          engine.map_from_words(wm([4, 12, 20]), words, 3).verses[2].end == "00:29.800")
    check("the map's times parse back to the same seconds",
          engine._seconds(v[0].start) == 4.0 and engine._seconds(v[2].end) == 27.8)


# ---------------------------------------------------------------- the whole listen step ------ #
def make_state(mode, tmp):
    cfg = engine.Settings()
    cfg.timing_source, cfg.api_key, cfg.retries = mode, "fake", 2
    logs = []
    cfg.log, cfg.status = logs.append, (lambda m: None)
    tafsir = Seg(np.zeros(SR * 40, dtype=np.int16))
    st = {"cfg": cfg, "job_cfg": cfg, "number": 12, "name": "yusuf", "verses": 3, "lang": "Spanish",
          "tafsir": tafsir, "taf_an": None, "audio_path": tmp / "a.mp3", "map_path": tmp / "x.map.json",
          "attempt": 0, "error": "", "reference": ""}
    return st, cfg, logs


def test_listen():
    tmp = Path(tempfile.mkdtemp())
    engine.SystemMessage = engine.HumanMessage = lambda *a, **k: (a[0] if a else k.get("content"))
    engine._chat = lambda *a, **k: None
    engine.TafsirMap.model_dump_json = lambda self, **k: "{}"
    words = [W(f"w{i}", i, i + 0.8) for i in range(30)]
    engine.get_words = lambda *a, **k: words

    # 1. a wrong answer first, then a right one: the complaint reaches the second request
    st, cfg, logs = make_state("transcribe", tmp)
    seen = []
    answers = [engine.WordMap(intro_heard="", verse_starts=[4, 12], outro_start=-1, outro_heard=""),
               engine.WordMap(intro_heard="begin", verse_starts=[4, 12, 20], outro_start=28, outro_heard="end")]

    def ask1(cfg_, chain, messages, what):
        seen.append(messages[1])
        return answers[len(seen) - 1]
    engine.ask = ask1
    r = engine.listen(st)
    check("the retry loop ends with a valid map of 3 verses", len(r["tmap"].verses) == 3, r)
    check("the second request carries the exact complaint",
          len(seen) == 2 and "2 start indices" in seen[1] and "need exactly 3" in seen[1], seen[1][-200:])
    check("the request holds the numbered transcript", "[29]w29" in seen[0])
    st.update(r)
    st["tafsir"] = Seg(np.zeros(SR * 40, dtype=np.int16))
    st.update(engine.check(st))
    check("the map passes the pipeline's own check() unchanged", st["error"] == "", st["error"])

    # 2. auto: transcription unusable -> falls back to the chat path once, then stays there
    st, cfg, logs = make_state("auto", tmp)

    def broken(*a, **k):
        raise engine.TranscribeError("no Interactions API")
    engine.get_words = broken
    engine._media = lambda a: {}
    chat = engine.TafsirMap(intro_heard="", outro_heard="", verses=[
        engine.VerseSpan(verse=i + 1, start=f"00:{i * 10:02d}.000", end=f"00:{i * 10 + 8:02d}.000",
                         first_words="x") for i in range(3)])
    engine.ask = lambda *a, **k: chat
    r = engine.listen(st)
    check("auto: falls back to the chat model", r["tmap"] is chat)
    check("auto: says why, and remembers it for the rest of the run",
          any("cannot be used" in l for l in logs) and cfg.usage.get("transcribe_off") is True, logs)

    # 3. transcribe-only: the failure is raised, not hidden
    st, cfg, logs = make_state("transcribe", tmp)
    try:
        engine.listen(st)
        check("transcribe: a transcription failure is raised", False)
    except engine.TranscribeError:
        check("transcribe: a transcription failure is raised", True)

    # 4. chat: the transcriber is never touched
    st, cfg, logs = make_state("chat", tmp)
    touched = []
    engine.get_words = lambda *a, **k: touched.append(1) or words
    r = engine.listen(st)
    check("chat: the old path, the transcriber is not called", r["tmap"] is chat and not touched)


# ---------------------------------------------------------------- cache -------------------- #
def test_cache():
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "a.mp3"
    src.write_bytes(b"x" * 100)
    cfg = engine.Settings()
    cfg.log = cfg.status = lambda *a: None
    cfg.api_key = "k"
    s = {"cfg": cfg, "number": 1, "name": "n", "lang": "L"}
    audio = Seg(np.zeros(SR * 3, dtype=np.int16))
    an = engine.Analysis(audio)
    calls = []

    def fake(cfg_, seg, code, what):
        calls.append(code)
        return [engine.Word(100, 500, "hello"), engine.Word(600, 900, "world")]
    engine.transcribe_segment = fake
    engine.say = lambda *a: None
    engine.doing = lambda *a: None
    cache = tmp / "a.words.json"
    w1 = REAL_GET_WORDS(s, cfg, audio, an, src, cache, "en-US", "x")
    w2 = REAL_GET_WORDS(s, cfg, audio, an, src, cache, "en-US", "x")
    check("second call is served from the saved transcript", len(calls) == 1 and w1 == w2)
    REAL_GET_WORDS(s, cfg, audio, an, src, cache, "es-419", "x")
    check("another language code invalidates the cache", len(calls) == 2)
    src.write_bytes(b"y" * 200)
    REAL_GET_WORDS(s, cfg, audio, an, src, cache, "es-419", "x")
    check("a changed audio file invalidates the cache", len(calls) == 3)
    cfg.retranscribe = True
    REAL_GET_WORDS(s, cfg, audio, an, src, cache, "es-419", "x")
    check("'Transcribe again' ignores the cache", len(calls) == 4)


if __name__ == "__main__":
    test_parse()
    test_lang()
    test_chunks()
    test_map()
    test_listen()
    test_cache()
    print()
    print("FAILURES:", fails if fails else "none")
    sys.exit(1 if fails else 0)
