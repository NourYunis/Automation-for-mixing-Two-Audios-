"""Tests for the save path: stream_encode, to_format, concat.   python test_save.py [--bench]

Uses the REAL ffmpeg. pydub itself is not needed: PCM below is a small stand-in that implements
just the parts of pydub's AudioSegment the engine touches, with pydub's own short-circuit
behaviour (set_channels / set_frame_rate / set_sample_width return self when nothing changes)
and audioop-based conversion when something does.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from test_equiv import engine          # installs the stand-in modules, then imports engine

try:
    import audioop
except ImportError:                     # removed in Python 3.13 (requirements.txt pulls in audioop-lts)
    audioop = None


class PCM:
    def __init__(self, data, frame_rate=44100, channels=2, sample_width=2):
        self._d, self.frame_rate, self.channels, self.sample_width = bytes(data), frame_rate, channels, sample_width
        self.frame_width = channels * sample_width
        self.exported = []

    raw_data = property(lambda self: self._d)

    def __len__(self):
        return round(1000 * (len(self._d) // self.frame_width) / self.frame_rate)

    def _spawn(self, data):
        return PCM(data, self.frame_rate, self.channels, self.sample_width)

    def set_frame_rate(self, r):
        if r == self.frame_rate:
            return self
        PCM.conversions += 1
        out, _ = audioop.ratecv(self._d, self.sample_width, self.channels, self.frame_rate, r, None)
        return PCM(out, r, self.channels, self.sample_width)

    def set_channels(self, c):
        if c == self.channels:
            return self
        fn = audioop.tostereo if c == 2 else audioop.tomono
        return PCM(fn(self._d, self.sample_width, *((1, 1) if c == 2 else (0.5, 0.5))),
                   self.frame_rate, c, self.sample_width)

    def set_sample_width(self, w):
        if w == self.sample_width:
            return self
        return PCM(audioop.lin2lin(self._d, self.sample_width, w), self.frame_rate, self.channels, w)

    converter = "ffmpeg"
    conversions = 0                                       # counts real sample-rate conversions

    @classmethod
    def silent(cls, duration=1000, frame_rate=11025):    # pydub: always mono, 16-bit
        return cls(b"\0\0" * int(frame_rate * (duration / 1000.0)), frame_rate, 1, 2)

    def __getitem__(self, sl):
        a = int((sl.start or 0) * (self.frame_rate / 1000.0)) * self.frame_width
        b = len(self._d) if sl.stop is None else min(len(self._d), int(sl.stop * (self.frame_rate / 1000.0)) * self.frame_width)
        return self._spawn(self._d[max(0, a):max(0, b)])

    def fade_in(self, _):
        return self

    fade_out = fade_in

    def get_array_of_samples(self):
        import array
        return array.array("h", self._d)

    def export(self, path, format="mp3", **kw):        # records the call; used to prove the fallback ran
        self.exported.append((path, format, kw))
        Path(path).write_bytes(b"EXPORTED-BY-PYDUB")


def speech_like(seconds, rate=44100, channels=2, seed=1):
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)) ** 2
    mono = (np.convolve(rng.normal(0, 1, n), np.ones(8) / 8, mode="same") * 0.6
            + 0.3 * np.sin(2 * np.pi * 180 * t)) * env * 9000
    pcm = np.stack([mono] * channels, axis=1).astype(np.int16)
    return PCM(pcm.tobytes(), rate, channels, 2)


class Cfg:
    bitrate, mp3_quality, fmt, stream_save = "192k", None, "mp3", True

    def __init__(self):
        self.stop = type("E", (), {"is_set": lambda self: self.v, "v": False})()


def pydub_style_export(audio, dst):
    """What AudioSegment.export() does for mp3 (pydub/audio_segment.py): temp WAV, ffmpeg to a
    second temp file, read it all back, write the destination."""
    data = tempfile.NamedTemporaryFile(mode="wb", delete=False)
    w = wave.open(data, "wb")
    w.setnchannels(audio.channels); w.setsampwidth(audio.sample_width); w.setframerate(audio.frame_rate)
    w.setnframes(len(audio.raw_data) // audio.frame_width); w.writeframesraw(audio.raw_data); w.close()
    out = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
    with open(os.devnull, "rb") as dn:
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "wav", "-i", data.name, "-b:a", "192k",
                              "-id3v2_version", "4", "-f", "mp3", out.name], stdin=dn,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.communicate()
    out.seek(0)
    Path(dst).write_bytes(out.read())
    for f in (data, out):
        f.close(); os.unlink(f.name)


fails = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)
        if detail:
            print("      " + str(detail))


md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()


def run_tests():
    tmp = Path(tempfile.mkdtemp())
    audio = speech_like(20)

    # 1. identical output ------------------------------------------------------------------
    a, b = tmp / "pydub.mp3", tmp / "stream.mp3"
    pydub_style_export(audio, a)
    engine.stream_encode(Cfg(), audio, b)
    check("streamed mp3 is byte-identical to pydub's export", md5(a) == md5(b),
          f"{a.stat().st_size} vs {b.stat().st_size} bytes")
    check("no .part file left behind", not list(tmp.glob("*.part")))

    # 2. progress -----------------------------------------------------------------------------
    seen = []
    engine._SAVE_CHUNK, keep = 256 * 1024, engine._SAVE_CHUNK
    engine.stream_encode(Cfg(), audio, tmp / "p.mp3", seen.append)
    engine._SAVE_CHUNK = keep
    check("progress is monotonic and finishes at 100%", seen == sorted(seen) and seen[-1] == 1.0 and len(seen) > 5,
          f"{len(seen)} updates, last={seen[-1] if seen else None}")

    # 3. Stop mid-save ----------------------------------------------------------------------
    cfg = Cfg()
    engine._SAVE_CHUNK = 128 * 1024
    n = [0]

    def stop_after_a_few(_):
        n[0] += 1
        if n[0] == 3:
            cfg.stop.v = True
    dst = tmp / "stopped.mp3"
    try:
        engine.stream_encode(cfg, audio, dst, stop_after_a_few)
        raised = False
    except engine.Stopped:
        raised = True
    engine._SAVE_CHUNK = keep
    check("Stop interrupts the save", raised)
    check("a stopped save leaves neither a .part nor a half file", not dst.exists() and not list(tmp.glob("stopped*")))

    # 4. an existing file survives a failed save ------------------------------------------------
    dst = tmp / "existing.mp3"
    dst.write_bytes(b"THE PREVIOUS GOOD FILE")
    bad = Cfg()
    bad.bitrate = "not-a-bitrate"
    try:
        engine.stream_encode(bad, audio, dst)
        failed = False
    except engine.SplitError as e:
        failed = "ffmpeg exited" in str(e)
    check("ffmpeg failure is reported with its own message", failed)
    check("...and the previous file is untouched (pydub's export would already have emptied it)",
          dst.read_bytes() == b"THE PREVIOUS GOOD FILE" and not list(tmp.glob("existing.mp3.part")))

    # 5. ffmpeg cannot be launched --------------------------------------------------------
    engine.AudioSegment.converter = "/nonexistent/ffmpeg"
    try:
        engine.stream_encode(Cfg(), audio, tmp / "nolaunch.mp3")
        launched = True
    except FileNotFoundError:
        launched = False
    engine.AudioSegment.converter = "ffmpeg"
    check("a missing ffmpeg raises cleanly (no leaked state)", not launched)

    # 6. _save falls back to pydub's export instead of failing -------------------------------
    logs = []
    s = {"cfg": type("C", (), {"bitrate": "192k", "mp3_quality": None, "fmt": "mp3", "stream_save": True,
                               "status": lambda self, m: None, "log": lambda self, m: logs.append(m)})(),
         "out_path": tmp / "fallback.mp3", "number": 1, "name": "x", "lang": "L"}
    s["cfg"].stop = Cfg().stop
    s["cfg"].bitrate = "not-a-bitrate"
    res = engine._save(s, audio)
    check("_save falls back to the standard export when streaming fails",
          len(audio.exported) == 1 and res["output"].endswith("fallback.mp3")
          and any("streaming save failed" in m for m in logs), logs)

    # 7. unsupported sample width goes to the fallback (rather than mis-encoding) ------------
    try:
        engine.stream_encode(Cfg(), PCM(b"\x00" * 800, 8000, 1, 1), tmp / "w1.mp3")
        rejected = False
    except engine.SplitError:
        rejected = True
    check("8-bit audio is refused by the streaming path (fallback handles it)", rejected)

    # 8. mp3_quality reaches ffmpeg only when set ---------------------------------------------
    script = tmp / "fake_ffmpeg.py"
    script.write_text("#!/usr/bin/env python3\nimport sys\nopen(sys.argv[-1],'wb').write(' '.join(sys.argv).encode()"
                      "+b'|'+sys.stdin.buffer.read()[:0])\n")
    script.chmod(0o755)
    engine.AudioSegment.converter = str(script)
    q = Cfg()
    engine.stream_encode(q, PCM(b"\x00" * 4000), tmp / "noq.mp3")
    q.mp3_quality = 7
    engine.stream_encode(q, PCM(b"\x00" * 4000), tmp / "q7.mp3")
    engine.AudioSegment.converter = "ffmpeg"
    check("default leaves ffmpeg's own quality alone (no -compression_level)",
          b"-compression_level" not in (tmp / "noq.mp3").read_bytes())
    check("mp3_quality=7 is passed through", b"-compression_level 7" in (tmp / "q7.mp3").read_bytes())

    # 9. concat / to_format -------------------------------------------------------------------
    if audioop is None:
        print("SKIP  concat/to_format tests need audioop (Python < 3.13, or pip install audioop-lts)")
        return
    rec = speech_like(1.0, 16000, 1)                       # the template: 16 kHz mono
    same = speech_like(0.5, 16000, 1, seed=2)
    other = speech_like(1.0, 8000, 1, seed=3)              # a different file: 8 kHz mono, 1.0 s
    check("to_format hands back the SAME object when the format already matches (no copy)",
          engine.to_format(same, rec) is same)
    joined = engine.concat([rec, same, other], rec)
    check("concat converts a differently-formatted part: total is 1.0 + 0.5 + 1.0 s",
          abs(len(joined) - 2500) <= 2, f"{len(joined)} ms")
    naive = rec._spawn(b"".join(p.raw_data for p in (rec, same, other)))   # the proposed 'Fix 1'
    check("...whereas appending raw bytes unconverted (the proposed fix) gets the length wrong",
          abs(len(naive) - 2500) > 200, f"{len(naive)} ms instead of 2500 - that audio would play at the wrong speed")
    # even when the LENGTH happens to survive (8 kHz stereo has the same byte rate as 16 kHz mono)
    # the content is still wrong: interleaved L/R samples read back as one fast mono stream
    st = speech_like(1.0, 8000, 2, seed=4)
    same_len = rec._spawn(b"".join(p.raw_data for p in (rec, st)))
    proper = engine.concat([rec, st], rec)
    check("...and even when the length coincides the samples are not the converted ones",
          abs(len(same_len) - len(proper)) <= 2 and same_len.raw_data != proper.raw_data)


def _timeline(spec, rate, channels, seed):
    """spec = [('v', seconds) speech | ('p', seconds) near-silence, ...] -> (PCM, [(start_s, end_s)] of speech)"""
    rng, chunks, marks, t = np.random.default_rng(seed), [], [], 0.0
    for k, (kind, sec) in enumerate(spec):
        n = int(sec * rate)
        if kind == "v":
            x = speech_like(sec, rate, 1, seed + k).raw_data
            x = np.frombuffer(x, dtype=np.int16)
            marks.append((t, t + sec))
        else:
            x = rng.normal(0, 2, n).astype(np.int16)
        chunks.append(np.stack([x] * channels, axis=1).reshape(-1) if channels > 1 else x)
        t += sec
    return PCM(np.concatenate(chunks).astype(np.int16).tobytes(), rate, channels, 2), marks


def test_assemble():
    """assemble() on a recitation and an interpretation in DIFFERENT formats, twice (a repair pass)."""
    import shutil
    print()
    if audioop is None:
        print("SKIP  assemble test needs audioop")
        return
    keep_api = engine.AudioSegment
    engine.AudioSegment = PCM
    try:
        N, logs = 6, []
        rec_spec = [("v", 2.0), ("p", 1.2)] * N
        taf_spec = [("p", 2.0)] + sum([[("v", 3.0), ("p", 1.3)] for _ in range(N)], [])[:-1] + [("p", 1.5)]
        rec, _ = _timeline(rec_spec, 44100, 2, 10)            # recitation: 44.1 kHz stereo
        taf, marks = _timeline(taf_spec, 22050, 1, 20)        # interpretation: 22.05 kHz mono
        cfg = engine.Settings()
        cfg.log, cfg.status = logs.append, lambda m: None
        an_rec, an_taf = engine.Analysis(rec), engine.Analysis(taf)
        edges = [0] + [int((2.0 * (i + 1) + 1.2 * i + 0.6) * 1000) for i in range(N - 1)] + [len(rec)]
        rec_clips = engine.cut_all(rec, an_rec, edges, cfg.pad)
        tm = engine.TafsirMap(intro_heard="begin", outro_heard="end", verses=[
            engine.VerseSpan(verse=i + 1, start="", end="", first_words=f"w{i + 1}") for i in range(N)])
        st = {"cfg": cfg, "job_cfg": cfg, "tafsir": taf, "taf_an": an_taf, "spans": marks, "tmap": tm,
              "rec": rec, "rec_clips": rec_clips, "lang": "L", "number": 1, "name": "x", "verses": N}

        PCM.conversions = 0
        r1 = engine.assemble(st)
        first = PCM.conversions
        check("assemble(): the interpretation is converted to the recitation's format ONCE (1 pass, 1 rate conversion)",
              first == 1, f"{first} conversions (the old per-clip path would do about {N})")
        out = r1["assembled_audio"]
        check("...and the result is in the recitation's format",
              (out.frame_rate, out.channels, out.sample_width) == (44100, 2, 2))
        check("...one [recitation, gap, interpretation, gap] block per verse",
              len(r1["unit_bounds"]) == N and len(r1["part_bounds"]) == N)
        check("...unit bounds add up to the finished length (within rounding)",
              abs(r1["unit_bounds"][-1][1] - len(out)) <= N * 2, f"{r1['unit_bounds'][-1][1]} vs {len(out)}")
        check("...a clean synthetic cut measures no defects", r1["local_problems"] == [], r1["local_problems"])

        st.update(r1)                                          # what the graph does between passes
        cfg2 = engine.Settings(); cfg2.log, cfg2.status = logs.append, lambda m: None
        cfg2.pad += 100                                        # what a 'cutoff' repair does
        st["job_cfg"] = cfg2
        r2 = engine.assemble(st)
        check("a repair pass re-assembles with ZERO further rate conversions (cached)",
              PCM.conversions == first, f"{PCM.conversions - first} extra")
        check("...reusing the very same converted buffer", r2["taf_fmt"][1] is r1["taf_fmt"][1])
        check("...and wider padding really did change the result", len(r2["assembled_audio"]) > len(out))

        # finally: save it and ask ffprobe how long the mp3 is
        tmp = Path(tempfile.mkdtemp())
        s2 = {"cfg": cfg, "out_path": tmp / "final.mp3", "number": 1, "name": "x", "lang": "L"}
        engine._save(s2, out)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                    "default=nw=1:nk=1", str(tmp / "final.mp3")], capture_output=True,
                                   text=True).stdout)
        check("saved mp3 has the assembled length (ffprobe)", abs(dur - len(out) / 1000) < 0.15,
              f"{dur:.2f}s vs {len(out) / 1000:.2f}s")
        check("_save logged how long it took and the speed",
              any("saved ->" in m and "x real time" in m for m in logs))
        shutil.rmtree(tmp, ignore_errors=True)
    finally:
        engine.AudioSegment = keep_api


def bench(minutes):
    """Reproduce the numbers quoted in the README."""
    audio = speech_like(minutes * 60)
    print(f"\n--- benchmark: {minutes} min of 44.1 kHz stereo ({len(audio.raw_data) / 1e6:.0f} MB PCM) ---")
    tmp = Path(tempfile.mkdtemp())
    engine.stream_encode(Cfg(), speech_like(5), tmp / "warm.mp3")

    print("\nLAME algorithm quality (ffmpeg -compression_level), 192k CBR:")
    for lvl in (None, 0, 2, 5, 7, 9):
        c = Cfg(); c.mp3_quality = lvl
        t = time.perf_counter(); engine.stream_encode(c, audio, tmp / "q.mp3"); dt = time.perf_counter() - t
        print(f"  {'default' if lvl is None else lvl!s:>7}: {dt:6.2f} s = {minutes * 60 / dt:5.0f}x real time")

    print("\nsave path:")
    t = time.perf_counter(); pydub_style_export(audio, tmp / "a.mp3"); ta = time.perf_counter() - t
    t = time.perf_counter(); engine.stream_encode(Cfg(), audio, tmp / "b.mp3"); tb = time.perf_counter() - t
    print(f"  pydub export (temp WAV, temp mp3, read back): {ta:6.2f} s")
    print(f"  raw PCM piped to ffmpeg                      : {tb:6.2f} s   identical output: {md5(tmp / 'a.mp3') == md5(tmp / 'b.mp3')}")

    print("\nconcat of 1,144 parts:")
    size = (len(audio.raw_data) // 1144) // 4 * 4
    parts = [audio.raw_data[i * size:(i + 1) * size] for i in range(1144)]
    t = time.perf_counter(); b"".join(parts); tj = time.perf_counter() - t
    t = time.perf_counter(); buf = bytearray(); [buf.extend(p) for p in parts]; bytes(buf); tb2 = time.perf_counter() - t
    print(f"  b''.join {tj * 1000:7.1f} ms     bytearray+bytes() {tb2 * 1000:7.1f} ms   ({tb2 / tj:.1f}x slower, copies twice)")


if __name__ == "__main__":
    run_tests()
    test_assemble()
    if "--bench" in sys.argv:
        bench(float(sys.argv[sys.argv.index("--minutes") + 1]) if "--minutes" in sys.argv else 5)
    print()
    print("FAILURES:", fails if fails else "none")
    sys.exit(1 if fails else 0)
