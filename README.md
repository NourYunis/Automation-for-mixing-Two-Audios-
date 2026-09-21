# Surah Audio Builder

Builds a Quran audio book where every verse is followed by its spoken interpretation (tafsir):

```
verse 1 recitation → interpretation of verse 1 → verse 2 recitation → interpretation of verse 2 → …
```

One finished file is produced per surah **and per language/dialect** (Spanish, Saudi, Persian, Moroccan, Egyptian, …). The languages are detected automatically from the file names.

The recitation is cut at its pauses. The interpretation is cut from a **word-timed transcript made by Gemini 3.5 Transcribe**: the transcription model supplies every word with its real start and end time, and a chat model reads the numbered transcript and says which word each verse starts at. A LangGraph pipeline validates the result and asks again if it is wrong. (The older way, where a chat model listens to the audio and guesses the times, is still available as **Timing source: chat**.)

---

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on your PATH (Windows: `winget install Gyan.FFmpeg`, then reopen the terminal)
- A Google AI Studio (Gemini) API key

```bash
pip install -r requirements.txt
python app.py
```

`flet` and `flet-desktop` are pinned to the same version (0.28.3). If you get `module 'flet_desktop' has no attribute 'version'`, an older Flet is mixed in. Fix it with:

```bash
pip uninstall -y flet flet-desktop flet-cli
pip install flet==0.28.3 flet-desktop==0.28.3
```

or use a fresh virtual environment (`python -m venv venv`).

`langchain-google-genai` must be **2.0 or newer**. The engine requests structured output with `include_raw=True`, which is what makes Gemini's token usage visible; older versions drop it and the cost report stays empty (nothing else breaks).

---

## Preparing your files

### Interpretation audio: one folder per surah

```
output/
├── سورة الناس/
│   ├── Spanish_114 سورة الناس_Tafseer.mp3
│   ├── Saudi_114 سورة الناس_Tafseer.mp3
│   ├── Persian_114 سورة الناس_Tafseer.mp3
│   └── Moroccan_114 سورة الناس_Tafseer.mp3
├── سورة الفلق/
│   └── …
```

- The **language/dialect is the word before the first `_`**: `<Language>_<number> سورة <name>_Tafseer`. Any name works, nothing is hard-coded. Case is ignored, and multi-word names such as `Brazilian Portuguese_9 …` are fine.
- The **surah folder** is recognised by its Arabic name or its number (`سورة الناس`, `114`, …).
- Audio files that do not follow the pattern, or that contain recitation words (`تلاوة`, `quran`, …), are ignored as interpretations.
- Each interpretation recording is expected to contain: a short opening announcement (e.g. "begin", the surah title) → the explanation of verse 1 … verse N → a short closing line (e.g. "end"). The announcement and the closing line are removed.

### Recitation audio

One full-surah recitation file per surah, in a single folder, with the surah name (or number) in the file name (`سورة الناس.mp3`, `114.mp3`, `تلاوة سورة الناس.mp3`, …). If no recitations folder is given, a recitation file inside the surah folder is used (file name contains `تلاوة`, `quran`, `recit`, …).

Alternatively, paste a **YouTube playlist link** and the app downloads them for you (see below).

### Reference text (optional)

A `.txt`, `.md`, `.srt`, `.vtt` or `.docx` in the surah folder, named with the same language pattern, is sent to Gemini as a hint for recognising where each verse's explanation begins. It is never used to count anything. For **Egyptian**, if nothing is in the folder, the app looks for a matching `.docx` in the "Tafsir .docx folder".

---

## Using the app

1. **Interpretation folder**: the `output/` folder above. The detected languages appear as checkboxes, e.g. `Persian (37)`, with the number of surahs each appears in. Untick any you do not want. **Detect** re-scans.
2. **Recitations folder**: where the recitation files are (or will be downloaded to).
3. **Save finished files to**: output folder.
4. *(optional)* **Tafsir .docx folder**: reference text for Egyptian.
5. *(optional)* **YouTube playlist link**: downloaded into the Recitations folder first.
6. **Gemini API key** and model (default `gemini-2.5-flash`). The key and folder paths are remembered on your computer between sessions.
7. **From surah / To surah**: the inclusive range to process (default 1 to 114). Choose the same surah twice for a single one.
8. **Basmala**: keep it before verse 1 (default), remove it, or "recording has none". What a recording actually opens with is *heard*, not assumed: before cutting, each leading stretch of speech is identified (isti'adha / basmala / spoken title / verse 1) and only the ones that come before verse 1 are skipped. That matters most in Al-Fatiha, whose basmala **is** verse 1, and At-Tawbah, which has none — a reciter who opens either with the isti'adha used to push every verse one clip out of place. Untick **Hear what the recitation opens with** to fall back to assuming it from the surah number.
9. **Recitation cutting**: *Auto* (default), *Always let Gemini mark the verses*, or *Silences only (never use Gemini)*.
10. **Timing source**: how the verses of the interpretation (and of a recitation Gemini has to mark) are found. *Auto* (default) uses Gemini 3.5 Transcribe and falls back to the chat model if the transcription model cannot be used; *Gemini 3.5 Transcribe only* never falls back; *Chat model listens* is the old way. **Transcribe again** ignores saved transcripts. See [Timing from a transcription model](#timing-from-a-transcription-model).
11. **Final check**: how the finished file is reviewed. See [The final check](#the-final-check-and-what-it-costs) below.
    - *Verse junctions only* (default) — the cheapest setting, and the recommended one.
    - *Listen to the whole finished file* — the original behaviour, kept unchanged.
    - *Measured checks only* — no audio is sent at all.
12. **Act on measured mid-word cuts**: off by default. Cut-off words are always measured and shown in the log; this switch decides whether they count as problems on their own or are left for the audio check to confirm.
13. Switches:
    - **Overwrite finished files**: rebuild files that already exist (otherwise they are skipped).
    - **Ask Gemini again**: ignore saved timing maps and re-run the listening step.
    - **Dry run**: analyse and print the timings, write no audio.
14. **Build**, or **Download recitations only**. **Stop** ends the run after the current item.

> Tip: start with a **Dry run** on one surah. It calls Gemini once, saves the timing map, and prints what was removed and the first words of verse 1, so you can confirm the first verse is still there. The real build afterwards reuses the map at no extra cost.

### YouTube playlist download

Every video is saved as mp3 (`<playlist number> - <title>.mp3`) in the Recitations folder. Videos already downloaded are recorded in `.downloaded.txt` and skipped next time. The surah is found from the surah name in the title. Files whose title has no recognisable surah name are listed in the log ("cannot tell which surah this is"); rename them, e.g. to `سورة النبأ.mp3`. YouTube downloads need a JavaScript runtime: install [Deno](https://deno.com) (`winget install DenoLand.Deno`, or the PowerShell installer). The app finds it on your PATH or in `%USERPROFILE%\.deno\bin`, so it also works if the app was already open when you installed it. `yt-dlp[default]` (in `requirements.txt`) includes the solver component it needs. If downloads suddenly fail, run `pip install -U "yt-dlp[default]"`.

---

## How it works

For each (surah, language) a LangGraph graph runs:

```
prepare → listen → check ──ok──→ assemble → verify ──ok──→ saved
             ↑        │                        │
             └────────┘ bad map                └─ problem found → repair → prepare (re-cut, re-check)
                        (Gemini gets the exact complaint)                └─ nothing left to try → NEEDS REVIEW
```

- **prepare**: loads the audio and cuts the recitation. By default (**Recitation cutting: Auto**) it cuts at the longest pauses; the pauses that follow whatever is recited *before* verse 1 are never verse ends (how many there are is heard once per surah and reused by every language of it, not assumed from the surah number — see Basmala above), the minimum spacing between cuts is relaxed rather than guessed around when a surah mixes one-word verses with very long ones, and any clip holding a pause that was rejected only by that spacing is put to Gemini as "is this one complete verse?", and a suspiciously long or short clip is confirmed by Gemini as one complete verse. If that fails (no usable pauses, or a clip Gemini says is several verses, which happens when a reciter runs verses together or breathes mid-verse), **Gemini listens to the whole recitation and marks every verse** instead, and each cut moves to the quietest spot around Gemini's boundary. That alignment is saved once per surah (`<no>_<name>.recitation.json` in the output folder) and reused for every language. Delete that file to have Gemini mark it again.
- **listen**: with the default timing source the interpretation is transcribed by `gemini-3.5-transcribe` (word by word, with real times), and a chat model reads that transcript and returns the index of the first word of every verse; the map's times are the transcriber's own. With *Chat model listens* it instead receives the audio (plus the optional reference text) and returns a start time, an end time and the first words for each verse. Either way the result is the same `.map.json`, so everything after this step is unchanged.
- **check**: rejects the answer if it does not have exactly N verses numbered 1…N, if verses overlap or are shorter than 0.8 s, if verse 1 starts later than 25 s, or if times exceed the audio length. Rejection sends Gemini the exact reason and it tries again (up to 3 extra attempts). A complaint about verse *i* is **local**, so only verses *i−1*…N are marked again rather than the whole recording, and the re-ask is made slightly warmer so a genuinely different answer is possible.
- **assemble**: moves each cut onto the nearest real pause, then merges `verse → 0.4 s → interpretation → 0.9 s → next verse`. The start cut can only move earlier and the end cut only later, so a word of verse 1 can never be trimmed off.
- **verify**: see below.
- **repair**: works out where things went wrong instead of starting over:
  - `spill` / `cutoff` heard at a clip edge: the **cut itself** is wrong, so it is moved to the next real pause in the direction the edge implies (see [A word of the next verse in the previous one](#a-word-of-the-next-verse-in-the-previous-one)). Only the two verses either side are checked again. No new listening.
  - `gap` (dead air) or a cut-off word with no edge information: wider padding and shorter gaps, re-cut with the saved timing map. No new listening.
  - `mismatch`: everything before the first mismatching verse was confirmed correct and is kept. Gemini is first asked whether the **recitation** clip there really is that verse. If not, Gemini is asked to mark the verses of the recitation (re-listening to the interpretation could never fix that), and the whole file is checked again. Otherwise Gemini re-listens **only from the verse before the first mismatch to the end**, and the result is merged into the map. If the same spot fails again, it starts one verse further back each time, finally the whole recording. If even that fails, it stops and tells you where to listen.
  - The next final check only listens to the parts that changed.

A file that fails all attempts is reported as `NEEDS REVIEW: <reason>` in the log and the summary. It is never silently skipped, and it never stops the rest of the batch. Temporary Gemini errors (rate limit, overload, timeouts) are shown, waited out (5 / 15 / 30 / 60 s) and asked again without using up a retry.

---

## Timing from a transcription model

`gemini-3.5-transcribe` (public preview since 26 August 2026) is a speech-to-text model that returns **every word with its start and end offset**. It does not reason about verses, so the work is split:

```
audio ──► gemini-3.5-transcribe ──► words with real times ──► saved as <n>.words.json
                                          │
                     numbered transcript  ▼  ("[57]word", a clock at every line)
                    chat model, TEXT only: "verse 1 starts at word 4, verse 2 at word 61, ..."
                                          │
        the engine turns the indices back into times from the transcript ──► the same .map.json as before
```

What this changes:

- **Times are measured, not guessed.** A chat model that listens estimates timestamps and drifts on long recordings; here every time in the map comes from the transcriber. The model only answers with **integers** (word indices), which are checked (exactly N, strictly increasing, inside the transcript) and sent back with the exact complaint if wrong.
- **Cheaper and faster to retry.** The mapping call reads text, not audio (audio is 32 tokens per second). A rejected answer, a repair or "Ask Gemini again" re-reads the saved transcript: **the audio is transcribed only once** per recording, language and model. The saved transcript is used again while the audio file, the model and the language code are unchanged; **Transcribe again** (or deleting `*.words.json`) redoes it.
- **The same for the recitation** when Gemini has to mark it (Recitation cutting: *Always*, or *Auto* after the silences fail): the recitation is transcribed and the chat model, which knows the Quran text, maps the words to verses. Saved as `<no>_<n>.recitation.words.json`.

Limits that shape the code (from the Gemini API documentation):

- Word timestamps work on **at most 30 minutes of audio per request**, so longer recordings are split into pieces of at most `transcribe_chunk_min` (25) minutes. Each cut is placed in the longest pause of its window, so it never falls inside a word, and every piece's times are shifted back onto the whole recording's clock.
- Word timestamps **may lower transcription accuracy slightly**. That is why the mapping model is asked to judge by meaning, and why the final check still listens to the joins.
- The model **detects the language by itself**, but a language code helps. The engine passes one when it knows it (Spanish `es-419`, French `fr-FR`, Persian `fa-IR`, Egyptian `ar-EG`, English `en-US`, and about thirty more, see `_LANG_CODES` in `engine.py`). A dialect the model does not list (Saudi, Moroccan, Urdu, ...) is left to automatic detection. The recitation uses automatic detection unless `rec_language_code` is set.
- It is in **public preview**. If it cannot be used (an SDK without the Interactions API, a rejected request), *Auto* logs why and lets the chat model listen for the rest of the run; *Gemini 3.5 Transcribe only* stops with the reason instead.

The cost report now has a line for it (`Transcription: N min of audio (billed by duration)`); see Google's pricing page for the rate.

```bash
python test_transcribe.py   # parsing, chunking, the numbered transcript, answer checks, retry loop, fallback, cache
```

---

## The final check, and what it costs

Gemini bills audio by **duration**: 32 tokens per second, 1,920 per minute. Not by file size, so sending a smaller mp3 of the same recording saves upload time and nothing else.

The final check used to send the **whole assembled file**, recitation *and* interpretation, once per language and again after every repair. That made it the most expensive step in the system, larger than the listening step it was checking. It asked three questions, and two of them do not need ears:

| Problem | How it is found |
|---|---|
| `gap`: dead air | **Measured.** The program inserted every silence itself, so it adds up what it actually put there. It also spots a clip that holds no speech at all, using the clip's level against the whole recording's. |
| a word sliced in half | **Measured.** If speech is still going at the instant of a cut, the cut landed inside a word. |
| `mismatch`: the wrong verse's explanation, and `spill` / `cutoff`: a word of the neighbouring verse at a clip edge, or a word of this verse missing | **Asked**, because only ears can answer it. |

Only that last question is sent as audio, and only the parts of the file that can answer it. For every verse, two stretches of the **finished file** (slices of the real output, so Gemini hears exactly what will be saved):

- **A**: the end of the recitation, the pause, the start of the interpretation. *Is this the right explanation for this verse, and is the recitation's last word its own?*
- **B**: the end of the interpretation, the pause, the start of the **next** verse's recitation. *Is the hand-over clean?*

Together they contain **every edge of every clip**. A short verse's two stretches are merged into one. Excerpts are separated by one second of silence so they cannot be miscounted, and the prompt says the outer ends of an excerpt are cut on purpose, so Gemini never reports those as faults.

| | whole file | both joins | |
|---|---|---|---|
| short surah (6 verses) | 10,426 tokens | 5,338 tokens | 51% |
| Yusuf-sized (111 verses) | 199,978 tokens | 100,426 tokens | 50% |
| medium surah (30 verses) | 68,448 tokens | 27,072 tokens | 40% |
| Al-Baqarah-sized (286 verses) | 808,122 tokens | 258,906 tokens | 32% |

The cost follows the **number of verses** (about 26 s each), not how long the commentary runs, which is the right scaling: what is being checked is a pair of joins per verse.

> An earlier version of this check sent only the end of each recitation and the *start* of each interpretation (about 20 s per verse, 23-38% of the file). It was cheaper, but it could not hear the start of any recitation, nor the end of any interpretation longer than 15 s, so a word of the next verse at those edges went unreported except on short verses. Hearing both sides costs about a third more; it is what makes the boundary check real.

Set **Final check** to *Listen to the whole finished file* to get the original whole-file behaviour. Its `cutoff` reports carry no edge, so the repair cannot tell which cut to move and falls back to padding; prefer the default.

### What every run tells you

The summary now ends with what the run actually cost:

```
Gemini: 14 call(s), 412,880 input + 9,431 output tokens
        3,120 of the output tokens were thinking tokens
        by step: listening 268,400, mismatch 131,050, checking 22,861
```

Thinking tokens are billed at the **output** rate, so the short yes/no calls (clip checks, identity checks, the mismatch check) now ask for a thinking budget of zero.

### The two levers that are still on the table

Neither is done here, and together they are larger than everything above:

- **Batch API** — half price on input *and* output, results within 24 hours. This pipeline is a batch job by nature: 114 surahs × N languages, nobody waiting. The first listening pass over every job belongs there, with only the failures falling back to interactive calls.
- **Files API + explicit context caching** — cache reads bill at 10% of the input rate. Today every retry re-pays full price for audio Gemini has already been sent. It would also lift the 18 MB inline limit, which is what makes a recitation longer than about 75 minutes fail outright.

One thing to leave alone: in `engine.py` the audio part of each message comes **before** the text part. That ordering is what lets Gemini's implicit cache match a retry's prefix. Do not reorder it.

---

## Speed

The loudness analysis of a recording is computed once and then sliced, instead of being recomputed for every clip, every repair pass and every language. Cutting one surah into verse clips:

| | before | after | second language of the same surah |
|---|---|---|---|
| 30 verses (1.8 min) | 19.9 ms | 5.3 ms (3.7×) | 1.7 ms (12×) |
| 120 verses (7.2 min) | 51.6 ms | 19.5 ms (2.6×) | 6.8 ms (7.6×) |
| 286 verses (17 min) | 128.3 ms | 56.0 ms (2.3×) | 15.7 ms (8.2×) |

The search for pauses makes 4 passes over the array where it used to make 24 (6.8× faster in the worst case): the silent runs only have to be found once per loudness threshold, since the minimum-length setting merely filters them.

The recitations folder is walked once per run instead of once per surah, and the name-matching helpers are memoised.

**These changes alter three numbers out of 108, on purpose.** Where the original code could not place its cuts under its own spacing floor it fell back to "the longest pauses, wherever they are" — an arbitrary cut, not a worse one. That fallback is now a relaxation of the floor, so a one-word verse keeps its own boundary; `test_equiv.py` checks those three cases against the new guarantee (every cut in a real pause) and the other 105 against the old numbers, unchanged.

**Everything else does not alter a single number.** `test_equiv.py` re-implements the original `_frame_db`, `_gaps`, `_find_gaps`, `trim_edges` and the edge-finding of `split_recitation` verbatim and compares them against the new code across four synthetic recitation shapes and every basmala / spacing combination — 108 checks, all identical.

```bash
python test_equiv.py     # byte-identical to the old implementation
python test_new.py       # benchmarks + the measured gate + digest sizing
```

---

## A word of the next verse in the previous one

The symptom: a word or phrase of verse *N+1* is audible at the end of verse *N*'s recitation or interpretation (or a word of verse *N* is missing from it). Three causes, each reproduced in `test_boundary.py` and each fixed:

**1. The interpretation cut could cross the boundary.** It used to snap to the nearest pause of 200 ms or more within 1.2 s of the midpoint between Gemini's end and start times, in either direction. When the real gap between two explanations was shorter than that, the nearest qualifying pause was often *inside the next explanation*, and its first words rode along in the previous clip (750 ms of it in the reproduction). Now the cut is confined to the interval Gemini gave, widened by `boundary_tol` (400 ms) on each side. Inside it, the pause covering the most of Gemini's own gap wins; if there is none, the quietest instant is used and the boundary is reported: `N of M verse boundaries have no clear pause...`. With accurate timestamps the result is the same as before (40 random boundaries: within 20 ms).

**2. The repair could not fix it.** A `cutoff` repair only widened the clips' padding, but padding keeps more silence *inside* a clip's window; it cannot move a cut (the clip is identical at 150 ms and at 5,000 ms). Now a problem heard at a clip edge moves that cut, by whole pauses, in the direction the edge implies:

| heard at… | means the cut is… | so it moves… |
|---|---|---|
| the **end** of a clip: words of the *next* verse (`spill`) | too late | earlier |
| the **end** of a clip: its own last word missing (`cutoff`) | too early | later |
| the **start** of a clip: words of the *previous* verse (`spill`) | too early | later |
| the **start** of a clip: its own first word missing (`cutoff`) | too late | earlier |

Only the two verses either side of the moved cut are checked again. If the fault persists it moves one pause further on the next attempt, up to `verify_retries` (now 3). A recitation cut that is moved stays moved for every language of that surah, so a fault in the shared recitation is fixed once, not once per language.

**3. The check could not hear it.** See the note under [the final check](#the-final-check-and-what-it-costs). Boundary errors on most verses were invisible; they showed up on short verses like Surah Yusuf's verse 99 because a short explanation fit inside the old digest's window.

What this does **not** do, and what to expect:

- Gemini's times on a long recording are only approximate. If they are off by more than `boundary_tol` *and* a wrong pause sits inside the interval, the placement can still choose it. The difference now is that the check hears it and the repair moves it.
- A stray phrase that spans more than one pause needs more than one move; after `verify_retries` attempts the file is reported as `NEEDS REVIEW` with the verse and the edge, never saved as if it were fine.
- The direction depends on Gemini reporting the edge and the kind correctly. That cannot be tested offline. If it leaves the edge out, the repair falls back to padding, which changes nothing for a misplaced cut, so you get the old outcome rather than a worse one.
- The first and last edges of a file are the intro and the outro, not verse boundaries; they are governed by `edge_pad`.

In the log, look for `N of M verse boundaries have no clear pause`, `junction check: both joins of ...`, and `repair: spill at the interpretation start of verse 100 - moving the interpretation cut between verses 99 and 100 one pause later`.

---

## Saving the finished file

The save step was suspected of being slow because of `concat` and pydub's export. Each suspect was measured (real ffmpeg 6.1, one CPU core, 192 kbps CBR):

**The encode is the floor, and it is single-threaded.** LAME's algorithm quality (`ffmpeg -compression_level`) on 5 minutes of audio:

| setting | time | speed |
|---|---|---|
| ffmpeg default (= 5) | 2.64 s | 114× real time |
| `0` | 10.0 s | 30× real time |
| `2` | 4.8 s | 63× |
| `5` | 2.0 s | 150× |
| `7` | 1.8 s | 169× |
| `9` | 1.7 s | 175× |

So a 3-hour finished file is about **1½ minutes of pure encoding** on this machine. Note that `-compression_level 0`, sometimes suggested as a speed-up, is the *slowest, best-quality* setting: 4–5× slower. If you want a faster encode, `mp3_quality=7` in `Settings` gives roughly 1.5× at a slightly lower-quality encode; the default (`None`) leaves ffmpeg's own setting and the output unchanged.

**`concat` is not the bottleneck.** Joining 1,144 parts (an Al-Baqarah-sized file) takes about 0.2 s with `b"".join`. The `bytearray` version that is sometimes suggested is about 2× slower and uses twice the extra memory, because it copies twice. It would also drop the format conversion, which is not optional: the interpretation may be a different sample rate or channel count from the recitation, and pydub's silences are always mono. Raw bytes appended at the wrong format play at the wrong speed (`test_save.py` demonstrates it).

**Saving through pydub versus piping to ffmpeg: same speed, identical file.** 30 minutes of audio, pydub's route (temp WAV, temp mp3, read back) 15.5 s against 15.7 s piped, with byte-identical output (same md5). It is encode-bound; the temporary WAV cost 0.14 s.

What streaming still buys, and why `stream_save` is on by default:

- **No size ceiling.** pydub's export goes through a WAV file, whose header stores its size in 32 bits. Past 4 GiB of PCM (about **6.7 hours** of 44.1 kHz stereo, which the longest surahs can reach once recitation and interpretation are added together) the export fails with `struct.error`. Raw PCM on a pipe has no header.
- **A failed or stopped save cannot destroy an existing file.** pydub opens the destination for writing before it encodes anything, which empties a file that was already there (with *Overwrite* ticked, that is your previous good build). The stream is written to `<name>.mp3.part` and moved into place only once ffmpeg has finished.
- **Progress and Stop.** The status line shows `saving the file... 37%`, and Stop ends the save at once instead of after several minutes.
- No temporary files: pydub writes the whole PCM to disk, then a second temporary mp3.

If streaming fails for any reason the engine logs why and falls back to pydub's export, so it is never worse than before.

**Reading the log.** Every save now ends with `written in 92s, 117x real time`. Around 100× means the encode is simply the cost and only `mp3_quality` changes it. Far below that points at something else: a slow CPU, a slow disk, or memory pressure. On the biggest surahs the assembled audio is held in RAM next to the decoded sources and the clips, so watching memory during the save is the first thing to check.

The interpretation is also brought into the recitation's format **once** instead of clip by clip, and that conversion is kept for the repair passes, which used to convert all ~570 clips again each time.

Not done, on purpose: writing an ffmpeg `concat` list and letting ffmpeg cut the clips itself. The cut positions and the 8 ms fades are decided sample-accurately in Python, mp3 seeking is not sample-accurate, the two sources are usually in different formats (which the concat demuxer does not tolerate), and it would not touch the encode, which is where the time goes. Splitting the encode across cores is not an option either: separately encoded mp3 pieces do not join gaplessly.

```bash
python test_save.py            # streaming save, format handling, assemble end to end (needs ffmpeg)
python test_save.py --bench    # reproduces the tables above on your machine
```

---

## Output

```
final/
├── 114_الناس_Spanish.mp3
├── 114_الناس_Spanish.map.json     ← Gemini's timings, reused on the next run
├── 114_الناس_Persian.mp3
├── 114_الناس_Persian.map.json
└── …
```

`*.map.json` files can be edited by hand to fix a boundary (times are `MM:SS.mmm`); the next run uses them unless "Ask Gemini again" is ticked. A map that fails the checks is replaced by a fresh Gemini answer. A map that produced a mismatch in the final check is renamed `*.map.rejected.json` (kept for inspection) so the next run asks Gemini again instead of trusting it.

Less common settings (`timing_source`, `transcribe_model`, `transcribe_chunk_min`, `rec_language_code`, gaps between clips, output format and bitrate, maximum intro length, retry counts, snap window, recitation pause length, `verify_retries`, `check_chunk_min`, `api_retries`, `stream_save`, `mp3_quality`, the digest window sizes `digest_rec_ms` / `digest_taf_ms` / `digest_taf_tail_ms` / `digest_rec_head_ms` / `digest_verses`, `boundary_tol`, and `retry_temperature`) are fields of `Settings` in `engine.py`.

---

## Troubleshooting

| Message | What to do |
|---|---|
| `the transcription model cannot be used (...)` | Needs `pip install -U google-genai` (the Interactions API), a key with access to the preview model, or a network that reaches it. In *Auto* the chat model listens instead; in *Gemini 3.5 Transcribe only* the job stops. |
| `mapping rejected: you returned N start indices…` | The mapping model gave the wrong number of verse starts. It is told exactly what was wrong and asked again (text only, cheap). If it keeps failing, the transcript probably lacks a verse's explanation: open `<n>.words.json` and check that part of the recording. |
| `NEEDS REVIEW: you returned N entries…` | Gemini could not find the right number of verses. Check that the file really is this surah and that the surah folder name is right; try "Ask Gemini again" or a stronger model. |
| `NEEDS REVIEW: verse 1 starts at …s` | Gemini merged verse 1 into the intro. Retry, or edit the `.map.json`. If your intro is genuinely longer than 25 s, raise `max_intro` in `Settings`. |
| `the … clip for verse N holds no audible speech` | Measured, not guessed: that verse's clip is silence. Usually the timing map is wrong there — check the `.map.json` around that verse, or tick "Ask Gemini again". |
| `the recitation of verse N is still being spoken where the clip ends` | A cut landed inside a word. It is only logged unless **Act on measured mid-word cuts** is on. Common where the reciter runs verses together; letting Gemini mark the recitation (**Recitation cutting: Always**) usually fixes it. |
| `[spill at the … of verse N]` or `NEEDS REVIEW: … (spill at interpretation start): …` | A word of the neighbouring verse is at that edge of that clip. The repair moves the cut one pause at a time; if it still fails after `verify_retries` attempts, the cut cannot be placed reliably from Gemini's times. Edit the `.map.json` for that verse (times are `MM:SS.mmm`), or for the recitation delete `<no>_<name>.recitation.json` and set **Recitation cutting** to *Always let Gemini mark the verses*. |
| `N of M verse boundaries have no clear pause between the explanations` | Informational. Those cuts had no real pause to sit in (the speaker runs the explanations together) and used the quietest instant inside Gemini's interval. They are the boundaries most likely to be reported by the check. |
| `unable to download video data: HTTP Error 403` | YouTube blocked that video for one client, often only temporarily. The app already retries failed videos through other YouTube clients. If some still fail: press the download button again later (finished videos are never re-downloaded), run `pip install -U --pre "yt-dlp[default]"`, or pick your browser in **Browser cookies**. |
| `NEEDS REVIEW: final check still failing after N repair attempt(s)…` | The log above it lists each problem with its position in the finished file. Listen there. Raise `verify_retries` in `Settings` for more attempts. |
| `NEEDS REVIEW: verse N's interpretation still does not match after re-listening to the whole recording` | The recording itself probably lacks or misorders that verse's explanation. Listen at the time shown in the message. |
| `recitation clip N is really verse M…` | The recitation cut is wrong even after a re-cut: wrong recitation file for this surah, or try a lower `min_silence`. |
| `Gemini error while …` | Gemini's own message is shown. Temporary errors (429/503/timeouts) are retried automatically; a rejected API key or blocked request is not. |
| `no recitation file found` | The recitation file name does not contain the surah name or number. Rename it or check the Recitations folder. |
| `recitation: could not find N pauses` / `verse N's clip is Xs …` | In **Auto** mode this triggers Gemini marking the verses automatically (you'll see "asking Gemini to mark every verse instead"). The message only stops the job if that fails too: wrong recitation file for this surah, or a recitation longer than about 75 minutes (too long for one request). |
| `audio is too long to send to Gemini inline (>18 MB…)` | Very long recordings (about 75 minutes or more) are sent in one request; split the audio. For a very long recitation, set **Recitation cutting** to *Silences only*. Lifting this properly needs the Files API — see the note above. |
| `no Gemini API key` | Paste the key in the app (or set `GEMINI_API_KEY` before starting it). |
| `skipped (already built…)` | The finished file exists. Tick **Overwrite finished files**. |
| Nothing detected | Interpretation file names must look like `<Language>_<number> …_Tafseer`. |
| `streaming save failed (...) - using the standard export instead` | ffmpeg rejected the streamed save (the reason is in the message). The file is still written through pydub, which has the 4 GiB limit described above. |
| A `<name>.mp3.part` file is left in the output folder | A save was killed mid-way (power cut, task killed). It is incomplete; delete it. The finished file is only ever created by renaming a complete `.part`. |
| The cost report is empty | `langchain-google-genai` is older than 2.0. `pip install -U "langchain-google-genai>=2.0"`. |

---

## Project files

| File | Purpose |
|---|---|
| `app.py` | Flet desktop app |
| `engine.py` | file discovery, YouTube download, audio cutting, the LangGraph pipeline |
| `test_equiv.py` | proves the audio analysis still returns exactly what it used to |
| `test_new.py` | benchmarks, the measured gate, and how much audio the check sends |
| `test_boundary.py` | cut placement, cut-moving repair and the two-join check, including a full assemble → verify → repair → assemble loop |
| `test_transcribe.py` | the transcription timing source: response parsing, chunking at pauses, the numbered transcript, answer checks, the retry loop, the fallback and the transcript cache |
| `test_save.py` | the streaming save, format handling and `assemble` end to end; `--bench` reproduces the timing tables |
| `requirements.txt` | Python dependencies (ffmpeg is installed separately) |
