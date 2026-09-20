# Automation-for-mixing-Two-Audios-

Builds a Quran audio book where every verse is followed by its spoken interpretation (tafsir):

```
verse 1 recitation → interpretation of verse 1 → verse 2 recitation → interpretation of verse 2 → …
```

One finished file is produced per surah **and per language/dialect** (Spanish, Saudi, Persian, Moroccan, Egyptian, …). The languages are detected automatically from the file names.

The recitation is cut at its pauses. The interpretation is cut by **Gemini, which listens to the audio** and marks where each verse's explanation starts and ends. A LangGraph pipeline validates the result and asks Gemini again if it is wrong.

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
8. **Basmala**: keep it before verse 1 (default), remove it, or "recording has none". Surahs 1 and 9 are always handled correctly.
9. Switches:
   - **Overwrite finished files**: rebuild files that already exist (otherwise they are skipped).
   - **Ask Gemini again**: ignore saved timing maps and re-run the listening step.
   - **Dry run**: analyse and print the timings, write no audio.
10. **Build**, or **Download recitations only**. **Stop** ends the run after the current item.

> Tip: start with a **Dry run** on one surah. It calls Gemini once, saves the timing map, and prints what was removed and the first words of verse 1, so you can confirm the first verse is still there. The real build afterwards reuses the map at no extra cost.

### YouTube playlist download

Every video is saved as mp3 (`<playlist number> - <title>.mp3`) in the Recitations folder. Videos already downloaded are recorded in `.downloaded.txt` and skipped next time. The surah is found from the surah name in the title. Files whose title has no recognisable surah name are listed in the log ("cannot tell which surah this is"); rename them, e.g. to `سورة النبأ.mp3`. If downloads suddenly fail, run `pip install -U yt-dlp`.

---

## How it works

For each (surah, language) a LangGraph graph runs:

```
prepare → listen → check ──ok──────────────────────→ assemble → done
             ↑        │
             └────────┴─ rejected, retries left (Gemini gets the exact complaint)
                      └─ rejected, no retries left → reported as NEEDS REVIEW
```

- **prepare**: loads the audio and cuts the recitation at its (verses − 1) longest pauses. The first pause (after the basmala) is never treated as a verse end.
- **listen**: Gemini receives the interpretation audio (plus the optional reference text) and returns, for each verse, a start time, an end time and its first words, together with what the announcement and closing line say.
- **check**: rejects the answer if it does not have exactly N verses numbered 1…N, if verses overlap or are shorter than 0.8 s, if verse 1 starts later than 25 s (the intro swallowed it), or if times exceed the audio length. Rejection sends Gemini the exact reason and it tries again (up to 3 extra attempts).
- **assemble**: moves each cut onto the nearest real pause, then merges `verse → 0.4 s → interpretation → 0.9 s → next verse`. The start cut can only move earlier and the end cut only later, so a word of verse 1 can never be trimmed off.

A file that fails all attempts is reported as `NEEDS REVIEW: <reason>` in the log and the summary. It is never silently skipped, and it never stops the rest of the batch.

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

`*.map.json` files can be edited by hand to fix a boundary (times are `MM:SS.mmm`); the next run uses them unless "Ask Gemini again" is ticked. A map that fails the checks is replaced by a fresh Gemini answer.

Less common settings (gaps between clips, output format and bitrate, maximum intro length, retry count, snap window, recitation pause length) are fields of `Settings` in `engine.py`.

---

## Troubleshooting

| Message | What to do |
|---|---|
| `NEEDS REVIEW: you returned N entries…` | Gemini could not find the right number of verses. Check that the file really is this surah and that the surah folder name is right; try "Ask Gemini again" or a stronger model. |
| `NEEDS REVIEW: verse 1 starts at …s` | Gemini merged verse 1 into the intro. Retry, or edit the `.map.json`. If your intro is genuinely longer than 25 s, raise `max_intro` in `Settings`. |
| `no recitation file found` | The recitation file name does not contain the surah name or number. Rename it or check the Recitations folder. |
| `recitation: could not find N pauses` | Wrong recitation file for this surah, or the recording has too little silence between verses. |
| `interpretation audio is too long (>18 MB)` | Very long recordings are sent in one request; split the audio. |
| `no Gemini API key` | Paste the key in the app (or set `GEMINI_API_KEY` before starting it). |
| `skipped (already built…)` | The finished file exists. Tick **Overwrite finished files**. |
| Nothing detected | Interpretation file names must look like `<Language>_<number> …_Tafseer`. |

---

## Project files

| File | Purpose |
|---|---|
| `app.py` | Flet desktop app |
| `engine.py` | file discovery, YouTube download, audio cutting, the LangGraph pipeline |
| `requirements.txt` | Python dependencies (ffmpeg is installed separately) |
