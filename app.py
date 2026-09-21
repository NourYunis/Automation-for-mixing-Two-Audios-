"""app.py - Flet front-end for engine.py     run:  python app.py"""
import os
import threading
from pathlib import Path

import flet as ft

from engine import SURAHS, Settings, run_batch, scan_languages


def main(page: ft.Page):
    page.title = "Surah Audio Builder"
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO
    store = page.client_storage

    def saved(key, default=""):
        v = store.get(key)
        return default if v is None else v

    def saved_bool(key, default=False):
        v = store.get(key)
        return default if v is None else bool(v)

    # ---- folder pickers -------------------------------------------------- #
    fields = {}

    def folder_row(key, label, on_pick=None):
        tf = ft.TextField(label=label, value=saved(key), expand=True, dense=True)
        fields[key] = tf

        def picked(e: ft.FilePickerResultEvent):
            if e.path:
                tf.value = e.path
                tf.update()
                if on_pick:
                    on_pick()

        picker = ft.FilePicker(on_result=picked)
        page.overlay.append(picker)
        return ft.Row([tf, ft.IconButton(ft.Icons.FOLDER_OPEN, tooltip="Browse",
                                         on_click=lambda _: picker.get_directory_path())])

    rows = [folder_row("output_dir", "Interpretation folder (one sub-folder per surah)", lambda: detect()),
            folder_row("recitations_dir", "Recitations folder (full-surah recitation files)"),
            folder_row("final_dir", "Save finished files to"),
            folder_row("docx_dir", "Tafsir .docx folder (optional - only a reference for Gemini)")]

    url = ft.TextField(label="YouTube playlist link (optional - downloaded into the Recitations folder first)",
                       value=saved("url"), dense=True)

    cookies = ft.Dropdown(label="Browser cookies (only if YouTube blocks the download)", value=saved("cookies", ""),
                          width=380, dense=True,
                          options=[ft.dropdown.Option("", "None"), ft.dropdown.Option("firefox", "Firefox"),
                                   ft.dropdown.Option("edge", "Edge"), ft.dropdown.Option("chrome", "Chrome"),
                                   ft.dropdown.Option("brave", "Brave")])

    # ---- options --------------------------------------------------------- #
    api = ft.TextField(label="Gemini API key", password=True, can_reveal_password=True, dense=True,
                       value=saved("api", os.environ.get("GEMINI_API_KEY", "")))
    model = ft.TextField(label="Gemini model", value=saved("model", "gemini-2.5-flash"),
                         dense=True, width=240)
    lang_row = ft.Row(wrap=True, spacing=10)
    lang_info = ft.Text("Choose the interpretation folder to detect languages.", size=12,
                        color=ft.Colors.GREY_500)

    def detect(_=None):
        """Languages/dialects come from the file names: '<Language>_<no> سورة <name>_Tafseer'."""
        d = fields["output_dir"].value
        lang_row.controls.clear()
        if d and Path(d).is_dir():
            found = scan_languages(d, *sorted((int(surah_from.value), int(surah_to.value))))
            for lang, n in found.items():
                lang_row.controls.append(ft.Checkbox(label=f"{lang} ({n})", data=lang, value=True))
            lang_info.value = (f"{len(found)} language(s) detected from the file names - untick any you "
                               f"don't want." if found else "No '<Language>_<no> ..._Tafseer' audio files found.")
        page.update()
    surah_opts = lambda: [ft.dropdown.Option(str(i), f"{i:03d} - {n}") for i, (n, _) in enumerate(SURAHS, 1)]
    surah_from = ft.Dropdown(label="From surah", value="1", width=240, dense=True, options=surah_opts())
    surah_to = ft.Dropdown(label="To surah", value="114", width=240, dense=True, options=surah_opts())
    basmala = ft.Dropdown(label="Basmala", value="keep", width=200, dense=True,
                          options=[ft.dropdown.Option("keep", "Keep before verse 1"),
                                   ft.dropdown.Option("drop", "Remove"),
                                   ft.dropdown.Option("none", "Recording has none")])
    rec_align = ft.Dropdown(label="Recitation cutting", value=saved("rec_align", "auto"), width=330, dense=True,
                            options=[ft.dropdown.Option("auto", "Auto - Gemini marks the verses only if silences fail"),
                                     ft.dropdown.Option("gemini", "Always let Gemini mark the verses"),
                                     ft.dropdown.Option("silence", "Silences only (never use Gemini)")])
    timing = ft.Dropdown(label="Timing source", value=saved("timing", "auto"), width=430, dense=True,
                         options=[ft.dropdown.Option("auto", "Auto - Gemini 3.5 Transcribe, chat model if unavailable"),
                                  ft.dropdown.Option("transcribe", "Gemini 3.5 Transcribe only"),
                                  ft.dropdown.Option("chat", "Chat model listens to the audio (old way)")])
    retranscribe = ft.Switch(label="Transcribe again (ignore saved transcripts)")
    opening_check = ft.Switch(label="Hear what the recitation opens with", value=saved_bool("opening_check", True),
                              tooltip="One short question per surah: is the first thing recited the isti'adha, "
                                      "the basmala, a spoken title, or verse 1 itself? Off: it is assumed from "
                                      "the surah number, and a reciter who opens Al-Fatiha or At-Tawbah with "
                                      "the isti'adha puts every verse one clip out of place.")
    # Dead air and cut-off words are measured directly, so the final check only has to ask the one
    # question a measurement cannot answer: is this the right verse's explanation?
    verify_mode = ft.Dropdown(label="Final check", value=saved("verify_mode", "digest"), width=370, dense=True,
                              options=[ft.dropdown.Option("digest", "Verse junctions only (default - cheapest)"),
                                       ft.dropdown.Option("full", "Listen to the whole finished file"),
                                       ft.dropdown.Option("local", "Measured checks only (no audio sent)")])
    local_cutoff = ft.Switch(label="Act on measured mid-word cuts", value=saved_bool("local_cutoff", False),
                             tooltip="Off: cut-off words are measured and logged, but the audio check "
                                     "decides. On: they count as problems straight away.")
    overwrite = ft.Switch(label="Overwrite finished files")
    relisten = ft.Switch(label="Ask Gemini again (ignore saved timing maps)")
    dry = ft.Switch(label="Dry run (analyse only, write no audio)")

    # ---- run / log ------------------------------------------------------- #
    bar = ft.ProgressBar(value=0)
    spinner = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
    status = ft.Text("Idle", size=14, weight=ft.FontWeight.W_500, expand=True)
    spend = ft.Text("", size=12, color=ft.Colors.GREY_500, selectable=True)

    def set_status(msg):
        status.value = msg
        page.update()
    log = ft.ListView(height=340, spacing=1, auto_scroll=True)
    run_btn = ft.FilledButton("Build", icon=ft.Icons.PLAY_ARROW)
    dl_btn = ft.OutlinedButton("Download recitations only", icon=ft.Icons.DOWNLOAD)
    stop_btn = ft.OutlinedButton("Stop", icon=ft.Icons.STOP, disabled=True)
    stop_ev = threading.Event()

    def line_colour(msg):
        """Red for a failure, amber for a warning, green for a measured saving.
        (The old version read as `A or (B and C)`, so most error lines were never coloured.)"""
        low, lead = msg.lower(), msg.lstrip()
        if "needs review" in low or lead.startswith("✗") or "-> error" in low:
            return ft.Colors.RED_400
        if lead.startswith("!") or lead.startswith("⚠"):
            return ft.Colors.AMBER_400
        if lead.startswith("Gemini:") or "instead of" in msg or "without asking Gemini" in msg:
            return ft.Colors.GREEN_300
        return None

    def add(msg):
        log.controls.append(ft.Text(msg, size=12, selectable=True, color=line_colour(msg),
                                    font_family="Consolas"))
        page.update()

    def progress(done, total):
        bar.value = done / max(total, 1)
        page.update()

    def start(download_only=False):
        if url.value.strip() and not fields["recitations_dir"].value:
            return add("✗ Choose the Recitations folder - the playlist is downloaded into it.")
        if download_only and not url.value.strip():
            return add("✗ Paste the YouTube playlist link first.")
        chosen = tuple(c.data for c in lang_row.controls if c.value)
        if not download_only:
            if not api.value.strip():
                return add("✗ Paste your Gemini API key first.")
            if not fields["output_dir"].value or not Path(fields["output_dir"].value).is_dir():
                return add("✗ Choose the interpretation folder.")
            if not chosen:
                return add("✗ No language ticked - press Detect first.")
            if not fields["final_dir"].value:
                return add("✗ Choose where to save the finished files.")
        for k, tf in fields.items():
            store.set(k, tf.value or "")
        store.set("api", api.value)
        store.set("model", model.value)
        store.set("url", url.value)
        store.set("cookies", cookies.value or "")
        store.set("rec_align", rec_align.value or "auto")
        store.set("verify_mode", verify_mode.value or "digest")
        store.set("local_cutoff", bool(local_cutoff.value))
        store.set("timing", timing.value or "auto")
        store.set("opening_check", bool(opening_check.value))

        lo, hi = int(surah_from.value), int(surah_to.value)
        if lo > hi:
            return add("✗ 'From surah' must not come after 'To surah'.")
        stop_ev.clear()
        cfg = Settings(
            api_key=api.value.strip(), model=model.value.strip(),
            playlist_url=url.value.strip(), download_only=download_only,
            cookies_browser=cookies.value or "",
            output_dir=Path(fields["output_dir"].value or "."),
            recitations_dir=Path(fields["recitations_dir"].value) if fields["recitations_dir"].value else None,
            final_dir=Path(fields["final_dir"].value or "final"),
            docx_dir=Path(fields["docx_dir"].value) if fields["docx_dir"].value else None,
            languages=chosen, surah_from=lo, surah_to=hi, basmala=basmala.value,
            rec_align=rec_align.value or "auto",
            timing_source=timing.value or "auto", retranscribe=bool(retranscribe.value),
            opening_check=bool(opening_check.value),
            verify_mode=verify_mode.value or "digest", local_cutoff=bool(local_cutoff.value),
            overwrite=overwrite.value, relisten=relisten.value, dry_run=dry.value,
            log=add, progress=progress, status=set_status, stop=stop_ev)
        log.controls.clear()
        bar.value = 0
        spend.value = ""
        run_btn.disabled = dl_btn.disabled = True
        stop_btn.disabled = False
        spinner.visible = True
        status.value = "Starting..."
        page.update()

        def work():
            try:
                run_batch(cfg)
            except Exception as e:
                add(f"✗ ERROR: {e}")
            if stop_ev.is_set():
                status.value = "Stopped"
            u = cfg.usage
            bits = []
            if u.get("transcribed_s"):
                bits.append(f"{u['transcribed_s'] / 60:,.1f} min transcribed")
            if u["calls"]:
                bits.append(f"{u['calls']} Gemini call(s), {u['input']:,} input + {u['output']:,} output tokens")
            if bits:
                spend.value = "This run: " + "; ".join(bits)
            run_btn.disabled = dl_btn.disabled = False
            stop_btn.disabled = True
            spinner.visible = False
            page.update()

        threading.Thread(target=work, daemon=True).start()

    surah_from.on_change = surah_to.on_change = detect
    run_btn.on_click = lambda _: start(False)
    dl_btn.on_click = lambda _: start(True)

    def stop(_):
        stop_ev.set()
        stop_btn.disabled = True
        status.value = "⏹ Stopping - finishing the current step..."
        add("⏹ Stop requested - it will stop as soon as the current step ends "
            "(a running Gemini request or download has to finish first).")

    stop_btn.on_click = stop

    page.add(
        ft.Text("Surah Audio Builder", size=24, weight=ft.FontWeight.BOLD),
        ft.Text("verse recitation → Gemini-cut interpretation → one merged file per language",
                color=ft.Colors.GREY_500),
        *rows,
        url,
        cookies,
        ft.Row([api, model], vertical_alignment=ft.CrossAxisAlignment.START),
        ft.Row([ft.Text("Languages / dialects:"), ft.TextButton("Detect", icon=ft.Icons.REFRESH, on_click=detect)]),
        lang_row, lang_info,
        ft.Row([surah_from, surah_to, basmala, rec_align], wrap=True),
        ft.Row([timing, retranscribe, opening_check], wrap=True),
        ft.Row([verify_mode, local_cutoff], wrap=True),
        ft.Row([overwrite, relisten, dry], wrap=True),
        ft.Row([run_btn, dl_btn, stop_btn]),
        ft.Row([spinner, status]),
        bar,
        spend,
        ft.Container(log, border=ft.border.all(1, ft.Colors.GREY_700), border_radius=6, padding=8),
    )

    detect()


if __name__ == "__main__":
    ft.app(target=main)
