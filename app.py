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
    overwrite = ft.Switch(label="Overwrite finished files")
    relisten = ft.Switch(label="Ask Gemini again (ignore saved timing maps)")
    dry = ft.Switch(label="Dry run (analyse only, write no audio)")

    # ---- run / log ------------------------------------------------------- #
    bar = ft.ProgressBar(value=0)
    spinner = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
    status = ft.Text("Idle", size=14, weight=ft.FontWeight.W_500, expand=True)

    def set_status(msg):
        status.value = msg
        page.update()
    log = ft.ListView(height=340, spacing=1, auto_scroll=True)
    run_btn = ft.FilledButton("Build", icon=ft.Icons.PLAY_ARROW)
    dl_btn = ft.OutlinedButton("Download recitations only", icon=ft.Icons.DOWNLOAD)
    stop_btn = ft.OutlinedButton("Stop", icon=ft.Icons.STOP, disabled=True)
    stop_ev = threading.Event()

    def add(msg):
        color = (ft.Colors.RED_400 if ("NEEDS REVIEW" in msg or "error" in msg.lower() and "->" in msg)
                 else ft.Colors.AMBER_400 if msg.strip().startswith("!") else None)
        log.controls.append(ft.Text(msg, size=12, selectable=True, color=color, font_family="Consolas"))
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
            overwrite=overwrite.value, relisten=relisten.value, dry_run=dry.value,
            log=add, progress=progress, status=set_status, stop=stop_ev)
        log.controls.clear()
        bar.value = 0
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
        ft.Row([surah_from, surah_to, basmala], wrap=True),
        ft.Row([overwrite, relisten, dry], wrap=True),
        ft.Row([run_btn, dl_btn, stop_btn]),
        ft.Row([spinner, status]),
        bar,
        ft.Container(log, border=ft.border.all(1, ft.Colors.GREY_700), border_radius=6, padding=8),
    )

    detect()


if __name__ == "__main__":
    ft.app(target=main)
