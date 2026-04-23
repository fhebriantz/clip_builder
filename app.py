"""Gradio Web UI untuk AI Video Clipper — akses semua fitur CLI via browser.

Jalankan:
  python app.py

Buka http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "Output_Clips"
TRANSCRIPT_DIR = PROJECT_ROOT / "transcripts"

HAS_GROQ = bool(os.environ.get("GROQ_API_KEY"))


def _list_existing_clips() -> set[str]:
    if not OUTPUT_DIR.exists():
        return set()
    return {p.name for p in OUTPUT_DIR.glob("*")}


def _load_meta(clip_path: Path) -> dict:
    meta = clip_path.with_suffix(".meta.json")
    if meta.exists():
        try:
            return json.loads(meta.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _format_meta_display(meta: dict) -> str:
    if not meta:
        return "_Metadata tidak tersedia (fitur AI tidak aktif)_"
    lines: list[str] = []
    if meta.get("titles"):
        lines.append("### 🎯 Title Options")
        for i, t in enumerate(meta["titles"], 1):
            lines.append(f"{i}. **{t}**")
    if meta.get("description"):
        lines.append("\n### 📝 Description")
        lines.append(meta["description"])
    if meta.get("hashtags"):
        lines.append("\n### 🏷️ Hashtags")
        lines.append(" ".join(meta["hashtags"]))
    if meta.get("hook"):
        h = meta["hook"]
        lines.append("\n### ⚡ Hook Moment")
        lines.append(f"- **Timestamp**: {h['hook_start']}s → {h['hook_end']}s (durasi {h['hook_duration']}s)")
        lines.append(f"- **Strength**: {h['strength']}/10")
        lines.append(f"- **Text**: *\"{h['text']}\"*")
        lines.append(f"- **Alasan**: {h['reason']}")
    return "\n".join(lines) if lines else "_Metadata kosong_"


def run_pipeline(
    url,
    batch_file,
    model,
    resolution,
    aspect,
    max_clips,
    # Subtitle
    subtitle_color,
    font_size,
    subtitle_margin_top,
    subtitle_min_words,
    subtitle_max_words,
    # Crop
    smart_crop,
    # Strategy
    strategy,
    target_duration,
    silence_threshold,
    min_clip_duration,
    highlight_keywords,
    min_score,
    # AI features
    use_metadata,
    use_hook,
    render_hook,
    open_hook,
    use_polish,
    polish_topic,
    polish_vocab,
    polish_fix,
    use_translate,
    translate_to,
    # Advanced
    language,
    initial_prompt,
    parallel,
    encoder,
    no_cache,
    no_viral,
    progress=gr.Progress(track_tqdm=False),
):
    """Build CLI args and run main.py subprocess, collect outputs."""
    if not url or not url.strip():
        if batch_file is None:
            return "❌ Error: kasih URL atau upload file batch .txt", None, None, None, None, None

    before = _list_existing_clips()
    cmd = [sys.executable, "main.py", "-y"]

    if batch_file is not None:
        batch_path = PROJECT_ROOT / "local_videos" / "batch_webui.txt"
        batch_path.parent.mkdir(exist_ok=True)
        src = Path(batch_file.name if hasattr(batch_file, "name") else str(batch_file))
        batch_path.write_bytes(src.read_bytes())
        cmd += ["--batch", str(batch_path)]
    else:
        cmd.append(url.strip())

    cmd += [
        "--model", model,
        "--output-resolution", str(int(resolution)),
        "--aspect", aspect,
        "--max-clips", str(int(max_clips)),
        "--subtitle-color", subtitle_color,
        "--font-size", str(int(font_size)),
        "--subtitle-margin-top", str(float(subtitle_margin_top)),
        "--subtitle-min-words", str(int(subtitle_min_words)),
        "--subtitle-max-words", str(int(subtitle_max_words)),
        "--strategy", strategy,
        "--target-duration", str(float(target_duration)),
        "--silence-threshold", str(float(silence_threshold)),
        "--min-clip-duration", str(float(min_clip_duration)),
        "--parallel", str(int(parallel)),
        "--encoder", encoder,
    ]
    if smart_crop:
        cmd.append("--smart-crop")
    if highlight_keywords and highlight_keywords.strip():
        cmd += ["--highlight-keywords", highlight_keywords.strip()]
    if min_score and int(min_score) != 1:
        cmd += ["--min-score", str(int(min_score))]
    if language and language.strip() and language.lower() != "auto":
        cmd += ["--language", language.strip()]
    if initial_prompt and initial_prompt.strip():
        cmd += ["--initial-prompt", initial_prompt.strip()]
    if no_cache:
        cmd.append("--no-cache")
    if no_viral:
        cmd.append("--no-viral")

    if use_metadata:
        cmd.append("--ai-metadata")
    if use_hook:
        cmd.append("--ai-hook")
    if render_hook:
        cmd.append("--render-hook")
    if open_hook:
        cmd.append("--open-hook")
    if use_polish:
        cmd.append("--ai-polish")
        if polish_topic and polish_topic.strip():
            cmd += ["--polish-topic", polish_topic.strip()]
        if polish_vocab and polish_vocab.strip():
            cmd += ["--polish-vocab", polish_vocab.strip()]
        if polish_fix and polish_fix.strip():
            cmd += ["--polish-fix", polish_fix.strip()]
    if use_translate and translate_to and translate_to.strip():
        cmd += ["--ai-translate", translate_to.strip()]

    progress(0.02, desc="Memulai pipeline...")
    print(f"\n{'='*60}\n[Web UI] Running: {' '.join(cmd)}\n{'='*60}\n", flush=True)

    _PHASES = [
        ("download", 0.08, "Download video..."),
        ("✓ video:", 0.12, "Download selesai"),
        ("✓ audio:", 0.15, "Audio ter-ekstrak"),
        ("Transkripsi", 0.20, "Whisper transcribe..."),
        ("Loading Whisper", 0.22, "Loading model Whisper..."),
        ("cache hit", 0.50, "Cache hit — skip Whisper"),
        ("ai polish", 0.55, "AI polish subtitle..."),
        ("AI translate", 0.60, "Translate subtitle..."),
        ("density", 0.65, "Pilih clip density..."),
        ("highlight", 0.65, "Pilih clip highlight..."),
        ("Smart crop", 0.75, "Face tracking + render..."),
        ("Clip 1:", 0.78, "Render clip 1..."),
        ("Clip 2:", 0.82, "Render clip 2..."),
        ("Clip 3:", 0.86, "Render clip 3..."),
        ("AI metadata", 0.90, "Generate metadata AI..."),
        ("hook teaser", 0.93, "Render hook teaser..."),
        ("open hook", 0.94, "Prepend open hook..."),
        ("Selesai", 0.97, "Hampir selesai..."),
    ]

    all_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except Exception as e:
        return f"❌ Gagal start pipeline: {e}", None, None, None, None, None

    try:
        for line in proc.stdout:
            all_lines.append(line)
            print(line, end="", flush=True)
            low = line.lower()
            for phrase, pct, desc in _PHASES:
                if phrase.lower() in low:
                    progress(pct, desc=desc)
                    break
        proc.wait(timeout=7200)
    except subprocess.TimeoutExpired:
        proc.kill()
        return "❌ Timeout setelah 2 jam", None, None, None, None, None

    if proc.returncode != 0:
        err_tail = "".join(all_lines[-20:])
        return f"### ❌ Error (exit {proc.returncode})\n```\n{err_tail}\n```", None, None, None, None, None

    progress(0.98, desc="Mengumpulkan hasil...")

    after = _list_existing_clips()
    new_files = after - before

    main_clips = sorted(
        OUTPUT_DIR / f for f in new_files
        if f.endswith(".mp4") and "_hook" not in f
    )
    hook_clips = sorted(
        OUTPUT_DIR / f for f in new_files
        if f.endswith("_hook.mp4")
    )

    if not main_clips:
        return "⚠️ Selesai tapi tidak ada clip baru terdeteksi", None, None, None, None, None

    first_clip = str(main_clips[0])
    all_files = [str(p) for p in main_clips] + [str(p) for p in hook_clips]

    first_meta = _load_meta(main_clips[0])
    meta_md = _format_meta_display(first_meta)
    all_meta = {p.name: _load_meta(p) for p in main_clips}

    stdout_tail = "".join(all_lines[-10:]).rstrip()
    status_md = (
        f"### ✅ Selesai — {len(main_clips)} clip dihasilkan\n\n"
        + (f"🎬 {len(hook_clips)} hook teaser tambahan\n\n" if hook_clips else "")
        + f"**Preview clip pertama di panel kanan.**\n\n"
        + f"<details><summary>Log terminal (10 baris terakhir)</summary>\n\n```\n{stdout_tail}\n```\n</details>"
    )

    progress(1.0, desc="Done")
    return status_md, first_clip, all_files, meta_md, all_meta, None


def build_app():
    # Nilai awal strategy menentukan visibilitas field turunannya
    _default_strategy = "ai" if HAS_GROQ else "density"
    _show_keywords = _default_strategy in ("highlight", "both")
    _show_silence = _default_strategy in ("density", "both")

    with gr.Blocks(title="AI Video Clipper") as app:
        gr.Markdown(
            "# 🎬 AI Video Clipper\n"
            "Ubah video YouTube panjang jadi clip viral siap upload ke TikTok, Reels, atau Shorts.\n\n"
            "💡 **Tip pertama kali pakai:** kasih 1 URL, pakai default semua, klik Generate. "
            "Setelah hasil muncul baru eksplor fitur advanced."
        )

        ai_banner = (
            "✅ **GROQ_API_KEY terdeteksi** — fitur AI tersedia"
            if HAS_GROQ
            else "⚠️ **GROQ_API_KEY tidak ketemu di `.env`.** Fitur AI akan di-skip. "
                 "[Daftar gratis di console.groq.com](https://console.groq.com)"
        )
        gr.Markdown(ai_banner)

        with gr.Row():
            # ═══ LEFT: INPUTS ═══
            with gr.Column(scale=1):

                # — Quick action (rate-limit) —
                with gr.Row():
                    disable_ai_btn = gr.Button(
                        "🚫 Matikan Semua AI",
                        variant="secondary", size="sm",
                        visible=HAS_GROQ,
                    )
                gr.Markdown(
                    "<sub>💡 Klik **Matikan AI** kalau rate-limit Groq tercapai atau mau run cepat tanpa API.</sub>",
                    visible=HAS_GROQ,
                )

                # — Input —
                gr.Markdown("### 📥 Input")
                url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://youtu.be/VIDEO_ID  atau channel URL",
                    info="Single video, channel (ambil N terbaru), atau Shorts URL",
                )
                batch_file = gr.File(
                    label="Atau upload file batch (.txt berisi list URL)",
                    file_types=[".txt"],
                    type="filepath",
                )
                gr.Markdown(
                    "<sub>📋 Format file batch: 1 URL per baris. Baris diawali `#` di-skip. "
                    "Kalau pakai batch, field URL di atas diabaikan.</sub>"
                )

                # — Basic Settings —
                gr.Markdown("### ⚙️ Basic Settings")
                with gr.Row():
                    model = gr.Dropdown(
                        ["tiny", "base", "small", "medium", "large-v3"],
                        value="small", label="Whisper Model",
                        info="small = balance terbaik bhs Indonesia",
                    )
                    resolution = gr.Dropdown(
                        [720, 1080], value=1080, label="Quality",
                        info="720p ~2x lebih cepat dari 1080p",
                    )
                    aspect = gr.Dropdown(
                        ["9:16", "1:1", "16:9"], value="9:16",
                        label="Aspect Ratio",
                        info="9:16 TikTok/Reels, 1:1 IG feed, 16:9 YouTube",
                    )

                gr.Markdown(
                    "<sub>⏱️ **Estimasi Whisper (CPU, audio 20 menit):** "
                    "tiny 3min · base 8min · **small 15min** · medium 30min · large-v3 50min. "
                    "Cache aktif — run kedua dengan setting sama instan skip Whisper.</sub>"
                )

                with gr.Row():
                    max_clips = gr.Slider(
                        1, 10, value=3, step=1, label="Jumlah Clip",
                        info="Waktu render proporsional dengan jumlah clip",
                    )
                    subtitle_color = gr.Radio(
                        ["yellow", "white"], value="yellow", label="🎨 Warna Subtitle",
                        info="Kuning klasik viral, putih netral",
                    )

                # — Subtitle Style —
                with gr.Accordion("🎨 Tuning Subtitle (font, posisi, panjang)", open=False):
                    font_size = gr.Slider(
                        10, 24, value=14, step=1, label="Font Size",
                        info="14 optimal mobile, 18+ kalau mau besar",
                    )
                    subtitle_margin_top = gr.Slider(
                        0.5, 0.95, value=0.75, step=0.05,
                        label="Posisi Subtitle (% dari atas)",
                        info="0.75 = lower-third (viral). 0.5 = tengah. 0.9 = bawah banget",
                    )
                    with gr.Row():
                        subtitle_min_words = gr.Slider(
                            2, 10, value=4, step=1, label="Min Kata/Chunk",
                            info="Min 4 kata per subtitle chunk",
                        )
                        subtitle_max_words = gr.Slider(
                            4, 12, value=6, step=1, label="Max Kata/Chunk",
                            info="Max 6 kata = gaya TikTok viral",
                        )

                # — Face Tracking —
                with gr.Accordion("📹 Face Tracking & Crop", open=True):
                    smart_crop = gr.Checkbox(
                        label="Smart crop (auto-follow speaker)",
                        value=True,
                        info="+30-60s per clip. Kamera otomatis follow wajah pembicara. "
                             "Wajib untuk podcast/interview.",
                    )

                # — Clipping Strategy —
                with gr.Accordion("✂️ Strategi Clipping", open=True):
                    strategy = gr.Radio(
                        ["density", "highlight", "both", "ai"],
                        value=_default_strategy,
                        label="Metode pilih clip",
                        info="density=urut dari awal · ai=LLM pilih viral moments (butuh Groq)",
                    )
                    with gr.Row():
                        target_duration = gr.Slider(
                            15, 120, value=60, step=5,
                            label="Target Durasi Clip (detik)",
                            info="60 detik optimal untuk Reels/Shorts",
                        )
                        min_clip_duration = gr.Slider(
                            10, 60, value=20, step=5,
                            label="Min Durasi",
                            info="Buang clip lebih pendek dari ini",
                        )
                    silence_threshold = gr.Slider(
                        0.5, 5, value=2, step=0.5,
                        label="Silence Threshold (detik)",
                        info="Jeda sunyi >= ini = batas clip (density/both only)",
                        visible=_show_silence,
                    )
                    with gr.Row():
                        highlight_keywords = gr.Textbox(
                            label="Highlight Keywords",
                            placeholder="bisnis,profit,strategi",
                            info="Dipisah koma. Hanya untuk strategy highlight/both.",
                            visible=_show_keywords,
                        )
                        min_score = gr.Slider(
                            1, 5, value=1, step=1, label="Min Score",
                            info="Threshold keyword match (highlight/both only)",
                            visible=_show_keywords,
                        )

                # — AI Features —
                ai_disabled = not HAS_GROQ
                with gr.Accordion(
                    f"🤖 AI Features{'  ·  Groq tidak terdeteksi' if ai_disabled else ''}",
                    open=HAS_GROQ,
                ):
                    if ai_disabled:
                        gr.Markdown(
                            "⚠️ Fitur AI tidak aktif. Tambahkan `GROQ_API_KEY` ke file `.env` "
                            "untuk mengaktifkan. [Daftar gratis](https://console.groq.com)"
                        )

                    use_metadata = gr.Checkbox(
                        label="Generate Title + Description + Hashtag",
                        value=HAS_GROQ,
                        interactive=not ai_disabled,
                        info="+1-2s per clip. 3 opsi title viral, caption, 8 hashtag disimpan ke .meta.json",
                    )

                    # Hook group
                    use_hook = gr.Checkbox(
                        label="Detect Hook Moment (3-5 detik puncak)",
                        value=HAS_GROQ,
                        interactive=not ai_disabled,
                        info="+1s per clip. LLM pilih kalimat punch terkuat, timestamp disimpan",
                    )
                    with gr.Group(visible=not ai_disabled):
                        gr.Markdown("<sub>↳ Opsi tambahan setelah hook terdeteksi:</sub>")
                        with gr.Row():
                            render_hook = gr.Checkbox(
                                label="Render Teaser (file *_hook.mp4 terpisah)",
                                value=False,
                                info="+5-10s per clip. File teaser 3-5 detik ultra-short",
                            )
                            open_hook = gr.Checkbox(
                                label="Open Hook (cold open di awal clip)",
                                value=False,
                                info="+10-20s per clip. Prepend hook ke awal — [hook 3-5s] + [clip penuh]",
                            )

                    # Polish group
                    use_polish = gr.Checkbox(
                        label="Polish Subtitle (fix typo + hapus filler)",
                        value=False,
                        interactive=not ai_disabled,
                        info="+5-15s total. Fix 'umm eh apa ya' dan typo Whisper",
                    )
                    with gr.Column(visible=False) as polish_fields:
                        polish_topic = gr.Textbox(
                            label="Konteks Topik",
                            placeholder="Video mindset bisnis membahas orang serakah dan sukses",
                            info="Bantu LLM disambiguasi typo ambigu",
                        )
                        polish_vocab = gr.Textbox(
                            label="Vocab Penting (dipisah koma)",
                            placeholder="serakah,rakus,koneksi,milenial,mindset",
                            info="LLM prefer ejaan ini saat Whisper salah dengar",
                        )
                        polish_fix = gr.Textbox(
                            label="Manual Fix (salah=benar, paling reliable)",
                            placeholder="straka=serakah,conesi=koneksi",
                            info="100% di-apply — pakai untuk koreksi yang pasti",
                        )

                    # Translate
                    use_translate = gr.Checkbox(
                        label="Translate Subtitle ke Bahasa Lain",
                        value=False,
                        interactive=not ai_disabled,
                        info="+10-30s & ~4k token. Burn subtitle bahasa target ke video",
                    )
                    with gr.Column(visible=False) as translate_container:
                        translate_to = gr.Dropdown(
                            choices=[
                                ("🇬🇧 English", "en"),
                                ("🇯🇵 Japanese", "ja"),
                                ("🇨🇳 Chinese (Simplified)", "zh"),
                                ("🇰🇷 Korean", "ko"),
                                ("🇸🇦 Arabic", "ar"),
                                ("🇪🇸 Spanish", "es"),
                                ("🇫🇷 French", "fr"),
                                ("🇩🇪 German", "de"),
                                ("🇮🇹 Italian", "it"),
                                ("🇵🇹 Portuguese", "pt"),
                                ("🇷🇺 Russian", "ru"),
                                ("🇮🇳 Hindi", "hi"),
                                ("🇹🇭 Thai", "th"),
                                ("🇻🇳 Vietnamese", "vi"),
                                ("🇲🇾 Malay", "ms"),
                                ("🇳🇱 Dutch", "nl"),
                                ("🇵🇱 Polish", "pl"),
                                ("🇹🇷 Turkish", "tr"),
                                ("🇸🇪 Swedish", "sv"),
                                ("🇩🇰 Danish", "da"),
                            ],
                            label="Bahasa Target",
                            info="Pilih bahasa untuk subtitle yang di-burn ke video.",
                            allow_custom_value=True,
                        )

                # — Advanced —
                with gr.Accordion("🔧 Advanced (biarkan default kalau ragu)", open=False):
                    with gr.Row():
                        language = gr.Dropdown(
                            ["auto", "id", "en", "ja", "zh", "es", "ko", "ar"],
                            value="auto", label="Force Language",
                            info="auto = deteksi otomatis (direkomendasi)",
                        )
                        initial_prompt = gr.Textbox(
                            label="Whisper Initial Prompt",
                            placeholder="mindset bisnis koneksi serakah",
                            info="Bias Whisper ke vocab ini — boost akurasi kata spesifik",
                        )
                    with gr.Row():
                        parallel = gr.Slider(
                            1, 4, value=1, step=1, label="Parallel Render",
                            info="2-3 lebih cepat kalau pakai hw encoder (NVENC/VAAPI)",
                        )
                        encoder = gr.Dropdown(
                            ["auto", "libx264", "h264_vaapi", "h264_nvenc", "h264_qsv"],
                            value="auto", label="FFmpeg Encoder",
                            info="auto pilih terbaik (hw > sw). Override kalau troubleshoot",
                        )
                    no_cache = gr.Checkbox(
                        label="Force re-transcribe (skip cache)",
                        value=False,
                        info="Paksa Whisper ulang. Aktifkan kalau ganti audio/kasus edge",
                    )
                    no_viral = gr.Checkbox(
                        label="Skip viral render (raw cut only)",
                        value=False,
                        info="Stream copy ~instant, tanpa 9:16/subtitle/smart-crop. Buat preview cepat",
                    )

                run_btn = gr.Button(
                    "🚀 Generate Clips", variant="primary", size="lg",
                )
                gr.Markdown(
                    "<sub>💡 **Estimasi waktu total:** video 20 menit default setting "
                    "= ~15 menit run pertama (Whisper), ~2-3 menit run berikutnya (cache).</sub>"
                )

            # ═══ RIGHT: OUTPUTS ═══
            with gr.Column(scale=1):
                with gr.Row():
                    open_folder_btn = gr.Button(
                        "📁 Buka Folder Output",
                        variant="secondary", size="sm",
                    )

                status = gr.Markdown("_Siap. Masukkan URL lalu klik Generate Clips._")

                with gr.Tabs():
                    with gr.Tab("🎬 Preview"):
                        video_preview = gr.Video(
                            label="Clip Pertama", autoplay=False,
                        )
                        meta_display = gr.Markdown(
                            "_Metadata AI akan muncul di sini setelah Generate._"
                        )

                    with gr.Tab("📁 Semua File"):
                        files_list = gr.Files(
                            label="Download clips & teasers (klik file untuk download)",
                        )

                    with gr.Tab("📊 Metadata JSON"):
                        meta_json = gr.JSON(label="Raw metadata per clip")

                gr.Markdown(
                    "📂 Output tersimpan di `Output_Clips/`.\n\n"
                    "⚡ **Quick Tips:**\n"
                    "- Eksperimen cepat: **max-clips 1** untuk preview, lalu scale up\n"
                    "- Typo Whisper: aktifkan **Polish** → isi Konteks Topik atau Manual Fix\n"
                    "- Multi-platform: render 3x dengan aspect berbeda (9:16 + 1:1 + 16:9)"
                )

        dummy = gr.State(None)

        # === Event: loading state pada tombol Generate ===
        def _btn_start():
            return gr.update(interactive=False, value="⏳ Memproses... (lihat terminal untuk progress)")

        def _btn_done():
            return gr.update(interactive=True, value="🚀 Generate Clips")

        run_btn.click(
            fn=_btn_start,
            outputs=run_btn,
        ).then(
            fn=run_pipeline,
            inputs=[
                url, batch_file,
                model, resolution, aspect, max_clips,
                subtitle_color, font_size, subtitle_margin_top,
                subtitle_min_words, subtitle_max_words,
                smart_crop,
                strategy, target_duration, silence_threshold,
                min_clip_duration, highlight_keywords, min_score,
                use_metadata, use_hook, render_hook, open_hook,
                use_polish, polish_topic, polish_vocab, polish_fix,
                use_translate, translate_to,
                language, initial_prompt, parallel, encoder,
                no_cache, no_viral,
            ],
            outputs=[status, video_preview, files_list, meta_display, meta_json, dummy],
            show_progress="full",
        ).then(
            fn=_btn_done,
            outputs=run_btn,
        )

        # === Event: show/hide polish sub-fields ===
        use_polish.change(
            fn=lambda v: gr.update(visible=v),
            inputs=use_polish,
            outputs=polish_fields,
            queue=False,
            show_progress="hidden",
        )

        # === Event: show/hide translate field ===
        use_translate.change(
            fn=lambda v: gr.update(visible=v),
            inputs=use_translate,
            outputs=translate_container,
            queue=False,
            show_progress="hidden",
        )

        # === Event: render_hook / open_hook auto-enable use_hook ===
        def _maybe_enable_hook(checked):
            return gr.update(value=True) if checked else gr.update()

        render_hook.change(fn=_maybe_enable_hook, inputs=render_hook, outputs=use_hook, queue=False, show_progress="hidden")
        open_hook.change(fn=_maybe_enable_hook, inputs=open_hook, outputs=use_hook, queue=False, show_progress="hidden")

        # === Event: strategy menentukan field yang relevan ===
        def _strategy_visibility(s):
            show_kw = s in ("highlight", "both")
            show_sil = s in ("density", "both")
            return (
                gr.update(visible=show_sil),   # silence_threshold
                gr.update(visible=show_kw),    # highlight_keywords
                gr.update(visible=show_kw),    # min_score
            )

        strategy.change(
            fn=_strategy_visibility,
            inputs=strategy,
            outputs=[silence_threshold, highlight_keywords, min_score],
            queue=False,
            show_progress="hidden",
        )

        # === Event: Matikan Semua AI ===
        def _disable_all_ai():
            gr.Info("🚫 Semua fitur AI dimatikan. Strategy di-set ke density.")
            return (
                False,       # use_metadata
                False,       # use_hook
                False,       # render_hook
                False,       # open_hook
                False,       # use_polish
                gr.update(visible=False),   # polish_fields
                False,       # use_translate
                gr.update(visible=False),   # translate_container
                "density",   # strategy
            )

        disable_ai_btn.click(
            fn=_disable_all_ai,
            outputs=[
                use_metadata, use_hook, render_hook, open_hook,
                use_polish, polish_fields,
                use_translate, translate_container,
                strategy,
            ],
            queue=False,
            show_progress="hidden",
        )

        # === Event: Buka Folder Output ===
        def _open_output_folder():
            OUTPUT_DIR.mkdir(exist_ok=True)
            system = platform.system()
            try:
                if system == "Windows":
                    os.startfile(str(OUTPUT_DIR))
                elif system == "Darwin":
                    subprocess.Popen(["open", str(OUTPUT_DIR)])
                else:
                    subprocess.Popen(["xdg-open", str(OUTPUT_DIR)])
                gr.Info(f"📁 Folder terbuka: {OUTPUT_DIR}")
            except Exception as e:
                gr.Warning(f"Gagal buka folder: {e}")

        open_folder_btn.click(fn=_open_output_folder, outputs=None)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
        inbrowser=True,
    )
