"""Gradio Web UI untuk AI Video Clipper.

Jalankan:
  python app.py

Lalu buka http://127.0.0.1:7860 di browser.

UI ini wrapper di atas main.py — semua fitur CLI tersedia via toggle/dropdown.
Output (clip mp4, hook teaser, metadata JSON) langsung ditampilkan untuk preview
dan download.
"""
from __future__ import annotations

import json
import os
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
    """Snapshot nama file Output_Clips sekarang — untuk deteksi file baru."""
    if not OUTPUT_DIR.exists():
        return set()
    return {p.name for p in OUTPUT_DIR.glob("*")}


def _load_meta(clip_path: Path) -> dict:
    """Load .meta.json sibling kalau ada."""
    meta = clip_path.with_suffix(".meta.json")
    if meta.exists():
        try:
            return json.loads(meta.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _format_meta_display(meta: dict) -> str:
    """Format metadata jadi markdown untuk display."""
    if not meta:
        return "_Metadata tidak tersedia_"
    lines: list[str] = []
    titles = meta.get("titles") or []
    if titles:
        lines.append("### Title Options")
        for i, t in enumerate(titles, 1):
            lines.append(f"{i}. **{t}**")
    if meta.get("description"):
        lines.append("\n### Description")
        lines.append(meta["description"])
    if meta.get("hashtags"):
        lines.append("\n### Hashtags")
        lines.append(" ".join(meta["hashtags"]))
    if meta.get("hook"):
        h = meta["hook"]
        lines.append("\n### Hook Moment")
        lines.append(f"- Timestamp: {h['hook_start']}s → {h['hook_end']}s ({h['hook_duration']}s)")
        lines.append(f"- Strength: {h['strength']}/10")
        lines.append(f"- Text: *{h['text']}*")
        lines.append(f"- Alasan: {h['reason']}")
    return "\n".join(lines) if lines else "_Metadata kosong_"


def run_pipeline(
    url,
    model,
    resolution,
    aspect,
    max_clips,
    smart_crop,
    use_ai_strategy,
    use_polish,
    polish_topic,
    polish_fix,
    use_metadata,
    use_hook,
    render_hook,
    subtitle_color,
    initial_prompt,
    progress=gr.Progress(track_tqdm=False),
):
    """Execute pipeline, yield progress updates, return outputs."""
    if not url or not url.strip():
        return "Error: URL kosong", None, None, None, None, None

    # Snapshot before run — detect new files after
    before = _list_existing_clips()

    cmd = [
        sys.executable, "main.py", url.strip(), "-y",
        "--model", model,
        "--output-resolution", str(int(resolution)),
        "--aspect", aspect,
        "--max-clips", str(int(max_clips)),
        "--subtitle-color", subtitle_color,
    ]
    if smart_crop:
        cmd.append("--smart-crop")
    if use_ai_strategy:
        cmd += ["--strategy", "ai"]
    if use_polish:
        cmd.append("--ai-polish")
        if polish_topic and polish_topic.strip():
            cmd += ["--polish-topic", polish_topic.strip()]
        if polish_fix and polish_fix.strip():
            cmd += ["--polish-fix", polish_fix.strip()]
    if use_metadata:
        cmd.append("--ai-metadata")
    if use_hook:
        cmd.append("--ai-hook")
    if render_hook:
        cmd.append("--render-hook")
    if initial_prompt and initial_prompt.strip():
        cmd += ["--initial-prompt", initial_prompt.strip()]

    progress(0.1, desc="Menjalankan pipeline...")

    # Run subprocess, stream stdout ke log biar tidak freeze
    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=3600,
        )
    except subprocess.TimeoutExpired:
        return "Error: timeout setelah 1 jam", None, None, None, None, None

    if result.returncode != 0:
        err_tail = "\n".join(result.stderr.strip().splitlines()[-20:])
        return f"### Error (exit {result.returncode})\n```\n{err_tail}\n```", None, None, None, None, None

    progress(0.9, desc="Mengumpulkan hasil...")

    # Detect new clip files
    after = _list_existing_clips()
    new_files = after - before

    # Organize by type
    main_clips = sorted(
        OUTPUT_DIR / f for f in new_files
        if f.endswith(".mp4") and "_hook" not in f
    )
    hook_clips = sorted(
        OUTPUT_DIR / f for f in new_files
        if f.endswith("_hook.mp4")
    )

    if not main_clips:
        return "Selesai tapi tidak ada clip baru terdeteksi", None, None, None, None, None

    # Build display output — first clip preview, all clips as file list, metadata markdown
    first_clip = str(main_clips[0])
    all_files = [str(p) for p in main_clips] + [str(p) for p in hook_clips]

    # Metadata dari clip pertama
    first_meta = _load_meta(main_clips[0])
    meta_md = _format_meta_display(first_meta)

    # JSON raw semua metadata
    all_meta = {p.name: _load_meta(p) for p in main_clips}

    # Log tail
    stdout_tail = "\n".join(result.stdout.strip().splitlines()[-15:])
    status_md = (
        f"### ✓ Selesai — {len(main_clips)} clip dihasilkan\n\n"
        + (f"{len(hook_clips)} hook teaser file.\n\n" if hook_clips else "")
        + f"**Preview:** clip pertama di panel kanan.\n\n"
        + f"<details><summary>Log terminal (15 baris terakhir)</summary>\n\n```\n{stdout_tail}\n```\n</details>"
    )

    progress(1.0, desc="Done")
    return status_md, first_clip, all_files, meta_md, all_meta, None


def build_app():
    with gr.Blocks(title="AI Video Clipper") as app:
        gr.Markdown(
            "# AI Video Clipper\n"
            "Ubah video YouTube jadi clip viral untuk TikTok/Reels/Shorts — "
            "lokal, gratis, dengan face tracking + AI smart highlight."
        )

        with gr.Row():
            # === Input Panel ===
            with gr.Column(scale=1):
                url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1,
                )

                with gr.Row():
                    model = gr.Dropdown(
                        ["tiny", "base", "small", "medium", "large-v3"],
                        value="small",
                        label="Whisper Model",
                        info="small = balance terbaik untuk bahasa Indonesia",
                    )
                    resolution = gr.Dropdown(
                        [720, 1080], value=1080, label="Quality",
                        info="720p lebih cepat, 1080p FHD",
                    )
                    aspect = gr.Dropdown(
                        ["9:16", "1:1", "16:9"], value="9:16",
                        label="Aspect Ratio",
                        info="9:16 TikTok/Reels, 1:1 IG feed, 16:9 YouTube",
                    )

                max_clips = gr.Slider(
                    1, 10, value=3, step=1,
                    label="Max Clips", info="Batasi jumlah clip per video",
                )

                with gr.Accordion("Subtitle & Crop", open=True):
                    subtitle_color = gr.Radio(
                        ["yellow", "white"], value="yellow", label="Subtitle Color",
                    )
                    smart_crop = gr.Checkbox(
                        label="Smart crop (face tracking auto-follow speaker)",
                        value=True,
                    )

                ai_enabled_note = (
                    "✓ GROQ_API_KEY terdeteksi — fitur AI tersedia"
                    if HAS_GROQ
                    else "⚠ GROQ_API_KEY tidak ditemukan. Fitur AI tidak akan jalan. "
                         "Daftar gratis di console.groq.com dan set di .env"
                )
                gr.Markdown(f"### AI Features\n{ai_enabled_note}")

                with gr.Group():
                    use_ai_strategy = gr.Checkbox(
                        label="AI Smart Highlight (LLM pilih momen viral-worthy)",
                        value=HAS_GROQ,
                    )
                    use_metadata = gr.Checkbox(
                        label="Generate Title + Description + Hashtag",
                        value=HAS_GROQ,
                    )
                    use_hook = gr.Checkbox(
                        label="Detect Hook Moment (3-5 detik puncak)",
                        value=HAS_GROQ,
                    )
                    render_hook = gr.Checkbox(
                        label="Render Hook Teaser (file terpisah 3-5 detik)",
                        value=False,
                    )
                    use_polish = gr.Checkbox(
                        label="Polish Subtitle (fix typo + hapus filler)",
                        value=False,
                    )

                with gr.Accordion("Polish & Context (advanced)", open=False):
                    polish_topic = gr.Textbox(
                        label="Polish Topic Hint",
                        placeholder="Video mindset bisnis membahas orang serakah",
                        info="Bantu LLM disambiguasi typo berdasar konteks",
                    )
                    polish_fix = gr.Textbox(
                        label="Manual Corrections",
                        placeholder="straka=serakah,conesi=koneksi",
                        info="Format 'salah=benar' dipisah koma",
                    )
                    initial_prompt = gr.Textbox(
                        label="Whisper Initial Prompt",
                        placeholder="mindset bisnis serakah koneksi",
                        info="Boost akurasi Whisper untuk vocab spesifik",
                    )

                run_btn = gr.Button("Generate Clips", variant="primary", size="lg")

            # === Output Panel ===
            with gr.Column(scale=1):
                status = gr.Markdown("_Belum mulai. Masukkan URL dan klik Generate._")
                with gr.Tabs():
                    with gr.Tab("Preview (Clip 1)"):
                        video_preview = gr.Video(label="First Clip")
                        meta_display = gr.Markdown("")
                    with gr.Tab("Semua File"):
                        files_list = gr.Files(label="Download clips & teasers")
                    with gr.Tab("Metadata JSON"):
                        meta_json = gr.JSON(label="Raw metadata per clip")

        dummy = gr.State(None)

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                url, model, resolution, aspect, max_clips, smart_crop,
                use_ai_strategy, use_polish, polish_topic, polish_fix,
                use_metadata, use_hook, render_hook,
                subtitle_color, initial_prompt,
            ],
            outputs=[status, video_preview, files_list, meta_display, meta_json, dummy],
        )

        gr.Markdown(
            "---\n"
            "*Output clips tersimpan di `Output_Clips/`. Jalankan dari terminal: "
            "`python app.py` → buka http://127.0.0.1:7860*"
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
