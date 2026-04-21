"""CLI entry point — AI Video Clipper (pipeline penuh).

URL → download → WAV → transkripsi → highlight detection → potong clip.

Mode interaktif aktif otomatis kalau user cuma kasih URL (tanpa flag lain).
Pakai flag `--yes` untuk skip prompt dan langsung jalan pakai default.
"""
import argparse
import sys
from pathlib import Path

from rich.console import Console

from accel import describe as describe_accel, set_encoder_override
from downloader import process_input
from highlighter import generate_clips, group_by_density, pick_highlights, save_highlights
from transcriber import save_srt, save_transcript, transcribe

console = Console()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI Video Clipper — download, transkrip, dan auto-clip highlight",
    )
    p.add_argument("url", help="Link video tunggal ATAU link channel YouTube")
    p.add_argument("-k", "--keyword", default="",
                   help="Filter judul video (contoh: 'Bisnis'). Kosong = semua.")
    p.add_argument("-n", "--limit", type=int, default=3,
                   help="Jumlah video terbaru dari channel (default: 3)")
    p.add_argument("--download-dir", default="downloads")
    p.add_argument("--audio-dir", default="audio")
    p.add_argument("--transcript-dir", default="transcripts")
    p.add_argument("--clip-dir", default="Output_Clips",
                   help="Folder output clip (default: Output_Clips)")
    p.add_argument("--model", default="small",
                   choices=["tiny", "base", "small", "medium", "large-v3"],
                   help="Ukuran model Whisper (default: small — akurasi baik untuk bahasa Indonesia)")
    p.add_argument("--language", default="auto",
                   help="Bahasa audio: 'auto' (default, deteksi otomatis), 'id', 'en', dll.")
    p.add_argument("--initial-prompt", default=None,
                   help="Kata kunci konteks untuk bias vocab Whisper "
                        "(contoh: 'mindset, koneksi, bisnis, uang'). "
                        "Membantu akurasi istilah spesifik tanpa ganti model.")

    # Clipping strategy
    p.add_argument("--strategy", default="density",
                   choices=["density", "highlight", "both"],
                   help="density=bagi 60s berbasis kepadatan bicara (default); "
                        "highlight=pilih berdasarkan keyword/hook; both=dua-duanya")

    # Density-based options
    p.add_argument("--target-duration", type=float, default=60.0,
                   help="Target durasi clip density (default 60s)")
    p.add_argument("--silence-threshold", type=float, default=2.0,
                   help="Gap antar segmen > ini (detik) dianggap batas clip")
    p.add_argument("--min-clip-duration", type=float, default=20.0,
                   help="Buang clip lebih pendek dari ini (detik)")

    # Highlight-based options
    p.add_argument("--highlight-keywords", default="",
                   help="Keyword highlight dipisah koma, contoh: 'bisnis,profit,strategi'")
    p.add_argument("--min-score", type=int, default=1, help="Skor minimal segmen lolos")
    p.add_argument("--max-clip-duration", type=float, default=60.0)
    p.add_argument("--merge-gap", type=float, default=2.0,
                   help="Segmen berdekatan (detik) digabung jadi satu clip")

    # Viral rendering options
    p.add_argument("--no-viral", action="store_true",
                   help="Skip 9:16 crop + subtitle burn-in, hanya potong mentah")
    p.add_argument("--subtitle-color", default="yellow", choices=["yellow", "white"],
                   help="Warna teks subtitle (default: yellow)")
    p.add_argument("--font-size", type=int, default=14,
                   help="Ukuran font subtitle (default: 14)")
    p.add_argument("--subtitle-margin-top", type=float, default=0.75,
                   help="Posisi subtitle sebagai fraksi dari atas (default: 0.75 = 75%% dari atas)")
    p.add_argument("--font", default=None,
                   help="Nama font subtitle (default: auto per platform — "
                        "DejaVu Sans di Linux, Arial di Windows, Helvetica di macOS)")
    p.add_argument("--subtitle-min-words", type=int, default=4,
                   help="Minimal kata per chunk subtitle (default: 4)")
    p.add_argument("--subtitle-max-words", type=int, default=6,
                   help="Maksimal kata per chunk subtitle (default: 6, gaya viral)")

    # Acceleration
    p.add_argument("--encoder", default="auto",
                   choices=["auto", "libx264", "h264_vaapi", "h264_nvenc", "h264_qsv"],
                   help="Override FFmpeg encoder (default: auto)")
    p.add_argument("--output-resolution", type=int, default=1080, choices=[720, 1080],
                   help="Tinggi target clip viral: 720 (HD, ~2x lebih cepat) atau 1080 (FHD)")
    p.add_argument("--parallel", type=int, default=1,
                   help="Render N clip bersamaan (disarankan 2-3 untuk hw encoder)")
    p.add_argument("--max-clips", type=int, default=0,
                   help="Batasi jumlah clip yang di-render per video (0 = semua)")

    # UX
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip mode interaktif, langsung pakai default/flag")

    # Skip flags
    p.add_argument("--no-transcribe", action="store_true",
                   help="Skip transkripsi & highlight (hanya download + WAV)")
    p.add_argument("--no-clip", action="store_true",
                   help="Skip pemotongan clip (tetap transkripsi)")
    return p


def _ask(prompt: str, default: str) -> str:
    try:
        val = input(f"{prompt} [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[red]Dibatalkan.[/red]")
        sys.exit(0)
    return val or default


def _ask_choice(prompt: str, options: list[tuple[str, str]], default_idx: int) -> str:
    """options = [(value, description)]. default_idx 0-based."""
    console.print(f"\n[bold]{prompt}[/bold]")
    for i, (val, desc) in enumerate(options, 1):
        marker = " [default]" if i - 1 == default_idx else ""
        console.print(f"  {i}) {val:12} {desc}{marker}")
    raw = _ask("Pilih", str(default_idx + 1))
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return options[idx][0]
    except ValueError:
        pass
    return options[default_idx][0]


def interactive_prompt(args: argparse.Namespace) -> argparse.Namespace:
    """Tanya user 4 opsi utama — yang lain tetap default."""
    console.print("\n[bold cyan]Mode Interaktif[/bold cyan] [dim](tekan Enter untuk pakai default)[/dim]")

    args.model = _ask_choice(
        "Model Whisper (akurasi vs kecepatan)",
        [
            ("tiny",     "75 MB,  ~3 menit  — akurasi rendah"),
            ("base",     "145 MB, ~8 menit  — akurasi sedang"),
            ("small",    "465 MB, ~15 menit — akurasi baik (RECOMMENDED)"),
            ("medium",   "1.5 GB, ~30 menit — akurasi tinggi"),
            ("large-v3", "3 GB,   ~50 menit — akurasi terbaik"),
        ],
        default_idx=2,  # small
    )

    res = _ask_choice(
        "Resolusi output",
        [("720", "720x1280   — ~4x lebih cepat"),
         ("1080", "1080x1920  — FHD")],
        default_idx=0,  # 720
    )
    args.output_resolution = int(res)

    mc = _ask("Batasi jumlah clip (0 = semua)", "0")
    try:
        args.max_clips = max(0, int(mc))
    except ValueError:
        args.max_clips = 0

    ip = _ask("Initial prompt untuk boost akurasi vocab (kosong = skip)", "")
    args.initial_prompt = ip or None

    console.print("\n[bold]Ringkasan:[/bold]")
    console.print(f"  Model        : [cyan]{args.model}[/cyan]")
    console.print(f"  Resolusi     : [cyan]{args.output_resolution}x{int(args.output_resolution * 16 / 9)}[/cyan]")
    console.print(f"  Max clips    : [cyan]{args.max_clips or 'semua'}[/cyan]")
    if args.initial_prompt:
        console.print(f"  Initial prompt: [cyan]{args.initial_prompt[:60]}{'...' if len(args.initial_prompt) > 60 else ''}[/cyan]")
    console.print(f"  Strategi     : {args.strategy} · target {args.target_duration}s · parallel {args.parallel}")

    confirm = _ask("\nMulai proses? (y/n)", "y").lower()
    if confirm not in ("y", "yes", "ya"):
        console.print("[red]Dibatalkan.[/red]")
        sys.exit(0)
    return args


def _only_url_given(argv: list[str]) -> bool:
    """True kalau user cuma kasih URL (tanpa flag konfigurasi)."""
    # argv[1] = URL positional. Flag apapun = tidak interactive.
    return len(argv) <= 2


def main() -> None:
    args = build_parser().parse_args()

    # Auto-aktifkan interactive kalau cuma URL yang di-pass, kecuali user pakai -y
    if not args.yes and _only_url_given(sys.argv):
        args = interactive_prompt(args)

    set_encoder_override(args.encoder)
    console.print(f"[dim]Akselerasi: {describe_accel()}[/dim]")

    # --- Step 1-2: download + WAV ---
    results = process_input(
        url=args.url,
        keyword=args.keyword or None,
        limit=args.limit,
        download_dir=Path(args.download_dir),
        audio_dir=Path(args.audio_dir),
    )
    if not results:
        console.print("[red]Tidak ada video diproses. Berhenti.[/red]")
        return

    if args.no_transcribe:
        console.print(f"\n[bold]Selesai (tanpa transkripsi): {len(results)} video.[/bold]")
        return

    # --- Step 3: transkripsi ---
    transcript_dir = Path(args.transcript_dir)
    clip_dir = Path(args.clip_dir)
    language = None if args.language.lower() == "auto" else args.language
    hl_keywords = [k.strip() for k in args.highlight_keywords.split(",") if k.strip()]

    console.print(f"\n[bold cyan]Transkripsi ({len(results)} audio)...[/bold cyan]")
    for i, r in enumerate(results, start=1):
        wav = Path(r["audio_path"])
        video = Path(r["video_path"])
        console.print(f"\n[{i}/{len(results)}] {wav.name}")

        transcript = transcribe(
            wav,
            language=language,
            model_size=args.model,
            initial_prompt=args.initial_prompt,
        )
        json_path = save_transcript(transcript, transcript_dir)
        srt_path = save_srt(transcript, transcript_dir)
        r["transcript_json"] = str(json_path)
        r["transcript_srt"] = str(srt_path)
        console.print(
            f"  [green]✓[/green] {len(transcript['segments'])} segmen · "
            f"{transcript['duration']}s · lang={transcript['language']}"
        )

        if args.no_clip:
            continue

        # --- Step 4: pilih clip sesuai strategi ---
        clips_meta: list[dict] = []

        if args.strategy in ("density", "both"):
            dens = group_by_density(
                transcript["segments"],
                target_duration=args.target_duration,
                silence_threshold=args.silence_threshold,
                min_duration=args.min_clip_duration,
            )
            console.print(f"  [cyan]density[/cyan] → {len(dens)} potongan")
            clips_meta.extend({**d, "kind": "density"} for d in dens)

        if args.strategy in ("highlight", "both"):
            hl = pick_highlights(
                transcript["segments"],
                keywords=hl_keywords,
                language=transcript["language"],
                min_score=args.min_score,
                min_duration=args.min_clip_duration,
                max_duration=args.max_clip_duration,
                merge_gap=args.merge_gap,
            )
            console.print(f"  [cyan]highlight[/cyan] → {len(hl)} potongan")
            clips_meta.extend({**h, "kind": "highlight"} for h in hl)

        if not clips_meta:
            console.print("  [yellow]⚠ Tidak ada potongan yang memenuhi kriteria.[/yellow]")
            r["clips"] = []
            continue

        if args.max_clips > 0 and len(clips_meta) > args.max_clips:
            console.print(
                f"  [yellow]Batasi ke {args.max_clips} clip pertama "
                f"(dari {len(clips_meta)} kandidat) sesuai --max-clips[/yellow]"
            )
            clips_meta = clips_meta[: args.max_clips]

        # 720 → 720x1280, 1080 → 1080x1920 (portrait 9:16)
        tw = args.output_resolution
        th = int(tw * 16 / 9)

        save_highlights(clips_meta, video.stem, transcript_dir)
        clip_paths = generate_clips(
            video, clips_meta, clip_dir,
            segments=transcript["segments"],
            viral=not args.no_viral,
            subtitle_color=args.subtitle_color,
            font_size=args.font_size,
            font=args.font,
            target_width=tw,
            target_height=th,
            parallel=args.parallel,
            subtitle_min_words=args.subtitle_min_words,
            subtitle_max_words=args.subtitle_max_words,
            subtitle_margin_bottom_pct=1.0 - args.subtitle_margin_top,
        )
        r["clips"] = [str(p) for p in clip_paths]

    # --- Ringkasan ---
    total_clips = sum(len(r.get("clips", [])) for r in results)
    console.print(
        f"\n[bold green]Selesai. {len(results)} video · {total_clips} clip dihasilkan.[/bold green]"
    )


if __name__ == "__main__":
    main()
