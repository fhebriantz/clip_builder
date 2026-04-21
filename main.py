"""CLI entry point — AI Video Clipper (pipeline penuh).

URL → download → WAV → transkripsi → highlight detection → potong clip.
"""
import argparse
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
    p.add_argument("--model", default="base",
                   choices=["tiny", "base", "small", "medium", "large-v3"],
                   help="Ukuran model Whisper (default: base)")
    p.add_argument("--language", default="auto",
                   help="Bahasa audio: 'auto' (default, deteksi otomatis), 'id', 'en', dll.")

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
    p.add_argument("--font-size", type=int, default=18,
                   help="Ukuran font subtitle (default: 18)")
    p.add_argument("--font", default=None,
                   help="Nama font subtitle (default: auto per platform — "
                        "DejaVu Sans di Linux, Arial di Windows, Helvetica di macOS)")

    # Acceleration
    p.add_argument("--encoder", default="auto",
                   choices=["auto", "libx264", "h264_vaapi", "h264_nvenc", "h264_qsv"],
                   help="Override FFmpeg encoder (default: auto)")
    p.add_argument("--output-resolution", type=int, default=1080, choices=[720, 1080],
                   help="Tinggi target clip viral: 720 (HD, ~2x lebih cepat) atau 1080 (FHD)")
    p.add_argument("--parallel", type=int, default=1,
                   help="Render N clip bersamaan (disarankan 2-3 untuk hw encoder)")

    # Skip flags
    p.add_argument("--no-transcribe", action="store_true",
                   help="Skip transkripsi & highlight (hanya download + WAV)")
    p.add_argument("--no-clip", action="store_true",
                   help="Skip pemotongan clip (tetap transkripsi)")
    return p


def main() -> None:
    args = build_parser().parse_args()

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

        transcript = transcribe(wav, language=language, model_size=args.model)
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
        )
        r["clips"] = [str(p) for p in clip_paths]

    # --- Ringkasan ---
    total_clips = sum(len(r.get("clips", [])) for r in results)
    console.print(
        f"\n[bold green]Selesai. {len(results)} video · {total_clips} clip dihasilkan.[/bold green]"
    )


if __name__ == "__main__":
    main()
