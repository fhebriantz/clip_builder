"""CLI entry point — AI Video Clipper (pipeline penuh).

URL → download → WAV → transkripsi → highlight detection → potong clip.

Mode interaktif aktif otomatis kalau user cuma kasih URL (tanpa flag lain).
Pakai flag `--yes` untuk skip prompt dan langsung jalan pakai default.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console


def _env_file_has_groq() -> bool:
    """Cek cepat apakah .env punya GROQ_API_KEY (tanpa load dotenv dulu)."""
    env = Path(".env")
    if not env.exists():
        return False
    try:
        return "GROQ_API_KEY=" in env.read_text() and "GROQ_API_KEY=\n" not in env.read_text()
    except Exception:
        return False

from accel import describe as describe_accel, set_encoder_override
from downloader import process_input
from highlighter import generate_clips, group_by_density, pick_highlights, save_highlights
from transcriber import load_cached_transcript, save_srt, save_transcript, transcribe

console = Console(legacy_windows=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AI Video Clipper — download, transkrip, dan auto-clip highlight",
    )
    p.add_argument("url", nargs="?", default=None,
                   help="Link video tunggal ATAU link channel YouTube (wajib kalau tanpa --batch)")
    p.add_argument("--batch", default=None,
                   help="Path ke file .txt berisi daftar URL (1 per baris). "
                        "Baris kosong & baris diawali '#' di-skip.")
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
                   choices=["density", "highlight", "both", "ai"],
                   help="density=bagi 60s berbasis kepadatan bicara (default); "
                        "highlight=pilih berdasarkan keyword/hook; both=dua-duanya; "
                        "ai=LLM pilih momen viral-worthy (butuh GROQ_API_KEY)")

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
    p.add_argument("--smart-crop", action="store_true",
                   help="Face tracking dynamic crop (anti-kaget via EMA + dead-zone)")
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
                   help="Kualitas target clip: 720 (HD, ~2x lebih cepat) atau 1080 (FHD)")
    p.add_argument("--aspect", default="9:16",
                   choices=["9:16", "1:1", "16:9"],
                   help="Aspect ratio output: 9:16 (TikTok/Reels/Shorts, default), "
                        "1:1 (Instagram feed), 16:9 (YouTube, desktop)")
    p.add_argument("--parallel", type=int, default=1,
                   help="Render N clip bersamaan (disarankan 2-3 untuk hw encoder)")
    p.add_argument("--max-clips", type=int, default=0,
                   help="Batasi jumlah clip yang di-render per video (0 = semua)")

    # UX
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip mode interaktif, langsung pakai default/flag")

    # AI features (Groq LLM)
    p.add_argument("--ai-metadata", action="store_true",
                   help="Generate viral title/description/hashtag via Groq LLM "
                        "(butuh GROQ_API_KEY di .env — gratis di console.groq.com)")
    p.add_argument("--ai-hook", action="store_true",
                   help="Detect 3-5 detik hook paling kuat di tiap clip, "
                        "simpan timestamp + text ke meta.json")
    p.add_argument("--ai-translate", default=None,
                   help="Translate subtitle ke bahasa target (contoh: 'en', 'ja', "
                        "'es', 'Chinese'). Burn-in subtitle terjemahan ke video.")
    p.add_argument("--render-hook", action="store_true",
                   help="Juga render hook clip 3-5 detik sebagai file teaser terpisah "
                        "(implies --ai-hook)")
    p.add_argument("--ai-polish", action="store_true",
                   help="Polish transkrip Whisper: hapus filler, fix typo, "
                        "konsistensi kapital (butuh GROQ_API_KEY)")
    p.add_argument("--polish-topic", default=None,
                   help="Deskripsi topik video untuk bantu polish fix typo ambigu "
                        "(contoh: 'mindset bisnis dan orang serakah')")
    p.add_argument("--polish-vocab", default="",
                   help="Kata-kata penting dipisah koma biar LLM pakai ejaan ini "
                        "(contoh: 'serakah,rakus,koneksi,milenial')")
    p.add_argument("--polish-fix", default="",
                   help="Manual corrections format 'salah=benar,salah2=benar2' "
                        "(contoh: 'straka=serakah,conesi=koneksi')")

    # Skip flags
    p.add_argument("--no-transcribe", action="store_true",
                   help="Skip transkripsi & highlight (hanya download + WAV)")
    p.add_argument("--no-clip", action="store_true",
                   help="Skip pemotongan clip (tetap transkripsi)")
    p.add_argument("--no-cache", action="store_true",
                   help="Paksa re-transcribe, abaikan cache di transcripts/")
    return p


def _ask(prompt: str, default: str) -> str:
    try:
        val = input(f"{prompt} [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[red]Dibatalkan.[/red]")
        sys.exit(0)
    return val or default


def _ask_yn(prompt: str, default: bool = False) -> bool:
    """Tanya y/n. Default jadi huruf besar di prompt."""
    default_str = "Y/n" if default else "y/N"
    ans = _ask(prompt + f" ({default_str})", "Y" if default else "N").lower()
    return ans in ("y", "yes", "ya", "1", "true")


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

    # --- Face tracking (non-AI, selalu available) ---
    args.smart_crop = _ask_yn(
        "Aktifkan smart crop (face tracking auto-follow speaker)?",
        default=True,
    )

    # --- AI features (hanya tampil kalau GROQ_API_KEY ada) ---
    has_groq = bool(os.environ.get("GROQ_API_KEY")) or _env_file_has_groq()
    if has_groq:
        console.print("\n[bold yellow]Fitur AI (Groq) terdeteksi:[/bold yellow]")

        use_ai_strategy = _ask_yn(
            "Pakai AI smart highlight (LLM pilih momen viral-worthy)?",
            default=True,
        )
        if use_ai_strategy:
            args.strategy = "ai"

        args.ai_polish = _ask_yn(
            "Polish subtitle (fix typo + hapus filler via LLM)?",
            default=False,
        )
        if args.ai_polish:
            topic = _ask("  Topik video (opsional, bantu fix typo ambigu)", "")
            args.polish_topic = topic or None

        args.ai_metadata = _ask_yn(
            "Generate title/description/hashtag viral?",
            default=True,
        )
    else:
        console.print(
            "\n[dim]Fitur AI di-skip — GROQ_API_KEY tidak ditemukan di .env. "
            "Daftar gratis di console.groq.com kalau mau pakai.[/dim]"
        )

    console.print("\n[bold]Ringkasan:[/bold]")
    console.print(f"  Model         : [cyan]{args.model}[/cyan]")
    console.print(f"  Resolusi      : [cyan]{args.output_resolution}x{int(args.output_resolution * 16 / 9)}[/cyan]")
    console.print(f"  Max clips     : [cyan]{args.max_clips or 'semua'}[/cyan]")
    console.print(f"  Smart crop    : [cyan]{'on' if args.smart_crop else 'off'}[/cyan]")
    console.print(f"  Strategy      : [cyan]{args.strategy}[/cyan]")
    console.print(f"  AI polish     : [cyan]{'on' if args.ai_polish else 'off'}[/cyan]")
    console.print(f"  AI metadata   : [cyan]{'on' if args.ai_metadata else 'off'}[/cyan]")
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


def _read_batch_urls(path: Path) -> list[str]:
    """Baca URL dari file .txt, skip baris kosong & comment (#)."""
    urls: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def main() -> None:
    args = build_parser().parse_args()

    # Validasi: butuh url ATAU batch, tidak boleh dua-duanya kosong
    if not args.url and not args.batch:
        console.print("[red]Error: butuh URL atau --batch path/to/urls.txt[/red]")
        sys.exit(1)
    if args.url and args.batch:
        console.print("[red]Error: pilih salah satu — URL positional ATAU --batch, bukan keduanya[/red]")
        sys.exit(1)

    # Auto-aktifkan interactive kalau cuma URL yang di-pass, kecuali user pakai -y / --batch
    if not args.yes and not args.batch and _only_url_given(sys.argv):
        args = interactive_prompt(args)

    set_encoder_override(args.encoder)
    console.print(f"[dim]Akselerasi: {describe_accel()}[/dim]")

    # --- Step 1-2: download + WAV (support batch: loop per URL) ---
    if args.batch:
        batch_path = Path(args.batch)
        if not batch_path.exists():
            console.print(f"[red]Batch file tidak ditemukan: {batch_path}[/red]")
            sys.exit(1)
        urls = _read_batch_urls(batch_path)
        if not urls:
            console.print(f"[red]File batch kosong: {batch_path}[/red]")
            sys.exit(1)

        console.print(f"\n[bold cyan]Batch mode: {len(urls)} URL[/bold cyan]")
        results: list[dict] = []
        batch_fail: list[tuple[str, str]] = []
        for bi, bu in enumerate(urls, 1):
            console.print(f"\n[bold yellow]═══ [{bi}/{len(urls)}] {bu}[/bold yellow]")
            try:
                partial = process_input(
                    url=bu,
                    keyword=args.keyword or None,
                    limit=args.limit,
                    download_dir=Path(args.download_dir),
                    audio_dir=Path(args.audio_dir),
                )
                results.extend(partial)
            except Exception as e:
                console.print(f"[red]✗ Gagal download URL {bi}: {e}[/red]")
                batch_fail.append((bu, str(e)))
    else:
        results = process_input(
            url=args.url,
            keyword=args.keyword or None,
            limit=args.limit,
            download_dir=Path(args.download_dir),
            audio_dir=Path(args.audio_dir),
        )
        batch_fail = []

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

        # Cek cache dulu — kalau parameter match, skip Whisper (hemat 15+ menit)
        transcript = None
        if not args.no_cache:
            cached = load_cached_transcript(
                wav, transcript_dir,
                model_size=args.model,
                language=language,
                initial_prompt=args.initial_prompt,
            )
            if cached is not None:
                transcript = cached
                console.print(
                    f"  [green]✓ cache hit[/green] {len(cached['segments'])} segmen "
                    f"(model={args.model}, skip Whisper)"
                )

        if transcript is None:
            transcript = transcribe(
                wav,
                language=language,
                model_size=args.model,
                initial_prompt=args.initial_prompt,
            )
            console.print(
                f"  [green]✓[/green] {len(transcript['segments'])} segmen · "
                f"{transcript['duration']}s · lang={transcript['language']}"
            )

        # --- AI Polish (opt-in): fix typo + hapus filler di transkrip ---
        # Polish = permanent upgrade, OK di-save ke cache (boleh di-reuse run berikutnya)
        if args.ai_polish:
            try:
                from ai_metadata import polish_subtitles
                vocab = [v.strip() for v in args.polish_vocab.split(",") if v.strip()]
                fix_pairs = {}
                for pair in args.polish_fix.split(","):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        fix_pairs[k.strip()] = v.strip()
                console.print(f"  [cyan]AI polish transkrip...[/cyan]")
                polished = polish_subtitles(
                    transcript["segments"],
                    topic_hint=args.polish_topic,
                    vocabulary=vocab or None,
                    corrections=fix_pairs or None,
                )
                transcript["segments"] = polished
                console.print(f"  [green]✓[/green] transkrip ter-polish ({len(polished)} segmen)")
            except Exception as e:
                console.print(f"  [yellow]⚠ AI polish gagal: {e}. Pakai transkrip raw.[/yellow]")

        # Save JSON + SRT dalam bahasa ASLI (polish OK, translate BELUM).
        # Ini jadi cache untuk run berikutnya — user yang tidak pakai --ai-translate
        # akan dapat transkrip asli, bukan translated.
        json_path = save_transcript(transcript, transcript_dir)
        srt_path = save_srt(transcript, transcript_dir)
        r["transcript_json"] = str(json_path)
        r["transcript_srt"] = str(srt_path)

        # --- AI Translate (opt-in): APPLY SETELAH save, volatile untuk run ini saja ---
        # Translate tidak ter-save ke JSON cache — next run tanpa --ai-translate
        # tetap dapat transkrip bahasa asli. SRT bahasa target disimpan terpisah.
        if args.ai_translate:
            try:
                from ai_metadata import translate_subtitles
                console.print(f"  [cyan]AI translate → {args.ai_translate}...[/cyan]")
                translated = translate_subtitles(
                    transcript["segments"],
                    target_language=args.ai_translate,
                )
                # In-memory copy untuk clip rendering downstream
                transcript = {
                    **transcript,
                    "segments": translated,
                    "language": args.ai_translate,
                }
                # Save translated SRT sebelah, suffix bahasa (tidak overwrite asli)
                translated_srt = transcript_dir / f"{wav.stem}.{args.ai_translate}.srt"
                from transcriber import transcript_to_srt
                translated_srt.write_text(transcript_to_srt(transcript))
                console.print(
                    f"  [green]✓[/green] {len(translated)} segmen translated · "
                    f"SRT: {translated_srt.name}"
                )
            except Exception as e:
                console.print(f"  [yellow]⚠ AI translate gagal: {e}. Pakai bahasa asli.[/yellow]")

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

        if args.strategy == "ai":
            try:
                from ai_metadata import generate_smart_highlights
                ai_max = args.max_clips if args.max_clips > 0 else 5
                ai_clips = generate_smart_highlights(
                    transcript["segments"],
                    max_clips=ai_max,
                    target_duration=args.target_duration,
                    min_duration=args.min_clip_duration,
                    max_duration=args.max_clip_duration,
                )
                console.print(f"  [cyan]ai[/cyan] → {len(ai_clips)} potongan viral-worthy")
                clips_meta.extend(ai_clips)
            except Exception as e:
                console.print(f"  [yellow]⚠ AI highlight gagal: {e}[/yellow]")
                console.print("  [dim]Fallback ke density strategy...[/dim]")
                dens = group_by_density(
                    transcript["segments"],
                    target_duration=args.target_duration,
                    silence_threshold=args.silence_threshold,
                    min_duration=args.min_clip_duration,
                )
                clips_meta.extend({**d, "kind": "density"} for d in dens)

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

        # Resolution × aspect lookup
        # 9:16 720→720x1280, 1080→1080x1920
        # 1:1  720→720x720, 1080→1080x1080
        # 16:9 720→1280x720, 1080→1920x1080
        _res_table = {
            ("9:16", 720):  (720, 1280),
            ("9:16", 1080): (1080, 1920),
            ("1:1", 720):   (720, 720),
            ("1:1", 1080):  (1080, 1080),
            ("16:9", 720):  (1280, 720),
            ("16:9", 1080): (1920, 1080),
        }
        tw, th = _res_table[(args.aspect, args.output_resolution)]

        # Timestamp per-run untuk anti-overwrite (clip file dengan nama sama
        # di Output_Clips/ akan ditambah suffix _YYYYmmdd_HHMMSS)
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

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
            smart_crop=args.smart_crop,
            run_timestamp=run_ts,
        )
        r["clips"] = [str(p) for p in clip_paths]

        # --- AI metadata + hook detection (opt-in) ---
        want_metadata = args.ai_metadata
        want_hook = args.ai_hook or args.render_hook
        if (want_metadata or want_hook) and clip_paths:
            try:
                from ai_metadata import (
                    generate_metadata, detect_hook_moment,
                    save_metadata, format_for_display,
                )
                import subprocess as _sp

                console.print(f"\n[bold cyan]Generate AI metadata ({len(clip_paths)} clip)...[/bold cyan]")
                for idx, (clip_path, meta_entry) in enumerate(zip(clip_paths, clips_meta), start=1):
                    clip_segs = [
                        s for s in transcript["segments"]
                        if s["end"] > meta_entry["start"] and s["start"] < meta_entry["end"]
                    ]
                    clip_text = " ".join(s["text"] for s in clip_segs)

                    combined: dict = {}
                    console.print(f"\n[{idx}/{len(clip_paths)}] {Path(clip_path).name}")

                    if want_metadata:
                        try:
                            combined.update(generate_metadata(clip_text))
                        except Exception as e:
                            console.print(f"  [yellow]⚠ metadata gagal: {e}[/yellow]")

                    if want_hook:
                        try:
                            hook = detect_hook_moment(clip_segs, clip_start_abs=meta_entry["start"])
                            if hook:
                                combined["hook"] = hook
                                if args.render_hook:
                                    # Cut hook dari rendered clip (portrait + subtitle + smart crop)
                                    # dengan re-encode untuk akurasi frame (stream copy cut di
                                    # keyframe, bisa meleset beberapa detik).
                                    hook_out = Path(clip_path).with_name(
                                        Path(clip_path).stem + "_hook.mp4"
                                    )
                                    _sp.run(
                                        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                                         "-ss", f"{hook['hook_start']:.3f}",
                                         "-i", str(Path(clip_path).resolve()),
                                         "-t", f"{hook['hook_duration']:.3f}",
                                         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                                         "-c:a", "aac",
                                         "-movflags", "+faststart",
                                         str(hook_out.resolve())],
                                        check=True,
                                    )
                                    console.print(f"  [green]✂[/green] hook teaser: {hook_out.name}")
                        except Exception as e:
                            console.print(f"  [yellow]⚠ hook detection gagal: {e}[/yellow]")

                    if combined:
                        out_meta = save_metadata(combined, Path(clip_path))
                        console.print(format_for_display(combined))
                        console.print(f"  [dim]saved: {out_meta.name}[/dim]")
            except RuntimeError as e:
                console.print(f"\n[red]AI features di-skip: {e}[/red]")

    # --- Ringkasan ---
    total_clips = sum(len(r.get("clips", [])) for r in results)
    console.print(
        f"\n[bold green]Selesai. {len(results)} video · {total_clips} clip dihasilkan.[/bold green]"
    )
    if batch_fail:
        console.print(f"\n[yellow]⚠ {len(batch_fail)} URL gagal di batch:[/yellow]")
        for url, err in batch_fail:
            console.print(f"  [dim]- {url}[/dim]")
            console.print(f"    [red]{err[:100]}[/red]")


if __name__ == "__main__":
    main()
