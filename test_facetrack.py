"""Quick face-tracking test — tanpa transkripsi & subtitle.

Cocok buat eksperimen parameter smart-crop di video baru tanpa nunggu Whisper.

Supported input:
  1. YouTube URL (regular OR Shorts)   : https://youtube.com/shorts/XXX
  2. Path lokal absolute/relative       : /home/user/video.mp4
  3. Nama file di folder local_videos/  : video.mp4 (auto-resolve)

Contoh:
  # Video lokal di folder local_videos/
  python test_facetrack.py my_video.mp4 --duration 30

  # YouTube Shorts
  python test_facetrack.py "https://youtube.com/shorts/XXX" --duration 60

  # Dari cache downloads/
  python test_facetrack.py downloads/XXX.mp4 --duration 30

  # Bandingkan center vs smart
  python test_facetrack.py video.mp4 --no-smart -o Output_Clips/center.mp4
  python test_facetrack.py video.mp4            -o Output_Clips/smart.mp4

  # Tuning parameter
  python test_facetrack.py video.mp4 --alpha 0.5 --dead-zone 0.02 --easing smoothstep
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from rich.console import Console

from accel import detect_ffmpeg_encoder, set_encoder_override
from downloader import download_video
from face_tracker import compute_smart_crop_x

console = Console()
LOCAL_DIR = Path("local_videos")


def _resolve_input(input_str: str) -> Path:
    """Resolve input: URL → download; else cari di CWD lalu local_videos/."""
    # URL YouTube (termasuk Shorts)
    if input_str.startswith(("http://", "https://")) or "youtube.com" in input_str or "youtu.be" in input_str:
        console.print(f"[cyan]Download: {input_str}[/cyan]")
        return download_video(input_str, Path("downloads"))

    # Local path: coba as-is, lalu di local_videos/
    candidates = [Path(input_str), LOCAL_DIR / input_str]
    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()

    tried = " | ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Video tidak ketemu. Dicari di: {tried}\n"
        f"Tip: taruh file di folder local_videos/ lalu panggil dengan nama filenya saja."
    )


def _get_aspect_ratio(video_path: Path) -> float:
    """Return width / height."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0:s=x", str(video_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = r.split("x")
    return int(w) / int(h)


def main() -> None:
    p = argparse.ArgumentParser(description="Quick face-tracking test (no transcribe/subtitle)")
    p.add_argument("input", help="YouTube URL atau path local .mp4")
    p.add_argument("--start", type=float, default=0.0, help="Start time (detik, default 0)")
    p.add_argument("--duration", type=float, default=30.0, help="Durasi test (detik, default 30)")
    p.add_argument("-o", "--output", default="Output_Clips/facetrack_test.mp4")
    p.add_argument("--target-width", type=int, default=720)
    p.add_argument("--target-height", type=int, default=1280)

    # Smart crop tuning
    p.add_argument("--trigger-threshold", type=float, default=0.06,
                   help="Delta minimal untuk trigger transisi (anti-jitter). Default 0.06 = 6%%")
    p.add_argument("--transition-duration", type=float, default=0.0,
                   help="Durasi transisi (detik). Default 0 = instant snap. "
                        "0.3 = sedikit smooth, 0.8 = smoother, 1.5 = cinematic")
    p.add_argument("--transition-curve", default="ease_out",
                   choices=["ease_out", "smoothstep", "linear"],
                   help="Kurva transisi: ease_out=cepat di awal (default), "
                        "smoothstep=slow-fast-slow, linear=konstan")
    p.add_argument("--sample-interval", type=float, default=0.5,
                   help="Sample wajah tiap N detik. Default 0.5")
    p.add_argument("--face-strategy", default="active_speaker",
                   choices=["active_speaker", "bbox_center", "biggest", "average"],
                   help="active_speaker=prioritas wajah frontal (default, ikuti yang bicara), "
                        "bbox_center=dua-duanya masuk frame, "
                        "biggest=wajah terbesar, average=rata-rata")
    p.add_argument("--no-smart", action="store_true",
                   help="Center crop (buat bandingkan)")

    p.add_argument("--encoder", default="auto")
    args = p.parse_args()

    video_path = _resolve_input(args.input)
    console.print(f"[dim]Source: {video_path}[/dim]")

    set_encoder_override(args.encoder)
    enc = detect_ffmpeg_encoder()

    end = args.start + args.duration

    # Auto-detect aspect — kalau sumber sudah portrait (misal Shorts), skip crop
    aspect = _get_aspect_ratio(video_path)
    target_aspect = args.target_width / args.target_height  # 9/16 default = 0.5625
    already_portrait = aspect <= target_aspect * 1.15  # toleransi 15%

    if already_portrait:
        console.print(
            f"[yellow]Input aspect {aspect:.2f} sudah portrait "
            f"(target {target_aspect:.2f}) — skip crop, langsung scale.[/yellow]"
        )
        crop_filter = None  # no crop, just scale
    elif args.no_smart:
        console.print("[yellow]Mode: center crop (no face tracking)[/yellow]")
        crop_filter = "crop=w=ih*9/16:h=ih:x=(iw-ih*9/16)/2:y=0"
    else:
        console.print(
            f"[cyan]Face tracking: strategy={args.face_strategy} · "
            f"sample tiap {args.sample_interval}s · "
            f"trigger={args.trigger_threshold} · "
            f"transition={args.transition_duration}s ({args.transition_curve})[/cyan]"
        )
        x_expr = compute_smart_crop_x(
            video_path, args.start, end,
            sample_interval=args.sample_interval,
            trigger_threshold=args.trigger_threshold,
            transition_duration=args.transition_duration,
            transition_curve=args.transition_curve,
            face_strategy=args.face_strategy,
        )
        crop_filter = f"crop=w=ih*9/16:h=ih:x={x_expr}:y=0"

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    scale = f"scale={args.target_width}:{args.target_height}"
    parts = [crop_filter, scale] if crop_filter else [scale]
    vf = ",".join(parts) + enc["filter_suffix"]

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *enc["input_args"],
        "-ss", f"{args.start:.3f}", "-i", str(video_path),
        "-t", f"{args.duration:.3f}",
        "-vf", vf,
        *enc["codec_args"],
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(output),
    ]

    console.print(f"[dim]Rendering to {output.name}...[/dim]")
    subprocess.run(cmd, check=True)
    size_mb = output.stat().st_size / 1024 / 1024
    console.print(f"[green]✓ Done: {output} ({size_mb:.1f} MB)[/green]")


if __name__ == "__main__":
    main()
