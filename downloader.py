"""Input handler: terima single video / channel URL, filter keyword, download, extract WAV."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable

import yt_dlp
from rich.console import Console

console = Console()

CHANNEL_URL_PATTERNS = [
    r"youtube\.com/@[\w\-\.]+",
    r"youtube\.com/channel/[\w\-]+",
    r"youtube\.com/c/[\w\-]+",
    r"youtube\.com/user/[\w\-]+",
]

SINGLE_VIDEO_MARKERS = ("watch?v=", "youtu.be/", "/shorts/")


def is_channel_url(url: str) -> bool:
    if any(m in url for m in SINGLE_VIDEO_MARKERS):
        return False
    return any(re.search(p, url) for p in CHANNEL_URL_PATTERNS)


def _normalize_channel_url(url: str) -> str:
    # /videos tab = uploads terurut dari terbaru
    url = url.rstrip("/")
    if url.endswith("/videos"):
        return url
    return f"{url}/videos"


def get_latest_videos(channel_url: str, limit: int = 3) -> list[dict]:
    opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
        "playlistend": limit,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(_normalize_channel_url(channel_url), download=False)

    entries = info.get("entries") or []
    videos: list[dict] = []
    for e in entries[:limit]:
        vid = e.get("id")
        if not vid:
            continue
        videos.append({
            "id": vid,
            "title": e.get("title", ""),
            "url": e.get("url") or f"https://www.youtube.com/watch?v={vid}",
        })
    return videos


def filter_by_keyword(videos: Iterable[dict], keyword: str | None) -> list[dict]:
    if not keyword:
        return list(videos)
    needle = keyword.lower()
    return [v for v in videos if needle in v["title"].lower()]


def download_video(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": False,
        "noprogress": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id = info["id"]
    for ext in ("mp4", "mkv", "webm"):
        candidate = output_dir / f"{video_id}.{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"File video untuk {video_id} tidak ditemukan di {output_dir}")


def extract_audio_wav(video_path: Path, audio_dir: Path, sample_rate: int = 16000) -> Path:
    # mono 16kHz PCM = format optimal untuk Faster-Whisper
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / f"{video_path.stem}.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(wav_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return wav_path


def fetch_single_video_meta(url: str) -> dict:
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return {"id": info["id"], "title": info.get("title", ""), "url": url}


def process_input(
    url: str,
    keyword: str | None = None,
    limit: int = 3,
    download_dir: Path = Path("downloads"),
    audio_dir: Path = Path("audio"),
) -> list[dict]:
    if is_channel_url(url):
        console.print(f"[cyan]Channel terdeteksi. Ambil {limit} video terbaru...[/cyan]")
        videos = get_latest_videos(url, limit=limit)
    else:
        console.print("[cyan]Single video terdeteksi.[/cyan]")
        videos = [fetch_single_video_meta(url)]

    if keyword:
        before = len(videos)
        videos = filter_by_keyword(videos, keyword)
        console.print(f"[yellow]Filter '{keyword}': {before} → {len(videos)} video lolos[/yellow]")

    if not videos:
        console.print("[red]Tidak ada video yang cocok dengan filter.[/red]")
        return []

    results: list[dict] = []
    for i, v in enumerate(videos, start=1):
        console.print(f"\n[bold][{i}/{len(videos)}] {v['title']}[/bold]")
        try:
            video_path = download_video(v["url"], download_dir)
            console.print(f"  [green]✓[/green] video: {video_path}")
            wav_path = extract_audio_wav(video_path, audio_dir)
            console.print(f"  [green]✓[/green] audio: {wav_path}")
            results.append({**v, "video_path": str(video_path), "audio_path": str(wav_path)})
        except Exception as exc:
            console.print(f"  [red]✗ gagal: {exc}[/red]")

    return results
