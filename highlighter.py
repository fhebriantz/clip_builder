"""Auto-detect highlight dari transkrip + potong jadi clip pendek.

Strategi scoring (lokal, tanpa LLM API):
1. Skor tiap segmen = (hit keyword user) + (hit hook phrase bahasa target) + (tanda tanya)
2. Ambil segmen >= min_score
3. Merge yang berdekatan (gap <= merge_gap detik)
4. Expand ke min_duration dengan tambah konteks sekitar
5. Cap ke max_duration
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console

from accel import detect_ffmpeg_encoder

console = Console()

# Font default per platform — DejaVu Sans hanya ada di Linux, Arial universal di Windows,
# Helvetica default di macOS. Kalau user mau override pakai flag --font.
DEFAULT_FONT = {
    "linux": "DejaVu Sans",
    "win32": "Arial",
    "darwin": "Helvetica",
}.get(sys.platform, "DejaVu Sans")

# Hook phrases umum di konten creator bahasa Indonesia
INDONESIAN_HOOKS = [
    "tips", "rahasia", "cara", "trik", "penting", "jangan pernah",
    "harus", "wajib", "kesalahan", "sukses", "gagal", "pengalaman",
    "pelajaran", "strategi", "langkah", "kunci", "hindari",
    "perhatikan", "ingat", "fakta", "bukti", "jujur", "pertama kali",
]

ENGLISH_HOOKS = [
    "tip", "secret", "how to", "important", "never", "always",
    "must", "mistake", "success", "fail", "lesson", "strategy",
    "key", "avoid", "remember", "fact", "proof", "truth", "honestly",
]


def score_segment(text: str, keywords: list[str], hooks: list[str]) -> int:
    t = text.lower()
    score = 0
    for kw in keywords:
        score += len(re.findall(rf"\b{re.escape(kw.lower())}\b", t))
    for h in hooks:
        if h in t:
            score += 1
    score += t.count("?")
    return score


def pick_highlights(
    segments: list[dict],
    keywords: list[str] | None = None,
    language: str = "id",
    min_score: int = 1,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    merge_gap: float = 2.0,
) -> list[dict]:
    keywords = keywords or []
    hooks = INDONESIAN_HOOKS if language.startswith("id") else ENGLISH_HOOKS

    scored = [
        {**seg, "score": score_segment(seg["text"], keywords, hooks)}
        for seg in segments
    ]
    flagged_idx = [i for i, s in enumerate(scored) if s["score"] >= min_score]
    if not flagged_idx:
        return []

    # Group adjacent flagged segments by time gap
    groups: list[list[int]] = []
    current = [flagged_idx[0]]
    for idx in flagged_idx[1:]:
        prev_end = scored[current[-1]]["end"]
        this_start = scored[idx]["start"]
        if this_start - prev_end <= merge_gap:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)

    highlights: list[dict] = []
    for group in groups:
        first, last = group[0], group[-1]
        start = scored[first]["start"]
        end = scored[last]["end"]

        # Expand konteks simetris sampai min_duration.
        # Hentikan kalau gap ke tetangga > merge_gap (hindari lintas bagian sunyi).
        left, right = first, last
        while (end - start) < min_duration:
            expanded = False
            if left > 0:
                gap_left = scored[left]["start"] - scored[left - 1]["end"]
                if gap_left <= merge_gap:
                    left -= 1
                    start = scored[left]["start"]
                    expanded = True
            if (end - start) >= min_duration:
                break
            if right < len(scored) - 1:
                gap_right = scored[right + 1]["start"] - scored[right]["end"]
                if gap_right <= merge_gap:
                    right += 1
                    end = scored[right]["end"]
                    expanded = True
            if not expanded:
                break

        if (end - start) > max_duration:
            end = start + max_duration

        highlights.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "score": sum(scored[i]["score"] for i in group),
            "text": " ".join(scored[i]["text"] for i in group),
        })

    return highlights


def group_by_density(
    segments: list[dict],
    target_duration: float = 60.0,
    silence_threshold: float = 2.0,
    min_duration: float = 20.0,
) -> list[dict]:
    """Bagi segmen jadi clip ~target_duration detik berdasarkan kepadatan bicara.

    Strategi:
      - Gabung segmen yang berurutan selama gap antar segmen <= silence_threshold.
      - Tutup grup kalau (a) durasi >= target_duration, atau (b) gap berikutnya > silence_threshold.
      - Buang grup yang terlalu pendek (< min_duration) — biasanya residu ending/salam.

    Return: [{start, end, duration, segment_count, text}]
    """
    if not segments:
        return []

    groups: list[list[dict]] = []
    current: list[dict] = [segments[0]]

    for seg in segments[1:]:
        gap = seg["start"] - current[-1]["end"]
        current_duration = current[-1]["end"] - current[0]["start"]

        # Tutup grup kalau sunyi terlalu panjang ATAU sudah cukup durasi
        if gap > silence_threshold or current_duration >= target_duration:
            groups.append(current)
            current = [seg]
        else:
            current.append(seg)
    groups.append(current)

    clips: list[dict] = []
    for g in groups:
        start = g[0]["start"]
        end = g[-1]["end"]
        dur = end - start
        if dur < min_duration:
            continue
        clips.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(dur, 2),
            "segment_count": len(g),
            "text": " ".join(s["text"] for s in g),
        })
    return clips


def _fmt_srt(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    ms = int(round((s - int(s)) * 1000))
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


def make_clip_subtitle(
    segments: list[dict],
    clip_start: float,
    clip_end: float,
    output_path: Path,
) -> Path:
    """Bikin SRT untuk 1 clip — timestamp digeser jadi relatif dari 0."""
    lines: list[str] = []
    idx = 1
    for seg in segments:
        if seg["end"] <= clip_start or seg["start"] >= clip_end:
            continue
        rel_start = max(0.0, seg["start"] - clip_start)
        rel_end = min(clip_end - clip_start, seg["end"] - clip_start)
        if rel_end <= rel_start:
            continue
        lines += [
            str(idx),
            f"{_fmt_srt(rel_start)} --> {_fmt_srt(rel_end)}",
            seg["text"],
            "",
        ]
        idx += 1
    output_path.write_text("\n".join(lines))
    return output_path


# ASS color = &HBBGGRR (BGR, no alpha). Dipakai di force_style.
_SUB_COLORS = {
    "yellow": "&H00FFFF",  # B=00 G=FF R=FF
    "white":  "&HFFFFFF",
}


def cut_clip(video_path: Path, start: float, end: float, output_path: Path) -> Path:
    """Raw cut dengan stream copy — near-instant (tidak re-encode).

    Trade-off: cut geser ke keyframe terdekat (biasanya ±2s). Cocok untuk
    density/highlight yang tidak butuh frame-accurate start.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{end - start:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def render_viral_clip(
    video_path: Path,
    start: float,
    end: float,
    segments: list[dict],
    output_path: Path,
    color: str = "yellow",
    font: str | None = None,
    font_size: int = 18,
    target_width: int = 1080,
    target_height: int = 1920,
) -> Path:
    """Cut + vertical 9:16 center crop + burn-in subtitle (bold, kuning/putih, outline hitam).

    Output: mp4 portrait siap-upload ke TikTok/Reels/Shorts.
    font=None → pakai DEFAULT_FONT per platform.
    """
    if font is None:
        font = DEFAULT_FONT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start

    # Pakai absolute path untuk input/output — karena subprocess dijalankan dengan
    # cwd=srt_tmp.parent agar subtitles filter bisa pakai nama file relatif
    # (menghindari drive letter Windows `C:` tabrakan dengan filter separator).
    video_abs = str(Path(video_path).resolve())
    output_abs = str(Path(output_path).resolve())

    srt_tmp = output_path.with_suffix(".tmp.srt")
    make_clip_subtitle(segments, start, end, srt_tmp)

    primary = _SUB_COLORS.get(color, _SUB_COLORS["yellow"])
    # Alignment=5 = middle-center (di tengah layar). Outline=3 = border tebal hitam.
    force_style = (
        f"FontName={font},"
        f"FontSize={font_size},"
        f"PrimaryColour={primary},"
        f"OutlineColour=&H000000,"
        f"BorderStyle=1,"
        f"Outline=3,"
        f"Shadow=0,"
        f"Bold=1,"
        f"Alignment=5,"
        f"MarginV=0"
    )

    enc = detect_ffmpeg_encoder()
    # Filter chain: center-crop 9:16 → scale → burn subtitle (software) → hwupload (kalau perlu)
    vf = (
        f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,"
        f"scale={target_width}:{target_height},"
        f"subtitles={srt_tmp.name}:force_style='{force_style}'"
        f"{enc['filter_suffix']}"
    )

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *enc["input_args"],
        "-ss", f"{start:.3f}",
        "-i", video_abs,
        "-t", f"{duration:.3f}",
        "-vf", vf,
        *enc["codec_args"],
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_abs,
    ]

    try:
        # cwd=srt_tmp.parent → subtitles filter pakai nama file tanpa path
        subprocess.run(cmd, check=True, cwd=str(srt_tmp.parent))
    finally:
        srt_tmp.unlink(missing_ok=True)

    return output_path


def generate_clips(
    video_path: Path,
    clips_meta: list[dict],
    output_dir: Path,
    segments: list[dict] | None = None,
    viral: bool = True,
    subtitle_color: str = "yellow",
    font_size: int = 18,
    font: str | None = None,
    target_width: int = 1080,
    target_height: int = 1920,
    parallel: int = 1,
) -> list[Path]:
    """Render semua potongan.

    viral=True → 9:16 + subtitle burn-in (segments wajib diisi).
    parallel>1 → render N clip bersamaan (cocok untuk hw encoder; libx264 sudah
    multi-thread per-proses jadi parallel tidak banyak membantu).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _render_one(idx: int, h: dict) -> tuple[int, Path]:
        out = output_dir / f"{video_path.stem}_clip_{idx:02d}.mp4"
        tag = h.get("score", h.get("segment_count", "?"))
        label = "score" if "score" in h else "segs"
        console.print(
            f"  [cyan]✂[/cyan] Clip {idx}: {h['start']:.1f}s → {h['end']:.1f}s "
            f"({h['duration']:.1f}s, {label}={tag})"
        )
        if viral:
            if segments is None:
                raise ValueError("segments wajib diisi saat viral=True")
            render_viral_clip(
                video_path, h["start"], h["end"], segments, out,
                color=subtitle_color, font=font, font_size=font_size,
                target_width=target_width, target_height=target_height,
            )
        else:
            cut_clip(video_path, h["start"], h["end"], out)
        return idx, out

    tasks = list(enumerate(clips_meta, start=1))

    if parallel <= 1:
        return [_render_one(i, h)[1] for i, h in tasks]

    results: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = [ex.submit(_render_one, i, h) for i, h in tasks]
        for fut in as_completed(futures):
            i, path = fut.result()
            results[i] = path
    return [results[i] for i in sorted(results)]


def save_highlights(highlights: list[dict], video_stem: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{video_stem}_highlights.json"
    out.write_text(json.dumps(highlights, ensure_ascii=False, indent=2))
    return out
