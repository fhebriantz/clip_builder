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

console = Console(legacy_windows=False)

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


_PUNCT_BREAK = set(".!?")
_PUNCT_SOFT = set(",;:")


def _find_split_indices(
    n: int,
    min_w: int,
    max_w: int,
    has_break_at: callable,
    has_soft_break_at: callable,
) -> list[tuple[int, int]]:
    """Greedy: cari split point di [min, max], prefer hard break > soft break > max.
    Return list of (start, end) exclusive."""
    chunks: list[tuple[int, int]] = []
    i = 0
    while i < n:
        upper = min(i + max_w, n)
        lower = min(i + min_w, n)

        # Kalau sisa kurang dari atau sama dengan max, semua masuk chunk terakhir
        if upper == n:
            chunks.append((i, n))
            break

        # Scan [lower-1 .. upper-1]: prioritas hard break (titik/tanya/seru)
        best_end = None
        for j in range(lower - 1, upper):
            if has_break_at(j):
                best_end = j + 1
                break
        # Kalau tidak ada hard break, coba soft break (koma)
        if best_end is None:
            for j in range(lower - 1, upper):
                if has_soft_break_at(j):
                    best_end = j + 1
                    break
        # Terakhir: tidak ada natural break. Kalau taking max bikin orphan,
        # redistribute rata. Kalau tidak, pakai max.
        if best_end is None:
            remaining_if_max = n - upper
            if 0 < remaining_if_max < min_w:
                total = n - i
                best_end = i + (total + 1) // 2  # round up
            else:
                best_end = upper

        chunks.append((i, best_end))
        i = best_end

    # Orphan merge: hanya merge kalau orphan sangat kecil (<= min-2),
    # misal min=4: merge orphan 1-2 kata. Orphan 3 tetap jadi chunk sendiri
    # (lebih baik dari 1 chunk kepanjangan).
    if len(chunks) >= 2:
        ls, le = chunks[-1]
        ps, pe = chunks[-2]
        orphan_size = le - ls
        if orphan_size <= max(1, min_w - 2) and (pe - ps) + orphan_size <= max_w + min_w // 2:
            chunks[-2] = (ps, le)
            chunks.pop()

    return chunks


def _rechunk_segment(
    seg: dict,
    min_words: int = 4,
    max_words: int = 6,
    pause_threshold: float = 0.4,
) -> list[dict]:
    """Pecah segment jadi chunks subtitle dengan range [min_words, max_words].

    Prioritas break:
      1. Hard break: titik/tanya/seru di akhir kata (.!?)
      2. Soft break: koma/titik-koma (,;:)
      3. Pause bicara (gap antar kata >= pause_threshold, hanya kalau ada word timestamp)
      4. Max_words (fallback)

    Orphan chunk terakhir (< min_words) di-merge ke chunk sebelumnya kalau muat.
    """
    text = seg["text"]
    start, end = seg["start"], seg["end"]

    words = seg.get("words")

    # Path 1: word-level timestamp (akurat, bisa deteksi pause)
    if words:
        n = len(words)
        if n <= max_words:
            return [seg]

        def has_hard_break(j: int) -> bool:
            t = words[j]["text"]
            if t and t[-1] in _PUNCT_BREAK:
                return True
            # Pause setelah kata juga dianggap hard break
            if j < n - 1:
                gap = words[j + 1]["start"] - words[j]["end"]
                if gap >= pause_threshold:
                    return True
            return False

        def has_soft_break(j: int) -> bool:
            t = words[j]["text"]
            return bool(t) and t[-1] in _PUNCT_SOFT

        splits = _find_split_indices(n, min_words, max_words, has_hard_break, has_soft_break)
        return [
            {
                "start": words[s]["start"],
                "end": words[e - 1]["end"],
                "text": " ".join(words[k]["text"] for k in range(s, e)).strip(),
            }
            for s, e in splits
        ]

    # Path 2: text saja — break di punctuation, timing proportional
    tokens = text.split()
    n = len(tokens)
    if n <= max_words:
        return [seg]

    def has_hard_break(j: int) -> bool:
        t = tokens[j]
        return bool(t) and t[-1] in _PUNCT_BREAK

    def has_soft_break(j: int) -> bool:
        t = tokens[j]
        return bool(t) and t[-1] in _PUNCT_SOFT

    splits = _find_split_indices(n, min_words, max_words, has_hard_break, has_soft_break)
    duration = max(end - start, 0.001)
    return [
        {
            "start": start + duration * (s / n),
            "end": start + duration * (e / n),
            "text": " ".join(tokens[s:e]),
        }
        for s, e in splits
    ]


def make_clip_subtitle(
    segments: list[dict],
    clip_start: float,
    clip_end: float,
    output_path: Path,
    min_words: int = 4,
    max_words: int = 6,
) -> Path:
    """Bikin SRT untuk 1 clip — subtitle dipecah jadi chunks [min_words, max_words]
    dengan prioritas break di titik/koma/pause natural."""
    lines: list[str] = []
    idx = 1
    for seg in segments:
        if seg["end"] <= clip_start or seg["start"] >= clip_end:
            continue
        for chunk in _rechunk_segment(seg, min_words=min_words, max_words=max_words):
            if chunk["end"] <= clip_start or chunk["start"] >= clip_end:
                continue
            rel_start = max(0.0, chunk["start"] - clip_start)
            rel_end = min(clip_end - clip_start, chunk["end"] - clip_start)
            if rel_end <= rel_start:
                continue
            lines += [
                str(idx),
                f"{_fmt_srt(rel_start)} --> {_fmt_srt(rel_end)}",
                chunk["text"],
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
    font_size: int = 14,
    target_width: int = 1080,
    target_height: int = 1920,
    subtitle_min_words: int = 4,
    subtitle_max_words: int = 6,
    subtitle_margin_bottom_pct: float = 0.25,
    smart_crop: bool = False,
    smart_crop_sample_interval: float = 1.0,
) -> Path:
    """Cut + vertical 9:16 center crop + burn-in subtitle (bold, kuning/putih, outline hitam).

    Output: mp4 portrait siap-upload ke TikTok/Reels/Shorts.
    font=None → pakai DEFAULT_FONT per platform.
    subtitle_margin_bottom_pct: jarak subtitle dari tepi bawah sebagai pct tinggi
        (0.3 = 30% dari bawah, posisi lower-third klasik gaya viral).
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
    make_clip_subtitle(
        segments, start, end, srt_tmp,
        min_words=subtitle_min_words, max_words=subtitle_max_words,
    )

    primary = _SUB_COLORS.get(color, _SUB_COLORS["yellow"])
    # Alignment=2 = bottom-center. MarginV diukur dalam script units libass
    # (PlayResY default = 288), BUKAN pixel video langsung. Jadi pct 0.3 →
    # MarginV ~86, yang setelah di-scale ke video 1280 tinggi jadi ~384 pixel
    # dari tepi bawah (≈ 30% dari bawah, sesuai yang diminta).
    margin_v = int(288 * subtitle_margin_bottom_pct)
    force_style = (
        f"FontName={font},"
        f"FontSize={font_size},"
        f"PrimaryColour={primary},"
        f"OutlineColour=&H000000,"
        f"BorderStyle=1,"
        f"Outline=3,"
        f"Shadow=0,"
        f"Bold=1,"
        f"Alignment=2,"
        f"MarginV={margin_v}"
    )

    enc = detect_ffmpeg_encoder()
    # Crop filter berbeda per aspect target
    target_ratio = target_width / target_height  # 9/16≈0.56, 1/1=1, 16/9≈1.78

    if target_ratio > 1.5:
        # Landscape 16:9 target → tidak perlu horizontal crop
        # (source biasanya 16:9, langsung scale. Kalau 4:3, hasilnya letterbox.)
        crop_filter = None
    else:
        # Portrait 9:16 atau Square 1:1 target → crop horizontal dari landscape source
        # crop_w sebagai fraksi dari iw (aspect_w/aspect_h × ih/iw, tapi lebih mudah: ih*ratio)
        if smart_crop:
            from face_tracker import compute_smart_crop_x
            console.print(f"  [dim]Smart crop: sampling wajah...[/dim]")
            # Parse target aspect dari width/height
            aw, ah = target_width, target_height
            x_expr = compute_smart_crop_x(
                video_path, start, end,
                target_aspect_w=aw, target_aspect_h=ah,
                sample_interval=smart_crop_sample_interval,
            )
            crop_filter = f"crop=w=ih*{aw}/{ah}:h=ih:x={x_expr}:y=0"
        else:
            crop_filter = f"crop=w=ih*{target_width}/{target_height}:h=ih:x=(iw-ih*{target_width}/{target_height})/2:y=0"

    parts = []
    if crop_filter:
        parts.append(crop_filter)
    parts.append(f"scale={target_width}:{target_height}")
    parts.append(f"subtitles={srt_tmp.name}:force_style='{force_style}'")

    vf = ",".join(parts) + enc["filter_suffix"]

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
    font_size: int = 14,
    font: str | None = None,
    target_width: int = 1080,
    target_height: int = 1920,
    parallel: int = 1,
    subtitle_min_words: int = 4,
    subtitle_max_words: int = 6,
    subtitle_margin_bottom_pct: float = 0.25,
    smart_crop: bool = False,
    run_timestamp: str | None = None,
) -> list[Path]:
    """Render semua potongan.

    viral=True → 9:16 + subtitle burn-in (segments wajib diisi).
    parallel>1 → render N clip bersamaan (cocok untuk hw encoder; libx264 sudah
    multi-thread per-proses jadi parallel tidak banyak membantu).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Anti-overwrite: kalau ada file clip dengan nama sama, append timestamp.
    # Decision-once di awal — semua clip dari run ini share timestamp yang sama
    # kalau ada conflict di salah satunya.
    _use_ts = False
    if run_timestamp:
        for _i in range(1, len(clips_meta) + 1):
            if (output_dir / f"{video_path.stem}_clip_{_i:02d}.mp4").exists():
                _use_ts = True
                break

    def _clip_output_path(idx: int) -> Path:
        if _use_ts and run_timestamp:
            return output_dir / f"{video_path.stem}_clip_{idx:02d}_{run_timestamp}.mp4"
        return output_dir / f"{video_path.stem}_clip_{idx:02d}.mp4"

    def _render_one(idx: int, h: dict) -> tuple[int, Path]:
        out = _clip_output_path(idx)
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
                subtitle_min_words=subtitle_min_words,
                subtitle_max_words=subtitle_max_words,
                subtitle_margin_bottom_pct=subtitle_margin_bottom_pct,
                smart_crop=smart_crop,
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
