"""Transkripsi WAV → segments timestamp pakai Faster-Whisper (CPU int8 mode)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from faster_whisper import WhisperModel
from rich.console import Console

from accel import detect_whisper_device

console = Console(legacy_windows=False)

# Model pertama kali dipanggil akan auto-download ke ~/.cache/huggingface/hub
# Ukuran: tiny(~75MB), base(~145MB), small(~465MB), medium(~1.5GB), large-v3(~3GB)
_model_cache: dict[str, WhisperModel] = {}


def load_model(
    size: str = "base",
    device: str | None = None,
    compute_type: str | None = None,
    cpu_threads: int | None = None,
) -> WhisperModel:
    """Load model — auto-detect GPU/CPU kalau device=None."""
    if device is None or compute_type is None or cpu_threads is None:
        auto_device, auto_compute, auto_threads = detect_whisper_device()
        device = device or auto_device
        compute_type = compute_type or auto_compute
        cpu_threads = cpu_threads if cpu_threads is not None else auto_threads

    key = f"{size}-{device}-{compute_type}-{cpu_threads}"
    if key not in _model_cache:
        _FALLBACK: list[str] = (
            ["int8_float16", "int8", "float32"] if device == "cuda"
            else ["int8", "float32"]
        )
        fallbacks = [compute_type] + [f for f in _FALLBACK if f != compute_type]
        for ct in fallbacks:
            console.print(
                f"[cyan]Loading Whisper '{size}' "
                f"(device={device}, compute={ct}"
                + (f", threads={cpu_threads}" if device == "cpu" else "")
                + ")...[/cyan]"
            )
            try:
                _model_cache[key] = WhisperModel(
                    size,
                    device=device,
                    compute_type=ct,
                    cpu_threads=cpu_threads if device == "cpu" else 0,
                    num_workers=1,
                )
                break
            except ValueError as e:
                console.print(f"  [yellow]compute={ct} tidak didukung: {e}. Coba berikutnya...[/yellow]")
        else:
            raise RuntimeError(f"Tidak ada compute type yang didukung untuk device={device}")
    return _model_cache[key]


def _cache_meta_match(cached: dict, model_size: str, language: str | None,
                      initial_prompt: str | None) -> bool:
    """Cek apakah cache valid untuk parameter sekarang."""
    meta = cached.get("_cache_meta") or {}
    return (
        meta.get("model_size") == model_size
        and meta.get("language") == language
        and meta.get("initial_prompt") == initial_prompt
    )


def load_cached_transcript(
    wav_path: Path,
    cache_dir: Path,
    model_size: str,
    language: str | None,
    initial_prompt: str | None,
) -> dict | None:
    """Return transkrip cached kalau ada dan parameter match. None kalau perlu re-transcribe."""
    cache_file = cache_dir / f"{wav_path.stem}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        if not _cache_meta_match(data, model_size, language, initial_prompt):
            return None
        # Update audio path ke lokasi aktual (kalau pindah workdir)
        data["audio"] = str(wav_path)
        return data
    except (json.JSONDecodeError, OSError):
        return None


def transcribe(
    wav_path: Path,
    language: str | None = None,
    model_size: str = "base",
    vad_filter: bool = True,
    initial_prompt: str | None = None,
) -> dict:
    """Return {audio, language, duration, segments: [{start, end, text}, ...]}.

    language: None (default) = auto-detect. Atau 'id', 'en', 'ja', dll untuk paksa.
    vad_filter: skip bagian sunyi → lebih cepat & akurat.
    initial_prompt: kata kunci konteks untuk bias vocab (contoh: "mindset, koneksi, bisnis").
                    Membantu mengoreksi typo pada istilah spesifik tanpa re-train model.
    """
    model = load_model(model_size)

    _transcribe_kwargs = dict(
        language=language,
        beam_size=5,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
        word_timestamps=True,
    )
    try:
        segments_iter, info = model.transcribe(str(wav_path), **_transcribe_kwargs)
        # Materialize sekarang agar RuntimeError CUDA tertangkap di sini
        segments_raw = list(segments_iter)
    except RuntimeError as e:
        _is_cuda_err = any(k in str(e).lower() for k in ("cuda", "cublas", "dll", "cudnn"))
        if not _is_cuda_err:
            raise
        console.print(f"  [yellow]CUDA runtime error: {e}[/yellow]")
        console.print("  [yellow]Fallback ke CPU/int8...[/yellow]")
        cpu_threads = min(os.cpu_count() or 4, 8)
        model = load_model(model_size, device="cpu", compute_type="int8", cpu_threads=cpu_threads)
        segments_iter, info = model.transcribe(str(wav_path), **_transcribe_kwargs)
        segments_raw = list(segments_iter)

    segments = []
    for s in segments_raw:
        seg = {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
        if s.words:
            seg["words"] = [
                {"start": round(w.start, 3), "end": round(w.end, 3), "text": w.word.strip()}
                for w in s.words
            ]
        segments.append(seg)

    return {
        "audio": str(wav_path),
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 2),
        "segments": segments,
        # Metadata untuk cache validation di run berikutnya
        "_cache_meta": {
            "model_size": model_size,
            "language": language,
            "initial_prompt": initial_prompt,
        },
    }


def save_transcript(transcript: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(transcript["audio"]).stem
    out = output_dir / f"{stem}.json"
    out.write_text(json.dumps(transcript, ensure_ascii=False, indent=2))
    return out


def _fmt_srt_time(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    ms = int(round((s - int(s)) * 1000))
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


def transcript_to_srt(transcript: dict) -> str:
    lines: list[str] = []
    for i, seg in enumerate(transcript["segments"], start=1):
        lines += [
            str(i),
            f"{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}",
            seg["text"],
            "",
        ]
    return "\n".join(lines)


def save_srt(transcript: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(transcript["audio"]).stem
    out = output_dir / f"{stem}.srt"
    out.write_text(transcript_to_srt(transcript))
    return out
