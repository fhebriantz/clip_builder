"""Transkripsi WAV → segments timestamp pakai Faster-Whisper (CPU int8 mode)."""
from __future__ import annotations

import json
from pathlib import Path

from faster_whisper import WhisperModel
from rich.console import Console

from accel import detect_whisper_device

console = Console()

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
        console.print(
            f"[cyan]Loading Whisper '{size}' "
            f"(device={device}, compute={compute_type}"
            + (f", threads={cpu_threads}" if device == "cpu" else "")
            + ")...[/cyan]"
        )
        _model_cache[key] = WhisperModel(
            size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads if device == "cpu" else 0,
            num_workers=1,
        )
    return _model_cache[key]


def transcribe(
    wav_path: Path,
    language: str | None = None,
    model_size: str = "base",
    vad_filter: bool = True,
) -> dict:
    """Return {audio, language, duration, segments: [{start, end, text}, ...]}.

    language: None (default) = auto-detect. Atau 'id', 'en', 'ja', dll untuk paksa.
    vad_filter: skip bagian sunyi → lebih cepat & akurat.
    """
    model = load_model(model_size)

    segments_iter, info = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=5,
        vad_filter=vad_filter,
    )

    segments = [
        {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
        for s in segments_iter
    ]

    return {
        "audio": str(wav_path),
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 2),
        "segments": segments,
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
