"""Auto-detect akselerasi hardware untuk FFmpeg + Faster-Whisper.

Strategi:
- Whisper : CUDA/float16 > CPU/int8 (sesuai torch.cuda.is_available()).
- FFmpeg  : NVENC > VAAPI > QSV > libx264. Masing-masing di-smoke-test (encode
            1 frame lewat `-f null`) untuk konfirmasi benar-benar jalan, bukan
            cuma listed di `-encoders`.

Hasil deteksi di-cache agar tidak di-probe ulang per clip. Override CLI via
set_encoder_override() paksa pilih encoder tertentu.
"""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

from rich.console import Console

console = Console(legacy_windows=False)

VAAPI_DEVICE = "/dev/dri/renderD128"

# Candidate list — tiap entry = spesifikasi lengkap untuk ffmpeg.
# Note VAAPI: cuma support CQP rate control (bukan -b:v), jadi pakai -qp langsung.
GPU_ENCODER_CANDIDATES: list[dict] = [
    {
        "name": "h264_nvenc",
        "input_args": [],
        "filter_suffix": "",
        "codec_args": ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-cq", "23"],
        "requires": lambda: True,
    },
    {
        "name": "h264_vaapi",
        "input_args": ["-vaapi_device", VAAPI_DEVICE],
        "filter_suffix": ",format=nv12,hwupload",
        "codec_args": ["-c:v", "h264_vaapi", "-qp", "23"],
        "requires": lambda: Path(VAAPI_DEVICE).exists(),
    },
    {
        "name": "h264_qsv",
        "input_args": ["-init_hw_device", "qsv=hw", "-filter_hw_device", "hw"],
        "filter_suffix": ",hwupload=extra_hw_frames=64,format=qsv",
        "codec_args": ["-c:v", "h264_qsv", "-preset", "faster", "-global_quality", "23"],
        "requires": lambda: True,
    },
]

SOFTWARE_ENCODER: dict = {
    "name": "libx264",
    "input_args": [],
    "filter_suffix": "",
    "codec_args": ["-c:v", "libx264", "-preset", "fast", "-crf", "23"],
}

# Module-level override + cache (menggantikan lru_cache agar cache key tidak mismatch).
_encoder_override: str = "auto"
_encoder_cache: dict | None = None


def set_encoder_override(name: str) -> None:
    """Paksa encoder tertentu. 'auto' = biarkan auto-detect memilih."""
    global _encoder_override, _encoder_cache
    _encoder_override = name
    _encoder_cache = None


@lru_cache(maxsize=1)
def detect_whisper_device() -> tuple[str, str, int]:
    """Return (device, compute_type, cpu_threads)."""
    try:
        import torch
        if torch.cuda.is_available():
            # int8_float16 lebih kompatibel dari float16 murni (support Pascal+)
            return "cuda", "int8_float16", 0
    except Exception:
        pass
    threads = min(os.cpu_count() or 4, 8)
    return "cpu", "int8", threads


def _ffmpeg_encoders() -> str:
    try:
        return subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return ""


def _smoke_test(enc: dict) -> bool:
    """Encode 1 frame dummy ke /dev/null — verifikasi driver+lib benar-benar jalan."""
    vf = f"null{enc['filter_suffix']}" if enc["filter_suffix"] else "null"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        *enc["input_args"],
        "-f", "lavfi", "-i", "color=black:size=320x240:duration=0.1",
        "-vf", vf,
        *enc["codec_args"],
        "-f", "null", "-",
    ]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def detect_ffmpeg_encoder() -> dict:
    """Return encoder spec terbaik yang tersedia. Hormati override global."""
    global _encoder_cache
    if _encoder_cache is not None:
        return _encoder_cache

    listed = _ffmpeg_encoders()

    # Filter kandidat sesuai override
    if _encoder_override == "auto":
        candidates = list(GPU_ENCODER_CANDIDATES)
    elif _encoder_override == "libx264":
        candidates = []
    else:
        candidates = [c for c in GPU_ENCODER_CANDIDATES if c["name"] == _encoder_override]

    for enc in candidates:
        if enc["name"] not in listed:
            continue
        if not enc["requires"]():
            continue
        if _smoke_test(enc):
            _encoder_cache = {k: enc[k] for k in ("name", "input_args", "codec_args", "filter_suffix")}
            return _encoder_cache
        console.print(f"  [dim][info] {enc['name']} listed tapi gagal smoke-test, coba berikutnya...[/dim]")

    _encoder_cache = dict(SOFTWARE_ENCODER)
    return _encoder_cache


def describe() -> str:
    device, compute, threads = detect_whisper_device()
    enc = detect_ffmpeg_encoder()
    whisper_line = (
        f"Whisper: {device}/{compute}" + (f" ({threads} threads)" if device == "cpu" else "")
    )
    return f"{whisper_line} · FFmpeg: {enc['name']}"
