# AI Video Clipper

Aplikasi Python lokal untuk mengubah video panjang dari YouTube menjadi klip pendek siap-upload ke TikTok, Instagram Reels, dan YouTube Shorts — lengkap dengan transkripsi otomatis dan subtitle burn-in ala konten viral.

Gratis, lokal, tanpa API berbayar. Auto-detect hardware (CPU/NVIDIA CUDA/Intel-AMD VAAPI) untuk performa maksimal di mesin manapun.

---

## Fitur Utama

- **Download fleksibel** — single video atau 3 video terbaru dari channel (yt-dlp)
- **Filter keyword judul** — hanya download video yang judulnya match
- **Auto-detect bahasa** — Faster-Whisper otomatis deteksi bahasa (Indonesia, Inggris, dll)
- **Transkripsi akurat** — timestamp per-kalimat, export JSON + SRT
- **Density-based clipping** — potong jadi clip ~60 detik berdasarkan kepadatan bicara (bukan asal potong)
- **Highlight detection** — opsi alternatif berdasarkan keyword/hook phrase viral
- **9:16 center crop** — otomatis portrait 720p atau 1080p
- **Subtitle burn-in viral style** — bold, kuning/putih, border hitam tebal, di tengah layar
- **Hardware acceleration** — auto-pilih NVENC > VAAPI > QSV > libx264
- **Parallel rendering** — render N clip bersamaan di hardware encoder
- **Cross-platform** — satu perintah yang sama di Linux, Windows, macOS

---

## Arsitektur Pipeline

```
URL YouTube
    │
    ▼
[downloader]    yt-dlp + ffmpeg ──► video.mp4 + audio.wav (16kHz mono)
    │
    ▼
[transcriber]   Faster-Whisper ───► segments[{start, end, text}] + bahasa
    │                                (auto-detect bahasa)
    ▼
[highlighter]   density | highlight ► clips_meta[{start, end, duration}]
    │                                 (berbasis kepadatan bicara)
    ▼
[render]        FFmpeg ────────────► Output_Clips/*.mp4
                • cut akurat           (1080x1920 atau 720x1280
                • 9:16 center crop      + subtitle kuning/putih
                • burn-in subtitle        bold outline hitam, center)
```

---

## Prasyarat

- **Python** 3.10 atau lebih baru
- **FFmpeg** 5.0+ (dengan filter `subtitles` dan salah satu encoder H.264)
- **Storage** ~2 GB (untuk Python dependencies + model Whisper base)

---

## Instalasi

### 1. Linux (Ubuntu/Debian)

```bash
# FFmpeg biasanya sudah ada; kalau belum:
sudo apt install ffmpeg python3-venv

# Setup project
git clone <repo-url> clip_builder
cd clip_builder
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Torch CPU (Intel/AMD tanpa GPU NVIDIA):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Windows (dengan GPU NVIDIA)

```powershell
# FFmpeg
winget install Gyan.FFmpeg

# Setup project
git clone <repo-url> clip_builder
cd clip_builder
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Torch CUDA 11.8 (kompatibel Pascal/Maxwell/Turing/Ampere):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# NVIDIA libs untuk Faster-Whisper (cuBLAS + cuDNN):
pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
```

### 3. macOS (Apple Silicon atau Intel)

```bash
# FFmpeg
brew install ffmpeg

# Setup project
git clone <repo-url> clip_builder
cd clip_builder
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Torch CPU (Apple Silicon pakai MPS otomatis via PyTorch):
pip install torch torchaudio
```

### Pilihan Torch Sesuai Hardware

| Hardware | Perintah |
|---|---|
| CPU only (Intel/AMD tanpa GPU) | `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu` |
| NVIDIA GPU (Pascal sampai Ada) | `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| NVIDIA GPU terbaru (Hopper+) | `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| Apple Silicon (M1/M2/M3) | `pip install torch torchaudio` (default pakai MPS) |

---

## Cara Pakai

### Quick Start

```bash
# Aktifkan venv (Linux/macOS)
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Download + transkripsi + auto-clip — pipeline lengkap
python main.py "https://youtu.be/VIDEO_ID"
```

Output siap pakai di folder `Output_Clips/`.

### Contoh Lebih Lengkap

**Channel dengan filter keyword:**
```bash
python main.py "https://youtube.com/@NamaChannel" \
  --keyword "Bisnis" \
  --limit 3
```

**Mode cepat (720p + parallel):**
```bash
python main.py "https://youtu.be/XXXX" \
  --model tiny \
  --output-resolution 720 \
  --parallel 2
```

**Custom durasi clip dan warna subtitle:**
```bash
python main.py "https://youtu.be/XXXX" \
  --target-duration 45 \
  --subtitle-color white \
  --font-size 22
```

**Strategi gabungan (density + highlight keyword):**
```bash
python main.py "https://youtu.be/XXXX" \
  --strategy both \
  --highlight-keywords "tips,rahasia,strategi"
```

**Raw cut tanpa subtitle (instant, stream copy):**
```bash
python main.py "https://youtu.be/XXXX" --no-viral
```

---

## CLI Options Lengkap

### Input
| Flag | Default | Deskripsi |
|---|---|---|
| `url` | — | Link video YouTube tunggal atau channel |
| `-k`, `--keyword` | `""` | Filter judul video (kosong = ambil semua) |
| `-n`, `--limit` | `3` | Jumlah video terbaru dari channel |

### Transkripsi
| Flag | Default | Deskripsi |
|---|---|---|
| `--model` | `base` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--language` | `auto` | `auto` (deteksi otomatis), `id`, `en`, dll |
| `--no-transcribe` | — | Skip transkripsi dan clipping |

### Strategi Clipping
| Flag | Default | Deskripsi |
|---|---|---|
| `--strategy` | `density` | `density`, `highlight`, `both` |
| `--target-duration` | `60` | Target durasi clip density (detik) |
| `--silence-threshold` | `2.0` | Gap bicara yang dianggap batas clip |
| `--min-clip-duration` | `20` | Buang clip lebih pendek dari ini |
| `--highlight-keywords` | `""` | Keyword highlight dipisah koma |
| `--min-score` | `1` | Skor minimal segmen untuk lolos (highlight) |
| `--max-clip-duration` | `60` | Cap durasi max (highlight) |
| `--merge-gap` | `2.0` | Gap antar segmen untuk digabung (highlight) |

### Viral Rendering
| Flag | Default | Deskripsi |
|---|---|---|
| `--no-viral` | — | Skip 9:16 + subtitle, raw cut saja (instant) |
| `--subtitle-color` | `yellow` | `yellow` atau `white` |
| `--font` | auto | Nama font — default per platform |
| `--font-size` | `18` | Ukuran font subtitle |
| `--output-resolution` | `1080` | `720` atau `1080` (tinggi 9:16) |

### Akselerasi
| Flag | Default | Deskripsi |
|---|---|---|
| `--encoder` | `auto` | `auto`, `libx264`, `h264_vaapi`, `h264_nvenc`, `h264_qsv` |
| `--parallel` | `1` | Render N clip bersamaan |

### Output
| Flag | Default | Deskripsi |
|---|---|---|
| `--download-dir` | `downloads` | Folder video mentah |
| `--audio-dir` | `audio` | Folder WAV |
| `--transcript-dir` | `transcripts` | Folder JSON + SRT + highlights |
| `--clip-dir` | `Output_Clips` | Folder clip final |

---

## Output Structure

```
clip_builder/
├── downloads/
│   └── <video_id>.mp4              video mentah dari YouTube
├── audio/
│   └── <video_id>.wav              audio 16kHz mono untuk Whisper
├── transcripts/
│   ├── <video_id>.json             transkrip lengkap
│   ├── <video_id>.srt              subtitle full video
│   └── <video_id>_highlights.json  metadata clip
└── Output_Clips/
    ├── <video_id>_clip_01.mp4      1080x1920 atau 720x1280
    ├── <video_id>_clip_02.mp4      subtitle ter-burn di tengah
    └── <video_id>_clip_NN.mp4
```

---

## Hardware Acceleration

Auto-detect saat program dijalankan — tidak perlu konfigurasi. Cek status dengan:

```bash
python main.py "URL" 2>&1 | head -1
# Output: Akselerasi: Whisper: cuda/float16 · FFmpeg: h264_nvenc
```

### Whisper

| Device | Compute | Dipakai kapan |
|---|---|---|
| `cuda` / `float16` | NVIDIA GPU + CUDA 11.8+ + cuDNN | Otomatis kalau tersedia |
| `cpu` / `int8` | CPU (8 threads) | Fallback, sudah cukup cepat dengan model `base` |

### FFmpeg Encoder

Priority: NVENC > VAAPI > QSV > libx264. Tiap kandidat di-smoke-test untuk memastikan benar-benar jalan (bukan sekadar listed).

| Encoder | Hardware | Platform |
|---|---|---|
| `h264_nvenc` | NVIDIA Pascal+ (GTX 900+) | Linux, Windows |
| `h264_vaapi` | Intel/AMD iGPU | Linux |
| `h264_qsv` | Intel Quick Sync Video | Linux, Windows |
| `libx264` | CPU (fallback) | Semua |

---

## Benchmark

Uji pada sumber 120 detik 1080p30, rendering 4 clip viral masing-masing 20 detik.

### CPU + VAAPI (Intel iGPU — Ubuntu 24.04)

| Konfigurasi | Waktu | Speedup |
|---|---|---|
| 1080p sequential (baseline) | 68.39s | 1.00x |
| 720p sequential | 16.92s | 4.04x |
| 1080p `--parallel 2` | 11.90s | 5.74x |
| **720p + `--parallel 2`** | **6.96s** | **9.82x** |
| Raw stream copy (`--no-viral`) | 0.22s | 316x |

### Ekspektasi di NVIDIA GTX 1080 Ti (Windows)

| Tahap | CPU-only Linux | 1080 Ti Windows |
|---|---|---|
| Whisper `base` transkripsi 10 menit | ~60s | ~6-8s |
| Render 4 clip viral 1080p | ~68s | ~15s |
| **Pipeline total** | baseline | **~5x lebih cepat** |

Kombinasi paling optimal:
```bash
python main.py "URL" --model tiny --output-resolution 720 --parallel 2
```

---

## Project Structure

```
clip_builder/
├── accel.py              auto-detect hardware (GPU, encoder, thread)
├── downloader.py         URL → video mp4 + audio wav
├── transcriber.py        Faster-Whisper wrapper → JSON + SRT
├── highlighter.py        density/highlight detector + renderer (cut, crop, subtitle)
├── main.py               CLI entry point (orchestrator)
├── requirements.txt
├── README.md
└── .gitignore
```

### Peran Setiap Modul

- **`accel.py`** — probe hardware sekali di awal, cache hasil. Module-level override untuk CLI `--encoder`.
- **`downloader.py`** — deteksi channel vs single, extract 3 terbaru, filter keyword, download + extract WAV 16kHz mono.
- **`transcriber.py`** — load Faster-Whisper model (cache in-memory), transcribe dengan VAD filter, export JSON/SRT.
- **`highlighter.py`** — dua strategi clipping: `group_by_density()` untuk pembagian 60s berbasis kepadatan, `pick_highlights()` untuk keyword-based. Plus `render_viral_clip()` dan `cut_clip()`.
- **`main.py`** — argparse, orchestrate seluruh pipeline, tampilkan progress.

---

## Troubleshooting

### FFmpeg tidak ketemu
```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```
Install FFmpeg: `sudo apt install ffmpeg` (Linux), `winget install Gyan.FFmpeg` (Windows), `brew install ffmpeg` (macOS). Pastikan ada di PATH.

### CUDA tidak terdeteksi di Windows
Output `Akselerasi: Whisper: cpu/int8` padahal punya GPU NVIDIA? Cek:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Kalau False: reinstall torch dengan CUDA wheel:
```powershell
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Faster-Whisper error "libcudnn not found"
Install NVIDIA libs bundle:
```bash
pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
```

### Subtitle tidak muncul di output
- Cek font tersedia: default auto per platform (DejaVu Sans/Arial/Helvetica)
- Override dengan flag `--font "Nama Font"`
- Cek FFmpeg compile dengan `libass`: `ffmpeg -filters | grep subtitles`

### NVENC listed tapi fallback ke libx264
Driver NVIDIA outdated atau GPU tidak support NVENC (hanya seri tertentu). Smoke-test saya bakal print info, cek output:
```
[info] h264_nvenc listed tapi gagal smoke-test, coba berikutnya...
```
Update driver NVIDIA atau terima fallback ke libx264.

### VAAPI error "Failed to initialize"
Device `/dev/dri/renderD128` belum di-grant ke user:
```bash
sudo usermod -aG video,render $USER
# Logout + login ulang
```

### Whisper model download lambat
Model disimpan di `~/.cache/huggingface/hub`. Download pertama sekitar 145 MB untuk `base`. Proxy setting via environment `HF_ENDPOINT` atau `HTTP_PROXY` kalau perlu.

---

## Tech Stack

- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — downloader YouTube/TikTok/Instagram
- **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)** — Whisper dengan backend CTranslate2 (4x lebih cepat dari OpenAI Whisper, support int8 quantization)
- **[FFmpeg](https://ffmpeg.org/)** — video/audio processing, hardware acceleration, subtitle burn-in via libass
- **[PyTorch](https://pytorch.org/)** — backend untuk Whisper (CPU/CUDA/MPS)
- **[Rich](https://github.com/Textualize/rich)** — terminal formatting

---

## Lisensi

Proyek portfolio pribadi. Dibuat untuk pembelajaran dan demonstrasi.

---

## Author

**Lutfi Febrianto** — Full Stack Developer (Node.js, React, Python, PHP)
