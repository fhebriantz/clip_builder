# AI Video Clipper

Aplikasi Python lokal untuk mengubah video panjang dari YouTube menjadi klip pendek viral siap-upload ke TikTok, Instagram Reels, dan YouTube Shorts — lengkap dengan transkripsi otomatis, face tracking, subtitle burn-in, dan AI metadata (title/description/hashtag).

Gratis + lokal (tidak ada API berbayar). Opsional: Groq API free tier untuk fitur AI advanced.

---

## Perintah Cepat yang Sering Dipakai

### 1. Mode Interaktif (paling direkomendasikan — tinggal tekan Enter)

```bash
source venv/bin/activate
python main.py "https://youtu.be/VIDEO_ID"
```

Akan tanya 1-per-1: model Whisper, resolusi, max clip, smart crop, AI features. Cukup tekan Enter untuk default.

### 2. Full Viral Workflow (terminal one-liner)

Workflow paling komplet — AI pilih clip terbaik, polish subtitle, smart crop, generate metadata:

```bash
python main.py "URL" -y \
  --strategy ai \
  --ai-polish \
  --ai-metadata \
  --smart-crop \
  --max-clips 3 \
  --aspect 9:16 \
  --subtitle-color yellow
```

Output 3 clip viral siap upload + file `.meta.json` per clip (berisi 3 pilihan title, description, hashtag).

### 3. Viral + Koreksi Typo Spesifik (paling akurat)

Saat Whisper salah dengar kata tertentu (misal "serakah" → "straka"), kasih topik + manual fix:

```bash
python main.py "URL" -y \
  --strategy ai --ai-polish --ai-metadata --smart-crop \
  --polish-topic "Video mindset bisnis membahas orang serakah dan sukses" \
  --polish-vocab "serakah,rakus,koneksi,milenial,mindset" \
  --polish-fix "straka=serakah,conesi=koneksi" \
  --max-clips 3
```

- `--polish-topic`: kasih LLM konteks topik video
- `--polish-vocab`: kata-kata penting yang mungkin muncul (ejaan benar)
- `--polish-fix`: manual override 100% reliable untuk typo yang Anda sudah tahu

### 4. Multi-Platform (render 3 aspect sekaligus)

```bash
# TikTok/Reels/Shorts
python main.py "URL" -y --strategy ai --ai-metadata --smart-crop --aspect 9:16 --max-clips 2

# Instagram feed
python main.py "URL" -y --strategy ai --ai-metadata --smart-crop --aspect 1:1 --max-clips 2

# YouTube clip / desktop
python main.py "URL" -y --strategy ai --ai-metadata --aspect 16:9 --max-clips 2
```

### 5. Quick Preview (1 clip saja, model cepat)

```bash
python main.py "URL" -y --model tiny --max-clips 1 --strategy ai
```

~3-5 menit untuk preview hasil sebelum commit ke pipeline full.

---

## Fitur Utama

### Core (tidak butuh API)
- **Download fleksibel** — single video atau N video terbaru dari channel (yt-dlp)
- **Support YouTube Shorts** — auto-detect URL Shorts
- **Filter keyword judul** — hanya download video yang judulnya match
- **Auto-detect bahasa** — Faster-Whisper (Indonesia, Inggris, dll)
- **Transkripsi akurat** — timestamp per-kalimat + word-level, export JSON + SRT
- **Density-based clipping** — potong 60 detik berbasis kepadatan bicara
- **Multi-aspect output** — 9:16 (TikTok/Reels), 1:1 (Instagram feed), 16:9 (YouTube/desktop)
- **Subtitle burn-in viral style** — bold, kuning/putih, outline hitam, lower-third
- **Smart subtitle chunking** — 4-6 kata per chunk, break di titik/koma/pause natural
- **Face tracking smart crop** — active speaker detection (frontal priority) + state machine smoothing (diam saat listening, snap saat scene cut)
- **Hardware acceleration** — auto NVENC > VAAPI > QSV > libx264
- **Parallel rendering** — render N clip bersamaan
- **Mode interaktif** — prompt y/n untuk setiap fitur, auto-detect Groq

### AI Features (butuh GROQ_API_KEY — gratis di console.groq.com)
- **AI Smart Highlight** — LLM baca transkrip penuh, pilih momen paling viral-worthy (skip intro/filler)
- **AI Subtitle Polish** — fix typo Whisper, hapus filler ("umm", "eh"), konsistensi kapital
- **Context-aware polish** — kasih topic hint + vocabulary + manual correction dict untuk akurasi maksimal
- **AI Metadata** — generate 3 opsi title viral + description + 8 hashtag per clip

### Graceful Fallback
Kalau `GROQ_API_KEY` tidak ada, semua fitur AI otomatis ter-skip. Pipeline utama jalan 100% dengan density strategy tanpa error.

---

## Arsitektur Pipeline

```
URL YouTube
    │
    ▼
[downloader]    yt-dlp + ffmpeg ──► video.mp4 + audio.wav (16kHz mono)
    │
    ▼
[transcriber]   Faster-Whisper ───► segments + word-level timestamps
    │                                (auto-detect bahasa)
    ▼
[ai_metadata]   Groq LLM ─────────► polish subtitle (opt-in)
    │                                fix typo, hapus filler
    ▼
[highlighter]   density | ai ───────► clips_meta
    │           LLM pilih viral        [{start, end, reason, viral_score}]
    │           moment (opt-in)
    ▼
[face_tracker]  OpenCV + state ────► dynamic crop x-expression
    │           machine smoothing     (opt-in via --smart-crop)
    ▼
[render]        FFmpeg ────────────► Output_Clips/*.mp4
                • cut + crop             (9:16 / 1:1 / 16:9)
                • smart crop (face)      + subtitle kuning/putih bold
                • burn-in subtitle         outline hitam lower-third
    │
    ▼
[ai_metadata]   Groq LLM ─────────► *.meta.json per clip
                (opt-in)             {titles, description, hashtags}
```

---

## Prasyarat

- **Python** 3.10 atau lebih baru
- **FFmpeg** 5.0+ (dengan filter `subtitles` dan salah satu encoder H.264)
- **Storage** ~2.5 GB (untuk Python dependencies + model Whisper small)

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

### Setup Groq API (Opsional — untuk fitur AI)

Fitur AI (smart highlight, polish, metadata) butuh Groq API key. **Gratis**, tidak perlu kartu kredit.

```bash
# 1. Daftar di https://console.groq.com (Google/GitHub login)
# 2. Buat API key di menu 'API Keys'
# 3. Simpan ke .env
echo 'GROQ_API_KEY=gsk_xxxxxxxxxxxxx' >> .env
```

Free tier: 14,400 request/hari (= ratusan video per hari). Tanpa key, fitur AI otomatis ter-skip, pipeline utama tetap jalan.

---

## Cara Pakai

### Quick Start (Mode Interaktif)

```bash
# Aktifkan venv (Linux/macOS)
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Cukup kasih URL, sisanya dipandu prompt
python main.py "https://youtu.be/VIDEO_ID"
```

Terminal akan menampilkan pilihan:
```
Mode Interaktif (tekan Enter untuk pakai default)

Model Whisper (akurasi vs kecepatan)
  1) tiny         75 MB,  ~3 menit  — akurasi rendah
  2) base         145 MB, ~8 menit  — akurasi sedang
  3) small        465 MB, ~15 menit — akurasi baik (RECOMMENDED) [default]
  4) medium       1.5 GB, ~30 menit — akurasi tinggi
  5) large-v3     3 GB,   ~50 menit — akurasi terbaik
Pilih [3]:

Resolusi output
  1) 720          720x1280   — ~4x lebih cepat [default]
  2) 1080         1080x1920  — FHD
Pilih [1]:

Batasi jumlah clip (0 = semua) [0]:
Initial prompt untuk boost akurasi vocab (kosong = skip) []:

Mulai proses? (y/n) [y]:
```

Output siap pakai di folder `Output_Clips/`.

### Skip Interaktif (Pakai Default)

```bash
# Langsung jalan dengan default semua (-y = yes, skip prompt)
python main.py "https://youtu.be/VIDEO_ID" -y
```

### Mode Advanced (Pakai Flag Spesifik)

Kalau kasih flag apapun selain URL, mode interaktif otomatis skip:
```bash
python main.py "https://youtu.be/VIDEO_ID" --model medium --output-resolution 1080 --parallel 2
```

### Contoh Lebih Lengkap

**Channel dengan filter keyword:**
```bash
python main.py "https://youtube.com/@NamaChannel" \
  --keyword "Bisnis" \
  --limit 3
```

**Preview 1 clip saja (untuk testing):**
```bash
python main.py "https://youtu.be/XXXX" \
  --model small \
  --max-clips 1 \
  --output-resolution 720
```

**Boost akurasi transkrip dengan initial prompt:**
```bash
python main.py "https://youtu.be/XXXX" \
  --model small \
  --initial-prompt "mindset, bisnis, koneksi, uang, sukses"
```

**Custom chunking subtitle (ultra-fast vs longgar):**
```bash
# High-energy TikTok style (3-4 kata per chunk)
python main.py "URL" --subtitle-min-words 3 --subtitle-max-words 4

# Lebih smooth untuk konten edukasi (6-9 kata)
python main.py "URL" --subtitle-min-words 6 --subtitle-max-words 9
```

**Mode cepat (parallel + 720p):**
```bash
python main.py "https://youtu.be/XXXX" \
  --model base \
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
| `--model` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--language` | `auto` | `auto` (deteksi otomatis), `id`, `en`, dll |
| `--initial-prompt` | — | Kata kunci konteks untuk boost akurasi vocab (contoh: `"mindset, bisnis, koneksi"`) |
| `--no-transcribe` | — | Skip transkripsi dan clipping |

### Strategi Clipping
| Flag | Default | Deskripsi |
|---|---|---|
| `--strategy` | `density` | `density`, `highlight`, `both`, **`ai`** (LLM pilih viral-worthy) |
| `--target-duration` | `60` | Target durasi clip (detik) |
| `--silence-threshold` | `2.0` | Gap bicara yang dianggap batas clip |
| `--min-clip-duration` | `20` | Buang clip lebih pendek dari ini |
| `--highlight-keywords` | `""` | Keyword highlight dipisah koma |
| `--min-score` | `1` | Skor minimal segmen untuk lolos (highlight) |
| `--max-clip-duration` | `60` | Cap durasi max (highlight) |
| `--merge-gap` | `2.0` | Gap antar segmen untuk digabung (highlight) |

### Viral Rendering
| Flag | Default | Deskripsi |
|---|---|---|
| `--aspect` | `9:16` | `9:16` (TikTok/Reels), `1:1` (Instagram feed), `16:9` (YouTube) |
| `--output-resolution` | `1080` | `720` (HD) atau `1080` (FHD) |
| `--no-viral` | — | Skip crop + subtitle, raw cut saja (instant) |
| `--subtitle-color` | `yellow` | `yellow` atau `white` |
| `--font` | auto | Nama font — default per platform |
| `--font-size` | `14` | Ukuran font subtitle |
| `--subtitle-min-words` | `4` | Minimal kata per chunk subtitle |
| `--subtitle-max-words` | `6` | Maksimal kata per chunk subtitle (gaya viral TikTok/Reels) |
| `--subtitle-margin-top` | `0.75` | Posisi subtitle sebagai fraksi dari atas (0.75 = 75% dari atas = lower-third) |

### AI Features (butuh GROQ_API_KEY)
| Flag | Default | Deskripsi |
|---|---|---|
| `--strategy ai` | off | LLM pilih clip viral-worthy dari transkrip penuh (ganti density/highlight) |
| `--ai-polish` | off | Polish subtitle: fix typo + hapus filler + konsistensi kapital |
| `--polish-topic` | — | Deskripsi topik video untuk disambiguasi typo (contoh: `"mindset bisnis serakah"`) |
| `--polish-vocab` | — | Kata-kata penting dipisah koma (contoh: `"serakah,koneksi,milenial"`) |
| `--polish-fix` | — | Manual correction `salah=benar,salah2=benar2` (100% reliable) |
| `--ai-metadata` | off | Generate title (3 opsi) + description + hashtag per clip |

### Face Tracking (opt-in)
| Flag | Default | Deskripsi |
|---|---|---|
| `--smart-crop` | off | Aktifkan face tracking crop (auto-follow speaker) |

Detail tuning tersedia di `test_facetrack.py` (script standalone untuk eksperimen cepat tanpa transkripsi). Parameter utama:
- `--face-strategy`: `active_speaker` (default, prioritas wajah frontal = yang bicara), `bbox_center` (frame dua orang), `biggest`, `average`
- `--trigger-threshold` (0.06): delta minimal untuk trigger transisi
- `--transition-duration` (0): 0=instant snap, 0.3-0.8=gradual, 1.5=cinematic
- `--transition-curve`: `ease_out`, `smoothstep`, `linear`

### Akselerasi
| Flag | Default | Deskripsi |
|---|---|---|
| `--encoder` | `auto` | `auto`, `libx264`, `h264_vaapi`, `h264_nvenc`, `h264_qsv` |
| `--parallel` | `1` | Render N clip bersamaan |
| `--max-clips` | `0` | Batasi jumlah clip per video (0 = semua) — berguna untuk preview |

### UX
| Flag | Default | Deskripsi |
|---|---|---|
| `-y`, `--yes` | — | Skip mode interaktif, pakai default/flag langsung |

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
| `cpu` / `int8` | CPU (8 threads) | Fallback, sudah cukup cepat dengan model `small` |

### FFmpeg Encoder

Priority: NVENC > VAAPI > QSV > libx264. Tiap kandidat di-smoke-test untuk memastikan benar-benar jalan (bukan sekadar listed).

| Encoder | Hardware | Platform |
|---|---|---|
| `h264_nvenc` | NVIDIA Pascal+ (GTX 900+) | Linux, Windows |
| `h264_vaapi` | Intel/AMD iGPU | Linux |
| `h264_qsv` | Intel Quick Sync Video | Linux, Windows |
| `libx264` | CPU (fallback) | Semua |

---

## Smart Subtitle Chunking

Subtitle tidak sekadar di-split per N kata (yang kaku), tapi pakai logika natural break + anti-orphan:

### Prioritas Break (saat cari posisi split)
1. **Hard break** — akhir kata ada `.` `!` `?` (akhir kalimat)
2. **Soft break** — akhir kata ada `,` `;` `:` (koma)
3. **Pause bicara** — gap antar kata ≥ 0.4 detik (hanya kalau word-timestamps aktif)
4. **Max words** — fallback kalau tidak ada break natural

### Anti-Orphan Logic
- Kalau mengambil max_words akan menyisakan orphan < min_words → **redistribusi rata** (contoh: 8 kata tidak jadi 6+2, tapi 4+4)
- Kalau setelah semua chunk masih ada orphan ≤ (min-2) → **merge ke chunk sebelumnya**

### Contoh Hasil (min=4, max=6)

| Input | Output | Alasan |
|---|---|---|
| `"Rahasia sukses itu adalah konsistensi, dalam hal kecil setiap hari"` | `[5, 5]` | Break di koma |
| `"Saya punya satu tips penting. Kita harus konsisten setiap hari ya"` | `[5, 6]` | Break di titik |
| `"Ini pelajaran sangat penting untuk kesuksesan dan mindset"` (8w, no punct) | `[4, 4]` | Redistribusi rata anti-orphan |
| `"Satu dua tiga empat lima enam tujuh delapan sembilan sepuluh sebelas duabelas tigabelas empatbelas"` (14w) | `[6, 4, 4]` | Redistribusi dari [6, 6, 2] |
| `"Halo teman-teman semua"` (3w, < min) | `[3]` | Tidak dipecah (sudah pendek) |

### Word-Level Timestamp
Saat transkripsi, `word_timestamps=True` otomatis aktif → setiap kata punya start/end sendiri, sehingga timing subtitle akurat per kata (bukan cuma proportional split). Menambah ~10-20% waktu transkripsi tapi worth it untuk kualitas viral.

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
| Whisper `small` transkripsi 10 menit | ~150s | ~12-18s |
| Render 4 clip viral 1080p | ~68s | ~15s |
| **Pipeline total** | baseline | **~5x lebih cepat** |

Kombinasi paling optimal (speed-quality balance):
```bash
python main.py "URL" --model small --output-resolution 720 --parallel 2
```

Kombinasi tercepat (akurasi rendah, untuk test/preview):
```bash
python main.py "URL" --model tiny --output-resolution 720 --parallel 2 --max-clips 1
```

---

## Project Structure

```
clip_builder/
├── accel.py              auto-detect hardware (GPU, encoder, thread)
├── downloader.py         URL → video mp4 + audio wav
├── transcriber.py        Faster-Whisper wrapper → JSON + SRT
├── highlighter.py        density/highlight detector + renderer (cut, crop, subtitle)
├── face_tracker.py       face detection + state machine smoothing untuk smart-crop
├── ai_metadata.py        Groq LLM: smart highlight + polish + title/desc/hashtag
├── test_facetrack.py     script standalone untuk eksperimen face tracking cepat
├── main.py               CLI entry point (orchestrator)
├── local_videos/         (opsional) folder video lokal untuk testing
├── .env                  (tidak ter-commit) GROQ_API_KEY disimpan di sini
├── requirements.txt
├── README.md
└── .gitignore
```

### Peran Setiap Modul

- **`accel.py`** — probe hardware sekali di awal, cache hasil. Module-level override untuk CLI `--encoder`.
- **`downloader.py`** — deteksi channel vs single, extract 3 terbaru, filter keyword, download + extract WAV 16kHz mono.
- **`transcriber.py`** — load Faster-Whisper model (cache in-memory), transcribe dengan VAD filter, export JSON/SRT.
- **`highlighter.py`** — dua strategi clipping: `group_by_density()` untuk pembagian 60s berbasis kepadatan, `pick_highlights()` untuk keyword-based. Plus `render_viral_clip()` dan `cut_clip()`.
- **`face_tracker.py`** — deteksi wajah multi-cascade (frontal + profile), strategi `active_speaker`, state machine smoothing (stable ↔ transitioning), FFmpeg crop x-expression builder.
- **`ai_metadata.py`** — Groq LLM integration: `generate_smart_highlights()` untuk pilih clip viral-worthy, `polish_subtitles()` untuk fix typo & filler, `generate_metadata()` untuk title/description/hashtag.
- **`test_facetrack.py`** — CLI terpisah untuk test cepat parameter face tracking tanpa jalan pipeline lengkap.
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
Model disimpan di `~/.cache/huggingface/hub`. Download pertama sekitar 465 MB untuk `small` (default), 145 MB untuk `base`, 1.5 GB untuk `medium`. Proxy setting via environment `HF_ENDPOINT` atau `HTTP_PROXY` kalau perlu.

---

## Tech Stack

- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — downloader YouTube/TikTok/Instagram
- **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)** — Whisper dengan backend CTranslate2 (4x lebih cepat dari OpenAI Whisper, support int8 quantization)
- **[FFmpeg](https://ffmpeg.org/)** — video/audio processing, hardware acceleration, subtitle burn-in via libass
- **[OpenCV](https://opencv.org/)** — face detection (Haar Cascade frontal + profile)
- **[Groq](https://groq.com/)** — LLM inference ultra-fast (Llama 3.3 70B) untuk AI features
- **[PyTorch](https://pytorch.org/)** — backend untuk Whisper (CPU/CUDA/MPS)
- **[Rich](https://github.com/Textualize/rich)** — terminal formatting

---

## Lisensi

Proyek portfolio pribadi. Dibuat untuk pembelajaran dan demonstrasi.

---

## Author

**Lutfi Febrianto** — Full Stack Developer (Node.js, React, Python, PHP)
