#!/usr/bin/env bash
# ============================================================
#  AI Video Clipper - Linux Launcher
#  Jalankan: ./run_linux.sh
#  (atau dari file manager: klik kanan → Run, kalau executable)
# ============================================================

set -e

# Pindah ke direktori script (biar bisa dijalankan dari mana saja)
cd "$(dirname "$(readlink -f "$0")")"

echo ""
echo "============================================================"
echo "  AI Video Clipper - Linux Launcher"
echo "============================================================"
echo ""

# === [1/4] Cek Python ===
if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] Python 3 tidak ketemu."
    echo "Install: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

# === [2/4] Cek FFmpeg ===
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[ERROR] FFmpeg tidak ketemu."
    echo "Install: sudo apt install ffmpeg"
    exit 1
fi

# === [3/4] Setup venv + install dependencies (sekali saja) ===
if [ ! -d venv ]; then
    echo "------------------------------------------------------------"
    echo "[3.1/4] Membuat virtual environment pertama kali..."
    echo "        (tunggu ~10-30 detik)"
    echo "------------------------------------------------------------"
    python3 -m venv venv
    echo "    > venv created"
    echo ""
fi

source venv/bin/activate

if [ ! -f venv/.installed ]; then
    echo "------------------------------------------------------------"
    echo "[3.2/4] Install dependencies dari requirements.txt"
    echo "        (tunggu ~3-5 menit, pip akan tampilkan progress bar)"
    echo "------------------------------------------------------------"
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "    > requirements.txt installed"

    echo ""
    echo "------------------------------------------------------------"
    echo "[3.3/4] Deteksi GPU NVIDIA + install PyTorch"
    echo "------------------------------------------------------------"
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "    > GPU NVIDIA terdeteksi"
        echo "    > Install torch CUDA 11.8 + cuBLAS/cuDNN (~3-5 menit, ~3GB)..."
        pip uninstall -y torch torchaudio
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
    else
        echo "    > Tidak ada GPU NVIDIA"
        echo "    > Install torch CPU-only (~1-2 menit)..."
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    echo "    > torch installed"

    touch venv/.installed
    echo ""
    echo "============================================================"
    echo "[SETUP SELESAI] Semua dependencies terinstall!"
    echo "============================================================"
    echo ""
    echo "TIP: untuk fitur AI, buat file .env dengan isi:"
    echo "     GROQ_API_KEY=gsk_xxxxxxxxxxxxx"
    echo ""
    echo "     Daftar gratis di https://console.groq.com"
    echo "     Tanpa .env, fitur AI auto-skip, pipeline utama tetap jalan."
    echo ""
fi

# === [4/4] Jalankan Web UI ===
echo ""
echo "============================================================"
echo "[4/4] Menjalankan Web UI"
echo "============================================================"
echo "  URL    : http://127.0.0.1:7860"
echo "  Browser: akan terbuka otomatis dalam beberapa detik"
echo "  Stop   : Ctrl+C di jendela ini"
echo "============================================================"
echo ""
echo "(saat klik Generate di browser, loading spinner akan muncul)"
echo "(progress detail Whisper/FFmpeg/AI terlihat di jendela ini)"
echo ""

python app.py
