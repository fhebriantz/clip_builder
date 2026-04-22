@echo off
REM ============================================================
REM  AI Video Clipper - Windows Launcher
REM  Double-click file ini untuk setup (sekali) lalu run Web UI.
REM ============================================================

setlocal enabledelayedexpansion

REM Pindah ke direktori file .bat (biar bisa di-klik dari mana saja)
cd /d "%~dp0"

echo.
echo ============================================================
echo   AI Video Clipper - Windows Launcher
echo ============================================================
echo.

REM === [1/4] Cek Python ===
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python tidak ditemukan di PATH.
    echo.
    echo Install Python 3.10+ dari:
    echo    https://www.python.org/downloads/
    echo Saat install, CENTANG kotak "Add Python to PATH".
    echo Setelah install, tutup jendela ini dan klik ulang file .bat.
    echo.
    pause
    exit /b 1
)

REM === [2/4] Cek FFmpeg ===
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [ERROR] FFmpeg tidak ditemukan di PATH.
    echo.
    echo Install dengan perintah berikut di PowerShell/CMD sebagai Administrator:
    echo    winget install Gyan.FFmpeg
    echo.
    echo Setelah install, tutup semua terminal lalu klik ulang file .bat ini.
    echo.
    pause
    exit /b 1
)

REM === [3/4] Setup virtual environment + install dependencies (hanya pertama kali) ===
if not exist venv (
    echo ------------------------------------------------------------
    echo [3.1/4] Membuat virtual environment pertama kali...
    echo         ^(tunggu ~10-30 detik^)
    echo ------------------------------------------------------------
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Gagal membuat venv. Pastikan Python 3.10+ terinstall.
        pause
        exit /b 1
    )
    echo     ^> venv created
    echo.
)

call venv\Scripts\activate.bat

if not exist venv\.installed (
    echo ------------------------------------------------------------
    echo [3.2/4] Install dependencies dari requirements.txt
    echo         ^(tunggu ~3-5 menit, pip akan tampilkan progress bar^)
    echo ------------------------------------------------------------
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Gagal install dependencies.
        pause
        exit /b 1
    )
    echo     ^> requirements.txt installed

    echo.
    echo ------------------------------------------------------------
    echo [3.3/4] Deteksi GPU NVIDIA + install PyTorch
    echo ------------------------------------------------------------
    where nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo    ^> Tidak ada GPU NVIDIA terdeteksi.
        echo    ^> Install torch CPU-only ^(~1-2 menit^)...
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    ) else (
        echo    ^> GPU NVIDIA terdeteksi.
        echo    ^> Install torch CUDA 11.8 + cuBLAS/cuDNN ^(~3-5 menit, ~3GB download^)...
        pip uninstall -y torch torchaudio
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
    )
    echo     ^> torch installed

    REM Marker: tandai sudah ter-install biar run berikutnya skip step ini
    type nul > venv\.installed
    echo.
    echo ============================================================
    echo [SETUP SELESAI] Semua dependencies terinstall!
    echo ============================================================
    echo.
    echo TIP: untuk fitur AI ^(smart highlight, metadata, polish, translate^),
    echo      buat file .env di folder ini dengan isi:
    echo.
    echo          GROQ_API_KEY=gsk_xxxxxxxxxxxxx
    echo.
    echo      Daftar gratis di https://console.groq.com
    echo      Tanpa .env, fitur AI auto-skip, pipeline utama tetap jalan.
    echo.
    pause
)

REM === [4/4] Jalankan Web UI ===
echo.
echo ============================================================
echo [4/4] Menjalankan Web UI
echo ============================================================
echo   URL    : http://127.0.0.1:7860
echo   Browser: akan terbuka otomatis dalam beberapa detik
echo   Stop   : Ctrl+C di jendela ini
echo ============================================================
echo.
echo ^(saat klik Generate di browser, loading spinner akan muncul^)
echo ^(progress detail Whisper/FFmpeg/AI terlihat di jendela ini^)
echo.

python app.py

echo.
echo Server berhenti. Tekan tombol apa saja untuk menutup jendela.
pause
