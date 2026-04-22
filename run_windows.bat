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
    echo [SETUP] Membuat virtual environment pertama kali...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Gagal membuat venv. Pastikan Python 3.10+ terinstall.
        pause
        exit /b 1
    )
    echo.
)

call venv\Scripts\activate.bat

if not exist venv\.installed (
    echo [SETUP] Install dependencies dari requirements.txt...
    echo.
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Gagal install dependencies.
        pause
        exit /b 1
    )

    echo.
    echo [SETUP] Deteksi GPU NVIDIA...
    where nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo    Tidak ada GPU NVIDIA terdeteksi.
        echo    Install torch CPU-only...
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    ) else (
        echo    GPU NVIDIA terdeteksi.
        echo    Install torch CUDA 11.8 + cuBLAS/cuDNN...
        pip uninstall -y torch torchaudio
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install nvidia-cublas-cu11 nvidia-cudnn-cu11
    )

    REM Marker: tandai sudah ter-install biar run berikutnya skip step ini
    type nul > venv\.installed
    echo.
    echo [SETUP] Setup selesai!
    echo.
    echo OPSIONAL: untuk fitur AI (smart highlight, metadata, polish, translate),
    echo           buat file .env di folder ini dengan isi:
    echo.
    echo               GROQ_API_KEY=gsk_xxxxxxxxxxxxx
    echo.
    echo   Daftar gratis di https://console.groq.com (pakai login Google/GitHub)
    echo   Tanpa .env, fitur AI auto-skip, pipeline utama tetap jalan.
    echo.
    pause
)

REM === [4/4] Jalankan Web UI ===
echo ============================================================
echo   Menjalankan Web UI di http://127.0.0.1:7860
echo   Browser akan terbuka otomatis.
echo   Tekan Ctrl+C di jendela ini untuk stop server.
echo ============================================================
echo.

python app.py

echo.
echo Server berhenti. Tekan tombol apa saja untuk menutup jendela.
pause
