# local_videos/

Taruh file video lokal di sini untuk testing tanpa download ulang.

## Cara Pakai

```bash
# Copy/move video ke folder ini
cp ~/Downloads/my_video.mp4 local_videos/

# Panggil dengan nama file saja (tidak perlu full path)
python test_facetrack.py my_video.mp4 --duration 30

# Atau pakai full path biasa (masih bekerja)
python test_facetrack.py local_videos/my_video.mp4 --duration 30
```

Script akan auto-resolve: kalau path tidak ada di working dir, dia cari di `local_videos/`.

## Format yang Didukung

Apapun yang FFmpeg bisa baca: `.mp4`, `.mkv`, `.mov`, `.webm`, `.avi`, `.flv`, dll.

## Aspect Ratio

- **Landscape** (16:9 dsb): face tracking + center crop jadi 9:16 portrait
- **Portrait** (9:16, sudah Shorts-ready): skip crop, langsung scale
- **Square** (1:1): crop horizontal dari tengah jadi 9:16 dengan face tracking
