"""Face detection + smoothing untuk dynamic crop anti-shake.

Workflow:
1. Sample frame per N detik (default 1s)
2. Deteksi wajah terbesar pakai OpenCV Haar Cascade (sudah bundled dengan opencv-python,
   tidak perlu model download terpisah; akurat enough untuk talking-head viral)
3. Smoothing: EMA (alpha rendah) + dead-zone (abaikan gerakan kecil)
4. Output: list keyframe (waktu, x_center_pct) untuk FFmpeg crop expression

Kenapa anti-shake penting: face detection kadang jitter beberapa persen per frame
bahkan kalau wajah diam. Tanpa smoothing, crop goyang terus. Solusi:
- EMA (alpha=0.15): perubahan pelan, ~85% bobot ke posisi lama
- Dead-zone (5%): perubahan < 5% width diabaikan total
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def _probe_video_info(video_path: Path) -> tuple[int, int, float]:
    """Return (width, height, duration) via ffprobe — hindari import cv2 yang berat."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration",
         "-of", "default=nw=1:nk=1", str(video_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    return int(out[0]), int(out[1]), float(out[2])


def _extract_frame(video_path: Path, t: float, tmp_png: Path) -> bool:
    """Extract 1 frame pada waktu t ke PNG. Return True kalau sukses."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-ss", f"{t:.3f}", "-i", str(video_path),
         "-frames:v", "1", "-update", "1", str(tmp_png)],
        capture_output=True,
    )
    return r.returncode == 0 and tmp_png.exists()


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _dedupe_tagged(
    tagged: list[tuple[tuple[int, int, int, int], str]],
    iou_threshold: float = 0.3,
) -> list[tuple[tuple[int, int, int, int], str]]:
    """Dedupe bbox yang overlap — prefer frontal over profile ketika ada konflik."""
    # Sort: frontal dulu, lalu size descending (biar frontal "menang" saat dedupe)
    sorted_tagged = sorted(
        tagged,
        key=lambda x: (0 if x[1] == "frontal" else 1, -x[0][2] * x[0][3]),
    )
    kept: list[tuple[tuple[int, int, int, int], str]] = []
    for face, kind in sorted_tagged:
        if all(_iou(face, k[0]) < iou_threshold for k in kept):
            kept.append((face, kind))
    return kept


def detect_face_positions(
    video_path: Path,
    start: float,
    end: float,
    sample_interval: float = 1.0,
    strategy: str = "active_speaker",
) -> list[tuple[float, float]]:
    """Return [(timestamp_absolute, face_center_x_as_fraction_of_width), ...].

    strategy:
      'active_speaker' : prioritas wajah FRONTAL (menghadap kamera = biasanya bicara).
                         Kalau ada frontal → track yang paling besar.
                         Kalau semua profile → track yang paling besar.
                         Cocok untuk interview/podcast 2 orang, auto-follow speaker.
      'bbox_center'    : center dari bounding box SEMUA wajah (2 orang masuk frame).
                         Trade-off: kalau 2 orang jauhan, crop di tengah kosong.
      'biggest'        : ambil wajah paling besar regardless of type.
      'average'        : rata-rata center semua wajah.

    Detection: kombinasi 3 Haar Cascade (frontal + profile-kiri + profile-kanan).
    """
    import cv2

    haar_dir = cv2.data.haarcascades
    frontal = cv2.CascadeClassifier(haar_dir + "haarcascade_frontalface_default.xml")
    profile = cv2.CascadeClassifier(haar_dir + "haarcascade_profileface.xml")
    if frontal.empty() or profile.empty():
        raise RuntimeError(f"Haar cascade gagal di-load dari {haar_dir}")

    positions: list[tuple[float, float]] = []
    tmp_png = Path("/tmp/_face_frame.png")

    try:
        t = start
        while t < end:
            if _extract_frame(video_path, t, tmp_png):
                img = cv2.imread(str(tmp_png))
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    h_img, w_img = gray.shape
                    min_size = (int(w_img * 0.06), int(h_img * 0.06))

                    # Kombinasi 3 detektor, tagged by type (frontal/profile)
                    frontals = [(tuple(f), "frontal") for f in frontal.detectMultiScale(gray, 1.1, 5, minSize=min_size)]
                    profs_l = [(tuple(f), "profile") for f in profile.detectMultiScale(gray, 1.1, 5, minSize=min_size)]
                    flipped = cv2.flip(gray, 1)
                    profs_r_flipped = profile.detectMultiScale(flipped, 1.1, 5, minSize=min_size)
                    profs_r = [((w_img - x - fw, y, fw, fh), "profile") for (x, y, fw, fh) in profs_r_flipped]

                    tagged = _dedupe_tagged(frontals + profs_l + profs_r, iou_threshold=0.3)
                    # Filter ukuran minimum
                    tagged = [(f, k) for f, k in tagged if f[2] >= w_img * 0.06]

                    if len(tagged) > 0:
                        if strategy == "active_speaker":
                            # Prioritas frontal — itu yang biasanya sedang bicara
                            frontals_only = [(f, k) for f, k in tagged if k == "frontal"]
                            candidates = frontals_only if frontals_only else tagged
                            biggest = max(candidates, key=lambda x: x[0][2] * x[0][3])[0]
                            center_x = (biggest[0] + biggest[2] / 2) / w_img
                        elif strategy == "biggest" or len(tagged) == 1:
                            biggest = max(tagged, key=lambda x: x[0][2] * x[0][3])[0]
                            center_x = (biggest[0] + biggest[2] / 2) / w_img
                        elif strategy == "average":
                            centers = [(f[0] + f[2] / 2) / w_img for f, _ in tagged]
                            center_x = sum(centers) / len(centers)
                        else:  # bbox_center
                            min_x = min(f[0] for f, _ in tagged) / w_img
                            max_x = max(f[0] + f[2] for f, _ in tagged) / w_img
                            center_x = (min_x + max_x) / 2
                        positions.append((t, center_x))
                    else:
                        positions.append((t, 0.5))
                else:
                    positions.append((t, 0.5))
            t += sample_interval
    finally:
        tmp_png.unlink(missing_ok=True)

    return positions


def smooth_positions(
    positions: list[tuple[float, float]],
    alpha: float = 0.15,
    dead_zone: float = 0.05,
    quantize_step: float = 0.0,
) -> list[tuple[float, float]]:
    """[Legacy] EMA smoothing + dead-zone + optional quantization."""
    if not positions:
        return []

    smoothed = [positions[0]]
    for t, x in positions[1:]:
        _, prev_x = smoothed[-1]
        if abs(x - prev_x) < dead_zone:
            new_x = prev_x
        else:
            new_x = alpha * x + (1 - alpha) * prev_x
        smoothed.append((t, new_x))

    if quantize_step > 0:
        smoothed = [(t, round(x / quantize_step) * quantize_step) for t, x in smoothed]

    return smoothed


def smooth_state_machine(
    positions: list[tuple[float, float]],
    trigger_threshold: float = 0.06,
    transition_duration: float = 0.8,
    transition_curve: str = "ease_out",
) -> list[tuple[float, float]]:
    """State machine smoothing — 'stable' atau 'transitioning'.

    Logika:
      - State 'stable' : posisi diam 100% sampai raw face pindah >= trigger_threshold.
      - State 'transitioning' : bergerak dari posisi lama ke TARGET AKTUAL dengan
        kurva easing selama transition_duration detik. Sampai TEPAT di target.
      - Setelah transisi selesai, kembali 'stable'.

    transition_curve:
      'ease_out'   : cepat di awal, lambat mendekati target (recommended untuk
                     snappy catch-up yang tetap tidak overshoot).
      'smoothstep' : slow-fast-slow S-curve (paling halus, tapi start pelan).
      'linear'     : kecepatan konstan.

    trigger_threshold: delta minimal untuk memicu transisi (0.06 = 6% width).
    transition_duration: lama transisi dari start sampai target (detik).
    """
    if not positions:
        return []

    def apply_curve(p: float) -> float:
        if transition_curve == "smoothstep":
            return p * p * (3 - 2 * p)
        if transition_curve == "linear":
            return p
        return 3 * p - 3 * p * p + p * p * p  # ease_out cubic (default)

    result = [positions[0]]
    state = "stable"
    trans_start_t = 0.0
    trans_start_x = positions[0][1]
    trans_target_x = positions[0][1]
    instant_mode = transition_duration <= 0.01

    for t, x in positions[1:]:
        _, current = result[-1]

        if state == "stable":
            if abs(x - current) >= trigger_threshold:
                if instant_mode:
                    # Langsung snap ke raw di sample trigger
                    new_x = x
                else:
                    state = "transitioning"
                    trans_start_t = t
                    trans_start_x = current
                    trans_target_x = x
                    new_x = current
            else:
                # Micro-correction: kalau raw beda 2-6%, tarik pelan
                delta = x - current
                if abs(delta) >= trigger_threshold * 0.33:
                    new_x = current + delta * 0.25
                else:
                    new_x = current
        else:  # transitioning
            elapsed = t - trans_start_t
            if abs(x - trans_target_x) >= trigger_threshold:
                trans_start_t = t
                trans_start_x = current
                trans_target_x = x
                new_x = current
            elif elapsed >= transition_duration:
                # Snap ke raw LATEST untuk centering presisi
                new_x = x
                state = "stable"
            else:
                p = elapsed / transition_duration
                new_x = trans_start_x + (trans_target_x - trans_start_x) * apply_curve(p)

        result.append((t, new_x))

    return result


def _compress_keyframes(
    positions: list[tuple[float, float]],
    min_delta: float = 0.005,
) -> list[tuple[float, float]]:
    """Buang keyframe yang identik/sangat dekat dengan tetangga.

    Kalau beberapa keyframe berturut-turut punya value SAMA (hasil quantize),
    cukup simpan yang pertama dan terakhir — ini bikin crop LOCK konstan
    sepanjang segment tanpa micro-drift.
    """
    if len(positions) <= 2:
        return positions

    kept = [positions[0]]
    for i in range(1, len(positions) - 1):
        _, prev_x = kept[-1]
        _, x = positions[i]
        _, next_x = positions[i + 1]
        # Skip kalau SAMA PERSIS dengan tetangga (kanan-kiri) — bisa di-interp
        if abs(x - prev_x) < min_delta and abs(next_x - prev_x) < min_delta:
            continue
        kept.append(positions[i])
    kept.append(positions[-1])
    return kept


def build_crop_x_expression(
    positions: list[tuple[float, float]],
    crop_width_fraction: float,
    clip_start: float,
    easing: str = "ease_out",
) -> str:
    """Build FFmpeg crop x-parameter expression dengan easing antar keyframe.

    easing:
      - "linear"    : konstan velocity (jerky di boundary keyframe)
      - "ease_out"  : cepat di awal, lambat mendekati tujuan (cubic)
      - "smoothstep": lambat-cepat-lambat (S-curve, paling mulus tapi kurang responsif)
    """
    half_crop = crop_width_fraction / 2
    rel = [(t - clip_start, x) for t, x in positions]

    if len(rel) == 1:
        x0 = rel[0][1]
        return f"max(0\\,min((1-{crop_width_fraction})*iw\\,({x0}-{half_crop})*iw))"

    def eased(p: str) -> str:
        """Return easing expression untuk p (normalized time 0..1)."""
        if easing == "ease_out":
            # cubic ease-out: 1-(1-p)^3 = 3p - 3p^2 + p^3 — cepat di awal, lambat di akhir
            return f"(3*{p}-3*{p}*{p}+{p}*{p}*{p})"
        if easing == "smoothstep":
            # 3p^2 - 2p^3 — lambat-cepat-lambat (S-curve)
            return f"(3*{p}*{p}-2*{p}*{p}*{p})"
        return p  # linear

    def recurse(i: int) -> str:
        if i >= len(rel) - 1:
            _, x_last = rel[-1]
            return f"({x_last}-{half_crop})*iw"
        t1, x1 = rel[i]
        t2, x2 = rel[i + 1]
        dt = max(t2 - t1, 0.001)
        # normalized progress p = (t-t1)/dt, clamp implicit karena if(lt(t,t2)) sudah filter
        p = f"((t-{t1:.3f})/{dt:.3f})"
        e = eased(p)
        expr = f"({x1}+({x2-x1:.6f})*{e}-{half_crop})*iw"
        return f"if(lt(t\\,{t2:.3f})\\,{expr}\\,{recurse(i+1)})"

    inner = recurse(0)
    max_x = 1.0 - crop_width_fraction
    return f"max(0\\,min({max_x}*iw\\,{inner}))"


def compute_smart_crop_x(
    video_path: Path,
    start: float,
    end: float,
    target_aspect_w: int = 9,
    target_aspect_h: int = 16,
    sample_interval: float = 0.5,
    trigger_threshold: float = 0.06,
    transition_duration: float = 0.0,
    transition_curve: str = "ease_out",
    face_strategy: str = "active_speaker",
) -> str:
    """End-to-end: detect → state machine smoothing → build expression.

    Default tuning (state machine — diam saat listening, smooth saat cut):
      sample_interval=0.5      → sample tiap 0.5 detik untuk respons cepat ke cut
      trigger_threshold=0.06   → >= 6% delta baru trigger transisi (anti-jitter)
      transition_duration=1.5  → transisi selesai dalam 1.5 detik (smoothstep)
      easing='smoothstep'      → no jerk di boundary keyframe
      face_strategy='active_speaker' → prioritas wajah frontal (yang bicara)
    """
    w, h, _ = _probe_video_info(video_path)
    crop_w = h * (target_aspect_w / target_aspect_h)
    crop_width_fraction = crop_w / w

    raw = detect_face_positions(video_path, start, end, sample_interval=sample_interval, strategy=face_strategy)
    if not raw:
        half_crop = crop_width_fraction / 2
        max_x = 1.0 - crop_width_fraction
        return f"max(0\\,min({max_x}*iw\\,(0.5-{half_crop})*iw))"

    smoothed = smooth_state_machine(
        raw,
        trigger_threshold=trigger_threshold,
        transition_duration=transition_duration,
        transition_curve=transition_curve,
    )
    compressed = _compress_keyframes(smoothed)
    # Linear interp di expression karena state machine sudah apply curve sendiri
    # (hindari double-easing yang bikin transisi terasa lebih lambat).
    return build_crop_x_expression(compressed, crop_width_fraction, start, easing="linear")
