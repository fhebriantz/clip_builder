"""Generate viral metadata (hook title, description, hashtag) via Groq LLM API.

Pakai Groq free tier (https://console.groq.com) — gratis ~14,400 req/hari.
Satu clip = 1 API call (combined prompt untuk hemat kuota).

Env var required: GROQ_API_KEY (taruh di .env file).

Model default: llama-3.3-70b-versatile (kualitas terbaik untuk task kreatif
di free tier). Bisa override via parameter.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


_SYSTEM_PROMPT = """Kamu social media expert untuk TikTok, Instagram Reels, YouTube Shorts di Indonesia.

Tugas: analisis transkrip clip pendek, generate metadata viral-friendly.

OUTPUT WAJIB JSON dengan struktur persis:
{
  "titles": ["judul 1", "judul 2", "judul 3"],
  "description": "caption 1-2 kalimat",
  "hashtags": ["tag1", "tag2", ..., "tag8"]
}

Rules:
- titles: 3 opsi, MAX 55 karakter per judul. Gaya hook: curiosity gap, angka spesifik, kontroversi,
  call-out common mistake, pertanyaan yang bikin penasaran. Bahasa Indonesia natural boleh campur
  kata English trending (tapi jangan berlebihan).
- description: MAX 150 karakter. Punchy hook + CTA singkat (contoh: 'Save buat nanti!',
  'Setuju ga?', 'Tag temanmu'). Bahasa Indonesia natural.
- hashtags: 8 hashtag TANPA tanda '#' (nanti ditambah otomatis). Mix:
    • 4 topic-specific sesuai konten
    • 3 trending umum Indonesia: fyp, viral, indonesia atau fypシ
    • 1 niche spesifik
  CamelCase kalau multi-kata (contoh: bisnisOnline).

JANGAN generate emoji berlebihan. JANGAN clickbait palsu. Fokus ke VALUE + curiosity."""


def _get_client():
    """Lazy import + check API key."""
    from groq import Groq  # noqa: lazy import

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY tidak ketemu.\n"
            "Setup:\n"
            "  1. Daftar gratis di https://console.groq.com\n"
            "  2. Buat API key di menu 'API Keys'\n"
            "  3. echo 'GROQ_API_KEY=gsk_...' >> .env"
        )
    return Groq(api_key=key)


def generate_metadata(
    transcript_text: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.85,
) -> dict:
    """Return {titles: [3], description: str, hashtags: [8]}.

    transcript_text: teks gabungan transkrip clip (max ~2000 karakter
    akan dipakai — cukup untuk context 60 detik bicara).
    """
    client = _get_client()
    content = transcript_text.strip()[:2000]

    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Transkrip clip:\n\n{content}"},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_completion_tokens=800,
    )
    raw = chat.choices[0].message.content
    data = json.loads(raw)

    # Normalisasi + validasi
    titles = data.get("titles") or []
    description = data.get("description") or ""
    hashtags = data.get("hashtags") or []

    # Pastikan hashtag pakai prefix '#'
    hashtags = [("#" + h.lstrip("#").replace(" ", "")) for h in hashtags]

    return {
        "titles": titles[:3],
        "description": description,
        "hashtags": hashtags[:10],
    }


def save_metadata(metadata: dict, clip_path: Path) -> Path:
    """Save metadata ke <clip>.meta.json (di folder yang sama dengan mp4)."""
    out = clip_path.with_suffix(".meta.json")
    out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return out


_HIGHLIGHT_SYSTEM_PROMPT = """Kamu video editor viral untuk TikTok, Instagram Reels, YouTube Shorts di Indonesia.

TUGAS: pilih bagian video yang layak jadi CLIP VIRAL dengan DURASI {duration} detik
(minimum {min_duration} detik, maksimum {max_duration} detik).

CARA KERJA:
1. Identify "momen puncak" — kalimat kunci yang punch (hook/insight/kontroversial/emotional)
2. EXPAND ke konteks sekitarnya untuk mencapai durasi target ({duration}s):
   - Ambil kalimat SEBELUM momen puncak (setup/intro konteks)
   - Ambil kalimat SESUDAH momen puncak (payoff/conclusion)
3. Pastikan clip STANDALONE — viewer paham tanpa konteks video full

CONTOH YANG SALAH:
- Clip durasi 5-10 detik — terlalu pendek, viewer bingung
- Clip cuma 1 kalimat — tidak ada konteks

CONTOH YANG BENAR:
- Clip 45-60 detik: setup (10s) + inti (30s) + payoff (10-20s)
- Start di awal kalimat, end di akhir kalimat

KRITERIA VIRAL-WORTHY:
- Emotional hook (cerita personal konkret)
- Insight praktis yang applicable
- Pendapat kontroversial yang call-out common mistake
- Quotable moment yang bisa jadi potongan viral

OUTPUT WAJIB JSON:
{
  "clips": [
    {
      "start": 125.3,
      "end": 187.5,
      "reason": "penjelasan viral-worthy",
      "viral_score": 9.2
    }
  ]
}

Rules:
- Tiap clip WAJIB durasi {min_duration}-{max_duration} detik — BUKAN lebih pendek
- Start match awal kalimat, end match akhir kalimat (lihat timestamp [start-end])
- Clip TIDAK boleh overlap
- viral_score 1-10 (10 = terbaik)
- Urutkan dari viral_score tertinggi
- Maksimal sesuai permintaan user"""


def _expand_clip_to_duration(
    clip_start: float,
    clip_end: float,
    segments: list[dict],
    target_duration: float,
    min_duration: float,
    max_duration: float,
) -> tuple[float, float]:
    """Post-process: kalau LLM balikin clip terlalu pendek, expand simetris
    dengan segmen sekitar sampai mencapai target duration."""
    if clip_end - clip_start >= min_duration:
        return clip_start, clip_end

    # Cari index segmen yang covered
    start_idx = next((i for i, s in enumerate(segments) if s["end"] > clip_start), 0)
    end_idx = next((i for i, s in enumerate(segments) if s["end"] >= clip_end), len(segments) - 1)

    # Expand kiri-kanan bergantian sampai mencapai target
    left, right = start_idx, end_idx
    new_start, new_end = segments[left]["start"], segments[right]["end"]
    while (new_end - new_start) < target_duration and (new_end - new_start) < max_duration:
        expanded = False
        if left > 0:
            left -= 1
            new_start = segments[left]["start"]
            expanded = True
        if (new_end - new_start) >= target_duration:
            break
        if right < len(segments) - 1:
            right += 1
            new_end = segments[right]["end"]
            expanded = True
        if not expanded:
            break

    if (new_end - new_start) > max_duration:
        new_end = new_start + max_duration

    return new_start, new_end


def generate_smart_highlights(
    segments: list[dict],
    max_clips: int = 5,
    target_duration: float = 60.0,
    min_duration: float = 30.0,
    max_duration: float = 75.0,
    model: str = "llama-3.3-70b-versatile",
) -> list[dict]:
    """Return list of {start, end, duration, reason, viral_score, text}.

    Analyze full transcript via LLM, pilih clip paling viral-worthy.
    Auto-expand clip yang terlalu pendek ke target_duration.
    """
    client = _get_client()

    lines = [f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments]
    transcript_formatted = "\n".join(lines)[:15000]

    system_prompt = (
        _HIGHLIGHT_SYSTEM_PROMPT
        .replace("{duration}", str(int(target_duration)))
        .replace("{min_duration}", str(int(min_duration)))
        .replace("{max_duration}", str(int(max_duration)))
    )

    user_msg = (
        f"Target durasi per clip: {int(target_duration)} detik "
        f"(minimum {int(min_duration)}, maksimum {int(max_duration)}).\n"
        f"Maksimal: {max_clips} clip.\n\n"
        f"Transkrip:\n{transcript_formatted}"
    )

    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.5,
        max_completion_tokens=2000,
    )
    raw = chat.choices[0].message.content
    data = json.loads(raw)

    clips_raw = data.get("clips") or []

    result: list[dict] = []
    for c in clips_raw[:max_clips]:
        try:
            start = float(c["start"])
            end = float(c["end"])
            if end <= start:
                continue
            # Post-process: expand kalau LLM kasih clip terlalu pendek
            start, end = _expand_clip_to_duration(
                start, end, segments, target_duration, min_duration, max_duration
            )
            text = " ".join(
                s["text"] for s in segments
                if s["end"] > start and s["start"] < end
            )
            result.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "duration": round(end - start, 2),
                "reason": c.get("reason", ""),
                "viral_score": float(c.get("viral_score", 0)),
                "text": text,
                "kind": "ai_highlight",
            })
        except (KeyError, ValueError, TypeError):
            continue

    return result


_POLISH_SYSTEM_PROMPT_BASE = """Kamu editor subtitle profesional untuk konten viral Indonesia.

TUGAS: polish subtitle raw (dari Whisper AI) jadi subtitle profesional yang siap ditampilkan.

RULES YANG HARUS DIIKUTI:
1. Hapus FILLER WORDS: umm, uh, eh, apa ya, ya kan, nah itu, gitu deh, anu, jadi gimana ya
2. Fix TYPO/misheard yang jelas. Pertimbangkan konteks kalimat & topik video — kata yang Whisper dengar
   mungkin hampir mirip fonetik tapi secara makna salah. Contoh:
     "conesi" → "koneksi", "straka" → bisa "strata" atau "serakah" (lihat konteks!)
3. Kapitalisasi: awal kalimat kapital, nama orang/merek kapital, sisanya kecil
4. Tanda baca minimal: titik akhir kalimat, koma jika perlu jeda
5. JANGAN ubah makna atau konteks asli
6. JANGAN terjemahkan — bahasa tetap sama dengan input
7. JANGAN merge atau split line — jumlah baris output HARUS SAMA persis dengan input
8. Pertahankan slang/gaya bicara natural (gue, lo, bro, dll) — JANGAN formalisasi berlebihan
9. Pertahankan nomor urut (idx) dari input

OUTPUT WAJIB JSON persis:
{
  "segments": [
    {"idx": 1, "text": "polished text"},
    {"idx": 2, "text": "polished text"},
    ...
  ]
}

CRITICAL: jumlah segments output HARUS SAMA dengan input."""


def _build_polish_prompt(topic_hint: str | None, vocabulary: list[str] | None) -> str:
    """Tambah topic context + vocabulary hint ke base prompt."""
    prompt = _POLISH_SYSTEM_PROMPT_BASE
    extras = []
    if topic_hint:
        extras.append(f"KONTEKS VIDEO: {topic_hint}")
    if vocabulary:
        extras.append(
            "KATA PENTING yang mungkin muncul (pakai ejaan ini kalau ada yang mirip): "
            + ", ".join(vocabulary)
        )
    if extras:
        prompt += "\n\n" + "\n\n".join(extras)
    return prompt


def _apply_corrections(text: str, corrections: dict[str, str]) -> str:
    """Replace case-insensitive dengan preserve leading capital kalau ada."""
    import re
    out = text
    for wrong, right in corrections.items():
        pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
        out = pattern.sub(right, out)
    return out


def _polish_batch(
    segments: list[dict],
    system_prompt: str,
    model: str,
    temperature: float,
) -> dict[int, str]:
    """Polish satu batch segments, return {idx: polished_text}."""
    client = _get_client()

    numbered = "\n".join(f"{i}. {s['text']}" for i, s in enumerate(segments, start=1))

    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Polish {len(segments)} baris subtitle berikut:\n\n{numbered}"},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_completion_tokens=4000,
    )
    raw = chat.choices[0].message.content
    data = json.loads(raw)
    polished = data.get("segments") or []
    return {int(p["idx"]): str(p.get("text", "")).strip() for p in polished if "idx" in p}


def polish_subtitles(
    segments: list[dict],
    topic_hint: str | None = None,
    vocabulary: list[str] | None = None,
    corrections: dict[str, str] | None = None,
    batch_size: int = 80,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
) -> list[dict]:
    """Polish subtitle text via LLM.

    topic_hint: deskripsi singkat topik video, misal "mindset bisnis vs serakah".
      LLM pakai ini saat fix typo ambigu (misal straka → serakah vs strata).

    vocabulary: list kata penting yang mungkin muncul (ejaan benar).
      LLM prefer ejaan ini untuk kata yang fonetik mirip.

    corrections: dict {wrong: right} untuk manual override 100% reliable.
      Diaplikasikan SETELAH LLM polish (jadi menang atas hasil LLM).
    """
    if not segments:
        return []

    # Apply corrections SEBELUM polish — biar LLM lihat kata benar, tidak ubah ulang
    if corrections:
        segments = [
            {**s, "text": _apply_corrections(s["text"], corrections)}
            for s in segments
        ]

    system_prompt = _build_polish_prompt(topic_hint, vocabulary)
    polished_all: dict[int, str] = {}

    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start : batch_start + batch_size]
        batch_result = _polish_batch(batch, system_prompt, model, temperature)
        for local_idx, text in batch_result.items():
            polished_all[batch_start + local_idx] = text

    result: list[dict] = []
    for i, seg in enumerate(segments, start=1):
        new_text = polished_all.get(i, seg["text"]).strip()
        if not new_text:
            new_text = seg["text"]
        # Apply corrections SEKALI LAGI setelah polish — safety net kalau LLM
        # entah kenapa revert kata yang sudah di-fix
        if corrections:
            new_text = _apply_corrections(new_text, corrections)
        result.append({"start": seg["start"], "end": seg["end"], "text": new_text})

    return result


_HOOK_SYSTEM_PROMPT = """Kamu video editor untuk konten viral TikTok/Reels/Shorts di Indonesia.

TUGAS: dari clip pendek, pilih MOMEN PUNCAK berdurasi 3-5 detik yang bisa berdiri
sendiri sebagai "hook teaser" — potongan paling catchy yang bikin viewer berhenti scroll.

Kriteria hook yang KUAT:
- Kalimat punch / kontroversial (contoh: "99% orang salah soal ini")
- Angka spesifik / fakta mengejutkan
- Pertanyaan yang hook curiosity
- Emotional peak (kemarahan, kegembiraan, shock)
- Call-out common mistake

Kriteria yang HARUS dihindari:
- Bagian intro/setup ("Halo teman-teman...")
- Bagian penjelasan panjang ("Jadi maksudnya adalah...")
- Kalimat yang tidak standalone tanpa konteks sebelumnya
- Filler ("umm eh apa ya")

OUTPUT WAJIB JSON:
{
  "hook_start": 12.5,
  "hook_end": 16.2,
  "text": "kalimat punch yang jadi hook",
  "reason": "kenapa ini hook paling kuat",
  "strength": 9.2
}

Rules:
- hook_start dan hook_end dalam SATUAN DETIK RELATIF TERHADAP AWAL CLIP (0 = start clip)
- Durasi 3-5 detik (minimum 2.5, maksimum 6)
- Harus di boundary kalimat (lihat timestamp di input)
- strength 1-10 (10 = terbaik)"""


def detect_hook_moment(
    clip_segments: list[dict],
    clip_start_abs: float = 0.0,
    model: str = "llama-3.3-70b-versatile",
) -> dict | None:
    """Detect 3-5 detik hook terkuat dalam clip. Return dict dengan timestamps
    RELATIF ke awal clip, atau None kalau tidak ketemu.

    clip_segments: segments yang masuk range clip (start-end sudah di-filter caller)
    clip_start_abs: absolute start time dari clip (untuk compute relative)
    """
    if not clip_segments:
        return None

    client = _get_client()

    # Format transcript dengan timestamp RELATIF
    lines = []
    for s in clip_segments:
        rel_start = s["start"] - clip_start_abs
        rel_end = s["end"] - clip_start_abs
        lines.append(f"[{rel_start:.1f}-{rel_end:.1f}] {s['text']}")
    transcript = "\n".join(lines)

    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _HOOK_SYSTEM_PROMPT},
            {"role": "user", "content": f"Clip transcript (timestamp relatif detik):\n\n{transcript}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.5,
        max_completion_tokens=500,
    )
    raw = chat.choices[0].message.content
    data = json.loads(raw)

    try:
        start = float(data["hook_start"])
        end = float(data["hook_end"])
        if end <= start or end - start < 1.0:
            return None
        return {
            "hook_start": round(start, 2),
            "hook_end": round(end, 2),
            "hook_duration": round(end - start, 2),
            "text": data.get("text", ""),
            "reason": data.get("reason", ""),
            "strength": float(data.get("strength", 0)),
        }
    except (KeyError, ValueError, TypeError):
        return None


_TRANSLATE_SYSTEM_PROMPT = """You are a professional subtitle translator for viral short-form video content.

TASK: translate subtitle text from source language to target language "{target_lang}".

STRICT RULES:
1. Preserve meaning, tone, and natural speaking style (don't formalize casual speech).
2. Keep each line as a standalone subtitle — DO NOT merge or split lines.
3. Output MUST have EXACT SAME number of idx as input.
4. Preserve the punch/emotion of viral content — translation should still be engaging.
5. If source has slang ("gue", "lo", "bro"), translate to equivalent casual register in target.
6. Keep proper nouns (names, brands) as-is unless target language has standard transliteration.
7. Do NOT add translator's notes, explanations, or commentary.

OUTPUT MUST BE JSON with structure:
{{
  "segments": [
    {{"idx": 1, "text": "translated text"}},
    {{"idx": 2, "text": "translated text"}},
    ...
  ]
}}

CRITICAL: number of segments output MUST equal number of input lines."""


def _translate_batch(
    segments: list[dict],
    target_lang: str,
    model: str,
    temperature: float,
) -> dict[int, str]:
    """Translate one batch, return {idx: translated_text}."""
    client = _get_client()

    numbered = "\n".join(f"{i}. {s['text']}" for i, s in enumerate(segments, start=1))
    system = _TRANSLATE_SYSTEM_PROMPT.replace("{target_lang}", target_lang)

    chat = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Translate {len(segments)} subtitle lines to {target_lang}:\n\n{numbered}"},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        max_completion_tokens=4000,
    )
    raw = chat.choices[0].message.content
    data = json.loads(raw)
    translated = data.get("segments") or []
    return {int(p["idx"]): str(p.get("text", "")).strip() for p in translated if "idx" in p}


def translate_subtitles(
    segments: list[dict],
    target_language: str,
    batch_size: int = 80,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.4,
) -> list[dict]:
    """Translate subtitle text to target language, preserve timing.

    target_language: ISO code ("en", "ja", "es", "zh", "fr") atau nama bahasa
        ("English", "Japanese", "Spanish", "Chinese", "French").

    Drop word-level timestamps (text berubah, word alignment tidak valid).
    """
    if not segments:
        return []

    translated_all: dict[int, str] = {}

    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start : batch_start + batch_size]
        batch_result = _translate_batch(batch, target_language, model, temperature)
        for local_idx, text in batch_result.items():
            translated_all[batch_start + local_idx] = text

    result: list[dict] = []
    for i, seg in enumerate(segments, start=1):
        new_text = translated_all.get(i, seg["text"]).strip() or seg["text"]
        result.append({"start": seg["start"], "end": seg["end"], "text": new_text})
    return result


def format_for_display(metadata: dict) -> str:
    """Format metadata untuk ditampilkan ke terminal."""
    lines = ["\n[bold]Metadata AI:[/bold]"]
    if metadata.get("titles"):
        lines.append("  [cyan]Title options:[/cyan]")
        for i, t in enumerate(metadata["titles"], 1):
            lines.append(f"    {i}. {t}")
    if metadata.get("description"):
        lines.append(f"  [cyan]Description:[/cyan] {metadata['description']}")
    if metadata.get("hashtags"):
        tags = " ".join(metadata["hashtags"])
        lines.append(f"  [cyan]Hashtags:[/cyan] {tags}")
    if metadata.get("hook"):
        h = metadata["hook"]
        lines.append(
            f"  [cyan]Hook:[/cyan] {h['hook_start']}s → {h['hook_end']}s "
            f"({h['hook_duration']}s, strength {h['strength']})"
        )
        lines.append(f"    [dim]text: {h['text'][:80]}[/dim]")
        lines.append(f"    [dim]reason: {h['reason']}[/dim]")
    return "\n".join(lines)
