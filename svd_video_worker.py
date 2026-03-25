# -*- coding: utf-8 -*-
"""
Post-traitement livraison : SVD produit ~2–4 s ; on boucle jusqu’à une durée cible,
voix off (edge-tts) à partir du formulaire, sous-titres ASS, logo optionnel (ffmpeg).
"""
from __future__ import annotations

import asyncio
import math
import os
import shutil
import subprocess
import tempfile
from typing import Any

_EDGE_VOICES = {
    "fr": "fr-FR-DeniseNeural",
    "en": "en-US-JennyNeural",
    "ar": "ar-SA-ZariyahNeural",
    "es": "es-ES-ElviraNeural",
    "de": "de-DE-KatjaNeural",
    "pt": "pt-BR-FranciscaNeural",
}


def _ffmpeg_exe() -> str:
    return (os.environ.get("FFMPEG_PATH") or "ffmpeg").strip() or "ffmpeg"


def _ffprobe_exe() -> str:
    explicit = (os.environ.get("FFPROBE_PATH") or "").strip()
    if explicit:
        return explicit
    main = _ffmpeg_exe()
    low = main.lower()
    if low.endswith("ffmpeg.exe"):
        return main[:-10] + "ffprobe.exe"
    if main.endswith("ffmpeg"):
        return main[: -len("ffmpeg")] + "ffprobe"
    return "ffprobe"


def _media_duration_sec(path: str) -> float:
    try:
        r = subprocess.run(
            [
                _ffprobe_exe(),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return float((r.stdout or "0").strip() or 0)
    except (ValueError, subprocess.TimeoutExpired, OSError):
        return 0.0


def _edge_voice(lang_code: str) -> str:
    key = (lang_code or "fr")[:2].lower()
    return _EDGE_VOICES.get(key, _EDGE_VOICES["fr"])


def build_spoken_script(payload: dict[str, Any]) -> str:
    """Texte pour la voix off (limité pour rester fluide)."""
    lang = (payload.get("language") or "fr")[:2].lower()
    store = str(payload.get("storeName") or "").strip()
    product = str(payload.get("productName") or "").strip()
    price = str(payload.get("price") or "").strip()
    cur = str(payload.get("currency") or "").strip()
    destock = str(payload.get("destockPeriod") or "").strip()
    phone = str(payload.get("phone") or "").strip()
    loc = str(payload.get("location") or "").strip()
    country = str(payload.get("country") or "").strip()
    other = str(payload.get("other") or "").strip()[:400]
    model = str(payload.get("modelId") or "").strip()

    price_line = f"{price} {cur}".strip() if price or cur else ""

    if lang == "en":
        parts = [
            "Special offer.",
            f"{store}. {product}." if store or product else "",
            f"Price: {price_line}." if price_line else "",
            f"Valid: {destock}." if destock else "",
            f"Call {phone}." if phone else "",
            f"{loc}, {country}." if loc or country else "",
            other,
            f"Offer type: {model}." if model else "",
        ]
    else:
        parts = [
            "Offre spéciale à ne pas manquer.",
            f"{store}. {product}." if store or product else "",
            f"Prix : {price_line}." if price_line else "",
            f"Valable : {destock}." if destock else "",
            f"Téléphone : {phone}." if phone else "",
            f"{loc}, {country}." if loc or country else "",
            other,
            f"Type d'offre : {model}." if model else "",
        ]
    text = " ".join(p for p in parts if p)
    return text[:4500] if text else "Découvrez notre offre."


def _ass_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r", "")
        .replace("\n", r"\N")
    )


def write_overlay_ass(path: str, payload: dict[str, Any], duration_sec: float) -> None:
    """Sous-titres avec les champs principaux du formulaire."""
    lines = []
    for label, key in (
        ("Boutique", "storeName"),
        ("Produit", "productName"),
        ("Prix", None),
        ("Tél.", "phone"),
        ("Lieu", "location"),
    ):
        if key:
            val = str(payload.get(key) or "").strip()
        else:
            p = str(payload.get("price") or "").strip()
            c = str(payload.get("currency") or "").strip()
            val = f"{p} {c}".strip()
        if val:
            lines.append(f"{label} : {_ass_escape(val)}")

    other = str(payload.get("other") or "").strip()
    if other:
        lines.append(_ass_escape(other[:200]))

    body = r"\N".join(lines) if lines else _ass_escape("WaaW")

    end_h = int(duration_sec // 3600)
    end_m = int((duration_sec % 3600) // 60)
    end_s = duration_sec % 60
    end_tc = f"{end_h:d}:{end_m:02d}:{end_s:05.2f}".replace(".", ":")

    content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1024
PlayResY: 576
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,32,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,40,40,36,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,{end_tc},Default,,0,0,0,,{{\\an2\\pos(512,500)}}{body}
"""
    with open(path, "w", encoding="utf8") as f:
        f.write(content)


async def _edge_tts_save(text: str, out_mp3: str, voice: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_mp3)


def generate_speech_file(text: str, out_path: str, lang_hint: str) -> bool:
    try:
        voice = _edge_voice(lang_hint)
        asyncio.run(_edge_tts_save(text, out_path, voice))
        return os.path.isfile(out_path) and os.path.getsize(out_path) > 80
    except Exception as e:
        print(f"[svd_compose] edge-tts échec: {e}", flush=True)
        return False


def generate_silence_aac(out_path: str, duration_sec: float) -> bool:
    """Piste silencieuse de secours (AAC dans MP4)."""
    try:
        subprocess.run(
            [
                _ffmpeg_exe(),
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r=44100:cl=stereo",
                "-t",
                str(max(1.0, duration_sec)),
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                out_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        return os.path.isfile(out_path)
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"[svd_compose] silence ffmpeg: {e}", flush=True)
        return False


def compose_delivery_mp4(
    raw_mp4: str,
    final_mp4: str,
    payload: dict[str, Any],
    logo_abs: str | None,
    min_duration_sec: float,
    enable_tts: bool,
) -> dict[str, Any]:
    """
    Remplace raw_mp4 par une version final_mp4 (même chemin possible : écrit dans tmp puis replace).
    Retourne métadonnées pour job.svdMeta.
    """
    meta: dict[str, Any] = {
        "composeMinSec": min_duration_sec,
        "tts": False,
        "ffmpeg": _ffmpeg_exe(),
    }

    ff = _ffmpeg_exe()
    if not (os.path.isfile(ff) or shutil.which(ff)):
        raise RuntimeError(
            "ffmpeg introuvable (FFMPEG_PATH). Installez ffmpeg pour la livraison ≥30 s + audio."
        )

    spoken = build_spoken_script(payload)
    lang = str(payload.get("language") or "fr")

    tmpdir = tempfile.mkdtemp(prefix="svd_compose_")
    try:
        speech_mp3 = os.path.join(tmpdir, "speech.mp3")
        speech_ok = False
        if enable_tts and spoken:
            speech_ok = generate_speech_file(spoken, speech_mp3, lang)

        audio_d = _media_duration_sec(speech_mp3) if speech_ok else 0.0
        if not speech_ok:
            meta["tts"] = False
            meta["ttsNote"] = "silence ou edge-tts indisponible"
        else:
            meta["tts"] = True
            meta["ttsDurationSec"] = round(audio_d, 2)

        target = max(float(min_duration_sec), math.ceil(audio_d) + 0.75 if audio_d > 0 else float(min_duration_sec))
        meta["deliveryDurationSec"] = round(target, 2)

        ass_path = os.path.join(tmpdir, "overlay.ass")
        write_overlay_ass(ass_path, payload, target)

        audio_proc = os.path.join(tmpdir, "audio.m4a")
        if speech_ok and audio_d > 0:
            # Étendre ou couper l’audio à target
            if audio_d >= target - 0.05:
                filt = f"[0:a]atrim=0:{target},asetpts=PTS-STARTPTS[a]"
            else:
                filt = f"[0:a]apad=whole_dur={target},atrim=0:{target},asetpts=PTS-STARTPTS[a]"
            subprocess.run(
                [
                    _ffmpeg_exe(),
                    "-y",
                    "-i",
                    speech_mp3,
                    "-filter_complex",
                    filt,
                    "-map",
                    "[a]",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    audio_proc,
                ],
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
            )
        else:
            if not generate_silence_aac(audio_proc, target):
                raise RuntimeError("Impossible de générer la piste audio.")

        subs_esc = ass_path.replace("\\", "/").replace(":", r"\:")
        tmp_out = os.path.join(tmpdir, "out_nologo.mp4")

        fc = (
            f"[0:v]scale=1024:576:force_original_aspect_ratio=decrease,"
            f"pad=1024:576:(ow-iw)/2:(oh-ih)/2,setsar=1,setpts=PTS-STARTPTS[vb];"
            f"[vb]subtitles='{subs_esc}'[vout]"
        )

        subprocess.run(
            [
                _ffmpeg_exe(),
                "-y",
                "-stream_loop",
                "-1",
                "-i",
                raw_mp4,
                "-i",
                audio_proc,
                "-filter_complex",
                fc,
                "-map",
                "[vout]",
                "-map",
                "1:a:0",
                "-t",
                str(target),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                tmp_out,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )

        current = tmp_out
        if logo_abs and os.path.isfile(logo_abs):
            with_logo = os.path.join(tmpdir, "out_logo.mp4")
            subprocess.run(
                [
                    _ffmpeg_exe(),
                    "-y",
                    "-i",
                    current,
                    "-loop",
                    "1",
                    "-i",
                    logo_abs,
                    "-filter_complex",
                    "[1:v]format=rgba,scale=160:-1[lg];[0:v][lg]overlay=W-w-16:16",
                    "-map",
                    "0:a",
                    "-t",
                    str(target),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "copy",
                    with_logo,
                ],
                capture_output=True,
                text=True,
                timeout=600,
                check=True,
            )
            current = with_logo
            meta["logoOverlay"] = True

        shutil.move(current, final_mp4)
        return meta
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "")[:1500]
        raise RuntimeError(f"ffmpeg compose: {err}") from e
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
