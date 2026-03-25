#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface Streamlit — même logique de formulaire que WaaW (vidéo publication) + génération SVD.
Lance : streamlit run svd_streamlit_app.py  (depuis ce dossier ou avec PYTHONPATH)

Prérequis : venv avec torch (CUDA), diffusers, ffmpeg dans le PATH pour la durée min. 30 s.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

# Répertoire de travail isolé (uploads + jobs)
_SCRIPT_DIR = Path(__file__).resolve().parent
_WORK_ROOT = _SCRIPT_DIR / "streamlit_work"
_UPLOAD_ROOT = _WORK_ROOT / "uploads"
_JOBS_DIR = _UPLOAD_ROOT / "video-generation-jobs"
_OUT_DIR = _UPLOAD_ROOT / "video-generation-output"
_TEST_DIR = _UPLOAD_ROOT / "test"

# Durée minimale de la vidéo finale (boucle ffmpeg si le clip SVD est plus court)
MIN_VIDEO_DURATION_SEC = float(os.environ.get("SVD_MIN_OUTPUT_DURATION_SEC", "30"))

# Données alignées sur frontend (videoPublicationTones / videoPublicationLists)
MODELS = [
    {"id": "urgence_stock_limite", "fr": "URGENCE – STOCK LIMITÉ"},
]
VIDEO_TYPES = [
    {"id": "peur_rater_urgence", "fr": "Crée la peur de rater, pousser à agir vite"},
]
TONE_OPTIONS: list[tuple[str, str]] = []
for _g in [
    ("commercial", [
        ("persuasif", "Persuasif"),
        ("accrocheur", "Accrocheur"),
        ("urgent", "Urgent"),
        ("promotionnel", "Promotionnel"),
    ]),
    ("autorite", [
        ("autoritaire", "Autoritaire"),
        ("critique", "Critique"),
        ("engage", "Engagé"),
    ]),
    ("relationnel", [
        ("amical", "Amical"),
        ("familier", "Familier"),
        ("respectueux", "Respectueux"),
    ]),
    ("informatif_pro", [
        ("informatif", "Informatif"),
        ("didactique", "Didactique / pédagogique"),
        ("professionnel", "Professionnel"),
        ("institutionnel", "Institutionnel"),
    ]),
    ("emotionnel", [
        ("emotionnel", "Émotionnel"),
        ("inspirant", "Inspirant / motivant"),
        ("empathique", "Empathique"),
        ("rassurant", "Rassurant"),
    ]),
    ("divertissant", [
        ("humoristique", "Humoristique"),
        ("leger_fun", "Léger / fun"),
        ("ironique", "Ironique / sarcastique"),
    ]),
]:
    for tid, label in _g[1]:
        TONE_OPTIONS.append((tid, label))

COUNTRIES = [
    ("SN", "Sénégal"), ("FR", "France"), ("BE", "Belgique"), ("CH", "Suisse"),
    ("CA", "Canada"), ("MA", "Maroc"), ("DZ", "Algérie"), ("TN", "Tunisie"),
    ("CI", "Côte d'Ivoire"), ("ML", "Mali"), ("BF", "Burkina Faso"), ("OTHER", "Autre"),
]
CURRENCIES = [
    ("XOF", "FCFA (XOF)"), ("EUR", "Euro"), ("USD", "Dollar US"),
]
GENDERS = [
    ("homme", "Homme"), ("femme", "Femme"), ("les_deux", "Les deux"), ("peu_importe", "Peu importe"),
]
LANGUAGES = [("fr", "Français"), ("en", "English"), ("ar", "العربية")]


def _ensure_dirs() -> None:
    for d in (_JOBS_DIR, _OUT_DIR, _TEST_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _ffmpeg_bin(name: str) -> str:
    env = (os.environ.get("FFMPEG_PATH") or "").strip()
    if name == "ffmpeg" and env:
        return env
    p = shutil.which(name)
    return p or name


def _video_duration_seconds(path: Path) -> float | None:
    ffprobe = _ffmpeg_bin("ffprobe")
    try:
        r = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode != 0:
            return None
        return float(r.stdout.strip())
    except (ValueError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def extend_to_min_duration(src: Path, dst: Path, min_sec: float) -> tuple[bool, str]:
    """
    Si la vidéo fait moins de min_sec, boucle avec ffmpeg jusqu'à atteindre min_sec.
    Si ffprobe indique déjà >= min_sec, copie sans ré-encoder (ou stream copy).
    """
    dur = _video_duration_seconds(src)
    ffmpeg = _ffmpeg_bin("ffmpeg")
    if dur is not None and dur >= min_sec - 0.05:
        shutil.copy2(src, dst)
        return True, f"Durée déjà ≥ {min_sec}s ({dur:.1f}s), fichier copié."

    # Boucle infinie puis coupe à min_sec (sans piste audio pour simplicité)
    cmd = [
        ffmpeg,
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(src),
        "-t",
        str(min_sec),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)
        return True, f"Vidéo étendue à {min_sec}s (boucle ffmpeg)."
    except FileNotFoundError:
        shutil.copy2(src, dst)
        return False, "ffmpeg introuvable : sortie SVD brute (durée < 30s possible)."
    except subprocess.CalledProcessError as e:
        shutil.copy2(src, dst)
        return False, f"ffmpeg erreur, sortie brute : {e.stderr[:300] if e.stderr else e}"


def run_worker(job_path: Path) -> int:
    os.environ["UPLOAD_DIR"] = str(_UPLOAD_ROOT)
    worker = _SCRIPT_DIR / "svd_video_worker.py"
    py = sys.executable
    return subprocess.run([py, str(worker), str(job_path)], cwd=str(_SCRIPT_DIR)).returncode


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="WaaW — Génération vidéo (SVD)", layout="wide")
    st.title("Génération vidéo produit (Stable Video Diffusion)")
    st.caption(
        "Formulaire aligné sur WaaW. Le modèle SVD produit un clip court ; "
        f"la sortie est prolongée à **minimum {MIN_VIDEO_DURATION_SEC:g}s** avec ffmpeg (boucle)."
    )

    _ensure_dirs()

    with st.form("gen"):
        model_id = st.selectbox("Modèle", options=[m["id"] for m in MODELS], format_func=lambda x: next(m["fr"] for m in MODELS if m["id"] == x))
        c1, c2 = st.columns(2)
        with c1:
            store_name = st.text_input("Nom de la boutique", "")
            product_name = st.text_input("Nom du produit", "")
            price = st.text_input("Prix", "")
            destock_period = st.text_input("Période destock / urgence", "")
        with c2:
            phone = st.text_input("Téléphone", "")
            location = st.text_input("Localisation", "")
            country = st.selectbox("Pays", options=[c[0] for c in COUNTRIES], format_func=lambda x: next(c[1] for c in COUNTRIES if c[0] == x))
            language = st.selectbox("Langue", options=[l[0] for l in LANGUAGES], format_func=lambda x: next(l[1] for l in LANGUAGES if l[0] == x))

        tone_id = st.selectbox(
            "Ton",
            options=[t[0] for t in TONE_OPTIONS],
            format_func=lambda x: next(t[1] for t in TONE_OPTIONS if t[0] == x),
        )
        video_type_id = st.selectbox(
            "Type de vidéo",
            options=[v["id"] for v in VIDEO_TYPES],
            format_func=lambda x: next(v["fr"] for v in VIDEO_TYPES if v["id"] == x),
        )
        currency = st.selectbox("Devise", options=[c[0] for c in CURRENCIES], format_func=lambda x: next(c[1] for c in CURRENCIES if c[0] == x))
        gender = st.selectbox("Public cible", options=[g[0] for g in GENDERS], format_func=lambda x: next(g[1] for g in GENDERS if g[0] == x))
        other = st.text_area("Autres précisions (optionnel)", "", max_chars=500)

        img = st.file_uploader("Image produit *", type=["jpg", "jpeg", "png", "webp"])
        logo = st.file_uploader("Logo (optionnel)", type=["jpg", "jpeg", "png", "webp"])

        submitted = st.form_submit_button("Générer la vidéo")

    if not submitted:
        return

    if not img:
        st.error("Ajoute une image produit.")
        return
    if not all([store_name.strip(), product_name.strip(), price.strip(), destock_period.strip(), phone.strip(), location.strip()]):
        st.error("Remplis boutique, produit, prix, période destock, téléphone et localisation.")
        return

    job_id = str(uuid.uuid4())
    ext = Path(img.name).suffix or ".jpg"
    product_rel = f"test/{job_id}{ext}"
    product_abs = _TEST_DIR / f"{job_id}{ext}"
    product_abs.parent.mkdir(parents=True, exist_ok=True)
    product_abs.write_bytes(img.getbuffer())

    logo_rel = None
    if logo:
        lext = Path(logo.name).suffix or ".png"
        logo_rel = f"test/logo_{job_id}{lext}"
        (_UPLOAD_ROOT / logo_rel.replace("/", os.sep)).parent.mkdir(parents=True, exist_ok=True)
        (_UPLOAD_ROOT / logo_rel.replace("/", os.sep)).write_bytes(logo.getbuffer())

    payload = {
        "modelId": model_id,
        "storeName": store_name,
        "productName": product_name,
        "price": price,
        "destockPeriod": destock_period,
        "phone": phone,
        "location": location,
        "country": country,
        "language": language,
        "toneId": tone_id,
        "videoTypeId": video_type_id,
        "other": other,
        "currency": currency,
        "gender": gender,
        "regenerateHint": "",
    }

    job = {
        "id": job_id,
        "userId": 1,
        "status": "running",
        "payload": payload,
        "persistedPaths": {"productImage": product_rel.replace("\\", "/"), "logo": logo_rel},
    }

    job_path = _JOBS_DIR / f"{job_id}.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf8")

    os.environ["UPLOAD_DIR"] = str(_UPLOAD_ROOT)
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(_WORK_ROOT / ".cache" / "huggingface")

    with st.spinner("Génération SVD en cours (première fois : téléchargement du modèle)…"):
        code = run_worker(job_path)

    raw_out = _OUT_DIR / f"{job_id}.mp4"
    final_out = _OUT_DIR / f"{job_id}_min{int(MIN_VIDEO_DURATION_SEC)}s.mp4"

    if code != 0 or not raw_out.is_file():
        st.error("Échec du worker. Vérifie les logs ci-dessus ou le fichier job.")
        if job_path.is_file():
            try:
                err_job = json.loads(job_path.read_text(encoding="utf8"))
                st.code(err_job.get("error") or "pas de détail", language="text")
            except Exception:
                pass
        return

    ok_ext, msg = extend_to_min_duration(raw_out, final_out, MIN_VIDEO_DURATION_SEC)
    st.info(msg)
    if not ok_ext:
        st.warning("Installe ffmpeg et ajoute-le au PATH pour la durée min. 30s.")

    st.success("Vidéo prête.")
    show_path = final_out if final_out.is_file() else raw_out
    st.video(str(show_path))
    st.caption(f"Fichier : `{show_path}` (brut SVD : `{raw_out}`)")

    st.download_button(
        "Télécharger la vidéo",
        data=show_path.read_bytes(),
        file_name=show_path.name,
        mime="video/mp4",
    )


if __name__ == "__main__":
    main()
