#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Worker Stable Video Diffusion (image → vidéo) pour WaaW.
Appelé par Node : python svd_video_worker.py <chemin_fichier_job.json>

Variables d'environnement utiles :
  UPLOAD_DIR          — racine uploads (déjà injectée par le process Node parent)
  HF_HOME             — cache Hugging Face (optionnel)
  SVD_MODEL_ID        — défaut: stabilityai/stable-video-diffusion-img2vid-xt
  SVD_FORCE_CPU       — si 1 : forcer le CPU même si CUDA est dispo (tests)
  SVD_CPU_FP16        — défaut: 1 sur CPU : charge les poids FP16 (moins de RAM)
  SVD_NUM_FRAMES      — défaut: 14 sur GPU, 6 sur CPU (rapide CPU : 4)
  SVD_DECODE_CHUNK    — défaut: 4 sur GPU, 1 sur CPU
  SVD_NUM_INFERENCE_STEPS — défaut: 25 GPU, 12 CPU (moins d’étapes = plus rapide, qualité moindre)
  SVD_FPS             — images/s à l’export (défaut 7) ; avec SVD_MAX_DURATION_SEC borne la durée
  SVD_MAX_DURATION_SEC — durée max de la vidéo en secondes (défaut 30) : num_frames ≤ floor(durée×fps)
  SVD_MAX_MODEL_FRAMES — plafond modèle SVD img2vid-xt (défaut 25)
  SVD_CPU_THREADS     — défaut: tous les cœurs (accélère un peu PyTorch sur CPU)
  OPENAI_API_KEY      — optionnel : enrichit motion_bucket_id via un court appel LLM
  OPENAI_MODEL        — défaut: gpt-4o-mini
  SVD_DELIVERY_MIN_SEC — durée minimale livrée (défaut 30) : boucle SVD + voix + sous-titres (ffmpeg)
  SVD_ENABLE_TTS      — 1 (défaut) : voix off edge-tts à partir du formulaire ; 0 : silence
  SVD_SKIP_COMPOSE    — si 1 : garde seulement le clip SVD court (pas de post-traitement)
  FFPROBE_PATH        — optionnel (voisin de ffmpeg)
  SVD_MARKETING_MODE  — ken_burns (défaut) : vidéo pub fluide depuis l’image produit ; loop : boucle du clip SVD
  SVD_MARKETING_FPS   — ips pour le Ken Burns (défaut 30)
"""
from __future__ import annotations

import gc
import json
import os
import sys
import traceback


def fail(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    sys.exit(code)


def load_job(path: str) -> dict:
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def upload_root() -> str:
    base = os.environ.get("UPLOAD_DIR", "").strip()
    if not base:
        fail("UPLOAD_DIR manquant dans l'environnement.")
    return os.path.abspath(base)


def abs_media(rel: str | None) -> str | None:
    if not rel:
        return None
    return os.path.normpath(os.path.join(upload_root(), rel.replace("/", os.sep)))


def map_payload_to_motion_noise(payload: dict) -> tuple[int, float]:
    """
    SVD img2vid ne prend pas de prompt texte : on mappe ton / urgence sur motion_bucket_id et noise_aug_strength.
    Plages usuelles : motion 1–255 (défaut ~127), noise 0–0.1.
    """
    tone = str(payload.get("toneId") or "").lower()
    vtype = str(payload.get("videoTypeId") or "").lower()
    motion = 127
    noise = 0.02
    if "urgence" in vtype or "flash" in vtype:
        motion = min(200, motion + 40)
        noise = 0.04
    if "douceur" in tone or "calme" in tone:
        motion = max(40, motion - 35)
        noise = 0.01
    if "energie" in tone or "impact" in tone:
        motion = min(220, motion + 50)
    return motion, noise


def optional_openai_motion(payload: dict) -> int | None:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    try:
        import urllib.request

        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        text = json.dumps(payload, ensure_ascii=False)[:4000]
        body = json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Tu réponds uniquement par un entier entre 40 et 220 : motion_bucket_id pour une vidéo produit courte (plus haut = plus dynamique).",
                    },
                    {"role": "user", "content": text},
                ],
                "temperature": 0.3,
                "max_tokens": 8,
            }
        ).encode("utf8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        raw = data["choices"][0]["message"]["content"].strip()
        n = int("".join(c for c in raw if c.isdigit()) or "127")
        return max(40, min(220, n))
    except Exception:
        return None


def write_job_update(job_path: str, job: dict) -> None:
    from datetime import datetime, timezone

    job["updatedAt"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with open(job_path, "w", encoding="utf8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)


def main() -> None:
    if len(sys.argv) < 2:
        fail("Usage: python svd_video_worker.py <job.json>")
    job_path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(job_path):
        fail(f"Fichier job introuvable: {job_path}")

    job = load_job(job_path)
    job_id = job.get("id")
    payload = job.get("payload") or {}
    paths = job.get("persistedPaths") or {}
    product = abs_media(paths.get("productImage"))
    if not product or not os.path.isfile(product):
        fail(f"Image produit introuvable: {product}")

    out_rel = os.path.join("video-generation-output", f"{job_id}.mp4").replace("\\", "/")
    out_abs = os.path.normpath(os.path.join(upload_root(), out_rel.replace("/", os.sep)))
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)

    try:
        import torch
    except ImportError as e:
        fail(f"PyTorch non installé: {e}\nInstallez: pip install -r requirements-svd.txt")

    def cuda_is_usable():
        """True seulement si PyTorch peut réellement allouer sur CUDA (évite CPU-only + is_available() trompeur)."""
        if not torch.cuda.is_available():
            return False
        try:
            x = torch.zeros(1, device="cuda")
            del x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except RuntimeError:
            return False

    force_cpu = os.environ.get("SVD_FORCE_CPU", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    cuda_ok = cuda_is_usable()
    if not cuda_ok and torch.cuda.is_available():
        print(
            "[svd_video_worker] CUDA détecté mais non utilisable (PyTorch CPU-only ou DLL CUDA manquantes) — mode CPU.",
            flush=True,
        )
    use_cpu = force_cpu or not cuda_ok

    if use_cpu:
        try:
            n_cpu = int(os.environ.get("SVD_CPU_THREADS", "0").strip() or "0")
            if n_cpu <= 0:
                n_cpu = os.cpu_count() or 8
            torch.set_num_threads(max(1, n_cpu))
            n_interop = max(1, min(8, n_cpu // 2))
            torch.set_num_interop_threads(n_interop)
            print(
                f"[svd_video_worker] PyTorch CPU threads={torch.get_num_threads()} "
                f"(interop={n_interop})",
                flush=True,
            )
        except Exception as e:
            print(f"[svd_video_worker] Impossible de régler les threads CPU: {e}", flush=True)

    if use_cpu and not force_cpu:
        print(
            "[svd_video_worker] AVERTISSEMENT : CUDA indisponible — exécution sur CPU "
            "(très lent, forte utilisation RAM). Préférez un GPU NVIDIA en production.",
            flush=True,
        )
    elif use_cpu and force_cpu:
        print("[svd_video_worker] SVD_FORCE_CPU=1 — mode CPU forcé.", flush=True)

    motion, noise = map_payload_to_motion_noise(payload)
    llm_motion = optional_openai_motion(payload)
    if llm_motion is not None:
        motion = llm_motion

    model_id = os.environ.get(
        "SVD_MODEL_ID", "stabilityai/stable-video-diffusion-img2vid-xt"
    )
    default_frames = "6" if use_cpu else "14"
    default_chunk = "1" if use_cpu else "4"
    fps = max(1, int(os.environ.get("SVD_FPS", "7")))
    max_duration_sec = float(os.environ.get("SVD_MAX_DURATION_SEC", "30"))
    if max_duration_sec <= 0:
        max_duration_sec = 30.0
    model_max_frames = int(os.environ.get("SVD_MAX_MODEL_FRAMES", "25"))
    max_frames_by_duration = max(1, int(max_duration_sec * fps))
    num_frames = int(os.environ.get("SVD_NUM_FRAMES", default_frames))
    num_frames = min(num_frames, max_frames_by_duration, model_max_frames)
    num_frames = max(1, num_frames)
    decode_chunk = int(os.environ.get("SVD_DECODE_CHUNK", default_chunk))
    default_infer = "12" if use_cpu else "25"
    num_inference_steps = int(
        os.environ.get("SVD_NUM_INFERENCE_STEPS", default_infer)
    )

    try:
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import export_to_video, load_image
    except ImportError as e:
        fail(f"diffusers non installé: {e}\n pip install -r requirements-svd.txt")

    device = "cpu" if use_cpu else "cuda"
    print(
        f"[svd_video_worker] model={model_id} device={device} motion={motion} "
        f"noise_aug={noise} num_frames={num_frames} fps={fps} max_duration_sec={max_duration_sec} "
        f"infer_steps={num_inference_steps}",
        flush=True,
    )

    # Chargement poids : limiter la RAM (serveurs sans GPU souvent peu dotés)
    _load_kw = {
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
    }

    if use_cpu:
        gc.collect()
        fp16_cpu = os.environ.get("SVD_CPU_FP16", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        pipe = None
        last_err = None
        if fp16_cpu:
            try:
                print(
                    "[svd_video_worker] Chargement FP16 + safetensors (moins de RAM que float32).",
                    flush=True,
                )
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    **_load_kw,
                )
            except MemoryError as e:
                fail(
                    "Mémoire RAM insuffisante pour charger le modèle (même en FP16).\n"
                    "Augmentez la RAM, ou déployez sur une machine avec GPU NVIDIA + PyTorch CUDA.\n"
                    f"Détail: {e}"
                )
            except Exception as e:
                last_err = e
                print(
                    f"[svd_video_worker] Échec chargement FP16: {e} — essai float32 + low_cpu_mem_usage.",
                    flush=True,
                )
        if pipe is None:
            try:
                gc.collect()
                pipe = StableVideoDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    **_load_kw,
                )
            except MemoryError as e:
                fail(
                    "Mémoire RAM insuffisante pour charger le modèle SVD.\n"
                    "Augmentez la RAM du serveur, utilisez un GPU avec build PyTorch CUDA, "
                    "ou définissez SVD_CPU_FP16=1 (défaut) si les poids fp16 sont disponibles.\n"
                    f"Détail: {e}"
                )
            except Exception as e:
                if last_err:
                    fail(f"Chargement modèle impossible: {e}\n(précédent: {last_err})")
                raise
        pipe.to("cpu")
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    else:
        dtype = torch.float16
        try:
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                variant="fp16",
                **_load_kw,
            )
        except OSError:
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                **_load_kw,
            )
        pipe.to("cuda")
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    from PIL import Image as PILImage

    image = load_image(product)
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Format attendu par SVD img2vid (16:9)
    try:
        resample = PILImage.Resampling.LANCZOS
    except AttributeError:
        resample = PILImage.LANCZOS
    image = image.resize((1024, 576), resample)

    # Même device que le pipeline (bug corrigé : ne pas forcer cuda si mode CPU)
    generator = torch.Generator(device=device).manual_seed(42)
    _call_kw = dict(
        num_frames=num_frames,
        decode_chunk_size=decode_chunk,
        motion_bucket_id=motion,
        noise_aug_strength=noise,
        generator=generator,
    )
    try:
        result = pipe(
            image,
            num_inference_steps=num_inference_steps,
            **_call_kw,
        )
    except TypeError:
        # Anciennes versions diffusers sans num_inference_steps
        result = pipe(image, **_call_kw)
    frames = result.frames[0]
    if hasattr(frames, "__len__") and len(frames) > num_frames:
        frames = frames[:num_frames]

    export_to_video(frames, out_abs, fps=fps)

    if not os.path.isfile(out_abs):
        fail("export_to_video n'a pas produit de fichier MP4.")

    skip_compose = os.environ.get("SVD_SKIP_COMPOSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    compose_meta: dict | None = None
    if not skip_compose:
        try:
            from svd_video_compose import compose_delivery_mp4

            min_delivery = float(os.environ.get("SVD_DELIVERY_MIN_SEC", "30"))
            if min_delivery <= 0:
                min_delivery = 30.0
            tts_on = os.environ.get("SVD_ENABLE_TTS", "1").strip().lower() not in (
                "0",
                "false",
                "no",
            )
            logo_abs = abs_media(paths.get("logo"))
            tmp_comp = out_abs + ".compose-tmp.mp4"
            compose_meta = compose_delivery_mp4(
                out_abs,
                tmp_comp,
                payload,
                logo_abs if logo_abs and os.path.isfile(logo_abs) else None,
                min_delivery,
                tts_on,
                product_image_abs=product,
            )
            os.replace(tmp_comp, out_abs)
            print(
                f"[svd_video_worker] Compose livraison ~{compose_meta.get('deliveryDurationSec')}s "
                f"(tts={compose_meta.get('tts')})",
                flush=True,
            )
        except Exception as e:
            if os.path.isfile(out_abs + ".compose-tmp.mp4"):
                try:
                    os.remove(out_abs + ".compose-tmp.mp4")
                except OSError:
                    pass
            fail(
                "Post-traitement ffmpeg / livraison longue durée échoué.\n"
                "Installez ffmpeg (+ libass pour les sous-titres), edge-tts (pip), "
                "ou définissez SVD_SKIP_COMPOSE=1 pour ne garder que le clip SVD court.\n"
                f"Détail: {e}"
            )

    job["status"] = "completed"
    job["outputVideoRelativePath"] = out_rel
    job["error"] = None
    job["svdMeta"] = {
        "modelId": model_id,
        "device": device,
        "motionBucketId": motion,
        "noiseAugStrength": noise,
        "numFrames": num_frames,
        "fps": fps,
        "maxDurationSec": max_duration_sec,
        "maxFramesByDuration": max_frames_by_duration,
        "modelMaxFrames": model_max_frames,
        "numInferenceSteps": num_inference_steps,
        "decodeChunkSize": decode_chunk,
        "usedOpenAiMotion": llm_motion is not None,
        "cpuMode": use_cpu,
        "composeSkipped": skip_compose,
        "compose": compose_meta,
    }
    write_job_update(job_path, job)
    print(f"[svd_video_worker] OK -> {out_abs}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        try:
            job_path = os.path.abspath(sys.argv[1])
            if os.path.isfile(job_path):
                job = load_job(job_path)
                job["status"] = "failed"
                job["error"] = (str(e) + "\n" + tb)[:2000]
                write_job_update(job_path, job)
        except Exception:
            pass
        fail(str(e), 1)
