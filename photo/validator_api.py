#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UMA Photo Validator API 

Behaviour:
- Accept JPG/PNG.
- Background must be plain white.
- Produce final 240x288 JPEG, <= 50 KB.
- /validate:
    * checks original size and background
    * crops to passport-style portrait
    * resizes to 240x288 and compresses
    * saves to photos/approved or photos/rejected
    * if approved, uploads to Supabase
- /fix-photo:
    * runs the SAME background + passport-crop + compression pipeline
    * does NOT save or upload
    * returns data_url for preview
    * if background is not white, returns same error text as /validate
"""

import base64
import io
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Iterable, cast

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

load_dotenv()

# ---------------- Config ----------------
TARGET_W, TARGET_H = 240, 288  # final dimensions

MAX_BYTES = int(os.getenv("UMA_MAX_BYTES", 50 * 1024))        # final JPEG
MAX_ORIGINAL_BYTES = int(os.getenv("UMA_MAX_ORIG_BYTES", MAX_BYTES))

PHOTOS_DIR = os.getenv("UMA_PHOTOS_DIR", "photos")

# background check (brightness 0-255 on grayscale)
BORDER_PIXELS = 12           # border thickness to inspect
WHITE_THRESHOLD = 210        # pixel >= this is considered "white"
BACKGROUND_MIN_WHITE = 0.55  # 60% of border pixels must be white

os.makedirs(os.path.join(PHOTOS_DIR, "approved"), exist_ok=True)
os.makedirs(os.path.join(PHOTOS_DIR, "rejected"), exist_ok=True)

# Pillow resampling constant
try: 
    from PIL.Image import Resampling
    
    RESAMPLE_LANCZOS = Resampling.LANCZOS
except Exception:  # pragma: no cover
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", getattr(Image, "BILINEAR", 2))

# Supabase config
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "student-photos")

def _log(*a: Any) -> None:
    print(*a, flush=True)


def _kb(num_bytes: int) -> float:
    return num_bytes / 1024.0


# ---------------- Supabase helper ----------------
def upload_to_supabase(jpg_bytes: bytes, path_in_bucket: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"ok": False}

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        info["error"] = "supabase_not_configured"
        _log("[supabase] missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
        return info

    try:
        object_path = f"{SUPABASE_BUCKET}/{path_in_bucket.lstrip('/')}"
        url = f"{SUPABASE_URL}/storage/v1/object/{object_path}"

        headers = {
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "image/jpeg",
            "x-upsert": "true",
        }

        resp = requests.post(url, headers=headers, data=jpg_bytes, timeout=30)

        if resp.status_code not in (200, 201):
            info["error"] = f"upload_failed_{resp.status_code}"
            info["details"] = resp.text[:200]
            _log("[supabase] upload failed:", info["error"], info.get("details", ""))
            return info

        public_url = (
            f"{SUPABASE_URL}/storage/v1/object/public/"
            f"{SUPABASE_BUCKET}/{path_in_bucket.lstrip('/')}"
        )

        info.update({"ok": True, "public_url": public_url, "status": resp.status_code})
        _log("[supabase] uploaded:", public_url)
        return info
    except Exception as e:  # pragma: no cover
        info["error"] = repr(e)
        _log("[supabase] upload exception:", repr(e))
        return info


# ---------------- Helpers ----------------
def sanitize_name(s: Optional[str]) -> str:
    return re.sub(r"[^\w\-]", "", (s or "").strip(), flags=re.ASCII)


def load_pil(upload: UploadFile, raw_bytes: Optional[bytes] = None) -> Tuple[Image.Image, bytes]:
    if raw_bytes is None:
        raw_bytes = upload.file.read()
    pil = Image.open(io.BytesIO(raw_bytes))
    if hasattr(ImageOps, "exif_transpose"):
        pil = ImageOps.exif_transpose(pil)
    return pil.convert("RGB"), raw_bytes


def border_white_ratio(pil_img: Image.Image) -> float:
    """
    Inspect a border around the image and return the fraction (0.0–1.0)
    of pixels that are "bright enough" to be considered white.
    """
    w, h = pil_img.size
    b = max(2, min(BORDER_PIXELS, w // 4, h // 4))

    gray = pil_img.convert("L")

    def frac_white(region: Image.Image) -> float:
        # getdata() returns an ImagingCore which *is* iterable at runtime,
        # but we cast it so static analysis is happy.
        data_iter = cast(Iterable[int], region.getdata())
        data_list = list(data_iter)
        total = len(data_list)
        if total == 0:
            return 0.0
        white = sum(1 for v in data_list if v >= WHITE_THRESHOLD)
        return white / total

    top = gray.crop((0, 0, w, b))
    bottom = gray.crop((0, h - b, w, h))
    left = gray.crop((0, 0, b, h))
    right = gray.crop((w - b, 0, w, h))

    vals = [frac_white(r) for r in (top, bottom, left, right)]
    return sum(vals) / len(vals)


def passport_crop(pil_img: Image.Image) -> Image.Image:
    """
    Crop image to a passport-style portrait:

    - Keep aspect ratio TARGET_W : TARGET_H (240x288).
    - Zoom in, but not too aggressively, so we keep space above the head.
    - Center vertically (no strong bias to the top).
    """
    w, h = pil_img.size
    target_ratio = TARGET_W / TARGET_H

    # Use about 90% of the height if the image is tall enough.
    crop_factor = 0.9 if h > TARGET_H * 1.2 else 1.0
    new_h = max(int(h * crop_factor), TARGET_H)
    new_w = int(new_h * target_ratio)

    # If that crop would be wider than the original, fall back to width-limited.
    if new_w > w:
        new_w = w
        new_h = int(new_w / target_ratio)

    # Horizontal: center
    left = max(0, (w - new_w) // 2)
    right = left + new_w

    # Vertical: center (no top bias -> more space above the head)
    max_top = h - new_h
    top = max_top // 2 if max_top > 0 else 0
    bottom = top + new_h

    return pil_img.crop((left, top, right, bottom))


def jpg_under_size(pil_img: Image.Image, limit: int = MAX_BYTES) -> bytes:
    lo, hi = 35, 95
    best: Optional[bytes] = None
    while lo <= hi:
        q = (lo + hi) // 2
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
        size = buf.tell()
        if size <= limit:
            best = buf.getvalue()
            lo = q + 1
        else:
            hi = q - 1
    if best is not None:
        return best
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=35, optimize=True, progressive=True)
    return buf.getvalue()


def run_pipeline(
    pil_img: Image.Image,
    *,
    require_white_bg: bool = True,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Core pipeline:
      - check background whiteness
      - crop to passport-style portrait
      - resize to TARGET_W x TARGET_H
      - compress under MAX_BYTES
    """
    issues: List[str] = []

    # 1) background check
    white_ratio = border_white_ratio(pil_img)
    background_ok = white_ratio >= BACKGROUND_MIN_WHITE

    if require_white_bg and not background_ok:
        issues.append(
            "La foto debe tomarse frente a una pared blanca lisa, "
            "sin objetos ni colores de fondo. Repita la foto con un fondo completamente blanco."
        )

    # 2) crop to passport style
    cropped = passport_crop(pil_img)

    # 3) resize to final size
    out_img = cropped.resize((TARGET_W, TARGET_H), resample=RESAMPLE_LANCZOS)

    # 4) compress under limit
    jpg = jpg_under_size(out_img, MAX_BYTES)

    info: Dict[str, Any] = {
        "issues": issues,
        "width": TARGET_W,
        "height": TARGET_H,
        "bytes": len(jpg),
        "white_ratio": white_ratio,
        "background_ok": background_ok,
    }
    return jpg, info


# ---------------- FastAPI app ----------------
app = FastAPI(title="UMA Photo Validator (no OpenCV)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint so Render / browsers don't see 404 on "/"
@app.get("/")
@app.head("/")   # <--- HEAD support for uptime monitors
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "message": "UMA Photo Validator running",
        "endpoints": {
            "health": "/health",
            "validate": "/validate",
            "fix_photo": "/fix-photo",
        },
    }


@app.get("/health")
@app.head("/health")   # <--- HEAD support here too
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "msg": "UMA validator healthy",
        "target": [TARGET_W, TARGET_H],
        "max_bytes": MAX_BYTES,
        "max_original_bytes": MAX_ORIGINAL_BYTES,
        "supabase": {
            "url": SUPABASE_URL,
            "bucket": SUPABASE_BUCKET,
            "configured": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY),
        },
    }


# ---------------- /validate ----------------
@app.post("/validate")
def validate(
    dni: Optional[str] = Form(None, description="Student DNI used as output filename"),
    image: UploadFile = File(...),
) -> Dict[str, Any]:
    dni = sanitize_name(dni) or "unknown_user"

    try:
        try:
            pil_in, raw_bytes = load_pil(image)
        except Exception:
            return {
                "ok": False,
                "issues": ["Archivo no es una imagen válida."],
                "bytes": 0,
            }

        original_size = len(raw_bytes)

        # Run main pipeline (includes background check + passport crop)
        jpg, info = run_pipeline(pil_in, require_white_bg=True)
        issues: List[str] = list(info.get("issues", []))

        # Extra check on original size for friendly message
        if original_size > MAX_ORIGINAL_BYTES:
            limit_kb = MAX_ORIGINAL_BYTES // 1024
            issues.insert(
                0,
                (
                    "Foto inválida: El archivo original pesa "
                    f"{_kb(original_size):.1f} KB; debe ser ≤ {limit_kb} KB. "
                    'Usa el botón "Arreglar con IA" o selecciona otra foto.'
                ),
            )

        ok = len(issues) == 0 and len(jpg) <= MAX_BYTES

        if not ok and not issues:
            issues.append("La foto no cumple con los criterios requeridos.")

        # Save locally
        bucket = "approved" if ok else "rejected"
        ts = int(time.time())
        fname = f"{dni}.jpg" if ok else f"{dni}_{ts}.jpg"
        save_dir = os.path.join(PHOTOS_DIR, bucket)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, fname)
        with open(save_path, "wb") as f:
            f.write(jpg)

        # Base64 data URL
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")

        # Supabase upload for approved photos
        supabase_info: Dict[str, Any] = {}
        supabase_url: Optional[str] = None
        if ok:
            object_path = f"approved/{fname}"
            supabase_info = upload_to_supabase(jpg, object_path)
            supabase_url = supabase_info.get("public_url")

        _log(
            "[validator]",
            "dni=",
            dni,
            "ok=",
            ok,
            "issues=",
            issues,
            "orig_bytes=",
            original_size,
            "bytes_final=",
            len(jpg),
            "local=",
            save_path,
            "supabase_used=",
            bool(supabase_url),
        )

        return {
            "ok": ok,
            "issues": issues,
            "width": info.get("width", TARGET_W),
            "height": info.get("height", TARGET_H),
            "bytes": info.get("bytes", len(jpg)),
            "category": bucket,
            "filename": fname,
            "relative_path": save_path,
            "data_url": data_url,
            "supabase_url": supabase_url,
            "supabase": supabase_info,
        }

    except Exception as e:  # pragma: no cover
        _log("[validator] unexpected error:", repr(e))
        return {
            "ok": False,
            "issues": [f"Error interno del validador: {repr(e)}"],
            "bytes": 0,
        }


# ---------------- /fix-photo ----------------
@app.post("/fix-photo")
def fix_photo(
    image: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Used by the student's "Arreglar con IA / Fix with AI" button.

    - Uses EXACTLY the same background rule as /validate.
    - Crops to passport style and compresses under 50 KB.
    - Does NOT save or upload anything.
    """
    try:
        try:
            pil_in, raw_bytes = load_pil(image)
        except Exception:
            return {
                "ok": False,
                "issues": ["Archivo no es una imagen válida."],
                "bytes": 0,
            }

        jpg, info = run_pipeline(pil_in, require_white_bg=True)
        issues: List[str] = list(info.get("issues", []))

        ok = len(issues) == 0 and len(jpg) <= MAX_BYTES

        data_url = "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")

        _log(
            "[fix-photo]",
            "ok=",
            ok,
            "orig_bytes=",
            len(raw_bytes),
            "bytes_final=",
            len(jpg),
            "white_ratio=",
            info.get("white_ratio"),
            "issues=",
            issues,
        )

        return {
            "ok": ok,
            "issues": issues,
            "width": info.get("width", TARGET_W),
            "height": info.get("height", TARGET_H),
            "bytes": info.get("bytes", len(jpg)),
            "data_url": data_url,
        }

    except Exception as e:  # pragma: no cover
        _log("[fix-photo] unexpected error:", repr(e))
        return {
            "ok": False,
            "issues": [f"Error interno al intentar corregir la foto: {repr(e)}"],
            "bytes": 0,
        }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    # Local run:  uvicorn validator_api:app --reload
    uvicorn.run("validator_api:app", host="127.0.0.1", port=8000, reload=True)
