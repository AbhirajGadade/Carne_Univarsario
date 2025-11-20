#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UMA Photo Validator API

Rules:
- Image must be a valid JPG/PNG
- Detect 1 face (Haar)
- Resize to 240x288
- Background mostly white/clear
- Final JPEG <= 50KB
- Save to photos/approved or photos/rejected
- If approved, upload to Supabase Storage (bucket: student-photos)
"""

import base64
import io
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import cv2
import numpy as np
from cv2 import data as cv2_data
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import requests  # <--- needed for Supabase REST calls

load_dotenv()

# ---------------- Config ----------------
TARGET_W, TARGET_H = 240, 288
MAX_BYTES = int(os.getenv("UMA_MAX_BYTES", 50 * 1024))  # 50 KB strict
PHOTOS_DIR = os.getenv("UMA_PHOTOS_DIR", "photos")

# Supabase config
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "student-photos")

# background / face config
BORDER = 10
WHITE_L_MIN = 75
LAB_BG_DIST = 16
FACE_CENTER_X = (0.28, 0.72)
FACE_CENTER_Y = (0.26, 0.72)
FACE_REL_H = (0.18, 0.72)

os.makedirs(os.path.join(PHOTOS_DIR, "approved"), exist_ok=True)
os.makedirs(os.path.join(PHOTOS_DIR, "rejected"), exist_ok=True)

# Pillow resampling constant
try:
    from PIL.Image import Resampling
    RESAMPLE_LANCZOS = Resampling.LANCZOS
except Exception:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", getattr(Image, "BILINEAR", 2))


def _log(*a: Any) -> None:
    print(*a, flush=True)


# ---------------- Supabase upload helper ----------------
def _supabase_upload_local_file(local_path: str, object_key: str) -> Dict[str, Any]:
    """
    Uploads a local JPEG file to Supabase Storage.

    - local_path: path on disk (e.g. photos/approved/12345678.jpg)
    - object_key: key in the bucket (e.g. approved/12345678.jpg)

    Uses:
      SUPABASE_URL
      SUPABASE_SERVICE_ROLE_KEY
      SUPABASE_BUCKET
    """
    info: Dict[str, Any] = {"used": False, "object_key": object_key}

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        info["error"] = "missing_supabase_config"
        return info

    if not os.path.exists(local_path):
        info["error"] = "file_not_found"
        return info

    try:
        with open(local_path, "rb") as f:
            data = f.read()

        endpoint = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{object_key}"

        # IMPORTANT:
        #  - Use secret key as "apikey"
        #  - Do NOT put it in Authorization as Bearer (that causes 'Invalid Compact JWS')
        headers = {
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Content-Type": "image/jpeg",
            "x-upsert": "true",
        }

        resp = requests.post(endpoint, headers=headers, data=data, timeout=30)
        info["status_code"] = resp.status_code

        if resp.ok:
            info["used"] = True
            # public URL (bucket is Public in your screenshots)
            info[
                "public_url"
            ] = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{object_key}"
        else:
            info["error"] = f"{resp.status_code} {resp.text}"

    except Exception as e:
        info["error"] = repr(e)

    if info.get("error"):
        _log("[supabase] upload failed:", info["error"])
    else:
        _log("[supabase] upload ok:", info.get("public_url"))

    return info


# ---------------- FastAPI app ----------------
app = FastAPI(title="UMA Photo Validator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # adjust for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Helpers ----------------
def sanitize_name(s: Optional[str]) -> str:
    return re.sub(r"[^\w\-]", "", (s or "").strip(), flags=re.ASCII)


def load_pil(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    pil = Image.open(io.BytesIO(data))
    if hasattr(ImageOps, "exif_transpose"):
        pil = ImageOps.exif_transpose(pil)
    return pil.convert("RGB")  # type: ignore


def to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)


def detect_face(bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return (x,y,w,h) of biggest detected face, or None."""
    try:
        cascade_path = os.path.join(
            cv2_data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        cas = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = cas.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda r: int(r[2]) * int(r[3]), reverse=True)
        x, y, w, h = [int(v) for v in faces[0]]
        return (x, y, w, h)
    except Exception:
        return None


def crop_to_ratio(rgb: np.ndarray, face: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """Crop image to TARGET_W:TARGET_H, trying to keep face centered."""
    h, w = rgb.shape[:2]
    target = TARGET_W / TARGET_H
    r = w / h
    if r > target:
        new_w = int(h * target)
        new_h = h
        cx = w // 2
        if face:
            cx = face[0] + face[2] // 2
        x1 = max(0, min(w - new_w, cx - new_w // 2))
        y1 = 0
    else:
        new_w = w
        new_h = int(w / target)
        cy = h // 2
        if face:
            cy = face[1] + face[3] // 2
        x1 = 0
        y1 = max(0, min(h - new_h, cy - new_h // 2))
    return rgb[y1: y1 + new_h, x1: x1 + new_w]


def rgb_to_lab(a: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(a, cv2.COLOR_RGB2LAB)


def whiten_background(rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """Make background whiter and return (image, % of border that is already bright)."""
    h, w = rgb.shape[:2]
    b = min(BORDER, h // 4, w // 4)
    border = np.concatenate(
        [
            rgb[:b, :, :].reshape(-1, 3),
            rgb[-b:, :, :].reshape(-1, 3),
            rgb[:, :b, :].reshape(-1, 3),
            rgb[:, -b:, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.uint8)

    lab_img = rgb_to_lab(rgb)
    lab_border = rgb_to_lab(border.reshape(-1, 1, 3)).reshape(-1, 3)
    bg_lab = np.median(lab_border, axis=0)

    white_pct = (lab_border[:, 0] >= WHITE_L_MIN).sum() / lab_border.shape[0] * 100.0
    dist = np.linalg.norm(lab_img - bg_lab[None, None, :], axis=2)
    mask = dist < LAB_BG_DIST

    out = rgb.copy()
    out[mask] = (255, 255, 255)
    return out, float(white_pct)


def jpg_under_size(pil_img: Image.Image, limit: int = MAX_BYTES) -> bytes:
    """Binary search JPEG quality so that file <= limit bytes if possible."""
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


def face_rules(
    rgb: np.ndarray, face: Optional[Tuple[int, int, int, int]]
) -> List[str]:
    """Return list of face-related issues, empty if OK."""
    issues: List[str] = []
    if face is None:
        issues.append("No se detectó un rostro claro.")
        return issues
    x, y, w, h = face
    H, W = rgb.shape[:2]
    cx, cy = x + w / 2, y + h / 2
    if not (W * FACE_CENTER_X[0] <= cx <= W * FACE_CENTER_X[1]):
        issues.append("Rostro no está centrado horizontalmente.")
    if not (H * FACE_CENTER_Y[0] <= cy <= H * FACE_CENTER_Y[1]):
        issues.append("Rostro no está centrado verticalmente.")
    if not (H * FACE_REL_H[0] <= h <= H * FACE_REL_H[1]):
        issues.append("Rostro demasiado pequeño o grande.")
    return issues


# ---------------- Routes ----------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "msg": "UMA validator healthy",
        "target": [TARGET_W, TARGET_H],
        "max_bytes": MAX_BYTES,
        "supabase": {
            "url": SUPABASE_URL,
            "bucket": SUPABASE_BUCKET,
            "has_key": bool(SUPABASE_SERVICE_ROLE_KEY),
        },
    }


@app.post("/validate")
def validate(
    dni: Optional[str] = Form(None, description="Student DNI used as output filename"),
    image: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Main validator endpoint.

    Request: multipart/form-data with fields:
      - image: file
      - dni: student identifier (used for filename)
    """
    dni = sanitize_name(dni) or "unknown_user"
    issues: List[str] = []

    try:
        try:
            pil_in = load_pil(image)
        except Exception:
            return {
                "ok": False,
                "issues": ["Archivo no es una imagen válida."],
                "bytes": 0,
            }

        rgb = to_np(pil_in)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        face = detect_face(bgr)
        issues += face_rules(rgb, face)

        cropped = crop_to_ratio(rgb, face)
        whitened, white_pct = whiten_background(cropped)
        if white_pct < 60:
            issues.append("Fondo no es suficientemente claro/blanco.")

        pil_out = Image.fromarray(whitened).resize(
            (TARGET_W, TARGET_H), resample=RESAMPLE_LANCZOS
        )
        jpg = jpg_under_size(pil_out, MAX_BYTES)

        if len(jpg) > MAX_BYTES:
            issues.append(
                f"La foto final debe pesar ≤ {MAX_BYTES // 1024} KB "
                f"(actual: {len(jpg) / 1024:.1f} KB)."
            )

        ok = (len(issues) == 0) and (len(jpg) <= MAX_BYTES)
        if not ok and not issues:
            issues.append("La foto no cumple con los criterios requeridos.")

        bucket = "approved" if ok else "rejected"
        ts = int(time.time())
        fname = f"{dni}.jpg" if ok else f"{dni}_{ts}.jpg"
        save_dir = os.path.join(PHOTOS_DIR, bucket)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, fname)
        with open(save_path, "wb") as f:
            f.write(jpg)

        data_url = "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")

        # --- Supabase upload (only for approved photos) ---
        supabase_info: Dict[str, Any] = {"used": False}
        http_url: Optional[str] = None

        if ok:
            object_key = f"{bucket}/{fname}"  # approved/<dni>.jpg
            supabase_info = _supabase_upload_local_file(save_path, object_key)
            if supabase_info.get("used"):
                http_url = supabase_info.get("public_url")

        _log(
            "[validator]",
            "dni=", dni,
            "ok=", ok,
            "issues=", issues,
            "bytes=", len(jpg),
            "local=", save_path,
            "supabase_used=", supabase_info.get("used"),
        )

        return {
            "ok": ok,
            "issues": issues,
            "width": TARGET_W,
            "height": TARGET_H,
            "bytes": len(jpg),
            "category": bucket,
            "filename": fname,
            "relative_path": save_path,
            "data_url": data_url,
            "http_url": http_url,
            "supabase": supabase_info,
        }

    except Exception as e:
        _log("[validator] unexpected error:", repr(e))
        return {
            "ok": False,
            "issues": [f"Error interno del validador: {repr(e)}"],
            "bytes": 0,
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("validator_api:app", host="127.0.0.1", port=8000, reload=True)
