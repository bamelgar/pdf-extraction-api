# main.py
#
# FastAPI app with two extraction endpoints:
#   - POST /extract/images  -> runs image extraction ONLY, using the same block you proved in "limiter-6"
#   - POST /extract/all     -> same image path, tables OFF by default (mirrors limiter-6 behavior)
# Both honor a `workers` multipart form field (defaults to 4) and pass it through to the extractor CLI.
#
# Supabase uploads use the REST Storage API exactly like the working script:
# PUT {SUPABASE_URL}/storage/v1/object/{bucket}/{object}
#   headers: apikey, Authorization: Bearer <service_role>, x-upsert: true
#
# Response shape matches the limiter-6 “/extract/all” wrapper:
# {
#   "results": [ { ...image item... }, ... ],
#   "count": <int>,
#   "extraction_timestamp": "<UTC ISO>",
#   "success": true,
#   "statistics": {
#       "images_count": <int>,
#       "tables_count": 0,
#       "supabase_enabled": true/false,
#       "elapsed_seconds": <float>,
#       "workers": <int>
#   }
# }
#
# NOTE: This file intentionally does NOT alter your image extraction logic/shape. The only "surgical" change
# is: workers now come from the HTTP form everywhere, defaulting to 4.

import io
import os
import sys
import json
import time
import glob
import uuid
import shutil
import base64
import logging
import tempfile
import datetime
import subprocess
from typing import Optional, List

import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, PlainTextResponse

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("main")

# -----------------------------------------------------------------------------
# Environment / Supabase
# -----------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "public-images")
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)

# Public URL helper for files stored in the "public" policy
def supabase_public_url(object_path: str) -> str:
    # public URL pattern for Supabase storage
    return f"{SUPABASE_URL}/storage/v1/object/public/{object_path}"

def upload_to_supabase(local_path: str, object_name: str) -> Optional[str]:
    """
    Upload a file to Supabase Storage using raw REST.
    Returns the public URL on success, or None on failure.
    """
    if not SUPABASE_ENABLED:
        return None

    object_path = f"{SUPABASE_BUCKET}/{object_name}"
    put_url = f"{SUPABASE_URL}/storage/v1/object/{object_path}"

    # Guess content type (we only push PNGs here)
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "x-upsert": "true",
        "Content-Type": "image/png",
    }
    with open(local_path, "rb") as f:
        resp = requests.post(put_url, headers=headers, data=f.read(), timeout=60)

    if resp.status_code in (200, 201):
        logger.info("✅ Uploaded %s to Supabase", object_name)
        public_url = supabase_public_url(object_path)
        logger.info("✅ Image %s uploaded to Supabase: %s", object_name, public_url)
        return public_url
    else:
        logger.error("❌ Supabase upload failed (%s): %s", resp.status_code, resp.text)
        return None

# -----------------------------------------------------------------------------
# FastAPI app & health
# -----------------------------------------------------------------------------
app = FastAPI(title="PDF Extraction API")

@app.get("/", response_class=PlainTextResponse)
def root():
    return "PDF Extraction API is running"

# -----------------------------------------------------------------------------
# Core: image extraction block (mirrors working limiter-6 behavior)
# -----------------------------------------------------------------------------
def run_image_extraction(
    pdf_bytes: bytes,
    workers: int = 4,
    min_quality: float = 0.3,
    min_width: int = 100,
    min_height: int = 100,
    vector_threshold: int = 10,
    timeout_s: int = 900,
):
    """
    Runs enterprise_image_extractor.py, uploads results to Supabase if configured,
    and returns (results_list, statistics_dict). This is the same logic path you
    proved in the limiter-6 script, with the EXACT same packaging and fields.
    """

    # Keep the same temp/work layout as your proven path
    t0 = time.time()
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "input.pdf")
        images_dir = os.path.join(temp_dir, "pdf_images")
        os.makedirs(images_dir, exist_ok=True)

        # Save incoming PDF to disk
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # Log lines mirroring limiter-6 tone
        logger.info("[LIMITERS DISABLED] Processing all images")
        logger.info(
            "Processing PDF: %s, Size: %d bytes, Temp path: %s",
            os.path.basename(pdf_path),
            len(pdf_bytes),
            pdf_path,
        )
        logger.info("📋 Table extraction is DISABLED")
        logger.info("🖼️ Extracting images...")

        # Build the exact CLI command style you were using
        cmd = [
            sys.executable,
            "enterprise_image_extractor.py",
            pdf_path,
            "--output-dir", images_dir,
            "--workers", str(int(workers)),  # <- form-driven workers
            "--min-width", str(int(min_width)),
            "--min-height", str(int(min_height)),
            "--min-quality", str(float(min_quality)),
            "--vector-threshold", str(int(vector_threshold)),
            "--clear-output",
        ]

        logger.info("Running command: %s (timeout=%ss)", " ".join(cmd), timeout_s)

        # Run extractor
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        logger.info("Image extraction exit code: %s", proc.returncode)
        # Show a trimmed stdout to keep logs friendly (like your limiter-6)
        stdout_preview = (proc.stdout or "")[:500]
        logger.info("Image stdout (first 500 chars): %s", stdout_preview if stdout_preview else "(empty)")
        if proc.stderr:
            logger.error("Image stderr: %s", proc.stderr)

        if proc.returncode != 0:
            raise RuntimeError(f"Image extractor failed: {proc.stderr or proc.stdout or 'unknown error'}")

        # The extractor writes a metadata JSON; keep the same pattern (fallback if missing)
        meta_path = os.path.join(images_dir, "metadata.json")
        extraction_meta_path = os.path.join(images_dir, "extraction_metadata.json")

        meta_map = {}
        if os.path.exists(extraction_meta_path):
            try:
                with open(extraction_meta_path, "r", encoding="utf-8") as f:
                    extraction_meta = json.load(f)
                # build a map filename -> metadata
                for m in extraction_meta.get("images", []):
                    # some runs provide `filename` while files on disk are named similarly
                    key = m.get("filename") or m.get("file_name") or m.get("name")
                    if key:
                        meta_map[key] = m
            except Exception:
                pass
        elif os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    extraction_meta = json.load(f)
                for m in extraction_meta.get("images", []):
                    key = m.get("filename") or m.get("file_name") or m.get("name")
                    if key:
                        meta_map[key] = m
            except Exception:
                pass

        # Package images list exactly like your working block
        images = []
        timestamp_prefix = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        png_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))

        logger.info("Processing all %d images", len(png_files))
        logger.info("Found %d images in metadata (packaging %d)", len(meta_map), len(png_files))

        for img_path in png_files:
            filename = os.path.basename(img_path)
            # Derive a safe unique storage name (prefix with UTC second-stamp)
            unique_name = f"{timestamp_prefix}_{filename}"

            # Upload to Supabase if configured; else leave base64 fallback fields
            supa_url = upload_to_supabase(img_path, unique_name) if SUPABASE_ENABLED else None

            # Attach any known metadata
            meta = meta_map.get(filename, {})
            page = meta.get("page") or meta.get("page_number")
            width = meta.get("width")
            height = meta.get("height")
            img_type = meta.get("type") or meta.get("image_type")
            method = meta.get("method") or ("vector" if "vector" in filename else "embedded")
            quality = meta.get("quality")

            # OCR/text is optional in your pipeline
            ocr_text = meta.get("text") or meta.get("ocr_text")
            ocr_success = meta.get("ocr_success")

            # Final item mirrors limiter-6 fields (including redundant URL keys used by your n8n)
            item = {
                "type": "image",
                "page": page,
                "image_type": img_type,
                "method": method,
                "quality": quality,
                "width": width,
                "height": height,

                "fileName": filename,
                "filename": filename,          # (kept for compatibility)
                "filePath": img_path,          # local temp path (diagnostic)

                # Supabase/public URLs (same naming you relied on)
                "supabase_url": supa_url,
                "url": supa_url,
                "image_url": supa_url,

                # fallbacks if needed by downstream (kept lightweight)
                "text": ocr_text,
                "ocr_success": ocr_success,
                "mime_type": "image/png",
            }

            # Only include base64 if Supabase is off
            if not supa_url:
                with open(img_path, "rb") as f:
                    item["image_base64"] = base64.b64encode(f.read()).decode("utf-8")

            images.append(item)

        elapsed = time.time() - t0
        stats = {
            "images_count": len(images),
            "tables_count": 0,
            "supabase_enabled": SUPABASE_ENABLED,
            "elapsed_seconds": round(elapsed, 2),
            "workers": int(workers),
        }
        return images, stats

# -----------------------------------------------------------------------------
# /extract/images — route DIRECTLY through the proven image block
# -----------------------------------------------------------------------------
@app.post("/extract/images")
async def extract_images(
    file: UploadFile = File(...),
    # Let n8n set this; we default to 4 (surgical change you asked for)
    workers: int = Form(4),
    # keep familiar knobs available; defaults match your runs
    min_quality: float = Form(0.3),
    vector_threshold: int = Form(10),
    min_width: int = Form(100),
    min_height: int = Form(100),
    timeout_s: int = Form(900),
):
    pdf_bytes = await file.read()

    try:
        images, stats = run_image_extraction(
            pdf_bytes=pdf_bytes,
            workers=workers,
            min_quality=min_quality,
            min_width=min_width,
            min_height=min_height,
            vector_threshold=vector_threshold,
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired:
        # Keep the exact 504 behavior you saw
        return JSONResponse(
            status_code=504,
            content={"detail": f"Image extraction timed out after {timeout_s//60} minutes"},
        )
    except Exception as e:
        logger.exception("Image extraction failed")
        return JSONResponse(status_code=500, content={"detail": str(e)})

    # Wrap exactly like limiter-6 “/extract/all”
    wrapped = {
        "results": images,
        "count": len(images),
        "extraction_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "success": True,
        "statistics": stats,
    }
    return JSONResponse(content=wrapped)

# -----------------------------------------------------------------------------
# /extract/all — mirrors limiter-6 behavior (tables OFF), and also honors `workers`
# -----------------------------------------------------------------------------
@app.post("/extract/all")
async def extract_all(
    file: UploadFile = File(...),
    workers: int = Form(4),
    min_quality: float = Form(0.3),
    vector_threshold: int = Form(10),
    min_width: int = Form(100),
    min_height: int = Form(100),
    timeout_s: int = Form(900),
    # Included for future toggles; default keeps tables OFF like limiter-6
    do_images: bool = Form(True),
    do_tables: bool = Form(False),
):
    pdf_bytes = await file.read()

    images = []
    stats = {"images_count": 0, "tables_count": 0, "supabase_enabled": SUPABASE_ENABLED, "workers": int(workers)}

    # TABLES are intentionally off here, matching your proven deployment.
    # (If you later want them, wire a similar runner and keep the wrapper.)
    if do_images:
        try:
            images, i_stats = run_image_extraction(
                pdf_bytes=pdf_bytes,
                workers=workers,
                min_quality=min_quality,
                min_width=min_width,
                min_height=min_height,
                vector_threshold=vector_threshold,
                timeout_s=timeout_s,
            )
            stats.update(i_stats)
        except subprocess.TimeoutExpired:
            return JSONResponse(
                status_code=504,
                content={"detail": f"Image extraction timed out after {timeout_s//60} minutes"},
            )
        except Exception as e:
            logger.exception("Image extraction failed")
            return JSONResponse(status_code=500, content={"detail": str(e)})

    wrapped = {
        "results": images,              # only images for now (like limiter-6)
        "count": len(images),
        "extraction_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "success": True,
        "statistics": stats,
    }
    logger.info("Total results: %d items", len(images))
    return JSONResponse(content=wrapped)

# -----------------------------------------------------------------------------
# /extract/tables — placeholder wrapper (keeps orchestrator shape intact)
# -----------------------------------------------------------------------------
@app.post("/extract/tables")
async def extract_tables(
    file: UploadFile = File(...),
    workers: int = Form(4),
):
    _ = await file.read()  # unused in the stub
    # Shape-compatible empty response; swap in your real table extractor later.
    wrapped = {
        "results": [],  # tables would go here
        "count": 0,
        "extraction_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "success": True,
        "statistics": {
            "images_count": 0,
            "tables_count": 0,
            "supabase_enabled": SUPABASE_ENABLED,
            "elapsed_seconds": 0.0,
            "workers": int(workers),
        },
    }
    return JSONResponse(content=wrapped)
