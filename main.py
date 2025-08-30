# main.py
#
# FastAPI app with three extraction endpoints:
#   - POST /extract/images  -> runs image extraction ONLY (unchanged from your working limiter-6 path)
#   - POST /extract/all     -> mirrors limiter-6 (images only by default), honors form-driven `workers`
#   - POST /extract/tables  -> NEW: real tables runner wired in, writes CSVs to /data/pdf_tables, wrapper shape
#
# Notes:
# - Image extraction logic and response fields are IDENTICAL to your proven limiter-6 path.
# - Tables return items with: type="table", fileName, filePath (CSV), plus optional csv_base64 when requested.
# - `workers` comes from the HTTP form everywhere (default 4).
# - Supabase is used for IMAGES only, exactly as before.

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
from typing import Optional

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
# Environment / Supabase (images only)
# -----------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "public-images")
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)

def supabase_public_url(object_path: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/object/public/{object_path}"

def upload_to_supabase(local_path: str, object_name: str) -> Optional[str]:
    if not SUPABASE_ENABLED:
        return None
    object_path = f"{SUPABASE_BUCKET}/{object_name}"
    put_url = f"{SUPABASE_URL}/storage/v1/object/{object_path}"
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
    logger.error("❌ Supabase upload failed (%s): %s", resp.status_code, resp.text)
    return None

# -----------------------------------------------------------------------------
# Shared constants
# -----------------------------------------------------------------------------
TABLE_OUTPUT_BASE = os.getenv("TABLE_OUTPUT_BASE", "/data/pdf_tables")

# -----------------------------------------------------------------------------
# FastAPI app & health
# -----------------------------------------------------------------------------
app = FastAPI(title="PDF Extraction API")

@app.get("/", response_class=PlainTextResponse)
def root():
    return "PDF Extraction API is running"

# -----------------------------------------------------------------------------
# Image extraction block (unchanged, mirrors your limiter-6 path)
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
    t0 = time.time()
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "input.pdf")
        images_dir = os.path.join(temp_dir, "pdf_images")
        os.makedirs(images_dir, exist_ok=True)

        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        logger.info("[LIMITERS DISABLED] Processing all images")
        logger.info(
            "Processing PDF: %s, Size: %d bytes, Temp path: %s",
            os.path.basename(pdf_path),
            len(pdf_bytes),
            pdf_path,
        )
        logger.info("📋 Table extraction is DISABLED")
        logger.info("🖼️ Extracting images...")

        cmd = [
            sys.executable,
            "enterprise_image_extractor.py",
            pdf_path,
            "--output-dir", images_dir,
            "--workers", str(int(workers)),
            "--min-width", str(int(min_width)),
            "--min-height", str(int(min_height)),
            "--min-quality", str(float(min_quality)),
            "--vector-threshold", str(int(vector_threshold)),
            "--clear-output",
        ]
        logger.info("Running command: %s (timeout=%ss)", " ".join(cmd), timeout_s)

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)

        logger.info("Image extraction exit code: %s", proc.returncode)
        stdout_preview = (proc.stdout or "")[:500]
        logger.info("Image stdout (first 500 chars): %s", stdout_preview if stdout_preview else "(empty)")
        if proc.stderr:
            logger.error("Image stderr: %s", proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"Image extractor failed: {proc.stderr or proc.stdout or 'unknown error'}")

        # metadata associations (best-effort)
        meta_map = {}
        for meta_name in ("extraction_metadata.json", "metadata.json"):
            mp = os.path.join(images_dir, meta_name)
            if os.path.exists(mp):
                try:
                    with open(mp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for m in data.get("images", []):
                        key = m.get("filename") or m.get("file_name") or m.get("name")
                        if key:
                            meta_map[key] = m
                except Exception:
                    pass

        images = []
        timestamp_prefix = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        png_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        logger.info("Processing all %d images", len(png_files))
        logger.info("Found %d images in metadata (packaging %d)", len(meta_map), len(png_files))

        for img_path in png_files:
            filename = os.path.basename(img_path)
            unique_name = f"{timestamp_prefix}_{filename}"
            supa_url = upload_to_supabase(img_path, unique_name) if SUPABASE_ENABLED else None

            meta = meta_map.get(filename, {})
            page = meta.get("page") or meta.get("page_number")
            width = meta.get("width")
            height = meta.get("height")
            img_type = meta.get("type") or meta.get("image_type")
            method = meta.get("method") or ("vector" if "vector" in filename else "embedded")
            quality = meta.get("quality")
            ocr_text = meta.get("text") or meta.get("ocr_text")
            ocr_success = meta.get("ocr_success")

            item = {
                "type": "image",
                "page": page,
                "image_type": img_type,
                "method": method,
                "quality": quality,
                "width": width,
                "height": height,
                "fileName": filename,
                "filename": filename,
                "filePath": img_path,
                "supabase_url": supa_url,
                "url": supa_url,
                "image_url": supa_url,
                "text": ocr_text,
                "ocr_success": ocr_success,
                "mime_type": "image/png",
            }
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
# NEW: Tables extraction block
# -----------------------------------------------------------------------------
def run_table_extraction(
    pdf_bytes: bytes,
    workers: int = 4,
    timeout_s: int = 900,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    page_max: Optional[int] = None,
    no_verification: bool = False,
    include_csv_base64: bool = False,
):
    """
    Runs enterprise_table_extractor.py, writes CSVs to TABLE_OUTPUT_BASE, and returns (results, stats).
    Items have: type="table", fileName, filePath (CSV). Optional csv_base64 if requested.
    """
    os.makedirs(TABLE_OUTPUT_BASE, exist_ok=True)
    t0 = time.time()

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "input.pdf")
        tables_dir = os.path.join(temp_dir, "pdf_tables")
        os.makedirs(tables_dir, exist_ok=True)

        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        cmd = [
            sys.executable,
            "enterprise_table_extractor.py",
            pdf_path,
            "--output-dir", tables_dir,
            "--workers", str(int(workers)),
            "--clear-output",
        ]
        if page_start is not None:
            cmd += ["--page-start", str(int(page_start))]
        if page_end is not None:
            cmd += ["--page-end", str(int(page_end))]
        if page_max is not None:
            cmd += ["--page-max", str(int(page_max))]
        if no_verification:
            cmd += ["--no-verification"]

        logger.info("[TABLES] Running: %s (timeout=%ss)", " ".join(cmd), timeout_s)

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)

        logger.info("Table extraction exit code: %s", proc.returncode)
        stdout_preview = (proc.stdout or "")[:500]
        logger.info("Table stdout (first 500 chars): %s", stdout_preview if stdout_preview else "(empty)")
        if proc.stderr:
            logger.error("Table stderr: %s", proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"Table extractor failed: {proc.stderr or proc.stdout or 'unknown error'}")

        # Collect CSVs and move them into /data/pdf_tables with unique prefix
        items = []
        ts_prefix = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        csv_files = sorted(glob.glob(os.path.join(tables_dir, "*.csv")))
        logger.info("Packaging %d CSV tables", len(csv_files))

        for src_path in csv_files:
            base = os.path.basename(src_path)
            safe_name = f"{ts_prefix}_{base}"
            dest_path = os.path.join(TABLE_OUTPUT_BASE, safe_name)

            # Move to shared/stable path
            shutil.move(src_path, dest_path)

            item = {
                "type": "table",
                "fileName": safe_name,
                "filePath": dest_path,
                "mime_type": "text/csv",
                "delimiter": ",",
            }
            if include_csv_base64:
                with open(dest_path, "rb") as f:
                    item["csv_base64"] = base64.b64encode(f.read()).decode("utf-8")

            items.append(item)

        elapsed = time.time() - t0
        stats = {
            "images_count": 0,
            "tables_count": len(items),
            "supabase_enabled": SUPABASE_ENABLED,  # images only, but kept for a consistent stats shape
            "elapsed_seconds": round(elapsed, 2),
            "workers": int(workers),
        }
        return items, stats

# -----------------------------------------------------------------------------
# /extract/images — unchanged (uses proven image block)
# -----------------------------------------------------------------------------
@app.post("/extract/images")
async def extract_images(
    file: UploadFile = File(...),
    workers: int = Form(4),
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
        return JSONResponse(
            status_code=504,
            content={"detail": f"Image extraction timed out after {timeout_s//60} minutes"},
        )
    except Exception as e:
        logger.exception("Image extraction failed")
        return JSONResponse(status_code=500, content={"detail": str(e)})

    wrapped = {
        "results": images,
        "count": len(images),
        "extraction_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "success": True,
        "statistics": stats,
    }
    return JSONResponse(content=wrapped)

# -----------------------------------------------------------------------------
# /extract/all — mirrors limiter-6 (images only), honors workers from form
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
    do_images: bool = Form(True),
    do_tables: bool = Form(False),
):
    pdf_bytes = await file.read()

    images = []
    stats = {"images_count": 0, "tables_count": 0, "supabase_enabled": SUPABASE_ENABLED, "workers": int(workers)}

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
        "results": images,
        "count": len(images),
        "extraction_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "success": True,
        "statistics": stats,
    }
    logger.info("Total results: %d items", len(images))
    return JSONResponse(content=wrapped)

# -----------------------------------------------------------------------------
# /extract/tables — REAL runner wired in, wrapper shape
# -----------------------------------------------------------------------------
@app.post("/extract/tables")
async def extract_tables(
    file: UploadFile = File(...),
    workers: int = Form(4),
    table_timeout_s: int = Form(900),
    page_start: Optional[int] = Form(None),
    page_end: Optional[int] = Form(None),
    page_max: Optional[int] = Form(None),
    no_verification: bool = Form(False),
    include_csv_base64: bool = Form(False),
):
    pdf_bytes = await file.read()
    try:
        tables, stats = run_table_extraction(
            pdf_bytes=pdf_bytes,
            workers=workers,
            timeout_s=table_timeout_s,
            page_start=page_start,
            page_end=page_end,
            page_max=page_max,
            no_verification=no_verification,
            include_csv_base64=include_csv_base64,
        )
    except subprocess.TimeoutExpired:
        return JSONResponse(
            status_code=504,
            content={"detail": f"Table extraction timed out after {table_timeout_s//60} minutes"},
        )
    except Exception as e:
        logger.exception("Table extraction failed")
        return JSONResponse(status_code=500, content={"detail": str(e)})

    wrapped = {
        "results": tables,
        "count": len(tables),
        "extraction_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "success": True,
        "statistics": stats,
    }
    return JSONResponse(content=wrapped)
