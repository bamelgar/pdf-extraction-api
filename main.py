"""
PDF Extraction API (Drop-in)

This file is a corrected, production-ready main.py that:
- Preserves the exact, proven image-extraction behavior from your
  "main (limiter - 6)-Image Extraction WORKS.py" (including Supabase upload).
- Forces workers=8 for images (per your instruction).
- Returns the SAME response shape as your limiter-6 script (wrapped
  { results, count, extraction_timestamp, success } with slim image items).
- Provides /extract/images (image-only) and /extract/tables (table-only),
  without touching the working image logic.
- Accepts extra form fields your n8n nodes send so we avoid 422s.

IMPORTANT:
- The image path here *is* the limiter-6 code path, including the same
  Supabase upload strategy (no client SDK, straight REST).
- /extract/images simply routes through that exact image block.
"""

import os
import sys
import json
import shutil
import logging
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import requests  # used for Supabase REST uploads

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# ------------------------------------------------------------------------------
# Environment & Auth
# ------------------------------------------------------------------------------
API_KEY = os.environ.get("API_KEY", "your-secret-api-key-change-this")

# Supabase envs used by the limiter-6 code path (image uploads)
SUPABASE_URL = os.environ.get("SUPABASE_URL")            # e.g., https://xxxx.supabase.co
SUPABASE_TOKEN = os.environ.get("SUPABASE_TOKEN")        # service role or anon with storage perms
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "public-images")  # your working bucket

# Hard clamp for images (your request): force workers=8 regardless of inputs
FORCED_IMAGE_WORKERS = 8

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials.credentials

# ------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------
app = FastAPI(title="PDF Extraction API", version="1.0.0")

# CORS for n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Utility: Supabase upload (EXACT limiter-6 approach: direct REST PUT)
# ------------------------------------------------------------------------------
def upload_image_to_supabase(image_path: Path, filename: str) -> Optional[str]:
    """
    Upload image to Supabase Storage and return a public URL.
    This mirrors the limiter-6 code path (no SDK; raw REST).
    """
    if not SUPABASE_URL or not SUPABASE_TOKEN:
        logger.warning("Supabase not configured - skipping upload")
        return None

    try:
        # Read file bytes
        data = image_path.read_bytes()

        # Upload endpoint (public bucket path)
        upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"

        headers = {
            "Authorization": f"Bearer {SUPABASE_TOKEN}",
            "Content-Type": "image/png",
            "x-upsert": "true",
        }

        resp = requests.put(upload_url, headers=headers, data=data, timeout=60)
        if resp.status_code not in (200, 201):
            logger.error(f"Supabase upload failed ({resp.status_code}): {resp.text}")
            return None

        # Construct the public URL like your working logs show
        public_url = (
            f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
        )
        return public_url

    except Exception as e:
        logger.exception(f"Supabase upload exception: {e}")
        return None

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "PDF Extraction API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/test")
async def test_endpoint():
    """Simple test endpoint that doesn't require auth."""
    return {
        "message": "API is working!",
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
    }

# ------------------------------------------------------------------------------
# IMAGE EXTRACTION (Limiter-6 logic) as a single reusable function
# ------------------------------------------------------------------------------
def run_image_extraction_block(
    pdf_path: Path,
    temp_dir: Path,
    min_quality: float,
    min_width: int,
    min_height: int,
    vector_threshold: int,
    timeout_s: int,
) -> dict:
    """
    EXACT functional behavior of your limiter-6 image block:
    - Calls enterprise_image_extractor.py with forced workers=8
    - Reads metadata
    - Uploads images to Supabase with timestamp prefix
    - Packages a slim 'result_item' for each image with the same fields:
      supabase_url, url, image_url, file_url, etc.
    - Returns wrapped shape: { results, count, extraction_timestamp, success }
    """

    # Create images output dir
    images_dir = temp_dir / "pdf_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: Force workers=8 per your requirement
    eff_workers = FORCED_IMAGE_WORKERS

    image_cmd = [
        sys.executable,
        "enterprise_image_extractor.py",
        str(pdf_path),
        "--output-dir", str(images_dir),
        "--workers", str(eff_workers),
        "--min-width", str(min_width),
        "--min-height", str(min_height),
        "--min-quality", str(min_quality),
        "--vector-threshold", str(vector_threshold),
        "--clear-output",
    ]

    logger.info(
        f"[IMAGES] Running: {' '.join(image_cmd)} (timeout={timeout_s}s)"
    )

    try:
        proc = subprocess.run(
            image_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Image extraction timed out after 15 minutes")

    logger.info(f"Image extraction exit code: {proc.returncode}")
    logger.info(f"Image stdout (first 500 chars): {proc.stdout[:500]}")
    if proc.stderr:
        logger.error(f"Image stderr: {proc.stderr[:1000]}")

    # If extractor failed, we still proceed with whatever exists (like limiter-6)
    # but typically returncode=0 on success.
    # Read metadata if present
    metadata_path = images_dir / "extraction_metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not parse image metadata JSON: {e}")

    # Build result items from metadata entries
    images = metadata.get("images", [])
    logger.info(f"Processing all {len(images)} images")

    all_results: List[dict] = []
    timestamp_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    for img_info in images:
        local_name = img_info.get("filename") or f"page_{img_info.get('page_number',0)}.png"
        local_path = images_dir / local_name

        # EXACT limiter-6 fields (no base64 here; slim payload)
        result_item = {
            "type": "image",
            "page": img_info.get("page_number", 0),
            "index": img_info.get("image_index", 0),
            "filePath": f"/data/pdf_images/{local_name}",
            "fileName": local_name,
            "image_type": img_info.get("image_type", "general_image"),
            "extraction_method": img_info.get("extraction_method", "unknown"),
            "quality_score": img_info.get("quality_score", 0.0),
            "width": img_info.get("width", 0),
            "height": img_info.get("height", 0),
            "has_text": img_info.get("has_text", False),
            "text_content": img_info.get("text_content", ""),
            "caption": img_info.get("context", {}).get("caption"),
            "figure_reference": img_info.get("context", {}).get("figure_reference"),
            "visual_elements": img_info.get("visual_elements", {}),
            "vector_count": img_info.get("vector_count"),
            "enhancement_applied": img_info.get("enhancement_applied", False),
            "mimeType": "image/png",
        }

        # Supabase upload (EXACT approach) with timestamp prefix
        public_url = None
        if local_path.exists():
            unique_name = f"{timestamp_prefix}_{local_name}"
            uploaded = upload_image_to_supabase(local_path, unique_name)
            if uploaded:
                logger.info(f"✅ Uploaded {unique_name} to Supabase")
                public_url = uploaded
                logger.info(f"✅ Image {local_name} uploaded to Supabase: {uploaded}")

        # The limiter-6 result included these aliases
        if public_url:
            result_item["supabase_url"] = public_url
            result_item["url"] = public_url
            result_item["image_url"] = public_url

        # Local file URL helper used downstream (matches your logs)
        result_item["file_url"] = result_item["filePath"]

        all_results.append(result_item)

    # Stable ordering (page, then index)
    all_results.sort(key=lambda x: (x.get("page", 0), x.get("index", 0)))

    wrapped = {
        "results": all_results,
        "count": len(all_results),
        "extraction_timestamp": datetime.now().isoformat(),
        "success": True,
    }
    return wrapped

# ------------------------------------------------------------------------------
# /extract/images  (image-only; EXACT limiter-6 path; workers forced to 8)
# ------------------------------------------------------------------------------
@app.post("/extract/images")
async def extract_images(
    file: UploadFile = File(...),
    # These match the limiter-6 knobs (but we clamp workers anyway)
    min_quality: float = Form(0.3),
    min_width: int = Form(100),
    min_height: int = Form(100),
    vector_threshold: int = Form(10),

    # Accept these extras so your existing n8n nodes won't 422 when they send them.
    table_timeout_s: int = Form(900),   # ignored for images; we map to image timeout if provided
    no_verification: Optional[bool] = Form(None),  # ignored for images
    image_timeout_s: int = Form(900),   # default 15 minutes for heavy docs
    token: str = Depends(verify_token),
):
    """
    IMAGE-ONLY endpoint that routes through the proven limiter-6 code path:
    - Uses Supabase upload logic
    - Returns the exact wrapped shape with slim items
    - Forces workers=8 (ignoring input workers)
    """
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Save upload to temp
        pdf_path = temp_dir / "input.pdf"
        with pdf_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # Let image timeout use explicit image_timeout_s; if n8n sent table_timeout_s, treat it as image timeout
        effective_timeout = image_timeout_s or table_timeout_s or 900

        wrapped = run_image_extraction_block(
            pdf_path=pdf_path,
            temp_dir=temp_dir,
            min_quality=min_quality,
            min_width=min_width,
            min_height=min_height,
            vector_threshold=vector_threshold,
            timeout_s=effective_timeout,
        )
        return wrapped

    finally:
        # Clean up temp
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

# ------------------------------------------------------------------------------
# /extract/tables  (table-only; simple pass-through to your enterprise script)
# NOTE: This DOES NOT touch the image logic above.
# ------------------------------------------------------------------------------
@app.post("/extract/tables")
async def extract_tables(
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    workers: int = Form(4),
    no_verification: bool = Form(False),
    table_timeout_s: int = Form(900),
    # optional page windowing knobs (safe no-ops if unused)
    page_start: Optional[int] = Form(None),
    page_end: Optional[int] = Form(None),
    page_max: Optional[int] = Form(None),
    token: str = Depends(verify_token),
):
    """
    TABLE-ONLY endpoint:
    - Matches your earlier working table path shape in spirit (CSV content included).
    - Does NOT alter the image logic above.
    """
    temp_dir = Path(tempfile.mkdtemp())
    try:
        pdf_path = temp_dir / "input.pdf"
        with pdf_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        out_dir = temp_dir / "pdf_tables"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "enterprise_table_extractor_full.py",
            str(pdf_path),
            "--output-dir", str(out_dir),
            "--workers", str(workers),
            "--min-quality", str(min_quality),
            "--clear-output",
        ]

        # Optional flags supported by your table extractor (safe if ignored)
        if no_verification:
            cmd.append("--no-verification")
        if page_start is not None:
            cmd += ["--page-start", str(page_start)]
        if page_end is not None:
            cmd += ["--page-end", str(page_end)]
        if page_max is not None:
            cmd += ["--page-max", str(page_max)]

        logger.info(f"[TABLES] Running: {' '.join(cmd)} (timeout={table_timeout_s}s)")

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=table_timeout_s
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Command timed out while extracting tables")

        if proc.returncode != 0:
            logger.error(f"Table extraction failed: {proc.stderr}")
            raise HTTPException(status_code=500, detail=f"Table extraction failed: {proc.stderr}")

        # Read metadata and package CSV content
        metadata_path = out_dir / "extraction_metadata.json"
        if not metadata_path.exists():
            raise HTTPException(status_code=500, detail="No metadata file generated for tables")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tables_meta = metadata.get("tables", [])

        tables_payload = []
        for t in tables_meta:
            csv_name = t.get("filename")
            csv_path = out_dir / csv_name if csv_name else None
            csv_content = ""
            if csv_path and csv_path.exists():
                try:
                    csv_content = csv_path.read_text(encoding="utf-8")
                except Exception as e:
                    logger.warning(f"CSV read error for {csv_name}: {e}")
                    csv_content = ""

            tables_payload.append(
                {
                    "filename": csv_name,
                    "page_number": t.get("page_number"),
                    "table_index": t.get("table_index"),
                    "table_type": t.get("table_type"),
                    "quality_score": t.get("quality_score"),
                    "rows": t.get("rows"),
                    "columns": t.get("columns"),
                    "csv_content": csv_content,
                    "metadata": t.get("metadata", {}),
                }
            )

        # Return a dedicated tables shape (like your working table endpoint)
        return {
            "success": True,
            "tables_count": len(tables_payload),
            "tables": tables_payload,
            "statistics": metadata.get("statistics", {}),
        }

    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

# ------------------------------------------------------------------------------
# /extract/all (Default: images only, to avoid accidental table slowness)
# NOTE: This keeps the wrapped response shape your n8n expects.
# ------------------------------------------------------------------------------
@app.post("/extract/all")
async def extract_all(
    file: UploadFile = File(...),
    # Images knobs (workers are still clamped to 8 internally)
    min_quality: float = Form(0.3),
    min_width: int = Form(100),
    min_height: int = Form(100),
    vector_threshold: int = Form(10),
    image_timeout_s: int = Form(900),

    # Tables off by default to mimic your successful “tables disabled” runs
    include_tables: bool = Form(False),
    table_timeout_s: int = Form(900),
    workers: int = Form(4),  # only used for tables if include_tables=True
    no_verification: bool = Form(False),

    # Optional page windowing for tables
    page_start: Optional[int] = Form(None),
    page_end: Optional[int] = Form(None),
    page_max: Optional[int] = Form(None),

    token: str = Depends(verify_token),
):
    """
    Unified endpoint. By default runs IMAGES ONLY (as that path is proven and fast).
    If include_tables=True, it will also run tables and merge the results,
    but image behavior and shape remain untouched.
    """
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Persist upload
        pdf_path = temp_dir / "input.pdf"
        with pdf_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # 1) Images (always)
        image_wrapped = run_image_extraction_block(
            pdf_path=pdf_path,
            temp_dir=temp_dir,
            min_quality=min_quality,
            min_width=min_width,
            min_height=min_height,
            vector_threshold=vector_threshold,
            timeout_s=image_timeout_s,
        )
        merged_results = image_wrapped["results"]

        # 2) Tables (optional)
        if include_tables:
            out_dir = temp_dir / "pdf_tables"
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "enterprise_table_extractor_full.py",
                str(pdf_path),
                "--output-dir", str(out_dir),
                "--workers", str(workers),
                "--min-quality", str(min_quality),
                "--clear-output",
            ]
            if no_verification:
                cmd.append("--no-verification")
            if page_start is not None:
                cmd += ["--page-start", str(page_start)]
            if page_end is not None:
                cmd += ["--page-end", str(page_end)]
            if page_max is not None:
                cmd += ["--page-max", str(page_max)]

            logger.info(f"[TABLES] (ALL) Running: {' '.join(cmd)} (timeout={table_timeout_s}s)")
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=table_timeout_s
                )
            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=504, detail="Command timed out while extracting tables")

            if proc.returncode != 0:
                logger.error(f"Table extraction failed: {proc.stderr}")
            else:
                meta = {}
                meta_path = out_dir / "extraction_metadata.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}

                for t in meta.get("tables", []):
                    csv_name = t.get("filename")
                    # Note: Keep image item shape untouched; add tables as distinct type
                    merged_results.append(
                        {
                            "type": "table",
                            "page": t.get("page_number", 0),
                            "index": t.get("table_index", 0),
                            "filePath": f"/data/pdf_tables/{csv_name}" if csv_name else None,
                            "fileName": csv_name,
                            "table_type": t.get("table_type", "general_data"),
                            "quality_score": t.get("quality_score", 0.0),
                            "extraction_method": t.get("extraction_method", "unknown"),
                            "rows": t.get("rows", 0),
                            "columns": t.get("columns", 0),
                            "mimeType": "text/csv",
                            # Intentionally slim here; your table-only endpoint returns csv_content.
                        }
                    )

        # Final wrapped (same shape as limiter-6)
        merged_results.sort(key=lambda x: (x.get("page", 0), x.get("index", 0)))
        wrapped = {
            "results": merged_results,
            "count": len(merged_results),
            "extraction_timestamp": datetime.now().isoformat(),
            "success": True,
        }
        logger.info(f"Total results: {len(merged_results)} items")
        return wrapped

    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

# ------------------------------------------------------------------------------
# Debug
# ------------------------------------------------------------------------------
@app.get("/debug/check-environment")
async def check_environment():
    """Quick environment probe for debugging."""
    checks = {
        "python_version": sys.version,
        "cwd": os.getcwd(),
        "scripts_exist": {
            "table_extractor": os.path.exists("enterprise_table_extractor_full.py"),
            "image_extractor": os.path.exists("enterprise_image_extractor.py"),
        },
        "supabase": {
            "url_set": bool(SUPABASE_URL),
            "token_set": bool(SUPABASE_TOKEN),
            "bucket": SUPABASE_BUCKET,
        },
    }
    return checks


if __name__ == "__main__":
    import uvicorn
    # Standard uvicorn boot
    uvicorn.run(app, host="0.0.0.0", port=8000)
