"""  # <-- keep this header in your file
PDF Extraction API - Split Endpoints Version (Images-only & Tables-only)
========================================================================
GOAL:
- Run images extraction ONLY when /extract/images is called.
- Run tables extraction ONLY when /extract/tables is called.
- Keep the IMAGE extraction path 100% IDENTICAL to your proven "Limiter-6" build:
  * Supabase upload with PUT and headers
  * Multiple URL fields on success: supabase_url, url, image_url, uploaded_filename
  * Base64 fallback (image_base64 + base64_content)
  * Same timeout, same logging

- Tables endpoint returns legacy shape used by your n8n tables branch:
  { success, tables_count, tables: [...], statistics }
  Each table has csv_content and csv_base64.

NOTES:
- We accept extra form fields on each endpoint (e.g., table_timeout_s on /extract/images)
  and simply ignore those that are irrelevant, so your existing HTTP nodes don't need edits.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import shutil
import json
import base64
from typing import Optional, List, Dict, Any
import subprocess
import sys
from datetime import datetime
import logging
import requests
from pathlib import Path

# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Extraction API - Split Endpoints (Supabase images intact)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

security = HTTPBearer()
API_KEY = os.environ.get("API_KEY", "your-secret-api-key-change-this")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials.credentials

# --------------------------------------------------------------------------------------
# Supabase (IMAGE PATH ONLY — UNCHANGED from Limiter-6)
# --------------------------------------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_TOKEN = os.environ.get("SUPABASE_TOKEN")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "public-images")

def upload_image_to_supabase(image_path: Path, filename: str) -> Optional[str]:
    """
    UNCHANGED: Upload PNG to Supabase, return public URL or None on failure.
    """
    if not SUPABASE_URL or not SUPABASE_TOKEN:
        logger.warning("Supabase not configured - skipping upload")
        return None

    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()

        upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_TOKEN}",
            "Content-Type": "image/png",
            "x-upsert": "true",
            "Cache-Control": "public, max-age=3600, immutable"
        }
        resp = requests.put(upload_url, data=image_data, headers=headers, timeout=30)
        if resp.status_code in (200, 201):
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
            logger.info(f"✅ Uploaded {filename} to Supabase")
            return public_url
        else:
            logger.error(f"❌ Supabase upload failed {resp.status_code}: {resp.text}")
            return None
    except Exception as e:
        logger.error(f"❌ Supabase upload error for {filename}: {e}")
        return None

def _limit_from(query_value: Optional[int], env_key: str) -> int:
    """
    Resolve limiter value:
      - prefer query param if > 0
      - else env var if > 0
      - else 0 (disabled)
    """
    try:
        if query_value is not None:
            v = int(query_value)
            if v > 0:
                return v
    except Exception:
        pass
    try:
        env_val = os.environ.get(env_key)
        if env_val:
            v = int(env_val)
            if v > 0:
                return v
    except Exception:
        pass
    return 0

# --------------------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------------------
@app.get("/")
async def health():
    return {
        "status": "healthy",
        "service": "PDF Extraction API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_TOKEN),
    }

# --------------------------------------------------------------------------------------
# /extract/images  — IMAGE extraction ONLY (Supabase logic intact)
# --------------------------------------------------------------------------------------
@app.post("/extract/images")
async def extract_images_only(
    file: UploadFile = File(...),
    # Image extractor knobs (kept identical)
    min_quality: float = Form(0.3),
    workers: int = Form(8),
    min_width: int = Form(100),
    min_height: int = Form(100),
    page_limit: Optional[int] = Form(None),
    # Accept but ignore table fields so existing nodes don't break:
    table_timeout_s: Optional[int] = Form(None),
    no_verification: Optional[bool] = Form(None),
    # Token
    token: str = Depends(verify_token),
):
    """
    Runs ONLY enterprise_image_extractor.py and packages results.
    IMAGE PATH IS KEPT IDENTICAL to Limiter-6 (Supabase upload + fallback).
    Returns legacy image shape:
      { success, images_count, images: [...], statistics }
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Save uploaded file
        pdf_path = os.path.join(temp_dir, "input.pdf")
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        images_dir = os.path.join(temp_dir, "pdf_images")
        os.makedirs(images_dir, exist_ok=True)

        # Build image extractor command (UNCHANGED args)
        image_cmd = [
            sys.executable,
            "enterprise_image_extractor.py",
            pdf_path,
            "--output-dir", images_dir,
            "--workers", str(workers),
            "--min-width", str(min_width),
            "--min-height", str(min_height),
            "--min-quality", str(min_quality),
            "--vector-threshold", "10",
            "--clear-output",
        ]
        if page_limit:
            image_cmd.extend(["--page-limit", str(page_limit)])

        logger.info(f"[IMAGES] Running: {' '.join(image_cmd)} (timeout=900s)")
        try:
            image_result = subprocess.run(
                image_cmd, capture_output=True, text=True, timeout=900  # 15 minutes
            )
            logger.info(f"[IMAGES] exit code: {image_result.returncode}")
            if image_result.stderr:
                logger.error(f"[IMAGES] stderr: {image_result.stderr[:2000]}")

            images: List[Dict[str, Any]] = []
            image_metadata: Dict[str, Any] = {}
            if image_result.returncode == 0:
                # Collect generated PNGs
                image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".png")]
                image_files.sort()

                # Read metadata if present
                meta_path = os.path.join(images_dir, "extraction_metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as mf:
                        image_metadata = json.load(mf)

                # Map filename -> metadata
                meta_map = {img.get("filename"): img for img in image_metadata.get("images", []) if isinstance(img, dict)}

                # Package each image (Supabase first, else base64)
                for img_file in image_files:
                    img_path = os.path.join(images_dir, img_file)

                    # unique filename to avoid collisions
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_filename = f"{timestamp}_{img_file}"

                    supabase_url = upload_image_to_supabase(Path(img_path), unique_filename)

                    info = meta_map.get(img_file, {})
                    item = {
                        "filename": img_file,
                        "page_number": info.get("page_number", 0),
                        "image_index": info.get("image_index", 0),
                        "image_type": info.get("image_type", "general_image"),
                        "quality_score": info.get("quality_score", 0.0),
                        "width": info.get("width", 0),
                        "height": info.get("height", 0),
                        "has_text": info.get("has_text", False),
                        "text_content": info.get("text_content", ""),
                        "metadata": info,
                    }

                    if supabase_url:
                        item["supabase_url"] = supabase_url
                        item["url"] = supabase_url
                        item["image_url"] = supabase_url
                        item["uploaded_filename"] = unique_filename
                        logger.info(f"✅ Image {img_file} uploaded to Supabase")
                    else:
                        with open(img_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        item["image_base64"] = b64
                        item["base64_content"] = b64
                        logger.info(f"⚠️ Image {img_file} using base64 fallback")

                    images.append(item)

            else:
                logger.warning("[IMAGES] extractor returned non-zero code")

            return {
                "success": True,
                "images_count": len(images),
                "images": images,
                "statistics": image_metadata.get("statistics", {}),
            }

        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Image extraction timed out after 15 minutes")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[IMAGES] unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction error: {e}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# --------------------------------------------------------------------------------------
# /extract/tables — TABLE extraction ONLY (legacy shape, csv_content included)
# --------------------------------------------------------------------------------------
@app.post("/extract/tables")
async def extract_tables_only(
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    workers: int = Form(4),
    page_limit: Optional[int] = Form(None),
    no_verification: bool = Form(False),
    table_timeout_s: int = Form(900),
    # Accept and ignore image-only fields so the same node body can be reused:
    min_width: Optional[int] = Form(None),
    min_height: Optional[int] = Form(None),
    # Token
    token: str = Depends(verify_token),
):
    """
    Runs ONLY enterprise_table_extractor_full.py and packages results.
    Returns legacy table shape expected by your n8n branch:
      { success, tables_count, tables: [...], statistics }
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Save uploaded file
        pdf_path = os.path.join(temp_dir, "input.pdf")
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        tables_dir = os.path.join(temp_dir, "pdf_tables")
        os.makedirs(tables_dir, exist_ok=True)

        table_cmd = [
            sys.executable,
            "enterprise_table_extractor_full.py",
            pdf_path,
            "--output-dir", tables_dir,
            "--workers", str(workers),
            "--min-quality", str(min_quality),
            "--clear-output",
        ]
        if page_limit:
            table_cmd.extend(["--page-limit", str(page_limit)])
        if no_verification:
            table_cmd.append("--no-verification")

        logger.info(f"[TABLES] Running: {' '.join(table_cmd)} (timeout={table_timeout_s}s)")
        try:
            table_result = subprocess.run(
                table_cmd, capture_output=True, text=True, timeout=table_timeout_s
            )
            logger.info(f"[TABLES] exit code: {table_result.returncode}")
            if table_result.stderr:
                logger.error(f"[TABLES] stderr: {table_result.stderr[:2000]}")

            if table_result.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Extraction failed: {table_result.stderr[:8000]}")

            # Metadata
            meta_path = os.path.join(tables_dir, "extraction_metadata.json")
            if not os.path.exists(meta_path):
                raise HTTPException(status_code=500, detail="No metadata file generated")

            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)

            tables: List[Dict[str, Any]] = []
            for t in meta.get("tables", []):
                item = {
                    "filename": t.get("filename"),
                    "page_number": t.get("page_number", 0),
                    "table_index": t.get("table_index", 0),
                    "table_type": t.get("table_type", "general_data"),
                    "quality_score": t.get("quality_score", 0.0),
                    "rows": t.get("rows", 0),
                    "columns": t.get("columns", 0),
                    "metadata": t.get("metadata", {}),
                }
                csv_path = os.path.join(tables_dir, t.get("filename", ""))
                csv_text, csv_b64 = "", ""
                if os.path.exists(csv_path):
                    try:
                        with open(csv_path, "r", encoding="utf-8") as cf:
                            csv_text = cf.read()
                    except Exception:
                        csv_text = ""
                    try:
                        with open(csv_path, "rb") as cf2:
                            csv_b64 = base64.b64encode(cf2.read()).decode("utf-8")
                    except Exception:
                        csv_b64 = ""
                item["csv_content"] = csv_text
                item["csv_base64"] = csv_b64
                tables.append(item)

            return {
                "success": True,
                "tables_count": len(tables),
                "tables": tables,
                "statistics": meta.get("statistics", {}),
            }

        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail=f"Table extraction timed out after {table_timeout_s} seconds")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TABLES] unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction error: {e}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# --------------------------------------------------------------------------------------
# Debug
# --------------------------------------------------------------------------------------
@app.get("/debug/check-environment")
async def check_environment():
    checks = {
        "python_version": sys.version,
        "current_directory": os.getcwd(),
        "scripts_exist": {
            "table_extractor": os.path.exists("enterprise_table_extractor_full.py"),
            "image_extractor": os.path.exists("enterprise_image_extractor.py"),
        },
    }
    # Quick imports
    required = [
        "pdfplumber", "pandas", "numpy", "camelot", "tabula", "fitz", "PIL", "cv2"
    ]
    checks["imports"] = {}
    for name in required:
        try:
            __import__(name if name not in ("fitz","PIL") else ("fitz" if name=="fitz" else "PIL.Image"))
            checks["imports"][name] = True
        except Exception:
            checks["imports"][name] = False

    # System deps
    checks["system"] = {
        "java_available": shutil.which("java") is not None,
        "tesseract_available": shutil.which("tesseract") is not None,
    }

    return checks

@app.get("/test")
async def test_endpoint():
    return {
        "message": "API is working!",
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
