"""
PDF Extraction API - Production Version with Supabase (Wrappers + Table Patches)
================================================================================
BASELINE:
- This file preserves the IMAGE extraction portion EXACTLY as in your
  "Production Version with Supabase" main (Supabase uploads, multi-URL fields,
  base64 fallback, limits).  DO NOT CHANGE IMAGE SECTION.

ADDITIONS:
1) Wrapper endpoints restored:
   - /extract/images  -> returns { success, images_count, images, statistics }
   - /extract/tables  -> returns { success, tables_count, tables, statistics }

2) Table extractor knobs (no behavior change unless you use them):
   - no_verification: bool (passes --no-verification to enterprise_table_extractor_full.py)
   - table_timeout_s: int (default 1800s here to match /all table timeout)
   - page_limit: int (forwarded to extractor as --page-limit)

3) /extract/all unchanged in overall shape (wrapped { results, count, ... }),
   and its IMAGE leg remains IDENTICAL to your Supabase version.

NOTE:
- Supabase image upload logic, headers, and URL fields are kept 100% intact.
- Comments retained; variable names preserved where possible.
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="PDF Extraction API - Production Version with Supabase", version="1.0.0")

# Add CORS middleware for n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple auth
security = HTTPBearer()
API_KEY = os.environ.get("API_KEY", "your-secret-api-key-change-this")

# Supabase configuration - using environment variables for security
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_TOKEN = os.environ.get("SUPABASE_TOKEN")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "public-images")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials.credentials

def upload_image_to_supabase(image_path: Path, filename: str) -> str:
    """Upload image to Supabase Storage and return public URL (UNCHANGED)"""
    if not SUPABASE_URL or not SUPABASE_TOKEN:
        logger.warning("Supabase not configured - skipping upload")
        return None

    try:
        # Read image data
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # Upload URL
        upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"

        # Headers
        headers = {
            "Authorization": f"Bearer {SUPABASE_TOKEN}",
            "Content-Type": "image/png",
            "x-upsert": "true",
            "Cache-Control": "public, max-age=3600, immutable"
        }

        # Upload to Supabase (UNCHANGED: PUT + headers)
        response = requests.put(upload_url, data=image_data, headers=headers, timeout=30)

        if response.status_code in [200, 201]:
            # Return public URL
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
            logger.info(f"✅ Uploaded {filename} to Supabase")
            return public_url
        else:
            logger.error(f"❌ Failed to upload {filename}: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logger.error(f"❌ Error uploading {filename}: {e}")
        return None

def _limit_from(query_value: Optional[int], env_key: str) -> int:
    """
    TESTING LIMITER:
    - If a positive integer is provided via query param, use it.
    - Else if env var exists and is positive, use it.
    - Else 0 (no limit).
    """
    try:
        if query_value is not None:
            val = int(query_value)
            return val if val > 0 else 0
    except Exception:
        pass
    try:
        env_val = os.environ.get(env_key)
        if env_val:
            val = int(env_val)
            return val if val > 0 else 0
    except Exception:
        pass
    return 0  # Return 0 instead of None to indicate "no limit"

@app.get("/")
async def health_check():
    # Status indicators
    testing_indicators = {
        "TESTING_IMAGE_LIMIT": os.environ.get("TESTING_IMAGE_LIMIT", "0"),
        "TESTING_TABLE_LIMIT": os.environ.get("TESTING_TABLE_LIMIT", "0"),
        "TEMP_LIMIT_TABLES": os.environ.get("TEMP_LIMIT_TABLES", "0"),
        "TEMP_LIMIT_IMAGES": os.environ.get("TEMP_LIMIT_IMAGES", "0"),
        "SUPABASE_CONFIGURED": "YES" if (SUPABASE_URL and SUPABASE_TOKEN) else "NO",
        "TABLES_ENABLED": "YES - Full functionality with limiter version"
    }

    return {
        "status": "healthy - PRODUCTION MODE with Supabase",
        "service": "PDF Extraction API - Production Version",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "testing_limits": testing_indicators
    }

@app.post("/extract/test")
async def test_extraction(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """Test endpoint to verify file upload works"""
    return {
        "success": True,
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "File received successfully",
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_TOKEN)
    }

# -----------------------------------------------------------------------------
# /extract/tables (NEW) — preserves legacy tables shape used by your n8n nodes
# -----------------------------------------------------------------------------
@app.post("/extract/tables")
async def extract_tables_only(
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    workers: int = Form(4),
    # Optional patches (no behavior change unless set)
    page_limit: Optional[int] = Form(None),
    no_verification: bool = Form(False),
    table_timeout_s: int = Form(1800),  # 30 minutes to mirror /all table leg
    token: str = Depends(verify_token)
):
    """
    Extract tables using the existing enterprise_table_extractor_full.py and
    return the legacy tables shape:
      { success, tables_count, tables[], statistics }
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Save uploaded file
        pdf_path = os.path.join(temp_dir, "input.pdf")
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Output directory
        tables_dir = os.path.join(temp_dir, "pdf_tables")
        os.makedirs(tables_dir, exist_ok=True)

        # TABLE extraction command (matches /extract/all logic; adds optional flags)
        table_cmd = [
            sys.executable,
            "enterprise_table_extractor_full.py",
            pdf_path,
            "--output-dir", tables_dir,
            "--workers", str(workers),
            "--min-quality", str(min_quality),
            "--clear-output"
        ]
        if page_limit:
            table_cmd.extend(["--page-limit", str(page_limit)])
        if no_verification:
            table_cmd.append("--no-verification")

        logger.info(f"[TABLES] Running: {' '.join(table_cmd)} (timeout={table_timeout_s}s)")
        try:
            table_result = subprocess.run(
                table_cmd,
                capture_output=True,
                text=True,
                timeout=table_timeout_s
            )
            logger.info(f"[TABLES] exit code: {table_result.returncode}")
            if table_result.stderr:
                logger.error(f"[TABLES] stderr: {table_result.stderr[:2000]}")

            if table_result.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Extraction failed: {table_result.stderr[:8000]}")

            # Read table metadata
            metadata_path = os.path.join(tables_dir, "extraction_metadata.json")
            if not os.path.exists(metadata_path):
                raise HTTPException(status_code=500, detail="No metadata file generated")

            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            tables: List[Dict[str, Any]] = []
            for table_info in meta.get('tables', []):
                item = {
                    'filename': table_info.get('filename'),
                    'page_number': table_info.get('page_number', 0),
                    'table_index': table_info.get('table_index', 0),
                    'table_type': table_info.get('table_type', 'general_data'),
                    'quality_score': table_info.get('quality_score', 0.0),
                    'rows': table_info.get('rows', 0),
                    'columns': table_info.get('columns', 0),
                    'metadata': table_info.get('metadata', {})
                }
                # Attach CSV content + base64 (legacy expects both)
                csv_path = os.path.join(tables_dir, table_info['filename'])
                csv_text, csv_b64 = "", ""
                if os.path.exists(csv_path):
                    try:
                        with open(csv_path, 'r', encoding='utf-8') as f:
                            csv_text = f.read()
                    except Exception:
                        csv_text = ""
                    try:
                        with open(csv_path, 'rb') as f:
                            csv_b64 = base64.b64encode(f.read()).decode('utf-8')
                    except Exception:
                        csv_b64 = ""
                item['csv_content'] = csv_text
                item['csv_base64'] = csv_b64
                tables.append(item)

            return {
                'success': True,
                'tables_count': len(tables),
                'tables': tables,
                'statistics': meta.get('statistics', {})
            }

        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail=f"Extraction timed out after {table_timeout_s} seconds")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TABLES] Extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction error: {e}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# -----------------------------------------------------------------------------
# /extract/images (NEW) — IMAGE SECTION UNCHANGED; returns legacy images shape
# -----------------------------------------------------------------------------
@app.post("/extract/images")
async def extract_images_only(
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    workers: int = Form(4),
    min_width: int = Form(100),
    min_height: int = Form(100),
    # (Optional) limiter parity with /all
    limit_images: Optional[int] = Form(None),
    page_limit: Optional[int] = Form(None),
    token: str = Depends(verify_token)
):
    """
    Extract images using the EXACT SAME logic as your Supabase build.
    Returns legacy images shape:
      { success, images_count, images[], statistics }
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Save uploaded file
        pdf_path = os.path.join(temp_dir, "input.pdf")
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Create output directory
        images_dir = os.path.join(temp_dir, "pdf_images")
        os.makedirs(images_dir, exist_ok=True)

        # Resolve limiters (same helper)
        eff_limit_images = _limit_from(limit_images, "TEMP_LIMIT_IMAGES")

        # ================================
        # IMAGE EXTRACTION - WITH SUPABASE
        # (IDENTICAL to your /extract/all image leg)
        # ================================
        logger.info("🖼️ Extracting images...")
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
            "--clear-output"
        ]
        if page_limit:
            image_cmd.extend(["--page-limit", str(page_limit)])

        logger.info(f"Running command: {' '.join(image_cmd)}")

        try:
            image_result = subprocess.run(
                image_cmd,
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes timeout (UNCHANGED)
            )

            logger.info(f"Image extraction exit code: {image_result.returncode}")
            logger.info(f"Image stdout (first 500 chars): {image_result.stdout[:500]}")
            if image_result.stderr:
                logger.error(f"Image stderr: {image_result.stderr[:1000]}")

            images: List[Dict[str, Any]] = []

            if image_result.returncode == 0:
                # List files in output directory
                image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]
                image_files.sort()
                if eff_limit_images > 0:
                    image_files = image_files[:eff_limit_images]
                    logger.info(f"Limiting to {eff_limit_images} images")
                else:
                    logger.info(f"Processing all {len(image_files)} images")

                # Read image metadata
                image_metadata_path = os.path.join(images_dir, "extraction_metadata.json")
                image_metadata = {}
                if os.path.exists(image_metadata_path):
                    with open(image_metadata_path, 'r') as f:
                        image_metadata = json.load(f)

                logger.info(f"Found {len(image_metadata.get('images', []))} images in metadata (packaging {len(image_files)})")

                # Pre-index metadata for filename lookup
                meta_map = {}
                for img_info in image_metadata.get('images', []):
                    fn = img_info.get('filename')
                    if fn:
                        meta_map[fn] = img_info

                for img_file in image_files:
                    img_path = os.path.join(images_dir, img_file)

                    # Create unique filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_filename = f"{timestamp}_{img_file}"

                    # Try to upload to Supabase
                    supabase_url = upload_image_to_supabase(Path(img_path), unique_filename)

                    # Attach metadata if present
                    info = meta_map.get(img_file, {})
                    result_item = {
                        "filename": img_file,  # legacy images shape uses 'filename'
                        "page_number": info.get('page_number', 0),
                        "image_index": info.get('image_index', 0),
                        "image_type": info.get('image_type', 'general_image'),
                        "quality_score": info.get('quality_score', 0.0),
                        "width": info.get('width', 0),
                        "height": info.get('height', 0),
                        "has_text": info.get('has_text', False),
                        "text_content": info.get('text_content', ''),
                        "metadata": info
                    }

                    if supabase_url:
                        # Multiple URL fields preserved for RAG flexibility
                        result_item['supabase_url'] = supabase_url
                        result_item['url'] = supabase_url
                        result_item['image_url'] = supabase_url
                        result_item['uploaded_filename'] = unique_filename
                        logger.info(f"✅ Image {img_file} uploaded to Supabase: {supabase_url}")
                    else:
                        # Fallback to base64 (legacy key name: image_base64)
                        with open(img_path, 'rb') as f:
                            img_base64 = base64.b64encode(f.read()).decode('utf-8')
                        result_item['image_base64'] = img_base64
                        logger.info(f"⚠️ Image {img_file} using base64 fallback")

                    images.append(result_item)

            else:
                logger.warning("Image extractor returned non-zero code")

            return {
                'success': True,
                'images_count': len(images),
                'images': images,
                'statistics': image_metadata.get('statistics', {}) if 'image_metadata' in locals() else {}
            }

        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="Image extraction timed out after 15 minutes")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[IMAGES] Extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction error: {e}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# -----------------------------------------------------------------------------
# /extract/all — keeps wrapped shape; IMAGE leg is unchanged (Supabase upload)
# -----------------------------------------------------------------------------
@app.post("/extract/all")
async def extract_all(
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    workers: int = Form(4),
    min_width: int = Form(100),
    min_height: int = Form(100),
    # LIMITERS (optional)
    limit_tables: Optional[int] = Form(None),
    limit_images: Optional[int] = Form(None),
    page_limit: Optional[int] = Form(None),
    token: str = Depends(verify_token)
):
    """Extract both tables and images from PDF (wrapped response)."""
    temp_dir = tempfile.mkdtemp()

    # Resolve temp limits
    eff_limit_tables = _limit_from(limit_tables, "TEMP_LIMIT_TABLES")
    eff_limit_images = _limit_from(limit_images, "TEMP_LIMIT_IMAGES")

    if eff_limit_tables > 0:
        logger.info(f"[LIMIT] /all: Tables packaging limited to {eff_limit_tables}.")
    if eff_limit_images > 0:
        logger.info(f"[LIMIT] /all: Images packaging limited to {eff_limit_images}.")
    else:
        logger.info("[LIMITERS DISABLED] Processing all images and tables")

    try:
        # Save uploaded file
        pdf_path = os.path.join(temp_dir,
