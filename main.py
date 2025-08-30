# main.py
#
# PDF Extraction API with:
# - Image endpoints that ALWAYS use the proven Supabase-uploading block
# - /extract/all keeps "tables OFF" behavior (like limiter-6) but still uploads images to Supabase
# - /extract/images reuses the same block as /extract/all (no base64 unless upload/env fails)
# - /extract/tables calls enterprise_table_extractor_full.py (exact filename)
# - Single source of truth for "workers" (HTTP form); default=4
#
# Env:
#   API_KEY (defaults to 'rockwood-reiss-api-2024-secure' for convenience)
#   SUPABASE_URL, SUPABASE_TOKEN, SUPABASE_BUCKET (default 'public-images')

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import os, tempfile, shutil, json, base64, subprocess, sys, logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# ---------- App & CORS ----------
app = FastAPI(title="PDF Extraction API (Supabase intact)", version="2025-08-30")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ---------- Auth ----------
security = HTTPBearer()
API_KEY = os.environ.get("API_KEY", "rockwood-reiss-api-2024-secure")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return True

# ---------- Supabase ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_TOKEN = os.environ.get("SUPABASE_TOKEN")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "public-images")

def upload_image_to_supabase(image_path: Path, filename: str) -> Optional[str]:
    """
    Upload image to Supabase Storage and return the PUBLIC URL.
    Matches the working limiter-6 behavior (headers, upsert, path, public URL).
    """
    if not SUPABASE_URL or not SUPABASE_TOKEN:
        logger.warning("Supabase not configured - skipping upload")
        return None

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_TOKEN}",
            "Content-Type": "image/png",
            "x-upsert": "true",
            "Cache-Control": "public, max-age=3600, immutable",
        }
        r = requests.put(upload_url, data=image_data, headers=headers, timeout=30)
        if r.status_code in (200, 201):
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
            logger.info(f"✅ Uploaded {filename} to Supabase")
            return public_url
        logger.error(f"❌ Supabase upload failed {filename}: {r.status_code} - {r.text}")
        return None
    except Exception as e:
        logger.error(f"❌ Supabase upload error for {filename}: {e}")
        return None

# ---------- Helpers ----------
def _limit_from(query_value: Optional[int], env_key: str) -> int:
    try:
        if query_value is not None:
            v = int(query_value)
            return v if v > 0 else 0
    except Exception:
        pass
    try:
        env_v = os.environ.get(env_key)
        if env_v:
            v = int(env_v)
            return v if v > 0 else 0
    except Exception:
        pass
    return 0

def _save_upload_to_temp(upload: UploadFile) -> (str, str):
    tmp = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp, "input.pdf")
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    logger.info(f"Processing PDF: {upload.filename}, Size: {os.path.getsize(pdf_path)} bytes, Temp path: {pdf_path}")
    return tmp, pdf_path

def _run_image_extractor(pdf_path: str, out_dir: str, workers: int, min_quality: float,
                         min_width: int, min_height: int, page_limit: Optional[int]) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "enterprise_image_extractor.py", pdf_path,
        "--output-dir", out_dir,
        "--workers", str(workers),
        "--min-width", str(min_width),
        "--min-height", str(min_height),
        "--min-quality", str(min_quality),
        "--vector-threshold", "10",
        "--clear-output",
    ]
    if page_limit:
        cmd.extend(["--page-limit", str(page_limit)])
    logger.info(f"[IMAGES] Running: {' '.join(cmd)} (timeout=900s)")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900)

def _package_images_with_supabase(images_dir: str, eff_limit_images: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".png")]
    image_files.sort()

    if eff_limit_images > 0:
        image_files = image_files[:eff_limit_images]
        logger.info(f"Limiting to {eff_limit_images} images")
    else:
        logger.info(f"Processing all {len(image_files)} images")

    # Map metadata by filename (if present)
    meta_map: Dict[str, Dict[str, Any]] = {}
    meta_path = os.path.join(images_dir, "extraction_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f) or {}
            for m in meta.get("images", []):
                fn = m.get("filename")
                if fn:
                    meta_map[fn] = m
        logger.info(f"Found {len(meta_map)} images in metadata (packaging {len(image_files)})")

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_name = f"{ts}_{img_file}"
        supa = upload_image_to_supabase(Path(img_path), unique_name)

        info = meta_map.get(img_file, {})
        item = {
            "type": "image",
            "page": info.get("page_number", 0),
            "index": info.get("image_index", 0),
            "filePath": f"/data/pdf_images/{img_file}",
            "fileName": img_file,
            "image_type": info.get("image_type", "general_image"),
            "extraction_method": info.get("extraction_method", "unknown"),
            "quality_score": info.get("quality_score", 0.0),
            "width": info.get("width", 0),
            "height": info.get("height", 0),
            "has_text": info.get("has_text", False),
            "text_content": info.get("text_content", ""),
            "caption": (info.get("context") or {}).get("caption"),
            "figure_reference": (info.get("context") or {}).get("figure_reference"),
            "visual_elements": info.get("visual_elements", {}),
            "vector_count": info.get("vector_count"),
            "enhancement_applied": info.get("enhancement_applied", False),
            "mimeType": "image/png",
        }

        if supa:
            item["supabase_url"] = supa
            item["url"] = supa
            item["image_url"] = supa
            item["uploaded_filename"] = unique_name
            logger.info(f"✅ Image {img_file} uploaded to Supabase: {supa}")
        else:
            # Fallback ONLY if Supabase upload/env fails — same as limiter-6.
            with open(img_path, "rb") as f:
                item["base64_content"] = base64.b64encode(f.read()).decode("utf-8")
            logger.info(f"⚠️ Image {img_file} using base64 fallback")

        results.append(item)

    return results

def _wrap_results(items: List[Dict[str, Any]], page_limit: Optional[int]) -> Dict[str, Any]:
    items.sort(key=lambda x: (x.get("page", 0), x.get("index", 0)))
    tcount = sum(1 for i in items if i.get("type") == "table")
    icount = sum(1 for i in items if i.get("type") == "image")
    return {
        "results": items,
        "count": len(items),
        "tables_count": tcount,
        "images_count": icount,
        "extraction_timestamp": datetime.now().isoformat(),
        "success": True,
        "supabase_enabled": bool(SUPABASE_URL and SUPABASE_TOKEN),
        "page_limit": page_limit,
    }

# ---------- Routes ----------
@app.get("/")
async def health():
    return {
        "status": "ok",
        "service": "PDF Extraction API",
        "version": "2025-08-30",
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_TOKEN),
    }

# ========== IMAGES ONLY (uses EXACT same Supabase packaging as /extract/all) ==========
@app.post("/extract/images")
async def extract_images_only(
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    workers: int = Form(4),
    min_width: int = Form(100),
    min_height: int = Form(100),
    page_limit: Optional[int] = Form(None),
    limit_images: Optional[int] = Form(None),
    token: bool = Depends(verify_token),
):
    temp_dir, pdf_path = _save_upload_to_temp(file)
    images_dir = os.path.join(temp_dir, "pdf_images")
    os.makedirs(images_dir, exist_ok=True)

    eff_limit_images = _limit_from(limit_images, "TEMP_LIMIT_IMAGES")
    try:
        proc = _run_image_extractor(pdf_path, images_dir, workers, min_quality, min_width, min_height, page_limit)
        logger.info(f"Image extraction exit code: {proc.returncode}")
        logger.info(f"Image stdout (first 500 chars): {proc.stdout[:500]}")
        if proc.stderr:
            logger.error(f"Image stderr: {proc.stderr[:1000]}")
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail="Image extractor failed")

        items = _package_images_with_supabase(images_dir, eff_limit_images)
        wrapped = _wrap_results(items, page_limit)
        return wrapped
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Image extraction timed out after 15 minutes")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ========== TABLES ONLY (calls enterprise_table_extractor_full.py) ==========
@app.post("/extract/tables")
async def extract_tables_only(
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    workers: int = Form(4),
    page_start: Optional[int] = Form(None),
    page_end: Optional[int] = Form(None),
    page_max: Optional[int] = Form(None),
    no_verification: bool = Form(False),
    table_timeout_s: int = Form(900),
    limit_tables: Optional[int] = Form(None),
    token: bool = Depends(verify_token),
):
    temp_dir, pdf_path = _save_upload_to_temp(file)
    tables_dir = os.path.join(temp_dir, "pdf_tables")
    os.makedirs(tables_dir, exist_ok=True)

    eff_limit_tables = _limit_from(limit_tables, "TEMP_LIMIT_TABLES")

    cmd = [
        sys.executable, "enterprise_table_extractor_full.py",  # REQUIRED name
        pdf_path,
        "--output-dir", tables_dir,
        "--workers", str(workers),
        "--min-quality", str(min_quality),
        "--clear-output",
    ]
    if no_verification:
        cmd.append("--no-verification")
    if page_start is not None:
        cmd.extend(["--page-start", str(page_start)])
    if page_end is not None:
        cmd.extend(["--page-end", str(page_end)])
    if page_max is not None:
        cmd.extend(["--page-max", str(page_max)])

    logger.info(f"[TABLES] Running: {' '.join(cmd)} (timeout={table_timeout_s}s)")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=table_timeout_s)
        logger.info(f"Table extraction exit code: {proc.returncode}")
        logger.info(f"Table stdout (first 500 chars): {proc.stdout[:500]}")
        if proc.stderr:
            logger.error(f"Table stderr: {proc.stderr[:1000]}")
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail="Table extractor failed")

        # Metadata (optional) — fall back to scanning CSVs if missing
        items: List[Dict[str, Any]] = []
        meta_path = os.path.join(tables_dir, "extraction_metadata.json")
        meta_tables: List[Dict[str, Any]] = []
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f) or {}
                meta_tables = meta.get("tables", [])

        csv_files = [f for f in os.listdir(tables_dir) if f.lower().endswith(".csv")]
        csv_files.sort()

        # If limiter requested, trim
        if eff_limit_tables > 0:
            csv_files = csv_files[:eff_limit_tables]
            logger.info(f"Limiting to {eff_limit_tables} tables")

        # Index meta by filename
        meta_by_name = {t.get("filename"): t for t in meta_tables if t.get("filename")}

        for csv_name in csv_files:
            m = meta_by_name.get(csv_name, {})
            item = {
                "type": "table",
                "page": m.get("page_number", 0),
                "index": m.get("table_index", 0),
                "filePath": f"/data/pdf_tables/{csv_name}",
                "fileName": csv_name,
                "table_type": m.get("table_type", "general_data"),
                "quality_score": m.get("quality_score", 0.0),
                "extraction_method": m.get("extraction_method", "unknown"),
                "rows": m.get("rows", 0),
                "columns": m.get("columns", 0),
                "has_headers": m.get("has_headers", True),
                "numeric_percentage": m.get("numeric_percentage", 0),
                "empty_cell_percentage": m.get("empty_cell_percentage", 0),
                "metadata": m.get("metadata", {}),
                "mimeType": "text/csv",
            }
            # Optionally embed CSV content (n8n can ignore if not needed)
            csv_path = os.path.join(tables_dir, csv_name)
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    item["csv_content"] = f.read()
            except Exception as e:
                logger.error(f"CSV read error {csv_name}: {e}")
                item["csv_content"] = ""

            items.append(item)

        return _wrap_results(items, page_limit=None)

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Table extraction timed out")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ========== ALL (tables OFF, images ON — same as your limiter-6 behavior) ==========
@app.post("/extract/all")
async def extract_all(
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    workers: int = Form(4),
    min_width: int = Form(100),
    min_height: int = Form(100),
    page_limit: Optional[int] = Form(None),
    limit_images: Optional[int] = Form(None),
    token: bool = Depends(verify_token),
):
    temp_dir, pdf_path = _save_upload_to_temp(file)
    images_dir = os.path.join(temp_dir, "pdf_images")
    os.makedirs(images_dir, exist_ok=True)

    eff_limit_images = _limit_from(limit_images, "TEMP_LIMIT_IMAGES")
    logger.info("📋 Table extraction is DISABLED")
    logger.info("🖼️ Extracting images...")

    try:
        proc = _run_image_extractor(pdf_path, images_dir, workers, min_quality, min_width, min_height, page_limit)
        logger.info(f"Image extraction exit code: {proc.returncode}")
        logger.info(f"Image stdout (first 500 chars): {proc.stdout[:500]}")
        if proc.stderr:
            logger.error(f"Image stderr: {proc.stderr[:1000]}")
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail="Image extractor failed")

        items = _package_images_with_supabase(images_dir, eff_limit_images)
        wrapped = _wrap_results(items, page_limit)
        logger.info(f"Total results: {wrapped['count']} items ({wrapped['tables_count']} tables, {wrapped['images_count']} images)")
        return wrapped
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Image extraction timed out after 15 minutes")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ---------- Debug ----------
@app.get("/debug/check-environment")
async def check_environment():
    return {
        "python_version": sys.version,
        "pwd": os.getcwd(),
        "scripts_exist": {
            "enterprise_table_extractor_full.py": os.path.exists("enterprise_table_extractor_full.py"),
            "enterprise_image_extractor.py": os.path.exists("enterprise_image_extractor.py"),
        },
        "supabase": {
            "url": bool(SUPABASE_URL),
            "token": bool(SUPABASE_TOKEN),
            "bucket": SUPABASE_BUCKET,
        },
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
