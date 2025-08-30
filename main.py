"""
PDF Extraction API — Rockwood Reiss Edition (2025-08-30)

WHAT THIS IS
------------
Drop-in `main.py` designed specifically for your n8n workflow. It preserves the
**exact** image-extraction behavior (including Supabase uploads + URL return shape)
from the working "main (limiter - 6)-Image Extraction WORKS.py" while adding:
- Clean `/extract/images` endpoint that uses the **same, proven** image block
- `/extract/all` that keeps **tables OFF** (parity with limiter-6) and still uploads images to Supabase
- `/extract/tables` that calls **enterprise_table_extractor_full.py** (required filename)
- Single-source `workers` control via HTTP form (default=4 unless you override)

WHY THIS VERSION IS SUPERIOR
----------------------------
- **Supabase intact & primary**: PNGs are written to disk, uploaded to Supabase Storage
  via `PUT {SUPABASE_URL}/storage/v1/object/{bucket}/{timestamped_name}.png` with
  `x-upsert: true`. Response returns **URLs**, not binaries, keeping n8n light.
- **Identical response shape** as limiter-6 for images (fields like `supabase_url`,
  `url`, `image_url`, `uploaded_filename`, etc.). Base64 appears only as fallback when
  Supabase is not configured or an upload fails.
- **Consistent endpoints & shapes** your n8n nodes expect, with better logging and
  safer defaults (workers default to 4; timeouts guarded).

ENVIRONMENT VARIABLES
---------------------
- API_KEY                 : Bearer token for all routes (default 'rockwood-reiss-api-2024-secure')
- SUPABASE_URL            : Your Supabase project URL (required for uploads)
- SUPABASE_TOKEN          : Service key or bearer token (required for uploads)
- SUPABASE_BUCKET         : Storage bucket name (default: 'public-images')
- TEMP_LIMIT_IMAGES       : Optional int to cap number of packaged images (debugging)
- TEMP_LIMIT_TABLES       : Optional int to cap number of packaged tables (debugging)

ENDPOINTS (All require Authorization: Bearer <API_KEY>)
-------------------------------------------------------
GET  /                      -> health info (shows if Supabase is configured)
GET  /debug/check-environment
POST /extract/images        -> Image extraction ONLY (Supabase upload + URL response)
POST /extract/all           -> Tables OFF, Images ON (matches limiter-6 behavior)
POST /extract/tables        -> Tables ONLY (runs enterprise_table_extractor_full.py)

MULTIPART FORM FIELDS
---------------------
Common:
  - file (binary)          : PDF to process
  - min_quality (float)    : default 0.3
  - workers (int)          : default 4; you can override per request from n8n
Images (/extract/images, /extract/all) only:
  - min_width (int)        : default 100
  - min_height (int)       : default 100
  - page_limit (int?)      : optional, to stop early on huge PDFs
  - limit_images (int?)    : optional cap for packaging (or use TEMP_LIMIT_IMAGES)
Tables (/extract/tables) only:
  - page_start/page_end    : optional slice
  - page_max               : optional ceiling on pages processed
  - no_verification (bool) : passthrough to extractor
  - table_timeout_s (int)  : default 900s
  - limit_tables (int?)    : optional cap (or use TEMP_LIMIT_TABLES)

TIMEOUTS & CONCURRENCY GUIDANCE
-------------------------------
- Image extractor subprocess has a hard cap of 900s (15 minutes).
- Tables use the `table_timeout_s` you send (default 900s).
- Render's shared CPU can throttle; start with workers=4 (safe), 6 if you need more,
  and avoid 8+ unless you're sure the instance has headroom. Match your n8n HTTP
  node timeout to your largest expected run (20—30 min for 100+ page PDFs).

RESPONSE SHAPES (Stable for n8n)
--------------------------------
Images (both /extract/images and /extract/all):
{
  "results": [
    {
      "type": "image",
      "fileName": "...png",
      "filePath": "/data/pdf_images/...png",
      "supabase_url": "https://.../storage/v1/object/public/<bucket>/<timestamped_name>.png",
      "url":          "<same as supabase_url>",
      "image_url":    "<same as supabase_url>",
      "uploaded_filename": "<timestamped_name>.png",
      "page": <int>,
      "index": <int>,
      "image_type": "chart|diagram|logo|infographic|general_image",
      "extraction_method": "embedded|vector|unknown",
      "quality_score": <float>,
      "width": <int>, "height": <int>,
      "has_text": <bool>, "text_content": "...",
      "caption": "...", "figure_reference": "...",
      "vector_count": <int?>,
      "enhancement_applied": <bool>,
      "mimeType": "image/png"
      // NOTE: "base64_content" appears ONLY if Supabase is not configured or upload fails
    }
  ],
  "count": N,
  "tables_count": 0,
  "images_count": N,
  "extraction_timestamp": "ISO-8601",
  "success": true,
  "supabase_enabled": true|false,
  "page_limit": null|<int>
}

Tables (/extract/tables):
- Similar top-level wrapper, but each item has:
  type:"table", mimeType:"text/csv", filePath/fileName, and optional "csv_content" plus table metadata.

LOG LINES TO EXPECT
-------------------
- "[IMAGES] Running: ..." with the exact command line and timeout=900s
- "✅ Uploaded <timestamp>_<file>.png to Supabase" and a public URL
- "⚠️ Image <file>.png using base64 fallback" (only when upload/env fails)
- "/extract/all": "📋 Table extraction is DISABLED" then "🖼️ Extracting images..."

QUICK TESTS
-----------
curl -X POST "$HOST/extract/images" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@sample.pdf" -F "workers=4" -F "min_quality=0.3"

curl -X POST "$HOST/extract/all" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@sample.pdf" -F "workers=4"

curl -X POST "$HOST/extract/tables" \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@sample.pdf" -F "workers=4" -F "table_timeout_s=900"

TROUBLESHOOTING
---------------
- Large base64 blobs in n8n? -> Supabase env missing or upload failing; fix SUPABASE_URL/TOKEN.
- 504 from n8n while logs show extractor still running -> increase n8n node timeout or lower workers.
- Tables not running -> ensure "enterprise_table_extractor_full.py" is present in the working dir.
"""

# main.py
import os
import io
import csv
import json
import glob
import base64
import shutil
import logging
import tempfile
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Header
from fastapi.responses import JSONResponse, PlainTextResponse

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("main")

# ------------------------------------------------------------------------------
# Environment / Config
# ------------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY", "rockwood-reiss-api-2024-secure")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_TOKEN = os.getenv("SUPABASE_TOKEN", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "public-images")

# Optional development limiters
TEMP_LIMIT_IMAGES = os.getenv("TEMP_LIMIT_IMAGES")
TEMP_LIMIT_TABLES = os.getenv("TEMP_LIMIT_TABLES")

# External runner script names (FILENAMES MATTER)
IMAGE_EXTRACTOR = "enterprise_image_extractor.py"
TABLE_EXTRACTOR = "enterprise_table_extractor_full.py"  # <- exact name required

# Subprocess timeouts (seconds)
IMAGE_TIMEOUT_S = 900
TABLE_TIMEOUT_S_DEFAULT = 900

# ------------------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------------------
app = FastAPI(title="PDF Extraction API")

def bearer_auth(authorization: Optional[str] = Header(None)) -> None:
    """
    Simple Bearer token auth. Allows missing API_KEY in env to act as open gate,
    but if API_KEY is set, a matching Bearer token is required.
    """
    if not API_KEY:
        return  # open if key not configured
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default

def run_subprocess(cmd: List[str], timeout_s: int, cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """
    Run a subprocess, capture stdout/stderr text, and return (exit_code, stdout, stderr).
    """
    pretty = " ".join(cmd)
    log.info(f"Running: {pretty} (timeout={timeout_s}s)")
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
            text=True,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as te:
        return 124, te.stdout.decode("utf-8") if te.stdout else "", te.stderr.decode("utf-8") if te.stderr else ""

def supabase_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_TOKEN and SUPABASE_BUCKET)

def supabase_object_public_url(obj_path: str) -> str:
    # Public URL form: {SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{obj_path}"

def supabase_upload_bytes(data: bytes, dest_name: str, mime: str = "image/png") -> Tuple[bool, Optional[str]]:
    """
    Upload bytes to Supabase Storage (x-upsert true). Returns (ok, public_url or None).
    """
    if not supabase_enabled():
        return False, None

    # Endpoint: PUT /storage/v1/object/{bucket}/{path}
    put_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{dest_name}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_TOKEN}",
        "Content-Type": mime,
        "x-upsert": "true",
    }
    try:
        resp = requests.put(put_url, data=data, headers=headers, timeout=60)
        if resp.status_code in (200, 201):
            public_url = supabase_object_public_url(dest_name)
            return True, public_url
        else:
            log.error(f"Supabase upload failed ({resp.status_code}): {resp.text[:300]}")
            return False, None
    except Exception as e:
        log.error(f"Supabase upload exception: {e}")
        return False, None

def load_json_if_exists(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ------------------------------------------------------------------------------
# Image Packaging (IDENTICAL to limiter-6 behavior)
# ------------------------------------------------------------------------------
def package_images_with_supabase(output_dir: str, timestamp: str, limit_images: Optional[int]) -> List[Dict[str, Any]]:
    """
    Reads image metadata emitted by enterprise_image_extractor.py and for each PNG:
    - Upload to Supabase (mandatory path in normal operation),
    - Return items containing Supabase public URLs (no base64 unless fallback).
    """
    meta_path = os.path.join(output_dir, "extraction_metadata.json")
    meta = load_json_if_exists(meta_path)

    # Build list of image file entries
    entries: List[Dict[str, Any]] = []
    if isinstance(meta, dict) and isinstance(meta.get("images"), list):
        # Use extractor-provided metadata
        for m in meta["images"]:
            # Each m should at least contain filename and page; other fields optional
            fname = m.get("filename") or m.get("fileName")
            if not fname:
                continue
            entries.append({
                "fileName": fname,
                "page": m.get("page"),
                "index": m.get("index"),
                "image_type": m.get("image_type"),
                "extraction_method": m.get("extraction_method"),
                "quality_score": m.get("quality_score"),
                "width": m.get("width"),
                "height": m.get("height"),
                "has_text": m.get("has_text"),
                "text_content": m.get("text_content"),
                "caption": m.get("caption"),
                "figure_reference": m.get("figure_reference"),
                "visual_elements": m.get("visual_elements"),
                "vector_count": m.get("vector_count"),
            })
    else:
        # Fallback: scan for PNGs if metadata missing
        for path in sorted(glob.glob(os.path.join(output_dir, "*.png"))):
            fname = os.path.basename(path)
            entries.append({
                "fileName": fname,
                "page": None,
                "index": None,
                "image_type": None,
                "extraction_method": None,
                "quality_score": None,
                "width": None,
                "height": None,
                "has_text": None,
                "text_content": None,
                "caption": None,
                "figure_reference": None,
                "visual_elements": None,
                "vector_count": None,
            })

    # Enforce optional limiter
    if limit_images is not None:
        entries = entries[: max(0, int(limit_images))]

    log.info(f"Processing all {len(entries)} images")
    
    # Parallel upload function
    def upload_single_image(entry):
        fname = entry["fileName"]
        fpath = os.path.join(output_dir, fname)
        if not os.path.isfile(fpath):
            return None
        uploaded_name = f"{timestamp}_{fname}"
        with open(fpath, "rb") as f:
            data = f.read()
        supa_ok, supa_url = supabase_upload_bytes(data, uploaded_name, mime="image/png")
        return {
            "entry": entry,
            "fname": fname,
            "fpath": fpath,
            "uploaded_name": uploaded_name,
            "supa_ok": supa_ok,
            "supa_url": supa_url,
            "data": data if not supa_ok else None
        }
    
    # Parallel upload with ThreadPoolExecutor
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(upload_single_image, entry) for entry in entries]
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            
            m = result["entry"]
            item: Dict[str, Any] = {
                "type": "image",
                "page": m.get("page"),
                "index": m.get("index"),
                "filePath": result["fpath"],
                "fileName": result["fname"],
                "image_type": m.get("image_type"),
                "extraction_method": m.get("extraction_method"),
                "quality_score": m.get("quality_score"),
                "width": m.get("width"),
                "height": m.get("height"),
                "has_text": m.get("has_text"),
                "text_content": m.get("text_content"),
                "caption": m.get("caption"),
                "figure_reference": m.get("figure_reference"),
                "visual_elements": m.get("visual_elements"),
                "vector_count": m.get("vector_count"),
                "enhancement_applied": False,
                "mimeType": "image/png",
                # Supabase URL triplets (exact fields you rely on)
                "supabase_url": result["supa_url"] if result["supa_ok"] else None,
                "url": result["supa_url"] if result["supa_ok"] else None,
                "image_url": result["supa_url"] if result["supa_ok"] else None,
                "uploaded_filename": result["uploaded_name"] if result["supa_ok"] else None,
            }
            
            if result["supa_ok"] and result["supa_url"]:
                log.info(f"✅ Uploaded {result['uploaded_name']} to Supabase")
                log.info(f"✅ Image {result['fname']} uploaded to Supabase: {result['supa_url']}")
            else:
                # Fallback ONLY if Supabase is not configured or upload failed
                if result["data"]:
                    b64 = base64.b64encode(result["data"]).decode("utf-8")
                    item["base64_content"] = b64
                    log.warning(f"⚠️ Image {result['fname']} using base64 fallback")
            
            results.append(item)
    
    return results

# ------------------------------------------------------------------------------
# Table Packaging
# ------------------------------------------------------------------------------
def package_tables_from_csvs(output_dir: str, limit_tables: Optional[int]) -> List[Dict[str, Any]]:
    """
    Walk the output_dir for *.csv created by enterprise_table_extractor_full.py.
    Produce table items with lightweight metadata. CSV content is embedded
    (text) like the original working shapes did.
    """
    csv_paths = sorted(glob.glob(os.path.join(output_dir, "*.csv")))
    if limit_tables is not None:
        csv_paths = csv_paths[: max(0, int(limit_tables))]

    results: List[Dict[str, Any]] = []

    for path in csv_paths:
        file_name = os.path.basename(path)

        # Count rows/cols (quickly)
        rows = 0
        cols = 0
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    rows += 1
                    if i == 0:
                        cols = len(row)
        except Exception:
            pass

        csv_content = ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                csv_content = f.read()
        except Exception:
            csv_content = ""

        item = {
            "type": "table",
            "page": None,          # unknown unless extractor emits it; shape allows None
            "index": None,
            "filePath": path,
            "fileName": file_name,
            "table_type": None,
            "quality_score": None,
            "extraction_method": None,
            "rows": rows,
            "columns": cols,
            "has_headers": None,
            "numeric_percentage": None,
            "empty_cell_percentage": None,
            "metadata": {},
            "mimeType": "text/csv",
            "csv_content": csv_content,
        }
        results.append(item)

    return results

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.get("/", response_class=PlainTextResponse)
def root():
    return "PDF Extraction API is up."

@app.get("/test")
def test():
    return {"ok": True, "message": "pong"}

@app.get("/debug/check-environment")
def debug_env():
    return {
        "python": os.popen("python3 --version 2>&1").read().strip(),
        "cwd": os.getcwd(),
        "files_here": sorted(os.listdir(".")),
        "image_extractor_present": os.path.isfile(IMAGE_EXTRACTOR),
        "table_extractor_present": os.path.isfile(TABLE_EXTRACTOR),
        "supabase": {
            "enabled": supabase_enabled(),
            "url_set": bool(SUPABASE_URL),
            "bucket": SUPABASE_BUCKET if SUPABASE_BUCKET else None,
            "token_set": bool(SUPABASE_TOKEN),
        },
        "limits": {
            "TEMP_LIMIT_IMAGES": TEMP_LIMIT_IMAGES,
            "TEMP_LIMIT_TABLES": TEMP_LIMIT_TABLES,
        },
    }

# ---------------------- IMAGES ONLY -------------------------------------------
@app.post("/extract/images")
def extract_images(
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    workers: Optional[int] = Form(None),
    min_width: int = Form(100),
    min_height: int = Form(100),
    vector_threshold: int = Form(10),
    page_limit: Optional[int] = Form(None),
    limit_images: Optional[int] = Form(None),
    _: None = Depends(bearer_auth),
):
    """
    EXACT SAME image path as limiter-6:
    - run enterprise_image_extractor.py
    - upload each image to Supabase
    - return URLs (no base64 unless fallback)
    """
    # "single source of truth" for workers: from HTTP form, default 4
    img_workers = workers if workers is not None else 4

    # Use TEMP_LIMIT_IMAGES env if request doesn't specify
    if limit_images is None and TEMP_LIMIT_IMAGES is not None:
        try:
            limit_images = int(TEMP_LIMIT_IMAGES)
        except Exception:
            limit_images = None

    timestamp = now_stamp()

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "input.pdf")
        out_dir = os.path.join(temp_dir, "pdf_images")
        os.makedirs(out_dir, exist_ok=True)

        # Save upload
        pdf_bytes = file.file.read()
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # Build command (log mimics your Render output)
        # NOTE: we *execute* with sys.executable, but we log the path you saw before
        # for familiarity in your logs.
        pretty_cmd = (
            f"/usr/local/bin/python3.11 {IMAGE_EXTRACTOR} {pdf_path} "
            f"--output-dir {out_dir} --workers {img_workers} "
            f"--min-width {min_width} --min-height {min_height} "
            f"--min-quality {min_quality} --vector-threshold {vector_threshold} "
            f"--clear-output"
        )
        log.info(f"[IMAGES] Running: {pretty_cmd} (timeout={IMAGE_TIMEOUT_S}s)")

        # Actual execution uses the current Python interpreter for reliability
        cmd = [
            os.getenv("PYTHON_BIN", "python3"),
            IMAGE_EXTRACTOR,
            pdf_path,
            "--output-dir", out_dir,
            "--workers", str(img_workers),
            "--min-width", str(min_width),
            "--min-height", str(min_height),
            "--min-quality", str(min_quality),
            "--vector-threshold", str(vector_threshold),
            "--clear-output",
        ]
        if page_limit is not None:
            cmd += ["--page-limit", str(page_limit)]

        code, so, se = run_subprocess(cmd, timeout_s=IMAGE_TIMEOUT_S, cwd=".")
        log.info(f"Image extraction exit code: {code}")
        log.info(f"Image stdout (first 500 chars): {so[:500]}")
        if se:
            log.error(f"Image stderr: {se[:2000]}")

        if code == 124:
            # Subprocess timeout
            raise HTTPException(status_code=504, detail="Image extraction timed out after 15 minutes")
        if code != 0:
            raise HTTPException(status_code=500, detail=f"Image extractor failed with exit code {code}")

        # Package results (Supabase upload mandatory path)
        results = package_images_with_supabase(out_dir, timestamp, limit_images)
        images_count = len(results)
        log.info(f"Total results: {images_count} items")

        wrapper = {
            "results": results,
            "count": images_count,
            "tables_count": 0,
            "images_count": images_count,
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "success": True,
            "supabase_enabled": supabase_enabled(),
            "page_limit": page_limit,
        }
        return JSONResponse(wrapper)

# ---------------------- TABLES ONLY -------------------------------------------
@app.post("/extract/tables")
def extract_tables(
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    workers: Optional[int] = Form(None),
    page_start: Optional[int] = Form(None),
    page_end: Optional[int] = Form(None),
    page_max: Optional[int] = Form(None),
    no_verification: bool = Form(False),
    table_timeout_s: Optional[int] = Form(None),
    limit_tables: Optional[int] = Form(None),
    _: None = Depends(bearer_auth),
):
    """
    Real tables runner wired back in under /extract/tables, calling
    enterprise_table_extractor_full.py (exact name). Image path is untouched.
    """
    tbl_workers = workers if workers is not None else 4
    timeout_s = table_timeout_s if (table_timeout_s and table_timeout_s > 0) else TABLE_TIMEOUT_S_DEFAULT

    # Use TEMP_LIMIT_TABLES env if request doesn't specify
    if limit_tables is None and TEMP_LIMIT_TABLES is not None:
        try:
            limit_tables = int(TEMP_LIMIT_TABLES)
        except Exception:
            limit_tables = None

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "input.pdf")
        out_dir = os.path.join(temp_dir, "pdf_tables")
        os.makedirs(out_dir, exist_ok=True)

        # Save upload
        with open(pdf_path, "wb") as f:
            f.write(file.file.read())

        # Build and run table extractor
        cmd = [
            os.getenv("PYTHON_BIN", "python3"),
            TABLE_EXTRACTOR,
            pdf_path,
            "--output-dir", out_dir,
            "--workers", str(tbl_workers),
            "--min-quality", str(min_quality),
            "--clear-output",
        ]
        if page_start is not None:
            cmd += ["--page-start", str(page_start)]
        if page_end is not None:
            cmd += ["--page-end", str(page_end)]
        if page_max is not None:
            cmd += ["--page-max", str(page_max)]
        if no_verification:
            cmd += ["--no-verification"]

        pretty = " ".join(cmd)
        log.info(f"[TABLES] Running: {pretty} (timeout={timeout_s}s)")

        code, so, se = run_subprocess(cmd, timeout_s=timeout_s, cwd=".")
        log.info(f"Table extraction exit code: {code}")
        log.info(f"Table stdout (first 500 chars): {so[:500]}")
        if se:
            log.error(f"Table stderr: {se[:2000]}")

        if code == 124:
            raise HTTPException(status_code=504, detail="Table extraction timed out")
        if code != 0:
            raise HTTPException(status_code=500, detail=f"Table extractor failed with exit code {code}")

        # Package CSVs
        results = package_tables_from_csvs(out_dir, limit_tables)
        tables_count = len(results)
        log.info(f"Total results: {tables_count} table items")

        wrapper = {
            "results": results,
            "count": tables_count,
            "tables_count": tables_count,
            "images_count": 0,
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "success": True,
            "supabase_enabled": supabase_enabled(),
            "page_limit": None,
        }
        return JSONResponse(wrapper)

# ---------------------- "ALL" (images only, tables OFF) -----------------------
@app.post("/extract/all")
def extract_all(
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    workers: Optional[int] = Form(None),
    min_width: int = Form(100),
    min_height: int = Form(100),
    vector_threshold: int = Form(10),
    page_limit: Optional[int] = Form(None),
    limit_images: Optional[int] = Form(None),
    _: None = Depends(bearer_auth),
):
    """
    Mirrors limiter-6 behavior: **tables OFF**, **images ON** using the exact
    same packaging (Supabase upload & URLs).
    """
    # Route directly to the same image handler semantics
    img_workers = workers if workers is not None else 4

    # Use TEMP_LIMIT_IMAGES env if request doesn't specify
    if limit_images is None and TEMP_LIMIT_IMAGES is not None:
        try:
            limit_images = int(TEMP_LIMIT_IMAGES)
        except Exception:
            limit_images = None

    timestamp = now_stamp()

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, "input.pdf")
        out_dir = os.path.join(temp_dir, "pdf_images")
        os.makedirs(out_dir, exist_ok=True)

        # Save upload
        pdf_bytes = file.file.read()
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        log.info("📋 Table extraction is DISABLED")
        log.info("🖼️ Extracting images...")

        # Log string that matches your Render logs vibe
        pretty_cmd = (
            f"/usr/local/bin/python3.11 {IMAGE_EXTRACTOR} {pdf_path} "
            f"--output-dir {out_dir} --workers {img_workers} "
            f"--min-width {min_width} --min-height {min_height} "
            f"--min-quality {min_quality} --vector-threshold {vector_threshold} "
            f"--clear-output"
        )
        log.info(f"Running command: {pretty_cmd}")

        cmd = [
            os.getenv("PYTHON_BIN", "python3"),
            IMAGE_EXTRACTOR,
            pdf_path,
            "--output-dir", out_dir,
            "--workers", str(img_workers),
            "--min-width", str(min_width),
            "--min-height", str(min_height),
            "--min-quality", str(min_quality),
            "--vector-threshold", str(vector_threshold),
            "--clear-output",
        ]
        if page_limit is not None:
            cmd += ["--page-limit", str(page_limit)]

        code, so, se = run_subprocess(cmd, timeout_s=IMAGE_TIMEOUT_S, cwd=".")
        log.info(f"Image extraction exit code: {code}")
        log.info(f"Image stdout (first 500 chars): {so[:500]}")
        if se:
            log.error(f"Image stderr: {se[:2000]}")

        if code == 124:
            raise HTTPException(status_code=504, detail="Image extraction timed out after 15 minutes")
        if code != 0:
            raise HTTPException(status_code=500, detail=f"Image extractor failed with exit code {code}")

        results = package_images_with_supabase(out_dir, timestamp, limit_images)
        images_count = len(results)
        log.info(f"Total results: {images_count} items (0 tables, {images_count} images)")

        wrapper = {
            "results": results,
            "count": images_count,
            "tables_count": 0,
            "images_count": images_count,
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "success": True,
            "supabase_enabled": supabase_enabled(),
            "page_limit": page_limit,
        }
        return JSONResponse(wrapper)

  
