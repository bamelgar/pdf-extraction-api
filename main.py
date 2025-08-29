"""
PDF Extraction API — Concurrent Production Version (Tables+Images) with Supabase Compatibility
==============================================================================================

WHAT THIS VERSION DOES (AT A GLANCE)
------------------------------------
1) Runs TABLE and IMAGE extractors **concurrently** (async) for lower wall-clock time.
2) **Preserves limiter flags** and extractor knobs:
   - Tables:  --workers, --min-quality, --page-limit (optional), --clear-output
   - Images:  --workers, --min-width, --min-height, --min-quality, --vector-threshold, --page-limit (optional), --clear-output
3) Keeps the **JSON response shape** your n8n graph expects:
   - Top-level: {"results": [...], "count": N, "success": true, "extraction_timestamp": "..."}
   - Each **table** item includes **csv_content** (required by your cloud “Read Table Files” node).
   - Each **image** item **passes through** Supabase URL fields exactly as produced upstream (no behavior change).
4) Adds an **OPTIONAL** CSV→Supabase storage helper for tables:
   - Controlled by env var UPLOAD_TABLE_CSVS=true (default: false).
   - Adds non-breaking field `table_url` while still returning **csv_content**.
   - If disabled or creds missing, tables behave exactly like before (no uploads).
5) Provides **runtime toggles** to test modalities without code edits:
   - Query params: include_images={true|false}, include_tables={true|false}
   - Example (tables only):  /extract/all?include_images=false&include_tables=true
6) Adds **convenience wrapper endpoints** that delegate to /extract/all:
   - /extract/tables  → runs tables only (images off)
   - /extract/images  → runs images only (tables off)

WHY THIS MATTERS FOR YOUR WORKFLOW
----------------------------------
• **Images path untouched.** Supabase image bucketing remains exactly as in your last working build. We do not change image upload logic or fields
  (critical for staying under n8n’s ~45MB cap). Your downstream “List All Image Files” continues to filter on `supabase_url`.

• **Tables path compatible.** Your cloud “Read Table Files” expects `csv_content`; this build **always includes it** (and only *optionally* adds `table_url`).

• **Safer testing.** You can validate DHI 10-K tables by setting include_images=false. When ready, flip images back on and you’ll get true concurrent timing.

TUNING & DEFAULTS (RENDER PROFESSIONAL)
---------------------------------------
• Recommended `workers`: **6–8** to avoid CPU thrash; 16 is often slower on small/medium instances.
• Avoid `page_limit` during real runs (10-Ks often have late tables). Use it only for debugging.
• `min_quality` 0.3 is a sensible floor for both modalities to reduce junk without dropping valid content.

ENV VARS & BEHAVIOR SWITCHES
----------------------------
• API_KEY                 : Simple Bearer auth for n8n HTTP Request node.
• SUPABASE_URL            : (images path already uses Supabase; used here only if you enable table CSV uploads)
• SUPABASE_TOKEN          : (same as above)
• SUPABASE_BUCKET         : defaults to "public-images" (images). CSVs (if enabled) are stored under "tables/<filename>" in the same bucket.
• UPLOAD_TABLE_CSVS       : "true"/"1" to enable optional table CSV uploads; **default false** (return `csv_content` as usual).

STARTUP LOGGING & DEBUG VISIBILITY
----------------------------------
• On boot, logs the following so you can verify config **without** hitting an endpoint:
  - “Tables enabled: YES…”
  - `UPLOAD_TABLE_CSVS=<True|False>`
  - Whether `SUPABASE_URL` is set and the bucket name
  - Java/Tesseract availability
• `/debug/check-environment` returns `env_flags` including `UPLOAD_TABLE_CSVS`, and shows script presence + Java/Tesseract checks.

COMPATIBILITY & NON-BREAKING GUARANTEES
---------------------------------------
• Response schema unchanged for downstream nodes:
  - **Tables**: still provide `csv_content`.
  - **Images**: Supabase URL fields (`supabase_url`, `image_url`, `url`, `uploaded_filename`) are preserved if present.
• Sorting is still by (page, index) to keep deterministic ordering.
• Fail-soft behavior: if one extractor exits non-zero, the other modality’s results are still returned (logged in stderr).

TROUBLESHOOTING CHECKLIST
-------------------------
• If **no tables** arrive downstream:
  - Confirm root ("/") shows `"TABLES_ENABLED": "YES..."`.
  - Ensure your n8n HTTP node **does not** send `page_limit=100` for long filings.
  - Use the tables-only toggle: include_images=false&include_tables=true (or call /extract/tables).
  - Check Render logs for extractor exit codes and stderr snippets.

• If **timeouts** occur:
  - Lower `workers` to 6–8.
  - Temporarily set a small `page_limit` for debugging only.
  - Verify Java presence at `/debug/check-environment` (Tabula needs it).

IMPORTANT GUARANTEE
-------------------
Per your directive: **Image extraction logic and Supabase image storage behavior are not altered.**
All changes here are confined to:
  (a) **concurrency orchestration** around the two subprocesses,
  (b) adding **optional** CSV→Supabase for tables (non-breaking),
  (c) adding **runtime toggles** for selective execution,
  (d) startup logging so Render shows UPLOAD_TABLE_CSVS state immediately.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os, sys, json, base64, shutil, tempfile, logging, asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

# --------------------------------------------------------------------------------------
# App & CORS
# --------------------------------------------------------------------------------------
app = FastAPI(title="PDF Extraction API", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------------------
security = HTTPBearer()
API_KEY = os.environ.get("API_KEY", "your-secret-api-key-change-this")
def verify(creds: HTTPAuthorizationCredentials = Security(security)):
    if creds.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return True

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

def _now() -> str:
    return datetime.now().isoformat()

# --------------------------------------------------------------------------------------
# Supabase config (images path remains as-is; CSV upload for tables is OPTIONAL)
# --------------------------------------------------------------------------------------
SUPABASE_URL    = os.environ.get("SUPABASE_URL", "")
SUPABASE_TOKEN  = os.environ.get("SUPABASE_TOKEN", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "public-images")
UPLOAD_TABLE_CSVS = os.environ.get("UPLOAD_TABLE_CSVS", "false").lower() in {"1", "true", "yes"}

# Lazy import for CSV uploads
try:
    import requests  # only needed if UPLOAD_TABLE_CSVS is true
except Exception:
    requests = None

# ----- Startup logging (show critical env flags without hitting debug endpoint) -----
log.info("=== PDF Extraction API starting ===")
log.info("Tables enabled: YES (full functionality with limiter version)")
log.info("UPLOAD_TABLE_CSVS=%s", UPLOAD_TABLE_CSVS)
log.info("SUPABASE_URL set: %s | bucket=%s", bool(SUPABASE_URL), SUPABASE_BUCKET)
log.info("Java available: %s", bool(shutil.which("java")))
log.info("Tesseract available: %s", bool(shutil.which("tesseract")))

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
async def run_proc(cmd: List[str], timeout: int) -> Dict[str, Any]:
    """Run a subprocess asynchronously. Returns {'exit','stdout','stderr'}."""
    log.info("Running: %s (timeout=%ss)", " ".join(cmd), timeout)
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(status_code=504, detail=f"Command timed out: {' '.join(cmd)}")
    return {
        "exit": proc.returncode,
        "stdout": stdout.decode("utf-8", errors="ignore"),
        "stderr": stderr.decode("utf-8", errors="ignore"),
    }

def safe_json_read(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("Could not read JSON %s: %s", path, e)
        return default

def maybe_upload_csv_to_supabase(csv_path: str, dest_name: str) -> Optional[str]:
    """
    OPTIONAL helper: upload table CSV to Supabase Storage.
    - Controlled by UPLOAD_TABLE_CSVS env (default false)
    - Returns a public URL if successful; None otherwise
    """
    if not UPLOAD_TABLE_CSVS:
        return None
    if not (requests and SUPABASE_URL and SUPABASE_TOKEN and SUPABASE_BUCKET):
        log.info("CSV upload skipped (missing requests or Supabase creds).")
        return None
    try:
        url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/{SUPABASE_BUCKET}/tables/{dest_name}"
        with open(csv_path, "rb") as f:
            data = f.read()
        headers = {
            "Authorization": f"Bearer {SUPABASE_TOKEN}",
            "Content-Type": "text/csv",
            "x-upsert": "true",
        }
        r = requests.post(url, headers=headers, data=data, timeout=60)
        if r.status_code in (200, 201):
            public = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{SUPABASE_BUCKET}/tables/{dest_name}"
            return public
        log.warning("CSV upload failed status=%s body=%s", r.status_code, r.text[:200])
        return None
    except Exception as e:
        log.warning("CSV upload error: %s", e)
        return None

# --------------------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "PDF Extraction API",
        "version": "1.3.0",
        "timestamp": _now(),
        "TABLES_ENABLED": "YES - Full functionality with limiter version",
    }

# silence Render's HEAD probe 405s
@app.head("/")
async def head_root():
    return {}

@app.get("/debug/check-environment")
async def check_environment():
    checks = {
        "python_version": sys.version,
        "current_directory": os.getcwd(),
        "scripts_exist": {
            "table_extractor": os.path.exists("enterprise_table_extractor_full.py"),
            "image_extractor": os.path.exists("enterprise_image_extractor.py"),
        },
        "system_checks": {
            "java_available": shutil.which("java") is not None,
            "tesseract_available": shutil.which("tesseract") is not None,
        },
        "env_flags": {
            "UPLOAD_TABLE_CSVS": os.environ.get("UPLOAD_TABLE_CSVS", "false"),
            "SUPABASE_URL_set": bool(SUPABASE_URL),
            "SUPABASE_BUCKET": SUPABASE_BUCKET,
        },
    }
    return checks

# --------------------------------------------------------------------------------------
# Core: /extract/all  (concurrent orchestration + configurable timeouts)
# --------------------------------------------------------------------------------------
@app.post("/extract/all")
async def extract_all(
    file: UploadFile = File(...),
    # shared knobs
    min_quality: float = 0.3,
    workers: int = 4,
    page_limit: Optional[int] = None,
    # image-only knobs
    min_width: int = 100,
    min_height: int = 100,
    # toggles
    include_images: Optional[bool] = True,
    include_tables: Optional[bool] = True,
    # timeouts (seconds)
    table_timeout_s: int = 900,
    image_timeout_s: int = 900,
    # optional speed knob for tables
    no_verification: bool = False,
    token: bool = Depends(verify),
):
    temp_dir = tempfile.mkdtemp()
    try:
        pdf_path = os.path.join(temp_dir, "input.pdf")
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        tables_dir = os.path.join(temp_dir, "pdf_tables")
        images_dir = os.path.join(temp_dir, "pdf_images")
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        # ----- Build commands (limiter flags preserved) -----
        table_cmd = [
            sys.executable, "enterprise_table_extractor_full.py", pdf_path,
            "--output-dir", tables_dir,
            "--workers", str(workers),
            "--min-quality", str(min_quality),
            "--clear-output",
        ]
        if page_limit and int(page_limit) > 0:
            table_cmd += ["--page-limit", str(int(page_limit))]
        if no_verification:
            table_cmd += ["--no-verification"]

        image_cmd = [
            sys.executable, "enterprise_image_extractor.py", pdf_path,
            "--output-dir", images_dir,
            "--workers", str(workers),
            "--min-width", str(min_width),
            "--min-height", str(min_height),
            "--min-quality", str(min_quality),
            "--vector-threshold", "10",
            "--clear-output",
        ]
        if page_limit and int(page_limit) > 0:
            image_cmd += ["--page-limit", str(int(page_limit))]

        # ----- Launch concurrently (respect toggles & timeouts) -----
        tasks = []
        if include_tables:
            tasks.append(run_proc(table_cmd, timeout=table_timeout_s))
        else:
            tasks.append(asyncio.sleep(0, result={"exit": 0, "stdout": "", "stderr": ""}))

        if include_images:
            tasks.append(run_proc(image_cmd, timeout=image_timeout_s))
        else:
            tasks.append(asyncio.sleep(0, result={"exit": 0, "stdout": "", "stderr": ""}))

        table_res, image_res = await asyncio.gather(*tasks)

        results: List[Dict[str, Any]] = []

        # ----- Package TABLES (keep csv_content; optional table_url) -----
        if include_tables and table_res["exit"] == 0:
            tmeta = safe_json_read(os.path.join(tables_dir, "extraction_metadata.json"), {})
            for t in tmeta.get("tables", []):
                csv_path = os.path.join(tables_dir, t["filename"])
                csv_content = ""
                if os.path.exists(csv_path):
                    try:
                        with open(csv_path, "r", encoding="utf-8") as cf:
                            csv_content = cf.read()
                    except Exception as e:
                        log.error("CSV read error %s: %s", csv_path, e)
                table_url = None
                if os.path.exists(csv_path):
                    # store as tables/<filename> in the same bucket
                    table_url = maybe_upload_csv_to_supabase(csv_path, t["filename"])
                results.append({
                    "type": "table",
                    "page": t.get("page_number", 0),
                    "index": t.get("table_index", 0),
                    "filePath": f"/data/pdf_tables/{t['filename']}",
                    "fileName": t["filename"],
                    "table_type": t.get("table_type", "general_data"),
                    "quality_score": t.get("quality_score", 0.0),
                    "extraction_method": t.get("extraction_method", "unknown"),
                    "rows": t.get("rows", 0),
                    "columns": t.get("columns", 0),
                    "size_bytes": t.get("size_bytes", 0),
                    "has_headers": t.get("has_headers", True),
                    "numeric_percentage": t.get("numeric_percentage", 0),
                    "empty_cell_percentage": t.get("empty_cell_percentage", 0),
                    "metadata": t.get("metadata", {}),
                    "mimeType": "text/csv",
                    "csv_content": csv_content,     # REQUIRED by your n8n cloud path
                    "table_url": table_url,         # OPTIONAL; safe for you to ignore
                })
        elif include_tables and table_res["exit"] != 0:
            log.error("Table extractor failed: %s", table_res["stderr"][:1000])

        # ----- Package IMAGES (pass-through; Supabase fields preserved) -----
        if include_images and image_res["exit"] == 0:
            imeta = safe_json_read(os.path.join(images_dir, "extraction_metadata.json"), {})
            for img in imeta.get("images", []):
                item = {
                    "type": "image",
                    "page": img.get("page_number", 0),
                    "index": img.get("image_index", 0),
                    "filePath": f"/data/pdf_images/{img.get('filename')}",
                    "fileName": img.get("filename"),
                    "image_type": img.get("image_type", "general_image"),
                    "extraction_method": img.get("extraction_method", "unknown"),
                    "quality_score": img.get("quality_score", 0.0),
                    "width": img.get("width", 0),
                    "height": img.get("height", 0),
                    "has_text": img.get("has_text", False),
                    "text_content": img.get("text_content", ""),
                    "caption": img.get("context", {}).get("caption"),
                    "figure_reference": img.get("context", {}).get("figure_reference"),
                    "visual_elements": img.get("visual_elements", {}),
                    "vector_count": img.get("vector_count"),
                    "enhancement_applied": img.get("enhancement_applied", False),
                    "mimeType": "image/png",
                }
                for k in ("supabase_url", "url", "image_url", "uploaded_filename"):
                    if k in img:
                        item[k] = img[k]
                results.append(item)
        elif include_images and image_res["exit"] != 0:
            log.error("Image extractor failed: %s", image_res["stderr"][:1000])

        # ----- Sort & return -----
        results.sort(key=lambda x: (x.get("page", 0), x.get("index", 0)))
        return {
            "results": results,
            "count": len(results),
            "extraction_timestamp": _now(),
            "success": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Extraction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Extraction error: {e}")
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

# --------------------------------------------------------------------------------------
# Convenience wrappers (delegate to /extract/all with your new timeout knobs)
# --------------------------------------------------------------------------------------
@app.post("/extract/tables")
async def extract_tables_only(
    file: UploadFile = File(...),
    min_quality: float = 0.3,
    workers: int = 4,
    page_limit: Optional[int] = None,
    table_timeout_s: int = 900,
    no_verification: bool = False,
    token: bool = Depends(verify),
):
    return await extract_all(
        file=file, min_quality=min_quality, workers=workers, page_limit=page_limit,
        include_images=False, include_tables=True,
        table_timeout_s=table_timeout_s, image_timeout_s=900,
        no_verification=no_verification,
        token=token
    )

@app.post("/extract/images")
async def extract_images_only(
    file: UploadFile = File(...),
    min_quality: float = 0.3,
    workers: int = 4,
    min_width: int = 100,
    min_height: int = 100,
    page_limit: Optional[int] = None,
    image_timeout_s: int = 900,
    token: bool = Depends(verify),
):
    return await extract_all(
        file=file, min_quality=min_quality, workers=workers,
        min_width=min_width, min_height=min_height, page_limit=page_limit,
        include_images=True, include_tables=False,
        table_timeout_s=900, image_timeout_s=image_timeout_s,
        token=token
    )

# --------------------------------------------------------------------------------------
# Local dev entry
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
