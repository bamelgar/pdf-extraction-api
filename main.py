import os
import sys
import json
import shutil
import base64
import logging
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# -----------------------------------------------------------------------------
# App & Security
# -----------------------------------------------------------------------------

app = FastAPI(title="PDF Extraction API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

logger = logging.getLogger("pdf-extraction-api")
logging.basicConfig(level=logging.INFO)

bearer = HTTPBearer(auto_error=False)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> str:
    api_key = os.environ.get("API_KEY", "").strip()
    if not api_key:
        return ""
    if not credentials or not credentials.credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    if credentials.credentials != api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return credentials.credentials

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent

def _script_path(name: str) -> str:
    p = REPO_DIR / name
    return str(p if p.exists() else name)

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def resolve_workers(request: Request, form_workers: int) -> int:
    try:
        qs = int(request.query_params.get("workers", form_workers))
    except Exception:
        qs = form_workers
    requested = max(int(form_workers or 1), qs)
    max_allowed = min(16, os.cpu_count() or 1)
    actual = max(1, min(requested, max_allowed))
    logger.info(f"Workers resolved -> form={form_workers}, query={qs}, cpus={os.cpu_count()}, using={actual}")
    return actual

def resolve_page_limit(request: Request, form_limit: Optional[int]) -> Optional[int]:
    """Honor both form and query; prefer the smaller positive (safer in cloud)."""
    qval = request.query_params.get("page_limit")
    qint = None
    try:
        if qval is not None:
            qint = int(qval)
    except Exception:
        qint = None

    candidates = [v for v in [form_limit, qint] if v and v > 0]
    if not candidates:
        logger.info("Page limit resolved -> none provided; processing all pages")
        return None
    # cap very high values defensively (no behavior change for typical tests)
    using = max(1, min(min(candidates), 2000))
    logger.info(f"Page limit resolved -> form={form_limit}, query={qint}, using={using}")
    return using

def supabase_public_url(bucket: str, object_name: str) -> Optional[str]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    if not url:
        return None
    return f"{url}/storage/v1/object/public/{bucket}/{object_name}"

def upload_to_supabase(bucket: str, object_name: str, file_path: Path, content_type: str = "application/octet-stream") -> Optional[str]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY") or ""
    if not url or not key:
        logger.info("Supabase not configured (SUPABASE_URL / SUPABASE_*_KEY missing). Skipping upload.")
        return None
    endpoint = f"{url}/storage/v1/object/{bucket}/{object_name}"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    try:
        with open(file_path, "rb") as f:
            resp = requests.put(endpoint, data=f, headers=headers, timeout=120)
        if 200 <= resp.status_code < 300:
            public = supabase_public_url(bucket, object_name)
            logger.info(f"Uploaded to Supabase: {public}")
            return public
        logger.error(f"Supabase upload failed [{resp.status_code}]: {resp.text}")
        return None
    except Exception as e:
        logger.exception(f"Supabase upload error: {e}")
        return None

def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# -----------------------------------------------------------------------------
# Health / Info (no auth)
# -----------------------------------------------------------------------------

@app.get("/")
async def root() -> Dict[str, Any]:
    return {"message": "PDF Extraction API", "status": "running", "timestamp": _now_iso()}

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "healthy", "cpus": os.cpu_count(), "timestamp": _now_iso()}

@app.get("/test")
async def test_endpoint() -> Dict[str, Any]:
    return {"message": "API is working!", "timestamp": _now_iso(), "python_version": sys.version}

@app.get("/debug/check-environment")
async def debug_check_environment(_: str = Depends(verify_token)) -> Dict[str, Any]:
    checks: Dict[str, Any] = {"timestamp": _now_iso()}
    commands = {
        "python": [sys.executable, "--version"],
        "tesseract": ["tesseract", "--version"],
        "ghostscript": ["gs", "--version"],
        "java": ["java", "-version"],
        "pdftoppm": ["pdftoppm", "-v"],
    }
    for name, cmd in commands.items():
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            out = res.stdout.strip() or res.stderr.strip()
            checks[name] = {"ok": res.returncode == 0, "output": out.splitlines()[:3]}
        except FileNotFoundError:
            checks[name] = {"ok": False, "error": "not found"}
        except Exception as e:
            checks[name] = {"ok": False, "error": str(e)}
    checks["cpus"] = os.cpu_count()
    return checks

# -----------------------------------------------------------------------------
# Table Extraction
# -----------------------------------------------------------------------------

@app.post("/extract/tables")
async def extract_tables(
    request: Request,
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    workers: int = Form(4),
    include_csv_base64: bool = Form(True),
    page_limit: Optional[int] = Form(None),
    _: str = Depends(verify_token),
) -> JSONResponse:
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    try:
        pdf_path = temp_path / "input.pdf"
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        tables_dir = temp_path / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        actual_workers = resolve_workers(request, workers)
        actual_limit = resolve_page_limit(request, page_limit)

        cmd = [
            sys.executable,
            _script_path("enterprise_table_extractor_full.py"),
            str(pdf_path),
            "--output-dir", str(tables_dir),
            "--workers", str(actual_workers),
            "--min-quality", str(min_quality),
            "--clear-output",
        ]
        if actual_limit:
            cmd.extend(["--page-limit", str(actual_limit)])

        logger.info("Running tables cmd: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=int(os.environ.get("TABLES_TIMEOUT_SEC", "1200")))
        if result.returncode != 0:
            logger.error("Tables extractor failed: %s", result.stderr)
            raise HTTPException(status_code=500, detail="Table extraction failed")

        meta_path = tables_dir / "extraction_metadata.json"
        if not meta_path.exists():
            raise HTTPException(status_code=500, detail="No table metadata produced")
        meta = _read_json(meta_path)

        tables: List[Dict[str, Any]] = []
        for t in meta.get("tables", []):
            csv_path = tables_dir / t["filename"]
            csv_content = _read_text(csv_path) if csv_path.exists() else ""
            item = {
                "filename": t.get("filename"),
                "page_number": t.get("page_number"),
                "table_index": t.get("table_index"),
                "table_type": t.get("table_type"),
                "quality_score": t.get("quality_score"),
                "rows": t.get("rows"),
                "columns": t.get("columns"),
                "csv_content": csv_content,
                "metadata": t.get("metadata", {}),
            }
            if include_csv_base64 and csv_path.exists():
                item["csv_base64"] = _b64(csv_path)
            tables.append(item)

        payload = {
            "success": True,
            "tables_count": len(tables),
            "tables": tables,
            "statistics": meta.get("statistics", {}),
            "extraction_timestamp": _now_iso(),
        }
        return JSONResponse(content=payload)
    except subprocess.TimeoutExpired:
        logger.error("Table extraction timed out")
        raise HTTPException(status_code=504, detail="Table extraction timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Table extraction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Table extraction error: {e}")
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)

# -----------------------------------------------------------------------------
# Image Extraction
# -----------------------------------------------------------------------------

@app.post("/extract/images")
async def extract_images(
    request: Request,
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    min_width: int = Form(64),
    min_height: int = Form(64),
    workers: int = Form(4),
    vector_threshold: int = Form(10),
    include_base64: bool = Form(False),
    skip_ocr: bool = Form(False),
    page_limit: Optional[int] = Form(None),
    supabase_bucket: str = Form("public-images"),
    supabase_prefix: str = Form(""),
    _: str = Depends(verify_token),
) -> JSONResponse:
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    try:
        pdf_path = temp_path / "input.pdf"
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        images_dir = temp_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        actual_workers = resolve_workers(request, workers)
        actual_limit = resolve_page_limit(request, page_limit)

        cmd = [
            sys.executable,
            _script_path("enterprise_image_extractor.py"),
            str(pdf_path),
            "--output-dir", str(images_dir),
            "--workers", str(actual_workers),
            "--min-width", str(min_width),
            "--min-height", str(min_height),
            "--min-quality", str(min_quality),
            "--vector-threshold", str(vector_threshold),
            "--clear-output",
        ]
        if actual_limit:
            cmd.extend(["--page-limit", str(actual_limit)])
        if skip_ocr:
            cmd.append("--no-ocr")  # v2 flag

        logger.info("Running images cmd: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=int(os.environ.get("IMAGES_TIMEOUT_SEC", "900")))
        if result.returncode != 0:
            logger.error("Images extractor failed: %s", result.stderr)
            raise HTTPException(status_code=500, detail="Image extraction failed")

        meta_path = images_dir / "extraction_metadata.json"
        if not meta_path.exists():
            raise HTTPException(status_code=500, detail="No image metadata produced")
        meta = _read_json(meta_path)
        images_meta = meta.get("images", [])

        images: List[Dict[str, Any]] = []
        for im in images_meta:
            filename = im.get("filename")
            img_path = images_dir / filename if filename else None
            item = {
                "filename": filename,
                "page_number": im.get("page_number"),
                "image_index": im.get("image_index"),
                "image_type": im.get("image_type"),
                "extraction_method": im.get("extraction_method"),
                "quality_score": im.get("quality_score"),
                "width": im.get("width"),
                "height": im.get("height"),
                "metadata": im.get("metadata", {}),
            }
            if img_path and img_path.exists():
                if os.environ.get("SUPABASE_URL") and (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")):
                    object_name = f"{supabase_prefix}/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}".lstrip("/").replace("//", "/")
                    public_url = upload_to_supabase(supabase_bucket, object_name, img_path, content_type="image/png")
                    if public_url:
                        item["supabase_url"] = public_url
                        item["uploaded_filename"] = object_name
                if include_base64:
                    try:
                        item["base64_content"] = _b64(img_path)
                    except Exception as e:
                        logger.warning(f"Failed to base64 image {filename}: {e}")
                        item["base64_content"] = None
            images.append(item)

        payload = {
            "success": True,
            "images_count": len(images),
            "images": images,
            "statistics": meta.get("statistics", {}),
            "extraction_timestamp": _now_iso(),
        }
        return JSONResponse(content=payload)
    except subprocess.TimeoutExpired:
        logger.error("Image extraction timed out")
        raise HTTPException(status_code=504, detail="Image extraction timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Image extraction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Image extraction error: {e}")
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)

# -----------------------------------------------------------------------------
# Combined Extraction
# -----------------------------------------------------------------------------

@app.post("/extract/all")
async def extract_all(
    request: Request,
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    min_width: int = Form(64),
    min_height: int = Form(64),
    workers: int = Form(4),
    vector_threshold: int = Form(10),
    include_base64: bool = Form(False),
    include_csv_base64: bool = Form(True),
    skip_ocr: bool = Form(False),
    page_limit: Optional[int] = Form(None),
    supabase_bucket: str = Form("public-images"),
    supabase_prefix: str = Form(""),
    _: str = Depends(verify_token),
) -> JSONResponse:
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    try:
        pdf_path = temp_path / "input.pdf"
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        actual_workers = resolve_workers(request, workers)
        actual_limit = resolve_page_limit(request, page_limit)

        # --- Tables ---
        tables_dir = temp_path / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        tables_cmd = [
            sys.executable, _script_path("enterprise_table_extractor_full.py"),
            str(pdf_path),
            "--output-dir", str(tables_dir),
            "--workers", str(actual_workers),
            "--min-quality", str(min_quality),
            "--clear-output",
        ]
        if actual_limit:
            tables_cmd.extend(["--page-limit", str(actual_limit)])
        logger.info("Running tables cmd: %s", " ".join(tables_cmd))
        tables_res = subprocess.run(tables_cmd, capture_output=True, text=True, timeout=int(os.environ.get("TABLES_TIMEOUT_SEC", "1200")))
        if tables_res.returncode != 0:
            logger.error("Tables extractor failed: %s", tables_res.stderr)
        tables_meta = {}
        if (tables_dir / "extraction_metadata.json").exists():
            tables_meta = _read_json(tables_dir / "extraction_metadata.json")

        tables_out: List[Dict[str, Any]] = []
        for t in tables_meta.get("tables", []):
            csv_path = tables_dir / t.get("filename", "")
            entry = {
                "filename": t.get("filename"),
                "page_number": t.get("page_number"),
                "table_index": t.get("table_index"),
                "table_type": t.get("table_type"),
                "quality_score": t.get("quality_score"),
                "rows": t.get("rows"),
                "columns": t.get("columns"),
                "csv_content": _read_text(csv_path) if csv_path.exists() else "",
                "metadata": t.get("metadata", {}),
            }
            if include_csv_base64 and csv_path.exists():
                entry["csv_base64"] = _b64(csv_path)
            tables_out.append(entry)

        # --- Images ---
        images_dir = temp_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        images_cmd = [
            sys.executable, _script_path("enterprise_image_extractor.py"),
            str(pdf_path),
            "--output-dir", str(images_dir),
            "--workers", str(actual_workers),
            "--min-width", str(min_width),
            "--min-height", str(min_height),
            "--min-quality", str(min_quality),
            "--vector-threshold", str(vector_threshold),
            "--clear-output",
        ]
        if actual_limit:
            images_cmd.extend(["--page-limit", str(actual_limit)])
        if skip_ocr:
            images_cmd.append("--no-ocr")
        logger.info("Running images cmd: %s", " ".join(images_cmd))
        images_res = subprocess.run(images_cmd, capture_output=True, text=True, timeout=int(os.environ.get("IMAGES_TIMEOUT_SEC", "900")))
        if images_res.returncode != 0:
            logger.error("Images extractor failed: %s", images_res.stderr)

        images_meta = {}
        if (images_dir / "extraction_metadata.json").exists():
            images_meta = _read_json(images_dir / "extraction_metadata.json")
        images_out: List[Dict[str, Any]] = []
        for im in images_meta.get("images", []):
            filename = im.get("filename")
            img_path = images_dir / filename if filename else None
            entry = {
                "filename": filename,
                "page_number": im.get("page_number"),
                "image_index": im.get("image_index"),
                "image_type": im.get("image_type"),
                "extraction_method": im.get("extraction_method"),
                "quality_score": im.get("quality_score"),
                "width": im.get("width"),
                "height": im.get("height"),
                "metadata": im.get("metadata", {}),
            }
            if img_path and img_path.exists():
                if os.environ.get("SUPABASE_URL") and (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")):
                    object_name = f"{supabase_prefix}/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{filename}".lstrip("/").replace("//", "/")
                    public_url = upload_to_supabase(supabase_bucket, object_name, img_path, content_type="image/png")
                    if public_url:
                        entry["supabase_url"] = public_url
                        entry["uploaded_filename"] = object_name
                if include_base64:
                    try:
                        entry["base64_content"] = _b64(img_path)
                    except Exception as e:
                        logger.warning(f"Failed to base64 image {filename}: {e}")
                        entry["base64_content"] = None
            images_out.append(entry)

        # --- Unified results ---
        results: List[Dict[str, Any]] = []
        for t in tables_out:
            results.append({
                "type": "table",
                "page": t.get("page_number"),
                "index": t.get("table_index"),
                "fileName": t.get("filename"),
                "filePath": f"/data/pdf_tables/{t.get('filename')}",
                "table_type": t.get("table_type"),
                "quality_score": t.get("quality_score"),
                "rows": t.get("rows"),
                "columns": t.get("columns"),
                "csv_content": t.get("csv_content"),
                **({"csv_base64": t.get("csv_base64")} if "csv_base64" in t else {}),
            })
        for im in images_out:
            results.append({
                "type": "image",
                "page": im.get("page_number"),
                "index": im.get("image_index"),
                "fileName": im.get("filename"),
                "filePath": f"/data/pdf_images/{im.get('filename')}",
                "image_type": im.get("image_type"),
                "extraction_method": im.get("extraction_method"),
                "quality_score": im.get("quality_score"),
                "width": im.get("width"),
                "height": im.get("height"),
                **({"supabase_url": im.get("supabase_url")} if "supabase_url" in im else {}),
                **({"base64_content": im.get("base64_content")} if "base64_content" in im else {}),
            })
        results.sort(key=lambda r: (r.get("page") or 0, (r.get("index") or 0)))

        payload = {
            "success": True,
            "tables": {"tables_count": len(tables_out), "tables": tables_out, "statistics": tables_meta.get("statistics", {})},
            "images": {"images_count": len(images_out), "images": images_out, "statistics": images_meta.get("statistics", {})},
            "results": results,
            "extraction_timestamp": _now_iso(),
        }
        return JSONResponse(content=payload)
    except subprocess.TimeoutExpired:
        logger.error("Combined extraction timed out")
        raise HTTPException(status_code=504, detail="Combined extraction timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Combined extraction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Combined extraction error: {e}")
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
