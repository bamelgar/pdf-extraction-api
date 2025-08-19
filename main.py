from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import shutil
import json
import base64
from typing import Optional
import subprocess
import sys
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed  # parallel IO/encode

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

# -------------------------------------------------------------------
# FastAPI App
# -------------------------------------------------------------------
app = FastAPI(title="PDF Extraction API", version="1.0.0")

# CORS for n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -------------------------------------------------------------------
# Auth
# -------------------------------------------------------------------
security = HTTPBearer()
API_KEY = os.environ.get("API_KEY", "your-secret-api-key-change-this")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials.credentials

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _resolve_workers(request_workers: Optional[int]) -> int:
    """
    Use caller-provided workers if >=1; otherwise auto-scale to CPU (cap 16).
    This lets you exploit Render's higher CPU tiers without changing n8n.
    """
    if request_workers and request_workers >= 1:
        return int(request_workers)
    cpu = os.cpu_count() or 4
    return min(cpu, 16)

def _safe_read(path: str, mode: str = "rb", bufsize: int = 1 << 20):
    return open(path, mode, buffering=bufsize)

# -------------------------------------------------------------------
# Health / Test
# -------------------------------------------------------------------
@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "PDF Extraction API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/test")
async def test_endpoint():
    return {
        "message": "API is working!",
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version
    }

@app.post("/extract/test")
async def test_extraction(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    return {
        "success": True,
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "File received successfully"
    }

# -------------------------------------------------------------------
# TABLES
# -------------------------------------------------------------------
@app.post("/extract/tables")
async def extract_tables(
    file: UploadFile = File(...),
    min_quality: float = 0.3,
    workers: Optional[int] = None,     # CHANGED default -> auto if not provided
    token: str = Depends(verify_token)
):
    temp_dir = tempfile.mkdtemp()
    try:
        # Save upload
        pdf_path = os.path.join(temp_dir, "input.pdf")
        with _safe_read(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Output
        output_dir = os.path.join(temp_dir, "tables")
        os.makedirs(output_dir, exist_ok=True)

        eff_workers = _resolve_workers(workers)
        cmd = [
            sys.executable, "enterprise_table_extractor_full.py",
            pdf_path,
            "--output-dir", output_dir,
            "--workers", str(eff_workers),
            "--min-quality", str(min_quality),
            "--clear-output",
        ]
        logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"Extraction failed: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Extraction failed: {result.stderr}")

        metadata_path = os.path.join(output_dir, "extraction_metadata.json")
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=500, detail="No metadata file generated")

        with _safe_read(metadata_path, "r") as f:
            metadata = json.load(f)

        tables = []
        for table_info in metadata.get('tables', []):
            csv_path = os.path.join(output_dir, table_info['filename'])
            if os.path.exists(csv_path):
                with _safe_read(csv_path, "r") as f:
                    csv_content = f.read()
                with _safe_read(csv_path, "rb") as f:
                    csv_base64 = base64.b64encode(f.read()).decode('ascii')

                tables.append({
                    'filename': table_info['filename'],
                    'page_number': table_info['page_number'],
                    'table_index': table_info['table_index'],
                    'table_type': table_info['table_type'],
                    'quality_score': table_info['quality_score'],
                    'rows': table_info['rows'],
                    'columns': table_info['columns'],
                    'csv_content': csv_content,
                    'csv_base64': csv_base64,
                    'metadata': table_info.get('metadata', {})
                })

        return {
            'success': True,
            'tables_count': len(tables),
            'tables': tables,
            'statistics': metadata.get('statistics', {})
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Extraction timed out after 5 minutes")
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# -------------------------------------------------------------------
# IMAGES
# -------------------------------------------------------------------
@app.post("/extract/images")
async def extract_images(
    file: UploadFile = File(...),
    min_quality: float = 0.3,
    min_width: int = 100,
    min_height: int = 100,
    workers: Optional[int] = None,     # CHANGED default -> auto if not provided
    token: str = Depends(verify_token)
):
    temp_dir = tempfile.mkdtemp()
    try:
        # Save upload
        pdf_path = os.path.join(temp_dir, "input.pdf")
        with _safe_read(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Output
        output_dir = os.path.join(temp_dir, "images")
        os.makedirs(output_dir, exist_ok=True)

        eff_workers = _resolve_workers(workers)
        cmd = [
            sys.executable, "enterprise_image_extractor.py",
            pdf_path,
            "--output-dir", output_dir,
            "--workers", str(eff_workers),
            "--min-quality", str(min_quality),
            "--min-width", str(min_width),
            "--min-height", str(min_height),
            "--clear-output",
        ]
        logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"Extraction failed: {result.stderr}")
            logger.warning("Continuing despite extraction errors")

        # Metadata
        metadata_path = os.path.join(output_dir, "extraction_metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with _safe_read(metadata_path, "r") as f:
                metadata = json.load(f)

        # ---------- PARALLEL IMAGE ENCODING ----------
        images = []
        meta_map = {}
        for img_info in metadata.get('images', []):
            fn = img_info.get('filename')
            if fn:
                meta_map[fn] = img_info

        image_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.png')]
        encode_workers = max(1, min(_resolve_workers(eff_workers), (os.cpu_count() or 4)))
        logger.info(f"Parallel encoding {len(image_files)} images with {encode_workers} threads")

        def build_image_record(img_file: str):
            img_path = os.path.join(output_dir, img_file)
            try:
                with _safe_read(img_path, "rb") as f:
                    img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode('ascii')
            except Exception as e:
                logger.error(f"Failed to read/encode image {img_path}: {e}")
                img_b64 = ""

            info = meta_map.get(img_file, {})
            rec = {
                'filename': img_file,
                'page_number': info.get('page_number', 0),
                'image_index': info.get('image_index', 0),
                'image_type': info.get('image_type', 'unknown'),
                'quality_score': info.get('quality_score', 0.5),
                'width': info.get('width', 0),
                'height': info.get('height', 0),
                'has_text': info.get('has_text', False),
                'text_content': info.get('text_content', ''),
                'image_base64': img_b64,          # original field (unchanged)
                'metadata': info
            }
            # Non-breaking mirror used by your cloud branch
            rec['base64_content'] = img_b64
            return rec

        with ThreadPoolExecutor(max_workers=encode_workers) as ex:
            futures = [ex.submit(build_image_record, f) for f in image_files]
            for fut in as_completed(futures):
                images.append(fut.result())
        # ---------- END PARALLEL ENCODING ----------

        return {
            'success': True,
            'images_count': len(images),
            'images': images,
            'statistics': metadata.get('statistics', {})
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Extraction timed out after 5 minutes")
    except Exception as e:
        logger.error(f"Extraction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# -------------------------------------------------------------------
# BOTH
# -------------------------------------------------------------------
@app.post("/extract/all")
async def extract_all(
    file: UploadFile = File(...),
    min_quality: float = 0.3,
    workers: Optional[int] = None,     # CHANGED default -> auto if not provided
    min_width: int = 100,
    min_height: int = 100,
    token: str = Depends(verify_token)
):
    temp_dir = tempfile.mkdtemp()
    try:
        # Save upload
        pdf_path = os.path.join(temp_dir, "input.pdf")
        with _safe_read(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_size = os.path.getsize(pdf_path)
        logger.info(f"Processing PDF: {file.filename}, Size: {file_size} bytes, Temp path: {pdf_path}")

        tables_dir = os.path.join(temp_dir, "pdf_tables")
        images_dir = os.path.join(temp_dir, "pdf_images")
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        all_results = []

        # ---- TABLES (unchanged shape) ----
        logger.info("Extracting tables...")
        eff_workers = _resolve_workers(workers)
        table_cmd = [
            sys.executable, "enterprise_table_extractor_full.py",
            pdf_path,
            "--output-dir", tables_dir,
            "--workers", str(eff_workers),
            "--min-quality", str(min_quality),
            "--clear-output",
        ]
        logger.info(f"Running command: {' '.join(table_cmd)}")

        try:
            table_result = subprocess.run(table_cmd, capture_output=True, text=True, timeout=300)
            logger.info(f"Table extraction exit code: {table_result.returncode}")
            logger.info(f"Table stdout (first 500 chars): {table_result.stdout[:500]}")
            if table_result.stderr:
                logger.error(f"Table stderr: {table_result.stderr[:1000]}")

            if table_result.returncode == 0:
                table_metadata_path = os.path.join(tables_dir, "extraction_metadata.json")
                if os.path.exists(table_metadata_path):
                    with _safe_read(table_metadata_path, "r") as f:
                        table_metadata = json.load(f)
                    logger.info(f"Found {len(table_metadata.get('tables', []))} tables in metadata")

                    for table_info in table_metadata.get('tables', []):
                        item = {
                            "type": "table",
                            "page": table_info['page_number'],
                            "index": table_info['table_index'],
                            "filePath": f"/data/pdf_tables/{table_info['filename']}",
                            "fileName": table_info['filename'],
                            "table_type": table_info.get('table_type', 'general_data'),
                            "quality_score": table_info.get('quality_score', 0.0),
                            "extraction_method": table_info.get('extraction_method', 'unknown'),
                            "rows": table_info.get('rows', 0),
                            "columns": table_info.get('columns', 0),
                            "size_bytes": table_info.get('size_bytes', 0),
                            "has_headers": table_info.get('has_headers', True),
                            "numeric_percentage": table_info.get('numeric_percentage', 0),
                            "empty_cell_percentage": table_info.get('empty_cell_percentage', 0),
                            "metadata": table_info.get('metadata', {}),
                            "mimeType": "text/csv"
                        }
                        csv_path = os.path.join(tables_dir, table_info['filename'])
                        if os.path.exists(csv_path):
                            try:
                                with _safe_read(csv_path, "r") as f:
                                    item["csv_content"] = f.read()
                            except Exception as e:
                                logger.error(f"Error reading CSV {csv_path}: {e}")
                                item["csv_content"] = ""
                        else:
                            logger.warning(f"CSV file not found: {csv_path}")
                            item["csv_content"] = ""
                        all_results.append(item)
                else:
                    logger.warning("No table metadata file found")
        except subprocess.TimeoutExpired:
            logger.error("Table extraction timed out")
        except Exception as e:
            logger.error(f"Table extraction error: {e}", exc_info=True)

        # ---- IMAGES (parallel base64 encoding) ----
        logger.info("Extracting images...")
        image_cmd = [
            sys.executable, "enterprise_image_extractor.py",
            pdf_path,
            "--output-dir", images_dir,
            "--workers", str(eff_workers),
            "--min-width", str(min_width),
            "--min-height", str(min_height),
            "--min-quality", str(min_quality),
            "--vector-threshold", "10",
            "--clear-output",
        ]
        logger.info(f"Running command: {' '.join(image_cmd)}")

        try:
            image_result = subprocess.run(image_cmd, capture_output=True, text=True, timeout=300)
            logger.info(f"Image extraction exit code: {image_result.returncode}")
            logger.info(f"Image stdout (first 500 chars): {image_result.stdout[:500]}")
            if image_result.stderr:
                logger.error(f"Image stderr: {image_result.stderr[:1000]}")

            if image_result.returncode == 0:
                image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]
                logger.info(f"Files in images directory: {image_files}")

                image_metadata_path = os.path.join(images_dir, "extraction_metadata.json")
                image_metadata = {}
                if os.path.exists(image_metadata_path):
                    with _safe_read(image_metadata_path, "r") as f:
                        image_metadata = json.load(f)
                logger.info(f"Found {len(image_metadata.get('images', []))} images in metadata")

                # Pre-index metadata and encode in parallel
                image_meta_map = {}
                for img_info in image_metadata.get('images', []):
                    fn = img_info.get('filename')
                    if fn:
                        image_meta_map[fn] = img_info

                encode_workers = max(1, min(_resolve_workers(eff_workers), (os.cpu_count() or 4)))
                logger.info(f"Parallel encoding {len(image_files)} images with {encode_workers} threads (/extract/all)")

                def build_result_item(img_file: str):
                    img_path = os.path.join(images_dir, img_file)
                    info = image_meta_map.get(img_file, {})
                    try:
                        with _safe_read(img_path, "rb") as f:
                            img_b64 = base64.b64encode(f.read()).decode('ascii')
                    except Exception as e:
                        logger.error(f"Failed to read/encode image {img_path}: {e}")
                        img_b64 = ""

                    return {
                        "type": "image",
                        "page": info.get('page_number', 0),
                        "index": info.get('image_index', 0),
                        "filePath": f"/data/pdf_images/{img_file}",
                        "fileName": img_file,
                        "image_type": info.get('image_type', 'general_image'),
                        "extraction_method": info.get('extraction_method', 'unknown'),
                        "quality_score": info.get('quality_score', 0.0),
                        "width": info.get('width', 0),
                        "height": info.get('height', 0),
                        "has_text": info.get('has_text', False),
                        "text_content": info.get('text_content', ''),
                        "caption": info.get('context', {}).get('caption'),
                        "figure_reference": info.get('context', {}).get('figure_reference'),
                        "visual_elements": info.get('visual_elements', {}),
                        "vector_count": info.get('vector_count'),
                        "enhancement_applied": info.get('enhancement_applied', False),
                        "mimeType": "image/png",
                        "base64_content": img_b64,  # cloud-normalized
                        "image_base64": img_b64     # original name
                    }

                with ThreadPoolExecutor(max_workers=encode_workers) as ex:
                    futures = [ex.submit(build_result_item, f) for f in image_files]
                    for fut in as_completed(futures):
                        all_results.append(fut.result())

        except subprocess.TimeoutExpired:
            logger.error("Image extraction timed out")
        except Exception as e:
            logger.error(f"Image extraction error: {e}", exc_info=True)

        # Sort results by page/index (original behavior)
        all_results.sort(key=lambda x: (x.get('page', 0), x.get('index', 0)))
        logger.info(f"Total results: {len(all_results)} items")

        return {
            "results": all_results,
            "count": len(all_results),
            "extraction_timestamp": datetime.now().isoformat(),
            "success": True
        }

    except Exception as e:
        logger.error(f"Extraction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# -------------------------------------------------------------------
# Env / System check (unchanged)
# -------------------------------------------------------------------
@app.get("/debug/check-environment")
async def check_environment():
    checks = {
        "python_version": sys.version,
        "current_directory": os.getcwd(),
        "scripts_exist": {
            "table_extractor": os.path.exists("enterprise_table_extractor_full.py"),
            "image_extractor": os.path.exists("enterprise_image_extractor.py")
        },
        "installed_packages": []
    }
    required_packages = [
        "pdfplumber", "pandas", "numpy", "camelot-py",
        "tabula-py", "PyMuPDF", "PIL", "cv2", "pytesseract"
    ]
    for package in required_packages:
        try:
            if package == "PyMuPDF": __import__("fitz")
            elif package == "PIL": __import__("PIL.Image")
            elif package == "cv2": __import__("cv2")
            elif package == "camelot-py": __import__("camelot")
            elif package == "tabula-py": __import__("tabula")
            else: __import__(package)
            checks["installed_packages"].append({"package": package, "installed": True})
        except ImportError:
            checks["installed_packages"].append({"package": package, "installed": False})

    checks["system_checks"] = {
        "java_available": shutil.which("java") is not None,
        "tesseract_available": shutil.which("tesseract") is not None
    }
    checks["files_in_directory"] = os.listdir(".")
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import pdfplumber; print('pdfplumber works')"],
            capture_output=True, text=True, timeout=5
        )
        checks["test_import"] = {
            "success": result.returncode == 0,
            "stdout": result.stdout, "stderr": result.stderr
        }
    except Exception as e:
        checks["test_import"] = {"error": str(e)}
    return checks

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
