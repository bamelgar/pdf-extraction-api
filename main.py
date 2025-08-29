"""
PDF Extraction API - Production Version with Supabase Integration
===============================================================
CHANGES IN THIS VERSION:
1. SKIPS TABLE EXTRACTION ENTIRELY (commented out but preserved)
2. Uploads images to Supabase during extraction
3. Returns URLs instead of base64 when Supabase succeeds
4. Processes all images by default (limiters set to 0)
5. Fallback to base64 if Supabase fails
6. Multiple URL fields for better compatibility with RAG systems
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
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

def upload_image_to_supabase(image_path: Path, filename: str) -> str:
    """Upload image to Supabase Storage and return public URL"""
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
        
        # Upload to Supabase
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

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials.credentials

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
        "TABLES_ENABLED": "NO - Tables extraction skipped entirely"
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

@app.post("/extract/all")
async def extract_all(
    file: UploadFile = File(...),
    min_quality: float = 0.3,
    workers: int = 8,
    min_width: int = 100,
    min_height: int = 100,
    # LIMITERS (optional)
    limit_tables: Optional[int] = None,
    limit_images: Optional[int] = None,
    page_limit: Optional[int] = None,
    token: str = Depends(verify_token)
):
    """Extract both tables and images from PDF"""
    temp_dir = tempfile.mkdtemp()

    # Resolve temp limits
    eff_limit_tables = _limit_from(limit_tables, "TEMP_LIMIT_TABLES")
    eff_limit_images = _limit_from(limit_images, "TEMP_LIMIT_IMAGES")
    
    if eff_limit_tables > 0:
        logger.info(f"[LIMIT] /all: Tables packaging limited to {eff_limit_tables}.")
    if eff_limit_images > 0:
        logger.info(f"[LIMIT] /all: Images packaging limited to {eff_limit_images}.")
    else:
        logger.info("[LIMITERS DISABLED] Processing all images")

    try:
        # Save uploaded file
        pdf_path = os.path.join(temp_dir, "input.pdf")
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Log file details
        file_size = os.path.getsize(pdf_path)
        logger.info(f"Processing PDF: {file.filename}, Size: {file_size} bytes, Temp path: {pdf_path}")

        # Create output directories
        tables_dir = os.path.join(temp_dir, "pdf_tables")
        images_dir = os.path.join(temp_dir, "pdf_images")
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        all_results = []

        # ============================================
        # TABLE EXTRACTION - SKIPPED BUT PRESERVED
        # ============================================
        logger.info("📋 Table extraction is DISABLED")
        
        """
        # This section is commented out to skip table extraction entirely
        
        logger.info("📋 Extracting tables...")
        table_cmd = [
            sys.executable,
            "enterprise_table_extractor_full.py",
            pdf_path,
            "--output-dir", tables_dir,
            "--workers", str(workers),
            "--min-quality", str(min_quality),
            "--clear-output"
        ]
        
        # Add page limit if specified
        if page_limit:
            table_cmd.extend(["--page-limit", str(page_limit)])
            
        logger.info(f"Running command: {' '.join(table_cmd)}")

        try:
            table_result = subprocess.run(
                table_cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )

            logger.info(f"Table extraction exit code: {table_result.returncode}")
            logger.info(f"Table stdout (first 500 chars): {table_result.stdout[:500]}")
            if table_result.stderr:
                logger.error(f"Table stderr: {table_result.stderr[:1000]}")

            if table_result.returncode == 0:
                # Read table metadata
                table_metadata_path = os.path.join(tables_dir, "extraction_metadata.json")
                if os.path.exists(table_metadata_path):
                    with open(table_metadata_path, 'r') as f:
                        table_metadata = json.load(f)

                    logger.info(f"Found {len(table_metadata.get('tables', []))} tables in metadata")

                    table_list = table_metadata.get('tables', [])
                    if eff_limit_tables > 0:
                        table_list = table_list[:eff_limit_tables]

                    for table_info in table_list:
                        result_item = {
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
                                with open(csv_path, 'r', encoding='utf-8') as f:
                                    result_item["csv_content"] = f.read()
                            except Exception as e:
                                logger.error(f"Error reading CSV {csv_path}: {e}")
                                result_item["csv_content"] = ""
                        else:
                            logger.warning(f"CSV file not found: {csv_path}")
                            result_item["csv_content"] = ""

                        all_results.append(result_item)
                else:
                    logger.warning("No table metadata file found")

        except subprocess.TimeoutExpired:
            logger.error("Table extraction timed out")
        except Exception as e:
            logger.error(f"Table extraction error: {e}", exc_info=True)
        """

        # ============================================
        # IMAGE EXTRACTION - WITH SUPABASE UPLOAD
        # ============================================
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
        
        # Add page limit if specified
        if page_limit:
            image_cmd.extend(["--page-limit", str(page_limit)])
            
        logger.info(f"Running command: {' '.join(image_cmd)}")

        try:
            image_result = subprocess.run(
                image_cmd,
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes timeout
            )

            logger.info(f"Image extraction exit code: {image_result.returncode}")
            logger.info(f"Image stdout (first 500 chars): {image_result.stdout[:500]}")
            if image_result.stderr:
                logger.error(f"Image stderr: {image_result.stderr[:1000]}")

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
                        "mimeType": "image/png"
                    }
                    
                    if supabase_url:
                        # Supabase upload successful - ADD MULTIPLE URL FIELDS
                        result_item['supabase_url'] = supabase_url
                        result_item['url'] = supabase_url  # Add standard URL field
                        result_item['image_url'] = supabase_url  # Add image-specific URL field
                        result_item['uploaded_filename'] = unique_filename
                        logger.info(f"✅ Image {img_file} uploaded to Supabase: {supabase_url}")
                    else:
                        # Fallback to base64
                        with open(img_path, 'rb') as f:
                            img_base64 = base64.b64encode(f.read()).decode('utf-8')
                        result_item['base64_content'] = img_base64
                        logger.info(f"⚠️ Image {img_file} using base64 fallback")

                    all_results.append(result_item)
            else:
                logger.warning("Image extractor returned non-zero code")

        except subprocess.TimeoutExpired:
            logger.error("Image extraction timed out")
        except Exception as e:
            logger.error(f"Image extraction error: {e}", exc_info=True)

        # Sort results by page and index
        all_results.sort(key=lambda x: (x.get('page', 0), x.get('index', 0)))
        
        # Count tables and images
        table_count = sum(1 for item in all_results if item.get('type') == 'table')
        image_count = sum(1 for item in all_results if item.get('type') == 'image')
        
        logger.info(f"Total results: {len(all_results)} items ({table_count} tables, {image_count} images)")

        # Return wrapped response
        return {
            "results": all_results,
            "count": len(all_results),
            "tables_count": table_count,
            "images_count": image_count,
            "extraction_timestamp": datetime.now().isoformat(),
            "success": True,
            "supabase_enabled": bool(SUPABASE_URL and SUPABASE_TOKEN),
            "page_limit": page_limit
        }

    except Exception as e:
        logger.error(f"Extraction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Extraction error: {str(e)}"
        )
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.get("/debug/check-environment")
async def check_environment():
    """Check if Python environment is set up correctly"""
    checks = {
        "python_version": sys.version,
        "current_directory": os.getcwd(),
        "scripts_exist": {
            "table_extractor": os.path.exists("enterprise_table_extractor_full.py"),
            "image_extractor": os.path.exists("enterprise_image_extractor.py")
        },
        "installed_packages": []
    }

    # Check for required packages
    required_packages = [
        "pdfplumber", "pandas", "numpy", "camelot-py",
        "tabula-py", "PyMuPDF", "PIL", "cv2", "pytesseract", "requests"
    ]

    for package in required_packages:
        try:
            if package == "PyMuPDF":
                __import__("fitz")
            elif package == "PIL":
                __import__("PIL.Image")
            elif package == "cv2":
                __import__("cv2")
            elif package == "camelot-py":
                __import__("camelot")
            elif package == "tabula-py":
                __import__("tabula")
            else:
                __import__(package)
            checks["installed_packages"].append({"package": package, "installed": True})
        except ImportError:
            checks["installed_packages"].append({"package": package, "installed": False})

    # Check for system dependencies
    checks["system_checks"] = {
        "java_available": shutil.which("java") is not None,
        "tesseract_available": shutil.which("tesseract") is not None
    }

    # List files in current directory
    checks["files_in_directory"] = os.listdir(".")

    # Test simple extraction
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import pdfplumber; print('pdfplumber works')"],
            capture_output=True,
            text=True,
            timeout=5
        )
        checks["test_import"] = {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        checks["test_import"] = {"error": str(e)}

    # LIMITER STATUS
    checks["limiter_status"] = {
        "TESTING_IMAGE_LIMIT": os.environ.get("TESTING_IMAGE_LIMIT", "0"),
        "TESTING_TABLE_LIMIT": os.environ.get("TESTING_TABLE_LIMIT", "0"),
        "TEMP_LIMIT_TABLES": os.environ.get("TEMP_LIMIT_TABLES", "0"),
        "TEMP_LIMIT_IMAGES": os.environ.get("TEMP_LIMIT_IMAGES", "0"),
        "TABLES_EXTRACTION": "DISABLED COMPLETELY"
    }
    
    # SUPABASE STATUS
    checks["supabase_status"] = {
        "SUPABASE_URL": "configured" if SUPABASE_URL else "not set",
        "SUPABASE_TOKEN": "configured" if SUPABASE_TOKEN else "not set",
        "SUPABASE_BUCKET": SUPABASE_BUCKET,
        "ready_for_upload": bool(SUPABASE_URL and SUPABASE_TOKEN)
    }

    return checks

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint that doesn't require auth"""
    return {
        "message": "API is working! - Production Mode with Supabase (Tables DISABLED)",
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "tables_enabled": False,
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_TOKEN)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
