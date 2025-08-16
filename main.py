import os
import tempfile
import subprocess
import json
import shutil
import base64
from pathlib import Path
from typing import List, Dict, Any
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import requests
from datetime import datetime
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Extraction API", version="1.0.0")

# Supabase configuration
SUPABASE_URL = "https://hdxxfknkzodgwzrcddug.supabase.co"
SUPABASE_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhkeHhma25rem9kZ3d6cmNkZHVnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTkyMDM0MiwiZXhwIjoyMDY3NDk2MzQyfQ.Oxj9cWTDm8hTo1zReh8JwPUnCMCSrAoRyyutON_iaNE"
SUPABASE_BUCKET = "public-images"

def test_supabase_connection() -> Dict[str, Any]:
    """Test Supabase connection and storage access"""
    logger.info("Testing Supabase connection...")
    
    try:
        # Create a small test image
        test_image = Image.new('RGB', (10, 10), color='red')
        img_buffer = BytesIO()
        test_image.save(img_buffer, format='PNG')
        test_data = img_buffer.getvalue()
        
        # Test filename
        test_filename = f"test_connection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        # Test upload
        upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{test_filename}"
        headers = {
            "Authorization": f"Bearer {SUPABASE_TOKEN}",
            "Content-Type": "image/png"
        }
        
        response = requests.put(upload_url, data=test_data, headers=headers, timeout=10)
        
        if response.status_code in [200, 201]:
            # Test public URL access
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{test_filename}"
            check_response = requests.head(public_url, timeout=10)
            
            # Clean up test file
            delete_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{test_filename}"
            requests.delete(delete_url, headers={"Authorization": f"Bearer {SUPABASE_TOKEN}"})
            
            logger.info("✅ Supabase connection test successful")
            return {
                "status": "success",
                "upload_status": response.status_code,
                "public_access": check_response.status_code == 200,
                "test_filename": test_filename,
                "bucket": SUPABASE_BUCKET,
                "message": "Supabase storage is working correctly"
            }
        else:
            logger.error(f"❌ Supabase upload failed: {response.status_code} - {response.text}")
            return {
                "status": "error",
                "upload_status": response.status_code,
                "error": response.text,
                "bucket": SUPABASE_BUCKET,
                "message": "Failed to upload to Supabase"
            }
            
    except Exception as e:
        logger.error(f"❌ Supabase connection test failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "bucket": SUPABASE_BUCKET,
            "message": "Connection test failed"
        }

def upload_image_to_supabase(image_path: Path, filename: str) -> str:
    """Upload image to Supabase Storage and return public URL"""
    try:
        # Read image data
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Upload URL
        upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
        
        # Headers
        headers = {
            "Authorization": f"Bearer {SUPABASE_TOKEN}",
            "Content-Type": "image/png"
        }
        
        # Upload to Supabase
        response = requests.put(upload_url, data=image_data, headers=headers)
        
        if response.status_code in [200, 201]:
            # Return public URL
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{filename}"
            logger.info(f"Successfully uploaded {filename} to Supabase")
            return public_url
        else:
            logger.error(f"Failed to upload {filename}: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error uploading {filename} to Supabase: {e}")
        return None

# Log system info
cpu_count = os.cpu_count()
logger.info(f"System has {cpu_count} CPUs available")

@app.get("/")
async def root():
    return {"message": "PDF Extraction API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "cpus": os.cpu_count()}

@app.get("/test-supabase")
async def test_supabase():
    """Test Supabase connection and storage functionality"""
    result = test_supabase_connection()
    return result

@app.get("/diagnostics")
async def diagnostics():
    """Comprehensive system diagnostics"""
    logger.info("Running system diagnostics...")
    
    diagnostics_result = {
        "system": {
            "cpus": os.cpu_count(),
            "python_version": subprocess.run(["python3", "--version"], capture_output=True, text=True).stdout.strip(),
            "disk_space": subprocess.run(["df", "-h", "/tmp"], capture_output=True, text=True).stdout.strip()
        },
        "supabase": test_supabase_connection(),
        "dependencies": {
            "requests_available": True,
            "pil_available": True
        }
    }
    
    # Test if extractor scripts exist
    scripts = ["enterprise_table_extractor_full.py", "enterprise_image_extractor.py"]
    diagnostics_result["scripts"] = {}
    
    for script in scripts:
        script_exists = os.path.exists(script)
        diagnostics_result["scripts"][script] = {
            "exists": script_exists,
            "executable": os.access(script, os.X_OK) if script_exists else False
        }
    
    return diagnostics_result

@app.post("/extract/all")
async def extract_all(
    file: UploadFile = File(...),
    min_quality: float = Form(0.3),
    workers: int = Form(4),
    page_limit: int = Form(None),  # ADD: Page limit support
    skip_ocr: bool = Form(False)
):
    """Extract both tables and images from PDF"""
    
    # Use 16 workers if available (CHANGE: from 4 to 16)
    max_workers = min(16, os.cpu_count() or 1)
    actual_workers = min(workers, max_workers)
    logger.info(f"Using {actual_workers} workers for extraction (CPUs available: {cpu_count})")
    
    temp_dir = None
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Save uploaded file
        pdf_path = temp_path / "input.pdf"
        with open(pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Processing PDF: {file.filename}, Size: {len(content)} bytes, Temp path: {pdf_path}")
        
        # Prepare directories
        tables_dir = temp_path / "pdf_tables"
        images_dir = temp_path / "pdf_images"
        
        # Extract tables
        logger.info("Extracting tables...")
        table_cmd = [
            "/usr/local/bin/python3.11",
            "enterprise_table_extractor_full.py",
            str(pdf_path),
            "--output-dir", str(tables_dir),
            "--workers", str(actual_workers),
            "--min-quality", str(min_quality),
            "--clear-output"
        ]
        
        # ADD: Page limit support for tables
        if page_limit:
            table_cmd.extend(["--page-limit", str(page_limit)])
        
        logger.info(f"Running command: {' '.join(table_cmd)}")
        
        # CHANGE: Increase timeout for large PDFs (from 600 to 1800 seconds)
        table_result = subprocess.run(
            table_cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes for table extraction
        )
        
        tables_extracted = []
        if table_result.returncode == 0:
            logger.info("Table extraction completed successfully")
            if tables_dir.exists():
                # Load table metadata
                metadata_file = tables_dir / "extraction_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        table_metadata = json.load(f)
                        tables_extracted = table_metadata.get('tables', [])
        else:
            logger.error("Table extraction failed")
            logger.error(f"Table stderr: {table_result.stderr}")
    
        # Extract images
        logger.info("Extracting images...")
        image_cmd = [
            "/usr/local/bin/python3.11",
            "enterprise_image_extractor.py",
            str(pdf_path),
            "--output-dir", str(images_dir),
            "--workers", str(actual_workers),
            "--min-width", "100",
            "--min-height", "100",
            "--min-quality", str(min_quality),
            "--vector-threshold", "10",
            "--clear-output"
        ]
        
        # ADD: Page limit support for images
        if page_limit:
            image_cmd.extend(["--page-limit", str(page_limit)])
        
        if skip_ocr:
            image_cmd.append("--no-ocr")
        
        logger.info(f"Running command: {' '.join(image_cmd)}")
        
        images_extracted = []
        try:
            # CHANGE: Increase timeout for images (from 600 to 900 seconds)
            image_result = subprocess.run(
                image_cmd,
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes for image extraction
            )
            
            logger.info(f"Image extraction exit code: {image_result.returncode}")
            logger.info(f"Image stdout (first 500 chars): {image_result.stdout[:500]}")
            
            if image_result.returncode != 0:
                logger.error(f"Image stderr: {image_result.stderr}")
            
            if images_dir.exists():
                # Get list of image files
                image_files = list(images_dir.glob("*.png"))
                logger.info(f"Files in images directory: {[f.name for f in image_files]}")
                
                # Load image metadata
                metadata_file = images_dir / "extraction_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        image_metadata = json.load(f)
                        images_list = image_metadata.get('images', [])
                        logger.info(f"Found {len(images_list)} images in metadata")
                        
                        # Upload images to Supabase instead of embedding base64
                        supabase_stats = {
                            "total_images": len(images_list),
                            "successful_uploads": 0,
                            "failed_uploads": 0,
                            "upload_errors": []
                        }
                        
                        for img_meta in images_list:
                            img_file = images_dir / img_meta['filename']
                            if img_file.exists():
                                # Create unique filename with timestamp
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                unique_filename = f"{timestamp}_{img_meta['filename']}"
                                
                                # Upload to Supabase
                                supabase_url = upload_image_to_supabase(img_file, unique_filename)
                                
                                if supabase_url:
                                    img_meta['supabase_url'] = supabase_url
                                    img_meta['uploaded_filename'] = unique_filename
                                    supabase_stats["successful_uploads"] += 1
                                    logger.info(f"✅ Uploaded {img_meta['filename']} to Supabase as {unique_filename}")
                                else:
                                    img_meta['supabase_url'] = None
                                    img_meta['uploaded_filename'] = None
                                    supabase_stats["failed_uploads"] += 1
                                    supabase_stats["upload_errors"].append(img_meta['filename'])
                                    logger.warning(f"❌ Failed to upload {img_meta['filename']} to Supabase")
                        
                        # Log Supabase upload statistics
                        logger.info(f"📊 Supabase Upload Summary: {supabase_stats['successful_uploads']}/{supabase_stats['total_images']} successful")
                        if supabase_stats["failed_uploads"] > 0:
                            logger.warning(f"⚠️  Failed uploads: {supabase_stats['upload_errors']}")
                        
                        images_extracted = images_list
        
        except subprocess.TimeoutExpired:
            logger.error("Image extraction timed out")
        except Exception as e:
            logger.error(f"Image extraction error: {e}")
        
        # Combine results
        all_results = []
        
        # Add tables
        for table in tables_extracted:
            result_item = {
                "type": "table",
                "page": table.get("page_number"),
                "index": table.get("table_index"),
                "extraction_method": table.get("extraction_method"),
                "quality_score": table.get("quality_score"),
                "table_type": table.get("table_type"),
                "rows": table.get("rows"),
                "columns": table.get("columns"),
                "filename": table.get("filename"),
                "metadata": table
            }
            all_results.append(result_item)
        
        # Add images
        for image in images_extracted:
            result_item = {
                "type": "image",
                "page": image.get("page_number"),
                "index": image.get("image_index"),
                "extraction_method": image.get("extraction_method"),
                "image_type": image.get("image_type"),
                "quality_score": image.get("quality_score"),
                "width": image.get("width"),
                "height": image.get("height"),
                "filename": image.get("filename"),
                "supabase_url": image.get("supabase_url"),
                "uploaded_filename": image.get("uploaded_filename"),
                "metadata": image
            }
            all_results.append(result_item)
        
        logger.info(f"Total results: {len(all_results)} items")
        
        # Cleanup
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        return [{
            "results": all_results,
            "count": len(all_results),
            "extraction_timestamp": subprocess.run(["date", "-Iseconds"], capture_output=True, text=True).stdout.strip(),
            "success": True,
            "workers_used": actual_workers,
            "page_limit": page_limit,  # ADD: Include page limit in response
            "skip_ocr": skip_ocr
        }]

    except subprocess.TimeoutExpired:
        logger.error("Table extraction timed out")
    except Exception as e:
        logger.error(f"Table extraction error: {e}")
    
    # Cleanup on error
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    return JSONResponse(
        status_code=500,
        content={"error": "Extraction failed", "success": False}
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
