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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Extraction API", version="1.0.0")

# Log system info
cpu_count = os.cpu_count()
logger.info(f"System has {cpu_count} CPUs available")

@app.get("/")
async def root():
    return {"message": "PDF Extraction API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "cpus": os.cpu_count()}

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
    
    except subprocess.TimeoutExpired:
        logger.error("Table extraction timed out")
    except Exception as e:
        logger.error(f"Table extraction error: {e}")
    
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
                    
                    # Add base64 data to each image
                    for img_meta in images_list:
                        img_file = images_dir / img_meta['filename']
                        if img_file.exists():
                            with open(img_file, 'rb') as f:
                                img_data = f.read()
                                img_base64 = base64.b64encode(img_data).decode('utf-8')
                                img_meta['image_base64'] = img_base64
                                logger.info(f"Added base64 for {img_meta['filename']}, size: {len(img_base64)} chars")
                    
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
            "image_base64": image.get("image_base64"),
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
