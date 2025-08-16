from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import shutil
import json
import base64
from typing import List, Dict, Optional
import subprocess
import sys
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="PDF Extraction API", version="1.0.0")

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

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return credentials.credentials

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "PDF Extraction API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
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
        "message": "File received successfully"
    }

@app.post("/extract/tables")
async def extract_tables(
    file: UploadFile = File(...),
    min_quality: float = 0.3,
    workers: int = 4,
    token: str = Depends(verify_token)
):
    """Extract tables from PDF using the enterprise extractor"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded file
        pdf_path = os.path.join(temp_dir, "input.pdf")
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "tables")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the extraction script
        cmd = [
            sys.executable,
            "enterprise_table_extractor_full.py",
            pdf_path,
            "--output-dir", output_dir,
            "--min-quality", str(min_quality),
            "--workers", str(workers)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            logger.error(f"Extraction failed: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Extraction failed: {result.stderr}")
        
        # Load metadata
        metadata_path = os.path.join(output_dir, "extraction_metadata.json")
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=500, detail="No metadata file generated")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Collect all CSV files with their content
        csv_files = {}
        for file in os.listdir(output_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(output_dir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    csv_files[file] = {
                        'content': f.read(),
                        'size': os.path.getsize(file_path)
                    }
        
        return {
            "success": True,
            "metadata": metadata,
            "csv_files": csv_files,
            "extraction_stats": {
                "total_tables": len(metadata.get('tables', [])),
                "total_csv_files": len(csv_files),
                "execution_time": metadata.get('execution_time', 0)
            }
        }
    
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Extraction timeout")
    except Exception as e:
        logger.error(f"Error during extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.post("/extract/images")
async def extract_images(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """Extract images from PDF"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded file
        pdf_path = os.path.join(temp_dir, "input.pdf")
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Create output directory
        output_dir = os.path.join(temp_dir, "images")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the extraction script
        cmd = [
            sys.executable,
            "enterprise_image_extractor_unique_ids.py",
            pdf_path,
            "--output-dir", output_dir
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            logger.error(f"Image extraction failed: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Image extraction failed: {result.stderr}")
        
        # Load metadata
        metadata_path = os.path.join(output_dir, "image_extraction_metadata.json")
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=500, detail="No metadata file generated")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Collect all images with base64 encoding
        images = {}
        for file in os.listdir(output_dir):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(output_dir, file)
                with open(file_path, 'rb') as f:
                    images[file] = {
                        'base64': base64.b64encode(f.read()).decode('utf-8'),
                        'size': os.path.getsize(file_path)
                    }
        
        return {
            "success": True,
            "metadata": metadata,
            "images": images,
            "extraction_stats": {
                "total_images": len(metadata.get('images', [])),
                "total_image_files": len(images),
                "execution_time": metadata.get('execution_time', 0)
            }
        }
    
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Image extraction timeout")
    except Exception as e:
        logger.error(f"Error during image extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.post("/extract/all")
async def extract_all(
    file: UploadFile = File(...),
    token: str = Depends(verify_token)
):
    """Extract both tables and images from PDF and return complete results"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded file
        pdf_path = os.path.join(temp_dir, "input.pdf")
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Create directories for outputs
        tables_dir = os.path.join(temp_dir, "tables")
        images_dir = os.path.join(temp_dir, "images")
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # Run table extraction
        table_cmd = [
            sys.executable,
            "enterprise_table_extractor_full.py",
            pdf_path,
            "--output-dir", tables_dir,
            "--min-quality", "0.3",
            "--workers", "4"
        ]
        
        table_result = subprocess.run(
            table_cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if table_result.returncode != 0:
            logger.error(f"Table extraction failed: {table_result.stderr}")
        
        # Run image extraction
        image_cmd = [
            sys.executable,
            "enterprise_image_extractor_unique_ids.py",
            pdf_path,
            "--output-dir", images_dir
        ]
        
        image_result = subprocess.run(
            image_cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if image_result.returncode != 0:
            logger.error(f"Image extraction failed: {image_result.stderr}")
        
        # Load table metadata
        table_metadata = {}
        table_metadata_path = os.path.join(tables_dir, "extraction_metadata.json")
        if os.path.exists(table_metadata_path):
            with open(table_metadata_path, 'r') as f:
                table_metadata = json.load(f)
        
        # Load image metadata
        image_metadata = {}
        image_metadata_path = os.path.join(images_dir, "image_extraction_metadata.json")
        if os.path.exists(image_metadata_path):
            with open(image_metadata_path, 'r') as f:
                image_metadata = json.load(f)
        
        # Build comprehensive results matching n8n expected format
        all_results = []
        
        # Process tables - INCLUDING CSV CONTENT (critical for cloud!)
        for table_info in table_metadata.get('tables', []):
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
            
            # CRITICAL: Add CSV content for cloud environment
            csv_path = os.path.join(tables_dir, table_info['filename'])
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        result_item["csv_content"] = f.read()
                    logger.info(f"Added CSV content for {table_info['filename']}")
                except Exception as e:
                    logger.error(f"Error reading CSV {csv_path}: {e}")
                    result_item["csv_content"] = ""
            else:
                logger.warning(f"CSV file not found: {csv_path}")
                result_item["csv_content"] = ""
            
            all_results.append(result_item)
        
        # Process images
        for image_info in image_metadata.get('images', []):
            result_item = {
                "type": "image",
                "page": image_info['page_number'],
                "index": image_info['image_index'],
                "filePath": f"/data/pdf_images/{image_info['filename']}",
                "fileName": image_info['filename'],
                "image_type": image_info.get('image_type', 'unknown'),
                "width": image_info.get('width', 0),
                "height": image_info.get('height', 0),
                "format": image_info.get('format', 'png'),
                "size_bytes": image_info.get('size_bytes', 0),
                "unique_id": image_info.get('unique_id', ''),
                "metadata": image_info.get('metadata', {}),
                "mimeType": f"image/{image_info.get('format', 'png')}"
            }
            
            # Add base64 image data
            image_path = os.path.join(images_dir, image_info['filename'])
            if os.path.exists(image_path):
                try:
                    with open(image_path, 'rb') as f:
                        result_item["base64"] = base64.b64encode(f.read()).decode('utf-8')
                    logger.info(f"Added base64 for {image_info['filename']}")
                except Exception as e:
                    logger.error(f"Error reading image {image_path}: {e}")
                    result_item["base64"] = ""
            
            all_results.append(result_item)
        
        return {
            "success": True,
            "total_items": len(all_results),
            "results": all_results,
            "summary": {
                "tables": len(table_metadata.get('tables', [])),
                "images": len(image_metadata.get('images', [])),
                "total_pages": max(
                    table_metadata.get('total_pages', 0),
                    image_metadata.get('total_pages', 0)
                ),
                "extraction_time": {
                    "tables": table_metadata.get('execution_time', 0),
                    "images": image_metadata.get('execution_time', 0)
                }
            }
        }
    
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Extraction timeout")
    except Exception as e:
        logger.error(f"Error during extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.get("/debug/environment")
async def debug_environment(token: str = Depends(verify_token)):
    """Debug endpoint to check environment and dependencies"""
    checks = {
        "python_version": sys.version,
        "working_dir": os.getcwd(),
        "temp_dir": tempfile.gettempdir(),
        "env_vars": {
            "API_KEY_SET": bool(os.environ.get("API_KEY")),
            "PATH": os.environ.get("PATH", "")
        }
    }
    
    # Check if scripts exist
    scripts = [
        "enterprise_table_extractor_full.py",
        "enterprise_image_extractor_unique_ids.py"
    ]
    
    for script in scripts:
        checks[f"script_{script}"] = os.path.exists(script)
    
    # Test imports
    import_tests = ["pdfplumber", "pandas", "numpy", "cv2", "PIL", "PyMuPDF"]
    for module in import_tests:
        try:
            __import__(module)
            checks[f"import_{module}"] = True
        except ImportError as e:
            checks[f"import_{module}"] = str(e)
    
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
    
    return checks

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint that doesn't require auth"""
    return {
        "message": "API is working!",
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
