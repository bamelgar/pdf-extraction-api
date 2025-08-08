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
            "--workers", str(workers),
            "--min-quality", str(min_quality),
            "--clear-output"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Extraction failed: {result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=f"Extraction failed: {result.stderr}"
            )
        
        # Read the metadata file
        metadata_path = os.path.join(output_dir, "extraction_metadata.json")
        if not os.path.exists(metadata_path):
            raise HTTPException(
                status_code=500,
                detail="No metadata file generated"
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Process each table
        tables = []
        for table_info in metadata.get('tables', []):
            csv_path = os.path.join(output_dir, table_info['filename'])
            
            if os.path.exists(csv_path):
                # Read CSV content
                with open(csv_path, 'r', encoding='utf-8') as f:
                    csv_content = f.read()
                
                # Read as base64
                with open(csv_path, 'rb') as f:
                    csv_base64 = base64.b64encode(f.read()).decode('utf-8')
                
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
        raise HTTPException(
            status_code=504,
            detail="Extraction timed out after 5 minutes"
        )
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

@app.post("/extract/images")
async def extract_images(
    file: UploadFile = File(...),
    min_quality: float = 0.3,
    min_width: int = 100,
    min_height: int = 100,
    workers: int = 4,
    token: str = Depends(verify_token)
):
    """Extract images from PDF using the enterprise extractor"""
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
            "enterprise_image_extractor.py",
            pdf_path,
            "--output-dir", output_dir,
            "--workers", str(workers),
            "--min-quality", str(min_quality),
            "--min-width", str(min_width),
            "--min-height", str(min_height),
            "--clear-output"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Extraction failed: {result.stderr}")
            # Don't fail completely, log the error
            logger.warning("Continuing despite extraction errors")
        
        # Read the metadata file if it exists
        metadata_path = os.path.join(output_dir, "extraction_metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Process each image
        images = []
        image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        
        for img_file in image_files:
            img_path = os.path.join(output_dir, img_file)
            
            # Read image as base64
            with open(img_path, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Find metadata for this image
            img_metadata = {}
            for img_info in metadata.get('images', []):
                if img_info.get('filename') == img_file:
                    img_metadata = img_info
                    break
            
            images.append({
                'filename': img_file,
                'page_number': img_metadata.get('page_number', 0),
                'image_index': img_metadata.get('image_index', 0),
                'image_type': img_metadata.get('image_type', 'unknown'),
                'quality_score': img_metadata.get('quality_score', 0.5),
                'width': img_metadata.get('width', 0),
                'height': img_metadata.get('height', 0),
                'has_text': img_metadata.get('has_text', False),
                'text_content': img_metadata.get('text_content', ''),
                'image_base64': img_base64,
                'metadata': img_metadata
            })
        
        return {
            'success': True,
            'images_count': len(images),
            'images': images,
            'statistics': metadata.get('statistics', {})
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Extraction timed out after 5 minutes"
        )
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

@app.post("/extract/all")
async def extract_all(
    file: UploadFile = File(...),
    min_quality: float = 0.3,
    workers: int = 4,
    min_width: int = 100,
    min_height: int = 100,
    token: str = Depends(verify_token)
):
    """Extract both tables and images from PDF - mimics the original orchestrator script"""
    temp_dir = tempfile.mkdtemp()
    
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
        
        # Extract tables
        logger.info("Extracting tables...")
        table_cmd = [
            sys.executable,
            "enterprise_table_extractor_full.py",
            pdf_path,
            "--output-dir", tables_dir,
            "--workers", str(workers),
            "--min-quality", str(min_quality),
            "--clear-output"
        ]
        
        logger.info(f"Running command: {' '.join(table_cmd)}")
        
        try:
            table_result = subprocess.run(
                table_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            logger.info(f"Table extraction exit code: {table_result.returncode}")
            logger.info(f"Table stdout (first 500 chars): {table_result.stdout[:500]}")
            if table_result.stderr:
                logger.error(f"Table stderr: {table_result.stderr[:1000]}")
            
            if table_result.returncode == 0:
                # List files in output directory
                table_files = os.listdir(tables_dir)
                logger.info(f"Files in tables directory: {table_files}")
                
                # Read table metadata
                table_metadata_path = os.path.join(tables_dir, "extraction_metadata.json")
                if os.path.exists(table_metadata_path):
                    with open(table_metadata_path, 'r') as f:
                        table_metadata = json.load(f)
                    
                    logger.info(f"Found {len(table_metadata.get('tables', []))} tables in metadata")
                    
                    for table_info in table_metadata.get('tables', []):
                        # Format exactly as the original Python script
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
                        
                        all_results.append(result_item)
                else:
                    logger.warning("No table metadata file found")
                        
        except subprocess.TimeoutExpired:
            logger.error("Table extraction timed out")
        except Exception as e:
            logger.error(f"Table extraction error: {e}", exc_info=True)
        
        # Extract images
        logger.info("Extracting images...")
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
        
        logger.info(f"Running command: {' '.join(image_cmd)}")
        
        try:
            image_result = subprocess.run(
                image_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            logger.info(f"Image extraction exit code: {image_result.returncode}")
            logger.info(f"Image stdout (first 500 chars): {image_result.stdout[:500]}")
            if image_result.stderr:
                logger.error(f"Image stderr: {image_result.stderr[:1000]}")
            
            if image_result.returncode == 0:
                # List files in output directory
                image_files = os.listdir(images_dir)
                logger.info(f"Files in images directory: {image_files}")
                
                # Read image metadata
                image_metadata_path = os.path.join(images_dir, "extraction_metadata.json")
                if os.path.exists(image_metadata_path):
                    with open(image_metadata_path, 'r') as f:
                        image_metadata = json.load(f)
                    
                    logger.info(f"Found {len(image_metadata.get('images', []))} images in metadata")
                    
                    for img_info in image_metadata.get('images', []):
                        # Format exactly as the original Python script
                        result_item = {
                            "type": "image",
                            "page": img_info['page_number'],
                            "index": img_info['image_index'],
                            "filePath": f"/data/pdf_images/{img_info['filename']}",
                            "fileName": img_info['filename'],
                            "image_type": img_info.get('image_type', 'general_image'),
                            "extraction_method": img_info.get('extraction_method', 'unknown'),
                            "quality_score": img_info.get('quality_score', 0.0),
                            "width": img_info.get('width', 0),
                            "height": img_info.get('height', 0),
                            "has_text": img_info.get('has_text', False),
                            "text_content": img_info.get('text_content', ''),
                            "caption": img_info.get('context', {}).get('caption'),
                            "figure_reference": img_info.get('context', {}).get('figure_reference'),
                            "visual_elements": img_info.get('visual_elements', {}),
                            "vector_count": img_info.get('vector_count'),
                            "enhancement_applied": img_info.get('enhancement_applied', False),
                            "mimeType": "image/png"
                        }
                        
                        all_results.append(result_item)
                else:
                    logger.warning("No image metadata file found")
                        
        except subprocess.TimeoutExpired:
            logger.error("Image extraction timed out")
        except Exception as e:
            logger.error(f"Image extraction error: {e}", exc_info=True)
        
        # Sort results by page and index (like the original script)
        all_results.sort(key=lambda x: (x.get('page', 0), x.get('index', 0)))
        
        logger.info(f"Total results: {len(all_results)} items")
        
        # CRITICAL FIX: Return wrapped response to prevent n8n from unwrapping single-item arrays
        return {
            "results": all_results,
            "count": len(all_results),
            "extraction_timestamp": datetime.now().isoformat(),
            "success": True
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
        "tabula-py", "PyMuPDF", "PIL", "cv2", "pytesseract"
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
