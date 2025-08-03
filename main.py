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
            "table_extractor.py",
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
            "image_extractor.py",
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
