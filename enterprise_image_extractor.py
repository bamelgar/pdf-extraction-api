#!/usr/bin/env python3
"""
Enterprise Image Extractor v3.5
- Advanced extraction of images, charts, and diagrams from PDFs
- Multi-process parallel extraction for performance
- Context-aware image processing with OCR and quality metrics
- Comprehensive metadata generation
- Properly handles both bitmap and vector images
"""

import os
import sys
import json
import time
import argparse
import concurrent.futures
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import logging
import shutil
import uuid
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ImageExtractor')

# Import specialized extraction libraries
try:
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    import fitz  # PyMuPDF
    import cv2
    from pdf2image import convert_from_path
    import pytesseract
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    logger.error("Please install required packages: pip install pymupdf pdf2image pytesseract opencv-python-headless pillow numpy")
    sys.exit(1)

# Try to load optional dependencies
try:
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTFigure, LTImage
    has_pdfminer = True
except ImportError:
    logger.warning("pdfminer.six not installed, some extraction methods will be unavailable")
    has_pdfminer = False


class ImageExtractor:
    """Enterprise PDF image extraction with multi-method approach"""
    
    def __init__(self, 
                 pdf_path: str, 
                 output_dir: str,
                 min_width: int = 100,
                 min_height: int = 100,
                 min_quality: float = 0.4,
                 dpi: int = 300,
                 max_workers: int = 4,
                 vector_threshold: int = 10,
                 extract_text: bool = True,
                 enhance_images: bool = True,
                 clear_output: bool = False):
        """
        Initialize the image extractor
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            min_width: Minimum image width to extract
            min_height: Minimum image height to extract
            min_quality: Minimum image quality score (0-1)
            dpi: DPI for PDF rendering
            max_workers: Maximum number of worker processes
            vector_threshold: Minimum vector elements to classify as vector
            extract_text: Whether to extract text from images
            enhance_images: Whether to enhance image quality
            clear_output: Whether to clear output directory before extraction
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.min_width = min_width
        self.min_height = min_height
        self.min_quality = min_quality
        self.dpi = dpi
        self.max_workers = max_workers
        self.vector_threshold = vector_threshold
        self.extract_text = extract_text
        self.enhance_images = enhance_images
        self.clear_output = clear_output
        
        # Initialize state
        self.extracted_images = []
        self.pages_total = 0
        self.start_time = time.time()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Clear output directory if requested
        if self.clear_output:
            for f in os.listdir(self.output_dir):
                if f.endswith('.png') or f.endswith('.json'):
                    os.remove(os.path.join(self.output_dir, f))
            
        # Verify PDF exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Initialized image extractor for {pdf_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Min dimensions: {min_width}x{min_height}, Quality: {min_quality}, Workers: {max_workers}")

    def _extract_pymupdf(self) -> List[Dict[str, Any]]:
        """Extract images using PyMuPDF (main extraction method)"""
        images = []
        
        try:
            pdf_document = fitz.open(self.pdf_path)
            self.pages_total = len(pdf_document)
            
            for page_index in range(len(pdf_document)):
                page = pdf_document[page_index]
                page_number = page_index + 1
                
                # Get page dimensions
                page_width, page_height = page.rect.width, page.rect.height
                
                # Process images on this page
                image_list = page.get_images(full=True)
                
                # Track images for this page
                page_images = []
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = pdf_document.extract_image(xref)
                        
                        if not base_image:
                            continue
                            
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Convert to PIL Image for processing
                        try:
                            from io import BytesIO
                            pil_image = Image.open(BytesIO(image_bytes))
                        except Exception as e:
                            logger.warning(f"Failed to open image on page {page_number}: {e}")
                            continue
                            
                        # Check dimensions
                        width, height = pil_image.size
                        if width < self.min_width or height < self.min_height:
                            continue
                            
                        # Get image position on page
                        bbox = None
                        for img_obj in page.get_images(full=True):
                            if img_obj[0] == xref:
                                # Get all instances of this image on the page
                                instances = page.get_image_rects(img_obj)
                                if instances:
                                    # Use the first instance's bbox
                                    bbox = instances[0]
                                    break
                                    
                        # Compute relative position
                        position = {}
                        if bbox:
                            position = {
                                "x1": bbox.x0 / page_width,
                                "y1": bbox.y0 / page_height,
                                "x2": bbox.x2 / page_width,
                                "y2": bbox.y2 / page_height,
                                "width": bbox.width / page_width,
                                "height": bbox.height / page_height,
                                "area": (bbox.width * bbox.height) / (page_width * page_height)
                            }
                            
                        # Generate unique filename
                        timestamp = int(time.time() * 1000)
                        filename = f"img_p{page_number:03d}_i{img_index+1:03d}_{timestamp}.png"
                        filepath = os.path.join(self.output_dir, filename)
                        
                        # Save the image
                        pil_image.save(filepath, "PNG")
                        
                        # Compute image quality metrics
                        quality_score = self._compute_image_quality(pil_image)
                        
                        # Only process images that meet quality threshold
                        if quality_score < self.min_quality:
                            if os.path.exists(filepath):
                                os.remove(filepath)
                            continue
                            
                        # Extract text from image if enabled
                        text_content = ""
                        has_text = False
                        if self.extract_text:
                            text_content = self._extract_text_from_image(pil_image)
                            has_text = bool(text_content.strip())
                            
                        # Categorize image type
                        image_type = self._categorize_image(pil_image, text_content)
                        
                        # Enhance image if enabled
                        enhancement_applied = False
                        if self.enhance_images and image_type != "photo":
                            try:
                                enhanced = self._enhance_image(pil_image)
                                enhanced.save(filepath, "PNG")
                                enhancement_applied = True
                            except Exception as e:
                                logger.warning(f"Image enhancement failed: {e}")
                        
                        # Gather metadata
                        image_info = {
                            "page_number": page_number,
                            "image_index": img_index + 1,
                            "filename": filename,
                            "filepath": filepath,
                            "width": width,
                            "height": height,
                            "position": position,
                            "image_type": image_type,
                            "quality_score": quality_score,
                            "has_text": has_text,
                            "text_content": text_content,
                            "extraction_method": "pymupdf",
                            "enhancement_applied": enhancement_applied,
                            "vector_elements": 0,  # Not a vector image
                            "size_bytes": os.path.getsize(filepath)
                        }
                        
                        page_images.append(image_info)
                        
                    except Exception as e:
                        logger.error(f"Error extracting image {img_index+1} on page {page_number}: {e}")
                        traceback.print_exc()
                
                # Check if we also have vector graphics on this page
                vector_images = self._extract_vector_graphics(page, page_number)
                page_images.extend(vector_images)
                
                # Add all valid images from this page
                images.extend(page_images)
                    
            pdf_document.close()
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            traceback.print_exc()
            
        return images

    def _extract_vector_graphics(self, page: fitz.Page, page_number: int) -> List[Dict[str, Any]]:
        """Extract vector graphics from the page"""
        vector_images = []
        
        try:
            # Render the page to an image at high DPI
            pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi/72, self.dpi/72))
            
            # Convert to PIL image
            from io import BytesIO
            img_data = pix.tobytes("png")
            page_image = Image.open(BytesIO(img_data))
            
            # Get page dimensions
            page_width, page_height = page.rect.width, page.rect.height
            
            # Get all paths, shapes, and vector elements
            paths = page.get_drawings()
            
            if len(paths) >= self.vector_threshold:
                # Likely has vector graphics, save the rendered page
                timestamp = int(time.time() * 1000)
                filename = f"vector_p{page_number:03d}_{timestamp}.png"
                filepath = os.path.join(self.output_dir, filename)
                
                page_image.save(filepath, "PNG")
                
                # Extract text content near vector graphics
                text_content = self._extract_text_from_image(page_image)
                has_text = bool(text_content.strip())
                
                # Compute image quality metrics
                quality_score = self._compute_image_quality(page_image)
                
                # Categorize image type (likely chart or diagram)
                image_type = self._categorize_image(page_image, text_content)
                if "vector" not in image_type:
                    image_type = f"vector_{image_type}"
                
                # Gather metadata
                image_info = {
                    "page_number": page_number,
                    "image_index": 0,  # Special index for full page vector
                    "filename": filename,
                    "filepath": filepath,
                    "width": page_image.width,
                    "height": page_image.height,
                    "position": {
                        "x1": 0, "y1": 0, "x2": 1, "y2": 1,
                        "width": 1, "height": 1, "area": 1
                    },
                    "image_type": image_type,
                    "quality_score": quality_score,
                    "has_text": has_text,
                    "text_content": text_content,
                    "extraction_method": "vector_render",
                    "enhancement_applied": False,
                    "vector_elements": len(paths),
                    "size_bytes": os.path.getsize(filepath)
                }
                
                vector_images.append(image_info)
                
        except Exception as e:
            logger.error(f"Vector graphics extraction failed on page {page_number}: {e}")
            traceback.print_exc()
            
        return vector_images
        
    def _extract_embedded_images(self) -> List[Dict[str, Any]]:
        """Extract images using PDF2Image (backup method)"""
        images = []
        
        try:
            # Convert PDF pages to images
            pages = convert_from_path(self.pdf_path, dpi=self.dpi)
            
            for page_index, page_image in enumerate(pages):
                page_number = page_index + 1
                
                # Process this page image
                width, height = page_image.size
                
                # Generate unique filename
                timestamp = int(time.time() * 1000)
                filename = f"embedded_p{page_number:03d}_{timestamp}.png"
                filepath = os.path.join(self.output_dir, filename)
                
                # Save the image
                page_image.save(filepath, "PNG")
                
                # Compute image quality metrics
                quality_score = self._compute_image_quality(page_image)
                
                # Extract text from image if enabled
                text_content = ""
                has_text = False
                if self.extract_text:
                    text_content = self._extract_text_from_image(page_image)
                    has_text = bool(text_content.strip())
                    
                # Categorize image type
                image_type = self._categorize_image(page_image, text_content)
                
                # Gather metadata
                image_info = {
                    "page_number": page_number,
                    "image_index": 0,  # Special index for embedded image
                    "filename": filename,
                    "filepath": filepath,
                    "width": width,
                    "height": height,
                    "position": {
                        "x1": 0, "y1": 0, "x2": 1, "y2": 1,
                        "width": 1, "height": 1, "area": 1
                    },
                    "image_type": image_type,
                    "quality_score": quality_score,
                    "has_text": has_text,
                    "text_content": text_content,
                    "extraction_method": "embedded",
                    "enhancement_applied": False,
                    "vector_elements": 0,
                    "size_bytes": os.path.getsize(filepath)
                }
                
                images.append(image_info)
                
        except Exception as e:
            logger.error(f"PDF2Image extraction failed: {e}")
            traceback.print_exc()
            
        return images
    
    def _compute_image_quality(self, img: Image.Image) -> float:
        """Compute image quality score based on various metrics"""
        try:
            # Convert to grayscale for analysis
            gray = img.convert('L')
            
            # Get image dimensions
            width, height = img.size
            
            # Calculate metrics
            aspect_ratio = width / height if height > 0 else 0
            size_score = min(1.0, (width * height) / (1000 * 1000))
            
            # Check if image is not just white/black
            pixels = np.array(gray)
            std_dev = np.std(pixels)
            contrast_score = min(1.0, std_dev / 50)
            
            # Calculate average quality score
            quality_score = (size_score + contrast_score) / 2
            
            return quality_score
            
        except Exception as e:
            logger.warning(f"Quality computation failed: {e}")
            return 0.5  # Default middle quality
    
    def _extract_text_from_image(self, img: Image.Image) -> str:
        """Extract text from image using OCR"""
        try:
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            return ""
    
    def _categorize_image(self, img: Image.Image, text_content: str) -> str:
        """Categorize image type based on content and text"""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(img)
            
            # Check image dimensions
            height, width = img_array.shape[:2]
            
            # Check if image contains a face
            try:
                # Use OpenCV's face detection
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    return "portrait"
            except:
                pass
            
            # Check if image is likely a chart/graph
            chart_keywords = ['chart', 'graph', 'figure', 'fig', 'plot', 'axis', 'trend', 
                             'bar', 'pie', 'line', 'scatter', 'series', 'data', 'statistics']
            
            text_lower = text_content.lower()
            chart_score = sum(1 for keyword in chart_keywords if keyword in text_lower)
            
            if chart_score >= 2:
                return "chart"
                
            # Check if image is likely a diagram
            diagram_keywords = ['diagram', 'flow', 'process', 'architecture', 'model', 
                              'system', 'framework', 'structure', 'network', 'map']
            
            diagram_score = sum(1 for keyword in diagram_keywords if keyword in text_lower)
            
            if diagram_score >= 2:
                return "diagram"
                
            # Check if image is likely a table
            table_keywords = ['table', 'column', 'row', 'cell', 'grid', 'data', 'value']
            
            table_score = sum(1 for keyword in table_keywords if keyword in text_lower)
            
            if table_score >= 2:
                return "table"
            
            # Default to general image type
            return "general_image"
            
        except Exception as e:
            logger.warning(f"Image categorization failed: {e}")
            return "general_image"
    
    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Enhance image quality"""
        try:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Apply enhancements
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)
            
            return img
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return img
            
    def _extract_page_images(self, page_number: int) -> List[Dict[str, Any]]:
        """Extract images from a single page using multiple methods"""
        images = []
        
        try:
            # Open the PDF document
            pdf_document = fitz.open(self.pdf_path)
            
            # Get the specific page
            page = pdf_document[page_number - 1]
            
            # Process images on this page using PyMuPDF
            image_list = page.get_images(full=True)
            
            # Track images for this page
            page_images = []
            
            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = pdf_document.extract_image(xref)
                    
                    if not base_image:
                        continue
                        
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Convert to PIL Image for processing
                    try:
                        from io import BytesIO
                        pil_image = Image.open(BytesIO(image_bytes))
                    except Exception as e:
                        logger.warning(f"Failed to open image on page {page_number}: {e}")
                        continue
                        
                    # Check dimensions
                    width, height = pil_image.size
                    if width < self.min_width or height < self.min_height:
                        continue
                        
                    # Generate unique filename
                    timestamp = int(time.time() * 1000)
                    filename = f"img_p{page_number:03d}_i{img_index+1:03d}_{timestamp}.png"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    # Save the image
                    pil_image.save(filepath, "PNG")
                    
                    # Compute image quality metrics
                    quality_score = self._compute_image_quality(pil_image)
                    
                    # Only process images that meet quality threshold
                    if quality_score < self.min_quality:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                        continue
                        
                    # Extract text from image if enabled
                    text_content = ""
                    has_text = False
                    if self.extract_text:
                        text_content = self._extract_text_from_image(pil_image)
                        has_text = bool(text_content.strip())
                        
                    # Categorize image type
                    image_type = self._categorize_image(pil_image, text_content)
                    
                    # Enhance image if enabled
                    enhancement_applied = False
                    if self.enhance_images and image_type != "photo":
                        try:
                            enhanced = self._enhance_image(pil_image)
                            enhanced.save(filepath, "PNG")
                            enhancement_applied = True
                        except Exception as e:
                            logger.warning(f"Image enhancement failed: {e}")
                    
                    # Gather metadata
                    image_info = {
                        "page_number": page_number,
                        "image_index": img_index + 1,
                        "filename": filename,
                        "filepath": filepath,
                        "width": width,
                        "height": height,
                        "image_type": image_type,
                        "quality_score": quality_score,
                        "has_text": has_text,
                        "text_content": text_content,
                        "extraction_method": "pymupdf",
                        "enhancement_applied": enhancement_applied,
                        "vector_elements": 0,  # Not a vector image
                        "size_bytes": os.path.getsize(filepath)
                    }
                    
                    page_images.append(image_info)
                    
                except Exception as e:
                    logger.error(f"Error extracting image {img_index+1} on page {page_number}: {e}")
                    traceback.print_exc()
            
            # Check if we also have vector graphics on this page
            vector_images = self._extract_vector_graphics(page, page_number)
            page_images.extend(vector_images)
            
            # Add all valid images from this page
            images.extend(page_images)
                
            pdf_document.close()
            
        except Exception as e:
            logger.error(f"Page {page_number} extraction failed: {e}")
            traceback.print_exc()
            
        return images
    
    def extract_images(self) -> List[Dict[str, Any]]:
        """Extract images from the PDF using multiple methods"""
        all_images = []
        
        # Try PyMuPDF method first (best for most PDFs)
        pymupdf_images = self._extract_pymupdf()
        if pymupdf_images:
            all_images.extend(pymupdf_images)
            logger.info(f"Extracted {len(pymupdf_images)} images using PyMuPDF")
        
        # If we got very few images, try the embedded method as backup
        if len(all_images) < 2:
            embedded_images = self._extract_embedded_images()
            if embedded_images:
                all_images.extend(embedded_images)
                logger.info(f"Extracted {len(embedded_images)} images using embedded method")
        
        # Gather all extracted images
        self.extracted_images = all_images
        
        # Sort images by page and index
        self.extracted_images.sort(key=lambda x: (x.get("page_number", 0), x.get("image_index", 0)))
        
        # Generate extraction metadata
        metadata = {
            "pdf_path": self.pdf_path,
            "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_pages": self.pages_total,
            "images": self.extracted_images,
            "statistics": {
                "total_images": len(self.extracted_images),
                "extraction_time_seconds": time.time() - self.start_time,
                "methods": {
                    "pymupdf": len([img for img in self.extracted_images if img.get("extraction_method") == "pymupdf"]),
                    "vector_render": len([img for img in self.extracted_images if img.get("extraction_method") == "vector_render"]),
                    "embedded": len([img for img in self.extracted_images if img.get("extraction_method") == "embedded"])
                },
                "types": {
                    "chart": len([img for img in self.extracted_images if "chart" in img.get("image_type", "")]),
                    "diagram": len([img for img in self.extracted_images if "diagram" in img.get("image_type", "")]),
                    "table": len([img for img in self.extracted_images if "table" in img.get("image_type", "")]),
                    "portrait": len([img for img in self.extracted_images if "portrait" in img.get("image_type", "")]),
                    "general": len([img for img in self.extracted_images if "general" in img.get("image_type", "")])
                }
            }
        }
        
        # Save metadata to JSON file
        metadata_path = os.path.join(self.output_dir, "extraction_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Extraction complete. Found {len(self.extracted_images)} images.")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return self.extracted_images
        
    def extract_images_parallel(self) -> List[Dict[str, Any]]:
        """Extract images in parallel using multiple processes"""
        all_images = []
        
        try:
            # Open PDF to get page count
            pdf_document = fitz.open(self.pdf_path)
            total_pages = len(pdf_document)
            pdf_document.close()
            
            self.pages_total = total_pages
            logger.info(f"Processing {total_pages} pages with {self.max_workers} workers")
            
            # Process pages in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit tasks for each page
                future_to_page = {
                    executor.submit(self._extract_page_images, page_num): page_num 
                    for page_num in range(1, total_pages + 1)
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        page_images = future.result()
                        if page_images:
                            all_images.extend(page_images)
                            logger.info(f"Page {page_num}: Extracted {len(page_images)} images")
                    except Exception as e:
                        logger.error(f"Page {page_num} processing failed: {e}")
                        traceback.print_exc()
            
            # Gather all extracted images
            self.extracted_images = all_images
            
            # Sort images by page and index
            self.extracted_images.sort(key=lambda x: (x.get("page_number", 0), x.get("image_index", 0)))
            
            # Generate extraction metadata
            metadata = {
                "pdf_path": self.pdf_path,
                "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_pages": self.pages_total,
                "images": self.extracted_images,
                "statistics": {
                    "total_images": len(self.extracted_images),
                    "extraction_time_seconds": time.time() - self.start_time,
                    "methods": {
                        "pymupdf": len([img for img in self.extracted_images if img.get("extraction_method") == "pymupdf"]),
                        "vector_render": len([img for img in self.extracted_images if img.get("extraction_method") == "vector_render"]),
                        "embedded": len([img for img in self.extracted_images if img.get("extraction_method") == "embedded"])
                    },
                    "types": {
                        "chart": len([img for img in self.extracted_images if "chart" in img.get("image_type", "")]),
                        "diagram": len([img for img in self.extracted_images if "diagram" in img.get("image_type", "")]),
                        "table": len([img for img in self.extracted_images if "table" in img.get("image_type", "")]),
                        "portrait": len([img for img in self.extracted_images if "portrait" in img.get("image_type", "")]),
                        "general": len([img for img in self.extracted_images if "general" in img.get("image_type", "")])
                    }
                }
            }
            
            # Save metadata to JSON file
            metadata_path = os.path.join(self.output_dir, "extraction_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Parallel extraction complete. Found {len(self.extracted_images)} images.")
            logger.info(f"Metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Parallel extraction failed: {e}")
            traceback.print_exc()
            
        return self.extracted_images


def main():
    """Main entry point for the CLI tool"""
    parser = argparse.ArgumentParser(description="Enterprise Image Extractor for PDFs")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output-dir", "-o", default="./pdf_images", help="Directory to save extracted images")
    parser.add_argument("--min-width", type=int, default=100, help="Minimum image width to extract")
    parser.add_argument("--min-height", type=int, default=100, help="Minimum image height to extract")
    parser.add_argument("--min-quality", type=float, default=0.4, help="Minimum image quality score (0-1)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF rendering")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of worker processes")
    parser.add_argument("--vector-threshold", type=int, default=10, help="Minimum vector elements to classify as vector")
    parser.add_argument("--no-text", action="store_false", dest="extract_text", help="Disable text extraction from images")
    parser.add_argument("--no-enhance", action="store_false", dest="enhance_images", help="Disable image enhancement")
    parser.add_argument("--clear-output", action="store_true", help="Clear output directory before extraction")
    
    args = parser.parse_args()
    
    try:
        # Create extractor
        extractor = ImageExtractor(
            pdf_path=args.pdf_path,
            output_dir=args.output_dir,
            min_width=args.min_width,
            min_height=args.min_height,
            min_quality=args.min_quality,
            dpi=args.dpi,
            max_workers=args.workers,
            vector_threshold=args.vector_threshold,
            extract_text=args.extract_text,
            enhance_images=args.enhance_images,
            clear_output=args.clear_output
        )
        
        # Extract images using parallel processing if multiple workers
        if args.workers > 1:
            extractor.extract_images_parallel()
        else:
            extractor.extract_images()
            
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        traceback.print_exc()
        sys.exit(1)
        
    # Success
    sys.exit(0)


if __name__ == "__main__":
    main()
