#!/usr/bin/env python3
"""
Enterprise PDF Image Extractor v1.0+ (Enhanced)
- Parallel processing with ThreadPoolExecutor
- Atomic file writes for images
- Comprehensive file verification
- Thread-safe operations
- Progress tracking
"""

import os
import sys
import json
import logging
import hashlib
import re
import tempfile
import shutil
import threading
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF
from io import BytesIO

# Try importing optional libraries
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("OpenCV not available, some features limited")

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    print("Tesseract not available, OCR disabled")

try:
    import easyocr
    HAS_EASYOCR = True
    reader = easyocr.Reader(['en'])
except ImportError:
    HAS_EASYOCR = False
    print("EasyOCR not available")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ImageMetadata:
    """Metadata for extracted images"""
    filename: str
    page_number: int
    image_index: int
    image_type: str
    extraction_method: str  # 'embedded' or 'vector_render'
    width: int
    height: int
    quality_score: float
    has_text: bool
    text_content: str
    visual_elements: Dict[str, Any]
    extraction_timestamp: str
    file_size: int
    dpi: int
    color_mode: str
    enhancement_applied: bool
    context: Dict[str, Any]
    vector_count: Optional[int] = None

class ImageClassifier:
    """Classify images by content type"""
    
    @staticmethod
    def classify_image(image: Image.Image, text_content: str = "", vector_count: int = 0) -> Tuple[str, Dict]:
        """Classify image type and extract relevant metadata"""
        
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Basic image properties
        width, height = image.size
        aspect_ratio = width / height if height > 0 else 0
        
        # Color analysis
        if len(img_array.shape) == 3:
            color_variance = np.var(img_array)
            mean_colors = np.mean(img_array, axis=(0, 1))
        else:
            color_variance = np.var(img_array)
            mean_colors = [np.mean(img_array)]
        
        # Edge detection for structure analysis
        edges = ImageClassifier._detect_edges(image)
        edge_density = np.sum(edges > 128) / edges.size if edges is not None else 0
        
        # Text analysis from OCR
        text_lower = text_content.lower() if text_content else ""
        
        # Vector graphics hint
        has_many_vectors = vector_count > 50
        
        # Classification logic
        classifications = {
            'chart': {
                'indicators': ['axis', 'legend', 'data', 'series', '%', 'chart', 'graph', 'exhibit',
                             'revenue', 'growth', 'trend', 'performance', 'quarterly', 'annual'],
                'edge_density_range': (0.05, 0.3),
                'aspect_ratio_range': (0.5, 2.0),
                'vector_hint': has_many_vectors,
                'metadata_extractors': ['chart_type', 'axis_labels', 'data_series', 'exhibit_number']
            },
            'diagram': {
                'indicators': ['flow', 'process', 'step', 'arrow', 'box', 'workflow',
                             'architecture', 'structure', 'hierarchy', 'relationship'],
                'edge_density_range': (0.1, 0.4),
                'aspect_ratio_range': (0.3, 3.0),
                'vector_hint': has_many_vectors,
                'metadata_extractors': ['diagram_type', 'components', 'flow_direction']
            },
            'infographic': {
                'indicators': ['infographic', 'data', 'visualization', 'icon', 'trend', 'statistic',
                             'comparison', 'timeline', 'fact', 'metric'],
                'edge_density_range': (0.05, 0.5),
                'aspect_ratio_range': (0.3, 3.0),
                'vector_hint': has_many_vectors,
                'metadata_extractors': ['info_sections', 'key_points', 'data_categories']
            },
            'table_image': {
                'indicators': ['table', 'row', 'column', 'cell', 'grid', 'spreadsheet'],
                'edge_density_range': (0.2, 0.6),
                'aspect_ratio_range': (0.5, 2.0),
                'metadata_extractors': ['table_structure', 'cell_count', 'header_detection']
            },
            'screenshot': {
                'indicators': ['screenshot', 'window', 'interface', 'button', 'menu', 'toolbar',
                             'application', 'software', 'screen'],
                'edge_density_range': (0.1, 0.5),
                'aspect_ratio_range': (0.5, 2.0),
                'metadata_extractors': ['ui_elements', 'application_type']
            },
            'logo': {
                'indicators': ['logo', 'brand', 'trademark', 'Â®', 'â„¢', 'copyright', 'Â©'],
                'edge_density_range': (0.0, 0.3),
                'aspect_ratio_range': (0.5, 2.0),
                'metadata_extractors': ['brand_name', 'logo_type']
            },
            'photograph': {
                'indicators': ['photo', 'image', 'picture'],
                'edge_density_range': (0.0, 0.2),
                'aspect_ratio_range': (0.3, 3.0),
                'metadata_extractors': ['subject', 'scene_type']
            },
            'scientific_figure': {
                'indicators': ['figure', 'fig.', 'experiment', 'result', 'scale', 'Î¼m', 'nm',
                             'microscopy', 'gel', 'blot', 'staining', 'fluorescence'],
                'edge_density_range': (0.05, 0.4),
                'aspect_ratio_range': (0.5, 2.0),
                'metadata_extractors': ['figure_type', 'scale_info', 'annotations', 'methodology']
            }
        }
        
        scores = {}
        for img_type, config in classifications.items():
            score = 0
            
            # Check text indicators
            for indicator in config['indicators']:
                if indicator in text_lower:
                    score += 2
            
            # Check edge density
            edge_min, edge_max = config['edge_density_range']
            if edge_min <= edge_density <= edge_max:
                score += 1
            
            # Check aspect ratio
            ar_min, ar_max = config['aspect_ratio_range']
            if ar_min <= aspect_ratio <= ar_max:
                score += 0.5
            
            # Vector hint bonus
            if config.get('vector_hint', False):
                score += 3
            
            scores[img_type] = score
        
        # Get best match
        best_type = max(scores.items(), key=lambda x: x[1])[0]
        if scores[best_type] == 0:
            best_type = 'general_image'
        
        # Extract metadata based on type
        metadata = ImageClassifier._extract_type_specific_metadata(
            image, text_content, best_type, classifications.get(best_type, {})
        )
        
        return best_type, metadata
    
    @staticmethod
    def _detect_edges(image: Image.Image) -> Optional[np.ndarray]:
        """Detect edges in image"""
        if HAS_OPENCV:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            return edges
        else:
            # Fallback to PIL edge detection
            return np.array(image.convert('L').filter(ImageFilter.FIND_EDGES))
    
    @staticmethod
    def _extract_type_specific_metadata(image: Image.Image, text: str, 
                                      img_type: str, config: Dict) -> Dict:
        """Extract metadata specific to image type"""
        metadata = {'image_classification': img_type}
        
        if img_type == 'chart':
            # Try to detect chart type
            if any(word in text.lower() for word in ['bar', 'column']):
                metadata['chart_type'] = 'bar_chart'
            elif any(word in text.lower() for word in ['line', 'trend']):
                metadata['chart_type'] = 'line_chart'
            elif any(word in text.lower() for word in ['pie', 'donut']):
                metadata['chart_type'] = 'pie_chart'
            elif any(word in text.lower() for word in ['scatter', 'bubble']):
                metadata['chart_type'] = 'scatter_plot'
            elif 'heat' in text.lower():
                metadata['chart_type'] = 'heatmap'
            else:
                metadata['chart_type'] = 'unknown'
            
            # Extract exhibit number if present
            exhibit_match = re.search(r'exhibit\s*(\d+)', text.lower())
            if exhibit_match:
                metadata['exhibit_number'] = exhibit_match.group(1)
            
            # Look for data period
            year_match = re.findall(r'\b(19|20)\d{2}\b', text)
            if year_match:
                metadata['years_referenced'] = year_match
        
        elif img_type == 'diagram':
            if 'flow' in text.lower():
                metadata['diagram_type'] = 'flowchart'
            elif 'process' in text.lower():
                metadata['diagram_type'] = 'process_diagram'
            elif 'architecture' in text.lower():
                metadata['diagram_type'] = 'architecture_diagram'
            else:
                metadata['diagram_type'] = 'general_diagram'
        
        elif img_type == 'table_image':
            # Count grid lines for table structure
            metadata['estimated_rows'] = ImageClassifier._estimate_table_rows(image)
            metadata['estimated_columns'] = ImageClassifier._estimate_table_cols(image)
        
        elif img_type == 'scientific_figure':
            # Look for common scientific figure elements
            if any(word in text.lower() for word in ['western', 'blot']):
                metadata['figure_type'] = 'western_blot'
            elif 'microscopy' in text.lower() or 'magnification' in text.lower():
                metadata['figure_type'] = 'microscopy'
            elif 'gel' in text.lower():
                metadata['figure_type'] = 'gel_electrophoresis'
            else:
                metadata['figure_type'] = 'general_scientific'
            
            # Extract scale information
            scale_match = re.search(r'(\d+)\s*(Î¼m|nm|mm|cm)', text)
            if scale_match:
                metadata['scale'] = scale_match.group(0)
        
        return metadata
    
    @staticmethod
    def _estimate_table_rows(image: Image.Image) -> int:
        """Estimate number of rows in a table image"""
        if HAS_OPENCV:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            if lines is not None:
                horizontal_lines = [l for l in lines if abs(l[0][1] - l[0][3]) < 5]
                return len(horizontal_lines)
        return 0
    
    @staticmethod
    def _estimate_table_cols(image: Image.Image) -> int:
        """Estimate number of columns in a table image"""
        if HAS_OPENCV:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            if lines is not None:
                vertical_lines = [l for l in lines if abs(l[0][0] - l[0][2]) < 5]
                return len(vertical_lines)
        return 0

class QualityAnalyzer:
    """Analyze and score image quality"""
    
    @staticmethod
    def calculate_quality_score(image: Image.Image) -> Tuple[float, Dict]:
        """Calculate comprehensive quality score"""
        metrics = {
            'resolution': QualityAnalyzer._score_resolution(image),
            'sharpness': QualityAnalyzer._score_sharpness(image),
            'contrast': QualityAnalyzer._score_contrast(image),
            'brightness': QualityAnalyzer._score_brightness(image),
            'noise': QualityAnalyzer._score_noise(image)
        }
        
        # Weighted average
        weights = {
            'resolution': 0.3,
            'sharpness': 0.2,
            'contrast': 0.2,
            'brightness': 0.2,
            'noise': 0.1
        }
        
        quality_score = sum(metrics[k] * weights[k] for k in metrics)
        
        return quality_score, metrics
    
    @staticmethod
    def _score_resolution(image: Image.Image) -> float:
        """Score based on image resolution"""
        width, height = image.size
        pixels = width * height
        
        if pixels >= 1920 * 1080:  # Full HD or better
            return 1.0
        elif pixels >= 1280 * 720:  # HD
            return 0.8
        elif pixels >= 640 * 480:   # VGA
            return 0.6
        elif pixels >= 320 * 240:   # QVGA
            return 0.4
        else:
            return 0.2
    
    @staticmethod
    def _score_sharpness(image: Image.Image) -> float:
        """Score based on image sharpness using Laplacian variance"""
        gray = image.convert('L')
        array = np.array(gray)
        
        if HAS_OPENCV:
            laplacian = cv2.Laplacian(array, cv2.CV_64F)
            variance = laplacian.var()
            # Normalize variance to 0-1 range
            return min(variance / 1000, 1.0)
        else:
            # Fallback: use edge detection
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_array = np.array(edges)
            return np.mean(edge_array) / 255
    
    @staticmethod
    def _score_contrast(image: Image.Image) -> float:
        """Score based on image contrast"""
        gray = image.convert('L')
        array = np.array(gray)
        
        # Calculate standard deviation as contrast measure
        std_dev = np.std(array)
        # Normalize to 0-1 range
        return min(std_dev / 127.5, 1.0)
    
    @staticmethod
    def _score_brightness(image: Image.Image) -> float:
        """Score based on image brightness"""
        gray = image.convert('L')
        array = np.array(gray)
        
        mean_brightness = np.mean(array)
        # Optimal brightness is around 127 (middle of 0-255 range)
        # Score decreases as we move away from optimal
        distance_from_optimal = abs(mean_brightness - 127.5) / 127.5
        return 1.0 - distance_from_optimal
    
    @staticmethod
    def _score_noise(image: Image.Image) -> float:
        """Score based on image noise (lower noise = higher score)"""
        if HAS_OPENCV:
            array = np.array(image.convert('L'))
            # Apply Gaussian blur and calculate difference
            blurred = cv2.GaussianBlur(array, (5, 5), 0)
            noise = np.mean(np.abs(array.astype(float) - blurred.astype(float)))
            # Normalize (lower noise = higher score)
            return max(1.0 - (noise / 50), 0)
        else:
            # Fallback: always return decent score
            return 0.8

class ImageEnhancer:
    """Enhance image quality for better extraction"""
    
    @staticmethod
    def enhance_image(image: Image.Image, image_type: str) -> Image.Image:
        """Apply type-specific enhancements"""
        enhanced = image.copy()
        
        if image_type in ['chart', 'diagram', 'table_image']:
            # Enhance contrast for better line detection
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.5)
            
            # Sharpen for clearer edges
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(2.0)
        
        elif image_type == 'screenshot':
            # Mild sharpening for text clarity
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.3)
        
        elif image_type == 'photograph':
            # Balance brightness
            enhancer = ImageEnhance.Brightness(enhanced)
            gray = enhanced.convert('L')
            mean_brightness = np.mean(np.array(gray))
            if mean_brightness < 100:
                enhanced = enhancer.enhance(1.2)
            elif mean_brightness > 155:
                enhanced = enhancer.enhance(0.8)
        
        return enhanced

class TextExtractor:
    """Extract text from images using OCR"""
    
    @staticmethod
    def extract_text(image: Image.Image, lang: str = 'eng') -> str:
        """Extract text using available OCR methods"""
        text = ""
        
        # Try Tesseract first
        if HAS_TESSERACT:
            try:
                text = pytesseract.image_to_string(image, lang=lang)
                text = text.strip()
            except Exception as e:
                logger.debug(f"Tesseract OCR failed: {e}")
        
        # Try EasyOCR as fallback
        if not text and HAS_EASYOCR:
            try:
                result = reader.readtext(np.array(image))
                text = ' '.join([item[1] for item in result])
            except Exception as e:
                logger.debug(f"EasyOCR failed: {e}")
        
        return text

class EnterpriseImageExtractor:
    """Enterprise-grade PDF image extractor with parallel processing"""
    
    def __init__(self, pdf_path: str, output_dir: str = "/data/pdf_images",
                 config: Optional[Dict] = None):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = config or {}
        self.min_size = (
            self.config.get('min_width', 100),
            self.config.get('min_height', 100)
        )
        self.min_quality_score = self.config.get('min_quality_score', 0.3)
        self.enable_ocr = self.config.get('enable_ocr', True)
        self.enable_enhancement = self.config.get('enable_enhancement', True)
        self.save_metadata = self.config.get('save_metadata', True)
        self.vector_threshold = self.config.get('vector_threshold', 10)
        # OPTIMIZED: Use all available CPUs up to 16 for Professional plan
        self.max_workers = self.config.get('max_workers', min(16, os.cpu_count() or 1))
        self.page_limit = self.config.get('page_limit', None)  # Add page limit support for testing
        self.clear_output = self.config.get('clear_output', False)
        
        # Results storage (thread-safe)
        self.extracted_images: List[ImageMetadata] = []
        self._images_lock = threading.Lock()
        
        self.extraction_stats = {
            'total_pages': 0,
            'pages_with_images': 0,
            'pages_with_vector_graphics': 0,
            'embedded_images_found': 0,
            'vector_graphics_found': 0,
            'image_types_found': defaultdict(int),
            'quality_distribution': defaultdict(int),
            'ocr_success_rate': 0,
            'total_extraction_time': 0,
            'extraction_errors': []
        }
        
        # Clear output if requested
        if self.clear_output:
            for file in self.output_dir.glob('*.png'):
                file.unlink()
            for file in self.output_dir.glob('*.json'):
                file.unlink()
    
    def _atomic_save_image(self, image: Image.Image, filepath: Path) -> bool:
        """Atomically save image to prevent corruption"""
        try:
            # Create temp file
            fd, tmp_path = tempfile.mkstemp(suffix='.png.tmp', dir=str(filepath.parent))
            os.close(fd)
            
            # Save to temp file
            image.save(tmp_path, 'PNG', optimize=True)
            
            # Verify temp file
            test_img = Image.open(tmp_path)
            test_img.verify()
            
            # Atomic rename
            shutil.move(tmp_path, str(filepath))
            return True
            
        except Exception as e:
            logger.error(f"Atomic save failed for {filepath}: {e}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
    
    def extract_all_images(self):
        """Main extraction method with parallel processing"""
        start_time = datetime.now()
        
        logger.info(f"Starting enhanced image extraction with {self.max_workers} workers")
        
        # Open PDF to get page count
        doc = fitz.open(self.pdf_path)
        total_pages = len(doc)
        if self.page_limit:
            total_pages = min(self.page_limit, total_pages)
            logger.info(f"Page limit set: processing first {total_pages} pages only")
        self.extraction_stats['total_pages'] = total_pages
        doc.close()
        
        # Process pages in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit pages up to limit
            future_to_page = {
                executor.submit(self._process_page_safe, page_num): page_num
                for page_num in range(self.extraction_stats['total_pages'])
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page_images = future.result()
                    if page_images:
                        self.extraction_stats['pages_with_images'] += 1
                        for img_info in page_images:
                            self._save_image(img_info)
                except Exception as e:
                    logger.error(f"Failed to process page {page_num + 1}: {e}")
                    self.extraction_stats['extraction_errors'].append({
                        'page': page_num + 1,
                        'error': str(e)
                    })
        
        # Calculate final statistics
        self.extraction_stats['total_extraction_time'] = (
            datetime.now() - start_time
        ).total_seconds()
        
        # Calculate OCR success rate
        images_with_text = sum(1 for img in self.extracted_images if img.has_text)
        total_images = len(self.extracted_images)
        self.extraction_stats['ocr_success_rate'] = (
            images_with_text / total_images if total_images > 0 else 0
        )
        
        # Verify saved files
        verification_issues = self._verify_saved_files()
        if verification_issues:
            logger.warning(f"Found {len(verification_issues)} verification issues")
            self.extraction_stats['verification_issues'] = verification_issues
        
        # Save metadata
        if self.save_metadata:
            self._save_extraction_metadata()
        
        logger.info(f"Extraction complete: {len(self.extracted_images)} images extracted")
        logger.info(f"  - Embedded images: {self.extraction_stats['embedded_images_found']}")
        logger.info(f"  - Vector graphics: {self.extraction_stats['vector_graphics_found']}")
        
        return self.extracted_images
    
    def _process_page_safe(self, page_num: int) -> List[Dict]:
        """Thread-safe wrapper for page processing"""
        try:
            return self._process_page(page_num)
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}", exc_info=True)
            raise
    
    def _process_page(self, page_num: int) -> List[Dict]:
        """Process a single page for images"""
        doc = fitz.open(self.pdf_path)
        page = doc[page_num]
        page_images = []
        
        # Extract embedded images
        embedded_images = self._extract_embedded_images(page, page_num + 1)
        if embedded_images:
            page_images.extend(embedded_images)
            self.extraction_stats['embedded_images_found'] += len(embedded_images)
        
        # Extract vector graphics
        vector_graphics = self._extract_vector_graphics(page, page_num + 1)
        if vector_graphics:
            page_images.extend(vector_graphics)
            self.extraction_stats['vector_graphics_found'] += len(vector_graphics)
            self.extraction_stats['pages_with_vector_graphics'] += 1
        
        doc.close()
        return page_images
    
    def _extract_embedded_images(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """Extract traditional embedded images from a page"""
        images = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                # Get image data
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                else:  # CMYK
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(BytesIO(img_data))
                
                # Check minimum size
                if image.width < self.min_size[0] or image.height < self.min_size[1]:
                    continue
                
                # Calculate quality score
                quality_score, quality_metrics = QualityAnalyzer.calculate_quality_score(image)
                
                if quality_score < self.min_quality_score:
                    continue
                
                # Extract text if OCR enabled
                text_content = ""
                if self.enable_ocr:
                    text_content = TextExtractor.extract_text(image)
                
                # Classify image
                image_type, type_metadata = ImageClassifier.classify_image(image, text_content)
                
                # Enhance image if enabled
                if self.enable_enhancement:
                    image = ImageEnhancer.enhance_image(image, image_type)
                
                # Get image position on page
                try:
                    img_rect = page.get_image_bbox(img)
                    position = {
                        'x': img_rect.x0,
                        'y': img_rect.y0,
                        'width': img_rect.width,
                        'height': img_rect.height
                    }
                except:
                    position = {'x': 0, 'y': 0, 'width': image.width, 'height': image.height}
                
                # Extract context
                context = self._extract_image_context(page, img_rect if 'img_rect' in locals() else None)
                
                # Create image info
                img_info = {
                    'image': image,
                    'page_number': page_num,
                    'image_index': img_index + 1,
                    'image_type': image_type,
                    'extraction_method': 'embedded',
                    'quality_score': quality_score,
                    'quality_metrics': quality_metrics,
                    'type_metadata': type_metadata,
                    'text_content': text_content,
                    'position': position,
                    'context': context
                }
                
                images.append(img_info)
                
            except Exception as e:
                logger.error(f"Failed to extract embedded image {img_index} from page {page_num}: {e}")
        
        return images
    
    def _extract_vector_graphics(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """Extract vector graphics by rendering page"""
        graphics = []
        
        # Get vector drawing count
        drawings = page.get_drawings()
        vector_count = len(drawings)
        
        # Only extract if significant vector content
        if vector_count >= self.vector_threshold:
            try:
                # Render page at high resolution
                mat = fitz.Matrix(3, 3)  # 3x zoom for quality
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(BytesIO(img_data))
                
                # Extract text if OCR enabled
                text_content = ""
                if self.enable_ocr:
                    text_content = TextExtractor.extract_text(image)
                
                # Classify image with vector hint
                image_type, type_metadata = ImageClassifier.classify_image(
                    image, text_content, vector_count
                )
                
                # Calculate quality score
                quality_score, quality_metrics = QualityAnalyzer.calculate_quality_score(image)
                
                # Extract context
                context = self._extract_page_context(page)
                
                # Create image info
                img_info = {
                    'image': image,
                    'page_number': page_num,
                    'image_index': 1,  # Vector renders are one per page
                    'image_type': image_type,
                    'extraction_method': 'vector_render',
                    'quality_score': quality_score,
                    'quality_metrics': quality_metrics,
                    'type_metadata': type_metadata,
                    'text_content': text_content,
                    'vector_count': vector_count,
                    'position': {
                        'x': 0,
                        'y': 0,
                        'width': image.width,
                        'height': image.height
                    },
                    'context': context
                }
                
                graphics.append(img_info)
                
            except Exception as e:
                logger.error(f"Failed to extract vector graphics from page {page_num}: {e}")
        
        return graphics
    
    def _extract_image_context(self, page: fitz.Page, img_rect: Optional[fitz.Rect]) -> Dict[str, Any]:
        """Extract context around image"""
        context = {
            'caption': None,
            'figure_reference': None,
            'surrounding_text': []
        }
        
        if not img_rect:
            return context
        
        # Get text blocks near image
        text_blocks = page.get_text("blocks")
        
        for block in text_blocks:
            if len(block) < 5:
                continue
            
            block_rect = fitz.Rect(block[:4])
            text = block[4].strip()
            
            # Check if block is below image (potential caption)
            if (block_rect.y0 > img_rect.y1 and 
                block_rect.y0 - img_rect.y1 < 50 and
                abs(block_rect.x0 - img_rect.x0) < 100):
                
                if re.search(r'(Figure|Fig\.|Figure\.|Exhibit)\s*\d+', text, re.IGNORECASE):
                    context['caption'] = text
                    # Extract figure reference
                    fig_match = re.search(r'(Figure|Fig\.?|Exhibit)\s*(\d+)', text, re.IGNORECASE)
                    if fig_match:
                        context['figure_reference'] = fig_match.group(0)
            
            # Check if block is near image
            elif (abs(block_rect.y0 - img_rect.y0) < 100 or 
                  abs(block_rect.y1 - img_rect.y1) < 100):
                context['surrounding_text'].append(text)
        
        return context
    
    def _extract_page_context(self, page: fitz.Page) -> Dict[str, Any]:
        """Extract overall page context"""
        context = {
            'page_title': None,
            'page_text_preview': "",
            'exhibits_mentioned': []
        }
        
        # Get all text from page
        text = page.get_text()
        
        # Look for title
        text_blocks = page.get_text("blocks")
        if text_blocks:
            for block in text_blocks[:3]:
                if len(block) >= 5:
                    potential_title = block[4].strip()
                    if len(potential_title) > 10 and len(potential_title) < 100:
                        context['page_title'] = potential_title
                        break
        
        # Get text preview
        context['page_text_preview'] = text[:200] + "..." if len(text) > 200 else text
        
        # Find exhibit references
        exhibit_matches = re.findall(r'Exhibit\s*\d+', text, re.IGNORECASE)
        context['exhibits_mentioned'] = list(set(exhibit_matches))
        
        return context
    
    def _save_image(self, img_info: Dict):
        """Save image with metadata"""
        page_num = img_info['page_number']
        img_index = img_info['image_index']
        image_type = img_info['image_type']
        extraction_method = img_info['extraction_method']
        
        # Generate filename
        if extraction_method == 'vector_render':
            filename = f"page_{page_num}_vector_{image_type}.png"
        else:
            filename = f"page_{page_num}_img_{img_index}_{image_type}.png"
        
        filepath = self.output_dir / filename
        
        # Atomic save
        if not self._atomic_save_image(img_info['image'], filepath):
            logger.error(f"Failed to save image {filename}")
            return
        
        # Calculate file size
        file_size = filepath.stat().st_size
        
        # Create metadata
        metadata = ImageMetadata(
            filename=filename,
            page_number=page_num,
            image_index=img_index,
            image_type=image_type,
            extraction_method=extraction_method,
            width=img_info['image'].width,
            height=img_info['image'].height,
            quality_score=img_info['quality_score'],
            has_text=bool(img_info['text_content']),
            text_content=img_info['text_content'],
            visual_elements=img_info['type_metadata'],
            extraction_timestamp=datetime.now().isoformat(),
            file_size=file_size,
            dpi=96,
            color_mode=img_info['image'].mode,
            enhancement_applied=self.enable_enhancement,
            context=img_info['context'],
            vector_count=img_info.get('vector_count')
        )
        
        # Thread-safe update
        with self._images_lock:
            self.extracted_images.append(metadata)
            
            # Update statistics
            self.extraction_stats['image_types_found'][image_type] += 1
            
            if metadata.quality_score >= 0.7:
                self.extraction_stats['quality_distribution']['high'] += 1
            elif metadata.quality_score >= 0.4:
                self.extraction_stats['quality_distribution']['medium'] += 1
            else:
                self.extraction_stats['quality_distribution']['low'] += 1
        
        logger.debug(f"Saved {filename} - Type: {image_type}, "
                    f"Method: {extraction_method}, "
                    f"Quality: {metadata.quality_score:.2f}")
    
    def _verify_saved_files(self) -> List[Dict]:
        """Verify all saved image files"""
        issues = []
        png_files = list(self.output_dir.glob("*.png"))
        
        logger.info(f"Verifying {len(png_files)} image files...")
        
        for png_file in png_files:
            try:
                # Check file size
                if png_file.stat().st_size == 0:
                    issues.append({
                        'file': png_file.name,
                        'issue': 'empty_file',
                        'severity': 'critical'
                    })
                    continue
                
                # Verify image integrity
                img = Image.open(png_file)
                img.verify()
                
                # Re-open for additional checks
                img = Image.open(png_file)
                
                # Check dimensions
                if img.width < 10 or img.height < 10:
                    issues.append({
                        'file': png_file.name,
                        'issue': 'too_small',
                        'width': img.width,
                        'height': img.height,
                        'severity': 'warning'
                    })
                
            except Exception as e:
                issues.append({
                    'file': png_file.name,
                    'issue': 'corrupt_image',
                    'error': str(e),
                    'severity': 'critical'
                })
        
        # Check for missing files
        metadata_files = {img.filename for img in self.extracted_images}
        actual_files = {f.name for f in png_files}
        
        missing_files = metadata_files - actual_files
        for missing in missing_files:
            issues.append({
                'file': missing,
                'issue': 'file_missing',
                'severity': 'critical'
            })
        
        if issues:
            critical_count = sum(1 for i in issues if i.get('severity') == 'critical')
            logger.warning(f"Verification found {len(issues)} issues ({critical_count} critical)")
        else:
            logger.info("All files verified successfully!")
        
        return issues
    
    def _save_extraction_metadata(self):
        """Save extraction metadata and statistics"""
        metadata_file = self.output_dir / 'extraction_metadata.json'
        
        metadata = {
            'pdf_file': str(self.pdf_path),
            'extraction_timestamp': datetime.now().isoformat(),
            'configuration': {
                'min_width': self.min_size[0],
                'min_height': self.min_size[1],
                'min_quality_score': self.min_quality_score,
                'vector_threshold': self.vector_threshold,
                'enable_ocr': self.enable_ocr,
                'enable_enhancement': self.enable_enhancement,
                'max_workers': self.max_workers,
                'page_limit': self.page_limit
            },
            'statistics': dict(self.extraction_stats),
            'images': [asdict(img) for img in self.extracted_images]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enterprise PDF Image Extractor v1.0+ (Enhanced)'
    )
    parser.add_argument('pdf_path', help='Path to PDF file', nargs='?', default='/data/input.pdf')
    parser.add_argument(
        '--output-dir',
        default='/data/pdf_images',
        help='Output directory for extracted images'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Number of parallel workers (0=auto)'
    )
    parser.add_argument(
        '--min-width',
        type=int,
        default=100,
        help='Minimum image width'
    )
    parser.add_argument(
        '--min-height',
        type=int,
        default=100,
        help='Minimum image height'
    )
    parser.add_argument(
        '--min-quality',
        type=float,
        default=0.3,
        help='Minimum quality score (0-1)'
    )
    parser.add_argument(
        '--vector-threshold',
        type=int,
        default=10,
        help='Minimum vector count to extract page as image'
    )
    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Disable OCR text extraction'
    )
    parser.add_argument(
        '--no-enhance',
        action='store_true',
        help='Disable image enhancement'
    )
    parser.add_argument(
        '--clear-output',
        action='store_true',
        help='Clear output directory before extraction'
    )
    parser.add_argument(
        '--page-limit',
        type=int,
        default=None,
        help='Limit extraction to first N pages (for testing)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect optimal workers
    if args.workers == 0:
        args.workers = min(16, os.cpu_count() or 1)
        logger.info(f"Auto-detected {args.workers} workers")
    
    # Configuration
    config = {
        'min_width': args.min_width,
        'min_height': args.min_height,
        'min_quality_score': args.min_quality,
        'enable_ocr': not args.no_ocr,
        'enable_enhancement': not args.no_enhance,
        'vector_threshold': args.vector_threshold,
        'save_metadata': True,
        'max_workers': args.workers,
        'clear_output': args.clear_output,
        'page_limit': args.page_limit
    }
    
    # Extract images
    extractor = EnterpriseImageExtractor(
        args.pdf_path,
        args.output_dir,
        config
    )
    
    images = extractor.extract_all_images()
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total pages processed: {extractor.extraction_stats['total_pages']}")
    print(f"Pages with images: {extractor.extraction_stats['pages_with_images']}")
    print(f"Pages with vector graphics: {extractor.extraction_stats['pages_with_vector_graphics']}")
    print(f"Total images extracted: {len(images)}")
    print(f"  - Embedded images: {extractor.extraction_stats['embedded_images_found']}")
    print(f"  - Vector graphics: {extractor.extraction_stats['vector_graphics_found']}")
    print(f"OCR success rate: {extractor.extraction_stats['ocr_success_rate']:.1%}")
    print(f"Extraction time: {extractor.extraction_stats['total_extraction_time']:.2f}s")
    print(f"Parallel workers used: {config['max_workers']}")
    
    if extractor.extraction_stats['extraction_errors']:
        print(f"\nExtraction errors: {len(extractor.extraction_stats['extraction_errors'])}")
    
    print("\nImage Types Found:")
    for img_type, count in sorted(
        extractor.extraction_stats['image_types_found'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {img_type}: {count}")
    
    print("\nQuality Distribution:")
    for quality, count in extractor.extraction_stats['quality_distribution'].items():
        print(f"  {quality.capitalize()}: {count}")
    
    if 'verification_issues' in extractor.extraction_stats:
        issues = extractor.extraction_stats['verification_issues']
        critical = sum(1 for i in issues if i.get('severity') == 'critical')
        print(f"\nVerification issues: {len(issues)} ({critical} critical)")

if __name__ == "__main__":
    main()
