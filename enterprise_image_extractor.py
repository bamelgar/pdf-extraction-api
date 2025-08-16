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
                'indicators': ['flow', 'process', 'step', 'arrow', 'â†'', 'box', 'workflow',
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
            edges = cv2.C
