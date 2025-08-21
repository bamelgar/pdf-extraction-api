#!/usr/bin/env python3
"""
Enterprise PDF Table Extractor v1.1+ (Complete Production Version) - TESTING VERSION
- TESTING ONLY: Early exit after N tables for speed
- Parallel processing with ThreadPoolExecutor
- Atomic CSV writes with validation
- Comprehensive file verification
- Thread-safe operations
- Progress tracking
"""

import pdfplumber
import pandas as pd
import numpy as np
import os
import sys
import json
import logging
import hashlib
import re
import concurrent.futures
import tempfile
import shutil
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try importing optional libraries
try:
    import camelot
    HAS_CAMELOT = True
except ImportError:
    HAS_CAMELOT = False
    
try:
    import tabula
    HAS_TABULA = True
except ImportError:
    HAS_TABULA = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TableMetadata:
    """Metadata for extracted tables"""
    filename: str
    page_number: int
    table_index: int
    extraction_method: str
    quality_score: float
    table_type: str
    rows: int
    columns: int
    size_bytes: int
    has_headers: bool
    numeric_percentage: float
    empty_cell_percentage: float
    extraction_timestamp: str
    metadata: Dict[str, Any]

class TableClassifier:
    """Classify tables by content type - FULL VERSION for institutional investing"""
    
    @staticmethod
    def classify_table(table_data: List[List[Any]], text_context: str = "") -> Tuple[str, Dict]:
        """Classify table type and extract relevant metadata"""
        # Convert table to string for analysis
        table_text = ' '.join(str(cell) for row in table_data for cell in row if cell).lower()
        headers = ' '.join(str(cell) for cell in table_data[0] if cell).lower() if table_data else ""
        
        # Classification patterns - COMPLETE SET FOR FINANCIAL/SCIENTIFIC DOCS
        classifications = {
            'financial_income': {
                'keywords': ['revenue', 'income', 'expense', 'profit', 'loss', 'earnings', 'ebitda', 
                           'margin', 'sales', 'cost', 'operating', 'net income', 'gross profit'],
                'patterns': [r'\$[\d,]+', r'million', r'billion', r'thousand', r'mn', r'bn'],
                'metadata_extractors': ['currency', 'fiscal_period', 'units', 'company_identifiers']
            },
            'financial_balance': {
                'keywords': ['assets', 'liabilities', 'equity', 'debt', 'capital', 'cash', 
                           'receivables', 'payables', 'inventory', 'goodwill'],
                'patterns': [r'\$[\d,]+', r'balance sheet', r'statement of financial position'],
                'metadata_extractors': ['currency', 'date', 'units', 'accounting_standard']
            },
            'financial_cashflow': {
                'keywords': ['cash flow', 'operating', 'investing', 'financing', 'free cash flow',
                           'capex', 'working capital', 'dividends'],
                'patterns': [r'cash', r'flow', r'fcf'],
                'metadata_extractors': ['currency', 'period', 'units', 'cash_flow_type']
            },
            'financial_ratios': {
                'keywords': ['ratio', 'margin', 'roe', 'roa', 'roi', 'eps', 'p/e', 'debt/equity',
                           'current ratio', 'quick ratio', 'leverage'],
                'patterns': [r'\d+\.\d+x', r'\d+%', r'times', r'percent'],
                'metadata_extractors': ['ratio_types', 'comparison_period', 'benchmarks']
            },
            'scientific_data': {
                'keywords': ['experiment', 'sample', 'control', 'mean', 'std', 'p-value', 
                           'significant', 'correlation', 'n=', 'error', 'ci', 'confidence'],
                'patterns': [r'±', r'p\s*[<=]\s*0\.\d+', r'\d+\.\d+\s*±\s*\d+\.\d+', 
                           r'r\s*=\s*[0-9.-]+', r'n\s*=\s*\d+'],
                'metadata_extractors': ['units', 'statistical_measures', 'sample_size', 'p_values']
            },
            'clinical_trial': {
                'keywords': ['patient', 'placebo', 'treatment', 'adverse', 'efficacy', 'safety',
                           'endpoint', 'phase', 'randomized', 'double-blind'],
                'patterns': [r'phase\s+[IVX123]', r'n\s*=\s*\d+', r'%\s*\([^)]+\)'],
                'metadata_extractors': ['trial_phase', 'patient_count', 'endpoints', 'drug_name']
            },
            'market_data': {
                'keywords': ['price', 'volume', 'market cap', 'shares', 'trading', 'close',
                           'open', 'high', 'low', 'bid', 'ask', 'yield'],
                'patterns': [r'\$\d+\.\d{2}', r'\d+[KMB]', r'\d{1,3}(,\d{3})*'],
                'metadata_extractors': ['ticker_symbols', 'date_range', 'exchange', 'currency']
            },
            'esg_metrics': {
                'keywords': ['carbon', 'emissions', 'scope', 'renewable', 'diversity', 'governance',
                           'sustainability', 'ghg', 'co2', 'environmental'],
                'patterns': [r'tco2e?', r'mwh', r'gj', r'scope\s*[123]'],
                'metadata_extractors': ['metric_type', 'reporting_standard', 'time_period']
            },
            'portfolio_holdings': {
                'keywords': ['holdings', 'position', 'weight', 'allocation', 'security', 'cusip',
                           'isin', 'sector', 'asset class'],
                'patterns': [r'\d+\.\d+%', r'[A-Z]{2}\d{10}', r'[A-Z]{12}'],
                'metadata_extractors': ['portfolio_date', 'total_positions', 'asset_classes']
            }
        }
        
        scores = {}
        for table_type, config in classifications.items():
            score = 0
            # Check keywords (weighted by relevance)
            for kw in config['keywords']:
                if kw in table_text:
                    score += 2
                if kw in headers:  # Headers are more important
                    score += 3
            
            # Check patterns
            for pattern in config['patterns']:
                matches = len(re.findall(pattern, table_text))
                score += matches * 2
            
            scores[table_type] = score
        
        # Get best match
        best_type = max(scores.items(), key=lambda x: x[1])[0]
        if scores[best_type] == 0:
            best_type = 'general_data'
        
        # Extract metadata based on type
        metadata = TableClassifier._extract_type_specific_metadata(
            table_data, best_type, classifications.get(best_type, {})
        )
        
        return best_type, metadata
    
    @staticmethod
    def _extract_type_specific_metadata(table_data, table_type, config):
        """Extract metadata specific to table type"""
        metadata = {'table_classification': table_type}
        
        if 'currency' in config.get('metadata_extractors', []):
            metadata['currency'] = TableClassifier._detect_currency(table_data)
        
        if 'units' in config.get('metadata_extractors', []):
            metadata['units'] = TableClassifier._detect_units(table_data)
            
        if 'statistical_measures' in config.get('metadata_extractors', []):
            text = str(table_data)
            metadata['has_p_values'] = bool(re.search(r'p\s*[<=]\s*0\.\d+', text))
            metadata['has_error_bars'] = bool(re.search(r'±', text))
            metadata['has_confidence_intervals'] = bool(re.search(r'(CI|confidence\s*interval)', text, re.I))
        
        if 'fiscal_period' in config.get('metadata_extractors', []):
            metadata['fiscal_period'] = TableClassifier._detect_fiscal_period(table_data)
        
        if 'ticker_symbols' in config.get('metadata_extractors', []):
            metadata['tickers'] = TableClassifier._detect_tickers(table_data)
        
        return metadata
    
    @staticmethod
    def _detect_currency(table_data):
        """Detect currency in table"""
        currencies = {
            '$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY', 'CHF': 'CHF',
            'Rs': 'INR', 'R$': 'BRL', 'C$': 'CAD', 'A$': 'AUD', 'HK$': 'HKD'
        }
        
        text = ' '.join(str(cell) for row in table_data for cell in row if cell)
        for symbol, code in currencies.items():
            if symbol in text:
                return code
        return None
    
    @staticmethod
    def _detect_units(table_data):
        """Detect measurement units in table"""
        unit_patterns = [
            # Financial
            r'million', r'billion', r'thousand', r'mn', r'bn', r'k',
            # Scientific
            r'mg/ml', r'μg/ml', r'ng/ml', r'mM', r'μM', r'nM',
            r'kDa', r'Da', r'°C', r'°F', r'K',
            # ESG
            r'tCO2e?', r'MWh', r'GWh', r'GJ', r'TJ',
            # General
            r'%', r'percent', r'bps', r'basis points'
        ]
        
        found_units = []
        text = ' '.join(str(cell) for row in table_data for cell in row if cell)
        
        for pattern in unit_patterns:
            if re.search(r'\b' + pattern + r'\b', text, re.I):
                found_units.append(pattern)
        
        return found_units
    
    @staticmethod
    def _detect_fiscal_period(table_data):
        """Detect fiscal period in financial tables"""
        text = ' '.join(str(cell) for row in table_data for cell in row if cell)
        
        # Look for quarters
        quarter_match = re.search(r'(Q[1-4])\s*(\d{4}|\d{2})', text)
        if quarter_match:
            return quarter_match.group(0)
        
        # Look for fiscal years
        fy_match = re.search(r'(FY|fiscal year)\s*(\d{4}|\d{2})', text, re.I)
        if fy_match:
            return fy_match.group(0)
        
        # Look for date ranges
        range_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})\s*-\s*(\d{1,2}/\d{1,2}/\d{2,4})', text)
        if range_match:
            return range_match.group(0)
        
        return None
    
    @staticmethod
    def _detect_tickers(table_data):
        """Detect stock ticker symbols"""
        tickers = set()
        text = ' '.join(str(cell) for row in table_data for cell in row if cell)
        
        # Common pattern: 1-5 uppercase letters, often followed by exchange
        ticker_matches = re.findall(r'\b[A-Z]{1,5}\b(?:\.[A-Z]{2})?', text)
        
        # Filter out common words that might match
        exclude = {'USD', 'EUR', 'GBP', 'CEO', 'CFO', 'COO', 'IPO', 'M&A', 'Q1', 'Q2', 'Q3', 'Q4'}
        tickers = [t for t in ticker_matches if t not in exclude]
        
        return list(set(tickers))[:10]  # Return up to 10 unique tickers

class QualityAnalyzer:
    """Analyze and score table quality - COMPLETE VERSION"""
    
    @staticmethod
    def calculate_quality_score(table_data: List[List[Any]]) -> Tuple[float, Dict]:
        """Calculate comprehensive quality score"""
        if not table_data or len(table_data) < 2:
            return 0.0, {'reason': 'insufficient_data'}
        
        metrics = {
            'completeness': QualityAnalyzer._score_completeness(table_data),
            'consistency': QualityAnalyzer._score_consistency(table_data),
            'structure': QualityAnalyzer._score_structure(table_data),
            'data_types': QualityAnalyzer._score_data_types(table_data),
            'size_appropriateness': QualityAnalyzer._score_size(table_data)
        }
        
        # Weighted average
        weights = {
            'completeness': 0.3,
            'consistency': 0.2,
            'structure': 0.2,
            'data_types': 0.2,
            'size_appropriateness': 0.1
        }
        
        quality_score = sum(metrics[k] * weights[k] for k in metrics)
        
        return quality_score, metrics
    
    @staticmethod
    def _score_completeness(table_data):
        """Score based on non-empty cells"""
        total_cells = sum(len(row) for row in table_data)
        non_empty = sum(1 for row in table_data for cell in row 
                       if cell and str(cell).strip())
        return non_empty / total_cells if total_cells > 0 else 0
    
    @staticmethod
    def _score_consistency(table_data):
        """Score based on consistent column count"""
        if not table_data:
            return 0
        
        col_counts = [len(row) for row in table_data]
        most_common = max(set(col_counts), key=col_counts.count)
        consistent_rows = sum(1 for count in col_counts if count == most_common)
        
        return consistent_rows / len(table_data)
    
    @staticmethod
    def _score_structure(table_data):
        """Score based on table structure quality"""
        if len(table_data) < 2:
            return 0.5
        
        # Check if first row looks like headers
        first_row = table_data[0]
        header_score = sum(1 for cell in first_row 
                          if cell and not re.match(r'^-?\d+\.?\d*$', str(cell).strip()))
        header_score = header_score / len(first_row) if first_row else 0
        
        # Check for reasonable dimensions
        rows = len(table_data)
        cols = len(table_data[0]) if table_data else 0
        
        dimension_score = 1.0
        if rows < 2 or cols < 2:
            dimension_score = 0.3
        elif rows > 1000 or cols > 50:
            dimension_score = 0.7
        
        return (header_score + dimension_score) / 2
    
    @staticmethod
    def _score_data_types(table_data):
        """Score based on consistent data types in columns"""
        if len(table_data) < 2:
            return 0.5
        
        # Analyze data types by column
        num_cols = max(len(row) for row in table_data)
        consistent_cols = 0
        
        for col_idx in range(num_cols):
            col_data = []
            for row in table_data[1:]:  # Skip header
                if col_idx < len(row) and row[col_idx]:
                    col_data.append(str(row[col_idx]).strip())
            
            if not col_data:
                continue
            
            # Check if column has consistent type
            is_numeric = sum(1 for val in col_data 
                           if re.match(r'^-?\d+\.?\d*$', val))
            
            if is_numeric > len(col_data) * 0.8 or is_numeric < len(col_data) * 0.2:
                consistent_cols += 1
        
        return consistent_cols / num_cols if num_cols > 0 else 0
    
    @staticmethod
    def _score_size(table_data):
        """Score based on table size appropriateness"""
        rows = len(table_data)
        cols = len(table_data[0]) if table_data else 0
        
        # Ideal size ranges
        if 2 <= rows <= 500 and 2 <= cols <= 30:
            return 1.0
        elif rows < 2 or cols < 2:
            return 0.1
        elif rows > 1000 or cols > 50:
            return 0.6
        else:
            return 0.8

class EnterpriseTableExtractor:
    """Enterprise-grade PDF table extractor with parallel processing - TESTING VERSION"""
    
    def __init__(self, pdf_path: str, output_dir: str = "/data/pdf_tables",
                 config: Optional[Dict] = None):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = config or {}
        self.min_quality_score = self.config.get('min_quality_score', 0.3)
        # OPTIMIZED: Use all available CPUs up to 16 for Professional plan
        self.max_workers = self.config.get('max_workers', min(16, os.cpu_count() or 1))
        self.page_limit = self.config.get('page_limit', None)  # Add page limit support for testing
        self.save_metadata = self.config.get('save_metadata', True)
        self.enforce_quality_filter = self.config.get('enforce_quality_filter', False)
        self.enable_verification = self.config.get('enable_verification', True)
        
        # TESTING ONLY - Early exit limiter
        self.testing_table_limit = int(os.environ.get('TESTING_TABLE_LIMIT', '0'))
        if self.testing_table_limit > 0:
            logger.info(f"[TESTING MODE] Will stop after extracting {self.testing_table_limit} tables")
        
        # Results storage (thread-safe)
        self.extracted_tables: List[TableMetadata] = []
        self._tables_lock = threading.Lock()
        
        # TESTING ONLY - Early exit flag
        self._should_stop_extraction = False
        
        self.extraction_stats = {
            'total_pages': 0,
            'pages_with_tables': 0,
            'extraction_methods_used': defaultdict(int),
            'table_types_found': defaultdict(int),
            'quality_distribution': defaultdict(int),
            'total_extraction_time': 0,
            'pages_processed': 0,
            'extraction_errors': [],
            'tables_filtered_by_quality': 0,
            'testing_mode': self.testing_table_limit > 0,
            'testing_limit_reached': False
        }
        
        # Internal counters to avoid filename collisions
        self._page_counts: Dict[int, int] = defaultdict(int)
        self._page_lock = threading.Lock()
        
        # Initialize PDF
        self._initialize_pdf()
    
    def _initialize_pdf(self):
        """Initialize PDF and get basic info"""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                self.extraction_stats['total_pages'] = len(pdf.pages)
                logger.info(f"[DIAGNOSTICS] Initialized PDF: {self.pdf_path.name} "
                          f"({self.extraction_stats['total_pages']} pages)")
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise
    
    def _next_index_for_page(self, page_num: int) -> int:
        """Thread-safe index generation"""
        with self._page_lock:
            self._page_counts[page_num] += 1
            return self._page_counts[page_num]
    
    def _atomic_write_csv(self, df: pd.DataFrame, filepath: Path) -> bool:
        """
        Atomically write CSV to prevent corruption.
        Returns True on success, False on failure.
        """
        try:
            # Create temp file in same directory
            fd, tmp_path = tempfile.mkstemp(
                suffix='.csv.tmp', 
                dir=str(filepath.parent)
            )
            os.close(fd)
            
            # Write to temp file
            df.to_csv(
                tmp_path, 
                index=False, 
                lineterminator='\n', 
                encoding='utf-8'
            )
            
            # Verify temp file is valid
            test_df = pd.read_csv(tmp_path)
            if len(test_df) != len(df):
                raise ValueError("Row count mismatch after write")
            
            # Atomic rename
            shutil.move(tmp_path, str(filepath))
            return True
            
        except Exception as e:
            logger.error(f"Atomic write failed for {filepath}: {e}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
    
    def extract_all_tables(self):
        """Main extraction method with parallel processing - TESTING VERSION"""
        start_time = datetime.now()
        
        # TESTING DIAGNOSTICS AND LIMITERS
        logger.info(f"[DIAGNOSTICS] TESTING_TABLE_LIMIT env var: {os.environ.get('TESTING_TABLE_LIMIT', 'NOT SET')}")
        logger.info(f"[DIAGNOSTICS] Parsed testing_table_limit: {self.testing_table_limit}")
        
        if self.testing_table_limit > 0:
            logger.info(f"[TESTING MODE] Will stop after extracting {self.testing_table_limit} tables")
            self._should_stop_extraction = False
        else:
            logger.info(f"[PRODUCTION MODE] No testing limit set - will process all tables")
        
        logger.info(f"[DIAGNOSTICS] Starting parallel extraction with {self.max_workers} workers")
        logger.info(f"[DIAGNOSTICS] Processing {self.extraction_stats['total_pages']} pages")
        logger.info(f"Quality filter: {'ENABLED' if self.enforce_quality_filter else 'DISABLED'}")
        
        # Determine pages to process
        total_pages = self.extraction_stats['total_pages']
        if self.page_limit:
            total_pages = min(self.page_limit, total_pages)
            logger.info(f"Page limit set: processing first {total_pages} pages only")
        
        # Process pages in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit pages up to limit
            future_to_page = {
                executor.submit(self._process_page_safe, page_num): page_num
                for page_num in range(1, total_pages + 1)
            }
            
            # Process results as they complete
            processed_pages = 0
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                processed_pages += 1
                
                # TESTING ONLY - Check early exit
                if self._should_stop_extraction:
                    logger.info(f"[TESTING MODE] Early exit triggered - cancelling remaining pages")
                    # Cancel remaining futures
                    for remaining_future in future_to_page:
                        if not remaining_future.done():
                            remaining_future.cancel()
                    break
                
                # Progress update
                if processed_pages % 10 == 0 or processed_pages == total_pages:
                    progress = (processed_pages / total_pages) * 100
                    logger.info(f"[DIAGNOSTICS] Progress: {processed_pages}/{total_pages} pages ({progress:.1f}%)")
                
                try:
                    page_tables = future.result()
                    if page_tables:
                        self.extraction_stats['pages_with_tables'] += 1
                        logger.info(f"[DIAGNOSTICS] Page {page_num} found {len(page_tables)} tables")
                        for table_info in page_tables:
                            self._save_table(table_info)
                            
                            # TESTING ONLY - Check if we should stop
                            if self._should_stop_extraction:
                                logger.info(f"[TESTING MODE] Breaking from table processing loop")
                                break
                        
                        if self._should_stop_extraction:
                            break
                    else:
                        logger.debug(f"[DIAGNOSTICS] Page {page_num} found no tables")
                            
                except Exception as e:
                    logger.error(f"Failed to process page {page_num}: {e}")
                    self.extraction_stats['extraction_errors'].append({
                        'page': page_num,
                        'error': str(e)
                    })
        
        # Calculate final statistics
        self.extraction_stats['total_extraction_time'] = (
            datetime.now() - start_time
        ).total_seconds()
        self.extraction_stats['pages_processed'] = processed_pages
        
        # Verify all saved files
        if self.enable_verification:
            logger.info("Verifying saved files...")
            verification_issues = self.verify_saved_files()
            if verification_issues:
                logger.warning(f"Found {len(verification_issues)} verification issues")
                self.extraction_stats['verification_issues'] = verification_issues
        
        # Save metadata
        if self.save_metadata:
            self._save_extraction_metadata()
        
        logger.info(f"[DIAGNOSTICS] Extraction complete: {len(self.extracted_tables)} tables extracted in {self.extraction_stats['total_extraction_time']:.1f}s")
        if self.enforce_quality_filter and self.extraction_stats['tables_filtered_by_quality'] > 0:
            logger.info(f"Tables filtered by quality: {self.extraction_stats['tables_filtered_by_quality']}")
        
        if self.testing_table_limit > 0:
            logger.info(f"[TESTING MODE] Limit was {self.testing_table_limit}, extracted {len(self.extracted_tables)}")
        
        return self.extracted_tables
    
    def _process_page_safe(self, page_num: int) -> List[Dict]:
        """Thread-safe wrapper for page processing"""
        try:
            return self._process_page(page_num)
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}", exc_info=True)
            raise
    
    def _process_page(self, page_num: int) -> List[Dict]:
        """Process a single page with multiple extraction methods"""
        
        # TESTING ONLY - Check early exit
        if self._should_stop_extraction:
            logger.info(f"[TESTING MODE] Skipping page {page_num} due to early exit")
            return []
            
        logger.debug(f"[DIAGNOSTICS] Processing page {page_num}")
        
        all_tables = []
        
        # Try each extraction method
        extraction_methods = [
            ('pdfplumber', self._extract_with_pdfplumber),
            ('camelot_lattice', self._extract_with_camelot_lattice),
            ('camelot_stream', self._extract_with_camelot_stream),
            ('tabula', self._extract_with_tabula),
            ('pymupdf', self._extract_with_pymupdf)
        ]
        
        for method_name, method_func in extraction_methods:
            
            # TESTING ONLY - Check early exit before each method
            if self._should_stop_extraction:
                logger.info(f"[TESTING MODE] Breaking from extraction methods on page {page_num}")
                break
                
            try:
                tables = method_func(page_num)
                
                for table_idx, table_data in enumerate(tables):
                    
                    # TESTING ONLY - Check early exit for each table
                    if self._should_stop_extraction:
                        break
                        
                    if self._is_valid_table(table_data):
                        # Clean and enhance table
                        cleaned_table = self._clean_table_data(table_data)
                        
                        # Calculate quality score
                        quality_score, quality_metrics = QualityAnalyzer.calculate_quality_score(
                            cleaned_table
                        )
                        
                        # Apply quality filter ONLY if enforce_quality_filter is True
                        if self.enforce_quality_filter and quality_score < self.min_quality_score:
                            self.extraction_stats['tables_filtered_by_quality'] += 1
                            logger.debug(f"Filtered table on page {page_num} with quality score {quality_score:.2f}")
                            continue
                        
                        # Classify table
                        table_type, type_metadata = TableClassifier.classify_table(
                            cleaned_table
                        )
                        
                        # Create table info with proper index
                        table_info = {
                            'data': cleaned_table,
                            'page': page_num,
                            'table_index': len(all_tables) + 1,  # Pre-calculated index
                            'extraction_method': method_name,
                            'quality_score': quality_score,
                            'quality_metrics': quality_metrics,
                            'table_type': table_type,
                            'type_metadata': type_metadata,
                            'has_headers': True
                        }
                        
                        all_tables.append(table_info)
                        logger.debug(f"[DIAGNOSTICS] Page {page_num} table {len(all_tables)}: {table_type}, quality {quality_score:.2f}")
                        
                        # Update statistics
                        with self._tables_lock:
                            self.extraction_stats['extraction_methods_used'][method_name] += 1
                
                if self._should_stop_extraction:
                    break
                            
            except Exception as e:
                logger.debug(f"Method {method_name} failed on page {page_num}: {e}")
                continue
        
        # Remove duplicates
        all_tables = self._remove_duplicate_tables(all_tables)
        
        return all_tables
    
    def _extract_with_pdfplumber(self, page_num: int) -> List[List[List[Any]]]:
        """Extract tables using pdfplumber"""
        tables = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            page = pdf.pages[page_num - 1]
            
            # Try different extraction settings
            settings = [
                {},  # Default
                {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines"
                },
                {
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 3,
                    "join_tolerance": 3
                }
            ]
            
            for setting in settings:
                
                # TESTING ONLY - Check early exit
                if self._should_stop_extraction:
                    break
                    
                extracted = page.extract_tables(table_settings=setting)
                for table in extracted:
                    if table and self._is_valid_table(table):
                        tables.append(table)
        
        return tables
    
    def _extract_with_camelot_lattice(self, page_num: int) -> List[List[List[Any]]]:
        """Extract tables using Camelot (lattice mode for bordered tables)"""
        if not HAS_CAMELOT or self._should_stop_extraction:
            return []
        
        try:
            tables_camelot = camelot.read_pdf(
                str(self.pdf_path),
                pages=str(page_num),
                flavor='lattice',
                suppress_stdout=True
            )
            
            return [table.df.values.tolist() for table in tables_camelot]
        except:
            return []
    
    def _extract_with_camelot_stream(self, page_num: int) -> List[List[List[Any]]]:
        """Extract tables using Camelot (stream mode for borderless tables)"""
        if not HAS_CAMELOT or self._should_stop_extraction:
            return []
        
        try:
            tables_camelot = camelot.read_pdf(
                str(self.pdf_path),
                pages=str(page_num),
                flavor='stream',
                suppress_stdout=True
            )
            
            return [table.df.values.tolist() for table in tables_camelot]
        except:
            return []
    
    def _extract_with_tabula(self, page_num: int) -> List[List[List[Any]]]:
        """Extract tables using Tabula"""
        if not HAS_TABULA or self._should_stop_extraction:
            return []
        
        try:
            # Try different extraction methods
            tables_list = []
            
            # Lattice method
            tables = tabula.read_pdf(
                self.pdf_path,
                pages=page_num,
                multiple_tables=True,
                lattice=True,
                pandas_options={'header': None}
            )
            tables_list.extend([table.values.tolist() for table in tables])
            
            if self._should_stop_extraction:
                return tables_list
            
            # Stream method
            tables = tabula.read_pdf(
                self.pdf_path,
                pages=page_num,
                multiple_tables=True,
                stream=True,
                pandas_options={'header': None}
            )
            tables_list.extend([table.values.tolist() for table in tables])
            
            return tables_list
        except:
            return []
    
    def _extract_with_pymupdf(self, page_num: int) -> List[List[List[Any]]]:
        """Extract tables using PyMuPDF (placeholder for future implementation)"""
        if not HAS_PYMUPDF or self._should_stop_extraction:
            return []
        
        try:
            doc = fitz.open(self.pdf_path)
            page = doc[page_num - 1]
            # PyMuPDF doesn't have built-in table extraction
            # This is a placeholder for potential future implementation
            doc.close()
            return []
        except:
            return []
    
    def _clean_table_data(self, table_data: List[List[Any]]) -> List[List[Any]]:
        """Clean and standardize table data"""
        cleaned = []
        
        for row in table_data:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append('')
                else:
                    # Clean the cell value
                    cell_str = str(cell).strip()
                    # Remove excess whitespace
                    cell_str = re.sub(r'\s+', ' ', cell_str)
                    # Remove non-printable characters
                    cell_str = ''.join(ch for ch in cell_str if ch.isprintable())
                    # Remove embedded newlines
                    cell_str = cell_str.replace('\n', ' ').replace('\r', ' ')
                    cleaned_row.append(cell_str)
            
            # Only add non-empty rows
            if any(cell for cell in cleaned_row):
                cleaned.append(cleaned_row)
        
        return cleaned
    
    def _is_valid_table(self, table_data: List[List[Any]]) -> bool:
        """Check if table has valid structure"""
        if not table_data or len(table_data) < 2:
            return False
        
        # Check for minimum content
        total_cells = sum(len(row) for row in table_data)
        non_empty = sum(1 for row in table_data for cell in row 
                       if cell and str(cell).strip())
        
        if total_cells == 0 or non_empty / total_cells < 0.2:
            return False
        
        # Check for consistent structure
        col_counts = [len(row) for row in table_data]
        most_common_cols = max(set(col_counts), key=col_counts.count)
        
        # At least 50% of rows should have the same column count
        consistent_rows = sum(1 for count in col_counts if count == most_common_cols)
        if consistent_rows / len(table_data) < 0.5:
            return False
        
        return True
    
    def _remove_duplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """Remove duplicate tables based on content hash"""
        seen_hashes = set()
        unique_tables = []
        
        for table in tables:
            # Create hash of table content
            table_str = json.dumps(table['data'], sort_keys=True)
            table_hash = hashlib.md5(table_str.encode()).hexdigest()
            
            if table_hash not in seen_hashes:
                seen_hashes.add(table_hash)
                unique_tables.append(table)
            else:
                logger.debug(f"Removed duplicate table on page {table['page']}")
        
        return unique_tables
    
    def _save_table(self, table_info: Dict):
        """Save table to CSV file with atomic write - TESTING VERSION"""
        
        # TESTING ONLY - Check if we should stop before saving
        if self.testing_table_limit > 0:
            current_count = len(self.extracted_tables)
            logger.info(f"[TESTING DIAGNOSTICS] Current table count: {current_count}, Limit: {self.testing_table_limit}")
            
            if current_count >= self.testing_table_limit:
                logger.info(f"[TESTING MODE] Reached limit of {self.testing_table_limit} tables - stopping extraction")
                self._should_stop_extraction = True
                self.extraction_stats['testing_limit_reached'] = True
                return  # Skip saving this table
        
        page_num = table_info['page']
        # Use the pre-calculated index from table_info
        table_idx = table_info.get('table_index')
        if table_idx is None:
            table_idx = self._next_index_for_page(page_num)
            table_info['table_index'] = table_idx
        
        # Add extraction method suffix to prevent collisions
        method_suffix = table_info.get('extraction_method', 'unknown')
        filename = f"table_p{page_num:03d}_t{table_idx:03d}_{method_suffix}.csv"
        filepath = self.output_dir / filename
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(table_info['data'])
            
            # Check if first row is headers
            if table_info.get('has_headers', True) and len(df) > 0:
                first_row = df.iloc[0]
                is_header_row = any(
                    not pd.api.types.is_numeric_dtype(type(val))
                    for val in first_row if pd.notna(val)
                )
                
                if is_header_row:
                    df.columns = df.iloc[0]
                    df = df[1:].reset_index(drop=True)
            
            # Atomic write
            if not self._atomic_write_csv(df, filepath):
                raise Exception("Atomic write failed")
            
            # Verify file exists
            if not filepath.exists():
                raise Exception("File not found after write")
            
            # Create metadata
            metadata = TableMetadata(
                filename=filename,
                page_number=page_num,
                table_index=table_idx,
                extraction_method=table_info['extraction_method'],
                quality_score=table_info['quality_score'],
                table_type=table_info['table_type'],
                rows=len(df),
                columns=len(df.columns),
                size_bytes=filepath.stat().st_size,
                has_headers=table_info.get('has_headers', True),
                numeric_percentage=self._calculate_numeric_percentage(df),
                empty_cell_percentage=self._calculate_empty_percentage(df),
                extraction_timestamp=datetime.now().isoformat(),
                metadata=table_info.get('type_metadata', {})
            )
            
            # Thread-safe update
            with self._tables_lock:
                self.extracted_tables.append(metadata)
                
                # Update statistics
                if metadata.quality_score >= 0.7:
                    self.extraction_stats['quality_distribution']['high'] += 1
                elif metadata.quality_score >= 0.4:
                    self.extraction_stats['quality_distribution']['medium'] += 1
                else:
                    self.extraction_stats['quality_distribution']['low'] += 1
                
                self.extraction_stats['table_types_found'][metadata.table_type] += 1
            
            logger.info(f"[DIAGNOSTICS] Saved {filename} - Type: {metadata.table_type}, "
                       f"Quality: {metadata.quality_score:.2f}, "
                       f"Rows: {metadata.rows}, Cols: {metadata.columns}")
            
        except Exception as e:
            logger.error(f"Failed to save table: {e}")
            self.extraction_stats['extraction_errors'].append({
                'page': page_num,
                'table': table_idx,
                'error': str(e)
            })
    
    def _calculate_numeric_percentage(self, df: pd.DataFrame) -> float:
        """Calculate percentage of numeric cells"""
        total_cells = df.size
        if total_cells == 0:
            return 0.0
        
        numeric_cells = 0
        for col in df.columns:
            numeric_cells += pd.to_numeric(df[col], errors='coerce').notna().sum()
        
        return numeric_cells / total_cells
    
    def _calculate_empty_percentage(self, df: pd.DataFrame) -> float:
        """Calculate percentage of empty cells"""
        total_cells = df.size
        if total_cells == 0:
            return 1.0
        
        empty_cells = df.isna().sum().sum() + (df == '').sum().sum()
        return empty_cells / total_cells
    
    def verify_saved_files(self) -> List[Dict]:
        """Comprehensive verification of all saved CSV files"""
        issues = []
        csv_files = list(self.output_dir.glob("table_p*.csv"))
        
        logger.info(f"Verifying {len(csv_files)} CSV files...")
        
        for csv_file in csv_files:
            try:
                # Check 1: File size
                if csv_file.stat().st_size == 0:
                    issues.append({
                        'file': csv_file.name,
                        'issue': 'empty_file',
                        'severity': 'critical'
                    })
                    continue
                
                # Check 2: Can pandas read it?
                df = pd.read_csv(csv_file)
                
                # Check 3: Has content?
                if df.empty:
                    issues.append({
                        'file': csv_file.name,
                        'issue': 'no_data',
                        'severity': 'warning'
                    })
                
                # Check 4: Reasonable structure?
                if len(df.columns) > 100:
                    issues.append({
                        'file': csv_file.name,
                        'issue': 'too_many_columns',
                        'columns': len(df.columns),
                        'severity': 'warning'
                    })
                
                # Check 5: Line endings
                with open(csv_file, 'rb') as f:
                    content = f.read()
                    unix_lines = content.count(b'\n')
                    windows_lines = content.count(b'\r\n')
                    
                    if windows_lines > 0:
                        issues.append({
                            'file': csv_file.name,
                            'issue': 'mixed_line_endings',
                            'unix_lines': unix_lines - windows_lines,
                            'windows_lines': windows_lines,
                            'severity': 'minor'
                        })
                    
                    if unix_lines <= 1:
                        issues.append({
                            'file': csv_file.name,
                            'issue': 'single_line_file',
                            'severity': 'critical'
                        })
                
            except pd.errors.EmptyDataError:
                issues.append({
                    'file': csv_file.name,
                    'issue': 'pandas_empty_data',
                    'severity': 'critical'
                })
            except Exception as e:
                issues.append({
                    'file': csv_file.name,
                    'issue': 'read_error',
                    'error': str(e),
                    'severity': 'critical'
                })
        
        # Check for missing files
        metadata_files = {t.filename for t in self.extracted_tables}
        actual_files = {f.name for f in csv_files}
        
        missing_files = metadata_files - actual_files
        for missing in missing_files:
            issues.append({
                'file': missing,
                'issue': 'file_missing',
                'severity': 'critical'
            })
        
        # Summary
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
                'max_workers': self.max_workers,
                'min_quality_score': self.min_quality_score,
                'enforce_quality_filter': self.enforce_quality_filter,
                'enable_verification': self.enable_verification,
                'page_limit': self.page_limit,
                'testing_table_limit': self.testing_table_limit
            },
            'statistics': dict(self.extraction_stats),
            'tables': [asdict(table) for table in self.extracted_tables]
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save summary CSV
        if self.extracted_tables:
            summary_df = pd.DataFrame([asdict(t) for t in self.extracted_tables])
            summary_df.to_csv(self.output_dir / 'extraction_summary.csv', index=False)
        
        logger.info(f"Saved metadata to {metadata_file}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enterprise PDF Table Extractor v1.1+ (Complete Production Version) - TESTING VERSION'
    )
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument(
        '--output-dir', 
        default='/data/pdf_tables',
        help='Output directory for extracted tables'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Number of parallel workers (0=auto)'
    )
    parser.add_argument(
        '--min-quality',
        type=float,
        default=0.3,
        help='Minimum quality score (0-1)'
    )
    parser.add_argument(
        '--clear-output',
        action='store_true',
        help='Clear output directory before extraction'
    )
    parser.add_argument(
        '--enforce-quality-filter',
        action='store_true',
        help='Only save tables meeting minimum quality score'
    )
    parser.add_argument(
        '--no-verification',
        action='store_true',
        help='Skip file verification step'
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
        'max_workers': args.workers,
        'min_quality_score': args.min_quality,
        'save_metadata': True,
        'enforce_quality_filter': args.enforce_quality_filter,
        'enable_verification': not args.no_verification,
        'page_limit': args.page_limit
    }
    
    # Clear output directory if requested
    if args.clear_output:
        output_path = Path(args.output_dir)
        if output_path.exists():
            logger.info(f"Clearing {output_path}")
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract tables
    extractor = EnterpriseTableExtractor(
        args.pdf_path,
        args.output_dir,
        config
    )
    
    tables = extractor.extract_all_tables()
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY - TESTING VERSION")
    print("="*60)
    print(f"Total pages processed: {extractor.extraction_stats['total_pages']}")
    print(f"Pages with tables: {extractor.extraction_stats['pages_with_tables']}")
    print(f"Total tables extracted: {len(tables)}")
    print(f"Extraction time: {extractor.extraction_stats['total_extraction_time']:.2f}s")
    print(f"Parallel workers used: {config['max_workers']}")
    print(f"Pages/second: {extractor.extraction_stats['total_pages'] / extractor.extraction_stats['total_extraction_time']:.1f}")
    
    if extractor.testing_table_limit > 0:
        print(f"[TESTING MODE] Limit was {extractor.testing_table_limit}, extracted {len(tables)}")
        print(f"[TESTING MODE] Limit reached: {extractor.extraction_stats['testing_limit_reached']}")
    
    if extractor.extraction_stats['extraction_errors']:
        print(f"\nExtraction errors: {len(extractor.extraction_stats['extraction_errors'])}")
        for err in extractor.extraction_stats['extraction_errors'][:5]:
            print(f"  - Page {err['page']}: {err['error']}")
    
    if extractor.enforce_quality_filter and extractor.extraction_stats['tables_filtered_by_quality'] > 0:
        print(f"\nTables filtered by quality: {extractor.extraction_stats['tables_filtered_by_quality']}")
    
    print("\nQuality Distribution:")
    for quality, count in extractor.extraction_stats['quality_distribution'].items():
        print(f"  {quality.capitalize()}: {count}")
    
    print("\nTable Types Found:")
    for table_type, count in sorted(
        extractor.extraction_stats['table_types_found'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {table_type}: {count}")
    
    print("\nExtraction Methods Used:")
    for method, count in sorted(
        extractor.extraction_stats['extraction_methods_used'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {method}: {count}")
    
    if 'verification_issues' in extractor.extraction_stats:
        issues = extractor.extraction_stats['verification_issues']
        critical = sum(1 for i in issues if i.get('severity') == 'critical')
        print(f"\nVerification issues: {len(issues)} ({critical} critical)")
        if critical > 0:
            print("  Critical issues:")
            for issue in issues[:5]:
                if issue.get('severity') == 'critical':
                    print(f"    - {issue['file']}: {issue['issue']}")
    
    print("\nTop 10 Largest Tables:")
    sorted_tables = sorted(tables, key=lambda x: x.size_bytes, reverse=True)[:10]
    for table in sorted_tables:
        print(f"  {table.filename}: {table.rows}x{table.columns} "
              f"({table.size_bytes:,} bytes) - {table.table_type}")

if __name__ == "__main__":
    main()
