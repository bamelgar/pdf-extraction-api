FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic tools
    wget \
    curl \
    # PDF tools
    poppler-utils \
    ghostscript \
    # Image processing
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # OCR
    tesseract-ocr \
    tesseract-ocr-eng \
    # Build tools
    gcc \
    g++ \
    python3-dev \
    # Java for Tabula
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional PDF processing libraries
RUN pip install --no-cache-dir \
    PyMuPDF==1.23.8 \
    tabula-py==2.8.2 \
    camelot-py[cv]==0.11.0

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/pdf_extraction

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
