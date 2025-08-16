FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    poppler-utils \
    ghostscript \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    gcc \
    g++ \
    python3-dev \
    default-jre \
    && rm -rf /var/lib/apt/lists/* || \
    apt-get update && apt-get install -y \
    wget \
    curl \
    poppler-utils \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    gcc \
    g++ \
    python3-dev \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    PyMuPDF==1.23.8 \
    tabula-py==2.8.2 \
    camelot-py[cv]==0.11.0

COPY . .

RUN mkdir -p /tmp/pdf_extraction

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
