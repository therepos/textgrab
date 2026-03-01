FROM python:3.11-slim

WORKDIR /app

# System dependencies for pdfplumber, pdf2image, OpenCV, doctr
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    poppler-utils \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    libjpeg62-turbo-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download doctr models during build so first request isn't slow
RUN python -c "from doctr.models import ocr_predictor; ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)"

COPY app ./app
COPY static ./static

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
