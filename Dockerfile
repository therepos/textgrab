# ---- Stage 1: Build ----
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libjpeg62-turbo-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Install Python deps with CUDA-enabled PyTorch (cu124).
# Runtime auto-detects GPU; CPU-only hosts still work because
# torch.cuda.is_available() returns False and we fall back to CPU.
# Image is ~1.5 GB larger than the CPU-only build.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    -r requirements.txt

# Pre-download doctr models into a known location
RUN python -c "from doctr.models import ocr_predictor; ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)"

# Pre-download Docling models (layout + TableFormer) and EasyOCR
# detection/recognition weights so the first request in production
# does not trigger a network fetch.
RUN python -c "\
from docling.document_converter import DocumentConverter, PdfFormatOption; \
from docling.datamodel.base_models import InputFormat; \
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions; \
opts = PdfPipelineOptions(generate_picture_images=True, images_scale=2.0, do_ocr=True, do_table_structure=True, ocr_options=EasyOcrOptions(lang=['en'], use_gpu=False)); \
conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}); \
conv.initialize_pipeline(InputFormat.PDF); \
print('Docling + EasyOCR models cached')"

# ---- Stage 2: Runtime ----
FROM python:3.11-slim

WORKDIR /app

# Runtime-only system deps (no build-essential/gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libglib2.0-0 \
    libgl1 \
    libjpeg62-turbo \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy cached model weights (doctr, Docling, EasyOCR all live under /root/.cache)
COPY --from=builder /root/.cache /root/.cache

# Docling respects OMP_NUM_THREADS for CPU inference; default is 4.
ENV OMP_NUM_THREADS=4

# Copy application code
COPY app ./app
COPY static ./static

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]