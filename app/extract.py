import io
import os
import logging
from pathlib import Path

import pdfplumber
from PIL import Image
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

# Lazy-load doctr to avoid slow import on every request
_doctr_model = None


def _get_doctr_model():
    """Lazy-load doctr OCR model (downloads on first use, cached after)."""
    global _doctr_model
    if _doctr_model is None:
        logger.info("Loading doctr OCR model...")
        from doctr.models import ocr_predictor
        _doctr_model = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
        )
        logger.info("doctr OCR model loaded.")
    return _doctr_model


def _preprocess_image(img: Image.Image) -> Image.Image:
    """Basic preprocessing to improve OCR quality."""
    import cv2
    import numpy as np

    # Convert to OpenCV format
    arr = np.array(img)

    # Convert to grayscale if color
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr

    # Adaptive thresholding for binarization
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary, h=10)

    return Image.fromarray(denoised)


def _ocr_with_doctr(images: list[Image.Image]) -> str:
    """Run doctr OCR on a list of PIL images."""
    import numpy as np

    model = _get_doctr_model()
    pages = []
    for img in images:
        # doctr expects numpy array in uint8
        arr = np.array(img.convert("RGB"))
        pages.append(arr)

    result = model(pages)
    text_parts = []
    for page in result.pages:
        page_text = []
        for block in page.blocks:
            for line in block.lines:
                line_text = " ".join(word.value for word in line.words)
                page_text.append(line_text)
        text_parts.append("\n".join(page_text))

    return "\n\n".join(text_parts)


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF. Try pdfplumber first, fall back to doctr OCR."""
    # First pass: pdfplumber for native/digital PDFs
    text_parts = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for p in pdf.pages:
            txt = p.extract_text(x_tolerance=2, y_tolerance=3) or ""
            text_parts.append(txt.rstrip())

    text = "\n".join(text_parts).strip()

    # If we got meaningful text, return it
    if text and len(text) > 50:
        return text

    # Fallback: convert PDF pages to images, preprocess, and OCR with doctr
    logger.info("pdfplumber returned sparse text, falling back to doctr OCR")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        images = convert_from_path(tmp_path, dpi=300)
        preprocessed = [_preprocess_image(img) for img in images]
        return _ocr_with_doctr(preprocessed)
    finally:
        os.unlink(tmp_path)


def extract_text_from_image(path: str) -> str:
    """Extract text from an image file using doctr with preprocessing."""
    img = Image.open(path)
    preprocessed = _preprocess_image(img)
    return _ocr_with_doctr([preprocessed])


def extract_text(path: str) -> str:
    """Extract text from a PDF or image file."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        with open(path, "rb") as f:
            return extract_text_from_pdf(f.read())
    elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"]:
        return extract_text_from_image(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
