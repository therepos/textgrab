# textgrab

Extract text from PDFs and images privately. Convert bank statements to categorized CSV.

## Features

- **Text extraction** from PDFs and images (JPG, PNG, TIFF, BMP, WebP)
- **OCR** powered by [doctr](https://github.com/mindee/doctr) with OpenCV preprocessing for scanned documents
- **Bank statement → CSV** conversion with automatic categorization (DBS credit card & deposit statements)
- **Fuzzy category matching** via customizable rules
- Simple web UI — no external dependencies, everything runs locally

## Quick Start

```yaml
services:
  textgrab:
    image: ghcr.io/therepos/textgrab:latest
    container_name: textgrab
    ports:
      - "8000:8000"
    volumes:
      - textgrab-data:/data/models
    restart: unless-stopped

volumes:
  textgrab-data:
```

```bash
docker compose up -d
```

Open `http://localhost:8000` in your browser.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/extract` | POST | Extract text from PDF/image |
| `/api/convert` | POST | Bank statement PDF → CSV |
| `/api/rules` | GET | Get category rules |
| `/api/rules` | POST | Update category rules |
| `/api/health` | GET | Health check |
| `/api/version` | GET | Version info |
| `/docs` | GET | Interactive API docs |

## Category Rules

Rules are stored in `/data/models/rules.json` (persisted via Docker volume). Update via API:

```bash
curl -X POST http://localhost:8000/api/rules \
  -H "Content-Type: application/json" \
  -d '{"Food": ["STARBUCKS", "MCDONALD"], "Transport": ["GRAB", "GOJEK"]}'
```
