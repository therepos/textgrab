# textgrab

Self-hosted text extraction and tabular data conversion.  
Deterministic reproducible output.

## Tabs

**Text** — Extract clean text from PDF, images, MHTML, HTML, DOCX. Optional scheme transforms for structured output (e.g. IM Classic for Government Instruction Manuals).

**Tabular** — Table detection and bank statement PDF to CSV conversion with auto-detect and template-based parsing.

## Schemes

Schemes are output transforms for the Text tab. They reshape extracted text into structured formats.

| Scheme | Accepts | Description |
|--------|---------|-------------|
| Raw Text | All | Default. Returns extracted text as-is. |
| IM Classic | .mhtml | Converts Government Instruction Manual content into structured markdown with hierarchical codes and glossary. |
| PDF → Markdown | .pdf | Converts PDFs into structured Markdown using Docling (IBM, MIT-licensed) as the primary engine — DocLayNet-trained layout model, TableFormer for table structure, EasyOCR for scanned pages. Extracts figures as base64 PNG sidecars. Falls back to a deterministic heuristic pipeline (font-size clustering + img2table + doctr) if Docling fails on a document. CPU-only inference; no LLM in the pipeline. |

Add new schemes by creating a Python file in `app/schemes/`.

## Deploy

```bash
docker compose up -d
```

Access at `http://localhost:3020`

### GPU acceleration (optional)

The image ships with CUDA-enabled PyTorch and auto-detects the GPU at startup. Hosts without a GPU fall back to CPU transparently. To grant the container GPU access, add this to the `textgrab` service in `docker-compose.yml`:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Prerequisites on the host: NVIDIA drivers + `nvidia-container-toolkit`. Expected speedup: ~5–10× for layout/OCR inference.

### Long-running conversions

PDFs larger than 5 pages (or 2 MB) are auto-routed to a background queue so the request isn't held open during inference. The UI polls for completion and shows a "Recent jobs" panel for results. Jobs are persisted in SQLite and retained for 7 days. Single-worker by design (prevents GPU OOM).

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/text` | POST | Text extraction with scheme transform (synchronous) |
| `/api/text/stream` | POST | Same, with SSE progress events; auto-routes large jobs to queue |
| `/api/jobs` | POST | Submit a scheme transform as a background job |
| `/api/jobs` | GET | List recent jobs |
| `/api/jobs/{id}` | GET | Fetch a job's status and result |
| `/api/schemes` | GET | List available schemes |
| `/api/tabular/extract` | POST | Structured extraction (text + tables) |
| `/api/tabular/convert` | POST | Bank statement PDF to CSV |
| `/api/tabular/convert-bulk` | POST | Multiple PDFs to merged CSV |
| `/api/templates` | GET | List tabular parser templates |
| `/api/version` | GET | Service version info |
| `/api/extract` | POST | Legacy — redirects to /api/text |
| `/api/convert` | POST | Legacy — redirects to /api/tabular/convert |

// Business
// Crypto
// Deposits
// Entertainment
// Equities
// Loans
// Others
// Rental

