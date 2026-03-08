# textgrab

Self-hosted text extraction and tabular data conversion.

## Tabs

**Text** — Extract clean text from PDF, images, MHTML, HTML, DOCX. Optional scheme transforms for structured output (e.g. IM Classic for Government Instruction Manuals).

**Tabular** — Table detection and bank statement PDF to CSV conversion with auto-detect and template-based parsing.

## Schemes

Schemes are output transforms for the Text tab. They reshape extracted text into structured formats.

| Scheme | Accepts | Description |
|--------|---------|-------------|
| Raw Text | All | Default. Returns extracted text as-is. |
| IM Classic | .mhtml | Converts Government Instruction Manual content into structured markdown with hierarchical codes and glossary. |

Add new schemes by creating a Python file in `app/schemes/`.

## Deploy

```bash
docker compose up -d
```

Access at `http://localhost:3020`

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/text` | POST | Text extraction with scheme transform |
| `/api/schemes` | GET | List available schemes |
| `/api/tabular/extract` | POST | Structured extraction (text + tables) |
| `/api/tabular/convert` | POST | Bank statement PDF to CSV |
| `/api/tabular/convert-bulk` | POST | Multiple PDFs to merged CSV |
| `/api/templates` | GET | List tabular parser templates |
| `/api/extract` | POST | Legacy — redirects to /api/text |
| `/api/convert` | POST | Legacy — redirects to /api/tabular/convert |
