# Docling Parser Component

Document parser component using Docling for extracting text and tables from various document formats.

## Overview

The Docling Parser component provides a standardized interface for parsing documents in multiple formats (PDF, DOCX, HTML, XLSX, PPTX) into the AI Toolkit Document schema. It uses the Docling library internally and exposes parsing capabilities through both a Python API and FastAPI web service.

## Features

- **Multiple Format Support**: PDF, DOCX, HTML, XLSX, PPTX
- **Table Extraction**: Automatically extracts tables from documents
- **Batch Processing**: Parse multiple documents in a single request
- **Standard Output**: Returns documents in AI Toolkit Document schema
- **REST API**: FastAPI-based web service
- **Pip Installable**: Available as a Python package
- **Docker Ready**: Containerized web service

## Installation

### As a Python Package

```bash
cd components/docling-parser
pip install -e .
```

### As a Docker Container

```bash
# Build from project root
docker build -f platform/docker/components/docling-parser/Dockerfile -t ai-toolkit/docling-parser:latest .

# Run
docker run -p 26000:26000 ai-toolkit/docling-parser:latest
```

### Using Docker Compose

```bash
# From project root
docker-compose up docling-parser
```

## Usage

### Python API

```python
from docling_parser.core import DoclingParser

# Initialize parser
parser = DoclingParser(extract_tables=True)

# Parse a single document
document = parser.parse_document(
    file_path="/path/to/document.pdf",
    doc_id="doc_123"
)

print(f"Document ID: {document.doc_id}")
print(f"Content: {document.content[:100]}...")
print(f"Tables: {len(document.tables)}")

# Parse multiple documents
documents = parser.parse_documents([
    "/path/to/doc1.pdf",
    "/path/to/doc2.docx",
])

# Get supported formats
formats = parser.get_supported_formats()
print(f"Supported formats: {formats}")
```

### REST API

#### Parse Single Document

```bash
curl -X POST "http://localhost:26000/api/v1/parse/single" \
  -H "Content-Type: application/json" \
  --data-urlencode "file_path=/path/to/document.pdf" \
  --data-urlencode "extract_tables=true"
```

Response:
```json
{
  "status": "success",
  "data": {
    "doc_id": "doc_abc123",
    "content": "Extracted document text...",
    "metadata": {
      "source": "/path/to/document.pdf",
      "file_type": "pdf",
      "page_count": null
    },
    "tables": []
  },
  "error": null,
  "metadata": {
    "processing_time_ms": 1234
  }
}
```

#### Parse Multiple Documents

```bash
curl -X POST "http://localhost:26000/api/v1/parse" \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": [
      "/path/to/doc1.pdf",
      "/path/to/doc2.docx"
    ],
    "extract_tables": true
  }'
```

#### Get Supported Formats

```bash
curl http://localhost:26000/api/v1/formats
```

## Configuration

Configuration is managed via environment variables using Pydantic Settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `DOCLING_PARSER_HOST` | API server host | 0.0.0.0 |
| `DOCLING_PARSER_PORT` | API server port | 26000 |
| `DOCLING_PARSER_LOG_LEVEL` | Logging level | INFO |
| `DOCLING_PARSER_MAX_FILE_SIZE_MB` | Maximum file size in MB | 100 |
| `DOCLING_PARSER_EXTRACT_TABLES` | Extract tables by default | true |
| `DOCLING_PARSER_EXTRACT_IMAGES` | Extract images (future) | false |

### Example Configuration

```bash
export DOCLING_PARSER_PORT=8080
export DOCLING_PARSER_LOG_LEVEL=DEBUG
export DOCLING_PARSER_MAX_FILE_SIZE_MB=200
```

## Data Models

### Input

- **file_path** (str): Path to document file
- **extract_tables** (bool): Whether to extract tables
- **doc_id** (str, optional): Custom document ID

### Output: Document Schema

```python
{
  "doc_id": str,           # Unique document identifier
  "content": str,          # Extracted text content
  "metadata": {            # Document metadata
    "source": str,         # Source file path
    "file_type": str,      # File extension
    "page_count": int,     # Number of pages (if available)
    "author": str,         # Author (if available)
    "created_at": str,     # Creation date (if available)
    "custom": dict         # Additional metadata
  },
  "tables": [              # Extracted tables
    {
      "data": list,        # Table data as list of dicts
      "caption": str,      # Table caption
      "num_rows": int,     # Number of rows
      "num_cols": int      # Number of columns
    }
  ]
}
```

## Development

### Setup Development Environment

```bash
cd components/docling-parser

# Install with dev dependencies
pip install -e ".[dev]"

# Or with uv
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=docling_parser --cov-report=html

# Run specific test file
pytest tests/test_core.py
```

### Running Locally

```bash
# Run the API server
python -m docling_parser.api.main

# Or with custom settings
DOCLING_PARSER_PORT=8080 python -m docling_parser.api.main
```

### Code Quality

```bash
# Format code
black .

# Lint
ruff check .

# Type check
mypy docling_parser/
```

## Architecture

### Directory Structure

```
docling-parser/
├── core/                    # Core parsing logic
│   ├── __init__.py
│   └── docling_parser.py   # DoclingParser class
├── api/                     # FastAPI web service
│   ├── __init__.py
│   ├── main.py             # FastAPI app
│   └── routes.py           # API endpoints
├── config/                  # Configuration
│   ├── __init__.py
│   └── settings.py         # Pydantic settings
├── tests/                   # Tests
│   ├── __init__.py
│   ├── test_core.py        # Core tests
│   └── test_api.py         # API tests
└── pyproject.toml          # Package configuration
```

### Key Classes

#### DoclingParser

Main parser class providing document parsing functionality.

**Methods:**
- `parse_document(file_path, doc_id)`: Parse single document
- `parse_documents(file_paths)`: Parse multiple documents
- `get_supported_formats()`: Get list of supported formats

## API Endpoints

### `POST /api/v1/parse/single`

Parse a single document.

**Query Parameters:**
- `file_path` (required): Path to document file
- `extract_tables` (optional): Extract tables (default: true)
- `doc_id` (optional): Custom document ID

**Response:** APIResponse with Document

### `POST /api/v1/parse`

Parse multiple documents.

**Request Body:**
```json
{
  "file_paths": ["path1", "path2"],
  "extract_tables": true
}
```

**Response:** APIResponse with list of Documents

### `GET /api/v1/formats`

Get supported file formats.

**Response:** APIResponse with list of format strings

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "docling-parser"
}
```

## Supported Formats

- **PDF**: Portable Document Format
- **DOCX**: Microsoft Word
- **HTML**: HyperText Markup Language
- **XLSX**: Microsoft Excel
- **PPTX**: Microsoft PowerPoint
- **MD**: Markdown (via Docling)
- **ASCIIDOC**: AsciiDoc (via Docling)

## Performance Considerations

- **File Size**: Default max 100MB (configurable)
- **Concurrent Requests**: FastAPI handles concurrency
- **Memory Usage**: Large documents may require significant memory
- **Processing Time**: Varies by document size and format

## Troubleshooting

### Import Errors

If you see `ImportError: docling not found`:
```bash
pip install docling
```

### File Not Found

Ensure file paths are accessible from the container:
- Mount volumes in docker-compose.yml
- Use absolute paths
- Check file permissions

### Table Extraction Issues

If tables aren't being extracted:
- Verify `extract_tables=true`
- Check document actually contains tables
- Some formats may not support table extraction

### Memory Issues

For large documents:
- Increase Docker memory limits
- Process documents in smaller batches
- Increase `max_file_size_mb` setting

## Integration Examples

### With LangChain Splitter

```python
from docling_parser.core import DoclingParser
from langchain_splitter.core import LangChainSplitter

# Parse document
parser = DoclingParser()
document = parser.parse_document("/path/to/doc.pdf")

# Split into chunks
splitter = LangChainSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_document(document)

print(f"Created {len(chunks)} chunks from document")
```

### In N8N Workflow

See [N8N Integration Guide](../../integrations/orchestrators/n8n/pipelines/README.md)

## Building and Deployment

### Build Python Package

```bash
python -m build
```

Wheel file will be in `dist/`

### Build Docker Image

```bash
# From project root
docker build -f platform/docker/components/docling-parser/Dockerfile \
  -t ai-toolkit/docling-parser:latest .
```

### Deploy to Registry

```bash
docker tag ai-toolkit/docling-parser:latest registry.company.com/ai-toolkit/docling-parser:latest
docker push registry.company.com/ai-toolkit/docling-parser:latest
```

## Contributing

When contributing to this component:

1. Follow PEP 8 style guidelines
2. Add tests for new functionality
3. Update documentation
4. Ensure all tests pass
5. Maintain >80% test coverage

## License

[Add license information]

## Support

For issues specific to this component:
- Check logs: `docker-compose logs docling-parser`
- Review API docs: http://localhost:26000/docs
- Create an issue in the project repository
