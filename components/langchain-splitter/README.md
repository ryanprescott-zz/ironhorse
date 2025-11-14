# LangChain Splitter Component

Text splitter component using LangChain's RecursiveCharacterTextSplitter for intelligent text chunking.

## Overview

The LangChain Splitter component provides a standardized interface for splitting text into chunks using LangChain's RecursiveCharacterTextSplitter. It converts split text into the AI Toolkit Chunk schema and exposes splitting capabilities through both a Python API and FastAPI web service.

## Features

- **Intelligent Splitting**: Uses recursive character-based splitting
- **Configurable Parameters**: Chunk size, overlap, separators
- **Batch Processing**: Split multiple documents in a single request
- **Standard Output**: Returns chunks in AI Toolkit Chunk schema
- **REST API**: FastAPI-based web service
- **Pip Installable**: Available as a Python package
- **Docker Ready**: Containerized web service

## Installation

### As a Python Package

```bash
cd components/langchain-splitter
pip install -e .
```

### As a Docker Container

```bash
# Build from project root
docker build -f platform/docker/components/langchain-splitter/Dockerfile -t ai-toolkit/langchain-splitter:latest .

# Run
docker run -p 26001:26001 ai-toolkit/langchain-splitter:latest
```

### Using Docker Compose

```bash
# From project root
docker-compose up langchain-splitter
```

## Usage

### Python API

```python
from langchain_splitter.core import LangChainSplitter
from shared.schemas import Document, DocumentMetadata

# Initialize splitter
splitter = LangChainSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# Split raw text
text = "Your long text content here..."
chunks = splitter.split_text(text, source_id="doc_123")

print(f"Created {len(chunks)} chunks")
for chunk in chunks[:3]:
    print(f"Chunk {chunk.chunk_id}: {chunk.text[:50]}...")

# Split a document
document = Document(
    doc_id="doc_123",
    content="Your document content...",
    metadata=DocumentMetadata(source="test.pdf")
)

chunks = splitter.split_document(document)

# Split multiple documents
documents = [doc1, doc2, doc3]
all_chunks = splitter.split_documents(documents)

# Analyze text without splitting
count = splitter.get_chunk_count(text)
print(f"Would create {count} chunks")
```

### REST API

#### Split Text

```bash
curl -X POST "http://localhost:26001/api/v1/split/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text content here...",
    "source_id": "doc_123",
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

Response:
```json
{
  "status": "success",
  "data": {
    "chunks": [
      {
        "chunk_id": "chunk_abc123_0",
        "text": "chunk text content...",
        "metadata": {
          "source_doc_id": "doc_123",
          "chunk_index": 0,
          "start_char": 0,
          "end_char": 1000
        }
      }
    ]
  },
  "error": null,
  "metadata": {
    "processing_time_ms": 45,
    "chunk_count": 5,
    "original_length": 4500
  }
}
```

#### Split Document

```bash
curl -X POST "http://localhost:26001/api/v1/split/document" \
  -H "Content-Type: application/json" \
  -d '{
    "document": {
      "doc_id": "doc_123",
      "content": "Your document content...",
      "metadata": {"source": "test.pdf", "file_type": "pdf"},
      "tables": []
    },
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

#### Split Multiple Documents

```bash
curl -X POST "http://localhost:26001/api/v1/split/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [...],
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

#### Analyze Text

```bash
curl -X POST "http://localhost:26001/api/v1/analyze?text=Your%20text%20here&chunk_size=1000&chunk_overlap=200"
```

## Configuration

Configuration is managed via environment variables using Pydantic Settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `LANGCHAIN_SPLITTER_HOST` | API server host | 0.0.0.0 |
| `LANGCHAIN_SPLITTER_PORT` | API server port | 26001 |
| `LANGCHAIN_SPLITTER_LOG_LEVEL` | Logging level | INFO |
| `LANGCHAIN_SPLITTER_DEFAULT_CHUNK_SIZE` | Default chunk size | 1000 |
| `LANGCHAIN_SPLITTER_DEFAULT_CHUNK_OVERLAP` | Default overlap | 200 |
| `LANGCHAIN_SPLITTER_DEFAULT_SEPARATORS` | Default separators | ["\n\n", "\n", " ", ""] |
| `LANGCHAIN_SPLITTER_DEFAULT_KEEP_SEPARATOR` | Keep separators | true |
| `LANGCHAIN_SPLITTER_DEFAULT_STRIP_WHITESPACE` | Strip whitespace | true |

### Example Configuration

```bash
export LANGCHAIN_SPLITTER_PORT=8081
export LANGCHAIN_SPLITTER_LOG_LEVEL=DEBUG
export LANGCHAIN_SPLITTER_DEFAULT_CHUNK_SIZE=500
export LANGCHAIN_SPLITTER_DEFAULT_CHUNK_OVERLAP=100
```

## Data Models

### Input Parameters

- **text/document** (str/Document): Content to split
- **source_id** (str, optional): Source identifier
- **chunk_size** (int): Maximum chunk size in characters
- **chunk_overlap** (int): Overlap between consecutive chunks
- **separators** (list[str], optional): Custom separators
- **keep_separator** (bool): Whether to keep separators in chunks
- **strip_whitespace** (bool): Whether to strip whitespace

### Output: Chunk Schema

```python
{
  "chunk_id": str,         # Unique chunk identifier
  "text": str,             # Chunk text content
  "metadata": {            # Chunk metadata
    "source_doc_id": str,  # Source document ID
    "chunk_index": int,    # Index in sequence
    "start_char": int,     # Start position
    "end_char": int,       # End position
    "custom": dict         # Additional metadata
  }
}
```

## Development

### Setup Development Environment

```bash
cd components/langchain-splitter

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
pytest --cov=langchain_splitter --cov-report=html

# Run specific test file
pytest tests/test_core.py
```

### Running Locally

```bash
# Run the API server
python -m langchain_splitter.api.main

# Or with custom settings
LANGCHAIN_SPLITTER_PORT=8081 python -m langchain_splitter.api.main
```

### Code Quality

```bash
# Format code
black .

# Lint
ruff check .

# Type check
mypy langchain_splitter/
```

## Architecture

### Directory Structure

```
langchain-splitter/
├── core/                        # Core splitting logic
│   ├── __init__.py
│   └── langchain_splitter.py   # LangChainSplitter class
├── api/                         # FastAPI web service
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   └── routes.py               # API endpoints
├── config/                      # Configuration
│   ├── __init__.py
│   └── settings.py             # Pydantic settings
├── tests/                       # Tests
│   ├── __init__.py
│   ├── test_core.py            # Core tests
│   └── test_api.py             # API tests
└── pyproject.toml              # Package configuration
```

### Key Classes

#### LangChainSplitter

Main splitter class providing text splitting functionality.

**Methods:**
- `split_text(text, source_id)`: Split raw text
- `split_document(document)`: Split a Document
- `split_documents(documents)`: Split multiple Documents
- `get_chunk_count(text)`: Get chunk count without splitting

## API Endpoints

### `POST /api/v1/split/text`

Split raw text into chunks.

**Request Body:**
```json
{
  "text": "text to split",
  "source_id": "optional_id",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "separators": ["\n\n", "\n"],
  "keep_separator": true,
  "strip_whitespace": true
}
```

**Response:** APIResponse with chunks

### `POST /api/v1/split/document`

Split a document into chunks.

**Request Body:**
```json
{
  "document": {...},
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

**Response:** APIResponse with chunks

### `POST /api/v1/split/documents`

Split multiple documents into chunks.

**Request Body:**
```json
{
  "documents": [...],
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

**Response:** APIResponse with all chunks

### `POST /api/v1/analyze`

Analyze text without splitting.

**Query Parameters:**
- `text`: Text to analyze
- `chunk_size`: Chunk size to use
- `chunk_overlap`: Overlap to use

**Response:** APIResponse with analysis data

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "langchain-splitter"
}
```

## Splitting Strategy

The component uses LangChain's RecursiveCharacterTextSplitter, which:

1. **Tries separators in order**: Attempts to split on larger separators first (e.g., "\n\n" before "\n")
2. **Respects chunk size**: Ensures chunks don't exceed the specified size
3. **Maintains overlap**: Creates overlap between chunks for context preservation
4. **Preserves meaning**: Tries to split at natural boundaries (paragraphs, sentences, words)

### Separator Hierarchy

Default separators (in order of priority):
1. `"\n\n"` - Paragraph breaks
2. `"\n"` - Line breaks
3. `" "` - Spaces
4. `""` - Characters (fallback)

### Recommended Settings

**General text:**
- Chunk size: 1000-2000 characters
- Overlap: 100-200 characters

**Code:**
- Chunk size: 500-1000 characters
- Overlap: 50-100 characters
- Custom separators: `["\n\n", "\n", ";", " "]`

**Structured documents:**
- Chunk size: 1500-2500 characters
- Overlap: 200-300 characters

## Performance Considerations

- **Processing Speed**: Very fast, typically <100ms for most texts
- **Memory Usage**: Minimal, processes text in-memory
- **Concurrent Requests**: FastAPI handles concurrency efficiently
- **Batch Processing**: More efficient than individual requests

## Troubleshooting

### Import Errors

If you see `ImportError: langchain-text-splitters not found`:
```bash
pip install langchain-text-splitters
```

### Chunks Too Large/Small

Adjust `chunk_size` parameter:
- Too large: Reduce chunk_size
- Too small: Increase chunk_size
- Uneven: Adjust separators

### Missing Context

Increase `chunk_overlap` to preserve more context between chunks.

### Incorrect Splitting

Customize `separators` for your content type:
```python
# For code
separators=["\n\n", "\nclass ", "\ndef ", "\n", " "]

# For lists
separators=["\n\n", "\n- ", "\n", " "]
```

## Integration Examples

### With Docling Parser

```python
from docling_parser.core import DoclingParser
from langchain_splitter.core import LangChainSplitter

# Parse document
parser = DoclingParser()
document = parser.parse_document("/path/to/doc.pdf")

# Split into chunks
splitter = LangChainSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_document(document)

# Process chunks
for chunk in chunks:
    print(f"Processing chunk {chunk.chunk_id}...")
    # Send to embedding service, vector store, etc.
```

### In N8N Workflow

See [N8N Integration Guide](../../integrations/orchestrators/n8n/pipelines/README.md)

### With Vector Database

```python
from langchain_splitter.core import LangChainSplitter

splitter = LangChainSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(long_text, source_id="doc_123")

# Store in vector database
for chunk in chunks:
    embedding = get_embedding(chunk.text)  # Your embedding function
    vector_db.insert({
        "id": chunk.chunk_id,
        "text": chunk.text,
        "embedding": embedding,
        "metadata": chunk.metadata.model_dump()
    })
```

## Building and Deployment

### Build Python Package

```bash
python -m build
```

Wheel file will be in `dist/`

### Build Docker Image

```bash
# From project root
docker build -f platform/docker/components/langchain-splitter/Dockerfile \
  -t ai-toolkit/langchain-splitter:latest .
```

### Deploy to Registry

```bash
docker tag ai-toolkit/langchain-splitter:latest registry.company.com/ai-toolkit/langchain-splitter:latest
docker push registry.company.com/ai-toolkit/langchain-splitter:latest
```

## Future Enhancements

Planned features for future versions:

- Support for additional LangChain splitters:
  - CharacterTextSplitter
  - TokenTextSplitter
  - MarkdownHeaderTextSplitter
  - LanguageTextSplitter (code-aware)
- Token-based chunking (OpenAI, HuggingFace)
- Semantic chunking
- Custom splitting strategies
- Chunk quality metrics

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
- Check logs: `docker-compose logs langchain-splitter`
- Review API docs: http://localhost:26001/docs
- Create an issue in the project repository
