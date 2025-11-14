# N8N Bindings for AI Toolkit Components

This directory contains Python bindings and helper functions for integrating AI Toolkit components with n8n workflows.

## Overview

The bindings provide convenient Python functions that wrap the HTTP APIs of AI Toolkit components, making it easier to call them from n8n HTTP Request nodes.

## Available Bindings

### Docling Parser Binding

Functions for parsing documents:

- `parse_document()`: Parse a single document
- `parse_documents()`: Parse multiple documents in batch
- `get_supported_formats()`: Get list of supported file formats

### LangChain Splitter Binding

Functions for splitting text:

- `split_text()`: Split raw text into chunks
- `split_document()`: Split a document into chunks
- `split_documents()`: Split multiple documents into chunks
- `analyze_text()`: Analyze text without splitting

## Usage in N8N

### Method 1: Direct HTTP Requests

Use n8n's built-in HTTP Request node to call the component APIs directly:

```javascript
// Parse a document
POST http://docling-parser:26000/api/v1/parse/single
Query Parameters:
  - file_path: {{$json["file_path"]}}
  - extract_tables: true

// Split text
POST http://langchain-splitter:26001/api/v1/split/text
Body (JSON):
{
  "text": {{$json["text"]}},
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

### Method 2: Python Function Node (if available)

If your n8n instance supports Python execution:

```python
from integrations.orchestrators.n8n.bindings import parse_document, split_document

# Parse document
result = parse_document(
    file_path="/path/to/document.pdf",
    extract_tables=True
)

# Extract parsed document
document = result["data"]

# Split document
chunks_result = split_document(
    document=document,
    chunk_size=1000,
    chunk_overlap=200
)

# Extract chunks
chunks = chunks_result["data"]["chunks"]
```

## Example Workflows

See the `pipelines/` directory for complete n8n workflow examples.

## Configuration

The bindings use default base URLs:
- Docling Parser: `http://docling-parser:26000`
- LangChain Splitter: `http://langchain-splitter:26001`

These can be overridden by passing the `base_url` parameter to any function.

## Error Handling

All binding functions will raise exceptions on HTTP errors. Wrap calls in try-except blocks for robust error handling:

```python
try:
    result = parse_document(file_path="/path/to/doc.pdf")
except requests.HTTPError as e:
    print(f"API error: {e}")
```
