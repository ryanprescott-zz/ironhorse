# N8N Pipeline Examples

This directory contains example n8n workflow definitions for AI Toolkit RAG pipelines.

## Available Workflows

### Basic RAG Pipeline

**File:** `basic_rag_pipeline.json`

A simple pipeline demonstrating document parsing and text splitting:

1. **Parse Document**: Uses Docling Parser to parse a document (PDF, DOCX, etc.)
2. **Split Document**: Uses LangChain Splitter to split the parsed document into chunks
3. **Extract Metadata**: Extracts processing metadata (chunk count, processing time)
4. **Split Out Chunks**: Splits chunks into individual items for downstream processing

**Input Format:**
```json
{
  "file_path": "/path/to/document.pdf"
}
```

**Output:**
Each execution produces multiple output items, one per chunk:
```json
{
  "chunk_id": "chunk_abc123_0",
  "text": "chunk text content...",
  "metadata": {
    "source_doc_id": "doc_xyz789",
    "chunk_index": 0,
    "start_char": 0,
    "end_char": 1000
  }
}
```

## Importing Workflows

1. Open your n8n instance
2. Click on "Workflows" in the sidebar
3. Click "Import from File"
4. Select the workflow JSON file
5. Configure the workflow as needed

## Configuring Workflows

### Service URLs

The workflows assume the following default service URLs:
- Docling Parser: `http://docling-parser:26000`
- LangChain Splitter: `http://langchain-splitter:26001`

If your services are running on different hosts/ports, update the HTTP Request nodes accordingly.

### Parameters

You can customize the following parameters in the workflows:

**Parse Document Node:**
- `file_path`: Path to the document file
- `extract_tables`: Whether to extract tables (true/false)
- `doc_id`: Optional custom document ID

**Split Document Node:**
- `chunk_size`: Size of each chunk in characters (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `separators`: Custom list of separators (optional)

## Extending Workflows

These are basic examples. You can extend them by adding:

1. **Vector Embedding**: Add an embedding component to generate embeddings for each chunk
2. **Vector Storage**: Store chunks and embeddings in a vector database
3. **RAG Retrieval**: Add retrieval nodes to find relevant chunks for queries
4. **LLM Integration**: Connect to an LLM for question answering

## Testing Workflows

### Local Testing with Docker Compose

1. Start all services:
   ```bash
   docker-compose up -d
   ```

2. Import the workflow into n8n

3. Trigger the workflow with test data:
   ```json
   {
     "file_path": "/path/to/test/document.pdf"
   }
   ```

4. Monitor the execution in n8n's execution log

### Manual Testing

You can test individual components using curl:

```bash
# Test Docling Parser
curl -X POST "http://localhost:26000/api/v1/parse/single?file_path=/test.pdf&extract_tables=true"

# Test LangChain Splitter
curl -X POST "http://localhost:26001/api/v1/split/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here", "chunk_size": 1000, "chunk_overlap": 200}'
```

## Troubleshooting

### Connection Errors

If you see connection errors:
1. Verify services are running: `docker-compose ps`
2. Check service logs: `docker-compose logs docling-parser langchain-splitter`
3. Verify network connectivity between n8n and services

### Processing Errors

If documents fail to process:
1. Check the file path is accessible to the Docling Parser container
2. Verify the file format is supported
3. Check service logs for detailed error messages

## Additional Resources

- [N8N Documentation](https://docs.n8n.io/)
- [AI Toolkit Component Documentation](../../../../README.md)
- [Docling Parser API](../../../../components/docling-parser/README.md)
- [LangChain Splitter API](../../../../components/langchain-splitter/README.md)
