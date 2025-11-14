# AI Toolkit Quick Start Guide

Welcome to AI Toolkit! This guide will help you get up and running quickly.

## What Was Built

The AI Toolkit project now includes:

### 1. Core Infrastructure
- **Shared Schemas**: Common data models (Document, Chunk, APIResponse)
- **Project Structure**: Organized directory layout
- **Cookie Cutter Template**: Template for creating new components

### 2. Components
- **Docling Parser**: Parse PDF, DOCX, HTML, XLSX, PPTX documents
- **LangChain Splitter**: Split text into chunks using LangChain

### 3. Orchestration
- **N8N Integration**: Bindings and workflows for n8n
- **Sample RAG Pipeline**: Example workflow for document processing

### 4. Infrastructure
- **Docker Support**: Dockerfiles for all components
- **Docker Compose**: Complete local development setup
- **Health Checks**: Monitoring endpoints for all services

## Quick Start (5 Minutes)

### Step 1: Start All Services

```bash
# From project root
docker-compose up -d
```

This starts:
- Docling Parser on port 26000
- LangChain Splitter on port 26001
- N8N on port 5678

### Step 2: Verify Services

```bash
# Check all services are running
docker-compose ps

# Health checks
curl http://localhost:26000/health
curl http://localhost:26001/health
```

Expected response: `{"status":"healthy","service":"..."}`

### Step 3: Try the Components

#### Parse a Document

First, place a test document in `data/documents/`:

```bash
# Example with a PDF (replace with your file)
curl -X POST "http://localhost:26000/api/v1/parse/single?file_path=/data/documents/test.pdf&extract_tables=true"
```

#### Split Text

```bash
curl -X POST "http://localhost:26001/api/v1/split/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test document. It has multiple sentences. We will split it into chunks for processing. This allows us to work with smaller pieces of text.",
    "chunk_size": 50,
    "chunk_overlap": 10
  }'
```

### Step 4: Access N8N

1. Open browser to http://localhost:5678
2. Login with:
   - Username: `admin`
   - Password: `admin`
3. Import the sample workflow from `integrations/orchestrators/n8n/pipelines/basic_rag_pipeline.json`

## Project Structure Overview

```
project-root/
├── components/              # Reusable components
│   ├── docling-parser/     # Document parser
│   └── langchain-splitter/ # Text splitter
├── shared/                  # Shared data models
├── integrations/           # Orchestrator integrations
│   └── orchestrators/n8n/  # N8N workflows
├── platform/               # Docker configs
├── templates/              # Cookie cutter templates
└── docker-compose.yml      # Local development
```

## Interactive API Documentation

Each component provides interactive API docs:

- **Docling Parser**: http://localhost:26000/docs
- **LangChain Splitter**: http://localhost:26001/docs

Try the endpoints directly in your browser!

## Creating Your First RAG Pipeline

### 1. Parse a Document

```python
import requests

response = requests.post(
    "http://localhost:26000/api/v1/parse/single",
    params={"file_path": "/data/documents/your-doc.pdf"}
)
document = response.json()["data"]
```

### 2. Split into Chunks

```python
response = requests.post(
    "http://localhost:26001/api/v1/split/document",
    json={
        "document": document,
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
)
chunks = response.json()["data"]["chunks"]
```

### 3. Process Chunks

```python
for chunk in chunks:
    print(f"Chunk {chunk['chunk_id']}: {chunk['text'][:100]}...")
    # Add your processing: embeddings, vector storage, etc.
```

## Next Steps

### For Development

1. **Create a New Component**:
   ```bash
   cd components
   cookiecutter ../templates/component
   ```

2. **Run Tests**:
   ```bash
   pytest components/docling-parser/tests/
   pytest components/langchain-splitter/tests/
   ```

3. **Local Development**:
   ```bash
   cd components/docling-parser
   pip install -e ".[dev]"
   python -m docling_parser.api.main
   ```

### For Production

1. Update docker-compose.yml with production settings
2. Replace default credentials
3. Configure SSL/TLS
4. Set up monitoring and logging
5. Deploy to your infrastructure

## Common Tasks

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f docling-parser
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart docling-parser
```

### Stop Services

```bash
docker-compose down
```

### Rebuild After Changes

```bash
docker-compose build
docker-compose up -d
```

## Configuration

All components support environment variables:

```bash
# Example: Change port
export DOCLING_PARSER_PORT=8080

# Example: Change chunk size
export LANGCHAIN_SPLITTER_DEFAULT_CHUNK_SIZE=2000
```

Update `docker-compose.yml` to persist changes.

## Troubleshooting

### Port Already in Use

Change ports in `docker-compose.yml`:
```yaml
ports:
  - "26000:26000"  # Change first number
```

### Docker Build Fails

```bash
# Clean Docker cache
docker system prune -a

# Rebuild
docker-compose build --no-cache
```

### Service Won't Start

```bash
# Check logs
docker-compose logs <service-name>

# Check Docker status
docker-compose ps
```

## Getting Help

- **Documentation**: See README.md files in each directory
- **API Docs**: http://localhost:26000/docs and http://localhost:26001/docs
- **Examples**: Check `integrations/orchestrators/n8n/pipelines/`

## What's Next?

Consider adding:
1. **Embedding Component**: Generate embeddings for chunks
2. **Vector Store Component**: Store and retrieve chunks
3. **LLM Component**: Answer questions using retrieved chunks
4. **Additional Parsers**: Support more document formats
5. **Monitoring**: Add logging and metrics

Happy building! 🚀
