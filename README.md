# AI Toolkit

Reusable AI development toolkit with components and orchestration framework.

## Overview

AI Toolkit is a comprehensive toolkit for building AI development projects, providing:

1. **Reusable Components**: A collection of modular, containerized components for common AI tasks
2. **Orchestration Framework**: Tools for defining, deploying, and monitoring processing pipelines
3. **Standard Data Models**: Consistent schemas and interfaces across all components
4. **Development Templates**: Cookie cutter templates for rapid component development

## Project Structure

```
project-root/
├── components/              # Reusable component collection
│   ├── docling-parser/     # Document parser using Docling
│   └── langchain-splitter/ # Text splitter using LangChain
├── shared/                  # Shared schemas and utilities
│   └── schemas/            # Common data models (Document, Chunk, APIResponse)
├── integrations/           # Orchestrator integrations
│   └── orchestrators/
│       └── n8n/           # N8N orchestrator bindings and workflows
├── platform/               # Deployment configurations
│   └── docker/            # Docker configurations
│       ├── components/    # Component Dockerfiles
│       └── orchestrators/ # Orchestrator Dockerfiles
├── templates/              # Cookie cutter templates
│   └── component/         # Component template
├── data/                   # Local data directory
│   └── documents/         # Test documents
└── docker-compose.yml      # Local development setup
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- uv package manager (optional, for local development)

### Running with Docker Compose

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd project-root
   ```

2. Start all services:
   ```bash
   docker-compose up -d
   ```

3. Access the services:
   - Docling Parser API: http://localhost:26000/docs
   - LangChain Splitter API: http://localhost:26001/docs
   - N8N Orchestrator: http://localhost:5678 (admin/admin)

4. Check service health:
   ```bash
   curl http://localhost:26000/health
   curl http://localhost:26001/health
   ```

### Using Components

#### Parse a Document

```bash
curl -X POST "http://localhost:26000/api/v1/parse/single?file_path=/data/documents/test.pdf&extract_tables=true"
```

#### Split Text

```bash
curl -X POST "http://localhost:26001/api/v1/split/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

## Components

### Docling Parser

Document parser supporting PDF, DOCX, HTML, XLSX, and PPTX formats.

- **Location**: `components/docling-parser/`
- **Port**: 26000
- **Documentation**: [Docling Parser README](components/docling-parser/README.md)

**Features:**
- Multiple format support
- Table extraction
- Batch processing
- Standard Document schema output

### LangChain Splitter

Text splitter using LangChain's RecursiveCharacterTextSplitter.

- **Location**: `components/langchain-splitter/`
- **Port**: 26001
- **Documentation**: [LangChain Splitter README](components/langchain-splitter/README.md)

**Features:**
- Configurable chunk size and overlap
- Custom separators
- Document and batch processing
- Standard Chunk schema output

## Orchestration

### N8N Workflows

The project includes n8n integration for building RAG pipelines.

- **Access**: http://localhost:5678
- **Credentials**: admin/admin (change in production!)
- **Workflows**: `integrations/orchestrators/n8n/pipelines/`

**Example Workflow:**
1. Parse document with Docling Parser
2. Split into chunks with LangChain Splitter
3. Process chunks downstream (embeddings, storage, etc.)

See [N8N Integration Guide](integrations/orchestrators/n8n/bindings/README.md) for details.

## Development

### Developer Documentation

Comprehensive guides for component development:

- **🐍 [Python Environment Setup](docs/PYTHON_ENVIRONMENT_SETUP.md)** - Complete guide for setting up your Python environment:
  - Virtual environment creation and activation
  - Installing dependencies and shared schemas
  - Running tests with pytest
  - Coverage reports and debugging
  - Common issues and solutions

- **📘 [Component Development Guide](docs/COMPONENT_DEVELOPMENT_GUIDE.md)** - Complete guide for creating new components from scratch:
  - Prerequisites and installation
  - Cookie cutter usage
  - Step-by-step implementation
  - Testing, Docker integration, and orchestration

- **⚡ [Developer Quick Reference](docs/DEVELOPER_QUICK_REFERENCE.md)** - Quick reference card for:
  - Common commands and shortcuts
  - File locations and structure
  - Testing and code quality tools
  - Docker commands
  - Troubleshooting tips

- **🎯 [Example Component Walkthrough](docs/EXAMPLE_COMPONENT_WALKTHROUGH.md)** - Complete real-world example:
  - Building an OpenAI Embedding Generator
  - Full implementation with code
  - Testing and deployment
  - End-to-end walkthrough

### Creating New Components

Use the cookie cutter template:

```bash
cd components
cookiecutter ../templates/component

# Follow the prompts:
# component_name: my-new-component
# component_class_name: MyNewComponent
# component_description: Description of the component
# component_port: 26002
```

See [Template Documentation](templates/README.md) for details.

### Local Development Setup

1. Install dependencies:
   ```bash
   pip install uv
   uv pip install -e ".[dev]"
   ```

2. Install component dependencies:
   ```bash
   cd components/docling-parser
   uv pip install -e ".[dev]"
   ```

3. Run tests:
   ```bash
   pytest components/docling-parser/tests/
   pytest components/langchain-splitter/tests/
   ```

4. Run a component locally:
   ```bash
   cd components/docling-parser
   python -m docling_parser.api.main
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=components --cov=shared --cov-report=html

# Run specific component tests
pytest components/docling-parser/tests/
```

### Code Quality

The project uses:
- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing

```bash
# Format code
black .

# Lint
ruff check .

# Type check
mypy components/ shared/
```

## Architecture

### Component Architecture

Each component follows a standard structure:

```
component-name/
├── core/           # Core Python module (pip installable)
├── api/            # FastAPI web service
├── config/         # Pydantic settings
├── tests/          # Pytests
└── pyproject.toml  # Package configuration
```

### Data Models

All components use standard Pydantic models from `shared/schemas/`:

- **Document**: Standard document structure (content, metadata, tables)
- **Chunk**: Standard text chunk structure (text, metadata, source)
- **APIResponse**: Standard API response wrapper (status, data, error)

### Technology Stack

- **Python**: 3.11+
- **Package Management**: uv
- **Data Models**: Pydantic v2
- **Configuration**: Pydantic Settings
- **Web Framework**: FastAPI
- **Server**: Uvicorn
- **Testing**: Pytest
- **Containerization**: Docker
- **Orchestration**: Docker Compose, N8N

## Deployment

### Building Docker Images

```bash
# Build all images
docker-compose build

# Build specific component
docker build -f platform/docker/components/docling-parser/Dockerfile -t docling-parser:latest .
```

### Production Deployment

For production deployments:

1. Use company-specific base images
2. Configure proper authentication and secrets
3. Set up SSL/TLS
4. Use environment-specific configurations
5. Implement monitoring and logging
6. Set up CI/CD pipelines (GitLab CI recommended)

See component READMEs for deployment details.

## Configuration

All components use Pydantic Settings with environment variable support.

### Environment Variables

#### Docling Parser
- `DOCLING_PARSER_HOST`: API host (default: 0.0.0.0)
- `DOCLING_PARSER_PORT`: API port (default: 26000)
- `DOCLING_PARSER_LOG_LEVEL`: Log level (default: INFO)
- `DOCLING_PARSER_MAX_FILE_SIZE_MB`: Max file size (default: 100)

#### LangChain Splitter
- `LANGCHAIN_SPLITTER_HOST`: API host (default: 0.0.0.0)
- `LANGCHAIN_SPLITTER_PORT`: API port (default: 26001)
- `LANGCHAIN_SPLITTER_LOG_LEVEL`: Log level (default: INFO)
- `LANGCHAIN_SPLITTER_DEFAULT_CHUNK_SIZE`: Chunk size (default: 1000)
- `LANGCHAIN_SPLITTER_DEFAULT_CHUNK_OVERLAP`: Overlap (default: 200)

## API Documentation

Each component provides interactive API documentation:

- Docling Parser: http://localhost:26000/docs
- LangChain Splitter: http://localhost:26001/docs

## Contributing

### Adding Components

1. Create component using cookie cutter template
2. Implement core functionality following project standards
3. Add comprehensive tests (>80% coverage)
4. Document all public APIs with Google-style docstrings
5. Ensure PEP 8 compliance
6. Create component README
7. Add Dockerfile and docker-compose service
8. Create n8n bindings if applicable

### Standards

- **Code Style**: PEP 8 compliant, Black formatted
- **Documentation**: Google-style docstrings
- **Testing**: Pytest with >80% coverage
- **Type Hints**: Required for all public APIs
- **Data Models**: Use Pydantic models and primitive types

## Roadmap

### Future Components

- Embedding component (OpenAI, HuggingFace)
- Vector store component (Pinecone, Weaviate, ChromaDB)
- LLM chat completion component
- RAG retrieval framework
- Document loader components

### Future Orchestrators

- Prefect integration
- Airflow integration
- Temporal integration

## Troubleshooting

### Common Issues

**Docker build fails:**
- Ensure Docker daemon is running
- Check available disk space
- Clear Docker cache: `docker system prune -a`

**Component health check fails:**
- Check logs: `docker-compose logs <service-name>`
- Verify port is not in use
- Ensure dependencies are installed

**N8N can't connect to components:**
- Verify all services are on same network
- Check service names match docker-compose.yml
- Test connectivity: `docker-compose exec n8n ping docling-parser`

## License

[Add license information]

## Support

For issues and questions:
- Create an issue in the project repository
- Contact the development team
- Check component-specific documentation

## Acknowledgments

- Docling: Document parsing library
- LangChain: LLM application framework
- N8N: Workflow automation tool
- FastAPI: Modern web framework
- Pydantic: Data validation library
