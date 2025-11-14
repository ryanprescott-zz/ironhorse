# Component Development Guide

Complete guide for creating and developing new AI Toolkit components from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Creating a New Component](#creating-a-new-component)
3. [Python Environment Setup](#python-environment-setup)
4. [Installing Dependencies](#installing-dependencies)
5. [Implementing Core Functionality](#implementing-core-functionality)
6. [Testing Your Component](#testing-your-component)
7. [Running Locally](#running-locally)
8. [Docker Integration](#docker-integration)
9. [Adding to Orchestration](#adding-to-orchestration)
10. [Best Practices](#best-practices)

---

## Prerequisites

### Required Software

1. **Python 3.11 or higher**
   ```bash
   python --version  # Should be 3.11+
   ```

   If you need to install Python 3.11:
   - **macOS**: `brew install python@3.11`
   - **Ubuntu**: `sudo apt install python3.11 python3.11-venv`
   - **Windows**: Download from python.org

2. **uv (Python package manager)**
   ```bash
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Or with pip
   pip install uv

   # Verify installation
   uv --version
   ```

3. **cookiecutter**
   ```bash
   pip install cookiecutter

   # Or with uv
   uv pip install cookiecutter

   # Verify installation
   cookiecutter --version
   ```

4. **Docker and Docker Compose** (for containerization)
   ```bash
   docker --version
   docker-compose --version
   ```

5. **Git** (recommended)
   ```bash
   git --version
   ```

---

## Creating a New Component

### Step 1: Navigate to Components Directory

```bash
cd /path/to/ai-toolkit/components
```

### Step 2: Run Cookie Cutter

```bash
cookiecutter ../templates/component
```

### Step 3: Answer the Prompts

You'll be asked for the following information:

```
component_name [my-component]: embedding-generator
component_class_name [MyComponent]: EmbeddingGenerator
component_description [A reusable component for AI Toolkit]: Generate embeddings using OpenAI or HuggingFace
component_port [26000]: 26002
author [AI Toolkit Team]: Your Name
python_version [3.11]: 3.11
```

**Naming Conventions**:
- `component_name`: kebab-case (lowercase with hyphens)
- `component_class_name`: PascalCase (capitalized words, no spaces)
- `component_port`: Pick next available port (26000, 26001, 26002, etc.)

### Example Output

```
components/
└── embedding-generator/
    ├── core/
    │   ├── __init__.py
    │   └── embedding_generator.py
    ├── api/
    │   ├── __init__.py
    │   ├── main.py
    │   └── routes.py
    ├── config/
    │   ├── __init__.py
    │   └── settings.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_core.py
    │   └── test_api.py
    ├── pyproject.toml
    └── README.md
```

---

## Python Environment Setup

### Option 1: Using venv (Standard Python)

#### 1.1 Create Virtual Environment

```bash
cd embedding-generator

# Create venv
python3.11 -m venv .venv

# Activate venv
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Verify activation (should show .venv path)
which python
```

#### 1.2 Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Option 2: Using uv (Recommended)

#### 2.1 Create Virtual Environment with uv

```bash
cd embedding-generator

# Create venv with uv (much faster)
uv venv

# Activate venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

---

## Installing Dependencies

### Step 1: Install Shared Schemas

The component needs access to the shared schemas. From your component directory:

```bash
# Navigate to project root
cd ../..

# Install shared schemas in editable mode
pip install -e shared/

# Or with uv (faster)
uv pip install -e shared/
```

### Step 2: Install Component Dependencies

Navigate back to your component:

```bash
cd components/embedding-generator

# Install component in editable mode with dev dependencies
pip install -e ".[dev]"

# Or with uv (faster)
uv pip install -e ".[dev]"
```

### Step 3: Add Component-Specific Dependencies

Edit `pyproject.toml` to add any additional dependencies:

```toml
[project]
name = "embedding-generator"
version = "0.1.0"
description = "Generate embeddings using OpenAI or HuggingFace"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    # Add your dependencies here
    "openai>=1.0.0",
    "sentence-transformers>=2.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.25.0",
]
```

Then reinstall:

```bash
pip install -e ".[dev]"
# or
uv pip install -e ".[dev]"
```

### Step 4: Verify Installation

```bash
# Check installed packages
pip list | grep ai-toolkit

# Should show:
# shared-schemas          0.1.0
# embedding-generator  0.1.0

# Verify imports work
python -c "from shared.schemas import Document, Chunk, APIResponse; print('Success!')"
python -c "from embedding_generator.core import EmbeddingGenerator; print('Success!')"
```

---

## Implementing Core Functionality

### Step 1: Update Core Implementation

Edit `core/embedding_generator.py`:

```python
"""EmbeddingGenerator implementation.

This module provides the core functionality for generating embeddings.
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.schemas import Chunk, Document


class EmbeddingGenerator:
    """Generate embeddings using OpenAI or HuggingFace.

    This class provides embedding generation capabilities for documents and chunks.

    Attributes:
        provider: Embedding provider ('openai' or 'huggingface').
        model: Model name to use for embeddings.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "text-embedding-ada-002"
    ) -> None:
        """Initialize the EmbeddingGenerator.

        Args:
            provider: Embedding provider ('openai' or 'huggingface').
            model: Model name to use.

        Raises:
            ValueError: If provider is not supported.
        """
        if provider not in ["openai", "huggingface"]:
            raise ValueError(f"Unsupported provider: {provider}")

        self.provider = provider
        self.model = model

        # Initialize provider-specific clients
        if provider == "openai":
            # Import and initialize OpenAI client
            from openai import OpenAI
            self.client = OpenAI()
        elif provider == "huggingface":
            # Import and initialize HuggingFace model
            from sentence_transformers import SentenceTransformer
            self.client = SentenceTransformer(model)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to generate embedding for.

        Returns:
            Embedding vector as list of floats.

        Raises:
            Exception: If embedding generation fails.
        """
        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding

        elif self.provider == "huggingface":
            embedding = self.client.encode(text)
            return embedding.tolist()

        raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to generate embeddings for.

        Returns:
            List of embedding vectors.
        """
        if self.provider == "openai":
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [item.embedding for item in response.data]

        elif self.provider == "huggingface":
            embeddings = self.client.encode(texts)
            return [emb.tolist() for emb in embeddings]

        raise ValueError(f"Unsupported provider: {self.provider}")

    def embed_chunks(self, chunks: List[Chunk]) -> List[dict]:
        """Generate embeddings for chunks.

        Args:
            chunks: List of chunks to embed.

        Returns:
            List of dicts with chunk info and embeddings.
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.generate_embeddings_batch(texts)

        results = []
        for chunk, embedding in zip(chunks, embeddings):
            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": embedding,
                "metadata": chunk.metadata.model_dump()
            })

        return results
```

### Step 2: Update Configuration

Edit `config/settings.py`:

```python
"""Settings for embedding-generator component."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for embedding-generator.

    All settings can be overridden using environment variables with the
    prefix 'EMBEDDING_GENERATOR_'.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_GENERATOR_",
        case_sensitive=False,
    )

    # API Settings
    host: str = "0.0.0.0"
    port: int = 26002
    log_level: str = "INFO"

    # Embedding Settings
    default_provider: str = "openai"
    default_model: str = "text-embedding-ada-002"
    openai_api_key: str = ""  # Set via environment variable


settings = Settings()
```

### Step 3: Update API Routes

Edit `api/routes.py`:

```python
"""API routes for embedding-generator component."""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Add shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.schemas import Chunk, APIResponse
from embedding_generator.core import EmbeddingGenerator
from embedding_generator.config import settings

router = APIRouter()


class EmbedRequest(BaseModel):
    """Request model for embedding endpoint."""

    text: str = Field(..., description="Text to embed")
    provider: str = Field(
        settings.default_provider,
        description="Embedding provider"
    )
    model: str = Field(
        settings.default_model,
        description="Model to use"
    )


class EmbedChunksRequest(BaseModel):
    """Request model for embedding chunks."""

    chunks: List[Chunk] = Field(..., description="Chunks to embed")
    provider: str = Field(
        settings.default_provider,
        description="Embedding provider"
    )
    model: str = Field(
        settings.default_model,
        description="Model to use"
    )


@router.post("/embed", response_model=APIResponse[Dict[str, Any]])
async def embed_text(request: EmbedRequest) -> Dict[str, Any]:
    """Generate embedding for text.

    Args:
        request: Embed request.

    Returns:
        API response with embedding.
    """
    start_time = time.time()

    try:
        generator = EmbeddingGenerator(
            provider=request.provider,
            model=request.model
        )

        embedding = generator.generate_embedding(request.text)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return APIResponse.success(
            data={
                "text": request.text,
                "embedding": embedding,
                "dimension": len(embedding)
            },
            metadata={"processing_time_ms": processing_time_ms}
        ).model_dump()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding failed: {str(e)}"
        )


@router.post("/embed/chunks", response_model=APIResponse[List[Dict[str, Any]]])
async def embed_chunks(request: EmbedChunksRequest) -> Dict[str, Any]:
    """Generate embeddings for chunks.

    Args:
        request: Embed chunks request.

    Returns:
        API response with embeddings.
    """
    start_time = time.time()

    try:
        generator = EmbeddingGenerator(
            provider=request.provider,
            model=request.model
        )

        results = generator.embed_chunks(request.chunks)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return APIResponse.success(
            data=results,
            metadata={
                "processing_time_ms": processing_time_ms,
                "chunk_count": len(results)
            }
        ).model_dump()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding failed: {str(e)}"
        )
```

---

## Testing Your Component

### Step 1: Write Unit Tests

Edit `tests/test_core.py`:

```python
"""Tests for embedding-generator core functionality."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.schemas import Chunk, ChunkMetadata
from embedding_generator.core import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator."""

    def test_initialization(self) -> None:
        """Test EmbeddingGenerator initialization."""
        with patch('embedding_generator.core.embedding_generator.OpenAI'):
            generator = EmbeddingGenerator(provider="openai")
            assert generator is not None
            assert generator.provider == "openai"

    def test_invalid_provider(self) -> None:
        """Test initialization with invalid provider."""
        with pytest.raises(ValueError):
            EmbeddingGenerator(provider="invalid")

    @patch('embedding_generator.core.embedding_generator.OpenAI')
    def test_generate_embedding(self, mock_openai: Mock) -> None:
        """Test generate_embedding method."""
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator(provider="openai")
        embedding = generator.generate_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    @patch('embedding_generator.core.embedding_generator.OpenAI')
    def test_embed_chunks(self, mock_openai: Mock) -> None:
        """Test embed_chunks method."""
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2]),
            Mock(embedding=[0.3, 0.4])
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator(provider="openai")

        chunks = [
            Chunk(
                chunk_id="chunk_1",
                text="text 1",
                metadata=ChunkMetadata()
            ),
            Chunk(
                chunk_id="chunk_2",
                text="text 2",
                metadata=ChunkMetadata()
            )
        ]

        results = generator.embed_chunks(chunks)

        assert len(results) == 2
        assert results[0]["chunk_id"] == "chunk_1"
        assert results[0]["embedding"] == [0.1, 0.2]
        assert results[1]["chunk_id"] == "chunk_2"
        assert results[1]["embedding"] == [0.3, 0.4]
```

### Step 2: Write API Tests

Edit `tests/test_api.py` (similar pattern).

### Step 3: Run Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=embedding_generator --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::TestEmbeddingGenerator::test_initialization
```

### Step 4: View Coverage Report

```bash
# Coverage report is generated in htmlcov/
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

---

## Running Locally

### Step 1: Set Environment Variables

```bash
# Create .env file (don't commit this!)
cat > .env << EOF
EMBEDDING_GENERATOR_HOST=0.0.0.0
EMBEDDING_GENERATOR_PORT=26002
EMBEDDING_GENERATOR_LOG_LEVEL=DEBUG
EMBEDDING_GENERATOR_DEFAULT_PROVIDER=openai
EMBEDDING_GENERATOR_OPENAI_API_KEY=your-api-key-here
EOF

# Load environment variables
export $(cat .env | xargs)
```

### Step 2: Run the API Server

```bash
# Method 1: Direct Python execution
python -m embedding_generator.api.main

# Method 2: With uvicorn (more control)
uvicorn embedding_generator.api.main:app --reload --host 0.0.0.0 --port 26002

# Method 3: With specific log level
uvicorn embedding_generator.api.main:app --reload --log-level debug
```

### Step 3: Test the API

```bash
# Health check
curl http://localhost:26002/health

# API documentation
open http://localhost:26002/docs

# Test embedding endpoint
curl -X POST "http://localhost:26002/api/v1/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "provider": "openai",
    "model": "text-embedding-ada-002"
  }'
```

---

## Docker Integration

### Step 1: Create Dockerfile

Create `platform/docker/components/embedding-generator/Dockerfile`:

```dockerfile
# Dockerfile for Embedding Generator component
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN pip install uv

# Copy shared schemas
COPY shared/ /app/shared/

# Copy component files
COPY components/embedding-generator/ /app/embedding-generator/

# Install dependencies
WORKDIR /app/embedding-generator
RUN uv pip install --system -e .

# Expose the API port
EXPOSE 26002

# Set Python path
ENV PYTHONPATH=/app

# Run the FastAPI application
CMD ["python", "-m", "embedding_generator.api.main"]
```

### Step 2: Create .dockerignore

Create `platform/docker/components/embedding-generator/.dockerignore`:

```
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/
.venv/
venv/
.DS_Store
*.swp
tests/test_data/
.env
```

### Step 3: Add to Docker Compose

Edit `docker-compose.yml` in project root:

```yaml
services:
  # ... existing services ...

  # Embedding Generator Component
  embedding-generator:
    build:
      context: .
      dockerfile: platform/docker/components/embedding-generator/Dockerfile
    container_name: embedding-generator
    ports:
      - "26002:26002"
    environment:
      - EMBEDDING_GENERATOR_HOST=0.0.0.0
      - EMBEDDING_GENERATOR_PORT=26002
      - EMBEDDING_GENERATOR_LOG_LEVEL=INFO
      - EMBEDDING_GENERATOR_DEFAULT_PROVIDER=openai
      - EMBEDDING_GENERATOR_OPENAI_API_KEY=${OPENAI_API_KEY}
    networks:
      - ai-toolkit-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:26002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Step 4: Build and Run

```bash
# Build the image
docker-compose build embedding-generator

# Run the service
docker-compose up -d embedding-generator

# View logs
docker-compose logs -f embedding-generator

# Test health check
curl http://localhost:26002/health
```

---

## Adding to Orchestration

### Step 1: Create N8N Binding

Create `integrations/orchestrators/n8n/bindings/embedding_generator_binding.py`:

```python
"""N8N binding for Embedding Generator component."""

from typing import Any, Dict, List
import requests


def generate_embedding(
    text: str,
    provider: str = "openai",
    model: str = "text-embedding-ada-002",
    base_url: str = "http://embedding-generator:26002"
) -> Dict[str, Any]:
    """Generate embedding for text.

    Args:
        text: Text to embed.
        provider: Embedding provider.
        model: Model to use.
        base_url: Base URL of the service.

    Returns:
        API response with embedding.
    """
    url = f"{base_url}/api/v1/embed"
    payload = {
        "text": text,
        "provider": provider,
        "model": model
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def embed_chunks(
    chunks: List[Dict[str, Any]],
    provider: str = "openai",
    model: str = "text-embedding-ada-002",
    base_url: str = "http://embedding-generator:26002"
) -> Dict[str, Any]:
    """Generate embeddings for chunks.

    Args:
        chunks: List of chunks to embed.
        provider: Embedding provider.
        model: Model to use.
        base_url: Base URL of the service.

    Returns:
        API response with embeddings.
    """
    url = f"{base_url}/api/v1/embed/chunks"
    payload = {
        "chunks": chunks,
        "provider": provider,
        "model": model
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()
```

### Step 2: Update N8N Binding __init__.py

Edit `integrations/orchestrators/n8n/bindings/__init__.py`:

```python
"""N8N bindings for AI Toolkit components."""

from integrations.orchestrators.n8n.bindings.docling_parser_binding import (
    parse_document,
    parse_documents,
    get_supported_formats,
)
from integrations.orchestrators.n8n.bindings.langchain_splitter_binding import (
    split_text,
    split_document,
    split_documents,
    analyze_text,
)
from integrations.orchestrators.n8n.bindings.embedding_generator_binding import (
    generate_embedding,
    embed_chunks,
)

__all__ = [
    "parse_document",
    "parse_documents",
    "get_supported_formats",
    "split_text",
    "split_document",
    "split_documents",
    "analyze_text",
    "generate_embedding",
    "embed_chunks",
]
```

---

## Best Practices

### Code Quality

1. **Always use type hints**:
   ```python
   def process(self, data: Dict[str, Any]) -> List[str]:
       pass
   ```

2. **Write docstrings** (Google style):
   ```python
   def method(self, param: str) -> int:
       """Brief description.

       Longer description if needed.

       Args:
           param: Description of param.

       Returns:
           Description of return value.

       Raises:
           ValueError: When param is invalid.
       """
       pass
   ```

3. **Follow PEP 8**:
   ```bash
   # Format with black
   black .

   # Lint with ruff
   ruff check .
   ```

4. **Use Pydantic for validation**:
   ```python
   from pydantic import BaseModel, Field

   class MyRequest(BaseModel):
       value: str = Field(..., min_length=1, max_length=100)
   ```

### Testing

1. **Aim for >80% coverage**
2. **Test edge cases**
3. **Use mocks for external dependencies**
4. **Write both unit and integration tests**

### Configuration

1. **Use Pydantic Settings**
2. **Support environment variables**
3. **Provide sensible defaults**
4. **Document all settings**

### Documentation

1. **Update README.md** with your component specifics
2. **Document all public APIs**
3. **Provide usage examples**
4. **Include troubleshooting section**

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/embedding-generator

# Stage changes
git add components/embedding-generator/

# Commit with descriptive message
git commit -m "Add embedding generator component

- Implement core embedding logic
- Add FastAPI endpoints
- Include comprehensive tests
- Add Docker support"

# Push to remote
git push origin feature/embedding-generator
```

---

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'shared'`

**Solution**:
```bash
# Make sure shared is installed
pip install -e ../../shared/

# Verify Python path includes project root
echo $PYTHONPATH
```

### Virtual Environment Issues

**Problem**: Wrong Python version in venv

**Solution**:
```bash
# Remove old venv
rm -rf .venv

# Create new with specific Python version
python3.11 -m venv .venv
source .venv/bin/activate
```

### Test Failures

**Problem**: Tests can't find shared schemas

**Solution**:
```python
# Add to test file
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
```

### Docker Build Issues

**Problem**: Docker can't find files

**Solution**: Build from project root, not component directory:
```bash
# Wrong
cd components/embedding-generator
docker build .

# Right
cd /path/to/ai-toolkit
docker build -f platform/docker/components/embedding-generator/Dockerfile .
```

---

## Complete Example Session

Here's a complete terminal session creating a new component:

```bash
# 1. Prerequisites
python3.11 --version
pip install uv cookiecutter

# 2. Navigate and create
cd ~/projects/ai-toolkit/components
cookiecutter ../templates/component
# Answer prompts...

# 3. Setup environment
cd embedding-generator
python3.11 -m venv .venv
source .venv/bin/activate

# 4. Install dependencies
cd ../..
pip install -e shared/
cd components/embedding-generator
pip install -e ".[dev]"

# 5. Implement your code
# Edit core/embedding_generator.py, api/routes.py, etc.

# 6. Run tests
pytest -v

# 7. Run locally
export EMBEDDING_GENERATOR_PORT=26002
python -m embedding_generator.api.main

# 8. Test API (in another terminal)
curl http://localhost:26002/health
open http://localhost:26002/docs

# 9. Build Docker image
cd ../..
docker build -f platform/docker/components/embedding-generator/Dockerfile \
  -t embedding-generator:latest .

# 10. Add to docker-compose and test
docker-compose up -d embedding-generator
docker-compose logs -f embedding-generator
```

---

## Additional Resources

- **Project README**: `/README.md`
- **Quick Start**: `/QUICKSTART.md`
- **Existing Components**: Study `docling-parser` and `langchain-splitter`
- **Shared Schemas**: `shared/schemas/`
- **Template**: `templates/component/`

---

## Next Steps

1. ✅ Create your component with cookie cutter
2. ✅ Set up Python environment
3. ✅ Implement core functionality
4. ✅ Write comprehensive tests
5. ✅ Create Docker support
6. ✅ Add to orchestration
7. ✅ Update documentation
8. ✅ Submit for review

Happy coding! 🚀
