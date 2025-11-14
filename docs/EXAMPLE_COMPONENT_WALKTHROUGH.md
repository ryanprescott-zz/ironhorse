# Example Component Walkthrough

Complete walkthrough of creating a real component from scratch: an Embedding Generator using OpenAI.

## Overview

We'll create a component that:
- Generates embeddings using OpenAI API
- Supports batch processing
- Follows all AI Toolkit standards
- Includes complete tests
- Is Docker-ready

**Component Name**: `embedding-generator`
**Port**: `26002`
**Purpose**: Generate embeddings for text and chunks

---

## Step 1: Create Component Structure

```bash
cd ~/projects/ai-toolkit/components

# Run cookiecutter
cookiecutter ../templates/component
```

**Answers to prompts**:
```
component_name: embedding-generator
component_class_name: EmbeddingGenerator
component_description: Generate embeddings using OpenAI API
component_port: 26002
author: Your Name
python_version: 3.11
```

**Result**:
```
components/embedding-generator/
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

## Step 2: Setup Python Environment

```bash
cd embedding-generator

# Create virtual environment
python3.11 -m venv .venv

# Activate it
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install shared schemas
cd ../..
pip install -e shared/

# Go back to component
cd components/embedding-generator

# Install component with dev dependencies
pip install -e ".[dev]"
```

**Verify installation**:
```bash
python -c "from shared.schemas import Document; print('✓ Shared schemas OK')"
python -c "from embedding_generator.core import EmbeddingGenerator; print('✓ Component OK')"
```

---

## Step 3: Add Dependencies

Edit `pyproject.toml`:

```toml
[project]
name = "embedding-generator"
version = "0.1.0"
description = "Generate embeddings using OpenAI API"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "openai>=1.0.0",  # Add OpenAI
    "numpy>=1.24.0",  # For vector operations
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.25.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Reinstall to get new dependencies**:
```bash
pip install -e ".[dev]"
```

---

## Step 4: Implement Configuration

Edit `config/settings.py`:

```python
"""Settings for embedding-generator component.

This module defines all configuration settings using Pydantic Settings,
allowing configuration via environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for embedding-generator.

    All settings can be overridden using environment variables with the
    prefix 'EMBEDDING_GENERATOR_'.

    Attributes:
        host: API server host.
        port: API server port.
        log_level: Logging level.
        openai_api_key: OpenAI API key.
        openai_model: OpenAI model to use.
        max_batch_size: Maximum batch size for embeddings.
        embedding_dimension: Expected embedding dimension.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_GENERATOR_",
        case_sensitive=False,
    )

    # API Settings
    host: str = "0.0.0.0"
    port: int = 26002
    log_level: str = "INFO"

    # OpenAI Settings
    openai_api_key: str = ""
    openai_model: str = "text-embedding-ada-002"
    openai_organization: str = ""

    # Processing Settings
    max_batch_size: int = 100
    embedding_dimension: int = 1536  # For text-embedding-ada-002


# Global settings instance
settings = Settings()
```

---

## Step 5: Implement Core Logic

Edit `core/embedding_generator.py`:

```python
"""EmbeddingGenerator implementation.

This module provides the core functionality for generating embeddings using OpenAI.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

# Add shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.schemas import Chunk, Document

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class EmbeddingGenerator:
    """Generate embeddings using OpenAI.

    This class provides embedding generation capabilities for text,
    documents, and chunks using the OpenAI API.

    Attributes:
        client: OpenAI client instance.
        model: OpenAI model name.
        max_batch_size: Maximum batch size for API calls.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-ada-002",
        max_batch_size: int = 100,
        organization: str = "",
    ) -> None:
        """Initialize the EmbeddingGenerator.

        Args:
            api_key: OpenAI API key.
            model: OpenAI model name.
            max_batch_size: Maximum batch size.
            organization: OpenAI organization ID.

        Raises:
            ImportError: If openai package is not installed.
            ValueError: If api_key is empty.
        """
        if OpenAI is None:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.model = model
        self.max_batch_size = max_batch_size

        # Initialize OpenAI client
        client_kwargs = {"api_key": api_key}
        if organization:
            client_kwargs["organization"] = organization

        self.client = OpenAI(**client_kwargs)

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to generate embedding for.

        Returns:
            Embedding vector as list of floats.

        Raises:
            Exception: If API call fails.
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        This method automatically batches requests to respect API limits.

        Args:
            texts: List of texts to generate embeddings for.

        Returns:
            List of embedding vectors.

        Raises:
            Exception: If API call fails.
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]

            response = self.client.embeddings.create(
                input=batch,
                model=self.model
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_chunks(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks.

        Args:
            chunks: List of chunks to embed.

        Returns:
            List of dicts with chunk info and embeddings.
        """
        if not chunks:
            return []

        # Extract texts
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts)

        # Combine with chunk data
        results = []
        for chunk, embedding in zip(chunks, embeddings):
            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "embedding": embedding,
                "embedding_dimension": len(embedding),
                "metadata": chunk.metadata.model_dump()
            })

        return results

    def embed_document(self, document: Document) -> Dict[str, Any]:
        """Generate embedding for a document's content.

        Args:
            document: Document to embed.

        Returns:
            Dict with document info and embedding.
        """
        embedding = self.generate_embedding(document.content)

        return {
            "doc_id": document.doc_id,
            "content": document.content,
            "embedding": embedding,
            "embedding_dimension": len(embedding),
            "metadata": document.metadata.model_dump()
        }

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the current model.

        Returns:
            Embedding dimension (e.g., 1536 for text-embedding-ada-002).
        """
        # Generate a test embedding to get dimension
        test_embedding = self.generate_embedding("test")
        return len(test_embedding)
```

---

## Step 6: Implement API Routes

Edit `api/routes.py`:

```python
"""API routes for embedding-generator component.

This module defines all FastAPI endpoints for the embedding generator.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Add shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.schemas import Chunk, Document, APIResponse
from embedding_generator.core import EmbeddingGenerator
from embedding_generator.config import settings

router = APIRouter()


class EmbedRequest(BaseModel):
    """Request model for embedding single text.

    Attributes:
        text: Text to embed.
    """

    text: str = Field(..., description="Text to embed", min_length=1)


class EmbedBatchRequest(BaseModel):
    """Request model for embedding multiple texts.

    Attributes:
        texts: List of texts to embed.
    """

    texts: List[str] = Field(
        ...,
        description="List of texts to embed",
        min_length=1
    )


class EmbedChunksRequest(BaseModel):
    """Request model for embedding chunks.

    Attributes:
        chunks: List of chunks to embed.
    """

    chunks: List[Chunk] = Field(
        ...,
        description="List of chunks to embed",
        min_length=1
    )


class EmbedDocumentRequest(BaseModel):
    """Request model for embedding a document.

    Attributes:
        document: Document to embed.
    """

    document: Document = Field(..., description="Document to embed")


@router.post("/embed", response_model=APIResponse[Dict[str, Any]])
async def embed_text(request: EmbedRequest) -> Dict[str, Any]:
    """Generate embedding for text.

    Args:
        request: Embed request.

    Returns:
        API response with embedding.

    Raises:
        HTTPException: If embedding generation fails.
    """
    start_time = time.time()

    try:
        generator = EmbeddingGenerator(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            organization=settings.openai_organization
        )

        embedding = generator.generate_embedding(request.text)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return APIResponse.success(
            data={
                "text": request.text,
                "embedding": embedding,
                "dimension": len(embedding),
                "model": settings.openai_model
            },
            metadata={"processing_time_ms": processing_time_ms}
        ).model_dump()

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {str(e)}"
        )


@router.post("/embed/batch", response_model=APIResponse[List[Dict[str, Any]]])
async def embed_texts_batch(request: EmbedBatchRequest) -> Dict[str, Any]:
    """Generate embeddings for multiple texts.

    Args:
        request: Batch embed request.

    Returns:
        API response with embeddings.

    Raises:
        HTTPException: If embedding generation fails.
    """
    start_time = time.time()

    try:
        generator = EmbeddingGenerator(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_batch_size=settings.max_batch_size,
            organization=settings.openai_organization
        )

        embeddings = generator.generate_embeddings_batch(request.texts)

        results = [
            {
                "text": text,
                "embedding": embedding,
                "dimension": len(embedding)
            }
            for text, embedding in zip(request.texts, embeddings)
        ]

        processing_time_ms = int((time.time() - start_time) * 1000)

        return APIResponse.success(
            data=results,
            metadata={
                "processing_time_ms": processing_time_ms,
                "text_count": len(request.texts),
                "model": settings.openai_model
            }
        ).model_dump()

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch embedding failed: {str(e)}"
        )


@router.post("/embed/chunks", response_model=APIResponse[List[Dict[str, Any]]])
async def embed_chunks(request: EmbedChunksRequest) -> Dict[str, Any]:
    """Generate embeddings for chunks.

    Args:
        request: Embed chunks request.

    Returns:
        API response with embeddings.

    Raises:
        HTTPException: If embedding generation fails.
    """
    start_time = time.time()

    try:
        generator = EmbeddingGenerator(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_batch_size=settings.max_batch_size,
            organization=settings.openai_organization
        )

        results = generator.embed_chunks(request.chunks)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return APIResponse.success(
            data=results,
            metadata={
                "processing_time_ms": processing_time_ms,
                "chunk_count": len(results),
                "model": settings.openai_model
            }
        ).model_dump()

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chunk embedding failed: {str(e)}"
        )


@router.post("/embed/document", response_model=APIResponse[Dict[str, Any]])
async def embed_document(request: EmbedDocumentRequest) -> Dict[str, Any]:
    """Generate embedding for a document.

    Args:
        request: Embed document request.

    Returns:
        API response with embedding.

    Raises:
        HTTPException: If embedding generation fails.
    """
    start_time = time.time()

    try:
        generator = EmbeddingGenerator(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            organization=settings.openai_organization
        )

        result = generator.embed_document(request.document)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return APIResponse.success(
            data=result,
            metadata={
                "processing_time_ms": processing_time_ms,
                "model": settings.openai_model
            }
        ).model_dump()

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document embedding failed: {str(e)}"
        )


@router.get("/info", response_model=APIResponse[Dict[str, Any]])
async def get_info() -> Dict[str, Any]:
    """Get embedding generator information.

    Returns:
        API response with generator info.
    """
    return APIResponse.success(
        data={
            "model": settings.openai_model,
            "max_batch_size": settings.max_batch_size,
            "expected_dimension": settings.embedding_dimension
        }
    ).model_dump()
```

---

## Step 7: Write Tests

Edit `tests/test_core.py`:

```python
"""Tests for embedding-generator core functionality."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.schemas import Chunk, ChunkMetadata, Document, DocumentMetadata
from embedding_generator.core import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator."""

    @patch('embedding_generator.core.embedding_generator.OpenAI')
    def test_initialization(self, mock_openai: Mock) -> None:
        """Test EmbeddingGenerator initialization."""
        generator = EmbeddingGenerator(api_key="test-key")
        assert generator is not None
        assert generator.model == "text-embedding-ada-002"
        mock_openai.assert_called_once()

    def test_initialization_no_api_key(self) -> None:
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="API key is required"):
            EmbeddingGenerator(api_key="")

    @patch('embedding_generator.core.embedding_generator.OpenAI')
    def test_generate_embedding(self, mock_openai: Mock) -> None:
        """Test generate_embedding method."""
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator(api_key="test-key")
        embedding = generator.generate_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    @patch('embedding_generator.core.embedding_generator.OpenAI')
    def test_generate_embeddings_batch(self, mock_openai: Mock) -> None:
        """Test generate_embeddings_batch method."""
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2]),
            Mock(embedding=[0.3, 0.4])
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator(api_key="test-key")
        embeddings = generator.generate_embeddings_batch(["text1", "text2"])

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2]
        assert embeddings[1] == [0.3, 0.4]

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

        generator = EmbeddingGenerator(api_key="test-key")

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

    @patch('embedding_generator.core.embedding_generator.OpenAI')
    def test_embed_document(self, mock_openai: Mock) -> None:
        """Test embed_document method."""
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = EmbeddingGenerator(api_key="test-key")

        document = Document(
            doc_id="doc_1",
            content="test document",
            metadata=DocumentMetadata()
        )

        result = generator.embed_document(document)

        assert result["doc_id"] == "doc_1"
        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert result["embedding_dimension"] == 3
```

Run tests:
```bash
pytest -v
```

---

## Step 8: Run Locally

```bash
# Set environment variables
export EMBEDDING_GENERATOR_OPENAI_API_KEY="your-api-key"
export EMBEDDING_GENERATOR_PORT=26002
export EMBEDDING_GENERATOR_LOG_LEVEL=DEBUG

# Run the server
python -m embedding_generator.api.main
```

Test it:
```bash
# Health check
curl http://localhost:26002/health

# API docs
open http://localhost:26002/docs

# Generate embedding
curl -X POST "http://localhost:26002/api/v1/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

---

## Step 9: Create Docker Support

Create `platform/docker/components/embedding-generator/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY shared/ /app/shared/
COPY components/embedding-generator/ /app/embedding-generator/

WORKDIR /app/embedding-generator
RUN uv pip install --system -e .

EXPOSE 26002

ENV PYTHONPATH=/app

CMD ["python", "-m", "embedding_generator.api.main"]
```

Update `docker-compose.yml`:

```yaml
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

Build and run:
```bash
cd /path/to/ai-toolkit
export OPENAI_API_KEY="your-key"
docker-compose build embedding-generator
docker-compose up -d embedding-generator
docker-compose logs -f embedding-generator
```

---

## Step 10: Complete!

You now have a fully functional component:

✅ Core logic implemented
✅ API endpoints created
✅ Tests written and passing
✅ Configuration manageable via environment
✅ Docker support added
✅ Following all AI Toolkit standards

---

## Next Steps

1. **Add N8N binding** in `integrations/orchestrators/n8n/bindings/`
2. **Update component README** with specific usage examples
3. **Create example workflow** showing integration with other components
4. **Add monitoring** and logging
5. **Optimize performance** for production use

---

## Summary

This walkthrough demonstrated:
- Using cookie cutter template
- Setting up Python environment
- Implementing core functionality
- Creating API endpoints
- Writing comprehensive tests
- Running locally for development
- Adding Docker support
- Integrating with the project

The same pattern applies to any component you create!
