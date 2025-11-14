"""Tests for langchain-splitter API endpoints."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.schemas import Document, DocumentMetadata
from langchain_splitter.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client.

    Returns:
        FastAPI test client.
    """
    return TestClient(app)


class TestAPI:
    """Test suite for API endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "langchain-splitter"

    def test_split_text(self, client: TestClient) -> None:
        """Test split text endpoint."""
        response = client.post(
            "/api/v1/split/text",
            json={
                "text": "This is a test sentence. " * 50,
                "source_id": "test_source",
                "chunk_size": 100,
                "chunk_overlap": 20,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "chunks" in data["data"]
        assert len(data["data"]["chunks"]) > 0
        assert data["metadata"]["chunk_count"] > 0

    def test_split_text_minimal(self, client: TestClient) -> None:
        """Test split text with minimal parameters."""
        response = client.post(
            "/api/v1/split/text",
            json={
                "text": "Short text.",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["data"]["chunks"]) == 1

    def test_split_text_custom_separators(self, client: TestClient) -> None:
        """Test split text with custom separators."""
        response = client.post(
            "/api/v1/split/text",
            json={
                "text": "Sentence 1. Sentence 2. Sentence 3.",
                "chunk_size": 50,
                "chunk_overlap": 0,
                "separators": [". "],
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_split_document(self, client: TestClient) -> None:
        """Test split document endpoint."""
        document = {
            "doc_id": "test_doc_1",
            "content": "This is a document. " * 100,
            "metadata": {
                "source": "test.pdf",
                "file_type": "pdf",
            },
            "tables": [],
        }

        response = client.post(
            "/api/v1/split/document",
            json={
                "document": document,
                "chunk_size": 100,
                "chunk_overlap": 20,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["data"]["chunks"]) > 0
        assert data["metadata"]["source_doc_id"] == "test_doc_1"

        # Verify chunks have correct source_doc_id
        for chunk in data["data"]["chunks"]:
            assert chunk["metadata"]["source_doc_id"] == "test_doc_1"

    def test_split_documents(self, client: TestClient) -> None:
        """Test split multiple documents endpoint."""
        documents = [
            {
                "doc_id": f"doc_{i}",
                "content": f"Document {i} content. " * 50,
                "metadata": {
                    "source": f"test_{i}.pdf",
                    "file_type": "pdf",
                },
                "tables": [],
            }
            for i in range(3)
        ]

        response = client.post(
            "/api/v1/split/documents",
            json={
                "documents": documents,
                "chunk_size": 100,
                "chunk_overlap": 20,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert len(data["data"]["chunks"]) > 0
        assert data["metadata"]["document_count"] == 3

        # Verify we have chunks from all documents
        source_ids = set(
            chunk["metadata"]["source_doc_id"]
            for chunk in data["data"]["chunks"]
        )
        assert len(source_ids) == 3

    def test_analyze_text(self, client: TestClient) -> None:
        """Test analyze text endpoint."""
        response = client.post(
            "/api/v1/analyze",
            params={
                "text": "This is a test. " * 100,
                "chunk_size": 100,
                "chunk_overlap": 20,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "chunk_count" in data["data"]
        assert "text_length" in data["data"]
        assert data["data"]["chunk_count"] > 0

    def test_analyze_text_default_params(self, client: TestClient) -> None:
        """Test analyze text with default parameters."""
        response = client.post(
            "/api/v1/analyze",
            params={
                "text": "Short text.",
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["chunk_count"] == 1

    def test_chunk_structure(self, client: TestClient) -> None:
        """Test that returned chunks have correct structure."""
        response = client.post(
            "/api/v1/split/text",
            json={
                "text": "Test text for structure validation.",
                "source_id": "struct_test",
            }
        )

        assert response.status_code == 200
        data = response.json()
        chunks = data["data"]["chunks"]

        for chunk in chunks:
            # Verify required fields
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk

            # Verify metadata structure
            metadata = chunk["metadata"]
            assert "source_doc_id" in metadata
            assert "chunk_index" in metadata
            assert metadata["source_doc_id"] == "struct_test"

    def test_processing_metadata(self, client: TestClient) -> None:
        """Test that response includes processing metadata."""
        response = client.post(
            "/api/v1/split/text",
            json={
                "text": "Test text.",
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Verify metadata
        assert "metadata" in data
        assert "processing_time_ms" in data["metadata"]
        assert "chunk_count" in data["metadata"]
        assert "original_length" in data["metadata"]
        assert data["metadata"]["processing_time_ms"] >= 0
