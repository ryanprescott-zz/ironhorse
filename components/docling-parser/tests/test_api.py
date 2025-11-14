"""Tests for docling-parser API endpoints."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.schemas import Document, DocumentMetadata
from docling_parser.api.main import app


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
        assert data["service"] == "docling-parser"

    @patch('docling_parser.api.routes.DoclingParser')
    def test_parse_single_document(
        self,
        mock_parser_class: MagicMock,
        client: TestClient
    ) -> None:
        """Test parse single document endpoint."""
        # Mock parser
        mock_parser = MagicMock()
        mock_doc = Document(
            doc_id="test_1",
            content="Test content",
            metadata=DocumentMetadata(source="test.pdf", file_type="pdf"),
            tables=[]
        )
        mock_parser.parse_document.return_value = mock_doc
        mock_parser_class.return_value = mock_parser

        # Create test file
        test_file = Path(__file__).parent / "test_data" / "api_test.pdf"
        test_file.parent.mkdir(exist_ok=True)
        test_file.touch()

        try:
            response = client.post(
                "/api/v1/parse/single",
                params={
                    "file_path": str(test_file),
                    "extract_tables": True,
                    "doc_id": "test_1"
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["data"]["doc_id"] == "test_1"
            assert "processing_time_ms" in data["metadata"]
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()

    @patch('docling_parser.api.routes.DoclingParser')
    def test_parse_documents_batch(
        self,
        mock_parser_class: MagicMock,
        client: TestClient
    ) -> None:
        """Test parse documents batch endpoint."""
        # Mock parser
        mock_parser = MagicMock()
        mock_docs = [
            Document(
                doc_id=f"test_{i}",
                content=f"Test content {i}",
                metadata=DocumentMetadata(source=f"test_{i}.pdf", file_type="pdf"),
                tables=[]
            )
            for i in range(2)
        ]
        mock_parser.parse_documents.return_value = mock_docs
        mock_parser_class.return_value = mock_parser

        # Create test files
        test_dir = Path(__file__).parent / "test_data"
        test_dir.mkdir(exist_ok=True)
        test_files = []

        for i in range(2):
            test_file = test_dir / f"batch_test_{i}.pdf"
            test_file.touch()
            test_files.append(str(test_file))

        try:
            response = client.post(
                "/api/v1/parse",
                json={
                    "file_paths": test_files,
                    "extract_tables": True
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert len(data["data"]["documents"]) == 2
            assert data["metadata"]["document_count"] == 2
        finally:
            # Cleanup
            for test_file in test_files:
                Path(test_file).unlink()

    @patch('docling_parser.api.routes.DoclingParser')
    def test_get_supported_formats(
        self,
        mock_parser_class: MagicMock,
        client: TestClient
    ) -> None:
        """Test get supported formats endpoint."""
        # Mock parser
        mock_parser = MagicMock()
        mock_parser.get_supported_formats.return_value = ["pdf", "docx", "html"]
        mock_parser_class.return_value = mock_parser

        response = client.get("/api/v1/formats")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "pdf" in data["data"]
        assert data["metadata"]["format_count"] == 3

    def test_parse_single_document_not_found(self, client: TestClient) -> None:
        """Test parse single document with non-existent file."""
        response = client.post(
            "/api/v1/parse/single",
            params={
                "file_path": "/nonexistent/file.pdf",
                "extract_tables": True
            }
        )

        assert response.status_code == 404
