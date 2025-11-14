"""Tests for docling-parser core functionality."""

import sys
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch, MagicMock

import pytest

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.schemas import Document, DocumentMetadata
from docling_parser.core import DoclingParser


class TestDoclingParser:
    """Test suite for DoclingParser."""

    @patch('docling_parser.core.docling_parser.DocumentConverter')
    def test_initialization(self, mock_converter: Mock) -> None:
        """Test DoclingParser initialization."""
        parser = DoclingParser(extract_tables=True)
        assert parser is not None
        assert parser.extract_tables is True
        mock_converter.assert_called_once()

    @patch('docling_parser.core.docling_parser.DocumentConverter')
    def test_parse_document(self, mock_converter: Mock) -> None:
        """Test parse_document method."""
        # Mock the converter result
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "# Test Document\n\nThis is a test."
        mock_result.document.tables = []

        mock_converter_instance = MagicMock()
        mock_converter_instance.convert.return_value = mock_result
        mock_converter.return_value = mock_converter_instance

        # Create test file
        test_file = Path(__file__).parent / "test_data" / "test.pdf"
        test_file.parent.mkdir(exist_ok=True)
        test_file.touch()

        try:
            parser = DoclingParser()
            document = parser.parse_document(str(test_file), doc_id="test_doc_1")

            assert isinstance(document, Document)
            assert document.doc_id == "test_doc_1"
            assert document.content == "# Test Document\n\nThis is a test."
            assert isinstance(document.metadata, DocumentMetadata)
            assert document.metadata.file_type == "pdf"
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()

    @patch('docling_parser.core.docling_parser.DocumentConverter')
    def test_parse_document_auto_id(self, mock_converter: Mock) -> None:
        """Test parse_document with auto-generated ID."""
        # Mock the converter result
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Test content"
        mock_result.document.tables = []

        mock_converter_instance = MagicMock()
        mock_converter_instance.convert.return_value = mock_result
        mock_converter.return_value = mock_converter_instance

        # Create test file
        test_file = Path(__file__).parent / "test_data" / "test.docx"
        test_file.parent.mkdir(exist_ok=True)
        test_file.touch()

        try:
            parser = DoclingParser()
            document = parser.parse_document(str(test_file))

            assert isinstance(document, Document)
            assert document.doc_id.startswith("doc_")
            assert len(document.doc_id) == 16  # "doc_" + 12 hex chars
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()

    @patch('docling_parser.core.docling_parser.DocumentConverter')
    def test_parse_document_file_not_found(self, mock_converter: Mock) -> None:
        """Test parse_document with non-existent file."""
        parser = DoclingParser()

        with pytest.raises(FileNotFoundError):
            parser.parse_document("/path/to/nonexistent/file.pdf")

    @patch('docling_parser.core.docling_parser.DocumentConverter')
    def test_parse_documents_batch(self, mock_converter: Mock) -> None:
        """Test parse_documents batch processing."""
        # Mock the converter result
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Test content"
        mock_result.document.tables = []

        mock_converter_instance = MagicMock()
        mock_converter_instance.convert.return_value = mock_result
        mock_converter.return_value = mock_converter_instance

        # Create test files
        test_files = []
        test_dir = Path(__file__).parent / "test_data"
        test_dir.mkdir(exist_ok=True)

        for i in range(3):
            test_file = test_dir / f"test_{i}.pdf"
            test_file.touch()
            test_files.append(str(test_file))

        try:
            parser = DoclingParser()
            documents = parser.parse_documents(test_files)

            assert len(documents) == 3
            assert all(isinstance(doc, Document) for doc in documents)
        finally:
            # Cleanup
            for test_file in test_files:
                Path(test_file).unlink()

    @patch('docling_parser.core.docling_parser.DocumentConverter')
    def test_get_supported_formats(self, mock_converter: Mock) -> None:
        """Test get_supported_formats method."""
        parser = DoclingParser()
        formats = parser.get_supported_formats()

        assert isinstance(formats, list)
        assert "pdf" in formats
        assert "docx" in formats
        assert "html" in formats

    @patch('docling_parser.core.docling_parser.DocumentConverter')
    def test_extract_tables(self, mock_converter: Mock) -> None:
        """Test table extraction."""
        # Mock table data
        mock_table = MagicMock()
        mock_df = MagicMock()
        mock_df.to_dict.return_value = [{"col1": "val1", "col2": "val2"}]
        mock_table.export_to_dataframe.return_value = mock_df
        mock_table.caption = "Test Table"
        mock_table.num_rows = 1
        mock_table.num_cols = 2

        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Test content"
        mock_result.document.tables = [mock_table]

        mock_converter_instance = MagicMock()
        mock_converter_instance.convert.return_value = mock_result
        mock_converter.return_value = mock_converter_instance

        # Create test file
        test_file = Path(__file__).parent / "test_data" / "test_tables.pdf"
        test_file.parent.mkdir(exist_ok=True)
        test_file.touch()

        try:
            parser = DoclingParser(extract_tables=True)
            document = parser.parse_document(str(test_file))

            assert len(document.tables) > 0
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()

    @patch('docling_parser.core.docling_parser.DocumentConverter')
    def test_process_with_string_input(self, mock_converter: Mock) -> None:
        """Test process method with string input (Component interface)."""
        # Mock the converter result
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Test content"
        mock_result.document.tables = []

        mock_converter_instance = MagicMock()
        mock_converter_instance.convert.return_value = mock_result
        mock_converter.return_value = mock_converter_instance

        # Create test file
        test_file = Path(__file__).parent / "test_data" / "test_process.pdf"
        test_file.parent.mkdir(exist_ok=True)
        test_file.touch()

        try:
            parser = DoclingParser()
            # Test with string input (file path)
            document = parser.process(str(test_file))

            assert isinstance(document, Document)
            assert document.content == "Test content"
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()

    @patch('docling_parser.core.docling_parser.DocumentConverter')
    def test_process_with_dict_input(self, mock_converter: Mock) -> None:
        """Test process method with dict input (Component interface)."""
        # Mock the converter result
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Test content"
        mock_result.document.tables = []

        mock_converter_instance = MagicMock()
        mock_converter_instance.convert.return_value = mock_result
        mock_converter.return_value = mock_converter_instance

        # Create test file
        test_file = Path(__file__).parent / "test_data" / "test_process_dict.pdf"
        test_file.parent.mkdir(exist_ok=True)
        test_file.touch()

        try:
            parser = DoclingParser()
            # Test with dict input
            document = parser.process({
                'file_path': str(test_file),
                'doc_id': 'custom_id_123'
            })

            assert isinstance(document, Document)
            assert document.doc_id == 'custom_id_123'
        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()

    @patch('docling_parser.core.docling_parser.DocumentConverter')
    def test_process_with_invalid_input(self, mock_converter: Mock) -> None:
        """Test process method with invalid input type."""
        parser = DoclingParser()

        with pytest.raises(ValueError, match="Invalid input type"):
            parser.process(12345)  # Invalid type

        with pytest.raises(ValueError, match="must contain 'file_path' key"):
            parser.process({'wrong_key': 'value'})
