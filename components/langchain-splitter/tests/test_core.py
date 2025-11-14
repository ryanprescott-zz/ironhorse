"""Tests for langchain-splitter core functionality."""

import sys
from pathlib import Path

import pytest

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.schemas import Chunk, Document, DocumentMetadata
from langchain_splitter.core import LangChainSplitter


class TestLangChainSplitter:
    """Test suite for LangChainSplitter."""

    def test_initialization(self) -> None:
        """Test LangChainSplitter initialization."""
        splitter = LangChainSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
        assert splitter is not None
        assert splitter.chunk_size == 500
        assert splitter.chunk_overlap == 100

    def test_initialization_with_custom_separators(self) -> None:
        """Test initialization with custom separators."""
        custom_separators = ["\n\n", ".", " "]
        splitter = LangChainSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=custom_separators,
        )
        assert splitter.separators == custom_separators

    def test_split_text(self) -> None:
        """Test split_text method."""
        splitter = LangChainSplitter(chunk_size=50, chunk_overlap=10)

        text = (
            "This is a test sentence. "
            "This is another test sentence. "
            "And here is yet another one to make it longer."
        )

        chunks = splitter.split_text(text, source_id="test_source")

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.metadata.source_doc_id == "test_source" for chunk in chunks)

        # Check chunk indices are sequential
        for idx, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == idx

    def test_split_text_small(self) -> None:
        """Test splitting small text that fits in one chunk."""
        splitter = LangChainSplitter(chunk_size=1000, chunk_overlap=200)

        text = "This is a small text."
        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_split_text_without_source_id(self) -> None:
        """Test split_text without providing source_id."""
        splitter = LangChainSplitter(chunk_size=50, chunk_overlap=10)

        text = "This is a test. " * 10
        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        assert all(chunk.metadata.source_doc_id is None for chunk in chunks)

    def test_split_document(self) -> None:
        """Test split_document method."""
        splitter = LangChainSplitter(chunk_size=50, chunk_overlap=10)

        document = Document(
            doc_id="doc_123",
            content="This is a test document. " * 10,
            metadata=DocumentMetadata(source="test.pdf", file_type="pdf"),
        )

        chunks = splitter.split_document(document)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.metadata.source_doc_id == "doc_123" for chunk in chunks)

    def test_split_documents(self) -> None:
        """Test split_documents method with multiple documents."""
        splitter = LangChainSplitter(chunk_size=50, chunk_overlap=10)

        documents = [
            Document(
                doc_id=f"doc_{i}",
                content=f"Document {i} content. " * 10,
                metadata=DocumentMetadata(source=f"test_{i}.pdf", file_type="pdf"),
            )
            for i in range(3)
        ]

        chunks = splitter.split_documents(documents)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

        # Check that chunks from different documents have different source_doc_ids
        source_ids = set(chunk.metadata.source_doc_id for chunk in chunks)
        assert len(source_ids) == 3
        assert all(f"doc_{i}" in source_ids for i in range(3))

    def test_get_chunk_count(self) -> None:
        """Test get_chunk_count method."""
        splitter = LangChainSplitter(chunk_size=50, chunk_overlap=10)

        text = "This is a test. " * 20
        count = splitter.get_chunk_count(text)

        assert count > 0
        assert isinstance(count, int)

        # Verify count matches actual split
        chunks = splitter.split_text(text)
        assert count == len(chunks)

    def test_chunk_metadata(self) -> None:
        """Test that chunk metadata is properly populated."""
        splitter = LangChainSplitter(chunk_size=30, chunk_overlap=5)

        text = "First chunk. Second chunk. Third chunk."
        chunks = splitter.split_text(text, source_id="meta_test")

        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_id.startswith("chunk_")
            assert chunk.metadata.source_doc_id == "meta_test"
            assert chunk.metadata.chunk_index == idx
            assert chunk.metadata.start_char is not None
            assert chunk.metadata.end_char is not None
            assert chunk.metadata.start_char < chunk.metadata.end_char

    def test_chunk_overlap(self) -> None:
        """Test that chunk overlap works correctly."""
        splitter = LangChainSplitter(chunk_size=50, chunk_overlap=20)

        text = "A" * 200

        chunks = splitter.split_text(text)

        # With overlap, we should have more chunks than without
        assert len(chunks) > 1

    def test_separator_options(self) -> None:
        """Test different separator configurations."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."

        # Test with default separators
        splitter1 = LangChainSplitter(chunk_size=50, chunk_overlap=0)
        chunks1 = splitter1.split_text(text)

        # Test with custom separators
        splitter2 = LangChainSplitter(
            chunk_size=50,
            chunk_overlap=0,
            separators=["\n\n"]
        )
        chunks2 = splitter2.split_text(text)

        # Both should produce chunks, possibly different numbers
        assert len(chunks1) > 0
        assert len(chunks2) > 0

    def test_strip_whitespace(self) -> None:
        """Test strip_whitespace option."""
        splitter = LangChainSplitter(
            chunk_size=50,
            chunk_overlap=0,
            strip_whitespace=True
        )

        text = "  Text with whitespace  \n\n  More text  "
        chunks = splitter.split_text(text)

        # Chunks should not have leading/trailing whitespace
        for chunk in chunks:
            assert chunk.text == chunk.text.strip()

    def test_process_with_string_input(self) -> None:
        """Test process method with string input (Component interface)."""
        splitter = LangChainSplitter(chunk_size=50, chunk_overlap=10)

        text = "This is a test sentence. " * 10
        chunks = splitter.process(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_process_with_document_input(self) -> None:
        """Test process method with Document input (Component interface)."""
        splitter = LangChainSplitter(chunk_size=50, chunk_overlap=10)

        document = Document(
            doc_id="doc_process_test",
            content="Test content. " * 10,
            metadata=DocumentMetadata(source="test.pdf", file_type="pdf"),
        )

        chunks = splitter.process(document)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.metadata.source_doc_id == "doc_process_test" for chunk in chunks)

    def test_process_with_document_list(self) -> None:
        """Test process method with list of Documents (Component interface)."""
        splitter = LangChainSplitter(chunk_size=50, chunk_overlap=10)

        documents = [
            Document(
                doc_id=f"doc_{i}",
                content=f"Content {i}. " * 10,
                metadata=DocumentMetadata(source=f"test_{i}.pdf", file_type="pdf"),
            )
            for i in range(2)
        ]

        chunks = splitter.process(documents)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_process_with_dict_text_input(self) -> None:
        """Test process method with dict containing text (Component interface)."""
        splitter = LangChainSplitter(chunk_size=50, chunk_overlap=10)

        data = {
            'text': "Test text. " * 10,
            'source_id': 'custom_source_123'
        }

        chunks = splitter.process(data)

        assert len(chunks) > 0
        assert all(chunk.metadata.source_doc_id == 'custom_source_123' for chunk in chunks)

    def test_process_with_dict_document_input(self) -> None:
        """Test process method with dict containing document (Component interface)."""
        splitter = LangChainSplitter(chunk_size=50, chunk_overlap=10)

        document = Document(
            doc_id="doc_dict_test",
            content="Dict test content. " * 10,
            metadata=DocumentMetadata(source="test.pdf", file_type="pdf"),
        )

        chunks = splitter.process({'document': document})

        assert len(chunks) > 0
        assert all(chunk.metadata.source_doc_id == "doc_dict_test" for chunk in chunks)

    def test_process_with_invalid_input(self) -> None:
        """Test process method with invalid input types."""
        splitter = LangChainSplitter(chunk_size=50, chunk_overlap=10)

        # Test with invalid type
        with pytest.raises(ValueError, match="Invalid input type"):
            splitter.process(12345)

        # Test with dict missing required keys
        with pytest.raises(ValueError, match="must contain 'text' or 'document' key"):
            splitter.process({'wrong_key': 'value'})

        # Test with list of non-Documents
        with pytest.raises(ValueError, match="List input must contain Document objects"):
            splitter.process(['string1', 'string2'])
