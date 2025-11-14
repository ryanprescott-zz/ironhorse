"""LangChainSplitter implementation.

This module provides the core functionality for splitting text using LangChain
text splitters and converting to the standard Chunk format.
"""

import sys
import uuid
from pathlib import Path
from typing import List, Optional

# Add shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.schemas import Chunk, ChunkMetadata, Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None


class LangChainSplitter:
    """Text splitter using LangChain.

    This class provides text splitting capabilities using LangChain's
    RecursiveCharacterTextSplitter, converting split text to the standard
    Chunk format.

    Attributes:
        chunk_size: Size of each chunk in characters.
        chunk_overlap: Overlap between consecutive chunks.
        separators: List of separator strings for splitting.
        keep_separator: Whether to keep separators in chunks.
        strip_whitespace: Whether to strip whitespace from chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        strip_whitespace: bool = True,
    ) -> None:
        """Initialize the LangChainSplitter.

        Args:
            chunk_size: Size of each chunk in characters.
            chunk_overlap: Overlap between consecutive chunks.
            separators: List of separator strings. If None, uses default.
            keep_separator: Whether to keep separators in chunks.
            strip_whitespace: Whether to strip whitespace from chunks.

        Raises:
            ImportError: If langchain-text-splitters is not installed.
        """
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain-text-splitters is not installed. "
                "Install it with: pip install langchain-text-splitters"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace

        # Initialize the LangChain splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            keep_separator=keep_separator,
            strip_whitespace=strip_whitespace,
        )

    def split_text(
        self,
        text: str,
        source_id: Optional[str] = None,
    ) -> List[Chunk]:
        """Split text into chunks.

        Args:
            text: Text to split.
            source_id: Optional source identifier (e.g., document ID).

        Returns:
            List of chunks in standard Chunk format.
        """
        # Split text using LangChain
        text_chunks = self.splitter.split_text(text)

        # Convert to standard Chunk format
        chunks = []
        current_position = 0

        for idx, chunk_text in enumerate(text_chunks):
            # Find the position of this chunk in the original text
            start_char = text.find(chunk_text, current_position)
            if start_char == -1:
                # Fallback if exact match not found
                start_char = current_position
            end_char = start_char + len(chunk_text)
            current_position = start_char + 1  # Move forward for next search

            # Generate chunk ID
            chunk_id = f"chunk_{uuid.uuid4().hex[:12]}_{idx}"

            # Create metadata
            metadata = ChunkMetadata(
                source_doc_id=source_id,
                chunk_index=idx,
                start_char=start_char,
                end_char=end_char,
            )

            # Create chunk
            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                metadata=metadata,
            )
            chunks.append(chunk)

        return chunks

    def split_document(
        self,
        document: Document,
    ) -> List[Chunk]:
        """Split a document into chunks.

        Args:
            document: Document to split (in standard Document format).

        Returns:
            List of chunks in standard Chunk format.
        """
        return self.split_text(document.content, source_id=document.doc_id)

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Chunk]:
        """Split multiple documents into chunks.

        Args:
            documents: List of documents to split.

        Returns:
            List of all chunks from all documents.
        """
        all_chunks = []
        for document in documents:
            chunks = self.split_document(document)
            all_chunks.extend(chunks)

        return all_chunks

    def get_chunk_count(self, text: str) -> int:
        """Get the number of chunks that would be created from text.

        Args:
            text: Text to analyze.

        Returns:
            Number of chunks that would be created.
        """
        chunks = self.splitter.split_text(text)
        return len(chunks)
