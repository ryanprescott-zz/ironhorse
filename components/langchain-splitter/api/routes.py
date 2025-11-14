"""API routes for langchain-splitter component.

This module defines all FastAPI endpoints for the text splitter component.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Add shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.schemas import Chunk, Document, APIResponse, ResponseStatus
from langchain_splitter.core import LangChainSplitter
from langchain_splitter.config import settings

router = APIRouter()


class SplitTextRequest(BaseModel):
    """Request model for splitting text.

    Attributes:
        text: Text to split.
        source_id: Optional source identifier.
        chunk_size: Chunk size in characters.
        chunk_overlap: Overlap between chunks.
        separators: List of separator strings.
        keep_separator: Whether to keep separators.
        strip_whitespace: Whether to strip whitespace.
    """

    text: str = Field(..., description="Text to split")
    source_id: Optional[str] = Field(None, description="Source identifier")
    chunk_size: int = Field(
        settings.default_chunk_size,
        description="Chunk size in characters"
    )
    chunk_overlap: int = Field(
        settings.default_chunk_overlap,
        description="Overlap between chunks"
    )
    separators: Optional[List[str]] = Field(
        None,
        description="List of separator strings"
    )
    keep_separator: bool = Field(
        settings.default_keep_separator,
        description="Whether to keep separators"
    )
    strip_whitespace: bool = Field(
        settings.default_strip_whitespace,
        description="Whether to strip whitespace"
    )


class SplitDocumentRequest(BaseModel):
    """Request model for splitting a document.

    Attributes:
        document: Document to split.
        chunk_size: Chunk size in characters.
        chunk_overlap: Overlap between chunks.
        separators: List of separator strings.
        keep_separator: Whether to keep separators.
        strip_whitespace: Whether to strip whitespace.
    """

    document: Document = Field(..., description="Document to split")
    chunk_size: int = Field(
        settings.default_chunk_size,
        description="Chunk size in characters"
    )
    chunk_overlap: int = Field(
        settings.default_chunk_overlap,
        description="Overlap between chunks"
    )
    separators: Optional[List[str]] = Field(
        None,
        description="List of separator strings"
    )
    keep_separator: bool = Field(
        settings.default_keep_separator,
        description="Whether to keep separators"
    )
    strip_whitespace: bool = Field(
        settings.default_strip_whitespace,
        description="Whether to strip whitespace"
    )


class SplitDocumentsRequest(BaseModel):
    """Request model for splitting multiple documents.

    Attributes:
        documents: List of documents to split.
        chunk_size: Chunk size in characters.
        chunk_overlap: Overlap between chunks.
        separators: List of separator strings.
        keep_separator: Whether to keep separators.
        strip_whitespace: Whether to strip whitespace.
    """

    documents: List[Document] = Field(..., description="Documents to split")
    chunk_size: int = Field(
        settings.default_chunk_size,
        description="Chunk size in characters"
    )
    chunk_overlap: int = Field(
        settings.default_chunk_overlap,
        description="Overlap between chunks"
    )
    separators: Optional[List[str]] = Field(
        None,
        description="List of separator strings"
    )
    keep_separator: bool = Field(
        settings.default_keep_separator,
        description="Whether to keep separators"
    )
    strip_whitespace: bool = Field(
        settings.default_strip_whitespace,
        description="Whether to strip whitespace"
    )


class ChunksResponse(BaseModel):
    """Response model for split operations.

    Attributes:
        chunks: List of text chunks.
    """

    chunks: List[Chunk]


@router.post("/split/text", response_model=APIResponse[ChunksResponse])
async def split_text(request: SplitTextRequest) -> Dict[str, Any]:
    """Split text into chunks.

    Args:
        request: Split text request.

    Returns:
        API response with chunks.

    Raises:
        HTTPException: If splitting fails.
    """
    start_time = time.time()

    try:
        # Initialize splitter
        splitter = LangChainSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            separators=request.separators,
            keep_separator=request.keep_separator,
            strip_whitespace=request.strip_whitespace,
        )

        # Split text
        chunks = splitter.split_text(request.text, source_id=request.source_id)

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        return APIResponse.success(
            data=ChunksResponse(chunks=chunks),
            metadata={
                "processing_time_ms": processing_time_ms,
                "chunk_count": len(chunks),
                "original_length": len(request.text),
            }
        ).model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Splitting failed: {str(e)}")


@router.post("/split/document", response_model=APIResponse[ChunksResponse])
async def split_document(request: SplitDocumentRequest) -> Dict[str, Any]:
    """Split a document into chunks.

    Args:
        request: Split document request.

    Returns:
        API response with chunks.

    Raises:
        HTTPException: If splitting fails.
    """
    start_time = time.time()

    try:
        # Initialize splitter
        splitter = LangChainSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            separators=request.separators,
            keep_separator=request.keep_separator,
            strip_whitespace=request.strip_whitespace,
        )

        # Split document
        chunks = splitter.split_document(request.document)

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        return APIResponse.success(
            data=ChunksResponse(chunks=chunks),
            metadata={
                "processing_time_ms": processing_time_ms,
                "chunk_count": len(chunks),
                "source_doc_id": request.document.doc_id,
            }
        ).model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Splitting failed: {str(e)}")


@router.post("/split/documents", response_model=APIResponse[ChunksResponse])
async def split_documents(request: SplitDocumentsRequest) -> Dict[str, Any]:
    """Split multiple documents into chunks.

    Args:
        request: Split documents request.

    Returns:
        API response with all chunks.

    Raises:
        HTTPException: If splitting fails.
    """
    start_time = time.time()

    try:
        # Initialize splitter
        splitter = LangChainSplitter(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            separators=request.separators,
            keep_separator=request.keep_separator,
            strip_whitespace=request.strip_whitespace,
        )

        # Split documents
        chunks = splitter.split_documents(request.documents)

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        return APIResponse.success(
            data=ChunksResponse(chunks=chunks),
            metadata={
                "processing_time_ms": processing_time_ms,
                "chunk_count": len(chunks),
                "document_count": len(request.documents),
            }
        ).model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Splitting failed: {str(e)}")


@router.post("/analyze")
async def analyze_text(
    text: str,
    chunk_size: int = settings.default_chunk_size,
    chunk_overlap: int = settings.default_chunk_overlap,
) -> Dict[str, Any]:
    """Analyze how text would be split without actually splitting.

    Args:
        text: Text to analyze.
        chunk_size: Chunk size to use.
        chunk_overlap: Chunk overlap to use.

    Returns:
        Analysis results including chunk count.

    Raises:
        HTTPException: If analysis fails.
    """
    try:
        splitter = LangChainSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunk_count = splitter.get_chunk_count(text)

        return APIResponse.success(
            data={
                "chunk_count": chunk_count,
                "text_length": len(text),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        ).model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
