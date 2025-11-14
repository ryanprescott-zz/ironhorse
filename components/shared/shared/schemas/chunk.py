"""Chunk data model for AI toolkit components.

This module defines the standard chunk structure used for text splitting
and RAG pipelines in the AI toolkit.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata associated with a text chunk.

    Attributes:
        source_doc_id: ID of the source document.
        chunk_index: Index of this chunk in the sequence.
        start_char: Starting character position in the source document.
        end_char: Ending character position in the source document.
        custom: Additional custom metadata fields.
    """

    source_doc_id: Optional[str] = Field(None, description="Source document ID")
    chunk_index: Optional[int] = Field(None, description="Chunk index in sequence")
    start_char: Optional[int] = Field(None, description="Start character position")
    end_char: Optional[int] = Field(None, description="End character position")
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")


class Chunk(BaseModel):
    """Standard chunk structure for AI toolkit components.

    This is the primary data structure for text chunks in splitting and
    RAG operations. All text processing components should use this format.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        text: Text content of the chunk.
        metadata: Chunk metadata.
    """

    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Text content of the chunk")
    metadata: ChunkMetadata = Field(
        default_factory=ChunkMetadata,
        description="Chunk metadata"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_12345_0",
                "text": "This is a chunk of text...",
                "metadata": {
                    "source_doc_id": "doc_12345",
                    "chunk_index": 0,
                    "start_char": 0,
                    "end_char": 100
                }
            }
        }
