"""Document data model for AI toolkit components.

This module defines the standard document structure used across all components
in the AI toolkit.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata associated with a document.

    Attributes:
        source: Source location or identifier of the document.
        file_type: Type of the file (e.g., pdf, docx, html).
        page_count: Number of pages in the document.
        author: Author of the document.
        created_at: Creation timestamp.
        custom: Additional custom metadata fields.
    """

    source: Optional[str] = Field(None, description="Source location or identifier")
    file_type: Optional[str] = Field(None, description="File type (pdf, docx, html, etc.)")
    page_count: Optional[int] = Field(None, description="Number of pages")
    author: Optional[str] = Field(None, description="Document author")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")


class Document(BaseModel):
    """Standard document structure for AI toolkit components.

    This is the primary data structure for documents flowing through the
    processing pipeline. All components should accept and return documents
    in this format.

    Attributes:
        doc_id: Unique identifier for the document.
        content: Text content of the document.
        metadata: Document metadata.
        tables: Extracted tables from the document (list of dicts).
    """

    doc_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Text content of the document")
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata,
        description="Document metadata"
    )
    tables: list[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extracted tables from the document"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "doc_id": "doc_12345",
                "content": "This is the document content...",
                "metadata": {
                    "source": "/path/to/document.pdf",
                    "file_type": "pdf",
                    "page_count": 10
                },
                "tables": []
            }
        }
