"""API routes for docling-parser component.

This module defines all FastAPI endpoints for the document parser component.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

# Add shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.schemas import Document, APIResponse, ResponseStatus
from docling_parser.core import DoclingParser
from docling_parser.config import settings

router = APIRouter()


class ParseRequest(BaseModel):
    """Request model for parsing endpoint.

    Attributes:
        file_paths: List of file paths to parse.
        extract_tables: Whether to extract tables.
    """

    file_paths: List[str] = Field(..., description="List of file paths to parse")
    extract_tables: bool = Field(
        True,
        description="Whether to extract tables from documents"
    )


class ParseResponse(BaseModel):
    """Response model for parsing endpoint.

    Attributes:
        documents: List of parsed documents.
    """

    documents: List[Document]


@router.post("/parse", response_model=APIResponse[ParseResponse])
async def parse_documents(request: ParseRequest) -> Dict[str, Any]:
    """Parse documents from file paths.

    Args:
        request: Parse request with file paths.

    Returns:
        API response with parsed documents.

    Raises:
        HTTPException: If parsing fails.
    """
    start_time = time.time()

    try:
        # Initialize parser
        parser = DoclingParser(extract_tables=request.extract_tables)

        # Parse documents
        documents = parser.parse_documents(request.file_paths)

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        return APIResponse.success(
            data=ParseResponse(documents=documents),
            metadata={
                "processing_time_ms": processing_time_ms,
                "document_count": len(documents)
            }
        ).model_dump()

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


@router.post("/parse/single", response_model=APIResponse[Document])
async def parse_single_document(
    file_path: str,
    extract_tables: bool = True,
    doc_id: Optional[str] = None
) -> Dict[str, Any]:
    """Parse a single document.

    Args:
        file_path: Path to the document file.
        extract_tables: Whether to extract tables.
        doc_id: Optional document ID.

    Returns:
        API response with parsed document.

    Raises:
        HTTPException: If parsing fails.
    """
    start_time = time.time()

    try:
        # Initialize parser
        parser = DoclingParser(extract_tables=extract_tables)

        # Parse document
        document = parser.parse_document(file_path, doc_id=doc_id)

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        return APIResponse.success(
            data=document,
            metadata={"processing_time_ms": processing_time_ms}
        ).model_dump()

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


@router.get("/formats", response_model=APIResponse[List[str]])
async def get_supported_formats() -> Dict[str, Any]:
    """Get list of supported file formats.

    Returns:
        API response with list of supported formats.
    """
    try:
        parser = DoclingParser()
        formats = parser.get_supported_formats()

        return APIResponse.success(
            data=formats,
            metadata={"format_count": len(formats)}
        ).model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get formats: {str(e)}")
