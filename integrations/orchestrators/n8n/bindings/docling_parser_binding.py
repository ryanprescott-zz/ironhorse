"""N8N binding for Docling Parser component.

This module provides helper functions for calling the Docling Parser API
from n8n workflows.
"""

from typing import Any, Dict, List, Optional
import requests


def parse_document(
    file_path: str,
    extract_tables: bool = True,
    doc_id: Optional[str] = None,
    base_url: str = "http://docling-parser:26000"
) -> Dict[str, Any]:
    """Parse a single document using Docling Parser.

    This function is designed to be called from n8n HTTP Request nodes.

    Args:
        file_path: Path to the document file.
        extract_tables: Whether to extract tables.
        doc_id: Optional document ID.
        base_url: Base URL of the Docling Parser API.

    Returns:
        API response with parsed document.

    Example n8n usage:
        Use an HTTP Request node with:
        - Method: POST
        - URL: http://docling-parser:26000/api/v1/parse/single
        - Query Parameters:
            - file_path: {{$json["file_path"]}}
            - extract_tables: true
            - doc_id: {{$json["doc_id"]}}
    """
    url = f"{base_url}/api/v1/parse/single"
    params = {
        "file_path": file_path,
        "extract_tables": extract_tables,
    }
    if doc_id:
        params["doc_id"] = doc_id

    response = requests.post(url, params=params)
    response.raise_for_status()
    return response.json()


def parse_documents(
    file_paths: List[str],
    extract_tables: bool = True,
    base_url: str = "http://docling-parser:26000"
) -> Dict[str, Any]:
    """Parse multiple documents using Docling Parser.

    Args:
        file_paths: List of file paths to parse.
        extract_tables: Whether to extract tables.
        base_url: Base URL of the Docling Parser API.

    Returns:
        API response with parsed documents.

    Example n8n usage:
        Use an HTTP Request node with:
        - Method: POST
        - URL: http://docling-parser:26000/api/v1/parse
        - Body (JSON):
            {
              "file_paths": {{$json["file_paths"]}},
              "extract_tables": true
            }
    """
    url = f"{base_url}/api/v1/parse"
    payload = {
        "file_paths": file_paths,
        "extract_tables": extract_tables,
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def get_supported_formats(
    base_url: str = "http://docling-parser:26000"
) -> Dict[str, Any]:
    """Get list of supported file formats.

    Args:
        base_url: Base URL of the Docling Parser API.

    Returns:
        API response with supported formats.

    Example n8n usage:
        Use an HTTP Request node with:
        - Method: GET
        - URL: http://docling-parser:26000/api/v1/formats
    """
    url = f"{base_url}/api/v1/formats"

    response = requests.get(url)
    response.raise_for_status()
    return response.json()
