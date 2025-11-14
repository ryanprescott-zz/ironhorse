"""N8N binding for LangChain Splitter component.

This module provides helper functions for calling the LangChain Splitter API
from n8n workflows.
"""

from typing import Any, Dict, List, Optional
import requests


def split_text(
    text: str,
    source_id: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
    base_url: str = "http://langchain-splitter:26001"
) -> Dict[str, Any]:
    """Split text into chunks using LangChain Splitter.

    This function is designed to be called from n8n HTTP Request nodes.

    Args:
        text: Text to split.
        source_id: Optional source identifier.
        chunk_size: Chunk size in characters.
        chunk_overlap: Overlap between chunks.
        separators: List of separator strings.
        base_url: Base URL of the LangChain Splitter API.

    Returns:
        API response with chunks.

    Example n8n usage:
        Use an HTTP Request node with:
        - Method: POST
        - URL: http://langchain-splitter:26001/api/v1/split/text
        - Body (JSON):
            {
              "text": {{$json["text"]}},
              "chunk_size": 1000,
              "chunk_overlap": 200
            }
    """
    url = f"{base_url}/api/v1/split/text"
    payload = {
        "text": text,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "keep_separator": True,
        "strip_whitespace": True,
    }
    if source_id:
        payload["source_id"] = source_id
    if separators:
        payload["separators"] = separators

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def split_document(
    document: Dict[str, Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
    base_url: str = "http://langchain-splitter:26001"
) -> Dict[str, Any]:
    """Split a document into chunks.

    Args:
        document: Document to split (standard Document format).
        chunk_size: Chunk size in characters.
        chunk_overlap: Overlap between chunks.
        separators: List of separator strings.
        base_url: Base URL of the LangChain Splitter API.

    Returns:
        API response with chunks.

    Example n8n usage:
        Use an HTTP Request node with:
        - Method: POST
        - URL: http://langchain-splitter:26001/api/v1/split/document
        - Body (JSON):
            {
              "document": {{$json["document"]}},
              "chunk_size": 1000,
              "chunk_overlap": 200
            }
    """
    url = f"{base_url}/api/v1/split/document"
    payload = {
        "document": document,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "keep_separator": True,
        "strip_whitespace": True,
    }
    if separators:
        payload["separators"] = separators

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def split_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
    base_url: str = "http://langchain-splitter:26001"
) -> Dict[str, Any]:
    """Split multiple documents into chunks.

    Args:
        documents: List of documents to split.
        chunk_size: Chunk size in characters.
        chunk_overlap: Overlap between chunks.
        separators: List of separator strings.
        base_url: Base URL of the LangChain Splitter API.

    Returns:
        API response with all chunks.

    Example n8n usage:
        Use an HTTP Request node with:
        - Method: POST
        - URL: http://langchain-splitter:26001/api/v1/split/documents
        - Body (JSON):
            {
              "documents": {{$json["documents"]}},
              "chunk_size": 1000,
              "chunk_overlap": 200
            }
    """
    url = f"{base_url}/api/v1/split/documents"
    payload = {
        "documents": documents,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "keep_separator": True,
        "strip_whitespace": True,
    }
    if separators:
        payload["separators"] = separators

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()


def analyze_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    base_url: str = "http://langchain-splitter:26001"
) -> Dict[str, Any]:
    """Analyze how text would be split without actually splitting.

    Args:
        text: Text to analyze.
        chunk_size: Chunk size to use.
        chunk_overlap: Chunk overlap to use.
        base_url: Base URL of the LangChain Splitter API.

    Returns:
        Analysis results including chunk count.

    Example n8n usage:
        Use an HTTP Request node with:
        - Method: POST
        - URL: http://langchain-splitter:26001/api/v1/analyze
        - Query Parameters:
            - text: {{$json["text"]}}
            - chunk_size: 1000
            - chunk_overlap: 200
    """
    url = f"{base_url}/api/v1/analyze"
    params = {
        "text": text,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }

    response = requests.post(url, params=params)
    response.raise_for_status()
    return response.json()
