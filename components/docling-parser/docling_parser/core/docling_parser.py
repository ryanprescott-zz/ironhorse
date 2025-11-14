"""DoclingParser implementation.

This module provides the core functionality for parsing documents using Docling
and converting them to the standard Document format.
"""

import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Union

# Add shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.component import Component
from shared.schemas import Document, DocumentMetadata

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
except ImportError:
    # Graceful fallback for when docling is not installed
    DocumentConverter = None
    InputFormat = None


class DoclingParser(Component):
    """Document parser using Docling.

    This class provides document parsing capabilities using the Docling library,
    converting parsed documents to the standard Document format.

    Implements the Component interface with process() method for document parsing.

    Attributes:
        converter: Docling DocumentConverter instance.
        extract_tables: Whether to extract tables from documents.
    """

    def __init__(self, extract_tables: bool = True) -> None:
        """Initialize the DoclingParser.

        Args:
            extract_tables: Whether to extract tables from documents.

        Raises:
            ImportError: If docling is not installed.
        """
        if DocumentConverter is None:
            raise ImportError(
                "Docling is not installed. Install it with: pip install docling"
            )

        self.converter = DocumentConverter()
        self.extract_tables = extract_tables

    def process(self, data: Any) -> Any:
        """Process input data (implements Component interface).

        This is the main entry point for the component. It accepts either
        a file path string or a dict with 'file_path' and optional 'doc_id'.

        Args:
            data: Either a string (file path) or dict with 'file_path' and
                  optional 'doc_id' keys.

        Returns:
            Parsed Document object.

        Raises:
            ValueError: If data format is invalid.
            FileNotFoundError: If file does not exist.
        """
        if isinstance(data, str):
            # Simple string input - treat as file path
            return self.parse_document(data)
        elif isinstance(data, dict):
            # Dict input - extract file_path and optional doc_id
            file_path = data.get('file_path')
            if not file_path:
                raise ValueError("Dict input must contain 'file_path' key")
            doc_id = data.get('doc_id')
            return self.parse_document(file_path, doc_id=doc_id)
        else:
            raise ValueError(
                f"Invalid input type: {type(data)}. "
                "Expected str (file path) or dict with 'file_path' key."
            )

    def parse_document(
        self,
        file_path: str,
        doc_id: Union[str, None] = None,
    ) -> Document:
        """Parse a single document.

        Args:
            file_path: Path to the document file.
            doc_id: Optional document ID. If not provided, a UUID will be generated.

        Returns:
            Parsed document in standard Document format.

        Raises:
            FileNotFoundError: If file_path does not exist.
            ValueError: If file format is not supported.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"

        # Convert document using Docling
        result = self.converter.convert(str(path))

        # Extract content
        content = result.document.export_to_markdown()

        # Extract tables if enabled
        tables = []
        if self.extract_tables:
            tables = self._extract_tables(result)

        # Build metadata
        metadata = DocumentMetadata(
            source=str(path),
            file_type=path.suffix.lstrip('.').lower(),
            custom={}
        )

        return Document(
            doc_id=doc_id,
            content=content,
            metadata=metadata,
            tables=tables
        )

    def parse_documents(
        self,
        file_paths: List[str],
    ) -> List[Document]:
        """Parse multiple documents in batch.

        Args:
            file_paths: List of paths to document files.

        Returns:
            List of parsed documents in standard Document format.

        Raises:
            FileNotFoundError: If any file_path does not exist.
            ValueError: If any file format is not supported.
        """
        documents = []
        for file_path in file_paths:
            doc = self.parse_document(file_path)
            documents.append(doc)

        return documents

    def _extract_tables(self, result: Any) -> List[Dict[str, Any]]:
        """Extract tables from Docling result.

        Args:
            result: Docling conversion result.

        Returns:
            List of table dictionaries.
        """
        tables = []

        try:
            # Docling stores tables in the document structure
            for table in result.document.tables:
                table_data = {
                    "data": table.export_to_dataframe().to_dict('records') if hasattr(table, 'export_to_dataframe') else {},
                    "caption": getattr(table, 'caption', ''),
                    "num_rows": getattr(table, 'num_rows', 0),
                    "num_cols": getattr(table, 'num_cols', 0),
                }
                tables.append(table_data)
        except Exception:
            # If table extraction fails, return empty list
            pass

        return tables

    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats.

        Returns:
            List of supported file extensions.
        """
        if InputFormat is None:
            return []

        # Return the formats Docling supports
        return ["pdf", "docx", "html", "xlsx", "pptx", "md", "asciidoc"]
