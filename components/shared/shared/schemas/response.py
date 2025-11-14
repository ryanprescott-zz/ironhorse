"""API response wrapper for AI toolkit components.

This module defines the standard API response structure used across all
FastAPI endpoints in the AI toolkit.
"""

from enum import Enum
from typing import Any, Dict, Optional, TypeVar, Generic
from pydantic import BaseModel, Field


class ResponseStatus(str, Enum):
    """Status codes for API responses.

    Attributes:
        SUCCESS: Request completed successfully.
        ERROR: Request failed with an error.
        PARTIAL: Request partially completed.
    """

    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper for all toolkit endpoints.

    This wrapper ensures consistent response format across all components,
    making it easier to integrate and debug.

    Attributes:
        status: Response status (success, error, partial).
        data: Response data payload.
        error: Error message if status is error.
        metadata: Additional response metadata.
    """

    status: ResponseStatus = Field(..., description="Response status")
    data: Optional[T] = Field(None, description="Response data payload")
    error: Optional[str] = Field(None, description="Error message if applicable")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "status": "success",
                "data": {"result": "some data"},
                "error": None,
                "metadata": {"processing_time_ms": 123}
            }
        }

    @classmethod
    def success(
        cls,
        data: T,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "APIResponse[T]":
        """Create a success response.

        Args:
            data: Response data payload.
            metadata: Optional response metadata.

        Returns:
            APIResponse with success status.
        """
        return cls(
            status=ResponseStatus.SUCCESS,
            data=data,
            error=None,
            metadata=metadata or {}
        )

    @classmethod
    def error(
        cls,
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "APIResponse[T]":
        """Create an error response.

        Args:
            error: Error message.
            metadata: Optional response metadata.

        Returns:
            APIResponse with error status.
        """
        return cls(
            status=ResponseStatus.ERROR,
            data=None,
            error=error,
            metadata=metadata or {}
        )

    @classmethod
    def partial(
        cls,
        data: T,
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "APIResponse[T]":
        """Create a partial response (some data with errors).

        Args:
            data: Partial response data.
            error: Error message describing what failed.
            metadata: Optional response metadata.

        Returns:
            APIResponse with partial status.
        """
        return cls(
            status=ResponseStatus.PARTIAL,
            data=data,
            error=error,
            metadata=metadata or {}
        )
