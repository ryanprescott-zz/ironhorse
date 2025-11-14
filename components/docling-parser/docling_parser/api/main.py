"""FastAPI application entry point for docling-parser.

This module sets up the FastAPI application and includes all routes.
"""

from fastapi import FastAPI
from docling_parser.api import routes
from docling_parser.config import settings

app = FastAPI(
    title="DoclingParser API",
    description="Document parser using Docling",
    version="0.1.0",
)

# Include routers
app.include_router(routes.router, prefix="/api/v1")


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint.

    Returns:
        Health status.
    """
    return {"status": "healthy", "service": "docling-parser"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
