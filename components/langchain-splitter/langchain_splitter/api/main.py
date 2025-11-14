"""FastAPI application entry point for langchain-splitter.

This module sets up the FastAPI application and includes all routes.
"""

from fastapi import FastAPI
from langchain_splitter.api import routes
from langchain_splitter.config import settings

app = FastAPI(
    title="LangChainSplitter API",
    description="Text splitter using LangChain",
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
    return {"status": "healthy", "service": "langchain-splitter"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
