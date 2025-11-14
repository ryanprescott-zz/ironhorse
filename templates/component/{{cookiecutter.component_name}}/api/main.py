"""FastAPI application entry point for {{cookiecutter.component_name}}.

This module sets up the FastAPI application and includes all routes.
"""

from fastapi import FastAPI
from {{cookiecutter.component_name.replace('-', '_')}}.api import routes
from {{cookiecutter.component_name.replace('-', '_')}}.config import settings

app = FastAPI(
    title="{{cookiecutter.component_class_name}} API",
    description="{{cookiecutter.component_description}}",
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
    return {"status": "healthy", "service": "{{cookiecutter.component_name}}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
