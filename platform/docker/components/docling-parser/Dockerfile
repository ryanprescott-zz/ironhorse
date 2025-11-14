# Dockerfile for Docling Parser component
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN pip install uv

# Copy shared schemas
COPY components/shared/ /app/shared/

# Copy component files
COPY components/docling-parser/ /app/docling-parser/

# Install dependencies
WORKDIR /app/docling-parser
RUN uv pip install --system -e .

# Expose the API port
EXPOSE 26000

# Set Python path
ENV PYTHONPATH=/app

# Run the FastAPI application
CMD ["python", "-m", "docling_parser.api.main"]
