# Dockerfile for LangChain Splitter component
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
COPY components/langchain-splitter/ /app/langchain-splitter/

# Install dependencies
WORKDIR /app/langchain-splitter
RUN uv pip install --system -e .

# Expose the API port
EXPOSE 26001

# Set Python path
ENV PYTHONPATH=/app

# Run the FastAPI application
CMD ["python", "-m", "langchain_splitter.api.main"]
