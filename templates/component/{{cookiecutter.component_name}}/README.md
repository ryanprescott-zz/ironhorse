# {{cookiecutter.component_class_name}}

{{cookiecutter.component_description}}

## Overview

This component provides [describe key capabilities here].

## Features

- Feature 1
- Feature 2
- Feature 3

## Installation

### As a Python Package

```bash
pip install {{cookiecutter.component_name}}
```

### As a Docker Container

```bash
docker pull {{cookiecutter.component_name}}:latest
docker run -p {{cookiecutter.component_port}}:{{cookiecutter.component_port}} {{cookiecutter.component_name}}:latest
```

## Usage

### Python API

```python
from {{cookiecutter.component_name.replace('-', '_')}}.core import {{cookiecutter.component_class_name}}

# Initialize component
component = {{cookiecutter.component_class_name}}()

# Process data
result = component.process({"input": "data"})
print(result)
```

### REST API

```bash
# Health check
curl http://localhost:{{cookiecutter.component_port}}/health

# Process data
curl -X POST http://localhost:{{cookiecutter.component_port}}/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{"data": {"input": "data"}}'
```

## Configuration

Configuration is managed via environment variables:

- `{{cookiecutter.component_name.replace('-', '_').upper()}}_HOST`: API server host (default: 0.0.0.0)
- `{{cookiecutter.component_name.replace('-', '_').upper()}}_PORT`: API server port (default: {{cookiecutter.component_port}})
- `{{cookiecutter.component_name.replace('-', '_').upper()}}_LOG_LEVEL`: Logging level (default: INFO)

## Development

### Setup

```bash
# Install shared schemas package (required dependency)
pip install -e ../shared

# Install component with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov={{cookiecutter.component_name.replace('-', '_')}} --cov-report=html
```

### Note on Shared Package

This component depends on the `shared` package for common data models and schemas. The shared package must be installed before this component can be used:

- **Document schema**: Standard document structure for parsed content
- **Chunk schema**: Text chunks for splitting/RAG operations
- **APIResponse**: Consistent API response wrapper

The shared package is located at `../shared` relative to this component.

### Running the API Server

```bash
python -m {{cookiecutter.component_name.replace('-', '_')}}.api.main
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:{{cookiecutter.component_port}}/docs
- ReDoc: http://localhost:{{cookiecutter.component_port}}/redoc

## Architecture

[Describe the component architecture, key classes, and design decisions]

## Building

### Python Wheel

```bash
python -m build
```

### Docker Image

```bash
docker build -t {{cookiecutter.component_name}}:latest .
```

## License

[License information]
