# Developer Quick Reference

Quick reference for common AI Toolkit development tasks.

## Quick Links

- 📖 [Full Development Guide](COMPONENT_DEVELOPMENT_GUIDE.md)
- 🚀 [Quick Start](../QUICKSTART.md)
- 📚 [Main README](../README.md)

---

## Installation & Setup

### Install Prerequisites

```bash
# Python 3.11+
python3.11 --version

# uv (fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# cookiecutter
pip install cookiecutter

# Docker
docker --version
```

### Create New Component

```bash
cd components/
cookiecutter ../templates/component
cd <your-component-name>
```

### Setup Python Environment

```bash
# Create venv
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or: .venv\Scripts\activate  # Windows

# Install shared schemas
pip install -e ../../shared/

# Install component with dev deps
pip install -e ".[dev]"
```

---

## Development Workflow

### Running Tests

```bash
# All tests
pytest

# Verbose mode
pytest -v

# With coverage
pytest --cov=<component_name> --cov-report=html

# Specific test file
pytest tests/test_core.py

# Specific test
pytest tests/test_core.py::TestClass::test_method

# Watch mode (install pytest-watch first)
ptw
```

### Code Quality

```bash
# Format code (install black first)
black .

# Lint code (install ruff first)
ruff check .

# Type check (install mypy first)
mypy <component_name>/

# All at once
black . && ruff check . && mypy <component_name>/
```

### Running Locally

```bash
# Run FastAPI server
python -m <component_name>.api.main

# With uvicorn (auto-reload)
uvicorn <component_name>.api.main:app --reload --port <port>

# Access API docs
open http://localhost:<port>/docs
```

---

## Docker Commands

### Build & Run

```bash
# Build specific component (from project root)
docker build -f platform/docker/components/<component>/Dockerfile \
  -t <component>:latest .

# Build all services
docker-compose build

# Run specific service
docker-compose up -d <service-name>

# Run all services
docker-compose up -d

# View logs
docker-compose logs -f <service-name>

# Stop services
docker-compose down
```

### Debugging

```bash
# Check service status
docker-compose ps

# Execute command in container
docker-compose exec <service-name> bash

# Check health
curl http://localhost:<port>/health

# View real-time logs
docker-compose logs -f <service-name>
```

---

## Common File Locations

### Component Structure

```
components/<component-name>/
├── core/              # Core logic
│   └── <component>.py
├── api/               # FastAPI endpoints
│   ├── main.py
│   └── routes.py
├── config/            # Configuration
│   └── settings.py
├── tests/             # Tests
│   ├── test_core.py
│   └── test_api.py
└── pyproject.toml     # Dependencies
```

### Shared Resources

```
shared/schemas/        # Common data models
├── document.py        # Document schema
├── chunk.py          # Chunk schema
└── response.py       # API response schema
```

### Docker Files

```
platform/docker/components/<component>/
├── Dockerfile
└── .dockerignore
```

### N8N Integration

```
integrations/orchestrators/n8n/
├── bindings/          # Python bindings
│   └── <component>_binding.py
└── pipelines/         # Workflow JSONs
    └── <workflow>.json
```

---

## Environment Variables

### Component Variables

```bash
# API settings
export <COMPONENT>_HOST=0.0.0.0
export <COMPONENT>_PORT=26000
export <COMPONENT>_LOG_LEVEL=INFO

# Component-specific settings
export <COMPONENT>_<SETTING>=value
```

### Using .env File

```bash
# Create .env file
cat > .env << EOF
COMPONENT_HOST=0.0.0.0
COMPONENT_PORT=26000
COMPONENT_LOG_LEVEL=DEBUG
EOF

# Load variables
export $(cat .env | xargs)

# Or use in docker-compose
docker-compose --env-file .env up
```

---

## API Endpoints

### Standard Endpoints

Every component should have:

```
GET  /health                # Health check
GET  /docs                  # Swagger UI
GET  /redoc                 # ReDoc
POST /api/v1/<endpoint>     # Main endpoints
```

### Testing Endpoints

```bash
# Health check
curl http://localhost:<port>/health

# POST request
curl -X POST "http://localhost:<port>/api/v1/<endpoint>" \
  -H "Content-Type: application/json" \
  -d '{"key": "value"}'

# GET request with params
curl "http://localhost:<port>/api/v1/<endpoint>?param=value"
```

---

## Common Tasks

### Add New Dependency

```bash
# Edit pyproject.toml
[project]
dependencies = [
    "existing-package>=1.0.0",
    "new-package>=2.0.0",  # Add here
]

# Reinstall
pip install -e ".[dev]"
```

### Update Shared Schemas

```bash
# Edit shared/schemas/<schema>.py
# Then reinstall in your component
cd components/<component>
pip install -e ../../shared/
```

### Add New API Endpoint

1. Add request/response models in `api/routes.py`
2. Add router endpoint function
3. Update docstrings
4. Add tests in `tests/test_api.py`
5. Test with curl or API docs

### Add New Core Method

1. Add method to core class in `core/<component>.py`
2. Add docstrings
3. Add tests in `tests/test_core.py`
4. Update API routes if needed

---

## Troubleshooting

### Module Not Found

```bash
# Ensure shared is installed
pip install -e ../../shared/

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Add to code if needed
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
```

### Port Already in Use

```bash
# Find process using port
lsof -i :<port>

# Kill process
kill -9 <PID>

# Or use different port
export COMPONENT_PORT=<new-port>
```

### Docker Build Fails

```bash
# Clear cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache <service>

# Check logs
docker-compose logs <service>
```

### Tests Failing

```bash
# Run with verbose output
pytest -v -s

# Run single test with output
pytest tests/test_core.py::test_name -v -s

# Check imports
python -c "from <component>.core import <Class>"
```

---

## Git Workflow

```bash
# Create branch
git checkout -b feature/<component-name>

# Stage changes
git add components/<component>/

# Commit
git commit -m "Add <component> component

- Brief description
- Key features"

# Push
git push origin feature/<component-name>
```

---

## Useful Python Snippets

### Testing API Endpoint

```python
from fastapi.testclient import TestClient
from <component>.api.main import app

client = TestClient(app)

def test_endpoint():
    response = client.post(
        "/api/v1/endpoint",
        json={"key": "value"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

@patch('<component>.core.<component>.<ExternalClass>')
def test_method(mock_class):
    mock_instance = Mock()
    mock_instance.method.return_value = "mocked"
    mock_class.return_value = mock_instance

    # Test your code
    result = your_function()
    assert result == "expected"
```

### Async Endpoint

```python
@router.post("/endpoint")
async def endpoint(request: RequestModel) -> Dict[str, Any]:
    """Endpoint description."""
    try:
        result = await process_async(request)
        return APIResponse.success(data=result).model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Port Assignments

Keep track of port assignments to avoid conflicts:

| Component | Port |
|-----------|------|
| docling-parser | 26000 |
| langchain-splitter | 26001 |
| embedding-generator | 26002 |
| vector-store | 26003 |
| llm-chat | 26004 |
| n8n | 5678 |

---

## Performance Tips

1. **Use async/await** for I/O operations
2. **Batch process** when possible
3. **Cache results** for expensive operations
4. **Use connection pooling** for databases
5. **Profile with cProfile** to find bottlenecks

```python
# Async example
async def process_batch(items: List[str]) -> List[Result]:
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)
```

---

## Security Checklist

- [ ] No hardcoded credentials
- [ ] Environment variables for secrets
- [ ] Input validation with Pydantic
- [ ] Rate limiting on endpoints
- [ ] HTTPS in production
- [ ] Authentication/authorization
- [ ] Don't commit .env files
- [ ] Use secrets management in production

---

## Before Submitting

```bash
# 1. Format and lint
black . && ruff check .

# 2. Run all tests
pytest

# 3. Check coverage
pytest --cov=<component> --cov-report=term-missing

# 4. Build Docker image
docker build -f platform/docker/components/<component>/Dockerfile .

# 5. Test in Docker
docker-compose up -d <component>
curl http://localhost:<port>/health

# 6. Update documentation
# - README.md
# - Docstrings
# - Examples

# 7. Commit and push
git add .
git commit -m "Descriptive message"
git push origin feature/<component>
```

---

## Help & Resources

- **Project README**: `/README.md`
- **Full Dev Guide**: `docs/COMPONENT_DEVELOPMENT_GUIDE.md`
- **Quick Start**: `/QUICKSTART.md`
- **Component Examples**: `components/docling-parser/`, `components/langchain-splitter/`
- **Shared Schemas**: `shared/schemas/`

---

## Keyboard Shortcuts (Terminal)

```bash
Ctrl+C        # Stop running process
Ctrl+Z        # Suspend process
Ctrl+D        # Exit shell/deactivate venv
Ctrl+R        # Search command history
Ctrl+A        # Go to start of line
Ctrl+E        # Go to end of line
```

---

**Last Updated**: 2024
**Version**: 0.1.0

For detailed information, see [Full Development Guide](COMPONENT_DEVELOPMENT_GUIDE.md)
