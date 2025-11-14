# AI Toolkit Component Templates

This directory contains cookiecutter templates for generating new AI Toolkit components.

## Usage

### Prerequisites

Install cookiecutter:

```bash
pip install cookiecutter
```

### Creating a New Component

From the project root directory:

```bash
cd components
cookiecutter ../templates/component
```

You will be prompted for:
- `component_name`: Name of the component (e.g., "docling-parser")
- `component_class_name`: Class name (e.g., "DoclingParser")
- `component_description`: Brief description
- `component_port`: API port number (default: 26000)
- `author`: Author name
- `python_version`: Python version (default: 3.11)

### Example

```bash
$ cd components
$ cookiecutter ../templates/component

component_name [my-component]: docling-parser
component_class_name [MyComponent]: DoclingParser
component_description [A reusable component for AI Toolkit]: Document parser using Docling
component_port [26000]: 26000
author [AI Toolkit Team]:
python_version [3.11]:
```

This will create a new component directory at `components/docling-parser/` with the standard structure.

## Template Structure

The component template includes:

```
{{cookiecutter.component_name}}/
├── core/                 # Core Python module (pip installable)
│   ├── __init__.py
│   └── {{component_name}}.py
├── api/                  # FastAPI web service
│   ├── __init__.py
│   ├── main.py
│   └── routes.py
├── config/               # Pydantic settings
│   ├── __init__.py
│   └── settings.py
├── tests/                # Pytests
│   ├── __init__.py
│   ├── test_core.py
│   └── test_api.py
├── pyproject.toml        # Package configuration
└── README.md             # Component documentation
```

## Standards

All generated components follow these standards:

- **Python**: 3.11+
- **Package Management**: uv
- **Data Models**: Pydantic v2
- **Configuration**: Pydantic Settings with environment variable support
- **Web Framework**: FastAPI
- **Server**: Uvicorn
- **Testing**: Pytest
- **Documentation**: Google-style docstrings, PEP 8 compliant

## Customization

After generation, customize the component by:

1. Implementing the core processing logic in `core/{{component_name}}.py`
2. Adding API endpoints in `api/routes.py`
3. Configuring settings in `config/settings.py`
4. Writing tests in `tests/`
5. Updating the README with component-specific documentation
