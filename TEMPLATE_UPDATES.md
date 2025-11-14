# Cookiecutter Template Updates

## Summary

Updated the cookiecutter template to properly use the restructured shared package with nested directory structure.

## Changes Made

### 1. Updated `api/routes.py`

**Before:**
```python
# Import shared schemas - adjust path as needed
# from shared.schemas import APIResponse, ResponseStatus
```

**After:**
```python
# Import shared schemas
from shared.schemas import APIResponse, ResponseStatus
```

**Also updated the `/process` endpoint to:**
- Use `APIResponse[ProcessResponse]` as response model
- Return `APIResponse.success()` instead of plain dict
- Properly type the response

### 2. Updated `README.md`

**Added to Development section:**
```bash
# Install shared schemas package (required dependency)
pip install -e ../shared

# Install component with dev dependencies
pip install -e ".[dev]"
```

**Added new section:**
```markdown
### Note on Shared Package

This component depends on the `shared` package for common data models and schemas. The shared package must be installed before this component can be used:

- **Document schema**: Standard document structure for parsed content
- **Chunk schema**: Text chunks for splitting/RAG operations
- **APIResponse**: Consistent API response wrapper

The shared package is located at `../shared` relative to this component.
```

### 3. Updated `pyproject.toml`

**Added comment in dependencies:**
```toml
dependencies = [
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    # Note: shared-schemas must be installed separately with: pip install -e ../shared
]
```

## Testing the Template

### Quick Test

Run the test script:
```bash
bash /Users/prescottrm/projects/ironhorse/test_template.sh
```

### Manual Test

```bash
cd /Users/prescottrm/projects/ironhorse/components

# Generate test component
cookiecutter ../templates/component --no-input \
    component_name=test-comp \
    component_description="Test component" \
    component_class_name=TestComp \
    component_port=8090 \
    python_version=3.11

# Set up and test
cd test-comp
python3 -m venv .venv
source .venv/bin/activate

# Install shared package (required!)
pip install -e ../shared

# Install component
pip install -e ".[dev]"

# Test imports
python3 -c "from shared.schemas import APIResponse; print('✓ Success!')"

# Run tests
pytest

# Clean up
cd ..
rm -rf test-comp
```

## Expected Behavior

Generated components will now:

1. **Import shared schemas directly**: No need to uncomment or adjust import paths
2. **Use APIResponse wrapper**: Endpoints return properly typed API responses
3. **Include setup instructions**: README clearly explains how to install shared package
4. **Work out of the box**: Following the README setup instructions will result in a working component

## Impact on Existing Components

Existing components (docling-parser, langchain-splitter) should:

1. Reinstall the shared package: `pip install -e ../shared`
2. Remove any `.pth` files that were manually created
3. Continue working as before

The new nested structure (`components/shared/shared/schemas/`) works with standard pip editable installs, eliminating the need for manual `.pth` file creation.

## Future Considerations

- Components can now be distributed with a proper dependency on `shared-schemas` if published to PyPI
- The shared package structure follows Python packaging best practices
- No special setup required for different Python environments (venv, conda, etc.)
