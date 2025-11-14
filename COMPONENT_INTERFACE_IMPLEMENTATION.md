# Component Interface Implementation

## Summary

Implemented a shared abstract base class (`Component`) that all AI toolkit components must implement. This provides a consistent interface across all components with a standardized `process()` method.

## What Was Implemented

### 1. Component ABC (`shared/component.py`)

Created an abstract base class with a single abstract method:

```python
from abc import ABC, abstractmethod
from typing import Any

class Component(ABC):
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process input data."""
        raise NotImplementedError("Subclasses must implement the process() method")
```

**Location**: `/Users/prescottrm/projects/ironhorse/components/shared/shared/component.py`

### 2. Updated Shared Package

Updated `shared/__init__.py` to export the Component class:

```python
from shared.component import Component

__all__ = ["Component"]
```

### 3. Updated DoclingParser

**File**: `components/docling-parser/docling_parser/core/docling_parser.py`

- Inherits from `Component`
- Implements `process()` method that accepts:
  - String (file path)
  - Dict with `file_path` and optional `doc_id`
- Delegates to existing `parse_document()` method

**Example usage**:
```python
parser = DoclingParser()

# Simple file path
doc = parser.process("/path/to/file.pdf")

# With custom doc_id
doc = parser.process({
    'file_path': '/path/to/file.pdf',
    'doc_id': 'custom_123'
})
```

### 4. Updated LangChainSplitter

**File**: `components/langchain-splitter/langchain_splitter/core/langchain_splitter.py`

- Inherits from `Component`
- Implements `process()` method that accepts:
  - String (text to split)
  - Document object
  - List of Document objects
  - Dict with `text` or `document` key

**Example usage**:
```python
splitter = LangChainSplitter()

# Simple text
chunks = splitter.process("Some text to split")

# Document object
chunks = splitter.process(document)

# Multiple documents
chunks = splitter.process([doc1, doc2, doc3])

# Dict with text and source_id
chunks = splitter.process({
    'text': 'Some text',
    'source_id': 'doc_123'
})
```

### 5. Updated Cookiecutter Template

**File**: `templates/component/{{cookiecutter.component_name}}/{{cookiecutter.component_name.replace('-', '_')}}/core/{{cookiecutter.component_name.replace('-', '_')}}.py`

- Class now inherits from `Component`
- Imports `Component` from shared
- Includes example implementation showing how to handle different input types
- Updated type hints to use `Any` instead of `Dict[str, Any]`

**Generated class structure**:
```python
from shared.component import Component
from typing import Any

class YourComponent(Component):
    def process(self, data: Any) -> Any:
        """Process input data (implements Component interface)."""
        if isinstance(data, dict):
            return {"result": "processed"}
        elif isinstance(data, str):
            return {"result": "processed", "input": data}
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")
```

## Tests Added

### DoclingParser Tests
Added 3 new test methods in `components/docling-parser/tests/test_core.py`:

1. `test_process_with_string_input` - Tests string file path input
2. `test_process_with_dict_input` - Tests dict with file_path and doc_id
3. `test_process_with_invalid_input` - Tests error handling

### LangChainSplitter Tests
Added 7 new test methods in `components/langchain-splitter/tests/test_core.py`:

1. `test_process_with_string_input` - Tests string text input
2. `test_process_with_document_input` - Tests Document object input
3. `test_process_with_document_list` - Tests list of Documents
4. `test_process_with_dict_text_input` - Tests dict with text key
5. `test_process_with_dict_document_input` - Tests dict with document key
6. `test_process_with_invalid_input` - Tests error handling

### Template Tests
Updated test template in `templates/component/{{cookiecutter.component_name}}/tests/test_core.py`:

1. `test_process_with_dict` - Tests dict input
2. `test_process_with_string` - Tests string input
3. `test_process_with_invalid_input` - Tests error handling

## Testing

### Quick Test
Run the comprehensive test script:

```bash
bash /Users/prescottrm/projects/ironhorse/test_component_interface.sh
```

This script:
1. Verifies Component ABC is correctly defined
2. Verifies DoclingParser implements Component
3. Verifies LangChainSplitter implements Component
4. Runs all process() method tests for docling-parser
5. Runs all process() method tests for langchain-splitter
6. Generates a test component from the template and verifies it implements Component

### Manual Testing

**Test docling-parser**:
```bash
cd components/docling-parser
.venv/bin/pytest tests/test_core.py -v -k "process"
```

**Test langchain-splitter**:
```bash
cd components/langchain-splitter
.venv/bin/pytest tests/test_core.py -v -k "process"
```

**Test template generation**:
```bash
cd components
cookiecutter ../templates/component --no-input \
    component_name=test-comp \
    component_class_name=TestComp \
    component_port=8099
cd test-comp
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../shared && pip install -e ".[dev]"
pytest tests/test_core.py -v
```

## Benefits

1. **Consistent Interface**: All components now have a standard `process()` method
2. **Type Safety**: ABC ensures all components implement the required method
3. **Flexibility**: `process()` accepts `Any` type, allowing components to define their own input formats
4. **Backward Compatible**: Existing methods (`parse_document`, `split_text`, etc.) still work
5. **Future Proof**: Easy to add new components that follow the same pattern
6. **Testable**: Interface makes it easy to mock components for testing

## Design Decisions

### Why `Any -> Any`?
- Allows maximum flexibility for different component types
- Each component can accept its natural input format
- Can be extended to use generics later if needed

### Why Keep Existing Methods?
- Maintains backward compatibility
- Provides more specific interfaces for component users
- `process()` serves as the unified entry point while specific methods offer convenience

### Why Minimal Interface?
- Keeps the interface simple and focused
- Avoids over-engineering
- Easy to extend in the future if needed

## Future Enhancements

Potential improvements:
1. Add generics for type safety: `Component[T, R]`
2. Add metadata properties: `name`, `version`, `description`
3. Add lifecycle hooks: `initialize()`, `shutdown()`
4. Add validation methods: `validate_input()`, `validate_output()`
5. Add common functionality: error handling, logging, metrics

## Migration Guide

For existing code using these components:

**Before**:
```python
parser = DoclingParser()
doc = parser.parse_document("/path/to/file.pdf")
```

**After (both work)**:
```python
parser = DoclingParser()

# Old way still works
doc = parser.parse_document("/path/to/file.pdf")

# New unified interface
doc = parser.process("/path/to/file.pdf")
```

No breaking changes - all existing code continues to work!

## Files Modified

1. `components/shared/shared/component.py` - Created
2. `components/shared/shared/__init__.py` - Updated to export Component
3. `components/docling-parser/docling_parser/core/docling_parser.py` - Updated
4. `components/docling-parser/tests/test_core.py` - Added tests
5. `components/langchain-splitter/langchain_splitter/core/langchain_splitter.py` - Updated
6. `components/langchain-splitter/tests/test_core.py` - Added tests
7. `templates/component/.../core/....py` - Updated
8. `templates/component/.../tests/test_core.py` - Updated

## Test Scripts Created

1. `/Users/prescottrm/projects/ironhorse/test_component_interface.sh` - Comprehensive test suite

## Documentation Created

1. `/Users/prescottrm/projects/ironhorse/COMPONENT_INTERFACE_IMPLEMENTATION.md` - This file
