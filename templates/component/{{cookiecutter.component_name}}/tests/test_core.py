"""Tests for {{cookiecutter.component_name}} core functionality."""

import pytest
from {{cookiecutter.component_name.replace('-', '_')}}.core import {{cookiecutter.component_class_name}}


class Test{{cookiecutter.component_class_name}}:
    """Test suite for {{cookiecutter.component_class_name}}."""

    def test_initialization(self) -> None:
        """Test {{cookiecutter.component_class_name}} initialization."""
        component = {{cookiecutter.component_class_name}}()
        assert component is not None

    def test_process_with_dict(self) -> None:
        """Test process method with dict input."""
        component = {{cookiecutter.component_class_name}}()
        result = component.process({"test": "data"})
        assert result is not None
        assert isinstance(result, dict)
        assert "result" in result

    def test_process_with_string(self) -> None:
        """Test process method with string input."""
        component = {{cookiecutter.component_class_name}}()
        result = component.process("test input")
        assert result is not None
        assert isinstance(result, dict)
        assert "input" in result

    def test_process_with_invalid_input(self) -> None:
        """Test process method with invalid input type."""
        component = {{cookiecutter.component_class_name}}()
        with pytest.raises(ValueError, match="Unsupported input type"):
            component.process(12345)  # Invalid type
