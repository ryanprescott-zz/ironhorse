"""Tests for {{cookiecutter.component_name}} core functionality."""

import pytest
from {{cookiecutter.component_name.replace('-', '_')}}.core import {{cookiecutter.component_class_name}}


class Test{{cookiecutter.component_class_name}}:
    """Test suite for {{cookiecutter.component_class_name}}."""

    def test_initialization(self) -> None:
        """Test {{cookiecutter.component_class_name}} initialization."""
        component = {{cookiecutter.component_class_name}}()
        assert component is not None

    def test_process(self) -> None:
        """Test process method."""
        component = {{cookiecutter.component_class_name}}()
        result = component.process({"test": "data"})
        assert result is not None
        assert isinstance(result, dict)
