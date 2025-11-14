"""{{cookiecutter.component_class_name}} implementation.

This module provides the core functionality for {{cookiecutter.component_description}}.
"""

from typing import Any, Dict, List
from pydantic import BaseModel

from shared.component import Component


class {{cookiecutter.component_class_name}}(Component):
    """{{cookiecutter.component_description}}.

    This class provides the core processing capabilities for this component.
    Implements the Component interface with the process() method.
    """

    def __init__(self) -> None:
        """Initialize the {{cookiecutter.component_class_name}}."""
        pass

    def process(self, data: Any) -> Any:
        """Process input data (implements Component interface).

        This is the main entry point for the component. Implement your
        specific processing logic here.

        Args:
            data: Input data to process. Type depends on component requirements.

        Returns:
            Processed output data. Type depends on component implementation.

        Raises:
            ValueError: If data is invalid.
        """
        # TODO: Implement processing logic
        # Example: Handle different input types
        if isinstance(data, dict):
            # Process dict input
            return {"result": "processed"}
        elif isinstance(data, str):
            # Process string input
            return {"result": "processed", "input": data}
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")
