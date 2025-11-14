"""Abstract base class for all AI toolkit components.

This module defines the Component interface that all toolkit components
must implement to ensure consistent processing interface across the system.
"""

from abc import ABC, abstractmethod
from typing import Any


class Component(ABC):
    """Abstract base class for AI toolkit components.

    All components in the AI toolkit should inherit from this class and
    implement the process() method. This ensures a consistent interface
    across all components for processing data.

    Example:
        class MyComponent(Component):
            def process(self, data: Any) -> Any:
                # Implementation here
                return processed_data
    """

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process input data.

        This is the main entry point for all component processing.
        Subclasses must implement this method to define their specific
        processing logic.

        Args:
            data: Input data to process. Type depends on the specific component.

        Returns:
            Processed output data. Type depends on the specific component.

        Raises:
            NotImplementedError: If subclass doesn't implement this method.
        """
        raise NotImplementedError("Subclasses must implement the process() method")
