"""
Extractor registry for managing and discovering extractors.
"""

from typing import Dict, Type, Optional, List
from .base import BaseExtractor


class ExtractorRegistry:
    """
    Registry for managing available extractors.

    Allows registration and discovery of extractor implementations.
    """

    _extractors: Dict[str, Type[BaseExtractor]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register an extractor class.

        Usage:
            @ExtractorRegistry.register('pdffigures2')
            class PDFFigures2Extractor(CombinedExtractor):
                ...

        Args:
            name: Unique identifier for this extractor
        """
        def decorator(extractor_class: Type[BaseExtractor]):
            cls._extractors[name] = extractor_class
            return extractor_class
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> Optional[BaseExtractor]:
        """
        Get an instance of a registered extractor.

        Args:
            name: Name of the extractor
            **kwargs: Arguments to pass to extractor constructor

        Returns:
            Extractor instance or None if not found
        """
        extractor_class = cls._extractors.get(name)
        if extractor_class is None:
            return None
        return extractor_class(**kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available extractors that have dependencies installed.

        Returns:
            List of extractor names that are ready to use
        """
        available = []
        for name, extractor_class in cls._extractors.items():
            try:
                instance = extractor_class()
                if instance.is_available():
                    available.append(name)
            except Exception:
                continue
        return available

    @classmethod
    def list_all(cls) -> List[str]:
        """
        List all registered extractors.

        Returns:
            List of all registered extractor names
        """
        return list(cls._extractors.keys())


def get_extractor(name: str, **kwargs) -> BaseExtractor:
    """
    Convenience function to get an extractor instance.

    Args:
        name: Name of the extractor
        **kwargs: Arguments to pass to extractor constructor

    Returns:
        Extractor instance

    Raises:
        ValueError: If extractor not found or not available
    """
    extractor = ExtractorRegistry.get(name, **kwargs)
    if extractor is None:
        raise ValueError(
            f"Extractor '{name}' not found. "
            f"Available: {ExtractorRegistry.list_all()}"
        )

    if not extractor.is_available():
        raise ValueError(
            f"Extractor '{name}' is not available. "
            f"Check that dependencies are installed."
        )

    return extractor
