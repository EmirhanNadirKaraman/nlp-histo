"""
PDF Parsers Package

This package contains specialized PDF parsers for extracting hierarchical text structure:
- MarkerParser: Primary parser using Marker for PDF→Markdown conversion
- NougatParser: Fallback parser using neural OCR for complex layouts
- PDFFiguresParser: Specialized parser for figure/table extraction
- EnsemblePDFParser: Orchestrator that coordinates all parsers

All parsers return standardized format matching the XML parser output:
{
    'path_list': [...],      # List of section hierarchy
    'path_string': str,      # "Section > Subsection" format
    'depth': int,            # Hierarchical depth
    'text': str              # Paragraph text content
}
"""

from .base_parser import BasePDFParser

__all__ = ['BasePDFParser']

# Parsers are imported lazily to avoid dependency issues
try:
    from .marker_parser import MarkerParser  # noqa: F401
    __all__.append('MarkerParser')
except ImportError:
    pass

try:
    from .nougat_parser import NougatParser  # noqa: F401
    __all__.append('NougatParser')
except ImportError:
    pass

try:
    from .pdffigures_parser import PDFFiguresParser  # noqa: F401
    __all__.append('PDFFiguresParser')
except ImportError:
    pass

try:
    from .ensemble_parser import EnsemblePDFParser  # noqa: F401
    __all__.append('EnsemblePDFParser')
except ImportError:
    pass
