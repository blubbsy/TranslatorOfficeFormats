"""
Factory for creating file processors based on file extension.
"""

import logging
from pathlib import Path
from typing import Type

from .base import BaseFileProcessor

logger = logging.getLogger("OfficeTranslator.ProcessorFactory")

# Registry of processors by extension
_PROCESSOR_REGISTRY: dict[str, Type[BaseFileProcessor]] = {}


def register_processor(extension: str, processor_class: Type[BaseFileProcessor]) -> None:
    """
    Register a processor class for a file extension.
    
    Args:
        extension: File extension (without dot, e.g., 'docx')
        processor_class: Processor class to register
    """
    ext = extension.lower().lstrip(".")
    _PROCESSOR_REGISTRY[ext] = processor_class
    logger.debug(f"Registered processor for .{ext}: {processor_class.__name__}")


def get_processor(file_path: str | Path) -> BaseFileProcessor:
    """
    Get the appropriate processor for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Instantiated processor for the file type
        
    Raises:
        ValueError: If no processor is registered for the file type
    """
    path = Path(file_path)
    extension = path.suffix.lower().lstrip(".")
    
    if extension not in _PROCESSOR_REGISTRY:
        supported = ", ".join(f".{ext}" for ext in _PROCESSOR_REGISTRY.keys())
        raise ValueError(
            f"Unsupported file type: .{extension}. Supported: {supported}"
        )
    
    processor_class = _PROCESSOR_REGISTRY[extension]
    logger.info(f"Creating {processor_class.__name__} for {path.name}")
    return processor_class()


def get_supported_extensions() -> list[str]:
    """
    Get list of supported file extensions.
    
    Returns:
        List of extensions (with dots, e.g., ['.docx', '.pptx'])
    """
    return [f".{ext}" for ext in sorted(_PROCESSOR_REGISTRY.keys())]


class ProcessorFactory:
    """
    Factory class for creating file processors.
    
    Provides a class-based interface to the processor registry.
    """
    
    @staticmethod
    def register(extension: str, processor_class: Type[BaseFileProcessor]) -> None:
        """Register a processor for a file extension."""
        register_processor(extension, processor_class)
    
    @staticmethod
    def get(file_path: str | Path) -> BaseFileProcessor:
        """Get a processor for a file."""
        return get_processor(file_path)
    
    @staticmethod
    def supported_extensions() -> list[str]:
        """Get list of supported extensions."""
        return get_supported_extensions()
    
    @staticmethod
    def is_supported(file_path: str | Path) -> bool:
        """Check if a file type is supported."""
        path = Path(file_path)
        extension = path.suffix.lower().lstrip(".")
        return extension in _PROCESSOR_REGISTRY


# Auto-register processors when their modules are imported
def _auto_register():
    """Auto-register all available processors."""
    try:
        from .docx_processor import DocxProcessor
        register_processor("docx", DocxProcessor)
    except ImportError:
        logger.debug("DocxProcessor not available")
    
    try:
        from .pptx_processor import PptxProcessor
        register_processor("pptx", PptxProcessor)
    except ImportError:
        logger.debug("PptxProcessor not available")
    
    try:
        from .xlsx_processor import XlsxProcessor
        register_processor("xlsx", XlsxProcessor)
    except ImportError:
        logger.debug("XlsxProcessor not available")
    
    try:
        from .pdf_processor import PdfProcessor
        register_processor("pdf", PdfProcessor)
    except ImportError:
        logger.debug("PdfProcessor not available")


# Run auto-registration
_auto_register()
