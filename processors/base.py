"""
Base processor interface and content chunk definitions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Optional


class ContentType(Enum):
    """Types of content that can be extracted from documents."""
    TEXT = "text"
    IMAGE = "image"
    TABLE_CELL = "table_cell"


@dataclass
class ContentChunk:
    """
    Represents a chunk of content extracted from a document.
    """
    id: str  # Unique identifier for this chunk
    content_type: ContentType
    text: Optional[str] = None  # For text content
    image_data: Optional[bytes] = None  # For image content
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Location info for progress reporting
    location: str = ""  # e.g., "Slide 3" or "Paragraph 15"
    
    def __post_init__(self):
        if self.content_type == ContentType.TEXT and not self.text:
            raise ValueError("TEXT content must have text")
        if self.content_type == ContentType.IMAGE and not self.image_data:
            raise ValueError("IMAGE content must have image_data")


class BaseFileProcessor(ABC):
    """
    Abstract base class for document processors.
    
    All file format plugins must inherit from this class and implement
    the required methods.
    """
    
    # File extensions this processor handles
    SUPPORTED_EXTENSIONS: list[str] = []
    
    def __init__(self):
        self._document: Any = None
        self._file_path: Optional[Path] = None
        self._chunks_cache: dict[str, ContentChunk] = {}
    
    @property
    def file_path(self) -> Optional[Path]:
        """Get the loaded file path."""
        return self._file_path
    
    @property
    def is_loaded(self) -> bool:
        """Check if a document is loaded."""
        return self._document is not None
    
    @abstractmethod
    def load(self, file_path: str | Path) -> None:
        """
        Load a document from file.
        
        Args:
            file_path: Path to the document file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        pass
    
    @abstractmethod
    def extract_content_generator(self) -> Generator[ContentChunk, None, None]:
        """
        Extract content chunks from the document.
        
        Yields chunks one at a time for progress tracking.
        
        Yields:
            ContentChunk objects representing document content
        """
        pass
    
    @abstractmethod
    def apply_translation(
        self,
        chunk_id: str,
        translated_content: str | bytes,
    ) -> None:
        """
        Apply a translation to a specific content chunk.
        
        Args:
            chunk_id: ID of the chunk to update
            translated_content: Translated text or image bytes
        """
        pass
    
    @abstractmethod
    def save(self, output_path: str | Path) -> None:
        """
        Save the modified document.
        
        Args:
            output_path: Path where to save the document
        """
        pass
    
    def get_total_chunks(self) -> int:
        """
        Get total number of content chunks (for progress bar).
        
        Default implementation iterates through all chunks.
        Override in subclasses for more efficient counting.
        
        Returns:
            Total number of chunks
        """
        count = 0
        for _ in self.extract_content_generator():
            count += 1
        return count
    
    def _validate_loaded(self) -> None:
        """Raise error if no document is loaded."""
        if not self.is_loaded:
            raise RuntimeError("No document loaded. Call load() first.")
