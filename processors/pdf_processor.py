"""
PDF document processor using PyMuPDF for text overlay approach.
"""

import io
import logging
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF

from .base import BaseFileProcessor, ContentChunk, ContentType

logger = logging.getLogger("OfficeTranslator.PdfProcessor")


class PdfProcessor(BaseFileProcessor):
    """
    Processor for PDF documents using overlay approach.
    
    Strategy: Instead of modifying original text (which is complex and
    often breaks layout), we:
    1. Extract text blocks with their positions
    2. Draw white rectangles over original text
    3. Draw translated text in same positions
    
    This preserves layout while showing translated content.
    """
    
    SUPPORTED_EXTENSIONS = ["pdf"]
    
    def __init__(self):
        super().__init__()
        self._document: fitz.Document = None
        self._text_blocks: dict[str, dict] = {}  # id -> {page, rect, text, ...}
    
    def load(self, file_path: str | Path) -> None:
        """Load a PDF file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        
        try:
            self._document = fitz.open(path)
            self._file_path = path
            self._text_blocks.clear()
            logger.info(f"Loaded PDF: {path.name} ({len(self._document)} pages)")
        except Exception as e:
            raise ValueError(f"Invalid PDF file: {e}")
    
    def extract_content_generator(self) -> Generator[ContentChunk, None, None]:
        """Extract text blocks from all pages."""
        self._validate_loaded()
        
        chunk_count = 0
        
        for page_idx, page in enumerate(self._document):
            page_num = page_idx + 1
            
            # Get text blocks with positions
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            
            for block_idx, block in enumerate(blocks):
                if block["type"] == 0:  # Text block
                    # Combine all lines/spans in the block
                    lines_text = []
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        lines_text.append(line_text)
                    
                    # Basic cleanup: Join lines, fix hyphenation
                    raw_text = "\n".join(lines_text)
                    # Replace hyphen-newline with empty string (de-hyphenate)
                    cleaned_text = raw_text.replace("-\n", "")
                    # Replace remaining newlines with space to form continuous text
                    cleaned_text = cleaned_text.replace("\n", " ")
                    
                    cleaned_text = cleaned_text.strip()
                    
                    if cleaned_text:
                        chunk_id = f"page{page_idx}_block{block_idx}"
                        
                        # Store block info for later modification
                        self._text_blocks[chunk_id] = {
                            "page_idx": page_idx,
                            "rect": fitz.Rect(block["bbox"]),
                            "original_text": cleaned_text,
                            "translated_text": None,
                            "lines": block.get("lines", []),
                        }
                        
                        yield ContentChunk(
                            id=chunk_id,
                            content_type=ContentType.TEXT,
                            text=cleaned_text,
                            location=f"Page {page_num}",
                            metadata={
                                "page": page_idx,
                                "block": block_idx,
                                "bbox": block["bbox"],
                            }
                        )
                        chunk_count += 1
                
                elif block["type"] == 1:  # Image block
                    try:
                        # Extract image
                        xref = block.get("xref", 0)
                        if xref:
                            img = self._document.extract_image(xref)
                            if img:
                                img_id = f"page{page_idx}_img{block_idx}"
                                self._text_blocks[img_id] = {
                                    "page_idx": page_idx,
                                    "rect": fitz.Rect(block["bbox"]),
                                    "type": "image",
                                    "xref": xref,
                                }
                                
                                yield ContentChunk(
                                    id=img_id,
                                    content_type=ContentType.IMAGE,
                                    image_data=img["image"],
                                    location=f"Page {page_num}, Image",
                                    metadata={
                                        "page": page_idx,
                                        "ext": img.get("ext", "png"),
                                    }
                                )
                                chunk_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to extract image: {e}")
        
        logger.info(f"Extracted {chunk_count} content blocks from PDF")
    
    def apply_translation(
        self,
        chunk_id: str,
        translated_content: str | bytes,
    ) -> None:
        """Store translation for later application during save."""
        self._validate_loaded()
        
        if chunk_id not in self._text_blocks:
            logger.warning(f"Block not found: {chunk_id}")
            return
        
        block_info = self._text_blocks[chunk_id]
        
        if block_info.get("type") == "image":
            # Store translated image
            if isinstance(translated_content, bytes):
                block_info["translated_image"] = translated_content
        else:
            # Store translated text
            block_info["translated_text"] = str(translated_content)
        
        logger.debug(f"Stored translation for {chunk_id}")
    
    def save(self, output_path: str | Path) -> None:
        """Save the modified PDF with text overlays."""
        self._validate_loaded()
        
        # Apply all translations as overlays
        for chunk_id, block_info in self._text_blocks.items():
            if block_info.get("translated_text"):
                self._apply_text_overlay(block_info)
            elif block_info.get("translated_image"):
                self._apply_image_overlay(block_info)
        
        path = Path(output_path)
        self._document.save(path)
        logger.info(f"Saved PDF: {path.name}")
    
    def _apply_text_overlay(self, block_info: dict) -> None:
        """Apply text overlay for a translated block."""
        try:
            page = self._document[block_info["page_idx"]]
            rect = block_info["rect"]
            translated_text = block_info["translated_text"]
            
            if not translated_text:
                return

            # Analyze original font size
            avg_font_size = 11.0
            if block_info.get("lines"):
                sizes = []
                for line in block_info["lines"]:
                    for span in line.get("spans", []):
                        sizes.append(span.get("size", 11.0))
                if sizes:
                    avg_font_size = sum(sizes) / len(sizes)
            
            # Draw white rectangle to cover original text
            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
            
            # Reduce padding (use rect directly or minimal padding)
            text_rect = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y1)
            
            # Try to fit text, starting slightly larger
            start_size = int(avg_font_size) + 2
            inserted = False
            
            # Loop down to very small size
            for size in range(start_size, 4, -1):
                rc = page.insert_textbox(
                    text_rect,
                    translated_text,
                    fontsize=size,
                    fontname="helv",
                    color=(0, 0, 0),
                    align=0 # Left align
                )
                if rc >= 0:  # Success
                    inserted = True
                    break
            
            if not inserted:
                # Fallback: Just insert text at top-left, ignoring overflow
                # This ensures text is visible even if wrapping failed
                page.insert_text(
                    text_rect.tl + (0, avg_font_size),
                    translated_text,
                    fontsize=max(5, avg_font_size),
                    fontname="helv",
                    color=(0, 0, 0)
                )
                logger.warning(f"Forced insertion for block on page {block_info['page_idx']}")
            
        except Exception as e:
            logger.error(f"Failed to apply text overlay: {e}")
    
    def _apply_image_overlay(self, block_info: dict) -> None:
        """Apply translated image overlay."""
        try:
            page = self._document[block_info["page_idx"]]
            rect = block_info["rect"]
            image_data = block_info["translated_image"]
            
            # Remove original image area
            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
            
            # Insert new image
            page.insert_image(rect, stream=image_data)
            
        except Exception as e:
            logger.error(f"Failed to apply image overlay: {e}")
    
    def get_total_chunks(self) -> int:
        """Count total blocks efficiently."""
        self._validate_loaded()
        
        count = 0
        for page in self._document:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:  # Text
                    block_text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            block_text += span.get("text", "")
                    if block_text.strip():
                        count += 1
                elif block["type"] == 1:  # Image
                    count += 1
        return count
