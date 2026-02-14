"""
PDF document processor using PyMuPDF for text overlay approach.
"""

import io
import logging
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
from PIL import Image

from .base import BaseFileProcessor, ContentChunk, ContentType
from settings import settings

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
        self._toc: list = []
        self._translated_toc: list = []
    
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
        """Extract text blocks and TOC from all pages."""
        self._validate_loaded()
        
        chunk_count = 0
        
        # 1. Extract TOC
        self._toc = self._document.get_toc()
        for i, entry in enumerate(self._toc):
            level, title, page = entry[:3]
            if title:
                chunk_id = f"toc_{i}"
                yield ContentChunk(
                    id=chunk_id,
                    content_type=ContentType.TEXT,
                    text=title,
                    location=f"Table of Contents",
                    metadata={"toc_index": i, "level": level}
                )
                chunk_count += 1

        # 2. Extract Page Content
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
                        img_data = None
                        
                        if xref > 0:
                            img = self._document.extract_image(xref)
                            if img:
                                img_data = img["image"]
                        
                        # Fallback: if no xref but it's an image type, try to render the area
                        if not img_data:
                            pix = page.get_pixmap(clip=block["bbox"], matrix=fitz.Matrix(2, 2))
                            img_data = pix.tobytes("png")
                        
                        if img_data:
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
                                image_data=img_data,
                                location=f"Page {page_num}, Image",
                                metadata={
                                    "page": page_idx,
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
        
        if chunk_id.startswith("toc_"):
            try:
                idx = int(chunk_id.split("_")[1])
                if not self._translated_toc:
                    self._translated_toc = [list(entry) for entry in self._toc]
                self._translated_toc[idx][1] = str(translated_content)
            except (ValueError, IndexError):
                logger.warning(f"Invalid TOC chunk ID: {chunk_id}")
            return

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
        """Save the modified PDF with text overlays and translated TOC."""
        self._validate_loaded()
        
        # 1. Apply TOC if translated
        if self._translated_toc:
            try:
                self._document.set_toc(self._translated_toc)
                logger.info("Applied translated Table of Contents")
            except Exception as e:
                logger.error(f"Failed to set TOC: {e}")

        # 2. Apply Page Overlays
        for chunk_id, block_info in self._text_blocks.items():
            if block_info.get("translated_text"):
                self._apply_text_overlay(block_info)
            elif block_info.get("translated_image"):
                self._apply_image_overlay(block_info)
        
        path = Path(output_path)
        self._document.save(path)
        logger.info(f"Saved PDF: {path.name}")
    
    def _apply_text_overlay(self, block_info: dict) -> None:
        """Apply text overlay for a translated block with color preservation."""
        try:
            page = self._document[block_info["page_idx"]]
            rect = block_info["rect"]
            translated_text = block_info["translated_text"]
            
            if not translated_text:
                return

            # Analyze original font size and color
            avg_font_size = 11.0
            orig_color_int = 0
            if block_info.get("lines"):
                sizes = []
                colors = []
                for line in block_info["lines"]:
                    for span in line.get("spans", []):
                        sizes.append(span.get("size", 11.0))
                        colors.append(span.get("color", 0))
                if sizes:
                    avg_font_size = sum(sizes) / len(sizes)
                if colors:
                    from collections import Counter
                    orig_color_int = Counter(colors).most_common(1)[0][0]
            
            # Convert color int to RGB floats (0-1)
            # fitz colors are often sRGB integers
            r = (orig_color_int >> 16) & 0xFF
            g = (orig_color_int >> 8) & 0xFF
            b = orig_color_int & 0xFF
            pdf_fg = (r/255.0, g/255.0, b/255.0)

            # Try to sample background color from the page
            try:
                pix = page.get_pixmap(clip=rect)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                # Sample edges for background color
                w, h = img.size
                samples = [img.getpixel((0,0)), img.getpixel((w-1,0)), img.getpixel((0,h-1)), img.getpixel((w-1,h-1)),
                           img.getpixel((w//2,0)), img.getpixel((w//2,h-1)), img.getpixel((0,h//2)), img.getpixel((w-1,h//2))]
                avg_bg = tuple(sum(s[i] for s in samples) // len(samples) for i in range(3))
                pdf_bg = tuple(v / 255.0 for v in avg_bg)
            except Exception as e:
                logger.debug(f"Failed to sample PDF colors: {e}")
                pdf_bg = (1, 1, 1)

            # Draw rectangle to cover original text
            page.draw_rect(rect, color=pdf_bg, fill=pdf_bg)
            
            # Determine font name based on language
            special_font = self.get_special_font()
            fontname = "helv" # Default
            
            if special_font:
                # Map system fonts to PyMuPDF built-ins or fonts that must be embedded
                # china-s, china-t, jpn, kor are the built-in CJK fonts in PyMuPDF
                lang = self.target_language.lower()
                if "chinese" in lang or "zh" in lang:
                    fontname = "china-t" if ("traditional" in lang or "hant" in lang) else "china-s"
                elif "japanese" in lang or "ja" in lang:
                    fontname = "jpn"
                elif "korean" in lang or "ko" in lang:
                    fontname = "kor"
            
            # Try to fit text in original box first
            inserted = False
            for size in range(int(avg_font_size) + 1, 4, -1):
                rc = page.insert_textbox(
                    rect,
                    translated_text,
                    fontsize=size,
                    fontname=fontname,
                    color=pdf_fg,
                    align=0
                )
                if rc >= 0:  # Everything fit (rc is remaining space)
                    inserted = True
                    break
            
            # If it didn't fit, try expanding height to page bottom
            if not inserted:
                # Expanded rect: same x-range, y0 to page height minus margin
                margin = 36 # ~0.5 inch
                expanded_rect = fitz.Rect(rect.x0, rect.y0, rect.x1, page.rect.height - margin)
                
                # Try fitting in expanded box
                for size in range(int(avg_font_size), 6, -1):
                    rc = page.insert_textbox(
                        expanded_rect,
                        translated_text,
                        fontsize=size,
                        fontname=fontname,
                        color=pdf_fg,
                        align=0
                    )
                    if rc >= 0:
                        inserted = True
                        break
                
                # Last resort fallback: force it into the expanded box anyway
                if not inserted:
                    page.insert_textbox(
                        expanded_rect,
                        translated_text,
                        fontsize=6,
                        fontname=fontname,
                        color=pdf_fg,
                        align=0
                    )
                    logger.warning(f"Forced overflow insertion for block on page {block_info['page_idx']}")
            
        except Exception as e:
            logger.error(f"Failed to apply text overlay: {e}")
    
    def _apply_image_overlay(self, block_info: dict) -> None:
        """Apply translated image overlay."""
        try:
            page = self._document[block_info["page_idx"]]
            rect = block_info["rect"]
            image_data = block_info["translated_image"]
            
            # Remove original image area by filling with sampled background from surroundings
            try:
                # Sample 5 pixels outside the rect to get the page background
                sample_rect = fitz.Rect(rect.x0 - 5, rect.y0 - 5, rect.x1 + 5, rect.y1 + 5)
                pix = page.get_pixmap(clip=sample_rect)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                # Sample corners of the pixmap (which are outside the original rect)
                w, h = img.size
                corners = [img.getpixel((0,0)), img.getpixel((w-1,0)), img.getpixel((0,h-1)), img.getpixel((w-1,h-1))]
                avg_bg = tuple(sum(c[i] for c in corners) // len(corners) for i in range(3))
                pdf_bg = tuple(v / 255.0 for v in avg_bg)
            except:
                pdf_bg = (1, 1, 1)

            page.draw_rect(rect, color=pdf_bg, fill=pdf_bg)
            
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
