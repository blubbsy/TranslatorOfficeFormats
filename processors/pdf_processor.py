"""
PDF document processor using PyMuPDF for text overlay approach.
"""

import io
import logging
from collections import Counter
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
    2. Use redaction annotations to remove original image content
    3. Draw translated text / images in same positions

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
            self._toc = []
            self._translated_toc = []
            logger.info(f"Loaded PDF: {path.name} ({len(self._document)} pages)")
        except Exception as e:
            raise ValueError(f"Invalid PDF file: {e}")

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_content_generator(self) -> Generator[ContentChunk, None, None]:
        """Extract text blocks, images, and TOC from all pages."""
        self._validate_loaded()

        chunk_count = 0

        # 1. Extract TOC
        try:
            self._toc = self._document.get_toc()
        except Exception as e:
            logger.warning(f"Failed to extract TOC: {e}")
            self._toc = []

        for i, entry in enumerate(self._toc):
            level, title, _page = entry[:3]
            if title and title.strip():
                chunk_id = f"toc_{i}"
                yield ContentChunk(
                    id=chunk_id,
                    content_type=ContentType.TEXT,
                    text=title,
                    location="Table of Contents",
                    metadata={"toc_index": i, "level": level},
                )
                chunk_count += 1

        # 2. Extract Page Content
        seen_image_xrefs: set[int] = set()  # avoid yielding duplicate xrefs

        for page_idx, page in enumerate(self._document):
            page_num = page_idx + 1

            # --- Text blocks ---
            try:
                blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)[
                    "blocks"
                ]
            except Exception as e:
                logger.warning(f"Failed to extract text on page {page_num}: {e}")
                blocks = []

            for block_idx, block in enumerate(blocks):
                if block["type"] == 0:  # Text block
                    try:
                        chunk = self._extract_text_block(
                            block, block_idx, page_idx, page_num
                        )
                        if chunk:
                            yield chunk
                            chunk_count += 1
                    except Exception as e:
                        logger.warning(
                            f"Failed to process text block {block_idx} on page {page_num}: {e}"
                        )

            # --- Images: use page.get_images() for reliable detection ---
            try:
                image_list = page.get_images(full=True)
            except Exception as e:
                logger.warning(f"Failed to list images on page {page_num}: {e}")
                image_list = []

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                if xref in seen_image_xrefs:
                    continue
                seen_image_xrefs.add(xref)

                try:
                    chunk = self._extract_image_by_xref(
                        page, xref, img_idx, page_idx, page_num
                    )
                    if chunk:
                        yield chunk
                        chunk_count += 1
                except Exception as e:
                    logger.debug(
                        f"Failed to extract image xref={xref} on page {page_num}: {e}"
                    )

            # --- Fallback: also scan dict-blocks for type-1 images ---
            # Some inline images don't appear in get_images()
            for block_idx, block in enumerate(blocks):
                if block["type"] == 1:
                    block_xref = block.get("xref", 0)
                    if block_xref and block_xref in seen_image_xrefs:
                        continue
                    try:
                        chunk = self._extract_image_block_fallback(
                            page, block, block_idx, page_idx, page_num
                        )
                        if chunk:
                            if block_xref:
                                seen_image_xrefs.add(block_xref)
                            yield chunk
                            chunk_count += 1
                    except Exception as e:
                        logger.debug(
                            f"Failed to extract image block {block_idx} on page {page_num}: {e}"
                        )

        logger.info(f"Extracted {chunk_count} content blocks from PDF")

    # --- helper: text block ---
    def _extract_text_block(
        self, block: dict, block_idx: int, page_idx: int, page_num: int
    ) -> ContentChunk | None:
        lines_text = []
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            lines_text.append(line_text)

        raw_text = "\n".join(lines_text)
        cleaned_text = raw_text.replace("-\n", "").replace("\n", " ").strip()

        if not cleaned_text:
            return None

        chunk_id = f"page{page_idx}_block{block_idx}"
        self._text_blocks[chunk_id] = {
            "page_idx": page_idx,
            "rect": fitz.Rect(block["bbox"]),
            "original_text": cleaned_text,
            "translated_text": None,
            "lines": block.get("lines", []),
        }

        return ContentChunk(
            id=chunk_id,
            content_type=ContentType.TEXT,
            text=cleaned_text,
            location=f"Page {page_num}",
            metadata={
                "page": page_idx,
                "block": block_idx,
                "bbox": block["bbox"],
            },
        )

    # --- helper: image via xref (preferred) ---
    def _extract_image_by_xref(
        self,
        page: fitz.Page,
        xref: int,
        img_idx: int,
        page_idx: int,
        page_num: int,
    ) -> ContentChunk | None:
        """Extract an image from the document by its xref number."""
        img_data = None
        img_rect = None

        # Try to get native image bytes
        try:
            img_info = self._document.extract_image(xref)
            if img_info:
                img_data = img_info["image"]
        except Exception as e:
            logger.debug(f"extract_image(xref={xref}) failed: {e}")

        # Determine bounding box on page
        try:
            img_rects = page.get_image_rects(xref)
            if img_rects:
                img_rect = img_rects[0]
        except Exception:
            pass

        # Fallback: render the area at 2x resolution
        if not img_data and img_rect and not img_rect.is_empty:
            try:
                pix = page.get_pixmap(clip=img_rect, matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
            except Exception as e:
                logger.debug(f"Pixmap fallback failed for xref={xref}: {e}")

        if not img_data:
            return None

        # Validate the image data is usable
        img_data = self._ensure_valid_image(img_data)
        if not img_data:
            return None

        # If we still have no rect, skip (can't place overlay)
        if not img_rect or img_rect.is_empty:
            logger.debug(f"No bounding rect for image xref={xref}, skipping")
            return None

        img_id = f"page{page_idx}_img{img_idx}"
        self._text_blocks[img_id] = {
            "page_idx": page_idx,
            "rect": img_rect,
            "type": "image",
            "xref": xref,
        }

        return ContentChunk(
            id=img_id,
            content_type=ContentType.IMAGE,
            image_data=img_data,
            location=f"Page {page_num}, Image",
            metadata={"page": page_idx},
        )

    # --- helper: fallback for dict-block images ---
    def _extract_image_block_fallback(
        self,
        page: fitz.Page,
        block: dict,
        block_idx: int,
        page_idx: int,
        page_num: int,
    ) -> ContentChunk | None:
        xref = block.get("xref", 0)
        img_data = None

        if xref and xref > 0:
            try:
                img_info = self._document.extract_image(xref)
                if img_info:
                    img_data = img_info["image"]
            except Exception:
                pass

        if not img_data:
            try:
                pix = page.get_pixmap(clip=block["bbox"], matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
            except Exception:
                return None

        img_data = self._ensure_valid_image(img_data)
        if not img_data:
            return None

        img_id = f"page{page_idx}_imgblk{block_idx}"
        self._text_blocks[img_id] = {
            "page_idx": page_idx,
            "rect": fitz.Rect(block["bbox"]),
            "type": "image",
            "xref": xref if xref and xref > 0 else 0,
        }

        return ContentChunk(
            id=img_id,
            content_type=ContentType.IMAGE,
            image_data=img_data,
            location=f"Page {page_num}, Image",
            metadata={"page": page_idx},
        )

    @staticmethod
    def _ensure_valid_image(data: bytes) -> bytes | None:
        """Validate image bytes and convert to PNG if needed.

        Returns PNG bytes on success, None on failure.
        Upscales tiny images so OCR has a better chance.
        """
        try:
            img = Image.open(io.BytesIO(data))
            img.verify()  # verify integrity
            # Re-open (verify consumes the file)
            img = Image.open(io.BytesIO(data)).convert("RGB")

            # Upscale very small images for OCR
            MIN_DIM = 150
            w, h = img.size
            if w < MIN_DIM or h < MIN_DIM:
                scale = max(MIN_DIM / w, MIN_DIM / h, 1.0)
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Translation application
    # ------------------------------------------------------------------

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
                if 0 <= idx < len(self._translated_toc):
                    self._translated_toc[idx][1] = str(translated_content)
            except (ValueError, IndexError):
                logger.warning(f"Invalid TOC chunk ID: {chunk_id}")
            return

        if chunk_id not in self._text_blocks:
            logger.warning(f"Block not found: {chunk_id}")
            return

        block_info = self._text_blocks[chunk_id]

        if block_info.get("type") == "image":
            if isinstance(translated_content, bytes):
                block_info["translated_image"] = translated_content
        else:
            block_info["translated_text"] = str(translated_content)

        logger.debug(f"Stored translation for {chunk_id}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, output_path: str | Path) -> None:
        """Save the modified PDF with text overlays and translated TOC."""
        self._validate_loaded()

        # 1. Apply TOC
        if self._translated_toc:
            try:
                self._document.set_toc(self._translated_toc)
                logger.info("Applied translated Table of Contents")
            except Exception as e:
                logger.error(f"Failed to set TOC: {e}")

        # 2. Group blocks by page for efficient processing
        pages_blocks: dict[int, list[tuple[str, dict]]] = {}
        for chunk_id, block_info in self._text_blocks.items():
            has_translation = block_info.get("translated_text") or block_info.get(
                "translated_image"
            )
            if has_translation:
                pidx = block_info["page_idx"]
                pages_blocks.setdefault(pidx, []).append((chunk_id, block_info))

        for page_idx in sorted(pages_blocks.keys()):
            page = self._document[page_idx]
            items = pages_blocks[page_idx]

            # --- Phase 1: redact original content for image overlays ---
            image_items = [(cid, bi) for cid, bi in items if bi.get("translated_image")]
            if image_items:
                for _cid, bi in image_items:
                    try:
                        page.add_redact_annot(bi["rect"])
                    except Exception as e:
                        logger.debug(f"Failed to add redact annotation: {e}")
                try:
                    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE)
                except Exception as e:
                    logger.warning(
                        f"apply_redactions failed on page {page_idx}, "
                        f"falling back to draw_rect: {e}"
                    )
                    # Fallback: cover with background colour
                    for _cid, bi in image_items:
                        bg = self._sample_background(page, bi["rect"])
                        page.draw_rect(bi["rect"], color=bg, fill=bg)

            # --- Phase 2: apply text overlays ---
            text_items = [(cid, bi) for cid, bi in items if bi.get("translated_text")]
            for _cid, bi in text_items:
                self._apply_text_overlay(bi)

            # --- Phase 3: insert translated images ---
            for _cid, bi in image_items:
                self._apply_image_overlay(bi)

        path = Path(output_path)
        self._document.save(path, garbage=3, deflate=True)
        logger.info(f"Saved PDF: {path.name}")

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    def _apply_text_overlay(self, block_info: dict) -> None:
        """Apply text overlay for a translated block with color preservation."""
        try:
            page = self._document[block_info["page_idx"]]
            rect = block_info["rect"]
            translated_text = block_info["translated_text"]

            if not translated_text:
                return

            # --- Analyse original font size and colour ---
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
                    orig_color_int = Counter(colors).most_common(1)[0][0]

            r_int = (orig_color_int >> 16) & 0xFF
            g_int = (orig_color_int >> 8) & 0xFF
            b_int = orig_color_int & 0xFF
            pdf_fg = (r_int / 255.0, g_int / 255.0, b_int / 255.0)

            # --- Sample background colour ---
            pdf_bg = self._sample_background(page, rect)

            # --- Cover original text ---
            page.draw_rect(rect, color=pdf_bg, fill=pdf_bg)

            # --- Font ---
            fontname = self._get_pdf_fontname()

            # --- Insert translated text with auto-fit ---
            inserted = False
            for size in range(int(avg_font_size) + 1, 4, -1):
                try:
                    rc = page.insert_textbox(
                        rect,
                        translated_text,
                        fontsize=size,
                        fontname=fontname,
                        color=pdf_fg,
                        align=0,
                    )
                    if rc >= 0:
                        inserted = True
                        break
                except Exception:
                    continue

            if not inserted:
                margin = 36
                expanded_rect = fitz.Rect(
                    rect.x0, rect.y0, rect.x1, page.rect.height - margin
                )
                for size in range(int(avg_font_size), 5, -1):
                    try:
                        rc = page.insert_textbox(
                            expanded_rect,
                            translated_text,
                            fontsize=size,
                            fontname=fontname,
                            color=pdf_fg,
                            align=0,
                        )
                        if rc >= 0:
                            inserted = True
                            break
                    except Exception:
                        continue

                if not inserted:
                    try:
                        page.insert_textbox(
                            expanded_rect,
                            translated_text,
                            fontsize=6,
                            fontname=fontname,
                            color=pdf_fg,
                            align=0,
                        )
                    except Exception as e:
                        logger.error(
                            f"Last-resort text insertion failed on page "
                            f"{block_info['page_idx']}: {e}"
                        )
                    logger.warning(
                        f"Forced overflow insertion for block on page "
                        f"{block_info['page_idx']}"
                    )

        except Exception as e:
            logger.error(f"Failed to apply text overlay: {e}")

    def _apply_image_overlay(self, block_info: dict) -> None:
        """Insert the translated image into the (already redacted) area."""
        try:
            page = self._document[block_info["page_idx"]]
            rect = block_info["rect"]
            image_data = block_info.get("translated_image")
            if not image_data:
                return

            page.insert_image(rect, stream=image_data)
        except Exception as e:
            logger.error(f"Failed to insert translated image overlay: {e}")

    def _get_pdf_fontname(self) -> str:
        """Return a PyMuPDF built-in fontname suitable for the target language."""
        lang = self.target_language.lower()
        if "chinese" in lang or "zh" in lang:
            return "china-t" if ("traditional" in lang or "hant" in lang) else "china-s"
        if "japanese" in lang or "ja" in lang:
            return "jpn"
        if "korean" in lang or "ko" in lang:
            return "kor"
        return "helv"

    @staticmethod
    def _sample_background(page: fitz.Page, rect: fitz.Rect) -> tuple:
        """Sample the page background colour around *rect*."""
        try:
            pix = page.get_pixmap(clip=rect)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            w, h = img.size
            if w < 2 or h < 2:
                return (1, 1, 1)
            samples = [
                img.getpixel((0, 0)),
                img.getpixel((w - 1, 0)),
                img.getpixel((0, h - 1)),
                img.getpixel((w - 1, h - 1)),
                img.getpixel((w // 2, 0)),
                img.getpixel((w // 2, h - 1)),
                img.getpixel((0, h // 2)),
                img.getpixel((w - 1, h // 2)),
            ]
            avg_bg = tuple(sum(s[i] for s in samples) // len(samples) for i in range(3))
            return tuple(v / 255.0 for v in avg_bg)
        except Exception as e:
            logger.debug(f"Background sampling failed: {e}")
            return (1, 1, 1)

    # ------------------------------------------------------------------
    # Counting
    # ------------------------------------------------------------------

    def get_total_chunks(self) -> int:
        """Count total content blocks efficiently."""
        self._validate_loaded()

        count = 0
        for page in self._document:
            try:
                blocks = page.get_text("dict")["blocks"]
            except Exception:
                continue
            for block in blocks:
                if block["type"] == 0:
                    block_text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            block_text += span.get("text", "")
                    if block_text.strip():
                        count += 1
                elif block["type"] == 1:
                    count += 1
        return count
