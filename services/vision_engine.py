"""
Vision engine for OCR text detection and overlay generation.

Supports two backends:
- **OCR** (default): EasyOCR for text detection → LLM for translation → Pillow overlay
- **VLM** (experimental): Send image to a Vision-Language Model for detection + translation
"""

import io
import json
import logging
import math
import base64
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps

from utils.text_utils import (
    fit_text_to_box,
    get_contrasting_colors,
    wrap_text,
    draw_rotated_text,
)

logger = logging.getLogger("OfficeTranslator.VisionEngine")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class TextRegion:
    """Represents a detected text region in an image."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    text: str
    confidence: float
    translated_text: Optional[str] = None
    angle: float = 0.0  # Rotation angle in degrees


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class VisionEngine:
    """
    Vision engine for OCR and text overlay generation.
    Uses EasyOCR for text detection and Pillow for overlay creation.
    """

    def __init__(self, use_gpu: bool = False):
        from settings import settings

        self.languages = [
            lang.strip() for lang in settings.ocr_source_languages.split(",")
        ]
        self.target_language = settings.target_language
        self.use_gpu = use_gpu
        self._reader = None  # Lazy-loaded
        self._reader_failed = False  # Set True if EasyOCR can't initialise

        logger.info(
            f"VisionEngine initialised for languages: {self.languages}, "
            f"Target: {self.target_language}"
        )

    # ------------------------------------------------------------------
    # EasyOCR reader (lazy with graceful fallback)
    # ------------------------------------------------------------------

    @property
    def reader(self):
        """Lazy-load EasyOCR reader. Returns None if unavailable."""
        if self._reader is not None:
            return self._reader
        if self._reader_failed:
            return None
        try:
            import easyocr  # noqa: deferred import to avoid crash if not installed

            logger.info("Loading EasyOCR model (this may take a moment)...")
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False,
            )
            logger.info("EasyOCR model loaded")
            return self._reader
        except Exception as e:
            logger.warning(f"EasyOCR could not be initialised: {e}")
            self._reader_failed = True
            return None

    # ------------------------------------------------------------------
    # OCR detection
    # ------------------------------------------------------------------

    def detect_text(self, image_data: bytes) -> List[TextRegion]:
        """Detect text regions in an image with pre-processing for better accuracy."""
        reader = self.reader
        if reader is None:
            logger.warning("EasyOCR unavailable – skipping text detection")
            return []

        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            # Upscale small images for better OCR accuracy
            MIN_OCR_DIM = 300
            w, h = image.size
            if w < MIN_OCR_DIM or h < MIN_OCR_DIM:
                scale = max(MIN_OCR_DIM / w, MIN_OCR_DIM / h, 1.0)
                new_size = (int(w * scale), int(h * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"Upscaled image for OCR: ({w},{h}) -> {new_size}")

            # Enhance contrast + sharpness
            image_processed = ImageEnhance.Contrast(image).enhance(1.5)
            image_processed = ImageEnhance.Sharpness(image_processed).enhance(1.5)

            image_np = np.array(image_processed)

            # Run OCR – paragraph=True groups text better for translation
            try:
                results = reader.readtext(image_np, paragraph=False)
            except Exception as e:
                logger.warning(f"readtext failed, retrying with defaults: {e}")
                results = reader.readtext(image_np)

            regions: List[TextRegion] = []
            for bbox_points, text, confidence in results:
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]

                dx = bbox_points[1][0] - bbox_points[0][0]
                dy = bbox_points[1][1] - bbox_points[0][1]
                angle = -math.degrees(math.atan2(dy, dx))

                bbox = (
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords)),
                )

                if text.strip() and confidence > 0.15:
                    regions.append(
                        TextRegion(
                            bbox=bbox,
                            text=text.strip(),
                            confidence=confidence,
                            angle=angle,
                        )
                    )

            logger.info(f"Detected {len(regions)} text regions in image")
            return regions

        except Exception as e:
            logger.error(f"OCR detection failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Overlay creation
    # ------------------------------------------------------------------

    def create_overlay(
        self,
        image_data: bytes,
        regions: List[TextRegion],
        smart_colors: bool = True,
        target_language: Optional[str] = None,
    ) -> bytes:
        """Create image with translated text overlays."""
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGBA")
            lang = target_language or self.target_language

            for region in regions:
                if not region.translated_text:
                    continue

                x1, y1, x2, y2 = region.bbox
                box_width = max(x2 - x1, 1)
                box_height = max(y2 - y1, 1)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Colours
                if smart_colors:
                    try:
                        bg_color, text_color = get_contrasting_colors(
                            image, region.bbox
                        )
                    except Exception:
                        bg_color = (0, 0, 0, 180)
                        text_color = (255, 255, 255)
                else:
                    bg_color = (0, 0, 0, 180)
                    text_color = (255, 255, 255)

                # Font
                try:
                    font, _font_size = fit_text_to_box(
                        region.translated_text,
                        box_width,
                        box_height,
                        max_font_size=min(32, max(8, box_height - 2)),
                        min_font_size=6,
                        language=lang,
                    )
                except Exception:
                    font = ImageFont.load_default()

                # Draw
                try:
                    draw_rotated_text(
                        image,
                        region.translated_text,
                        center,
                        region.angle,
                        font,
                        text_color,
                        bg_fill=bg_color,
                        original_bbox=region.bbox,
                    )
                except Exception as e:
                    logger.debug(f"draw_rotated_text failed, using simple draw: {e}")
                    draw = ImageDraw.Draw(image)
                    draw.rectangle(region.bbox, fill=bg_color)
                    draw.text(
                        (x1 + 2, y1 + 2),
                        region.translated_text,
                        font=font,
                        fill=text_color,
                    )

            output = io.BytesIO()
            image.save(output, format="PNG")
            output.seek(0)

            logger.info(
                f"Created overlay with "
                f"{sum(1 for r in regions if r.translated_text)} translations"
            )
            return output.getvalue()

        except Exception as e:
            logger.error(f"Overlay creation failed: {e}")
            logger.error(traceback.format_exc())
            return image_data  # Return original on failure

    # ------------------------------------------------------------------
    # OCR pipeline
    # ------------------------------------------------------------------

    def process_image(
        self,
        image_data: bytes,
        translate_func: Callable[[str], str],
        smart_colors: bool = True,
        target_language: Optional[str] = None,
    ) -> Tuple[bytes, List[TextRegion]]:
        """Full pipeline: detect text → translate → create overlay."""
        regions = self.detect_text(image_data)

        if not regions:
            logger.debug("No text detected in image")
            return image_data, regions

        # Translate each region
        for region in regions:
            try:
                region.translated_text = translate_func(region.text)
            except Exception as e:
                logger.warning(f"Translation failed for '{region.text[:30]}...': {e}")
                region.translated_text = region.text  # Keep original as fallback

        result = self.create_overlay(
            image_data, regions, smart_colors, target_language=target_language
        )
        return result, regions

    # ------------------------------------------------------------------
    # VLM pipeline
    # ------------------------------------------------------------------

    def process_image_vlm(
        self,
        image_data: bytes,
        translate_image_func: Callable[[str], str],
        smart_colors: bool = True,
        target_language: Optional[str] = None,
    ) -> Tuple[bytes, List[TextRegion]]:
        """VLM pipeline with automatic resizing for API compatibility."""
        try:
            # 1. Load and optionally resize
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            orig_size = image.size

            MAX_DIM = 1024
            if max(orig_size) > MAX_DIM:
                scale = MAX_DIM / max(orig_size)
                new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image for VLM: {orig_size} -> {new_size}")
            else:
                scale = 1.0

            # 2. Encode to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            b64_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # 3. Call VLM
            json_response = translate_image_func(b64_img)

            # 4. Parse response
            regions = self._parse_vlm_response(json_response, scale)

            if not regions:
                logger.info("VLM returned no regions, falling back to OCR")
                return self._vlm_fallback_to_ocr(
                    image_data, smart_colors, target_language
                )

            logger.info(f"VLM returned {len(regions)} regions")

            # 5. Create overlay on ORIGINAL image data
            result = self.create_overlay(
                image_data, regions, smart_colors, target_language=target_language
            )
            return result, regions

        except Exception as e:
            logger.error(f"VLM processing failed: {e}")
            logger.info("Falling back to OCR pipeline")
            return self._vlm_fallback_to_ocr(image_data, smart_colors, target_language)

    def _vlm_fallback_to_ocr(
        self,
        image_data: bytes,
        smart_colors: bool,
        target_language: Optional[str],
    ) -> Tuple[bytes, List[TextRegion]]:
        """When VLM fails, try OCR as a fallback. Return original on total failure."""
        try:
            if self.reader is not None:
                # We need a translate_func; just return original text (no LLM here)
                regions = self.detect_text(image_data)
                if regions:
                    # Mark each region with its original text as "translated" text
                    # (the text won't actually be translated, but at least detection works)
                    for r in regions:
                        r.translated_text = r.text
                    result = self.create_overlay(
                        image_data, regions, smart_colors, target_language
                    )
                    return result, regions
        except Exception as e:
            logger.debug(f"OCR fallback also failed: {e}")
        return image_data, []

    @staticmethod
    def _parse_vlm_response(raw_response: str, scale: float) -> List[TextRegion]:
        """Robustly parse the JSON response from a VLM."""
        if not raw_response or not raw_response.strip():
            return []

        clean = raw_response.strip()

        # Strip markdown code fences (various formats)
        for prefix in ("```json", "```JSON", "```"):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
                break
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        # Try to find JSON object / array boundaries
        data = None
        for attempt_str in (clean, _find_json_object(clean)):
            if not attempt_str:
                continue
            try:
                data = json.loads(attempt_str)
                break
            except json.JSONDecodeError:
                continue

        if data is None:
            logger.warning("Could not parse VLM JSON response")
            logger.debug(f"Raw VLM response: {raw_response[:500]}")
            return []

        # Accept {"regions": [...]} or just [...]
        items: list = []
        if isinstance(data, dict):
            items = data.get("regions", data.get("results", []))
        elif isinstance(data, list):
            items = data

        regions: List[TextRegion] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox") or item.get("bounding_box") or item.get("box")
            text = (
                item.get("text")
                or item.get("translation")
                or item.get("translated_text")
            )
            if not bbox or not text:
                continue
            try:
                coords = [int(int(c) / scale) for c in bbox]
                if len(coords) == 4:
                    regions.append(
                        TextRegion(
                            bbox=tuple(coords),
                            text="",
                            confidence=1.0,
                            translated_text=str(text),
                        )
                    )
            except (ValueError, TypeError):
                continue

        return regions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_json_object(text: str) -> Optional[str]:
    """Try to extract the first JSON object or array from *text*."""
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None
