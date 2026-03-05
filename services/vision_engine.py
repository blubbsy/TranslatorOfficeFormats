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
from typing import List, Tuple, Optional, Callable, Any

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
# EasyOCR language code mapping
# Maps common locale / ISO-639 codes to the identifiers EasyOCR actually accepts.
# ---------------------------------------------------------------------------

_EASYOCR_LANG_MAP: dict[str, str] = {
    # Chinese
    "zh": "ch_sim",
    "zh-cn": "ch_sim",
    "zh_cn": "ch_sim",
    "zh-hans": "ch_sim",
    "chinese": "ch_sim",
    "ch_sim": "ch_sim",
    "zh-tw": "ch_tra",
    "zh_tw": "ch_tra",
    "zh-hant": "ch_tra",
    "ch_tra": "ch_tra",
    # Japanese / Korean
    "ja": "ja",
    "jp": "ja",
    "japanese": "ja",
    "ko": "ko",
    "kr": "ko",
    "korean": "ko",
    # European
    "en": "en",
    "english": "en",
    "de": "de",
    "german": "de",
    "fr": "fr",
    "french": "fr",
    "es": "es",
    "spanish": "es",
    "it": "it",
    "italian": "it",
    "pt": "pt",
    "portuguese": "pt",
    "ru": "ru",
    "russian": "ru",
    "ar": "ar",
    "arabic": "ar",
    "nl": "nl",
    "dutch": "nl",
    "pl": "pl",
    "polish": "pl",
    "tr": "tr",
    "turkish": "tr",
    "vi": "vi",
    "vietnamese": "vi",
    "th": "th",
    "thai": "th",
    "hi": "hi",
    "hindi": "hi",
}


def _normalise_ocr_languages(raw_codes: list[str]) -> tuple[list[str], bool]:
    """Convert user-supplied language codes to valid EasyOCR identifiers.

    Matches EasyOCR's compatibility constraints: some languages (like Chinese
    or Japanese) can only be combined with English. If such a language is
    detected, only it and English are returned.

    Returns:
        tuple: (list of valid identifiers, bool indicating if a restriction was applied)
    """
    seen: set[str] = set()
    mapped_codes: list[str] = []

    for code in raw_codes:
        mapped = _EASYOCR_LANG_MAP.get(code.lower().strip(), code.lower().strip())
        if mapped not in seen:
            seen.add(mapped)
            mapped_codes.append(mapped)

    # EasyOCR constraint: Some languages can ONLY be paired with English.
    # If we have one of these, we must strip all other non-English languages.
    restricted_langs = {"ch_sim", "ch_tra", "ja", "ko", "th", "hi"}
    has_restricted = any(lang in restricted_langs for lang in mapped_codes)

    if has_restricted:
        # Keep only English + the FIRST restricted language found
        main_restricted = next(
            lang for lang in mapped_codes if lang in restricted_langs
        )
        result = [main_restricted]

        # Check if we are actually stripping anything other than 'en'
        original_non_en = [
            l for l in mapped_codes if l != "en" and l != main_restricted
        ]
        was_restricted = len(original_non_en) > 0

        if "en" in mapped_codes or "en" not in result:
            result.append("en")
        return result, was_restricted

    return mapped_codes, False


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

    # OCR tuning constants
    MIN_OCR_DIM = 300  # Minimum dimension (px) before upscaling for OCR
    OCR_CONFIDENCE_THRESHOLD = 0.15  # Minimum confidence to accept a detection
    CONTRAST_ENHANCE = 1.5  # Contrast enhancement factor for OCR pre-processing
    SHARPNESS_ENHANCE = 1.5  # Sharpness enhancement factor for OCR pre-processing

    # VLM constants
    VLM_MAX_DIM = 1024  # Maximum image dimension sent to VLM
    VLM_JPEG_QUALITY = 85  # JPEG quality for VLM encoding

    # Reader recovery
    MAX_READER_RETRIES = 3  # How many times to retry EasyOCR init before giving up

    def __init__(self, use_gpu: bool = False):
        from settings import settings
        import threading
        from pathlib import Path

        raw_langs = [lang.strip() for lang in settings.ocr_source_languages.split(",")]
        self.languages, self.was_restricted = _normalise_ocr_languages(raw_langs)
        self.target_language = settings.target_language
        self.use_gpu = use_gpu
        self._reader = None  # Lazy-loaded
        self._reader_init_failures = 0  # Track consecutive init failures
        self._init_lock = threading.Lock()

        # Local model storage setup
        self._model_dir = str(Path(__file__).parent.parent / ".easyocr_models")

        logger.info(
            f"VisionEngine initialised for languages: {self.languages}, "
            f"Target: {self.target_language}"
        )

    # ------------------------------------------------------------------
    # EasyOCR reader (lazy with graceful fallback)
    # ------------------------------------------------------------------

    @property
    def reader(self):
        """Lazy-load EasyOCR reader. Returns None if unavailable. Thread-safe."""
        if self._reader is not None:
            return self._reader

        with self._init_lock:
            # Re-check after acquiring the lock in case another thread loaded it
            if self._reader is not None:
                return self._reader

            if self._reader_init_failures >= self.MAX_READER_RETRIES:
                return None
            try:
                import easyocr  # type: ignore # noqa: deferred import to avoid crash if not installed

                logger.info("Loading EasyOCR model (this may take a moment)...")
                self._reader = easyocr.Reader(
                    self.languages,
                    gpu=self.use_gpu,
                    model_storage_directory=self._model_dir,
                    verbose=False,
                )
                self._reader_init_failures = 0
                logger.info("EasyOCR model loaded")
                return self._reader
            except Exception as e:
                self._reader_init_failures += 1
            remaining = self.MAX_READER_RETRIES - self._reader_init_failures
            logger.warning(
                f"EasyOCR could not be initialised: {e} "
                f"({remaining} retries remaining)"
            )
            return None

    # ------------------------------------------------------------------
    # OCR detection
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_nearby_regions(regions: List[TextRegion]) -> List[TextRegion]:
        """
        Merge text regions that are on the same line and close together.
        This helps with formulas and consistent text that OCR splits up.
        """
        if not regions:
            return regions

        # Sort by y-coordinate (top to bottom)
        sorted_regions = sorted(regions, key=lambda r: r.bbox[1])

        merged: List[TextRegion] = []
        current_line: List[TextRegion] = [sorted_regions[0]]

        for region in sorted_regions[1:]:
            # Check if this region is on approximately the same line as the current line
            current_y = sum(r.bbox[1] + r.bbox[3] for r in current_line) / (
                2 * len(current_line)
            )
            region_y = (region.bbox[1] + region.bbox[3]) / 2
            y_threshold = (
                max(r.bbox[3] - r.bbox[1] for r in current_line) * 0.3
            )  # 30% of height - only merge text on exact same baseline

            if abs(region_y - current_y) <= y_threshold:
                # Same line - add to current line group
                current_line.append(region)
            else:
                # Different line - process current line and start new one
                merged.extend(VisionEngine._process_line(current_line))
                current_line = [region]

        # Process the last line
        if current_line:
            merged.extend(VisionEngine._process_line(current_line))

        return merged

    @staticmethod
    def _process_line(line_regions: List[TextRegion]) -> List[TextRegion]:
        """
        Process regions on the same line: merge if they're close together horizontally.
        """
        if not line_regions:
            return []

        # Sort by x-coordinate (left to right)
        sorted_line = sorted(line_regions, key=lambda r: r.bbox[0])

        merged: List[TextRegion] = []
        current = sorted_line[0]

        for next_region in sorted_line[1:]:
            # Calculate horizontal gap
            gap = next_region.bbox[0] - current.bbox[2]
            avg_height = (
                current.bbox[3]
                - current.bbox[1]
                + next_region.bbox[3]
                - next_region.bbox[1]
            ) / 2

            # Merge if gap is small (less than 0.8x average character width estimate)
            # Estimate character width as height * 0.6 (typical aspect ratio)
            # Only merge text that's very close (like formula components)
            char_width_estimate = avg_height * 0.6

            if gap < char_width_estimate * 0.8:
                # Merge regions
                current = TextRegion(
                    bbox=(
                        min(current.bbox[0], next_region.bbox[0]),
                        min(current.bbox[1], next_region.bbox[1]),
                        max(current.bbox[2], next_region.bbox[2]),
                        max(current.bbox[3], next_region.bbox[3]),
                    ),
                    text=current.text + " " + next_region.text,
                    confidence=min(current.confidence, next_region.confidence),
                    angle=(current.angle + next_region.angle) / 2,
                )
            else:
                # Gap too large - keep as separate regions
                merged.append(current)
                current = next_region

        # Add the last region
        merged.append(current)
        return merged

    def detect_text(self, image_data: bytes) -> List[TextRegion]:
        """Detect text regions in an image with pre-processing for better accuracy."""
        reader = self.reader
        if reader is None:
            logger.warning("EasyOCR unavailable – skipping text detection")
            return []

        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            # Upscale small images for better OCR accuracy
            w, h = image.size
            if w < self.MIN_OCR_DIM or h < self.MIN_OCR_DIM:
                scale = max(self.MIN_OCR_DIM / w, self.MIN_OCR_DIM / h, 1.0)
                new_size = (int(w * scale), int(h * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"Upscaled image for OCR: ({w},{h}) -> {new_size}")

            # Enhance contrast + sharpness
            image_processed = ImageEnhance.Contrast(image).enhance(
                self.CONTRAST_ENHANCE
            )
            image_processed = ImageEnhance.Sharpness(image_processed).enhance(
                self.SHARPNESS_ENHANCE
            )

            image_np = np.array(image_processed)

            # Run OCR with paragraph=False to detect individual text elements
            # Custom merging logic will combine text on the same line (for formulas)
            # while keeping separate lines apart (for tables of contents, etc.)
            try:
                results = reader.readtext(image_np, paragraph=False)
            except Exception as e:
                logger.error(f"readtext failed: {e}")
                return []

            regions: List[TextRegion] = []
            for result in results:
                try:
                    # EasyOCR can return different formats
                    if len(result) == 3:
                        bbox_points, text, confidence = result
                    elif len(result) == 2:
                        bbox_points, text = result
                        confidence = 1.0  # No confidence provided
                    else:
                        logger.debug(f"Unexpected result format: {result}")
                        continue

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

                    if text.strip() and confidence > self.OCR_CONFIDENCE_THRESHOLD:
                        regions.append(
                            TextRegion(
                                bbox=bbox,
                                text=text.strip(),
                                confidence=confidence,
                                angle=angle,
                            )
                        )
                except Exception as e:
                    logger.debug(f"Failed to process OCR result: {e}")

            logger.info(f"Detected {len(regions)} text regions (before merging)")

            # Merge nearby regions on the same line to keep formulas/text together
            regions = self._merge_nearby_regions(regions)

            logger.info(f"Final {len(regions)} text regions after merging")
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
                        angle=region.angle,
                    )
                except Exception:
                    default_font: Any = ImageFont.load_default()
                    font = default_font

                # Check font type for mypy/type-safety in older Pillow versions
                if not isinstance(font, ImageFont.FreeTypeFont):
                    # Fallback to a concrete FreeTypeFont if possible,
                    # though modern Pillow load_default() usually is one.
                    try:
                        from utils.text_utils import get_font_for_language

                        font = get_font_for_language(lang, 12)
                    except Exception:
                        pass  # Keep the default

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

            MAX_DIM = self.VLM_MAX_DIM
            if max(orig_size) > MAX_DIM:
                scale = MAX_DIM / max(orig_size)
                new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image for VLM: {orig_size} -> {new_size}")
            else:
                scale = 1.0

            # 2. Encode to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=self.VLM_JPEG_QUALITY)
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
                    # Cast to fixed size tuple for mypy/type-safety
                    bbox_tuple = (coords[0], coords[1], coords[2], coords[3])
                    regions.append(
                        TextRegion(
                            bbox=bbox_tuple,
                            text="",
                            confidence=1.0,
                            translated_text=str(text),
                        )
                    )
            except (ValueError, TypeError, IndexError):
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
