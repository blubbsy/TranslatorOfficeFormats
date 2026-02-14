"""
Vision engine for OCR text detection and overlay generation.
"""

import io
import logging
import base64
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.text_utils import fit_text_to_box, get_contrasting_colors, wrap_text

logger = logging.getLogger("OfficeTranslator.VisionEngine")


import math

@dataclass
class TextRegion:
    """Represents a detected text region in an image."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    text: str
    confidence: float
    translated_text: Optional[str] = None
    angle: float = 0.0  # Rotation angle in degrees


class VisionEngine:
    """
    Vision engine for OCR and text overlay generation.
    Uses EasyOCR for text detection and Pillow for overlay creation.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize vision engine.
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        from settings import settings
        self.languages = [lang.strip() for lang in settings.ocr_source_languages.split(',')]
        self.target_language = settings.target_language
        self.use_gpu = use_gpu
        self._reader: Optional[easyocr.Reader] = None
        
        logger.info(f"VisionEngine initialized for languages: {self.languages}, Target: {self.target_language}")
    
    @property
    def reader(self) -> easyocr.Reader:
        """Lazy-load EasyOCR reader."""
        if self._reader is None:
            logger.info("Loading EasyOCR model (this may take a moment)...")
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False,
            )
            logger.info("EasyOCR model loaded")
        return self._reader
    
    def detect_text(self, image_data: bytes) -> List[TextRegion]:
        """
        Detect text regions in an image with pre-processing for better accuracy.
        """
        try:
            from PIL import ImageOps, ImageEnhance
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Enhance image for better OCR: Increase contrast and sharpness
            enhancer = ImageEnhance.Contrast(image)
            image_processed = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Sharpness(image_processed)
            image_processed = enhancer.enhance(1.5)
            
            # Convert to numpy array for EasyOCR
            image_np = np.array(image_processed)
            
            # Run OCR
            results = self.reader.readtext(image_np)
            
            regions = []
            for bbox_points, text, confidence in results:
                # Convert polygon to rectangular bbox and calculate angle
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]
                
                # Calculate angle based on the top edge (p0 to p1)
                dx = bbox_points[1][0] - bbox_points[0][0]
                dy = bbox_points[1][1] - bbox_points[0][1]
                angle = -math.degrees(math.atan2(dy, dx))
                
                bbox = (
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords)),
                )
                
                if text.strip() and confidence > 0.2: # Filter low confidence noise
                    regions.append(TextRegion(
                        bbox=bbox,
                        text=text,
                        confidence=confidence,
                        angle=angle
                    ))
            
            logger.info(f"Detected {len(regions)} text regions in image")
            return regions
            
        except Exception as e:
            logger.error(f"OCR detection failed: {e}")
            return []
    
    def create_overlay(
        self,
        image_data: bytes,
        regions: List[TextRegion],
        smart_colors: bool = True,
        target_language: Optional[str] = None,
    ) -> bytes:
        """
        Create image with translated text overlays.
        
        Args:
            image_data: Original image as bytes
            regions: List of TextRegion objects with translated_text set
            smart_colors: Use dynamic colors based on background luminance
            target_language: Target language for font selection
            
        Returns:
            Modified image as bytes (PNG format)
        """
        try:
            from utils.text_utils import fit_text_to_box, get_contrasting_colors, wrap_text, draw_rotated_text
            
            image = Image.open(io.BytesIO(image_data)).convert("RGBA")
            # We will draw directly onto the image using alpha_composite in draw_rotated_text
            
            lang = target_language or self.target_language
            
            for region in regions:
                if not region.translated_text:
                    continue
                
                x1, y1, x2, y2 = region.bbox
                box_width = x2 - x1
                box_height = y2 - y1
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Determine colors
                if smart_colors:
                    bg_color, text_color = get_contrasting_colors(image, region.bbox)
                else:
                    bg_color = (0, 0, 0, 180)
                    text_color = (255, 255, 255)
                
                # Fit font
                font, font_size = fit_text_to_box(
                    region.translated_text,
                    box_width,
                    box_height,
                    max_font_size=min(32, max(8, box_height - 2)),
                    min_font_size=6,
                    language=lang
                )
                
                # Draw text (handling rotation)
                draw_rotated_text(
                    image,
                    region.translated_text,
                    center,
                    region.angle,
                    font,
                    text_color,
                    bg_fill=bg_color,
                    original_bbox=region.bbox
                )
            
            # Convert to bytes
            output = io.BytesIO()
            image.save(output, format="PNG")
            output.seek(0)
            
            logger.info(f"Created overlay with {len([r for r in regions if r.translated_text])} translations")
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Overlay creation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return image_data  # Return original on failure
    
    def process_image(
        self,
        image_data: bytes,
        translate_func: Callable[[str], str],
        smart_colors: bool = True,
        target_language: Optional[str] = None,
    ) -> Tuple[bytes, List[TextRegion]]:
        """
        Full pipeline: detect text, translate, and create overlay.
        
        Args:
            image_data: Input image as bytes
            translate_func: Function that takes text and returns translated text
            smart_colors: Use dynamic background colors
            target_language: Target language
            
        Returns:
            Tuple of (processed image bytes, list of text regions)
        """
        # Detect text
        regions = self.detect_text(image_data)
        
        if not regions:
            logger.debug("No text detected in image")
            return image_data, regions
        
        # Translate each region
        for region in regions:
            try:
                region.translated_text = translate_func(region.text)
            except Exception as e:
                logger.warning(f"Translation failed for '{region.text[:20]}...': {e}")
                region.translated_text = region.text  # Keep original
        
        # Create overlay
        result = self.create_overlay(image_data, regions, smart_colors, target_language=target_language)
        
        return result, regions

    def process_image_vlm(
        self,
        image_data: bytes,
        translate_image_func: Callable[[str], str],
        smart_colors: bool = True,
        target_language: Optional[str] = None,
    ) -> Tuple[bytes, List[TextRegion]]:
        """
        VLM pipeline with automatic resizing for API compatibility.
        """
        try:
            # 1. Load and optionally resize image
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            orig_size = image.size
            
            # Max dimension for VLM (e.g., 1024)
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
            b64_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # 3. Call VLM
            json_response = translate_image_func(b64_img)
            
            # 4. Parse response
            clean_json = json_response.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.startswith("```"):
                clean_json = clean_json[3:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
                
            data = json.loads(clean_json)
            regions = []
            
            for item in data.get("regions", []):
                bbox = item.get("bbox")
                text = item.get("text")
                if bbox and text:
                    try:
                        # Scale bbox back to original size if resized
                        valid_bbox = tuple(map(lambda x: int(int(x) / scale), bbox))
                        if len(valid_bbox) == 4:
                            regions.append(TextRegion(
                                bbox=valid_bbox,
                                text="", 
                                confidence=1.0,
                                translated_text=text
                            ))
                    except (ValueError, TypeError):
                        continue
            
            logger.info(f"VLM returned {len(regions)} regions")
            
            # 5. Create overlay on ORIGINAL image data
            result = self.create_overlay(image_data, regions, smart_colors, target_language=target_language)
            return result, regions
            
        except Exception as e:
            logger.error(f"VLM processing failed: {e}")
            return image_data, []
