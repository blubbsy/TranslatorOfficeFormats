"""
Vision engine for OCR text detection and overlay generation.
"""

import io
import logging
import base64
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional

import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.text_utils import fit_text_to_box, get_contrasting_colors, wrap_text

logger = logging.getLogger("OfficeTranslator.VisionEngine")


@dataclass
class TextRegion:
    """Represents a detected text region in an image."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    text: str
    confidence: float
    translated_text: Optional[str] = None


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
        self.use_gpu = use_gpu
        self._reader: Optional[easyocr.Reader] = None
        
        logger.info(f"VisionEngine initialized for languages: {self.languages}")
    
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
        Detect text regions in an image.
        
        Args:
            image_data: Image as bytes
            
        Returns:
            List of TextRegion objects with bounding boxes and text
        """
        try:
            # Convert bytes to numpy array
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image.convert("RGB"))
            
            # Run OCR
            results = self.reader.readtext(image_np)
            
            regions = []
            for bbox_points, text, confidence in results:
                # Convert polygon to rectangular bbox
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]
                bbox = (
                    int(min(x_coords)),
                    int(min(y_coords)),
                    int(max(x_coords)),
                    int(max(y_coords)),
                )
                
                if text.strip():  # Skip empty detections
                    regions.append(TextRegion(
                        bbox=bbox,
                        text=text,
                        confidence=confidence,
                    ))
            
            logger.info(f"Detected {len(regions)} text regions")
            return regions
            
        except Exception as e:
            logger.error(f"OCR detection failed: {e}")
            return []
    
    def create_overlay(
        self,
        image_data: bytes,
        regions: List[TextRegion],
        smart_colors: bool = True,
    ) -> bytes:
        """
        Create image with translated text overlays.
        
        Args:
            image_data: Original image as bytes
            regions: List of TextRegion objects with translated_text set
            smart_colors: Use dynamic colors based on background luminance
            
        Returns:
            Modified image as bytes (PNG format)
        """
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGBA")
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            for region in regions:
                if not region.translated_text:
                    continue
                
                x1, y1, x2, y2 = region.bbox
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Add padding
                padding = 4
                x1 -= padding
                y1 -= padding
                x2 += padding
                y2 += padding
                
                # Determine colors
                if smart_colors:
                    bg_color, text_color = get_contrasting_colors(image, region.bbox)
                else:
                    bg_color = (0, 0, 0, 180)
                    text_color = (255, 255, 255)
                
                # Draw background rectangle
                draw.rectangle([x1, y1, x2, y2], fill=bg_color)
                
                # Fit and draw text
                font, font_size = fit_text_to_box(
                    region.translated_text,
                    box_width,
                    box_height,
                    max_font_size=min(24, box_height - 4),
                    min_font_size=8,
                )
                
                # Wrap text if needed
                lines = wrap_text(region.translated_text, font, box_width - 4)
                
                # Calculate text position (centered)
                total_text_height = len(lines) * (font_size + 2)
                y_offset = y1 + padding + (box_height - total_text_height) // 2
                
                for line in lines:
                    # Get line width for centering
                    dummy = Image.new("RGB", (1, 1))
                    dummy_draw = ImageDraw.Draw(dummy)
                    line_bbox = dummy_draw.textbbox((0, 0), line, font=font)
                    line_width = line_bbox[2] - line_bbox[0]
                    x_offset = x1 + padding + (box_width - line_width) // 2
                    
                    draw.text(
                        (x_offset, y_offset),
                        line,
                        font=font,
                        fill=text_color,
                    )
                    y_offset += font_size + 2
            
            # Composite overlay onto original image
            result = Image.alpha_composite(image, overlay)
            
            # Convert to bytes
            output = io.BytesIO()
            result.save(output, format="PNG")
            output.seek(0)
            
            logger.info(f"Created overlay with {len([r for r in regions if r.translated_text])} translations")
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Overlay creation failed: {e}")
            return image_data  # Return original on failure
    
    def process_image(
        self,
        image_data: bytes,
        translate_func,
        smart_colors: bool = True,
    ) -> Tuple[bytes, List[TextRegion]]:
        """
        Full pipeline: detect text, translate, and create overlay.
        
        Args:
            image_data: Input image as bytes
            translate_func: Function that takes text and returns translated text
            smart_colors: Use dynamic background colors
            
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
        result = self.create_overlay(image_data, regions, smart_colors)
        
        return result, regions

    def process_image_vlm(
        self,
        image_data: bytes,
        translate_image_func,
        smart_colors: bool = True,
    ) -> Tuple[bytes, List[TextRegion]]:
        """
        VLM pipeline: Send image to VLM, get bboxes+text, create overlay.
        
        Args:
            image_data: Input image as bytes
            translate_image_func: Function taking base64 image and returning JSON string
            smart_colors: Use dynamic background colors
        """
        try:
            # Encode image
            b64_img = base64.b64encode(image_data).decode('utf-8')
            
            # Call VLM
            json_response = translate_image_func(b64_img)
            
            # Parse response
            # Cleanup potential markdown code blocks
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
                    # Ensure bbox is valid tuple of ints
                    try:
                        valid_bbox = tuple(map(int, bbox))
                        if len(valid_bbox) == 4:
                            regions.append(TextRegion(
                                bbox=valid_bbox,
                                text="", # Original text unknown/irrelevant for overlay
                                confidence=1.0,
                                translated_text=text
                            ))
                    except (ValueError, TypeError):
                        continue
            
            logger.info(f"VLM returned {len(regions)} regions")
            
            # Create overlay
            result = self.create_overlay(image_data, regions, smart_colors)
            return result, regions
            
        except Exception as e:
            logger.error(f"VLM processing failed: {e}")
            return image_data, []
