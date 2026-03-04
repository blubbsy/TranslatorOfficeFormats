"""
Text utilities for fitting text to bounding boxes and color calculations.
"""

import math
from typing import Tuple, List, Optional

from PIL import Image, ImageDraw, ImageFont


def get_font_for_language(language: str, font_size: int) -> ImageFont.FreeTypeFont:
    """
    Get a suitable font for the given language.
    """
    lang_lower = language.lower()
    
    # Common font paths for different OS
    font_candidates = []
    
    if "chinese" in lang_lower or "zh" in lang_lower:
        if "traditional" in lang_lower or "hant" in lang_lower:
            font_candidates = [
                "msjh.ttc",      # Windows: Microsoft JhengHei
                "mingliu.ttc",   # Windows: MingLiU
                "DroidSansFallback.ttf"
            ]
        else:
            font_candidates = [
                "msyh.ttc",      # Windows: Microsoft YaHei
                "simsun.ttc",    # Windows: SimSun
                "simhei.ttf",    # Windows: SimHei
                "wqy-microhei.ttc", # Linux: WenQuanYi Micro Hei
                "DroidSansFallback.ttf" # Common fallback
            ]
    elif "japanese" in lang_lower or "ja" in lang_lower:
        font_candidates = ["msgothic.ttc", "meiryo.ttc", "msmincho.ttc", "DroidSansFallback.ttf"]
    elif "korean" in lang_lower or "ko" in lang_lower:
        font_candidates = ["malgun.ttf", "batang.ttc", "gulim.ttc", "DroidSansFallback.ttf"]
    
    # Try candidates
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
            
    # Default fallbacks
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            return ImageFont.load_default()


def calculate_luminance(rgb: Tuple[int, int, int]) -> float:
    """
    Calculate relative luminance of an RGB color.
    
    Uses the formula from WCAG 2.0:
    L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    
    Args:
        rgb: Tuple of (R, G, B) values in range 0-255
        
    Returns:
        Luminance value between 0 (black) and 1 (white)
    """
    r, g, b = [v / 255.0 for v in rgb]
    
    # Apply gamma correction
    r = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
    
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def sample_region_background(region: Image.Image) -> Tuple[int, int, int]:
    """
    Sample average background colour from corners and edge midpoints of a region.

    Args:
        region: An RGB PIL Image to sample.

    Returns:
        Average RGB tuple (0-255 per channel).
    """
    pixels = region.load()
    w, h = region.size
    if w < 2 or h < 2:
        return pixels[0, 0][:3] if w >= 1 and h >= 1 else (255, 255, 255)

    samples = [
        pixels[0, 0],
        pixels[w - 1, 0],
        pixels[0, h - 1],
        pixels[w - 1, h - 1],
        pixels[w // 2, 0],
        pixels[w // 2, h - 1],
        pixels[0, h // 2],
        pixels[w - 1, h // 2],
    ]
    return tuple(sum(s[i] for s in samples) // len(samples) for i in range(3))


def get_contrasting_colors(
    image: Image.Image,
    bbox: Tuple[int, int, int, int]
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int]]:
    """
    Determine optimal background and text colors based on image region.
    """
    x1, y1, x2, y2 = bbox
    
    # Ensure bbox is within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)
    
    if x2 <= x1 or y2 <= y1:
        return (255, 255, 255, 255), (0, 0, 0)

    # Crop the region
    region = image.crop((x1, y1, x2, y2)).convert("RGB")
    
    # 1. Detect Background Color
    avg_bg = sample_region_background(region)
    
    # 2. Detect Original Text Color
    # We look for the color that is MOST different from the background
    small_region = region.resize((20, 20), Image.Resampling.NEAREST)
    colors = small_region.getcolors(small_region.size[0] * small_region.size[1])
    
    text_color = (0, 0, 0) # Default
    max_diff = -1
    
    if colors:
        for count, color in colors:
            # Simple Euclidean distance in RGB
            diff = sum((color[i] - avg_bg[i])**2 for i in range(3))**0.5
            if diff > max_diff:
                max_diff = diff
                text_color = color
                
    # If the detected text color is too similar to background, fallback to contrast
    if max_diff < 40:
        lum = calculate_luminance(avg_bg)
        text_color = (255, 255, 255) if lum < 0.5 else (0, 0, 0)
    
    # Return background with full opacity for a solid cover
    return (*avg_bg, 255), text_color


def fit_text_to_box(
    text: str,
    box_width: int,
    box_height: int,
    max_font_size: int = 24,
    min_font_size: int = 8,
    language: str = "English",
    angle: float = 0.0,
) -> Tuple[ImageFont.FreeTypeFont, int]:
    """
    Find the largest font size that fits text within a bounding box.

    When *angle* is non-zero the available width/height are adjusted to
    account for the rotated text footprint inside the axis-aligned box.
    """
    # When text is rotated, the axis-aligned bbox of the rotated text is
    # larger than the text itself.  We compute the effective dimensions
    # that are available along the text's own axes.
    if abs(angle) > 0.5:
        rad = math.radians(abs(angle))
        cos_a = abs(math.cos(rad))
        sin_a = abs(math.sin(rad))
        # Solve for the maximum text dimensions (tw, th) such that the
        # rotated bounding box fits inside the axis-aligned box:
        #   tw * cos(a) + th * sin(a) <= box_width
        #   tw * sin(a) + th * cos(a) <= box_height
        # Solving this 2×2 system gives a denominator of cos²(a) - sin²(a),
        # which equals cos(2a).  At 45° this is zero, so we fall back.
        denom = cos_a * cos_a - sin_a * sin_a
        if abs(denom) > 1e-6:
            eff_w = abs((box_width * cos_a - box_height * sin_a) / denom)
            eff_h = abs((box_height * cos_a - box_width * sin_a) / denom)
        else:
            # ~45° – both axes contribute equally
            diag = (box_width + box_height) / 2.0
            eff_w = eff_h = diag / math.sqrt(2)
        eff_w = max(int(eff_w), 1)
        eff_h = max(int(eff_h), 1)
    else:
        eff_w, eff_h = box_width, box_height

    font_size = max_font_size
    
    while font_size >= min_font_size:
        font = get_font_for_language(language, font_size)
        
        # Calculate text size
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        if text_width <= eff_w and text_height <= eff_h:
            return font, font_size
        
        font_size -= 1
    
    return get_font_for_language(language, min_font_size), min_font_size


def draw_rotated_text(
    image: Image.Image,
    text: str,
    center: Tuple[int, int],
    angle: float,
    font: ImageFont.FreeTypeFont,
    fill: Tuple[int, int, int],
    bg_fill: Optional[Tuple[int, int, int, int]] = None,
    original_bbox: Optional[Tuple[int, int, int, int]] = None
):
    """
    Draw text rotated around its center.
    If original_bbox is provided, it clears that area first with bg_fill.
    Handles edge-case angles (near 0°, 90°, etc.) and clips to image bounds.
    """
    draw = ImageDraw.Draw(image)

    # 1. Clear original area if requested
    if original_bbox and bg_fill:
        draw.rectangle(original_bbox, fill=bg_fill)

    # 2. Measure text
    dummy = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    if w <= 0 or h <= 0:
        return

    # 3. Render text on transparent canvas
    pad = 4
    txt_img = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    txt_draw = ImageDraw.Draw(txt_img)
    # Offset by -bbox origin to compensate for fonts with non-zero bearing
    # (textbbox may return negative x0/y0 for descenders or left-bearing).
    txt_draw.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=fill)

    # 4. Rotate (skip if angle is negligible for crispness)
    if abs(angle) > 0.5:
        rotated = txt_img.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    else:
        rotated = txt_img

    # 5. Composite at centre, clipped to image bounds
    rw, rh = rotated.size
    tx = int(center[0] - rw / 2)
    ty = int(center[1] - rh / 2)

    # Clip to image dimensions
    tx = max(0, min(tx, image.width - rw))
    ty = max(0, min(ty, image.height - rh))

    image.alpha_composite(rotated, (tx, ty))


def wrap_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int
) -> list[str]:
    """
    Wrap text to fit within a maximum width.
    
    Args:
        text: Text to wrap
        font: Font to use for measurement
        max_width: Maximum width in pixels
        
    Returns:
        List of text lines
    """
    words = text.split()
    lines = []
    current_line = []
    
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    
    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        width = bbox[2] - bbox[0]
        
        if width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines if lines else [text]


def split_text_smart(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks that respect a maximum character limit.
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    
    # Try splitting by paragraph
    paragraphs = text.split("\n\n")
    if len(paragraphs) > 1:
        current_chunk = []
        current_len = 0
        
        for p in paragraphs:
            if len(p) > max_chars:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                chunks.extend(split_text_smart(p, max_chars))
            elif current_len + len(p) + 2 > max_chars:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [p]
                current_len = len(p)
            else:
                current_chunk.append(p)
                current_len += len(p) + 2
                
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return chunks

    # Try splitting by line
    lines = text.split("\n")
    if len(lines) > 1:
        current_chunk = []
        current_len = 0
        for l in lines:
            if len(l) > max_chars:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                chunks.extend(split_text_smart(l, max_chars))
            elif current_len + len(l) + 1 > max_chars:
                chunks.append("\n".join(current_chunk))
                current_chunk = [l]
                current_len = len(l)
            else:
                current_chunk.append(l)
                current_len += len(l) + 1
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks

    # Hard split by characters
    for i in range(0, len(text), max_chars):
        chunks.append(text[i : i + max_chars])
    
    return chunks


def format_size(size_bytes: int) -> str:
    """Format bytes into a human-readable string (KB, MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
