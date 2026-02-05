"""
Text utilities for fitting text to bounding boxes and color calculations.
"""

from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


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


def get_contrasting_colors(
    image: Image.Image,
    bbox: Tuple[int, int, int, int]
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int]]:
    """
    Determine optimal background and text colors based on image region.
    
    Args:
        image: PIL Image to analyze
        bbox: Bounding box (x1, y1, x2, y2) of the region
        
    Returns:
        Tuple of (background_rgba, text_rgb) for maximum contrast
    """
    x1, y1, x2, y2 = bbox
    
    # Crop the region and calculate average color
    region = image.crop(bbox)
    region_small = region.resize((1, 1), Image.Resampling.LANCZOS)
    avg_color = region_small.getpixel((0, 0))
    
    if len(avg_color) == 4:
        avg_color = avg_color[:3]  # Remove alpha if present
    
    luminance = calculate_luminance(avg_color)
    
    # Choose contrasting colors
    if luminance > 0.5:
        # Light background -> dark overlay with white text
        bg_color = (0, 0, 0, 180)  # More opaque black
        text_color = (255, 255, 255)
    else:
        # Dark background -> light overlay with dark text
        bg_color = (255, 255, 255, 200)  # More opaque white
        text_color = (0, 0, 0)
    
    return bg_color, text_color


def fit_text_to_box(
    text: str,
    box_width: int,
    box_height: int,
    max_font_size: int = 24,
    min_font_size: int = 8,
    font_path: str = None
) -> Tuple[ImageFont.FreeTypeFont, int]:
    """
    Find the largest font size that fits text within a bounding box.
    
    Args:
        text: Text to fit
        box_width: Maximum width in pixels
        box_height: Maximum height in pixels
        max_font_size: Starting font size
        min_font_size: Minimum acceptable font size
        font_path: Path to TTF font file (uses default if None)
        
    Returns:
        Tuple of (font object, actual font size used)
    """
    font_size = max_font_size
    
    while font_size >= min_font_size:
        try:
            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                # Try to use a system font, fall back to default
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except OSError:
                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                    except OSError:
                        font = ImageFont.load_default()
                        # Default font doesn't support size, return immediately
                        return font, 10
        except OSError:
            font = ImageFont.load_default()
            return font, 10
        
        # Calculate text size
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        if text_width <= box_width and text_height <= box_height:
            return font, font_size
        
        font_size -= 1
    
    # Return minimum size font
    try:
        if font_path:
            font = ImageFont.truetype(font_path, min_font_size)
        else:
            font = ImageFont.truetype("arial.ttf", min_font_size)
    except OSError:
        font = ImageFont.load_default()
    
    return font, min_font_size


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


def format_size(size_bytes: int) -> str:
    """Format bytes into a human-readable string (KB, MB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
