"""
Image Utilities

Helper functions for image processing and validation.
"""

from pathlib import Path
from typing import Optional
from PIL import Image


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}


def validate_image(path: str) -> tuple[bool, Optional[str]]:
    """
    Validate that file exists and is a supported image format.
    
    Args:
        path: Path to image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    file_path = Path(path)
    
    if not file_path.exists():
        return False, f"File not found: {path}"
    
    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {file_path.suffix}. Supported: {SUPPORTED_FORMATS}"
    
    try:
        with Image.open(path) as img:
            img.verify()
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {e}"


def get_image_info(path: str) -> dict:
    """
    Get image metadata.
    
    Args:
        path: Path to image
        
    Returns:
        Dict with width, height, format, mode
    """
    with Image.open(path) as img:
        return {
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "mode": img.mode
        }
