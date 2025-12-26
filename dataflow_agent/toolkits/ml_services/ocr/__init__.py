"""
PaddleOCR service.

Client and server for OCR text recognition.
"""

from .client import OCRClient
from .server import create_ocr_server

__all__ = ["OCRClient", "create_ocr_server"]
