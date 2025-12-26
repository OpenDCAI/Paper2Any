"""
PaddleOCR service client.

Usage:
    client = OCRClient("http://localhost:8003", api_key="secret")
    response = await client.recognize("image.png")
    for line in response.lines:
        print(line.text, line.confidence)
"""

import base64
from pathlib import Path
from typing import Optional

from ..common.client import MLServiceClient
from ..common.schemas import (
    ImageInput,
    OCRRequest,
    OCRResponse,
)


class OCRClient(MLServiceClient):
    """Client for OCR service."""

    SERVICE_NAME = "ocr"

    async def recognize(
        self,
        image_path: str,
        lang: str = "ch",
        drop_score: int = 30,
        with_layout: bool = True,
        request_id: Optional[str] = None,
    ) -> OCRResponse:
        """
        Perform OCR on an image.

        Args:
            image_path: Path to image file
            lang: Language code (ch=Chinese+English, en=English)
            drop_score: Confidence threshold 0-100
            with_layout: Include layout analysis
            request_id: Optional tracking ID

        Returns:
            OCRResponse with text lines and layout info
        """
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        request = OCRRequest(
            image=ImageInput(base64_data=image_data, filename=Path(image_path).name),
            lang=lang,
            drop_score=drop_score,
            with_layout=with_layout,
            request_id=request_id,
        )

        return await self.post("/ocr", request.model_dump(exclude_none=True), OCRResponse)

    async def recognize_simple(
        self,
        image_path: str,
        lang: str = "ch",
    ) -> str:
        """
        Simple OCR returning just the text.

        Args:
            image_path: Path to image file
            lang: Language code

        Returns:
            All recognized text as single string
        """
        response = await self.recognize(image_path, lang=lang, with_layout=False)
        return "\n".join(line.text for line in response.lines)

    # Sync convenience methods

    def recognize_sync(self, image_path: str, **kwargs) -> OCRResponse:
        """Synchronous OCR."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.recognize(image_path, **kwargs)
        )

    def recognize_simple_sync(self, image_path: str, lang: str = "ch") -> str:
        """Synchronous simple OCR."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.recognize_simple(image_path, lang)
        )
