"""
RMBG-2.0 background removal client.

Usage:
    client = RMBGClient("http://localhost:8004", api_key="secret")
    response = await client.remove_background("image.png")

    # Save result
    import base64
    with open("output.png", "wb") as f:
        f.write(base64.b64decode(response.image_base64))
"""

import base64
from pathlib import Path
from typing import Optional

from ..common.client import MLServiceClient
from ..common.schemas import (
    ImageInput,
    RMBGRequest,
    RMBGResponse,
)


class RMBGClient(MLServiceClient):
    """Client for background removal service."""

    SERVICE_NAME = "rmbg"

    async def remove_background(
        self,
        image_path: str,
        request_id: Optional[str] = None,
    ) -> RMBGResponse:
        """
        Remove background from an image.

        Args:
            image_path: Path to image file
            request_id: Optional tracking ID

        Returns:
            RMBGResponse with base64 encoded PNG with alpha channel
        """
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        request = RMBGRequest(
            image=ImageInput(base64_data=image_data, filename=Path(image_path).name),
            request_id=request_id,
        )

        return await self.post("/remove", request.model_dump(exclude_none=True), RMBGResponse)

    async def remove_background_to_file(
        self,
        image_path: str,
        output_path: str,
    ) -> str:
        """
        Remove background and save to file.

        Args:
            image_path: Input image path
            output_path: Output PNG path

        Returns:
            Output file path
        """
        response = await self.remove_background(image_path)

        if not response.success:
            raise RuntimeError(response.error)

        with open(output_path, "wb") as f:
            f.write(base64.b64decode(response.image_base64))

        return output_path

    # Sync convenience methods

    def remove_background_sync(self, image_path: str) -> RMBGResponse:
        """Synchronous background removal."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.remove_background(image_path)
        )

    def remove_background_to_file_sync(
        self,
        image_path: str,
        output_path: str,
    ) -> str:
        """Synchronous background removal to file."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.remove_background_to_file(image_path, output_path)
        )
