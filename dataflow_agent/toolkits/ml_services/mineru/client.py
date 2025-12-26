"""
MinerU service client.

Usage:
    client = MinerUClient("http://localhost:8001", api_key="secret")

    # Single image
    response = await client.parse_image("page.png")
    for block in response.blocks:
        print(block.type, block.text)

    # Batch images
    response = await client.parse_images(["page1.png", "page2.png"])

    # PDF (file must be accessible to server)
    response = await client.parse_pdf("/shared/paper.pdf")
"""

import base64
from pathlib import Path
from typing import List, Optional, Union

from ..common.client import MLServiceClient
from ..common.schemas import (
    ImageInput,
    MinerURequest,
    MinerUResponse,
    HealthResponse,
)


class MinerUClient(MLServiceClient):
    """Client for MinerU parsing service."""

    SERVICE_NAME = "mineru"

    async def parse_image(
        self,
        image_path: str,
        output_format: str = "blocks",
        request_id: Optional[str] = None,
    ) -> MinerUResponse:
        """
        Parse a single image.

        Args:
            image_path: Path to image file
            output_format: "blocks" or "markdown"
            request_id: Optional tracking ID

        Returns:
            MinerUResponse with blocks or markdown
        """
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        request = MinerURequest(
            image=ImageInput(
                base64_data=image_data,
                filename=Path(image_path).name,
            ),
            output_format=output_format,
            request_id=request_id,
        )

        return await self.post("/parse", request.model_dump(), MinerUResponse)

    async def parse_images(
        self,
        image_paths: List[str],
        output_format: str = "blocks",
        request_id: Optional[str] = None,
    ) -> MinerUResponse:
        """
        Parse multiple images (batch).

        Args:
            image_paths: List of image file paths
            output_format: "blocks" or "markdown"
            request_id: Optional tracking ID

        Returns:
            MinerUResponse with combined results
        """
        images = []
        for path in image_paths:
            with open(path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            images.append(ImageInput(
                base64_data=image_data,
                filename=Path(path).name,
            ))

        request = MinerURequest(
            images=images,
            output_format=output_format,
            request_id=request_id,
        )

        return await self.post("/parse/batch", request.model_dump(), MinerUResponse)

    async def parse_pdf(
        self,
        pdf_path: str,
        output_format: str = "markdown",
        request_id: Optional[str] = None,
    ) -> MinerUResponse:
        """
        Parse a PDF file.

        Note: The PDF path must be accessible to the server.
        For remote servers, upload the file first or use shared storage.

        Args:
            pdf_path: Path to PDF file (server-accessible)
            output_format: "blocks" or "markdown"
            request_id: Optional tracking ID

        Returns:
            MinerUResponse with parsed content
        """
        request = MinerURequest(
            pdf_path=pdf_path,
            output_format=output_format,
            request_id=request_id,
        )

        return await self.post("/parse/pdf", request.model_dump(), MinerUResponse)

    # Sync convenience methods

    def parse_image_sync(
        self,
        image_path: str,
        output_format: str = "blocks",
    ) -> MinerUResponse:
        """Synchronous image parsing."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.parse_image(image_path, output_format)
        )

    def parse_images_sync(
        self,
        image_paths: List[str],
        output_format: str = "blocks",
    ) -> MinerUResponse:
        """Synchronous batch image parsing."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.parse_images(image_paths, output_format)
        )
