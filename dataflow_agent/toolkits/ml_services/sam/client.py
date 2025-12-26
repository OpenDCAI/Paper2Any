"""
SAM/YOLO service clients.

Usage:
    # SAM client
    client = SAMClient("http://localhost:8002", api_key="secret")
    response = await client.segment_auto("image.png")
    for item in response.items:
        print(item.bbox, item.score)

    # YOLO client
    client = YOLOClient("http://localhost:8002", api_key="secret")
    response = await client.segment("image.png")
"""

import base64
from pathlib import Path
from typing import List, Optional

from ..common.client import MLServiceClient
from ..common.schemas import (
    ImageInput,
    SAMRequest,
    SAMResponse,
    YOLORequest,
    YOLOResponse,
)


class SAMClient(MLServiceClient):
    """Client for SAM segmentation service."""

    SERVICE_NAME = "sam"

    async def segment_auto(
        self,
        image_path: str,
        min_area: Optional[int] = None,
        min_score: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> SAMResponse:
        """
        Auto segmentation - finds all objects.

        Args:
            image_path: Path to image file
            min_area: Filter masks smaller than this
            min_score: Filter masks with lower confidence
            nms_threshold: NMS IoU threshold for deduplication
            top_k: Return only top K results
            request_id: Optional tracking ID

        Returns:
            SAMResponse with segmentation items
        """
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        request = SAMRequest(
            image=ImageInput(base64_data=image_data, filename=Path(image_path).name),
            mode="auto",
            min_area=min_area,
            min_score=min_score,
            nms_threshold=nms_threshold,
            top_k=top_k,
            request_id=request_id,
        )

        return await self.post("/segment", request.model_dump(exclude_none=True), SAMResponse)

    async def segment_boxes(
        self,
        image_path: str,
        boxes: List[List[float]],
        request_id: Optional[str] = None,
    ) -> SAMResponse:
        """
        Box-guided segmentation.

        Args:
            image_path: Path to image file
            boxes: List of bounding boxes [[x1,y1,x2,y2], ...]
            request_id: Optional tracking ID

        Returns:
            SAMResponse with one mask per box
        """
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        request = SAMRequest(
            image=ImageInput(base64_data=image_data, filename=Path(image_path).name),
            mode="box",
            boxes=boxes,
            request_id=request_id,
        )

        return await self.post("/segment", request.model_dump(exclude_none=True), SAMResponse)

    async def segment_points(
        self,
        image_path: str,
        points: List[List[float]],
        point_labels: List[int],
        request_id: Optional[str] = None,
    ) -> SAMResponse:
        """
        Point-guided segmentation.

        Args:
            image_path: Path to image file
            points: List of points [[x, y], ...]
            point_labels: Labels for each point (1=foreground, 0=background)
            request_id: Optional tracking ID

        Returns:
            SAMResponse with segmentation result
        """
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        request = SAMRequest(
            image=ImageInput(base64_data=image_data, filename=Path(image_path).name),
            mode="point",
            points=points,
            point_labels=point_labels,
            request_id=request_id,
        )

        return await self.post("/segment", request.model_dump(exclude_none=True), SAMResponse)

    # Sync convenience methods

    def segment_auto_sync(self, image_path: str, **kwargs) -> SAMResponse:
        """Synchronous auto segmentation."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.segment_auto(image_path, **kwargs)
        )


class YOLOClient(MLServiceClient):
    """Client for YOLO segmentation service."""

    SERVICE_NAME = "yolo"

    async def segment(
        self,
        image_path: str,
        weights: str = "yolov8n-seg.pt",
        conf_threshold: float = 0.25,
        request_id: Optional[str] = None,
    ) -> YOLOResponse:
        """
        YOLO instance segmentation.

        Args:
            image_path: Path to image file
            weights: YOLO weights file name
            conf_threshold: Confidence threshold
            request_id: Optional tracking ID

        Returns:
            YOLOResponse with labeled segmentation items
        """
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        request = YOLORequest(
            image=ImageInput(base64_data=image_data, filename=Path(image_path).name),
            weights=weights,
            conf_threshold=conf_threshold,
            request_id=request_id,
        )

        return await self.post("/segment/yolo", request.model_dump(exclude_none=True), YOLOResponse)

    def segment_sync(self, image_path: str, **kwargs) -> YOLOResponse:
        """Synchronous segmentation."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.segment(image_path, **kwargs)
        )
