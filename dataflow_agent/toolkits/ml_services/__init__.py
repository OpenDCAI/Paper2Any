"""
ML Services - Client-Server Architecture for ML Models.

This package provides a clean separation between ML model servers and clients,
enabling:
- Testing without heavy ML dependencies (use mock server)
- Distributed deployment (models on GPU servers, clients anywhere)
- Simple API key authentication

## Services

| Service | Port | Description |
|---------|------|-------------|
| MinerU  | 8001 | PDF/image parsing |
| SAM     | 8002 | Image segmentation |
| OCR     | 8003 | Text recognition |
| RMBG    | 8004 | Background removal |
| Mock    | 8000 | All services mocked |

## Quick Start

### Run Mock Server (for testing)

```bash
# All services on one port
uvicorn dataflow_agent.toolkits.ml_services.mock_server:app --port 8000

# With API key
ML_SERVICE_API_KEY=secret uvicorn ... --port 8000
```

### Run Real Servers (requires ML dependencies)

```bash
# MinerU
uvicorn dataflow_agent.toolkits.ml_services.mineru.server:app --port 8001

# SAM/YOLO
uvicorn dataflow_agent.toolkits.ml_services.sam.server:app --port 8002

# OCR
uvicorn dataflow_agent.toolkits.ml_services.ocr.server:app --port 8003

# RMBG
uvicorn dataflow_agent.toolkits.ml_services.rmbg.server:app --port 8004
```

### Use Clients

```python
from dataflow_agent.toolkits.ml_services import MinerUClient, SAMClient, OCRClient, RMBGClient

# Connect to mock server
client = MinerUClient("http://localhost:8000", api_key="secret")
response = await client.parse_image("document.png")
print(response.blocks)

# Or real server
client = SAMClient("http://localhost:8002", api_key="secret")
response = await client.segment_auto("image.png", top_k=5)
```

## Authentication

All servers use simple API key authentication via the `X-API-Key` header.

- Set `ML_SERVICE_API_KEY` env var on server
- Pass `api_key` to client constructor
- If no key is set, authentication is disabled (dev mode)
"""

# Re-export clients for convenience
from .mineru import MinerUClient
from .sam import SAMClient, YOLOClient
from .ocr import OCRClient
from .rmbg import RMBGClient

# Re-export common utilities
from .common import MLServiceClient, MLServiceClientPool
from .common.auth import APIKeyMiddleware, verify_api_key

# Re-export schemas
from .common.schemas import (
    ImageInput,
    MinerURequest,
    MinerUResponse,
    MinerUBlock,
    SAMRequest,
    SAMResponse,
    YOLORequest,
    YOLOResponse,
    SegmentationItem,
    OCRRequest,
    OCRResponse,
    OCRLine,
    RMBGRequest,
    RMBGResponse,
    HealthResponse,
)

__all__ = [
    # Clients
    "MinerUClient",
    "SAMClient",
    "YOLOClient",
    "OCRClient",
    "RMBGClient",
    "MLServiceClient",
    "MLServiceClientPool",
    # Auth
    "APIKeyMiddleware",
    "verify_api_key",
    # Schemas
    "ImageInput",
    "MinerURequest",
    "MinerUResponse",
    "MinerUBlock",
    "SAMRequest",
    "SAMResponse",
    "YOLORequest",
    "YOLOResponse",
    "SegmentationItem",
    "OCRRequest",
    "OCRResponse",
    "OCRLine",
    "RMBGRequest",
    "RMBGResponse",
    "HealthResponse",
]
