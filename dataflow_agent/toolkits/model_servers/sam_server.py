from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
import numpy as np
import base64
import zlib
import os
import sys

# Add project root to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataflow_agent.toolkits.imtool.sam_tool import run_sam_auto, free_sam_model

try:
    import torch
except ImportError:
    torch = None

app = FastAPI(title="SAM Model Server")

# Check CUDA device on startup
@app.on_event("startup")
async def startup_event():
    print("SAM Server Startup Check:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    if torch and torch.cuda.is_available():
        print(f"Torch CUDA available: {torch.cuda.is_available()}")
        print(f"Current Device Count: {torch.cuda.device_count()}")
        print(f"Current Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA NOT AVAILABLE")

class SAMRequest(BaseModel):
    image_path: str
    checkpoint: str = "sam_b.pt"
    device: str = "cuda"

class SAMItemResponse(BaseModel):
    mask_b64: str
    mask_shape: List[int]
    bbox: List[float]
    score: Optional[float] = None
    area: int

class SAMResponse(BaseModel):
    items: List[SAMItemResponse]

@app.post("/predict", response_model=SAMResponse)
async def predict(req: SAMRequest):
    """
    Run SAM auto segmentation on the given image path.
    """
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=404, detail=f"Image path not found: {req.image_path}")

    try:
        # Force device to cuda:0 if it's generic cuda, to align with CUDA_VISIBLE_DEVICES env
        target_device = req.device
        if target_device == "cuda":
            target_device = "cuda:0"

        # 调用本地的 sam_tool 函数
        # 注意：这里会利用 sam_tool 内部的 caching 机制
        # 如果启动了多个 sam_server 进程，每个进程会维护自己的 cache
        items = run_sam_auto(
            image_path=req.image_path,
            checkpoint=req.checkpoint,
            device=target_device
        )
        
        # 序列化结果
        serialized_items = []
        for it in items:
            mask = it.get("mask")
            if mask is None:
                continue
                
            # Convert mask to base64
            # mask is numpy array (bool or uint8)
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask)
            
            # Use bool type for serialization consistency
            mask_bool = mask.astype(bool)
            mask_bytes = mask_bool.tobytes()
            # Compress using zlib to reduce payload size
            compressed_bytes = zlib.compress(mask_bytes)
            mask_b64 = base64.b64encode(compressed_bytes).decode('utf-8')
            
            serialized_items.append(SAMItemResponse(
                mask_b64=mask_b64,
                mask_shape=list(mask.shape),
                bbox=it.get("bbox", []),
                score=it.get("score"),
                area=it.get("area", 0)
            ))
            
        return SAMResponse(items=serialized_items)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Aggressive cleanup to prevent OOM
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.post("/free_model")
async def free_model(checkpoint: str = "sam_b.pt"):
    try:
        free_sam_model(checkpoint)
        return {"status": "ok", "message": f"Model {checkpoint} freed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
