from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any, Tuple, Union
import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataflow_agent.toolkits.imtool.ppt_tool import paddle_ocr_page_with_layout

app = FastAPI(title="OCR Model Server")

class OCRRequest(BaseModel):
    image_path: str

class OCRLine(BaseModel):
    bbox: List[float]  # [x1, y1, x2, y2]
    text: str
    conf: float

class OCRResponse(BaseModel):
    image_size: Tuple[int, int]
    lines: List[OCRLine]
    body_h_px: Optional[float] = None
    bg_color: Optional[Tuple[int, int, int]] = None

@app.post("/predict", response_model=OCRResponse)
async def predict(req: OCRRequest):
    """
    Run PaddleOCR on the given image path and analyze layout.
    """
    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=404, detail=f"Image path not found: {req.image_path}")

    try:
        # 调用本地 ppt_tool 函数
        # 注意：PADDLE_OCR 是全局初始化的，所以服务启动时加载一次
        result = paddle_ocr_page_with_layout(req.image_path)
        
        # result structure:
        # {
        #     "image_size": (w, h),
        #     "lines": [(bbox, text, conf), ...],
        #     "body_h_px": float/None,
        #     "bg_color": (r,g,b)/None,
        # }

        # Transform lines format to match Pydantic model
        # from tuple to dict/object
        transformed_lines = []
        for line in result.get("lines", []):
            bbox, text, conf = line
            transformed_lines.append(OCRLine(
                bbox=bbox,
                text=text,
                conf=conf
            ))
            
        return OCRResponse(
            image_size=result.get("image_size", (0, 0)),
            lines=transformed_lines,
            body_h_px=result.get("body_h_px"),
            bg_color=result.get("bg_color")
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
