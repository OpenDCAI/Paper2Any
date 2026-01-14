from __future__ import annotations

from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi_app.services.pricing_service import get_pricing_service
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

router = APIRouter()


@router.get("/pricing", response_model=Dict[str, Any])
async def get_pricing():
    """
    获取价格配置
    
    返回完整的价格配置 JSON，供前端显示使用
    """
    try:
        pricing_service = get_pricing_service()
        config = pricing_service.get_pricing_config()
        
        log.info("[Pricing API] Pricing config requested")
        return config
        
    except Exception as e:
        log.error(f"[Pricing API] Failed to get pricing config: {e}")
        raise HTTPException(
            status_code=500,
            detail="获取价格配置失败"
        )


@router.post("/pricing/reload")
async def reload_pricing():
    """
    重新加载价格配置（管理员接口）
    
    用于热更新价格配置，无需重启服务
    """
    try:
        pricing_service = get_pricing_service()
        pricing_service.reload_config()
        
        log.info("[Pricing API] Pricing config reloaded")
        return {
            "success": True,
            "message": "价格配置已重新加载"
        }
        
    except Exception as e:
        log.error(f"[Pricing API] Failed to reload pricing config: {e}")
        raise HTTPException(
            status_code=500,
            detail="重新加载价格配置失败"
        )
