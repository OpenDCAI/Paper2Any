"""
计费工具函数
"""

from __future__ import annotations

import json
from typing import Dict, Any

from fastapi import HTTPException

from dataflow_agent.logger import get_logger

log = get_logger(__name__)


def extract_page_count_from_param(kwargs: Dict[str, Any]) -> int:
    """
    从请求参数中提取 page_count
    
    Args:
        kwargs: 函数参数字典
        
    Returns:
        页数
        
    Raises:
        HTTPException: 页数无效时抛出 400 错误
    """
    page_count = kwargs.get("page_count", 1)
    
    if page_count <= 0:
        log.error(f"[Billing] Invalid page_count: {page_count}")
        raise HTTPException(
            status_code=400,
            detail=f"页数必须大于 0，当前值: {page_count}"
        )
    
    log.info(f"[Billing] Extracted page_count from param: {page_count}")
    return page_count


def extract_page_count_from_pagecontent(kwargs: Dict[str, Any]) -> int:
    """
    从 pagecontent JSON 字符串中提取页数
    
    Args:
        kwargs: 函数参数字典
        
    Returns:
        页数（编辑单页时返回 1）
        
    Raises:
        HTTPException: 页数无效时抛出 400 错误
    """
    # 检查是否是编辑单页模式
    get_down = kwargs.get("get_down", "false")
    if get_down.lower() == "true":
        log.info("[Billing] Edit single page mode, page_count=1")
        return 1
    
    # 从 pagecontent 解析页数
    pagecontent_str = kwargs.get("pagecontent")
    if not pagecontent_str:
        log.warning("[Billing] No pagecontent provided, defaulting to 1 page")
        return 1
    
    try:
        pagecontent = json.loads(pagecontent_str)
        if isinstance(pagecontent, list):
            page_count = len(pagecontent)
        else:
            page_count = 1
        
        if page_count <= 0:
            log.error(f"[Billing] Invalid page_count from pagecontent: {page_count}")
            raise HTTPException(
                status_code=400,
                detail=f"页数必须大于 0，当前值: {page_count}"
            )
        
        log.info(f"[Billing] Extracted page_count from pagecontent: {page_count}")
        return page_count
        
    except json.JSONDecodeError as e:
        log.error(f"[Billing] Failed to parse pagecontent JSON: {e}")
        raise HTTPException(
            status_code=400,
            detail="pagecontent 格式错误，无法解析 JSON"
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[Billing] Unexpected error extracting page_count: {e}")
        return 1


def extract_page_count_from_result(result: Dict[str, Any]) -> int:
    """
    从返回结果中提取页数
    
    Args:
        result: 接口返回结果
        
    Returns:
        页数
        
    Raises:
        HTTPException: 页数无效时抛出 400 错误
    """
    # 尝试从 pagecontent 字段提取
    if "pagecontent" in result:
        pagecontent = result["pagecontent"]
        if isinstance(pagecontent, list):
            page_count = len(pagecontent)
        elif isinstance(pagecontent, str):
            try:
                pagecontent_list = json.loads(pagecontent)
                page_count = len(pagecontent_list) if isinstance(pagecontent_list, list) else 1
            except:
                page_count = 1
        else:
            page_count = 1
    else:
        log.warning("[Billing] No pagecontent in result, defaulting to 1 page")
        page_count = 1
    
    if page_count <= 0:
        log.error(f"[Billing] Invalid page_count from result: {page_count}")
        raise HTTPException(
            status_code=400,
            detail=f"页数必须大于 0，当前值: {page_count}"
        )
    
    log.info(f"[Billing] Extracted page_count from result: {page_count}")
    return page_count


def calculate_hybrid_billing(
    kwargs: Dict[str, Any],
    base_price: int,
    ai_per_page_price: int
) -> int:
    """
    计算混合计费：基础价格 + AI 增强按页计费
    
    Args:
        kwargs: 函数参数字典
        base_price: 基础价格
        ai_per_page_price: AI 增强单页价格
        
    Returns:
        总费用
        
    Raises:
        HTTPException: 页数无效时抛出 400 错误
    """
    # 基础费用
    total = base_price
    
    # 检查是否启用 AI 增强
    use_ai_edit = kwargs.get("use_ai_edit", False)
    
    if use_ai_edit:
        # 获取页数
        page_count = kwargs.get("page_count", 8)
        
        if page_count <= 0:
            log.error(f"[Billing] Invalid page_count for AI edit: {page_count}")
            raise HTTPException(
                status_code=400,
                detail=f"页数必须大于 0，当前值: {page_count}"
            )
        
        # AI 增强费用 = 单页价格 × 页数
        ai_cost = ai_per_page_price * page_count
        total += ai_cost
        
        log.info(f"[Billing] Hybrid billing: base={base_price}, ai_per_page={ai_per_page_price}, page_count={page_count}, ai_cost={ai_cost}, total={total}")
    else:
        log.info(f"[Billing] Hybrid billing: base={base_price}, no AI enhancement, total={total}")
    
    return total
