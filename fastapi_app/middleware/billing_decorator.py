from __future__ import annotations

import os
from functools import wraps
from typing import Callable, Optional
from fastapi import Request, HTTPException
from dataflow_agent.logger import get_logger
from fastapi_app.services.billing_service import BillingService
from fastapi_app.services.idempotency_service import get_idempotency_service
from fastapi_app.services.pricing_service import PricingService

log = get_logger(__name__)


def with_billing(service: str, endpoint: str):
    """
    计费装饰器：在请求成功后扣费（固定价格）
    
    Args:
        service: 服务名（如 "paper2figure"）
        endpoint: 端点名（如 "generate_json"）
        
    Usage:
        @with_billing("paper2figure", "generate_json")
        async def my_endpoint(...):
            ...
    
    注意：
    - 用户刷新页面或退出登录不影响扣费
    - 只要业务逻辑执行成功，就会扣费
    - 幂等性保护防止 5 分钟内的重复提交
    - 价格从 config/pricing.json 读取
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 提取 request 对象
            request: Request = kwargs.get("request") or next(
                (arg for arg in args if isinstance(arg, Request)), None
            )
            
            if not request:
                raise RuntimeError("Cannot find Request object in function arguments")
            
            billing_service = BillingService()
            idempotency_service = get_idempotency_service()
            pricing_service = PricingService()
            
            # 检查是否启用计费
            if not billing_service.enabled:
                log.info(f"[Billing] Billing disabled, executing {func.__name__} without charge")
                return await func(*args, **kwargs)
            
            # 获取用户凭证并生成 bizNo
            try:
                access_key, client_name = billing_service.get_user_credentials(request)
                biz_no = billing_service.generate_biz_no(client_name)
            except HTTPException:
                # 如果获取凭证失败，直接抛出异常（401）
                raise
            
            # 幂等性检查（防止 5 分钟内重复提交）
            if idempotency_service.is_duplicate(biz_no):
                log.warning(f"[Billing] Duplicate request detected: biz_no={biz_no}")
                raise HTTPException(
                    status_code=409,
                    detail="请求重复，请勿在短时间内重复提交"
                )
            
            # 标记为处理中
            idempotency_service.mark_processed(biz_no)
            
            try:
                # 执行业务逻辑
                result = await func(*args, **kwargs)
                
                # 从 pricing.json 获取价格
                pricing_config = pricing_service.get_pricing_config()
                event_value = pricing_config.get("pricing", {}).get(service, {}).get(endpoint, {}).get("price", 1)
                
                if event_value == 1:
                    log.warning(f"[Billing] Price not found for {service}.{endpoint}, using default value 1")
                
                log.info(f"[Billing] Fixed billing: service={service}, endpoint={endpoint}, price={event_value}")
                
                await billing_service.charge(
                    request=request,
                    event_value=event_value,
                    biz_no=biz_no
                )
                
                return result
                
            except HTTPException:
                # 业务失败（HTTP 异常），移除幂等性标记，不扣费
                idempotency_service.remove(biz_no)
                raise
            except Exception as e:
                # 其他异常，移除幂等性标记，不扣费
                idempotency_service.remove(biz_no)
                log.error(f"[Billing] Unexpected error in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator



def with_dynamic_billing(
    service: str,
    endpoint: str,
    page_count_extractor: Callable
):
    """
    动态计费装饰器：根据页数计算费用
    
    Args:
        service: 服务名（如 "paper2ppt"）
        endpoint: 端点名（如 "pagecontent"）
        page_count_extractor: 提取页数的函数
            - 接收 (kwargs, result) 参数
            - 返回页数（int）
        
    Usage:
        @with_dynamic_billing(
            "paper2ppt",
            "pagecontent",
            lambda kwargs, result: kwargs.get("page_count", 1)
        )
        async def my_endpoint(...):
            ...
    
    注意：
    - 实际扣费 = 单页价格 × 页数
    - 页数必须 > 0，否则抛出 400 错误
    - 编辑单页时按 1 页计费
    - 价格从 config/pricing.json 读取
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 提取 request 对象
            request: Request = kwargs.get("request") or next(
                (arg for arg in args if isinstance(arg, Request)), None
            )
            
            if not request:
                raise RuntimeError("Cannot find Request object in function arguments")
            
            billing_service = BillingService()
            idempotency_service = get_idempotency_service()
            pricing_service = PricingService()
            
            # 检查是否启用计费
            if not billing_service.enabled:
                log.info(f"[Billing] Billing disabled, executing {func.__name__} without charge")
                return await func(*args, **kwargs)
            
            # 获取用户凭证并生成 bizNo
            try:
                access_key, client_name = billing_service.get_user_credentials(request)
                biz_no = billing_service.generate_biz_no(client_name)
            except HTTPException:
                # 如果获取凭证失败，直接抛出异常（401）
                raise
            
            # 幂等性检查（防止 5 分钟内重复提交）
            if idempotency_service.is_duplicate(biz_no):
                log.warning(f"[Billing] Duplicate request detected: biz_no={biz_no}")
                raise HTTPException(
                    status_code=409,
                    detail="请求重复，请勿在短时间内重复提交"
                )
            
            # 标记为处理中
            idempotency_service.mark_processed(biz_no)
            
            try:
                # 执行业务逻辑
                result = await func(*args, **kwargs)
                
                # 提取页数
                try:
                    page_count = page_count_extractor(kwargs, result)
                    if page_count <= 0:
                        raise HTTPException(
                            status_code=400,
                            detail=f"页数必须大于 0，当前值: {page_count}"
                        )
                except HTTPException:
                    # 页数提取失败，移除幂等性标记
                    idempotency_service.remove(biz_no)
                    raise
                except Exception as e:
                    log.error(f"[Billing] Failed to extract page_count: {e}")
                    # 提取失败默认为 1 页
                    page_count = 1
                
                # 从 pricing.json 获取单页价格
                pricing_config = pricing_service.get_pricing_config()
                unit_price = pricing_config.get("pricing", {}).get(service, {}).get(endpoint, {}).get("price_per_page", 1)
                
                if unit_price == 1:
                    log.warning(f"[Billing] Price not found for {service}.{endpoint}, using default value 1")
                
                # 计算实际扣费金额
                total_event_value = unit_price * page_count
                
                log.info(f"[Billing] Dynamic billing: service={service}, endpoint={endpoint}, unit_price={unit_price}, page_count={page_count}, total={total_event_value}")
                
                # 业务成功后扣费
                await billing_service.charge(
                    request=request,
                    event_value=total_event_value,
                    biz_no=biz_no
                )
                
                return result
                
            except HTTPException:
                # 业务失败（HTTP 异常），移除幂等性标记，不扣费
                idempotency_service.remove(biz_no)
                raise
            except Exception as e:
                # 其他异常，移除幂等性标记，不扣费
                idempotency_service.remove(biz_no)
                log.error(f"[Billing] Unexpected error in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator



def with_hybrid_billing(service: str, endpoint: str):
    """
    混合计费装饰器：基础价格 + AI 增强按页计费
    
    Args:
        service: 服务名（如 "pdf2ppt"）
        endpoint: 端点名（如 "pdf2ppt"）
        
    Usage:
        @with_hybrid_billing("pdf2ppt", "pdf2ppt")
        async def my_endpoint(...):
            ...
    
    计费规则：
    - 不使用 AI：基础价格
    - 使用 AI：基础价格 + (AI单页价格 × 页数)
    - 价格从 config/pricing.json 读取
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 提取 request 对象
            request: Request = kwargs.get("request") or next(
                (arg for arg in args if isinstance(arg, Request)), None
            )
            
            if not request:
                raise RuntimeError("Cannot find Request object in function arguments")
            
            billing_service = BillingService()
            idempotency_service = get_idempotency_service()
            pricing_service = PricingService()
            
            # 检查是否启用计费
            if not billing_service.enabled:
                log.info(f"[Billing] Billing disabled, executing {func.__name__} without charge")
                return await func(*args, **kwargs)
            
            # 获取用户凭证并生成 bizNo
            try:
                access_key, client_name = billing_service.get_user_credentials(request)
                biz_no = billing_service.generate_biz_no(client_name)
            except HTTPException:
                # 如果获取凭证失败，直接抛出异常（401）
                raise
            
            # 幂等性检查（防止 5 分钟内重复提交）
            if idempotency_service.is_duplicate(biz_no):
                log.warning(f"[Billing] Duplicate request detected: biz_no={biz_no}")
                raise HTTPException(
                    status_code=409,
                    detail="请求重复，请勿在短时间内重复提交"
                )
            
            # 标记为处理中
            idempotency_service.mark_processed(biz_no)
            
            try:
                # 执行业务逻辑
                result = await func(*args, **kwargs)
                
                # 从 pricing.json 获取价格配置
                pricing_config = pricing_service.get_pricing_config()
                pricing_info = pricing_config.get("pricing", {}).get(service, {}).get(endpoint, {})
                
                base_price = pricing_info.get("base_price", 1)
                ai_per_page_price = pricing_info.get("ai_price_per_page", 1)
                
                if base_price == 1:
                    log.warning(f"[Billing] Base price not found for {service}.{endpoint}, using default value 1")
                if ai_per_page_price == 1:
                    log.warning(f"[Billing] AI per page price not found for {service}.{endpoint}, using default value 1")
                
                # 计算总费用
                from fastapi_app.utils.billing_utils import calculate_hybrid_billing
                
                try:
                    total_event_value = calculate_hybrid_billing(
                        kwargs=kwargs,
                        base_price=base_price,
                        ai_per_page_price=ai_per_page_price
                    )
                except HTTPException:
                    # 计费计算失败，移除幂等性标记
                    idempotency_service.remove(biz_no)
                    raise
                
                log.info(f"[Billing] Hybrid billing: service={service}, endpoint={endpoint}, total={total_event_value}")
                
                # 业务成功后扣费
                await billing_service.charge(
                    request=request,
                    event_value=total_event_value,
                    biz_no=biz_no
                )
                
                return result
                
            except HTTPException:
                # 业务失败（HTTP 异常），移除幂等性标记，不扣费
                idempotency_service.remove(biz_no)
                raise
            except Exception as e:
                # 其他异常，移除幂等性标记，不扣费
                idempotency_service.remove(biz_no)
                log.error(f"[Billing] Unexpected error in {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator
