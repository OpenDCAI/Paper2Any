from __future__ import annotations

import os
import time
import random
import httpx
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


class BillingService:
    """计费服务：负责从 Cookie 获取用户信息并调用扣费 API"""
    
    def __init__(self):
        self.api_url = os.getenv(
            "BILLING_API_URL",
            "https://openapi.dp.tech/openapi/v1/api/integral/consume"
        )
        self.enabled = os.getenv("BILLING_ENABLED", "true").lower() == "true"
        self.sku_id = int(os.getenv("BILLING_SKU_ID", "0"))
        
        # 开发者默认凭证（用于本地调试）
        self.dev_access_key = os.getenv("DEV_ACCESS_KEY")
        self.dev_client_name = os.getenv("DEV_CLIENT_NAME")
        
        if self.enabled and self.sku_id == 0:
            log.warning("[Billing] BILLING_SKU_ID not configured, billing may fail")
        
    def get_user_credentials(self, request: Request) -> tuple[str, str]:
        """
        从 Cookie 获取用户凭证，如果没有则使用开发者默认凭证
        
        优先级：
        1. Cookie 中的用户凭证（生产环境）
        2. 环境变量中的开发者凭证（本地调试）
        3. 抛出异常（未配置）
        """
        access_key = request.cookies.get("appAccessKey")
        client_name = request.cookies.get("clientName")
        
        # 如果 Cookie 中有凭证，优先使用
        if access_key and client_name:
            log.info(f"[Billing] Using credentials from Cookie: client={client_name}")
            return access_key, client_name
        
        # 否则尝试使用开发者默认凭证
        if self.dev_access_key and self.dev_client_name:
            log.warning(f"[Billing] Using developer credentials: client={self.dev_client_name}")
            return self.dev_access_key, self.dev_client_name
        
        # 都没有，抛出异常
        log.error("[Billing] No credentials found in Cookie or environment variables")
        raise HTTPException(
            status_code=401,
            detail="未找到用户凭证，请先登录或配置开发者凭证"
        )
    
    def generate_biz_no(self, client_name: str) -> int:
        """生成业务流水号：timestamp + clientName前4位hash + random"""
        timestamp = int(time.time())
        client_prefix = client_name[:4].ljust(4, '0')  # 取前4位，不足补0
        # 将前4位转为数字（ASCII 码求和取模）
        client_hash = sum(ord(c) for c in client_prefix) % 10000
        rand_part = random.randint(1000, 9999)
        biz_no = int(f"{timestamp}{client_hash:04d}{rand_part}")
        return biz_no
    
    async def charge(
        self,
        request: Request,
        event_value: int,
        biz_no: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行扣费操作
        
        Args:
            request: FastAPI Request 对象
            event_value: 扣费金额
            biz_no: 业务流水号（可选，不传则自动生成）
            
        Returns:
            扣费结果
            
        Raises:
            HTTPException: 扣费失败时抛出 402
        """
        if not self.enabled:
            log.info("[Billing] Billing is disabled, skipping charge")
            return {"success": True, "message": "Billing disabled"}
        
        # 获取用户凭证
        access_key, client_name = self.get_user_credentials(request)
        
        # 生成业务流水号
        if biz_no is None:
            biz_no = self.generate_biz_no(client_name)
        
        # 构建请求
        headers = {
            "accessKey": access_key,
            "x-app-key": client_name,
            "Content-Type": "application/json"
        }
        
        payload = {
            "bizNo": biz_no,
            "changeType": 1,
            "eventValue": event_value,
            "skuId": self.sku_id,
            "scene": "appCustomizeCharge"
        }
        
        log.info(f"[Billing] Charging user {client_name}, bizNo={biz_no}, skuId={self.sku_id}, eventValue={event_value}")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                )
                
                result = response.json()
                log.info(f"[Billing] API Response: status={response.status_code}, body={result}")
                
                if response.status_code != 200:
                    log.error(f"[Billing] Charge failed: HTTP {response.status_code} {response.text}")
                    raise HTTPException(
                        status_code=402,
                        detail=f"扣费失败: {response.text}"
                    )
                
                # 检查业务返回码，code=0 表示成功
                if result.get("code") != 0:
                    log.error(f"[Billing] Charge failed: code={result.get('code')}, data={result.get('data')}")
                    raise HTTPException(
                        status_code=402,
                        detail=f"扣费失败: {result.get('data', '未知错误')}"
                    )
                
                log.info(f"[Billing] Charge successful! client={client_name}, bizNo={biz_no}, amount={event_value}")
                return result
                
        except httpx.TimeoutException:
            log.error("[Billing] Charge timeout")
            raise HTTPException(
                status_code=402,
                detail="扣费超时，请稍后重试"
            )
        except httpx.RequestError as e:
            log.error(f"[Billing] Charge request error: {e}")
            raise HTTPException(
                status_code=402,
                detail=f"扣费请求失败: {str(e)}"
            )
