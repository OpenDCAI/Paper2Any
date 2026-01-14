from __future__ import annotations

import time
from typing import Dict
from dataflow_agent.logger import get_logger

log = get_logger(__name__)


class IdempotencyService:
    """
    幂等性服务：防止重复扣费（基于内存实现）
    
    注意：
    - 用户刷新页面或退出登录不影响扣费，因为业务逻辑已经执行
    - 只防止短时间内（5分钟）的重复提交
    """
    
    def __init__(self, ttl: int = 300):
        """
        Args:
            ttl: 幂等性保护时长（秒），默认 5 分钟
        """
        self._cache: Dict[int, float] = {}  # {biz_no: expire_time}
        self.ttl = ttl
    
    def _cleanup_expired(self):
        """清理过期的记录"""
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if v < now]
        for k in expired_keys:
            del self._cache[k]
    
    def is_duplicate(self, biz_no: int) -> bool:
        """检查是否重复"""
        self._cleanup_expired()
        return biz_no in self._cache
    
    def mark_processed(self, biz_no: int):
        """标记已处理"""
        self._cache[biz_no] = time.time() + self.ttl
        log.info(f"[Idempotency] Marked biz_no={biz_no} as processed")
    
    def remove(self, biz_no: int):
        """移除记录（业务失败时调用，避免扣费）"""
        if biz_no in self._cache:
            del self._cache[biz_no]
            log.info(f"[Idempotency] Removed biz_no={biz_no}")


# 全局单例
_idempotency_service: IdempotencyService | None = None

def get_idempotency_service() -> IdempotencyService:
    global _idempotency_service
    if _idempotency_service is None:
        _idempotency_service = IdempotencyService()
    return _idempotency_service
