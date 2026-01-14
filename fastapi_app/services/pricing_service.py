from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)


class PricingService:
    """价格配置服务：从 JSON 文件加载价格配置"""
    
    _instance: Optional['PricingService'] = None
    _pricing_config: Optional[Dict[str, Any]] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._pricing_config is None:
            self._load_pricing_config()
    
    def _load_pricing_config(self):
        """从 JSON 文件加载价格配置"""
        try:
            # 获取配置文件路径
            config_path = os.getenv("PRICING_CONFIG_PATH", "config/pricing.json")
            
            # 如果是相对路径，则相对于项目根目录
            if not os.path.isabs(config_path):
                project_root = get_project_root()
                config_path = project_root / config_path
            else:
                config_path = Path(config_path)
            
            # 读取 JSON 文件
            if not config_path.exists():
                log.error(f"[Pricing] Config file not found: {config_path}")
                self._pricing_config = self._get_default_config()
                return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self._pricing_config = json.load(f)
            
            log.info(f"[Pricing] Loaded pricing config from: {config_path}")
            log.info(f"[Pricing] Config version: {self._pricing_config.get('version', 'unknown')}")
            
        except Exception as e:
            log.error(f"[Pricing] Failed to load pricing config: {e}")
            self._pricing_config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置（降级方案）"""
        return {
            "version": "1.0",
            "currency": "积分",
            "pricing": {}
        }
    
    def get_pricing_config(self) -> Dict[str, Any]:
        """获取完整的价格配置"""
        return self._pricing_config or self._get_default_config()
    
    def get_price(self, service: str, endpoint: str, key: str) -> Any:
        """
        获取特定价格
        
        Args:
            service: 服务名称（如 'paper2ppt'）
            endpoint: 端点名称（如 'pagecontent'）
            key: 价格键（如 'price_per_page'）
            
        Returns:
            价格值，如果不存在则返回 None
        """
        try:
            pricing = self._pricing_config.get('pricing', {})
            service_pricing = pricing.get(service, {})
            endpoint_pricing = service_pricing.get(endpoint, {})
            return endpoint_pricing.get(key)
        except Exception as e:
            log.error(f"[Pricing] Failed to get price for {service}.{endpoint}.{key}: {e}")
            return None
    
    def reload_config(self):
        """重新加载配置（用于热更新）"""
        log.info("[Pricing] Reloading pricing config...")
        self._pricing_config = None
        self._load_pricing_config()


# 全局单例
_pricing_service: Optional[PricingService] = None

def get_pricing_service() -> PricingService:
    global _pricing_service
    if _pricing_service is None:
        _pricing_service = PricingService()
    return _pricing_service
