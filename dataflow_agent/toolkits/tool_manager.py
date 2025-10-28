from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Callable, Set
from langchain_core.tools import Tool
from dataflow_agent.logger import get_logger

# from dataflow_agent.agentroles.base_agent import BaseAgent
# from dataflow_agent.state import DFState

log = get_logger(__name__)

class ToolManager:
    """工具管理器 - 支持不同角色的工具管理"""
    
    def __init__(self):
        self.role_pre_tools: Dict[str, Dict[str, Callable]] = {}
        self.role_post_tools: Dict[str, List[Tool]] = {}
        self.global_pre_tools: Dict[str, Callable] = {}
        self.global_post_tools: List[Tool] = []
    
    def register_pre_tool(self, name: str, func: Callable, role: Optional[str] = None):
        """注册前置工具"""
        if role:
            if role not in self.role_pre_tools:
                self.role_pre_tools[role] = {}
            self.role_pre_tools[role][name] = func
            log.info(f"为角色 '{role}' 注册前置工具: {name}")
        else:
            self.global_pre_tools[name] = func
            log.info(f"注册全局前置工具: {name}")
    
    def register_post_tool(self, tool: Tool, role: Optional[str] = None):
        """注册后置工具"""
        if role:
            if role not in self.role_post_tools:
                self.role_post_tools[role] = []
            self.role_post_tools[role].append(tool)
            log.info(f"为角色 '{role}' 注册后置工具: {tool.name}")
        else:
            self.global_post_tools.append(tool)
            log.info(f"注册全局后置工具: {tool.name}")
    
    def get_pre_tools(self, role: str) -> Dict[str, Callable]:
        """获取指定角色的前置工具（包含全局工具）"""
        tools = self.global_pre_tools.copy()
        if role in self.role_pre_tools:
            tools.update(self.role_pre_tools[role])
        return tools
    
    def get_post_tools(self, role: str) -> List[Tool]:
        """获取指定角色的后置工具（包含全局工具）"""
        tools = self.global_post_tools.copy()
        if role in self.role_post_tools:
            tools.extend(self.role_post_tools[role])
        return tools
    
    async def execute_pre_tools(self, role: str) -> Dict[str, Any]:
        """执行指定角色的前置工具"""
        tools = self.get_pre_tools(role)
        results = {}
        
        for name, func in tools.items():
            try:
                if asyncio.iscoroutinefunction(func):
                    results[name] = await func()
                else:
                    results[name] = func()
                log.info(f"角色 '{role}' 前置工具 '{name}' 执行成功")
            except Exception as e:
                log.error(f"角色 '{role}' 前置工具 '{name}' 执行失败: {e}")
                results[name] = None
        
        return results
    
    def get_available_roles(self) -> Set[str]:
        roles = set(self.role_pre_tools.keys())
        roles.update(self.role_post_tools.keys())
        return roles
    
    # ==================== Agent-as-Tool 支持 ====================

    def register_agent_as_tool(self, agent, state, role: Optional[str] = None):
        """
        将 agent 注册为后置工具
        
        Args:
            agent: BaseAgent 实例
            state: DFState 实例
            role: 要注册到哪个角色的后置工具，None 表示全局工具
        """
        tool = agent.as_tool(state)
        self.register_post_tool(tool, role)
        log.info(f"Agent '{agent.role_name}' 注册为工具 '{tool.name}' (角色: {role or 'global'})")
        return tool

    def register_multiple_agents_as_tools(self, agents: List, state, role: Optional[str] = None):
        """
        批量注册多个 agent 作为后置工具
        
        Args:
            agents: BaseAgent 实例列表
            state: DFState 实例
            role: 目标角色
        """
        tools = []
        for agent in agents:
            tool = self.register_agent_as_tool(agent, state, role)
            tools.append(tool)
        log.info(f"批量注册了 {len(tools)} 个 agent 作为工具")
        return tools

_tool_manager_instance = None

def get_tool_manager() -> ToolManager:
    global _tool_manager_instance
    if _tool_manager_instance is None:
        _tool_manager_instance = ToolManager()
    return _tool_manager_instance