# dataflow_agent/agentroles/__init__.py
import importlib
from pathlib import Path
from dataflow_agent.toolkits.tool_manager import get_tool_manager

# 1) 自动 import 所有 .py 文件
_pkg_path = Path(__file__).resolve().parent
for py in _pkg_path.glob("*.py"):
    if py.stem not in {"__init__", "registry", "base_agent"}:
        importlib.import_module(f"{__name__}.{py.stem}")


from .registry import AgentRegistry
# 2) 对外接口
def get_agent_cls(name: str):
    return AgentRegistry.get(name)

def create_agent(name: str, tool_manager=get_tool_manager(), **kwargs):
    """
    
    Args:
        name (str): 代理角色名称
        tool_manager (ToolManager, optional): 工具管理器. Defaults to get_tool_manager().
        **kwargs: 其他参数
        model_name: 模型名称，如 "gpt-4"
        tool_manager: 工具管理器实例
        temperature: 采样温度，控制随机性 (0.0-1.0)
        max_tokens: 最大生成token数
        tool_mode: 工具调用模式 ("auto", "none", "required")
        react_mode: 是否启用ReAct推理模式
        react_max_retries: ReAct模式下最大重试次数
        parser_type: 解析器类型 ("json", "xml", "text")，这个允许你在提示词中定义LLM不同的返回，xml还是json，还是直出；
        parser_config: 解析器配置字典（如XML的root_tag）
        use_vlm: 是否使用视觉语言模型，使用了视觉模型，其余的参数失效；
        vlm_config: VLM配置字典
        use_agent: 是否使用agent模式
        **kwargs: 其他传递给execute的参数
    
    Returns:
        Agent: 代理角色实例
    """
    cls = get_agent_cls(name)
    return cls.create(tool_manager=tool_manager, **kwargs)

list_agents = AgentRegistry.all