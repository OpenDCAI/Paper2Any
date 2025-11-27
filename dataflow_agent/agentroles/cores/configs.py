from typing import Any, Dict, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ExecutionMode(Enum):
    """执行模式枚举"""
    SIMPLE = "simple"           # 简单模式：单次LLM调用
    REACT = "react"             # ReAct模式：带验证的循环
    GRAPH = "graph"             # 图模式：子图+工具调用
    VLM = "vlm"                 # 视觉语言模型模式
    PARALLEL = "parallel"       # 并行模式：同时调用多个LLM
    PLAN_SOLVE = "plan_solve"   # Plan-and-Solve模式：一次性生成计划，按顺序执行
    PLAN_EXECUTE = "plan_execute"  # Plan-and-Execute模式：动态调整计划
    CUSTOM = "custom"           # 自定义模式（预留）


@dataclass
class BaseAgentConfig:
    """Agent基础配置"""
    # 核心参数
    model_name: Optional[str] = None
    chat_api_url: Optional[str] = None  # 新增chat_api_url参数
    temperature: float = 0.0
    max_tokens: int = 16384
    
    # 工具相关
    tool_mode: str = "auto"
    tool_manager: Optional[Any] = None  # ToolManager
    
    # 解析器相关
    parser_type: str = "json"
    parser_config: Optional[Dict[str, Any]] = None
    
    # JSON Schema 快捷配置（用于 JSONParser）
    response_schema: Optional[Dict[str, Any]] = None  # 期望的返回结构，如 {"code": "string", "files": "list"}
    response_schema_description: Optional[str] = None  # Schema 的文字描述
    response_example: Optional[Dict[str, Any]] = None  # 返回示例
    required_fields: Optional[List[str]] = None       # 必填字段列表
    
    # 消息历史
    ignore_history: bool = True
    message_history: Optional[Any] = None  # AdvancedMessageHistory


@dataclass
class SimpleConfig(BaseAgentConfig):
    """简单模式配置"""
    mode: ExecutionMode = field(default=ExecutionMode.SIMPLE, init=False)


@dataclass
class ReactConfig(BaseAgentConfig):
    """ReAct模式配置"""
    mode: ExecutionMode = field(default=ExecutionMode.REACT, init=False)
    max_retries: int = 3
    validators: Optional[List[Callable]] = None  # 自定义验证器


@dataclass
class GraphConfig(BaseAgentConfig):
    """图模式配置"""
    mode: ExecutionMode = field(default=ExecutionMode.GRAPH, init=False)
    enable_react_validation: bool = False  # 是否在图模式中启用ReAct验证
    react_max_retries: int = 3


@dataclass
class VLMConfig(BaseAgentConfig):
    """视觉语言模型配置"""
    mode: ExecutionMode = field(default=ExecutionMode.VLM, init=False)
    vlm_mode: str = "understanding"  # understanding/generation/edit
    image_detail: str = "auto"       # low/high/auto
    max_image_size: tuple = (1024, 1024)
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelConfig(BaseAgentConfig):
    """并行模式配置"""
    mode: ExecutionMode = field(default=ExecutionMode.PARALLEL, init=False)
    concurrency_limit: int = 5  # 并行度限制，默认同时执行5个任务


# ==================== Planning Agent 配置 ====================

@dataclass
class PlanSolveConfig(BaseAgentConfig):
    """
    Plan-and-Solve 模式配置
    
    一次性生成完整计划，然后按顺序执行，不回头调整。
    适用于确定性较高的任务。
    
    工作流程:
    1. Planner 分析任务，生成完整的步骤计划
    2. (可选) Human-in-the-Loop: 用户审批计划
    3. Executor 按顺序执行每个步骤
    4. 返回最终结果
    
    Example:
        >>> config = PlanSolveConfig(
        ...     planner_model="gpt-4",
        ...     executor_model="gpt-4-turbo",
        ...     require_plan_approval=True,
        ...     max_plan_steps=5
        ... )
        >>> agent = create_plan_solve_agent("task_planner", config=config)
    """
    mode: ExecutionMode = field(default=ExecutionMode.PLAN_SOLVE, init=False)
    
    # ----- 规划器配置 -----
    planner_model: Optional[str] = None         # 规划器使用的模型（默认使用 model_name）
    planner_temperature: float = 0.0            # 规划器温度（通常较低以保证稳定性）
    planner_prompt_template: Optional[str] = None  # 自定义规划器 prompt 模板名
    
    # ----- 执行器配置 -----
    executor_model: Optional[str] = None        # 执行器使用的模型
    executor_temperature: float = 0.0           # 执行器温度
    executor_as_react: bool = True              # 执行器是否使用 ReAct 模式
    executor_max_retries: int = 2               # 执行器 ReAct 重试次数
    executor_tools: Optional[List[Any]] = None  # 执行器可用的工具列表
    
    # ----- Human-in-the-Loop 配置 -----
    require_plan_approval: bool = True          # 是否需要用户审批计划
    show_plan_details: bool = True              # 审批时是否显示详细信息
    
    # ----- 执行控制 -----
    max_plan_steps: int = 10                    # 最大计划步骤数
    continue_on_error: bool = False             # 某步骤失败时是否继续执行
    collect_step_results: bool = True           # 是否收集每步的执行结果


@dataclass
class PlanExecuteConfig(BaseAgentConfig):
    """
    Plan-and-Execute (Replanning) 模式配置
    
    动态生成和调整计划。执行一步后评估结果，决定是继续执行、
    调整计划还是完成任务。适用于复杂、不确定性较高的任务。
    
    工作流程:
    1. Planner 分析任务，生成初始计划
    2. (可选) Human-in-the-Loop: 用户审批计划
    3. Executor 执行当前步骤
    4. (可选) Human-in-the-Loop: 用户确认/干预
    5. Replanner 评估执行结果，决定:
       - 继续执行下一步
       - 调整剩余计划
       - 任务已完成，返回结果
    6. 循环 3-5 直到完成或达到最大轮数
    
    Example:
        >>> config = PlanExecuteConfig(
        ...     planner_model="gpt-4",
        ...     executor_model="gpt-4-turbo",
        ...     replanner_model="gpt-4",
        ...     interrupt_before_step=True,
        ...     max_replanning_rounds=3
        ... )
        >>> agent = create_plan_execute_agent("adaptive_planner", config=config)
    """
    mode: ExecutionMode = field(default=ExecutionMode.PLAN_EXECUTE, init=False)
    
    # ----- 规划器配置 -----
    planner_model: Optional[str] = None         # 规划器使用的模型
    planner_temperature: float = 0.0            # 规划器温度
    planner_prompt_template: Optional[str] = None  # 自定义规划器 prompt 模板名
    
    # ----- 执行器配置 -----
    executor_model: Optional[str] = None        # 执行器使用的模型
    executor_temperature: float = 0.0           # 执行器温度
    executor_as_react: bool = True              # 执行器是否使用 ReAct 模式
    executor_max_retries: int = 2               # 执行器 ReAct 重试次数
    executor_tools: Optional[List[Any]] = None  # 执行器可用的工具列表
    
    # ----- 重规划器配置 -----
    replanner_model: Optional[str] = None       # 重规划器使用的模型
    replanner_temperature: float = 0.0          # 重规划器温度
    replanner_prompt_template: Optional[str] = None  # 自定义重规划器 prompt 模板名
    max_replanning_rounds: int = 3              # 最大重规划轮数
    
    # ----- Human-in-the-Loop 配置 -----
    require_plan_approval: bool = True          # 是否需要用户审批初始计划
    interrupt_before_step: bool = True          # 每步执行前是否中断等待用户确认
    interrupt_after_step: bool = False          # 每步执行后是否中断
    interrupt_on_replan: bool = True            # 重规划时是否中断
    
    # ----- 执行控制 -----
    max_plan_steps: int = 10                    # 最大计划步骤数
    continue_on_error: bool = False             # 某步骤失败时是否继续
    auto_replan_on_error: bool = True           # 失败时是否自动触发重规划
    collect_step_results: bool = True           # 是否收集每步的执行结果


# ==================== 配置验证函数 ====================

def validate_planning_config(config: BaseAgentConfig) -> Tuple[bool, Optional[str]]:
    """
    验证 Planning 配置是否有效
    
    Args:
        config: 配置对象
    
    Returns:
        (是否有效, 错误信息)
    """
    if isinstance(config, (PlanSolveConfig, PlanExecuteConfig)):
        if config.max_plan_steps <= 0:
            return False, "max_plan_steps 必须大于 0"
        
        if isinstance(config, PlanExecuteConfig):
            if config.max_replanning_rounds <= 0:
                return False, "max_replanning_rounds 必须大于 0"
    
    return True, None
