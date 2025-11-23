from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Callable, Tuple
from dataflow_agent.llm_callers.base import BaseLLMCaller
from dataflow_agent.parsers.parsers import BaseParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
from dataflow_agent.graphbuilder.message_history import AdvancedMessageHistory


from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow_agent.state import MainState
from dataflow_agent.utils import robust_parse_json
from dataflow_agent.toolkits.tool_manager import ToolManager
import pickle
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root
from dataflow_agent.agentroles.cores.strategies import ExecutionStrategy

PROJDIR = get_project_root()

log = get_logger(__name__)

# 验证器类型定义：返回 (是否通过, 错误信息)
ValidatorFunc = Callable[[str, Dict[str, Any]], Tuple[bool, Optional[str]]]


class BaseAgent(ABC):
    """Agent基类 - 定义通用的agent执行模式"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        try:
            tmp = cls(tool_manager=None)          # BaseAgent 的 __init__ 很轻
            name = tmp.role_name
            from dataflow_agent.agentroles.registry import AgentRegistry
            AgentRegistry.register(name.lower(), cls)
        except Exception as e:
            pass
    
    def __init__(self, 
                 tool_manager: Optional[ToolManager] = None,
                 model_name: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 16384,
                 tool_mode: str = "auto",
                 react_mode: bool = False,
                 react_max_retries: int = 3,
                 # 新增参数
                 parser_type: str = "json",
                 parser_config: Optional[Dict[str, Any]] = None,
                 use_vlm: bool = False,
                 vlm_config: Optional[Dict[str, Any]] = None,
                 ignore_history: bool = True,
                 message_history: Optional[AdvancedMessageHistory] = None,

                # 新增参数，用于策略控制； 
                 execution_config: Optional[Any] = None,
                 chat_api_url: Optional[str] = None
                 ):
        """
        Args:
            parser_type: 解析器类型 ("json", "xml", "text")
            parser_config: 解析器配置（如XML的root_tag）
            use_vlm: 是否使用视觉语言模型
            vlm_config: VLM配置字典
            ignore_history: 是否忽略历史消息
            message_history: 消息历史管理器
            execution_config: 执行配置，用于策略控制
        """
        self.tool_manager = tool_manager
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_mode = tool_mode
        self.react_mode = react_mode
        self.react_max_retries = react_max_retries
        self.chat_api_url = chat_api_url
        
        # 解析器配置
        self.parser_type = parser_type
        self.parser_config = parser_config or {}
        self._parser = None  # 懒加载
        
        # VLM配置
        self.use_vlm = use_vlm
        self.vlm_config = vlm_config or {}
        
        # 消息历史配置
        self.ignore_history = ignore_history
        self.message_history = message_history or AdvancedMessageHistory()

        # ========== 新增：策略模式支持 ==========
        self._execution_strategy: Optional[ExecutionStrategy] = None
        if execution_config:
            # 如果提供了执行配置，则使用其中的值更新agent实例的属性
            # 这解决了通过 create_simple_agent 等函数创建时参数不生效的问题
            for f in execution_config.__dataclass_fields__:
                config_value = getattr(execution_config, f)
                if hasattr(self, f) and config_value is not None:
                    setattr(self, f, config_value)

            from dataflow_agent.agentroles.cores.strategies import StrategyFactory
            self._execution_strategy = StrategyFactory.create(
                execution_config.mode.value,
                self,
                execution_config
            )

    # 暂时没用到这个LLM caller；还是create_llm；
    def get_llm_caller(self, state: MainState) -> BaseLLMCaller:
        """根据配置返回对应的LLM Caller"""
        if self.use_vlm:
            from dataflow_agent.llm_callers import VisionLLMCaller
            log.info(f"使用 VisionLLMCaller，模式: {self.vlm_config.get('mode', 'understanding')}")
            return VisionLLMCaller(
                state,
                vlm_config=self.vlm_config,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tool_mode=self.tool_mode,
                tool_manager=self.tool_manager,
                chat_api_url=self.chat_api_url
            )
        else:
            from dataflow_agent.llm_callers import TextLLMCaller
            return TextLLMCaller(
                state,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tool_mode=self.tool_mode,
                tool_manager=self.tool_manager,
                chat_api_url=self.chat_api_url
            )

    @property
    def parser(self) -> BaseParser:
        """获取解析器实例（懒加载）"""
        if self._parser is None:
            from dataflow_agent.parsers import ParserFactory
            self._parser = ParserFactory.create(self.parser_type, **self.parser_config)
        return self._parser
    
    def parse_result(self, content: str) -> Dict[str, Any]:
        """使用配置的解析器解析结果"""
        try:
            parsed = self.parser.parse(content)
            log.info(f"{self.role_name} 使用 {self.parser_type} 解析器解析成功")
            log.critical(f"[parse_result 解析结果] : {parsed}")
            return parsed
        except Exception as e:
            log.exception(f"解析失败: {e}")
            return {"raw": content, "error": str(e)}

    @classmethod
    def create(cls, tool_manager: Optional[ToolManager] = None, **kwargs) -> "BaseAgent":
        """
        工厂方法：保持所有 Agent 统一的创建入口。

        ----
        BaseAgent 的具体子类实例（cls）
        """
        return cls(tool_manager=tool_manager, **kwargs)
    
    @property
    @abstractmethod
    def role_name(self) -> str:
        """角色名称 - 子类必须实现"""
        pass
    
    @property
    @abstractmethod
    def system_prompt_template_name(self) -> str:
        """系统提示词模板名称 - 子类必须实现"""
        pass
    
    @property
    @abstractmethod
    def task_prompt_template_name(self) -> str:
        """任务提示词模板名称 - 子类必须实现"""
        pass
    
    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取任务提示词参数 - 子类可重写
        
        Args:
            pre_tool_results: 前置工具结果
            
        Returns:
            提示词参数字典
        """
        return pre_tool_results
    
    # def parse_result(self, content: str) -> Dict[str, Any]:
    #     """
    #     解析结果 - 子类可重写自定义解析逻辑
        
    #     Args:
    #         content: LLM输出内容
            
    #     Returns:
    #         解析后的结果
    #     """
    #     try:
    #         parsed = robust_parse_json(content)
    #         # log.info(f'content是什么？？{content}')
    #         log.info(f"{self.role_name} 结果解析成功")
    #         return parsed
    #     except ValueError as e:
    #         log.warning(f"JSON解析失败: {e}")
    #         return {"raw": content}
    #     except Exception as e:
    #         log.warning(f"解析过程出错: {e}")
    #         return {"raw": content}
    
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """获取默认前置工具结果 - 子类可重写"""
        return {}
    
    # ==================== 并行模式 ========================================================================================================================
    async def process_parallel_mode(self, state: MainState, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        并行模式处理
        
        自动检测前置工具结果中的列表数据，进行并行LLM调用
        不需要强制使用特定的字段名
        """
        import asyncio
        log.info(f"执行 {self.role_name} 并行模式...")
        
        # 智能检测并行数据
        parallel_items = []
        
        # 情况1: 如果pre_tool_results中直接包含列表类型的主要数据
        if isinstance(pre_tool_results, list):
            parallel_items = pre_tool_results
        
        # 情况2: 如果有明确的parallel_items字段
        elif "parallel_items" in pre_tool_results:
            parallel_items = pre_tool_results["parallel_items"]
        
        # 情况3: 检查是否有任何值为列表的字段（取第一个找到的非空列表）
        elif isinstance(pre_tool_results, dict):
            for key, value in pre_tool_results.items():
                if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                    parallel_items = value
                    break
        
        # 如果没有找到合适的并行数据
        if not parallel_items:
            log.warning("未找到合适的并行数据，回退到简单模式")
            return await self.process_simple_mode(state, pre_tool_results)
        
        log.info(f"找到 {len(parallel_items)} 条数据用于并行处理")
        
        # 获取并发限制（从执行策略配置中获取）
        concurrency_limit = 5  # 默认值
        if hasattr(self, '_execution_strategy') and hasattr(self._execution_strategy, 'config'):
            if hasattr(self._execution_strategy.config, 'concurrency_limit'):
                concurrency_limit = self._execution_strategy.config.concurrency_limit
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        # 定义单个并行任务处理函数
        async def process_item(item: dict) -> tuple[int, dict]:
            """处理单个并行项，返回索引和结果"""
            async with semaphore:
                try:
                    # 为每个并行项创建独立的上下文
                    item_pre_tool_results = {}
                    
                    # 如果是字典，将其所有键值对添加到前置工具结果
                    if isinstance(item, dict):
                        item_pre_tool_results.update(item)
                    
                    # 也保留原始前置工具结果中的非列表字段
                    if isinstance(pre_tool_results, dict):
                        for key, value in pre_tool_results.items():
                            if not isinstance(value, list):
                                item_pre_tool_results[key] = value
                    
                    # 使用简单模式处理单个项
                    result = await self.process_simple_mode(state, item_pre_tool_results)
                    return result
                except Exception as e:
                    log.error(f"并行处理单个项失败: {e}")
                    return {"error": str(e)}
        
        # 并行执行所有任务
        tasks = [process_item(item) for item in parallel_items]
        results = await asyncio.gather(*tasks)
        
        log.info(f"并行模式执行完成，共处理 {len(results)} 个任务")
        
        # 返回结果列表
        return {
            "parallel_results": results,
            "total_processed": len(results)
        }
    # ==================== Agent-as-Tool 功能 ========================================================================================================================

    def get_tool_name(self) -> str:
        """获取作为工具时的名称 - 子类可重写"""
        return f"call_{self.role_name.lower()}_agent"

    def get_tool_description(self) -> str:
        """获取作为工具时的描述 - 子类应该重写提供更具体的描述"""
        return f"调用 {self.role_name} agent 来执行特定任务。该 agent 会根据输入参数执行相应的分析和处理。"

    def get_tool_args_schema(self) -> Type[BaseModel]:
        """获取作为工具时的参数模式 - 子类可以重写定义自己的参数结构"""
        from pydantic import BaseModel, Field
        
        class DefaultAgentToolArgs(BaseModel):
            """默认 Agent 工具参数"""
            task_description: str = Field(
                description=f"传递给 {self.role_name} 的任务描述或指令"
            )
            additional_params: Optional[Dict[str, Any]] = Field(
                default=None,
                description="额外的参数，会被合并到前置工具结果中"
            )
        
        return DefaultAgentToolArgs

    def prepare_tool_execution_params(self, **tool_kwargs) -> Dict[str, Any]:
        """准备工具执行时的参数 - 子类可重写自定义参数处理逻辑"""
        params = {}
        
        if 'additional_params' in tool_kwargs and tool_kwargs['additional_params']:
            params.update(tool_kwargs['additional_params'])
        
        for key, value in tool_kwargs.items():
            if key != 'additional_params':
                params[key] = value
        
        return params

    def extract_tool_result(self, state: MainState) -> Dict[str, Any]:
        """从状态中提取工具调用的结果 - 子类可重写自定义结果提取逻辑"""
        agent_result = state.agent_results.get(self.role_name, {})
        return agent_result.get('results', {})

    async def _execute_as_tool(self, state: MainState, **tool_kwargs) -> Dict[str, Any]:
        """作为工具执行的内部方法"""
        try:
            log.info(f"[Agent-as-Tool] 调用 {self.role_name}，参数: {tool_kwargs}")
            
            exec_params = self.prepare_tool_execution_params(**tool_kwargs)
            
            result_state = await self.execute(
                state, 
                use_agent=False,
                **exec_params
            )
            
            result = self.extract_tool_result(result_state)
            
            log.info(f"[Agent-as-Tool] {self.role_name} 执行完成")
            return result
            
        except Exception as e:
            log.exception(f"[Agent-as-Tool] {self.role_name} 执行失败: {e}")
            return {
                "error": str(e),
                "agent": self.role_name,
                "status": "failed"
            }

    def as_tool(self, state: MainState) -> Tool:
        """将 agent 包装成可被调用的工具"""
        
        async def agent_tool_func(**kwargs) -> Dict[str, Any]:
            return await self._execute_as_tool(state, **kwargs)
        
        def sync_agent_tool_func(**kwargs) -> Dict[str, Any]:
            return asyncio.run(agent_tool_func(**kwargs))
        
        return Tool(
            name=self.get_tool_name(),
            description=self.get_tool_description(),
            func=sync_agent_tool_func,
            coroutine=agent_tool_func,
            args_schema=self.get_tool_args_schema()
        )
    
    # ==================== ReAct 模式相关方法 ========================================================================================================================
    
    def get_react_validators(self) -> List[ValidatorFunc]:
        """
        获取ReAct模式的验证器列表 - 子类可重写添加自定义验证器
        
        Returns:
            验证器函数列表，每个函数签名为: (content: str, parsed_result: Dict) -> (bool, Optional[str])
            - bool: 是否通过验证
            - Optional[str]: 未通过时的错误提示信息
        """
        return [
            self._default_json_validator,
            # 子类可以添加更多验证器，例如：
            # self._check_required_fields,
            # self._check_data_format,
        ]
    
    @staticmethod
    def _default_json_validator(content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        默认JSON格式验证器
        
        Args:
            content: LLM原始输出
            parsed_result: parse_result()的结果
            
        Returns:
            (是否通过, 错误信息)
        """
        # 如果解析结果中只有raw字段，说明JSON解析失败
        if "raw" in parsed_result and len(parsed_result) == 1:
            return False, (
                "你返回的内容不是有效的JSON格式。请确保返回纯JSON格式的数据，"
                "不要包含其他文字说明。正确的格式示例：\n"
                '{"key1": "value1", "key2": "value2"}'
            )
        
        # 检查是否为空字典
        if not parsed_result or (isinstance(parsed_result, dict) and not parsed_result):
            return False, "你返回的JSON为空，请提供完整的结果数据。"
        
        return True, None
    
    def _run_validators(self, content: str, parsed_result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        运行所有验证器
        
        Args:
            content: LLM原始输出
            parsed_result: 解析后的结果
            
        Returns:
            (是否全部通过, 错误信息列表)
        """
        validators = self.get_react_validators()
        errors = []
        
        for i, validator in enumerate(validators):
            try:
                passed, error_msg = validator(content, parsed_result)
                if not passed:
                    validator_name = getattr(validator, '__name__', f'validator_{i}')
                    log.warning(f"验证器 {validator_name} 未通过: {error_msg}")
                    if error_msg:
                        errors.append(error_msg)
            except Exception as e:
                log.exception(f"验证器执行出错: {e}")
                errors.append(f"验证过程出错: {str(e)}")
        
        return len(errors) == 0, errors
    
    async def process_react_mode(self, state: MainState, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        ReAct模式处理 - 循环调用LLM直到验证通过
        
        Args:
            state: DFState实例
            pre_tool_results: 前置工具结果
            
        Returns:
            处理结果
        """
        log.info(f"执行 {self.role_name} ReAct模式 (最大重试: {self.react_max_retries})")
        
        # 构建初始消息
        messages = self.build_messages(state, pre_tool_results)
        
        # 消息历史管理
        if not self.ignore_history:
            history_messages = self.message_history.get_messages()
            if history_messages:
                messages = self.message_history.merge_histories(history_messages, messages)
                log.info(f"合并了 {len(history_messages)} 条历史消息")
        
        llm = self.create_llm(state, bind_post_tools=False)
        
        for attempt in range(self.react_max_retries + 1):
            try:
                # 调用LLM
                log.info(f"ReAct尝试 {attempt + 1}/{self.react_max_retries + 1}")
                answer_msg = await llm.ainvoke(messages)
                answer_text = answer_msg.content
                log.info(f'LLM原始输出：{answer_text[:200]}...' if len(answer_text) > 200 else f'LLM原始输出：{answer_text}')
                
                # 解析结果
                parsed_result = self.parse_result(answer_text)
                
                # 运行验证器
                all_passed, errors = self._run_validators(answer_text, parsed_result)
                
                if all_passed:
                    log.info(f"✓ {self.role_name} ReAct验证通过，共尝试 {attempt + 1} 次")
                    
                    # 更新消息历史
                    if not self.ignore_history:
                        self.message_history.add_messages([answer_msg])
                        log.info("已更新消息历史")
                    
                    return parsed_result
                
                # 验证未通过
                if attempt < self.react_max_retries:
                    # 构建反馈消息
                    feedback = self._build_validation_feedback(errors)
                    log.warning(f"[process_react_mode] : 验证未通过 (尝试 {attempt + 1}): {feedback}")
                    
                    # 添加LLM的回复和人类的反馈到消息列表
                    messages.append(AIMessage(content=answer_text))
                    messages.append(HumanMessage(content=feedback))
                else:
                    # 达到最大重试次数
                    log.error(f"[process_react_mode] : {self.role_name} ReAct达到最大重试次数，验证仍未通过")
                    return {
                        "error": "ReAct验证失败",
                        "attempts": attempt + 1,
                        "last_errors": errors,
                        "last_result": parsed_result
                    }
                    
            except Exception as e:
                log.exception(f"ReAct模式LLM调用失败 (尝试 {attempt + 1}): {e}")
                if attempt >= self.react_max_retries:
                    return {"error": f"LLM调用失败: {str(e)}"}
                # 继续重试
                continue
        
        # 理论上不会到这里
        return {"error": "ReAct处理异常终止"}
    
    def _build_validation_feedback(self, errors: List[str]) -> str:
        """
        构建验证失败的反馈消息
        
        Args:
            errors: 错误信息列表
            
        Returns:
            反馈消息
        """
        if not errors:
            return "输出格式有误，请按要求重新生成。"
        
        feedback_parts = ["你的输出存在以下问题，请修正后重新生成：\n"]
        for i, error in enumerate(errors, 1):
            feedback_parts.append(f"{i}. {error}")
        
        feedback_parts.append("\n请仔细检查并重新输出正确的结果。")
        return "\n".join(feedback_parts)
    
    # ==================== 原有方法 ============================================================================================================================================
    async def _execute_react_graph(self, state: MainState, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        自动构建和执行ReAct子图
        
        Args:
            state: 主状态
            pre_tool_results: 前置工具结果
            
        Returns:
            执行结果
        """
        from langgraph.graph import StateGraph
        from langgraph.prebuilt import ToolNode, tools_condition
        
        log.info(f"开始构建 {self.role_name} 的子图...")
        
        # 1. 获取后置工具
        post_tools = self.get_post_tools()
        if not post_tools:
            log.warning(f"{self.role_name} 没有后置工具，回退到简单模式")
            return await self.process_simple_mode(state, pre_tool_results)
                
        # 2. 使用 DFState 作为子图状态（与现有代码一致）
        log.critical(f"state: {state.agent_results}")

        subgraph = StateGraph(type(state))
        
        # 3. 直接复用 create_assistant_node_func ， 这里没有写历史存储的代码；
        assistant_func = self.create_assistant_node_func(state, pre_tool_results)
        
        # 4. 添加节点
        subgraph.add_node("assistant", assistant_func)
        subgraph.add_node("tools", ToolNode(post_tools))
        
        # 5. 添加边
        subgraph.add_conditional_edges("assistant", tools_condition)
        subgraph.add_edge("tools", "assistant")
        
        # 6. 设置入口点
        subgraph.set_entry_point("assistant")
        
        # 7. 编译并执行
        compiled_graph = subgraph.compile()
        log.info(f"{self.role_name} 子图编译完成")
        
        try:
            # 执行子图，传入当前 state
            final_state = await compiled_graph.ainvoke(state)
            
            log.info(f"{self.role_name} 子图执行完成")
            
            # 8. 从 final_state 中提取结果
            # create_assistant_node_func 已经把结果放到 state.agent_results 中了
            result = final_state["agent_results"].get(self.role_name.lower(), {}).get("results", {})
            
            if not result:
                log.error("子图执行后未找到结果")
                return {"error": "子图执行异常：未找到结果"}
            
            # 如果需要 react mode 验证，这里可以加react对LLM Resp的

            # if self.react_mode:
            #     # 从 messages 中获取最后的 AI 响应
            #     messages = final_state.get("messages", [])
            #     if messages:
            #         last_msg = messages[-1]
            #         if hasattr(last_msg, 'content'):
            #             all_passed, errors = self._run_validators(last_msg.content, result)
            #             if not all_passed:
            #                 log.warning(f"ReAct 验证未通过: {errors}")
            #                 result["validation_errors"] = errors
            
            log.info(f"{self.role_name} 子图结果解析完成")
            return result
            
        except Exception as e:
            log.exception(f"{self.role_name} 子图执行失败: {e}")
            return {"error": f"子图执行失败: {str(e)}"}



    def store_outputs(self, data, file_name: str = None) -> str:
        """保存输出结果到文件"""
        out_dir = f"{PROJDIR}/outputs/{self.role_name.lower()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if not file_name:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{ts}.pkl"
        
        file_path = out_dir / file_name
        
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        log.info(f"已保存->: {file_path}")
        return str(file_path)
    
    async def execute_pre_tools(self, state: MainState) -> Dict[str, Any]:
        """执行前置工具"""
        log.info(f"开始执行 {self.role_name} 的前置工具...")
        
        if not self.tool_manager:
            log.info("未提供工具管理器，使用默认值")
            return self.get_default_pre_tool_results()
        
        results = await self.tool_manager.execute_pre_tools(self.role_name)
        
        # 设置默认值
        defaults = self.get_default_pre_tool_results()
        for key, default_value in defaults.items():
            if key not in results or results[key] is None:
                results[key] = default_value
                
        log.info(f"前置工具执行完成，获得: {list(results.keys())}")
        return results
    
    # def build_messages(self, 
    #                   state: MainState, 
    #                   pre_tool_results: Dict[str, Any]) -> List[BaseMessage]:
    #     """构建消息列表"""
    #     log.info("构建提示词消息...")
        
    #     ptg = PromptsTemplateGenerator(state.request.language)
    #     sys_prompt = ptg.render(self.system_prompt_template_name)
        
    #     task_params = self.get_task_prompt_params(pre_tool_results)
    #     task_prompt = ptg.render(self.task_prompt_template_name, **task_params)
    #     # log.info(f"系统提示词: {sys_prompt}")
    #     log.debug(f"[base agent]: 任务提示词: {task_prompt}")
        
    #     messages = [
    #         SystemMessage(content=sys_prompt),
    #         HumanMessage(content=task_prompt),
    #     ]
        
    #     log.info("提示词消息构建完成")
    #     return messages

    # 多模态版本
    def build_messages(self, 
                    state: MainState, 
                    pre_tool_results: Dict[str, Any]) -> List[BaseMessage]:
        """构建消息列表 - 添加格式说明"""
        log.info("构建提示词消息...")
        
        ptg = PromptsTemplateGenerator(state.request.language)
        sys_prompt = ptg.render(self.system_prompt_template_name)
        
        # 添加解析器格式说明
        format_instruction = self.parser.get_format_instruction()
        if format_instruction and not self.use_vlm:  # VLM模式可能不需要格式说明
            sys_prompt += f"\n\n{format_instruction}"
        
        task_params = self.get_task_prompt_params(pre_tool_results)
        task_prompt = ptg.render(self.task_prompt_template_name, **task_params)
        # log.info(f"系统提示词: {sys_prompt}")
        log.info(f"[build_messages]任务提示词: {task_prompt}")
        
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=task_prompt),
        ]
        
        log.info("提示词消息构建完成")
        return messages

    def create_llm(self, state: MainState, bind_post_tools: bool = False) -> ChatOpenAI:
        """创建LLM实例"""
        actual_model = self.model_name or state.request.model
        actual_url = self.chat_api_url or state.request.chat_api_url
        log.info(f"[create_llm:]创建LLM实例，温度: {self.temperature}, 最大token: {self.max_tokens}, 模型: {actual_model}, 接口URL: {actual_url}, API Key: {state.request.api_key}")
        llm = ChatOpenAI(
            openai_api_base=actual_url,
            openai_api_key=state.request.api_key,
            model_name=actual_model,
            temperature=self.temperature,
            # max_tokens=self.max_tokens,
        )
        
        if bind_post_tools and self.tool_manager:
            post_tools = self.get_post_tools()
            if post_tools:
                llm = llm.bind_tools(post_tools, tool_choice=self.tool_mode)
                log.info(f"[create_llm]:为LLM绑定了 {len(post_tools)} 个后置工具: {[t.name for t in post_tools]}")
        return llm

    # def create_llm(self, state: MainState, bind_post_tools: bool = False) -> ChatOpenAI:
    #     """创建LLM实例"""
    #     import httpx
    #     actual_model = self.model_name or state.request.model
    #     actual_url = self.chat_api_url or state.request.chat_api_url
    #     log.info(f"[create_llm:]创建LLM实例，模型: {actual_model}")
        
    #     log.info(f"[create_llm:]创建LLM实例，温度: {self.temperature}, 最大token: {self.max_tokens}, 模型: {actual_model}, 接口URL: {actual_url}, API Key: {state.request.api_key}")

    #     # 1. 复用你之前的逻辑：配置超时和 SSL
    #     # 注意：这里不需要 async with，因为 ChatOpenAI 会接管客户端的生命周期
    #     timeout = 120.0 # 或者从 self.vlm_config 获取，保持一致性

    #     # 2. 创建干净的异步客户端 (针对报错: _AsyncHttpxClientWrapper)
    #     clean_async_client = httpx.AsyncClient(
    #         timeout=httpx.Timeout(timeout),
    #         proxies=None,        # === 关键修复：显式禁用代理 ===
    #         trust_env=False,     # === 关键修复：忽略环境变量 ===
    #         verify=False,        # 保持和你之前代码一致，关闭 SSL 验证（如需）
    #         follow_redirects=True
    #     )

    #     # 3. 创建干净的同步客户端 (防止同步调用时出错)
    #     clean_sync_client = httpx.Client(
    #         timeout=httpx.Timeout(timeout),
    #         proxies=None,
    #         trust_env=False,
    #         verify=False,
    #         follow_redirects=True
    #     )

    #     llm = ChatOpenAI(
    #         base_url=actual_url,
    #         api_key=state.request.api_key,
    #         model_name=actual_model,
    #         temperature=self.temperature,
    #         # === 注入修复后的客户端 ===
    #         http_client=clean_sync_client,       # 覆盖同步调用
    #         http_async_client=clean_async_client # 覆盖异步调用 (解决 AttributeError 问题)
    #         # ========================
    #     )


    #     log.critical(f"ChatOpenAI创建完成！")
        
    #     if bind_post_tools and self.tool_manager:
    #         post_tools = self.get_post_tools()
    #         if post_tools:
    #             llm = llm.bind_tools(post_tools, tool_choice=self.tool_mode)
    #             log.info(f"[create_llm]:为LLM绑定了 {len(post_tools)} 个后置工具: {[t.name for t in post_tools]}")
    #     return llm
    
    def get_post_tools(self) -> List[Tool]:
        if not self.tool_manager:
            return []
        tools = self.tool_manager.get_post_tools(self.role_name)
        uniq, seen = [], set()
        for t in tools:
            if t.name not in seen:
                uniq.append(t)
                seen.add(t.name)
        return uniq
    
    async def process_simple_mode(self, state: MainState, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """简单模式处理"""
        log.info(f"执行 {self.role_name} 简单模式...")
        
        messages = self.build_messages(state, pre_tool_results)
        
        # 消息历史管理
        if not self.ignore_history:
            history_messages = self.message_history.get_messages()
            if history_messages:
                messages = self.message_history.merge_histories(history_messages, messages)
                log.info(f"合并了 {len(history_messages)} 条历史消息")
        
        llm = self.create_llm(state, bind_post_tools=False)
        
        try:
            answer_msg = await llm.ainvoke(messages)
            answer_text = answer_msg.content
            log.info(f'LLM原始输出：{answer_text}')
            log.info("LLM调用成功，开始解析结果")
            
            # 更新消息历史
            if not self.ignore_history:
                self.message_history.add_messages([answer_msg])
                log.info("已更新消息历史")
                
        except Exception as e:
            log.exception("LLM调用失败: %s", e)
            return {"error": str(e)}
        
        return self.parse_result(answer_text)
    
    async def process_with_llm_for_graph(self, messages: List[BaseMessage], state: MainState) -> BaseMessage:
        llm = self.create_llm(state, bind_post_tools=True)
        try:
            response = await llm.ainvoke(messages)
            log.info(response)
            log.info(f"{self.role_name} 图模式LLM调用成功")
            return response
        except Exception as e:
            log.exception(f"{self.role_name} 图模式LLM调用失败: {e}")
            raise
    
    def has_tool_calls(self, message: BaseMessage) -> bool:
        """检查消息是否包含工具调用"""
        return hasattr(message, 'tool_calls') and bool(getattr(message, 'tool_calls', None))
    
    def update_state_result(self, state: MainState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
        """
        更新状态结果 - 子类可重写以自定义状态更新逻辑
        
        Args:
            state: 状态对象
            result: 处理结果
            pre_tool_results: 前置工具结果
        """
        # 默认行为：将结果存储到与角色名对应的属性中
        setattr(state, self.role_name.lower(), result)
        state.agent_results[self.role_name] = {
            "pre_tool_results": pre_tool_results,
            "post_tools": [t.name for t in self.get_post_tools()],
            "results": result
        }
# =================================================================================execute 部分！ 核心！！！=================================================================================
    async def execute(self, state: MainState, use_agent: bool = False, **kwargs) -> MainState:
        """
        统一执行入口
        
        Args:
            state: DFState实例
            use_agent: 是否使用代理模式（图模式）
            **kwargs: 额外参数
            
        Returns:
            更新后的DFState
        """
        # 获取一些状态信息
        self.state = state
        # 如果配置了执行策略，优先使用
        if self._execution_strategy:
            log.info(f"使用策略模式执行: {self._execution_strategy.__class__.__name__}")
            try:
                pre_tool_results = await self.execute_pre_tools(state)
                result = await self._execution_strategy.execute(state, **kwargs)
                self.update_state_result(state, result, pre_tool_results)
                return state
            except Exception as e:
                log.exception(f"策略执行失败: {e}")
                error_result = {"error": str(e)}
                self.update_state_result(state, error_result, {})
                return state
            
        # ================之前的代码，非策略支持部分 ========================
        log.info(f"开始执行 {self.role_name} (ReAct模式: {self.react_mode}, 图模式: {use_agent})")

        if getattr(self, "use_vlm", False):
            # 直接走多模态链路
            log.critical(f'[base agent]: 走多模态路径 ')
            result = await self._execute_vlm(state, **kwargs)
            # 和简单/React 逻辑保持一致，更新到 state
            self.update_state_result(state, result, {})
            log.info(f"{self.role_name} 多模态执行完成")
            return state
        try:
            # 1. 执行前置工具
            pre_tool_results = await self.execute_pre_tools(state)
            # 1.1 统一写入 temp_data，确保简单模式/图模式均可用
            try:
                if not hasattr(state, 'temp_data') or state.temp_data is None:
                    state.temp_data = {}
                state.temp_data['pre_tool_results'] = pre_tool_results
            except Exception:
                pass
            # 合并 kwargs 到前置工具结果中， 这一步是 agent as tool 时，需要将 kwargs 作为前置工具的输入； 也可以在正常调用时，将 kwargs 作为前置工具的结果，比如 {"content": "xxxx报告"}
            pre_tool_results.update(kwargs)
            # 2. 检查是否有后置工具
            post_tools = self.get_post_tools()
            
            # 3. 根据模式和工具情况选择处理方式
            if use_agent and post_tools:
                log.info(f"[子图新模式] 自动构建 {self.role_name} 的子图，后置工具: {[t.name for t in post_tools]}")
                result = await self._execute_react_graph(state, pre_tool_results)
                self.update_state_result(state, result, pre_tool_results)
                log.info(f"[子图新模式] {self.role_name} 子图模式执行完成")
                if not hasattr(state, 'temp_data'):
                    state.temp_data = {}
                state.temp_data['pre_tool_results'] = pre_tool_results
                state.temp_data[f'{self.role_name}_instance'] = self
                
            elif self.react_mode:
                # ReAct模式 - 带验证的循环调用
                log.info("ReAct模式 - 带验证循环")
                result = await self.process_react_mode(state, pre_tool_results)
                self.update_state_result(state, result, pre_tool_results)
                log.info(f"{self.role_name} ReAct模式执行完成")
                
            else:
                # 简单模式 - 单次调用
                if use_agent and not post_tools:
                    log.info("图模式无可用后置工具，回退到简单模式")
                result = await self.process_simple_mode(state, pre_tool_results)
                self.update_state_result(state, result, pre_tool_results)
                log.info(f"{self.role_name} 简单模式执行完成")
            
        except Exception as e:
            log.exception(f"{self.role_name} 执行失败: {e}")
            import traceback
            traceback.print_exc()
            error_result = {"error": str(e)}
            self.update_state_result(state, error_result, {})
            
        return state
    
    async def _execute_vlm(self,
                        state: MainState,
                        **kwargs) -> Dict[str, Any]:
        """
        Vision-LLM 专用执行流程
        与文本链路完全解耦，不碰原有 llm / process_simple_mode 等函数
        """
        # 1. 前置工具
        pre_tool_results = await self.execute_pre_tools(state)
    
        # 2. 构建消息（若是图像生成/编辑仅用 prompt 就行，
        #    若是图像理解可和文本一样）——示例给两种典型写法:
        mode = self.vlm_config.get("mode", "understanding")
    
        # if mode in {"generation", "edit"}:
        #     # 只需要最后一条 prompt
        #     messages = [
        #         HumanMessage(content=self.build_generation_prompt(pre_tool_results))
        #     ]
        # else:
        messages = self.build_messages(state, pre_tool_results)
    
        # 3. 调用 VisionLLMCaller
        from dataflow_agent.llm_callers import VisionLLMCaller
        vlm_caller = VisionLLMCaller(
            state,
            vlm_config=self.vlm_config,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tool_mode=self.tool_mode,
            tool_manager=self.tool_manager,
        )
        response = await vlm_caller.call(messages)
        log.info(f"{self.role_name} 多模态原始响应: {response}")
    
        # 4. 解析
        parsed = self.parse_result(response.content)
    
        # 5. 如有附加信息（例如图像路径 / base64），一起返回
        if hasattr(response, "additional_kwargs"):
            log.info(f"{self.role_name} 多模态附加信息: {response.additional_kwargs}")
            
            if isinstance(parsed, dict):
                parsed.update(response.additional_kwargs)
            elif isinstance(parsed, list):
                log.warning(f"{self.role_name} parsed 是列表类型，无法调用 update 方法，跳过附加参数合并")
                log.info(f"parsed 类型: {type(parsed).__name__}, 内容: {parsed}")
                log.info(f"additional_kwargs 内容: {response.additional_kwargs}")
            else:
                log.warning(f"{self.role_name} parsed 是 {type(parsed).__name__} 类型，无法调用 update 方法，跳过附加参数合并")
                log.info(f"parsed 内容: {parsed}")
                log.info(f"additional_kwargs 内容: {response.additional_kwargs}")
        
        return parsed
# 这个暂时用不到
    def build_generation_prompt(self, pre_tool_results: Dict[str, Any]) -> str:
        """示例：把工具结果拼进 prompt"""
        return (
            f"{self.vlm_config.get('prompt', '')}"
        )
    
    def create_assistant_node_func(self, state: MainState, pre_tool_results: Dict[str, Any]):
        async def assistant_node(graph_state):
            messages = graph_state.get("messages", [])
            if not messages:
                messages = self.build_messages(state, pre_tool_results)
                log.info(f"构建 {self.role_name} 初始消息，包含前置工具结果")

            response = await self.process_with_llm_for_graph(messages, state)

            if self.has_tool_calls(response):
                log.info(f"[create_assistant_node_func]: {self.role_name} LLM选择调用工具: ...")
                return {"messages": messages + [response]}
            else:
                log.info(f"[create_assistant_node_func]: {self.role_name} LLM本次未调用工具，解析最终结果")
                result = self.parse_result(response.content)
                # **这里同步了一次 agent_results**
                state.agent_results[self.role_name.lower()] = {
                    "pre_tool_results": pre_tool_results,
                    "post_tools": [t.name for t in self.get_post_tools()],
                    "results": result
                }
                return {
                    "messages": messages + [response],
                    self.role_name.lower(): result,
                    "finished": True
                }
        return assistant_node
    
# # ---------------------------------- 在 BaseAgent 顶部添加 --------------------------
# from langchain_core.messages import AIMessage, BaseMessage

# # ---------------------------------- 添加统一调用助手 ------------------------------
# async def _call_llm(self,
#                     messages: List[BaseMessage],
#                     state: MainState,
#                     bind_post_tools: bool = False) -> AIMessage:
#     if self.use_vlm:
#         caller = self.get_llm_caller(state)       # VisionLLMCaller
#         return await caller.call(messages)

#     llm = self.create_llm(state, bind_post_tools=bind_post_tools)
#     return await llm.ainvoke(messages)
# # -------------------------------------------------------------------------------

# # process_simple_mode / process_react_mode 内全部换成：
# answer_msg = await self._call_llm(messages, state, bind_post_tools=False)