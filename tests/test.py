from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Callable, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langchain_core.tools import Tool
from pydantic import BaseModel, Field

from dataflow_agent.promptstemplates.prompt_template import PromptsTemplateGenerator
from dataflow_agent.state import DFState
from dataflow_agent.utils import robust_parse_json
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow import get_logger

log = get_logger()

# 验证器类型定义：返回 (是否通过, 错误信息)
ValidatorFunc = Callable[[str, Dict[str, Any]], Tuple[bool, Optional[str]]]


class BaseAgent(ABC):
    """Agent基类 - 定义通用的agent执行模式"""
    
    def __init__(self, 
                 tool_manager: Optional[ToolManager] = None,
                 model_name: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 tool_mode: str = "auto",
                 react_mode: bool = False,
                 react_max_retries: int = 3):
        """
        初始化Agent
        
        Args:
            tool_manager: 工具管理器
            model_name: 模型名称
            temperature: 模型温度
            max_tokens: 最大token数
            tool_mode: 工具选择模式 ("auto", "required", "none")
            react_mode: 是否启用ReAct模式
            react_max_retries: ReAct模式最大重试次数
        """
        self.tool_manager = tool_manager
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_mode = tool_mode
        self.react_mode = react_mode
        self.react_max_retries = react_max_retries

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
    
    def parse_result(self, content: str) -> Dict[str, Any]:
        """
        解析结果 - 子类可重写自定义解析逻辑
        
        Args:
            content: LLM输出内容
            
        Returns:
            解析后的结果
        """
        try:
            parsed = robust_parse_json(content)
            log.info(f'content是什么？？{content}')
            log.info(f"{self.role_name} 结果解析成功")
            return parsed
        except ValueError as e:
            log.warning(f"JSON解析失败: {e}")
            return {"raw": content}
        except Exception as e:
            log.warning(f"解析过程出错: {e}")
            return {"raw": content}
    
    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """获取默认前置工具结果 - 子类可重写"""
        return {}
    
    # ==================== Agent-as-Tool 功能 ====================
    
    def get_tool_name(self) -> str:
        """
        获取作为工具时的名称 - 子类可重写
        
        Returns:
            工具名称
        """
        return f"call_{self.role_name.lower()}_agent"
    
    def get_tool_description(self) -> str:
        """
        获取作为工具时的描述 - 子类应该重写提供更具体的描述
        
        Returns:
            工具描述
        """
        return f"调用 {self.role_name} agent 来执行特定任务。该 agent 会根据输入参数执行相应的分析和处理。"
    
    def get_tool_args_schema(self) -> Type[BaseModel]:
        """
        获取作为工具时的参数模式 - 子类可以重写定义自己的参数结构
        
        Returns:
            Pydantic BaseModel 类型
        """
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
        """
        准备工具执行时的参数 - 子类可重写自定义参数处理逻辑
        
        Args:
            **tool_kwargs: 工具调用时传入的参数
            
        Returns:
            处理后的参数字典，会被传递给 execute 方法
        """
        # 默认实现：提取所有参数
        params = {}
        
        # 如果有 additional_params，展开它
        if 'additional_params' in tool_kwargs and tool_kwargs['additional_params']:
            params.update(tool_kwargs['additional_params'])
        
        # 其他参数也添加进去
        for key, value in tool_kwargs.items():
            if key != 'additional_params':
                params[key] = value
        
        return params
    
    def extract_tool_result(self, state: DFState) -> Dict[str, Any]:
        """
        从状态中提取工具调用的结果 - 子类可重写自定义结果提取逻辑
        
        Args:
            state: 执行后的状态
            
        Returns:
            要返回给调用方的结果
        """
        # 默认从 agent_results 中提取
        agent_result = state.agent_results.get(self.role_name, {})
        return agent_result.get('results', {})
    
    async def _execute_as_tool(self, state: DFState, **tool_kwargs) -> Dict[str, Any]:
        """
        作为工具执行的内部方法
        
        Args:
            state: DFState 实例
            **tool_kwargs: 工具调用参数
            
        Returns:
            执行结果
        """
        try:
            log.info(f"[Agent-as-Tool] 调用 {self.role_name}，参数: {tool_kwargs}")
            
            # 准备执行参数
            exec_params = self.prepare_tool_execution_params(**tool_kwargs)
            
            # 执行 agent（通常使用简单模式，不使用图模式）
            result_state = await self.execute(
                state, 
                use_agent=False,  # 工具模式下通常不使用图模式
                **exec_params
            )
            
            # 提取结果
            result = self.extract_tool_result(result_state)
            
            log.info(f"[Agent-as-Tool] {self.role_name} 执行完成，结果: {list(result.keys())}")
            return result
            
        except Exception as e:
            log.exception(f"[Agent-as-Tool] {self.role_name} 执行失败: {e}")
            return {
                "error": str(e),
                "agent": self.role_name,
                "status": "failed"
            }
    
    def as_tool(self, state: DFState) -> Tool:
        """
        将 agent 包装成可被调用的工具
        
        Args:
            state: DFState 实例，会被传递给执行方法
            
        Returns:
            Tool 对象，可以被其他 agent 或 LLM 调用
            
        Example:
            ```python
            # 在一个 agent 中调用另一个 agent
            analyzer_agent = AnalyzerAgent.create()
            analyzer_tool = analyzer_agent.as_tool(state)
            
            # 将该工具添加到工具管理器或直接使用
            tool_manager.add_tool(analyzer_tool)
            ```
        """
        # 创建异步执行函数
        async def agent_tool_func(**kwargs) -> Dict[str, Any]:
            return await self._execute_as_tool(state, **kwargs)
        
        # 创建同步包装（如果需要同步调用）
        def sync_agent_tool_func(**kwargs) -> Dict[str, Any]:
            return asyncio.run(agent_tool_func(**kwargs))
        
        return Tool(
            name=self.get_tool_name(),
            description=self.get_tool_description(),
            func=sync_agent_tool_func,  # 同步版本
            coroutine=agent_tool_func,   # 异步版本
            args_schema=self.get_tool_args_schema()
        )
    
    def register_as_tool(self, state: DFState, tool_manager: Optional[ToolManager] = None) -> Tool:
        """
        将自己注册为工具到工具管理器
        
        Args:
            state: DFState 实例
            tool_manager: 工具管理器，如果为 None 则使用 self.tool_manager
            
        Returns:
            创建的 Tool 对象
            
        Example:
            ```python
            # 创建 agent 并注册为工具
            analyzer_agent = AnalyzerAgent.create(tool_manager=tm)
            analyzer_tool = analyzer_agent.register_as_tool(state)
            
            # 现在其他 agent 可以通过工具管理器调用这个 agent
            ```
        """
        target_tm = tool_manager or self.tool_manager
        if not target_tm:
            raise ValueError("需要提供 tool_manager 参数或在初始化时设置 tool_manager")
        
        tool = self.as_tool(state)
        
        # 注册到工具管理器
        # 假设 ToolManager 有 add_tool 或类似方法
        if hasattr(target_tm, 'add_tool'):
            target_tm.add_tool(tool)
            log.info(f"[Agent-as-Tool] {self.role_name} 已注册为工具: {tool.name}")
        else:
            log.warning(f"ToolManager 没有 add_tool 方法，无法自动注册")
        
        return tool
    
    # ==================== ReAct 模式相关方法 ====================
    
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
    
    async def process_react_mode(self, state: DFState, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
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
                    return parsed_result
                
                # 验证未通过
                if attempt < self.react_max_retries:
                    # 构建反馈消息
                    feedback = self._build_validation_feedback(errors)
                    log.warning(f"✗ 验证未通过 (尝试 {attempt + 1}): {feedback}")
                    
                    # 添加LLM的回复和人类的反馈到消息列表
                    messages.append(AIMessage(content=answer_text))
                    messages.append(HumanMessage(content=feedback))
                else:
                    # 达到最大重试次数
                    log.error(f"✗ {self.role_name} ReAct达到最大重试次数，验证仍未通过")
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
    
    # ==================== 原有方法 ====================
    
    async def execute_pre_tools(self, state: DFState) -> Dict[str, Any]:
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
    
    def build_messages(self, 
                      state: DFState, 
                      pre_tool_results: Dict[str, Any]) -> List[BaseMessage]:
        """构建消息列表"""
        log.info("构建提示词消息...")
        
        ptg = PromptsTemplateGenerator(state.request.language)
        sys_prompt = ptg.render(self.system_prompt_template_name)
        
        task_params = self.get_task_prompt_params(pre_tool_results)
        task_prompt = ptg.render(self.task_prompt_template_name, **task_params)
        # log.info(f"系统提示词: {sys_prompt}")
        log.info(f"任务提示词: {task_prompt}")
        
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=task_prompt),
        ]
        
        log.info("提示词消息构建完成")
        return messages
    
    def create_llm(self, state: DFState, bind_post_tools: bool = False) -> ChatOpenAI:
        """创建LLM实例"""
        actual_model = self.model_name or state.request.model
        log.info(f"创建LLM实例，模型: {actual_model}")
        
        llm = ChatOpenAI(
            openai_api_base=state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=actual_model,
            temperature=self.temperature,
            # max_tokens=self.max_tokens,
        )
        
        if bind_post_tools and self.tool_manager:
            post_tools = self.get_post_tools()
            if post_tools:
                llm = llm.bind_tools(post_tools, tool_choice=self.tool_mode)
                log.info(f"为LLM绑定了 {len(post_tools)} 个后置工具: {[t.name for t in post_tools]}")
        return llm
    
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
    
    async def process_simple_mode(self, state: DFState, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """简单模式处理"""
        log.info(f"执行 {self.role_name} 简单模式...")
        
        messages = self.build_messages(state, pre_tool_results)
        llm = self.create_llm(state, bind_post_tools=False)
        
        try:
            answer_msg = await llm.ainvoke(messages)
            answer_text = answer_msg.content
            log.info(f'LLM原始输出：{answer_text}')
            log.info("LLM调用成功，开始解析结果")
        except Exception as e:
            log.exception("LLM调用失败: %s", e)
            return {"error": str(e)}
        
        return self.parse_result(answer_text)
    
    async def process_with_llm_for_graph(self, messages: List[BaseMessage], state: DFState) -> BaseMessage:
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
    
    def update_state_result(self, state: DFState, result: Dict[str, Any], pre_tool_results: Dict[str, Any]):
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
    
    async def execute(self, state: DFState, use_agent: bool = False, **kwargs) -> DFState:
        """
        统一执行入口
        
        Args:
            state: DFState实例
            use_agent: 是否使用代理模式（图模式）
            **kwargs: 额外参数
            
        Returns:
            更新后的DFState
        """
        log.info(f"开始执行 {self.role_name} (ReAct模式: {self.react_mode}, 图模式: {use_agent})")
        
        try:
            # 1. 执行前置工具
            pre_tool_results = await self.execute_pre_tools(state)
            
            # 2. 检查是否有后置工具
            post_tools = self.get_post_tools()
            
            # 3. 根据模式和工具情况选择处理方式
            if use_agent and post_tools:
                # 图模式 - 需要外部图构建器处理
                log.info("图模式 - 需要外部图构建器处理，暂存必要数据")
                if not hasattr(state, 'temp_data'):
                    state.temp_data = {}
                state.temp_data['pre_tool_results'] = pre_tool_results
                state.temp_data[f'{self.role_name}_instance'] = self
                log.info(f"已暂存前置工具结果和 {self.role_name} 实例，后置工具: {[t.name for t in post_tools]}")
                
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
            error_result = {"error": str(e)}
            self.update_state_result(state, error_result, {})
            
        return state
    
    def create_assistant_node_func(self, state: DFState, pre_tool_results: Dict[str, Any]):
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