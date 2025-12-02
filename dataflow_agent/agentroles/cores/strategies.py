from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING, List
from dataflow_agent.logger import get_logger
import asyncio  # 添加导入

if TYPE_CHECKING:
    from dataflow_agent.agentroles.cores.base_agent import BaseAgent
    from dataflow_agent.state import MainState

log = get_logger(__name__)


class ExecutionStrategy(ABC):
    """执行策略基类"""
    
    def __init__(self, agent: "BaseAgent", config: Any):
        self.agent = agent
        self.config = config
    
    @abstractmethod
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        """执行策略的核心方法"""
        pass


class SimpleStrategy(ExecutionStrategy):
    """简单模式策略"""
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[SimpleStrategy] 执行 {self.agent.role_name}")
        pre_tool_results = await self.agent.execute_pre_tools(state)
        result = await self.agent.process_simple_mode(state, pre_tool_results)
        return result


class ReactStrategy(ExecutionStrategy):
    """ReAct模式策略"""
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[ReactStrategy] 执行 {self.agent.role_name}，最大重试: {self.config.max_retries}")
        
        # 注入自定义验证器
        if self.config.validators:
            original_validators = self.agent.get_react_validators
            def custom_validators():
                return original_validators() + self.config.validators
            self.agent.get_react_validators = custom_validators
        
        pre_tool_results = await self.agent.execute_pre_tools(state)
        
        # 临时覆盖 react_max_retries
        original_retries = self.agent.react_max_retries
        self.agent.react_max_retries = self.config.max_retries
        
        result = await self.agent.process_react_mode(state, pre_tool_results)
        
        self.agent.react_max_retries = original_retries
        return result


class GraphStrategy(ExecutionStrategy):
    """图模式策略"""
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[GraphStrategy] 执行 {self.agent.role_name} 子图模式")
        pre_tool_results = await self.agent.execute_pre_tools(state)
        
        post_tools = self.agent.get_post_tools()
        if not post_tools:
            log.warning("无后置工具，回退到简单模式")
            return await self.agent.process_simple_mode(state, pre_tool_results)
        
        result = await self.agent._execute_react_graph(state, pre_tool_results)
        return result


class VLMStrategy(ExecutionStrategy):
    """视觉语言模型策略"""
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[VLMStrategy] 执行 {self.agent.role_name} VLM模式: {self.config.vlm_mode}")
        
        # 构建 VLM 配置字典
        vlm_config = {
            "mode": self.config.vlm_mode,
            "image_detail": self.config.image_detail,
            "max_image_size": self.config.max_image_size,
            **self.config.additional_params
        }
        
        # 临时注入配置
        # original_vlm_config = getattr(self.agent, 'vlm_config', {})

        self.agent.vlm_config.update(vlm_config)
        self.agent.model_name = self.config.model_name
        self.agent.temperature = self.config.temperature
        self.agent.max_tokens = self.config.max_tokens

        log.info(f"VLMStrategy 执行 {self.agent.role_name} VLM模式: {self.agent.vlm_config} + {kwargs},self.config={self.config}")
        result = await self.agent._execute_vlm(state, **kwargs)
        log.critical(f"VLMStrategy 执行 {self.agent.role_name} VLM模式结果: {result}")
        
        # self.agent.vlm_config = original_vlm_config
        return result


class ParallelStrategy(ExecutionStrategy):
    """并行模式策略"""
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        log.info(f"[ParallelStrategy] 执行 {self.agent.role_name}，并行度限制: {self.config.concurrency_limit}")
        
        # 执行前置工具获取结果
        pre_tool_results = await self.agent.execute_pre_tools(state)
        
        # 检查是否有parallel_items，如果没有且pre_tool_results是列表，自动转换为parallel_items
        if "parallel_items" not in pre_tool_results and isinstance(pre_tool_results, list):
            pre_tool_results = {"parallel_items": pre_tool_results}
        
        # 执行并行模式
        return await self.agent.process_parallel_mode(state, pre_tool_results)


# ==================== Planning Agent 策略 ====================

class PlanSolveStrategy(ExecutionStrategy):
    """
    Plan-and-Solve 策略
    
    一次性生成完整计划，然后按顺序执行，不回头调整。
    
    流程:
    1. 调用 Planner 生成完整计划
    2. (可选) Human-in-the-Loop: 等待用户审批计划
    3. 按顺序执行每个步骤
    4. 收集结果并返回
    """
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        from dataflow_agent.state import PlanningState
        
        log.info(f"[PlanSolveStrategy] 执行 {self.agent.role_name}")
        
        # 确保状态类型正确
        if not isinstance(state, PlanningState):
            log.warning("状态不是 PlanningState 类型，可能会缺少某些功能")
        
        # Step 1: 生成计划
        log.info("[PlanSolveStrategy] Step 1: 生成计划")
        plan = await self._generate_plan(state)
        
        if not plan:
            return {"error": "无法生成计划", "status": "failed"}
        
        log.info(f"[PlanSolveStrategy] 生成了 {len(plan)} 步计划")
        
        # 更新状态
        if hasattr(state, 'plan'):
            state.plan = plan
        
        # Step 2: Human-in-the-Loop - 计划审批
        if self.config.require_plan_approval:
            log.info("[PlanSolveStrategy] Step 2: 等待用户审批计划")
            # 使用 LangGraph 的 interrupt 机制
            from langgraph.types import interrupt
            
            approval_response = interrupt({
                "type": "plan_approval",
                "message": "请审批以下计划:",
                "plan": plan,
                "options": ["approve", "reject", "modify"]
            })
            
            if approval_response.get("action") == "reject":
                return {
                    "status": "rejected",
                    "message": "计划被用户拒绝",
                    "plan": plan
                }
            elif approval_response.get("action") == "modify":
                plan = approval_response.get("modified_plan", plan)
                if hasattr(state, 'plan'):
                    state.plan = plan
        
        # 标记计划已审批
        if hasattr(state, 'plan_approved'):
            state.plan_approved = True
        
        # Step 3: 按顺序执行计划
        log.info("[PlanSolveStrategy] Step 3: 执行计划")
        results = []
        
        for i, step in enumerate(plan):
            log.info(f"[PlanSolveStrategy] 执行步骤 {i+1}/{len(plan)}: {step[:50]}...")
            
            try:
                step_result = await self._execute_step(state, step, i)
                results.append({
                    "step_index": i,
                    "step": step,
                    "result": step_result,
                    "status": "completed"
                })
                
                # 更新状态
                if hasattr(state, 'mark_step_complete'):
                    state.mark_step_complete(str(step_result))
                    
            except Exception as e:
                log.error(f"[PlanSolveStrategy] 步骤 {i+1} 执行失败: {e}")
                results.append({
                    "step_index": i,
                    "step": step,
                    "error": str(e),
                    "status": "failed"
                })
                
                if not self.config.continue_on_error:
                    break
        
        # Step 4: 返回结果
        return {
            "status": "completed",
            "plan": plan,
            "results": results,
            "total_steps": len(plan),
            "completed_steps": sum(1 for r in results if r.get("status") == "completed")
        }
    
    async def _generate_plan(self, state: "MainState") -> List[str]:
        """
        使用 LLM 生成计划
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, Field
        
        # 获取任务描述
        task = state.request.target if hasattr(state, 'request') else ""
        if hasattr(state, 'original_task') and state.original_task:
            task = state.original_task
        
        # 获取可用工具信息
        tools_info = ""
        if hasattr(state, 'executor_tools') and state.executor_tools:
            tools_info = f"\n可用工具: {', '.join(state.executor_tools)}"
        
        # 构建 Planner 的提示词
        system_prompt = """你是一个任务规划专家。你的职责是分析用户的任务，并生成一个清晰、可执行的分步计划。

规则:
1. 每个步骤应该是独立的、可执行的操作
2. 步骤之间应该有逻辑顺序
3. 步骤描述应该清晰具体
4. 步骤数量不宜过多，通常 3-7 步为宜
5. 不要包含无关的步骤"""

        task_prompt = f"""请为以下任务生成执行计划:

任务: {task}
{tools_info}

请以 JSON 格式返回计划，格式如下:
{{"steps": ["步骤1描述", "步骤2描述", ...]}}"""

        # 创建 LLM
        planner_model = self.config.planner_model or self.config.model_name or state.request.model
        llm = ChatOpenAI(
            openai_api_base=self.config.chat_api_url or state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=planner_model,
            temperature=self.config.planner_temperature,
        )
        
        # 使用结构化输出
        class PlanOutput(BaseModel):
            steps: List[str] = Field(description="计划步骤列表")
        
        try:
            structured_llm = llm.with_structured_output(PlanOutput)
            response = await structured_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=task_prompt)
            ])
            
            # 限制最大步骤数
            steps = response.steps[:self.config.max_plan_steps]
            return steps
            
        except Exception as e:
            log.error(f"[PlanSolveStrategy] 生成计划失败: {e}")
            # 回退到非结构化输出
            try:
                response = await llm.ainvoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=task_prompt)
                ])
                # 尝试解析 JSON
                import json
                import re
                content = response.content
                # 提取 JSON
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                    return data.get("steps", [])[:self.config.max_plan_steps]
            except Exception as e2:
                log.error(f"[PlanSolveStrategy] 回退解析也失败: {e2}")
            return []
    
    async def _execute_step(self, state: "MainState", step: str, step_index: int) -> str:
        """
        执行单个计划步骤
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_openai import ChatOpenAI
        
        # 获取上下文
        context = ""
        if hasattr(state, 'past_steps') and state.past_steps:
            context = "\n已完成的步骤:\n"
            for prev_step, prev_result in state.past_steps:
                context += f"- {prev_step}: {prev_result[:100]}...\n"
        
        # 构建 Executor 的提示词
        system_prompt = """你是一个任务执行专家。你需要执行给定的步骤，并返回执行结果。

规则:
1. 专注于当前步骤
2. 提供清晰的执行结果
3. 如果遇到问题，说明问题所在"""

        task_prompt = f"""请执行以下步骤:

步骤 {step_index + 1}: {step}
{context}

请执行这个步骤并返回结果。"""

        # 创建 LLM
        executor_model = self.config.executor_model or self.config.model_name or state.request.model
        llm = ChatOpenAI(
            openai_api_base=self.config.chat_api_url or state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=executor_model,
            temperature=self.config.executor_temperature,
        )
        
        # 如果配置了工具，绑定工具
        if self.config.executor_tools:
            llm = llm.bind_tools(self.config.executor_tools)
        
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_prompt)
        ])
        
        return response.content


class PlanExecuteStrategy(ExecutionStrategy):
    """
    Plan-and-Execute (Replanning) 策略
    
    动态生成和调整计划。执行一步后评估结果，决定是继续执行、
    调整计划还是完成任务。
    
    流程:
    1. 调用 Planner 生成初始计划
    2. (可选) Human-in-the-Loop: 等待用户审批计划
    3. 执行当前步骤
    4. (可选) Human-in-the-Loop: 用户确认/干预
    5. 调用 Replanner 评估并决定下一步
    6. 循环 3-5 直到完成
    """
    
    async def execute(self, state: "MainState", **kwargs) -> Dict[str, Any]:
        from dataflow_agent.state import PlanningState
        
        log.info(f"[PlanExecuteStrategy] 执行 {self.agent.role_name}")
        
        # 确保状态类型正确
        if not isinstance(state, PlanningState):
            log.warning("状态不是 PlanningState 类型，可能会缺少某些功能")
        
        replanning_count = 0
        max_rounds = self.config.max_replanning_rounds
        
        # Step 1: 生成初始计划
        log.info("[PlanExecuteStrategy] Step 1: 生成初始计划")
        plan = await self._generate_plan(state)
        
        if not plan:
            return {"error": "无法生成计划", "status": "failed"}
        
        log.info(f"[PlanExecuteStrategy] 生成了 {len(plan)} 步计划")
        
        # 更新状态
        if hasattr(state, 'plan'):
            state.plan = plan
        
        # Step 2: Human-in-the-Loop - 计划审批
        if self.config.require_plan_approval:
            log.info("[PlanExecuteStrategy] Step 2: 等待用户审批计划")
            from langgraph.types import interrupt
            
            approval_response = interrupt({
                "type": "plan_approval",
                "message": "请审批以下计划:",
                "plan": plan,
                "options": ["approve", "reject", "modify"]
            })
            
            if approval_response.get("action") == "reject":
                return {
                    "status": "rejected",
                    "message": "计划被用户拒绝",
                    "plan": plan
                }
            elif approval_response.get("action") == "modify":
                plan = approval_response.get("modified_plan", plan)
                if hasattr(state, 'plan'):
                    state.plan = plan
        
        if hasattr(state, 'plan_approved'):
            state.plan_approved = True
        
        # Step 3-5: 执行循环
        current_step_index = 0
        results = []
        
        while current_step_index < len(plan) and replanning_count <= max_rounds:
            current_step = plan[current_step_index]
            log.info(f"[PlanExecuteStrategy] 执行步骤 {current_step_index + 1}/{len(plan)}: {current_step[:50]}...")
            
            # Human-in-the-Loop: 执行前中断
            if self.config.interrupt_before_step:
                from langgraph.types import interrupt
                
                step_approval = interrupt({
                    "type": "step_approval",
                    "message": f"即将执行步骤 {current_step_index + 1}:",
                    "step": current_step,
                    "past_steps": results,
                    "remaining_steps": plan[current_step_index:],
                    "options": ["continue", "skip", "abort", "modify"]
                })
                
                if step_approval.get("action") == "abort":
                    return {
                        "status": "aborted",
                        "message": "用户中止执行",
                        "results": results
                    }
                elif step_approval.get("action") == "skip":
                    current_step_index += 1
                    continue
                elif step_approval.get("action") == "modify":
                    current_step = step_approval.get("modified_step", current_step)
            
            # 执行步骤
            try:
                step_result = await self._execute_step(state, current_step, current_step_index)
                results.append({
                    "step_index": current_step_index,
                    "step": current_step,
                    "result": step_result,
                    "status": "completed"
                })
                
                # 更新状态
                if hasattr(state, 'past_steps'):
                    state.past_steps.append((current_step, step_result))
                    
            except Exception as e:
                log.error(f"[PlanExecuteStrategy] 步骤执行失败: {e}")
                results.append({
                    "step_index": current_step_index,
                    "step": current_step,
                    "error": str(e),
                    "status": "failed"
                })
                
                if self.config.auto_replan_on_error:
                    if hasattr(state, 'is_replanning_needed'):
                        state.is_replanning_needed = True
                elif not self.config.continue_on_error:
                    return {
                        "status": "failed",
                        "error": str(e),
                        "results": results
                    }
            
            # Human-in-the-Loop: 执行后中断
            if self.config.interrupt_after_step:
                from langgraph.types import interrupt
                
                post_step = interrupt({
                    "type": "step_completed",
                    "message": f"步骤 {current_step_index + 1} 执行完成",
                    "step": current_step,
                    "result": results[-1] if results else None,
                    "options": ["continue", "replan", "abort"]
                })
                
                if post_step.get("action") == "abort":
                    return {
                        "status": "aborted",
                        "message": "用户中止执行",
                        "results": results
                    }
                elif post_step.get("action") == "replan":
                    if hasattr(state, 'is_replanning_needed'):
                        state.is_replanning_needed = True
            
            # Step 5: Replanner 决策
            replan_decision = await self._replan_decision(state, plan, current_step_index, results)
            
            if replan_decision["action"] == "finish":
                log.info("[PlanExecuteStrategy] Replanner 判断任务已完成")
                return {
                    "status": "completed",
                    "response": replan_decision.get("response", ""),
                    "plan": plan,
                    "results": results
                }
            elif replan_decision["action"] == "replan":
                log.info("[PlanExecuteStrategy] 触发重规划")
                replanning_count += 1
                
                if hasattr(state, 'replanning_count'):
                    state.replanning_count = replanning_count
                
                # Human-in-the-Loop: 重规划确认
                if self.config.interrupt_on_replan:
                    from langgraph.types import interrupt
                    
                    replan_approval = interrupt({
                        "type": "replan_approval",
                        "message": "Replanner 建议调整计划",
                        "reason": replan_decision.get("reason", ""),
                        "new_plan": replan_decision.get("new_plan", []),
                        "options": ["approve", "reject", "modify"]
                    })
                    
                    if replan_approval.get("action") == "reject":
                        # 继续原计划
                        current_step_index += 1
                        continue
                    elif replan_approval.get("action") == "modify":
                        replan_decision["new_plan"] = replan_approval.get("modified_plan", 
                                                                          replan_decision.get("new_plan", []))
                
                # 应用新计划
                new_plan = replan_decision.get("new_plan", [])
                if new_plan:
                    plan = new_plan
                    if hasattr(state, 'plan'):
                        state.plan = plan
                    current_step_index = 0  # 重置到新计划的开始
                else:
                    current_step_index += 1
            else:
                # 继续下一步
                current_step_index += 1
        
        # 检查是否因为达到最大轮数而退出
        if replanning_count > max_rounds:
            return {
                "status": "max_rounds_reached",
                "message": f"达到最大重规划轮数 ({max_rounds})",
                "results": results
            }
        
        return {
            "status": "completed",
            "plan": plan,
            "results": results
        }
    
    async def _generate_plan(self, state: "MainState") -> List[str]:
        """
        使用 LLM 生成计划 (与 PlanSolveStrategy 类似)
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, Field
        
        task = state.request.target if hasattr(state, 'request') else ""
        if hasattr(state, 'original_task') and state.original_task:
            task = state.original_task
        
        tools_info = ""
        if hasattr(state, 'executor_tools') and state.executor_tools:
            tools_info = f"\n可用工具: {', '.join(state.executor_tools)}"
        
        # 如果有历史执行记录，包含在上下文中
        history_info = ""
        if hasattr(state, 'past_steps') and state.past_steps:
            history_info = "\n\n已执行的步骤:\n"
            for step, result in state.past_steps:
                history_info += f"- {step}: {result[:100]}...\n"
        
        system_prompt = """你是一个任务规划专家。你的职责是分析用户的任务，并生成一个清晰、可执行的分步计划。

规则:
1. 每个步骤应该是独立的、可执行的操作
2. 步骤之间应该有逻辑顺序
3. 步骤描述应该清晰具体
4. 如果有已执行的步骤，基于其结果规划后续步骤
5. 步骤数量不宜过多，通常 3-7 步为宜"""

        task_prompt = f"""请为以下任务生成执行计划:

任务: {task}
{tools_info}
{history_info}

请以 JSON 格式返回计划，格式如下:
{{"steps": ["步骤1描述", "步骤2描述", ...]}}"""

        planner_model = self.config.planner_model or self.config.model_name or state.request.model
        llm = ChatOpenAI(
            openai_api_base=self.config.chat_api_url or state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=planner_model,
            temperature=self.config.planner_temperature,
        )
        
        class PlanOutput(BaseModel):
            steps: List[str] = Field(description="计划步骤列表")
        
        try:
            structured_llm = llm.with_structured_output(PlanOutput)
            response = await structured_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=task_prompt)
            ])
            return response.steps[:self.config.max_plan_steps]
        except Exception as e:
            log.error(f"[PlanExecuteStrategy] 生成计划失败: {e}")
            return []
    
    async def _execute_step(self, state: "MainState", step: str, step_index: int) -> str:
        """
        执行单个计划步骤 (与 PlanSolveStrategy 类似)
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_openai import ChatOpenAI
        
        context = ""
        if hasattr(state, 'past_steps') and state.past_steps:
            context = "\n已完成的步骤:\n"
            for prev_step, prev_result in state.past_steps:
                context += f"- {prev_step}: {prev_result[:100]}...\n"
        
        system_prompt = """你是一个任务执行专家。你需要执行给定的步骤，并返回执行结果。"""

        task_prompt = f"""请执行以下步骤:

步骤 {step_index + 1}: {step}
{context}

请执行这个步骤并返回结果。"""

        executor_model = self.config.executor_model or self.config.model_name or state.request.model
        llm = ChatOpenAI(
            openai_api_base=self.config.chat_api_url or state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=executor_model,
            temperature=self.config.executor_temperature,
        )
        
        if self.config.executor_tools:
            llm = llm.bind_tools(self.config.executor_tools)
        
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_prompt)
        ])
        
        return response.content
    
    async def _replan_decision(
        self, 
        state: "MainState", 
        current_plan: List[str], 
        current_index: int,
        results: List[Dict]
    ) -> Dict[str, Any]:
        """
        Replanner: 决定是继续执行、重规划还是完成
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_openai import ChatOpenAI
        from pydantic import BaseModel, Field
        from typing import Union, Literal
        
        task = state.request.target if hasattr(state, 'request') else ""
        if hasattr(state, 'original_task') and state.original_task:
            task = state.original_task
        
        # 构建执行历史
        history = ""
        for r in results:
            status = r.get("status", "unknown")
            if status == "completed":
                history += f"✓ {r['step']}: {r['result'][:100]}...\n"
            else:
                history += f"✗ {r['step']}: 失败 - {r.get('error', 'unknown')}\n"
        
        # 剩余计划
        remaining = current_plan[current_index + 1:]
        remaining_str = "\n".join(f"- {s}" for s in remaining) if remaining else "无"
        
        system_prompt = """你是一个任务评估专家。根据任务目标和已执行的步骤，你需要决定:
1. finish - 任务已完成，可以返回最终结果
2. continue - 继续执行剩余计划
3. replan - 需要调整计划

只有当任务目标已经达成时才选择 finish。
如果执行结果显示需要调整后续步骤，选择 replan 并提供新计划。
否则选择 continue。"""

        task_prompt = f"""原始任务: {task}

已执行步骤和结果:
{history}

剩余计划:
{remaining_str}

请评估并决定下一步行动。如果选择 replan，请提供新的计划步骤。

以 JSON 格式返回:
{{"action": "finish|continue|replan", "reason": "决策原因", "response": "最终回答(仅finish时)", "new_plan": ["新步骤"](仅replan时)}}"""

        replanner_model = self.config.replanner_model or self.config.model_name or state.request.model
        llm = ChatOpenAI(
            openai_api_base=self.config.chat_api_url or state.request.chat_api_url,
            openai_api_key=state.request.api_key,
            model_name=replanner_model,
            temperature=self.config.replanner_temperature,
        )
        
        class ReplanDecision(BaseModel):
            action: Literal["finish", "continue", "replan"] = Field(description="决策动作")
            reason: str = Field(description="决策原因")
            response: Optional[str] = Field(default=None, description="最终回答，仅当 action=finish 时")
            new_plan: Optional[List[str]] = Field(default=None, description="新计划，仅当 action=replan 时")
        
        try:
            structured_llm = llm.with_structured_output(ReplanDecision)
            decision = await structured_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=task_prompt)
            ])
            
            return {
                "action": decision.action,
                "reason": decision.reason,
                "response": decision.response,
                "new_plan": decision.new_plan
            }
        except Exception as e:
            log.error(f"[PlanExecuteStrategy] Replanner 决策失败: {e}")
            # 默认继续执行
            return {"action": "continue", "reason": f"决策失败: {e}"}


class StrategyFactory:
    """策略工厂"""
    
    _strategies = {
        "simple": SimpleStrategy,
        "react": ReactStrategy,
        "graph": GraphStrategy,
        "vlm": VLMStrategy,
        "parallel": ParallelStrategy,
        "plan_solve": PlanSolveStrategy,
        "plan_execute": PlanExecuteStrategy,
    }
    
    @classmethod
    def create(cls, mode: str, agent: "BaseAgent", config: Any) -> ExecutionStrategy:
        strategy_cls = cls._strategies.get(mode.lower())
        if not strategy_cls:
            raise ValueError(f"不支持的执行模式: {mode}，可选: {list(cls._strategies.keys())}")
        return strategy_cls(agent, config)
    
    @classmethod
    def register(cls, mode: str, strategy_cls: type):
        """注册自定义策略"""
        cls._strategies[mode.lower()] = strategy_cls
