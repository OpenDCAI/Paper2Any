from __future__ import annotations

from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.state import PromptWritingState, DFRequest
from dataflow_agent.agentroles.data_agents.prompt_writer import create_prompt_writer
from dataflow_agent.agentroles.data_agents.pipelinebuilder import create_pipeline_builder
from dataflow_agent.logger import get_logger
from dataflow_agent.workflow.registry import register

from dataflow_agent.toolkits.optool.op_tools import (
    local_tool_for_get_purpose,
)

from dataflow_agent.toolkits.basetool.file_tools import (
    get_otherinfo_code,
)

from dataflow_agent.toolkits.optool.op_tools import (
    get_prompt_sources_of_operator,
)


log = get_logger()

@register("pipeline_prompt_writing")
def create_operator_prompt_writing_graph() -> GenericGraphBuilder:
    """Build the operator prompt writing workflow graph.

    Flow: prompt_writer -> pipeline_builder
          -> (code_debugger -> rewriter -> after_rewrite -> prompt_writing)*
    """
    builder = GenericGraphBuilder(state_model=PromptWritingState, entry_point="prompt_writer")

    # ---------------- 前置工具：prompt_writer ----------------
    @builder.pre_tool("operator_code", "prompt_writer")
    def pre_get_operator_code(state: PromptWritingState):
        op_name = state.prompt_op_name
        return get_otherinfo_code([op_name])[op_name]

    @builder.pre_tool("task_description", "prompt_writer")
    def pre_get_task_description(state: PromptWritingState):
        return local_tool_for_get_purpose(state.request)

    @builder.pre_tool("prompt_example", "prompt_writer")
    def pre_get_prompt_example(state: PromptWritingState):
        return get_prompt_sources_of_operator(state.prompt_op_name)

    # ---------------- 前置工具：pipeline_builder ----------------
    @builder.pre_tool("recommendation", "pipeline_builder")
    def pre_get_recommendation(state: PromptWritingState):
        op_name = state.prompt_op_name
        log.info(f"pre_get_recommendation : {[op_name]}")
        return [op_name]
    
    
    
    

    # ---------------- 节点实现 ----------------
    async def prompt_writing_node(s: PromptWritingState) -> PromptWritingState:
        agent = create_prompt_writer()
        return await agent.execute(s, use_agent=False)
    
    async def builder_node(s: PromptWritingState) -> PromptWritingState:
        builder_agent = create_pipeline_builder()
        skip = bool(s.temp_data.get("rewritten", False))
        log.warning(f"[builder_node] skip_assemble = {skip}")
        return await builder_agent.execute(
            s,
            skip_assemble=skip,
            # file_path=s.temp_data.get("pipeline_file_path"),
            file_path= s.request.python_file_path,
            assembler_kwargs={"file_path": s.request.json_file, "chat_api_url": s.request.chat_api_url},
        )

    nodes = {
        "prompt_writer": prompt_writing_node,
        # "pipeline_builder": builder_node,
    }

    edges = [
        # ("prompt_writer", "pipeline_builder"),
    ]

    builder.add_nodes(nodes).add_edges(edges)
    return builder