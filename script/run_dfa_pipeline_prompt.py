import argparse
import asyncio
import os
from typing import Optional

from dataflow_agent.state import DFRequest, PromptWritingState
from dataflow_agent.workflow.wf_pipeline_prompt import create_operator_prompt_writing_graph

def parse_args():
    p = argparse.ArgumentParser(description="Run operator flow: match -> write -> (optional debug loop)")
    p.add_argument('--chat-api-url',        default='http://123.129.219.111:3000/v1/', help='LLM Chat API base')
    p.add_argument('--model',               default='gpt-4o', help='LLM model name')
    p.add_argument('--language',            default='en', help='Prompt output language')
    p.add_argument('--task-description',    required=True, help='Task description')
    p.add_argument('--op-name',             required=True, help='Operator name')
    # 这两项在算子不拥有任何一个预置提示词时才需要提供，否则会仿照已有提示词生成
    p.add_argument('--output-format',       default='', help='Output format for prompt')
    p.add_argument('--arguments',           default=[], nargs='*', help='Arguments for prompt')
    # 缓存目录，用于存储测试数据和提示词
    p.add_argument('--cache-dir',           default='./pa_cache', help='File path for cache directory')
    p.add_argument('--delete-test-files',   action='store_true', default=True, help='Delete files for operator test')
    return p.parse_args()

async def main():
    args = parse_args()

    req = DFRequest(
        language=args.language,
        chat_api_url=args.chat_api_url,
        api_key=os.getenv("DF_API_KEY", "sk-dummy"),
        model=args.model,
        target=args.task_description,
        cache_dir=args.cache_dir,
    )
    state = PromptWritingState(request=req, messages=[])
    state.prompt_op_name = args.op_name
    state.prompt_args = args.arguments
    state.prompt_output_format = args.output_format
    state.temp_data["pipeline_file_path"] = args.cache_dir
    state.temp_data["cache_dir"] = args.cache_dir
    state.temp_data["delete_test_files"] = args.delete_test_files
    

    graph = create_operator_prompt_writing_graph().build()
    final_state: PromptWritingState = await graph.ainvoke(state)
    
if __name__ == "__main__":
    asyncio.run(main())
    
# python /mnt/DataFlow/lz/proj/agentgroup/ziyi/dataflow-agent/DataFlow-Agent/script/run_dfa_pipeline_prompt.py --task-description 我想写一个适用于金融问题的过滤器提示词 --op-name ReasoningQuestionFilter --cache-dir /mnt/DataFlow/lz/proj/agentgroup/ziyi/dataflow-agent/cache
