from typing import Any, Dict, Optional, List
from pathlib import Path
import inspect
import os
import re
import time
import json
import sys
import subprocess

from dataflow_agent.state import PromptWritingState
from dataflow_agent.toolkits.tool_manager import ToolManager
from dataflow_agent.logger import get_logger
from dataflow.utils.registry import PROMPT_REGISTRY, OPERATOR_REGISTRY
from dataflow_agent.agentroles.cores.registry import register

from dataflow_agent.toolkits.pipetool.pipe_tools import extract_op_params, indent_block, snake_case

from dataflow_agent.agentroles.cores.base_agent import BaseAgent
from pydantic.type_adapter import P


"""
cache_dir: 优先用request.python_file_path or temp_data["pipeline_file_path"]作为缓存路径，否则用"./cache_local"
"""

log = get_logger()

@register("prompt_writer")
class PromptWriter(BaseAgent):
    """
    指定一个算子，根据任务描述、算子的参数列表和输出格式，生成一个针对新任务的提示词。
    前置工具:
        - target : local_tool_for_get_purpose
        - arguments : 算子的参数列表
        - output_format : 算子的输出格式
    后置工具:
        - 无
    """
    @property
    def role_name(self) -> str:
        return "prompt_writer"
    
    @property
    def system_prompt_template_name(self) -> str:
        return "system_prompt_for_prompt_writer"
    
    @property
    def task_prompt_template_name(self) -> str:
        return "task_prompt_for_prompt_writer"

    def get_task_prompt_params(self, pre_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """提示词生成器特有的提示词参数"""
        return {
            'operator_code': pre_tool_results.get('operator_code', ''),
            'task_description': pre_tool_results.get('task_description', ''),
            'prompt_example': pre_tool_results.get('prompt_example', ''),
        }

    def get_default_pre_tool_results(self) -> Dict[str, Any]:
        """提示词生成器的默认前置工具结果"""
        return {
            'operator_code': '',
            'task_description': '',
            'prompt_example': '',
        }
        
    def parse_result(self, content: str) -> Dict[str, Any]:
        import re
        # 通过正则抽取代码块 ```python ... ``` 内部代码
        pattern = r"```python(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            code = match.group(1).strip()
            return {"code": code}
        else:
            # 若未匹配到，则报错
            raise ValueError("未匹配到代码块，解析失败")
    
    # ---------------- 写入prompt代码 -------------------
    
    # 未使用
    def _get_prompt_dir(self, state: PromptWritingState) -> str:
        op_name = state.prompt_op_name
        op_cls = OPERATOR_REGISTRY.get(op_name)
        if op_cls is None:
            raise KeyError(f"Operator <{op_name}> not in OPERATOR_REGISTRY")
        # 通过算子类得到允许的提示词类
        if getattr(op_cls, "ALLOWED_PROMPTS", None):
            prompt_classes = op_cls.ALLOWED_PROMPTS
        else:
            raise ValueError("Operator has no ALLOWED_PROMPTS")
        prompt_cls = prompt_classes[0]
        path = inspect.getsourcefile(prompt_cls) or inspect.getfile(prompt_cls)
        path = os.path.abspath(path) if path else None
        prompt_dir = os.path.join(os.path.dirname(path), "diy_prompt")
        return prompt_dir
    
    # 未使用
    def _init_diy_prompt_module(self, state: PromptWritingState, prompt_dir: str):
        # 初始化diy_prompt模块，并创建自动读取的init文件
        if not os.path.exists(prompt_dir + "/diy_prompt/__init__.py"):
            os.makedirs(prompt_dir + "/diy_prompt", exist_ok=True)
            with open(prompt_dir + "/diy_prompt/__init__.py", "w", encoding="utf-8") as f:
                f.write("""
from importlib import import_module
from pkgutil import iter_modules
import inspect as _inspect

__all__ = []
_pkg_name = __name__
_pkg_path = __path__

for _, _mod, _ispkg in iter_modules(_pkg_path):
    if _ispkg or _mod.startswith("_"):
        continue
    m = import_module(f".{_mod}", _pkg_name)
    names = getattr(m, "__all__", None)
    items = ((n, getattr(m, n)) for n in names) if names else vars(m).items()
    for name, obj in items:
        if _inspect.isclass(obj) and not name.startswith("_"):
            if name in globals():
                raise RuntimeError(f"重复导出类名：{name}（发生在模块 {_mod}）")
            globals()[name] = obj
            __all__.append(name)""")
            
            log.info(f"已初始化diy_prompt模块: {prompt_dir}" + "/diy_prompt")
        else:
            log.info(f"diy_prompt模块已存在: {prompt_dir}" + "/diy_prompt")
        state.temp_data["prompt_dir"] = os.path.join(prompt_dir, "diy_prompt")
        
        return prompt_dir
        
    def _build_diy_prompt_file_path(self, state: PromptWritingState) -> str:
        """
        1. 初始化diy_prompt模块
        2. 从代码文本中提取类名，并将类名转化为蛇形命名法，添加时间戳保证唯一性，构建为文件名，返回完整保存路径。

        Args:
            state: PromptWritingState

        Returns:
            str: 完整的保存路径
        """
        # prompt_dir = self._get_prompt_dir(state)
        # self._init_diy_prompt_module(state, prompt_dir)
        # diy_prompt_dir = state.temp_data["prompt_dir"]
        diy_prompt_dir = state.request.cache_dir
        prompt_code = state.draft_prompt_code
        
        # 尝试通过__all__获取类名
        match = re.search(r'__all__\s*=\s*\[(.*?)\]', prompt_code)
        if not match:
            raise ValueError("未能从代码中提取到类名")
        class_names = match.group(1).strip().split(",")
        class_names = [name.strip() for name in class_names]
        class_names = [name.strip("'") for name in class_names]
        class_names = [name.strip('"') for name in class_names]
        class_name = class_names[0]
        
        # 将类名转换为蛇形命名法
        snake_case_name = snake_case(class_name)
        file_name = snake_case_name + str(time.strftime("%Y%m%d%H%M%S")) + ".py"
        file_path = os.path.join(diy_prompt_dir, file_name)
        
        state.temp_data["prompt_class_name"] = class_name
        log.info(f"prompt类名: {class_name}")
        
        return file_path
    
    def _dump_code(self, state: PromptWritingState, file_path_str: str, new_code: str) -> Path | None:
        """
        将新代码写入目标文件。如果未提供路径，则仅返回 None（不强制落盘）。
        """
        file_path = Path(file_path_str)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_code)
            log.info(f"已将新代码写入 {file_path}")
            return file_path
        except Exception as e:
            log.error(f"写入文件 {file_path} 失败: {e}")
            return None
        
        
    # ---------------- 测试prompt -------------------
    def _init_cache_dir(self, state: PromptWritingState) -> str:
        """
        初始化缓存目录
        """
        cache_dir = (
            state.request.cache_dir
            or state.temp_data.get("cache_dir")
            or state.temp_data.get("pipeline_file_path")
            or "./pa_cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        
        state.temp_data["cache_dir"] = cache_dir
        
        log.info(f"初始化缓存目录: {cache_dir}")
        
        return cache_dir
    
    async def _build_test_data_by_llm(self, state: PromptWritingState) -> str:
        """
        通过LLM生成测试数据
        """
        prompt_code = state.draft_prompt_code
        operator_code = state.temp_data['pre_tool_results'].get("operator_code", "")
        cache_dir = state.temp_data["cache_dir"]
        
        user_prompt = f"""
        # 角色
        - 你是DataFlow项目的一个测试数据工程师
        
        # 背景
        - DataFlow的算子负责对数据进行某种处理，以制造适用于大模型训练的优质数据。算子的工作过程是通过提示词来控制大模型进行处理数据。
        
        # 具体任务
        - 根据以下提示词和算子代码，生成一组测试数据
        算子代码：
        {operator_code}
        提示词代码：
        {prompt_code}
        # 要求
        - 这组数据是用于测试提示词的，因此在内容上需要符合提示词的场景。
        - 需要注意数据的多样性，在10组数据中需要包含多种不同的答案类型。
        - 你需要参考算子代码中的run函数的input_XXX_key参数(如input_key, input_instruction_key, input_db_id_key等等)，来生成测试数据的字段。
        - 如果run函数中有多个input_XXX_key参数，在一组测试数据中，需要包含这多个字段。
        - 你需要为每一条数据添加一个唯一的id字段，用于标识该条数据。
        - 你需要为每一条数据生成一个golden_answer字段，标识该条数据的正确答案。
        
        # 输出格式
        - 输出格式为jsonl格式，每一行为一组测试数据，你共需要输出10组测试数据
        - 你需要直接输出测试数据（用代码块包裹jsonl格式），不需要任何其他内容
        - 样例：
        ```jsonl
        {json.dumps({"<input_key1>": "XXX"})}
        {json.dumps({"<input_key2>": "XXX"})}
        ...
        ```
        - 样例2：
        ```jsonl
        {json.dumps({"<input_key1>": "XXX", "<input_key2>": "XXX"})}
        {json.dumps({"<input_key1>": "XXX", "<input_key2>": "XXX"})}
        ...
        ```
        """
        
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        
        llm = self.create_llm(state, bind_post_tools=False)
        answer_msg = await llm.ainvoke(messages)
        answer_text = answer_msg.content
        pattern = r"```jsonl(.*?)```"
        match = re.search(pattern, answer_text, re.DOTALL)
        if match:
            test_data = match.group(1).strip()
            file_path = os.path.join(cache_dir, "prompt_test_data.jsonl")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(test_data)
            log.info(f"已将测试数据写入 {file_path}")
            state.temp_data["test_data_file_path"] = file_path
            return file_path
        else:
            log.error(f"未匹配到代码块，解析失败: {answer_text}")
            return ""

    def _get_imports(self, state: PromptWritingState) -> List[str]:
        op_name = state.prompt_op_name
        prompt_name = state.temp_data["prompt_class_name"]
        
        import_lines = []
        # 获取算子类
        op_cls = OPERATOR_REGISTRY.get(op_name)
        if op_cls is None:
            raise KeyError(f"Operator <{op_name}> not in OPERATOR_REGISTRY")
        # 获取算子导入（优先精确模块，失败则回退到上层聚合）
        parts = op_cls.__module__.split('.')
        parent2_module = '.'.join(parts[:-2]) if len(parts) > 2 else (parts[0] if parts else '')
        op_import_block = (
            f"try:\n"
            f"    from {op_cls.__module__} import {op_cls.__qualname__}\n"
            f"except Exception:\n"
            f"    from {parent2_module} import {op_cls.__qualname__}"
        )
        import_lines.append(op_import_block)
        
        # 获取DIY提示词导入：把算子测试代码与提示词代码放在同一个目录里
        prompt_file_path = state.temp_data.get("prompt_file_path", "")
        prompt_file_name = os.path.basename(prompt_file_path).split('.')[0]
        prompt_import_line = f"from {prompt_file_name} import {prompt_name}"
        import_lines.append(prompt_import_line)
        
        return import_lines

    def _render_operator_blocks(self, op_name: str, state: PromptWritingState) -> str:
        op_cls = OPERATOR_REGISTRY.get(op_name)
        prompt_name = state.temp_data["prompt_class_name"]
        if op_cls is None:
            raise KeyError(f"Operator <{op_name}> not in OPERATOR_REGISTRY")
        init_kwargs, run_kwargs, run_has_storage = extract_op_params(op_cls)
        # Inject pipeline context where appropriate
        var_name = snake_case(op_cls.__name__)
        rendered_init_args: List[str] = []
        for k, v in init_kwargs:
            if k == "llm_serving":
                rendered_init_args.append(f"{k}=self.llm_serving")
            elif k == "prompt_template":
                p_t = f"{prompt_name}()"
                rendered_init_args.append(f'{k}={p_t}')
            else:
                rendered_init_args.append(f"{k}={v}")

        init_line = f"self.{var_name} = {op_cls.__name__}(" + ", ".join(rendered_init_args) + ")"

        # Build run call
        run_args: List[str] = []
        if run_has_storage:
            run_args.append("storage=self.storage.step()")
        run_args.extend([f"{k}={v}" for k, v in run_kwargs])

        if run_args:
            forward_line = (
                f"self.{var_name}.run(\n"
                f"    " + ", ".join(run_args) + "\n"
                f")"
            )
        else:
            forward_line = f"self.{var_name}.run()"
        
        return init_line, forward_line

    def _build_test_code(
        self,
        op_name: str,
        state: PromptWritingState,
        file_path: str, # 测试数据文件路径
        chat_api_url: str = None,
        *,
        cache_dir: str = None,
        llm_local: bool = False,
        local_model_path: str = "",
        model_name: str = "gpt-4o",
    ) -> str:
        if cache_dir is None:
            cache_dir = (
                state.request.cache_dir
                or state.temp_data.get("cache_dir")
                or state.request.python_file_path
                or state.temp_data.get("pipeline_file_path")
                or "./pa_cache"
            )
        
        if chat_api_url is None:
            chat_api_url = state.request.chat_api_url
            
        # 1) 根据 file_path 后缀判断 cache_type
        file_suffix = Path(file_path).suffix.lower() if file_path else ""
        if file_suffix == ".jsonl":
            cache_type = "jsonl"
        elif file_suffix == ".json":
            cache_type = "json"
        elif file_suffix == ".csv":
            cache_type = "csv"
        else:
            cache_type = "jsonl"
            log.warning(f"[pipeline_assembler] Unknown file suffix '{file_suffix}', defaulting to 'jsonl'")

        # 2) 收集导入与类
        import_lines = self._get_imports(state)

        # 3) 渲染 operator 代码片段（无缩进）
        ops_init_block_raw, forward_block_raw = self._render_operator_blocks(op_name, state)
        
        import_section = "\n".join(import_lines)

        # 4) LLM-Serving 片段（无缩进，统一在模板中缩进）
        if llm_local:
            llm_block_raw = f"""
# -------- LLM Serving (Local) --------
self.llm_serving = LocalModelLLMServing_vllm(
    hf_model_name_or_path="{local_model_path}",
    vllm_tensor_parallel_size=1,
    vllm_max_tokens=8192,
    hf_local_dir="local",
    model_name="{model_name}",
)
"""
        else:
            llm_block_raw = f"""
# -------- LLM Serving (Remote) --------
self.llm_serving = APILLMServing_request(
    api_url="{chat_api_url}chat/completions",
    key_name_of_api_key="DF_API_KEY",
    model_name="{model_name}",
    max_workers=100,
)
"""

        # 5) 统一缩进
        llm_block = indent_block(llm_block_raw, 8)
        ops_init_block = indent_block(ops_init_block_raw, 8)
        forward_block = indent_block(forward_block_raw, 8)

        # 6) 模板（使用 {cache_type} 占位符）
        template = '''"""
Auto-generated by prompt_writer
"""
from dataflow.pipeline import PipelineABC
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm

{import_section}

class RecommendPipeline(PipelineABC):
    def __init__(self):
        super().__init__()
        # -------- FileStorage --------
        self.storage = FileStorage(
            first_entry_file_name="{file_path}",
            cache_path="{cache_dir}",
            file_name_prefix="dataflow_cache_step",
            cache_type="{cache_type}",
        )
{llm_block}

{ops_init_block}

    def forward(self):
{forward_block}

if __name__ == "__main__":
    pipeline = RecommendPipeline()
    pipeline.compile()
    pipeline.forward()
'''

        # 7) 格式化并返回
        code = template.format(
            file_path=file_path,
            import_section=import_section,
            cache_dir=cache_dir,
            cache_type=cache_type, 
            llm_block=llm_block,
            ops_init_block=ops_init_block,
            forward_block=forward_block,
        )
        return code
        
    # ---------------- 更新 State -------------------
    def update_state_result(
        self,
        state: PromptWritingState,
        result: Dict[str, Any],
        pre_tool_results: Dict[str, Any],
    ):
        code_str = ""
        if isinstance(result, dict):
            code_str = result.get("code", "")
        # 将生成代码写入状态，并同步到 temp_data 以便后续执行/调试节点复用
        state.draft_prompt_code = code_str
        if code_str:
            file_path_str = self._build_diy_prompt_file_path(state)
            log.info(f"写入提示词代码文件: {file_path_str}")
            new_code = state.draft_prompt_code
            
            saved_path = self._dump_code(state, file_path_str, new_code)
            
            if saved_path is not None:
                state.temp_data["prompt_file_path"] = str(saved_path)
        
            state.temp_data["prompt_code"] = code_str
            
        super().update_state_result(state, result, pre_tool_results)


    async def execute(self, state: PromptWritingState, use_agent: bool = False, **kwargs) -> PromptWritingState:
        # 先初始化缓存目录，确保后续写入提示词文件不会因目录不存在而失败
        self._init_cache_dir(state)
        await super().execute(state, use_agent=use_agent, **kwargs)
        
        # 前置校验：确保已生成可用的提示词代码
        prompt_code = state.temp_data.get("prompt_code", "")
        if not prompt_code:
            raise ValueError("提示词未生成或格式不正确：需要使用 ```python 包裹完整源码。")

        # 确保存在 prompt_class_name；若缺失，尝试从代码回溯提取
        if not state.temp_data.get("prompt_class_name"):
            name = None
            try:
                m = re.search(r"__all__\s*=\s*\[(.*?)\]", prompt_code, re.S)
                if m:
                    first = m.group(1).split(",")[0].strip().strip("'\"")
                    if first:
                        name = first
                if not name:
                    m2 = re.search(r"class\s+([A-Za-z_]\w*)\s*\(", prompt_code)
                    if m2:
                        name = m2.group(1)
            except Exception:
                name = None
            if name:
                state.temp_data["prompt_class_name"] = name
            else:
                raise ValueError("无法确定提示词类名：请确保源码包含 __all__ = ['YourClass'] 或明确的 class 定义。")

        log.info("开始进行算子测试流程")
        cache_dir = self._init_cache_dir(state)
        test_data_file_path = await self._build_test_data_by_llm(state)
        op_name = state.prompt_op_name
        test_code = self._build_test_code(op_name, state, file_path=test_data_file_path, cache_dir=cache_dir)
        state.temp_data["prompt_test_code"] = test_code
        
        prompt_name = state.temp_data["prompt_class_name"]
        
        test_file_path = os.path.join(cache_dir, "test_" + prompt_name + ".py")
        
        saved_path = self._dump_code(state, test_file_path, test_code)
        if saved_path is not None:
            state.temp_data["prompt_test_file_path"] = str(saved_path)

        # 执行测试代码
        log.info(f"开始执行测试代码: {test_file_path}")
        print("当前工作目录: " + os.getcwd())
        result = subprocess.run(["python", test_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 收集结果文件
        try:
            cache_dir = state.temp_data.get("cache_dir") or "./pa_cache"
            candidates = []
            for name in os.listdir(cache_dir):
                if not name.startswith("dataflow_cache_step"):
                    continue
                if not (name.endswith(".jsonl") or name.endswith(".json") or name.endswith(".csv")):
                    continue
                full_path = os.path.join(cache_dir, name)
                try:
                    mtime = os.path.getmtime(full_path)
                except Exception:
                    mtime = 0
                candidates.append((mtime, full_path))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                latest_path = candidates[0][1]
                state.temp_data["prompt_result_file_path"] = latest_path
                try:
                    with open(latest_path, "r", encoding="utf-8") as f:
                        state.temp_data["prompt_result_preview"] = f.read()
                except Exception:
                    state.temp_data["prompt_result_preview"] = ""
        except Exception:
            pass
        
        if result.returncode != 0:
            log.error(f"执行测试代码失败: {result.stderr}")
        else:
            log.info(f"执行测试代码成功: {result.stdout}")
        
        return state
    
    async def revise_with_feedback(self, state: PromptWritingState, feedback: str) -> PromptWritingState:
        # 若没有已有代码，回退到初次生成
        current_code = state.temp_data.get("prompt_code") or getattr(state, "draft_prompt_code", "")
        if not current_code:
            return await self.execute(state, use_agent=False)
        
        # 1) 通过LLM根据反馈改写现有代码
        try:
            prompt_text = f"""
你是一个高级提示词模板工程师。请根据“用户反馈”对下方的 Python 提示词类代码进行改写：
- 保持原有类名与 __all__ 不变
- 不改变导入路径风格
- 仅改写提示词内容/逻辑以满足反馈
- 输出必须是完整的 Python 源码，并使用 ```python 代码块 包裹

用户反馈：
{feedback or "（无）"}

现有代码：
```python
{current_code}
```
"""
            llm = self.create_llm(state, bind_post_tools=False)
            answer_msg = await llm.ainvoke([{"role": "user", "content": prompt_text}])
            answer_text = answer_msg.content
            result = self.parse_result(answer_text)  # 期望返回 {"code": "..."}
        except Exception as e:
            log.error(f"基于反馈改写提示词失败: {e}")
            return state
        
        # 2) 写回新代码并保存
        pre_tool_results = state.temp_data.get("pre_tool_results", {})
        self.update_state_result(state, result, pre_tool_results)
        
        # 3) 复用测试流程：生成测试数据、生成并执行测试代码
        log.info("基于反馈改写后，开始进行算子测试流程")
        cache_dir = self._init_cache_dir(state)
        test_data_file_path = await self._build_test_data_by_llm(state)
        op_name = state.prompt_op_name
        test_code = self._build_test_code(op_name, state, file_path=test_data_file_path, cache_dir=cache_dir)
        state.temp_data["prompt_test_code"] = test_code
        
        prompt_name = state.temp_data["prompt_class_name"]
        test_file_path = os.path.join(cache_dir, "test_" + prompt_name + ".py")
        saved_path = self._dump_code(state, test_file_path, test_code)
        if saved_path is not None:
            state.temp_data["prompt_test_file_path"] = str(saved_path)
        
        log.info(f"开始执行测试代码(反馈改写后): {test_file_path}")
        env = os.environ.copy()
        base = os.getcwd()
        env["PYTHONPATH"] = base + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        result_run = subprocess.run([sys.executable, test_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        
        # 收集结果文件
        try:
            cache_dir = state.temp_data.get("cache_dir") or base
            candidates = []
            for name in os.listdir(cache_dir):
                if not name.startswith("dataflow_cache_step"):
                    continue
                if not (name.endswith(".jsonl") or name.endswith(".json") or name.endswith(".csv")):
                    continue
                full_path = os.path.join(cache_dir, name)
                try:
                    mtime = os.path.getmtime(full_path)
                except Exception:
                    mtime = 0
                candidates.append((mtime, full_path))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                latest_path = candidates[0][1]
                state.temp_data["prompt_result_file_path"] = latest_path
                try:
                    with open(latest_path, "r", encoding="utf-8") as f:
                        state.temp_data["prompt_result_preview"] = f.read()
                except Exception:
                    state.temp_data["prompt_result_preview"] = ""
        except Exception:
            pass
        
        if result_run.returncode != 0:
            log.error(f"执行测试代码失败(反馈改写后): {result_run.stderr}")
        else:
            log.info(f"执行测试代码成功(反馈改写后): {result_run.stdout}")
        
        return state
        
async def prompt_writing(
    state: PromptWritingState,
    model_name: Optional[str] = None,
    tool_manager: Optional[ToolManager] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    use_agent: bool = False,
    **kwargs,
    ) -> PromptWritingState:
    inst = create_prompt_writer(
        tool_manager=tool_manager,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return await inst.execute(state, use_agent=use_agent, **kwargs)

def create_prompt_writer(tool_manager: Optional[ToolManager] = None, **kwargs) -> PromptWriter:
    if tool_manager is None:
        from dataflow_agent.toolkits.tool_manager import get_tool_manager
        tool_manager = get_tool_manager()
    return PromptWriter(tool_manager=tool_manager, **kwargs)


if __name__ == "__main__":
    pass