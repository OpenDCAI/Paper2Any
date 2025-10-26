# dataflow_agent/workflow/__init__.py

import importlib
from pathlib import Path

from .registry import RuntimeRegistry

# ---- 1. 自动发现并导入所有工作流定义模块 ---------------------------------
# 遍历当前包目录下所有以 wf_*.py 命名的 Python 文件，并动态导入。
# 通过 importlib 以全限定名加载模块，从而确保每个工作流文件中的 @register 装饰器
# 能够在导入时将相应工作流注册到 RuntimeRegistry。
_pkg_path = Path(__file__).resolve().parent
for py in _pkg_path.glob("wf_*.py"):
    # importlib 需要模块的点分路径（dotted-path），例如 dataflow_agent.workflow.wf_xxx
    mod_name = f"{__name__}.{py.stem}"
    importlib.import_module(mod_name)
# 模块导入后，各 wf_*.py 文件内的 @register 装饰器会自动注册工作流到 RuntimeRegistry

# ---- 2. 工作流的统一接口 ---------------------------------------------
def get_workflow(name: str):
    """
    根据工作流名称获取 create_pipeline_graph 工厂方法。

    Args:
        name (str): 工作流名称（注册名）

    Returns:
        Callable: 用于构建该工作流图的工厂函数
    """
    return RuntimeRegistry.get(name)

async def run_workflow(name: str, state):
    """
    执行指定名称的工作流。

    Args:
        name (str): 工作流名称（注册名）
        state (Any): 初始状态数据（将传递给工作流）

    Returns:
        Any: 工作流执行结果（通常是终节点的输出）
    """
    factory = get_workflow(name)             # 获取 create_pipeline_graph 工厂
    graph_builder = factory()                # 构建图生成器实例
    graph = graph_builder.compile()          # 编译生成可执行的 Graph 实例
    return await graph.ainvoke(state)        # 异步执行工作流

# ---- 3. 工作流注册信息公开接口 -------------------------------------------
# 提供所有已注册工作流的列表，便于外部查询与 introspection
list_workflows = RuntimeRegistry.all