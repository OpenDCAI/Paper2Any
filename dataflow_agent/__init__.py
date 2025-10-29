# # ─── compatibility shim ──────────────────────────────────────────────────────────
# import sys, importlib, types

# proxy = types.ModuleType("dataflow.dataflowagent")
# sys.modules["dataflow.dataflowagent"] = proxy           # 顶级包

# for sub in [
#     "state",
#     "agentroles",
#     "toolkits",
#     "workflow",
#     "promptstemplates",
#     "graghbuilder",
#     "storage",
#     "utils",
# ]:
#     try:
#         sys.modules[f"dataflow.dataflowagent.{sub}"] = importlib.import_module(
#             f"dataflow_agent.{sub}"
#         )
#     except ModuleNotFoundError:
#         pass