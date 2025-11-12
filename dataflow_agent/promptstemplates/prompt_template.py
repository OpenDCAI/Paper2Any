from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import re
import sys
from pathlib import Path
from string import Formatter
from typing import Any, Dict, Sequence
import warnings
from dataflow_agent.logger import get_logger
from dataflow_agent.utils import get_project_root


log = get_logger(__name__)




class PromptsTemplateGenerator:
    ANSWER_SUFFIX = ".(Answer in {lang}!!!)"

    # ---------- Singleton ----------
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    # ---------- Init ----------
    def __init__(
        self,
        output_language: str,
        *,
        python_modules: Sequence[str] | None = ["prompts_repo"],
        template_dirs: Sequence[str] | None = [f'{get_project_root()}/dataflow_agent/promptstemplates/resources'],
    ) -> None:
        """
        Parameters
        ----------
        output_language : str
            The language in which the model should answer finally.
        python_modules : Sequence[str] | None, optional
            A list of module names to be scanned (can be more than one).
            Defaults to ["prompts_repo"].
        template_dirs : Sequence[str] | None, optional
            A list of directory paths to scan for pt_*.py files.
            Example: [""]
        """
        self.output_language = output_language
        self.templates: Dict[str, str] = {}
        self.json_form_templates: Dict[str, str] = {}
        self.code_debug_templates: Dict[str, str] = {}
        self.operator_templates: Dict[str, Dict] = {}
        
        # 用于跟踪模板来源，检测重复
        self._template_sources: Dict[str, list] = {}

        # 加载模块
        if python_modules:
            self._load_python_templates(python_modules)
        
        # 加载目录中的文件
        if template_dirs:
            self._load_from_directories(template_dirs)

    # ---------- Safe formatter ----------
    @staticmethod
    def _safe_format(tpl: str, **kwargs) -> str:
        class _Missing(dict):
            def __missing__(self, k):
                return "{" + k + "}"

        try:
            return Formatter().vformat(tpl, [], _Missing(**kwargs))
        except Exception:
            for k in re.findall(r"{(.*?)}", tpl):
                tpl = tpl.replace("{" + k + "}", str(kwargs.get(k, "{"+k+"}")))
            return tpl

    # ---------- 新增：从目录加载 ----------
    def _load_from_directories(self, dirs: Sequence[str]) -> None:
        """
        扫描指定目录下所有 pt_*.py 文件并加载模板
        """
        for dir_path in dirs:
            path = Path(dir_path)
            if not path.exists() or not path.is_dir():
                warnings.warn(f"Template directory not found: {dir_path}")
                continue
            
            # 查找所有 pt_*.py 文件
            for file_path in path.glob("pt_*.py"):
                self._load_file_as_module(file_path)

    def _load_file_as_module(self, file_path: Path) -> None:
        """
        动态加载单个 Python 文件作为模块
        """
        module_name = f"_dynamic_template_{file_path.stem}"
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                
                # 收集该文件的模板
                source_info = str(file_path)
                
                # 1. 类属性
                for _, cls in inspect.getmembers(mod, inspect.isclass):
                    if cls.__module__ != module_name:
                        continue
                    self._collect_from_mapping(vars(cls), source_info)
                
                # 2. 顶层变量
                self._collect_from_mapping(vars(mod), source_info)
                
        except Exception as e:
            warnings.warn(f"Failed to load template file {file_path}: {e}")

    def _load_python_templates(self, modules: Sequence[str]) -> None:
        """
        Scan all classes and top-level variables in the given modules
        """
        for mod_name in modules:
            try:
                if __package__:
                    mod = importlib.import_module(f'.{mod_name}', package=__package__)
                else:
                    try:
                        mod = importlib.import_module(f'dataflow_agent.promptstemplates.{mod_name}')
                    except ImportError:
                        mod = importlib.import_module(mod_name)
                
                source_info = f"module:{mod_name}"
                for _, cls in inspect.getmembers(mod, inspect.isclass):
                    if cls.__module__ != mod.__name__:
                        continue
                    self._collect_from_mapping(vars(cls), source_info)

                # 2. Top-level variables
                self._collect_from_mapping(vars(mod), source_info)
                
            except ImportError as e:
                warnings.warn(f"Failed to import module {mod_name}: {e}")


    def _collect_from_mapping(self, mapping: dict, source: str) -> None:
        """
        收集模板并记录来源，检测重复
        
        Parameters
        ----------
        mapping : dict
            包含模板的字典（通常是 vars(cls) 或 vars(mod)）
        source : str
            来源信息（用于重复警告）
        """
        for attr, value in mapping.items():
            if attr.startswith("_"):
                continue
            
            # ---- operator dict ----
            if attr == "operator_templates" and isinstance(value, dict):
                self._track_and_add("operator_templates", source)
                self.operator_templates.update(value)
                continue
            
            # ---- string templates ----
            if not isinstance(value, str):
                continue
            
            # 处理不同类型的模板
            if attr.startswith("system_prompt_for_") or attr.startswith("task_prompt_for_"):
                self._track_and_add(attr, source)
                self.templates[attr] = value
                
            elif attr.startswith("json_form_template_for_"):
                key = attr.replace("json_form_template_for_", "")
                full_key = f"json_form:{key}"
                self._track_and_add(full_key, source)
                self.json_form_templates[key] = value
                
            elif attr.startswith("code_debug_template_for_"):
                key = attr.replace("code_debug_template_for_", "")
                full_key = f"code_debug:{key}"
                self._track_and_add(full_key, source)
                self.code_debug_templates[key] = value
                
            else:
                # 其他字符串也收录到 templates
                self._track_and_add(attr, source)
                self.templates[attr] = value

    def _track_and_add(self, template_key: str, source: str) -> None:
        """
        跟踪模板来源，检测重复并发出警告
        """
        if template_key not in self._template_sources:
            self._template_sources[template_key] = []
        
        self._template_sources[template_key].append(source)
        
        # 如果有重复，发出警告
        if len(self._template_sources[template_key]) > 1:
            sources = self._template_sources[template_key]
            warnings.warn(
                f"   Duplicate template detected: '{template_key}'\n"
                f"   Found in: {sources}\n"
                f"   Latest source will override previous ones.",
                UserWarning,
                stacklevel=3
            )

    # ---------- 查看重复模板的辅助方法 ----------
    def get_duplicate_templates(self) -> Dict[str, list]:
        """
        返回所有有重复定义的模板及其来源
        
        Returns
        -------
        Dict[str, list]
            键是模板名，值是来源列表（长度>1表示有重复）
        """
        return {
            key: sources 
            for key, sources in self._template_sources.items() 
            if len(sources) > 1
        }

    def print_duplicate_report(self) -> None:
        """
        打印重复模板报告
        """
        duplicates = self.get_duplicate_templates()
        if not duplicates:
            log.info(" No duplicate templates found.")
            return
        
        log.critical("  你的提示词key已经存在了！！请修改！！！")
        log.info("=" * 60)
        for template_name, sources in duplicates.items():
            log.info(f"\n    Template: {template_name}")
            log.warning(f"   Found in {len(sources)} locations:")
            for i, src in enumerate(sources, 1):
                log.info(f"   {i}. {src}")
        log.info("=" * 60)

    # ---------- 其余方法保持不变 ----------
    def render(self, template_name: str, *, add_suffix: bool = False, **kwargs) -> str:
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        txt = self._safe_format(self.templates[template_name], **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=self.output_language) if add_suffix else "")

    def render_json_form(self, template_name: str, *, add_suffix=False, **kwargs) -> str:
        if template_name not in self.json_form_templates:
            raise ValueError(f"JSON-form template '{template_name}' not found")
        txt = self._safe_format(self.json_form_templates[template_name], **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=self.output_language) if add_suffix else "")

    def render_code_debug(self, template_name: str, *, add_suffix=False, **kwargs) -> str:
        if template_name not in self.code_debug_templates:
            raise ValueError(f"Code-debug template '{template_name}' not found")
        txt = self._safe_format(self.code_debug_templates[template_name], **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=self.output_language) if add_suffix else "")

    def render_operator_prompt(
        self,
        operator_name: str,
        prompt_type: str = "task",
        language: str | None = None,
        *,
        add_suffix: bool = False,
        **kwargs,
    ) -> str:
        lang = language or self.output_language
        op = self.operator_templates.get(operator_name)
        if not op:
            raise ValueError(f"Operator '{operator_name}' not found")
        try:
            tpl = op["prompts"][lang][prompt_type]
        except KeyError:
            raise KeyError(
                f"Missing prompt (operator={operator_name}, lang={lang}, type={prompt_type})"
            )
        txt = self._safe_format(tpl, **kwargs)
        return txt + (self.ANSWER_SUFFIX.format(lang=lang) if add_suffix else "")

    def add_sys_template(self, name: str, template: str) -> None:
        self.templates[f"system_prompt_for_{name}"] = template

    def add_task_template(self, name: str, template: str) -> None:
        self.templates[f"task_prompt_for_{name}"] = template

    def add_json_form_template(self, task_name: str, template: str | dict) -> None:
        if isinstance(template, dict):
            import json
            template = json.dumps(template, ensure_ascii=False, indent=2)
        self.json_form_templates[task_name] = template

if __name__ == "__main__":
    # 初始化，同时指定模块和目录
    ptg = PromptsTemplateGenerator(
        output_language="zh",
        python_modules=["prompts_repo"]
    )
    
    # 查看是否有重复模板
    ptg.print_duplicate_report()
    
    # 或者程序化处理
    duplicates = ptg.get_duplicate_templates()
    if duplicates:
        log.info(f"Found {len(duplicates)} duplicate template(s)")
        for name, sources in duplicates.items():
            log.info(f"  - {name}: {sources}")
    
    # 正常使用
    result = ptg.render("system_prompt_for_data_cleaning_and_analysis", 
                        language="zh", 
                        history_data="{...}",
                        user_question="分析这些数据",
                        target_language="中文")
    # log.critical(result)