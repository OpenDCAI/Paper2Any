# dataflow_agent/parsers.py
from abc import ABC, abstractmethod
from typing import Any, Dict
import json
import xml.etree.ElementTree as ET
from dataflow_agent.logger import get_logger

log = get_logger(__name__)

class BaseParser(ABC):
    """解析器基类"""
    
    @abstractmethod
    def parse(self, content: str) -> Dict[str, Any]:
        """解析LLM输出内容"""
        pass
    
    @abstractmethod
    def get_format_instruction(self) -> str:
        """返回格式说明，用于添加到提示词中"""
        pass


class JSONParser(BaseParser):
    """JSON解析器 - 当前默认方式"""
    
    def parse(self, content: str) -> Dict[str, Any]:
        from dataflow_agent.utils import robust_parse_json
        try:
            parsed = robust_parse_json(content)
            log.info("JSON 解析成功")
            return parsed
        except ValueError as e:
            log.warning(f"JSON解析失败: {e}")
            return {"raw": content}
        except Exception as e:
            log.warning(f"解析过程出错: {e}")
            return {"raw": content}
    
    def get_format_instruction(self) -> str:
        return "请以JSON格式返回结果，不要包含其他文字说明。"


class XMLParser(BaseParser):
    """XML解析器 - 解析标签内容"""
    
    def __init__(self, root_tag: str = "result"):
        self.root_tag = root_tag
    
    def parse(self, content: str) -> Dict[str, Any]:
        try:
            # 清理可能的markdown代码块
            content = content.strip()
            if content.startswith("```xml"):
                content = content[6:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # 解析XML
            root = ET.fromstring(content)
            result = self._parse_element(root)
            log.info("XML 解析成功")
            return result
            
        except ET.ParseError as e:
            log.warning(f"XML解析失败: {e}")
            return {"raw": content}
        except Exception as e:
            log.warning(f"XML解析过程出错: {e}")
            return {"raw": content}
    
    def _parse_element(self, element: ET.Element) -> Dict[str, Any]:
        """递归解析XML元素"""
        result = {}
        
        # 处理属性
        if element.attrib:
            result.update(element.attrib)
        
        # 处理子元素
        children = list(element)
        if children:
            for child in children:
                child_data = self._parse_element(child)
                if child.tag in result:
                    # 如果已存在，转为列表
                    if not isinstance(result[child.tag], list):
                        result[child.tag] = [result[child.tag]]
                    result[child.tag].append(child_data)
                else:
                    result[child.tag] = child_data
        else:
            # 叶子节点，获取文本
            text = element.text.strip() if element.text else ""
            if text:
                result["value"] = text
        
        # 如果result只有value，直接返回value
        if len(result) == 1 and "value" in result:
            return result["value"]
        
        return result if result else element.text
    
    def get_format_instruction(self) -> str:
        return f"请以XML格式返回结果，根标签为<{self.root_tag}>，不要包含其他文字说明。"


class TextParser(BaseParser):
    """文本解析器 - 不做任何解析"""
    
    def parse(self, content: str) -> Dict[str, Any]:
        log.info("使用文本解析器，不做处理")
        return {"text": content}
    
    def get_format_instruction(self) -> str:
        return "请以自然语言文本形式返回结果。"


# 解析器工厂
class ParserFactory:
    """解析器工厂"""
    
    _parsers = {
        "json": JSONParser,
        "xml": XMLParser,
        "text": TextParser,
    }
    
    @classmethod
    def create(cls, parser_type: str, **kwargs) -> BaseParser:
        """创建解析器实例"""
        parser_type = parser_type.lower()
        if parser_type not in cls._parsers:
            raise ValueError(f"不支持的解析器类型: {parser_type}，可用类型: {list(cls._parsers.keys())}")
        
        parser_class = cls._parsers[parser_type]
        return parser_class(**kwargs)
    
    @classmethod
    def register(cls, name: str, parser_class: type):
        """注册新的解析器类型"""
        cls._parsers[name.lower()] = parser_class