"""LangGraph utilities"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import json
import json_repair

from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from src.state.poster_state import ModelConfig
from utils.src.logging_utils import log_agent_error, log_agent_warning

load_dotenv(override=True) # reload env every time


def create_model(config: ModelConfig):
    """create chat model from config"""
    # common timeout settings for all providers
    timeout_settings = {
        'request_timeout': 500,  # 2 minutes for request timeout
        'max_retries': 2,        # reduce retries at model level since we have tenacity
    }
    
    if config.provider == 'openai':
        from langchain_openai import ChatOpenAI

        openai_kwargs = {
            'model_name': config.model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'api_key': os.getenv('OPENAI_API_KEY'),
            'request_timeout': timeout_settings['request_timeout'],
            'max_retries': timeout_settings['max_retries'],
        }
        base_url = os.getenv('OPENAI_BASE_URL')
        if base_url:
            openai_kwargs['base_url'] = base_url
            
        return ChatOpenAI(**openai_kwargs)
    elif config.provider == 'anthropic':
        from langchain_anthropic import ChatAnthropic

        anthropic_kwargs = {
            'model': config.model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'api_key': os.getenv('ANTHROPIC_API_KEY'),
            'timeout': timeout_settings['request_timeout'],
            'max_retries': timeout_settings['max_retries'],
        }
        base_url = os.getenv('ANTHROPIC_BASE_URL')
        if base_url:
            anthropic_kwargs['base_url'] = base_url
            
        return ChatAnthropic(**anthropic_kwargs)
    elif config.provider == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI

        google_kwargs = {
            'model': config.model_name,
            'temperature': config.temperature,
            'max_output_tokens': config.max_tokens,
            'google_api_key': os.getenv('GOOGLE_API_KEY'),
            'timeout': timeout_settings['request_timeout'],
            'max_retries': timeout_settings['max_retries'],
        }
        base_url = os.getenv('GOOGLE_BASE_URL')
        if base_url:
            google_kwargs['base_url'] = base_url
            
        return ChatGoogleGenerativeAI(**google_kwargs)
    elif config.provider == 'zhipu':
        from langchain_openai import ChatOpenAI

        zhipu_kwargs = {
            'model': config.model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'api_key': os.getenv('ZHIPU_API_KEY'),
            'timeout': timeout_settings['request_timeout'],
            'max_retries': timeout_settings['max_retries'],
        }
        base_url = os.getenv('ZHIPU_BASE_URL')
        if base_url:
            zhipu_kwargs['base_url'] = base_url
            
        return ChatOpenAI(**zhipu_kwargs)
    elif config.provider == 'moonshot':
        from langchain_openai import ChatOpenAI

        moonshot_kwargs = {
            'model': config.model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'api_key': os.getenv('MOONSHOT_API_KEY'),
            'timeout': timeout_settings['request_timeout'],
            'max_retries': timeout_settings['max_retries'],
        }
        base_url = os.getenv('MOONSHOT_BASE_URL')
        if base_url:
            moonshot_kwargs['base_url'] = base_url
            
        return ChatOpenAI(**moonshot_kwargs)
    elif config.provider == 'Minimax':
        from langchain_openai import ChatOpenAI

        minimax_kwargs = {
            'model': config.model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'api_key': os.getenv('MINIMAX_API_KEY'),
            'timeout': timeout_settings['request_timeout'],
            'max_retries': timeout_settings['max_retries'],
        }
        base_url = os.getenv('MINIMAX_BASE_URL')
        if base_url:
            minimax_kwargs['base_url'] = base_url
            
        return ChatOpenAI(**minimax_kwargs)
    elif config.provider == 'Alibaba':
        from langchain_openai import ChatOpenAI

        alibaba_kwargs = {
            'model': config.model_name,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'api_key': os.getenv('ALIBABA_API_KEY'),
            'timeout': timeout_settings['request_timeout'],
            'max_retries': timeout_settings['max_retries'],
        }
        base_url = os.getenv('ALIBABA_BASE_URL')
        if base_url:
            alibaba_kwargs['base_url'] = base_url
            
        return ChatOpenAI(**alibaba_kwargs)
    else:
        raise ValueError(f"unsupported provider: {config.provider}")


def _extract_token_usage(response: Any, *, fallback_input: float, fallback_output_text: str) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None) or {}
    response_meta = getattr(response, "response_metadata", None) or {}
    token_usage = response_meta.get("token_usage") or {}

    input_tokens = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or token_usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
        or int(fallback_input)
    )
    output_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or token_usage.get("completion_tokens")
        or token_usage.get("output_tokens")
        or max(1, int(len(fallback_output_text.split()) * 1.3))
    )
    return int(input_tokens), int(output_tokens)


class LangGraphAgent:
    """langgraph agent wrapper"""
    
    def __init__(self, system_msg: str, config: ModelConfig):
        self.system_msg = system_msg
        self.config = config
        self.model = create_model(config)
        self.history = [SystemMessage(content=system_msg)]
    
    def reset(self):
        """reset conversation"""
        self.history = [SystemMessage(content=self.system_msg)]
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def step(self, message: str) -> 'AgentResponse':
        """process message and return response"""
        # check if message is json with image data
        try:
            msg_data = json.loads(message)
            if isinstance(msg_data, list) and any("image_url" in item for item in msg_data):
                # vision model call
                return self._step_vision(msg_data)
        except:
            pass
        
        # regular text call
        self.history.append(HumanMessage(content=message))
        
        # keep conversation window
        if len(self.history) > 10:
            self.history = [self.history[0]] + self.history[-9:]
        
        # get response with token tracking
        input_tokens, output_tokens = 0, 0
        try:
            response = self.model.invoke(self.history)
            input_tokens, output_tokens = _extract_token_usage(
                response,
                fallback_input=len(message.split()) * 1.3,
                fallback_output_text=response.content,
            )
        except Exception as e:
            error_msg = f"model call failed: {e}"
            log_agent_error("langgraph_utils", error_msg)
            
            # provide more specific error information
            if "timeout" in str(e).lower() or "read operation timed out" in str(e).lower():
                log_agent_warning("langgraph_utils", f"Timeout error detected for {self.config.provider} {self.config.model_name}")
                log_agent_warning("langgraph_utils", "Possible solutions: check internet, verify API key, switch provider, or increase timeout.")
            elif "rate limit" in str(e).lower():
                log_agent_warning("langgraph_utils", f"Rate limit exceeded for {self.config.provider}")
            elif "authentication" in str(e).lower() or "api key" in str(e).lower():
                log_agent_warning("langgraph_utils", f"Authentication error for {self.config.provider}; check API key configuration.")
            
            input_tokens = len(message.split()) * 1.3
            output_tokens = 100
            raise
        
        self.history.append(response)
        
        return AgentResponse(response.content, input_tokens, output_tokens)
    
    def _step_vision(self, messages: List[Dict]) -> 'AgentResponse':
        """handle vision model calls"""
        # convert to proper format
        content = []
        for msg in messages:
            if msg.get("type") == "text":
                content.append({"type": "text", "text": msg["text"]})
            elif msg.get("type") == "image_url":
                content.append({
                    "type": "image_url",
                    "image_url": msg["image_url"]
                })
        
        human_msg = HumanMessage(content=content)
        
        # get response
        input_tokens, output_tokens = 0, 0
        try:
            response = self.model.invoke([self.history[0], human_msg])
            input_tokens, output_tokens = _extract_token_usage(
                response,
                fallback_input=200,
                fallback_output_text=response.content,
            )
        except Exception as e:
            error_msg = f"vision model call failed: {e}"
            log_agent_error("langgraph_utils", error_msg)
            
            # provide more specific error information for vision calls
            if "timeout" in str(e).lower() or "read operation timed out" in str(e).lower():
                log_agent_warning("langgraph_utils", f"Vision timeout error detected for {self.config.provider} {self.config.model_name}")
                log_agent_warning("langgraph_utils", "Vision calls may take longer; consider another model or check image size/format.")
            elif "rate limit" in str(e).lower():
                log_agent_warning("langgraph_utils", f"Rate limit exceeded for vision calls on {self.config.provider}")
            elif "authentication" in str(e).lower() or "api key" in str(e).lower():
                log_agent_warning("langgraph_utils", f"Authentication error for vision calls on {self.config.provider}")
            
            raise
        
        return AgentResponse(response.content, input_tokens, output_tokens)


class AgentResponse:
    """agent response with token tracking"""
    def __init__(self, content: str, input_tokens: int, output_tokens: int):
        self.content = content
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def extract_json(response: str) -> Dict[str, Any]:
    """extract json from model response"""
    
    # find json code block
    start = response.find("```json")
    end = response.rfind("```")
    
    if start != -1 and end != -1 and end > start:
        json_content = response[start + 7:end].strip()
    else:
        json_content = response.strip()
    
    try:
        return json_repair.loads(json_content)
    except Exception as e:
        raise ValueError(f"failed to parse json: {e}")


def load_prompt(path: str) -> str:
    """load prompt template from file"""
    prompt_path = Path(path).expanduser()
    if not prompt_path.is_absolute():
        postertool_root = Path(__file__).resolve().parents[1]
        prompt_path = postertool_root / prompt_path

    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read()
