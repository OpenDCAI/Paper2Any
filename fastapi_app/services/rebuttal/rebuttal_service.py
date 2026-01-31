import sys
import os
import re
import json
import time
import threading
from typing import Tuple, List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum

from fastapi_app.services.rebuttal.llm import LLMClient, TokenUsageTracker
from fastapi_app.services.rebuttal.arxiv import search_relevant_papers
from fastapi_app.services.rebuttal.tools import _read_text, load_prompt, pdf_to_md, download_pdf_and_convert_md, _fix_json_escapes


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Use a directory under the project root for sessions
try:
    from dataflow_agent.utils import get_project_root
    project_root = get_project_root()
    SESSIONS_BASE_DIR = os.path.join(project_root, "rebuttal_sessions")
except ImportError:
    # Fallback if get_project_root is not available
    SESSIONS_BASE_DIR = os.path.join(_CURRENT_DIR, "..", "..", "..", "rebuttal_sessions")
QUESTIONS_UPPER_BOUND = 100

os.makedirs(SESSIONS_BASE_DIR, exist_ok=True)


token_tracker = TokenUsageTracker()

llm_client: Optional[LLMClient] = None


def init_llm_client(api_key: str, chat_api_url: str = None, provider: str = None, model: str = "gpt-5.1") -> LLMClient:
    """Initialize the LLM client
    
    Args:
        api_key: API key for the LLM service
        chat_api_url: API URL (e.g., https://api.apiyi.com/v1). If provided, will use this URL directly.
        provider: Provider name (openrouter, qwen, etc.). Only used if chat_api_url is not provided.
        model: Model name to use
    """
    global llm_client
    
    # If chat_api_url is provided, infer provider from URL or use "openrouter" as default
    if chat_api_url:
        # Infer provider from URL
        url_lower = chat_api_url.lower()
        if 'apiyi.com' in url_lower or 'openrouter' in url_lower:
            inferred_provider = "openrouter"
        elif 'ai520.ai' in url_lower:
            inferred_provider = "openrouter"  # AI520 uses OpenAI-compatible API
        elif 'dashscope' in url_lower or 'qwen' in url_lower:
            inferred_provider = "qwen"
        elif 'deepseek' in url_lower:
            inferred_provider = "deepseek"
        elif 'openai.com' in url_lower:
            inferred_provider = "openai"
        elif 'zhipu' in url_lower or 'bigmodel.cn' in url_lower:
            inferred_provider = "zhipu"
        else:
            # Default to openrouter for OpenAI-compatible APIs
            inferred_provider = "openrouter"
        
        llm_client = LLMClient(
            provider=inferred_provider,
            api_key=api_key,
            base_url=chat_api_url,
            default_model=model,
            site_url="https://rebuttal-assistant.local",
            site_name="Rebuttal Assistant",
            token_tracker=token_tracker
        )
    else:
        # Fallback to provider-based initialization
        provider = provider or "openrouter"
        llm_client = LLMClient(
            provider=provider,
            api_key=api_key,
            default_model=model,
            site_url="https://rebuttal-assistant.local",
            site_name="Rebuttal Assistant",
            token_tracker=token_tracker
        )
    return llm_client


def get_llm_client() -> LLMClient:
    """Get the LLM client. Raises error if not initialized."""
    if llm_client is None:
        raise RuntimeError("LLM client not initialized. Please configure API Key via the Gradio interface first.")
    return llm_client


class LogCollector:
    """Thread-safe log collector for real-time Gradio UI display"""
    
    def __init__(self, max_lines: int = 500):
        self._logs: List[str] = []
        self._lock = threading.Lock()
        self._max_lines = max_lines
    
    def add(self, message: str):
        """Add a log entry"""
        with self._lock:
            timestamp = time.strftime("%H:%M:%S")
            self._logs.append(f"[{timestamp}] {message}")

            if len(self._logs) > self._max_lines:
                self._logs = self._logs[-self._max_lines:]
    
    def get_all(self) -> str:
        """Get all logs (returns concatenated string)"""
        with self._lock:
            return "\n".join(self._logs)
    
    def get_recent(self, n: int = 50) -> str:
        """Get the most recent n log entries"""
        with self._lock:
            return "\n".join(self._logs[-n:])
    
    def clear(self):
        """Clear all logs"""
        with self._lock:
            self._logs.clear()


class ProcessStatus(Enum):
    """Processing status enum"""
    NOT_STARTED = "not_started"
    PROCESSING = "processing"
    WAITING_FEEDBACK = "waiting_feedback"  
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class QuestionState:
    """State of a single question"""
    question_id: int
    question_text: str
    status: ProcessStatus = ProcessStatus.NOT_STARTED
    
    reference_paper_summary: str = ""
    agent6_output: str = ""  
    agent7_output: str = "" 
    
    # Structured todo list (parsed from JSON)
    todo_list: List[Dict] = field(default_factory=list)
    strategy_text: str = ""  # Strategy explanation text
    draft_response: str = ""  # Draft response snippets
    
    # Papers information
    searched_papers: List[Dict] = field(default_factory=list)  # All papers from search
    selected_papers: List[Dict] = field(default_factory=list)  # Papers selected by Agent4
    analyzed_papers: List[Dict] = field(default_factory=list)  # Papers analyzed by Agent5
    
    # History tracking
    history: List[Dict] = field(default_factory=list)  # Full history of revisions
    
    feedback_history: List[Dict] = field(default_factory=list) 
    revision_count: int = 0 
    is_satisfied: bool = False 


@dataclass
class SessionState:
    """Session state containing processing status for all questions"""
    session_id: str
    paper_file_path: str = ""
    review_file_path: str = ""
    paper_summary: str = ""
    
    session_dir: str = ""       
    logs_dir: str = ""        
    arxiv_papers_dir: str = ""  
    
    questions: List[QuestionState] = field(default_factory=list)
    current_question_idx: int = 0
    
    overall_status: ProcessStatus = ProcessStatus.NOT_STARTED
    final_rebuttal: str = ""

    progress_message: str = ""
    

    log_collector: Optional[LogCollector] = None


class Agent1:
    def __init__(self, paper_file_path: str, temperature: float = 0.4, log_dir: str = None):
        self.paper_file_path = paper_file_path
        self.temperature = temperature
        self.log_dir = log_dir
        self.final_text = None
        self.thinking_text = None
    
    def _build_context(self, paper_text: str) -> str:
        instructions = load_prompt("1.txt")
        return f"{instructions}[paper original text]\n\n{paper_text}\n\n"

    def run(self) -> str:
        paper_text = _read_text(self.paper_file_path)
        model_input = self._build_context(paper_text)
        instructions_text = "Please think very carefully and rigorously before answering, and never fabricate anything."
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, "agent1_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, self.thinking_text = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name="Agent1_paper_summary",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, "agent1_output.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== FINAL TEXT ===\n{self.final_text or '(empty)'}\n\n=== THINKING ===\n{self.thinking_text or '(empty)'}")
        
        return self.final_text


class Agent2:
    """Extract review questions"""
    def __init__(self, paper_summary: str, review_file_path: str, temperature: float = 0.4, log_dir: str = None):
        self.paper_summary = paper_summary
        self.review_file_path = review_file_path
        self.temperature = temperature
        self.log_dir = log_dir
        self.final_text = None
    
    def _build_context(self, paper_summary: str, review_text: str) -> str:
        instructions = load_prompt("2.txt")
        return (
            f"{instructions}"
            f"[compressed paper]\n\n{paper_summary}\n```\n\n"
            f"[review original text]\n\n{review_text}\n```\n"
            f"\n**Begin extraction now.**\n"
        )

    def run(self) -> str:
        review_text = _read_text(self.review_file_path)
        model_input = self._build_context(self.paper_summary, review_text)
        instructions_text = "Please think very carefully and rigorously before answering, and never fabricate anything"
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, "agent2_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name="Agent2_extract_questions",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, "agent2_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text


class Agent2Checker:
    """Check and correct extracted questions"""
    def __init__(self, paper_summary: str, review_file_path: str, agent2_output: str, temperature: float = 0.4, log_dir: str = None):
        self.paper_summary = paper_summary
        self.review_file_path = review_file_path
        self.agent2_output = agent2_output
        self.temperature = temperature
        self.log_dir = log_dir
        self.final_text = None
    
    def _build_context(self) -> str:
        review_text = _read_text(self.review_file_path)
        instructions = load_prompt("2_c.txt")
        return (
            f"{instructions}"
            f"[compressed paper]\n\n{self.paper_summary}\n```\n\n"
            f"[review original text]\n\n{review_text}\n```\n"
            f"[student's output]\n\n{self.agent2_output}"
            f"\n**Begin now.**\n"
        )

    def run(self) -> str:
        model_input = self._build_context()
        instructions_text = "Please think very carefully and rigorously before answering, and never fabricate anything"
        

        if self.log_dir:
            with open(os.path.join(self.log_dir, "agent2_checker_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name="Agent2_checker",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, "agent2_checker_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text


class Agent3:
    """Determine search queries"""
    def __init__(self, paper_summary: str, review_question: str, temperature: float = 0.5, num: int = 1, log_dir: str = None):
        self.paper_summary = paper_summary
        self.review_question = review_question
        self.temperature = temperature
        self.num = num
        self.log_dir = log_dir
        self.final_text = None

    def _build_context(self) -> str:
        instructions = load_prompt("3.txt")
        return (
            f"{instructions}"
            f"[compressed paper]\n```paper\n{self.paper_summary}\n```\n"
            f"[review_question]\n```review\n{self.review_question}\n```\n"
        )

    def run(self) -> str:
        model_input = self._build_context()
        instructions_text = "Be rigorous, don't overly trust your own internal knowledge."
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent3_q{self.num}_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name=f"Agent3_search_q{self.num}",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent3_q{self.num}_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text

    def extract(self) -> Tuple[bool, List[str], List[str], str]:
        text = self.final_text or ""
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        
        if json_start == -1 or json_end <= json_start:
            return False, [], [], ""
        
        try:
            json_str = _fix_json_escapes(text[json_start:json_end])
            data = json.loads(json_str)
            
            need_search = bool(data.get("need_search", False))
            queries = data.get("queries", [])
            links = data.get("links", [])
            reason = data.get("reason", "")
            
            return need_search, queries, links, reason
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Agent3] JSON parsing failed: {e}")
            return False, [], [], ""


class Agent4:
    """Filter relevant papers"""
    def __init__(self, paper_list: str, paper_summary: str, review_question: str, 
                 reason: str, temperature: float = 0.5, num: int = 1, log_dir: str = None):
        self.paper_list = paper_list
        self.paper_summary = paper_summary
        self.review_question = review_question
        self.reason = reason
        self.temperature = temperature
        self.num = num
        self.log_dir = log_dir
        self.final_text = None
    
    def _build_context(self) -> str:
        instructions = load_prompt("4.txt")
        return (
            f"{instructions}"
            f"[compressed paper]\n```paper\n{self.paper_summary}\n```\n"
            f"[review_question]\n```review\n{self.review_question}\n```\n"
            f"[papers retrieved]\n```paper\n{self.paper_list}\n```\n"
            f"[search reasons]\n```paper\n{self.reason}\n```\n"
        )
    
    def run(self) -> str:
        model_input = self._build_context()
        instructions_text = "Be rigorous."
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent4_q{self.num}_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name=f"Agent4_select_q{self.num}",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent4_q{self.num}_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text


class Agent5:
    """Analyze reference papers"""
    def __init__(self, paper_summary: str, review_question: str, reference_paper: str,
                 paper_url: str, temperature: float = 0.5, num: int = 1, log_dir: str = None):
        self.paper_summary = paper_summary
        self.review_question = review_question
        self.reference_paper = reference_paper
        self.paper_url = paper_url
        self.temperature = temperature
        self.num = num
        self.log_dir = log_dir
        self.final_text = None
    
    def _build_context(self) -> str:
        instructions = load_prompt("5.txt")
        return (
            f"{instructions}"
            f"[compressed paper]\n```paper\n{self.paper_summary}\n```\n"
            f"[review_question]\n```review\n{self.review_question}\n```\n"
            f"[reference paper]\n```paper\n{self.reference_paper}\n```\n"
            f"[reference paper URL]\n```paper\n{self.paper_url}\n```\n"
        )
    
    def run(self) -> str:
        model_input = self._build_context()
        instructions_text = "Be rigorous."
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent5_ref{self.num}_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name=f"Agent5_analyze_ref_{self.num}",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent5_ref{self.num}_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text


class Agent6:
    """Generate initial rebuttal strategy"""
    def __init__(self, paper_summary: str, review_question: str, 
                 reference_summary: str, temperature: float = 0.4, num: int = 1, log_dir: str = None):
        self.paper_summary = paper_summary
        self.review_question = review_question
        self.reference_summary = reference_summary
        self.temperature = temperature
        self.num = num
        self.log_dir = log_dir
        self.final_text = None
    
    def _build_context(self) -> str:
        instructions = load_prompt("6.txt")
        return (
            f"{instructions}"
            f"[original paper]\n\n{self.paper_summary}\n```\n"
            f"[review_question]\n\n{self.review_question}\n```\n"
            f"[reference papers summary]\n\n{self.reference_summary}\n```\n"
            f"\n**Begin now.**\n"
        )

    def run(self) -> str:
        model_input = self._build_context()
        instructions_text = "Be rigorous."
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent6_q{self.num}_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name=f"Agent6_rebuttal_q{self.num}",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent6_q{self.num}_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text


class Agent7:
    """Check and optimize rebuttal strategy"""
    def __init__(self, to_do_list: str, paper_summary: str, review_question: str,
                 reference_summary: str, temperature: float = 0.4, num: int = 1, log_dir: str = None):
        self.to_do_list = to_do_list
        self.paper_summary = paper_summary
        self.review_question = review_question
        self.reference_summary = reference_summary
        self.temperature = temperature
        self.num = num
        self.log_dir = log_dir
        self.final_text = None
    
    def _build_context(self) -> str:
        instructions = load_prompt("7.txt")
        return (
            f"{instructions}"
            f"[original paper]\n```paper\n{self.paper_summary}\n```\n"
            f"[review_question]\n```review\n{self.review_question}\n```\n"
            f"[reference papers summary]\n```\n{self.reference_summary}\n```\n"
            f"[student's rebuttal strategy and to-do list]\n```\n{self.to_do_list}\n```\n"
            f"\nplease now output the final version."
        )

    def run(self) -> str:
        model_input = self._build_context()
        instructions_text = "Be rigorous."
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent7_q{self.num}_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name=f"Agent7_check_q{self.num}",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent7_q{self.num}_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text


class Agent7WithHumanFeedback:

    def __init__(self, current_strategy: str, paper_summary: str, review_question: str,
                 reference_summary: str, human_feedback: str, temperature: float = 0.4, num: int = 1, log_dir: str = None):
        self.current_strategy = current_strategy
        self.paper_summary = paper_summary
        self.review_question = review_question
        self.reference_summary = reference_summary
        self.human_feedback = human_feedback 
        self.temperature = temperature
        self.num = num
        self.log_dir = log_dir
        self.final_text = None
    
    def _build_context(self) -> str:
        instructions = load_prompt("7_h.txt")
        return (
            f"{instructions}"
            f"[original paper]\n```paper\n{self.paper_summary}\n```\n"
            f"[review_question]\n```review\n{self.review_question}\n```\n"
            f"[reference papers summary]\n```\n{self.reference_summary}\n```\n"
            f"[current rebuttal strategy and to-do list]\n```\n{self.current_strategy}\n```\n"
            f"[human's feedback]\n```\n{self.human_feedback}\n```\n"
            f"\nPlease incorporate the human feedback and output the revised version. "
            f"Do not include comments on the previous version. Output only the rebuttal strategy and to-do list."
        )

    def run(self) -> str:
        model_input = self._build_context()
        instructions_text = "Be rigorous. Carefully consider the human feedback."
        
        if self.log_dir:
            revision_num = len([f for f in os.listdir(self.log_dir) if f.startswith(f"agent7_hitl_q{self.num}_") and f.endswith("_input.txt")]) + 1
            with open(os.path.join(self.log_dir, f"agent7_hitl_q{self.num}_r{revision_num}_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name=f"Agent7_HITL_q{self.num}",
        )
        
        if self.log_dir:
            revision_num = len([f for f in os.listdir(self.log_dir) if f.startswith(f"agent7_hitl_q{self.num}_") and f.endswith("_output.txt")]) + 1
            with open(os.path.join(self.log_dir, f"agent7_hitl_q{self.num}_r{revision_num}_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text


class Agent8:
    def __init__(self, to_do_list: str, paper_summary: str, review_file_path: str,
                 temperature: float = 0.4, num: int = 1, log_dir: str = None):
        self.to_do_list = to_do_list
        self.paper_summary = paper_summary
        self.review_file_path = review_file_path
        self.temperature = temperature
        self.num = num
        self.log_dir = log_dir
        self.final_text = None
    
    def _build_context(self) -> str:
        review_text = _read_text(self.review_file_path)
        instructions = load_prompt("8.txt")
        return (
            f"{instructions}"
            f"[original paper]\n```paper\n{self.paper_summary}\n```\n\n"
            f"[review original text]\n```review\n{review_text}\n```\n\n"
            f"[rebuttal strategies]\n```rebuttal\n{self.to_do_list}\n```\n"
            f"Please generate the formal ICLR rebuttal response now."
        )

    def run(self) -> str:
        model_input = self._build_context()
        instructions_text = "Be rigorous."
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent8_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name=f"Agent8_final_{self.num}",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent8_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text


class Agent9:
    def __init__(self, draft: str, to_do_list: str, paper_summary: str, 
                 review_file_path: str, temperature: float = 0.4, num: int = 1, log_dir: str = None):
        self.draft = draft
        self.to_do_list = to_do_list
        self.paper_summary = paper_summary
        self.review_file_path = review_file_path
        self.temperature = temperature
        self.num = num
        self.log_dir = log_dir
        self.final_text = None
    
    def _build_context(self) -> str:
        review_text = _read_text(self.review_file_path)
        instructions = load_prompt("9.txt")
        return (
            f"{instructions}"
            f"[original paper]\n```\n{self.paper_summary}\n```\n\n"
            f"[review original text]\n```\n{review_text}\n```\n\n"
            f"[rebuttal strategies]\n```\n{self.to_do_list}\n```\n"
            f"[student's version]\n```\n{self.draft}\n```\n"
            f"Please generate the final ICLR rebuttal response."
        )

    def run(self) -> str:
        model_input = self._build_context()
        instructions_text = "Be rigorous."
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent9_input.txt"), "w", encoding="utf-8") as f:
                f.write(f"=== INSTRUCTIONS ===\n{instructions_text}\n\n=== MODEL INPUT ===\n{model_input}")
        
        self.final_text, _ = get_llm_client().generate(
            instructions=instructions_text,
            input_text=model_input,
            enable_reasoning=True,
            temperature=self.temperature,
            agent_name=f"Agent9_final_{self.num}",
        )
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f"agent9_output.txt"), "w", encoding="utf-8") as f:
                f.write(self.final_text or "(empty)")
        
        return self.final_text




def extract_review_questions(review_questions_text: str) -> Tuple[List[str], int]:
    """Extract question list from Agent2 output"""
    all_questions = []
    last_index = 0
    text = review_questions_text.strip()
    
    paired_found = False
    for i in range(1, QUESTIONS_UPPER_BOUND + 1):
        open_tag = rf"\[\s*(?!/)\s*q{i}\s*\]"
        close_tag = rf"\[\s*/?\s*q{i}\s*\]"
        pattern = re.compile(rf"{open_tag}\s*(.+?)\s*{close_tag}", re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(text) or []
        if matches:
            paired_found = True
            all_questions.extend([m.strip() for m in matches])
            last_index = i
        elif paired_found:
            break

    if paired_found and all_questions:
        return all_questions, last_index

    all_questions = []
    question_tags = []
    for i in range(1, QUESTIONS_UPPER_BOUND + 1):
        open_tag = rf"\[\s*(?!/)\s*q{i}\s*\]"
        pattern = re.compile(open_tag, re.IGNORECASE)
        match = pattern.search(text)
        if match:
            question_tags.append((i, match.start(), match.end()))

    question_tags.sort(key=lambda x: x[1])
    
    if not question_tags:
        return [], 0

    for idx, (q_num, start_pos, end_pos) in enumerate(question_tags):
        if idx + 1 < len(question_tags):
            content_end = question_tags[idx + 1][1]
        else:
            content_end = len(text)
        content = text[end_pos:content_end].strip()
        if content:
            all_questions.append(content)
            last_index = q_num
    
    return all_questions, last_index


def extract_reference_paper_indices(agent4_output: str) -> List[int]:
    json_start = agent4_output.find('{')
    json_end = agent4_output.rfind('}') + 1
    if json_start != -1 and json_end > json_start:
        try:
            json_str = _fix_json_escapes(agent4_output[json_start:json_end])
            data = json.loads(json_str)
            numbers = data.get("selected_papers", [])
            numbers = [int(n) for n in numbers if isinstance(n, (int, str)) and str(n).isdigit()]
            return list(dict.fromkeys(numbers))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[extract_reference_paper_indices] JSON parsing failed: {e}")
            return []
    return []


def parse_strategy_json(agent_output: str) -> Tuple[str, List[Dict], str]:
    """Parse Agent6/Agent7 JSON output to extract strategy, todo_list, and draft_response
    
    Returns:
        Tuple of (strategy_text, todo_list, draft_response)
    """
    if not agent_output:
        return "", [], ""
    
    json_start = agent_output.find('{')
    json_end = agent_output.rfind('}') + 1
    
    if json_start == -1 or json_end <= json_start:
        # Fallback: treat entire output as strategy text
        return agent_output, [], ""
    
    try:
        json_str = _fix_json_escapes(agent_output[json_start:json_end])
        data = json.loads(json_str)
        
        strategy_text = data.get("strategy", "")
        todo_list = data.get("todo_list", [])
        draft_response = data.get("draft_response", "")
        
        # Validate todo_list structure
        if isinstance(todo_list, list):
            validated_todos = []
            for item in todo_list:
                if isinstance(item, dict):
                    validated_todos.append({
                        "id": item.get("id", len(validated_todos) + 1),
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "type": item.get("type", "experiment"),
                        "status": item.get("status", "pending"),
                        "related_papers": item.get("related_papers", [])
                    })
            todo_list = validated_todos
        
        return strategy_text, todo_list, draft_response
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[parse_strategy_json] JSON parsing failed: {e}")
        # Fallback: treat entire output as strategy text
        return agent_output, [], ""


class RebuttalService:

    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def _read_text_safe(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""

    def _load_json_safe(self, file_path: str) -> Optional[Dict]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _resolve_existing_path(self, candidates: List[Optional[str]]) -> str:
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return ""

    def _parse_questions_from_logs(self, logs_dir: str) -> List[str]:
        for filename in ("agent2_checker_output.txt", "agent2_output.txt"):
            path = os.path.join(logs_dir, filename)
            if os.path.exists(path):
                text = self._read_text_safe(path)
                if text:
                    questions, _ = extract_review_questions(text)
                    if questions:
                        return questions
        return []

    def _collect_question_ids_from_logs(self, logs_dir: str) -> List[int]:
        qids = set()
        if not os.path.isdir(logs_dir):
            return []
        for fname in os.listdir(logs_dir):
            m = re.match(r"agent(?:3|4|6|7)_q(\d+)_", fname)
            if m:
                qids.add(int(m.group(1)))
                continue
            m = re.match(r"agent7_hitl_q(\d+)_r", fname)
            if m:
                qids.add(int(m.group(1)))
                continue
            m = re.match(r"interaction_q(\d+)\.json", fname)
            if m:
                qids.add(int(m.group(1)))
        return sorted(qids)

    def _find_latest_agent7_output(self, logs_dir: str, question_id: int) -> Tuple[str, int]:
        latest_text = ""
        max_rev = 0
        if not os.path.isdir(logs_dir):
            return "", 0
        for fname in os.listdir(logs_dir):
            m = re.match(rf"agent7_hitl_q{question_id}_r(\d+)_output\.txt", fname)
            if m:
                rev = int(m.group(1))
                if rev >= max_rev:
                    max_rev = rev
                    latest_text = self._read_text_safe(os.path.join(logs_dir, fname))
        if latest_text:
            return latest_text, max_rev
        base_path = os.path.join(logs_dir, f"agent7_q{question_id}_output.txt")
        if os.path.exists(base_path):
            return self._read_text_safe(base_path), 0
        return "", max_rev

    def _extract_hitl_feedback(self, text: str) -> str:
        marker = "[human's feedback]"
        idx = text.lower().find(marker)
        if idx == -1:
            return ""
        tail = text[idx + len(marker):]
        lines = tail.splitlines()
        cleaned = []
        for line in lines:
            if "Please incorporate" in line:
                break
            if line.strip().startswith("```"):
                continue
            cleaned.append(line)
        feedback = "\n".join(cleaned).strip()
        return feedback

    def _load_hitl_feedback_history(self, logs_dir: str, question_id: int) -> Tuple[List[Dict], int]:
        history = []
        max_rev = 0
        if not os.path.isdir(logs_dir):
            return history, max_rev
        for fname in os.listdir(logs_dir):
            m = re.match(rf"agent7_hitl_q{question_id}_r(\d+)_input\.txt", fname)
            if not m:
                continue
            rev = int(m.group(1))
            path = os.path.join(logs_dir, fname)
            text = self._read_text_safe(path)
            feedback = self._extract_hitl_feedback(text)
            if not feedback:
                continue
            try:
                ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
            except Exception:
                ts = ""
            history.append({"feedback": feedback, "timestamp": ts, "_rev": rev})
            max_rev = max(max_rev, rev)
        history.sort(key=lambda x: x.get("_rev", 0))
        for item in history:
            item.pop("_rev", None)
        return history, max_rev

    def _load_interaction_history(self, logs_dir: str, question_id: int) -> Tuple[List[Dict], int]:
        path = os.path.join(logs_dir, f"interaction_q{question_id}.json")
        if not os.path.exists(path):
            return [], 0
        data = self._load_json_safe(path) or {}
        interactions = data.get("interactions", [])
        history = []
        max_rev = 0
        for item in interactions:
            feedback = item.get("user_feedback") or item.get("feedback") or ""
            timestamp = item.get("timestamp", "")
            if feedback:
                entry = {"feedback": feedback, "timestamp": timestamp}
                rev = item.get("revision_number")
                if isinstance(rev, int):
                    entry["_rev"] = rev
                history.append(entry)
            rev = item.get("revision_number")
            if isinstance(rev, int):
                max_rev = max(max_rev, rev)
        if history:
            history.sort(key=lambda x: x.get("_rev", 0))
            for item in history:
                item.pop("_rev", None)
        if max_rev == 0 and history:
            max_rev = len(history)
        return history, max_rev

    def _hydrate_question_from_logs(self, q_state: QuestionState, logs_dir: str) -> None:
        strategy, hitl_rev = self._find_latest_agent7_output(logs_dir, q_state.question_id)
        if strategy and (hitl_rev > 0 or not q_state.agent7_output):
            q_state.agent7_output = strategy
            # Parse strategy JSON if not already parsed
            if not q_state.strategy_text and not q_state.todo_list:
                strategy_text, todo_list, draft_response = parse_strategy_json(strategy)
                q_state.strategy_text = strategy_text
                q_state.todo_list = todo_list
                q_state.draft_response = draft_response
        if hitl_rev > 0 and q_state.revision_count < hitl_rev:
            q_state.revision_count = hitl_rev
        if not q_state.feedback_history:
            history, max_rev = self._load_interaction_history(logs_dir, q_state.question_id)
            hitl_history, hitl_max = self._load_hitl_feedback_history(logs_dir, q_state.question_id)
            if history or hitl_history:
                merged = []
                seen = set()
                for item in history + hitl_history:
                    fb = (item.get("feedback") or "").strip()
                    if not fb:
                        continue
                    if fb in seen:
                        continue
                    seen.add(fb)
                    merged.append(item)
                q_state.feedback_history = merged
            if q_state.revision_count == 0:
                q_state.revision_count = max(max_rev, hitl_max, len(q_state.feedback_history))
        
        # Load history if not already loaded
        if not q_state.history and q_state.agent7_output:
            strategy_text, todo_list, draft_response = parse_strategy_json(q_state.agent7_output)
            q_state.history.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "revision": q_state.revision_count,
                "strategy_text": strategy_text,
                "todo_list": todo_list,
                "draft_response": draft_response,
                "agent7_output": q_state.agent7_output
            })
        
        if q_state.is_satisfied:
            q_state.status = ProcessStatus.COMPLETED
        elif q_state.agent7_output:
            q_state.status = ProcessStatus.WAITING_FEEDBACK

    def _load_session_from_dir(self, session_id: str, session_dir: str) -> Optional[SessionState]:
        if not os.path.isdir(session_dir):
            return None

        logs_dir = os.path.join(session_dir, "logs")
        if not os.path.isdir(logs_dir):
            logs_dir = session_dir
        arxiv_papers_dir = os.path.join(session_dir, "arxiv_papers")
        if not os.path.isdir(arxiv_papers_dir):
            arxiv_papers_dir = session_dir

        token_tracker.log_file = os.path.join(logs_dir, "token_usage.json")

        summary_path = os.path.join(logs_dir, "session_summary.json")
        summary_data = self._load_json_safe(summary_path) if os.path.exists(summary_path) else None

        session = SessionState(
            session_id=session_id,
            session_dir=session_dir,
            logs_dir=logs_dir,
            arxiv_papers_dir=arxiv_papers_dir,
            log_collector=LogCollector(),
        )

        paper_path = ""
        review_path = ""
        if summary_data:
            paper_path = summary_data.get("paper_path", "")
            review_path = summary_data.get("review_path", "")

        session.paper_file_path = self._resolve_existing_path([
            paper_path,
            os.path.join(session_dir, "paper.md"),
            os.path.join(session_dir, "paper.pdf"),
        ])
        session.review_file_path = self._resolve_existing_path([
            review_path,
            os.path.join(session_dir, "review.txt"),
        ])

        final_rebuttal_path = os.path.join(logs_dir, "final_rebuttal.txt")
        if os.path.exists(final_rebuttal_path):
            session.final_rebuttal = self._read_text_safe(final_rebuttal_path)

        questions: List[QuestionState] = []
        if summary_data and isinstance(summary_data.get("questions"), list):
            for q in summary_data.get("questions", []):
                q_state = QuestionState(
                    question_id=int(q.get("question_id", 0) or 0),
                    question_text=q.get("question_text", "") or "",
                    revision_count=int(q.get("revision_count", 0) or 0),
                    is_satisfied=bool(q.get("is_satisfied", False)),
                )
                q_state.agent7_output = q.get("final_strategy", "") or ""
                q_state.feedback_history = q.get("feedback_history", []) or []
                # Load structured data
                q_state.strategy_text = q.get("strategy_text", "") or ""
                q_state.todo_list = q.get("todo_list", []) or []
                q_state.draft_response = q.get("draft_response", "") or ""
                q_state.searched_papers = q.get("searched_papers", []) or []
                q_state.selected_papers = q.get("selected_papers", []) or []
                q_state.analyzed_papers = q.get("analyzed_papers", []) or []
                q_state.history = q.get("history", []) or []
                if q_state.question_id > 0:
                    questions.append(q_state)

            questions.sort(key=lambda x: x.question_id)

        if not questions:
            parsed_questions = self._parse_questions_from_logs(logs_dir)
            if parsed_questions:
                for idx, text in enumerate(parsed_questions, start=1):
                    questions.append(QuestionState(question_id=idx, question_text=text))
            else:
                for qid in self._collect_question_ids_from_logs(logs_dir):
                    questions.append(QuestionState(question_id=qid, question_text=f"Question {qid} (restored)"))

        session.questions = questions
        for q_state in session.questions:
            self._hydrate_question_from_logs(q_state, logs_dir)

        if session.final_rebuttal or (session.questions and all(q.is_satisfied for q in session.questions)):
            session.overall_status = ProcessStatus.COMPLETED
        elif any(q.agent7_output for q in session.questions):
            session.overall_status = ProcessStatus.WAITING_FEEDBACK
        elif session.questions:
            session.overall_status = ProcessStatus.PROCESSING
        else:
            session.overall_status = ProcessStatus.NOT_STARTED

        session.progress_message = "Restored from disk"
        return session

    def restore_session_from_disk(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            existing = self.sessions.get(session_id)
        if existing:
            return existing

        session_dir = os.path.join(SESSIONS_BASE_DIR, session_id)
        session = self._load_session_from_dir(session_id, session_dir)
        if not session:
            return None
        with self._lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = session
        return session

    def restore_sessions_from_disk(self) -> int:
        restored = 0
        try:
            for entry in os.listdir(SESSIONS_BASE_DIR):
                session_dir = os.path.join(SESSIONS_BASE_DIR, entry)
                if not os.path.isdir(session_dir):
                    continue
                with self._lock:
                    already_loaded = entry in self.sessions
                if already_loaded:
                    continue
                session = self._load_session_from_dir(entry, session_dir)
                if session:
                    with self._lock:
                        if entry not in self.sessions:
                            self.sessions[entry] = session
                            restored += 1
        except Exception as e:
            print(f"[ERROR] Failed to restore sessions from disk: {e}")
        return restored
    
    def _get_session_log_dir(self, session_id: str) -> str:
        session = self.get_session(session_id)
        if session and session.logs_dir:
            return session.logs_dir
        if session and session.session_dir:
            return session.session_dir
        return SESSIONS_BASE_DIR
    
    def _get_session_arxiv_dir(self, session_id: str) -> str:
        session = self.get_session(session_id)
        if session and session.arxiv_papers_dir:
            return session.arxiv_papers_dir
        return SESSIONS_BASE_DIR
    
    def _save_interaction_log(self, session_id: str, question_idx: int, 
                               feedback: str, ai_response: str) -> None:

        try:
            session = self.get_session(session_id)
            if not session:
                return
            
            log_dir = self._get_session_log_dir(session_id)
            q_state = session.questions[question_idx]
            

            interaction_log_path = os.path.join(log_dir, f"interaction_q{q_state.question_id}.json")

            interaction_data = {
                "session_id": session_id,
                "question_id": q_state.question_id,
                "question_text": q_state.question_text,
                "interactions": []
            }
            
            if os.path.exists(interaction_log_path):
                try:
                    with open(interaction_log_path, 'r', encoding='utf-8') as f:
                        interaction_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass  
            interaction_record = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "revision_number": q_state.revision_count,
                "user_feedback": feedback,
                "ai_response": ai_response
            }
            interaction_data["interactions"].append(interaction_record)
            

            with open(interaction_log_path, 'w', encoding='utf-8') as f:
                json.dump(interaction_data, f, ensure_ascii=False, indent=2)
            
            print(f"[LOG] Interaction log saved to: {interaction_log_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save interaction log: {e}")
    
    def _save_session_summary(self, session_id: str) -> None:
        """Save session summary containing final strategies and interaction stats for all questions"""
        try:
            session = self.get_session(session_id)
            if not session:
                return
            
            log_dir = self._get_session_log_dir(session_id)
            summary_path = os.path.join(log_dir, "session_summary.json")
            
            # Use lock to prevent concurrent writes
            with self._lock:
                summary_data = {
                    "session_id": session_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "paper_path": session.paper_file_path,
                    "review_path": session.review_file_path,
                    "total_questions": len(session.questions),
                    "questions": []
                }
                
                for q in session.questions:
                    q_summary = {
                        "question_id": q.question_id,
                        "question_text": q.question_text,
                        "revision_count": q.revision_count,
                        "is_satisfied": q.is_satisfied,
                        "final_strategy": q.agent7_output or "",
                        "strategy_text": getattr(q, 'strategy_text', '') or "",
                        "todo_list": getattr(q, 'todo_list', []) or [],
                        "draft_response": getattr(q, 'draft_response', '') or "",
                        "searched_papers": getattr(q, 'searched_papers', []) or [],
                        "selected_papers": getattr(q, 'selected_papers', []) or [],
                        "analyzed_papers": getattr(q, 'analyzed_papers', []) or [],
                        "history": getattr(q, 'history', []) or [],
                        "feedback_history": [
                            {
                                "feedback": h.get("feedback", ""),
                                "timestamp": h.get("timestamp", "")
                            }
                            for h in (q.feedback_history or [])
                        ]
                    }
                    summary_data["questions"].append(q_summary)
                
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, ensure_ascii=False, indent=2)
                
                print(f"[LOG] Session summary saved to: {summary_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save session summary: {e}")
            import traceback
            traceback.print_exc()
    
    def create_session(self, session_id: str, paper_path: str, review_path: str) -> SessionState:

        session_dir = os.path.join(SESSIONS_BASE_DIR, session_id)
        logs_dir = os.path.join(session_dir, "logs")
        arxiv_papers_dir = os.path.join(session_dir, "arxiv_papers")
        
        os.makedirs(session_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(arxiv_papers_dir, exist_ok=True)
        
        token_tracker.log_file = os.path.join(logs_dir, "token_usage.json")

        log_collector = LogCollector()
        
        session = SessionState(
            session_id=session_id,
            paper_file_path=paper_path,
            review_file_path=review_path,
            session_dir=session_dir,
            logs_dir=logs_dir,
            arxiv_papers_dir=arxiv_papers_dir,
            log_collector=log_collector,
        )
        with self._lock:
            self.sessions[session_id] = session

        log_collector.add(f"Session created: {session_id}")
        log_collector.add(f"Session directory: {session_dir}")
        
        print(f"[Session] Created session: {session_id}")
        print(f"  - Session directory: {session_dir}")
        print(f"  - Logs directory: {logs_dir}")
        print(f"  - Papers directory: {arxiv_papers_dir}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self.sessions.get(session_id)
    
    def list_active_sessions(self) -> List[Dict]:

        self.restore_sessions_from_disk()

        sessions_info = []
        with self._lock:
            for session_id, session in self.sessions.items():
                # Calculate progress
                total_questions = len(session.questions)
                completed_questions = sum(1 for q in session.questions if q.is_satisfied)
                processed_questions = sum(1 for q in session.questions if q.agent7_output)
                
                # Determine status text
                if session.overall_status == ProcessStatus.ERROR:
                    status_text = "❌ Error"
                elif total_questions == 0:
                    status_text = "⏳ Initializing..."
                elif completed_questions == total_questions:
                    status_text = "✅ Completed"
                elif processed_questions > 0:
                    status_text = f"📝 Reviewing ({processed_questions}/{total_questions})"
                else:
                    status_text = "⏳ Processing..."
                
                sessions_info.append({
                    "session_id": session_id,
                    "status": status_text,
                    "total_questions": total_questions,
                    "completed_questions": completed_questions,
                    "processed_questions": processed_questions,
                    "current_idx": session.current_question_idx,
                    "display_text": f"[{session_id}] {status_text} - {processed_questions}/{total_questions} questions"
                })
        
        return sessions_info
    
    def run_initial_analysis(
        self, 
        session_id: str, 
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> SessionState:

        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        session.overall_status = ProcessStatus.PROCESSING
        
        def update_progress(msg: str):
            session.progress_message = msg
            if session.log_collector:
                session.log_collector.add(msg)
            if progress_callback:
                progress_callback(msg)
            print(f"[Progress] {msg}")
        
        try:
            update_progress("Converting PDF to Markdown...")
            paper_md_path = pdf_to_md(session.paper_file_path, session.session_dir)
            if not paper_md_path:
                raise RuntimeError("PDF conversion failed")
            session.paper_file_path = paper_md_path  
            update_progress("Agent1: Generating paper summary...")
            agent1 = Agent1(session.paper_file_path, log_dir=session.logs_dir)
            session.paper_summary = agent1.run()

            update_progress("Agent2: Extracting review questions...")
            agent2 = Agent2(session.paper_summary, session.review_file_path, log_dir=session.logs_dir)
            questions_raw = agent2.run()
            
            update_progress("Agent2-Checker: Validating question extraction...")
            checker = Agent2Checker(session.paper_summary, session.review_file_path, questions_raw, log_dir=session.logs_dir)
            questions_checked = checker.run()

            update_progress("Parsing question list...")
            questions, num = extract_review_questions(questions_checked)
            
            if not questions:
                raise RuntimeError("Failed to extract questions from Review")

            session.questions = [
                QuestionState(question_id=i+1, question_text=q)
                for i, q in enumerate(questions)
            ]
            session.current_question_idx = 0
            
            update_progress(f"Analysis complete! Extracted {len(questions)} questions.")
            
        except Exception as e:
            session.overall_status = ProcessStatus.ERROR
            session.progress_message = f"Error: {str(e)}"
            raise
        
        return session
    
    def process_single_question(
        self, 
        session_id: str, 
        question_idx: int,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> QuestionState:

        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if question_idx >= len(session.questions):
            raise ValueError(f"Question index {question_idx} out of range")
        
        q_state = session.questions[question_idx]
        q_state.status = ProcessStatus.PROCESSING
        
        def update_progress(msg: str):
            session.progress_message = msg
            if session.log_collector:
                session.log_collector.add(f"Q{q_state.question_id}: {msg}")
            if progress_callback:
                progress_callback(msg)
            print(f"[Q{q_state.question_id}] {msg}")
        
        try:
            num = q_state.question_id
            question_text = q_state.question_text
            paper_summary = session.paper_summary
            paper_path = session.paper_file_path
            
            update_progress("🔎 正在分析问题并确定搜索策略...")
            agent3 = Agent3(paper_summary, question_text, num=num, log_dir=session.logs_dir)
            agent3.run()
            need_search, queries, links, reason = agent3.extract()
            
            final_target_papers = []

            if links:
                update_progress(f"📎 发现 {len(links)} 个直接提供的论文链接")
                for idx, link in enumerate(links):
                    paper_obj = {
                        "title": f"Provided_Link_Ref_{idx+1}",
                        "arxiv_id": "",
                        "pdf_url": link,
                        "abs_url": link,
                        "authors": ["Reference Link"],
                        "summary": "Directly provided link by analysis agent."
                    }
                    final_target_papers.append(paper_obj)
            
            papers_list_for_agent4_text = ""
            papers_pool_from_search = []
            
            if need_search and queries:
                update_progress(f"📚 正在搜索相关论文（{len(queries)} 个查询）...")
                current_paper_idx = 0
                
                for query_idx, query in enumerate(queries, 1):
                    update_progress(f"🔍 查询 {query_idx}/{len(queries)}: {query[:50]}...")
                    print(f"Searching: {query}")
                    papers = search_relevant_papers(query, max_results=6)
                    print(f"Found {len(papers)} papers")
                    update_progress(f"✓ 找到 {len(papers)} 篇相关论文")
                    
                    for paper in papers:
                        current_paper_idx += 1
                        papers_list_for_agent4_text += "\n------------------\n"
                        papers_list_for_agent4_text += f"[{current_paper_idx}]:{paper}\n"
                        papers_pool_from_search.append(paper)
                
                # Save searched papers
                q_state.searched_papers = papers_pool_from_search.copy()
                

                if papers_pool_from_search:
                    update_progress(f"📑 正在从 {len(papers_pool_from_search)} 篇论文中筛选最相关的...")
                    agent4 = Agent4(papers_list_for_agent4_text, paper_summary, question_text, reason, num=num, log_dir=session.logs_dir)
                    agent4.run()
                    

                    selected_indices = extract_reference_paper_indices(agent4.final_text)
                    print(f"Agent4 selected indices: {selected_indices}")
                    selected_papers_list = []
                    if selected_indices:
                        update_progress(f"✓ 已筛选出 {len(selected_indices)} 篇相关论文")
                        for idx in selected_indices:
                            if 1 <= idx <= len(papers_pool_from_search):
                                paper = papers_pool_from_search[idx-1]
                                final_target_papers.append(paper)
                                selected_papers_list.append(paper)
                    else:
                        # 相关论文即选中论文：Agent4 未返回选中时，将全部搜索到的论文视为选中并参与分析
                        update_progress(f"✓ 将全部搜索到的 {len(papers_pool_from_search)} 篇论文视为选中并分析")
                        selected_papers_list = [dict(p) for p in papers_pool_from_search]
                        for paper in papers_pool_from_search:
                            final_target_papers.append(paper)
                    q_state.selected_papers = selected_papers_list
            
            unique_papers = []
            seen_urls = set()
            for p in final_target_papers:
                url = p.get('pdf_url', '').strip()
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_papers.append(p)
                elif not url:
                    unique_papers.append(p)
            
            # 仅通过链接提供参考文献时，也填充 selected_papers，便于前端展示
            if not q_state.selected_papers and unique_papers:
                q_state.selected_papers = [dict(p) for p in unique_papers]
            
            print(f"[INFO] Final papers to process: {len(unique_papers)}")
            
            reference_paper_summary = []
            
            def _process_single_reference(ti: int, paper_obj: dict) -> Tuple[int, str]:
                try:
                    print(f"\n{'='*80}")
                    print(f"[INFO] Processing reference #{ti} Title: {paper_obj.get('title', 'N/A')[:50]}")
                    print(f"{'='*80}")
                    
                    md_path = download_pdf_and_convert_md(paper_obj, output_dir=session.arxiv_papers_dir)
                    
                    if not md_path:
                        print(f"[ERROR] Paper #{ti} processing failed, skipping")
                        return (ti, "")
                    
                    md_content = ""
                    try:
                        with open(md_path, 'r', encoding='utf-8', errors='ignore') as rf:
                            md_content = rf.read(150000) 
                    except Exception as e:
                        print(f"[ERROR] Failed to read Markdown: {e}")
                        return (ti, "")
                    
                    if not md_content or len(md_content.strip()) < 20:
                        print(f"[ERROR] Markdown content is empty or too short, skipping")
                        return (ti, "")
                    
                    print(f"[STEP 3] Starting Agent5 to analyze reference paper...")
                    agent5 = Agent5(paper_summary, question_text, md_content,
                                   paper_obj.get('abs_url', ''), num=num*100+ti, log_dir=session.logs_dir)
                    agent5_output = agent5.run()
                    print(f"[SUCCESS] Agent5 complete, output length: {len(agent5_output)} characters")
                    
                    return (ti, agent5_output)
                    
                except Exception as e:
                    print(f"[ERROR] Processing reference #{ti} failed: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    return (ti, "")
            
            analysis_by_ti: Dict[int, str] = {}
            if unique_papers:
                update_progress(f"📖 正在分析 {len(unique_papers)} 篇参考文献（并行处理）...")

                # Use all papers as workers for maximum parallelism
                max_workers_ref = len(unique_papers)
                
                with ThreadPoolExecutor(max_workers=max_workers_ref) as pool:
                    futures_ref = [
                        pool.submit(_process_single_reference, idx, paper_obj)
                        for idx, paper_obj in enumerate(unique_papers, start=1)
                    ]
                    

                    ref_results = [f.result() for f in as_completed(futures_ref)]

                    ref_results.sort(key=lambda x: x[0])
                    # ti 为 1-based 索引，建立 ti -> analysis 映射，保证与 unique_papers 顺序一致
                    analysis_by_ti = {ti: r for ti, r in ref_results if r}
            
            reference_paper_summary_list = [analysis_by_ti.get(i, "") for i in range(1, len(unique_papers) + 1)] if unique_papers else []
            q_state.reference_paper_summary = "\n\n".join(s for s in reference_paper_summary_list if s)
            
            # Save analyzed papers info（按 unique_papers 顺序，analysis 与索引一一对应）
            analyzed_papers_list = []
            for idx, paper_obj in enumerate(unique_papers):
                ti = idx + 1
                analysis_text = analysis_by_ti.get(ti, "")
                paper_info = {
                    "title": paper_obj.get("title", ""),
                    "arxiv_id": paper_obj.get("arxiv_id", ""),
                    "abs_url": paper_obj.get("abs_url", ""),
                    "pdf_url": paper_obj.get("pdf_url", ""),
                    "authors": paper_obj.get("authors", []),
                    "abstract": paper_obj.get("abstract", ""),
                    "analysis": analysis_text,
                }
                analyzed_papers_list.append(paper_info)
            q_state.analyzed_papers = analyzed_papers_list
            
            update_progress("💡 正在生成反驳策略...")
            original_paper = _read_text(paper_path)
            agent6 = Agent6(original_paper, question_text, q_state.reference_paper_summary, num=num, log_dir=session.logs_dir)
            q_state.agent6_output = agent6.run()
            
            # Parse Agent6 output
            strategy_text_6, todo_list_6, draft_response_6 = parse_strategy_json(q_state.agent6_output)
            
            update_progress("✨ 正在优化反驳策略...")
            agent7 = Agent7(q_state.agent6_output, original_paper, question_text,
                           q_state.reference_paper_summary, num=num, log_dir=session.logs_dir)
            q_state.agent7_output = agent7.run()
            
            # Parse Agent7 output and save structured data
            strategy_text, todo_list, draft_response = parse_strategy_json(q_state.agent7_output)
            q_state.strategy_text = strategy_text
            q_state.todo_list = todo_list
            q_state.draft_response = draft_response
            
            # Save initial history state
            q_state.history.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "revision": 0,
                "strategy_text": strategy_text,
                "todo_list": todo_list,
                "draft_response": draft_response,
                "agent6_output": q_state.agent6_output,
                "agent7_output": q_state.agent7_output,
                "searched_papers": q_state.searched_papers.copy() if q_state.searched_papers else [],
                "selected_papers": q_state.selected_papers.copy() if q_state.selected_papers else [],
                "analyzed_papers": [p.copy() for p in q_state.analyzed_papers] if q_state.analyzed_papers else []
            })
            
            # Save session summary after each question is processed
            # Use service lock to prevent concurrent writes
            try:
                self._save_session_summary(session_id)
            except Exception as e:
                print(f"[WARNING] Failed to save session summary: {e}")
            
            q_state.status = ProcessStatus.WAITING_FEEDBACK
            update_progress("✅ 处理完成，等待您的反馈...")
            
        except Exception as e:
            q_state.status = ProcessStatus.ERROR
            session.progress_message = f"Question {q_state.question_id} processing error: {str(e)}"
            raise
        
        return q_state
    
    def revise_with_feedback(
        self, 
        session_id: str, 
        question_idx: int, 
        human_feedback: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> QuestionState:

        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        q_state = session.questions[question_idx]
        
        def update_progress(msg: str):
            session.progress_message = msg
            if session.log_collector:
                session.log_collector.add(f"Q{q_state.question_id}: {msg}")
            if progress_callback:
                progress_callback(msg)
            print(f"[HITL Q{q_state.question_id}] {msg}")
        
        try:
            update_progress("🔄 正在根据您的反馈修订策略...")
            
            original_paper = _read_text(session.paper_file_path)
            
            agent7h = Agent7WithHumanFeedback(
                current_strategy=q_state.agent7_output,
                paper_summary=original_paper,
                review_question=q_state.question_text,
                reference_summary=q_state.reference_paper_summary,
                human_feedback=human_feedback,
                num=q_state.question_id,
                log_dir=session.logs_dir
            )
            
            new_strategy = agent7h.run()

            # Parse new strategy JSON
            strategy_text, todo_list, draft_response = parse_strategy_json(new_strategy)
            
            q_state.feedback_history.append({
                "feedback": human_feedback,
                "previous_strategy": q_state.agent7_output,
                "new_strategy": new_strategy,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            
            # Save to history
            q_state.history.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "revision": q_state.revision_count + 1,
                "feedback": human_feedback,
                "strategy_text": strategy_text,
                "todo_list": todo_list,
                "draft_response": draft_response,
                "agent7_output": new_strategy
            })

            q_state.agent7_output = new_strategy
            q_state.strategy_text = strategy_text
            q_state.todo_list = todo_list
            q_state.draft_response = draft_response
            q_state.revision_count += 1
            
            self._save_interaction_log(session_id, question_idx, human_feedback, new_strategy)
            
            update_progress(f"✅ 修订完成！这是第 {q_state.revision_count} 次修订。")
            
        except Exception as e:
            session.progress_message = f"Revision failed: {str(e)}"
            raise
        
        return q_state
    
    def mark_question_satisfied(self, session_id: str, question_idx: int) -> QuestionState:
        """Mark question as satisfied, ready to proceed to next question"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        q_state = session.questions[question_idx]
        q_state.is_satisfied = True
        q_state.status = ProcessStatus.COMPLETED
        
        if question_idx + 1 < len(session.questions):
            session.current_question_idx = question_idx + 1
        
        self._save_session_summary(session_id)
        
        return q_state
    
    def process_all_questions_parallel(
        self,
        session_id: str,
        max_workers: int = 3,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[QuestionState]:

        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        def update_progress(msg: str):
            session.progress_message = msg
            if session.log_collector:
                session.log_collector.add(f"[并行处理] {msg}")
            if progress_callback:
                progress_callback(msg)
            print(f"[Parallel] {msg}")
        
        num_questions = len(session.questions)
        if num_questions == 0:
            raise RuntimeError("No questions to process")
        
        max_workers_actual = min(max_workers, num_questions)
        
        update_progress(f"⚙️ 准备并行处理 {num_questions} 个问题（使用 {max_workers_actual} 个工作线程）")
        
        results_map: Dict[int, QuestionState] = {}
        
        def _process_question_wrapper(idx: int) -> Tuple[int, QuestionState]:
            """Wrapper function for parallel processing"""
            try:
                q_state = self.process_single_question(session_id, idx)
                # Note: _save_session_summary is already called in process_single_question
                return (idx, q_state)
            except Exception as e:
                print(f"[ERROR] Question {idx+1} processing failed: {e}")
                import traceback
                traceback.print_exc()
                session = self.get_session(session_id)
                if session and idx < len(session.questions):
                    session.questions[idx].status = ProcessStatus.ERROR
                    # Save error state
                    try:
                        self._save_session_summary(session_id)
                    except:
                        pass
                    return (idx, session.questions[idx])
                # Return empty state if session is gone
                return (idx, QuestionState(question_id=idx+1, question_text="", status=ProcessStatus.ERROR))
        
        with ThreadPoolExecutor(max_workers=max_workers_actual) as executor:
            future_to_idx = {}
            for i in range(num_questions):
                fut = executor.submit(_process_question_wrapper, i)
                future_to_idx[fut] = i
            
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    _, q_state = fut.result()
                    results_map[idx] = q_state
                    update_progress(f"✓ 问题 {idx+1}/{num_questions} 处理完成")
                except Exception as e:
                    print(f"[ERROR] Failed to get result for question {idx+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    # 仍写入当前 session 中该问题的状态，避免后续覆盖时丢失
                    session_ref = self.get_session(session_id)
                    if session_ref and idx < len(session_ref.questions):
                        results_map[idx] = session_ref.questions[idx]
        
        ordered_results = []
        for i in range(num_questions):
            if i in results_map:
                ordered_results.append(results_map[i])
                session.questions[i] = results_map[i]
            else:
                ordered_results.append(session.questions[i])
        
        session = self.get_session(session_id)
        if session:
            # Set status to WAITING_FEEDBACK so SSE stream can detect completion
            session.overall_status = ProcessStatus.WAITING_FEEDBACK
            session.progress_message = f"✅ 所有 {num_questions} 个问题处理完成！"
            # Final save after all questions are processed
            self._save_session_summary(session_id)
        
        update_progress(f"✅ 所有 {num_questions} 个问题处理完成！")
        
        return ordered_results
    
    def generate_summary_markdown(self, session_id: str) -> str:
        """Generate a markdown summary of all questions, strategies, and rebuttal"""
        session = self.get_session(session_id)
        if not session:
            session = self.restore_session_from_disk(session_id)
        if not session:
            return ""
        
        lines = []
        lines.append("# 反驳信生成报告\n\n")
        lines.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Session ID**: {session_id}\n")
        lines.append(f"**问题总数**: {len(session.questions)}\n\n")
        lines.append("---\n\n")
        
        # Process each question
        for q in session.questions:
            lines.append(f"## 问题 {q.question_id}\n\n")
            lines.append(f"### 评审问题\n\n{q.question_text}\n\n")
            
            # Strategy
            strategy_text = getattr(q, 'strategy_text', '') or q.agent7_output or ""
            if strategy_text:
                lines.append(f"### 反驳策略\n\n{strategy_text}\n\n")
            
            # Todo list
            todo_list = getattr(q, 'todo_list', []) or []
            if todo_list:
                lines.append("### 待办事项列表\n\n")
                for todo in todo_list:
                    status_icon = "✅" if todo.get("status") == "completed" else "⏳"
                    type_emoji = {
                        "experiment": "🧪",
                        "analysis": "📊",
                        "clarification": "📝",
                        "comparison": "⚖️",
                        "ablation": "🔬"
                    }.get(todo.get("type", ""), "📋")
                    lines.append(f"- {status_icon} {type_emoji} **{todo.get('title', 'N/A')}**\n")
                    if todo.get('description'):
                        desc = todo.get('description', '').strip()
                        # Indent description
                        for desc_line in desc.split('\n'):
                            if desc_line.strip():
                                lines.append(f"  - {desc_line.strip()}\n")
                    if todo.get('related_papers'):
                        papers_str = ', '.join(todo.get('related_papers', []))
                        lines.append(f"  - 📚 相关论文: {papers_str}\n")
                lines.append("\n")
            
            # Related papers
            analyzed_papers = getattr(q, 'analyzed_papers', []) or []
            searched_papers = getattr(q, 'searched_papers', []) or []
            selected_papers = getattr(q, 'selected_papers', []) or []
            
            if analyzed_papers or searched_papers:
                lines.append("### 相关论文\n\n")
                
                if analyzed_papers:
                    lines.append("#### 已分析论文\n\n")
                    for paper in analyzed_papers:
                        lines.append(f"- **{paper.get('title', 'N/A')}**\n")
                        if paper.get('authors'):
                            authors_str = ', '.join(paper.get('authors', [])[:3])
                            if len(paper.get('authors', [])) > 3:
                                authors_str += " et al."
                            lines.append(f"  - 作者: {authors_str}\n")
                        if paper.get('abs_url'):
                            lines.append(f"  - 链接: {paper.get('abs_url')}\n")
                        if paper.get('analysis'):
                            analysis = paper.get('analysis', '').strip()
                            if analysis:
                                # Limit analysis length
                                if len(analysis) > 300:
                                    analysis = analysis[:300] + "..."
                                lines.append(f"  - 分析摘要: {analysis}\n")
                        lines.append("\n")
                
                if searched_papers and len(searched_papers) > len(analyzed_papers):
                    lines.append("#### 搜索到的其他论文\n\n")
                    for paper in searched_papers[:5]:  # Limit to 5
                        if paper.get('title'):
                            lines.append(f"- **{paper.get('title', 'N/A')}**\n")
                            if paper.get('abs_url'):
                                lines.append(f"  - 链接: {paper.get('abs_url')}\n")
            
            lines.append("---\n\n")
        
        # Final rebuttal
        if session.final_rebuttal:
            lines.append("## 最终反驳信\n\n")
            lines.append(session.final_rebuttal)
            lines.append("\n")
        
        return "".join(lines)
    
    def generate_final_rebuttal(
        self, 
        session_id: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:

        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        def update_progress(msg: str):
            session.progress_message = msg
            if progress_callback:
                progress_callback(msg)
            print(f"[Final] {msg}")
        
        unsatisfied = [q for q in session.questions if not q.is_satisfied]
        if unsatisfied:
            raise RuntimeError(f"{len(unsatisfied)} questions not yet confirmed as satisfied")
        
        try:
            all_strategies = []
            for q in session.questions:
                block = (
                    f"\n## Q[{q.question_id}]:\n"
                    f"```review_question\n{q.question_text}\n```\n"
                    f"\n[Rebuttal Strategy & To-Do List]:\n{q.agent7_output}\n"
                )
                all_strategies.append(block)
            
            combined = "\n".join(all_strategies)
            
            update_progress("📝 正在生成反驳信草稿...")
            original_paper = _read_text(session.paper_file_path)
            agent8 = Agent8(combined, original_paper, session.review_file_path, log_dir=session.logs_dir)
            draft = agent8.run()
            

            update_progress("🔍 正在校对并生成最终版本...")
            agent9 = Agent9(draft, combined, original_paper, session.review_file_path, log_dir=session.logs_dir)
            session.final_rebuttal = agent9.run()
            
            session.overall_status = ProcessStatus.COMPLETED
            update_progress("✅ 反驳信生成完成！")
            
            with open(os.path.join(session.logs_dir, "final_rebuttal.txt"), "w", encoding="utf-8") as f:
                f.write(session.final_rebuttal)
            
            # Generate and save summary markdown
            summary_md = self.generate_summary_markdown(session_id)
            summary_path = os.path.join(session.logs_dir, "summary.md")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_md)
            print(f"[LOG] Summary markdown saved to: {summary_path}")
            
            token_tracker.print_summary()
            token_tracker.export_to_file()
            
        except Exception as e:
            session.overall_status = ProcessStatus.ERROR
            session.progress_message = f"Final rebuttal generation failed: {str(e)}"
            raise
        
        return session.final_rebuttal

rebuttal_service = RebuttalService()
