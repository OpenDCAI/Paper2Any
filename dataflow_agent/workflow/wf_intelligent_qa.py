from __future__ import annotations
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from dataflow_agent.workflow.registry import register
from dataflow_agent.graphbuilder.graph_builder import GenericGraphBuilder
from dataflow_agent.logger import get_logger
from dataflow_agent.state import IntelligentQAState
from dataflow_agent.agentroles import create_vlm_agent, create_simple_agent
from dataflow_agent.utils import get_project_root

log = get_logger(__name__)

# Try importing office libraries
try:
    from docx import Document
except ImportError:
    Document = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None

@register("intelligent_qa")
def create_intelligent_qa_graph() -> GenericGraphBuilder:
    """
    Workflow for Intelligent Q&A
    Steps:
    1. Parse uploaded files (PDF/Office/Image/Video)
    2. Aggregate context
    3. Generate answer using LLM
    """
    builder = GenericGraphBuilder(state_model=IntelligentQAState, entry_point="_start_")

    def _start_(state: IntelligentQAState) -> IntelligentQAState:
        # Ensure request fields
        if not state.request.files:
            state.request.files = []
        if not state.request.query:
            state.request.query = ""
        return state

    async def parallel_parse_node(state: IntelligentQAState) -> IntelligentQAState:
        """
        Parallel parsing of all files in state.request.files
        """
        files = state.request.files
        if not files:
            state.context_content = ""
            return state

        parsed_contents = []

        async def parse_file(file_path: str) -> str:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return f"[Error: File not found {file_path}]"
            
            suffix = file_path_obj.suffix.lower()
            filename = file_path_obj.name

            content = f"--- File: {filename} ---\n"
            
            try:
                # 1. PDF Parsing (PyMuPDF)
                if suffix == ".pdf":
                    try:
                        doc = fitz.open(file_path)
                        text = ""
                        for page in doc:
                            text += page.get_text() + "\n"
                        content += text
                    except Exception as e:
                        log.error(f"Error parsing PDF {file_path}: {e}")
                        content += f"[Error parsing PDF: {e}]"

                # 2. Word Parsing (python-docx)
                elif suffix in [".docx", ".doc"]:
                    if Document is None:
                         content += "[Error: python-docx not installed]"
                    else:
                        try:
                            doc = Document(file_path)
                            text = "\n".join([p.text for p in doc.paragraphs])
                            content += text
                        except Exception as e:
                             log.error(f"Error parsing Docx {file_path}: {e}")
                             content += f"[Error parsing Docx: {e}]"

                # 3. PPT Parsing (python-pptx)
                elif suffix in [".pptx", ".ppt"]:
                    if Presentation is None:
                        content += "[Error: python-pptx not installed]"
                    else:
                        try:
                            prs = Presentation(file_path)
                            text = ""
                            for slide in prs.slides:
                                for shape in slide.shapes:
                                    if hasattr(shape, "text"):
                                        text += shape.text + "\n"
                            content += text
                        except Exception as e:
                            log.error(f"Error parsing PPT {file_path}: {e}")
                            content += f"[Error parsing PPT: {e}]"
                
                # 4. Image/Video Parsing (VLM)
                elif suffix in [".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi"]:
                    try:
                        # Use create_vlm_agent
                        # For video, mode might need to be "video_understanding"
                        # But create_vlm_agent default "understanding" supports images.
                        # For video, we check suffix.
                        vlm_mode = "understanding"
                        if suffix in [".mp4", ".mov", ".avi"]:
                             vlm_mode = "video_understanding"
                             input_key = "input_video"
                        else:
                             input_key = "input_image"

                        # Create temp state for agent execution
                        # We need to construct a state that the agent can use.
                        # create_vlm_agent returns an Agent. 
                        # Agent.execute(state) requires a state that has request.chat_api_url etc.
                        
                        agent = create_vlm_agent(
                            name=f"vlm_parser_{filename}",
                            vlm_mode=vlm_mode,
                            model_name="gemini-2.5-flash", # Use Gemini Flash for speed/multimodal
                            chat_api_url=state.request.chat_api_url,
                            additional_params={
                                input_key: file_path
                            }
                        )
                        
                        # We need to pass a prompt to VLM to describe the content
                        # Since BaseAgent usually takes state and looks for 'messages' or 'target',
                        # we should construct a prompt.
                        
                        # HACK: construct a temporary state with a user message
                        from dataflow_agent.state import MainState
                        from langchain_core.messages import HumanMessage
                        
                        temp_req = state.request
                        temp_state = MainState(request=temp_req)
                        temp_state.messages = [HumanMessage(content="Please describe the content of this media file in detail.")]
                        
                        # Execute
                        res_state = await agent.execute(temp_state)
                        
                        # Extract result
                        # Simple/VLM Agent puts result in messages (last AI Message)
                        if res_state.messages and res_state.messages[-1].type == "ai":
                             content += res_state.messages[-1].content
                        else:
                             content += "[VLM returned no content]"

                    except Exception as e:
                        log.error(f"Error parsing Media {file_path}: {e}")
                        content += f"[Error parsing Media: {e}]"
                
                else:
                    # Text fallback
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content += f.read()
                    except:
                        content += "[Unsupported file type]"

            except Exception as e:
                 content += f"[Generic Error: {e}]"
            
            return content + "\n"

        # Run in parallel
        tasks = [parse_file(f) for f in files]
        results = await asyncio.gather(*tasks)
        
        state.context_content = "\n".join(results)
        return state

    async def chat_node(state: IntelligentQAState) -> IntelligentQAState:
        """
        Chat with context
        """
        # Construct history string
        history_str = ""
        for msg in state.request.history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
            
        system_prompt = f"""You are a helpful intelligent assistant.
You have access to the following file contents as context:

{state.context_content}

Conversation History:
{history_str}

User Question: {state.request.query}

Please answer the user's question based on the provided context. If the answer is not in the context, use your general knowledge but prioritize the context.
Answer in the language of the user's question (mostly Chinese).
"""
        
        # Use Simple Agent for QA
        agent = create_simple_agent(
            name="qa_agent",
            model_name=state.request.model,
            chat_api_url=state.request.chat_api_url,
            temperature=0.7
        )
        
        # We need to pass the prompt. 
        # SimpleConfig agents usually use 'target' or messages.
        # Let's override state.messages with our prompt.
        from langchain_core.messages import HumanMessage
        
        # Clear messages in state to avoid carrying over internal noise, set our prompt
        state.messages = [HumanMessage(content=system_prompt)]
        
        new_state = await agent.execute(state)
        
        if new_state.messages and new_state.messages[-1].type == "ai":
            state.answer = new_state.messages[-1].content
        else:
            state.answer = "Sorry, I couldn't generate an answer."
            
        return state

    nodes = {
        "_start_": _start_,
        "parallel_parse": parallel_parse_node,
        "chat": chat_node,
        "_end_": lambda s: s
    }

    edges = [
        ("_start_", "parallel_parse"),
        ("parallel_parse", "chat"),
        ("chat", "_end_")
    ]

    builder.add_nodes(nodes).add_edges(edges)
    return builder
