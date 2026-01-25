"""
Prompt Templates for qa_agent
Generated at: 2026-01-21 18:15:27
"""

# --------------------------------------------------------------------------- #
# 1. QaAgent - qa_agent 相关提示词
# --------------------------------------------------------------------------- #
class QaAgent:
    """
    qa_agent 任务的提示词模板
    """
    
    # ----------------------------------------------------------------------
    # System Prompt
    # ----------------------------------------------------------------------
    system_prompt_for_qa_agent = """
You are an intelligent knowledge base assistant. 
Your goal is to help users understand files and answer their questions based on the provided content.
"""

    # ----------------------------------------------------------------------
    # Task Prompt 1: Single File Analysis
    # ----------------------------------------------------------------------
    file_analysis_prompt = """
You are provided with the content of a single file.
Filename: {filename}
File Type: {file_type}

Content:
{content}

User Question: {query}

Please analyze this file content specifically in the context of the user's question.
If the file contains information relevant to the question, summarize it and explain how it relates.
If the file is irrelevant, briefly state that it contains no relevant information.
"""

    # ----------------------------------------------------------------------
    # Task Prompt 2: Final Synthesis
    # ----------------------------------------------------------------------
    final_qa_prompt = """
You are provided with analyses from multiple files regarding a user's question.

User Question: {query}

File Analyses:
{file_analyses}

Conversation History:
{history}

Based on the above analyses and history, provide a comprehensive and final answer to the user's question.
Cite specific files where information is drawn from.
Answer in the same language as the user's question (likely Chinese).
"""

    # Default task prompt if needed
    task_prompt_for_qa_agent = """
Your task description here.
Input: {input_data}
"""
