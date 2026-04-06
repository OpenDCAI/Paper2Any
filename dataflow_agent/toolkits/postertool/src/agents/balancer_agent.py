"""
column space balancer
"""

import json
from typing import Dict, List, Any
from src.state.poster_state import PosterState
from utils.langgraph_utils import load_prompt, LangGraphAgent, extract_json
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error

class BalancerAgent:
    def __init__(self):
        self.name = "balancer_agent"
        self.balancer_prompt = load_prompt("config/prompts/layout_balancer.txt")

    def __call__(self, initial_layout_data: Dict, column_analysis: Dict, 
                 state: PosterState) -> Dict:
        """optimize column space distribution"""
        
        log_agent_info(self.name, "optimizing column balance")
        
        structured_sections = state.get("structured_sections")
        story_board = state.get("story_board")
        
        columns = column_analysis['columns']
        left_rate = columns['left']['utilization_rate']
        middle_rate = columns['middle']['utilization_rate'] 
        right_rate = columns['right']['utilization_rate']
        
        log_agent_info(self.name, f"utilization - left: {left_rate:.1%}, middle: {middle_rate:.1%}, right: {right_rate:.1%}")
        
        agent = LangGraphAgent("layout optimization specialist", state["text_model"])
        
        variables = {
            "structured_sections": json.dumps(structured_sections, indent=2),
            "current_story_board": json.dumps(story_board, indent=2),
            "column_analysis": json.dumps(column_analysis, indent=2),
            "available_height": column_analysis["available_height"],
            "left_utilization": f"{left_rate:.1%}",
            "middle_utilization": f"{middle_rate:.1%}",
            "right_utilization": f"{right_rate:.1%}",
            "left_status": columns['left']['status'],
            "middle_status": columns['middle']['status'], 
            "right_status": columns['right']['status']
        }
        
        MAX_ATTEMPTS = 3
        for attempt in range(MAX_ATTEMPTS):
            prompt = self.balancer_prompt.format(**variables)
            response = agent.step(prompt)
            
            log_agent_info(self.name, f"attempt {attempt + 1}: response {len(response.content)} chars")
            
            try:
                optimized_story_board = extract_json(response.content)
                
                if self._validate_story_board(optimized_story_board):
                    log_agent_success(self.name, f"optimized on attempt {attempt + 1}")
                    return {
                        "optimized_story_board": optimized_story_board,
                        "balancer_decisions": self._extract_decisions(response.content),
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens
                    }
                else:
                    log_agent_error(self.name, f"attempt {attempt + 1}: validation failed")
                    
            except Exception as e:
                log_agent_error(self.name, f"attempt {attempt + 1}: json extraction failed - {str(e)}")
        
        log_agent_error(self.name, f"failed after {MAX_ATTEMPTS} attempts")
        return {"optimized_story_board": story_board, "balancer_decisions": {}}

    def _validate_story_board(self, story_board: Dict) -> bool:
        """validate story board structure"""
        if "spatial_content_plan" not in story_board:
            return False
        
        scp = story_board["spatial_content_plan"]
        if "sections" not in scp or not isinstance(scp["sections"], list):
            return False
            
        for section in scp["sections"]:
            if section is None:
                log_agent_error(self.name, "null section found")
                return False
            if not isinstance(section, dict):
                log_agent_error(self.name, f"invalid section type: {type(section)}")
                return False
            if "column_assignment" not in section:
                return False
            if section["column_assignment"] not in ["left", "middle", "right"]:
                return False
                
        return True

    def _extract_decisions(self, response_content: str) -> Dict:
        """extract optimization decisions from response"""
        decisions = {
            "text_adjustments": [],
            "section_additions": [],
            "section_removals": [],
            "optimizations": []
        }
        
        content_patterns = ["expanded text", "added detail", "enhanced content", "increased content",
                          "reduced text", "shortened", "condensed content", "decreased content"]
        addition_patterns = ["added section", "included section", "new section"]
        removal_patterns = ["removed section", "deleted section", "eliminated section"]
        optimization_patterns = ["within column", "column optimization", "adjusted in", "optimized in"]
        
        for line in response_content.split('\n'):
            line_lower = line.lower()
            if any(p in line_lower for p in content_patterns):
                decisions["text_adjustments"].append(line.strip())
            elif any(p in line_lower for p in addition_patterns):
                decisions["section_additions"].append(line.strip())
            elif any(p in line_lower for p in removal_patterns):
                decisions["section_removals"].append(line.strip())
            elif any(p in line_lower for p in optimization_patterns):
                decisions["optimizations"].append(line.strip())
                
        return decisions


def balancer_agent_node(state: PosterState) -> Dict[str, Any]:
    """balancer agent node for langgraph"""
    try:
        agent = BalancerAgent()
        result = agent(state.get("initial_layout_data"), 
                      state.get("column_analysis"), 
                      state)
        
        state["tokens"].add_text(
            result.get("input_tokens", 0),
            result.get("output_tokens", 0)
        )
        
        return {
            **state,
            "optimized_story_board": result["optimized_story_board"],
            "balancer_decisions": result["balancer_decisions"],
            "current_agent": "balancer_agent"
        }
    except Exception as e:
        log_agent_error("balancer_agent", f"error: {e}")
        return {**state, "errors": state.get("errors", []) + [f"balancer_agent: {e}"]}