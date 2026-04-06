"""
3-phase layout optimization orchestrator
"""

import json
from pathlib import Path
from typing import Dict, Any
from src.state.poster_state import PosterState
from src.agents.layout_agent import LayoutAgent
from src.agents.balancer_agent import BalancerAgent
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error

class LayoutWithBalancerAgent:
    def __init__(self):
        self.name = "layout_with_balancer"
        self.layout_agent = LayoutAgent()
        self.balancer_agent = BalancerAgent()

    def __call__(self, state: PosterState) -> PosterState:
        """execute 3-phase layout optimization"""
        log_agent_info(self.name, "starting 3-phase layout optimization")
        
        try:
            # phase 1: initial layout generation
            log_agent_info(self.name, "phase 1: generating initial layout")
            initial_state = self.layout_agent(state, mode="initial")
            if initial_state.get("errors"):
                return initial_state
            
            # phase 2: balancer optimization  
            log_agent_info(self.name, "phase 2: optimizing with balancer")
            balancer_result = self.balancer_agent(
                initial_layout_data=initial_state["initial_layout_data"],
                column_analysis=initial_state["column_analysis"],
                state=initial_state
            )
            
            # save balancer decisions
            self._save_balancer_output(balancer_result, initial_state)
            
            # update state with optimized story board
            initial_state["optimized_story_board"] = balancer_result["optimized_story_board"]
            initial_state["balancer_decisions"] = balancer_result["balancer_decisions"]
            
            # phase 3: final layout generation
            log_agent_info(self.name, "phase 3: generating final layout")
            final_state = self.layout_agent(initial_state, mode="final")
            if final_state.get("errors"):
                return final_state
            
            # update token counts
            final_state["tokens"].add_text(
                balancer_result.get("input_tokens", 0),
                balancer_result.get("output_tokens", 0)
            )
            
            log_agent_success(self.name, "3-phase layout optimization complete")
            return final_state
            
        except Exception as e:
            log_agent_error(self.name, f"3-phase optimization error: {e}")
            return {"errors": [f"{self.name}: {e}"]}

    def _save_balancer_output(self, balancer_result: Dict, state: PosterState):
        """save balancer optimization results"""
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "optimized_story_board.json", "w", encoding='utf-8') as f:
            json.dump(balancer_result["optimized_story_board"], f, indent=2)
        
        with open(output_dir / "balancer_decisions.json", "w", encoding='utf-8') as f:
            json.dump(balancer_result["balancer_decisions"], f, indent=2)


def layout_with_balancer_node(state: PosterState) -> Dict[str, Any]:
    """layout with balancer node for langgraph"""
    try:
        agent = LayoutWithBalancerAgent()
        result = agent(state)
        
        return {
            **state,
            "design_layout": result.get("design_layout"),
            "optimized_column_assignment": result.get("optimized_column_assignment"),
            "optimized_story_board": result.get("optimized_story_board"),
            "balancer_decisions": result.get("balancer_decisions"),
            "tokens": result.get("tokens"),
            "current_agent": result.get("current_agent"),
            "errors": result.get("errors", [])
        }
    except Exception as e:
        log_agent_error("layout_with_balancer", f"node error: {e}")
        return {**state, "errors": state.get("errors", []) + [f"layout_with_balancer: {e}"]}