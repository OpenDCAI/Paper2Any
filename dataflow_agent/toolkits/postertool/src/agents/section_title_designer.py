"""
Section title Designer
- Fixed style for current version
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from src.state.poster_state import PosterState
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error


class SectionTitleDesigner:
    def __init__(self):
        self.name = "section_title_designer"

    def __call__(self, state: PosterState) -> PosterState:
        log_agent_info(self.name, "generating section title styling (code-based, Style 2 only)")
        
        try:
            story_board = state.get("story_board")
            color_scheme = state.get("color_scheme")
            
            if not story_board:
                log_agent_error(self.name, "missing story_board")
                raise ValueError("missing story_board from curator")
            
            if not color_scheme:
                log_agent_error(self.name, "missing color_scheme")
                raise ValueError("missing color_scheme from color agent")
            
            title_design = self._generate_colorblock_design(story_board, color_scheme)
            
            state["section_title_design"] = title_design
            state["current_agent"] = self.name
            
            self._save_title_design(state)
            
            log_agent_success(self.name, "generated section title styling")
            log_agent_info(self.name, f"theme color: {color_scheme.get('theme', 'unknown')}")

        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")
            
        return state

    def _generate_colorblock_design(self, story_board: Dict, color_scheme: Dict) -> Dict:
        """Generate colorblock design"""
        
        sections = story_board.get("spatial_content_plan", {}).get("sections", [])
        
        # color mapping from color_scheme for rectangle_left template
        colors = self._map_rectangle_colors(color_scheme)
        
        # applications for all sections
        applications = self._generate_rectangle_applications(sections, colors)
        
        return {
            "section_title_design": {
                "selected_template": "rectangle_left",
                "design_rationale": "Code-generated rectangle_left template for modern, design-forward appearance with color accent",
                "color_palette": colors,
                "spacing_specifications": {
                    "title_left_padding": "4_spaces",
                    "rectangle_to_content_gap": 0.15
                },
                "section_applications": applications
            }
        }

    def _map_rectangle_colors(self, color_scheme: Dict) -> Dict:
        """Map color scheme to rectangle_left template colors"""
        
        theme_color = color_scheme.get("theme", "#1E3A8A")
        mono_light = color_scheme.get("mono_light", "#335f91")
        mono_dark = color_scheme.get("mono_dark", "#002c5e")
        
        return {
            "theme_color": theme_color,
            "mono_light": mono_light,
            "mono_dark": mono_dark,
            "title_text_color": "#000000",  # black for readability on colored background
            "accent_rectangle_color": theme_color,
            "background_color": "#FFFFFF"
        }

    def _generate_rectangle_applications(self, sections: List[Dict], colors: Dict) -> List[Dict]:
       
        applications = []
        
        for section in sections:
            application = {
                "section_id": section["section_id"],
                "section_title": section.get("section_title", ""),
                "title_styling": {
                    "font_family": "Helvetica Neue",
                    "font_size": 48,  # this will be overridden by styling_interfaces font_sizes
                    "font_weight": "bold",
                    "color": colors["title_text_color"],
                    "alignment": "left"
                },
                "accent_styling": {
                    "type": "rectangle",
                    "color": colors["accent_rectangle_color"],
                    "dimensions": {"width": "golden_ratio_based_on_height", "height": "title_height"},
                    "position": "same_row"
                }
            }
            
            applications.append(application)
        
        return applications

    def _save_title_design(self, state: PosterState):
        """Save title design to json file"""
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "section_title_design.json", "w", encoding='utf-8') as f:
            json.dump(state.get("section_title_design", {}), f, indent=2)


def section_title_designer_node(state: PosterState) -> Dict[str, Any]:
    result = SectionTitleDesigner()(state)
    return {
        **state,
        "section_title_design": result["section_title_design"],
        "current_agent": result["current_agent"],
        "errors": result["errors"]
    }