"""
precise layout generation using css box model
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from src.state.poster_state import PosterState
from utils.langgraph_utils import LangGraphAgent, extract_json, load_prompt
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error, log_agent_warning
from src.layout.text_height_measurement import measure_text_height
from src.config.poster_config import load_config

class LayoutAgent:
    """creates optimized layouts using css box model"""
    
    def __init__(self):
        self.name = "layout_agent"
        self.config = load_config()
        self.poster_margin = self.config["layout"]["poster_margin"]
        self.column_spacing = self.config["layout"]["column_spacing"]
        self.title_height_fraction = self.config["layout"]["title_height_fraction"]
        self.title_font_family = self.config["typography"]["fonts"]["title"]
        self.authors_font_family = self.config["typography"]["fonts"]["authors"]
        self.section_title_font_family = self.config["typography"]["fonts"]["section_title"]
        self.body_text_font_family = self.config["typography"]["fonts"]["body_text"]
        # layout constants
        self.layout_constants = self.config["layout_constants"]
        self.column_balancing = self.config["column_balancing"]
        
        # debug configuration
        self.show_debug_borders = self.config["rendering"]["debug_borders"]  ## enable to see section boundaries for debugging
        

    def __call__(self, state: PosterState, mode: str = "initial") -> PosterState:
        if mode == "initial":
            return self._generate_initial_layout(state)
        else:
            return self._generate_final_layout(state)
    
    def _generate_initial_layout(self, state: PosterState) -> PosterState:
        """generate initial layout without optimization - direct curator mapping"""
        log_agent_info(self.name, "generating initial layout from story board")
        
        try:
            story_board = state.get("story_board")
            if not story_board:
                raise ValueError("missing story_board from curator")
            
            # organize sections from story board for layout creation
            sections = story_board["spatial_content_plan"]["sections"]
            optimized_layout = self._organize_sections_by_column(sections)
            
            # create layout directly from curator output - no optimization
            layout_data = self._create_precise_layout(
                story_board=story_board,
                optimized_layout=optimized_layout,
                state=state
            )
            
            # generate column analysis for balancer
            column_analysis = self._generate_column_analysis(layout_data, state)
            
            state["initial_layout_data"] = layout_data
            state["column_analysis"] = column_analysis
            state["current_agent"] = self.name
            
            self._save_initial_layout(state)
            
            log_agent_success(self.name, "initial layout generated")
            return state
            
        except Exception as e:
            log_agent_error(self.name, f"initial layout error: {e}")
            state["errors"].append(f"{self.name}: {e}")
            return state
    
    def _generate_final_layout(self, state: PosterState) -> PosterState:
        """generate final layout from optimized story board"""
        log_agent_info(self.name, "generating final layout from optimized story board")
        
        try:
            optimized_story_board = state.get("optimized_story_board")
            if not optimized_story_board:
                raise ValueError("missing optimized_story_board from balancer")
            
            # organize sections from optimized story board
            sections = optimized_story_board["spatial_content_plan"]["sections"]
            organized_layout = self._organize_sections_by_column(sections)
            
            # create final layout from optimized story board
            layout_data = self._create_precise_layout(
                story_board=optimized_story_board,
                optimized_layout=organized_layout,
                state=state
            )
            
            # generate final column analysis to verify optimization success
            final_column_analysis = self._generate_column_analysis(layout_data, state)
            
            # validate final layout
            validation = self._validate_precise_layout(layout_data, state["poster_width"], state["poster_height"])
            
            state["design_layout"] = layout_data
            state["final_column_analysis"] = final_column_analysis
            state["optimized_column_assignment"] = organized_layout["optimized_layout"]["column_assignments"]
            state["current_agent"] = self.name
            
            self._save_final_layout(state)
            
            log_agent_success(self.name, "final layout complete")
            return state
            
        except Exception as e:
            log_agent_error(self.name, f"final layout error: {e}")
            state["errors"].append(f"{self.name}: {e}")
            return state
    
    def _optimize_column_distribution(self, story_board: Dict, poster_width: int, poster_height: int, config, state) -> Dict:
        """rule-based column distribution for optimal space utilization"""
        log_agent_info(self.name, "optimizing column distribution")
        
        # calculate available space
        effective_height = poster_height - 2 * self.poster_margin  # total height minus margins
        title_region_height = effective_height * self.title_height_fraction  # 18% of effective height
        available_height = effective_height - title_region_height  # remaining height for sections
        column_width = (poster_width - 2 * self.poster_margin - 2 * self.column_spacing) / 3
        
        # handle new spatial content plan format
        if "spatial_content_plan" in story_board:
            sections = story_board["spatial_content_plan"]["sections"]
            column_distribution = story_board.get("column_distribution", {})
        else:
            # fallback to old format
            sections = story_board.get("story_board", {}).get("sections", [])
            column_distribution = {}
        
        # create precise spatial layout using css-like calculations
        optimized_layout = self._create_spatial_layout(
            sections, column_distribution, available_height, column_width, state
        )
        
        log_agent_success(self.name, f"created rule-based optimized layout")
        
        return {
            "optimized_layout": {
                "column_assignments": optimized_layout,
                "strategy": "rule_based_intelligent",
                "space_utilization_target": 0.90,
                "column_dimensions": {
                    "width": column_width,
                    "height": available_height
                }
            }
        }
    
    def _apply_adjustments(self, adjustments: Dict):
        """apply critic-requested adjustments to layout parameters"""
        if adjustments.get("increase_spacing"):
            log_agent_info(self.name, "increased spacing: adjusting layout constants")
        
        if adjustments.get("reduce_sizes"):
            log_agent_info(self.name, "reduced spacing: adjusting layout constants")
        
        if adjustments.get("poster_margin"):
            self.poster_margin = adjustments["poster_margin"]
        
        if adjustments.get("column_spacing"):
            self.column_spacing = adjustments["column_spacing"]

    def _save_initial_layout(self, state: PosterState):
        """save initial layout data"""
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "initial_layout_data.json", "w", encoding='utf-8') as f:
            json.dump(state.get("initial_layout_data", []), f, indent=2)
        
        with open(output_dir / "column_analysis.json", "w", encoding='utf-8') as f:
            json.dump(state.get("column_analysis", {}), f, indent=2)
    
    def _save_final_layout(self, state: PosterState):
        """save final layout data"""
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "final_design_layout.json", "w", encoding='utf-8') as f:
            json.dump(state.get("design_layout", []), f, indent=2)
        
        with open(output_dir / "optimized_layout.json", "w", encoding='utf-8') as f:
            json.dump(state.get("optimized_column_assignment", {}), f, indent=2)
        
        # save final column analysis to show optimization success
        if state.get("final_column_analysis"):
            with open(output_dir / "final_column_analysis.json", "w", encoding='utf-8') as f:
                json.dump(state.get("final_column_analysis", {}), f, indent=2)
    
    def _generate_column_analysis(self, layout_data: List[Dict], state: PosterState) -> Dict:
        """generate detailed column utilization analysis using exact column calculation method"""
        poster_width = state["poster_width"]
        poster_height = state["poster_height"]
        effective_height = poster_height - 2 * self.poster_margin
        title_region_height = effective_height * self.title_height_fraction
        available_height = effective_height - title_region_height
        
        # calculate precise column x coordinates using global constants
        column_width = (poster_width - 2 * self.poster_margin - 2 * self.column_spacing) / 3
        left_column_x = self.poster_margin
        middle_column_x = self.poster_margin + column_width + self.column_spacing
        right_column_x = self.poster_margin + 2 * (column_width + self.column_spacing)
        
        columns = {"left": [], "middle": [], "right": []}
        
        # group elements by column using calculated column boundaries
        for element in layout_data:
            if element.get("type") == "section_container":
                element_x = element.get("x", 0)
                # use midpoint boundaries to categorize elements
                if element_x < (left_column_x + middle_column_x) / 2:
                    columns["left"].append(element)
                elif element_x < (middle_column_x + right_column_x) / 2:
                    columns["middle"].append(element)
                else:
                    columns["right"].append(element)
        
        # calculate utilization for each column
        column_analysis = {
            "available_height": available_height,
            "columns": {}
        }
        
        for col_name, elements in columns.items():
            if elements:
                max_bottom = max(elem["y"] + elem["height"] for elem in elements)
                min_top = min(elem["y"] for elem in elements) 
                used_height = max_bottom - min_top
            else:
                used_height = 0
            
            utilization_rate = used_height / available_height if available_height > 0 else 0
            
            status = "overflow" if utilization_rate > 1.0 else "underutilized" if utilization_rate < 0.7 else "balanced"
            
            column_analysis["columns"][col_name] = {
                "utilization_rate": utilization_rate,
                "total_height": used_height,
                "status": status,
                "available_space": max(0, available_height - used_height),
                "excess_height": max(0, used_height - available_height)
            }
        
        return column_analysis
    
    def _organize_sections_by_column(self, sections: List[Dict]) -> Dict:
        """organize sections by column assignment for layout creation"""
        columns = {"left": [], "middle": [], "right": []}
        
        for section in sections:
            column = section.get("column_assignment", "left")
            if column in columns:
                columns[column].append(section)
        
        column_assignments = [
            {"column_id": 0, "sections": columns["left"]},
            {"column_id": 1, "sections": columns["middle"]}, 
            {"column_id": 2, "sections": columns["right"]}
        ]
        
        return {
            "optimized_layout": {
                "column_assignments": column_assignments
            }
        }
    
    def _create_precise_layout(self, story_board: Dict, optimized_layout: Dict, state: PosterState) -> List[Dict]:
        """create precise layout with exact positioning using measurements"""
        layout_elements = []
        
        # poster dimensions
        poster_width = state["poster_width"]
        poster_height = state["poster_height"]
        
        # calculate layout dimensions
        effective_height = poster_height - 2 * self.poster_margin
        title_region_height = effective_height * self.title_height_fraction  # 18% fixed region
        available_height = effective_height - title_region_height  # remaining for sections
        column_width = (poster_width - 2 * self.poster_margin - 2 * self.column_spacing) / 3
        
        # add title element (still uses actual measured height, not fixed region height)
        title_element = self._create_title_element(state, poster_width, title_region_height)
        if title_element:
            layout_elements.append(title_element)
        
        # add logo elements
        logo_elements = self._create_logo_elements(state, poster_width)
        layout_elements.extend(logo_elements)
        
        # process each column
        column_assignments = optimized_layout.get("optimized_layout", {}).get("column_assignments", [])
        
        for col_idx, column in enumerate(column_assignments):
            column_x = self.poster_margin + col_idx * (column_width + self.column_spacing)
            column_y = self.poster_margin + title_region_height  # fixed at poster_margin + 18%
            
            current_y = column_y
            
            # process each section in this column
            for section in column.get("sections", []):
                section_start_y = current_y
                section_elements = self._create_section_elements(
                    section, column_x, current_y, column_width, state, available_height
                )
                
                # calculate section height from actual element positions
                section_height = 0
                if section_elements:
                    # find the bottommost element
                    max_bottom = 0
                    for element in section_elements:
                        element_bottom = element["y"] + element["height"]
                        max_bottom = max(max_bottom, element_bottom)
                    section_height = max_bottom - section_start_y
                
                # create section container for layout structure
                section_container = {
                    "type": "section_container",
                    "x": column_x,
                    "y": section_start_y,
                    "width": column_width,
                    "height": section_height,
                    "section_id": section.get("section_id", "unknown"),
                    "importance_level": section.get("importance_level", 2),  # importance level for background styling
                    "priority": 0.1
                }
                
                # add debug border only if enabled
                if self.show_debug_borders:
                    section_container["debug_border"] = True
                
                layout_elements.append(section_container)
                
                layout_elements.extend(section_elements)
                current_y += section_height + 1.0  # 1" section spacing for stability
                
                log_agent_info(self.name, f"placed section '{section.get('section_id')}' at y={section_start_y:.2f}, height={section_height:.2f}")
        
        return layout_elements
    
    def _create_title_element(self, state: PosterState, poster_width: float, title_height: float) -> Dict:
        """create title element with exact positioning"""
        # calculate 2/3 width (2 columns + 1 margin width)
        column_width = (poster_width - 2 * self.poster_margin - 2 * self.column_spacing) / 3
        title_width = 2 * column_width + self.column_spacing  # 2 columns + 1 spacing
        
        # extract title and authors from narrative content
        narrative = state.get("narrative_content", {})
        meta = narrative.get("meta", {})
        poster_title = meta.get("poster_title", state.get('poster_name', 'Title'))
        authors = meta.get("authors", state.get('authors', 'Authors'))
        
        return {
            "type": "title",
            "x": self.poster_margin,
            "y": self.poster_margin,
            "width": title_width,
            "height": title_height - 1.0,
            "content": f"{poster_title}\n{authors}",
            "font_family": self.title_font_family,
            "font_size": 100,
            "author_font_size": 72,
            "priority": 1.0
        }
    
    def _create_logo_elements(self, state: PosterState, poster_width: float) -> List[Dict]:
        """create logo elements with exact positioning"""
        elements = []

        # get aspect ratio of logos
        from PIL import Image
        conf_logo_aspect_ratio = self.layout_constants["default_logo_aspect_ratio"]
        aff_logo_aspect_ratio = self.layout_constants["default_logo_aspect_ratio"]
        if state.get("logo_path") and Path(state["logo_path"]).exists():
            with Image.open(state["logo_path"]) as img:
                conf_logo_aspect_ratio = img.size[0] / img.size[1]
        if state.get("aff_logo_path") and Path(state["aff_logo_path"]).exists():
            with Image.open(state["aff_logo_path"]) as img:
                aff_logo_aspect_ratio = img.size[0] / img.size[1]

        # calculate logo heights based on fit in 1/3 of poster width
        column_width = (poster_width - 2 * self.poster_margin - 2 * self.column_spacing) / 3
        logo_height = (column_width - 1) / (conf_logo_aspect_ratio + aff_logo_aspect_ratio)
        # widths based on aspect ratios
        conf_logo_width = logo_height * conf_logo_aspect_ratio
        aff_logo_width = logo_height * aff_logo_aspect_ratio

        conf_logo_x = poster_width - self.poster_margin - conf_logo_width
        aff_logo_x = conf_logo_x - 1.0 - aff_logo_width
        
        if state.get("aff_logo_path"):
            elements.append({
                "type": "aff_logo", 
                "x": aff_logo_x,
                "y": self.poster_margin,
                "width": aff_logo_width,
                "height": logo_height,
                "priority": 0.9
            })
        
        if state.get("logo_path"):
            elements.append({
                "type": "conf_logo",
                "x": conf_logo_x,
                "y": self.poster_margin,
                "width": conf_logo_width,
                "height": logo_height,
                "priority": 0.9
            })
        
        return elements
    
    def _create_section_elements(self, section: Dict, column_x: float, start_y: float, 
                               column_width: float, state: PosterState, available_height: float = None) -> List[Dict]:
        """create all elements for a section with precise positioning"""
        elements = []
        current_y = start_y
        
        # enhanced section title with design styling
        section_title = section.get("section_title", "")
        if section_title:
            title_elements = self._create_section_title_design(
                section, column_x, current_y, column_width, state
            )
            elements.extend(title_elements)
            
            # calculate total height used by title and accent elements
            title_total_height = max(elem["y"] + elem["height"] - current_y for elem in title_elements)
            current_y += title_total_height + self.config["layout"]["title_to_content_spacing"]
        
        # visual assets first (after title, before text)
        visual_assets = section.get("visual_assets", [])
        for visual_asset in visual_assets:
            visual_id = visual_asset.get("visual_id", "")
            # apply same padding as text elements
            visual_padding = self.config["layout"]["text_padding"]["left_right"]  # left/right padding
            visual_width = column_width - (2 * visual_padding)
            final_visual_width, final_visual_height, scale_factor = self._calculate_visual_height(visual_id, visual_width, state, available_height)
            
            # center the visual within the section (important for scaled visuals)
            section_content_width = column_width - (2 * visual_padding)
            if final_visual_width < section_content_width:
                # center horizontally within the section
                visual_x = column_x + visual_padding + (section_content_width - final_visual_width) / 2
            else:
                # use left alignment if visual fills the section
                visual_x = column_x + visual_padding
            
            elements.append({
                "type": "visual",
                "x": visual_x,
                "y": current_y,
                "width": final_visual_width,
                "height": final_visual_height,
                "visual_id": visual_id,
                "scale_factor": scale_factor,  # for renderer to apply proper scaling
                "priority": 0.6,
                "id": f"{section.get('section_id')}_{visual_id}",
                "font_family": self.body_text_font_family,
                "font_color": "#000000",
                "font_size": 44,
                "line_spacing": 1.0
            })
            # use the already-scaled height for positioning (no double scaling)
            current_y += final_visual_height + self.config["layout"]["visual_spacing"]["below_visual"]
        
        # text content (after visuals)
        text_content = section.get("text_content", [])
        if text_content:
            combined_text = "\n".join(text_content)
            text_padding = self.config["layout"]["text_padding"]["left_right"]  # consistent with layout positioning
            text_measurement = measure_text_height(
                text_content=combined_text,
                width_inches=column_width - (2 * text_padding),
                font_name=self.body_text_font_family,
                font_size=44,
                line_spacing=1.0
            )
            text_height = text_measurement["optimal_height"] + 0.1
            
            # apply text padding to match measurement calculation
            elements.append({
                "type": "text",
                "x": column_x + text_padding,
                "y": current_y,
                "width": column_width - (2 * text_padding),
                "height": text_height,
                "content": combined_text,
                "font_family": self.body_text_font_family,
                "font_size": 44,
                "font_color": "#000000",
                "priority": 0.5,
                "id": f"{section.get('section_id')}_text",
                "line_spacing": 1.0
            })
            current_y += text_height + 0.3
            
        return elements
    
    def _create_section_title_design(self, section: Dict, column_x: float, start_y: float, column_width: float, state: PosterState) -> List[Dict]:
        """create section title with colorblock styling"""
        elements = []
        section_title = section.get("section_title", "")
        section_id = section.get("section_id", "")
        
        # get section title design from state
        title_design = state.get("section_title_design", {}).get("section_title_design", {})
        
        # find specific section application
        section_app = None
        for app in title_design.get("section_applications", []):
            if app.get("section_id") == section_id:
                section_app = app
                break
        
        # extract styling information
        title_styling = section_app.get("title_styling", {})
        accent_styling = section_app.get("accent_styling", {})
        
        # create title textbox positioning
        title_padding = self.layout_constants["title_padding"]
        title_width = column_width - (2 * title_padding)
        base_title_x = column_x + title_padding
        
        # use font size from styling_interfaces
        styling_interfaces = state.get("styling_interfaces", {})
        section_title_font_size = styling_interfaces.get("font_sizes", {}).get("section_title", 64)
        
        # calculate rectangle dimensions (same as before)
        rect_height = section_title_font_size / 72  # convert pt to inches precisely
        rect_width = rect_height * 0.618  # golden ratio width
        
        # apply user-requested coordinate modifications
        rect_y_offset = 10 / 72  # 10pt converted to inches
        title_x_offset = rect_height  # offset by rectangle height
        
        # create rectangle background element with y offset
        rectangle_element = {
            "type": "title_accent_block",
            "x": base_title_x,
            "y": start_y + rect_y_offset,  # user modification: y + 10pt
            "width": rect_width,
            "height": rect_height,
            "color": accent_styling.get("color", "#335f91"),
            "priority": 0.7
        }
        elements.append(rectangle_element)
        
        # adjust title content (add 4 spaces prefix for rectangle_left template)
        display_title = "    " + section_title
        
        # create title element with x offset using precise font-based height
        precise_title_height = section_title_font_size / 72  # pt to inches
        
        title_element = {
            "type": "section_title",
            "x": base_title_x + title_x_offset,  # x + rectangle height
            "y": start_y,
            "width": title_width,
            "height": precise_title_height,
            "section_title": display_title,
            "font_family": title_styling.get("font_family", self.section_title_font_family),
            "font_size": section_title_font_size,
            "font_weight": title_styling.get("font_weight", "bold"),
            "font_color": title_styling.get("color", "#000000"),
            "alignment": title_styling.get("alignment", "left"),
            "priority": 0.8
        }
        elements.append(title_element)
        
        return elements
    
    def _validate_precise_layout(self, layout_data: List[Dict], poster_width: float, 
                               poster_height: float) -> Dict[str, Any]:
        """validate layout for overlaps and overflow"""
        issues = []
        valid = True
        
        # check for overflow
        for element in layout_data:
            right_edge = element["x"] + element["width"]
            bottom_edge = element["y"] + element["height"]
            
            if right_edge > poster_width:
                issues.append(f"Element {element.get('id', 'unknown')} overflows right edge")
                valid = False
            
            if bottom_edge > poster_height:
                issues.append(f"Element {element.get('id', 'unknown')} overflows bottom edge")
                valid = False
        
        # check for overlaps (simplified check)
        for i, elem1 in enumerate(layout_data):
            for j, elem2 in enumerate(layout_data[i+1:], i+1):
                if self._elements_overlap(elem1, elem2):
                    issues.append(f"Elements {elem1.get('id', 'unknown')} and {elem2.get('id', 'unknown')} overlap")
                    valid = False
        
        # calculate space utilization
        total_used_area = sum(elem["width"] * elem["height"] for elem in layout_data)
        total_poster_area = poster_width * poster_height
        space_utilization = total_used_area / total_poster_area if total_poster_area > 0 else 0
        
        return {
            "valid": valid,
            "issues": issues,
            "space_utilization": space_utilization,
            "total_elements": len(layout_data)
        }
    
    def _elements_overlap(self, elem1: Dict, elem2: Dict) -> bool:
        """check if two elements overlap"""
        return not (
            elem1["x"] + elem1["width"] <= elem2["x"] or
            elem2["x"] + elem2["width"] <= elem1["x"] or
            elem1["y"] + elem1["height"] <= elem2["y"] or
            elem2["y"] + elem2["height"] <= elem1["y"]
        )
    
    def _create_spatial_layout(self, sections: List[Dict], column_distribution: Dict, 
                             available_height: float, column_width: float, state: PosterState) -> List[Dict]:
        """create precise spatial layout using css-like calculations"""
        
        log_agent_info(self.name, "creating spatial layout with css-like precision")
        
        # organize sections by spatial assignment
        columns = {
            "left": {"sections": [], "total_height": 0.0},
            "middle": {"sections": [], "total_height": 0.0}, 
            "right": {"sections": [], "total_height": 0.0}
        }
        
        for section in sections:
            column = section.get("column_assignment", "left")
            if column in columns:
                columns[column]["sections"].append(section)
        
        log_agent_info(self.name, f"organized sections: left={len(columns['left']['sections'])}, middle={len(columns['middle']['sections'])}, right={len(columns['right']['sections'])}")
        
        # calculate precise heights for each section
        for column_name, column_data in columns.items():
            for section in column_data["sections"]:
                section_height = self._calculate_precise_section_height(section, column_width, state, available_height)
                section["calculated_height"] = section_height
                column_data["total_height"] += section_height
        
        
        # return layout in expected format
        return [{
            "column_id": 0,
            "sections": [s for s in sections if s.get("column_assignment") == "left"],
            "estimated_height": columns["left"]["total_height"]
        }, {
            "column_id": 1, 
            "sections": [s for s in sections if s.get("column_assignment") == "middle"],
            "estimated_height": columns["middle"]["total_height"]
        }, {
            "column_id": 2,
            "sections": [s for s in sections if s.get("column_assignment") == "right"], 
            "estimated_height": columns["right"]["total_height"]
        }]
    
    def _calculate_precise_section_height(self, section: Dict, column_width: float, state: PosterState, available_height: float = None) -> float:
        """calculate precise section height using css box model"""
        
        total_height = 0.0
        
        # section title height (if exists)
        title = section.get("section_title", "")
        if title:
            title_padding = self.layout_constants["title_padding"]  # consistent with layout positioning
            title_measurement = measure_text_height(
                text_content=title,
                width_inches=column_width - (2 * title_padding),  # account for padding
                font_name="Helvetica Neue",
                font_size=64,
                line_spacing=1.0
            )
            title_height = title_measurement["optimal_height"] + 0.3  # title margin
            total_height += title_height
        
        # text content height with fixed line spacing
        text_content = section.get("text_content", [])
        if text_content:
            # join all bullet points with proper paragraph separation
            full_text = "\n\n".join(text_content)  # double newline between paragraphs
            
            text_padding = self.config["layout"]["text_padding"]["left_right"]  # consistent with layout positioning
            text_measurement = measure_text_height(
                text_content=full_text,
                width_inches=column_width - (2 * text_padding),  # account for padding
                font_name=self.body_text_font_family, 
                font_size=44,
                line_spacing=1.0
            )
            text_height = text_measurement["optimal_height"] + 0.2  # text margin
            total_height += text_height
        
        # visual assets height (fixed aspect ratio)
        visual_assets = section.get("visual_assets", [])
        for visual in visual_assets:
            visual_id = visual.get("visual_id", "")
            if visual_id:
                visual_padding = self.layout_constants["visual_padding"]  # consistent with layout positioning
                visual_width = column_width - (2 * visual_padding)
                final_visual_width, final_visual_height, scale_factor = self._calculate_visual_height(visual_id, visual_width, state, available_height)
                # use the already-scaled height for section sizing (no double scaling)
                total_height += final_visual_height + 0.3  # visual margin
        
        # section padding and margins
        section_padding = self.layout_constants["section_padding"]
        total_height += section_padding
        
        return total_height
    
    def _calculate_visual_height(self, visual_id: str, visual_width: float, state, available_height: float = None) -> tuple:
        """calculate proper visual width and height based on aspect ratio with auto-shrinking for large visuals
        
        returns: (final_width, final_height, scale_factor)
        """
        # visual width already accounts for padding (passed from caller)
        
        # get aspect ratio from state data
        images = state.get("images", {})
        tables = state.get("tables", {})
        
        # better default aspect ratios based on visual type
        if visual_id.startswith("table_"):
            aspect_ratio = 1.5  # tables are often wider than tall
        elif visual_id.startswith("figure_"):
            aspect_ratio = 1.2  # figures vary but often slightly wider
        else:
            aspect_ratio = self.layout_constants["default_logo_aspect_ratio"]  # default square
        
        # handle both formats: "figure_1"/"table_1" and "1"/"2" etc.
        lookup_id = visual_id
        
        # if visual_id has prefix, extract the number
        if visual_id.startswith("figure_"):
            lookup_id = visual_id.replace("figure_", "")
        elif visual_id.startswith("table_"):
            lookup_id = visual_id.replace("table_", "")
        
        # check in appropriate collection first based on visual_id prefix
        if visual_id.startswith("table_") and lookup_id in tables:
            found_aspect = tables[lookup_id].get("aspect", aspect_ratio)
            aspect_ratio = found_aspect
            log_agent_info(self.name, f"found visual {visual_id} -> {lookup_id} in tables, aspect={aspect_ratio:.2f}")
        elif visual_id.startswith("figure_") and lookup_id in images:
            found_aspect = images[lookup_id].get("aspect", aspect_ratio)
            aspect_ratio = found_aspect
            log_agent_info(self.name, f"found visual {visual_id} -> {lookup_id} in images, aspect={aspect_ratio:.2f}")
        elif lookup_id in images:
            found_aspect = images[lookup_id].get("aspect", aspect_ratio)
            aspect_ratio = found_aspect
            log_agent_info(self.name, f"found visual {visual_id} -> {lookup_id} in images, aspect={aspect_ratio:.2f}")
        elif lookup_id in tables:
            found_aspect = tables[lookup_id].get("aspect", aspect_ratio)
            aspect_ratio = found_aspect
            log_agent_info(self.name, f"found visual {visual_id} -> {lookup_id} in tables, aspect={aspect_ratio:.2f}")
        else:
            log_agent_warning(self.name, f"visual {visual_id} (lookup: {lookup_id}) not found in state data, using fallback aspect={aspect_ratio:.2f}")
            # debug: log available visual data
            log_agent_info(self.name, f"available images: {list(images.keys())}")
            log_agent_info(self.name, f"available tables: {list(tables.keys())}")
        
        # calculate original height from aspect ratio
        original_height = visual_width / aspect_ratio
        
        # check if shrinking is needed (height > 40% of column height)
        scale_factor = 1.0
        if available_height and original_height > (available_height * 0.4):
            scale_factor = 0.8  # shrink to 80% of original size
            log_agent_info(self.name, f"visual {visual_id} too large ({original_height:.2f}\" > 40% of {available_height:.2f}\"), shrinking to 80%")
        
        # apply scaling to both width and height to maintain aspect ratio
        final_width = visual_width * scale_factor
        final_height = original_height * scale_factor
        
        log_agent_info(self.name, f"visual {visual_id}: orig_w={visual_width:.2f}\", orig_h={original_height:.2f}\", scale={scale_factor:.1f}, final_w={final_width:.2f}\", final_h={final_height:.2f}\"")
        
        # return final width, height and scale factor for rendering
        return final_width, final_height, scale_factor
    


def layout_agent_node(state: PosterState) -> Dict[str, Any]:
    result = LayoutAgent()(state)
    return {
        **state,
        "design_layout": result["design_layout"],
        "column_assignment": result.get("column_assignment"),
        "tokens": result["tokens"],
        "current_agent": result["current_agent"],
        "errors": result["errors"]
    }