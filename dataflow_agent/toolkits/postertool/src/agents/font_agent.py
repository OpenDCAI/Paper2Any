"""
font styling and keyword highlighting
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List

from src.state.poster_state import PosterState
from utils.langgraph_utils import LangGraphAgent, extract_json, load_prompt
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error, log_agent_warning
from src.config.poster_config import load_config
from jinja2 import Template


class FontAgent:
    """handles text styling and keyword highlighting"""
    
    def __init__(self):
        self.name = "font_agent"
        self.keyword_extraction_prompt = load_prompt("config/prompts/extract_keywords.txt")

    def __call__(self, state: PosterState) -> PosterState:
        log_agent_info(self.name, "starting font styling")
        
        try:
            design_layout = state.get("design_layout", [])
            color_scheme = state.get("color_scheme", {})
            story_board = state.get("story_board", {})
            
            if not design_layout:
                raise ValueError("missing design_layout from layout agent")
            if not color_scheme:
                raise ValueError("missing color_scheme from color agent")
            if not story_board:
                raise ValueError("missing story_board from story board curator")
            
            # identify keywords to highlight
            keywords = self._identify_keywords(story_board, state)
            
            # apply styling to layout
            styled_layout = self._apply_styling(design_layout, color_scheme, keywords, state)
            
            state["styled_layout"] = styled_layout
            state["keywords"] = keywords
            state["current_agent"] = self.name
            
            self._save_styled_layout(state)
            
            # count total keywords across all sections
            total_keywords = sum(len(kw_list) for kw_list in keywords.get("section_keywords", {}).values())
            
            log_agent_success(self.name, f"applied enhanced styling to {len(styled_layout)} elements")
            log_agent_success(self.name, f"identified {total_keywords} keywords for highlighting")

        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")
            
        return state

    def _identify_keywords(self, story_board: Dict, state: PosterState) -> Dict[str, Any]:
        """identify keywords using story board content and enhanced narrative"""
        
        enhanced_narrative = state.get("enhanced_narrative", {})
        
        # extract keywords using LLM with external prompt
        log_agent_info(self.name, "identifying keywords for highlighting")
        
        agent = LangGraphAgent("expert at identifying key terms for visual highlighting", state["text_model"])
        
        template_data = {
            "enhanced_narrative": json.dumps(enhanced_narrative, indent=2),
            "curated_content": json.dumps(story_board, indent=2)
        }
        
        prompt = Template(self.keyword_extraction_prompt).render(**template_data)
        response = agent.step(prompt)
        result = extract_json(response.content)
        
        # add token usage
        state["tokens"].add_text(response.input_tokens, response.output_tokens)
        
        return result

    def _apply_styling(self, layout: List[Dict], colors: Dict, keywords: Dict, state: PosterState) -> List[Dict]:
        """apply styling with proper bullet point and bold formatting"""
        styled_layout = []
        section_keywords = keywords.get("section_keywords", {})
        
        # process all elements with enhanced styling
        for element in layout:
            styled_element = element.copy()
            
            # apply element-specific styling
            if element.get("type") == "title":
                self._apply_title_styling(styled_element, colors)
            
            elif element.get("type") in ["section_title", "title_accent_block", "title_accent_line"]:
                # these are handled by the section title designer
                pass
            
            elif element.get("type") == "section_container":
                self._apply_section_container_styling(styled_element, colors)
                
            elif element.get("type") in ["text", "visual", "mixed"]:
                self._apply_content_styling(styled_element, colors, section_keywords)
            
            elif element.get("type") in ["conf_logo", "aff_logo"]:
                # logos don't need text styling
                pass
            
            styled_layout.append(styled_element)
        
        # sort by priority for proper rendering order
        styled_layout.sort(key=lambda x: x.get("priority", 0.5))
        
        return styled_layout

    def _apply_title_styling(self, element: Dict, colors: Dict):
        """apply styling to title elements"""
        element["font_family"] = "Helvetica Neue"
        element["font_color"] = colors.get("text_on_theme", "#FFFFFF")
        element["font_size"] = 100
        element["author_font_size"] = 72
        element["font_weight"] = "bold"

    def _apply_section_container_styling(self, element: Dict, colors: Dict):
        """apply styling to section container elements"""
        element["border_color"] = colors.get("mono_light", "#CCCCCC")
        element["border_width"] = 1
        element["fill_color"] = "#FFFFFF"  # white background

    def _apply_content_styling(self, element: Dict, colors: Dict, section_keywords: Dict):
        """apply styling to content elements with keyword highlighting"""
        # determine parent section for keyword lookup
        parent_section = self._extract_parent_section(element)
        keywords_for_section = section_keywords.get(parent_section, {})
        
        # ensure proper bullet point formatting first (before keyword highlighting to preserve formatting)
        if element.get("content"):
            element["content"] = self._format_bullet_points(element["content"])
        
        # apply keyword highlighting to content (after bullet formatting)
        if keywords_for_section and element.get("content"):
            content = element["content"]
            original_content = content
            content = self._apply_keyword_highlighting(content, keywords_for_section, colors)
            element["content"] = content
            
            # debug logging
            if content != original_content:
                total_keywords = sum(len(kw_list) for kw_list in keywords_for_section.values() if isinstance(kw_list, list))
                log_agent_info(self.name, f"Applied highlighting to {parent_section}: found {total_keywords} keywords")
            elif keywords_for_section:
                total_keywords = sum(len(kw_list) for kw_list in keywords_for_section.values() if isinstance(kw_list, list))
                log_agent_warning(self.name, f"Keywords found for {parent_section} ({total_keywords} total) but no highlighting applied")
        
        # apply base text styling
        element["font_family"] = "Arial"
        element["font_color"] = colors.get("text", "#000000")
        element["font_size"] = 44

    def _extract_parent_section(self, element: Dict) -> str:
        """extract parent section id from element"""
        element_id = element.get("id", "")
        
        # extract section id from element id
        if "_" in element_id and element_id.endswith("_text"):
            # remove the "_text" suffix to get the section ID
            return element_id[:-5]  # remove last 5 characters ("_text")
        elif "_" in element_id:
            # fallback: remove last part after underscore
            parts = element_id.split("_")
            if len(parts) > 1:
                return "_".join(parts[:-1])
        
        return ""

    def _apply_keyword_highlighting(self, content: str, keywords: Dict, colors: Dict) -> str:
        """apply semantic-based keyword highlighting with three distinct styles"""
        # use contrast color for highlighting
        highlight_color = colors.get("contrast", colors.get("theme", "#1E3A8A"))
        
        # define highlighting styles based on semantic categories
        style_functions = {
            "bold_contrast": lambda text: f"<color:{highlight_color}>{text}</color>",  # contrast color (bold applied automatically in renderer)
            "bold": lambda text: f"**{text}**",  # just bold
            "italic": lambda text: f"*{text}*"  # italic
        }
        
        # apply each style category
        for style_type, style_func in style_functions.items():
            keyword_list = keywords.get(style_type, [])
            for keyword in keyword_list:
                if not keyword.strip():
                    continue
                content = self._highlight_keyword_in_content(content, keyword, style_func)
        
        return content

    def _highlight_keyword_in_content(self, content: str, keyword: str, style_func) -> str:
        """highlight a specific keyword in content"""
        if f"<color:" in content and keyword.lower() in content.lower():
            return content
        
        escaped_keyword = re.escape(keyword.strip())
        
        # first try to match keyword with existing bold formatting
        bold_pattern = rf'\*\*([^*]*?{escaped_keyword}[^*]*?)\*\*'
        bold_match = re.search(bold_pattern, content, re.IGNORECASE)
        
        if bold_match:
            # extract the full bold text, replace only the keyword part
            full_bold_text = bold_match.group(1)
            keyword_in_bold = re.search(escaped_keyword, full_bold_text, re.IGNORECASE)
            if keyword_in_bold:
                # replace just the keyword within the bold text
                original_keyword = keyword_in_bold.group(0)
                new_keyword_formatted = style_func(original_keyword)
                
                # check if style_func returns color format
                if '<color:' in new_keyword_formatted:
                    # remove the outer ** since color already implies bold
                    new_bold_text = full_bold_text.replace(original_keyword, new_keyword_formatted, 1)
                    old_full_bold = bold_match.group(0)
                    return content.replace(old_full_bold, new_bold_text, 1)
                else:
                    # for regular bold/italic, keep the ** wrapper
                    new_keyword = new_keyword_formatted.replace('**', '').replace('**', '')  # remove any extra bold markers
                    new_bold_text = full_bold_text.replace(original_keyword, new_keyword, 1)
                    old_full_bold = bold_match.group(0)
                    new_full_bold = f'**{new_bold_text}**'
                    return content.replace(old_full_bold, new_full_bold, 1)
        
        # then match keyword with existing italic formatting  
        italic_pattern = rf'\*({escaped_keyword})\*'
        italic_match = re.search(italic_pattern, content, re.IGNORECASE)
        
        if italic_match:
            old_formatted = italic_match.group(0)
            new_formatted = style_func(keyword)
            return content.replace(old_formatted, new_formatted, 1)
        
        plain_pattern = rf'\b{escaped_keyword}\b'
        plain_match = re.search(plain_pattern, content, re.IGNORECASE)
        
        if plain_match:
            matched_text = plain_match.group(0)
            new_formatted = style_func(matched_text)
            return content.replace(matched_text, new_formatted, 1)
        
        return content

    def _format_bullet_points(self, content: str) -> str:
        """ensure proper bullet point formatting"""
        if not content:
            return content
        
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # ensure start with '•' or preserve existing '•'
            if line.startswith('• '):
                formatted_lines.append(line)
            elif line.startswith('- '):
                # dash -> bullet
                formatted_lines.append('• ' + line[2:])
            elif line.startswith('* '):
                # asterisk -> bullet
                formatted_lines.append('• ' + line[2:])
            elif not line.startswith('•'):
                # add bullet if missing (for content that should be bulleted)
                if any(line.lower().startswith(word) for word in ['the ', 'this ', 'our ', 'we ', 'new ', 'key ', 'main ']):
                    formatted_lines.append('• ' + line)
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    def get_styling_interfaces(self) -> Dict[str, Any]:
        """return interfaces for renderer to properly handle styled content"""
        config = load_config()
        font_params = config["typography"]
        
        return {
            "bullet_point_marker": "•",
            "bold_start_tag": "**",
            "bold_end_tag": "**",
            "italic_start_tag": "*",
            "italic_end_tag": "*",
            "color_start_tag": "<color:",
            "color_end_tag": "</color>",
            "line_spacing": font_params["line_spacing"],  # from config
            "paragraph_spacing": font_params["paragraph_spacing"],
            "font_sizes": {
                "title": font_params["sizes"]["title"],
                "authors": font_params["sizes"]["authors"],
                "section_title": font_params["sizes"]["section_title"],
                "body_text": font_params["sizes"]["body_text"]
            }
        }

    def _save_styled_layout(self, state: PosterState):
        """save styled layout and keywords"""
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # styled layout
        with open(output_dir / "styled_layout.json", "w", encoding='utf-8') as f:
            json.dump(state.get("styled_layout", []), f, indent=2)
        
        # keywords
        with open(output_dir / "keywords.json", "w", encoding='utf-8') as f:
            json.dump(state.get("keywords", {}), f, indent=2)
        
        # styling interfaces
        with open(output_dir / "styling_interfaces.json", "w", encoding='utf-8') as f:
            json.dump(self.get_styling_interfaces(), f, indent=2)


def font_agent_node(state: PosterState) -> Dict[str, Any]:
    result = FontAgent()(state)
    return {
        **state,
        "styled_layout": result["styled_layout"],
        "keywords": result.get("keywords"),
        "tokens": result["tokens"],
        "current_agent": result["current_agent"],
        "errors": result["errors"]
    }