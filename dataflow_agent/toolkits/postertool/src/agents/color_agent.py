"""
color extraction and palette generation
"""

import json
import base64
from pathlib import Path
from typing import Dict, Any, Tuple

from src.state.poster_state import PosterState
from utils.langgraph_utils import LangGraphAgent, extract_json, load_prompt
from utils.src.logging_utils import log_agent_info, log_agent_success, log_agent_error, log_agent_warning
from src.config.poster_config import load_config


class ColorAgent:
    """extracts theme colors and generates color schemes"""
    
    def __init__(self):
        self.name = "color_agent"
        self.logo_extraction_prompt = load_prompt("config/prompts/extract_theme_from_logo.txt")
        self.figure_color_prompt = load_prompt("config/prompts/extract_color_from_figure.txt")
        self.config = load_config()
        self.color_config = self.config["colors"]

    def __call__(self, state: PosterState) -> PosterState:
        log_agent_info(self.name, "starting color analysis")
        
        try:
            aff_logo_path = state.get("aff_logo_path")
            
            if aff_logo_path and Path(aff_logo_path).exists():
                log_agent_info(self.name, "extracting theme from affiliation logo")
                theme_color = self._extract_theme_from_logo(aff_logo_path, state)
            else:
                log_agent_info(self.name, "no logo found, using visual fallback")
                theme_color = self._extract_theme_from_visuals(state)
            
            color_scheme = self._generate_color_scheme(theme_color)
            color_scheme = self._add_contrast_color(color_scheme)
            
            state["color_scheme"] = color_scheme
            state["current_agent"] = self.name
            
            self._save_color_scheme(state)
            
            log_agent_success(self.name, f"theme: {theme_color}, {len(color_scheme)} colors")

        except Exception as e:
            log_agent_error(self.name, f"failed: {e}")
            state["errors"].append(f"{self.name}: {e}")
            
        return state

    def _extract_theme_from_logo(self, logo_path: str, state: PosterState) -> str:
        """extract theme color from affiliation logo using vision LLM"""
        log_agent_info(self.name, f"analyzing affiliation logo: {Path(logo_path).name}")
        
        try:
            # encode logo image
            with open(logo_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            
            agent = LangGraphAgent(
                "color extraction specialist for academic institutions",
                state["vision_model"]
            )
            
            messages = [
                {"type": "text", "text": self.logo_extraction_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
            ]
            
            response = agent.step(json.dumps(messages))
            result = extract_json(response.content)
            
            # add token usage
            state["tokens"].add_vision(response.input_tokens, response.output_tokens)
            
            extracted_color = result.get("extracted_color", load_config()["colors"]["fallback_theme"])
            suitability_score = result.get("suitability_score", 0)
            
            log_agent_info(self.name, f"logo analysis: {result.get('color_name', 'unknown')} (score: {suitability_score})")
            
            if result.get("adjustment_made") != "none":
                log_agent_info(self.name, f"color adjusted: {result.get('adjustment_made')}")
            
            return extracted_color
            
        except Exception as e:
            log_agent_warning(self.name, f"logo extraction failed: {e}, using fallback")
            return self._extract_theme_from_visuals(state)

    def _extract_theme_from_visuals(self, state: PosterState) -> str:
        """fallback: extract theme from key visuals"""
        classified = state.get("classified_visuals", {})
        key_visual = classified.get("key_visual")
        
        if not key_visual:
            log_agent_warning(self.name, "no key visual found, using default navy color")
            return load_config()["colors"]["fallback_theme"]
        
        # get path to key visual
        images = state.get("images", {})
        visual_path = None
        
        if key_visual.startswith("figure_"):
            fig_id = key_visual.replace("figure_", "")
            if fig_id in images:
                visual_path = images[fig_id].get("path")
        
        if not visual_path or not Path(visual_path).exists():
            log_agent_warning(self.name, "key visual path not found, using default navy color")
            return load_config()["colors"]["fallback_theme"]
        
        # analyze figure to extract prominent color
        try:
            theme_color = self._analyze_figure_for_color(visual_path, state)
            return theme_color
        except Exception as e:
            log_agent_warning(self.name, f"visual color extraction failed: {e}, using default navy color")
            return load_config()["colors"]["fallback_theme"]

    def _analyze_figure_for_color(self, image_path: str, state: PosterState) -> str:
        """analyze figure to extract theme color"""
        log_agent_info(self.name, "analyzing figure for color extraction")
        
        # encode image
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
        
        agent = LangGraphAgent(
            "color extraction expert for academic poster design",
            state["vision_model"]
        )
        
        prompt = self.figure_color_prompt

        messages = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
        ]
        
        response = agent.step(json.dumps(messages))
        result = extract_json(response.content)
        
        # add token usage
        state["tokens"].add_vision(response.input_tokens, response.output_tokens)
        
        return result.get("theme_color", load_config()["colors"]["fallback_theme"])

    def _generate_color_scheme(self, theme_color: str) -> Dict[str, str]:
        # hex to rgb
        hex_color = theme_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        # generate monochromatic variations
        # mono_light: medium/high saturation + brighter variant  
        mono_light = self._generate_enhanced_light_variant(r, g, b)
        
        # mono_dark: medium saturation + darker variant
        mono_dark = self._generate_enhanced_dark_variant(r, g, b)
        
        return {
            "theme": theme_color,
            "mono_light": mono_light,
            "mono_dark": mono_dark,
            "text": self.color_config["constants"]["black_text"],
            "text_on_theme": self._get_contrast_text_color(theme_color)
        }

    def _add_contrast_color(self, color_scheme: Dict[str, str]) -> Dict[str, str]:
        """add contrast color for keyword highlighting"""
        theme_color = color_scheme["theme"]
        hex_color = theme_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        comp_r, comp_g, comp_b = self._generate_complementary_color(r, g, b)
        contrast_color = self._reduce_saturation_brightness(comp_r, comp_g, comp_b)
        
        color_scheme["contrast"] = contrast_color
        return color_scheme
    
    def _generate_enhanced_light_variant(self, r: int, g: int, b: int) -> str:
        """generate light background color"""
        h, s, v = self._rgb_to_hsv(r, g, b)
        light_s = self.color_config["mono_light"]["saturation"]
        light_v = self.color_config["mono_light"]["brightness"]
        
        new_r, new_g, new_b = self._hsv_to_rgb(h, light_s, light_v)
        return f"#{int(new_r):02x}{int(new_g):02x}{int(new_b):02x}"
    
    def _generate_enhanced_dark_variant(self, r: int, g: int, b: int) -> str:
        """generate darker variant"""
        color_params = self.config["colors"]["saturation_adjustments"]
        bounds = self.config["colors"]["hsv_bounds"]
        
        h, s, v = self._rgb_to_hsv(r, g, b)
        
        s_range = self.color_config["mono_dark"]["saturation_range"]
        enhanced_s = min(1.0, max(s_range[0], self.color_config["mono_dark"]["saturation_default"]))
        enhanced_v = max(bounds["brightness_min"], v - color_params["dark_decrease"])
        
        new_r, new_g, new_b = self._hsv_to_rgb(h, enhanced_s, enhanced_v)
        return f"#{int(new_r):02x}{int(new_g):02x}{int(new_b):02x}"
    
    def _generate_complementary_color(self, r: int, g: int, b: int) -> Tuple[int, int, int]:
        """generate complementary color"""
        h, s, v = self._rgb_to_hsv(r, g, b)
        comp_h = (h + self.color_config["complementary"]["hue_offset"]) % 1.0
        comp_r, comp_g, comp_b = self._hsv_to_rgb(comp_h, s, v)
        return int(comp_r), int(comp_g), int(comp_b)
    
    def _reduce_saturation_brightness(self, r: int, g: int, b: int) -> str:
        """optimize contrast color for readability"""
        h, s, v = self._rgb_to_hsv(r, g, b)
        
        font_s = self.color_config["contrast_color"]["saturation"]
        font_v = self.color_config["contrast_color"]["brightness_start"]
        
        max_brightness = self.color_config["contrast_color"]["brightness_max"] 
        step = self.color_config["contrast_color"]["brightness_step"]
        required_ratio = self.color_config["contrast_color"]["wcag_contrast_ratio"]
        
        white_rgb = self.color_config["constants"]["white_rgb"]
        while font_v < max_brightness:
            test_r, test_g, test_b = self._hsv_to_rgb(h, font_s, font_v)
            if self._calculate_contrast_ratio(test_r, test_g, test_b, *white_rgb) >= required_ratio:
                break
            font_v += step
        
        final_r, final_g, final_b = self._hsv_to_rgb(h, font_s, font_v)
        contrast_color = f"#{int(final_r):02x}{int(final_g):02x}{int(final_b):02x}"
        
        return contrast_color
    
    def _calculate_contrast_ratio(self, r1: int, g1: int, b1: int, r2: int, g2: int, b2: int) -> float:
        """calculate WCAG contrast ratio between two colors"""
        l1 = self._get_relative_luminance(r1, g1, b1)
        l2 = self._get_relative_luminance(r2, g2, b2)
        if l1 < l2:
            l1, l2 = l2, l1
        
        return (l1 + self.color_config["srgb"]["gamma_offset"]) / (l2 + self.color_config["srgb"]["gamma_offset"])
    
    def _get_relative_luminance(self, r: int, g: int, b: int) -> float:
        """calculate relative luminance for WCAG contrast calculations"""
        max_rgb = self.color_config["constants"]["max_rgb"]
        r_norm = r / max_rgb
        g_norm = g / max_rgb
        b_norm = b / max_rgb
        
        def gamma_correct(c):
            threshold = self.color_config["srgb"]["gamma_threshold"]
            if c <= threshold:
                return c / self.color_config["constants"]["gamma_linear_divisor"]
            else:
                offset = self.color_config["srgb"]["gamma_offset"]
                divisor = self.color_config["srgb"]["gamma_divisor"]
                exponent = self.color_config["srgb"]["gamma_exponent"]
                return pow((c + offset) / divisor, exponent)
        
        r_linear = gamma_correct(r_norm)
        g_linear = gamma_correct(g_norm)
        b_linear = gamma_correct(b_norm)
        
        weights = self.color_config["luminance_weights"]
        return weights["red"] * r_linear + weights["green"] * g_linear + weights["blue"] * b_linear
    
    def _rgb_to_hsv(self, r: int, g: int, b: int) -> Tuple[float, float, float]:
        """convert rgb to hsv"""
        max_rgb = self.color_config["constants"]["max_rgb"]
        r, g, b = r/max_rgb, g/max_rgb, b/max_rgb
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val

        if diff == 0:
            h = 0
        elif max_val == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360

        s = 0 if max_val == 0 else diff / max_val
        v = max_val
        
        return h/360.0, s, v
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        h = h * 360  # convert back to degrees
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        max_rgb = self.color_config["constants"]["max_rgb"]
        return (r + m) * max_rgb, (g + m) * max_rgb, (b + m) * max_rgb

    def _get_contrast_text_color(self, bg_color: str) -> str:
        """determine appropriate text color for given background"""
        hex_color = bg_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        constants = self.color_config["constants"]
        brightness = (r * constants["red_weight"] + 
                     g * constants["green_weight"] + 
                     b * constants["blue_weight"]) / constants["brightness_divisor"]
        
        return (constants["white_text"] if brightness < constants["brightness_threshold"] 
                else constants["black_text"])

    def _save_color_scheme(self, state: PosterState):
        """save color scheme to json file"""
        output_dir = Path(state["output_dir"]) / "content"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "color_scheme.json", "w", encoding='utf-8') as f:
            json.dump(state.get("color_scheme", {}), f, indent=2)


def color_agent_node(state: PosterState) -> Dict[str, Any]:
    result = ColorAgent()(state)
    return {
        **state,
        "color_scheme": result["color_scheme"],
        "tokens": result["tokens"],
        "current_agent": result["current_agent"],
        "errors": result["errors"]
    }