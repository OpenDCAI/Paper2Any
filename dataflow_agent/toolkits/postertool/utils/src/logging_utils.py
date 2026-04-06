"""logging utility for all LangGraph agents"""

import inspect
from pathlib import Path
from rich.console import Console
from rich.text import Text

console = Console()

def _get_caller_info():
    """get the caller file and line number, skipping this logging_utils.py file"""
    frame = inspect.currentframe()
    try:
        current_frame = frame.f_back
        while current_frame:
            filepath = Path(current_frame.f_code.co_filename)
            filename = filepath.name
            
            if filename != "logging_utils.py":
                line_number = current_frame.f_lineno
                
                # try to get relative path from current working directory or project root
                try:
                    relative_path = filepath.relative_to(Path.cwd())
                except ValueError:
                    try:
                        current_dir = Path.cwd()
                        while current_dir.parent != current_dir:  
                            if (current_dir / "README.md").exists() or (current_dir / "requirements.txt").exists():
                                relative_path = filepath.relative_to(current_dir)
                                break
                            current_dir = current_dir.parent
                        else:
                            relative_path = filepath.name
                    except ValueError:
                        relative_path = filepath.name
                
                return f"{relative_path}:{line_number}"
            current_frame = current_frame.f_back
        return "unknown:0"
    finally:
        del frame

def log(agent_name: str, level: str, message: str, max_width: int = 15, show_location: bool = True):
    """
    centralized logging function for all agents
    
    args:
        agent_name: name of the agent (e.g., "parser", "color_agent")
        level: log level (e.g., "info", "warning", "error", "success")
        message: the message to log
        max_width: maximum width for the agent name padding
        show_location: whether to show file location info
    """
    # clean agent name for display
    display_name = agent_name.replace("_agent", "").replace("_node", "").replace("_", " ").title()
    
    # color scheme based on level
    level_colors = {
        "info": "cyan",
        "warning": "yellow", 
        "error": "red",
        "success": "green",
        "debug": "blue"
    }
    
    level_color = level_colors.get(level.lower(), "white")
    
    # create header with fixed width
    header = Text(f"[ {display_name:^{max_width}} ]", style=f"bold {level_color}")
    
    # add location info if requested
    if show_location:
        location_info = _get_caller_info()
        location_text = Text(f" [{location_info}] ", style="dim")
        header.append(location_text)
    
    body = Text(message)
    
    console.print(header, body)


def log_agent_start(agent_name: str, show_location: bool = True):
    """log agent start with separator"""
    log(agent_name, "info", f"starting {agent_name}...", show_location=show_location)


def log_agent_success(agent_name: str, message: str, show_location: bool = True):
    """log agent success"""
    log(agent_name, "success", f"✅ {message}", show_location=show_location)


def log_agent_error(agent_name: str, message: str, show_location: bool = True):
    """log agent error"""
    log(agent_name, "error", f"❌ {message}", show_location=show_location)


def log_agent_warning(agent_name: str, message: str, show_location: bool = True):
    """log agent warning"""
    log(agent_name, "warning", f"⚠️ {message}", show_location=show_location)


def log_agent_info(agent_name: str, message: str, show_location: bool = True):
    """log agent info"""
    log(agent_name, "info", message, show_location=show_location)