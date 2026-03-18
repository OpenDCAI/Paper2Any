"""
configuration loader
"""

import yaml
from pathlib import Path

def load_config():
    """Load configuration from poster_config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "poster_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)