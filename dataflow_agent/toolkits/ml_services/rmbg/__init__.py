"""
RMBG-2.0 background removal service.

Client and server for background removal.
"""

from .client import RMBGClient
from .server import create_rmbg_server

__all__ = ["RMBGClient", "create_rmbg_server"]
