import logging
import os
from logging.handlers import RotatingFileHandler

DEFAULT_LOG_FILE = os.getenv("DATAFLOW_LOG_FILE", "dataflow_agent.log")
DEFAULT_LOG_LEVEL = os.getenv("DATAFLOW_LOG_LEVEL", "INFO").upper()
MAX_LOG_SIZE = 10 * 1024 * 1024
BACKUP_COUNT = 5

# ANSI 颜色码
COLOR_MAP = {
    "DEBUG": "\033[36m",    # 青色
    "INFO": "\033[32m",     # 绿色
    "WARNING": "\033[33m",  # 黄色
    "ERROR": "\033[31m",    # 红色
    "CRITICAL": "\033[41m", # 红底
    "RESET": "\033[0m",
}

class ColorFormatter(logging.Formatter):
    """
    支持不同日志级别高亮显示的 Formatter，仅限控制台输出。
    """
    def format(self, record):
        level_name = record.levelname
        color = COLOR_MAP.get(level_name, "")
        reset = COLOR_MAP["RESET"]
        msg = super().format(record)
        if color:
            msg = f"{color}{msg}{reset}"
        return msg

def _create_handler():
    """创建控制台和文件的日志处理器。"""
    # 控制台输出（带颜色）
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(DEFAULT_LOG_LEVEL)
    color_formatter = ColorFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(color_formatter)

    # 文件输出（不带颜色）
    file_handler = RotatingFileHandler(DEFAULT_LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8")
    file_handler.setLevel(DEFAULT_LOG_LEVEL)
    plain_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(plain_formatter)
    return [stream_handler, file_handler]

def get_logger(name: str = "dataflow_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handlers = _create_handler()
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(DEFAULT_LOG_LEVEL)
        logger.propagate = False
    return logger

log = get_logger()

if __name__ == "__main__":
    log.info("Logger 初始化成功")
    log.debug("This is a debug message.")
    log.warning("This is a warning.")
    log.error("This is an error.")
    log.critical("This is CRITICAL!")