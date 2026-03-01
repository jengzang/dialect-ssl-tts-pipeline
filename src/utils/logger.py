"""日志配置模块

提供统一的日志配置和管理功能。
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class LoggerManager:
    """日志管理器

    负责创建和配置日志记录器。
    """

    _loggers: Dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(
        cls,
        name: str,
        log_dir: Optional[str] = None,
        level: str = "INFO",
        log_format: Optional[str] = None
    ) -> logging.Logger:
        """获取或创建日志记录器

        Args:
            name: 日志记录器名称
            log_dir: 日志文件目录
            level: 日志级别
            log_format: 日志格式

        Returns:
            logging.Logger 实例
        """
        # 如果已存在，直接返回
        if name in cls._loggers:
            return cls._loggers[name]

        # 创建新的日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        # 避免重复添加处理器
        if logger.handlers:
            return logger

        # 默认日志格式
        if log_format is None:
            log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        formatter = logging.Formatter(log_format)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件处理器
        if log_dir is not None:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # 创建日志文件名（包含日期）
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = log_path / f"{name}_{timestamp}.log"

            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # 缓存日志记录器
        cls._loggers[name] = logger

        return logger


def setup_logger(
    name: str,
    config: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """设置日志记录器（便捷函数）

    Args:
        name: 日志记录器名称
        config: 配置字典，包含 log_dir, level, format 等

    Returns:
        logging.Logger 实例
    """
    if config is None:
        config = {}

    log_dir = config.get('log_dir', 'logs')
    level = config.get('level', 'INFO')
    log_format = config.get('format')

    return LoggerManager.get_logger(
        name=name,
        log_dir=log_dir,
        level=level,
        log_format=log_format
    )
