"""配置加载器模块

提供统一的配置文件加载和访问接口。
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """配置加载器

    负责加载和管理项目配置文件。
    """

    def __init__(self, config_path: Optional[str] = None):
        """初始化配置加载器

        Args:
            config_path: 配置文件路径，默认为项目根目录的 config.yaml
        """
        if config_path is None:
            # 默认配置文件路径
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None

    def load(self) -> Dict[str, Any]:
        """加载配置文件

        Returns:
            配置字典

        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: 配置文件格式错误
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        return self._config

    @property
    def config(self) -> Dict[str, Any]:
        """获取配置字典

        Returns:
            配置字典
        """
        if self._config is None:
            self.load()
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项

        Args:
            key: 配置键，支持点号分隔的嵌套键（如 'paths.data_dir'）
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_lesson_config(self, lesson_name: str) -> Dict[str, Any]:
        """获取特定课程的配置

        Args:
            lesson_name: 课程名称（如 'svm', 'lstm', 'wav2vec'）

        Returns:
            课程配置字典
        """
        return self.config.get(lesson_name, {})


# 全局配置实例
_global_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_path: Optional[str] = None) -> ConfigLoader:
    """获取全局配置加载器实例

    Args:
        config_path: 配置文件路径

    Returns:
        ConfigLoader 实例
    """
    global _global_config_loader

    if _global_config_loader is None:
        _global_config_loader = ConfigLoader(config_path)

    return _global_config_loader
