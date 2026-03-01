"""数据处理抽象基类

定义数据处理器的通用接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


class BaseDataProcessor(ABC):
    """数据处理器抽象基类

    所有数据处理器都应继承此类并实现抽象方法。
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化数据处理器

        Args:
            config: 配置字典
        """
        self.config = config

    @abstractmethod
    def load_data(self, data_path: str) -> Any:
        """加载数据

        Args:
            data_path: 数据路径

        Returns:
            加载的数据
        """
        pass

    @abstractmethod
    def process(self, data: Any) -> Any:
        """处理数据

        Args:
            data: 输入数据

        Returns:
            处理后的数据
        """
        pass

    @abstractmethod
    def save(self, data: Any, output_path: str):
        """保存数据

        Args:
            data: 要保存的数据
            output_path: 输出路径
        """
        pass


class BaseFeatureExtractor(ABC):
    """特征提取器抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        """初始化特征提取器

        Args:
            config: 配置字典
        """
        self.config = config

    @abstractmethod
    def extract(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """提取特征

        Args:
            audio_path: 音频文件路径
            **kwargs: 其他参数

        Returns:
            特征字典
        """
        pass


class BaseDatasetBuilder(ABC):
    """数据集构建器抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        """初始化数据集构建器

        Args:
            config: 配置字典
        """
        self.config = config

    @abstractmethod
    def build(
        self,
        data: Any,
        test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Tuple[Any, Any, Any, Any]:
        """构建训练集和测试集

        Args:
            data: 输入数据
            test_size: 测试集比例
            random_state: 随机种子

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        pass

    @abstractmethod
    def save_dataset(self, dataset: Any, output_path: str):
        """保存数据集

        Args:
            dataset: 数据集
            output_path: 输出路径
        """
        pass
