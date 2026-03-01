"""模型抽象基类

定义模型的通用接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from pathlib import Path


class BaseModel(ABC):
    """模型抽象基类

    所有模型都应继承此类并实现抽象方法。
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化模型

        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None

    @abstractmethod
    def build(self):
        """构建模型"""
        pass

    @abstractmethod
    def train(self, X_train: Any, y_train: Any, **kwargs):
        """训练模型

        Args:
            X_train: 训练数据
            y_train: 训练标签
            **kwargs: 其他参数
        """
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """预测

        Args:
            X: 输入数据

        Returns:
            预测结果
        """
        pass

    @abstractmethod
    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, float]:
        """评估模型

        Args:
            X_test: 测试数据
            y_test: 测试标签

        Returns:
            评估指标字典
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """保存模型

        Args:
            path: 保存路径
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """加载模型

        Args:
            path: 模型路径
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息

        Returns:
            模型信息字典
        """
        return {
            'model_type': self.__class__.__name__,
            'config': self.config
        }
