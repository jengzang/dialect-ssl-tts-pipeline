"""训练器抽象基类

定义训练器的通用接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import logging


class BaseTrainer(ABC):
    """训练器抽象基类

    所有训练器都应继承此类并实现抽象方法。
    """

    def __init__(
        self,
        model: Any,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """初始化训练器

        Args:
            model: 模型实例
            config: 配置字典
            logger: 日志记录器
        """
        self.model = model
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def train(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """训练模型

        Args:
            train_data: 训练数据
            val_data: 验证数据
            **kwargs: 其他参数

        Returns:
            训练历史字典
        """
        pass

    @abstractmethod
    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """评估模型

        Args:
            test_data: 测试数据

        Returns:
            评估指标字典
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: str, **kwargs):
        """保存检查点

        Args:
            path: 保存路径
            **kwargs: 其他参数
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path: str):
        """加载检查点

        Args:
            path: 检查点路径
        """
        pass

    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """记录指标

        Args:
            metrics: 指标字典
            prefix: 前缀
        """
        for key, value in metrics.items():
            self.logger.info(f"{prefix}{key}: {value:.4f}")


class EarlyStopping:
    """早停机制

    用于在验证集性能不再提升时提前停止训练。
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """初始化早停机制

        Args:
            patience: 容忍的 epoch 数
            min_delta: 最小改进量
            mode: 'min' 或 'max'，表示指标越小越好还是越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """检查是否应该早停

        Args:
            score: 当前指标值

        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
