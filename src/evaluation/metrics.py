"""评估指标模块

提供各种评估指标的计算功能。
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Optional
import logging


class MetricsCalculator:
    """评估指标计算器"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """计算分类指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            average: 平均方式 ('micro', 'macro', 'weighted')

        Returns:
            指标字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
        }

        return metrics

    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """获取混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签

        Returns:
            混淆矩阵
        """
        return confusion_matrix(y_true, y_pred)

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """获取分类报告

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            target_names: 类别名称列表

        Returns:
            分类报告字符串
        """
        return classification_report(
            y_true, y_pred,
            target_names=target_names,
            zero_division=0
        )

    def print_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ):
        """打印评估指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            target_names: 类别名称列表
        """
        # 计算基本指标
        metrics = self.calculate_classification_metrics(y_true, y_pred)

        self.logger.info("=" * 50)
        self.logger.info("评估指标")
        self.logger.info("=" * 50)
        self.logger.info(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        self.logger.info(f"精确率 (Precision): {metrics['precision']:.4f}")
        self.logger.info(f"召回率 (Recall): {metrics['recall']:.4f}")
        self.logger.info(f"F1 分数: {metrics['f1_score']:.4f}")
        self.logger.info("=" * 50)

        # 打印分类报告
        report = self.get_classification_report(y_true, y_pred, target_names)
        self.logger.info("\n分类报告:\n" + report)
