"""SVM 分类器模块

提供基于 sklearn 的 SVM 分类器。
"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np
from typing import Dict, Any
import logging

from .base import BaseModel


class SVMClassifier(BaseModel):
    """SVM 元音分类器

    基于 sklearn 的 SVM 分类器，用于元音分类任务。
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化 SVM 分类器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.scaler = None
        self.label_encoder = None

    def build(self):
        """构建 SVM 模型"""
        svm_config = self.config.get('svm', {})

        self.model = SVC(
            kernel=svm_config.get('kernel', 'rbf'),
            C=svm_config.get('C', 1.0),
            gamma=svm_config.get('gamma', 'auto'),
            random_state=svm_config.get('random_state', 42),
            probability=True  # 启用概率估计
        )

        self.logger.info(f"SVM 模型已构建: kernel={self.model.kernel}, C={self.model.C}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """训练模型

        Args:
            X_train: 训练数据
            y_train: 训练标签
            **kwargs: 其他参数
        """
        if self.model is None:
            self.build()

        self.logger.info("开始训练 SVM 模型...")
        self.model.fit(X_train, y_train)
        self.logger.info("SVM 模型训练完成")

        # 计算训练集准确率
        train_acc = self.model.score(X_train, y_train)
        self.logger.info(f"训练集准确率: {train_acc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测

        Args:
            X: 输入数据

        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train() 方法")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率

        Args:
            X: 输入数据

        Returns:
            预测概率
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train() 方法")

        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """评估模型

        Args:
            X_test: 测试数据
            y_test: 测试标签

        Returns:
            评估指标字典
        """
        y_pred = self.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        self.logger.info(f"测试集准确率: {accuracy:.4f}")

        # 打印分类报告
        if self.label_encoder is not None:
            target_names = self.label_encoder.classes_
        else:
            target_names = None

        report = classification_report(
            y_test, y_pred,
            target_names=target_names,
            output_dict=True
        )

        return {
            'accuracy': accuracy,
            'report': report
        }

    def save(self, path: str):
        """保存模型

        Args:
            path: 保存路径
        """
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'config': self.config
        }, path)

        self.logger.info(f"模型已保存到: {path}")

    def load(self, path: str):
        """加载模型

        Args:
            path: 模型路径
        """
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data.get('scaler')
        self.label_encoder = data.get('label_encoder')
        self.config = data.get('config', {})

        self.logger.info(f"模型已加载: {path}")

    def set_preprocessors(self, scaler, label_encoder):
        """设置预处理器

        Args:
            scaler: 标准化器
            label_encoder: 标签编码器
        """
        self.scaler = scaler
        self.label_encoder = label_encoder
