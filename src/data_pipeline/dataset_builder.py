"""数据集构建模块

提供数据集划分、归一化等功能。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import joblib

from .base import BaseDatasetBuilder


class SVMDatasetBuilder(BaseDatasetBuilder):
    """SVM 数据集构建器

    负责数据集的划分、归一化和保存。
    """

    def __init__(self, config: Dict):
        """初始化数据集构建器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def build(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        label_column: str,
        test_size: float = 0.2,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """构建训练集和测试集

        Args:
            data: 输入数据 DataFrame
            feature_columns: 特征列名列表
            label_column: 标签列名
            test_size: 测试集比例
            random_state: 随机种子

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        self.logger.info(f"构建数据集，特征列: {feature_columns}")
        self.logger.info(f"标签列: {label_column}")

        # 提取特征和标签
        X = data[feature_columns].values
        y = data[label_column].values

        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)

        self.logger.info(f"数据集大小: {len(X)}")
        self.logger.info(f"类别数量: {len(self.label_encoder.classes_)}")
        self.logger.info(f"类别: {self.label_encoder.classes_}")

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded
        )

        # 归一化特征
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.logger.info(f"训练集大小: {len(X_train)}")
        self.logger.info(f"测试集大小: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def save_dataset(
        self,
        dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        output_dir: str
    ):
        """保存数据集

        Args:
            dataset: (X_train, X_test, y_train, y_test)
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        X_train, X_test, y_train, y_test = dataset

        # 保存数据
        np.save(output_path / "X_train.npy", X_train)
        np.save(output_path / "X_test.npy", X_test)
        np.save(output_path / "y_train.npy", y_train)
        np.save(output_path / "y_test.npy", y_test)

        # 保存标准化器和标签编码器
        joblib.dump(self.scaler, output_path / "scaler.pkl")
        joblib.dump(self.label_encoder, output_path / "label_encoder.pkl")

        self.logger.info(f"数据集已保存到: {output_path}")

    def load_dataset(
        self,
        dataset_dir: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载数据集

        Args:
            dataset_dir: 数据集目录

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        dataset_path = Path(dataset_dir)

        X_train = np.load(dataset_path / "X_train.npy")
        X_test = np.load(dataset_path / "X_test.npy")
        y_train = np.load(dataset_path / "y_train.npy")
        y_test = np.load(dataset_path / "y_test.npy")

        # 加载标准化器和标签编码器
        self.scaler = joblib.load(dataset_path / "scaler.pkl")
        self.label_encoder = joblib.load(dataset_path / "label_encoder.pkl")

        self.logger.info(f"数据集已加载: {dataset_path}")

        return X_train, X_test, y_train, y_test
