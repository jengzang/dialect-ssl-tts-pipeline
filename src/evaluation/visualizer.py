"""可视化模块

提供混淆矩阵、t-SNE 等可视化功能。
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
from typing import List, Optional
import logging


class Visualizer:
    """可视化工具"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        output_path: str,
        title: str = "Confusion Matrix",
        figsize: tuple = (10, 8)
    ):
        """绘制混淆矩阵

        Args:
            cm: 混淆矩阵
            class_names: 类别名称列表
            output_path: 输出路径
            title: 标题
            figsize: 图像大小
        """
        plt.figure(figsize=figsize)

        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 绘制热力图
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Normalized Count'}
        )

        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        # 保存图像
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"混淆矩阵已保存到: {output_file}")

    def plot_tsne(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: List[str],
        output_path: str,
        title: str = "t-SNE Visualization",
        figsize: tuple = (12, 10),
        perplexity: int = 30,
        random_state: int = 42
    ):
        """绘制 t-SNE 可视化

        Args:
            X: 特征数据
            y: 标签
            class_names: 类别名称列表
            output_path: 输出路径
            title: 标题
            figsize: 图像大小
            perplexity: t-SNE 参数
            random_state: 随机种子
        """
        self.logger.info("计算 t-SNE...")

        # 计算 t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000
        )
        X_tsne = tsne.fit_transform(X)

        # 绘制散点图
        plt.figure(figsize=figsize)

        for i, class_name in enumerate(class_names):
            mask = y == i
            plt.scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                label=class_name,
                alpha=0.6,
                s=50
            )

        plt.title(title)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # 保存图像
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"t-SNE 图已保存到: {output_file}")

    def plot_feature_distribution(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
        output_path: str,
        figsize: tuple = (15, 10)
    ):
        """绘制特征分布图

        Args:
            X: 特征数据
            y: 标签
            feature_names: 特征名称列表
            class_names: 类别名称列表
            output_path: 输出路径
            figsize: 图像大小
        """
        n_features = X.shape[1]
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i, feature_name in enumerate(feature_names):
            ax = axes[i]

            for j, class_name in enumerate(class_names):
                mask = y == j
                ax.hist(
                    X[mask, i],
                    alpha=0.5,
                    label=class_name,
                    bins=20
                )

            ax.set_title(feature_name)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()

        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        # 保存图像
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"特征分布图已保存到: {output_file}")
