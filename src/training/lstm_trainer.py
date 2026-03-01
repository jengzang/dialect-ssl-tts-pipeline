"""LSTM 训练器模块

提供 LSTM 模型的训练、验证和早停功能。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import logging
from tqdm import tqdm
import numpy as np

from .base_trainer import BaseTrainer, EarlyStopping


class LSTMTrainer(BaseTrainer):
    """LSTM 训练器

    负责 LSTM 模型的训练、验证和早停。
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        """初始化训练器

        Args:
            model: LSTM 模型
            config: 配置字典
            device: 计算设备
            logger: 日志记录器
        """
        super().__init__(model, config, logger)
        self.device = device
        self.model.to(self.device)

        # 获取训练配置
        self.lstm_config = config.get('lstm', {})
        self.training_config = config.get('training', {})

        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lstm_config.get('learning_rate', 0.001)
        )

        # 初始化损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 初始化学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            **kwargs: 其他参数

        Returns:
            训练历史字典
        """
        epochs = epochs or self.lstm_config.get('epochs', 50)
        patience = self.lstm_config.get('early_stopping_patience', 5)

        # 初始化早停
        early_stopping = EarlyStopping(patience=patience, mode='min')

        self.logger.info("=" * 60)
        self.logger.info("开始训练 LSTM 模型")
        self.logger.info("=" * 60)
        self.logger.info(f"训练轮数: {epochs}")
        self.logger.info(f"批次大小: {train_loader.batch_size}")
        self.logger.info(f"学习率: {self.lstm_config.get('learning_rate', 0.001)}")
        self.logger.info(f"早停耐心: {patience}")
        self.logger.info("=" * 60)

        for epoch in range(epochs):
            # 训练阶段
            train_loss, train_acc = self._train_epoch(train_loader, epoch, epochs)

            # 验证阶段
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader)

                # 更新学习率
                self.scheduler.step(val_loss)

                # 记录历史
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # 打印进度
                self.logger.info(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

                # 早停检查
                if early_stopping(val_loss):
                    self.logger.info(f"早停触发，停止训练")
                    break
            else:
                # 没有验证集
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)

                self.logger.info(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )

        self.logger.info("=" * 60)
        self.logger.info("训练完成")
        self.logger.info("=" * 60)

        return self.history

    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> tuple:
        """训练一个 epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前 epoch
            total_epochs: 总 epoch 数

        Returns:
            (平均损失, 准确率)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")

        for batch in pbar:
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            seq_lengths = batch.get('seq_lens')

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(features, seq_lengths)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _validate_epoch(self, val_loader: DataLoader) -> tuple:
        """验证一个 epoch

        Args:
            val_loader: 验证数据加载器

        Returns:
            (平均损失, 准确率)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                seq_lengths = batch.get('seq_lens')

                # 前向传播
                outputs = self.model(features, seq_lengths)
                loss = self.criterion(outputs, labels)

                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """评估模型

        Args:
            test_loader: 测试数据加载器

        Returns:
            评估指标字典
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                seq_lengths = batch.get('seq_lens')

                # 前向传播
                outputs = self.model(features, seq_lengths)
                loss = self.criterion(outputs, labels)

                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels
        }

    def save_checkpoint(self, path: str, **kwargs):
        """保存检查点

        Args:
            path: 保存路径
            **kwargs: 其他参数
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config
        }
        checkpoint.update(kwargs)

        torch.save(checkpoint, path)
        self.logger.info(f"检查点已保存到: {path}")

    def load_checkpoint(self, path: str):
        """加载检查点

        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', {})
        self.config = checkpoint.get('config', {})

        self.logger.info(f"检查点已加载: {path}")
