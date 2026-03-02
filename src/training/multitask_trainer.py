"""
多任务训练器

支持翻译和分类任务的联合训练。

特性：
- 任务平衡策略
- 损失加权
- 梯度归一化
- 多任务评估
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class MultitaskTrainer:
    """多任务训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cuda",
        loss_weights: Optional[Dict[str, float]] = None,
        gradient_normalization: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "multitask-dialect",
        output_dir: str = "checkpoints/multitask"
    ):
        """
        初始化多任务训练器

        Args:
            model: 多任务模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            device: 设备
            loss_weights: 任务损失权重 {'translation': 1.0, 'classification': 1.0}
            gradient_normalization: 是否使用梯度归一化
            use_wandb: 是否使用 WandB
            wandb_project: WandB 项目名
            output_dir: 输出目录
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_normalization = gradient_normalization
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 损失权重
        self.loss_weights = loss_weights or {
            'translation': 1.0,
            'classification': 1.0
        }

        # 优化器
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=5e-5,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer

        # WandB
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project=wandb_project, config={
                'loss_weights': self.loss_weights,
                'gradient_normalization': gradient_normalization
            })

        # 训练统计
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        logger.info("MultitaskTrainer initialized")
        logger.info(f"Loss weights: {self.loss_weights}")
        logger.info(f"Gradient normalization: {gradient_normalization}")

    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        translation_loss = 0.0
        classification_loss = 0.0
        translation_count = 0
        classification_count = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch in progress_bar:
            # 移动到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # 前向传播
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                task_id=batch['task_id'],
                labels=batch.get('labels'),
                classification_label=batch.get('classification_label')
            )

            loss = outputs['loss']
            task_type = outputs['task_type']

            # 应用损失权重
            weighted_loss = loss * self.loss_weights.get(task_type, 1.0)

            # 反向传播
            self.optimizer.zero_grad()
            weighted_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            # 统计
            total_loss += weighted_loss.item()
            if task_type == 'translation':
                translation_loss += loss.item()
                translation_count += 1
            else:
                classification_loss += loss.item()
                classification_count += 1

            self.global_step += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': weighted_loss.item(),
                'task': task_type
            })

            # WandB 日志
            if self.use_wandb and self.global_step % 10 == 0:
                wandb.log({
                    'train/loss': weighted_loss.item(),
                    'train/task': task_type,
                    'global_step': self.global_step
                })

        # 计算平均损失
        avg_total_loss = total_loss / len(self.train_loader)
        avg_translation_loss = translation_loss / translation_count if translation_count > 0 else 0
        avg_classification_loss = classification_loss / classification_count if classification_count > 0 else 0

        metrics = {
            'train/total_loss': avg_total_loss,
            'train/translation_loss': avg_translation_loss,
            'train/classification_loss': avg_classification_loss,
            'train/translation_samples': translation_count,
            'train/classification_samples': classification_count
        }

        logger.info(f"Epoch {self.epoch} - Train Loss: {avg_total_loss:.4f}")
        logger.info(f"  Translation Loss: {avg_translation_loss:.4f} ({translation_count} samples)")
        logger.info(f"  Classification Loss: {avg_classification_loss:.4f} ({classification_count} samples)")

        return metrics

    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        translation_loss = 0.0
        classification_loss = 0.0
        translation_count = 0
        classification_count = 0
        classification_correct = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # 移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # 前向传播
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    task_id=batch['task_id'],
                    labels=batch.get('labels'),
                    classification_label=batch.get('classification_label')
                )

                loss = outputs['loss']
                task_type = outputs['task_type']

                # 统计
                if loss is not None:
                    total_loss += loss.item()

                if task_type == 'translation':
                    translation_loss += loss.item()
                    translation_count += 1
                else:
                    classification_loss += loss.item()
                    classification_count += 1

                    # 计算分类准确率
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                    classification_correct += (predictions == batch['classification_label']).sum().item()

        # 计算平均指标
        avg_total_loss = total_loss / len(self.val_loader)
        avg_translation_loss = translation_loss / translation_count if translation_count > 0 else 0
        avg_classification_loss = classification_loss / classification_count if classification_count > 0 else 0
        classification_accuracy = classification_correct / classification_count if classification_count > 0 else 0

        metrics = {
            'val/total_loss': avg_total_loss,
            'val/translation_loss': avg_translation_loss,
            'val/classification_loss': avg_classification_loss,
            'val/classification_accuracy': classification_accuracy,
            'val/translation_samples': translation_count,
            'val/classification_samples': classification_count
        }

        logger.info(f"Validation - Total Loss: {avg_total_loss:.4f}")
        logger.info(f"  Translation Loss: {avg_translation_loss:.4f} ({translation_count} samples)")
        logger.info(f"  Classification Loss: {avg_classification_loss:.4f} ({classification_count} samples)")
        logger.info(f"  Classification Accuracy: {classification_accuracy:.4f}")

        return metrics

    def train(self, num_epochs: int, save_every: int = 1):
        """
        训练模型

        Args:
            num_epochs: 训练轮数
            save_every: 每隔多少轮保存一次
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # 训练
            train_metrics = self.train_epoch()

            # 评估
            val_metrics = self.evaluate()

            # 合并指标
            all_metrics = {**train_metrics, **val_metrics, 'epoch': epoch}

            # WandB 日志
            if self.use_wandb:
                wandb.log(all_metrics)

            # 保存检查点
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch+1}")

            # 保存最佳模型
            if val_metrics and val_metrics['val/total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val/total_loss']
                self.save_checkpoint("best")
                logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")

        logger.info("Training completed")

        # 保存最终模型
        self.save_checkpoint("final")

        # 保存训练摘要
        self.save_training_summary()

    def save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        self.model.save_pretrained(str(checkpoint_dir))

        # 保存优化器状态
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(self.optimizer.state_dict(), optimizer_path)

        # 保存训练状态
        state_path = checkpoint_dir / "trainer_state.json"
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def save_training_summary(self):
        """保存训练摘要"""
        summary_path = self.output_dir / "training_summary.json"
        summary = {
            'total_epochs': self.epoch + 1,
            'total_steps': self.global_step,
            'best_val_loss': self.best_val_loss,
            'loss_weights': self.loss_weights,
            'gradient_normalization': self.gradient_normalization
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training summary saved to {summary_path}")

