"""方言口音识别模型

基于 wav2vec 2.0 的口音分类器。
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

from .base import BaseModel


class AccentClassifier(BaseModel):
    """方言口音分类器

    基于 wav2vec 2.0 的口音识别模型。
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化模型

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

        self.model_name = config.get('wav2vec', {}).get(
            'model_name', 'facebook/wav2vec2-base'
        )
        self.num_classes = None
        self.processor = None
        self.label_encoder = None

    def build(self, num_classes: int):
        """构建模型

        Args:
            num_classes: 口音类别数量
        """
        self.logger.info(f"构建口音分类器: {self.model_name}")
        self.num_classes = num_classes

        # 加载预训练的 wav2vec 2.0
        wav2vec_model = Wav2Vec2Model.from_pretrained(self.model_name)

        # 构建分类器
        self.model = AccentClassifierModel(
            wav2vec_model=wav2vec_model,
            num_classes=num_classes,
            freeze_feature_encoder=self.config.get('wav2vec', {}).get(
                'freeze_feature_encoder', True
            )
        )

        # 加载 processor
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)

        self.logger.info(f"模型已构建，类别数: {num_classes}")

    def train(self, *args, **kwargs):
        """训练模型

        注意：训练逻辑在 AccentTrainer 中实现
        """
        raise NotImplementedError("请使用 AccentTrainer 进行训练")

    def predict(self, audio_input: torch.Tensor) -> int:
        """预测口音类别

        Args:
            audio_input: 音频输入张量

        Returns:
            预测的类别索引
        """
        self.model.eval()

        with torch.no_grad():
            # 处理音频
            if audio_input.dim() == 1:
                inputs = self.processor(
                    audio_input,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                audio_input = inputs.input_values

            # 前向传播
            logits = self.model(audio_input)
            predicted_class = torch.argmax(logits, dim=-1).item()

        return predicted_class

    def predict_proba(self, audio_input: torch.Tensor) -> torch.Tensor:
        """预测口音概率

        Args:
            audio_input: 音频输入张量

        Returns:
            预测概率 (num_classes,)
        """
        self.model.eval()

        with torch.no_grad():
            if audio_input.dim() == 1:
                inputs = self.processor(
                    audio_input,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                audio_input = inputs.input_values

            logits = self.model(audio_input)
            probs = torch.softmax(logits, dim=-1).squeeze()

        return probs

    def evaluate(self, test_loader, criterion=None) -> Dict[str, float]:
        """评估模型

        Args:
            test_loader: 测试数据加载器
            criterion: 损失函数

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
                audio = batch['audio'].to(self.model.device)
                labels = batch['label'].to(self.model.device)

                # 前向传播
                logits = self.model(audio)

                # 计算损失
                if criterion is not None:
                    loss = criterion(logits, labels)
                    total_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0

        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'predictions': all_preds,
            'labels': all_labels
        }

    def save(self, path: str):
        """保存模型

        Args:
            path: 保存路径
        """
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'config': self.config,
            'label_encoder': self.label_encoder
        }, output_dir / 'model.pth')

        # 保存 processor
        self.processor.save_pretrained(output_dir)

        self.logger.info(f"模型已保存到: {output_dir}")

    def load(self, path: str, device: torch.device = None):
        """加载模型

        Args:
            path: 模型路径
            device: 计算设备
        """
        model_dir = Path(path)

        # 加载 processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir)

        # 加载模型
        checkpoint = torch.load(
            model_dir / 'model.pth',
            map_location=device or torch.device('cpu')
        )

        self.num_classes = checkpoint['num_classes']
        self.config = checkpoint.get('config', {})
        self.label_encoder = checkpoint.get('label_encoder')

        # 重建模型
        self.build(self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if device:
            self.model.to(device)

        self.logger.info(f"模型已加载: {model_dir}")

    def set_label_encoder(self, label_encoder):
        """设置标签编码器

        Args:
            label_encoder: 标签编码器
        """
        self.label_encoder = label_encoder


class AccentClassifierModel(nn.Module):
    """口音分类器模型

    基于 wav2vec 2.0 的分类器。
    """

    def __init__(
        self,
        wav2vec_model: Wav2Vec2Model,
        num_classes: int,
        freeze_feature_encoder: bool = True
    ):
        """初始化模型

        Args:
            wav2vec_model: 预训练的 wav2vec 2.0 模型
            num_classes: 类别数量
            freeze_feature_encoder: 是否冻结特征提取器
        """
        super().__init__()

        self.wav2vec = wav2vec_model
        self.num_classes = num_classes

        # 冻结特征提取器
        if freeze_feature_encoder:
            self.wav2vec.feature_extractor._freeze_parameters()

        # 分类头
        hidden_size = self.wav2vec.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

        self.device = torch.device('cpu')

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            input_values: 音频输入 (batch_size, seq_len)

        Returns:
            logits (batch_size, num_classes)
        """
        # wav2vec 2.0 编码
        outputs = self.wav2vec(input_values)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # 平均池化
        pooled = torch.mean(hidden_states, dim=1)  # (batch_size, hidden_size)

        # 分类
        logits = self.classifier(pooled)  # (batch_size, num_classes)

        return logits

    def to(self, device):
        """移动到设备

        Args:
            device: 目标设备
        """
        self.device = device
        return super().to(device)
