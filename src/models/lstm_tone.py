"""LSTM 声调分类模型

基于 PyTorch 的 LSTM 模型，用于声调分类任务。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import logging

from .base import BaseModel


class ToneLSTM(nn.Module):
    """LSTM 声调分类模型

    使用双向 LSTM 和注意力机制进行声调分类。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """初始化 LSTM 模型

        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM 层数
            num_classes: 类别数量
            dropout: Dropout 比例
            bidirectional: 是否使用双向 LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # 输入投影层
        self.input_fc = nn.Linear(input_dim, hidden_dim)

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # 注意力机制
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

        # 分类层
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征 (batch_size, seq_len, input_dim)
            seq_lengths: 序列长度 (batch_size,)

        Returns:
            输出 logits (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.size()

        # 输入投影
        x = self.input_fc(x)  # (batch_size, seq_len, hidden_dim)

        # LSTM
        if seq_lengths is not None:
            # Pack padded sequence
            x = nn.utils.rnn.pack_padded_sequence(
                x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        lstm_out, _ = self.lstm(x)

        if seq_lengths is not None:
            # Unpack sequence
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )

        # 注意力机制
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # 加权求和
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, lstm_output_dim)

        # 分类
        output = self.fc(context)  # (batch_size, num_classes)

        return output


class LSTMToneClassifier(BaseModel):
    """LSTM 声调分类器

    封装 ToneLSTM 模型，提供训练、预测、评估接口。
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化分类器

        Args:
            config: 配置字典
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.device = None
        self.label_encoder = None
        self.scaler = None

    def build(
        self,
        input_dim: int,
        num_classes: int,
        device: torch.device = None
    ):
        """构建模型

        Args:
            input_dim: 输入特征维度
            num_classes: 类别数量
            device: 计算设备
        """
        lstm_config = self.config.get('lstm', {})

        self.model = ToneLSTM(
            input_dim=input_dim,
            hidden_dim=lstm_config.get('hidden_size', 128),
            num_layers=lstm_config.get('num_layers', 2),
            num_classes=num_classes,
            dropout=lstm_config.get('dropout', 0.3),
            bidirectional=True
        )

        self.device = device or torch.device('cpu')
        self.model.to(self.device)

        self.logger.info(f"LSTM 模型已构建")
        self.logger.info(f"  - 输入维度: {input_dim}")
        self.logger.info(f"  - 隐藏层维度: {lstm_config.get('hidden_size', 128)}")
        self.logger.info(f"  - 层数: {lstm_config.get('num_layers', 2)}")
        self.logger.info(f"  - 类别数: {num_classes}")
        self.logger.info(f"  - 设备: {self.device}")

    def train(self, *args, **kwargs):
        """训练模型

        注意：LSTM 的训练逻辑在 LSTMTrainer 中实现
        """
        raise NotImplementedError("请使用 LSTMTrainer 进行训练")

    def predict(self, X: torch.Tensor, seq_lengths: torch.Tensor = None) -> torch.Tensor:
        """预测

        Args:
            X: 输入特征 (batch_size, seq_len, input_dim)
            seq_lengths: 序列长度 (batch_size,)

        Returns:
            预测类别 (batch_size,)
        """
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            if seq_lengths is not None:
                seq_lengths = seq_lengths.to(self.device)

            logits = self.model(X, seq_lengths)
            predictions = torch.argmax(logits, dim=1)

        return predictions.cpu()

    def predict_proba(self, X: torch.Tensor, seq_lengths: torch.Tensor = None) -> torch.Tensor:
        """预测概率

        Args:
            X: 输入特征 (batch_size, seq_len, input_dim)
            seq_lengths: 序列长度 (batch_size,)

        Returns:
            预测概率 (batch_size, num_classes)
        """
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            if seq_lengths is not None:
                seq_lengths = seq_lengths.to(self.device)

            logits = self.model(X, seq_lengths)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu()

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
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                seq_lengths = batch.get('seq_lens')

                # 前向传播
                outputs = self.model(features, seq_lengths)

                # 计算损失
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(outputs, 1)
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
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler
        }, path)

        self.logger.info(f"模型已保存到: {path}")

    def load(self, path: str, device: torch.device = None):
        """加载模型

        Args:
            path: 模型路径
            device: 计算设备
        """
        checkpoint = torch.load(path, map_location=device or self.device)

        self.config = checkpoint.get('config', {})
        self.label_encoder = checkpoint.get('label_encoder')
        self.scaler = checkpoint.get('scaler')

        # 需要先构建模型
        # 这里假设模型已经构建好了
        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)

        self.logger.info(f"模型已加载: {path}")

    def set_preprocessors(self, scaler, label_encoder):
        """设置预处理器

        Args:
            scaler: 标准化器
            label_encoder: 标签编码器
        """
        self.scaler = scaler
        self.label_encoder = label_encoder
