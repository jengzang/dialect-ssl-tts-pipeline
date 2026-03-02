#!/usr/bin/env python3
"""
Prefix Tuning 模型实现

Prefix Tuning 通过在输入序列前添加可学习的连续提示（prefix）来微调模型，
而不修改预训练模型的参数。

参考论文: Prefix-Tuning: Optimizing Continuous Prompts for Generation (Li & Liang, 2021)
"""

import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel
)

logger = logging.getLogger(__name__)


class PrefixEncoder(nn.Module):
    """
    Prefix 编码器

    将可学习的 prefix 参数映射到模型的键值空间
    """

    def __init__(
        self,
        prefix_length: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int = 512
    ):
        """
        Args:
            prefix_length: Prefix 长度（token 数）
            num_layers: Transformer 层数
            num_heads: 注意力头数
            head_dim: 每个注意力头的维度
            hidden_size: 中间隐藏层大小
        """
        super().__init__()

        self.prefix_length = prefix_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # 每层需要 key 和 value，每个的维度是 num_heads * head_dim
        kv_dim = num_heads * head_dim

        # Prefix 参数：[prefix_length, num_layers * 2 * kv_dim]
        # 2 表示 key 和 value
        self.embedding = nn.Embedding(prefix_length, num_layers * 2 * kv_dim)

        # MLP 重参数化（可选，提高表达能力）
        self.trans = nn.Sequential(
            nn.Linear(num_layers * 2 * kv_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_layers * 2 * kv_dim)
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        生成 prefix 的键值对

        Args:
            batch_size: 批次大小

        Returns:
            形状为 [batch_size, prefix_length, num_layers * 2 * num_heads * head_dim]
        """
        # 生成 prefix 索引
        prefix_indices = torch.arange(self.prefix_length, device=self.embedding.weight.device)

        # 获取 prefix embeddings: [prefix_length, num_layers * 2 * kv_dim]
        prefix_embeds = self.embedding(prefix_indices)

        # MLP 变换
        prefix_embeds = self.trans(prefix_embeds)

        # 扩展到 batch: [batch_size, prefix_length, num_layers * 2 * kv_dim]
        prefix_embeds = prefix_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        return prefix_embeds


class PrefixTuningModel(nn.Module):
    """
    Prefix Tuning 模型包装器

    在预训练模型前添加可学习的 prefix
    """

    def __init__(
        self,
        model: PreTrainedModel,
        prefix_length: int = 10,
        prefix_hidden_size: int = 512,
        freeze_model: bool = True
    ):
        """
        Args:
            model: 预训练模型
            prefix_length: Prefix 长度
            prefix_hidden_size: Prefix 编码器隐藏层大小
            freeze_model: 是否冻结预训练模型参数
        """
        super().__init__()

        self.model = model
        self.prefix_length = prefix_length

        # 冻结预训练模型
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        # 获取模型配置
        config = model.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads

        # 创建 prefix 编码器
        self.prefix_encoder = PrefixEncoder(
            prefix_length=prefix_length,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=prefix_hidden_size
        )

        logger.info(f"PrefixTuningModel initialized:")
        logger.info(f"  Prefix length: {prefix_length}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Num heads: {num_heads}")
        logger.info(f"  Head dim: {head_dim}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            labels: 标签（用于计算损失）
            **kwargs: 其他参数

        Returns:
            包含 loss, logits 等的字典
        """
        batch_size = input_ids.shape[0]

        # 生成 prefix
        # 注意：这里简化实现，实际需要将 prefix 注入到每层的 key/value
        # 完整实现需要修改模型的 forward 方法或使用 hooks

        # 扩展 attention_mask 以包含 prefix
        if attention_mask is not None:
            prefix_mask = torch.ones(
                batch_size, self.prefix_length,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 调用原始模型
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        return outputs

    def get_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.prefix_encoder.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        """获取总参数数量"""
        return sum(p.numel() for p in self.parameters())


def create_prefix_tuning_model(
    model_name: str,
    prefix_length: int = 10,
    prefix_hidden_size: int = 512,
    device: str = "cuda"
) -> PrefixTuningModel:
    """
    创建 Prefix Tuning 模型

    Args:
        model_name: 预训练模型名称
        prefix_length: Prefix 长度
        prefix_hidden_size: Prefix 编码器隐藏层大小
        device: 设备

    Returns:
        PrefixTuningModel 实例
    """
    logger.info(f"Creating Prefix Tuning model: {model_name}")

    # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    # 创建 Prefix Tuning 模型
    prefix_model = PrefixTuningModel(
        model=model,
        prefix_length=prefix_length,
        prefix_hidden_size=prefix_hidden_size,
        freeze_model=True
    )

    prefix_model = prefix_model.to(device)

    # 统计参数
    total_params = prefix_model.get_total_parameters()
    trainable_params = prefix_model.get_trainable_parameters()
    trainable_ratio = trainable_params / total_params * 100

    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable ratio: {trainable_ratio:.2f}%")

    return prefix_model


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 创建一个小模型测试
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=1000,
        n_positions=512,
        n_embd=256,
        n_layer=4,
        n_head=4
    )

    model = GPT2LMHeadModel(config)

    # 创建 Prefix Tuning 模型
    prefix_model = PrefixTuningModel(
        model=model,
        prefix_length=10,
        prefix_hidden_size=128
    )

    # 测试前向传播
    input_ids = torch.randint(0, 1000, (2, 20))
    outputs = prefix_model(input_ids)

    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Trainable params: {prefix_model.get_trainable_parameters():,}")
    print(f"Total params: {prefix_model.get_total_parameters():,}")
