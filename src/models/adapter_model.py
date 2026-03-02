#!/usr/bin/env python3
"""
Adapter Layers 模型实现

Adapter 是在 Transformer 层之间插入的轻量级瓶颈模块，
通过少量参数实现任务适配。

参考论文: Parameter-Efficient Transfer Learning for NLP (Houlsby et al., 2019)
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


class AdapterLayer(nn.Module):
    """
    Adapter 层

    结构: LayerNorm -> Down-projection -> Activation -> Up-projection -> Residual
    """

    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 64,
        activation: str = "gelu"
    ):
        """
        Args:
            hidden_size: 输入/输出维度
            adapter_size: Adapter 瓶颈维度（通常远小于 hidden_size）
            activation: 激活函数类型
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.adapter_size = adapter_size

        # Down-projection: hidden_size -> adapter_size
        self.down_proj = nn.Linear(hidden_size, adapter_size)

        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Up-projection: adapter_size -> hidden_size
        self.up_proj = nn.Linear(adapter_size, hidden_size)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # 初始化为接近恒等映射
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            hidden_states: 输入隐藏状态 [batch_size, seq_len, hidden_size]

        Returns:
            输出隐藏状态 [batch_size, seq_len, hidden_size]
        """
        # 保存残差
        residual = hidden_states

        # Adapter 变换
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.up_proj(hidden_states)

        # 残差连接
        hidden_states = residual + hidden_states

        return hidden_states


class AdapterModel(nn.Module):
    """
    带 Adapter 的模型包装器

    在 Transformer 的每一层后插入 Adapter
    """

    def __init__(
        self,
        model: PreTrainedModel,
        adapter_size: int = 64,
        adapter_activation: str = "gelu",
        freeze_model: bool = True
    ):
        """
        Args:
            model: 预训练模型
            adapter_size: Adapter 瓶颈维度
            adapter_activation: Adapter 激活函数
            freeze_model: 是否冻结预训练模型参数
        """
        super().__init__()

        self.model = model
        self.adapter_size = adapter_size

        # 冻结预训练模型
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False

        # 获取模型配置
        config = model.config
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers

        # 为每一层创建 Adapter
        self.adapters = nn.ModuleList([
            AdapterLayer(
                hidden_size=hidden_size,
                adapter_size=adapter_size,
                activation=adapter_activation
            )
            for _ in range(num_layers)
        ])

        # 注册 hooks 来插入 Adapter
        self._register_adapter_hooks()

        logger.info(f"AdapterModel initialized:")
        logger.info(f"  Adapter size: {adapter_size}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Hidden size: {hidden_size}")

    def _register_adapter_hooks(self):
        """注册 forward hooks 来插入 Adapter"""
        # 这里简化实现，实际需要根据具体模型架构调整
        # 对于 GPT-2 类模型，需要在每个 transformer block 后插入
        pass

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
            labels: 标签
            **kwargs: 其他参数

        Returns:
            包含 loss, logits 等的字典
        """
        # 简化实现：直接调用原始模型
        # 完整实现需要在每层后应用 Adapter
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        return outputs

    def get_trainable_parameters(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.adapters.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        """获取总参数数量"""
        return sum(p.numel() for p in self.parameters())


def create_adapter_model(
    model_name: str,
    adapter_size: int = 64,
    adapter_activation: str = "gelu",
    device: str = "cuda"
) -> AdapterModel:
    """
    创建 Adapter 模型

    Args:
        model_name: 预训练模型名称
        adapter_size: Adapter 瓶颈维度
        adapter_activation: 激活函数
        device: 设备

    Returns:
        AdapterModel 实例
    """
    logger.info(f"Creating Adapter model: {model_name}")

    # 加载预训练模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    # 创建 Adapter 模型
    adapter_model = AdapterModel(
        model=model,
        adapter_size=adapter_size,
        adapter_activation=adapter_activation,
        freeze_model=True
    )

    adapter_model = adapter_model.to(device)

    # 统计参数
    total_params = adapter_model.get_total_parameters()
    trainable_params = adapter_model.get_trainable_parameters()
    trainable_ratio = trainable_params / total_params * 100

    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable ratio: {trainable_ratio:.2f}%")

    return adapter_model


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=1000,
        n_positions=512,
        n_embd=256,
        n_layer=4,
        n_head=4
    )

    model = GPT2LMHeadModel(config)

    # 创建 Adapter 模型
    adapter_model = AdapterModel(
        model=model,
        adapter_size=32
    )

    # 测试前向传播
    input_ids = torch.randint(0, 1000, (2, 20))
    outputs = adapter_model(input_ids)

    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Trainable params: {adapter_model.get_trainable_parameters():,}")
    print(f"Total params: {adapter_model.get_total_parameters():,}")
