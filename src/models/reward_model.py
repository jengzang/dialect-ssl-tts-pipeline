#!/usr/bin/env python3
"""
奖励模型

用于 RLHF 的奖励模型，预测人类偏好
"""

import logging
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel
)

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    奖励模型

    基于预训练模型，添加一个标量输出头来预测奖励分数
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        hidden_size: Optional[int] = None
    ):
        """
        Args:
            base_model: 预训练基础模型
            hidden_size: 隐藏层大小（如果为 None，从模型配置获取）
        """
        super().__init__()

        self.base_model = base_model

        # 获取隐藏层大小
        if hidden_size is None:
            hidden_size = base_model.config.hidden_size

        # 奖励头：将隐藏状态映射到标量奖励
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

        logger.info(f"RewardModel initialized with hidden_size={hidden_size}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码
            **kwargs: 其他参数

        Returns:
            奖励分数 [batch_size]
        """
        # 获取基础模型的输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        # 获取最后一层的隐藏状态
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]

        # 使用最后一个 token 的隐藏状态
        # 或者使用平均池化
        if attention_mask is not None:
            # 找到每个序列的最后一个有效 token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            last_hidden = hidden_states[torch.arange(batch_size), sequence_lengths]
        else:
            last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]

        # 计算奖励
        reward = self.reward_head(last_hidden).squeeze(-1)  # [batch_size]

        return reward


def create_reward_model(
    model_name: str,
    device: str = "cuda"
) -> Tuple[RewardModel, AutoTokenizer]:
    """
    创建奖励模型

    Args:
        model_name: 预训练模型名称
        device: 设备

    Returns:
        (奖励模型, tokenizer)
    """
    logger.info(f"Creating reward model from: {model_name}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型
    base_model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    # 创建奖励模型
    reward_model = RewardModel(base_model)
    reward_model = reward_model.to(device)

    # 统计参数
    total_params = sum(p.numel() for p in reward_model.parameters())
    trainable_params = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)

    logger.info(f"Reward model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    return reward_model, tokenizer


class RewardModelTrainer:
    """
    奖励模型训练器

    使用偏好数据训练奖励模型
    """

    def __init__(
        self,
        reward_model: RewardModel,
        tokenizer: AutoTokenizer,
        device: str = "cuda"
    ):
        """
        Args:
            reward_model: 奖励模型
            tokenizer: Tokenizer
            device: 设备
        """
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device

        logger.info("RewardModelTrainer initialized")

    def compute_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        计算偏好损失

        使用 Bradley-Terry 模型：
        P(chosen > rejected) = sigmoid(reward_chosen - reward_rejected)

        损失：-log(sigmoid(reward_chosen - reward_rejected))

        Args:
            chosen_rewards: 被选择响应的奖励 [batch_size]
            rejected_rewards: 被拒绝响应的奖励 [batch_size]

        Returns:
            损失值
        """
        # Bradley-Terry 损失
        # 等价于：-log_sigmoid(chosen_rewards - rejected_rewards)
        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()

        return loss

    def train_step(
        self,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str],
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        单步训练

        Args:
            prompts: 提示列表
            chosen_responses: 被选择的响应列表
            rejected_responses: 被拒绝的响应列表
            optimizer: 优化器

        Returns:
            损失值
        """
        self.reward_model.train()

        # 构建输入
        chosen_texts = [p + " " + r for p, r in zip(prompts, chosen_responses)]
        rejected_texts = [p + " " + r for p, r in zip(prompts, rejected_responses)]

        # Tokenize
        chosen_inputs = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        rejected_inputs = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # 前向传播
        chosen_rewards = self.reward_model(**chosen_inputs)
        rejected_rewards = self.reward_model(**rejected_inputs)

        # 计算损失
        loss = self.compute_loss(chosen_rewards, rejected_rewards)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def evaluate(
        self,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str]
    ) -> Dict[str, float]:
        """
        评估模型

        Args:
            prompts: 提示列表
            chosen_responses: 被选择的响应列表
            rejected_responses: 被拒绝的响应列表

        Returns:
            评估指标
        """
        self.reward_model.eval()

        # 构建输入
        chosen_texts = [p + " " + r for p, r in zip(prompts, chosen_responses)]
        rejected_texts = [p + " " + r for p, r in zip(prompts, rejected_responses)]

        # Tokenize
        chosen_inputs = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        rejected_inputs = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # 前向传播
        with torch.no_grad():
            chosen_rewards = self.reward_model(**chosen_inputs)
            rejected_rewards = self.reward_model(**rejected_inputs)

            # 计算损失
            loss = self.compute_loss(chosen_rewards, rejected_rewards)

            # 计算准确率（chosen 的奖励是否高于 rejected）
            correct = (chosen_rewards > rejected_rewards).float().sum()
            accuracy = correct / len(prompts)

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "avg_chosen_reward": chosen_rewards.mean().item(),
            "avg_rejected_reward": rejected_rewards.mean().item()
        }


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    from transformers import GPT2Config, GPT2Model

    # 创建小模型测试
    config = GPT2Config(
        vocab_size=1000,
        n_positions=512,
        n_embd=256,
        n_layer=4,
        n_head=4
    )

    base_model = GPT2Model(config)

    # 创建奖励模型
    reward_model = RewardModel(base_model)

    # 测试前向传播
    input_ids = torch.randint(0, 1000, (2, 20))
    rewards = reward_model(input_ids)

    print(f"Rewards shape: {rewards.shape}")
    print(f"Rewards: {rewards}")
    print(f"Model params: {sum(p.numel() for p in reward_model.parameters()):,}")
