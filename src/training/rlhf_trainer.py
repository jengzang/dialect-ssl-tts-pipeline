#!/usr/bin/env python3
"""
RLHF 训练器

实现基于人类反馈的强化学习（简化版本）
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import json

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

from src.models.reward_model import RewardModel

logger = logging.getLogger(__name__)


class RLHFTrainer:
    """
    RLHF 训练器

    简化版本：使用奖励模型指导策略模型的训练
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: RewardModel,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        kl_coef: float = 0.1
    ):
        """
        Args:
            policy_model: 策略模型（要训练的模型）
            reward_model: 奖励模型（已训练好）
            tokenizer: Tokenizer
            device: 设备
            kl_coef: KL 散度系数（防止偏离太远）
        """
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device
        self.kl_coef = kl_coef

        # 冻结奖励模型
        for param in self.reward_model.parameters():
            param.requires_grad = False

        self.reward_model.eval()

        # 保存参考模型（用于计算 KL 散度）
        self.ref_model = self._create_reference_model()

        logger.info("RLHFTrainer initialized")
        logger.info(f"  KL coefficient: {kl_coef}")

    def _create_reference_model(self) -> nn.Module:
        """创建参考模型（策略模型的副本）"""
        # 简化：直接使用策略模型
        # 实际应该创建一个冻结的副本
        return self.policy_model

    def generate_responses(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 0.8
    ) -> List[str]:
        """
        生成响应

        Args:
            prompts: 提示列表
            max_length: 最大长度
            temperature: 温度

        Returns:
            响应列表
        """
        self.policy_model.eval()

        responses = []

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            responses.append(response.strip())

        return responses

    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> torch.Tensor:
        """
        计算奖励

        Args:
            prompts: 提示列表
            responses: 响应列表

        Returns:
            奖励张量 [batch_size]
        """
        # 构建完整文本
        texts = [p + " " + r for p, r in zip(prompts, responses)]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # 计算奖励
        with torch.no_grad():
            rewards = self.reward_model(**inputs)

        return rewards

    def train_step_simple(
        self,
        prompts: List[str],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        简化的训练步骤

        使用奖励加权的监督学习（不是完整的 PPO）

        Args:
            prompts: 提示列表
            optimizer: 优化器

        Returns:
            训练统计
        """
        self.policy_model.train()

        # 生成响应
        responses = self.generate_responses(prompts)

        # 计算奖励
        rewards = self.compute_rewards(prompts, responses)

        # 构建训练数据
        texts = [p + " " + r for p, r in zip(prompts, responses)]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # 前向传播
        outputs = self.policy_model(**inputs, labels=inputs.input_ids)

        # 奖励加权损失
        # 简化：使用奖励作为权重
        loss = outputs.loss

        # 添加奖励项（最大化奖励）
        reward_loss = -rewards.mean()

        total_loss = loss + self.kl_coef * reward_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
            "reward_loss": reward_loss.item(),
            "total_loss": total_loss.item(),
            "avg_reward": rewards.mean().item()
        }

    def train(
        self,
        train_prompts: List[str],
        val_prompts: Optional[List[str]] = None,
        epochs: int = 3,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        训练模型

        Args:
            train_prompts: 训练提示列表
            val_prompts: 验证提示列表
            epochs: 训练轮数
            learning_rate: 学习率
            batch_size: 批次大小
            output_dir: 输出目录

        Returns:
            训练统计
        """
        logger.info("Starting RLHF training...")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Batch size: {batch_size}")

        # 优化器
        optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=learning_rate
        )

        # 训练循环
        start_time = time.time()
        training_stats = {
            "epochs": epochs,
            "training_time": 0,
            "final_avg_reward": 0
        }

        for epoch in range(epochs):
            epoch_stats = []

            # 分批训练
            for i in range(0, len(train_prompts), batch_size):
                batch_prompts = train_prompts[i:i+batch_size]

                stats = self.train_step_simple(batch_prompts, optimizer)
                epoch_stats.append(stats)

            # 计算平均统计
            avg_loss = sum(s["loss"] for s in epoch_stats) / len(epoch_stats)
            avg_reward = sum(s["avg_reward"] for s in epoch_stats) / len(epoch_stats)

            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

            # 验证
            if val_prompts:
                val_reward = self.evaluate(val_prompts)
                logger.info(f"Validation Avg Reward: {val_reward:.4f}")

        # 训练结束
        training_time = time.time() - start_time
        training_stats["training_time"] = training_time
        training_stats["final_avg_reward"] = avg_reward

        logger.info(f"Training completed in {training_time:.2f} seconds")

        # 保存模型
        if output_dir:
            self.save_model(output_dir)
            with open(Path(output_dir) / "training_stats.json", "w") as f:
                json.dump(training_stats, f, indent=2)

        return training_stats

    def evaluate(self, prompts: List[str]) -> float:
        """
        评估模型

        Args:
            prompts: 提示列表

        Returns:
            平均奖励
        """
        self.policy_model.eval()

        # 生成响应
        responses = self.generate_responses(prompts)

        # 计算奖励
        rewards = self.compute_rewards(prompts, responses)

        return rewards.mean().item()

    def save_model(self, output_dir: str):
        """保存模型"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存策略模型
        torch.save(self.policy_model.state_dict(), output_path / "policy_model.pt")

        # 保存 tokenizer
        self.tokenizer.save_pretrained(str(output_path))

        logger.info(f"Model saved to {output_dir}")


def create_rlhf_trainer(
    policy_model_name: str,
    reward_model: RewardModel,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
    kl_coef: float = 0.1
) -> RLHFTrainer:
    """
    创建 RLHF 训练器

    Args:
        policy_model_name: 策略模型名称
        reward_model: 奖励模型
        tokenizer: Tokenizer
        device: 设备
        kl_coef: KL 系数

    Returns:
        RLHF 训练器
    """
    logger.info(f"Creating RLHF trainer with policy model: {policy_model_name}")

    # 加载策略模型
    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    policy_model = policy_model.to(device)

    # 创建训练器
    trainer = RLHFTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=device,
        kl_coef=kl_coef
    )

    return trainer


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    print("RLHF Trainer module loaded successfully")
    print("Note: Full testing requires trained reward model")
