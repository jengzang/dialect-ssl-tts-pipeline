#!/usr/bin/env python3
"""
偏好数据集构建器

用于 RLHF 训练的人类偏好数据收集和处理
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class PreferenceDataset:
    """
    偏好数据集

    存储 (prompt, chosen_response, rejected_response) 三元组
    """

    def __init__(self):
        self.data = []

    def add_preference(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        添加偏好样本

        Args:
            prompt: 输入提示
            chosen: 被选择的响应（更好）
            rejected: 被拒绝的响应（更差）
            metadata: 元数据
        """
        self.data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": metadata or {}
        })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def save(self, path: str):
        """保存数据集"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.data)} preferences to {path}")

    @classmethod
    def load(cls, path: str) -> "PreferenceDataset":
        """加载数据集"""
        dataset = cls()
        with open(path, "r", encoding="utf-8") as f:
            dataset.data = json.load(f)
        logger.info(f"Loaded {len(dataset.data)} preferences from {path}")
        return dataset


class PreferenceCollector:
    """
    偏好数据收集器

    生成模型响应并收集人类偏好
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda"
    ):
        """
        Args:
            model_name: 模型名称
            device: 设备
        """
        self.model_name = model_name
        self.device = device

        logger.info(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

        self.model = self.model.to(device)
        self.model.eval()

        logger.info("Model loaded successfully")

    def generate_responses(
        self,
        prompt: str,
        num_responses: int = 2,
        max_length: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9
    ) -> List[str]:
        """
        生成多个响应

        Args:
            prompt: 输入提示
            num_responses: 响应数量
            max_length: 最大长度
            temperature: 温度
            top_p: Top-p 采样

        Returns:
            响应列表
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        responses = []

        with torch.no_grad():
            for _ in range(num_responses):
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )

                responses.append(response.strip())

        return responses

    def collect_preferences_interactive(
        self,
        prompts: List[str],
        output_path: str
    ) -> PreferenceDataset:
        """
        交互式收集偏好

        Args:
            prompts: 提示列表
            output_path: 输出路径

        Returns:
            偏好数据集
        """
        dataset = PreferenceDataset()

        logger.info(f"Collecting preferences for {len(prompts)} prompts")

        for i, prompt in enumerate(prompts):
            logger.info(f"\n{'='*60}")
            logger.info(f"Prompt {i+1}/{len(prompts)}")
            logger.info(f"{'='*60}")
            logger.info(f"Prompt: {prompt}")

            # 生成两个响应
            responses = self.generate_responses(prompt, num_responses=2)

            logger.info(f"\nResponse A: {responses[0]}")
            logger.info(f"\nResponse B: {responses[1]}")

            # 这里应该让人类选择，但我们简化为自动选择
            # 实际应用中需要人工标注界面
            logger.info("\n[Simulated human preference: A is better]")

            dataset.add_preference(
                prompt=prompt,
                chosen=responses[0],
                rejected=responses[1],
                metadata={"method": "simulated"}
            )

        # 保存数据集
        dataset.save(output_path)

        return dataset


def simulate_preference_dataset(
    prompts: List[str],
    output_path: str,
    quality_criteria: Optional[Dict[str, float]] = None
) -> PreferenceDataset:
    """
    模拟偏好数据集

    根据启发式规则生成偏好数据（用于测试）

    Args:
        prompts: 提示列表
        output_path: 输出路径
        quality_criteria: 质量标准

    Returns:
        偏好数据集
    """
    logger.info(f"Simulating preference dataset for {len(prompts)} prompts")

    dataset = PreferenceDataset()

    for prompt in prompts:
        # 生成两个响应（模拟）
        # 实际应该使用模型生成
        response_a = f"这是对'{prompt}'的详细回答，包含了相关信息和解释。"
        response_b = f"回答：{prompt}的答案。"

        # 启发式规则：更长、更详细的响应通常更好
        if len(response_a) > len(response_b):
            chosen = response_a
            rejected = response_b
        else:
            chosen = response_b
            rejected = response_a

        # 随机翻转一些偏好（模拟噪声）
        if random.random() < 0.1:
            chosen, rejected = rejected, chosen

        dataset.add_preference(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata={"method": "simulated", "rule": "length"}
        )

    # 保存数据集
    dataset.save(output_path)

    return dataset


def create_dialect_translation_preferences(
    translation_pairs: List[Dict[str, str]],
    output_path: str
) -> PreferenceDataset:
    """
    创建方言翻译偏好数据集

    Args:
        translation_pairs: 翻译对列表 [{"dialect": "...", "mandarin": "..."}]
        output_path: 输出路径

    Returns:
        偏好数据集
    """
    logger.info(f"Creating dialect translation preferences for {len(translation_pairs)} pairs")

    dataset = PreferenceDataset()

    for pair in translation_pairs:
        dialect = pair["dialect"]
        mandarin = pair["mandarin"]

        # 创建提示
        prompt = f"请将以下方言翻译成普通话：{dialect}"

        # 好的翻译（正确）
        chosen = mandarin

        # 差的翻译（添加错误）
        # 简化：直接复制或添加噪声
        rejected_variants = [
            dialect,  # 没有翻译
            f"{mandarin}吧",  # 添加不必要的语气词
            mandarin[:len(mandarin)//2],  # 不完整
        ]

        rejected = random.choice(rejected_variants)

        dataset.add_preference(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata={"task": "dialect_translation"}
        )

    # 保存数据集
    dataset.save(output_path)

    return dataset


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    # 测试 1: 模拟偏好数据集
    prompts = [
        "什么是机器学习？",
        "如何学习 Python？",
        "深度学习的应用有哪些？"
    ]

    dataset = simulate_preference_dataset(
        prompts=prompts,
        output_path="test_preferences.json"
    )

    print(f"Created dataset with {len(dataset)} preferences")
    print(f"Sample: {dataset[0]}")

    # 测试 2: 方言翻译偏好
    translation_pairs = [
        {"dialect": "你食咗饭未？", "mandarin": "你吃饭了吗？"},
        {"dialect": "我哋去边度？", "mandarin": "我们去哪里？"}
    ]

    dialect_dataset = create_dialect_translation_preferences(
        translation_pairs=translation_pairs,
        output_path="test_dialect_preferences.json"
    )

    print(f"Created dialect dataset with {len(dialect_dataset)} preferences")
