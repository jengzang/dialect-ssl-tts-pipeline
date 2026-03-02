#!/usr/bin/env python3
"""
RLHF 训练 CLI

命令行工具用于 RLHF 训练流程
"""

import argparse
import logging
from pathlib import Path
import sys
import json

import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.preference_dataset import (
    PreferenceDataset,
    simulate_preference_dataset,
    create_dialect_translation_preferences
)
from src.models.reward_model import (
    create_reward_model,
    RewardModelTrainer
)
from src.training.rlhf_trainer import create_rlhf_trainer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_preference_data(args):
    """创建偏好数据"""
    logger.info("="*60)
    logger.info("Creating Preference Dataset")
    logger.info("="*60)

    if args.data_type == "simulated":
        # 模拟数据
        prompts = [
            "什么是机器学习？",
            "如何学习 Python？",
            "深度学习的应用有哪些？",
            "什么是神经网络？",
            "如何训练一个模型？"
        ] * 10  # 重复以增加数据量

        dataset = simulate_preference_dataset(
            prompts=prompts,
            output_path=args.output_path
        )

    elif args.data_type == "dialect":
        # 方言翻译数据
        # 这里应该从文件加载，简化为硬编码
        translation_pairs = [
            {"dialect": "你食咗饭未？", "mandarin": "你吃饭了吗？"},
            {"dialect": "我哋去边度？", "mandarin": "我们去哪里？"},
            {"dialect": "呢个几多钱？", "mandarin": "这个多少钱？"},
        ] * 10

        dataset = create_dialect_translation_preferences(
            translation_pairs=translation_pairs,
            output_path=args.output_path
        )

    else:
        raise ValueError(f"Unknown data type: {args.data_type}")

    logger.info(f"Created dataset with {len(dataset)} preferences")
    logger.info(f"Saved to {args.output_path}")


def train_reward_model(args):
    """训练奖励模型"""
    logger.info("="*60)
    logger.info("Training Reward Model")
    logger.info("="*60)

    # 加载偏好数据
    dataset = PreferenceDataset.load(args.preference_data)

    logger.info(f"Loaded {len(dataset)} preferences")

    # 创建奖励模型
    reward_model, tokenizer = create_reward_model(
        model_name=args.model_name,
        device=args.device
    )

    # 创建训练器
    trainer = RewardModelTrainer(
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=args.device
    )

    # 准备数据
    prompts = [d["prompt"] for d in dataset.data]
    chosen = [d["chosen"] for d in dataset.data]
    rejected = [d["rejected"] for d in dataset.data]

    # 划分训练/验证集
    split_idx = int(len(prompts) * 0.8)
    train_prompts = prompts[:split_idx]
    train_chosen = chosen[:split_idx]
    train_rejected = rejected[:split_idx]

    val_prompts = prompts[split_idx:]
    val_chosen = chosen[split_idx:]
    val_rejected = rejected[split_idx:]

    # 优化器
    optimizer = torch.optim.AdamW(
        reward_model.parameters(),
        lr=args.learning_rate
    )

    # 训练循环
    logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        # 训练
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_prompts), args.batch_size):
            batch_prompts = train_prompts[i:i+args.batch_size]
            batch_chosen = train_chosen[i:i+args.batch_size]
            batch_rejected = train_rejected[i:i+args.batch_size]

            loss = trainer.train_step(
                prompts=batch_prompts,
                chosen_responses=batch_chosen,
                rejected_responses=batch_rejected,
                optimizer=optimizer
            )

            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches

        # 验证
        val_metrics = trainer.evaluate(
            prompts=val_prompts,
            chosen_responses=val_chosen,
            rejected_responses=val_rejected
        )

        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"  Train Loss: {avg_loss:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")

    # 保存模型
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(reward_model.state_dict(), output_dir / "reward_model.pt")
    tokenizer.save_pretrained(str(output_dir))

    # 保存统计
    stats = {
        "final_val_loss": val_metrics["loss"],
        "final_val_accuracy": val_metrics["accuracy"],
        "epochs": args.epochs
    }

    with open(output_dir / "reward_model_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Reward model saved to {args.output_dir}")


def train_rlhf(args):
    """RLHF 训练"""
    logger.info("="*60)
    logger.info("RLHF Training")
    logger.info("="*60)

    # 加载奖励模型
    logger.info(f"Loading reward model from {args.reward_model_path}")

    reward_model, tokenizer = create_reward_model(
        model_name=args.base_model,
        device=args.device
    )

    reward_model.load_state_dict(
        torch.load(Path(args.reward_model_path) / "reward_model.pt")
    )

    # 创建 RLHF 训练器
    rlhf_trainer = create_rlhf_trainer(
        policy_model_name=args.policy_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=args.device,
        kl_coef=args.kl_coef
    )

    # 准备训练数据（提示）
    # 这里应该从文件加载
    train_prompts = [
        "什么是机器学习？",
        "如何学习 Python？",
        "深度学习的应用有哪些？",
    ] * 10

    val_prompts = [
        "什么是神经网络？",
        "如何训练一个模型？",
    ] * 5

    # 训练
    stats = rlhf_trainer.train(
        train_prompts=train_prompts,
        val_prompts=val_prompts,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )

    logger.info("RLHF training completed")
    logger.info(f"Final avg reward: {stats['final_avg_reward']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="RLHF 训练工具")

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # 创建偏好数据
    create_parser = subparsers.add_parser("create_data", help="创建偏好数据")
    create_parser.add_argument(
        "--data_type",
        type=str,
        choices=["simulated", "dialect"],
        default="simulated",
        help="数据类型"
    )
    create_parser.add_argument(
        "--output_path",
        type=str,
        default="data/preferences.json",
        help="输出路径"
    )

    # 训练奖励模型
    reward_parser = subparsers.add_parser("train_reward", help="训练奖励模型")
    reward_parser.add_argument(
        "--preference_data",
        type=str,
        required=True,
        help="偏好数据路径"
    )
    reward_parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="基础模型名称"
    )
    reward_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数"
    )
    reward_parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="批次大小"
    )
    reward_parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="学习率"
    )
    reward_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备"
    )
    reward_parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/reward_model",
        help="输出目录"
    )

    # RLHF 训练
    rlhf_parser = subparsers.add_parser("train_rlhf", help="RLHF 训练")
    rlhf_parser.add_argument(
        "--reward_model_path",
        type=str,
        required=True,
        help="奖励模型路径"
    )
    rlhf_parser.add_argument(
        "--base_model",
        type=str,
        default="gpt2",
        help="基础模型（用于加载奖励模型架构）"
    )
    rlhf_parser.add_argument(
        "--policy_model",
        type=str,
        default="gpt2",
        help="策略模型名称"
    )
    rlhf_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练轮数"
    )
    rlhf_parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="批次大小"
    )
    rlhf_parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="学习率"
    )
    rlhf_parser.add_argument(
        "--kl_coef",
        type=float,
        default=0.1,
        help="KL 散度系数"
    )
    rlhf_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备"
    )
    rlhf_parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/rlhf_model",
        help="输出目录"
    )

    args = parser.parse_args()

    if args.command == "create_data":
        create_preference_data(args)
    elif args.command == "train_reward":
        train_reward_model(args)
    elif args.command == "train_rlhf":
        train_rlhf(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
