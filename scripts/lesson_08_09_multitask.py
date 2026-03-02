#!/usr/bin/env python3
"""
Lesson 8+9: 多任务学习 - 方言翻译 + 口音分类

结合 Lesson 8（方言翻译）和 Lesson 9（口音识别）的多任务学习。

用法：
    # 训练
    python scripts/lesson_08_09_multitask.py --mode train \\
        --translation_data data/dialect_parallel_train.json \\
        --classification_data data/accent_train.json \\
        --output_dir checkpoints/multitask

    # 评估
    python scripts/lesson_08_09_multitask.py --mode evaluate \\
        --model_path checkpoints/multitask/best \\
        --translation_data data/dialect_parallel_test.json \\
        --classification_data data/accent_test.json

    # 推理
    python scripts/lesson_08_09_multitask.py --mode inference \\
        --model_path checkpoints/multitask/best \\
        --task translation \\
        --input "侬好啊"
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.multitask_dataset import create_multitask_dataloaders
from src.models.multitask_dialect_model import create_multitask_model
from src.training.multitask_trainer import MultitaskTrainer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train(args):
    """训练模式"""
    logger.info("=== 多任务训练模式 ===")

    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 加载 tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建数据加载器
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_multitask_dataloaders(
        translation_train_path=args.translation_data,
        translation_val_path=args.translation_val_data,
        classification_train_path=args.classification_data,
        classification_val_path=args.classification_val_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        task_sampling=args.task_sampling
    )

    # 创建模型
    logger.info("Creating multitask model...")
    model = create_multitask_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_task_specific_lora=args.use_task_specific_lora,
        device=device
    )

    # 损失权重
    loss_weights = {
        'translation': args.translation_weight,
        'classification': args.classification_weight
    }

    # 创建训练器
    logger.info("Creating trainer...")
    trainer = MultitaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        loss_weights=loss_weights,
        gradient_normalization=args.gradient_normalization,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        output_dir=args.output_dir
    )

    # 训练
    logger.info(f"Starting training for {args.num_epochs} epochs...")
    trainer.train(
        num_epochs=args.num_epochs,
        save_every=args.save_every
    )

    logger.info("Training completed!")


def evaluate(args):
    """评估模式"""
    logger.info("=== 多任务评估模式 ===")

    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 加载 tokenizer
    logger.info(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建数据加载器
    logger.info("Creating dataloaders...")
    _, test_loader = create_multitask_dataloaders(
        translation_train_path=args.translation_data,
        translation_val_path=None,
        classification_train_path=args.classification_data,
        classification_val_path=None,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        task_sampling="proportional"
    )

    # 加载模型
    logger.info(f"Loading model from: {args.model_path}")
    from src.models.multitask_dialect_model import MultitaskDialectModel
    model = MultitaskDialectModel.from_pretrained(
        args.model_path,
        num_classes=args.num_classes,
        device=device
    )

    # 创建训练器（仅用于评估）
    trainer = MultitaskTrainer(
        model=model,
        train_loader=test_loader,
        val_loader=test_loader,
        device=device,
        output_dir=args.output_dir
    )

    # 评估
    logger.info("Evaluating...")
    metrics = trainer.evaluate()

    # 打印结果
    logger.info("=== Evaluation Results ===")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")


def inference(args):
    """推理模式"""
    logger.info("=== 多任务推理模式 ===")

    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 加载 tokenizer
    logger.info(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    logger.info(f"Loading model from: {args.model_path}")
    from src.models.multitask_dialect_model import MultitaskDialectModel
    model = MultitaskDialectModel.from_pretrained(
        args.model_path,
        num_classes=args.num_classes,
        device=device
    )
    model.eval()

    # 推理
    if args.task == "translation":
        # 翻译任务
        input_text = f"[翻译任务] 请将以下方言翻译成普通话：\\n方言：{args.input}\\n普通话："
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=args.max_length,
                num_beams=4,
                early_stopping=True
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取翻译结果
        if "普通话：" in result:
            result = result.split("普通话：")[-1].strip()

        logger.info(f"输入: {args.input}")
        logger.info(f"翻译: {result}")

    elif args.task == "classification":
        # 分类任务
        input_text = f"[分类任务] 请识别以下文本的方言类型：\\n文本：{args.input}\\n方言类型："
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True,
            padding=True
        ).to(device)

        task_id = torch.tensor([1], device=device)  # 1 = classification

        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                task_id=task_id
            )

        predictions = torch.argmax(outputs['logits'], dim=-1)
        label_map = {0: 'shanghai', 1: 'cantonese', 2: 'mandarin', 3: 'other'}
        predicted_label = label_map.get(predictions.item(), 'unknown')

        logger.info(f"输入: {args.input}")
        logger.info(f"预测方言: {predicted_label}")

    else:
        logger.error(f"Unknown task: {args.task}")


def main():
    parser = argparse.ArgumentParser(description="多任务学习：方言翻译 + 口音分类")

    # 模式
    parser.add_argument("--mode", type=str, required=True,
                       choices=["train", "evaluate", "inference"],
                       help="运行模式")

    # 数据路径
    parser.add_argument("--translation_data", type=str,
                       help="翻译数据路径")
    parser.add_argument("--translation_val_data", type=str,
                       help="翻译验证数据路径")
    parser.add_argument("--classification_data", type=str,
                       help="分类数据路径")
    parser.add_argument("--classification_val_data", type=str,
                       help="分类验证数据路径")

    # 模型参数
    parser.add_argument("--model_name", type=str,
                       default="uer/gpt2-chinese-cluecorpussmall",
                       help="预训练模型名称")
    parser.add_argument("--model_path", type=str,
                       help="模型路径（用于评估/推理）")
    parser.add_argument("--num_classes", type=int, default=4,
                       help="分类任务的类别数")

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    parser.add_argument("--use_task_specific_lora", action="store_true",
                       help="使用任务特定的 LoRA")

    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="批次大小")
    parser.add_argument("--max_length", type=int, default=512,
                       help="最大序列长度")
    parser.add_argument("--save_every", type=int, default=1,
                       help="保存检查点的频率")

    # 多任务参数
    parser.add_argument("--task_sampling", type=str, default="balanced",
                       choices=["balanced", "proportional", "translation_heavy", "classification_heavy"],
                       help="任务采样策略")
    parser.add_argument("--translation_weight", type=float, default=1.0,
                       help="翻译任务损失权重")
    parser.add_argument("--classification_weight", type=float, default=1.0,
                       help="分类任务损失权重")
    parser.add_argument("--gradient_normalization", action="store_true",
                       help="使用梯度归一化")

    # 推理参数
    parser.add_argument("--task", type=str,
                       choices=["translation", "classification"],
                       help="推理任务类型")
    parser.add_argument("--input", type=str,
                       help="推理输入文本")

    # 输出
    parser.add_argument("--output_dir", type=str,
                       default="checkpoints/multitask",
                       help="输出目录")

    # WandB
    parser.add_argument("--use_wandb", action="store_true",
                       help="使用 WandB 日志")
    parser.add_argument("--wandb_project", type=str,
                       default="multitask-dialect",
                       help="WandB 项目名")

    args = parser.parse_args()

    # 执行对应模式
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "inference":
        inference(args)


if __name__ == "__main__":
    main()
