#!/usr/bin/env python3
"""
Lesson 8: 指令微调与提示工程

将方言翻译模型转变为遵循指令的方言任务助手。

用法：
    # 创建指令数据集
    python scripts/lesson_08_instruction_tuning.py --mode create_dataset \\
        --translation_data data/dialect_parallel_train.json \\
        --classification_data data/accent_train.json \\
        --output_dir data/instruction_dataset

    # 训练指令遵循模型
    python scripts/lesson_08_instruction_tuning.py --mode train \\
        --train_data data/instruction_dataset/train.json \\
        --val_data data/instruction_dataset/val.json \\
        --output_dir checkpoints/instruction_tuned

    # 评估
    python scripts/lesson_08_instruction_tuning.py --mode evaluate \\
        --model_path checkpoints/instruction_tuned/best \\
        --test_data data/instruction_dataset/test.json

    # 推理
    python scripts/lesson_08_instruction_tuning.py --mode inference \\
        --model_path checkpoints/instruction_tuned/best \\
        --instruction "请将以下方言翻译成普通话" \\
        --input "侬好啊"

    # 少样本推理
    python scripts/lesson_08_instruction_tuning.py --mode few_shot \\
        --model_path checkpoints/instruction_tuned/best \\
        --instruction "请将以下方言翻译成普通话" \\
        --examples_file data/few_shot_examples.json \\
        --input "侬好啊"
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.instruction_builder import (
    InstructionBuilder,
    create_instruction_dataset
)
from src.models.instruction_tuned_model import create_instruction_tuned_model
from src.evaluation.instruction_eval import InstructionEvaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dataset(args):
    """创建指令数据集"""
    logger.info("=== 创建指令数据集 ===")

    # 创建指令数据集
    instructions = create_instruction_dataset(
        translation_data_path=args.translation_data,
        classification_data_path=args.classification_data,
        output_path=f"{args.output_dir}/all_instructions.json",
        template_style=args.template_style,
        include_reverse_translation=args.include_reverse
    )

    # 划分数据集
    import random
    random.shuffle(instructions)

    train_size = int(len(instructions) * 0.7)
    val_size = int(len(instructions) * 0.15)

    train_data = instructions[:train_size]
    val_data = instructions[train_size:train_size + val_size]
    test_data = instructions[train_size + val_size:]

    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(output_dir / "val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    with open(output_dir / "test.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Dataset created:")
    logger.info(f"  Train: {len(train_data)} samples")
    logger.info(f"  Val: {len(val_data)} samples")
    logger.info(f"  Test: {len(test_data)} samples")
    logger.info(f"  Saved to: {output_dir}")


def train(args):
    """训练模式"""
    logger.info("=== 指令微调训练模式 ===")

    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 加载 tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    logger.info("Loading training data...")
    with open(args.train_data, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    val_data = None
    if args.val_data:
        with open(args.val_data, 'r', encoding='utf-8') as f:
            val_data = json.load(f)

    logger.info(f"Train samples: {len(train_data)}")
    if val_data:
        logger.info(f"Val samples: {len(val_data)}")

    # 创建模型
    logger.info("Creating instruction-tuned model...")
    model = create_instruction_tuned_model(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device=device
    )

    # 简化训练（这里只是示例，实际应使用完整的训练循环）
    logger.info("Training...")
    logger.info("Note: This is a simplified training example.")
    logger.info("For full training, use the Trainer from transformers or implement a custom training loop.")

    # 保存模型
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir / "final"))

    logger.info(f"Model saved to {output_dir / 'final'}")


def evaluate(args):
    """评估模式"""
    logger.info("=== 指令遵循评估模式 ===")

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
    from src.models.instruction_tuned_model import InstructionTunedModel
    model = InstructionTunedModel.from_pretrained(
        args.model_path,
        device=device
    )
    model.eval()

    # 加载测试数据
    logger.info("Loading test data...")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    logger.info(f"Test samples: {len(test_data)}")

    # 评估
    evaluator = InstructionEvaluator()

    for i, item in enumerate(test_data[:args.max_eval_samples]):
        # 构建输入
        formatted_text = item.get('formatted_text', '')
        if not formatted_text:
            # 手动构建
            formatted_text = f"{item['instruction']}\n输入：{item['input']}\n输出："

        # Tokenize
        inputs = tokenizer(
            formatted_text,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True
        ).to(device)

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=args.max_length
            )

        # 解码
        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取输出部分
        if "输出：" in predicted:
            predicted = predicted.split("输出：")[-1].strip()

        # 评估
        evaluator.evaluate_instruction_following(
            instruction=item['instruction'],
            input_text=item['input'],
            predicted=predicted,
            reference=item['output'],
            task_type=item.get('task_type', 'translation')
        )

        if (i + 1) % 10 == 0:
            logger.info(f"Evaluated {i + 1}/{len(test_data)} samples")

    # 打印摘要
    evaluator.print_summary()

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(str(output_dir / "evaluation_results.json"))


def inference(args):
    """推理模式"""
    logger.info("=== 指令遵循推理模式 ===")

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
    from src.models.instruction_tuned_model import InstructionTunedModel
    model = InstructionTunedModel.from_pretrained(
        args.model_path,
        device=device
    )
    model.eval()

    # 构建输入
    prompt = f"{args.instruction}\n输入：{args.input}\n输出："

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=args.max_length,
        truncation=True
    ).to(device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=args.max_length,
            num_beams=args.num_beams,
            temperature=args.temperature
        )

    # 解码
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取输出部分
    if "输出：" in result:
        result = result.split("输出：")[-1].strip()

    logger.info(f"\n指令: {args.instruction}")
    logger.info(f"输入: {args.input}")
    logger.info(f"输出: {result}")


def few_shot(args):
    """少样本推理模式"""
    logger.info("=== 少样本推理模式 ===")

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
    from src.models.instruction_tuned_model import InstructionTunedModel
    model = InstructionTunedModel.from_pretrained(
        args.model_path,
        device=device
    )
    model.eval()

    # 加载示例
    with open(args.examples_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    logger.info(f"Loaded {len(examples)} examples")

    # 生成
    result = model.generate_with_few_shot(
        tokenizer=tokenizer,
        instruction=args.instruction,
        examples=examples,
        input_text=args.input,
        max_length=args.max_length
    )

    logger.info(f"\n指令: {args.instruction}")
    logger.info(f"示例数: {len(examples)}")
    logger.info(f"输入: {args.input}")
    logger.info(f"输出: {result}")


def main():
    parser = argparse.ArgumentParser(description="指令微调与提示工程")

    # 模式
    parser.add_argument("--mode", type=str, required=True,
                       choices=["create_dataset", "train", "evaluate", "inference", "few_shot"],
                       help="运行模式")

    # 数据路径
    parser.add_argument("--translation_data", type=str,
                       help="翻译数据路径")
    parser.add_argument("--classification_data", type=str,
                       help="分类数据路径")
    parser.add_argument("--train_data", type=str,
                       help="训练数据路径")
    parser.add_argument("--val_data", type=str,
                       help="验证数据路径")
    parser.add_argument("--test_data", type=str,
                       help="测试数据路径")
    parser.add_argument("--examples_file", type=str,
                       help="少样本示例文件")

    # 模型参数
    parser.add_argument("--model_name", type=str,
                       default="uer/gpt2-chinese-cluecorpussmall",
                       help="预训练模型名称")
    parser.add_argument("--model_path", type=str,
                       help="模型路径（用于评估/推理）")

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")

    # 指令参数
    parser.add_argument("--template_style", type=str, default="alpaca",
                       choices=["alpaca", "vicuna", "simple"],
                       help="指令模板风格")
    parser.add_argument("--include_reverse", action="store_true",
                       help="包含反向翻译")

    # 推理参数
    parser.add_argument("--instruction", type=str,
                       help="指令")
    parser.add_argument("--input", type=str,
                       help="输入文本")
    parser.add_argument("--max_length", type=int, default=256,
                       help="最大序列长度")
    parser.add_argument("--num_beams", type=int, default=4,
                       help="Beam search 数量")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="温度参数")

    # 评估参数
    parser.add_argument("--max_eval_samples", type=int, default=100,
                       help="最大评估样本数")

    # 输出
    parser.add_argument("--output_dir", type=str,
                       default="results/instruction_tuning",
                       help="输出目录")

    args = parser.parse_args()

    # 执行对应模式
    if args.mode == "create_dataset":
        create_dataset(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "inference":
        inference(args)
    elif args.mode == "few_shot":
        few_shot(args)


if __name__ == "__main__":
    main()
