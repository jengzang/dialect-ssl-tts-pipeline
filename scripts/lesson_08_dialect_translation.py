"""
Lesson 8: 方言平行语料翻译

使用 LoRA 微调大语言模型，实现方言到普通话的翻译任务。

功能：
- 训练翻译模型
- 推理翻译
- 批量翻译

数据格式（JSON）：
[
    {
        "dialect": "我今日好开心啊",
        "mandarin": "我今天很开心"
    },
    ...
]

使用示例：
    # 训练模型
    python scripts/lesson_08_dialect_translation.py --mode train \\
        --model_name Qwen/Qwen-7B-Chat \\
        --train_data data/dialect_parallel.json \\
        --output_dir checkpoints/dialect_translator \\
        --epochs 3

    # 推理翻译
    python scripts/lesson_08_dialect_translation.py --mode inference \\
        --model_path checkpoints/dialect_translator/best \\
        --dialect_text "我今日好开心啊"

    # 批量翻译
    python scripts/lesson_08_dialect_translation.py --mode batch \\
        --model_path checkpoints/dialect_translator/best \\
        --input_file data/test_dialect.txt \\
        --output_file results/translations.txt
"""

import argparse
import logging
from pathlib import Path
import sys
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.models.dialect_translator import DialectTranslator
from src.training.dialect_translation_trainer import DialectTranslationTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Lesson 8: Dialect Translation with LoRA'
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'inference', 'batch'],
        help='运行模式'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/Qwen-7B-Chat',
        help='基础模型名称'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        help='LoRA 模型路径（用于推理）'
    )

    parser.add_argument(
        '--train_data',
        type=str,
        help='训练数据路径（JSON 格式）'
    )

    parser.add_argument(
        '--val_data',
        type=str,
        help='验证数据路径（JSON 格式）'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints/dialect_translator',
        help='输出目录'
    )

    parser.add_argument(
        '--dialect_text',
        type=str,
        help='方言文本（用于推理）'
    )

    parser.add_argument(
        '--input_file',
        type=str,
        help='输入文件（用于批量翻译）'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        help='输出文件（用于批量翻译）'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='训练轮数'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='批次大小'
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-4,
        help='学习率'
    )

    parser.add_argument(
        '--lora_r',
        type=int,
        default=8,
        help='LoRA rank'
    )

    parser.add_argument(
        '--lora_alpha',
        type=int,
        default=32,
        help='LoRA alpha'
    )

    parser.add_argument(
        '--quantization',
        action='store_true',
        help='使用 4-bit 量化'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径'
    )

    return parser.parse_args()


def train_mode(args, logger):
    """训练模式"""
    logger.info("=== Training Mode ===")

    if not args.train_data:
        logger.error("--train_data is required")
        return False

    config = {
        'model_name': args.model_name,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'max_length': 512
    }

    # 初始化训练器
    trainer = DialectTranslationTrainer(config)

    # 准备模型
    trainer.prepare_model(quantization=args.quantization)

    # 训练
    try:
        trainer.train(args.train_data, args.val_data)
        logger.info("[OK] Training completed")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}")
        return False


def inference_mode(args, logger):
    """推理模式"""
    logger.info("=== Inference Mode ===")

    if not args.model_path or not args.dialect_text:
        logger.error("--model_path and --dialect_text are required")
        return False

    config = {
        'model_name': args.model_name,
        'max_length': 512
    }

    # 初始化翻译器
    translator = DialectTranslator(config)

    # 加载模型
    try:
        translator.load_lora_model(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

    # 翻译
    try:
        translation = translator.translate(args.dialect_text)
        logger.info(f"方言: {args.dialect_text}")
        logger.info(f"普通话: {translation}")
        return True
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return False


def batch_mode(args, logger):
    """批量翻译模式"""
    logger.info("=== Batch Translation Mode ===")

    if not all([args.model_path, args.input_file, args.output_file]):
        logger.error("--model_path, --input_file, --output_file are required")
        return False

    config = {
        'model_name': args.model_name,
        'max_length': 512
    }

    # 初始化翻译器
    translator = DialectTranslator(config)

    # 加载模型
    try:
        translator.load_lora_model(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

    # 读取输入文件
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            dialect_texts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return False

    logger.info(f"Loaded {len(dialect_texts)} texts")

    # 批量翻译
    try:
        translations = translator.batch_translate(
            dialect_texts,
            batch_size=args.batch_size
        )

        # 保存结果
        output_path = Path(args.output_file)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for dialect, mandarin in zip(dialect_texts, translations):
                f.write(f"方言: {dialect}\n")
                f.write(f"普通话: {mandarin}\n")
                f.write("\n")

        logger.info(f"[OK] Translations saved to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Batch translation failed: {e}")
        return False


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    log_config = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_dir': 'logs'
    }
    logger = setup_logger('lesson_08', log_config)

    logger.info("Starting Lesson 8: Dialect Translation with LoRA")
    logger.info(f"Mode: {args.mode}")

    # 根据模式执行
    if args.mode == 'train':
        success = train_mode(args, logger)
    elif args.mode == 'inference':
        success = inference_mode(args, logger)
    elif args.mode == 'batch':
        success = batch_mode(args, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        success = False

    if success:
        logger.info("=== Completed Successfully ===")
        sys.exit(0)
    else:
        logger.error("=== Failed ===")
        sys.exit(1)


if __name__ == '__main__':
    main()
