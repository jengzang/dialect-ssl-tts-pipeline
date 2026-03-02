"""
Lesson 5: MFA 声学模型训练

Montreal Forced Aligner (MFA) 声学模型训练实战

功能：
- 语料验证
- 声学模型训练
- 模型测试（对齐）

使用示例：
    # 验证语料
    python scripts/lesson_05_mfa_training.py --mode validate \\
        --corpus_dir data/mfa_corpus \\
        --dictionary_path data/dict.txt

    # 训练模型
    python scripts/lesson_05_mfa_training.py --mode train \\
        --corpus_dir data/mfa_corpus \\
        --dictionary_path data/dict.txt \\
        --output_dir checkpoints/mfa_model

    # 测试对齐
    python scripts/lesson_05_mfa_training.py --mode align \\
        --model_path checkpoints/mfa_model/acoustic_model.zip \\
        --test_corpus data/test_corpus \\
        --dictionary_path data/dict.txt \\
        --output_dir results/alignments
"""

import argparse
import logging
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.training.mfa_trainer import MFATrainer
from src.data_pipeline.mfa_wrapper import MFAWrapper


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Lesson 5: MFA Acoustic Model Training'
    )

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['validate', 'train', 'align', 'list_models'],
        help='运行模式'
    )

    parser.add_argument(
        '--corpus_dir',
        type=str,
        help='语料目录'
    )

    parser.add_argument(
        '--dictionary_path',
        type=str,
        help='发音词典路径'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints/mfa_model',
        help='输出目录'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        help='模型路径（用于 align 模式）'
    )

    parser.add_argument(
        '--test_corpus',
        type=str,
        help='测试语料目录（用于 align 模式）'
    )

    parser.add_argument(
        '--num_jobs',
        type=int,
        default=4,
        help='并行任务数'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default='acoustic',
        choices=['acoustic', 'dictionary'],
        help='模型类型（用于 list_models 模式）'
    )

    return parser.parse_args()


def validate_mode(args, logger):
    """验证模式"""
    logger.info("=== Validation Mode ===")

    if not args.corpus_dir or not args.dictionary_path:
        logger.error("--corpus_dir and --dictionary_path are required")
        return False

    config = {
        'corpus_dir': args.corpus_dir,
        'dictionary_path': args.dictionary_path,
        'output_dir': args.output_dir,
        'mfa': {
            'num_jobs': args.num_jobs
        }
    }

    trainer = MFATrainer(config)

    # 准备语料
    if not trainer.prepare_corpus():
        logger.error("Corpus preparation failed")
        return False

    # 验证
    if trainer.validate():
        logger.info("[OK] Validation passed")
        return True
    else:
        logger.warning("[WARNING] Validation found issues")
        return False


def train_mode(args, logger):
    """训练模式"""
    logger.info("=== Training Mode ===")

    if not args.corpus_dir or not args.dictionary_path:
        logger.error("--corpus_dir and --dictionary_path are required")
        return False

    config = {
        'corpus_dir': args.corpus_dir,
        'dictionary_path': args.dictionary_path,
        'output_dir': args.output_dir,
        'mfa': {
            'num_jobs': args.num_jobs
        }
    }

    trainer = MFATrainer(config)

    # 训练
    model_path = trainer.train(validate_first=True)

    if model_path:
        logger.info(f"[OK] Training completed: {model_path}")
        return True
    else:
        logger.error("[ERROR] Training failed")
        return False


def align_mode(args, logger):
    """对齐模式"""
    logger.info("=== Alignment Mode ===")

    if not all([args.model_path, args.test_corpus, args.dictionary_path]):
        logger.error("--model_path, --test_corpus, --dictionary_path required")
        return False

    config = {
        'mfa': {
            'num_jobs': args.num_jobs
        }
    }

    mfa = MFAWrapper(config['mfa'])

    # 对齐
    success = mfa.align(
        args.test_corpus,
        args.dictionary_path,
        args.model_path,
        args.output_dir
    )

    if success:
        logger.info(f"[OK] Alignment completed: {args.output_dir}")
        return True
    else:
        logger.error("[ERROR] Alignment failed")
        return False


def list_models_mode(args, logger):
    """列出可用模型"""
    logger.info("=== List Models Mode ===")

    config = {
        'mfa': {
            'num_jobs': args.num_jobs
        }
    }

    mfa = MFAWrapper(config['mfa'])

    # 检查 MFA 安装
    if not mfa.check_mfa_installed():
        logger.error("MFA is not installed")
        return False

    # 列出模型
    models = mfa.list_available_models(args.model_type)

    if models:
        logger.info(f"Available {args.model_type} models:")
        for model in models:
            logger.info(f"  - {model}")
        return True
    else:
        logger.warning("No models found")
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
    logger = setup_logger('lesson_05', log_config)

    logger.info("Starting Lesson 5: MFA Acoustic Model Training")
    logger.info(f"Mode: {args.mode}")

    # 根据模式执行
    if args.mode == 'validate':
        success = validate_mode(args, logger)
    elif args.mode == 'train':
        success = train_mode(args, logger)
    elif args.mode == 'align':
        success = align_mode(args, logger)
    elif args.mode == 'list_models':
        success = list_models_mode(args, logger)
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
