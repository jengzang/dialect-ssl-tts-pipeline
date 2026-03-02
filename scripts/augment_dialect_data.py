"""
数据增强脚本

将小规模方言数据集扩展到更大规模，并划分为训练/验证/测试集。

使用示例：
    python scripts/augment_dialect_data.py \
        --input material/lesson_8/dialect2mandarin.csv \
        --output_dir data/dialect_translation \
        --target_size 500 \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --test_ratio 0.15
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.dialect_augmentation import augment_dialect_data
from src.utils.logger import setup_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Augment dialect translation dataset'
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入数据路径（CSV 或 JSON）'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录'
    )

    parser.add_argument(
        '--target_size',
        type=int,
        default=500,
        help='目标数据集大小（默认 500）'
    )

    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='训练集比例（默认 0.7）'
    )

    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='验证集比例（默认 0.15）'
    )

    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='测试集比例（默认 0.15）'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认 42）'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    log_config = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_dir': 'logs'
    }
    logger = setup_logger('augment_dialect_data', log_config)

    logger.info("=== Dialect Data Augmentation ===")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Target size: {args.target_size}")
    logger.info(f"Split: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")

    try:
        augment_dialect_data(
            input_path=args.input,
            output_dir=args.output_dir,
            target_size=args.target_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )

        logger.info("=== Augmentation Completed Successfully ===")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Augmentation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
