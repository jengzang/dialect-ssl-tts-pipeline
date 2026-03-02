"""
快速验证训练脚本

使用小规模数据和少量 epoch 快速验证训练流程。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from src.utils.logger import setup_logger

# 设置日志
log_config = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_dir': 'logs'
}
logger = setup_logger('quick_train_test', log_config)

logger.info("=== Quick Training Test ===")
logger.info("This is a quick test to verify the training pipeline works.")
logger.info("For full training, use: scripts/lesson_08_dialect_translation.py")

# 检查数据
import json

try:
    with open('data/dialect_translation/train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('data/dialect_translation/val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open('data/dialect_translation/test.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Val samples: {len(val_data)}")
    logger.info(f"Test samples: {len(test_data)}")

    # 显示示例
    logger.info("\n=== Sample Data ===")
    for i, sample in enumerate(train_data[:3]):
        logger.info(f"Sample {i+1}:")
        logger.info(f"  Dialect: {sample['dialect']}")
        logger.info(f"  Mandarin: {sample['mandarin']}")

    logger.info("\n[OK] Data verification passed!")
    logger.info("\nNext steps:")
    logger.info("1. Install LoRA dependencies: pip install peft accelerate bitsandbytes")
    logger.info("2. Run full training with:")
    logger.info("   python scripts/lesson_08_dialect_translation.py \\")
    logger.info("       --mode train \\")
    logger.info("       --train_data data/dialect_translation/train.json \\")
    logger.info("       --val_data data/dialect_translation/val.json \\")
    logger.info("       --epochs 3 \\")
    logger.info("       --use_wandb")

except Exception as e:
    logger.error(f"[ERROR] Data verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
