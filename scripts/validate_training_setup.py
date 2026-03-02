"""
训练配置验证脚本

验证训练流程的所有组件是否正确配置，无需实际训练。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging
from src.utils.logger import setup_logger

# 设置日志
log_config = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_dir': 'logs'
}
logger = setup_logger('training_validation', log_config)

logger.info("=" * 60)
logger.info("Training Configuration Validation")
logger.info("=" * 60)

# 1. 检查 GPU
logger.info("\n[1/6] Checking GPU availability...")
cuda_available = torch.cuda.is_available()
logger.info(f"CUDA available: {cuda_available}")
if cuda_available:
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA version: {torch.version.cuda}")
else:
    logger.warning("No GPU detected. Training will be very slow on CPU.")
    logger.warning("For actual training, please use a machine with GPU.")

# 2. 检查依赖
logger.info("\n[2/6] Checking dependencies...")
try:
    import peft
    import accelerate
    import bitsandbytes
    import transformers
    logger.info(f"peft: {peft.__version__}")
    logger.info(f"accelerate: {accelerate.__version__}")
    logger.info(f"transformers: {transformers.__version__}")
    logger.info("[OK] All LoRA dependencies installed")
except ImportError as e:
    logger.error(f"[ERROR] Missing dependency: {e}")
    sys.exit(1)

# 3. 检查数据
logger.info("\n[3/6] Checking training data...")
import json

try:
    with open('data/dialect_translation/train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('data/dialect_translation/val.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Val samples: {len(val_data)}")

    # 验证数据格式
    sample = train_data[0]
    assert 'dialect' in sample and 'mandarin' in sample
    logger.info("[OK] Data format valid")

except Exception as e:
    logger.error(f"[ERROR] Data check failed: {e}")
    sys.exit(1)

# 4. 检查模型文件
logger.info("\n[4/6] Checking model files...")
try:
    from src.models.dialect_translator import DialectTranslator
    from src.training.dialect_translation_trainer import DialectTranslationTrainer
    logger.info("[OK] Model classes imported successfully")
except Exception as e:
    logger.error(f"[ERROR] Model import failed: {e}")
    sys.exit(1)

# 5. 检查评估模块
logger.info("\n[5/6] Checking evaluation modules...")
try:
    from src.evaluation.mt_metrics import MTMetrics
    logger.info("[OK] Evaluation modules available")
except Exception as e:
    logger.error(f"[ERROR] Evaluation import failed: {e}")
    sys.exit(1)

# 6. 测试配置
logger.info("\n[6/6] Testing training configuration...")
try:
    config = {
        'model_name': 'Qwen/Qwen-7B-Chat',
        'output_dir': 'checkpoints/dialect_translation_test',
        'epochs': 1,
        'batch_size': 2,
        'learning_rate': 2e-4,
        'lora_r': 8,
        'lora_alpha': 32,
        'max_length': 512,
        'use_wandb': False
    }

    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    logger.info("[OK] Configuration valid")

except Exception as e:
    logger.error(f"[ERROR] Configuration test failed: {e}")
    sys.exit(1)

# 总结
logger.info("\n" + "=" * 60)
logger.info("Validation Summary")
logger.info("=" * 60)

if cuda_available:
    logger.info("[OK] All checks passed! Ready for GPU training.")
    logger.info("\nTo start training, run:")
    logger.info("python scripts/lesson_08_dialect_translation.py \\")
    logger.info("    --mode train \\")
    logger.info("    --train_data data/dialect_translation/train.json \\")
    logger.info("    --val_data data/dialect_translation/val.json \\")
    logger.info("    --output_dir checkpoints/dialect_translation_v2 \\")
    logger.info("    --epochs 3 \\")
    logger.info("    --batch_size 4 \\")
    logger.info("    --use_wandb")
else:
    logger.info("[WARNING] All checks passed, but no GPU detected.")
    logger.info("\nOptions:")
    logger.info("1. Run on a machine with GPU for actual training")
    logger.info("2. Use a smaller model for CPU testing (not recommended)")
    logger.info("3. Use cloud GPU services (Google Colab, AWS, etc.)")

    logger.info("\nFor demonstration purposes, you can try CPU training with:")
    logger.info("python scripts/lesson_08_dialect_translation.py \\")
    logger.info("    --mode train \\")
    logger.info("    --train_data data/dialect_translation/train.json \\")
    logger.info("    --val_data data/dialect_translation/val.json \\")
    logger.info("    --output_dir checkpoints/dialect_translation_cpu \\")
    logger.info("    --epochs 1 \\")
    logger.info("    --batch_size 1 \\")
    logger.info("    --quantization")
    logger.info("\nWARNING: This will be VERY slow and may take hours!")

logger.info("\n" + "=" * 60)
