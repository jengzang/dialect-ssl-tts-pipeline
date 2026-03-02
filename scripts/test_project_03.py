#!/usr/bin/env python3
"""
测试 Project 3: 多任务学习

生成合成数据并测试多任务模型。
"""

import json
import logging
from pathlib import Path
import sys
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_classification_data():
    """创建合成分类数据"""
    logger.info("Creating synthetic classification data...")

    # 合成数据
    data = [
        {"text": "侬好啊，今朝天气老好额", "label": "shanghai", "accent": 0},
        {"text": "你好啊，今天天气很好", "label": "mandarin", "accent": 2},
        {"text": "你好呀，今日天气好靓", "label": "cantonese", "accent": 1},
        {"text": "侬晓得伐，阿拉上海宁", "label": "shanghai", "accent": 0},
        {"text": "我知道，我们是上海人", "label": "mandarin", "accent": 2},
        {"text": "我知道，我哋系广东人", "label": "cantonese", "accent": 1},
        {"text": "今朝吃啥好呢", "label": "shanghai", "accent": 0},
        {"text": "今天吃什么好呢", "label": "mandarin", "accent": 2},
        {"text": "今日食乜嘢好呢", "label": "cantonese", "accent": 1},
        {"text": "侬屋里厢有几个人", "label": "shanghai", "accent": 0},
        {"text": "你家里有几个人", "label": "mandarin", "accent": 2},
        {"text": "你屋企有几多人", "label": "cantonese", "accent": 1},
        {"text": "阿拉一道去白相", "label": "shanghai", "accent": 0},
        {"text": "我们一起去玩", "label": "mandarin", "accent": 2},
        {"text": "我哋一齐去玩", "label": "cantonese", "accent": 1},
        {"text": "侬做啥工作额", "label": "shanghai", "accent": 0},
        {"text": "你做什么工作", "label": "mandarin", "accent": 2},
        {"text": "你做乜嘢工作", "label": "cantonese", "accent": 1},
        {"text": "今朝夜里厢见", "label": "shanghai", "accent": 0},
        {"text": "今天晚上见", "label": "mandarin", "accent": 2},
        {"text": "今晚见", "label": "cantonese", "accent": 1},
    ]

    # 保存数据
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    train_data = data[:15]
    val_data = data[15:18]
    test_data = data[18:]

    train_path = data_dir / "accent_train.json"
    val_path = data_dir / "accent_val.json"
    test_path = data_dir / "accent_test.json"

    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Created {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
    logger.info(f"Saved to: {data_dir}")

    return train_path, val_path, test_path


def test_multitask_dataset():
    """测试多任务数据集"""
    logger.info("\\n=== Testing Multitask Dataset ===")

    from transformers import AutoTokenizer
    from src.data_pipeline.multitask_dataset import MultitaskDialectDataset

    # 创建分类数据
    train_path, val_path, test_path = create_synthetic_classification_data()

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "uer/gpt2-chinese-cluecorpussmall",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建数据集
    translation_data = project_root / "data" / "dialect_parallel_train.json"

    if not translation_data.exists():
        logger.warning(f"Translation data not found: {translation_data}")
        logger.info("Using only classification data for testing")
        translation_data = None

    dataset = MultitaskDialectDataset(
        translation_data_path=str(translation_data) if translation_data else None,
        classification_data_path=str(train_path),
        tokenizer=tokenizer,
        max_length=128,
        task_sampling="balanced"
    )

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Task distribution: {dataset.get_task_distribution()}")

    # 测试获取样本
    sample = dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Task type: {sample.get('task_type', 'N/A')}")
    logger.info(f"Input shape: {sample['input_ids'].shape}")

    logger.info("✓ Multitask dataset test passed")


def test_multitask_model():
    """测试多任务模型"""
    logger.info("\\n=== Testing Multitask Model ===")

    from src.models.multitask_dialect_model import create_multitask_model

    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 创建模型
    model = create_multitask_model(
        model_name="uer/gpt2-chinese-cluecorpussmall",
        num_classes=4,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        use_task_specific_lora=False,
        device=device
    )

    logger.info(f"Total parameters: {model.count_parameters():,}")
    logger.info(f"Trainable parameters: {model.count_trainable_parameters():,}")

    # 测试前向传播 - 翻译任务
    logger.info("\\nTesting translation forward pass...")
    batch_size = 2
    seq_length = 32

    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    labels = input_ids.clone()
    task_id = torch.zeros(batch_size, dtype=torch.long).to(device)  # 0 = translation

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task_id=task_id,
        labels=labels
    )

    logger.info(f"Translation loss: {outputs['loss'].item():.4f}")
    logger.info(f"Translation logits shape: {outputs['logits'].shape}")

    # 测试前向传播 - 分类任务
    logger.info("\\nTesting classification forward pass...")
    task_id = torch.ones(batch_size, dtype=torch.long).to(device)  # 1 = classification
    classification_label = torch.randint(0, 4, (batch_size,)).to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task_id=task_id,
        classification_label=classification_label
    )

    logger.info(f"Classification loss: {outputs['loss'].item():.4f}")
    logger.info(f"Classification logits shape: {outputs['logits'].shape}")

    logger.info("✓ Multitask model test passed")


def test_multitask_trainer():
    """测试多任务训练器"""
    logger.info("\\n=== Testing Multitask Trainer ===")

    from transformers import AutoTokenizer
    from src.data_pipeline.multitask_dataset import create_multitask_dataloaders
    from src.models.multitask_dialect_model import create_multitask_model
    from src.training.multitask_trainer import MultitaskTrainer

    # 创建分类数据
    train_path, val_path, test_path = create_synthetic_classification_data()

    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "uer/gpt2-chinese-cluecorpussmall",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建数据加载器
    translation_train = project_root / "data" / "dialect_parallel_train.json"
    translation_val = project_root / "data" / "dialect_parallel_val.json"

    train_loader, val_loader = create_multitask_dataloaders(
        translation_train_path=str(translation_train) if translation_train.exists() else None,
        translation_val_path=str(translation_val) if translation_val.exists() else None,
        classification_train_path=str(train_path),
        classification_val_path=str(val_path),
        tokenizer=tokenizer,
        batch_size=2,
        max_length=128,
        task_sampling="balanced"
    )

    # 创建模型
    model = create_multitask_model(
        model_name="uer/gpt2-chinese-cluecorpussmall",
        num_classes=4,
        lora_r=8,
        lora_alpha=16,
        device=device
    )

    # 创建训练器
    output_dir = project_root / "results" / "test_multitask"
    trainer = MultitaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        loss_weights={'translation': 1.0, 'classification': 1.0},
        gradient_normalization=True,
        use_wandb=False,
        output_dir=str(output_dir)
    )

    # 训练 1 个 epoch
    logger.info("Training for 1 epoch...")
    trainer.train(num_epochs=1, save_every=1)

    logger.info("✓ Multitask trainer test passed")


def main():
    """运行所有测试"""
    import sys
    logger.info("=== Project 3: Multitask Learning Tests ===\\n")

    try:
        # 测试 1: 数据集
        test_multitask_dataset()

        # 测试 2: 模型
        test_multitask_model()

        # 测试 3: 训练器（可选，需要更多时间）
        import sys
        if "--full" in sys.argv:
            test_multitask_trainer()
        else:
            logger.info("\\nSkipping trainer test (use --full to run)")

        logger.info("\\n=== All tests passed! ===")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
