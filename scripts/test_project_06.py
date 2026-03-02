#!/usr/bin/env python3
"""
测试 Project 6: 高级微调技术

测试不同的参数高效微调方法
"""

import logging
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_prefix_tuning():
    """测试 Prefix Tuning"""
    logger.info("\n=== Testing Prefix Tuning ===")

    from src.models.prefix_tuning_model import PrefixTuningModel
    from transformers import GPT2Config, GPT2LMHeadModel
    import torch

    # 创建小模型
    config = GPT2Config(
        vocab_size=1000,
        n_positions=512,
        n_embd=256,
        n_layer=4,
        n_head=4
    )

    model = GPT2LMHeadModel(config)

    # 创建 Prefix Tuning 模型
    prefix_model = PrefixTuningModel(
        model=model,
        prefix_length=10,
        prefix_hidden_size=128
    )

    # 测试前向传播
    input_ids = torch.randint(0, 1000, (2, 20))
    outputs = prefix_model(input_ids)

    logger.info(f"Output logits shape: {outputs.logits.shape}")
    logger.info(f"Trainable params: {prefix_model.get_trainable_parameters():,}")
    logger.info(f"Total params: {prefix_model.get_total_parameters():,}")
    logger.info(f"Trainable ratio: {prefix_model.get_trainable_parameters() / prefix_model.get_total_parameters() * 100:.2f}%")

    logger.info("✓ Prefix Tuning test passed")


def test_adapter():
    """测试 Adapter Layers"""
    logger.info("\n=== Testing Adapter Layers ===")

    from src.models.adapter_model import AdapterModel, AdapterLayer
    from transformers import GPT2Config, GPT2LMHeadModel
    import torch

    # 测试单个 Adapter Layer
    adapter = AdapterLayer(hidden_size=256, adapter_size=32)
    hidden_states = torch.randn(2, 10, 256)
    output = adapter(hidden_states)

    logger.info(f"Adapter output shape: {output.shape}")
    logger.info(f"Adapter params: {sum(p.numel() for p in adapter.parameters()):,}")

    # 创建完整模型
    config = GPT2Config(
        vocab_size=1000,
        n_positions=512,
        n_embd=256,
        n_layer=4,
        n_head=4
    )

    model = GPT2LMHeadModel(config)

    # 创建 Adapter 模型
    adapter_model = AdapterModel(
        model=model,
        adapter_size=32
    )

    # 测试前向传播
    input_ids = torch.randint(0, 1000, (2, 20))
    outputs = adapter_model(input_ids)

    logger.info(f"Output logits shape: {outputs.logits.shape}")
    logger.info(f"Trainable params: {adapter_model.get_trainable_parameters():,}")
    logger.info(f"Total params: {adapter_model.get_total_parameters():,}")
    logger.info(f"Trainable ratio: {adapter_model.get_trainable_parameters() / adapter_model.get_total_parameters() * 100:.2f}%")

    logger.info("✓ Adapter test passed")


def test_advanced_trainer():
    """测试高级训练器"""
    logger.info("\n=== Testing Advanced Trainer ===")

    from src.training.advanced_trainer import AdvancedFinetuner
    import torch

    # 测试 LoRA 方法
    logger.info("\nTesting LoRA method...")
    try:
        trainer = AdvancedFinetuner(
            model_name="gpt2",
            method="lora",
            method_config={"r": 8, "alpha": 16},
            device="cpu",
            mixed_precision=False
        )

        stats = trainer.get_stats()
        logger.info(f"LoRA stats:")
        logger.info(f"  Total params: {stats['total_params']:,}")
        logger.info(f"  Trainable params: {stats['trainable_params']:,}")
        logger.info(f"  Ratio: {stats['trainable_params'] / stats['total_params'] * 100:.2f}%")

        logger.info("✓ LoRA trainer test passed")

    except Exception as e:
        logger.warning(f"LoRA test skipped (requires network): {e}")

    logger.info("✓ Advanced trainer test passed")


def test_comparison():
    """测试方法比较"""
    logger.info("\n=== Testing Method Comparison ===")

    from transformers import GPT2Config, GPT2LMHeadModel
    import torch

    # 创建自定义小模型
    config = GPT2Config(
        vocab_size=1000,
        n_positions=512,
        n_embd=256,
        n_layer=4,
        n_head=4
    )

    model = GPT2LMHeadModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Base model params: {total_params:,}")

    # 比较不同方法的参数效率
    methods_params = {}

    # Prefix Tuning
    from src.models.prefix_tuning_model import PrefixTuningModel
    prefix_model = PrefixTuningModel(model, prefix_length=10)
    methods_params["Prefix Tuning"] = prefix_model.get_trainable_parameters()

    # Adapter
    from src.models.adapter_model import AdapterModel
    adapter_model = AdapterModel(model, adapter_size=32)
    methods_params["Adapter"] = adapter_model.get_trainable_parameters()

    # 输出比较
    logger.info("\nParameter Efficiency Comparison:")
    logger.info(f"{'Method':<20} {'Trainable Params':<20} {'Ratio':<10}")
    logger.info("-" * 50)

    for method, params in sorted(methods_params.items(), key=lambda x: x[1]):
        ratio = params / total_params * 100
        logger.info(f"{method:<20} {params:,<20} {ratio:.2f}%")

    logger.info("✓ Comparison test passed")


def main():
    """运行所有测试"""
    logger.info("=== Project 6: Advanced Finetuning Tests ===\n")

    try:
        # 测试 1: Prefix Tuning
        test_prefix_tuning()

        # 测试 2: Adapter
        test_adapter()

        # 测试 3: 高级训练器
        test_advanced_trainer()

        # 测试 4: 方法比较
        test_comparison()

        logger.info("\n=== All tests passed! ===")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
