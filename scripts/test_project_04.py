#!/usr/bin/env python3
"""
测试 Project 4: 指令微调与提示工程

测试指令数据集构建、模型创建和评估功能。
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


def test_instruction_builder():
    """测试指令构建器"""
    logger.info("\\n=== Testing Instruction Builder ===")

    from src.data_pipeline.instruction_builder import InstructionBuilder

    # 创建构建器
    builder = InstructionBuilder(template_style="alpaca")

    # 测试翻译指令
    translation_inst = builder.build_translation_instruction(
        dialect_text="侬好啊",
        mandarin_text="你好",
        reverse=False
    )

    logger.info(f"Translation instruction: {translation_inst}")

    # 测试分类指令
    classification_inst = builder.build_classification_instruction(
        text="侬好啊，今朝天气老好额",
        dialect_label="上海话"
    )

    logger.info(f"Classification instruction: {classification_inst}")

    # 测试格式化
    formatted = builder.format_instruction(translation_inst)
    logger.info(f"Formatted instruction:\\n{formatted}")

    logger.info("✓ Instruction builder test passed")


def test_instruction_dataset_creation():
    """测试指令数据集创建"""
    logger.info("\\n=== Testing Instruction Dataset Creation ===")

    from src.data_pipeline.instruction_builder import create_instruction_dataset

    # 创建合成数据
    translation_data = [
        {"dialect": "侬好啊", "mandarin": "你好"},
        {"dialect": "今朝天气老好额", "mandarin": "今天天气很好"},
        {"dialect": "侬吃饭了伐", "mandarin": "你吃饭了吗"}
    ]

    classification_data = [
        {"text": "侬好啊", "label": "shanghai"},
        {"text": "你好呀", "label": "cantonese"},
        {"text": "你好", "label": "mandarin"}
    ]

    # 保存临时数据
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / "temp_translation.json", 'w', encoding='utf-8') as f:
        json.dump(translation_data, f, ensure_ascii=False)

    with open(data_dir / "temp_classification.json", 'w', encoding='utf-8') as f:
        json.dump(classification_data, f, ensure_ascii=False)

    # 创建指令数据集
    instructions = create_instruction_dataset(
        translation_data_path=str(data_dir / "temp_translation.json"),
        classification_data_path=str(data_dir / "temp_classification.json"),
        output_path=str(data_dir / "temp_instructions.json"),
        template_style="alpaca",
        include_reverse_translation=True
    )

    logger.info(f"Created {len(instructions)} instructions")
    logger.info(f"Sample instruction: {instructions[0]}")

    logger.info("✓ Instruction dataset creation test passed")


def test_instruction_tuned_model():
    """测试指令遵循模型"""
    logger.info("\\n=== Testing Instruction-Tuned Model ===")

    from src.models.instruction_tuned_model import create_instruction_tuned_model

    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 创建模型
    model = create_instruction_tuned_model(
        model_name="uer/gpt2-chinese-cluecorpussmall",
        lora_r=8,
        lora_alpha=16,
        device=device
    )

    logger.info(f"Total parameters: {model.count_parameters():,}")
    logger.info(f"Trainable parameters: {model.count_trainable_parameters():,}")

    # 测试前向传播
    logger.info("\\nTesting forward pass...")
    batch_size = 2
    seq_length = 32

    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    labels = input_ids.clone()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    logger.info(f"Loss: {outputs['loss'].item():.4f}")
    logger.info(f"Logits shape: {outputs['logits'].shape}")

    logger.info("✓ Instruction-tuned model test passed")


def test_instruction_evaluator():
    """测试指令评估器"""
    logger.info("\\n=== Testing Instruction Evaluator ===")

    from src.evaluation.instruction_eval import InstructionEvaluator

    evaluator = InstructionEvaluator()

    # 测试翻译评估
    translation_result = evaluator.evaluate_translation(
        predicted="你好",
        reference="你好"
    )

    logger.info(f"Translation evaluation: {translation_result}")

    # 测试分类评估
    classification_result = evaluator.evaluate_classification(
        predicted="上海话",
        reference="上海话"
    )

    logger.info(f"Classification evaluation: {classification_result}")

    # 测试指令遵循评估
    inst_result = evaluator.evaluate_instruction_following(
        instruction="请将以下方言翻译成普通话",
        input_text="侬好啊",
        predicted="你好",
        reference="你好",
        task_type="translation"
    )

    logger.info(f"Instruction following result: {inst_result}")

    # 计算聚合指标
    aggregate = evaluator.compute_aggregate_metrics()
    logger.info(f"Aggregate metrics: {aggregate}")

    logger.info("✓ Instruction evaluator test passed")


def main():
    """运行所有测试"""
    logger.info("=== Project 4: Instruction Tuning Tests ===\\n")

    try:
        # 测试 1: 指令构建器
        test_instruction_builder()

        # 测试 2: 指令数据集创建
        test_instruction_dataset_creation()

        # 测试 3: 指令遵循模型
        test_instruction_tuned_model()

        # 测试 4: 指令评估器
        test_instruction_evaluator()

        logger.info("\\n=== All tests passed! ===")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
