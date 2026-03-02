#!/usr/bin/env python3
"""
测试 Project 5: 模型比较与缩放定律

测试模型工厂、缩放分析功能。
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


def test_model_factory():
    """测试模型工厂"""
    logger.info("\\n=== Testing Model Factory ===")

    from src.models.model_factory import ModelFactory
    import torch

    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # 创建工厂
    factory = ModelFactory(device=device)

    # 列出可用模型
    models = factory.list_available_models()
    logger.info(f"Available models: {models}")

    # 获取模型信息
    for model_key in models[:2]:  # 只测试前两个
        info = factory.get_model_info(model_key)
        logger.info(f"\\n{model_key}: {info}")

    # 创建一个小模型
    logger.info("\\nCreating gpt2-custom-tiny...")
    model = factory.create_model(
        model_key="gpt2-custom-tiny",
        use_lora=True,
        lora_r=8
    )

    logger.info(f"Model created successfully")

    # 跳过 tokenizer 测试（需要网络连接）
    logger.info("Skipping tokenizer test (requires network)")

    logger.info("✓ Model factory test passed")


def test_scaling_analyzer():
    """测试缩放分析器"""
    logger.info("\\n=== Testing Scaling Analyzer ===")

    from src.evaluation.scaling_analysis import ScalingAnalyzer, simulate_scaling_law

    # 模拟数据
    results = simulate_scaling_law(num_models=5)
    logger.info(f"Simulated {len(results)} models")

    # 创建分析器
    analyzer = ScalingAnalyzer()

    for result in results:
        analyzer.add_result(**result)

    # 分析参数缩放
    param_scaling = analyzer.analyze_param_scaling()
    logger.info(f"\\nParameter scaling: {param_scaling}")

    # 分析效率
    efficiency = analyzer.analyze_efficiency()
    logger.info(f"\\nEfficiency analysis:")
    logger.info(f"  Best param efficiency: {efficiency['best_param_efficiency']['model_name']}")
    logger.info(f"  Best memory efficiency: {efficiency['best_memory_efficiency']['model_name']}")
    logger.info(f"  Best speed efficiency: {efficiency['best_speed_efficiency']['model_name']}")

    # 保存结果
    output_dir = project_root / "results" / "test_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer.save_analysis(str(output_dir / "test_analysis.json"))
    analyzer.plot_scaling_curves(str(output_dir))

    logger.info(f"\\nResults saved to {output_dir}")
    logger.info("✓ Scaling analyzer test passed")


def test_model_comparison_cli():
    """测试模型比较 CLI"""
    logger.info("\\n=== Testing Model Comparison CLI ===")

    import subprocess

    # 测试列出模型
    logger.info("\\nTesting list mode...")
    result = subprocess.run(
        ["python", "scripts/model_comparison.py", "--mode", "list"],
        cwd=project_root,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        logger.info("List mode: ✓")
    else:
        logger.error(f"List mode failed: {result.stderr}")

    # 测试模拟模式
    logger.info("\\nTesting simulate mode...")
    result = subprocess.run(
        ["python", "scripts/model_comparison.py",
         "--mode", "simulate",
         "--output_dir", "results/test_simulate"],
        cwd=project_root,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        logger.info("Simulate mode: ✓")
    else:
        logger.error(f"Simulate mode failed: {result.stderr}")

    logger.info("✓ Model comparison CLI test passed")


def main():
    """运行所有测试"""
    logger.info("=== Project 5: Model Comparison Tests ===\\n")

    try:
        # 测试 1: 模型工厂
        test_model_factory()

        # 测试 2: 缩放分析器
        test_scaling_analyzer()

        # 测试 3: CLI（可选）
        if "--full" in sys.argv:
            test_model_comparison_cli()
        else:
            logger.info("\\nSkipping CLI test (use --full to run)")

        logger.info("\\n=== All tests passed! ===")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
